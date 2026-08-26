"""
Lightning DataModule for reaction-conditioned residue pooling.
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Optional

import torch.distributed as dist
from torch.utils.data import BatchSampler, DataLoader, Dataset

from horizyn.data_module import HorizynDataModule
from horizyn.datasets.base import BaseDataset
from horizyn.datasets.collection import TupleDataset
from horizyn.datasets.csv import CSVDataset
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from horizyn.utils import dict_collate_fn, residue_collate_fn


class DualResidueEmbedDataset(BaseDataset[str]):
    """
    Pair value residue embeddings with separate scorer residue embeddings.

    The value embeddings are exposed as ``residue_embeddings`` and are pooled
    into the enzyme representation. The scorer embeddings are exposed as
    ``score_residue_embeddings`` and must align one-to-one with the same
    truncated residue positions.
    """

    def __init__(
        self,
        value_dataset: ResidueEmbedDataset,
        score_dataset: ResidueEmbedDataset,
        **kwargs: Any,
    ):
        self.value_dataset = value_dataset
        self.score_dataset = score_dataset
        self.vec_dim = value_dataset.vec_dim
        self.score_vec_dim = score_dataset.vec_dim
        common_keys = sorted(set(value_dataset.keys) & set(score_dataset.keys))
        if not common_keys:
            raise ValueError("No common protein IDs found between value and score residue HDF5s")
        super().__init__(keys=common_keys, use_key_to_idx=True, **kwargs)

    def __getitem__(self, key: str | int) -> dict[str, Any]:
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            actual_key = self.keys[key]
        else:
            actual_key = key

        value_sample = dict(self.value_dataset[actual_key])
        score_sample = self.score_dataset[actual_key]
        residue_embeddings = value_sample["residue_embeddings"]
        score_embeddings = score_sample["residue_embeddings"]
        if residue_embeddings.shape[0] != score_embeddings.shape[0]:
            raise ValueError(
                "Value and score residue embeddings must have the same truncated length "
                f"for protein {actual_key}: value={residue_embeddings.shape[0]}, "
                f"score={score_embeddings.shape[0]}"
            )
        value_sample["score_residue_embeddings"] = score_embeddings
        return self._apply_transforms(actual_key, value_sample)


class KeySubsetDataset(BaseDataset[str]):
    """Expose a key-ordered subset of another key-addressable dataset."""

    def __init__(self, source_dataset: BaseDataset[str], keys: list[str]):
        self.source_dataset = source_dataset
        super().__init__(keys=list(keys), use_key_to_idx=True)

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            actual_key = self.keys[key]
        else:
            actual_key = key
            self._get_idx(actual_key)
        return self.source_dataset[actual_key]


class IndexedMappingDataset(Dataset):
    """Return a mapping sample plus stable id and lookup-row metadata."""

    def __init__(
        self,
        dataset: BaseDataset[str],
        *,
        index_key: str,
        id_key: str,
    ):
        self.dataset = dataset
        self.index_key = index_key
        self.id_key = id_key

    @property
    def keys(self) -> list[str]:
        return list(self.dataset.keys)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_key = self.dataset.keys[index]
        sample = self.dataset[sample_key]
        if isinstance(sample, dict):
            output = dict(sample)
        else:
            output = {"value": sample}
        output[self.id_key] = sample_key
        output[self.index_key] = index
        return output


def _strip_direction_suffix(query_id: str) -> str:
    if query_id.endswith("_f") or query_id.endswith("_r"):
        return query_id[:-2]
    return query_id


def _load_id_list(path: str | Path) -> list[str]:
    ids: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value and not value.startswith("#"):
                ids.append(value.split(",")[0].split()[0])
    return ids


def _load_hard_negative_map(path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get(
            "anchor_to_negatives",
            payload.get("query_to_negatives", payload),
        )
    if not isinstance(payload, dict):
        raise ValueError(f"Hard-negative pool must be a JSON object: {path}")
    result: dict[str, list[str]] = {}
    for query_id, negatives in payload.items():
        if not isinstance(negatives, list):
            raise ValueError(f"Hard-negative candidates for {query_id!r} must be a JSON list")
        cleaned = [str(value).strip() for value in negatives if str(value).strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError(f"Hard-negative candidates for {query_id!r} contain duplicates")
        if cleaned:
            result[str(query_id)] = cleaned
    return result


class DirectionalHardNegativeBatchSampler(BatchSampler):
    """Build batches with hard negatives on either retrieval candidate axis."""

    def __init__(
        self,
        dataset: TupleDataset,
        *,
        batch_size: int,
        hard_negative_pools_path: str | Path,
        direction: str = "reaction_to_enzyme",
        anchor_queries_per_batch: int = 48,
        positives_per_query: int = 2,
        negatives_per_query: int = 8,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if anchor_queries_per_batch <= 0:
            raise ValueError("anchor_queries_per_batch must be positive")
        if positives_per_query <= 0:
            raise ValueError("positives_per_query must be positive")
        if negatives_per_query <= 0:
            raise ValueError("negatives_per_query must be positive")
        if direction not in {"reaction_to_enzyme", "enzyme_to_reaction"}:
            raise ValueError("direction must be reaction_to_enzyme or enzyme_to_reaction")
        if len(dataset) == 0:
            raise ValueError("Hard-negative sampling requires a non-empty dataset")
        if batch_size > len(dataset):
            raise ValueError(
                "batch_size cannot exceed the number of distinct pair rows when "
                f"sampling without replacement: batch_size={batch_size}, rows={len(dataset)}"
            )
        self.dataset = dataset
        self.direction = direction
        self.batch_size = int(batch_size)
        self.hard_negative_map = _load_hard_negative_map(hard_negative_pools_path)
        self.anchor_queries_per_batch = int(anchor_queries_per_batch)
        self.positives_per_query = int(positives_per_query)
        self.negatives_per_query = int(negatives_per_query)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        self.query_to_indices: dict[str, list[int]] = defaultdict(list)
        self.target_to_indices: dict[str, list[int]] = defaultdict(list)
        self.query_positive_targets: dict[str, set[str]] = defaultdict(set)
        self.target_positive_queries: dict[str, set[str]] = defaultdict(set)
        for row_idx, pair_key in enumerate(dataset.keys):
            pair = dataset.tuple_dataset[pair_key]
            query_id = str(pair["query_id"])
            target_id = str(pair["target_id"])
            self.query_to_indices[query_id].append(row_idx)
            self.target_to_indices[target_id].append(row_idx)
            self.query_positive_targets[query_id].add(target_id)
            self.target_positive_queries[target_id].add(query_id)

        anchor_ids = (
            self.query_to_indices
            if self.direction == "reaction_to_enzyme"
            else self.target_to_indices
        )
        hard_anchors = []
        for anchor_id in sorted(anchor_ids):
            base_anchor_id = _strip_direction_suffix(anchor_id)
            if anchor_id in self.hard_negative_map or base_anchor_id in self.hard_negative_map:
                hard_anchors.append(anchor_id)
        if not hard_anchors:
            raise ValueError(
                "No train anchors matched the hard-negative pool. "
                f"direction={self.direction}, anchors={len(anchor_ids)}, "
                f"pools={len(self.hard_negative_map)}"
            )
        self.anchor_queries = hard_anchors
        self.all_indices = list(range(len(dataset)))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rank_info(self) -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), int(dist.get_world_size())
        return 0, 1

    def _negative_candidates_for_anchor(self, anchor_id: str) -> list[str]:
        base_anchor_id = _strip_direction_suffix(anchor_id)
        return self.hard_negative_map.get(anchor_id) or self.hard_negative_map.get(
            base_anchor_id,
            [],
        )

    def _build_anchor_rows(self, anchor_id: str, rng: random.Random) -> list[int]:
        rows: list[int] = []
        if self.direction == "reaction_to_enzyme":
            positives = list(self.query_to_indices[anchor_id])
            positive_candidates = self.query_positive_targets[anchor_id]
            candidate_to_indices = self.target_to_indices
        else:
            positives = list(self.target_to_indices[anchor_id])
            positive_candidates = self.target_positive_queries[anchor_id]
            candidate_to_indices = self.query_to_indices
        rng.shuffle(positives)
        rows.extend(positives[: self.positives_per_query])

        hard_candidates = list(self._negative_candidates_for_anchor(anchor_id))
        rng.shuffle(hard_candidates)
        negative_rows = []
        for candidate_id in hard_candidates:
            if candidate_id in positive_candidates:
                continue
            candidate_rows = candidate_to_indices.get(candidate_id)
            if not candidate_rows:
                continue
            negative_rows.append(rng.choice(candidate_rows))
            if len(negative_rows) >= self.negatives_per_query:
                break
        rows.extend(negative_rows)
        return rows

    def __iter__(self):
        rank, world_size = self._rank_info()
        rng = random.Random(self.seed + self.epoch)
        query_order = list(self.anchor_queries)
        rng.shuffle(query_order)
        for batch_number in range(self._ddp_padded_total_batches()):
            start = batch_number * self.anchor_queries_per_batch
            anchor_queries = query_order[start : start + self.anchor_queries_per_batch]
            batch: list[int] = []
            seen: set[int] = set()
            for query_id in anchor_queries:
                for row_idx in self._build_anchor_rows(query_id, rng):
                    if row_idx not in seen:
                        batch.append(row_idx)
                        seen.add(row_idx)
                    if len(batch) >= self.batch_size:
                        break
                if len(batch) >= self.batch_size:
                    break
            remaining_indices = [row_idx for row_idx in self.all_indices if row_idx not in seen]
            rng.shuffle(remaining_indices)
            missing_count = self.batch_size - len(batch)
            if missing_count > len(remaining_indices):
                raise RuntimeError(
                    "Cannot complete a distinct-row hard-negative batch: "
                    f"need={missing_count}, available={len(remaining_indices)}"
                )
            batch.extend(remaining_indices[:missing_count])
            if self.drop_last and len(batch) < self.batch_size:
                continue
            if batch_number % world_size == rank:
                yield batch[: self.batch_size]
        self.epoch += 1

    def _total_batches(self) -> int:
        hard_negative_batches = math.ceil(len(self.anchor_queries) / self.anchor_queries_per_batch)
        standard_batches = math.ceil(len(self.dataset) / self.batch_size)
        return max(hard_negative_batches, standard_batches)

    def _ddp_padded_total_batches(self) -> int:
        total = self._total_batches()
        _rank, world_size = self._rank_info()
        if world_size <= 1:
            return total
        return math.ceil(total / world_size) * world_size

    def __len__(self) -> int:
        total = self._ddp_padded_total_batches()
        _rank, world_size = self._rank_info()
        return math.ceil(total / world_size)


ReactionHardNegativeBatchSampler = DirectionalHardNegativeBatchSampler


class ReactionDegreeBalancedBatchSampler(BatchSampler):
    """Sample DDP-global batches of distinct reactions with tempered degree weights."""

    def __init__(
        self,
        dataset: TupleDataset,
        *,
        batch_size: int,
        degree_exponent: float = 0.5,
        seed: int = 42,
        drop_last: bool = False,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if degree_exponent < 0.0:
            raise ValueError("degree_exponent must be non-negative")
        if (rank is None) != (world_size is None):
            raise ValueError("rank and world_size must be provided together")
        if world_size is not None and (
            world_size <= 0 or rank is None or not 0 <= rank < world_size
        ):
            raise ValueError("rank must be in [0, world_size)")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.degree_exponent = float(degree_exponent)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.fixed_rank = rank
        self.fixed_world_size = world_size
        self.epoch = 0
        self.query_to_indices: dict[str, list[int]] = defaultdict(list)
        for row_idx, pair_key in enumerate(dataset.keys):
            pair = dataset.tuple_dataset[pair_key]
            self.query_to_indices[str(pair["query_id"])].append(row_idx)
        if not self.query_to_indices:
            raise ValueError("Reaction-balanced sampling requires a non-empty dataset")
        self.query_ids = sorted(self.query_to_indices)
        self.query_weights = {
            query_id: len(indices) ** self.degree_exponent
            for query_id, indices in self.query_to_indices.items()
        }

    def _rank_info(self) -> tuple[int, int]:
        if self.fixed_rank is not None and self.fixed_world_size is not None:
            return int(self.fixed_rank), int(self.fixed_world_size)
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), int(dist.get_world_size())
        return 0, 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _weighted_without_replacement(
        values: list[str],
        weights: dict[str, float],
        count: int,
        rng: random.Random,
    ) -> list[str]:
        if count > len(values):
            raise ValueError(
                "A reaction-unique global batch is larger than the number of reactions: "
                f"batch={count}, reactions={len(values)}"
            )
        keyed = []
        for value in values:
            uniform = max(rng.random(), 1e-15)
            keyed.append((math.log(uniform) / weights[value], value))
        keyed.sort(reverse=True)
        return [value for _key, value in keyed[:count]]

    def __iter__(self):
        rank, world_size = self._rank_info()
        global_batch_size = self.batch_size * world_size
        rng = random.Random(self.seed + self.epoch)
        for _step in range(len(self)):
            queries = self._weighted_without_replacement(
                self.query_ids,
                self.query_weights,
                global_batch_size,
                rng,
            )
            global_indices = [rng.choice(self.query_to_indices[query_id]) for query_id in queries]
            start = rank * self.batch_size
            batch = global_indices[start : start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            yield batch
        self.epoch += 1

    def __len__(self) -> int:
        _rank, world_size = self._rank_info()
        denominator = self.batch_size * world_size
        if self.drop_last:
            return len(self.dataset) // denominator
        return math.ceil(len(self.dataset) / denominator)


class ReactionConditionedDataModule(HorizynDataModule):
    """
    DataModule for unpooled residue embeddings and reaction-conditioned pooling.

    This keeps the reaction fingerprint path from ``HorizynDataModule`` but
    replaces pooled protein vectors with padded residue-level tensors.
    """

    def __init__(
        self,
        train_pairs_path: str,
        test_pairs_path: str,
        train_reactions_path: str,
        test_reactions_path: str,
        protein_residue_embeds_path: str,
        protein_score_residue_embeds_path: str | None = None,
        train_batch_size: int = 64,
        retrieval_batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = False,
        rdkit_fp_dim: int = 1024,
        drfp_dim: int = 1024,
        reaction_representation: str = "fingerprint",
        reaction_embeds_path: str | None = None,
        reaction_t5v2_embeds_path: str | None = None,
        reaction_model_embeds_path: str | None = None,
        reaction_unimol2_embeds_path: str | None = None,
        reaction_chiro_embeds_path: str | None = None,
        reaction_chirality_embeds_path: str | None = None,
        reaction_chienn_embeds_path: str | None = None,
        reaction_chemistry_vectors_path: str | None = None,
        reaction_directional_vectors_path: str | None = None,
        train_reaction_embeds_path: str | None = None,
        validation_reaction_embeds_path: str | None = None,
        train_reaction_t5v2_embeds_path: str | None = None,
        validation_reaction_t5v2_embeds_path: str | None = None,
        train_reaction_model_embeds_path: str | None = None,
        validation_reaction_model_embeds_path: str | None = None,
        train_reaction_unimol2_embeds_path: str | None = None,
        validation_reaction_unimol2_embeds_path: str | None = None,
        train_reaction_chiro_embeds_path: str | None = None,
        validation_reaction_chiro_embeds_path: str | None = None,
        train_reaction_chirality_embeds_path: str | None = None,
        validation_reaction_chirality_embeds_path: str | None = None,
        train_reaction_chienn_embeds_path: str | None = None,
        validation_reaction_chienn_embeds_path: str | None = None,
        train_reaction_chemistry_vectors_path: str | None = None,
        validation_reaction_chemistry_vectors_path: str | None = None,
        train_reaction_directional_vectors_path: str | None = None,
        validation_reaction_directional_vectors_path: str | None = None,
        reaction_model_dim: int | None = None,
        reaction_unimol_dim: int = 768,
        reaction_chiro_dim: int | None = None,
        reaction_chirality_dim: int | None = None,
        reaction_chienn_dim: int | None = None,
        reaction_chemistry_dim: int | None = None,
        reaction_directional_dim: int | None = None,
        reaction_use_model: bool = True,
        reaction_use_chiro: bool | None = None,
        reaction_use_chirality: bool | None = None,
        reaction_use_chienn: bool = True,
        reaction_use_chemistry: bool = False,
        reaction_use_directional: bool = False,
        reaction_load_directional: bool = False,
        reaction_allow_missing_unimol2: bool = False,
        reaction_allow_missing_chiro: bool | None = None,
        reaction_allow_missing_chirality: bool | None = None,
        reaction_allow_missing_chienn: bool = False,
        reaction_allow_missing_chemistry: bool = True,
        reaction_allow_missing_directional: bool = True,
        reaction_embedding_in_memory: bool = True,
        residue_dim: int = 1024,
        score_residue_dim: int | None = None,
        max_protein_tokens: int | None = 1024,
        protein_truncation: str = "ends_center",
        standardize_reactions: bool = True,
        standardize_hypervalent: bool = True,
        standardize_remove_hs: bool = True,
        standardize_kekulize: bool = False,
        standardize_uncharge: bool = True,
        standardize_metals: bool = True,
        normalize_molecule_sets_as_self_reactions: bool = False,
        enzyme_ec_labels_path: str | None = None,
        protein_capability_vectors_path: str | None = None,
        protein_capability_metadata_path: str | None = None,
        capability_vector_in_memory: bool = True,
        capability_missing_policy: str = "zero_with_mask",
        protein_factorized_capability_vectors_path: str | None = None,
        factorized_capability_missing_policy: str = "zero_with_mask",
        protein_text_vectors_path: str | None = None,
        protein_text_metadata_path: str | None = None,
        text_vector_missing_policy: str = "zero_with_mask",
        protein_biofp_targets_path: str | None = None,
        protein_biofp_vocab_path: str | None = None,
        biofp_missing_policy: str = "zero_with_mask",
        validation_retrieval_metrics: bool = False,
        validation_retrieval_candidate_set: str = "validation",
        validation_retrieval_candidate_ids_path: str | None = None,
        validation_retrieval_batch_size: int | None = None,
        validation_retrieval_directions: Optional[List[str]] = None,
        hard_negative_pools_path: str | None = None,
        hard_negative_direction: str = "reaction_to_enzyme",
        hard_negative_anchor_queries_per_batch: int = 48,
        hard_negative_positives_per_query: int = 2,
        hard_negative_negatives_per_query: int = 8,
        hard_negative_seed: int = 42,
        reaction_balanced_sampling: bool = False,
        reaction_balanced_degree_exponent: float = 0.5,
        reaction_balanced_seed: int = 42,
        reaction_direction_mode: str = "bidirectional",
    ):
        super().__init__(
            train_pairs_path=train_pairs_path,
            test_pairs_path=test_pairs_path,
            train_reactions_path=train_reactions_path,
            test_reactions_path=test_reactions_path,
            protein_embeds_path=protein_residue_embeds_path,
            train_batch_size=train_batch_size,
            retrieval_batch_size=retrieval_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            rdkit_fp_dim=rdkit_fp_dim,
            drfp_dim=drfp_dim,
            reaction_representation=reaction_representation,
            reaction_embeds_path=reaction_embeds_path,
            reaction_t5v2_embeds_path=reaction_t5v2_embeds_path,
            reaction_model_embeds_path=reaction_model_embeds_path,
            reaction_unimol2_embeds_path=reaction_unimol2_embeds_path,
            reaction_chiro_embeds_path=reaction_chiro_embeds_path,
            reaction_chirality_embeds_path=reaction_chirality_embeds_path,
            reaction_chienn_embeds_path=reaction_chienn_embeds_path,
            reaction_chemistry_vectors_path=reaction_chemistry_vectors_path,
            reaction_directional_vectors_path=reaction_directional_vectors_path,
            train_reaction_embeds_path=train_reaction_embeds_path,
            validation_reaction_embeds_path=validation_reaction_embeds_path,
            train_reaction_t5v2_embeds_path=train_reaction_t5v2_embeds_path,
            validation_reaction_t5v2_embeds_path=validation_reaction_t5v2_embeds_path,
            train_reaction_model_embeds_path=train_reaction_model_embeds_path,
            validation_reaction_model_embeds_path=validation_reaction_model_embeds_path,
            train_reaction_unimol2_embeds_path=train_reaction_unimol2_embeds_path,
            validation_reaction_unimol2_embeds_path=validation_reaction_unimol2_embeds_path,
            train_reaction_chiro_embeds_path=train_reaction_chiro_embeds_path,
            validation_reaction_chiro_embeds_path=validation_reaction_chiro_embeds_path,
            train_reaction_chirality_embeds_path=train_reaction_chirality_embeds_path,
            validation_reaction_chirality_embeds_path=validation_reaction_chirality_embeds_path,
            train_reaction_chienn_embeds_path=train_reaction_chienn_embeds_path,
            validation_reaction_chienn_embeds_path=validation_reaction_chienn_embeds_path,
            train_reaction_chemistry_vectors_path=train_reaction_chemistry_vectors_path,
            validation_reaction_chemistry_vectors_path=validation_reaction_chemistry_vectors_path,
            train_reaction_directional_vectors_path=train_reaction_directional_vectors_path,
            validation_reaction_directional_vectors_path=validation_reaction_directional_vectors_path,
            reaction_model_dim=reaction_model_dim,
            reaction_unimol_dim=reaction_unimol_dim,
            reaction_chiro_dim=reaction_chiro_dim,
            reaction_chirality_dim=reaction_chirality_dim,
            reaction_chienn_dim=reaction_chienn_dim,
            reaction_chemistry_dim=reaction_chemistry_dim,
            reaction_directional_dim=reaction_directional_dim,
            reaction_use_model=reaction_use_model,
            reaction_use_chiro=reaction_use_chiro,
            reaction_use_chirality=reaction_use_chirality,
            reaction_use_chienn=reaction_use_chienn,
            reaction_use_chemistry=reaction_use_chemistry,
            reaction_use_directional=reaction_use_directional,
            reaction_load_directional=reaction_load_directional,
            reaction_allow_missing_unimol2=reaction_allow_missing_unimol2,
            reaction_allow_missing_chiro=reaction_allow_missing_chiro,
            reaction_allow_missing_chirality=reaction_allow_missing_chirality,
            reaction_allow_missing_chienn=reaction_allow_missing_chienn,
            reaction_allow_missing_chemistry=reaction_allow_missing_chemistry,
            reaction_allow_missing_directional=reaction_allow_missing_directional,
            reaction_embedding_in_memory=reaction_embedding_in_memory,
            standardize_reactions=standardize_reactions,
            standardize_hypervalent=standardize_hypervalent,
            standardize_remove_hs=standardize_remove_hs,
            standardize_kekulize=standardize_kekulize,
            standardize_uncharge=standardize_uncharge,
            standardize_metals=standardize_metals,
            normalize_molecule_sets_as_self_reactions=normalize_molecule_sets_as_self_reactions,
            enzyme_ec_labels_path=enzyme_ec_labels_path,
            protein_capability_vectors_path=protein_capability_vectors_path,
            protein_capability_metadata_path=protein_capability_metadata_path,
            capability_vector_in_memory=capability_vector_in_memory,
            capability_missing_policy=capability_missing_policy,
            protein_factorized_capability_vectors_path=protein_factorized_capability_vectors_path,
            factorized_capability_missing_policy=factorized_capability_missing_policy,
            protein_text_vectors_path=protein_text_vectors_path,
            protein_text_metadata_path=protein_text_metadata_path,
            text_vector_missing_policy=text_vector_missing_policy,
            protein_biofp_targets_path=protein_biofp_targets_path,
            protein_biofp_vocab_path=protein_biofp_vocab_path,
            biofp_missing_policy=biofp_missing_policy,
            reaction_direction_mode=reaction_direction_mode,
        )
        self.protein_residue_embeds_path = Path(protein_residue_embeds_path)
        self.protein_score_residue_embeds_path = (
            None
            if protein_score_residue_embeds_path is None
            else Path(protein_score_residue_embeds_path)
        )
        self.residue_dim = residue_dim
        self.score_residue_dim = score_residue_dim
        self.max_protein_tokens = max_protein_tokens
        self.protein_truncation = protein_truncation
        if validation_retrieval_candidate_set not in {"validation", "screening", "custom"}:
            raise ValueError(
                "validation_retrieval_candidate_set must be one of: "
                "validation, screening, custom"
            )
        self.validation_retrieval_metrics = bool(validation_retrieval_metrics)
        directions = (
            ["reaction_to_enzyme", "enzyme_to_reaction"]
            if validation_retrieval_directions is None
            else list(validation_retrieval_directions)
        )
        valid_directions = {"reaction_to_enzyme", "enzyme_to_reaction"}
        invalid_directions = sorted(set(directions) - valid_directions)
        if invalid_directions:
            raise ValueError(
                "validation_retrieval_directions entries must be drawn from "
                f"{sorted(valid_directions)}; got {invalid_directions}"
            )
        if not directions:
            raise ValueError("validation_retrieval_directions must not be empty")
        self.validation_retrieval_directions = tuple(dict.fromkeys(directions))
        self.validation_retrieval_candidate_set = validation_retrieval_candidate_set
        self.validation_retrieval_candidate_ids_path = (
            None
            if validation_retrieval_candidate_ids_path is None
            else Path(validation_retrieval_candidate_ids_path)
        )
        self.validation_retrieval_batch_size = (
            retrieval_batch_size
            if validation_retrieval_batch_size is None
            else int(validation_retrieval_batch_size)
        )
        self.hard_negative_pools_path = (
            None if hard_negative_pools_path is None else Path(hard_negative_pools_path)
        )
        if hard_negative_direction not in {
            "reaction_to_enzyme",
            "enzyme_to_reaction",
        }:
            raise ValueError(
                "hard_negative_direction must be reaction_to_enzyme or enzyme_to_reaction"
            )
        self.hard_negative_direction = hard_negative_direction
        self.hard_negative_anchor_queries_per_batch = int(hard_negative_anchor_queries_per_batch)
        self.hard_negative_positives_per_query = int(hard_negative_positives_per_query)
        self.hard_negative_negatives_per_query = int(hard_negative_negatives_per_query)
        self.hard_negative_seed = int(hard_negative_seed)
        self.reaction_balanced_sampling = bool(reaction_balanced_sampling)
        self.reaction_balanced_degree_exponent = float(reaction_balanced_degree_exponent)
        self.reaction_balanced_seed = int(reaction_balanced_seed)
        if self.reaction_balanced_degree_exponent < 0.0:
            raise ValueError("reaction_balanced_degree_exponent must be non-negative")
        if self.reaction_balanced_sampling and self.hard_negative_pools_path is not None:
            raise ValueError(
                "Reaction-balanced and hard-negative batch samplers are mutually exclusive"
            )
        self._val_metric_target_lookup_data = None
        self._val_metric_query_lookup_data = None
        self._val_metric_target_query_data = None
        self._val_metric_query_data = None
        self._target_to_queries = None
        self._val_retrieval_target_candidate_ids = None
        self._val_retrieval_query_candidate_ids = None

    def _create_target_dataset(self):
        value_dataset = ResidueEmbedDataset(
            file_path=str(self.protein_residue_embeds_path),
            in_memory=False,
            max_tokens=self.max_protein_tokens,
            truncation=self.protein_truncation,
        )
        if value_dataset.vec_dim != self.residue_dim:
            raise ValueError(
                f"Residue embedding dim mismatch: expected {self.residue_dim}, "
                f"got {value_dataset.vec_dim}"
            )
        if self.protein_score_residue_embeds_path is None:
            return value_dataset

        score_dataset = ResidueEmbedDataset(
            file_path=str(self.protein_score_residue_embeds_path),
            in_memory=False,
            max_tokens=self.max_protein_tokens,
            truncation=self.protein_truncation,
        )
        expected_score_dim = (
            score_dataset.vec_dim if self.score_residue_dim is None else self.score_residue_dim
        )
        if score_dataset.vec_dim != expected_score_dim:
            raise ValueError(
                f"Score residue embedding dim mismatch: expected {expected_score_dim}, "
                f"got {score_dataset.vec_dim}"
            )
        return DualResidueEmbedDataset(value_dataset=value_dataset, score_dataset=score_dataset)

    def _setup_validation_data(self):
        """Setup validation pairs and query groupings for this variant."""
        print("Setting up reaction-conditioned validation data...")

        val_pairs = CSVDataset(
            file_path=str(self.test_pairs_path),
            key_column="pr_id",
            columns=["reaction_id", "protein_id"],
            rename_map={"reaction_id": "query_id", "protein_id": "target_id"},
        )
        original_val_pair_count = len(val_pairs)
        print(f"  Loaded {original_val_pair_count} validation pairs")

        val_pairs = self._augment_pairs_for_direction_mode(val_pairs)
        print(
            f"  Prepared {len(val_pairs)} pairs with "
            f"reaction_direction_mode={self.reaction_direction_mode}"
        )

        self._val_query_data_raw = self._create_query_dataset(
            self.test_reactions_path,
            split_name="validation",
        )
        print(f"  Loaded {len(self._val_query_data_raw)} test reactions")
        val_pairs = self._filter_pairs_to_available_queries(
            val_pairs,
            self._val_query_data_raw,
            split_name="validation",
        )

        if self._target_data is None:
            raise RuntimeError("Training data must be setup before validation data")

        self._screening_target_data = self._maybe_attach_target_side_vectors(
            self._create_target_dataset()
        )

        train_pairs_for_stats = CSVDataset(
            file_path=str(self.train_pairs_path),
            key_column="pr_id",
            columns=["reaction_id", "protein_id"],
            rename_map={"reaction_id": "query_id", "protein_id": "target_id"},
        )
        train_protein_ids = set(
            train_pairs_for_stats[k]["target_id"] for k in train_pairs_for_stats.keys
        )
        val_protein_ids = set(val_pairs[k]["target_id"] for k in val_pairs.keys)
        screening_protein_ids = set(self._screening_target_data.keys)

        print(f"  Training proteins: {len(train_protein_ids)}")
        print(f"  Validation proteins: {len(val_protein_ids)}")
        print(f"  Screening set (full): {len(screening_protein_ids)} proteins")
        print(f"  Overlap (train ∩ val): {len(train_protein_ids & val_protein_ids)}")
        print(f"  Val-only proteins: {len(val_protein_ids - train_protein_ids)}")

        self._val_data = TupleDataset(
            tuple_dataset=val_pairs,
            key_name_to_dataset={
                "query_id": self._val_query_data_raw,
                "target_id": self._target_data,
            },
            rename_map={"query_id": "query_vec"},
        )

        valid_val_pair_keys = list(self._val_data.keys)
        filtered_val_pair_count = len(val_pairs) - len(valid_val_pair_keys)
        if filtered_val_pair_count > 0:
            print(
                "  Filtered "
                f"{filtered_val_pair_count}/{len(val_pairs)} validation retrieval pairs "
                "with missing query or target records"
            )

        query_to_targets, target_to_queries = self._build_pair_lookup_maps(
            val_pairs,
            valid_val_pair_keys,
        )

        unique_query_ids = sorted(query_to_targets.keys())
        if not unique_query_ids:
            raise ValueError(
                "No validation retrieval queries remain after filtering missing "
                "query/target records"
            )
        self._query_to_targets = query_to_targets
        self._target_to_queries = target_to_queries
        self._val_query_data = TupleDataset(
            tuple_dataset=BaseDataset(
                keys=unique_query_ids,
                array_data=[{"query_id": qid} for qid in unique_query_ids],
            ),
            key_name_to_dataset={"query_id": self._val_query_data_raw},
            rename_map={"query_id": "query_vec"},
        )
        self._val_retrieval_targets = BaseDataset(
            keys=unique_query_ids,
            array_data=[query_to_targets[qid] for qid in unique_query_ids],
        )

        print(f"Validation dataset ready: {len(self._val_data)} samples")
        print(f"  Unique validation queries: {len(unique_query_ids)}")
        valid_positive_count = sum(len(targets) for targets in query_to_targets.values())
        print(f"  Avg valid targets per query: {valid_positive_count / len(unique_query_ids):.2f}")

        if self.validation_retrieval_metrics:
            screening_available_ids = set(self._screening_target_data.keys)
            target_query_ids = sorted(
                target_id
                for target_id in target_to_queries.keys()
                if target_id in screening_available_ids
            )
            missing_target_query_ids = len(target_to_queries) - len(target_query_ids)
            if missing_target_query_ids > 0:
                print(
                    "  Filtered "
                    f"{missing_target_query_ids} validation metric target anchors "
                    "missing from the screening target dataset"
                )
            if self.validation_retrieval_candidate_set == "screening":
                target_candidate_ids = sorted(screening_protein_ids)
            elif self.validation_retrieval_candidate_set == "custom":
                if self.validation_retrieval_candidate_ids_path is None:
                    raise ValueError(
                        "validation_retrieval_candidate_set='custom' requires "
                        "validation_retrieval_candidate_ids_path"
                    )
                requested_candidate_ids = _load_id_list(
                    self.validation_retrieval_candidate_ids_path
                )
                target_candidate_ids = [
                    target_id
                    for target_id in requested_candidate_ids
                    if target_id in screening_available_ids
                ]
                missing_candidate_ids = len(requested_candidate_ids) - len(target_candidate_ids)
                if missing_candidate_ids > 0:
                    print(
                        "  Filtered "
                        f"{missing_candidate_ids} custom validation candidates missing "
                        "from the screening target dataset"
                    )
            else:
                target_candidate_ids = target_query_ids
            query_candidate_ids = unique_query_ids
            if not target_candidate_ids:
                raise ValueError(
                    "No validation retrieval target candidates remain after filtering "
                    "missing target records"
                )

            target_candidate_dataset = KeySubsetDataset(
                self._screening_target_data,
                target_candidate_ids,
            )
            self._val_metric_target_lookup_data = IndexedMappingDataset(
                target_candidate_dataset,
                index_key="target_lookup_row_idx",
                id_key="target_id",
            )
            self._val_metric_target_query_data = IndexedMappingDataset(
                KeySubsetDataset(self._screening_target_data, target_query_ids),
                index_key="target_query_row_idx",
                id_key="target_id",
            )
            self._val_metric_query_lookup_data = IndexedMappingDataset(
                self._val_query_data,
                index_key="query_lookup_row_idx",
                id_key="query_id",
            )
            self._val_metric_query_data = IndexedMappingDataset(
                self._val_query_data,
                index_key="query_metric_row_idx",
                id_key="query_id",
            )
            self._val_retrieval_target_candidate_ids = target_candidate_ids
            self._val_retrieval_query_candidate_ids = query_candidate_ids
            print("  Validation retrieval metrics: enabled")
            print(
                "  Validation retrieval directions: "
                f"{', '.join(self.validation_retrieval_directions)}"
            )
            print(
                "  Retrieval target candidates: "
                f"{len(target_candidate_ids)} ({self.validation_retrieval_candidate_set})"
            )
            print(f"  Retrieval query candidates: {len(query_candidate_ids)}")

    def train_dataloader(self) -> DataLoader:
        if self._train_data is None:
            raise RuntimeError("Training data not setup. Call setup() first.")

        if self.hard_negative_pools_path is not None:
            sampler = DirectionalHardNegativeBatchSampler(
                self._train_data,
                batch_size=self.train_batch_size,
                hard_negative_pools_path=self.hard_negative_pools_path,
                direction=self.hard_negative_direction,
                anchor_queries_per_batch=self.hard_negative_anchor_queries_per_batch,
                positives_per_query=self.hard_negative_positives_per_query,
                negatives_per_query=self.hard_negative_negatives_per_query,
                seed=self.hard_negative_seed,
            )
            print(
                "Using directional hard-negative batch sampler: "
                f"direction={sampler.direction}, "
                f"anchors={sampler.anchor_queries_per_batch}, "
                f"positives/query={sampler.positives_per_query}, "
                f"negatives/query={sampler.negatives_per_query}, "
                f"matched_queries={len(sampler.anchor_queries)}"
            )
            return DataLoader(
                self._train_data,
                batch_sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=residue_collate_fn,
            )

        if self.reaction_balanced_sampling:
            sampler = ReactionDegreeBalancedBatchSampler(
                self._train_data,
                batch_size=self.train_batch_size,
                degree_exponent=self.reaction_balanced_degree_exponent,
                seed=self.reaction_balanced_seed,
            )
            print(
                "Using reaction degree-balanced batch sampler: "
                f"degree_exponent={sampler.degree_exponent}, "
                f"reactions={len(sampler.query_ids)}"
            )
            return DataLoader(
                self._train_data,
                batch_sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=residue_collate_fn,
            )

        return DataLoader(
            self._train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=residue_collate_fn,
        )

    def val_dataloader(self) -> List[DataLoader]:
        if self._val_data is None:
            raise RuntimeError("Validation data not setup. Call setup() first.")

        loaders: list[DataLoader] = [
            DataLoader(
                self._val_data,
                batch_size=self.train_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=residue_collate_fn,
            )
        ]
        if not self.validation_retrieval_metrics:
            return loaders
        if (
            self._val_metric_target_lookup_data is None
            or self._val_metric_query_data is None
            or self._val_metric_query_lookup_data is None
            or self._val_metric_target_query_data is None
        ):
            raise RuntimeError("Validation retrieval datasets were not initialized")

        metric_batch_size = max(1, int(self.validation_retrieval_batch_size))
        loaders.extend(
            [
                DataLoader(
                    self._val_metric_target_lookup_data,
                    batch_size=self.train_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    collate_fn=residue_collate_fn,
                ),
                DataLoader(
                    self._val_metric_query_lookup_data,
                    batch_size=metric_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    collate_fn=dict_collate_fn,
                ),
            ]
        )
        if "reaction_to_enzyme" in self.validation_retrieval_directions:
            loaders.append(
                DataLoader(
                    self._val_metric_query_data,
                    batch_size=metric_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    collate_fn=dict_collate_fn,
                )
            )
        if "enzyme_to_reaction" in self.validation_retrieval_directions:
            loaders.append(
                DataLoader(
                    self._val_metric_target_query_data,
                    batch_size=metric_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    collate_fn=residue_collate_fn,
                )
            )
        return loaders
