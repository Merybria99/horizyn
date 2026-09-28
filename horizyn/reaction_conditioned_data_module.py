"""
Lightning DataModule for reaction-conditioned residue pooling.
"""

import csv
import json
import math
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Iterator, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, DataLoader, Dataset

from horizyn.data_module import HorizynDataModule
from horizyn.balanced_anchor_sampler import BalancedAnchorBatchSampler
from horizyn.biological_residual import FunctionalTokenH5Dataset
from horizyn.config import DotDict
from horizyn.datasets.base import BaseDataset
from horizyn.datasets.collection import MergeDataset, TupleDataset
from horizyn.datasets.csv import CSVDataset
from horizyn.datasets.hdf5 import EmbedDataset
from horizyn.datasets.indexed_pairs import (
    IndexedPairDataset,
    IndexedPairs,
    IndexedTypedNegativeBatchSampler,
)
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from horizyn.reaction_features import build_reaction_feature_dataset
from horizyn.utils import dict_collate_fn, residue_collate_fn


def _initialize_loader_worker(worker_id: int, *, num_threads: int) -> None:
    """Bound CPU work per rank's worker and retain Lightning's worker seeding."""
    from lightning.fabric.utilities.seed import pl_worker_init_function
    torch.set_num_threads(num_threads)
    pl_worker_init_function(worker_id)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    threadpool_limits(limits=num_threads)


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
        mismatched_lengths = [
            (
                protein_id,
                value_dataset.length_by_key[protein_id],
                score_dataset.length_by_key[protein_id],
            )
            for protein_id in common_keys
            if value_dataset.length_by_key[protein_id] != score_dataset.length_by_key[protein_id]
        ]
        if mismatched_lengths:
            preview = ", ".join(
                f"{protein_id} (value={value_length}, score={score_length})"
                for protein_id, value_length, score_length in mismatched_lengths[:5]
            )
            raise ValueError(
                "Value and score residue HDF5s must contain one embedding per residue "
                f"before truncation; mismatched proteins: {preview}"
            )
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


class NamedEmbeddingDataset(BaseDataset[str]):
    """Expose a dense embedding cache as a named dictionary field."""

    def __init__(self, file_path: str | Path, field_name: str) -> None:
        self.embedding_dataset = EmbedDataset(
            file_path=str(file_path),
            in_memory=True,
        )
        self.field_name = str(field_name)
        self.vec_dim = self.embedding_dataset.vec_dim
        super().__init__(
            keys=list(self.embedding_dataset.keys),
            use_key_to_idx=True,
        )

    def __getitem__(self, key: str | int) -> dict[str, Any]:
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            actual_key = self.keys[key]
        else:
            actual_key = key
            self._get_idx(actual_key)
        return {self.field_name: self.embedding_dataset[actual_key]}


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


def _directional_validation_queries(path, query_ids, target_ids):
    """Keep fixed per-direction anchors while retaining full gold/candidates."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"reaction_to_enzyme", "enzyme_to_reaction"}:
        raise ValueError("Validation query panel requires reaction_to_enzyme and enzyme_to_reaction lists")
    result = []
    for direction, available in (("reaction_to_enzyme", set(query_ids)), ("enzyme_to_reaction", set(target_ids))):
        requested = payload[direction]
        if not isinstance(requested, list) or not requested or not all(isinstance(value, str) and value for value in requested):
            raise ValueError(f"Validation {direction} query panel must be a nonempty ID list")
        resolved = []
        for value in requested:
            actual = value
            if actual not in available and direction == "reaction_to_enzyme":
                actual = f"{value}_f"
            if actual not in available:
                raise ValueError(f"Requested validation {direction} query lacks gold/features: {value}")
            resolved.append(actual)
        if len(set(resolved)) != len(resolved):
            raise ValueError(f"Duplicate validation {direction} query IDs")
        result.append(resolved)
    return tuple(result)


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


class _CombinedPairView(BaseDataset[str]):
    """Pair metadata view for a primary dataset plus a replay subset."""

    def __init__(
        self,
        primary: TupleDataset,
        replay: TupleDataset,
        replay_indices: list[int],
    ) -> None:
        self.primary = primary
        self.replay = replay
        self.replay_indices = list(replay_indices)
        keys = [f"primary:{key}" for key in primary.keys] + [
            f"replay:{replay.keys[index]}" for index in replay_indices
        ]
        super().__init__(keys=keys, use_key_to_idx=True)

    def __getitem__(self, key: str | int) -> dict[str, Any]:
        index = self._get_idx(key) if isinstance(key, str) else key
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} is out of bounds for dataset of length {len(self)}")
        if index < len(self.primary):
            pair_key = self.primary.keys[index]
            return dict(self.primary.tuple_dataset[pair_key])
        replay_index = self.replay_indices[index - len(self.primary)]
        pair_key = self.replay.keys[replay_index]
        return dict(self.replay.tuple_dataset[pair_key])


class ReplayConcatDataset(Dataset):
    """Concatenate primary samples with a deterministic subset of source replay."""

    def __init__(
        self,
        primary: TupleDataset,
        replay: TupleDataset,
        *,
        replay_fraction: float,
        seed: int,
    ) -> None:
        if not 0.0 < replay_fraction < 1.0:
            raise ValueError("replay_fraction must lie in (0, 1)")
        desired = math.ceil(len(primary) * replay_fraction / (1.0 - replay_fraction))
        count = min(desired, len(replay))
        rng = random.Random(int(seed))
        replay_indices = sorted(rng.sample(range(len(replay)), k=count))
        self.primary = primary
        self.replay = replay
        self.replay_indices = replay_indices
        self.tuple_dataset = _CombinedPairView(primary, replay, replay_indices)
        self.keys = list(self.tuple_dataset.keys)
        self.realized_replay_fraction = count / max(1, len(primary) + count)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, key: str | int) -> dict[str, Any]:
        index = self.tuple_dataset._get_idx(key) if isinstance(key, str) else key
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} is out of bounds for dataset of length {len(self)}")
        if index < len(self.primary):
            return self.primary[index]
        return self.replay[self.replay_indices[index - len(self.primary)]]


def _union_positive_lookups(
    first: dict[str, list[str]],
    second: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Union complete annotation graphs without depending on sampled rows."""

    merged: dict[str, set[str]] = defaultdict(set)
    for lookup in (first, second):
        for key, values in lookup.items():
            merged[str(key)].update(str(value) for value in values)
    return {key: sorted(values) for key, values in merged.items()}


def _construct_replay_data_module(
    kwargs: dict[str, Any],
) -> "ReactionConditionedDataModule":
    """Construct replay outside a data-module method's Lightning frame chain.

    Lightning's argument collector walks every enclosing method whose ``self``
    is a HyperparametersMixin. Constructing the same data-module class directly
    inside ``_setup_training_data`` therefore makes it inspect that non-init
    frame as though it were ``__init__`` and raises a spurious KeyError.
    """

    return ReactionConditionedDataModule(**kwargs)


def _strip_direction_suffix(query_id: str) -> str:
    if query_id.endswith("_f") or query_id.endswith("_r"):
        return query_id[:-2]
    return query_id


def _resolve_typed_pool_query_id(
    pool_query_id: str,
    available_query_ids: set[str],
) -> str | None:
    if pool_query_id in available_query_ids:
        return pool_query_id
    forward_query_id = f"{pool_query_id}_f"
    return forward_query_id if forward_query_id in available_query_ids else None


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


def _canonical_protein_id(protein_id: str) -> str:
    value = str(protein_id).strip()
    for prefix in ("prot_", "uprot_", "nr90_"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def _load_typed_negative_map(path: str | Path) -> dict[str, dict[str, list[str]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Typed-negative pool must use schema_version=1")
    pools = payload.get("reaction_to_negatives")
    if not isinstance(pools, dict):
        raise ValueError("Typed-negative pool must contain reaction_to_negatives")
    result: dict[str, dict[str, list[str]]] = {}
    for query_id, pool in pools.items():
        if not isinstance(pool, dict):
            raise ValueError(f"Negative pool for {query_id!r} must be an object")
        biological = pool.get("biological", [])
        random_negatives = pool.get("random", [])
        if not isinstance(biological, list) or not isinstance(random_negatives, list):
            raise ValueError(f"Negative categories for {query_id!r} must be lists")
        biological = [str(value).strip() for value in biological if str(value).strip()]
        random_negatives = [str(value).strip() for value in random_negatives if str(value).strip()]
        if len(biological) != len(set(biological)):
            raise ValueError(f"Biological negatives for {query_id!r} contain duplicates")
        if len(random_negatives) != len(set(random_negatives)):
            raise ValueError(f"Random negatives for {query_id!r} contain duplicates")
        if set(biological) & set(random_negatives):
            raise ValueError(f"Negative categories for {query_id!r} overlap")
        if biological or random_negatives:
            result[str(query_id)] = {
                "biological": biological,
                "random": random_negatives,
            }
    return result


class TypedNegativeBatchSampler(BatchSampler):
    """Pair every positive row with one typed negative on every epoch."""

    def __init__(
        self,
        dataset: TupleDataset,
        *,
        batch_size: int,
        positive_fraction: float = 0.5,
        biological_negative_fraction: float = 0.5,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0 or batch_size % 2:
            raise ValueError("50/50 typed-negative sampling requires a positive even batch_size")
        if positive_fraction != 0.5:
            raise ValueError("Typed-negative positive_fraction must be exactly 0.5")
        if not 0.0 <= biological_negative_fraction <= 1.0:
            raise ValueError("biological_negative_fraction must be in [0, 1]")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.positive_fraction = float(positive_fraction)
        self.biological_negative_fraction = float(biological_negative_fraction)
        self.positives_per_batch = self.batch_size // 2
        self.negatives_per_batch = self.batch_size - self.positives_per_batch
        self.seed = int(seed)
        self.epoch = 0
        self.rows: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        self.positive_rows: list[int] = []
        self.positive_row_queries: dict[int, str] = {}
        valid_types = {"positive", "biological_negative", "random_negative"}
        for row_idx, pair_key in enumerate(dataset.keys):
            pair = dataset.tuple_dataset[pair_key]
            pair_type = str(pair.get("pair_type", "positive"))
            if pair_type not in valid_types:
                raise ValueError(f"Unsupported pair_type: {pair_type!r}")
            query_id = str(pair["query_id"])
            self.rows[query_id][pair_type].append(row_idx)
            if pair_type == "positive":
                self.positive_rows.append(row_idx)
                self.positive_row_queries[row_idx] = query_id
        if not self.positive_rows:
            raise ValueError("Typed-negative sampling requires positive rows")
        missing_negative_queries = sorted(
            query_id
            for query_id, typed_rows in self.rows.items()
            if typed_rows["positive"]
            and not (typed_rows["biological_negative"] or typed_rows["random_negative"])
        )
        if missing_negative_queries:
            raise ValueError(
                "Every positive query requires a typed negative; missing for "
                f"{len(missing_negative_queries)} queries"
            )

    def _rank_info(self) -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), int(dist.get_world_size())
        return 0, 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _negative_rows(
        self,
        positive_rows: list[int],
        rng: random.Random,
    ) -> list[int]:
        only_biological: list[int] = []
        flexible: list[int] = []
        for position, positive_row in enumerate(positive_rows):
            query_id = self.positive_row_queries[positive_row]
            typed_rows = self.rows[query_id]
            has_biological = bool(typed_rows["biological_negative"])
            has_random = bool(typed_rows["random_negative"])
            if has_biological and has_random:
                flexible.append(position)
            elif has_biological:
                only_biological.append(position)

        desired_biological = round(len(positive_rows) * self.biological_negative_fraction)
        desired_biological = min(
            max(desired_biological, len(only_biological)),
            len(only_biological) + len(flexible),
        )
        rng.shuffle(flexible)
        biological_positions = set(only_biological)
        biological_positions.update(flexible[: desired_biological - len(only_biological)])

        negative_rows: list[int] = []
        for position, positive_row in enumerate(positive_rows):
            query_id = self.positive_row_queries[positive_row]
            category = (
                "biological_negative" if position in biological_positions else "random_negative"
            )
            candidates = self.rows[query_id][category]
            if not candidates:
                fallback = (
                    "random_negative"
                    if category == "biological_negative"
                    else "biological_negative"
                )
                candidates = self.rows[query_id][fallback]
            negative_rows.append(rng.choice(candidates))
        return negative_rows

    @staticmethod
    def _pad(values: list[int], target_size: int) -> list[int]:
        padded = list(values)
        while len(padded) < target_size:
            missing = target_size - len(padded)
            padded.extend(values[:missing])
        return padded

    def __iter__(self) -> Iterator[list[int]]:
        rank, world_size = self._rank_info()
        rng = random.Random(self.seed + self.epoch)
        positive_order = list(self.positive_rows)
        rng.shuffle(positive_order)
        global_positives_per_batch = self.positives_per_batch * world_size
        total_positives = len(self) * global_positives_per_batch
        positive_order = self._pad(positive_order, total_positives)
        for start in range(0, total_positives, global_positives_per_batch):
            global_positives = positive_order[start : start + global_positives_per_batch]
            global_negatives = self._negative_rows(global_positives, rng)
            rank_start = rank * self.positives_per_batch
            rank_end = rank_start + self.positives_per_batch
            batch = global_positives[rank_start:rank_end] + global_negatives[rank_start:rank_end]
            rng.shuffle(batch)
            yield batch
        self.epoch += 1

    def __len__(self) -> int:
        _rank, world_size = self._rank_info()
        global_positives_per_batch = self.positives_per_batch * world_size
        return math.ceil(len(self.positive_rows) / global_positives_per_batch)


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


class EnzymeGroupedBatchSampler(BatchSampler):
    """Create DDP-global batches containing several positives per enzyme.

    The grouped rows expose multi-reaction supervision to a prototype head.
    Remaining capacity is filled with ordinary association rows, retaining
    singleton enzymes and broad reaction coverage in every epoch.
    """

    def __init__(
        self,
        dataset: TupleDataset,
        *,
        batch_size: int,
        anchors_per_batch: int = 64,
        positives_per_anchor: int = 4,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if anchors_per_batch <= 0:
            raise ValueError("anchors_per_batch must be positive")
        if positives_per_anchor < 2:
            raise ValueError("positives_per_anchor must be at least 2")
        if anchors_per_batch * positives_per_anchor > batch_size:
            raise ValueError(
                "anchors_per_batch * positives_per_anchor must not exceed batch_size"
            )
        if len(dataset) == 0:
            raise ValueError("Enzyme-grouped sampling requires a non-empty dataset")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.anchors_per_batch = int(anchors_per_batch)
        self.positives_per_anchor = int(positives_per_anchor)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.target_to_indices: dict[str, list[int]] = defaultdict(list)
        for row_idx, pair_key in enumerate(dataset.keys):
            pair = dataset.tuple_dataset[pair_key]
            self.target_to_indices[str(pair["target_id"])].append(row_idx)
        self.multi_reaction_targets = sorted(
            target_id
            for target_id, indices in self.target_to_indices.items()
            if len(indices) >= 2
        )
        if not self.multi_reaction_targets:
            raise ValueError("Enzyme-grouped sampling found no multi-reaction enzymes")
        self.all_indices = list(range(len(dataset)))

    def _rank_info(self) -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), int(dist.get_world_size())
        return 0, 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _take_cyclic(
        values: list[Any],
        count: int,
        cursor: int,
        rng: random.Random,
    ) -> tuple[list[Any], int]:
        selected: list[Any] = []
        while len(selected) < count:
            if cursor >= len(values):
                rng.shuffle(values)
                cursor = 0
            take = min(count - len(selected), len(values) - cursor)
            selected.extend(values[cursor : cursor + take])
            cursor += take
        return selected, cursor

    def __iter__(self):
        rank, world_size = self._rank_info()
        rng = random.Random(self.seed + self.epoch)
        anchor_order = list(self.multi_reaction_targets)
        filler_order = list(self.all_indices)
        rng.shuffle(anchor_order)
        rng.shuffle(filler_order)
        anchor_cursor = 0
        filler_cursor = 0
        global_batch_size = self.batch_size * world_size
        global_anchor_count = self.anchors_per_batch * world_size

        for _step in range(len(self)):
            anchors, anchor_cursor = self._take_cyclic(
                anchor_order,
                global_anchor_count,
                anchor_cursor,
                rng,
            )
            global_rows: list[int] = []
            seen: set[int] = set()
            for target_id in anchors:
                candidates = list(self.target_to_indices[target_id])
                rng.shuffle(candidates)
                for row_idx in candidates[: self.positives_per_anchor]:
                    if row_idx in seen:
                        continue
                    global_rows.append(row_idx)
                    seen.add(row_idx)

            while len(global_rows) < global_batch_size:
                fillers, filler_cursor = self._take_cyclic(
                    filler_order,
                    1,
                    filler_cursor,
                    rng,
                )
                row_idx = int(fillers[0])
                if row_idx in seen:
                    continue
                global_rows.append(row_idx)
                seen.add(row_idx)
            rng.shuffle(global_rows)
            start = rank * self.batch_size
            batch = global_rows[start : start + self.batch_size]
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


class HypergraphBatchSampler(BatchSampler):
    """Mix reaction- and enzyme-centred hyperedges inside every local DDP batch.

    Rows belonging to one selected hyperedge stay on the same rank.  This is
    important for the biological residual objective, which intentionally uses
    local (memory-bounded) late-interaction negatives while CIRCE can continue
    to use the existing cross-rank global objective.
    """

    def __init__(
        self,
        dataset: TupleDataset,
        *,
        batch_size: int,
        anchors_per_batch: int = 16,
        positives_per_anchor: int = 4,
        reaction_anchor_fraction: float = 0.5,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0 or anchors_per_batch <= 0:
            raise ValueError("batch_size and anchors_per_batch must be positive")
        if positives_per_anchor < 2:
            raise ValueError("positives_per_anchor must be at least two")
        if anchors_per_batch * positives_per_anchor > batch_size:
            raise ValueError(
                "anchors_per_batch * positives_per_anchor must not exceed batch_size"
            )
        if not 0.0 <= reaction_anchor_fraction <= 1.0:
            raise ValueError("reaction_anchor_fraction must lie in [0, 1]")
        if len(dataset) == 0:
            raise ValueError("Hypergraph sampling requires a non-empty dataset")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.anchors_per_batch = int(anchors_per_batch)
        self.positives_per_anchor = int(positives_per_anchor)
        self.reaction_anchor_fraction = float(reaction_anchor_fraction)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if isinstance(dataset, ReplayConcatDataset):
            primary_count = len(dataset.primary)
            self.primary_indices = list(range(primary_count))
            self.replay_indices = list(range(primary_count, len(dataset)))
            self.replay_rows_per_batch = min(
                len(self.replay_indices),
                int(round(self.batch_size * dataset.realized_replay_fraction)),
            )
        else:
            self.primary_indices = list(range(len(dataset)))
            self.replay_indices = []
            self.replay_rows_per_batch = 0
        primary_rows_per_batch = self.batch_size - self.replay_rows_per_batch
        if anchors_per_batch * positives_per_anchor > primary_rows_per_batch:
            raise ValueError(
                "Grouped hyperedge capacity exceeds the non-replay rows available "
                "per batch"
            )
        query_to_indices: dict[str, list[int]] = defaultdict(list)
        target_to_indices: dict[str, list[int]] = defaultdict(list)
        for row_idx in self.primary_indices:
            pair_key = dataset.keys[row_idx]
            pair = dataset.tuple_dataset[pair_key]
            if str(pair.get("pair_type", "positive")) != "positive":
                continue
            query_to_indices[str(pair["query_id"])].append(row_idx)
            target_to_indices[str(pair["target_id"])].append(row_idx)
        self.reaction_hyperedges = {
            key: values for key, values in query_to_indices.items() if len(values) >= 2
        }
        self.enzyme_hyperedges = {
            key: values for key, values in target_to_indices.items() if len(values) >= 2
        }
        if not self.reaction_hyperedges and not self.enzyme_hyperedges:
            raise ValueError("Hypergraph sampling found no degree>=2 reaction or enzyme")
        self.all_indices = list(range(len(dataset)))

    @staticmethod
    def _rank_info() -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), int(dist.get_world_size())
        return 0, 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _anchor_counts(self) -> tuple[int, int]:
        reaction_count = int(round(self.anchors_per_batch * self.reaction_anchor_fraction))
        enzyme_count = self.anchors_per_batch - reaction_count
        if not self.reaction_hyperedges:
            return 0, self.anchors_per_batch
        if not self.enzyme_hyperedges:
            return self.anchors_per_batch, 0
        return reaction_count, enzyme_count

    @staticmethod
    def _sample_groups(
        groups: dict[str, list[int]],
        count: int,
        positives_per_anchor: int,
        rng: random.Random,
    ) -> list[list[int]]:
        if count <= 0 or not groups:
            return []
        keys = list(groups)
        selected = rng.sample(keys, k=min(count, len(keys)))
        while len(selected) < count:
            selected.append(rng.choice(keys))
        output: list[list[int]] = []
        for key in selected:
            values = list(groups[key])
            rng.shuffle(values)
            output.append(values[:positives_per_anchor])
        return output

    def __iter__(self):
        rank, _world_size = self._rank_info()
        rng = random.Random(self.seed + 100_003 * self.epoch + 7_919 * rank)
        reaction_count, enzyme_count = self._anchor_counts()
        primary_filler_order = list(self.primary_indices)
        replay_filler_order = list(self.replay_indices)
        rng.shuffle(primary_filler_order)
        rng.shuffle(replay_filler_order)
        primary_filler_cursor = 0
        replay_filler_cursor = 0
        for _step in range(len(self)):
            grouped_rows = self._sample_groups(
                self.reaction_hyperedges,
                reaction_count,
                self.positives_per_anchor,
                rng,
            ) + self._sample_groups(
                self.enzyme_hyperedges,
                enzyme_count,
                self.positives_per_anchor,
                rng,
            )
            rng.shuffle(grouped_rows)
            batch: list[int] = []
            seen: set[int] = set()
            for group in grouped_rows:
                for row_idx in group:
                    if row_idx not in seen:
                        batch.append(row_idx)
                        seen.add(row_idx)
            primary_target = self.batch_size - self.replay_rows_per_batch
            attempts = 0
            while len(batch) < primary_target:
                if primary_filler_cursor >= len(primary_filler_order):
                    rng.shuffle(primary_filler_order)
                    primary_filler_cursor = 0
                row_idx = primary_filler_order[primary_filler_cursor]
                primary_filler_cursor += 1
                attempts += 1
                if row_idx in seen:
                    if (
                        attempts > 2 * len(primary_filler_order)
                        and len(seen) >= len(primary_filler_order)
                    ):
                        break
                    continue
                batch.append(row_idx)
                seen.add(row_idx)
            replay_added = 0
            while replay_added < self.replay_rows_per_batch:
                if replay_filler_cursor >= len(replay_filler_order):
                    rng.shuffle(replay_filler_order)
                    replay_filler_cursor = 0
                row_idx = replay_filler_order[replay_filler_cursor]
                replay_filler_cursor += 1
                if row_idx in seen:
                    continue
                batch.append(row_idx)
                seen.add(row_idx)
                replay_added += 1
            if self.drop_last and len(batch) < self.batch_size:
                continue
            rng.shuffle(batch)
            yield batch
        self.epoch += 1

    def __len__(self) -> int:
        _rank, world_size = self._rank_info()
        denominator = self.batch_size * world_size
        if self.drop_last:
            return len(self.dataset) // denominator
        return math.ceil(len(self.dataset) / denominator)


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
        cached_enzyme_base_embeds_path: str | None = None,
        cached_train_reaction_base_embeds_path: str | None = None,
        cached_validation_reaction_base_embeds_path: str | None = None,
        protein_functional_tokens_path: str | None = None,
        train_batch_size: int = 64,
        retrieval_batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        worker_num_threads: int | None = None,
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
        validation_retrieval_query_ids_path: str | None = None,
        validation_retrieval_batch_size: int | None = None,
        validation_retrieval_directions: Optional[List[str]] = None,
        hard_negative_pools_path: str | None = None,
        hard_negative_direction: str = "reaction_to_enzyme",
        hard_negative_anchor_queries_per_batch: int = 48,
        hard_negative_positives_per_query: int = 2,
        hard_negative_negatives_per_query: int = 8,
        hard_negative_seed: int = 42,
        typed_negative_pools_path: str | None = None,
        indexed_pairs_dir: str | None = None,
        typed_negative_positive_fraction: float | None = None,
        typed_negative_biological_fraction: float = 0.5,
        typed_negative_seed: int = 42,
        reaction_balanced_sampling: bool = False,
        balanced_anchor_sampling: bool = False,
        balanced_anchor_positives: int = 4,
        balanced_anchor_seed: int = 42,
        reaction_balanced_degree_exponent: float = 0.5,
        reaction_balanced_seed: int = 42,
        enzyme_grouped_sampling: bool = False,
        enzyme_grouped_anchors_per_batch: int = 64,
        enzyme_grouped_positives_per_anchor: int = 4,
        enzyme_grouped_seed: int = 42,
        hypergraph_sampling: bool = False,
        hypergraph_anchors_per_batch: int = 16,
        hypergraph_positives_per_anchor: int = 4,
        hypergraph_reaction_anchor_fraction: float = 0.5,
        hypergraph_seed: int = 42,
        replay_config: dict[str, Any] | None = None,
        replay_fraction: float = 0.0,
        replay_seed: int = 42,
        reaction_direction_mode: str = "bidirectional",
        validation_enabled: bool = True,
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
            validation_reaction_directional_vectors_path=(
                validation_reaction_directional_vectors_path
            ),
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
            validation_enabled=validation_enabled,
        )
        self.protein_residue_embeds_path = Path(protein_residue_embeds_path)
        self.protein_score_residue_embeds_path = (
            None
            if protein_score_residue_embeds_path is None
            else Path(protein_score_residue_embeds_path)
        )
        self.cached_enzyme_base_embeds_path = (
            None
            if cached_enzyme_base_embeds_path is None
            else Path(cached_enzyme_base_embeds_path)
        )
        self.cached_train_reaction_base_embeds_path = (
            None
            if cached_train_reaction_base_embeds_path is None
            else Path(cached_train_reaction_base_embeds_path)
        )
        self.cached_validation_reaction_base_embeds_path = (
            None
            if cached_validation_reaction_base_embeds_path is None
            else Path(cached_validation_reaction_base_embeds_path)
        )
        self.protein_functional_tokens_path = (
            None
            if protein_functional_tokens_path is None
            else Path(protein_functional_tokens_path)
        )
        cache_paths = (
            self.cached_enzyme_base_embeds_path,
            self.cached_train_reaction_base_embeds_path,
            self.cached_validation_reaction_base_embeds_path,
        )
        if self.protein_functional_tokens_path is not None and not any(
            path is not None for path in cache_paths
        ):
            raise ValueError(
                "protein_functional_tokens_path currently requires frozen base embeddings"
            )
        if any(path is not None for path in cache_paths):
            if self.cached_enzyme_base_embeds_path is None:
                raise ValueError(
                    "cached_enzyme_base_embeds_path is required when using base caches"
                )
            if self.cached_train_reaction_base_embeds_path is None:
                raise ValueError(
                    "cached_train_reaction_base_embeds_path is required when using base caches"
                )
            if (
                self.validation_enabled
                and self.cached_validation_reaction_base_embeds_path is None
            ):
                raise ValueError(
                    "cached_validation_reaction_base_embeds_path is required when "
                    "validation is enabled with base caches"
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
        self.validation_retrieval_query_ids_path = (
            None if validation_retrieval_query_ids_path is None
            else Path(validation_retrieval_query_ids_path)
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
        self.typed_negative_pools_path = (
            None if typed_negative_pools_path is None else Path(typed_negative_pools_path)
        )
        self.indexed_pairs_dir = None if indexed_pairs_dir is None else Path(indexed_pairs_dir)
        if not isinstance(persistent_workers, bool):
            raise ValueError("persistent_workers must be boolean")
        for name, value in (("prefetch_factor", prefetch_factor), ("worker_num_threads", worker_num_threads)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.worker_num_threads = worker_num_threads
        self.indexed_training_pairs = None
        self.typed_negative_positive_fraction = float(
            (0.85 if self.indexed_pairs_dir is not None else 0.5)
            if typed_negative_positive_fraction is None else typed_negative_positive_fraction
        )
        self.typed_negative_biological_fraction = float(typed_negative_biological_fraction)
        self.typed_negative_seed = int(typed_negative_seed)
        self.reaction_balanced_sampling = bool(reaction_balanced_sampling)
        self.balanced_anchor_sampling = bool(balanced_anchor_sampling)
        self.balanced_anchor_positives = int(balanced_anchor_positives)
        self.balanced_anchor_seed = int(balanced_anchor_seed)
        if self.balanced_anchor_sampling and replay_config is not None:
            raise ValueError("Balanced anchors currently require a single training dataset")
        self.reaction_balanced_degree_exponent = float(reaction_balanced_degree_exponent)
        self.reaction_balanced_seed = int(reaction_balanced_seed)
        self.enzyme_grouped_sampling = bool(enzyme_grouped_sampling)
        self.enzyme_grouped_anchors_per_batch = int(enzyme_grouped_anchors_per_batch)
        self.enzyme_grouped_positives_per_anchor = int(enzyme_grouped_positives_per_anchor)
        self.enzyme_grouped_seed = int(enzyme_grouped_seed)
        self.hypergraph_sampling = bool(hypergraph_sampling)
        self.hypergraph_anchors_per_batch = int(hypergraph_anchors_per_batch)
        self.hypergraph_positives_per_anchor = int(hypergraph_positives_per_anchor)
        self.hypergraph_reaction_anchor_fraction = float(
            hypergraph_reaction_anchor_fraction
        )
        self.hypergraph_seed = int(hypergraph_seed)
        self.replay_config = None if replay_config is None else dict(replay_config)
        self.replay_fraction = float(replay_fraction)
        self.replay_seed = int(replay_seed)
        self._replay_data_module: ReactionConditionedDataModule | None = None
        if self.replay_config is None and self.replay_fraction != 0.0:
            raise ValueError("replay_fraction requires replay_config")
        if self.replay_config is not None and not 0.0 < self.replay_fraction < 1.0:
            raise ValueError("replay_fraction must lie in (0, 1) when replay is enabled")
        if self.replay_config is not None and (
            self.typed_negative_pools_path is not None
            or self.hard_negative_pools_path is not None
            or self.indexed_pairs_dir is not None
        ):
            raise ValueError("Source replay is incompatible with typed/hard-negative samplers")
        if self.reaction_balanced_degree_exponent < 0.0:
            raise ValueError("reaction_balanced_degree_exponent must be non-negative")
        enabled_samplers = sum(
            (
                self.reaction_balanced_sampling,
                self.balanced_anchor_sampling,
                self.enzyme_grouped_sampling,
                self.hypergraph_sampling,
                self.hard_negative_pools_path is not None,
                self.typed_negative_pools_path is not None,
                self.indexed_pairs_dir is not None,
            )
        )
        if enabled_samplers > 1:
            raise ValueError(
                "Reaction-balanced, enzyme-grouped, legacy hard-negative, and "
                "typed-negative samplers "
                "are mutually exclusive"
            )
        self._val_metric_target_lookup_data = None
        self._val_metric_query_lookup_data = None
        self._val_metric_target_query_data = None
        self._val_metric_query_data = None
        self._target_to_queries = None
        self._val_retrieval_target_candidate_ids = None
        self._val_retrieval_query_candidate_ids = None

    @property
    def uses_cached_base_embeddings(self) -> bool:
        return self.cached_enzyme_base_embeds_path is not None

    @property
    def _base_collate_fn(self):
        return dict_collate_fn if self.uses_cached_base_embeddings else residue_collate_fn

    def _setup_training_data(self) -> None:
        self._train_typed_negative_targets: dict[str, dict[str, frozenset[str]]] = {}
        if self.indexed_pairs_dir is not None:
            if self.reaction_direction_mode != "forward_only":
                raise ValueError("Indexed pair sampling requires reaction_direction_mode=forward_only")
            self.indexed_training_pairs = IndexedPairs(self.indexed_pairs_dir)
            self._train_query_data = self._create_query_dataset(self.train_reactions_path, split_name="train")
            self._target_data = self._maybe_attach_target_side_vectors(self._create_target_dataset())
            self._train_data = IndexedPairDataset(self.indexed_training_pairs, self._train_query_data, self._target_data)
            self._train_query_to_targets = self.indexed_training_pairs.query_to_targets
            self._train_target_to_queries = self.indexed_training_pairs.target_to_queries
            # EC/cofactor evidence is used by the proposal policy, not soft
            # positive fabrication or an additional dense classification loss.
            self._train_enzyme_ec_sets = {}
            self._train_reaction_ec_sets = {}
            print(f"Indexed training graph ready: {len(self._train_data)} positive edges; typed negatives sampled on demand")
        else:
            super()._setup_training_data()
        if self.uses_cached_base_embeddings:
            query_dim = getattr(self._train_query_data, "vec_dim", None)
            target_dim = getattr(self._target_data, "vec_dim", None)
            if query_dim is None or target_dim is None or int(query_dim) != int(target_dim):
                raise ValueError(
                    "Cached reaction and enzyme base embeddings must have the same "
                    f"dimension, got reaction={query_dim}, enzyme={target_dim}"
                )
            print(f"  Using cached frozen base embeddings: dim={query_dim}")
        if self.replay_config is not None:
            replay_kwargs = dict(self.replay_config)
            replay_kwargs.setdefault("test_pairs_path", replay_kwargs["train_pairs_path"])
            replay_kwargs.setdefault("test_reactions_path", replay_kwargs["train_reactions_path"])
            replay_kwargs["validation_enabled"] = False
            replay_kwargs["replay_config"] = None
            replay_kwargs["replay_fraction"] = 0.0
            self._replay_data_module = _construct_replay_data_module(replay_kwargs)
            self._replay_data_module.setup(stage="fit")
            replay_data = self._replay_data_module.train_data
            if replay_data is None or self._train_data is None:
                raise RuntimeError("Primary or replay training data was not initialized")
            self._train_data = ReplayConcatDataset(
                self._train_data,
                replay_data,
                replay_fraction=self.replay_fraction,
                seed=self.replay_seed,
            )
            # Replay rows are subsampled for compute, but label coverage must
            # not be subsampled: every annotated source edge remains protected
            # if its endpoints happen to co-occur in a Stage-B batch.
            self._train_query_to_targets = _union_positive_lookups(
                self._train_query_to_targets,
                self._replay_data_module._train_query_to_targets,
            )
            self._train_target_to_queries = _union_positive_lookups(
                self._train_target_to_queries,
                self._replay_data_module._train_target_to_queries,
            )
            print(
                "  Source replay: "
                f"requested={self.replay_fraction:.1%}, "
                f"realized={self._train_data.realized_replay_fraction:.1%}, "
                f"rows={len(self._train_data.replay_indices)}"
            )
        if self.typed_negative_pools_path is None:
            return
        if self.reaction_direction_mode != "forward_only":
            raise ValueError("Typed negative sampling currently requires forward_only reactions")

        pools = _load_typed_negative_map(self.typed_negative_pools_path)
        positive_pairs = self._train_data.tuple_dataset
        available_pair_keys = list(self._train_data.keys)
        target_ids = set(self._target_data.keys)
        target_by_canonical: dict[str, list[str]] = defaultdict(list)
        for target_id in target_ids:
            target_by_canonical[_canonical_protein_id(target_id)].append(target_id)

        def resolve_target(pool_target_id: str) -> str | None:
            if pool_target_id in target_ids:
                return pool_target_id
            matches = target_by_canonical.get(_canonical_protein_id(pool_target_id), [])
            return matches[0] if len(matches) == 1 else None

        keys: list[str] = []
        rows: list[dict[str, str]] = []
        for pair_key in available_pair_keys:
            pair = positive_pairs[pair_key]
            keys.append(f"positive:{pair_key}")
            rows.append(
                {
                    "query_id": str(pair["query_id"]),
                    "target_id": str(pair["target_id"]),
                    "pair_type": "positive",
                }
            )

        positive_map = {
            query_id: set(targets) for query_id, targets in self._train_query_to_targets.items()
        }
        added = defaultdict(int)
        resolved_pools: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: {"biological": set(), "random": set()}
        )
        skipped_missing = 0
        available_query_ids = set(positive_map)
        skipped_queries = 0
        for pool_query_id, typed_pool in pools.items():
            query_id = _resolve_typed_pool_query_id(
                pool_query_id,
                available_query_ids,
            )
            if query_id is None:
                skipped_queries += 1
                continue
            for pool_name, pair_type in (
                ("biological", "biological_negative"),
                ("random", "random_negative"),
            ):
                for pool_target_id in typed_pool[pool_name]:
                    target_id = resolve_target(pool_target_id)
                    if target_id is None:
                        skipped_missing += 1
                        continue
                    if target_id in positive_map[query_id]:
                        raise ValueError(
                            "Typed-negative pool contains a known positive: "
                            f"query={query_id}, target={target_id}"
                        )
                    other_pool = "random" if pool_name == "biological" else "biological"
                    if target_id in resolved_pools[query_id][other_pool]:
                        raise ValueError(
                            "Typed-negative categories overlap after resolving protein IDs: "
                            f"query={query_id}, target={target_id}"
                        )
                    if target_id in resolved_pools[query_id][pool_name]:
                        continue
                    resolved_pools[query_id][pool_name].add(target_id)
                    key = f"{pair_type}:{query_id}:{target_id}"
                    keys.append(key)
                    rows.append(
                        {
                            "query_id": query_id,
                            "target_id": target_id,
                            "pair_type": pair_type,
                        }
                    )
                    added[pair_type] += 1

        self._train_typed_negative_targets = {
            query_id: {pool_name: frozenset(targets) for pool_name, targets in typed_pool.items()}
            for query_id, typed_pool in resolved_pools.items()
        }

        typed_pairs = BaseDataset(keys=keys, array_data=rows)
        self._train_data = TupleDataset(
            tuple_dataset=typed_pairs,
            key_name_to_dataset={
                "query_id": self._train_query_data,
                "target_id": self._target_data,
            },
            rename_map={"query_id": "query_vec", "target_id": "target_vec"},
            skip_missing=False,
        )
        print(
            "  Typed-negative rows: "
            f"biological={added['biological_negative']}, "
            f"random={added['random_negative']}, missing_targets={skipped_missing}, "
            f"unmatched_queries={skipped_queries}"
        )

    def _create_query_dataset(
        self,
        reactions_path: Path,
        split_name: str | None = None,
    ):
        if self.uses_cached_base_embeddings:
            if self._is_train_split(split_name):
                cache_path = self.cached_train_reaction_base_embeds_path
            elif self._is_validation_split(split_name):
                cache_path = self.cached_validation_reaction_base_embeds_path
            else:
                cache_path = self.cached_train_reaction_base_embeds_path
            if cache_path is None:
                raise ValueError(f"No cached reaction base embeddings for split={split_name}")
            base_dataset = NamedEmbeddingDataset(
                cache_path,
                field_name="cached_query_embedding",
            )
            if self.protein_functional_tokens_path is None:
                return base_dataset
            if self.reaction_representation != "multimodal_reaction_attention":
                raise ValueError(
                    "Biological residual caches require multimodal_reaction_attention data"
                )
            unimol2_path = self._select_reaction_unimol2_path(split_name)
            if unimol2_path is None:
                raise ValueError("Biological residual requires UniMol2 molecule embeddings")
            chirality_path = self._select_reaction_chirality_path(split_name)
            biological_dataset = build_reaction_feature_dataset(
                reactions_path=reactions_path,
                config=DotDict(
                    {
                        "data": {
                            "reaction_representation": "multimodal_reaction_attention",
                            "reaction_use_model": False,
                            "reaction_unimol2_embeds_path": str(unimol2_path),
                            "reaction_unimol_dim": self.reaction_unimol_dim,
                            "reaction_use_chiro": chirality_path is not None,
                            "reaction_use_chirality": chirality_path is not None,
                            "reaction_use_chienn": chirality_path is not None,
                            "reaction_chiro_embeds_path": (
                                None if chirality_path is None else str(chirality_path)
                            ),
                            "reaction_chirality_embeds_path": (
                                None if chirality_path is None else str(chirality_path)
                            ),
                            "reaction_chienn_embeds_path": (
                                None if chirality_path is None else str(chirality_path)
                            ),
                            "reaction_chiro_dim": self.reaction_chienn_dim,
                            "reaction_chirality_dim": self.reaction_chienn_dim,
                            "reaction_chienn_dim": self.reaction_chienn_dim,
                            "reaction_allow_missing_unimol2": self.reaction_allow_missing_unimol2,
                            "reaction_allow_missing_chiro": self.reaction_allow_missing_chienn,
                            "reaction_allow_missing_chirality": self.reaction_allow_missing_chienn,
                            "reaction_allow_missing_chienn": self.reaction_allow_missing_chienn,
                            "reaction_embedding_in_memory": False,
                        }
                    }
                ),
                bidirectional=self.indexed_pairs_dir is None,
                transforms=self._normalize_reaction_smiles,
                split_name=split_name,
            )
            merged = MergeDataset(
                {"base": base_dataset, "biological": biological_dataset},
                add_prefix=False,
            )
            merged.vec_dim = base_dataset.vec_dim
            return merged
        if self.indexed_pairs_dir is not None:
            # New extraction artifacts use the original forward IDs. Avoid the
            # legacy feature constructor's unconditional _f/_r augmentation.
            names = (
                "reaction_representation", "reaction_model_dim", "reaction_unimol_dim",
                "reaction_chienn_dim", "reaction_chemistry_dim", "reaction_directional_dim",
                "reaction_use_model", "reaction_use_chienn", "reaction_use_chemistry",
                "reaction_use_directional", "reaction_load_directional",
                "reaction_allow_missing_unimol2", "reaction_allow_missing_chienn",
                "reaction_allow_missing_chemistry", "reaction_allow_missing_directional",
                "reaction_embedding_in_memory", "rdkit_fp_dim", "drfp_dim",
            )
            data = {name: getattr(self, name) for name in names}
            paths = {
                "reaction_embeds_path": self._select_reaction_embeds_path(split_name),
                "reaction_t5v2_embeds_path": self._select_reaction_model_path(split_name),
                "reaction_unimol2_embeds_path": self._select_reaction_unimol2_path(split_name),
                "reaction_chiro_embeds_path": self._select_reaction_chirality_path(split_name),
                "reaction_chemistry_vectors_path": self._select_reaction_chemistry_path(split_name),
                "reaction_directional_vectors_path": self._select_reaction_directional_path(split_name),
            }
            data.update({key: None if value is None else str(value) for key, value in paths.items()})
            data.update(reaction_use_chiro=self.reaction_use_chienn,
                        reaction_chiro_dim=self.reaction_chienn_dim,
                        reaction_allow_missing_chiro=self.reaction_allow_missing_chienn)
            return build_reaction_feature_dataset(
                reactions_path=reactions_path, config=DotDict({"data": data}),
                bidirectional=False, transforms=self._normalize_reaction_smiles,
                split_name=split_name,
            )
        return super()._create_query_dataset(reactions_path, split_name=split_name)

    def _augment_pairs_for_direction_mode(self, pairs):
        if self.indexed_pairs_dir is not None:
            return pairs
        return super()._augment_pairs_for_direction_mode(pairs)

    def _create_target_dataset(self):
        if self.uses_cached_base_embeddings:
            if self.cached_enzyme_base_embeds_path is None:
                raise RuntimeError("Cached enzyme embedding path was not initialized")
            base_dataset = NamedEmbeddingDataset(
                self.cached_enzyme_base_embeds_path,
                field_name="cached_target_embedding",
            )
            if self.protein_functional_tokens_path is None:
                return base_dataset
            functional_dataset = FunctionalTokenH5Dataset(
                self.protein_functional_tokens_path
            )
            merged = MergeDataset(
                {"base": base_dataset, "functional": functional_dataset},
                add_prefix=False,
            )
            merged.vec_dim = base_dataset.vec_dim
            return merged
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
        if self.indexed_pairs_dir is not None:
            with self.test_reactions_path.open(newline="", encoding="utf-8") as handle:
                expected_queries = [row["reaction_id"].strip() for row in csv.DictReader(handle)]
            available_queries = set(self._val_query_data_raw.keys)
            if len(set(expected_queries)) != len(expected_queries):
                raise ValueError("Indexed validation reaction candidate table contains duplicate IDs")
            if set(expected_queries) != available_queries:
                missing = sorted(set(expected_queries) - available_queries)
                raise ValueError(f"Indexed validation candidate reactions lack features: {missing[:10]}")
            if any(val_pairs[key]["query_id"] not in available_queries for key in val_pairs.keys):
                raise ValueError("Indexed validation gold references an undeclared/missing reaction")
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

        val_protein_ids = set(val_pairs[k]["target_id"] for k in val_pairs.keys)
        screening_protein_ids = set(self._screening_target_data.keys)
        if self.indexed_pairs_dir is not None and not val_protein_ids.issubset(screening_protein_ids):
            raise ValueError("Indexed validation gold proteins lack features")
        if self.indexed_training_pairs is not None:
            train_count = len(self.indexed_training_pairs.protein_ids)
            overlap_count = sum(pid in self._train_target_to_queries for pid in val_protein_ids)
        else:
            train_pairs_for_stats = CSVDataset(
                file_path=str(self.train_pairs_path),
                key_column="pr_id",
                columns=["reaction_id", "protein_id"],
                rename_map={"reaction_id": "query_id", "protein_id": "target_id"},
            )
            train_protein_ids = set(train_pairs_for_stats[k]["target_id"] for k in train_pairs_for_stats.keys)
            train_count = len(train_protein_ids)
            overlap_count = len(train_protein_ids & val_protein_ids)

        print(f"  Training proteins: {train_count}")
        print(f"  Validation proteins: {len(val_protein_ids)}")
        print(f"  Screening set (full): {len(screening_protein_ids)} proteins")
        print(f"  Overlap (train ∩ val): {overlap_count}")
        print(f"  Val-only proteins: {len(val_protein_ids) - overlap_count}")

        self._val_data = TupleDataset(
            tuple_dataset=val_pairs,
            key_name_to_dataset={
                "query_id": self._val_query_data_raw,
                "target_id": self._target_data,
            },
            rename_map={"query_id": "query_vec"},
            skip_missing=self.indexed_pairs_dir is None,
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
                    if self.indexed_pairs_dir is not None:
                        raise ValueError("Indexed validation candidate proteins lack features")
                    print(
                        "  Filtered "
                        f"{missing_candidate_ids} custom validation candidates missing "
                        "from the screening target dataset"
                    )
            else:
                target_candidate_ids = target_query_ids
            query_candidate_ids = unique_query_ids
            query_candidate_dataset = self._val_query_data
            if self.indexed_pairs_dir is not None:
                # Incidental panel gold is not the candidate universe: enzyme
                # queries retrieve against ALL declared validation reactions.
                query_candidate_ids = list(self._val_query_data_raw.keys)
                query_candidate_dataset = TupleDataset(
                    tuple_dataset=BaseDataset(
                        keys=query_candidate_ids,
                        array_data=[{"query_id": qid} for qid in query_candidate_ids],
                    ),
                    key_name_to_dataset={"query_id": self._val_query_data_raw},
                    rename_map={"query_id": "query_vec"}, skip_missing=False,
                )
            metric_query_ids = unique_query_ids
            if self.validation_retrieval_query_ids_path is not None:
                metric_query_ids, target_query_ids = _directional_validation_queries(
                    self.validation_retrieval_query_ids_path, unique_query_ids, target_query_ids,
                )
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
                query_candidate_dataset,
                index_key="query_lookup_row_idx",
                id_key="query_id",
            )
            self._val_metric_query_data = IndexedMappingDataset(
                KeySubsetDataset(self._val_query_data, metric_query_ids),
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

    def _loader_worker_kwargs(self, *, training: bool = False) -> dict[str, Any]:
        kwargs = {"num_workers": self.num_workers, "pin_memory": self.pin_memory}
        if self.num_workers > 0:
            # Validation has up to five loaders; retaining all their worker
            # pools would multiply host RAM and processes between validations.
            kwargs["persistent_workers"] = self.persistent_workers and training
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.prefetch_factor
            if self.worker_num_threads is not None:
                kwargs["worker_init_fn"] = partial(_initialize_loader_worker, num_threads=self.worker_num_threads)
        return kwargs

    def train_dataloader(self) -> DataLoader:
        if self._train_data is None:
            raise RuntimeError("Training data not setup. Call setup() first.")

        if self.balanced_anchor_sampling:
            sampler = BalancedAnchorBatchSampler(
                self._train_data, batch_size=self.train_batch_size,
                positives_per_anchor=self.balanced_anchor_positives,
                seed=self.balanced_anchor_seed)
            print(f"Uniform reaction/enzyme anchors including singletons: "
                  f"{len(sampler.reactions)} reactions, {len(sampler.enzymes)} enzymes")
            return DataLoader(self._train_data, batch_sampler=sampler,
                              **self._loader_worker_kwargs(training=True),
                              collate_fn=self._base_collate_fn)

        if self.indexed_pairs_dir is not None:
            sampler = IndexedTypedNegativeBatchSampler(
                self.indexed_training_pairs, batch_size=self.train_batch_size,
                positive_fraction=self.typed_negative_positive_fraction,
                biological_negative_fraction=self.typed_negative_biological_fraction,
                seed=self.typed_negative_seed,
            )
            return DataLoader(self._train_data, batch_sampler=sampler,
                              **self._loader_worker_kwargs(training=True),
                              collate_fn=self._base_collate_fn)

        if self.typed_negative_pools_path is not None:
            sampler = TypedNegativeBatchSampler(
                self._train_data,
                batch_size=self.train_batch_size,
                positive_fraction=self.typed_negative_positive_fraction,
                biological_negative_fraction=self.typed_negative_biological_fraction,
                seed=self.typed_negative_seed,
            )
            print(
                "Using typed-negative batch sampler: "
                f"positive_fraction={sampler.positive_fraction:.2f}, "
                f"negative_fraction={1.0 - sampler.positive_fraction:.2f}, "
                "negative_mix=(biological/random), "
                f"positive_rows/epoch={len(sampler.positive_rows)}"
            )
            return DataLoader(
                self._train_data,
                batch_sampler=sampler,
                **self._loader_worker_kwargs(training=True),
                collate_fn=self._base_collate_fn,
            )

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
                **self._loader_worker_kwargs(training=True),
                collate_fn=self._base_collate_fn,
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
                **self._loader_worker_kwargs(training=True),
                collate_fn=self._base_collate_fn,
            )

        if self.enzyme_grouped_sampling:
            sampler = EnzymeGroupedBatchSampler(
                self._train_data,
                batch_size=self.train_batch_size,
                anchors_per_batch=self.enzyme_grouped_anchors_per_batch,
                positives_per_anchor=self.enzyme_grouped_positives_per_anchor,
                seed=self.enzyme_grouped_seed,
            )
            print(
                "Using enzyme-grouped batch sampler: "
                f"anchors/batch={sampler.anchors_per_batch}, "
                f"positives/anchor<={sampler.positives_per_anchor}, "
                f"multi-reaction enzymes={len(sampler.multi_reaction_targets)}"
            )
            return DataLoader(
                self._train_data,
                batch_sampler=sampler,
                **self._loader_worker_kwargs(training=True),
                collate_fn=self._base_collate_fn,
            )

        if self.hypergraph_sampling:
            sampler = HypergraphBatchSampler(
                self._train_data,
                batch_size=self.train_batch_size,
                anchors_per_batch=self.hypergraph_anchors_per_batch,
                positives_per_anchor=self.hypergraph_positives_per_anchor,
                reaction_anchor_fraction=self.hypergraph_reaction_anchor_fraction,
                seed=self.hypergraph_seed,
            )
            print(
                "Using mixed hypergraph batch sampler: "
                f"anchors/batch={sampler.anchors_per_batch}, "
                f"positives/anchor<={sampler.positives_per_anchor}, "
                f"reaction_fraction={sampler.reaction_anchor_fraction:.2f}, "
                f"reaction_hyperedges={len(sampler.reaction_hyperedges)}, "
                f"enzyme_hyperedges={len(sampler.enzyme_hyperedges)}"
            )
            return DataLoader(
                self._train_data,
                batch_sampler=sampler,
                **self._loader_worker_kwargs(training=True),
                collate_fn=self._base_collate_fn,
            )

        return DataLoader(
            self._train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            **self._loader_worker_kwargs(training=True),
            collate_fn=self._base_collate_fn,
        )

    def val_dataloader(self) -> List[DataLoader]:
        if not self.validation_enabled:
            return []
        if self._val_data is None:
            raise RuntimeError("Validation data not setup. Call setup() first.")

        loaders: list[DataLoader] = [
            DataLoader(
                self._val_data,
                batch_size=self.train_batch_size,
                shuffle=False,
                **self._loader_worker_kwargs(),
                collate_fn=self._base_collate_fn,
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
                    **self._loader_worker_kwargs(),
                    collate_fn=self._base_collate_fn,
                ),
                DataLoader(
                    self._val_metric_query_lookup_data,
                    batch_size=metric_batch_size,
                    shuffle=False,
                    **self._loader_worker_kwargs(),
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
                    **self._loader_worker_kwargs(),
                    collate_fn=dict_collate_fn,
                )
            )
        if "enzyme_to_reaction" in self.validation_retrieval_directions:
            loaders.append(
                DataLoader(
                    self._val_metric_target_query_data,
                    batch_size=metric_batch_size,
                    shuffle=False,
                    **self._loader_worker_kwargs(),
                    collate_fn=self._base_collate_fn,
                )
            )
        return loaders
