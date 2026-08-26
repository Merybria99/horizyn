#!/usr/bin/env python3
"""
Evaluate a residue-level protein-pooling Horizyn checkpoint.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from horizyn.config import load_config
from horizyn.capability.enzyme_capability_dataset import (
    CapabilityVectorDataset,
    FactorizedCapabilityVectorDataset,
    TargetWithCapabilityDataset,
    TargetWithFactorizedCapabilityDataset,
)
from horizyn.datasets.base import BaseDataset
from horizyn.datasets.csv import CSVDataset
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from horizyn.reaction_features import build_reaction_feature_dataset
from horizyn.utils import residue_collate_fn, unimol2_reaction_collate_fn


LEGACY_ALL_CANDIDATES = "legacy_all_candidates"
PAPER_TEST_CANDIDATES = "paper_test_candidates"
CONFIGURED_FORWARD_CANDIDATES = "configured_forward_candidates"
EVALUATION_PROTOCOLS = (
    LEGACY_ALL_CANDIDATES,
    PAPER_TEST_CANDIDATES,
    CONFIGURED_FORWARD_CANDIDATES,
)
HIT_RATE_CUTOFFS = (1, 2, 3, 4, 5, 10, 20, 100, 1000)


def deduplicate_preserving_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def append_retrieval_metrics(
    metric_results: dict[str, list[float]],
    scores: torch.Tensor,
    target_idx: torch.Tensor,
) -> None:
    """Record all metrics from one ranking of the candidate scores."""
    valid_targets = target_idx[target_idx >= 0]
    if not valid_targets.numel():
        for cutoff in HIT_RATE_CUTOFFS:
            metric_results[f"top_{cutoff}"].append(0.0)
        for key in (
            "mrr",
            "first_positive_mrr",
            "reactzyme_mrr",
            "r_precision",
            "avg_precision",
        ):
            metric_results[key].append(0.0)
        return

    sorted_indices = torch.argsort(scores, descending=True)
    relevance = (sorted_indices[:, None] == valid_targets[None, :]).any(dim=1)
    positive_positions = torch.nonzero(
        relevance,
        as_tuple=False,
    ).flatten()
    first_positive_rank = int(positive_positions[0].item()) + 1
    for cutoff in HIT_RATE_CUTOFFS:
        metric_results[f"top_{cutoff}"].append(float(first_positive_rank <= cutoff))

    positive_ranks = positive_positions.to(dtype=torch.float32) + 1.0
    first_positive_rr = torch.reciprocal(positive_ranks[0])
    reactzyme_mrr = torch.reciprocal(positive_ranks).mean()

    num_relevant = len(valid_targets)
    r_precision_value = relevance[: min(num_relevant, scores.numel())].float().sum()
    r_precision_value = r_precision_value / num_relevant

    cumulative_relevant = torch.cumsum(relevance.float(), dim=0)
    ranking_positions = torch.arange(
        1,
        scores.numel() + 1,
        device=scores.device,
        dtype=torch.float32,
    )
    avg_precision_value = (
        (cumulative_relevant / ranking_positions) * relevance.float()
    ).sum() / num_relevant

    first_rr, all_positive_mrr, r_prec, avg_prec = (
        torch.stack(
            (
                first_positive_rr,
                reactzyme_mrr,
                r_precision_value,
                avg_precision_value,
            )
        )
        .cpu()
        .tolist()
    )
    metric_results["mrr"].append(first_rr)
    metric_results["first_positive_mrr"].append(first_rr)
    metric_results["reactzyme_mrr"].append(all_positive_mrr)
    metric_results["r_precision"].append(r_prec)
    metric_results["avg_precision"].append(avg_prec)


def read_id_list(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate ID file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_candidate_keys(
    *,
    evaluation_protocol: str,
    available_keys: list[str],
    configured_candidate_ids: list[str],
    test_target_ids: list[str],
) -> tuple[list[str], dict[str, int]]:
    """Select the enzyme screening set for an explicit evaluation protocol."""
    if evaluation_protocol not in EVALUATION_PROTOCOLS:
        raise ValueError(
            f"Unknown evaluation protocol {evaluation_protocol!r}; "
            f"expected one of {EVALUATION_PROTOCOLS}"
        )

    available_keys = deduplicate_preserving_order(available_keys)
    configured_candidate_ids = deduplicate_preserving_order(configured_candidate_ids)
    test_target_ids = deduplicate_preserving_order(test_target_ids)
    available_key_set = set(available_keys)

    if evaluation_protocol == PAPER_TEST_CANDIDATES:
        missing_test_targets = [
            target_id for target_id in test_target_ids if target_id not in available_key_set
        ]
        if missing_test_targets:
            preview = ", ".join(missing_test_targets[:5])
            raise ValueError(
                "Paper-protocol evaluation requires every unique test protein to be "
                f"available, but {len(missing_test_targets)} are missing. Examples: {preview}"
            )
        selected_keys = list(test_target_ids)
        requested_count = len(test_target_ids)
        missing_count = 0
    else:
        requested_keys = configured_candidate_ids or available_keys
        selected_keys = [key for key in requested_keys if key in available_key_set]
        requested_count = len(requested_keys)
        missing_count = requested_count - len(selected_keys)
        if not selected_keys:
            raise ValueError("No configured candidate proteins are available for evaluation")

    return selected_keys, {
        "available_count": len(available_keys),
        "configured_count": len(configured_candidate_ids),
        "test_count": len(test_target_ids),
        "requested_count": requested_count,
        "selected_count": len(selected_keys),
        "missing_count": missing_count,
    }


class KeySubsetDataset:
    def __init__(self, dataset, keys: list[str]):
        self.dataset = dataset
        self.keys = list(keys)
        self.vec_dim = getattr(dataset, "vec_dim", None)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, key: str | int):
        if isinstance(key, int):
            key = self.keys[key]
        return self.dataset[key]


def maybe_attach_capability_vectors(config, target_dataset):
    capability_path = config.data.get("protein_capability_vectors_path", None)
    if not capability_path:
        return target_dataset
    capability_dataset = CapabilityVectorDataset(capability_path)
    merged = TargetWithCapabilityDataset(
        target_dataset=target_dataset,
        capability_dataset=capability_dataset,
        capability_key="capability_vec",
        mask_key="capability_mask",
        missing_policy=config.data.get("capability_missing_policy", "zero_with_mask"),
    )
    if merged.missing_count:
        print(
            "Capability vectors missing for "
            f"{merged.missing_count}/{len(target_dataset.keys)} target proteins; "
            "using zero vectors with capability_mask=False"
        )
    return merged


def maybe_attach_factorized_capability_vectors(config, target_dataset):
    capability_path = config.data.get("protein_factorized_capability_vectors_path", None)
    if not capability_path:
        return target_dataset
    capability_dataset = FactorizedCapabilityVectorDataset(capability_path)
    merged = TargetWithFactorizedCapabilityDataset(
        target_dataset=target_dataset,
        capability_dataset=capability_dataset,
        missing_policy=config.data.get(
            "factorized_capability_missing_policy",
            "zero_with_mask",
        ),
    )
    if merged.missing_count:
        print(
            "Factorized capability vectors missing for "
            f"{merged.missing_count}/{len(target_dataset.keys)} target proteins; "
            "using zero vectors with family masks=False"
        )
    return merged


def build_bidirectional_pairs(val_pairs: CSVDataset) -> BaseDataset:
    augmented_keys = []
    augmented_data = []
    for pair_key in val_pairs.keys:
        pair_data = val_pairs[pair_key]
        query_id = pair_data["query_id"]
        target_id = pair_data["target_id"]
        augmented_keys.append(f"{pair_key}_f")
        augmented_data.append({"query_id": f"{query_id}_f", "target_id": target_id})
        augmented_keys.append(f"{pair_key}_r")
        augmented_data.append({"query_id": f"{query_id}_r", "target_id": target_id})
    return BaseDataset(keys=augmented_keys, array_data=augmented_data)


def build_forward_pairs(val_pairs: CSVDataset) -> BaseDataset:
    """Use one canonical query per original reaction while retaining HDF5 `_f` keys."""
    augmented_keys = []
    augmented_data = []
    for pair_key in val_pairs.keys:
        pair_data = val_pairs[pair_key]
        augmented_keys.append(f"{pair_key}_f")
        augmented_data.append(
            {
                "query_id": f"{pair_data['query_id']}_f",
                "target_id": pair_data["target_id"],
            }
        )
    return BaseDataset(keys=augmented_keys, array_data=augmented_data)


def build_bidirectional_reactions(reactions: CSVDataset) -> BaseDataset:
    augmented_keys = []
    augmented_data = []
    for rxn_id in reactions.keys:
        rxn_data = reactions[rxn_id]
        smiles = rxn_data["reaction_smiles"]
        augmented_keys.append(f"{rxn_id}_f")
        augmented_data.append({"reaction_smiles": smiles})
        if ">>" in smiles:
            parts = smiles.split(">>")
            if len(parts) == 2:
                augmented_keys.append(f"{rxn_id}_r")
                augmented_data.append({"reaction_smiles": f"{parts[1]}>>{parts[0]}"})
    return BaseDataset(keys=augmented_keys, array_data=augmented_data)


def build_reaction_fingerprints(config):
    return build_reaction_feature_dataset(
        reactions_path=config.data.test_reactions_path,
        config=config,
        bidirectional=True,
    )


def build_query_inputs(reaction_features, query_ids: list[str], device: str):
    samples = [reaction_features[query_id] for query_id in query_ids]
    if isinstance(samples[0], dict):
        batch = unimol2_reaction_collate_fn(samples)
        return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    return torch.stack(samples).to(device, non_blocking=True)


def mean_metric_results(metric_results: dict) -> dict:
    return {
        metric_name: sum(values) / len(values) if values else 0.0
        for metric_name, values in metric_results.items()
    }


def encode_targets(
    module: ProteinPooledLitModule,
    residue_dataset,
    device: str,
    target_batch_size: int,
    store_on_device: bool = True,
    retrieval_direction: str = "reaction_to_enzyme",
) -> torch.Tensor:
    output_dim = module.model.target_encoder.output_dim
    storage_device = device if store_on_device else "cpu"
    target_embeds = torch.empty(
        len(residue_dataset),
        output_dim,
        dtype=torch.float32,
        device=storage_device,
    )

    for batch_start in tqdm(
        range(0, len(residue_dataset), target_batch_size),
        desc="Encoding target proteins",
    ):
        batch_end = min(batch_start + target_batch_size, len(residue_dataset))
        target_keys = residue_dataset.keys[batch_start:batch_end]
        samples = []
        for target_id in target_keys:
            sample = residue_dataset[target_id]
            sample["target_id"] = target_id
            samples.append(sample)

        batch = residue_collate_fn(samples)
        residues = batch["residue_embeddings"].to(device, non_blocking=True)
        mask = batch["residue_padding_mask"].to(device, non_blocking=True)
        enzyme_input_mode = getattr(module.model, "enzyme_input_mode", "")
        needs_capability = "capability" in str(enzyme_input_mode)
        capability_vectors = None
        capability_mask = None
        factorized_capability_vectors = {}
        factorized_capability_masks = {}
        if needs_capability:
            if (
                getattr(module.model, "enzyme_input_mode", "")
                == "raw_mean_sleec_hyperbolic_factorized_capability_blockwise"
            ):
                for family in ("cofactor", "center", "transition"):
                    vector_key = f"capability_{family}_vec"
                    mask_key = f"capability_{family}_mask"
                    if vector_key not in batch:
                        raise ValueError(
                            f"{vector_key} is required for "
                            f"enzyme_input_mode={enzyme_input_mode}"
                        )
                    factorized_capability_vectors[family] = batch[vector_key].to(
                        device,
                        non_blocking=True,
                    )
                    if mask_key in batch:
                        factorized_capability_masks[family] = batch[mask_key].to(
                            device,
                            non_blocking=True,
                        )
            elif "capability_vec" not in batch:
                raise ValueError(
                    f"capability_vec is required for enzyme_input_mode={enzyme_input_mode}"
                )
            else:
                capability_vectors = batch["capability_vec"].to(device, non_blocking=True)
                if "capability_mask" in batch:
                    capability_mask = batch["capability_mask"].to(device, non_blocking=True)
        encoded = module.model.encode_targets(
            residues,
            residue_padding_mask=mask,
            capability_vectors=capability_vectors,
            capability_mask=capability_mask,
            factorized_capability_vectors=factorized_capability_vectors,
            factorized_capability_masks=factorized_capability_masks,
            retrieval_direction=retrieval_direction,
        )
        target_embeds[batch_start:batch_end] = encoded.to(storage_device)

    return target_embeds


def load_target_embedding_cache(
    cache_path: str | Path,
    expected_target_keys: list[str],
    checkpoint_path: str | Path,
    config_path: str | Path,
    device: str,
    store_on_device: bool,
    retrieval_direction: str = "reaction_to_enzyme",
) -> torch.Tensor | None:
    cache_path = Path(cache_path)
    if not cache_path.is_file() or cache_path.stat().st_size == 0:
        return None
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"Ignoring unreadable target embedding cache {cache_path}: {exc}")
        return None
    if not isinstance(payload, dict):
        print(f"Ignoring target embedding cache without dict payload: {cache_path}")
        return None
    expected_checkpoint = str(Path(checkpoint_path).resolve())
    expected_config = str(Path(config_path).resolve())
    if payload.get("checkpoint") != expected_checkpoint:
        print(f"Ignoring target cache with different checkpoint: {cache_path}")
        return None
    if payload.get("config") != expected_config:
        print(f"Ignoring target cache with different config: {cache_path}")
        return None
    cached_direction = payload.get("retrieval_direction", "reaction_to_enzyme")
    if cached_direction != retrieval_direction:
        print(f"Ignoring target cache with different retrieval direction: {cache_path}")
        return None
    cached_keys = list(payload.get("target_keys") or [])
    target_embeds = payload.get("target_embeds")
    if not isinstance(target_embeds, torch.Tensor):
        print(f"Ignoring target cache without tensor target_embeds: {cache_path}")
        return None
    if target_embeds.shape[0] != len(cached_keys):
        print(f"Ignoring target cache with wrong row count: {cache_path}")
        return None

    if cached_keys == list(expected_target_keys):
        cache_description = "cached target embeddings"
    else:
        cached_key_to_idx = {key: idx for idx, key in enumerate(cached_keys)}
        if len(cached_key_to_idx) != len(cached_keys) or any(
            key not in cached_key_to_idx for key in expected_target_keys
        ):
            print(f"Ignoring target cache with incompatible target keys: {cache_path}")
            return None
        subset_indices = torch.tensor(
            [cached_key_to_idx[key] for key in expected_target_keys],
            dtype=torch.long,
        )
        target_embeds = target_embeds.index_select(0, subset_indices)
        cache_description = "target embedding subset from superset cache"

    target_embeds = target_embeds.float().contiguous()
    print(f"Loaded {cache_description}: {cache_path}")
    return target_embeds.to(device) if store_on_device else target_embeds.cpu()


def save_target_embedding_cache(
    cache_path: str | Path,
    target_embeds: torch.Tensor,
    target_keys: list[str],
    checkpoint_path: str | Path,
    config_path: str | Path,
    evaluation_protocol: str,
    retrieval_direction: str = "reaction_to_enzyme",
) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "config": str(Path(config_path).resolve()),
        "target_keys": list(target_keys),
        "target_embeds": target_embeds.detach().float().cpu(),
        "embedding_shape": list(target_embeds.shape),
        "evaluation_protocol": evaluation_protocol,
        "retrieval_direction": retrieval_direction,
    }
    temp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temp_path)
    os.replace(temp_path, cache_path)
    print(f"Saved cached target embeddings: {cache_path}")


def encode_reactions(
    module: ProteinPooledLitModule,
    reaction_features,
    query_ids: list[str],
    device: str,
    batch_size: int,
    retrieval_direction: str = "reaction_to_enzyme",
) -> torch.Tensor:
    output_dim = module.model.query_encoder.output_dim
    query_embeds = torch.empty(
        len(query_ids),
        output_dim,
        dtype=torch.float32,
        device=device,
    )
    for batch_start in tqdm(
        range(0, len(query_ids), batch_size),
        desc="Encoding reactions",
    ):
        batch_end = min(batch_start + batch_size, len(query_ids))
        batch_ids = query_ids[batch_start:batch_end]
        query_inputs = build_query_inputs(reaction_features, batch_ids, device)
        query_embeds[batch_start:batch_end] = module.model.encode_queries(
            query_inputs,
            retrieval_direction=retrieval_direction,
        )
    return query_embeds


def format_results_table(results: dict) -> str:
    lines = [
        "=" * 70,
        "PROTEIN-POOLING HORIZYN EVALUATION RESULTS",
        "=" * 70,
        "",
        f"Checkpoint: {results.get('checkpoint', 'N/A')}",
        f"Config: {results.get('config', 'N/A')}",
        f"Evaluation protocol: {results.get('evaluation_protocol', 'N/A')}",
        f"Screening set size: {results.get('num_targets', 'N/A')}",
        "",
    ]

    metric_keys = [
        *((f"Top-{cutoff} HR", f"top_{cutoff}") for cutoff in HIT_RATE_CUTOFFS),
        ("ReactZyme MRR", "reactzyme_mrr"),
        ("First-positive MRR", "first_positive_mrr"),
        ("R-precision", "r_precision"),
        ("Avg. precision", "avg_precision"),
    ]
    if results.get("direction") == "both":
        sections = [
            ("REACTION TO ENZYME", "reaction_to_enzyme"),
            ("ENZYME TO REACTION", "enzyme_to_reaction"),
        ]
    else:
        sections = [("RETRIEVAL METRICS", "")]

    for title, prefix in sections:
        prefix_with_separator = f"{prefix}/" if prefix else ""
        lines.extend(
            [
                "-" * 70,
                title,
                "-" * 70,
                f"Queries evaluated: "
                f"{results.get(prefix_with_separator + 'num_queries', 'N/A')}",
                "",
                f"{'Metric':<20} {'Value':<15} {'Value (%)':<15}",
                "-" * 50,
            ]
        )
        for display_name, key in metric_keys:
            value = results.get(prefix_with_separator + key, 0.0)
            lines.append(f"{display_name:<20} {value:.4f}         {value * 100:.1f}%")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def evaluate_checkpoint(
    checkpoint_path: str,
    config_path: str,
    device: str,
    batch_size: int,
    target_batch_size: int,
    store_targets_on_device: bool,
    direction: str = "reaction_to_enzyme",
    target_embeds_cache: str | None = None,
    target_embeds_source_cache: str | None = None,
    evaluation_protocol: str = LEGACY_ALL_CANDIDATES,
    hard_negative_output: str | None = None,
    hard_negative_top_k: int = 0,
) -> dict:
    if evaluation_protocol not in EVALUATION_PROTOCOLS:
        raise ValueError(
            f"Unknown evaluation protocol {evaluation_protocol!r}; "
            f"expected one of {EVALUATION_PROTOCOLS}"
        )
    print(f"Evaluation protocol: {evaluation_protocol}")
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    print(f"Loading checkpoint from: {checkpoint_path}")
    module = ProteinPooledLitModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    module.eval()
    module.to(device)

    print("Setting up validation data...")
    raw_val_pairs = CSVDataset(
        file_path=config.data.test_pairs_path,
        key_column="pr_id",
        columns=["reaction_id", "protein_id"],
        rename_map={"reaction_id": "query_id", "protein_id": "target_id"},
    )
    test_target_ids = [raw_val_pairs[pair_key]["target_id"] for pair_key in raw_val_pairs.keys]
    if evaluation_protocol in {
        PAPER_TEST_CANDIDATES,
        CONFIGURED_FORWARD_CANDIDATES,
    }:
        val_pairs = build_forward_pairs(raw_val_pairs)
        reaction_query_expansion = "canonical_forward_only"
    else:
        val_pairs = build_bidirectional_pairs(raw_val_pairs)
        reaction_query_expansion = "forward_and_reverse"

    query_to_targets = defaultdict(list)
    for pair_key in val_pairs.keys:
        pair = val_pairs[pair_key]
        query_to_targets[pair["query_id"]].append(pair["target_id"])

    reaction_fps = build_reaction_fingerprints(config)

    residue_dataset = ResidueEmbedDataset(
        file_path=config.data.protein_residue_embeds_path,
        in_memory=False,
        max_tokens=config.data.get("max_protein_tokens", 1024),
        truncation=config.data.get("protein_truncation", "ends_center"),
    )
    candidate_ids_path = config.data.get(
        "validation_retrieval_candidate_ids_path",
        config.data.get("candidate_ids_path", None),
    )
    candidate_ids = read_id_list(candidate_ids_path)
    candidate_keys, candidate_stats = select_candidate_keys(
        evaluation_protocol=evaluation_protocol,
        available_keys=list(residue_dataset.keys),
        configured_candidate_ids=candidate_ids,
        test_target_ids=test_target_ids,
    )
    residue_dataset = KeySubsetDataset(residue_dataset, candidate_keys)
    print(
        "Candidate filter: "
        f"configured={candidate_stats['configured_count']}, "
        f"test={candidate_stats['test_count']}, "
        f"requested={candidate_stats['requested_count']}, "
        f"usable={candidate_stats['selected_count']}, "
        f"missing={candidate_stats['missing_count']}"
    )
    residue_dataset = maybe_attach_capability_vectors(config, residue_dataset)
    residue_dataset = maybe_attach_factorized_capability_vectors(config, residue_dataset)
    target_id_to_idx = {target_id: idx for idx, target_id in enumerate(residue_dataset.keys)}

    available_query_ids = set(reaction_fps.keys)
    filtered_query_to_targets = defaultdict(list)
    missing_query_pairs = 0
    missing_target_pairs = 0
    for query_id, target_ids in query_to_targets.items():
        if query_id not in available_query_ids:
            missing_query_pairs += len(target_ids)
            continue
        for target_id in target_ids:
            if target_id in target_id_to_idx:
                filtered_query_to_targets[query_id].append(target_id)
            else:
                missing_target_pairs += 1

    query_to_targets = filtered_query_to_targets
    unique_query_ids = sorted(query_to_targets.keys())
    target_to_queries = defaultdict(list)
    for query_id, target_ids in query_to_targets.items():
        for target_id in target_ids:
            target_to_queries[target_id].append(query_id)

    if missing_query_pairs or missing_target_pairs:
        print(
            "Filtered validation pairs unavailable in evaluation data: "
            f"missing_queries={missing_query_pairs}, "
            f"missing_targets={missing_target_pairs}"
        )
    if evaluation_protocol == PAPER_TEST_CANDIDATES and missing_query_pairs:
        raise ValueError(
            "Paper-protocol evaluation requires every test reaction query to have "
            f"features, but {missing_query_pairs} test pairs reference unavailable queries"
        )

    print(f"Screening set size: {len(residue_dataset)} proteins")
    print(f"Evaluating {len(unique_query_ids)} validation queries")

    if hard_negative_top_k < 0:
        raise ValueError("hard_negative_top_k must be non-negative")
    if hard_negative_output and direction == "both":
        raise ValueError("Hard-negative export requires one explicit retrieval direction")
    r2e_metric_results = defaultdict(list)
    e2r_metric_results = defaultdict(list)
    anchor_to_hard_negatives: dict[str, list[str]] = {}
    with torch.inference_mode():
        primary_target_direction = (
            "reaction_to_enzyme"
            if direction in {"reaction_to_enzyme", "both"}
            else "enzyme_to_reaction"
        )
        target_embeds = None
        if target_embeds_cache:
            target_embeds = load_target_embedding_cache(
                cache_path=target_embeds_cache,
                expected_target_keys=list(residue_dataset.keys),
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                device=device,
                store_on_device=store_targets_on_device,
                retrieval_direction=primary_target_direction,
            )
        loaded_from_source_cache = False
        if target_embeds is None and target_embeds_source_cache:
            target_embeds = load_target_embedding_cache(
                cache_path=target_embeds_source_cache,
                expected_target_keys=list(residue_dataset.keys),
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                device=device,
                store_on_device=store_targets_on_device,
                retrieval_direction=primary_target_direction,
            )
            loaded_from_source_cache = target_embeds is not None
        if target_embeds is None:
            target_embeds = encode_targets(
                module=module,
                residue_dataset=residue_dataset,
                device=device,
                target_batch_size=target_batch_size,
                store_on_device=store_targets_on_device,
                retrieval_direction=primary_target_direction,
            )
            if target_embeds_cache:
                save_target_embedding_cache(
                    cache_path=target_embeds_cache,
                    target_embeds=target_embeds,
                    target_keys=list(residue_dataset.keys),
                    checkpoint_path=checkpoint_path,
                    config_path=config_path,
                    evaluation_protocol=evaluation_protocol,
                    retrieval_direction=primary_target_direction,
                )
        elif loaded_from_source_cache and target_embeds_cache:
            save_target_embedding_cache(
                cache_path=target_embeds_cache,
                target_embeds=target_embeds,
                target_keys=list(residue_dataset.keys),
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                evaluation_protocol=evaluation_protocol,
                retrieval_direction=primary_target_direction,
            )

        if target_embeds.device.type == "cpu":
            target_embeds_for_scoring = target_embeds.to(device)
        else:
            target_embeds_for_scoring = target_embeds

        r2e_target_embeds_for_scoring = target_embeds_for_scoring
        e2r_target_embeds_for_scoring = target_embeds_for_scoring
        if direction == "both" and getattr(module.model, "r2e_adapter", None) is not None:
            e2r_target_embeds = encode_targets(
                module=module,
                residue_dataset=residue_dataset,
                device=device,
                target_batch_size=target_batch_size,
                store_on_device=store_targets_on_device,
                retrieval_direction="enzyme_to_reaction",
            )
            e2r_target_embeds_for_scoring = (
                e2r_target_embeds.to(device)
                if e2r_target_embeds.device.type == "cpu"
                else e2r_target_embeds
            )

        if direction in {"reaction_to_enzyme", "both"}:
            for query_start in tqdm(
                range(0, len(unique_query_ids), batch_size),
                desc="Scoring reaction_to_enzyme queries",
            ):
                query_end = min(query_start + batch_size, len(unique_query_ids))
                query_ids = unique_query_ids[query_start:query_end]
                query_inputs = build_query_inputs(reaction_fps, query_ids, device)
                query_embeds = module.model.encode_queries(
                    query_inputs,
                    retrieval_direction="reaction_to_enzyme",
                )
                scores = torch.matmul(query_embeds, r2e_target_embeds_for_scoring.T)

                for row_idx, query_id in enumerate(query_ids):
                    target_indices = [
                        target_id_to_idx[target_id]
                        for target_id in query_to_targets[query_id]
                        if target_id in target_id_to_idx
                    ]
                    if not target_indices:
                        continue
                    target_idx = torch.tensor(target_indices, dtype=torch.long, device=device)
                    query_scores = scores[row_idx]
                    if hard_negative_output and hard_negative_top_k > 0:
                        negative_scores = query_scores.clone()
                        negative_scores[target_idx] = -torch.inf
                        top_k = min(hard_negative_top_k, len(candidate_keys) - len(target_indices))
                        if top_k > 0:
                            top_indices = torch.topk(negative_scores, k=top_k).indices.tolist()
                            anchor_to_hard_negatives[query_id] = [
                                candidate_keys[index] for index in top_indices
                            ]
                    append_retrieval_metrics(
                        r2e_metric_results,
                        query_scores,
                        target_idx,
                    )

        if direction in {"enzyme_to_reaction", "both"}:
            reaction_embeds = encode_reactions(
                module=module,
                reaction_features=reaction_fps,
                query_ids=unique_query_ids,
                device=device,
                batch_size=batch_size,
                retrieval_direction="enzyme_to_reaction",
            )
            reaction_id_to_idx = {query_id: idx for idx, query_id in enumerate(unique_query_ids)}
            valid_target_ids = sorted(
                target_id for target_id in target_to_queries if target_id in target_id_to_idx
            )
            for start in tqdm(
                range(0, len(valid_target_ids), target_batch_size),
                desc="Scoring enzyme_to_reaction queries",
            ):
                end = min(start + target_batch_size, len(valid_target_ids))
                batch_target_ids = valid_target_ids[start:end]
                protein_indices = torch.tensor(
                    [target_id_to_idx[target_id] for target_id in batch_target_ids],
                    dtype=torch.long,
                    device=e2r_target_embeds_for_scoring.device,
                )
                protein_embeds = e2r_target_embeds_for_scoring.index_select(
                    0,
                    protein_indices,
                )
                scores = torch.matmul(protein_embeds.to(device), reaction_embeds.T)
                for row_idx, target_id in enumerate(batch_target_ids):
                    reaction_indices = [
                        reaction_id_to_idx[query_id]
                        for query_id in target_to_queries[target_id]
                        if query_id in reaction_id_to_idx
                    ]
                    if not reaction_indices:
                        continue
                    target_idx = torch.tensor(
                        reaction_indices,
                        dtype=torch.long,
                        device=device,
                    )
                    query_scores = scores[row_idx]
                    if hard_negative_output and hard_negative_top_k > 0:
                        negative_scores = query_scores.clone()
                        negative_scores[target_idx] = -torch.inf
                        top_k = min(
                            hard_negative_top_k,
                            len(unique_query_ids) - len(reaction_indices),
                        )
                        if top_k > 0:
                            top_indices = torch.topk(
                                negative_scores,
                                k=top_k,
                            ).indices.tolist()
                            anchor_to_hard_negatives[target_id] = [
                                unique_query_ids[index] for index in top_indices
                            ]
                    append_retrieval_metrics(
                        e2r_metric_results,
                        query_scores,
                        target_idx,
                    )

    if direction == "enzyme_to_reaction":
        results = mean_metric_results(e2r_metric_results)
        results["num_queries"] = len(e2r_metric_results["top_1"])
    else:
        results = mean_metric_results(r2e_metric_results)
        results["num_queries"] = len(r2e_metric_results["top_1"])
    if direction == "both":
        r2e = mean_metric_results(r2e_metric_results)
        e2r = mean_metric_results(e2r_metric_results)
        r2e["num_queries"] = len(r2e_metric_results["top_1"])
        e2r["num_queries"] = len(e2r_metric_results["top_1"])
        results = {
            **{f"reaction_to_enzyme/{key}": value for key, value in r2e.items()},
            **{f"enzyme_to_reaction/{key}": value for key, value in e2r.items()},
            "balanced_mrr": (r2e.get("mrr", 0.0) + e2r.get("mrr", 0.0)) / 2.0,
            "balanced_first_positive_mrr": (
                r2e.get("first_positive_mrr", 0.0) + e2r.get("first_positive_mrr", 0.0)
            )
            / 2.0,
            "balanced_reactzyme_mrr": (
                r2e.get("reactzyme_mrr", 0.0) + e2r.get("reactzyme_mrr", 0.0)
            )
            / 2.0,
            "balanced_top_1": (r2e.get("top_1", 0.0) + e2r.get("top_1", 0.0)) / 2.0,
        }
    results["num_targets"] = len(residue_dataset)
    results["checkpoint"] = checkpoint_path
    results["config"] = config_path
    results["target_batch_size"] = target_batch_size
    results["direction"] = direction
    results["evaluation_protocol"] = evaluation_protocol
    results["hit_rate_cutoffs"] = list(HIT_RATE_CUTOFFS)
    results["ground_truth_pairs"] = "test_pairs_only"
    results["reaction_query_expansion"] = reaction_query_expansion
    results["num_enzyme_candidates"] = len(residue_dataset)
    results["num_reaction_candidates"] = len(unique_query_ids)
    results["configured_candidate_count"] = candidate_stats["configured_count"]
    results["test_candidate_count"] = candidate_stats["test_count"]
    if target_embeds_cache:
        results["target_embeds_cache"] = str(target_embeds_cache)
    if target_embeds_source_cache:
        results["target_embeds_source_cache"] = str(target_embeds_source_cache)
    if hard_negative_output:
        hard_negative_path = Path(hard_negative_output)
        hard_negative_path.parent.mkdir(parents=True, exist_ok=True)
        hard_negative_path.write_text(
            json.dumps(
                {
                    "schema_version": "model_ranked_hard_negatives_v2",
                    "checkpoint": str(Path(checkpoint_path).resolve()),
                    "config": str(Path(config_path).resolve()),
                    "direction": direction,
                    "top_k": int(hard_negative_top_k),
                    "anchor_to_negatives": anchor_to_hard_negatives,
                    **(
                        {"query_to_negatives": anchor_to_hard_negatives}
                        if direction == "reaction_to_enzyme"
                        else {}
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        results["hard_negative_output"] = str(hard_negative_path)
        results["hard_negative_queries"] = len(anchor_to_hard_negatives)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate protein-pooling Horizyn checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Query batch size")
    parser.add_argument(
        "--target-batch-size",
        type=int,
        default=512,
        help="Number of target proteins to encode per batch",
    )
    parser.add_argument(
        "--store-targets-on-cpu",
        action="store_true",
        help="Store encoded target matrix on CPU before scoring",
    )
    parser.add_argument(
        "--direction",
        choices=["reaction_to_enzyme", "enzyme_to_reaction", "both"],
        default="reaction_to_enzyme",
        help="Retrieval direction to evaluate",
    )
    parser.add_argument(
        "--target-embeds-cache",
        default=None,
        help="Optional checkpoint-specific target embedding cache path",
    )
    parser.add_argument(
        "--target-embeds-source-cache",
        default=None,
        help="Optional compatible superset cache used when the primary cache is absent",
    )
    parser.add_argument(
        "--evaluation-protocol",
        choices=EVALUATION_PROTOCOLS,
        default=LEGACY_ALL_CANDIDATES,
        help="Candidate/query construction used for retrieval evaluation",
    )
    parser.add_argument(
        "--hard-negative-output",
        default=None,
        help="Optional JSON path for top non-positive candidates in the selected direction",
    )
    parser.add_argument(
        "--hard-negative-top-k",
        type=int,
        default=0,
        help="Number of non-positive candidates exported per retrieval anchor",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}")
        sys.exit(1)

    results = evaluate_checkpoint(
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        device=args.device,
        batch_size=args.batch_size,
        target_batch_size=args.target_batch_size,
        store_targets_on_device=not args.store_targets_on_cpu,
        direction=args.direction,
        target_embeds_cache=args.target_embeds_cache,
        target_embeds_source_cache=args.target_embeds_source_cache,
        evaluation_protocol=args.evaluation_protocol,
        hard_negative_output=args.hard_negative_output,
        hard_negative_top_k=args.hard_negative_top_k,
    )
    print()
    print(format_results_table(results))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as output_file:
            json.dump(results, output_file, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
