#!/usr/bin/env python3
"""Evaluate unimodal Lorentz enzyme hierarchy separation diagnostics."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from horizyn.hyperbolic_enzyme import (  # noqa: E402
    LorentzEnzymeProjector,
    SLEECGuidedAttentionPool,
    compute_ec_shared_depth,
    format_ec_prefixes,
    known_ec_depth,
    parse_ec_prefixes,
)
from horizyn.lorentz import (  # noqa: E402
    lorentz_constraint_error,
    lorentz_distance,
    lorentz_origin,
    pairwise_lorentz_distance,
)
from scripts.pretrain_hyperbolic_enzyme import (  # noqa: E402
    DEFAULT_CONFIG,
    build_training_dataset,
    hyperbolic_enzyme_collate_fn,
)


DEFAULT_CHECKPOINT = (
    "checkpoints/hyperbolic_enzyme/"
    "hyperbolic-enzyme-esmc-lorentz-c0p25-stabilized-b512-4gpu-20260604_102625/"
    "best.ckpt"
)


MetricFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
EC_PLOT_LEVELS = ("ec1", "ec2", "ec3", "ec4")
EC_HIERARCHY_LEVELS = (1, 2, 3, 4)


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _device_from_config(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _resolve_checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config.update(checkpoint.get("config", {}))
    for key in ("input_dim", "hyp_dim", "curvature", "p0_sleec_threshold"):
        if key in checkpoint:
            config[key] = checkpoint[key]
    return config


def _load_models(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[SLEECGuidedAttentionPool, LorentzEnzymeProjector, dict[str, Any]]:
    config = _resolve_checkpoint_config(checkpoint)
    input_dim = int(config["input_dim"])
    projector_state = checkpoint.get("hyperbolic_projector_state_dict") or checkpoint.get(
        "projector_state_dict"
    )
    attention_state = checkpoint.get("attention_pooler_state_dict") or checkpoint.get(
        "sleec_guided_attention_pool_state_dict"
    )
    if projector_state is None:
        raise KeyError("Checkpoint does not contain a hyperbolic projector state dict")
    if attention_state is None:
        raise KeyError("Checkpoint does not contain a SLEEC-guided attention pooler state dict")

    inferred_norm = "layernorm" if any(key.startswith("input_norm.") for key in projector_state) else "none"
    input_normalization = str(config.get("projector_input_normalization", inferred_norm))
    if input_normalization == "none" and inferred_norm == "layernorm":
        input_normalization = "layernorm"

    attention_pooler = SLEECGuidedAttentionPool(
        input_dim=input_dim,
        scorer_hidden_dim=int(config.get("sleec_scorer_hidden_dim", 256)),
        p0=float(config.get("p0_sleec_threshold", 0.34)),
        sleec_checkpoint_path=None,
        freeze_sleec=True,
        attention_bias=bool(config.get("attention_bias", True)),
        initial_prior_strength=float(config.get("initial_prior_strength", 1.0)),
        train_prior_strength=bool(config.get("train_prior_strength", True)),
        eps=float(config.get("eps", 1e-6)),
    )
    projector = LorentzEnzymeProjector(
        input_dim=input_dim,
        hyp_dim=int(config["hyp_dim"]),
        curvature=float(config["curvature"]),
        tangent_clip=config.get("tangent_clip", 5.0),
        tangent_clip_mode=str(config.get("tangent_clip_mode", "hard")),
        input_normalization=input_normalization,
        eps=float(config.get("eps", 1e-6)),
    )
    attention_pooler.load_state_dict(attention_state)
    projector.load_state_dict(projector_state)
    attention_pooler.to(device).eval()
    projector.to(device).eval()
    return attention_pooler, projector, config


def _select_indices(length: int, max_samples: int, seed: int) -> list[int]:
    if max_samples <= 0 or length <= max_samples:
        return list(range(length))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(length, size=max_samples, replace=False).tolist())


def _load_eval_dataset(config: dict[str, Any], args: argparse.Namespace):
    residue_embeddings_path = args.residue_embeddings_path or config["residue_embeddings_path"]
    ec_labels_path = args.ec_labels_path or config["ec_labels_path"]
    sleec_logits_path = args.sleec_logits_path
    if sleec_logits_path is None:
        sleec_logits_path = config.get("sleec_logits_path")

    dataset, metadata = build_training_dataset(
        residue_embeddings_path=residue_embeddings_path,
        ec_labels_path=ec_labels_path,
        input_dim=int(config["input_dim"]),
        sleec_logits_path=sleec_logits_path,
        id_column=args.id_column if args.id_column is not None else config.get("id_column"),
        ec_column=args.ec_column if args.ec_column is not None else config.get("ec_column"),
        max_protein_tokens=args.max_protein_tokens
        if args.max_protein_tokens is not None
        else config.get("max_protein_tokens"),
        protein_truncation=str(config.get("protein_truncation", "ends_center")),
    )
    selected = _select_indices(len(dataset), int(args.max_samples), int(args.seed))
    if len(selected) != len(dataset):
        dataset.rows = [dataset.rows[idx] for idx in selected]
    metadata.update(
        {
            "eval_samples": len(dataset),
            "eval_residue_embeddings_path": str(residue_embeddings_path),
            "eval_ec_labels_path": str(ec_labels_path),
            "eval_sleec_logits_path": None if sleec_logits_path is None else str(sleec_logits_path),
            "max_samples": int(args.max_samples),
        }
    )
    return dataset, metadata


def extract_representations(
    *,
    dataset,
    attention_pooler: SLEECGuidedAttentionPool,
    projector: LorentzEnzymeProjector,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=hyperbolic_enzyme_collate_fn,
    )
    pooled_batches: list[torch.Tensor] = []
    z_batches: list[torch.Tensor] = []
    tangent_batches: list[torch.Tensor] = []
    attention_entropy: list[torch.Tensor] = []
    attention_mass_above_threshold: list[torch.Tensor] = []
    prior_strengths: list[torch.Tensor] = []
    mean_sleec_logits: list[torch.Tensor] = []
    enzyme_ids: list[str] = []
    ec_labels: list[str] = []

    with torch.no_grad():
        for batch in loader:
            residue_embeddings = batch["residue_embeddings"].to(device=device, non_blocking=True)
            residue_mask = batch["residue_mask"].to(device=device, non_blocking=True)
            sleec_logits = batch.get("sleec_logits")
            if sleec_logits is not None:
                sleec_logits = sleec_logits.to(device=device, non_blocking=True)

            pooled, details = attention_pooler(
                residue_embeddings,
                residue_mask=residue_mask,
                sleec_logits=sleec_logits,
                return_details=True,
            )
            z_hyp, z_tangent = projector(pooled)
            weights = details["attention_weights"]
            valid_logits = details["sleec_logits"].masked_fill(~residue_mask, 0.0)
            valid_count = residue_mask.sum(dim=1).clamp_min(1)
            threshold_logit = attention_pooler.threshold_logit.to(
                device=device,
                dtype=valid_logits.dtype,
            )
            mass_above = (
                weights.masked_fill(valid_logits <= threshold_logit, 0.0).sum(dim=1)
            )
            entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=1)
            mean_logit = valid_logits.sum(dim=1) / valid_count

            pooled_batches.append(pooled.detach().cpu())
            z_batches.append(z_hyp.detach().cpu())
            tangent_batches.append(z_tangent.detach().cpu())
            attention_entropy.append(entropy.detach().cpu())
            attention_mass_above_threshold.append(mass_above.detach().cpu())
            prior_strengths.append(
                torch.full(
                    (pooled.shape[0],),
                    float(details["prior_strength"].detach().cpu().item()),
                )
            )
            mean_sleec_logits.append(mean_logit.detach().cpu())
            enzyme_ids.extend(batch["enzyme_id"])
            ec_labels.extend(batch["ec_labels"])

    return {
        "enzyme_ids": enzyme_ids,
        "ec_labels": ec_labels,
        "pooled": torch.cat(pooled_batches, dim=0),
        "z_hyp": torch.cat(z_batches, dim=0),
        "z_tangent": torch.cat(tangent_batches, dim=0),
        "attention_entropy": torch.cat(attention_entropy, dim=0),
        "attention_mass_above_threshold": torch.cat(attention_mass_above_threshold, dim=0),
        "prior_strength": torch.cat(prior_strengths, dim=0),
        "mean_sleec_logit": torch.cat(mean_sleec_logits, dim=0),
    }


def _parse_all_prefixes(ec_labels: list[str]) -> list[tuple[str | None, str | None, str | None, str | None]]:
    return [parse_ec_prefixes(label) for label in ec_labels]


def _shared_depth(
    lhs: tuple[str | None, str | None, str | None, str | None],
    rhs: tuple[str | None, str | None, str | None, str | None],
) -> int:
    depth = 0
    for left, right in zip(lhs, rhs):
        if left is None or right is None or left != right:
            break
        depth += 1
    return depth


def _sample_pairs(num_items: int, max_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if num_items < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    total_pairs = num_items * (num_items - 1) // 2
    if max_pairs <= 0 or total_pairs <= max_pairs:
        rows, cols = np.triu_indices(num_items, k=1)
        return rows.astype(np.int64), cols.astype(np.int64)

    rng = np.random.default_rng(seed)
    left = rng.integers(0, num_items, size=max_pairs, dtype=np.int64)
    right = rng.integers(0, num_items - 1, size=max_pairs, dtype=np.int64)
    right = right + (right >= left)
    return left, right


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    rx = _rankdata_average(x[mask])
    ry = _rankdata_average(y[mask])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "q05": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "q95": float("nan"),
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _cosine_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_norm = F.normalize(left, dim=-1)
    right_norm = F.normalize(right, dim=-1)
    return 1.0 - (left_norm * right_norm).sum(dim=-1)


def _euclidean_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left - right).norm(dim=-1)


def _lorentz_pair_distance(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    curvature: float,
    eps: float,
) -> torch.Tensor:
    return lorentz_distance(left, right, kappa=curvature, eps=eps)


def _compute_pair_metric_values(
    values: torch.Tensor,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    distance_fn: MetricFn,
    *,
    batch_size: int,
) -> np.ndarray:
    distances: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(left_idx), batch_size):
            end = min(start + batch_size, len(left_idx))
            left = values[torch.from_numpy(left_idx[start:end]).long()]
            right = values[torch.from_numpy(right_idx[start:end]).long()]
            distances.append(distance_fn(left, right).detach().cpu())
    return torch.cat(distances, dim=0).numpy() if distances else np.empty(0, dtype=np.float32)


def compute_pair_distance_diagnostics(
    *,
    ec_labels: list[str],
    representations: dict[str, tuple[torch.Tensor, MetricFn]],
    max_pairs: int,
    seed: int,
    pair_batch_size: int,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, np.ndarray]]:
    prefixes = _parse_all_prefixes(ec_labels)
    left_idx, right_idx = _sample_pairs(len(ec_labels), max_pairs=max_pairs, seed=seed)
    depths = np.asarray(
        [_shared_depth(prefixes[int(i)], prefixes[int(j)]) for i, j in zip(left_idx, right_idx)],
        dtype=np.float64,
    )
    metrics: dict[str, float] = {"num_pair_samples": float(len(depths))}
    rows: list[dict[str, Any]] = []
    distance_cache: dict[str, np.ndarray] = {"depth": depths}

    for metric_name, (values, distance_fn) in representations.items():
        distances = _compute_pair_metric_values(
            values,
            left_idx,
            right_idx,
            distance_fn,
            batch_size=pair_batch_size,
        )
        distance_cache[metric_name] = distances
        metrics[f"{metric_name}_spearman_depth_neg_distance"] = _spearman(depths, -distances)
        for depth in range(5):
            selected = distances[depths == depth]
            stats = _quantiles(selected)
            rows.append(
                {
                    "metric": metric_name,
                    "shared_depth": depth,
                    "count": int(selected.shape[0]),
                    **stats,
                }
            )

        means = {
            depth: float(np.mean(distances[depths == depth]))
            for depth in range(5)
            if np.any(depths == depth)
        }
        adjacent = [
            float(means[depth + 1] < means[depth])
            for depth in range(4)
            if depth in means and (depth + 1) in means
        ]
        metrics[f"{metric_name}_adjacent_depth_mean_order_fraction"] = (
            float(np.mean(adjacent)) if adjacent else float("nan")
        )
    return metrics, rows, distance_cache


def _prefix_groups(
    prefixes: list[tuple[str | None, str | None, str | None, str | None]],
) -> dict[tuple[int, str], list[int]]:
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for idx, prefix_tuple in enumerate(prefixes):
        for level, prefix in enumerate(prefix_tuple, start=1):
            if prefix is not None:
                groups[(level, prefix)].append(idx)
    return groups


def _sample_exact_depth_index(
    *,
    anchor_idx: int,
    target_depth: int,
    prefixes: list[tuple[str | None, str | None, str | None, str | None]],
    groups: dict[tuple[int, str], list[int]],
    rng: random.Random,
    max_attempts: int = 200,
) -> int | None:
    if target_depth < 0 or target_depth > 4:
        return None
    if target_depth == 0:
        candidates = range(len(prefixes))
    else:
        prefix = prefixes[anchor_idx][target_depth - 1]
        if prefix is None:
            return None
        candidates = groups.get((target_depth, prefix), [])
    candidates_list = list(candidates)
    if len(candidates_list) <= 1:
        return None
    for _ in range(max_attempts):
        candidate = int(rng.choice(candidates_list))
        if candidate == anchor_idx:
            continue
        if _shared_depth(prefixes[anchor_idx], prefixes[candidate]) == target_depth:
            return candidate
    return None


def sample_hierarchy_triplets(
    ec_labels: list[str],
    *,
    max_triplet_anchors: int,
    max_triplets_per_anchor: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prefixes = _parse_all_prefixes(ec_labels)
    groups = _prefix_groups(prefixes)
    rng = random.Random(seed)
    anchor_indices = list(range(len(ec_labels)))
    rng.shuffle(anchor_indices)
    if max_triplet_anchors > 0:
        anchor_indices = anchor_indices[:max_triplet_anchors]

    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    positive_depths: list[int] = []
    negative_depths: list[int] = []

    for anchor_idx in anchor_indices:
        triplets_for_anchor = 0
        possible_positive_depths = list(range(min(4, known_ec_depth(ec_labels[anchor_idx])), 0, -1))
        while triplets_for_anchor < max_triplets_per_anchor:
            added = False
            for pos_depth in possible_positive_depths:
                pos_idx = _sample_exact_depth_index(
                    anchor_idx=anchor_idx,
                    target_depth=pos_depth,
                    prefixes=prefixes,
                    groups=groups,
                    rng=rng,
                )
                if pos_idx is None:
                    continue
                for neg_depth in range(pos_depth - 1, -1, -1):
                    neg_idx = _sample_exact_depth_index(
                        anchor_idx=anchor_idx,
                        target_depth=neg_depth,
                        prefixes=prefixes,
                        groups=groups,
                        rng=rng,
                    )
                    if neg_idx is None:
                        continue
                    anchors.append(anchor_idx)
                    positives.append(pos_idx)
                    negatives.append(neg_idx)
                    positive_depths.append(pos_depth)
                    negative_depths.append(neg_depth)
                    triplets_for_anchor += 1
                    added = True
                    break
                if added:
                    break
            if not added:
                break

    return (
        np.asarray(anchors, dtype=np.int64),
        np.asarray(positives, dtype=np.int64),
        np.asarray(negatives, dtype=np.int64),
        np.asarray(positive_depths, dtype=np.float64),
        np.asarray(negative_depths, dtype=np.float64),
    )


def compute_triplet_diagnostics(
    *,
    ec_labels: list[str],
    representations: dict[str, tuple[torch.Tensor, MetricFn]],
    base_margin: float,
    max_triplet_anchors: int,
    max_triplets_per_anchor: int,
    seed: int,
    pair_batch_size: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    anchor_idx, pos_idx, neg_idx, depth_pos, depth_neg = sample_hierarchy_triplets(
        ec_labels,
        max_triplet_anchors=max_triplet_anchors,
        max_triplets_per_anchor=max_triplets_per_anchor,
        seed=seed,
    )
    metrics: dict[str, float] = {"num_triplet_samples": float(len(anchor_idx))}
    rows: list[dict[str, Any]] = []
    if len(anchor_idx) == 0:
        return metrics, rows

    margins = base_margin * np.maximum(depth_pos - depth_neg, 0.0)
    depth_gaps = (depth_pos - depth_neg).astype(int)
    for metric_name, (values, distance_fn) in representations.items():
        pos_dist = _compute_pair_metric_values(
            values,
            anchor_idx,
            pos_idx,
            distance_fn,
            batch_size=pair_batch_size,
        )
        neg_dist = _compute_pair_metric_values(
            values,
            anchor_idx,
            neg_idx,
            distance_fn,
            batch_size=pair_batch_size,
        )
        ranking_ok = pos_dist < neg_dist
        margin_ok = margins + pos_dist - neg_dist <= 0.0
        ranking_loss = np.maximum(margins + pos_dist - neg_dist, 0.0)

        metrics[f"{metric_name}_triplet_accuracy"] = float(np.mean(ranking_ok))
        metrics[f"{metric_name}_triplet_margin_satisfaction"] = float(np.mean(margin_ok))
        metrics[f"{metric_name}_triplet_margin_loss"] = float(np.mean(ranking_loss))
        metrics[f"{metric_name}_triplet_mean_positive_distance"] = float(np.mean(pos_dist))
        metrics[f"{metric_name}_triplet_mean_negative_distance"] = float(np.mean(neg_dist))
        metrics[f"{metric_name}_triplet_mean_positive_depth"] = float(np.mean(depth_pos))
        metrics[f"{metric_name}_triplet_mean_negative_depth"] = float(np.mean(depth_neg))

        for gap in sorted(set(depth_gaps.tolist())):
            selected = depth_gaps == gap
            rows.append(
                {
                    "metric": metric_name,
                    "depth_gap": int(gap),
                    "count": int(selected.sum()),
                    "accuracy": float(np.mean(ranking_ok[selected])),
                    "margin_satisfaction": float(np.mean(margin_ok[selected])),
                    "margin_loss": float(np.mean(ranking_loss[selected])),
                    "mean_positive_distance": float(np.mean(pos_dist[selected])),
                    "mean_negative_distance": float(np.mean(neg_dist[selected])),
                }
            )
    return metrics, rows


def _topk_lorentz(
    values: torch.Tensor,
    *,
    k: int,
    curvature: float,
    eps: float,
    batch_size: int,
) -> np.ndarray:
    indices: list[torch.Tensor] = []
    for start in range(0, values.shape[0], batch_size):
        end = min(start + batch_size, values.shape[0])
        distances = pairwise_lorentz_distance(
            values[start:end],
            values,
            kappa=curvature,
            eps=eps,
        )
        row_idx = torch.arange(end - start)
        distances[row_idx, torch.arange(start, end)] = float("inf")
        indices.append(torch.topk(distances, k=k, largest=False, dim=1).indices.cpu())
    return torch.cat(indices, dim=0).numpy()


def _topk_cosine(values: torch.Tensor, *, k: int, batch_size: int) -> np.ndarray:
    normalized = F.normalize(values, dim=-1)
    indices: list[torch.Tensor] = []
    for start in range(0, values.shape[0], batch_size):
        end = min(start + batch_size, values.shape[0])
        similarity = normalized[start:end] @ normalized.T
        row_idx = torch.arange(end - start)
        similarity[row_idx, torch.arange(start, end)] = -float("inf")
        indices.append(torch.topk(similarity, k=k, largest=True, dim=1).indices.cpu())
    return torch.cat(indices, dim=0).numpy()


def _topk_euclidean(values: torch.Tensor, *, k: int, batch_size: int) -> np.ndarray:
    indices: list[torch.Tensor] = []
    for start in range(0, values.shape[0], batch_size):
        end = min(start + batch_size, values.shape[0])
        distances = torch.cdist(values[start:end], values)
        row_idx = torch.arange(end - start)
        distances[row_idx, torch.arange(start, end)] = float("inf")
        indices.append(torch.topk(distances, k=k, largest=False, dim=1).indices.cpu())
    return torch.cat(indices, dim=0).numpy()


def _exact_depth_counts(
    query_idx: int,
    prefixes: list[tuple[str | None, str | None, str | None, str | None]],
    groups: dict[tuple[int, str], list[int]],
) -> dict[int, int]:
    num_items = len(prefixes)
    query = prefixes[query_idx]
    ge = {0: num_items}
    for level in range(1, 5):
        prefix = query[level - 1]
        ge[level] = len(groups.get((level, prefix), [])) if prefix is not None else 0

    exact: dict[int, int] = {}
    exact[0] = max(num_items - ge[1], 0)
    for depth in range(1, 5):
        if query[depth - 1] is None:
            exact[depth] = 0
            continue
        if depth == 4 or query[depth] is None:
            exact[depth] = max(ge[depth] - 1, 0)
        else:
            exact[depth] = max(ge[depth] - ge[depth + 1], 0)
    return exact


def _dcg(relevances: np.ndarray) -> float:
    if len(relevances) == 0:
        return float("nan")
    discounts = 1.0 / np.log2(np.arange(2, len(relevances) + 2, dtype=np.float64))
    gains = np.power(2.0, relevances.astype(np.float64)) - 1.0
    return float(np.sum(gains * discounts))


def _ideal_dcg(
    query_idx: int,
    k: int,
    prefixes: list[tuple[str | None, str | None, str | None, str | None]],
    groups: dict[tuple[int, str], list[int]],
) -> float:
    counts = _exact_depth_counts(query_idx, prefixes, groups)
    ideal: list[int] = []
    for depth in range(4, -1, -1):
        take = min(k - len(ideal), counts.get(depth, 0))
        ideal.extend([depth] * take)
        if len(ideal) >= k:
            break
    return _dcg(np.asarray(ideal, dtype=np.float64))


def compute_knn_diagnostics(
    *,
    ec_labels: list[str],
    representations: dict[str, tuple[torch.Tensor, str]],
    k_values: list[int],
    max_knn_samples: int,
    seed: int,
    batch_size: int,
    curvature: float,
    eps: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if len(ec_labels) < 2:
        return {}, []
    selected_indices = _select_indices(len(ec_labels), max_knn_samples, seed)
    selected_labels = [ec_labels[idx] for idx in selected_indices]
    prefixes = _parse_all_prefixes(selected_labels)
    groups = _prefix_groups(prefixes)
    max_k = min(max(k_values), len(selected_indices) - 1)
    metrics: dict[str, float] = {"num_knn_samples": float(len(selected_indices))}
    rows: list[dict[str, Any]] = []
    if max_k <= 0:
        return metrics, rows

    index_tensor = torch.tensor(selected_indices, dtype=torch.long)
    for metric_name, (all_values, mode) in representations.items():
        values = all_values[index_tensor]
        if mode == "lorentz":
            topk = _topk_lorentz(
                values,
                k=max_k,
                curvature=curvature,
                eps=eps,
                batch_size=batch_size,
            )
        elif mode == "cosine":
            topk = _topk_cosine(values, k=max_k, batch_size=batch_size)
        elif mode == "euclidean":
            topk = _topk_euclidean(values, k=max_k, batch_size=batch_size)
        else:
            raise ValueError(f"Unknown kNN mode: {mode}")

        for k in k_values:
            if k > max_k:
                continue
            neighbor_idx = topk[:, :k]
            depth_values = np.empty((len(selected_indices), k), dtype=np.int64)
            for query_idx in range(len(selected_indices)):
                query_prefix = prefixes[query_idx]
                for rank_idx, neighbor in enumerate(neighbor_idx[query_idx]):
                    depth_values[query_idx, rank_idx] = _shared_depth(
                        query_prefix,
                        prefixes[int(neighbor)],
                    )

            ndcgs = []
            for query_idx in range(len(selected_indices)):
                ideal = _ideal_dcg(query_idx, k, prefixes, groups)
                ndcgs.append(_dcg(depth_values[query_idx].astype(np.float64)) / ideal if ideal > 0 else np.nan)
            ndcg = float(np.nanmean(ndcgs)) if np.isfinite(ndcgs).any() else float("nan")
            metrics[f"{metric_name}_ndcg_depth_at_{k}"] = ndcg

            for level in range(1, 5):
                matches = depth_values >= level
                purity = matches.mean(axis=1)
                value = float(np.mean(purity))
                metrics[f"{metric_name}_ec{level}_purity_at_{k}"] = value
                rows.append(
                    {
                        "metric": metric_name,
                        "k": int(k),
                        "level": f"EC{level}",
                        "purity": value,
                        "num_queries": int(len(selected_indices)),
                    }
                )
            rows.append(
                {
                    "metric": metric_name,
                    "k": int(k),
                    "level": "depth_ndcg",
                    "purity": ndcg,
                    "num_queries": int(len(selected_indices)),
                }
            )
    return metrics, rows


def compute_manifold_metrics(
    *,
    z_hyp: torch.Tensor,
    z_tangent: torch.Tensor,
    curvature: float,
    eps: float,
) -> tuple[dict[str, float], np.ndarray]:
    origin = lorentz_origin(
        z_hyp.shape[-1] - 1,
        kappa=curvature,
        dtype=z_hyp.dtype,
        device=z_hyp.device,
    )
    with torch.no_grad():
        errors = lorentz_constraint_error(z_hyp, kappa=curvature)
        radii = lorentz_distance(
            z_hyp,
            origin.unsqueeze(0).expand_as(z_hyp),
            kappa=curvature,
            eps=eps,
        )
        tangent_norm = z_tangent.norm(dim=-1)
    metrics = {
        "num_embeddings": float(z_hyp.shape[0]),
        "finite_z_hyp_fraction": float(torch.isfinite(z_hyp).all(dim=1).float().mean().item()),
        "finite_z_tangent_fraction": float(torch.isfinite(z_tangent).all(dim=1).float().mean().item()),
        "mean_lorentz_constraint_error": float(errors.mean().item()),
        "max_lorentz_constraint_error": float(errors.max().item()),
        "mean_radius": float(radii.mean().item()),
        "std_radius": float(radii.std(unbiased=False).item()),
        "min_radius": float(radii.min().item()),
        "max_radius": float(radii.max().item()),
        "mean_tangent_norm": float(tangent_norm.mean().item()),
        "std_tangent_norm": float(tangent_norm.std(unbiased=False).item()),
        "max_tangent_norm": float(tangent_norm.max().item()),
    }
    return metrics, radii.cpu().numpy()


def _pca_coordinates(values: torch.Tensor, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(_select_indices(values.shape[0], max_points, seed), dtype=np.int64)
    x = values[torch.from_numpy(selected)].float().numpy()
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    if coords.shape[1] == 1:
        coords = np.concatenate([coords, np.zeros((coords.shape[0], 1))], axis=1)
    return selected, coords[:, :2]


def _pcoa_coordinates(
    z_hyp: torch.Tensor,
    *,
    max_points: int,
    seed: int,
    curvature: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(_select_indices(z_hyp.shape[0], max_points, seed), dtype=np.int64)
    values = z_hyp[torch.from_numpy(selected)]
    distances = pairwise_lorentz_distance(values, values, kappa=curvature, eps=eps).numpy()
    squared = distances * distances
    n = squared.shape[0]
    centering = np.eye(n) - np.full((n, n), 1.0 / n)
    gram = -0.5 * centering @ squared @ centering
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    coords = eigvecs[:, :2] * np.sqrt(np.clip(eigvals[:2], 0.0, None))
    if coords.shape[1] == 1:
        coords = np.concatenate([coords, np.zeros((coords.shape[0], 1))], axis=1)
    return selected, coords[:, :2]


def _projection_rows(
    *,
    name: str,
    selected: np.ndarray,
    coords: np.ndarray,
    enzyme_ids: list[str],
    ec_labels: list[str],
    radii: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local_idx, sample_idx in enumerate(selected):
        prefixes = parse_ec_prefixes(ec_labels[int(sample_idx)])
        rows.append(
            {
                "projection": name,
                "sample_index": int(sample_idx),
                "enzyme_id": enzyme_ids[int(sample_idx)],
                "ec_label": ec_labels[int(sample_idx)],
                "ec1": prefixes[0],
                "ec2": prefixes[1],
                "ec3": prefixes[2],
                "ec4": prefixes[3],
                "x": float(coords[local_idx, 0]),
                "y": float(coords[local_idx, 1]),
                "lorentz_radius": float(radii[int(sample_idx)]),
            }
        )
    return rows


def compute_projection_rows(
    *,
    enzyme_ids: list[str],
    ec_labels: list[str],
    pooled: torch.Tensor,
    z_tangent: torch.Tensor,
    z_hyp: torch.Tensor,
    radii: np.ndarray,
    max_visualization_points: int,
    max_pcoa_points: int,
    seed: int,
    curvature: float,
    eps: float,
) -> dict[str, list[dict[str, Any]]]:
    raw_selected, raw_coords = _pca_coordinates(pooled, max_visualization_points, seed)
    tangent_selected, tangent_coords = _pca_coordinates(z_tangent, max_visualization_points, seed)
    pcoa_selected, pcoa_coords = _pcoa_coordinates(
        z_hyp,
        max_points=max_pcoa_points,
        seed=seed,
        curvature=curvature,
        eps=eps,
    )
    return {
        "raw_pooled_pca": _projection_rows(
            name="raw_pooled_pca",
            selected=raw_selected,
            coords=raw_coords,
            enzyme_ids=enzyme_ids,
            ec_labels=ec_labels,
            radii=radii,
        ),
        "tangent_pca": _projection_rows(
            name="tangent_pca",
            selected=tangent_selected,
            coords=tangent_coords,
            enzyme_ids=enzyme_ids,
            ec_labels=ec_labels,
            radii=radii,
        ),
        "lorentz_pcoa": _projection_rows(
            name="lorentz_pcoa",
            selected=pcoa_selected,
            coords=pcoa_coords,
            enzyme_ids=enzyme_ids,
            ec_labels=ec_labels,
            radii=radii,
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_visualization_notes(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Hyperbolic Enzyme Evaluation Outputs",
                "",
                "- `metrics.json`: scalar evaluation metrics for manifold validity, EC-depth ordering, triplet ranking, kNN purity, and negative controls.",
                "- `distance_by_ec_depth.csv`: mean/median distance grouped by EC shared-prefix depth.",
                "- `triplet_metrics_by_gap.csv`: hierarchy triplet accuracy and margin satisfaction by EC-depth gap.",
                "- `knn_purity.csv`: EC1/EC2/EC3/EC4 purity@k and depth-NDCG@k.",
                "- `raw_pooled_pca.csv`: PCA of the SLEEC-guided pooled encoder representation before the Lorentz projector.",
                "- `tangent_pca.csv`: PCA of `Log_o(z_hyp)`, the Euclidean tangent representation to use later in Horizyn.",
                "- `lorentz_pcoa.csv`: 2D classical MDS/PCoA over Lorentz distances; this best reflects the trained metric.",
                "- `projection_ec4_sample_separation.csv`: for each projected point with a repeated complete EC4 label, nearest same-EC4 versus nearest different-EC4 distance.",
                "- `projection_ec4_centroid_separation.csv`: per-complete-EC4 projection centroid radius and nearest other EC4 centroid distance.",
                "- `projection_ec_hierarchy_centroids.csv`: EC1/EC2/EC3/EC4 centroids in each 2D projection, with parent labels for hierarchy overlays.",
                "",
                "A useful projection should show monotonic Lorentz distances by EC depth, higher Lorentz triplet accuracy than raw pooled cosine distance, and higher EC kNN purity/NDCG than shuffled-label controls.",
            ]
        )
        + "\n"
    )


def _is_complete_ec4(label: str) -> bool:
    label = str(label)
    return bool(label) and "-" not in label and len(label.split(".")) == 4


def _is_known_ec_prefix(label: str, level: int) -> bool:
    label = str(label)
    return bool(label) and "-" not in label and len(label.split(".")) == level


def _projection_ec4_separation_rows(
    projection_name: str,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    complete_rows = [row for row in rows if _is_complete_ec4(str(row.get("ec4", "")))]
    if len(complete_rows) < 2:
        return [], []

    coords = np.asarray(
        [[float(row["x"]), float(row["y"])] for row in complete_rows],
        dtype=np.float64,
    )
    labels = np.asarray([str(row["ec4"]) for row in complete_rows], dtype=object)
    unique_labels = sorted(set(labels.tolist()))
    label_counts = {label: int(np.sum(labels == label)) for label in unique_labels}

    sample_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(complete_rows):
        label = labels[idx]
        same_mask = labels == label
        same_mask[idx] = False
        if not bool(np.any(same_mask)):
            continue
        diff_mask = labels != label
        distances = np.linalg.norm(coords - coords[idx], axis=1)
        nearest_same = float(np.min(distances[same_mask]))
        nearest_diff = float(np.min(distances[diff_mask]))
        margin = nearest_diff - nearest_same
        sample_rows.append(
            {
                "projection": projection_name,
                "sample_index": row.get("sample_index", idx),
                "enzyme_id": row.get("enzyme_id", ""),
                "ec4": label,
                "ec4_count_in_projection": label_counts[label],
                "nearest_same_ec4_distance": nearest_same,
                "nearest_different_ec4_distance": nearest_diff,
                "separation_margin": margin,
                "same_closer_than_different": int(margin > 0.0),
            }
        )

    group_indices = {label: np.where(labels == label)[0] for label in unique_labels}
    centroid_labels = []
    centroid_values = []
    centroid_rows: list[dict[str, Any]] = []
    for label in unique_labels:
        indices = group_indices[label]
        centroid = coords[indices].mean(axis=0)
        centroid_labels.append(label)
        centroid_values.append(centroid)
    centroids = np.asarray(centroid_values, dtype=np.float64)

    for label_idx, label in enumerate(centroid_labels):
        indices = group_indices[label]
        centroid = centroids[label_idx]
        within = np.linalg.norm(coords[indices] - centroid, axis=1)
        if len(centroids) > 1:
            centroid_distances = np.linalg.norm(centroids - centroid, axis=1)
            centroid_distances[label_idx] = np.inf
            nearest_other_idx = int(np.argmin(centroid_distances))
            nearest_other_distance = float(centroid_distances[nearest_other_idx])
            nearest_other_label = centroid_labels[nearest_other_idx]
        else:
            nearest_other_distance = float("nan")
            nearest_other_label = ""
        median_within = float(np.median(within)) if len(within) else 0.0
        max_within = float(np.max(within)) if len(within) else 0.0
        centroid_rows.append(
            {
                "projection": projection_name,
                "ec4": label,
                "ec4_count_in_projection": int(len(indices)),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "median_within_ec4_radius": median_within,
                "max_within_ec4_radius": max_within,
                "nearest_different_ec4": nearest_other_label,
                "nearest_different_ec4_centroid_distance": nearest_other_distance,
                "median_radius_separation_margin": nearest_other_distance - median_within,
                "max_radius_separation_margin": nearest_other_distance - max_within,
                "median_radius_separated": int(nearest_other_distance > median_within),
                "max_radius_separated": int(nearest_other_distance > max_within),
            }
        )
    return sample_rows, centroid_rows


def _projection_hierarchy_centroid_rows(
    projection_name: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    centroid_rows: list[dict[str, Any]] = []
    for level in EC_HIERARCHY_LEVELS:
        key = f"ec{level}"
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            label = str(row.get(key, ""))
            if _is_known_ec_prefix(label, level):
                grouped[label].append(row)
        for label, label_rows in sorted(grouped.items()):
            x_values = [float(row["x"]) for row in label_rows]
            y_values = [float(row["y"]) for row in label_rows]
            parts = label.split(".")
            parent_label = ".".join(parts[:-1]) if level > 1 else ""
            centroid_rows.append(
                {
                    "projection": projection_name,
                    "ec_level": level,
                    "ec_label": label,
                    "parent_level": level - 1 if level > 1 else "",
                    "parent_label": parent_label,
                    "ec1": parts[0],
                    "count_in_projection": len(label_rows),
                    "centroid_x": float(np.mean(x_values)),
                    "centroid_y": float(np.mean(y_values)),
                }
            )
    return centroid_rows


def _try_write_plots(
    output_dir: Path,
    *,
    distance_rows: list[dict[str, Any]],
    projection_rows: dict[str, list[dict[str, Any]]],
) -> None:
    _write_svg_plots(output_dir, distance_rows=distance_rows, projection_rows=projection_rows)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = sorted({row["metric"] for row in distance_rows})
    for metric in metrics:
        rows = [row for row in distance_rows if row["metric"] == metric]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: int(row["shared_depth"]))
        x = [int(row["shared_depth"]) for row in rows]
        y = [float(row["mean"]) for row in rows]
        q25 = [float(row["q25"]) for row in rows]
        q75 = [float(row["q75"]) for row in rows]
        plt.figure(figsize=(6, 4))
        plt.plot(x, y, marker="o")
        plt.fill_between(x, q25, q75, alpha=0.2)
        plt.xlabel("EC shared-prefix depth")
        plt.ylabel("distance")
        plt.title(metric)
        plt.tight_layout()
        plt.savefig(plot_dir / f"distance_by_depth_{metric}.png", dpi=160)
        plt.close()

    for name, rows in projection_rows.items():
        if not rows:
            continue
        x = [float(row["x"]) for row in rows]
        y = [float(row["y"]) for row in rows]
        for level in EC_PLOT_LEVELS:
            labels = [str(row.get(level, "unknown")) for row in rows]
            colors = [_svg_color(label) for label in labels]
            plt.figure(figsize=(6, 5))
            plt.scatter(x, y, c=colors, s=8, alpha=0.75, linewidths=0)
            plt.xlabel("component 1")
            plt.ylabel("component 2")
            plt.title(f"{name} colored by {level.upper()}")
            plt.tight_layout()
            plt.savefig(plot_dir / f"{name}_{level}.png", dpi=180)
            plt.close()


def _scale(values: list[float], low: float, high: float, *, invert: bool = False) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if not math.isfinite(min_value) or not math.isfinite(max_value) or max_value == min_value:
        midpoint = 0.5 * (low + high)
        return [midpoint for _ in values]
    scaled = [low + (value - min_value) / (max_value - min_value) * (high - low) for value in values]
    if invert:
        return [high - (value - low) for value in scaled]
    return scaled


def _svg_color(label: str) -> str:
    palette = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#4f46e5",
        "#be123c",
        "#65a30d",
        "#0f766e",
        "#7c2d12",
        "#334155",
        "#a21caf",
        "#ca8a04",
        "#059669",
        "#0284c7",
        "#c026d3",
        "#db2777",
        "#4d7c0f",
        "#0e7490",
    ]
    stable_hash = sum((idx + 1) * ord(char) for idx, char in enumerate(label))
    return palette[stable_hash % len(palette)]


def _write_svg_line_chart(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    width, height = 720, 420
    left, right, top, bottom = 70, 30, 45, 60
    rows = sorted(rows, key=lambda row: int(row["shared_depth"]))
    x_values = [float(row["shared_depth"]) for row in rows]
    y_values = [float(row["mean"]) for row in rows]
    x_scaled = _scale(x_values, left, width - right)
    y_scaled = _scale(y_values, top, height - bottom, invert=True)
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(x_scaled, y_scaled))
    circles = "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#2563eb" />'
        f'<text x="{x:.2f}" y="{height - 32}" text-anchor="middle" font-size="12">{int(depth)}</text>'
        for x, y, depth in zip(x_scaled, y_scaled, x_values)
    )
    labels = "\n".join(
        f'<text x="{left - 12}" y="{y:.2f}" text-anchor="end" font-size="11">{value:.3f}</text>'
        for y, value in zip(y_scaled, y_values)
    )
    path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{left}" y="26" font-size="18" font-family="Arial">{html.escape(title)}</text>
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>
  <polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/>
  {circles}
  {labels}
  <text x="{(left + width - right) / 2:.2f}" y="{height - 12}" text-anchor="middle" font-size="13">EC shared-prefix depth</text>
  <text x="18" y="{height / 2:.2f}" transform="rotate(-90 18 {height / 2:.2f})" text-anchor="middle" font-size="13">distance</text>
</svg>
''',
    )


def _write_svg_scatter(path: Path, rows: list[dict[str, Any]], title: str, *, color_key: str) -> None:
    width, height = 720, 560
    left, right, top, bottom = 55, 25, 45, 45
    x_values = [float(row["x"]) for row in rows]
    y_values = [float(row["y"]) for row in rows]
    x_scaled = _scale(x_values, left, width - right)
    y_scaled = _scale(y_values, top, height - bottom, invert=True)
    points = "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.8" fill="{_svg_color(str(row.get(color_key, "unknown")))}" fill-opacity="0.72">'
        f'<title>{html.escape(str(row["enzyme_id"]))} {html.escape(str(row["ec_label"]))}</title></circle>'
        for x, y, row in zip(x_scaled, y_scaled, rows)
    )
    path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{left}" y="26" font-size="18" font-family="Arial">{html.escape(title)}</text>
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>
  {points}
  <text x="{(left + width - right) / 2:.2f}" y="{height - 12}" text-anchor="middle" font-size="13">component 1</text>
  <text x="18" y="{height / 2:.2f}" transform="rotate(-90 18 {height / 2:.2f})" text-anchor="middle" font-size="13">component 2</text>
</svg>
''',
    )


def _write_svg_diagonal_scatter(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
    x_key: str,
    y_key: str,
    good_key: str,
    x_label: str,
    y_label: str,
) -> None:
    finite_rows = [
        row
        for row in rows
        if math.isfinite(float(row[x_key])) and math.isfinite(float(row[y_key]))
    ]
    if not finite_rows:
        return
    width, height = 720, 560
    left, right, top, bottom = 70, 30, 45, 60
    x_values = [float(row[x_key]) for row in finite_rows]
    y_values = [float(row[y_key]) for row in finite_rows]
    max_value = max(max(x_values), max(y_values), 1e-12)
    x_span = (width - right) - left
    y_span = (height - bottom) - top
    x_scaled = [left + value / max_value * x_span for value in x_values]
    y_scaled = [(height - bottom) - value / max_value * y_span for value in y_values]
    diag_start_x = left
    diag_start_y = height - bottom
    diag_end_x = width - right
    diag_end_y = top
    points = "\n".join(
        (
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.4" '
            f'fill="{"#16a34a" if int(row[good_key]) else "#dc2626"}" fill-opacity="0.52">'
            f'<title>{html.escape(str(row.get("enzyme_id", row.get("ec4", ""))))} '
            f'{html.escape(str(row.get("ec4", "")))} '
            f'x={float(row[x_key]):.4g} y={float(row[y_key]):.4g}</title></circle>'
        )
        for x, y, row in zip(x_scaled, y_scaled, finite_rows)
    )
    path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{left}" y="26" font-size="18" font-family="Arial">{html.escape(title)}</text>
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>
  <line x1="{diag_start_x}" y1="{diag_start_y}" x2="{diag_end_x}" y2="{diag_end_y}" stroke="#64748b" stroke-dasharray="6 5"/>
  {points}
  <text x="{left}" y="{height - 36}" font-size="11" fill="#475569">0</text>
  <text x="{width - right}" y="{height - 36}" font-size="11" text-anchor="end" fill="#475569">{max_value:.3g}</text>
  <text x="{left - 8}" y="{height - bottom}" font-size="11" text-anchor="end" fill="#475569">0</text>
  <text x="{left - 8}" y="{top}" font-size="11" text-anchor="end" fill="#475569">{max_value:.3g}</text>
  <text x="{(left + width - right) / 2:.2f}" y="{height - 12}" text-anchor="middle" font-size="13">{html.escape(x_label)}</text>
  <text x="18" y="{height / 2:.2f}" transform="rotate(-90 18 {height / 2:.2f})" text-anchor="middle" font-size="13">{html.escape(y_label)}</text>
  <text x="{width - right}" y="{top + 18}" text-anchor="end" font-size="12" fill="#16a34a">above diagonal = separated</text>
</svg>
''',
    )


def _write_ec4_projection_separation_outputs(
    output_dir: Path,
    projection_rows: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_sample_rows: list[dict[str, Any]] = []
    all_centroid_rows: list[dict[str, Any]] = []
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for name, rows in projection_rows.items():
        sample_rows, centroid_rows = _projection_ec4_separation_rows(name, rows)
        all_sample_rows.extend(sample_rows)
        all_centroid_rows.extend(centroid_rows)
        _write_svg_diagonal_scatter(
            plot_dir / f"{name}_ec4_sample_separation.svg",
            sample_rows,
            title=f"{name}: nearest same EC4 vs different EC4",
            x_key="nearest_same_ec4_distance",
            y_key="nearest_different_ec4_distance",
            good_key="same_closer_than_different",
            x_label="nearest same full EC distance",
            y_label="nearest different full EC distance",
        )
        _write_svg_diagonal_scatter(
            plot_dir / f"{name}_ec4_centroid_separation.svg",
            centroid_rows,
            title=f"{name}: EC4 centroid separation",
            x_key="median_within_ec4_radius",
            y_key="nearest_different_ec4_centroid_distance",
            good_key="median_radius_separated",
            x_label="median within full EC radius",
            y_label="nearest different full EC centroid distance",
        )

    write_csv(output_dir / "projection_ec4_sample_separation.csv", all_sample_rows)
    write_csv(output_dir / "projection_ec4_centroid_separation.csv", all_centroid_rows)
    return all_sample_rows, all_centroid_rows


def _write_svg_hierarchy_overlay(
    path: Path,
    *,
    projection_name: str,
    point_rows: list[dict[str, Any]],
    centroid_rows: list[dict[str, Any]],
) -> None:
    if not point_rows or not centroid_rows:
        return
    width, height = 860, 660
    left, right, top, bottom = 60, 30, 55, 60
    all_x = [float(row["x"]) for row in point_rows] + [
        float(row["centroid_x"]) for row in centroid_rows
    ]
    all_y = [float(row["y"]) for row in point_rows] + [
        float(row["centroid_y"]) for row in centroid_rows
    ]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * ((width - right) - left)

    def sy(value: float) -> float:
        return (height - bottom) - (value - y_min) / (y_max - y_min) * ((height - bottom) - top)

    centroid_lookup = {
        (int(row["ec_level"]), str(row["ec_label"])): row for row in centroid_rows
    }
    point_elements = "\n".join(
        f'<circle cx="{sx(float(row["x"])):.2f}" cy="{sy(float(row["y"])):.2f}" r="1.6" '
        f'fill="{_svg_color(str(row.get("ec1", "")))}" fill-opacity="0.16">'
        f'<title>{html.escape(str(row.get("enzyme_id", "")))} {html.escape(str(row.get("ec_label", "")))}</title></circle>'
        for row in point_rows
    )

    level_width = {2: 1.8, 3: 1.1, 4: 0.55}
    level_opacity = {2: 0.48, 3: 0.30, 4: 0.15}
    link_elements: list[str] = []
    for row in sorted(centroid_rows, key=lambda item: int(item["ec_level"])):
        level = int(row["ec_level"])
        if level <= 1:
            continue
        parent = centroid_lookup.get((level - 1, str(row["parent_label"])))
        if parent is None:
            continue
        color = _svg_color(str(row["ec1"]))
        link_elements.append(
            f'<line x1="{sx(float(parent["centroid_x"])):.2f}" y1="{sy(float(parent["centroid_y"])):.2f}" '
            f'x2="{sx(float(row["centroid_x"])):.2f}" y2="{sy(float(row["centroid_y"])):.2f}" '
            f'stroke="{color}" stroke-width="{level_width[level]:.2f}" stroke-opacity="{level_opacity[level]:.2f}">'
            f'<title>{html.escape(str(parent["ec_label"]))} -> {html.escape(str(row["ec_label"]))}</title></line>'
        )

    radius_by_level = {1: 8.0, 2: 5.2, 3: 3.3, 4: 2.0}
    opacity_by_level = {1: 0.98, 2: 0.84, 3: 0.72, 4: 0.52}
    centroid_elements: list[str] = []
    for row in sorted(centroid_rows, key=lambda item: int(item["ec_level"]), reverse=True):
        level = int(row["ec_level"])
        label = str(row["ec_label"])
        color = _svg_color(str(row["ec1"]))
        stroke = "#111827" if level == 1 else "#ffffff"
        centroid_elements.append(
            f'<circle cx="{sx(float(row["centroid_x"])):.2f}" cy="{sy(float(row["centroid_y"])):.2f}" '
            f'r="{radius_by_level[level]:.1f}" fill="{color}" fill-opacity="{opacity_by_level[level]:.2f}" '
            f'stroke="{stroke}" stroke-width="{1.2 if level <= 2 else 0.6}">'
            f'<title>EC{level} {html.escape(label)} count={int(row["count_in_projection"])}</title></circle>'
        )

    ec1_labels = "\n".join(
        f'<text x="{sx(float(row["centroid_x"])) + 10:.2f}" y="{sy(float(row["centroid_y"])) + 4:.2f}" '
        f'font-size="13" font-family="Arial" fill="#111827">EC1 {html.escape(str(row["ec_label"]))}</text>'
        for row in centroid_rows
        if int(row["ec_level"]) == 1
    )
    path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{left}" y="28" font-size="19" font-family="Arial">{html.escape(projection_name)} EC hierarchy overlay</text>
  <text x="{left}" y="47" font-size="12" font-family="Arial" fill="#475569">faint points are proteins; circles are EC centroids; colored lines connect EC1 -> EC2 -> EC3 -> EC4</text>
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>
  {point_elements}
  {"".join(link_elements)}
  {"".join(centroid_elements)}
  {ec1_labels}
  <text x="{(left + width - right) / 2:.2f}" y="{height - 16}" text-anchor="middle" font-size="13">component 1</text>
  <text x="18" y="{height / 2:.2f}" transform="rotate(-90 18 {height / 2:.2f})" text-anchor="middle" font-size="13">component 2</text>
</svg>
''',
    )


def _write_ec_hierarchy_outputs(
    output_dir: Path,
    projection_rows: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    all_centroid_rows: list[dict[str, Any]] = []
    for name, rows in projection_rows.items():
        centroid_rows = _projection_hierarchy_centroid_rows(name, rows)
        all_centroid_rows.extend(centroid_rows)
        _write_svg_hierarchy_overlay(
            plot_dir / f"{name}_ec_hierarchy_overlay.svg",
            projection_name=name,
            point_rows=rows,
            centroid_rows=centroid_rows,
        )
    write_csv(output_dir / "projection_ec_hierarchy_centroids.csv", all_centroid_rows)
    return all_centroid_rows


def _write_svg_plots(
    output_dir: Path,
    *,
    distance_rows: list[dict[str, Any]],
    projection_rows: dict[str, list[dict[str, Any]]],
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric in sorted({row["metric"] for row in distance_rows}):
        rows = [row for row in distance_rows if row["metric"] == metric and int(row["count"]) > 0]
        if rows:
            _write_svg_line_chart(
                plot_dir / f"distance_by_depth_{metric}.svg",
                rows,
                f"Distance by EC depth: {metric}",
            )
    for name, rows in projection_rows.items():
        if rows:
            for level in EC_PLOT_LEVELS:
                _write_svg_scatter(
                    plot_dir / f"{name}_{level}.svg",
                    rows,
                    f"{name} colored by {level.upper()}",
                    color_key=level,
                )
    _write_ec4_projection_separation_outputs(output_dir, projection_rows)
    _write_ec_hierarchy_outputs(output_dir, projection_rows)


def _shuffle_labels(labels: list[str], seed: int) -> list[str]:
    rng = random.Random(seed)
    shuffled = list(labels)
    rng.shuffle(shuffled)
    return shuffled


def _random_projector_representations(
    pooled: torch.Tensor,
    config: dict[str, Any],
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    input_normalization = str(config.get("projector_input_normalization", "none"))
    projector = LorentzEnzymeProjector(
        input_dim=int(config["input_dim"]),
        hyp_dim=int(config["hyp_dim"]),
        curvature=float(config["curvature"]),
        tangent_clip=config.get("tangent_clip", 5.0),
        tangent_clip_mode=str(config.get("tangent_clip_mode", "hard")),
        input_normalization=input_normalization,
        eps=float(config.get("eps", 1e-6)),
    )
    projector.eval()
    z_batches: list[torch.Tensor] = []
    tangent_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, pooled.shape[0], 512):
            z_hyp, z_tangent = projector(pooled[start : start + 512])
            z_batches.append(z_hyp.cpu())
            tangent_batches.append(z_tangent.cpu())
    return torch.cat(z_batches, dim=0), torch.cat(tangent_batches, dim=0)


def evaluate(args: argparse.Namespace) -> Path:
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else Path("results/hyperbolic_enzyme/eval") / checkpoint_path.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _device_from_config(args.device)
    checkpoint = _torch_load(checkpoint_path, device=torch.device("cpu"))
    attention_pooler, projector, config = _load_models(checkpoint, device=device)
    config.update(
        {
            "residue_embeddings_path": args.residue_embeddings_path or config["residue_embeddings_path"],
            "ec_labels_path": args.ec_labels_path or config["ec_labels_path"],
        }
    )
    dataset, dataset_metadata = _load_eval_dataset(config, args)
    extracted = extract_representations(
        dataset=dataset,
        attention_pooler=attention_pooler,
        projector=projector,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
    )

    enzyme_ids = extracted["enzyme_ids"]
    ec_labels = extracted["ec_labels"]
    pooled = extracted["pooled"].float()
    z_hyp = extracted["z_hyp"].float()
    z_tangent = extracted["z_tangent"].float()
    curvature = float(config["curvature"])
    eps = float(config.get("eps", 1e-6))

    metrics, radii = compute_manifold_metrics(
        z_hyp=z_hyp,
        z_tangent=z_tangent,
        curvature=curvature,
        eps=eps,
    )
    metrics.update(
        {
            "checkpoint_epoch": float(checkpoint.get("epoch", float("nan"))),
            "checkpoint_global_step": float(checkpoint.get("global_step", float("nan"))),
            "checkpoint_best_epoch_loss": float(checkpoint.get("best_epoch_loss", float("nan"))),
            "mean_attention_entropy": float(extracted["attention_entropy"].mean().item()),
            "mean_attention_mass_above_sleec_threshold": float(
                extracted["attention_mass_above_threshold"].mean().item()
            ),
            "mean_prior_strength": float(extracted["prior_strength"].mean().item()),
            "mean_sleec_logit": float(extracted["mean_sleec_logit"].mean().item()),
        }
    )

    lorentz_distance_fn = lambda left, right: _lorentz_pair_distance(  # noqa: E731
        left,
        right,
        curvature=curvature,
        eps=eps,
    )
    representations: dict[str, tuple[torch.Tensor, MetricFn]] = {
        "lorentz": (z_hyp, lorentz_distance_fn),
        "tangent_cosine": (z_tangent, _cosine_distance),
        "tangent_euclidean": (z_tangent, _euclidean_distance),
        "raw_pooled_cosine": (pooled, _cosine_distance),
    }
    knn_representations: dict[str, tuple[torch.Tensor, str]] = {
        "lorentz": (z_hyp, "lorentz"),
        "tangent_cosine": (z_tangent, "cosine"),
        "raw_pooled_cosine": (pooled, "cosine"),
    }

    if bool(args.include_negative_controls):
        random_z_hyp, random_z_tangent = _random_projector_representations(
            pooled,
            config,
            seed=int(args.seed) + 17,
        )
        representations["random_lorentz"] = (random_z_hyp, lorentz_distance_fn)
        representations["random_tangent_cosine"] = (random_z_tangent, _cosine_distance)
        knn_representations["random_lorentz"] = (random_z_hyp, "lorentz")

    pair_metrics, distance_rows, _ = compute_pair_distance_diagnostics(
        ec_labels=ec_labels,
        representations=representations,
        max_pairs=int(args.max_pairs),
        seed=int(args.seed),
        pair_batch_size=int(args.pair_batch_size),
    )
    metrics.update(pair_metrics)

    triplet_metrics, triplet_rows = compute_triplet_diagnostics(
        ec_labels=ec_labels,
        representations=representations,
        base_margin=float(config.get("base_margin", 0.05)),
        max_triplet_anchors=int(args.max_triplet_anchors),
        max_triplets_per_anchor=int(args.max_triplets_per_anchor),
        seed=int(args.seed),
        pair_batch_size=int(args.pair_batch_size),
    )
    metrics.update(triplet_metrics)

    k_values = sorted({int(k) for k in args.k_values if int(k) > 0})
    knn_metrics, knn_rows = compute_knn_diagnostics(
        ec_labels=ec_labels,
        representations=knn_representations,
        k_values=k_values,
        max_knn_samples=int(args.max_knn_samples),
        seed=int(args.seed),
        batch_size=int(args.knn_batch_size),
        curvature=curvature,
        eps=eps,
    )
    metrics.update(knn_metrics)

    if bool(args.include_negative_controls):
        shuffled_labels = _shuffle_labels(ec_labels, int(args.seed) + 29)
        shuffled_pair_metrics, _, _ = compute_pair_distance_diagnostics(
            ec_labels=shuffled_labels,
            representations={"lorentz": (z_hyp, lorentz_distance_fn)},
            max_pairs=int(args.max_pairs),
            seed=int(args.seed),
            pair_batch_size=int(args.pair_batch_size),
        )
        metrics.update({f"shuffled_labels_{key}": value for key, value in shuffled_pair_metrics.items()})
        shuffled_triplet_metrics, _ = compute_triplet_diagnostics(
            ec_labels=shuffled_labels,
            representations={"lorentz": (z_hyp, lorentz_distance_fn)},
            base_margin=float(config.get("base_margin", 0.05)),
            max_triplet_anchors=int(args.max_triplet_anchors),
            max_triplets_per_anchor=int(args.max_triplets_per_anchor),
            seed=int(args.seed),
            pair_batch_size=int(args.pair_batch_size),
        )
        metrics.update({f"shuffled_labels_{key}": value for key, value in shuffled_triplet_metrics.items()})
        shuffled_knn_metrics, _ = compute_knn_diagnostics(
            ec_labels=shuffled_labels,
            representations={"lorentz": (z_hyp, "lorentz")},
            k_values=k_values,
            max_knn_samples=int(args.max_knn_samples),
            seed=int(args.seed),
            batch_size=int(args.knn_batch_size),
            curvature=curvature,
            eps=eps,
        )
        metrics.update({f"shuffled_labels_{key}": value for key, value in shuffled_knn_metrics.items()})

    projection_rows = compute_projection_rows(
        enzyme_ids=enzyme_ids,
        ec_labels=ec_labels,
        pooled=pooled,
        z_tangent=z_tangent,
        z_hyp=z_hyp,
        radii=radii,
        max_visualization_points=int(args.max_visualization_points),
        max_pcoa_points=int(args.max_pcoa_points),
        seed=int(args.seed),
        curvature=curvature,
        eps=eps,
    )

    per_sample_rows = []
    for idx, enzyme_id in enumerate(enzyme_ids):
        prefixes = parse_ec_prefixes(ec_labels[idx])
        per_sample_rows.append(
            {
                "sample_index": idx,
                "enzyme_id": enzyme_id,
                "ec_label": ec_labels[idx],
                "known_ec_depth": known_ec_depth(ec_labels[idx]),
                "ec1": prefixes[0],
                "ec2": prefixes[1],
                "ec3": prefixes[2],
                "ec4": prefixes[3],
                "lorentz_radius": float(radii[idx]),
                "attention_entropy": float(extracted["attention_entropy"][idx].item()),
                "attention_mass_above_sleec_threshold": float(
                    extracted["attention_mass_above_threshold"][idx].item()
                ),
                "mean_sleec_logit": float(extracted["mean_sleec_logit"][idx].item()),
            }
        )

    write_csv(output_dir / "distance_by_ec_depth.csv", distance_rows)
    write_csv(output_dir / "triplet_metrics_by_gap.csv", triplet_rows)
    write_csv(output_dir / "knn_purity.csv", knn_rows)
    write_csv(output_dir / "per_sample_metrics.csv", per_sample_rows)
    for name, rows in projection_rows.items():
        write_csv(output_dir / f"{name}.csv", rows)

    metrics_payload = {
        "checkpoint": str(checkpoint_path),
        "config": config,
        "dataset": dataset_metadata,
        "metrics": metrics,
    }
    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(metrics_payload, handle, indent=2, default=_json_default, sort_keys=True)
    _write_visualization_notes(output_dir / "visualization_guide.md")
    _try_write_plots(output_dir, distance_rows=distance_rows, projection_rows=projection_rows)

    print(f"Wrote evaluation outputs to {output_dir}")
    print(
        " ".join(
            [
                f"lorentz_triplet_accuracy={metrics.get('lorentz_triplet_accuracy', math.nan):.4f}",
                f"raw_triplet_accuracy={metrics.get('raw_pooled_cosine_triplet_accuracy', math.nan):.4f}",
                f"lorentz_ec4_purity@10={metrics.get('lorentz_ec4_purity_at_10', math.nan):.4f}",
                f"raw_ec4_purity@10={metrics.get('raw_pooled_cosine_ec4_purity_at_10', math.nan):.4f}",
                f"spearman_depth=-dist={metrics.get('lorentz_spearman_depth_neg_distance', math.nan):.4f}",
                f"max_constraint_error={metrics.get('max_lorentz_constraint_error', math.nan):.6f}",
            ]
        )
    )
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--residue-embeddings-path", default=None)
    parser.add_argument("--data-path", "--ec-labels-path", dest="ec_labels_path", default=None)
    parser.add_argument("--sleec-logits-path", default=None)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--ec-column", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--max-protein-tokens", type=int, default=None)
    parser.add_argument("--max-pairs", type=int, default=200000)
    parser.add_argument("--pair-batch-size", type=int, default=8192)
    parser.add_argument("--max-triplet-anchors", type=int, default=5000)
    parser.add_argument("--max-triplets-per-anchor", type=int, default=16)
    parser.add_argument("--max-knn-samples", type=int, default=5000)
    parser.add_argument("--knn-batch-size", type=int, default=256)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 5, 10, 50])
    parser.add_argument("--max-visualization-points", type=int, default=5000)
    parser.add_argument("--max-pcoa-points", type=int, default=1000)
    parser.add_argument(
        "--include-negative-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
