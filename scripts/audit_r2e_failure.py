#!/usr/bin/env python3
"""Audit reaction-to-enzyme failure modes for a pinned retrieval checkpoint."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from horizyn.benchmarks.retrieval import (
    BenchmarkTask,
    build_reaction_inputs,
    encode_reactions,
    encode_residue_targets,
    filter_candidate_keys_by_score_residue,
    group_pairs,
    load_benchmark_suite,
    load_candidate_keys_from_residue,
    load_repo_checkpoint,
    l2_normalize_embeddings,
    model_kind_from_config,
    needs_score_residue_embeddings,
    prepare_target_embedding_cache,
    read_pairs,
    release_target_embedding_cache_lock,
    score_embeddings,
    select_residue_h5,
    select_score_residue_h5,
    write_target_embedding_cache,
    _target_cache_base_metadata,
)
from horizyn.config import load_config
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset


FAMILIES = ("center", "cofactor", "transition")
RANK_TOP_K = (1, 10, 50, 100, 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit R->E retrieval geometry, candidate pools, and BioFP overlap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True, help="Run directory to audit")
    parser.add_argument("--checkpoint", required=True, help="Pinned Lightning checkpoint")
    parser.add_argument("--config", required=True, help="Pinned model config")
    parser.add_argument(
        "--suite",
        default="configs/benchmarks/retrieval_source_collapse_tests_reactiont5v2_unimol2_chiro_raw.yaml",
        help="Held-out benchmark suite",
    )
    parser.add_argument("--output-dir", required=True, help="Audit output directory")
    parser.add_argument("--target-cache-dir", default=None, help="Held-out target cache directory")
    parser.add_argument("--internal-target-cache-dir", default=None)
    parser.add_argument("--protein-embedding", default="prott5")
    parser.add_argument("--score-protein-embedding", default="prott5")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--target-batch-size", type=int, default=512)
    parser.add_argument("--score-query-batch-size", type=int, default=64)
    parser.add_argument("--top-k-predictions", type=int, default=50)
    parser.add_argument("--max-geometry-sample", type=int, default=10000)
    parser.add_argument("--geometry-neighbors", type=int, default=10)
    parser.add_argument("--biofp-threshold", type=float, default=0.0)
    parser.add_argument("--train-query-sample", type=int, default=512)
    parser.add_argument("--train-candidate-limit", type=int, default=12000)
    parser.add_argument("--skip-internal-gap", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def checkpoint_epoch(path: str | Path) -> int | None:
    match = re.search(r"epoch=(\d+)", str(path))
    return None if match is None else int(match.group(1))


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def format_float(value: Any) -> str:
    if value in {"", None}:
        return "nan"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class BioFPLookup:
    def __init__(self, npz_path: Path, vocab_path: Path, threshold: float = 0.0) -> None:
        self.npz_path = npz_path
        self.vocab_path = vocab_path
        self.threshold = float(threshold)
        payload = np.load(npz_path, allow_pickle=True)
        self.ids = [str(value) for value in payload["ids"]]
        self.index = {value: idx for idx, value in enumerate(self.ids)}
        self.targets = {family: payload[f"{family}_targets"].astype(np.float32) for family in FAMILIES}
        self.masks = {family: payload[f"{family}_mask"].astype(bool) for family in FAMILIES}
        raw_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        families = raw_vocab.get("families", raw_vocab)
        self.labels = {
            family: list(families.get(family, []))
            for family in FAMILIES
        }

    @staticmethod
    def normalize_id(protein_id: str) -> str:
        if protein_id.startswith("prot_"):
            return "uprot_" + protein_id[len("prot_") :]
        return protein_id

    def raw_id(self, protein_id: str) -> str:
        return self.normalize_id(str(protein_id))

    def has_id(self, protein_id: str) -> bool:
        return self.raw_id(protein_id) in self.index

    def signature(self, protein_id: str, family: str) -> tuple[int, ...] | None:
        idx = self.index.get(self.raw_id(protein_id))
        if idx is None or not bool(self.masks[family][idx]):
            return None
        values = self.targets[family][idx]
        return tuple(int(i) for i in np.flatnonzero(values > self.threshold))

    def signature_names(self, protein_id: str, family: str) -> str:
        signature = self.signature(protein_id, family)
        if signature is None:
            return ""
        labels = self.labels.get(family, [])
        return ";".join(labels[i] if i < len(labels) else str(i) for i in signature)

    def combined_signature(self, protein_id: str) -> tuple[tuple[int, ...], ...] | None:
        parts = []
        for family in FAMILIES:
            signature = self.signature(protein_id, family)
            if signature is None:
                return None
            parts.append(signature)
        return tuple(parts)


def jaccard(a: tuple[int, ...] | None, b: tuple[int, ...] | None) -> float | None:
    if a is None or b is None:
        return None
    a_set = set(a)
    b_set = set(b)
    if not a_set and not b_set:
        return 1.0
    union = a_set | b_set
    return float(len(a_set & b_set) / len(union)) if union else 0.0


def signature_overlap(
    lookup: BioFPLookup,
    protein_id: str,
    positives: list[str],
    family: str,
) -> dict[str, Any]:
    candidate_signature = lookup.signature(protein_id, family)
    positive_signatures = [
        sig for positive in positives if (sig := lookup.signature(positive, family)) is not None
    ]
    nonempty_positive_signatures = [sig for sig in positive_signatures if len(sig) > 0]
    jaccards = [
        value
        for positive_signature in positive_signatures
        if (value := jaccard(candidate_signature, positive_signature)) is not None
    ]
    return {
        f"{family}_known": candidate_signature is not None,
        f"{family}_nonempty": bool(candidate_signature),
        f"{family}_labels": lookup.signature_names(protein_id, family),
        f"positive_{family}_known_count": len(positive_signatures),
        f"positive_{family}_nonempty_count": len(nonempty_positive_signatures),
        f"same_{family}_as_any_positive": (
            candidate_signature is not None and candidate_signature in positive_signatures
        ),
        f"same_nonempty_{family}_as_any_positive": (
            candidate_signature is not None
            and len(candidate_signature) > 0
            and candidate_signature in nonempty_positive_signatures
        ),
        f"max_{family}_jaccard_to_positive": max(jaccards) if jaccards else "",
    }


def load_target_embeddings_for_task(
    *,
    task: BenchmarkTask,
    config: Any,
    checkpoint: Path,
    config_path: Path,
    module: torch.nn.Module | None,
    kind: str,
    protein_embedding: str,
    score_protein_embedding: str,
    device: str,
    target_batch_size: int,
    target_cache_dir: Path | None,
) -> tuple[list[str], torch.Tensor, dict[str, Any], torch.nn.Module | None]:
    if kind != "residue":
        raise ValueError("This audit currently supports residue-pooling checkpoints only")

    residue_h5 = select_residue_h5(task, protein_embedding)
    max_tokens = config.data.get("max_protein_tokens", 1024)
    truncation = config.data.get("protein_truncation", "ends_center")
    target_dataset = ResidueEmbedDataset(
        str(residue_h5),
        in_memory=False,
        max_tokens=max_tokens,
        truncation=truncation,
    )
    candidate_keys, candidate_stats = load_candidate_keys_from_residue(
        target_dataset,
        task.candidate_ids,
    )

    score_target_dataset = None
    score_residue_h5 = None
    if needs_score_residue_embeddings(config):
        score_residue_h5 = select_score_residue_h5(task, score_protein_embedding)
        score_target_dataset = ResidueEmbedDataset(
            str(score_residue_h5),
            in_memory=False,
            max_tokens=max_tokens,
            truncation=truncation,
        )
        candidate_keys, score_candidate_stats = filter_candidate_keys_by_score_residue(
            candidate_keys,
            score_target_dataset,
        )
        candidate_stats.update(score_candidate_stats)

    base_metadata = _target_cache_base_metadata(
        kind=kind,
        checkpoint=checkpoint,
        config_path=config_path,
        protein_embedding=protein_embedding,
        score_protein_embedding=score_protein_embedding,
        residue_h5=residue_h5,
        score_residue_h5=score_residue_h5,
        max_tokens=max_tokens,
        truncation=truncation,
    )
    target_embeds, cache_info = prepare_target_embedding_cache(
        target_cache_dir,
        base_metadata,
        candidate_keys,
        device=device,
        store_on_device=True,
    )
    if target_embeds is None:
        if module is None:
            module, loaded_kind = load_repo_checkpoint(checkpoint, config, device)
            if loaded_kind != kind:
                raise RuntimeError(f"Loaded model kind changed from {kind} to {loaded_kind}")
        try:
            target_embeds = encode_residue_targets(
                module,
                target_dataset,
                candidate_keys,
                device,
                target_batch_size,
                store_on_device=True,
                score_dataset=score_target_dataset,
            )
        except Exception:
            release_target_embedding_cache_lock(cache_info)
            raise
        write_target_embedding_cache(cache_info, base_metadata, candidate_keys, target_embeds)

    cache_info = dict(cache_info or {})
    cache_info.update(candidate_stats)
    return candidate_keys, target_embeds, cache_info, module


def positive_ranks(row_scores: torch.Tensor, positive_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    positive_tensor = torch.as_tensor(positive_indices, dtype=torch.long, device=row_scores.device)
    positive_scores = row_scores.index_select(0, positive_tensor)
    ranks = (row_scores.unsqueeze(0) > positive_scores.unsqueeze(1)).sum(dim=1) + 1
    order = torch.argsort(ranks)
    return ranks.index_select(0, order), positive_scores.index_select(0, order)


def audit_retrieval_task(
    *,
    task: BenchmarkTask,
    config: Any,
    checkpoint: Path,
    config_path: Path,
    module: torch.nn.Module,
    kind: str,
    biofp: BioFPLookup,
    protein_embedding: str,
    score_protein_embedding: str,
    device: str,
    query_batch_size: int,
    score_query_batch_size: int,
    target_batch_size: int,
    target_cache_dir: Path | None,
    top_k_predictions: int,
    collect_predictions: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], torch.nn.Module]:
    reaction_inputs = build_reaction_inputs(task, config)
    eval_pairs = read_pairs(task.pairs, task.reaction_id_col, task.protein_id_col)
    candidate_keys, target_embeds, cache_info, module = load_target_embeddings_for_task(
        task=task,
        config=config,
        checkpoint=checkpoint,
        config_path=config_path,
        module=module,
        kind=kind,
        protein_embedding=protein_embedding,
        score_protein_embedding=score_protein_embedding,
        device=device,
        target_batch_size=target_batch_size,
        target_cache_dir=target_cache_dir,
    )
    reaction_to_proteins, _protein_to_reactions = group_pairs(
        eval_pairs,
        allowed_reactions=set(reaction_inputs.keys),
        allowed_proteins=set(candidate_keys),
    )
    reaction_ids = sorted(reaction_to_proteins)
    candidate_to_idx = {protein_id: idx for idx, protein_id in enumerate(candidate_keys)}
    target_embeds = target_embeds.to(device)
    reaction_embeds = encode_reactions(
        module,
        reaction_inputs,
        reaction_ids,
        device,
        query_batch_size,
    )

    query_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    margin_values: list[float] = []
    best_ranks: list[float] = []
    mrr_values: list[float] = []
    top_hit_values = {k: [] for k in RANK_TOP_K}
    positive_counts: list[int] = []
    best_positive_scores: list[float] = []
    best_false_scores: list[float] = []

    with torch.inference_mode():
        for batch_start in range(0, len(reaction_ids), score_query_batch_size):
            batch_end = min(batch_start + score_query_batch_size, len(reaction_ids))
            batch_ids = reaction_ids[batch_start:batch_end]
            score_batch = score_embeddings(
                reaction_embeds[batch_start:batch_end],
                target_embeds,
                "cosine",
            )
            top_count = min(top_k_predictions, score_batch.shape[1])
            top_values, top_indices = torch.topk(score_batch, k=top_count, dim=1)
            top_values_cpu = top_values.detach().float().cpu().numpy()
            top_indices_cpu = top_indices.detach().cpu().numpy()
            for local_idx, reaction_id in enumerate(batch_ids):
                positives = reaction_to_proteins[reaction_id]
                positive_indices = [
                    candidate_to_idx[protein_id]
                    for protein_id in positives
                    if protein_id in candidate_to_idx
                ]
                if not positive_indices:
                    continue
                row_scores = score_batch[local_idx]
                ranks, positive_scores = positive_ranks(row_scores, positive_indices)
                best_rank = int(ranks[0].item())
                best_positive_score = float(positive_scores[0].item())
                positive_index_set = set(positive_indices)
                best_false_score = float("nan")
                for idx, value in zip(top_indices_cpu[local_idx], top_values_cpu[local_idx]):
                    if int(idx) not in positive_index_set:
                        best_false_score = float(value)
                        break
                if math.isnan(best_false_score):
                    best_false_score = float(row_scores.masked_fill(
                        torch.as_tensor(
                            [idx in positive_index_set for idx in range(row_scores.shape[0])],
                            dtype=torch.bool,
                            device=row_scores.device,
                        ),
                        -float("inf"),
                    ).max().item())
                margin = best_positive_score - best_false_score

                query_row = {
                    "task": task.name,
                    "dataset": task.dataset,
                    "task_label": task.task_label,
                    "reaction_id": reaction_id,
                    "num_candidates": len(candidate_keys),
                    "num_positives": len(positive_indices),
                    "best_rank": best_rank,
                    "best_rank_percentile": best_rank / max(1, len(candidate_keys)),
                    "mrr": 1.0 / best_rank,
                    "mean_positive_rank": float(ranks.float().mean().item()),
                    "best_positive_score": best_positive_score,
                    "mean_positive_score": float(positive_scores.float().mean().item()),
                    "best_false_score": best_false_score,
                    "score_margin": margin,
                }
                for k in RANK_TOP_K:
                    query_row[f"top_{k}"] = float(best_rank <= k)
                query_rows.append(query_row)
                best_ranks.append(float(best_rank))
                mrr_values.append(1.0 / best_rank)
                margin_values.append(margin)
                positive_counts.append(len(positive_indices))
                best_positive_scores.append(best_positive_score)
                best_false_scores.append(best_false_score)
                for k in RANK_TOP_K:
                    top_hit_values[k].append(float(best_rank <= k))

                if collect_predictions:
                    for rank_offset, (idx, value) in enumerate(
                        zip(top_indices_cpu[local_idx], top_values_cpu[local_idx]),
                        start=1,
                    ):
                        protein_id = candidate_keys[int(idx)]
                        is_positive = int(idx) in positive_index_set
                        top_row = {
                            "task": task.name,
                            "dataset": task.dataset,
                            "task_label": task.task_label,
                            "reaction_id": reaction_id,
                            "rank": rank_offset,
                            "protein_id": protein_id,
                            "score": float(value),
                            "is_positive": is_positive,
                            "num_positives": len(positive_indices),
                            "best_positive_rank": best_rank,
                            "best_positive_score": best_positive_score,
                            "best_false_score": best_false_score,
                            "score_margin": margin,
                        }
                        for family in FAMILIES:
                            top_row.update(signature_overlap(biofp, protein_id, positives, family))
                        combined = biofp.combined_signature(protein_id)
                        positive_combined = [
                            sig
                            for positive_id in positives
                            if (sig := biofp.combined_signature(positive_id)) is not None
                        ]
                        top_row["combined_known"] = combined is not None
                        top_row["same_combined_as_any_positive"] = (
                            combined is not None and combined in positive_combined
                        )
                        top_rows.append(top_row)

    candidate_stats = biofp_candidate_stats(biofp, candidate_keys)
    summary = {
        "task": task.name,
        "dataset": task.dataset,
        "task_label": task.task_label,
        "split": task.split,
        "num_queries": len(query_rows),
        "num_candidates": len(candidate_keys),
        "pair_count": len(eval_pairs),
        "mean_positives_per_query": mean([float(v) for v in positive_counts]),
        "median_positives_per_query": median([float(v) for v in positive_counts]),
        "p90_positives_per_query": percentile([float(v) for v in positive_counts], 90),
        "mean_best_rank": mean(best_ranks),
        "median_best_rank": median(best_ranks),
        "p90_best_rank": percentile(best_ranks, 90),
        "mean_best_rank_percentile": mean([row["best_rank_percentile"] for row in query_rows]),
        "mrr": mean(mrr_values),
        "mean_best_positive_score": mean(best_positive_scores),
        "mean_best_false_score": mean(best_false_scores),
        "mean_score_margin": mean(margin_values),
        "target_cache_status": cache_info.get("status", ""),
        **{f"top_{k}": mean(top_hit_values[k]) for k in RANK_TOP_K},
        **candidate_stats,
    }
    return summary, query_rows, top_rows, cache_info, module


def biofp_candidate_stats(biofp: BioFPLookup, protein_ids: list[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "biofp_id_overlap": sum(biofp.has_id(protein_id) for protein_id in protein_ids),
        "biofp_id_overlap_rate": mean([float(biofp.has_id(protein_id)) for protein_id in protein_ids]),
    }
    for family in FAMILIES:
        signatures = [biofp.signature(protein_id, family) for protein_id in protein_ids]
        known = [sig for sig in signatures if sig is not None]
        nonempty = [sig for sig in known if len(sig) > 0]
        stats[f"biofp_{family}_known_count"] = len(known)
        stats[f"biofp_{family}_known_rate"] = len(known) / max(1, len(protein_ids))
        stats[f"biofp_{family}_nonempty_count"] = len(nonempty)
        stats[f"biofp_{family}_nonempty_rate"] = len(nonempty) / max(1, len(protein_ids))
        stats[f"biofp_{family}_unique_signatures"] = len(set(known))
    return stats


def summarize_biofp_overlap(top_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in top_rows:
        by_task[str(row["task"])].append(row)
    for task, task_rows in sorted(by_task.items()):
        false_rows = [row for row in task_rows if not bool(row.get("is_positive"))]
        positive_rows = [row for row in task_rows if bool(row.get("is_positive"))]
        base = {
            "task": task,
            "top_prediction_rows": len(task_rows),
            "top_false_positive_rows": len(false_rows),
            "top_true_positive_rows": len(positive_rows),
            "same_combined_false_positive_rate": mean(
                [float(bool(row.get("same_combined_as_any_positive"))) for row in false_rows]
            ),
            "combined_known_false_positive_rate": mean(
                [float(bool(row.get("combined_known"))) for row in false_rows]
            ),
        }
        for family in FAMILIES:
            base[f"{family}_known_false_positive_rate"] = mean(
                [float(bool(row.get(f"{family}_known"))) for row in false_rows]
            )
            base[f"{family}_nonempty_false_positive_rate"] = mean(
                [float(bool(row.get(f"{family}_nonempty"))) for row in false_rows]
            )
            base[f"same_{family}_false_positive_rate"] = mean(
                [float(bool(row.get(f"same_{family}_as_any_positive"))) for row in false_rows]
            )
            base[f"same_nonempty_{family}_false_positive_rate"] = mean(
                [
                    float(bool(row.get(f"same_nonempty_{family}_as_any_positive")))
                    for row in false_rows
                ]
            )
            jaccards = [
                float(row[f"max_{family}_jaccard_to_positive"])
                for row in false_rows
                if row.get(f"max_{family}_jaccard_to_positive") not in {"", None}
            ]
            base[f"mean_max_{family}_jaccard_false_positive"] = mean(jaccards)
        rows.append(base)
    return rows


def geometry_for_candidate_set(
    *,
    task_name: str,
    target_embeds: torch.Tensor,
    candidate_keys: list[str],
    biofp: BioFPLookup,
    seq_dim: int,
    bio_dim: int,
    max_sample: int,
    neighbors: int,
    device: str,
) -> dict[str, Any]:
    embeds = l2_normalize_embeddings(target_embeds.detach().float()).to(device)
    row: dict[str, Any] = {
        "task": task_name,
        "num_candidates": len(candidate_keys),
        "embedding_dim": int(embeds.shape[1]),
    }
    if embeds.shape[1] >= seq_dim + bio_dim:
        seq_norm = embeds[:, :seq_dim].norm(dim=1).detach().cpu().numpy()
        bio_norm = embeds[:, seq_dim : seq_dim + bio_dim].norm(dim=1).detach().cpu().numpy()
        row.update(
            {
                "mean_seq_subspace_norm": float(seq_norm.mean()),
                "std_seq_subspace_norm": float(seq_norm.std()),
                "mean_biofp_subspace_norm": float(bio_norm.mean()),
                "std_biofp_subspace_norm": float(bio_norm.std()),
                "mean_biofp_to_seq_norm_ratio": float((bio_norm / np.maximum(seq_norm, 1e-12)).mean()),
            }
        )

    labeled_indices = [
        idx for idx, protein_id in enumerate(candidate_keys) if biofp.has_id(protein_id)
    ]
    row["biofp_labeled_candidates"] = len(labeled_indices)
    if not labeled_indices:
        return row
    sample_indices = sorted(labeled_indices, key=lambda idx: stable_key(candidate_keys[idx]))
    if max_sample > 0:
        sample_indices = sample_indices[: min(max_sample, len(sample_indices))]
    sample_tensor = torch.as_tensor(sample_indices, dtype=torch.long, device=device)
    sample_embeds = embeds.index_select(0, sample_tensor)
    sample_keys = [candidate_keys[idx] for idx in sample_indices]
    row["geometry_sample_size"] = len(sample_keys)
    row["geometry_neighbors"] = int(neighbors)

    signatures = {
        family: [biofp.signature(protein_id, family) for protein_id in sample_keys]
        for family in FAMILIES
    }
    combined = [biofp.combined_signature(protein_id) for protein_id in sample_keys]
    k = min(neighbors + 1, len(sample_keys))
    purity_values = {family: [] for family in FAMILIES}
    nonempty_purity_values = {family: [] for family in FAMILIES}
    combined_values: list[float] = []
    chunk_size = 512
    with torch.inference_mode():
        for start in range(0, len(sample_keys), chunk_size):
            end = min(start + chunk_size, len(sample_keys))
            scores = torch.matmul(sample_embeds[start:end], sample_embeds.T)
            top_idx = torch.topk(scores, k=k, dim=1).indices.detach().cpu().numpy()
            for local_row, neighbor_indices in enumerate(top_idx):
                anchor_idx = start + local_row
                filtered = [int(idx) for idx in neighbor_indices if int(idx) != anchor_idx][:neighbors]
                for family in FAMILIES:
                    anchor_signature = signatures[family][anchor_idx]
                    if anchor_signature is None:
                        continue
                    neighbor_signatures = [signatures[family][idx] for idx in filtered]
                    comparable = [sig for sig in neighbor_signatures if sig is not None]
                    if comparable:
                        purity_values[family].append(
                            sum(sig == anchor_signature for sig in comparable) / len(comparable)
                        )
                    if anchor_signature:
                        comparable_nonempty = [sig for sig in comparable if sig]
                        if comparable_nonempty:
                            nonempty_purity_values[family].append(
                                sum(sig == anchor_signature for sig in comparable_nonempty)
                                / len(comparable_nonempty)
                            )
                anchor_combined = combined[anchor_idx]
                if anchor_combined is not None:
                    comparable_combined = [
                        combined[idx] for idx in filtered if combined[idx] is not None
                    ]
                    if comparable_combined:
                        combined_values.append(
                            sum(sig == anchor_combined for sig in comparable_combined)
                            / len(comparable_combined)
                        )
    for family in FAMILIES:
        row[f"top{neighbors}_{family}_signature_purity"] = mean(purity_values[family])
        row[f"top{neighbors}_{family}_nonempty_signature_purity"] = mean(
            nonempty_purity_values[family]
        )
    row[f"top{neighbors}_combined_signature_purity"] = mean(combined_values)
    return row


def latest_metrics_row(metrics_csv: Path, max_epoch: int | None) -> dict[str, str] | None:
    if not metrics_csv.exists():
        return None
    rows = read_csv_rows(metrics_csv)
    candidates = []
    for row in rows:
        try:
            epoch = int(float(row.get("epoch", "")))
        except ValueError:
            continue
        if max_epoch is not None and epoch > max_epoch:
            continue
        if row.get("val/reaction_to_enzyme/mrr"):
            candidates.append(row)
    return candidates[-1] if candidates else None


def latest_train_loss_row(metrics_csv: Path, max_epoch: int | None) -> dict[str, str] | None:
    if not metrics_csv.exists():
        return None
    rows = read_csv_rows(metrics_csv)
    candidates = []
    for row in rows:
        try:
            epoch = int(float(row.get("epoch", "")))
        except ValueError:
            continue
        if max_epoch is not None and epoch > max_epoch:
            continue
        if row.get("train/loss_epoch") or row.get("train/loss_mlnce_epoch"):
            candidates.append(row)
    return candidates[-1] if candidates else None


def build_gap_rows_from_existing(
    *,
    run_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    epoch = checkpoint_epoch(checkpoint_path)
    gap_rows: list[dict[str, Any]] = []
    test_summary = run_dir / "results" / "original_chiro_test_epoch20_best" / "summary_wide.csv"
    if test_summary.exists():
        for row in read_csv_rows(test_summary):
            gap_rows.append(
                {
                    "source": str(test_summary),
                    "split_group": "heldout_test",
                    "epoch": epoch,
                    "dataset": row.get("dataset", ""),
                    "task": row.get("task", ""),
                    "task_label": row.get("task_label", ""),
                    "direction": row.get("direction", ""),
                    "candidate_pool_size": row.get("candidate_pool_size", ""),
                    "num_queries": row.get("num_queries", ""),
                    "top_1": row.get("top_1", ""),
                    "top_10": row.get("top_10", ""),
                    "top_50": row.get("top_50", ""),
                    "top_100": row.get("top_100", ""),
                    "top_1000": row.get("top_1000", ""),
                    "mrr": row.get("mrr", ""),
                    "mean_rank": row.get("mean_rank", ""),
                    "note": "pinned held-out benchmark summary",
                }
            )

    metrics_csv = (
        run_dir
        / "logs"
        / "joint_biofp_retrieval"
        / "protein_pooling_training"
        / "version_1"
        / "metrics.csv"
    )
    val_row = latest_metrics_row(metrics_csv, epoch)
    if val_row is not None:
        for prefix, direction in (
            ("val/reaction_to_enzyme", "reaction_to_enzyme"),
            ("val/enzyme_to_reaction", "enzyme_to_reaction"),
        ):
            gap_rows.append(
                {
                    "source": str(metrics_csv),
                    "split_group": "training_validation_log",
                    "epoch": val_row.get("epoch", ""),
                    "dataset": "ClipzymeEval",
                    "task": "clipzyme_validation",
                    "task_label": "validation",
                    "direction": direction,
                    "candidate_pool_size": "validation",
                    "num_queries": val_row.get(f"{prefix}/num_queries", ""),
                    "top_1": val_row.get(f"{prefix}/top_1", ""),
                    "top_10": val_row.get(f"{prefix}/top_10", ""),
                    "top_100": val_row.get(f"{prefix}/top_100", ""),
                    "top_1000": val_row.get(f"{prefix}/top_1000", ""),
                    "mrr": val_row.get(f"{prefix}/mrr", ""),
                    "mean_rank": val_row.get(f"{prefix}/mean_rank", ""),
                    "note": "latest validation row at or before pinned checkpoint epoch",
                }
            )
    train_row = latest_train_loss_row(metrics_csv, epoch)
    if train_row is not None:
        gap_rows.append(
            {
                "source": str(metrics_csv),
                "split_group": "train_log",
                "epoch": train_row.get("epoch", ""),
                "dataset": "train",
                "task": "joint_training_loss",
                "task_label": "loss",
                "direction": "loss",
                "candidate_pool_size": "",
                "num_queries": "",
                "mrr": "",
                "top_1": "",
                "top_10": "",
                "mean_rank": "",
                "note": (
                    "loss-only train logger row: "
                    f"train/loss_epoch={train_row.get('train/loss_epoch', '')}, "
                    f"train/loss_mlnce_epoch={train_row.get('train/loss_mlnce_epoch', '')}"
                ),
            }
        )
    return gap_rows


def copy_reactions_with_forward_ids(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8", newline="") as src, dest.open(
        "w", encoding="utf-8", newline=""
    ) as out:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {source}")
        writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            row = dict(row)
            row["reaction_id"] = f"{row['reaction_id']}_f"
            writer.writerow(row)


def write_pairs_with_forward_ids(rows: list[dict[str, str]], fieldnames: list[str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["reaction_id"] = f"{item['reaction_id']}_f"
            writer.writerow(item)


def write_id_file(ids: list[str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(ids) + "\n", encoding="utf-8")


def build_internal_gap_tasks(
    *,
    config: Any,
    output_dir: Path,
    train_query_sample: int,
    train_candidate_limit: int,
) -> list[BenchmarkTask]:
    internal_dir = output_dir / "internal_gap_inputs"
    copied_config = copy.deepcopy(config)
    copied_config.data.reaction_allow_missing_unimol2 = True
    copied_config.data.reaction_allow_missing_chiro = True
    copied_config.data.reaction_allow_missing_chirality = True
    copied_config.data.reaction_allow_missing_chienn = True

    tasks: list[BenchmarkTask] = []
    val_pairs_path = Path(config.data.test_pairs_path)
    val_reactions_path = Path(config.data.test_reactions_path)
    val_rows = read_csv_rows(val_pairs_path)
    val_fieldnames = list(val_rows[0].keys()) if val_rows else ["reaction_id", "protein_id"]
    val_pairs_out = internal_dir / "validation_pairs_forward.csv"
    val_rxns_out = internal_dir / "validation_reactions_forward.csv"
    val_candidates_out = internal_dir / "validation_candidate_ids.txt"
    write_pairs_with_forward_ids(val_rows, val_fieldnames, val_pairs_out)
    copy_reactions_with_forward_ids(val_reactions_path, val_rxns_out)
    write_id_file(sorted({row["protein_id"] for row in val_rows}), val_candidates_out)
    tasks.append(
        BenchmarkTask(
            name="internal_validation_clipzyme",
            task_type="retrieval",
            dataset="ClipzymeEval",
            task_label="validation",
            split="internal_validation_forward",
            pairs=val_pairs_out,
            reactions=val_rxns_out,
            candidate_ids=val_candidates_out,
            candidate_residue_h5={"prott5": Path(config.data.protein_residue_embeds_path)},
            candidate_score_residue_h5={"prott5": Path(config.data.protein_residue_embeds_path)},
            reaction_model_embeds_h5=Path(config.data.reaction_t5v2_embeds_path),
            reaction_unimol2_embeds_h5=Path(config.data.reaction_unimol2_embeds_path),
            reaction_chiro_embeds_h5=Path(config.data.reaction_chiro_embeds_path),
            directions=("reaction_to_enzyme",),
            top_k=RANK_TOP_K,
        )
    )

    train_pairs_path = Path(config.data.train_pairs_path)
    train_reactions_path = Path(config.data.train_reactions_path)
    train_rows_all = read_csv_rows(train_pairs_path)
    train_fieldnames = list(train_rows_all[0].keys()) if train_rows_all else ["reaction_id", "protein_id"]
    reaction_ids = sorted({row["reaction_id"] for row in train_rows_all}, key=stable_key)
    selected_reactions = set(reaction_ids[: max(0, train_query_sample)])
    sampled_rows = [row for row in train_rows_all if row["reaction_id"] in selected_reactions]
    positive_ids = sorted({row["protein_id"] for row in sampled_rows})
    train_candidate_file = Path(config.data.train_pairs_path).with_name("candidate_ids.txt")
    if train_candidate_file.exists():
        negative_pool = [
            line.strip()
            for line in train_candidate_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and line.strip() not in set(positive_ids)
        ]
    else:
        negative_pool = sorted({row["protein_id"] for row in train_rows_all} - set(positive_ids))
    negative_pool = sorted(negative_pool, key=stable_key)
    candidate_limit = max(len(positive_ids), int(train_candidate_limit))
    train_candidates = positive_ids + negative_pool[: max(0, candidate_limit - len(positive_ids))]

    train_pairs_out = internal_dir / "train_sample_pairs_forward.csv"
    train_rxns_out = internal_dir / "train_sample_reactions_forward.csv"
    train_candidates_out = internal_dir / "train_sample_candidate_ids.txt"
    write_pairs_with_forward_ids(sampled_rows, train_fieldnames, train_pairs_out)
    selected_rxn_rows = []
    for row in read_csv_rows(train_reactions_path):
        if row["reaction_id"] in selected_reactions:
            selected_rxn_rows.append(row)
    selected_rxn_raw = internal_dir / "train_sample_reactions_raw.csv"
    write_csv(selected_rxn_raw, selected_rxn_rows, fieldnames=list(selected_rxn_rows[0].keys()))
    copy_reactions_with_forward_ids(selected_rxn_raw, train_rxns_out)
    write_id_file(train_candidates, train_candidates_out)
    tasks.append(
        BenchmarkTask(
            name="internal_train_sample",
            task_type="retrieval",
            dataset="Train",
            task_label="train_sample",
            split="internal_train_forward_sample",
            pairs=train_pairs_out,
            reactions=train_rxns_out,
            candidate_ids=train_candidates_out,
            candidate_residue_h5={"prott5": Path(config.data.protein_residue_embeds_path)},
            candidate_score_residue_h5={"prott5": Path(config.data.protein_residue_embeds_path)},
            reaction_model_embeds_h5=Path(config.data.reaction_t5v2_embeds_path),
            reaction_unimol2_embeds_h5=Path(config.data.reaction_unimol2_embeds_path),
            reaction_chiro_embeds_h5=Path(config.data.reaction_chiro_embeds_path),
            directions=("reaction_to_enzyme",),
            top_k=RANK_TOP_K,
        )
    )
    return tasks, copied_config


def append_internal_gap_rows(
    gap_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> None:
    for summary in summaries:
        gap_rows.append(
            {
                "source": "audit_internal_eval",
                "split_group": summary.get("split", ""),
                "epoch": "",
                "dataset": summary.get("dataset", ""),
                "task": summary.get("task", ""),
                "task_label": summary.get("task_label", ""),
                "direction": "reaction_to_enzyme",
                "candidate_pool_size": summary.get("num_candidates", ""),
                "num_queries": summary.get("num_queries", ""),
                "top_1": summary.get("top_1", ""),
                "top_10": summary.get("top_10", ""),
                "top_50": summary.get("top_50", ""),
                "top_100": summary.get("top_100", ""),
                "top_1000": summary.get("top_1000", ""),
                "mrr": summary.get("mrr", ""),
                "mean_rank": summary.get("mean_best_rank", ""),
                "note": "internal exact/sampled audit using forward reaction IDs",
            }
        )


def write_report(
    *,
    output_dir: Path,
    checkpoint: Path,
    candidate_rows: list[dict[str, Any]],
    biofp_rows: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    notes: list[str],
) -> None:
    lines = [
        "# R->E Failure Audit",
        "",
        f"Checkpoint: `{checkpoint}`",
        "",
        "## Candidate Pool Difficulty",
        "",
    ]
    for row in candidate_rows:
        lines.append(
            "- "
            f"{row.get('task')}: MRR={format_float(row.get('mrr'))}, "
            f"top1={format_float(row.get('top_1'))}, "
            f"top10={format_float(row.get('top_10'))}, "
            f"median_rank={format_float(row.get('median_best_rank'))}, "
            f"candidates={row.get('num_candidates')}"
        )
    lines.extend(["", "## BioFP Overlap", ""])
    for row in biofp_rows:
        lines.append(
            "- "
            f"{row.get('task')}: false-positive same center="
            f"{format_float(row.get('same_center_false_positive_rate'))}, "
            f"same cofactor="
            f"{format_float(row.get('same_cofactor_false_positive_rate'))}, "
            f"same transition="
            f"{format_float(row.get('same_transition_false_positive_rate'))}, "
            f"same combined="
            f"{format_float(row.get('same_combined_false_positive_rate'))}"
        )
    lines.extend(["", "## Embedding Geometry", ""])
    for row in geometry_rows:
        lines.append(
            "- "
            f"{row.get('task')}: labeled={row.get('biofp_labeled_candidates')}/"
            f"{row.get('num_candidates')}, "
            f"bio/seq norm ratio={format_float(row.get('mean_biofp_to_seq_norm_ratio'))}, "
            f"center purity={format_float(row.get('top10_center_signature_purity'))}, "
            f"cofactor purity={format_float(row.get('top10_cofactor_signature_purity'))}, "
            f"transition purity={format_float(row.get('top10_transition_signature_purity'))}"
        )
    lines.extend(["", "## Gap Rows", ""])
    for row in gap_rows:
        if row.get("direction") == "loss":
            lines.append(f"- {row.get('split_group')} {row.get('task')}: {row.get('note')}")
        else:
            lines.append(
                "- "
                f"{row.get('split_group')} {row.get('task')} {row.get('direction')}: "
                f"MRR={row.get('mrr')}, top1={row.get('top_1')}, top10={row.get('top_10')}"
            )
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    target_cache_dir = (
        Path(args.target_cache_dir).resolve()
        if args.target_cache_dir
        else run_dir / "results" / "target_cache_epoch20_original_chiro"
    )
    internal_cache_dir = (
        Path(args.internal_target_cache_dir).resolve()
        if args.internal_target_cache_dir
        else output_dir / "target_cache_internal"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    config = load_config(str(config_path))
    kind = model_kind_from_config(config)
    biofp_path = Path(config.data.protein_biofp_targets_path)
    vocab_path = Path(config.data.protein_biofp_vocab_path)
    biofp = BioFPLookup(biofp_path, vocab_path, threshold=args.biofp_threshold)
    module, loaded_kind = load_repo_checkpoint(checkpoint, config, args.device)
    if loaded_kind != kind:
        raise RuntimeError(f"Loaded model kind changed from {kind} to {loaded_kind}")

    tasks = load_benchmark_suite(args.suite, Path.cwd())
    candidate_rows: list[dict[str, Any]] = []
    query_rows_all: list[dict[str, Any]] = []
    top_rows_all: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    cache_by_task: dict[str, tuple[list[str], torch.Tensor]] = {}

    for task in tasks:
        print(f"Auditing held-out task: {task.name}", flush=True)
        summary, query_rows, top_rows, _cache_info, module = audit_retrieval_task(
            task=task,
            config=config,
            checkpoint=checkpoint,
            config_path=config_path,
            module=module,
            kind=kind,
            biofp=biofp,
            protein_embedding=args.protein_embedding,
            score_protein_embedding=args.score_protein_embedding,
            device=args.device,
            query_batch_size=args.query_batch_size,
            score_query_batch_size=args.score_query_batch_size,
            target_batch_size=args.target_batch_size,
            target_cache_dir=target_cache_dir,
            top_k_predictions=args.top_k_predictions,
            collect_predictions=True,
        )
        candidate_rows.append(summary)
        query_rows_all.extend(query_rows)
        top_rows_all.extend(top_rows)

        candidate_keys, target_embeds, _cache_info_2, module = load_target_embeddings_for_task(
            task=task,
            config=config,
            checkpoint=checkpoint,
            config_path=config_path,
            module=module,
            kind=kind,
            protein_embedding=args.protein_embedding,
            score_protein_embedding=args.score_protein_embedding,
            device=args.device,
            target_batch_size=args.target_batch_size,
            target_cache_dir=target_cache_dir,
        )
        cache_key = hashlib.sha256("\0".join(candidate_keys).encode("utf-8")).hexdigest()
        if cache_key not in cache_by_task:
            geometry_rows.append(
                geometry_for_candidate_set(
                    task_name=task.name,
                    target_embeds=target_embeds,
                    candidate_keys=candidate_keys,
                    biofp=biofp,
                    seq_dim=int(config.model.biofp.seq_dim),
                    bio_dim=int(config.model.biofp.dim),
                    max_sample=args.max_geometry_sample,
                    neighbors=args.geometry_neighbors,
                    device=args.device,
                )
            )
            cache_by_task[cache_key] = (candidate_keys, target_embeds.detach().cpu())
        else:
            base = dict(geometry_rows[-1])
            base["task"] = task.name
            base["note"] = "same candidate pool as previous geometry row"
            geometry_rows.append(base)

    biofp_rows = summarize_biofp_overlap(top_rows_all)
    gap_rows = build_gap_rows_from_existing(
        run_dir=run_dir,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
    )

    if not args.skip_internal_gap:
        try:
            internal_tasks, internal_config = build_internal_gap_tasks(
                config=config,
                output_dir=output_dir,
                train_query_sample=args.train_query_sample,
                train_candidate_limit=args.train_candidate_limit,
            )
            internal_summaries: list[dict[str, Any]] = []
            for task in internal_tasks:
                print(f"Auditing internal gap task: {task.name}", flush=True)
                summary, query_rows, _top_rows, _cache_info, module = audit_retrieval_task(
                    task=task,
                    config=internal_config,
                    checkpoint=checkpoint,
                    config_path=config_path,
                    module=module,
                    kind=kind,
                    biofp=biofp,
                    protein_embedding=args.protein_embedding,
                    score_protein_embedding=args.score_protein_embedding,
                    device=args.device,
                    query_batch_size=args.query_batch_size,
                    score_query_batch_size=args.score_query_batch_size,
                    target_batch_size=args.target_batch_size,
                    target_cache_dir=internal_cache_dir,
                    top_k_predictions=args.top_k_predictions,
                    collect_predictions=False,
                )
                internal_summaries.append(summary)
                query_rows_all.extend(query_rows)
            append_internal_gap_rows(gap_rows, internal_summaries)
        except Exception as exc:
            notes.append(f"Internal gap audit failed: {type(exc).__name__}: {exc}")

    write_csv(output_dir / "candidate_pool_difficulty.csv", candidate_rows)
    write_csv(output_dir / "query_rank_audit.csv", query_rows_all)
    write_csv(output_dir / "biofp_overlap_summary.csv", biofp_rows)
    write_csv(output_dir / "embedding_geometry_summary.csv", geometry_rows)
    write_csv(output_dir / "gap_audit_summary.csv", gap_rows)
    write_csv(output_dir / "top_false_positives.csv", top_rows_all)
    parquet_path = output_dir / "top_false_positives.parquet"
    try:
        import pandas as pd

        top_frame = pd.DataFrame(top_rows_all)
        for column in top_frame.columns:
            if column.endswith("_labels"):
                top_frame[column] = top_frame[column].fillna("").astype(str)
            elif column.startswith("max_") and column.endswith("_jaccard_to_positive"):
                top_frame[column] = pd.to_numeric(top_frame[column], errors="coerce")
        top_frame.to_parquet(parquet_path, index=False)
    except Exception as exc:
        notes.append(f"Could not write Parquet top false positives: {type(exc).__name__}: {exc}")

    write_report(
        output_dir=output_dir,
        checkpoint=checkpoint,
        candidate_rows=candidate_rows,
        biofp_rows=biofp_rows,
        geometry_rows=geometry_rows,
        gap_rows=gap_rows,
        notes=notes,
    )
    print(f"Saved audit outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
