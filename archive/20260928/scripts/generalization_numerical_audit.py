#!/usr/bin/env python3
"""Frozen F3 FP32 sensitivity and independent retrieval/tie audit.

This diagnostic never selects a method. It recomputes positive ranks by direct
NumPy comparisons, independently of the benchmark evaluator's Torch argsort.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generalization_export import atomic_json, digest, identity
from scripts.generalization_official_evaluate import SPLITS, DIRECTIONS, checked_official_truth, checked_scores


def summarize_ranks(ranks, anchor, count):
    counts = np.bincount(anchor, minlength=count)
    if (counts == 0).any():
        raise ValueError("Independent audit requires the complete positive query universe")
    per_query = np.bincount(anchor, weights=1 / ranks, minlength=count) / counts
    first = np.full(count, np.inf)
    np.minimum.at(first, anchor, ranks)
    summary = dict(reactzyme_mrr=float(per_query.mean()), first_positive_mrr=float((1 / first).mean()))
    summary.update({f"top_{k}": float((first <= k).mean()) for k in (1, 5, 10)})
    return summary, dict(reactzyme_mrr=per_query, first_rank=first)


def independent_ranks(matrix, anchor, candidate, batch_size=256):
    count = len(anchor)
    strict = np.empty(count, np.int64)
    equal = np.empty(count, np.int64)
    preceding = np.empty(count, np.int64)
    candidate_index = np.arange(matrix.shape[1])
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        rows = matrix[anchor[start:end]]
        thresholds = rows[np.arange(end - start), candidate[start:end]][:, None]
        ties = rows == thresholds
        strict[start:end] = (rows > thresholds).sum(1)
        equal[start:end] = ties.sum(1)
        preceding[start:end] = (ties & (candidate_index < candidate[start:end, None])).sum(1)
    ranks = dict(stable=1 + strict + preceding, optimistic=1 + strict,
                 average=1 + strict + (equal - 1) / 2, pessimistic=strict + equal)
    # Ties anywhere in the pool can be harmless; report separately from ties
    # involving a positive or the top-ranked score.
    any_tied_queries = top_tied_queries = adjacent_equal_pairs = 0
    for start in range(0, len(matrix), batch_size):
        ordered = np.sort(matrix[start:start + batch_size], axis=1)
        adjacent = ordered[:, 1:] == ordered[:, :-1]
        any_tied_queries += int(adjacent.any(1).sum())
        top_tied_queries += int((ordered[:, -1] == ordered[:, -2]).sum()) if matrix.shape[1] > 1 else 0
        adjacent_equal_pairs += int(adjacent.sum())
    summaries, per_query = {}, {}
    for policy, values in ranks.items():
        summaries[policy], per_query[policy] = summarize_ranks(values, anchor, len(matrix))
    incidence = dict(queries=len(matrix), positive_edges=count,
        queries_with_any_score_tie=any_tied_queries,
        queries_with_top_score_tie=top_tied_queries,
        equal_adjacent_candidate_pairs=adjacent_equal_pairs,
        positive_edges_with_tied_score=int((equal > 1).sum()),
        queries_with_positive_score_tie=int(len(np.unique(anchor[equal > 1]))),
        maximum_positive_score_tie_size=int(equal.max()))
    return dict(metrics_by_tie_policy=summaries, incidence=incidence,
        tie_policy_delta_from_stable={policy: {key: value - summaries["stable"][key] for key, value in values.items()}
                                     for policy, values in summaries.items() if policy != "stable"}), ranks, per_query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Numerical audits are immutable; use a fresh output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    freeze_path = args.campaign / "frozen_recipe.json"
    freeze_sha = digest(freeze_path)
    freeze = json.loads(freeze_path.read_text())
    if not freeze["frozen_before_held_out_evaluation"] or freeze["primary_seed"] != 42:
        raise ValueError("Expected the frozen primary seed42 campaign")
    manifest_path = args.campaign / "official_evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["freeze"]["sha256"] != freeze_sha:
        raise ValueError("Evaluation manifest and frozen recipe disagree")
    provenance = {str(path): identity(path, True) for path in (freeze_path, manifest_path)}
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    results = {}
    for split in SPLITS:
        directory = args.campaign / f"features_test_{split}"
        catalog, edges = checked_official_truth(directory, split, freeze_sha, provenance)
        shape = (len(catalog["reactions"]), len(catalog["proteins"]))
        methods = {}
        for label in ("F3", "combined_seed42"):
            entry = next(m for m in manifest["methods"] if m["label"] == label)["panels"][split]
            methods[label] = checked_scores(entry, args.campaign, directory / "catalog.json", shape, freeze_sha, provenance)
        feature_receipt = json.loads((directory / "feature_bundle_receipt.json").read_text())
        base_path = directory / "f3_features.npz"
        if digest(base_path) != feature_receipt["inputs"]["base"]["sha256"]:
            raise ValueError("Frozen parent feature checksum changed")
        provenance[str(base_path)] = identity(base_path, True)
        with np.load(base_path, allow_pickle=False) as base:
            enzymes = F.normalize(torch.tensor(base["proteins"], device=args.device), dim=1)
            reactions = F.normalize(torch.tensor(base["reactions"], device=args.device), dim=1)
        with torch.inference_mode():
            canonical = (reactions.double() @ enzymes.double().T).float().cpu().numpy()
            native = (reactions @ enzymes.T).cpu().numpy()
        if not np.array_equal(canonical, methods["F3"]):
            raise ValueError("Independent parent normalization/canonical dot does not reproduce frozen predictions exactly")
        methods["F3_native_fp32"] = native
        native_path = args.output / f"{split}_f3_native_fp32.npz"
        np.savez(native_path, scores=native)
        atomic_json(args.output / f"{split}_f3_native_fp32_receipt.json", dict(
            schema="frozen_f3_native_fp32_sensitivity_v1", freeze_sha256=freeze_sha,
            catalog=identity(directory / "catalog.json", True), base=identity(base_path, True),
            feature_receipt=identity(directory / "feature_bundle_receipt.json", True),
            output=identity(native_path, True), device=args.device, torch_version=torch.__version__,
            tf32_enabled=False, matmul_precision="highest", normalized_endpoints="Identical to canonical baseline",
            purpose="Preregistered numerical sensitivity only; no method selection"))
        results[split] = dict(native_fp32_score_difference=dict(
            maximum_absolute=float(np.abs(native - canonical).max()),
            changed_entries=int(np.count_nonzero(native != canonical)), total_entries=int(native.size)), methods={})
        for label, scores in methods.items():
            results[split]["methods"][label] = {}
            for direction in DIRECTIONS:
                r2e = direction == "reaction_to_enzyme"
                matrix = scores if r2e else scores.T
                anchor, candidate = (edges[:, 0], edges[:, 1]) if r2e else (edges[:, 1], edges[:, 0])
                output, ranks, per_query = independent_ranks(matrix, anchor, candidate)
                if label != "F3_native_fp32":
                    saved_path = args.evaluation / f"{label}_{split}_per_query.npz"
                    with np.load(saved_path, allow_pickle=False) as saved:
                        first_equal = np.array_equal(saved[f"{direction}__first_rank"], per_query["stable"]["first_rank"])
                        mrr_error = float(np.abs(saved[f"{direction}__reactzyme_mrr"] - per_query["stable"]["reactzyme_mrr"]).max())
                    if not first_equal or mrr_error > 5e-7:
                        raise ValueError("Independent rank calculation disagrees with official evaluation")
                    output["independent_validation"] = dict(first_ranks_exact=first_equal,
                        maximum_per_query_mrr_error=mrr_error, tolerance=5e-7,
                        note="Only float32 versus float64 reciprocal-rank aggregation rounding is allowed")
                    provenance[str(saved_path)] = identity(saved_path, True)
                results[split]["methods"][label][direction] = output
                np.savez_compressed(args.output / f"{split}_{label}_{direction}_positive_ranks.npz", **ranks)
            print(f"Independent rank/tie audit complete: {split} {label}", flush=True)
        results[split]["native_fp32_metric_delta"] = {
            direction: {metric: value - results[split]["methods"]["F3"][direction]["metrics_by_tie_policy"]["stable"][metric]
                        for metric, value in results[split]["methods"]["F3_native_fp32"][direction]["metrics_by_tie_policy"]["stable"].items()}
            for direction in DIRECTIONS}
    if digest(freeze_path) != freeze_sha:
        raise ValueError("Frozen recipe changed during audit")
    atomic_json(args.output / "summary.json", dict(schema="frozen_numerical_sensitivity_audit_v1",
        frozen_recipe_sha256=freeze_sha, results=results,
        tie_policy="Stable candidate-index order remains primary. Alternative policies are diagnostics only; average means reciprocal of the average rank, not average reciprocal over permutations.",
        native_fp32="Same normalized FP32 F3 endpoints, TF32 disabled, CUDA FP32 dot versus FP64 dot rounded to FP32; not new feature extraction.",
        model_selection="None; frozen primary recipe unchanged"))
    atomic_json(args.output / "complete.json", dict(inputs=provenance, source=identity(__file__, True),
        output=identity(args.output / "summary.json", True), freeze_sha256=freeze_sha))


if __name__ == "__main__":
    main()
