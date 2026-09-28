#!/usr/bin/env python3
"""Paired P450 query comparison of a fresh F3 base against released arms."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "runs/enzymecage_p450_reproduction_20260918"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def measures(ranks: np.ndarray) -> dict[str, np.ndarray]:
    return {"top4": (ranks <= 4).astype(float),
            "top14": (ranks <= 14).astype(float),
            "top24": (ranks <= 24).astype(float),
            "mrr": 1 / ranks.astype(float)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    export = json.loads((args.export / "export.json").read_text())
    evaluation = json.loads((args.evaluation / "summary.json").read_text())
    if (export["schema"] != "fresh_matched_enzymecage_f3_p450_score_export_v1"
            or evaluation["schema"] != "fresh_matched_enzymecage_f3_p450_evaluation_v1"
            or evaluation["score_matrix_sha256"] != export["score_matrix_sha256"]):
        raise ValueError("Fresh score/evaluation lineage mismatch")
    scores_path = args.export / "scores.npz"
    if sha256(scores_path) != export["score_matrix_sha256"]:
        raise ValueError("Fresh scores changed")
    with np.load(scores_path, allow_pickle=False) as source:
        qids = source["reaction_ids"].tolist()
        pids = source["protein_ids"].tolist()
        scores = source["scores"].astype(np.float64)
    labels_path = ROOT / "runs/generalization_20260919_2251/p450_audit/labels.json"
    labels = json.loads(labels_path.read_text())["known_positive_ids"]
    pindex = {protein: i for i, protein in enumerate(pids)}
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int32)
    np.put_along_axis(ranks, order,
                       np.broadcast_to(np.arange(1, len(pids) + 1), order.shape), axis=1)
    fresh = np.array([min(ranks[i, pindex[p]] for p in labels[q])
                      for i, q in enumerate(qids)])
    fresh_metrics = measures(fresh)
    for name in ("top4", "top14", "top24"):
        if abs(float(fresh_metrics[name].mean()) - evaluation["raw_metrics"][name]) > 1e-12:
            raise ValueError("Fresh rank disagrees with official evaluation")
    baseline_path = PILOT / "comparison_per_query.csv"
    baselines = {}
    with baseline_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["evaluation"] != "raw_diagnostic" or row["method"] not in (
                    "pretrained", "p450_finetuned"):
                continue
            baselines.setdefault(row["method"], {})[row["reaction_id"]] = int(
                row["best_positive_rank"]
            )
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(qids), (10000, len(qids)))
    comparisons = {}
    for method, row_by_query in baselines.items():
        if set(row_by_query) != set(qids):
            raise ValueError("Comparator query IDs differ")
        original = np.asarray([row_by_query[q] for q in qids])
        ref_metrics = measures(original)
        comparisons[method] = {}
        for key, values in fresh_metrics.items():
            delta = values - ref_metrics[key]
            ci = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
            comparisons[method][key] = {
                "fresh": float(values.mean()),
                "comparator": float(ref_metrics[key].mean()),
                "paired_difference": float(delta.mean()),
                "query_bootstrap_95pct": ci.tolist(),
            }
    args.output.mkdir(parents=True)
    receipt = {
        "schema": "matched_enzymecage_f3_p450_paired_comparison_v1",
        "fresh_export_sha256": sha256(args.export / "export.json"),
        "fresh_evaluation_sha256": sha256(args.evaluation / "summary.json"),
        "baseline_per_query_sha256": sha256(baseline_path),
        "known_positive_labels_sha256": sha256(labels_path),
        "query_count": len(qids), "candidate_count": len(pids),
        "comparisons": comparisons,
        "bootstrap_seed": 42, "bootstrap_replicates": 10000,
        "limitation": "Exploratory same-panel paired query uncertainty; no training-seed or external-dataset uncertainty. P450-finetuned comparator has extra P450 supervision and belongs to a separate arm.",
        "test_metrics_used_for_model_selection": False,
        "source_code_sha256": sha256(Path(__file__)),
    }
    (args.output / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: {m: round(v[m]["paired_difference"], 6)
                           for m in ("top4", "top14", "top24")}
                      for k, v in comparisons.items()}), flush=True)


if __name__ == "__main__":
    main()
