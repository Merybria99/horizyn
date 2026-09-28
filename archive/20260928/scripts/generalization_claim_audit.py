#!/usr/bin/env python3
"""Independently recompute headline claims without importing campaign metrics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def official_mrr(scores, pairs):
    """Count strictly better and earlier tied candidates for each positive."""
    outputs = {}
    for direction, matrix, edges in (
        ("reaction_to_enzyme", scores, pairs),
        ("enzyme_to_reaction", scores.T, pairs[:, ::-1]),
    ):
        positive = defaultdict(set)
        for q, c in edges:
            positive[int(q)].add(int(c))
        result = np.zeros(len(matrix), dtype=np.float64)
        candidate_indices = np.arange(matrix.shape[1])
        for q, candidates in positive.items():
            row = matrix[q]
            reciprocal = []
            for c in sorted(candidates):
                rank = 1 + np.count_nonzero(row > row[c])
                rank += np.count_nonzero((row == row[c]) & (candidate_indices < c))
                reciprocal.append(1.0 / rank)
            result[q] = np.mean(reciprocal)
        outputs[direction] = result
    return outputs


def assay_metrics(rows, query_key):
    """Pairwise AUROC and uniform-score-tie reciprocal-rank expectation."""
    groups = defaultdict(list)
    for row in rows:
        groups[row[query_key]].append(row)
    mrr, auc, ap, tie_queries = [], [], [], 0
    for query in sorted(groups):
        records = groups[query]
        scores = np.array([float(r["score"]) for r in records])
        labels = np.array([int(r["label"]) for r in records], dtype=bool)
        assert labels.any() and (~labels).any()
        positive, negative = scores[labels], scores[~labels]
        auc.append(float(np.mean((positive[:, None] > negative[None, :]) +
                                 .5 * (positive[:, None] == negative[None, :]))))
        reciprocals = []
        for value in positive:
            start = 1 + np.count_nonzero(scores > value)
            end = start + np.count_nonzero(scores == value)
            reciprocals.append(np.mean(1.0 / np.arange(start, end)))
        mrr.append(float(np.mean(reciprocals)))
        # Threshold AP, with all members of a score tie included together.
        precisions, weights = [], []
        for value in np.unique(positive):
            mask = scores >= value
            precisions.append(np.count_nonzero(labels & mask) / np.count_nonzero(mask))
            weights.append(np.count_nonzero(positive == value) / len(positive))
        ap.append(float(np.dot(precisions, weights)))
        tie_queries += int(len(np.unique(scores)) < len(scores))
    return {"queries": len(groups), "all_positive_mrr": float(np.mean(mrr)),
            "auroc": float(np.mean(auc)), "average_precision": float(np.mean(ap)),
            "queries_with_any_score_tie": tie_queries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign.resolve()
    out = root / "claim_validation"
    out.mkdir(exist_ok=True)
    sources, checks = {}, []

    def read_json(relative):
        path = root / relative
        sources[str(path)] = sha(path)
        return json.loads(path.read_text())

    def close(name, actual, expected, tolerance=1e-7):
        error = float(abs(actual - expected))
        checks.append({"claim": name, "recomputed": float(actual),
                       "reported": float(expected), "absolute_error": error,
                       "tolerance": tolerance, "pass": error <= tolerance})

    summary = read_json("phase4/official_evaluation/summary.json")
    for method, results in summary["methods"].items():
        cells = [results[s][d]["reactzyme_mrr"]
                 for s in ("reaction_smi", "enzyme_smi", "time")
                 for d in ("reaction_to_enzyme", "enzyme_to_reaction")]
        close(f"{method}: six-cell macro", np.mean(cells), summary["six_cell_macro"][method], 1e-12)

    pairs_path = root / "features_test_reaction_smi/pairs.npz"
    sources[str(pairs_path)] = sha(pairs_path)
    pairs = np.load(pairs_path)["test"]
    metrics = {}
    specs = {"phase4_seed42": ("phase4", "selected"),
             "phase2_seed42": ("phase2", "selected"),
             "F3_fp64": ("phase4", "baseline_fp64")}
    for method, (phase, key) in specs.items():
        path = root / phase / "predictions/reaction_smi/seed42/scores.npz"
        sources[str(path)] = sha(path)
        scores = np.load(path)[key]
        assert scores.shape == (386, 14688) and np.isfinite(scores).all()
        computed = official_mrr(scores, pairs)
        metrics[method] = computed
        for direction, values in computed.items():
            close(f"{method}: Reaction-Sim {direction}", values.mean(),
                  summary["methods"][method]["reaction_smi"][direction]["reactzyme_mrr"])

    # Describe the regression using fixed groups; this does not select a model.
    catalog = read_json("features_test_reaction_smi/catalog.json")
    groups = defaultdict(list)
    for enzyme in range(14688):
        groups[int(pairs[pairs[:, 1] == enzyme, 0].min())].append(enzyme)
    old = metrics["phase2_seed42"]["enzyme_to_reaction"]
    new = metrics["phase4_seed42"]["enzyme_to_reaction"]
    contributions = []
    for reaction, enzymes in groups.items():
        contributions.append({"reaction_id": catalog["reactions"][reaction],
                              "enzyme_queries": len(enzymes),
                              "phase2_mrr": float(old[enzymes].mean()),
                              "phase4_mrr": float(new[enzymes].mean()),
                              "pooled_delta_contribution": float((new[enzymes] - old[enzymes]).sum() / len(old))})
    contributions.sort(key=lambda r: r["pooled_delta_contribution"])
    with (out / "reaction_e2r_group_contributions.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contributions[0]))
        writer.writeheader()
        writer.writerows(contributions)

    assays = read_json("phase4/aminotransferase_evaluation/summary.json")
    csv_path = root / "phase4/aminotransferase_evaluation/all_assay_scores.csv"
    sources[str(csv_path)] = sha(csv_path)
    with csv_path.open() as handle:
        records = list(csv.DictReader(handle))
    panel_checks = {}
    for method in specs:
        rows = [r for r in records if r["method"] == method]
        assert len(rows) == 450
        assert len({(r["query_id"], r["protein_id"]) for r in rows}) == 450
        assert sum(int(r["label"]) for r in rows) == 186
        assert sum(r["censored"] == "True" for r in rows) == 264
        panel_checks[method] = {}
        for direction, query_key in (("reaction_to_enzyme", "query_id"),
                                     ("enzyme_to_reaction", "protein_id")):
            actual = assay_metrics(rows, query_key)
            reported = assays["universes"]["full_25_primary"]["metrics"][method][direction]
            panel_checks[method][direction] = actual
            close(f"{method}: aminotransferase {direction} MRR", actual["all_positive_mrr"],
                  reported["all_queries"]["all_positive_mrr"], 1e-12)
            for metric in ("auroc", "average_precision"):
                close(f"{method}: aminotransferase {direction} {metric}", actual[metric],
                      reported["mixed_class_queries_only"][metric], 1e-12)

    result = {"schema": "independent_campaign_claim_audit_v1",
              "at_utc": datetime.now(timezone.utc).isoformat(),
              "method": "Independent NumPy/Python count-based formulas; no campaign evaluator imports.",
              "checks": checks, "all_pass": all(c["pass"] for c in checks),
              "aminotransferase": panel_checks, "sources": sources,
              "script_sha256": sha(__file__),
              "limits": "Verifies saved-score calculations, not the biological truth of labels, causal generalization, or competitor protocol equivalence."}
    (out / "independent_calculations.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checks": len(checks), "all_pass": result["all_pass"], "output": str(out)}))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
