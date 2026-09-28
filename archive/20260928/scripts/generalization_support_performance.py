#!/usr/bin/env python3
"""Descriptive frozen-test performance by training-only chemical support."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign.resolve()
    phase = root / "phase2"
    source = phase / "support_distribution/per_query.csv"
    support = pd.read_csv(source)
    thresholds_path = phase / "support_distribution/summary.json"
    thresholds = json.loads(thresholds_path.read_text())
    rows, joined, provenance = [], [], [source, thresholds_path]
    rng = np.random.default_rng(20260920)
    for split in ("reaction_smi", "enzyme_smi", "time"):
        catalog_path = root / f"features_test_{split}/catalog.json"
        catalog = json.loads(catalog_path.read_text())
        methods = {}
        for method in ("F3_fp64", "phase2_seed42"):
            path = phase / f"official_evaluation/{method}_{split}_per_query.npz"
            provenance.append(path)
            with np.load(path, allow_pickle=False) as arrays:
                methods[method] = {key.split("__", 1)[1]: arrays[key] for key in arrays.files if key.startswith("reaction_to_enzyme__")}
        index = methods["F3_fp64"]["query_index"]
        if not np.array_equal(index, methods["phase2_seed42"]["query_index"]):
            raise ValueError("Methods have different query order")
        expected_ids = [catalog["reactions"][int(i)] for i in index]
        s = support[support["panel"] == f"{split}/official_test"].set_index("query_id")
        if len(s) != len(index) or not s.index.is_unique or set(s.index) != set(expected_ids):
            raise ValueError("Support and metric query identities differ")
        s = s.loc[expected_ids]
        cos = s["nearest_training_raw_cosine"].to_numpy()
        threshold = thresholds["splits"][split]["thresholds"]
        strata = np.where(cos < threshold["train_loo_p05"], "below_train_p05",
                          np.where(cos < threshold["train_loo_p25"], "train_p05_to_p25", "at_or_above_train_p25"))
        for i, rid in enumerate(expected_ids):
            joined.append(dict(split=split, reaction_id=rid, raw_support=cos[i], stratum=strata[i],
                F3_fp64_mrr=float(methods["F3_fp64"]["reactzyme_mrr"][i]),
                phase2_seed42_mrr=float(methods["phase2_seed42"]["reactzyme_mrr"][i])))
        for name in ("all", "below_train_p05", "train_p05_to_p25", "at_or_above_train_p25"):
            mask = np.ones(len(cos), bool) if name == "all" else strata == name
            count = int(mask.sum())
            row = dict(split=split, direction="reaction_to_enzyme", stratum=name, query_count=count,
                train_p05=threshold["train_loo_p05"], train_p25=threshold["train_loo_p25"])
            for metric in ("reactzyme_mrr", "top_1", "top_10"):
                a = methods["F3_fp64"][metric][mask].astype(float)
                b = methods["phase2_seed42"][metric][mask].astype(float)
                row[f"F3_fp64_{metric}"] = float(a.mean()) if count else None
                row[f"phase2_seed42_{metric}"] = float(b.mean()) if count else None
                row[f"delta_{metric}"] = float((b-a).mean()) if count else None
                if count:
                    draws = []
                    for _ in range(20):
                        draws.extend((b-a)[rng.integers(0, count, (500, count))].mean(1).tolist())
                    low, high = np.quantile(draws, [.025, .975]).tolist()
                else:
                    low, high = None, None
                row[f"delta_{metric}_query_ci_low"] = low
                row[f"delta_{metric}_query_ci_high"] = high
            rows.append(row)
        provenance.append(catalog_path)
    out = phase / "support_performance"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(joined).to_csv(out / "per_query.csv", index=False)
    (out / "provenance.json").write_text(json.dumps(dict(
        schema="phase2_posthoc_support_stratified_performance_v1",
        status="Descriptive post-evaluation audit; no model, checkpoint or threshold selected from test performance.",
        strata="Same-split training leave-one-out raw support p05 and p25, computed before this performance join.",
        uncertainty="10000 paired reaction-query bootstrap draws, seed20260920; descriptive unadjusted intervals, not independent confirmatory tests.",
        unchanged_candidate_universe=True,
        inputs=[dict(path=str(p), sha256=sha(p)) for p in provenance],
        script_sha256=sha(Path(__file__)),
    ), indent=2)+"\n")
    report = ["# Descriptive official R→E performance by raw training support", "",
        "Post-evaluation audit using the frozen primary and precision-matched F3. Strata use each split's training leave-one-out fifth and25th percentiles. Candidate pools are unchanged. No threshold or model is selected from these results.", "",
        "| Split | Support stratum | Queries | F3 MRR | Primary MRR | Delta | Paired query95% interval |",
        "|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        if not r["query_count"]:
            report.append(f"| {r['split']} | {r['stratum']} |0| — | — | — | — |")
        else:
            report.append(f"| {r['split']} | {r['stratum']} |{r['query_count']}|{r['F3_fp64_reactzyme_mrr']:.6f}|{r['phase2_seed42_reactzyme_mrr']:.6f}|{r['delta_reactzyme_mrr']:+.6f}|[{r['delta_reactzyme_mrr_query_ci_low']:.6f},{r['delta_reactzyme_mrr_query_ci_high']:.6f}]|")
    report.extend(["", "Intervals are descriptive, unadjusted and conditional on these test queries. Biological and chemical dependence remains. Low raw cosine is a representation-specific support measure, not proof of reaction novelty or a calibrated activity probability. The external assay panels remain necessary."])
    (out / "README.md").write_text("\n".join(report)+"\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
