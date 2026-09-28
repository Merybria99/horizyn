"""Post-evaluation E→R readout by fixed train-only sequence-hit thresholds."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/generalization_20260919_2251/esterase_audit/public_horizyn1_dev"
PROTEINS = ROOT / "runs/generalization_20260919_2251/esterase_audit/inputs/proteins.csv"


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    target = BASE / "homology/performance_strata.json"
    if target.exists():
        raise ValueError("Preserve existing stratum readout")
    summary_path, hits_path = BASE / "summary.json", BASE / "homology/hits.tsv"
    summary = json.loads(summary_path.read_text())
    with PROTEINS.open(newline="") as handle:
        proteins = [row["protein_id"] for row in csv.DictReader(handle)]
    with hits_path.open(newline="") as handle:
        hits = list(csv.DictReader(handle, delimiter="\t", fieldnames=("query", "target", "fident", "alnlen", "qcov", "tcov", "evalue", "bits")))
    close = {row["query"] for row in hits if float(row["fident"]) >= 0.3 and float(row["qcov"]) >= 0.8}
    if len(proteins) != 145 or len(close) != 70:
        raise ValueError("Changed audit population")
    models = {"Horizyn1_dev": summary["metrics"], **summary["prior_metrics"]}
    strata = {}
    for name, value in models.items():
        per = value["enzyme_to_reaction"]["per_query"]
        if len(per) != 145:
            raise ValueError("Incomplete E→R query metrics")
        strata[name] = {}
        for label, matches_close in (("below_30pct_or_below_80pct_coverage", False), ("at_least_30pct_and_80pct_coverage", True)):
            selected = [row for row in per if (proteins[row["query_index"]] in close) == matches_close]
            strata[name][label] = {
                "query_count": len(selected),
                "mixed_class_query_count": sum(row["auroc"] is not None for row in selected),
                "auroc": statistics.mean(row["auroc"] for row in selected if row["auroc"] is not None),
                "average_precision": statistics.mean(row["average_precision"] for row in selected if row["average_precision"] is not None),
            }
    report = {
        "schema": "esterase_horizyn1_exploratory_homology_strata_v1",
        "threshold": "MMseqs2 local hit to official supervised train-only FASTA: sequence identity >= 30% and query coverage >= 80%",
        "endpoints": "E→R measured activity ranking on original fixed 86-reaction pool; mixed-class query AUROC/AP",
        "post_evaluation": True,
        "limitations": "Descriptive strata chosen after panel evaluation; no homology proof for low-hit group and no independent family uncertainty.",
        "sources": {str(p.relative_to(ROOT)): sha256(p) for p in (summary_path, hits_path, PROTEINS)},
        "implementation_sha256": sha256(Path(__file__)),
        "strata": strata,
    }
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(target), "strata": strata}), flush=True)


if __name__ == "__main__":
    main()
