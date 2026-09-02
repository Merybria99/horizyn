#!/usr/bin/env python3
"""Select the auxiliary weight from completed F3 no-EC calibration runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAL_ROOT = ROOT / "runs/reactzyme_f3_biological_no_ec_v1/calibration"
TAGS = ("lambda_0p03", "lambda_0p10", "lambda_0p30", "lambda_1p00")
WEIGHTS = {
    "lambda_0p03": 0.03,
    "lambda_0p10": 0.10,
    "lambda_0p30": 0.30,
    "lambda_1p00": 1.00,
}
METRIC = "val/mean_bidirectional_reactzyme_mrr"


def completed(tag: str) -> None:
    path = CAL_ROOT / tag / "status.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not events or events[-1].get("event") != "end" or events[-1].get("returncode") != 0:
        raise RuntimeError(f"Calibration is not complete and successful: {tag}")


def best_row(tag: str) -> dict[str, float | int | str]:
    paths = list((CAL_ROOT / tag / "logs").glob("protein_pooling_training/*/metrics.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one metrics.csv for {tag}, found {paths}")
    candidates = []
    with paths[0].open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get(METRIC):
                continue
            candidates.append(
                {
                    "tag": tag,
                    "weight": WEIGHTS[tag],
                    "epoch": int(row["epoch"]),
                    "macro": float(row[METRIC]),
                    "e2r": float(row["val/enzyme_to_reaction/reactzyme_mrr"]),
                    "r2e": float(row["val/reaction_to_enzyme/reactzyme_mrr"]),
                }
            )
    if not candidates:
        raise RuntimeError(f"No validation metric found for {tag}")
    return max(candidates, key=lambda row: (float(row["macro"]), -int(row["epoch"])))


def main() -> None:
    for tag in TAGS:
        completed(tag)
    rows = [best_row(tag) for tag in TAGS]
    # Prefer the smaller multiplier only for an exact numerical tie.
    selected = max(rows, key=lambda row: (float(row["macro"]), -float(row["weight"])))
    payload = {
        "schema_version": "f3_no_ec_aux_weight_selection_v1",
        "selection_metric": METRIC,
        "selection_rule": "maximum validation macro; smaller lambda breaks exact ties",
        "rows": rows,
        "selected": selected,
    }
    output = CAL_ROOT / "selected_aux_weight.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(selected["weight"])


if __name__ == "__main__":
    main()
