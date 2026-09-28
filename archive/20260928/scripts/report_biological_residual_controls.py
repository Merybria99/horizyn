#!/usr/bin/env python3
"""Paired R0/R1/R2/R3 reports; bootstrap uncertainty is conditional on seed 42."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

VARIANTS = ("R1_full", "R2_no_source_pretrain", "R3_unguided")
DIRECTIONS = ("enzyme_to_reaction", "reaction_to_enzyme")
METRICS = ("reactzyme_mrr", "first_positive_mrr", "top_1", "precision_at_10", "mean_rank")
KEYS = ["direction", "query_id"]


def bootstrap_difference(values: np.ndarray, *, seed: int = 42, repetitions: int = 2000) -> dict:
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Paired differences must be nonempty and finite")
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions)
    # Bounded temporary memory, even when the query catalog becomes large.
    batch = max(1, min(64, 1_000_000 // len(values)))
    for start in range(0, repetitions, batch):
        count = min(batch, repetitions - start)
        samples[start : start + count] = values[
            rng.integers(len(values), size=(count, len(values)))
        ].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "delta": float(values.mean()),
        "ci95": [float(low), float(high)],
        "n_queries": len(values),
    }


def selected_rows(path: Path, alpha: float) -> dict:
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not np.isclose(float(row["alpha"]), alpha, rtol=0, atol=1e-12):
                continue
            key = tuple(row[k] for k in KEYS)
            if key in result:
                raise ValueError(f"Duplicated selected-alpha query: {path}: {key}")
            for metric in METRICS:
                row[metric] = float(row[metric])
            result[key] = row
    if not result:
        raise ValueError(f"Missing/duplicated selected-alpha queries: {path}")
    return dict(sorted(result.items()))


def count_bin(value) -> str:
    if value is None or value == "":
        return "unknown"
    value = int(value)
    return (
        "0"
        if value == 0
        else "1" if value == 1 else "2-5" if value <= 5 else "6-20" if value <= 20 else ">20"
    )


def report(run_root: Path, query_metadata: Path | None = None) -> dict:
    frames, results = {}, {}
    for variant in VARIANTS:
        path = run_root / variant / "results/test.json"
        result = json.loads(path.read_text())
        selection = json.loads(path.with_name("validation.json").read_text())
        if (
            result.get("evaluation_split") != "test"
            or result.get("selection_metric") != "balanced_reactzyme_mrr"
            or result["best_alpha"] != selection["best_alpha"]
            or result["checkpoint"] != selection["checkpoint"]
        ):
            raise ValueError(f"Test selection/protocol mismatch: {path}")
        frames[variant] = selected_rows(path.with_suffix(".per_query.csv"), result["best_alpha"])
        baseline = selected_rows(path.with_suffix(".per_query.csv"), 0)
        if "R0_F3" not in frames:
            frames["R0_F3"] = baseline
        elif baseline.keys() != frames["R0_F3"].keys() or any(
            abs(row[metric] - frames["R0_F3"][key][metric]) > 1e-7
            for key, row in baseline.items()
            for metric in METRICS
        ):
            raise ValueError("Controls do not recover the same paired F3 baseline")
        if frames[variant].keys() != frames["R0_F3"].keys():
            raise ValueError(f"Query sets differ: {variant}")
        results[variant] = result

    metadata = {}
    if query_metadata:
        with query_metadata.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not set(KEYS) <= set(reader.fieldnames or []):
                raise ValueError("Metadata needs direction/query_id keys")
            columns = [c for c in reader.fieldnames if c.startswith("stratum_")]
            if not columns:
                raise ValueError("Metadata needs at least one stratum_* column")
            for row in reader:
                key = tuple(row[k] for k in KEYS)
                if key in metadata:
                    raise ValueError("Duplicated metadata query")
                metadata[key] = {c: row[c] or "unknown" for c in columns}

    table, subgroups = [], []
    for variant in ("R0_F3", *VARIANTS):
        frame = {key: dict(row) for key, row in frames[variant].items()}
        for key, row in frame.items():
            row.update(
                stratum_known_associations=count_bin(row["known_positive_count"]),
                stratum_training_associations=count_bin(row["train_known_association_count"]),
                stratum_unimol2_available=row["has_unimol2"] or "not_applicable",
                stratum_chiro_available=row["has_chiro"] or "not_applicable",
            )
            if query_metadata:
                if set(columns) & set(row):
                    raise ValueError("External metadata cannot overwrite built-in strata")
                row.update(metadata.get(key, {c: "unknown" for c in columns}))
        alpha = 0 if variant == "R0_F3" else results[variant]["best_alpha"]
        for direction in DIRECTIONS:
            part = {key: row for key, row in frame.items() if key[0] == direction}
            if not part:
                raise ValueError(f"No queries for direction: {direction}")
            table.append(
                {
                    "variant": variant,
                    "direction": direction,
                    "alpha": alpha,
                    "n_queries": len(part),
                    **{
                        name: float(np.mean([row[name] for row in part.values()]))
                        for name in METRICS
                    },
                }
            )
            for column in (c for c in next(iter(part.values())) if c.startswith("stratum_")):
                groups = defaultdict(list)
                for key, row in part.items():
                    groups[str(row[column])].append((key, row))
                for group, rows in sorted(groups.items()):
                    subgroups.append(
                        {
                            "variant": variant,
                            "direction": direction,
                            "stratum": column,
                            "group": str(group),
                            "n_queries": len(rows),
                            "low_support": len(rows) < 20,
                            "reactzyme_mrr": float(np.mean([r["reactzyme_mrr"] for _, r in rows])),
                            "delta_vs_F3": float(
                                np.mean(
                                    [
                                        r["reactzyme_mrr"] - frames["R0_F3"][k]["reactzyme_mrr"]
                                        for k, r in rows
                                    ]
                                )
                            ),
                        }
                    )
    comparisons = {}
    for left, right in [(v, "R0_F3") for v in VARIANTS] + [("R1_full", v) for v in VARIANTS[1:]]:
        label = f"{left}_minus_{right}"
        comparisons[label] = {}
        for direction in DIRECTIONS:
            differences = np.array(
                [
                    row["reactzyme_mrr"] - frames[right][key]["reactzyme_mrr"]
                    for key, row in frames[left].items()
                    if key[0] == direction
                ]
            )
            comparisons[label][direction] = bootstrap_difference(differences)
    metadata_coverage = {
        column: {
            direction: sum(
                str(metadata.get(key, {}).get(column, "unknown")).strip().lower()
                not in {"", "unknown", "nan", "none"}
                for key in frames["R0_F3"]
                if key[0] == direction
            )
            for direction in DIRECTIONS
        }
        for column in ("stratum_train_similarity", "stratum_annotation_coverage")
    }
    payload = {
        "schema": "biological_residual_controls_report_v1",
        "table": table,
        "paired_query_bootstrap": comparisons,
        "uncertainty_scope": "Query bootstrap conditional on seed 42; not training-seed uncertainty or proof of independence between related queries",
        "missing_analysis": [
            name
            for name, column in (
                ("training_similarity_bins", "stratum_train_similarity"),
                ("annotation_coverage_bins", "stratum_annotation_coverage"),
            )
            if not all(metadata_coverage[column].values())
        ],
        "metadata_coverage_queries": metadata_coverage,
        "query_metadata": str(query_metadata) if query_metadata else None,
    }
    output = run_root / "reports"
    output.mkdir(exist_ok=True)
    for name, rows in (("test_summary.csv", table), ("stratified_metrics.csv", subgroups)):
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Biological residual: matched reaction-smi controls",
        "",
        "All alphas are validation-selected. R0 uses the exact parent F3 checkpoint.",
        "",
        "| Variant | Direction | Alpha | Hit@1 | ReactZyme MRR | First+ MRR | P@10 | Mean rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['variant']} | {row['direction']} | {row['alpha']:g} | "
            + " | ".join(
                f"{row[key]:.4f}"
                for key in (
                    "top_1",
                    "reactzyme_mrr",
                    "first_positive_mrr",
                    "precision_at_10",
                    "mean_rank",
                )
            )
            + " |"
        )
    lines += ["", "Query-bootstrap differences and 95% intervals:", ""]
    for pair, directions in comparisons.items():
        for direction, stats in directions.items():
            lines.append(
                f"- {pair}, {direction}: {stats['delta']:+.4f} [{stats['ci95'][0]:+.4f}, {stats['ci95'][1]:+.4f}]"
            )
    lines += [
        "",
        payload["uncertainty_scope"] + ".",
        "Known association counts are annotation-based proxies, not complete promiscuity measurements.",
        "Similarity/annotation strata require audited training-relative metadata; missing values are never invented.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Report: {output / 'summary.md'}", flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--query-metadata", type=Path)
    args = parser.parse_args()
    report(args.run_root, args.query_metadata)
