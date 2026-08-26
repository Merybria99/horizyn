#!/usr/bin/env python3
"""Validate and report the F0-F6 ReactZyme reaction-feature test ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.reactzyme_tiger_comparison import (
    DIRECTIONS,
    SPLITS,
    TIGER_NO_TEXT,
    TIGER_PAPER_URL,
    TIGER_RESULTS,
)


EXPECTED_LABELS = tuple(f"F{index}" for index in range(7))
PAPER_PROTOCOL = "paper_test_candidates"
PAPER_GROUND_TRUTH = "test_pairs_only"
PAPER_REACTION_EXPANSION = "canonical_forward_only"
METRICS = ("top_1", "reactzyme_mrr", "first_positive_mrr")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing required artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _resolve(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _metric(payload: dict[str, Any], key: str, source: Path) -> float:
    value = payload.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{source}: invalid or missing metric {key!r}: {value!r}")
    return float(value)


def _load_plan(plan_path: Path) -> list[dict[str, str]]:
    with plan_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = {(label, split) for label in EXPECTED_LABELS for split in SPLITS}
    actual = {(row.get("label", ""), row.get("split", "")) for row in rows}
    duplicates = len(actual) != len(rows)
    if duplicates or actual != expected or len(rows) != len(expected):
        raise ValueError(
            "Reaction-feature plan must contain exactly the F0-F6 x "
            f"{len(SPLITS)} split grid; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}, duplicates={duplicates}"
        )

    variants_by_label: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        variants_by_label[row["label"]].add(row["variant"])
        if int(row["wave"]) != int(row["label"][1:]):
            raise ValueError(
                f"{plan_path}: wave/label mismatch for {row['run_id']}: "
                f"wave={row['wave']}, label={row['label']}"
            )
    inconsistent = {
        label: variants for label, variants in variants_by_label.items() if len(variants) != 1
    }
    if inconsistent:
        raise ValueError(f"Each label must map to one variant: {inconsistent}")

    return sorted(
        rows,
        key=lambda row: (int(row["label"][1:]), SPLITS.index(row["split"])),
    )


def _validate_result(row: dict[str, str]) -> dict[str, Any]:
    eval_path = _resolve(row["eval_json"])
    config_path = _resolve(row["test_config"])
    sidecar_path = Path(f"{eval_path}.checkpoint.json")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing test config for {row['run_id']}: {config_path}")

    sidecar = _read_json(sidecar_path)
    if sidecar.get("run_id") != row["run_id"]:
        raise ValueError(
            f"{sidecar_path}: expected run_id={row['run_id']!r}, "
            f"got {sidecar.get('run_id')!r}"
        )
    checkpoint_path = _resolve(str(sidecar.get("checkpoint") or ""))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing selected checkpoint for {row['run_id']}: {checkpoint_path}"
        )

    payload = _read_json(eval_path)
    expected_metadata: dict[str, Any] = {
        "evaluation_protocol": PAPER_PROTOCOL,
        "ground_truth_pairs": PAPER_GROUND_TRUTH,
        "direction": "both",
        "reaction_query_expansion": PAPER_REACTION_EXPANSION,
    }
    for key, expected in expected_metadata.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"{eval_path}: expected {key}={expected!r}, got {payload.get(key)!r}"
            )
    for key, expected_path in (
        ("checkpoint", checkpoint_path),
        ("config", config_path),
    ):
        value = payload.get(key)
        if not value or _resolve(str(value)) != expected_path:
            raise ValueError(
                f"{eval_path}: expected {key}={str(expected_path)!r}, got {value!r}"
            )

    metrics: dict[str, dict[str, float]] = {}
    for direction in DIRECTIONS:
        metrics[direction] = {
            metric: _metric(payload, f"{direction}/{metric}", eval_path)
            for metric in METRICS
        }

    balanced = _metric(payload, "balanced_reactzyme_mrr", eval_path)
    expected_balanced = sum(
        metrics[direction]["reactzyme_mrr"] for direction in DIRECTIONS
    ) / len(DIRECTIONS)
    if not math.isclose(balanced, expected_balanced, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{eval_path}: balanced_reactzyme_mrr={balanced} does not equal "
            f"the directional mean {expected_balanced}"
        )

    candidate_counts = {}
    for key in ("num_enzyme_candidates", "num_reaction_candidates"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{eval_path}: invalid {key}={value!r}")
        candidate_counts[key] = value
    query_expectations = {
        "enzyme_to_reaction/num_queries": candidate_counts["num_enzyme_candidates"],
        "reaction_to_enzyme/num_queries": candidate_counts["num_reaction_candidates"],
    }
    for key, expected in query_expectations.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"{eval_path}: expected {key}={expected}, got {payload.get(key)!r}"
            )

    return {
        "label": row["label"],
        "variant": row["variant"],
        "description": row["description"],
        "split": row["split"],
        "run_id": row["run_id"],
        "metrics": metrics,
        "balanced_reactzyme_mrr": balanced,
        "candidate_counts": candidate_counts,
        "checkpoint": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "config": str(config_path),
        "evaluation": str(eval_path),
    }


def _best_tiger(split: str, direction: str, metric: str) -> float:
    return max(
        float(values[split][direction][metric]) for values in TIGER_RESULTS.values()
    )


def _load_directional_coverage(
    plan_path: Path,
    candidate_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    coverage = {}
    for split in SPLITS:
        path = plan_path.parent / "data" / split / "reaction_directional" / "schema.json"
        payload = _read_json(path)
        if payload.get("schema_version") != "reactzyme_directional_vectors_v1":
            raise ValueError(f"{path}: unsupported directional schema")
        split_coverage = {}
        for partition in ("train", "validation", "test"):
            values = payload.get("splits", {}).get(partition)
            if not isinstance(values, dict):
                raise ValueError(f"{path}: missing coverage for {partition}")
            num_reactions = values.get("num_reactions")
            num_resolved = values.get("num_resolved")
            fraction = values.get("coverage")
            if (
                isinstance(num_reactions, bool)
                or not isinstance(num_reactions, int)
                or num_reactions <= 0
                or isinstance(num_resolved, bool)
                or not isinstance(num_resolved, int)
                or not 0 <= num_resolved <= num_reactions
                or isinstance(fraction, bool)
                or not isinstance(fraction, (int, float))
                or not math.isclose(
                    float(fraction),
                    num_resolved / num_reactions,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"{path}: invalid directional coverage for {partition}: {values}"
                )
            split_coverage[partition] = {
                "num_reactions": num_reactions,
                "num_resolved": num_resolved,
                "coverage": float(fraction),
                "with_center": int(values.get("counts", {}).get("with_center", 0)),
            }
        expected_test_reactions = candidate_counts[split]["num_reaction_candidates"]
        if split_coverage["test"]["num_reactions"] != expected_test_reactions:
            raise ValueError(
                f"{path}: test reaction count "
                f"{split_coverage['test']['num_reactions']} does not match paper-test "
                f"candidate count {expected_test_reactions}"
            )
        coverage[split] = split_coverage
    return coverage


def build_report(plan_path: Path) -> dict[str, Any]:
    plan_path = _resolve(plan_path)
    records = [_validate_result(row) for row in _load_plan(plan_path)]

    candidate_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        split_counts = {
            json.dumps(record["candidate_counts"], sort_keys=True)
            for record in records
            if record["split"] == split
        }
        if len(split_counts) != 1:
            raise ValueError(
                f"Candidate universe changed across F0-F6 for split {split}: "
                f"{sorted(split_counts)}"
            )
        candidate_counts[split] = json.loads(next(iter(split_counts)))
    directional_coverage = _load_directional_coverage(plan_path, candidate_counts)

    variants: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    for label in EXPECTED_LABELS:
        label_records = [record for record in records if record["label"] == label]
        first = label_records[0]
        cells: dict[str, Any] = {}
        top1_values = []
        mrr_values = []
        cells_above_tiger = 0
        for record in label_records:
            split = record["split"]
            cells[split] = {}
            for direction, direction_label in DIRECTIONS.items():
                metrics = record["metrics"][direction]
                deltas = {
                    metric: metrics[metric] - _best_tiger(split, direction, metric)
                    for metric in ("top_1", "reactzyme_mrr")
                }
                cells[split][direction] = {
                    **metrics,
                    "delta_vs_best_tiger": deltas,
                }
                top1_values.append(metrics["top_1"])
                mrr_values.append(metrics["reactzyme_mrr"])
                cells_above_tiger += int(
                    deltas["top_1"] > 0.0 and deltas["reactzyme_mrr"] > 0.0
                )
                flat_rows.append(
                    {
                        "source": "ours",
                        "method": label,
                        "variant": first["variant"],
                        "split": split,
                        "direction": direction_label,
                        "top_1": metrics["top_1"],
                        "reactzyme_mrr": metrics["reactzyme_mrr"],
                        "first_positive_mrr": metrics["first_positive_mrr"],
                        "top_1_delta_vs_best_tiger": deltas["top_1"],
                        "reactzyme_mrr_delta_vs_best_tiger": deltas["reactzyme_mrr"],
                    }
                )
        variants[label] = {
            "variant": first["variant"],
            "description": first["description"],
            "seed": 42,
            "cells": cells,
            "descriptive_test_macro": {
                "top_1": sum(top1_values) / len(top1_values),
                "reactzyme_mrr": sum(mrr_values) / len(mrr_values),
                "cells_beating_best_tiger_on_both_metrics": cells_above_tiger,
                "num_cells": len(top1_values),
            },
            "artifacts": [
                {
                    key: record[key]
                    for key in (
                        "split",
                        "run_id",
                        "checkpoint",
                        "checkpoint_size_bytes",
                        "config",
                        "evaluation",
                    )
                }
                for record in label_records
            ],
        }

    for method, values in TIGER_RESULTS.items():
        for split in SPLITS:
            for direction, direction_label in DIRECTIONS.items():
                metrics = values[split][direction]
                flat_rows.append(
                    {
                        "source": "tiger_table_1",
                        "method": method,
                        "variant": "",
                        "split": split,
                        "direction": direction_label,
                        "top_1": metrics["top_1"],
                        "reactzyme_mrr": metrics["reactzyme_mrr"],
                        "first_positive_mrr": "",
                        "top_1_delta_vs_best_tiger": (
                            metrics["top_1"] - _best_tiger(split, direction, "top_1")
                        ),
                        "reactzyme_mrr_delta_vs_best_tiger": (
                            metrics["reactzyme_mrr"]
                            - _best_tiger(split, direction, "reactzyme_mrr")
                        ),
                    }
                )

    return {
        "schema_version": "reactzyme_reaction_feature_tiger_ablation_v1",
        "source_plan": str(plan_path),
        "status": {
            "complete": True,
            "expected_evaluations": len(EXPECTED_LABELS) * len(SPLITS),
            "validated_evaluations": len(records),
        },
        "protocol": {
            "evaluation_protocol": PAPER_PROTOCOL,
            "ground_truth_pairs": PAPER_GROUND_TRUTH,
            "reaction_query_expansion": PAPER_REACTION_EXPANSION,
            "direction": "both",
            "seed": 42,
            "test_set_used_for_ablation_comparison": True,
            "checkpoint_selection_metric": (
                "validation arithmetic mean of bidirectional first-positive MRR"
            ),
            "checkpoint_selection_consistent_across_f0_f6": True,
            "selection_warning": (
                "F0-F6 are a single-seed test ablation. Any best variant identified "
                "from this report is descriptive and post-hoc, not validation-selected."
            ),
            "enzyme_text": False,
            "tiger_table_1_uses_generated_enzyme_text": True,
        },
        "metric_definitions": {
            "paper_mrr": (
                "ReactZyme all-positive MRR: reciprocal ranks are averaged over "
                "every positive candidate for each query."
            ),
            "diagnostic_mrr": (
                "First-positive MRR is retained as a diagnostic and is not used "
                "for comparison with TIGER."
            ),
        },
        "candidate_counts": candidate_counts,
        "directional_feature_coverage": directional_coverage,
        "tiger_paper": TIGER_PAPER_URL,
        "tiger_table_1": TIGER_RESULTS,
        "tiger_no_text_ablation": TIGER_NO_TEXT,
        "variants": variants,
        "rows": flat_rows,
    }


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _table(rows: list[list[str]], alignment: list[str]) -> str:
    return "\n".join(
        ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(alignment) + " |"]
        + ["| " + " | ".join(row) + " |" for row in rows[1:]]
    )


def _table1_header() -> tuple[list[str], list[str]]:
    header = ["Method"]
    alignment = [":--"]
    for split in SPLITS:
        split_label = split.replace("_", " ").title()
        for direction_label in DIRECTIONS.values():
            header.extend(
                (
                    f"{split_label} {direction_label} Hit@1",
                    f"{split_label} {direction_label} MRR",
                )
            )
            alignment.extend(("--:", "--:"))
    return header, alignment


def _tiger_table_row(method: str, values: dict[str, Any]) -> list[str]:
    row = [method]
    for split in SPLITS:
        for direction in DIRECTIONS:
            row.extend(
                (
                    _fmt(values[split][direction]["top_1"]),
                    _fmt(values[split][direction]["reactzyme_mrr"]),
                )
            )
    return row


def _variant_table_row(label: str, variant: dict[str, Any]) -> list[str]:
    row = [label]
    for split in SPLITS:
        for direction in DIRECTIONS:
            cell = variant["cells"][split][direction]
            row.extend((_fmt(cell["top_1"]), _fmt(cell["reactzyme_mrr"])))
    return row


def _write_outputs(
    report: dict[str, Any],
    *,
    output_json: Path,
    output_csv: Path,
    output_markdown: Path,
) -> None:
    for output in (output_json, output_csv, output_markdown):
        output.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fields = [
        "source",
        "method",
        "variant",
        "split",
        "direction",
        "top_1",
        "reactzyme_mrr",
        "first_positive_mrr",
        "top_1_delta_vs_best_tiger",
        "reactzyme_mrr_delta_vs_best_tiger",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["rows"])

    header, alignment = _table1_header()
    paper_rows = [header]
    paper_rows.extend(
        _tiger_table_row(method, values)
        for method, values in report["tiger_table_1"].items()
    )
    paper_rows.extend(
        _variant_table_row(label, report["variants"][label])
        for label in EXPECTED_LABELS
    )

    aggregate_rows = [
        [
            "Variant",
            "Macro Hit@1",
            "Macro Paper MRR",
            "Cells above best TIGER on both",
            "RXNsim E->R Hit@1",
            "RXNsim E->R MRR",
            "Delta Hit@1",
            "Delta MRR",
        ]
    ]
    for label in EXPECTED_LABELS:
        variant = report["variants"][label]
        aggregate = variant["descriptive_test_macro"]
        focus = variant["cells"]["reaction_smi"]["enzyme_to_reaction"]
        aggregate_rows.append(
            [
                label,
                _fmt(aggregate["top_1"]),
                _fmt(aggregate["reactzyme_mrr"]),
                f"{aggregate['cells_beating_best_tiger_on_both_metrics']}/"
                f"{aggregate['num_cells']}",
                _fmt(focus["top_1"]),
                _fmt(focus["reactzyme_mrr"]),
                f"{focus['delta_vs_best_tiger']['top_1']:+.3f}",
                f"{focus['delta_vs_best_tiger']['reactzyme_mrr']:+.3f}",
            ]
        )

    variant_rows = [["Label", "Variant", "Change"]]
    for label in EXPECTED_LABELS:
        variant = report["variants"][label]
        variant_rows.append([label, variant["variant"], variant["description"]])

    coverage_rows = [
        ["Split", "Train active", "Validation active", "Test active", "Test center labels"]
    ]
    for split in SPLITS:
        values = report["directional_feature_coverage"][split]
        coverage_rows.append(
            [
                split,
                f"{values['train']['num_resolved']}/{values['train']['num_reactions']} "
                f"({values['train']['coverage']:.1%})",
                f"{values['validation']['num_resolved']}/"
                f"{values['validation']['num_reactions']} "
                f"({values['validation']['coverage']:.1%})",
                f"{values['test']['num_resolved']}/{values['test']['num_reactions']} "
                f"({values['test']['coverage']:.1%})",
                f"{values['test']['with_center']}/{values['test']['num_reactions']}",
            ]
        )

    markdown = "\n".join(
        [
            "# ReactZyme reaction-feature ablation vs TIGER",
            "",
            "All 21 F0-F6 evaluations were validated against the official paper-test "
            "candidate protocol with test-only ground-truth pairs. MRR is ReactZyme's "
            "all-positive paper metric, not first-positive MRR.",
            "",
            "## Table 1 format",
            "",
            _table(paper_rows, alignment),
            "",
            "TIGER values are its Table 1 point estimates. F0-F6 are single-seed "
            "(seed 42) test results and use no enzyme text; TIGER Table 1 uses "
            "generated enzyme text.",
            "",
            f"TIGER source: {report['tiger_paper']}",
            "",
            "## Descriptive ablation view",
            "",
            _table(
                aggregate_rows,
                [":--", "--:", "--:", "--:", "--:", "--:", "--:", "--:"],
            ),
            "",
            "The macro columns average all six split/direction cells. Deltas use the "
            "better value of TIGER (ESM2Text) and TIGER (ProtT3) independently for "
            "each metric. These are post-hoc test-set descriptions, not a model-selection "
            "criterion.",
            "",
            "## Configurations",
            "",
            _table(variant_rows, [":--", ":--", ":--"]),
            "",
            "## Directional feature coverage",
            "",
            _table(coverage_rows, [":--", "--:", "--:", "--:", "--:"]),
            "",
            "F5 and F6 apply their gated residual only to reactions marked active by "
            "this mapping. F6 appends train-vocabulary reaction-center features; rows "
            "without center labels retain the other F5 directional features.",
            "",
            "## Provenance",
            "",
            f"- Plan: `{report['source_plan']}`",
            f"- Validated evaluations: `{report['status']['validated_evaluations']}/"
            f"{report['status']['expected_evaluations']}`",
            f"- Evaluation protocol: `{report['protocol']['evaluation_protocol']}`",
            f"- Ground truth: `{report['protocol']['ground_truth_pairs']}`",
            f"- Seed: `{report['protocol']['seed']}`",
            "- Checkpoint selection: validation arithmetic mean of bidirectional "
            "first-positive MRR (identical for F0-F6)",
            "- TIGER table comparison: ReactZyme all-positive test MRR",
            "- Test used for this ablation comparison: `true`",
            "- First-positive MRR: diagnostic only and excluded from TIGER deltas",
        ]
    ) + "\n"
    output_markdown.write_text(markdown, encoding="utf-8")


def generate_report(
    *,
    plan_path: Path,
    output_json: Path,
    output_csv: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    report = build_report(plan_path)
    _write_outputs(
        report,
        output_json=output_json,
        output_csv=output_csv,
        output_markdown=output_markdown,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=_resolve, required=True)
    parser.add_argument("--plan", type=_resolve)
    parser.add_argument("--output-json", type=_resolve)
    parser.add_argument("--output-csv", type=_resolve)
    parser.add_argument("--output-markdown", type=_resolve)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_root = args.run_root / "eval"
    report = generate_report(
        plan_path=args.plan or args.run_root / "train_plan.tsv",
        output_json=args.output_json or eval_root / "tiger_comparison.json",
        output_csv=args.output_csv or eval_root / "tiger_comparison.csv",
        output_markdown=args.output_markdown or eval_root / "tiger_comparison.md",
    )
    print(
        "Wrote complete F0-F6 TIGER comparison: "
        f"{report['status']['validated_evaluations']} evaluations"
    )


if __name__ == "__main__":
    main()
