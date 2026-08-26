#!/usr/bin/env python3
"""Prepare and aggregate a validation-selected ReactZyme/TIGER comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("time", "enzyme_smi", "reaction_smi")
SEEDS = (7, 42, 137)
DIRECTIONS = {
    "enzyme_to_reaction": "E->R",
    "reaction_to_enzyme": "R->E",
}
TIGER_PAPER_URL = "https://arxiv.org/pdf/2605.24489"

# TIGER Table 1 point estimates. ReactZyme's MRR averages reciprocal rank over
# all positives for each query; it is not conventional first-positive MRR.
TIGER_RESULTS = {
    "TIGER (ESM2Text)": {
        "time": {
            "enzyme_to_reaction": {"top_1": 0.581, "reactzyme_mrr": 0.690},
            "reaction_to_enzyme": {"top_1": 0.454, "reactzyme_mrr": 0.366},
        },
        "enzyme_smi": {
            "enzyme_to_reaction": {"top_1": 0.931, "reactzyme_mrr": 0.956},
            "reaction_to_enzyme": {"top_1": 0.792, "reactzyme_mrr": 0.592},
        },
        "reaction_smi": {
            "enzyme_to_reaction": {"top_1": 0.416, "reactzyme_mrr": 0.518},
            "reaction_to_enzyme": {"top_1": 0.430, "reactzyme_mrr": 0.319},
        },
    },
    "TIGER (ProtT3)": {
        "time": {
            "enzyme_to_reaction": {"top_1": 0.583, "reactzyme_mrr": 0.683},
            "reaction_to_enzyme": {"top_1": 0.454, "reactzyme_mrr": 0.372},
        },
        "enzyme_smi": {
            "enzyme_to_reaction": {"top_1": 0.908, "reactzyme_mrr": 0.940},
            "reaction_to_enzyme": {"top_1": 0.784, "reactzyme_mrr": 0.579},
        },
        "reaction_smi": {
            "enzyme_to_reaction": {"top_1": 0.386, "reactzyme_mrr": 0.472},
            "reaction_to_enzyme": {"top_1": 0.428, "reactzyme_mrr": 0.337},
        },
    },
}

# TIGER Figure 4 / text-source analysis. This is the fairer contextual baseline
# for this repository's intentionally no-enzyme-text setting, but Table 1 stays
# the primary SOTA comparison.
TIGER_NO_TEXT = {
    "time": {
        "enzyme_to_reaction": {"top_1": 0.478, "reactzyme_mrr": 0.612},
        "reaction_to_enzyme": {"top_1": 0.358, "reactzyme_mrr": 0.306},
    },
    "enzyme_smi": {
        "enzyme_to_reaction": {"top_1": 0.887, "reactzyme_mrr": 0.932},
        "reaction_to_enzyme": {"top_1": 0.726, "reactzyme_mrr": 0.549},
    },
    "reaction_smi": {
        "enzyme_to_reaction": {"top_1": 0.256, "reactzyme_mrr": 0.343},
        "reaction_to_enzyme": {"top_1": 0.324, "reactzyme_mrr": 0.267},
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_rows(promotion_path: Path, plan_path: Path) -> tuple[str, list[dict[str, str]]]:
    promotion = _read_json(promotion_path)
    if promotion.get("stage") != "recipe_confirm":
        raise ValueError(
            "Paper comparison requires the three-seed recipe_confirm promotion; "
            f"got stage={promotion.get('stage')!r}"
        )
    winner = str(promotion.get("winner") or "")
    if not winner:
        raise ValueError(f"Promotion has no winner: {promotion_path}")

    with plan_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t") if row["variant"] == winner]
    expected = {(seed, split) for seed in SEEDS for split in SPLITS}
    actual = {(int(row["seed"]), row["split"]) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError(
            f"Selected recipe {winner} must have exactly the 3x3 seed/split grid; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return winner, sorted(rows, key=lambda row: (int(row["seed"]), SPLITS.index(row["split"])))


def _checkpoint_from_sidecar(path: Path) -> Path:
    payload = _read_json(path)
    checkpoint = Path(str(payload.get("checkpoint") or "")).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint}")
    return checkpoint


def build_jobs(
    promotion_path: Path,
    plan_path: Path,
    output_root: Path,
) -> tuple[str, list[dict[str, str]]]:
    winner, rows = _selected_rows(promotion_path, plan_path)
    jobs = []
    for row in rows:
        checkpoint = _checkpoint_from_sidecar(Path(row["checkpoint_sidecar"]))
        config = Path(row["test_config"]).resolve()
        if not config.is_file():
            raise FileNotFoundError(f"Test config is missing: {config}")
        seed = int(row["seed"])
        split = row["split"]
        jobs.append(
            {
                "run_id": row["run_id"],
                "variant": winner,
                "seed": str(seed),
                "split": split,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "config": str(config),
                "output": str((output_root / winner / f"seed{seed}" / f"{split}.json").resolve()),
            }
        )
    return winner, jobs


def write_jobs(args: argparse.Namespace) -> None:
    winner, jobs = build_jobs(args.promotion, args.plan, args.output_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(jobs[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(jobs)
    print(f"Prepared {len(jobs)} paper-test jobs for validation-selected recipe {winner}: {args.output}")


def _validate_result(payload: dict[str, Any], job: dict[str, str]) -> None:
    expected_metadata = {
        "evaluation_protocol": "paper_test_candidates",
        "ground_truth_pairs": "test_pairs_only",
        "direction": "both",
        "checkpoint": str(Path(job["checkpoint"]).resolve()),
        "config": str(Path(job["config"]).resolve()),
    }
    for key, expected in expected_metadata.items():
        value = payload.get(key)
        if key in {"checkpoint", "config"} and value:
            value = str(Path(str(value)).resolve())
        if value != expected:
            raise ValueError(
                f"{job['output']}: expected {key}={expected!r}, got {value!r}"
            )
    for prefix in DIRECTIONS:
        for metric in ("top_1", "reactzyme_mrr", "first_positive_mrr"):
            key = f"{prefix}/{metric}"
            value = payload.get(key)
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{job['output']}: invalid or missing metric {key}")


def validate_output(args: argparse.Namespace) -> None:
    job = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "output": str(args.output),
    }
    _validate_result(_read_json(args.output), job)


def _mean_std(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values),
        "n": len(values),
    }


def _fmt_stat(stat: dict[str, Any]) -> str:
    return f"{stat['mean']:.3f} +/- {stat['std']:.3f}"


def _fmt_point(value: float) -> str:
    return f"{value:.3f}"


def _table_header() -> tuple[list[str], list[str]]:
    columns = ["Method"]
    separator = [":--"]
    for split in SPLITS:
        label = split.replace("_", " ").title()
        for direction in DIRECTIONS.values():
            columns.extend((f"{label} {direction} Hit@1", f"{label} {direction} MRR"))
            separator.extend(("--:", "--:"))
    return columns, separator


def _paper_row(method: str, values: dict[str, Any], aggregated: bool) -> list[str]:
    row = [method]
    for split in SPLITS:
        for direction in DIRECTIONS:
            metrics = values[split][direction]
            if aggregated:
                row.extend((_fmt_stat(metrics["top_1"]), _fmt_stat(metrics["reactzyme_mrr"])))
            else:
                row.extend((_fmt_point(metrics["top_1"]), _fmt_point(metrics["reactzyme_mrr"])))
    return row


def _markdown_table(rows: list[list[str]], alignment: list[str]) -> str:
    return "\n".join(
        ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(alignment) + " |"]
        + ["| " + " | ".join(row) + " |" for row in rows[1:]]
    )


def aggregate(args: argparse.Namespace) -> None:
    winner, jobs = build_jobs(args.promotion, args.plan, args.output_root)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    checkpoints = []
    candidate_counts: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for job in jobs:
        output = Path(job["output"])
        if not output.is_file():
            raise FileNotFoundError(f"Missing paper-test result: {output}")
        payload = _read_json(output)
        _validate_result(payload, job)
        grouped[(job["split"], job["seed"])].append(payload)
        checkpoints.append(
            {
                "seed": int(job["seed"]),
                "split": job["split"],
                "checkpoint": job["checkpoint"],
                "checkpoint_sha256": job["checkpoint_sha256"],
                "evaluation": job["output"],
            }
        )
        candidate_counts[job["split"]]["enzymes"].add(int(payload["num_enzyme_candidates"]))
        candidate_counts[job["split"]]["reactions"].add(int(payload["num_reaction_candidates"]))

    aggregate_values: dict[str, Any] = {}
    for split in SPLITS:
        aggregate_values[split] = {}
        for direction in DIRECTIONS:
            aggregate_values[split][direction] = {}
            for metric in ("top_1", "reactzyme_mrr", "first_positive_mrr"):
                values = [
                    float(grouped[(split, str(seed))][0][f"{direction}/{metric}"])
                    for seed in SEEDS
                ]
                aggregate_values[split][direction][metric] = _mean_std(values)
        for candidate_type, values in candidate_counts[split].items():
            if len(values) != 1:
                raise ValueError(f"Candidate count changed across seeds for {split}/{candidate_type}: {values}")

    best_tiger = {}
    deltas = {}
    for split in SPLITS:
        best_tiger[split] = {}
        deltas[split] = {}
        for direction in DIRECTIONS:
            best_tiger[split][direction] = {}
            deltas[split][direction] = {}
            for metric in ("top_1", "reactzyme_mrr"):
                value = max(
                    method[split][direction][metric] for method in TIGER_RESULTS.values()
                )
                best_tiger[split][direction][metric] = value
                deltas[split][direction][metric] = (
                    aggregate_values[split][direction][metric]["mean"] - value
                )

    report = {
        "schema_version": "reactzyme_tiger_comparison_v1",
        "selection": {
            "variant": winner,
            "promotion": str(args.promotion.resolve()),
            "plan": str(args.plan.resolve()),
            "criterion": "validation-only recipe_confirm promotion",
            "test_set_used_for_selection": False,
            "seeds": list(SEEDS),
        },
        "metric_definitions": {
            "paper_mrr": "mean reciprocal rank over every positive candidate per query",
            "diagnostic_mrr": "reciprocal rank of the first positive candidate",
        },
        "candidate_counts": {
            split: {key: next(iter(values)) for key, values in counts.items()}
            for split, counts in candidate_counts.items()
        },
        "checkpoints": checkpoints,
        "ours": aggregate_values,
        "tiger_table_1": TIGER_RESULTS,
        "tiger_no_text_ablation": TIGER_NO_TEXT,
        "best_tiger": best_tiger,
        "delta_vs_best_tiger": deltas,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "method", "split", "direction", "top1_mean", "top1_std",
            "reactzyme_mrr_mean", "reactzyme_mrr_std", "first_positive_mrr_mean",
            "first_positive_mrr_std", "seeds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for split in SPLITS:
            for direction, direction_label in DIRECTIONS.items():
                metrics = aggregate_values[split][direction]
                writer.writerow(
                    {
                        "method": winner,
                        "split": split,
                        "direction": direction_label,
                        "top1_mean": metrics["top_1"]["mean"],
                        "top1_std": metrics["top_1"]["std"],
                        "reactzyme_mrr_mean": metrics["reactzyme_mrr"]["mean"],
                        "reactzyme_mrr_std": metrics["reactzyme_mrr"]["std"],
                        "first_positive_mrr_mean": metrics["first_positive_mrr"]["mean"],
                        "first_positive_mrr_std": metrics["first_positive_mrr"]["std"],
                        "seeds": len(SEEDS),
                    }
                )

    header, alignment = _table_header()
    paper_rows = [header]
    paper_rows.extend(_paper_row(name, values, aggregated=False) for name, values in TIGER_RESULTS.items())
    paper_rows.append(_paper_row(f"Ours ({winner}, 3 seeds)", aggregate_values, aggregated=True))

    diagnostic_rows = [["Split", "Direction", "Paper MRR", "First-positive MRR", "Difference"]]
    for split in SPLITS:
        for direction, direction_label in DIRECTIONS.items():
            paper = aggregate_values[split][direction]["reactzyme_mrr"]
            first = aggregate_values[split][direction]["first_positive_mrr"]
            diagnostic_rows.append(
                [
                    split,
                    direction_label,
                    _fmt_stat(paper),
                    _fmt_stat(first),
                    f"{first['mean'] - paper['mean']:+.3f}",
                ]
            )

    delta_rows = [["Split", "Direction", "Hit@1 delta", "MRR delta"]]
    for split in SPLITS:
        for direction, direction_label in DIRECTIONS.items():
            delta_rows.append(
                [
                    split,
                    direction_label,
                    f"{deltas[split][direction]['top_1']:+.3f}",
                    f"{deltas[split][direction]['reactzyme_mrr']:+.3f}",
                ]
            )

    no_text_rows = [["Split", "Direction", "Ours Hit@1", "TIGER no-text Hit@1", "Ours MRR", "TIGER no-text MRR"]]
    for split in SPLITS:
        for direction, direction_label in DIRECTIONS.items():
            ours = aggregate_values[split][direction]
            tiger = TIGER_NO_TEXT[split][direction]
            no_text_rows.append(
                [
                    split,
                    direction_label,
                    _fmt_stat(ours["top_1"]),
                    _fmt_point(tiger["top_1"]),
                    _fmt_stat(ours["reactzyme_mrr"]),
                    _fmt_point(tiger["reactzyme_mrr"]),
                ]
            )

    markdown = "\n".join(
        [
            "# Validation-selected ReactZyme comparison with TIGER",
            "",
            f"Recipe **{winner}** was selected using validation metrics only, before these test results were read. "
            "Our entries are mean +/- sample standard deviation over seeds 7, 42, and 137; TIGER entries are "
            "the point estimates reported in Table 1.",
            "",
            "## Paper-compatible test results",
            "",
            _markdown_table(paper_rows, alignment),
            "",
            "MRR above uses ReactZyme's all-positive definition. Hit@1 is positive-set aware.",
            "",
            "## Delta from the best TIGER variant",
            "",
            _markdown_table(delta_rows, [":--", ":--", "--:", "--:"]),
            "",
            "Positive values favor our selected recipe.",
            "",
            "## No-enzyme-text context",
            "",
            _markdown_table(no_text_rows, [":--", ":--", "--:", "--:", "--:", "--:"]),
            "",
            "TIGER's Table 1 systems use generated enzyme text. The no-text ablation is included as context because our selected recipe intentionally excludes enzyme text; it is not a replacement for the primary Table 1 comparison.",
            "",
            "## Metric diagnostic",
            "",
            _markdown_table(diagnostic_rows, [":--", ":--", "--:", "--:", "--:"]),
            "",
            "First-positive MRR is retained only as a ranking diagnostic and must not be compared to TIGER's MRR.",
            "",
            "## Selection provenance",
            "",
            f"- Promotion: `{args.promotion.resolve()}`",
            f"- Confirmation plan: `{args.plan.resolve()}`",
            f"- Recipe: `{winner}`",
            f"- Seeds: `{', '.join(map(str, SEEDS))}`",
            f"- Test used for selection: `false`",
        ]
    ) + "\n"
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown, encoding="utf-8")
    print(f"Wrote TIGER comparison for {winner}: {args.output_markdown}")


def _path(value: str) -> Path:
    return Path(value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    jobs_parser = subparsers.add_parser("write-jobs")
    jobs_parser.add_argument("--promotion", required=True, type=_path)
    jobs_parser.add_argument("--plan", required=True, type=_path)
    jobs_parser.add_argument("--output-root", required=True, type=_path)
    jobs_parser.add_argument("--output", required=True, type=_path)
    jobs_parser.set_defaults(func=write_jobs)

    validate_parser = subparsers.add_parser("validate-output")
    validate_parser.add_argument("--output", required=True, type=_path)
    validate_parser.add_argument("--checkpoint", required=True, type=_path)
    validate_parser.add_argument("--config", required=True, type=_path)
    validate_parser.set_defaults(func=validate_output)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--promotion", required=True, type=_path)
    aggregate_parser.add_argument("--plan", required=True, type=_path)
    aggregate_parser.add_argument("--output-root", required=True, type=_path)
    aggregate_parser.add_argument("--output-json", required=True, type=_path)
    aggregate_parser.add_argument("--output-tsv", required=True, type=_path)
    aggregate_parser.add_argument("--output-markdown", required=True, type=_path)
    aggregate_parser.set_defaults(func=aggregate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
