"""Small runtime utilities used by the ReactZyme ablation launcher."""

from __future__ import annotations

import argparse
import csv
import json
import math
import hashlib
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "variant",
    "label",
    "split",
    "run_id",
    "r2e_mrr",
    "r2e_first_positive_mrr",
    "r2e_top1",
    "r2e_top10",
    "r2e_top100",
    "e2r_mrr",
    "e2r_first_positive_mrr",
    "e2r_top1",
    "balanced_mrr",
    "balanced_first_positive_mrr",
    "eval_json",
]
CHECKPOINT_HASH_WORKERS = 12


def _protein_suffix(value: str) -> str:
    return value.split("_", 1)[1] if "_" in value else value


def build_train_ec_subset(
    *,
    train_pairs_path: Path,
    ec_source_path: Path,
    output_csv: Path,
    output_report: Path,
) -> dict[str, Any]:
    """Filter an EC catalog to proteins observed in one training split."""

    train_proteins: list[str] = []
    seen: set[str] = set()
    with train_pairs_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            protein_id = str(row.get("protein_id", "")).strip()
            if protein_id and protein_id not in seen:
                seen.add(protein_id)
                train_proteins.append(protein_id)

    suffix_to_ec: dict[str, set[str]] = defaultdict(set)
    with ec_source_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            protein_id = str(row.get("protein_id", "")).strip()
            ec_numbers = {
                value.strip() for value in str(row.get("ec_number", "")).split(";") if value.strip()
            }
            if protein_id and ec_numbers:
                suffix_to_ec[_protein_suffix(protein_id)].update(ec_numbers)

    rows = []
    for protein_id in train_proteins:
        ec_numbers = sorted(suffix_to_ec.get(_protein_suffix(protein_id), set()))
        if ec_numbers:
            rows.append({"protein_id": protein_id, "ec_number": ";".join(ec_numbers)})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protein_id", "ec_number"])
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "source_train_pairs": str(train_pairs_path),
        "source_ec_labels": str(ec_source_path),
        "train_proteins": len(train_proteins),
        "train_proteins_with_ec": len(rows),
        "coverage": (len(rows) / len(train_proteins)) if train_proteins else 0.0,
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def read_checkpoint_sidecar(path: Path) -> str | None:
    """Return a checkpoint only when the sidecar and referenced file are valid."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    checkpoint = payload.get("checkpoint", "")
    return checkpoint if checkpoint and Path(checkpoint).is_file() else None


def write_checkpoint_sidecar(path: Path, *, run_id: str, checkpoint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"run_id": run_id, "checkpoint": checkpoint}, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_hashes(entries: list[dict[str, Any]]) -> list[str]:
    if not entries:
        return []
    with ThreadPoolExecutor(
        max_workers=min(CHECKPOINT_HASH_WORKERS, len(entries))
    ) as executor:
        return list(
            executor.map(
                lambda entry: _sha256(Path(entry["checkpoint"])),
                entries,
            )
        )


def freeze_selection(
    *,
    plan_path: Path,
    output_path: Path,
    variants: set[str] | None,
) -> dict[str, Any]:
    """Freeze checkpoint identities before any ReactZyme test evaluation."""

    entries = []
    with plan_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if variants is not None and row["variant"] not in variants:
                continue
            sidecar = Path(row["eval_json"] + ".checkpoint.json")
            checkpoint = read_checkpoint_sidecar(sidecar)
            if checkpoint is None:
                raise FileNotFoundError(
                    f"No completed checkpoint sidecar for {row['run_id']}: {sidecar}"
                )
            checkpoint_path = Path(checkpoint)
            entries.append(
                {
                    "variant": row["variant"],
                    "label": row["label"],
                    "split": row["split"],
                    "run_id": row["run_id"],
                    "checkpoint": str(checkpoint_path),
                    "test_config": row["test_config"],
                    "eval_json": row["eval_json"],
                    "master_port": row["master_port"],
                }
            )
    if not entries:
        raise ValueError("Selection contains no runs")
    for entry, checkpoint_hash in zip(
        entries, _checkpoint_hashes(entries), strict=True
    ):
        entry["checkpoint_sha256"] = checkpoint_hash
    payload = {
        "schema_version": "reactzyme_frozen_selection_v1",
        "source_plan": str(plan_path),
        "source_plan_sha256": _sha256(plan_path),
        "variants": sorted({entry["variant"] for entry in entries}),
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def selection_rows(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "reactzyme_frozen_selection_v1":
        raise ValueError(f"Unsupported selection manifest: {path}")
    entries = payload.get("entries", [])
    for entry in entries:
        checkpoint = Path(entry["checkpoint"])
        if not checkpoint.is_file():
            raise ValueError(f"Checkpoint changed after selection was frozen: {checkpoint}")
    checkpoint_hashes = _checkpoint_hashes(entries)
    rows = []
    for entry, checkpoint_hash in zip(entries, checkpoint_hashes, strict=True):
        checkpoint = Path(entry["checkpoint"])
        if checkpoint_hash != entry["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed after selection was frozen: {checkpoint}")
        rows.append({key: str(value) for key, value in entry.items()})
    return rows


def list_plan_variants(plan_path: Path) -> list[str]:
    variants: list[str] = []
    with plan_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            variant = row["variant"]
            if variant not in variants:
                variants.append(variant)
    return variants


def _number(value: Any) -> float | None:
    if value == "" or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def _format_metric(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.4f}"


def _summary_rows(plan_path: Path) -> list[dict[str, Any]]:
    rows = []
    with plan_path.open(newline="", encoding="utf-8") as handle:
        for plan_row in csv.DictReader(handle, delimiter="\t"):
            eval_path = Path(plan_row["eval_json"])
            payload = (
                json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
            )
            r2e_mrr = payload.get(
                "reaction_to_enzyme/reactzyme_mrr",
                payload.get("reaction_to_enzyme/mrr", payload.get("mrr", "")),
            )
            r2e_first_positive_mrr = payload.get(
                "reaction_to_enzyme/first_positive_mrr",
                payload.get("reaction_to_enzyme/mrr", payload.get("mrr", "")),
            )
            e2r_mrr = payload.get(
                "enzyme_to_reaction/reactzyme_mrr",
                payload.get("enzyme_to_reaction/mrr", ""),
            )
            e2r_first_positive_mrr = payload.get(
                "enzyme_to_reaction/first_positive_mrr",
                payload.get("enzyme_to_reaction/mrr", ""),
            )
            balanced_mrr = payload.get(
                "balanced_reactzyme_mrr",
                payload.get("balanced_mrr", ""),
            )
            if balanced_mrr == "" and r2e_mrr != "" and e2r_mrr != "":
                balanced_mrr = (float(r2e_mrr) + float(e2r_mrr)) / 2.0
            balanced_first_positive_mrr = payload.get("balanced_mrr", "")
            if (
                balanced_first_positive_mrr == ""
                and r2e_first_positive_mrr != ""
                and e2r_first_positive_mrr != ""
            ):
                balanced_first_positive_mrr = (
                    float(r2e_first_positive_mrr) + float(e2r_first_positive_mrr)
                ) / 2.0
            rows.append(
                {
                    "variant": plan_row["variant"],
                    "label": plan_row["label"],
                    "split": plan_row["split"],
                    "run_id": plan_row["run_id"],
                    "r2e_mrr": r2e_mrr,
                    "r2e_first_positive_mrr": r2e_first_positive_mrr,
                    "r2e_top1": payload.get("reaction_to_enzyme/top_1", payload.get("top_1", "")),
                    "r2e_top10": payload.get(
                        "reaction_to_enzyme/top_10", payload.get("top_10", "")
                    ),
                    "r2e_top100": payload.get(
                        "reaction_to_enzyme/top_100", payload.get("top_100", "")
                    ),
                    "e2r_mrr": e2r_mrr,
                    "e2r_first_positive_mrr": e2r_first_positive_mrr,
                    "e2r_top1": payload.get("enzyme_to_reaction/top_1", ""),
                    "balanced_mrr": balanced_mrr,
                    "balanced_first_positive_mrr": balanced_first_positive_mrr,
                    "eval_json": str(eval_path),
                }
            )
    return rows


def write_summary(*, plan_path: Path, output_csv: Path, output_markdown: Path) -> None:
    """Write per-run and aggregate metrics from an ablation train plan."""

    rows = _summary_rows(plan_path)
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)

    aggregate = {}
    for variant, items in by_variant.items():
        r2e_values = [_number(item["r2e_mrr"]) for item in items]
        balanced_values = [_number(item["balanced_mrr"]) for item in items]
        r2e_values = [value for value in r2e_values if value is not None]
        balanced_values = [value for value in balanced_values if value is not None]
        if len(r2e_values) == 3 and len(balanced_values) == 3:
            mean_r2e = sum(r2e_values) / 3.0
            mean_balanced = sum(balanced_values) / 3.0
            worst_r2e = min(r2e_values)
            aggregate[variant] = {
                # One symmetric metric across all three official split protocols.
                # Variant selection is frozen before test evaluation; this value is
                # only the corresponding aggregate reported after unblinding.
                "primary_score": mean_balanced,
                "mean_r2e_mrr": mean_r2e,
                "mean_balanced_mrr": mean_balanced,
                "worst_r2e_mrr": worst_r2e,
            }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# ReactZyme Ablation Summary",
        "",
        "## Aggregate",
        "",
        "MRR columns use ReactZyme's all-positive paper definition. "
        "First-positive MRR is retained below only as a ranking diagnostic.",
        "",
        "| Variant | Primary | Mean R2E Paper MRR | Mean Balanced Paper MRR | "
        "Worst R2E Paper MRR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for variant, stats in sorted(
        aggregate.items(), key=lambda item: item[1]["primary_score"], reverse=True
    ):
        lines.append(
            f"| {variant} | {_format_metric(stats['primary_score'])} | "
            f"{_format_metric(stats['mean_r2e_mrr'])} | "
            f"{_format_metric(stats['mean_balanced_mrr'])} | "
            f"{_format_metric(stats['worst_r2e_mrr'])} |"
        )
    lines.append("")

    for split in sorted({row["split"] for row in rows}):
        lines.extend(
            [
                f"## {split}",
                "",
                "| Variant | R2E Paper MRR | R2E First+ MRR | R2E Top1 | "
                "R2E Top10 | R2E Top100 | E2R Paper MRR | E2R First+ MRR | "
                "E2R Top1 | Balanced Paper MRR | Balanced First+ MRR |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: |",
            ]
        )
        for row in (item for item in rows if item["split"] == split):
            lines.append(
                f"| {row['variant']} | {_format_metric(row['r2e_mrr'])} | "
                f"{_format_metric(row['r2e_first_positive_mrr'])} | "
                f"{_format_metric(row['r2e_top1'])} | {_format_metric(row['r2e_top10'])} | "
                f"{_format_metric(row['r2e_top100'])} | {_format_metric(row['e2r_mrr'])} | "
                f"{_format_metric(row['e2r_first_positive_mrr'])} | "
                f"{_format_metric(row['e2r_top1'])} | {_format_metric(row['balanced_mrr'])} | "
                f"{_format_metric(row['balanced_first_positive_mrr'])} |"
            )
        lines.append("")
    output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ec_parser = subparsers.add_parser("build-ec-subset")
    ec_parser.add_argument("train_pairs", type=Path)
    ec_parser.add_argument("ec_source", type=Path)
    ec_parser.add_argument("output_csv", type=Path)
    ec_parser.add_argument("output_report", type=Path)

    read_parser = subparsers.add_parser("read-checkpoint")
    read_parser.add_argument("sidecar", type=Path)

    write_parser = subparsers.add_parser("write-checkpoint")
    write_parser.add_argument("sidecar", type=Path)
    write_parser.add_argument("run_id")
    write_parser.add_argument("checkpoint")

    variants_parser = subparsers.add_parser("list-variants")
    variants_parser.add_argument("plan", type=Path)

    summary_parser = subparsers.add_parser("write-summary")
    summary_parser.add_argument("plan", type=Path)
    summary_parser.add_argument("output_csv", type=Path)
    summary_parser.add_argument("output_markdown", type=Path)

    freeze_parser = subparsers.add_parser("freeze-selection")
    freeze_parser.add_argument("plan", type=Path)
    freeze_parser.add_argument("output", type=Path)
    freeze_parser.add_argument("--variants", nargs="*", default=None)

    selection_parser = subparsers.add_parser("selection-rows")
    selection_parser.add_argument("selection", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build-ec-subset":
        report = build_train_ec_subset(
            train_pairs_path=args.train_pairs,
            ec_source_path=args.ec_source,
            output_csv=args.output_csv,
            output_report=args.output_report,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "read-checkpoint":
        checkpoint = read_checkpoint_sidecar(args.sidecar)
        if checkpoint:
            print(checkpoint)
    elif args.command == "write-checkpoint":
        write_checkpoint_sidecar(
            args.sidecar,
            run_id=args.run_id,
            checkpoint=args.checkpoint,
        )
    elif args.command == "list-variants":
        print(" ".join(list_plan_variants(args.plan)))
    elif args.command == "write-summary":
        write_summary(
            plan_path=args.plan,
            output_csv=args.output_csv,
            output_markdown=args.output_markdown,
        )
        print(f"Wrote {args.output_csv}")
        print(f"Wrote {args.output_markdown}")
    elif args.command == "freeze-selection":
        payload = freeze_selection(
            plan_path=args.plan,
            output_path=args.output,
            variants=None if args.variants is None else set(args.variants),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "selection-rows":
        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=[
                "variant", "label", "split", "run_id", "checkpoint",
                "checkpoint_sha256", "test_config", "eval_json", "master_port",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(selection_rows(args.selection))


if __name__ == "__main__":
    main()
