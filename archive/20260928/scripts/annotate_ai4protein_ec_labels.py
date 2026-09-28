#!/usr/bin/env python3
"""Annotate the AI4Protein/EC CSV splits with UniProt EC labels.

The HuggingFace AI4Protein/EC dataset stores numeric class IDs in the ``label``
column. Its ``name`` column has the form ``pdb_chain-UniProtAccession``. This
script extracts that accession and joins against the same UniProt EC-label cache
format produced by ``scripts/download_uniprot_ec_labels.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET_DIR = Path("data/huggingface/AI4Protein_EC")
DEFAULT_EC_CACHE = DEFAULT_DATASET_DIR / "uniprot_ai4protein_ec_labels.csv"
COMPLETE_EC_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")


def extract_uniprot_accession(name: str) -> str:
    """Extract UniProt accession from AI4Protein row names.

    Examples:
        ``1s4d_A-P21631`` -> ``P21631``
        ``5agy_A-I1MJ34`` -> ``I1MJ34``
    """

    value = (name or "").strip()
    return value.rsplit("-", 1)[-1].strip() if "-" in value else value


def split_ec_numbers(value: str | None) -> tuple[list[str], list[str]]:
    """Return all EC strings and complete EC-4 labels from a UniProt EC cell."""

    all_ecs: list[str] = []
    complete_ecs: list[str] = []
    for item in re.split(r"[;,]", value or ""):
        ec = item.strip()
        if not ec:
            continue
        all_ecs.append(ec)
        if COMPLETE_EC_RE.fullmatch(ec):
            complete_ecs.append(ec)
    return all_ecs, complete_ecs


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def read_split_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = {"name", "aa_seq", "label"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return rows


def write_accession_csv(dataset_dir: Path, output_path: Path) -> dict[str, object]:
    split_counts: dict[str, Counter[str]] = {}
    all_accessions: list[str] = []
    for split in ("train", "valid", "test"):
        rows = read_split_rows(dataset_dir / f"{split}.csv")
        counter = Counter(extract_uniprot_accession(row["name"]) for row in rows)
        split_counts[split] = counter
        all_accessions.extend(counter.keys())

    unique_accessions = unique_preserve_order(all_accessions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["protein_id", "train_rows", "valid_rows", "test_rows", "total_rows"],
        )
        writer.writeheader()
        for accession in unique_accessions:
            train_rows = split_counts["train"][accession]
            valid_rows = split_counts["valid"][accession]
            test_rows = split_counts["test"][accession]
            writer.writerow(
                {
                    "protein_id": accession,
                    "train_rows": train_rows,
                    "valid_rows": valid_rows,
                    "test_rows": test_rows,
                    "total_rows": train_rows + valid_rows + test_rows,
                }
            )

    return {
        "unique_accessions": len(unique_accessions),
        "accession_csv": str(output_path),
        "split_unique_accessions": {
            split: len(counter) for split, counter in split_counts.items()
        },
    }


def read_ec_cache(path: Path) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"protein_id", "uniprot_accession", "ec_number"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            keys = [
                row.get("protein_id", "").strip(),
                row.get("uniprot_accession", "").strip(),
            ]
            for key in keys:
                if key and key not in lookup:
                    lookup[key] = row
    return lookup


def annotate_splits(dataset_dir: Path, ec_cache_path: Path, output_dir: Path) -> dict[str, object]:
    ec_lookup = read_ec_cache(ec_cache_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "dataset_dir": str(dataset_dir),
        "ec_cache_path": str(ec_cache_path),
        "output_dir": str(output_dir),
        "splits": {},
    }

    fieldnames = [
        "name",
        "uniprot_accession",
        "aa_seq",
        "label",
        "uniprot_entry_name",
        "uniprot_reviewed",
        "ec_number",
        "complete_ec_numbers",
        "has_complete_ec",
    ]
    for split in ("train", "valid", "test"):
        rows = read_split_rows(dataset_dir / f"{split}.csv")
        output_path = output_dir / f"{split}_with_ec.csv"
        counters = Counter()
        unique_with_complete: set[str] = set()
        with open(output_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                counters["rows"] += 1
                accession = extract_uniprot_accession(row["name"])
                ec_row = ec_lookup.get(accession)
                if ec_row:
                    counters["accession_mapped"] += 1
                ec_number = ec_row.get("ec_number", "") if ec_row else ""
                _, complete_ecs = split_ec_numbers(ec_number)
                if complete_ecs:
                    counters["rows_with_complete_ec"] += 1
                    unique_with_complete.add(accession)
                writer.writerow(
                    {
                        "name": row["name"],
                        "uniprot_accession": accession,
                        "aa_seq": row["aa_seq"],
                        "label": row["label"],
                        "uniprot_entry_name": ec_row.get("entry_name", "") if ec_row else "",
                        "uniprot_reviewed": ec_row.get("reviewed", "") if ec_row else "",
                        "ec_number": ec_number,
                        "complete_ec_numbers": "; ".join(complete_ecs),
                        "has_complete_ec": bool(complete_ecs),
                    }
                )
        counters["rows_without_complete_ec"] = counters["rows"] - counters["rows_with_complete_ec"]
        counters["accessions_with_complete_ec"] = len(unique_with_complete)
        summary["splits"][split] = {**counters, "output_path": str(output_path)}

    summary_csv_path = output_dir / "ec_label_coverage_summary.csv"
    with open(summary_csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "rows",
                "accession_mapped",
                "rows_with_complete_ec",
                "rows_without_complete_ec",
                "accessions_with_complete_ec",
                "output_path",
            ],
        )
        writer.writeheader()
        for split, values in summary["splits"].items():
            writer.writerow({"split": split, **values})
    summary["summary_csv"] = str(summary_csv_path)

    summary_json_path = output_dir / "ec_label_coverage_summary.json"
    with open(summary_json_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    summary["summary_json"] = str(summary_json_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--ec-cache-path", default=str(DEFAULT_EC_CACHE))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--accession-output",
        default=None,
        help="CSV of unique UniProt accessions to feed to download_uniprot_ec_labels.py",
    )
    parser.add_argument(
        "--write-accessions-only",
        action="store_true",
        help="Only write the accession CSV and skip EC annotation.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir
    accession_output = (
        Path(args.accession_output)
        if args.accession_output
        else output_dir / "ai4protein_uniprot_accessions.csv"
    )

    accession_summary = write_accession_csv(dataset_dir, accession_output)
    print(json.dumps(accession_summary, indent=2, sort_keys=True))
    if args.write_accessions_only:
        return

    summary = annotate_splits(dataset_dir, Path(args.ec_cache_path), output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
