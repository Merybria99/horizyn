#!/usr/bin/env python3
"""Build the Case1 restricted Homolog candidate pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = SCRIPT_DIR.parent / "sequence_pool/final_entry_sequences.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "candidate_pool"
ALLOWED_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYX")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sheet", "entry_id", "name", "status", "length", "sha256", "sequence"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        return [dict(row) for row in reader]


def select_homolog_rows(
    rows: Iterable[dict[str, str]], *, expected_count: int | None = 144
) -> list[dict[str, str]]:
    """Return validated, resolved Homolog rows while preserving workbook order."""

    selected = [
        dict(row)
        for row in rows
        if row.get("sheet") == "Homologs"
        and row.get("sequence", "").strip()
        and row.get("status", "").startswith("resolved")
    ]
    if expected_count is not None and len(selected) != expected_count:
        raise ValueError(f"Expected {expected_count} resolved Homolog rows, found {len(selected)}")

    identifiers = [row["entry_id"].strip() for row in selected]
    duplicate_ids = sorted(
        identifier for identifier, count in Counter(identifiers).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"Duplicate Homolog entry IDs: {', '.join(duplicate_ids)}")

    for row in selected:
        entry_id = row["entry_id"].strip()
        sequence = re.sub(r"\s+", "", row["sequence"].upper())
        invalid = sorted(set(sequence).difference(ALLOWED_RESIDUES))
        if invalid:
            raise ValueError(f"{entry_id} has unsupported residues: {''.join(invalid)}")
        if int(row["length"]) != len(sequence):
            raise ValueError(
                f"{entry_id} length mismatch: CSV={row['length']}, sequence={len(sequence)}"
            )
        actual_sha = _sequence_sha256(sequence)
        if row["sha256"] != actual_sha:
            raise ValueError(f"{entry_id} sequence SHA-256 does not match the source table")
        row["entry_id"] = entry_id
        row["sequence"] = sequence

    return selected


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_candidate_pool(
    source: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    expected_count: int | None = 144,
) -> dict[str, object]:
    """Create FASTA, metadata, ID order, and provenance files for F3 inference."""

    source = source.resolve()
    output_dir = output_dir.resolve()
    rows = select_homolog_rows(_load_rows(source), expected_count=expected_count)
    output_dir.mkdir(parents=True, exist_ok=True)

    ids_by_sha: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        ids_by_sha[row["sha256"]].append(row["entry_id"])

    metadata_fields = [
        "entry_id",
        "name",
        "excel_row",
        "status",
        "sequence_source",
        "parent_identifier",
        "applied_changes",
        "length",
        "sha256",
        "sequence_duplicate_count",
        "sequence_duplicate_ids",
    ]
    metadata_rows: list[dict[str, str]] = []
    for row in rows:
        duplicate_ids = ids_by_sha[row["sha256"]]
        metadata_rows.append(
            {
                field: row.get(field, "")
                for field in metadata_fields
                if not field.startswith("sequence_duplicate_")
            }
            | {
                "sequence_duplicate_count": str(len(duplicate_ids)),
                "sequence_duplicate_ids": ";".join(duplicate_ids),
            }
        )

    fasta_path = output_dir / "proteins.fasta"
    with fasta_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            safe_name = re.sub(r"\s+", " ", row["name"].strip())
            handle.write(f">{row['entry_id']} {safe_name}\n")
            sequence = row["sequence"]
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")

    ids_path = output_dir / "candidate_ids_prott5_order.txt"
    ids_path.write_text("".join(f"{row['entry_id']}\n" for row in rows), encoding="utf-8")
    metadata_path = output_dir / "proteins.csv"
    _write_csv(metadata_path, metadata_rows, metadata_fields)

    duplicate_groups = [ids for ids in ids_by_sha.values() if len(ids) > 1]
    manifest: dict[str, object] = {
        "schema_version": "case1_restricted_candidate_pool_v1",
        "source": str(source),
        "source_sha256": _file_sha256(source),
        "selection": "sheet == Homologs and status starts with resolved and sequence is non-empty",
        "candidate_rows": len(rows),
        "unique_entry_ids": len({row["entry_id"] for row in rows}),
        "unique_sequences": len(ids_by_sha),
        "duplicate_sequence_groups": len(duplicate_groups),
        "rows_in_duplicate_sequence_groups": sum(len(group) for group in duplicate_groups),
        "artifacts": {
            "fasta": str(fasta_path),
            "metadata_csv": str(metadata_path),
            "candidate_ids": str(ids_path),
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-count", type=int, default=144)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_candidate_pool(
        args.source,
        args.output_dir,
        expected_count=args.expected_count,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
