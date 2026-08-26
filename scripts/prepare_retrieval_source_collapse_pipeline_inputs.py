#!/usr/bin/env python3
"""Prepare FASTA and CSV inputs for collapsed-source Horizyn retrieval training."""

from __future__ import annotations

import argparse
import csv
import os
from collections import OrderedDict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create training/validation embedding FASTAs and small adapter files "
            "for data/standardized/retrieval_training_source_collapse."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        default="data/standardized/retrieval_training_source_collapse",
        help="Retrieval source-collapse artifact root",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["exact", "nr90", "nr50"],
        choices=["exact", "nr90", "nr50"],
        help="Training variants to prepare",
    )
    parser.add_argument(
        "--shared-eval-root",
        default="data/standardized/horizyn_reactzyme_eval",
        help="Shared Horizyn + ReactZyme candidate FASTA/candidate-id root",
    )
    return parser.parse_args()


def read_fasta(path: Path) -> OrderedDict[str, str]:
    records: OrderedDict[str, str] = OrderedDict()
    current_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = "".join(chunks)
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        records[current_id] = "".join(chunks)
    return records


def write_fasta(records: OrderedDict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for protein_id, sequence in records.items():
            handle.write(f">{protein_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(f"{sequence[start:start + 80]}\n")


def merge_fastas(paths: list[Path], output_path: Path) -> dict[str, int]:
    merged: OrderedDict[str, str] = OrderedDict()
    duplicate_same = 0
    duplicate_conflict = 0
    for path in paths:
        records = read_fasta(path)
        for protein_id, sequence in records.items():
            previous = merged.get(protein_id)
            if previous is None:
                merged[protein_id] = sequence
            elif previous == sequence:
                duplicate_same += 1
            else:
                duplicate_conflict += 1
                raise ValueError(
                    f"Protein ID {protein_id!r} has conflicting sequences while merging {path}"
                )
    write_fasta(merged, output_path)
    return {
        "input_records": sum(len(read_fasta(path)) for path in paths),
        "output_records": len(merged),
        "duplicate_same_id_same_sequence": duplicate_same,
        "duplicate_same_id_conflict": duplicate_conflict,
    }


def write_id_list_from_fasta(fasta_path: Path, output_path: Path) -> int:
    records = read_fasta(fasta_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for protein_id in records:
            handle.write(f"{protein_id}\n")
    return len(records)


def write_pairs_horizyn(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", newline="", encoding="utf-8") as in_handle:
        reader = csv.DictReader(in_handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_path}")
        missing = {"reaction_id", "protein_id"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns in {input_path}: {sorted(missing)}")
        extra_fields = [
            field
            for field in reader.fieldnames
            if field not in {"pr_id", "reaction_id", "protein_id"}
        ]
        fieldnames = ["pr_id", "reaction_id", "protein_id", *extra_fields]
        count = 0
        with output_path.open("w", newline="", encoding="utf-8") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
            writer.writeheader()
            for idx, row in enumerate(reader):
                out_row = {
                    "pr_id": row.get("pr_id") or str(idx),
                    "reaction_id": row["reaction_id"],
                    "protein_id": row["protein_id"],
                }
                for field in extra_fields:
                    out_row[field] = row.get(field, "")
                writer.writerow(out_row)
                count += 1
    return count


def ensure_symlink_or_copy(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if dest.resolve() == source.resolve():
            return
        dest.unlink()
    rel_source = os.path.relpath(source.resolve(), start=dest.parent.resolve())
    dest.symlink_to(rel_source)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    shared_eval_root = Path(args.shared_eval_root)
    if not root.exists():
        raise FileNotFoundError(root)
    if not shared_eval_root.exists():
        raise FileNotFoundError(shared_eval_root)

    validation_root = root / "validation" / "clipzyme_eval"
    clipzyme_pairs = validation_root / "pairs.csv"
    clipzyme_proteins = validation_root / "proteins.fasta"
    pairs_horizyn = validation_root / "pairs_horizyn.csv"
    pair_count = write_pairs_horizyn(clipzyme_pairs, pairs_horizyn)
    print(f"Wrote validation pair adapter: {pairs_horizyn} ({pair_count} rows)")

    for variant in args.variants:
        train_root = root / f"train_{variant}"
        train_fasta = train_root / "train_proteins.fasta"
        fit_fasta = train_root / "fit_proteins_with_clipzyme_eval.fasta"
        stats = merge_fastas([train_fasta, clipzyme_proteins], fit_fasta)
        print(
            f"Wrote {variant} fit FASTA: {fit_fasta} "
            f"({stats['output_records']} proteins; {stats['duplicate_same_id_same_sequence']} "
            "same-ID duplicates skipped)"
        )

    horizyn_fasta = root / "test" / "horizyn" / "proteins.fasta"
    horizyn_ids = root / "test" / "horizyn" / "candidate_ids.txt"
    id_count = write_id_list_from_fasta(horizyn_fasta, horizyn_ids)
    print(f"Wrote Horizyn test candidate IDs: {horizyn_ids} ({id_count} IDs)")

    shared_root = root / "test" / "horizyn_reactzyme_shared_candidates"
    shared_files = [
        "proteins.fasta",
        "candidate_ids.txt",
        "horizyn_sota_candidate_ids.txt",
        "reactzyme_time_candidate_ids.txt",
        "reactzyme_enzyme_smi_candidate_ids.txt",
        "reactzyme_reaction_smi_candidate_ids.txt",
    ]
    for filename in shared_files:
        ensure_symlink_or_copy(shared_eval_root / filename, shared_root / filename)
    print(f"Prepared shared test candidate symlinks under: {shared_root}")


if __name__ == "__main__":
    main()
