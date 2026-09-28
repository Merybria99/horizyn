#!/usr/bin/env python3
"""Materialize official ReactZyme protocol datasets without derived val splits.

The ReactZyme release provides three independent benchmark protocols:
``time``, ``enzyme_smi``, and ``reaction_smi``. Each protocol has an official
train split and an official test split. This script preserves that recipe and
only converts the paper files into the Horizyn CSV layout expected by the
training/evaluation code.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data/paper/reactzyme/eval"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data/revised_protocols/reactzyme_official"
PROTOCOLS = ("time", "enzyme_smi", "reaction_smi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value and not value.startswith("#"):
                ids.append(value.split(",")[0].split()[0])
    return ids


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def read_fasta(path: Path) -> OrderedDict[str, str]:
    records: OrderedDict[str, str] = OrderedDict()
    current_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = "".join(chunks).upper()
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if current_id is not None:
            records[current_id] = "".join(chunks).upper()
    return records


def write_fasta(path: Path, records: OrderedDict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record_id, sequence in records.items():
            handle.write(f">{record_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def canonical_pair_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        output.append(
            {
                "pr_id": str(idx),
                "reaction_id": row["reaction_id"],
                "protein_id": row["protein_id"],
                "reaction_smiles": row["reaction_smiles"],
                "protein_sequence": row["protein_sequence"],
            }
        )
    return output


def reaction_smiles_from_pairs(pair_rows: list[dict[str, str]]) -> OrderedDict[str, str]:
    by_reaction: OrderedDict[str, str] = OrderedDict()
    for row in pair_rows:
        reaction_id = row["reaction_id"]
        reaction_smiles = row["reaction_smiles"]
        previous = by_reaction.get(reaction_id)
        if previous is not None and previous != reaction_smiles:
            raise ValueError(f"Conflicting SMILES for reaction {reaction_id}")
        by_reaction.setdefault(reaction_id, reaction_smiles)
    return by_reaction


def read_reaction_table(path: Path) -> OrderedDict[str, str]:
    rows = read_csv_rows(path)
    table: OrderedDict[str, str] = OrderedDict()
    for row in rows:
        reaction_id = row["reaction_id"]
        reaction_smiles = row["reaction_smiles"]
        previous = table.get(reaction_id)
        if previous is not None and previous != reaction_smiles:
            raise ValueError(f"Conflicting SMILES for reaction {reaction_id} in {path}")
        table.setdefault(reaction_id, reaction_smiles)
    return table


def reaction_rows_for_pairs(
    pair_rows: list[dict[str, str]],
    reaction_table: OrderedDict[str, str],
) -> list[dict[str, str]]:
    pair_reactions = reaction_smiles_from_pairs(pair_rows)
    rows: list[dict[str, str]] = []
    for reaction_id, pair_smiles in pair_reactions.items():
        table_smiles = reaction_table.get(reaction_id, pair_smiles)
        if table_smiles != pair_smiles:
            raise ValueError(f"Reaction table disagrees with pairs for {reaction_id}")
        rows.append({"reaction_id": reaction_id, "reaction_smiles": table_smiles})
    return rows


def pair_set(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {(row["protein_id"], row["reaction_id"]) for row in rows}


def sequence_smiles_pair_set(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {(row["protein_sequence"], row["reaction_smiles"]) for row in rows}


def collect_counts(
    rows: list[dict[str, str]],
    reaction_rows: list[dict[str, str]],
    candidate_ids: list[str] | None = None,
) -> dict[str, int]:
    counts = {
        "pairs": len(rows),
        "proteins": len({row["protein_id"] for row in rows}),
        "reactions": len({row["reaction_id"] for row in rows}),
        "reaction_smiles": len({row["reaction_smiles"] for row in rows}),
        "reaction_table_rows": len(reaction_rows),
    }
    if candidate_ids is not None:
        counts["candidate_proteins"] = len(candidate_ids)
    return counts


def write_protocol(protocol: str, input_root: Path, out_root: Path) -> dict[str, Any]:
    source_dir = input_root / protocol
    train_rows = canonical_pair_rows(read_csv_rows(source_dir / "train_pairs.csv"))
    test_rows = canonical_pair_rows(read_csv_rows(source_dir / "test_pairs.csv"))
    reaction_table = read_reaction_table(source_dir / "reactions.csv")
    train_rxns = reaction_rows_for_pairs(train_rows, reaction_table)
    test_rxns = reaction_rows_for_pairs(test_rows, reaction_table)

    candidate_ids = read_ids(source_dir / "candidate_ids.txt")
    candidate_records = read_fasta(source_dir / "proteins.fasta")
    missing_candidates = sorted(set(candidate_ids) - set(candidate_records))
    if missing_candidates:
        examples = ", ".join(missing_candidates[:5])
        raise ValueError(
            f"{protocol} has {len(missing_candidates)} candidate IDs missing from "
            f"proteins.fasta: {examples}"
        )

    out_dir = out_root / protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_fields = ["pr_id", "reaction_id", "protein_id", "reaction_smiles", "protein_sequence"]
    reaction_fields = ["reaction_id", "reaction_smiles"]
    write_csv(out_dir / "train_pairs.csv", train_rows, pair_fields)
    write_csv(out_dir / "test_pairs.csv", test_rows, pair_fields)
    write_csv(out_dir / "train_rxns.csv", train_rxns, reaction_fields)
    write_csv(out_dir / "test_rxns.csv", test_rxns, reaction_fields)
    write_ids(out_dir / "candidate_ids.txt", candidate_ids)
    write_ids(out_dir / "train_positive_ids.txt", sorted({row["protein_id"] for row in train_rows}))
    write_ids(out_dir / "test_positive_ids.txt", sorted({row["protein_id"] for row in test_rows}))
    write_fasta(
        out_dir / "proteins.fasta",
        OrderedDict((protein_id, candidate_records[protein_id]) for protein_id in candidate_ids),
    )

    overlaps = {
        "train_test_exact_pair_overlap": len(pair_set(train_rows) & pair_set(test_rows)),
        "train_test_sequence_smiles_overlap": len(
            sequence_smiles_pair_set(train_rows) & sequence_smiles_pair_set(test_rows)
        ),
        "train_test_reaction_id_overlap": len(
            {row["reaction_id"] for row in train_rows}
            & {row["reaction_id"] for row in test_rows}
        ),
        "train_test_protein_id_overlap": len(
            {row["protein_id"] for row in train_rows}
            & {row["protein_id"] for row in test_rows}
        ),
    }
    if overlaps["train_test_exact_pair_overlap"]:
        raise RuntimeError(f"Unexpected exact pair leakage for {protocol}: {overlaps}")

    manifest = {
        "protocol": protocol,
        "description": (
            "Official ReactZyme protocol. Train is the released train split and "
            "test is the released test split. No random validation split and no "
            "cross-protocol pooling are materialized here."
        ),
        "source_files": {
            "train_pairs": str(source_dir / "train_pairs.csv"),
            "test_pairs": str(source_dir / "test_pairs.csv"),
            "reactions": str(source_dir / "reactions.csv"),
            "candidate_ids": str(source_dir / "candidate_ids.txt"),
            "proteins_fasta": str(source_dir / "proteins.fasta"),
        },
        "outputs": {
            "train_pairs": str(out_dir / "train_pairs.csv"),
            "test_pairs": str(out_dir / "test_pairs.csv"),
            "train_reactions": str(out_dir / "train_rxns.csv"),
            "test_reactions": str(out_dir / "test_rxns.csv"),
            "candidate_ids": str(out_dir / "candidate_ids.txt"),
            "proteins_fasta": str(out_dir / "proteins.fasta"),
        },
        "counts": {
            "train": collect_counts(train_rows, train_rxns),
            "test": collect_counts(test_rows, test_rxns, candidate_ids),
        },
        "leakage_checks": overlaps,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_cross_protocol_report(input_root: Path, out_root: Path) -> list[dict[str, Any]]:
    split_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    for protocol in PROTOCOLS:
        source_dir = input_root / protocol
        split_rows[(protocol, "train")] = read_csv_rows(source_dir / "train_pairs.csv")
        split_rows[(protocol, "test")] = read_csv_rows(source_dir / "test_pairs.csv")

    report_rows: list[dict[str, Any]] = []
    for held_protocol in PROTOCOLS:
        held_test = split_rows[(held_protocol, "test")]
        held_pairs = pair_set(held_test)
        held_seq_smiles = sequence_smiles_pair_set(held_test)
        for train_protocol in PROTOCOLS:
            train_rows = split_rows[(train_protocol, "train")]
            train_pairs = pair_set(train_rows)
            train_seq_smiles = sequence_smiles_pair_set(train_rows)
            report_rows.append(
                {
                    "heldout_protocol": held_protocol,
                    "train_protocol": train_protocol,
                    "heldout_test_pairs": len(held_pairs),
                    "train_pairs": len(train_pairs),
                    "exact_pair_overlap": len(held_pairs & train_pairs),
                    "sequence_smiles_overlap": len(held_seq_smiles & train_seq_smiles),
                    "note": (
                        "same protocol"
                        if held_protocol == train_protocol
                        else "cross-protocol overlap: do not mix protocols"
                    ),
                }
            )

    write_csv(
        out_root / "cross_protocol_overlap.csv",
        report_rows,
        [
            "heldout_protocol",
            "train_protocol",
            "heldout_test_pairs",
            "train_pairs",
            "exact_pair_overlap",
            "sequence_smiles_overlap",
            "note",
        ],
    )
    return report_rows


def write_readme(out_root: Path, manifests: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Official ReactZyme Protocol Datasets",
        "",
        "This directory contains the released ReactZyme protocol splits converted",
        "to the Horizyn CSV layout. It intentionally does not contain random",
        "validation splits.",
        "",
        "Use exactly one protocol folder per run. Do not merge `time`,",
        "`enzyme_smi`, and `reaction_smi` train files for split-specific",
        "benchmark reporting.",
        "",
        "| Protocol | Train pairs | Test pairs | Train reactions | Test reactions | Candidates |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for protocol in PROTOCOLS:
        counts = manifests[protocol]["counts"]
        lines.append(
            f"| `{protocol}` | {counts['train']['pairs']:,} | "
            f"{counts['test']['pairs']:,} | {counts['train']['reactions']:,} | "
            f"{counts['test']['reactions']:,} | {counts['test']['candidate_proteins']:,} |"
        )
    lines.extend(
        [
            "",
            "Each protocol folder contains:",
            "",
            "- `train_pairs.csv`, `test_pairs.csv`",
            "- `train_rxns.csv`, `test_rxns.csv`",
            "- `candidate_ids.txt` and `proteins.fasta` copied from the release",
            "- `manifest.json` with counts and leakage checks",
            "",
            "`cross_protocol_overlap.csv` records why the three protocol train",
            "splits must not be pooled for paper-comparable evaluation.",
            "",
        ]
    )
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {args.out_root}")
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    missing = [protocol for protocol in PROTOCOLS if not (args.input_root / protocol).exists()]
    if missing:
        raise FileNotFoundError(f"Missing ReactZyme protocol folders: {missing}")

    manifests = {
        protocol: write_protocol(protocol, args.input_root, args.out_root)
        for protocol in PROTOCOLS
    }
    cross_protocol = build_cross_protocol_report(args.input_root, args.out_root)
    top_manifest = {
        "output_root": str(args.out_root),
        "input_root": str(args.input_root),
        "protocols": manifests,
        "cross_protocol_overlap_csv": str(args.out_root / "cross_protocol_overlap.csv"),
        "cross_protocol_overlap": cross_protocol,
        "guardrail": (
            "Official ReactZyme runs train on exactly one released train split "
            "and evaluate on the matching released test split."
        ),
    }
    (args.out_root / "manifest.json").write_text(
        json.dumps(top_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(args.out_root, manifests)
    print(json.dumps(top_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
