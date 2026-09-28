#!/usr/bin/env python3
"""Build Horizyn-ready train/test files from the global unique split outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/test/global_unique_train_test_splits"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/standardized/global_unique_retrieval"
POLICIES = ("exact", "nr90", "nr50")


PAIR_OUTPUT_FIELDS = [
    "pr_id",
    "reaction_id",
    "protein_id",
    "source_task",
    "source_dataset",
    "source_split",
    "source_reaction_id",
    "source_protein_id",
    "source_pair_index",
    "reaction_smiles",
]
REACTION_OUTPUT_FIELDS = [
    "reaction_id",
    "reaction_smiles",
    "source_task",
    "source_dataset",
    "source_split",
    "source_reaction_id",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
            count += 1
    return count


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


def write_ids(path: Path, ids: Iterable[str]) -> None:
    ids_list = list(ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids_list) + ("\n" if ids_list else ""), encoding="utf-8")


def standardize_side(
    *,
    policy: str,
    side: str,
    input_root: Path,
    output_dir: Path,
) -> dict[str, int | str]:
    input_pairs = input_root / policy / "pairs" / f"{side}_pairs.csv"
    input_fasta = input_root / policy / "pairs" / f"{side}_candidate_proteins.fasta"
    if not input_pairs.exists():
        raise FileNotFoundError(input_pairs)
    if not input_fasta.exists():
        raise FileNotFoundError(input_fasta)

    raw_rows = read_csv(input_pairs)
    protein_records = read_fasta(input_fasta)
    seen_reactions: OrderedDict[str, dict[str, str]] = OrderedDict()
    pair_rows: list[dict[str, str]] = []
    missing_protein_ids: set[str] = set()
    reaction_conflicts: list[str] = []

    for idx, row in enumerate(raw_rows):
        reaction_id = row.get("global_reaction_id") or row.get("reaction_id")
        protein_id = row.get("split_protein_id") or row.get("protein_id")
        reaction_smiles = row.get("reaction_smiles", "")
        if not reaction_id:
            raise ValueError(f"Missing reaction id in {input_pairs} row {idx}")
        if not protein_id:
            raise ValueError(f"Missing protein id in {input_pairs} row {idx}")
        if protein_id not in protein_records:
            missing_protein_ids.add(protein_id)

        pair_rows.append(
            {
                "pr_id": str(idx),
                "reaction_id": reaction_id,
                "protein_id": protein_id,
                "source_task": row.get("source_task", ""),
                "source_dataset": row.get("source_dataset", ""),
                "source_split": row.get("source_split", ""),
                "source_reaction_id": row.get("reaction_id", ""),
                "source_protein_id": row.get("source_protein_id", ""),
                "source_pair_index": row.get("source_pair_index", ""),
                "reaction_smiles": reaction_smiles,
            }
        )

        reaction_row = {
            "reaction_id": reaction_id,
            "reaction_smiles": reaction_smiles,
            "source_task": row.get("source_task", ""),
            "source_dataset": row.get("source_dataset", ""),
            "source_split": row.get("source_split", ""),
            "source_reaction_id": row.get("reaction_id", ""),
        }
        previous = seen_reactions.get(reaction_id)
        if previous is None:
            seen_reactions[reaction_id] = reaction_row
        elif previous["reaction_smiles"] != reaction_smiles:
            reaction_conflicts.append(reaction_id)

    if missing_protein_ids:
        examples = ", ".join(sorted(missing_protein_ids)[:5])
        raise ValueError(
            f"{len(missing_protein_ids)} proteins in {input_pairs} are missing from "
            f"{input_fasta}; examples: {examples}"
        )
    if reaction_conflicts:
        examples = ", ".join(sorted(set(reaction_conflicts))[:5])
        raise ValueError(
            f"{len(set(reaction_conflicts))} reaction IDs have conflicting SMILES in "
            f"{input_pairs}; examples: {examples}"
        )

    prefix = "train" if side == "train_unique" else "test"
    pairs_written = write_csv(output_dir / f"{prefix}_pairs.csv", PAIR_OUTPUT_FIELDS, pair_rows)
    reactions_written = write_csv(
        output_dir / f"{prefix}_rxns.csv",
        REACTION_OUTPUT_FIELDS,
        seen_reactions.values(),
    )
    write_fasta(output_dir / f"{prefix}_proteins.fasta", protein_records)
    write_ids(output_dir / f"{prefix}_candidate_ids.txt", protein_records.keys())

    return {
        f"{prefix}_pairs": pairs_written,
        f"{prefix}_reactions": reactions_written,
        f"{prefix}_proteins": len(protein_records),
        f"{prefix}_input_pairs": str(input_pairs),
        f"{prefix}_input_fasta": str(input_fasta),
    }


def combine_proteins(output_dir: Path) -> int:
    combined: OrderedDict[str, str] = OrderedDict()
    for path in (output_dir / "train_proteins.fasta", output_dir / "test_proteins.fasta"):
        for protein_id, sequence in read_fasta(path).items():
            previous = combined.get(protein_id)
            if previous is not None and previous != sequence:
                raise ValueError(f"Protein id {protein_id} has conflicting sequences")
            combined.setdefault(protein_id, sequence)
    write_fasta(output_dir / "proteins.fasta", combined)
    write_ids(output_dir / "candidate_ids.txt", combined.keys())
    return len(combined)


def write_readme(output_root: Path, summary_rows: list[dict[str, int | str]]) -> None:
    lines = [
        "# Standardized Global Unique Retrieval Datasets",
        "",
        "These folders adapt `data/test/global_unique_train_test_splits` into the",
        "CSV/FASTA layout expected by Horizyn training.",
        "",
        "For each policy, the standardized files are:",
        "",
        "```text",
        "<policy>/train_pairs.csv",
        "<policy>/test_pairs.csv",
        "<policy>/train_rxns.csv",
        "<policy>/test_rxns.csv",
        "<policy>/train_proteins.fasta",
        "<policy>/test_proteins.fasta",
        "<policy>/proteins.fasta",
        "<policy>/candidate_ids.txt",
        "<policy>/metadata.json",
        "```",
        "",
        "The pair files use the Horizyn-compatible columns",
        "`pr_id,reaction_id,protein_id`. `reaction_id` is the global reaction ID",
        "from the unique split, and `protein_id` is the standardized split protein",
        "ID (`uprot_*`, `nr90_*`, or `nr50_*`) used in the FASTA headers.",
        "",
        "| Policy | Train pairs | Train reactions | Train proteins | Test pairs | Test reactions | Test proteins | Combined proteins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| `{policy}` | {train_pairs:,} | {train_reactions:,} | {train_proteins:,} "
            "| {test_pairs:,} | {test_reactions:,} | {test_proteins:,} | "
            "{combined_proteins:,} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Use `proteins.fasta` for residue embedding extraction, then point",
            "`data.protein_residue_embeds_path` at the resulting HDF5 file.",
            "",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, int | str]] = []
    for policy in args.policies:
        output_dir = output_root / policy
        output_dir.mkdir(parents=True, exist_ok=True)
        stats: dict[str, int | str] = {"policy": policy}
        stats.update(
            standardize_side(
                policy=policy,
                side="train_unique",
                input_root=input_root,
                output_dir=output_dir,
            )
        )
        stats.update(
            standardize_side(
                policy=policy,
                side="test_unique",
                input_root=input_root,
                output_dir=output_dir,
            )
        )
        stats["combined_proteins"] = combine_proteins(output_dir)
        (output_dir / "metadata.json").write_text(
            json.dumps(stats, indent=2) + "\n",
            encoding="utf-8",
        )
        summary_rows.append(stats)

    write_csv(
        output_root / "summary_counts.csv",
        [
            "policy",
            "train_pairs",
            "train_reactions",
            "train_proteins",
            "test_pairs",
            "test_reactions",
            "test_proteins",
            "combined_proteins",
        ],
        summary_rows,
    )
    write_readme(output_root, summary_rows)
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
