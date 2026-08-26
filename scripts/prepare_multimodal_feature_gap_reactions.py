#!/usr/bin/env python3
"""Prepare high-impact reaction feature gap lists for multimodal training.

The multimodal reaction encoder can train with missing UniMol2/ChIRo features,
but high-pair-count missing reactions still reduce how often those modalities are
available. This script ranks missing reactions by retrieval-pair count and writes
CSV files that can be passed directly to the UniMol2/ChIRo extractors.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="Training pairs CSV")
    parser.add_argument("--reactions", required=True, help="Training reactions CSV")
    parser.add_argument("--unimol2-h5", required=True, help="Existing UniMol2 reaction HDF5")
    parser.add_argument("--chiro-h5", required=True, help="Existing ChIRo reaction HDF5")
    parser.add_argument(
        "--reaction-model-h5",
        default=None,
        help="Optional ReactionT5/RXN model HDF5 for coverage reporting",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for output CSVs")
    parser.add_argument("--top-n", type=int, default=None, help="Optional top-N truncation")
    parser.add_argument("--pair-id-column", default="reaction_id")
    parser.add_argument("--reaction-id-column", default="reaction_id")
    parser.add_argument("--smiles-column", default="reaction_smiles")
    return parser.parse_args()


def decode_id(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_h5_ids(path: str | Path) -> set[str]:
    with h5py.File(path, "r") as handle:
        if "ids" in handle:
            return {decode_id(value) for value in handle["ids"][:]}
        return {str(key) for key in handle.keys()}


def expected_direction_ids(reaction_id: str, reaction_smiles: str) -> list[str]:
    if ">>" in reaction_smiles and len(reaction_smiles.split(">>")) == 2:
        return [f"{reaction_id}_f", f"{reaction_id}_r"]
    return [f"{reaction_id}_f", f"{reaction_id}_r"]


def read_pair_counts(path: str | Path, reaction_id_column: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reaction_id_column not in (reader.fieldnames or []):
            raise KeyError(f"{path} has no column {reaction_id_column!r}")
        for row in reader:
            counts[str(row[reaction_id_column])] += 1
    return counts


def read_reactions(
    path: str | Path,
    reaction_id_column: str,
    smiles_column: str,
) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reaction_id_column not in (reader.fieldnames or []):
            raise KeyError(f"{path} has no column {reaction_id_column!r}")
        if smiles_column not in (reader.fieldnames or []):
            raise KeyError(f"{path} has no column {smiles_column!r}")
        return [dict(row) for row in reader]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_counts = read_pair_counts(args.pairs, args.pair_id_column)
    reactions = read_reactions(args.reactions, args.reaction_id_column, args.smiles_column)
    unimol2_ids = load_h5_ids(args.unimol2_h5)
    chiro_ids = load_h5_ids(args.chiro_h5)
    reaction_model_ids = load_h5_ids(args.reaction_model_h5) if args.reaction_model_h5 else None

    rows: list[dict[str, object]] = []
    for row in reactions:
        reaction_id = str(row[args.reaction_id_column])
        smiles = str(row[args.smiles_column])
        expected_ids = expected_direction_ids(reaction_id, smiles)
        missing_unimol2 = [key for key in expected_ids if key not in unimol2_ids]
        missing_chiro = [key for key in expected_ids if key not in chiro_ids]
        missing_reaction_model = (
            [key for key in expected_ids if key not in reaction_model_ids]
            if reaction_model_ids is not None
            else []
        )
        pair_count = int(pair_counts.get(reaction_id, 0))
        rows.append(
            {
                args.reaction_id_column: reaction_id,
                args.smiles_column: smiles,
                "pair_count": pair_count,
                "bidirectional_pair_count": pair_count * len(expected_ids),
                "expected_direction_ids": "|".join(expected_ids),
                "missing_unimol2": bool(missing_unimol2),
                "missing_chiro": bool(missing_chiro),
                "missing_any_side_modality": bool(missing_unimol2 or missing_chiro),
                "missing_reaction_model": bool(missing_reaction_model),
                "missing_unimol2_ids": "|".join(missing_unimol2),
                "missing_chiro_ids": "|".join(missing_chiro),
            }
        )

    rows.sort(key=lambda item: (-int(item["pair_count"]), str(item[args.reaction_id_column])))
    missing_unimol2 = [row for row in rows if row["missing_unimol2"]]
    missing_chiro = [row for row in rows if row["missing_chiro"]]
    missing_any = [row for row in rows if row["missing_any_side_modality"]]
    if args.top_n is not None:
        missing_unimol2 = missing_unimol2[: args.top_n]
        missing_chiro = missing_chiro[: args.top_n]
        missing_any = missing_any[: args.top_n]

    write_rows(output_dir / "missing_unimol2_reactions.csv", missing_unimol2)
    write_rows(output_dir / "missing_chiro_reactions.csv", missing_chiro)
    write_rows(output_dir / "missing_any_side_modality_reactions.csv", missing_any)
    write_rows(output_dir / "all_reaction_feature_coverage.csv", rows)

    total_pair_count = sum(pair_counts.values())
    summary = {
        "num_reactions": len(rows),
        "num_pairs": total_pair_count,
        "unimol2_feature_ids": len(unimol2_ids),
        "chiro_feature_ids": len(chiro_ids),
        "missing_unimol2_reactions": sum(bool(row["missing_unimol2"]) for row in rows),
        "missing_chiro_reactions": sum(bool(row["missing_chiro"]) for row in rows),
        "missing_any_side_modality_reactions": sum(
            bool(row["missing_any_side_modality"]) for row in rows
        ),
        "missing_unimol2_pair_count": sum(
            int(row["pair_count"]) for row in rows if row["missing_unimol2"]
        ),
        "missing_chiro_pair_count": sum(
            int(row["pair_count"]) for row in rows if row["missing_chiro"]
        ),
        "missing_any_side_modality_pair_count": sum(
            int(row["pair_count"]) for row in rows if row["missing_any_side_modality"]
        ),
    }
    if total_pair_count:
        summary["missing_unimol2_pair_fraction"] = (
            summary["missing_unimol2_pair_count"] / total_pair_count
        )
        summary["missing_chiro_pair_fraction"] = (
            summary["missing_chiro_pair_count"] / total_pair_count
        )
        summary["missing_any_side_modality_pair_fraction"] = (
            summary["missing_any_side_modality_pair_count"] / total_pair_count
        )

    (output_dir / "coverage_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
