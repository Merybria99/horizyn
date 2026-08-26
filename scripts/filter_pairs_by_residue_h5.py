#!/usr/bin/env python3
"""Filter protein-reaction pair CSVs to proteins with non-empty residue embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep only pair rows whose protein_id is present in a residue HDF5 and "
            "has at least one residue vector."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pairs", required=True, help="Input pair CSV")
    parser.add_argument("--output", required=True, help="Filtered output pair CSV")
    parser.add_argument("--h5", required=True, help="Residue HDF5 with ids/vectors/offsets")
    parser.add_argument("--protein-column", default="protein_id")
    parser.add_argument("--metadata-output", default=None)
    return parser.parse_args()


def load_nonempty_ids(h5_path: Path) -> tuple[set[str], dict[str, int]]:
    with h5py.File(h5_path, "r") as handle:
        ids = handle["ids"][:]
        offsets = handle["offsets"][:]
        lengths = offsets[1:] - offsets[:-1]
        decoded = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in ids
        ]
        nonempty = {protein_id for protein_id, length in zip(decoded, lengths) if int(length) > 0}
        return nonempty, {
            "h5_ids": len(decoded),
            "h5_nonempty_ids": len(nonempty),
            "h5_empty_ids": len(decoded) - len(nonempty),
        }


def main() -> None:
    args = parse_args()
    pair_path = Path(args.pairs)
    output_path = Path(args.output)
    h5_path = Path(args.h5)
    metadata_path = (
        Path(args.metadata_output)
        if args.metadata_output is not None
        else output_path.with_suffix(output_path.suffix + ".metadata.json")
    )

    nonempty_ids, h5_metadata = load_nonempty_ids(h5_path)

    with pair_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {pair_path}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    if args.protein_column not in fieldnames:
        raise KeyError(f"Column '{args.protein_column}' not found in {pair_path}")

    kept = [row for row in rows if row.get(args.protein_column, "") in nonempty_ids]
    dropped = [row for row in rows if row.get(args.protein_column, "") not in nonempty_ids]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    dropped_reasons = {"missing_or_empty_embedding": len(dropped)}
    metadata = {
        "input_pairs": len(rows),
        "output_pairs": len(kept),
        "dropped_pairs": len(dropped),
        "dropped_reasons": dropped_reasons,
        "input_pairs_path": str(pair_path),
        "output_pairs_path": str(output_path),
        "h5_path": str(h5_path),
        **h5_metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
