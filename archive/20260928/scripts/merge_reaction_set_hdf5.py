#!/usr/bin/env python3
"""Merge ragged reaction-side embedding HDF5 files.

The input/output schema is the one used by UniMol2ReactionEmbedDataset:

    ids
    reactant_vectors, reactant_offsets
    product_vectors, product_offsets

Inputs are consumed in order. Duplicate reaction IDs are kept from the first
file unless --prefer-last is supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Input HDF5 files")
    parser.add_argument("--output", required=True, help="Merged output HDF5")
    parser.add_argument("--force", action="store_true", help="Overwrite output")
    parser.add_argument(
        "--prefer-last",
        action="store_true",
        help="For duplicate IDs, keep the last input occurrence instead of the first",
    )
    parser.add_argument(
        "--compression",
        choices=("none", "gzip", "lzf"),
        default="none",
        help="Compression for vector datasets",
    )
    return parser.parse_args()


def decode_id(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_records(path: str | Path) -> list[dict[str, np.ndarray | str]]:
    records = []
    with h5py.File(path, "r") as handle:
        required = {
            "ids",
            "reactant_vectors",
            "reactant_offsets",
            "product_vectors",
            "product_offsets",
        }
        missing = sorted(required - set(handle.keys()))
        if missing:
            raise KeyError(f"{path} is missing required datasets: {missing}")
        ids = [decode_id(value) for value in handle["ids"][:]]
        reactant_offsets = handle["reactant_offsets"][:]
        product_offsets = handle["product_offsets"][:]
        for idx, reaction_id in enumerate(ids):
            r0, r1 = int(reactant_offsets[idx]), int(reactant_offsets[idx + 1])
            p0, p1 = int(product_offsets[idx]), int(product_offsets[idx + 1])
            records.append(
                {
                    "id": reaction_id,
                    "reactants": handle["reactant_vectors"][r0:r1],
                    "products": handle["product_vectors"][p0:p1],
                }
            )
    return records


def write_records(
    records: list[dict[str, np.ndarray | str]],
    output: str | Path,
    compression: str,
    force: bool,
) -> None:
    output_path = Path(output)
    if output_path.exists() and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    h5_compression = None if compression == "none" else compression
    string_dtype = h5py.string_dtype(encoding="utf-8")

    if not records:
        raise ValueError("No records to write")
    reactant_vectors = np.concatenate([record["reactants"] for record in records], axis=0)
    product_vectors = np.concatenate([record["products"] for record in records], axis=0)
    reactant_offsets = [0]
    product_offsets = [0]
    for record in records:
        reactant_offsets.append(reactant_offsets[-1] + int(record["reactants"].shape[0]))
        product_offsets.append(product_offsets[-1] + int(record["products"].shape[0]))

    with h5py.File(output_path, "w") as handle:
        handle.create_dataset(
            "ids",
            data=np.asarray([record["id"] for record in records], dtype=object),
            dtype=string_dtype,
        )
        handle.create_dataset("reactant_vectors", data=reactant_vectors, compression=h5_compression)
        handle.create_dataset("reactant_offsets", data=np.asarray(reactant_offsets, dtype=np.int64))
        handle.create_dataset("product_vectors", data=product_vectors, compression=h5_compression)
        handle.create_dataset("product_offsets", data=np.asarray(product_offsets, dtype=np.int64))


def main() -> None:
    args = parse_args()
    by_id: dict[str, dict[str, np.ndarray | str]] = {}
    order: list[str] = []
    for input_path in args.inputs:
        for record in read_records(input_path):
            reaction_id = str(record["id"])
            if reaction_id not in by_id:
                order.append(reaction_id)
                by_id[reaction_id] = record
            elif args.prefer_last:
                by_id[reaction_id] = record
    records = [by_id[reaction_id] for reaction_id in order]
    write_records(records, args.output, args.compression, args.force)
    print(f"Wrote {len(records)} merged reaction records to {args.output}")


if __name__ == "__main__":
    main()
