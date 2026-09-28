#!/usr/bin/env python3
"""
Extract fixed DRFP reaction vectors to an HDF5 file.

Output schema:
    /ids      string reaction IDs
    /vectors  float reaction vectors [N, D]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from horizyn.chemistry.standardizer import Standardizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reactions", nargs="+", required=True, help="Reaction CSV files")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument("--key-column", default="reaction_id")
    parser.add_argument("--smiles-column", default="reaction_smiles")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--rings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--standardize-hypervalent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--standardize-remove-hs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--standardize-kekulize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--standardize-uncharge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--standardize-metals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-pseudo-reactions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Convert molecule-set strings with no reaction arrow into self reactions.",
    )
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    return parser.parse_args()


def normalize_smiles(
    smiles: str,
    standardizer: Standardizer | None,
    allow_pseudo_reactions: bool,
) -> str:
    if smiles.count(">") < 2:
        if not allow_pseudo_reactions:
            return smiles
        smiles = f"{smiles}>>{smiles}"
    return standardizer.standardize_reaction(smiles) if standardizer is not None else smiles


def reverse_reaction(smiles: str) -> str:
    parts = smiles.split(">>", maxsplit=1)
    if len(parts) != 2:
        return smiles
    return f"{parts[1]}>>{parts[0]}"


def read_reactions(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    standardizer = (
        Standardizer(
            standardize_hypervalent=args.standardize_hypervalent,
            standardize_remove_hs=args.standardize_remove_hs,
            standardize_kekulize=args.standardize_kekulize,
            standardize_uncharge=args.standardize_uncharge,
            standardize_metals=args.standardize_metals,
        )
        if args.standardize
        else None
    )
    ids: list[str] = []
    smiles_values: list[str] = []
    seen: set[str] = set()
    for path_str in args.reactions:
        path = Path(path_str)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if args.key_column not in reader.fieldnames or args.smiles_column not in reader.fieldnames:
                raise KeyError(
                    f"{path} must contain columns {args.key_column!r} and "
                    f"{args.smiles_column!r}; found {reader.fieldnames}"
                )
            for row in reader:
                base_id = str(row[args.key_column])
                smiles = normalize_smiles(
                    str(row[args.smiles_column]),
                    standardizer,
                    args.allow_pseudo_reactions,
                )
                directions = (
                    ((base_id, smiles),)
                    if not args.bidirectional
                    else (
                        (f"{base_id}_f", smiles),
                        (f"{base_id}_r", reverse_reaction(smiles)),
                    )
                )
                for reaction_id, reaction_smiles in directions:
                    if reaction_id in seen:
                        continue
                    seen.add(reaction_id)
                    ids.append(reaction_id)
                    smiles_values.append(reaction_smiles)
    return ids, smiles_values


def encode_reactions(
    smiles_values: list[str],
    batch_size: int,
    dim: int,
    radius: int,
    rings: bool,
) -> np.ndarray:
    from drfp import DrfpEncoder

    chunks = []
    total = len(smiles_values)
    for start in range(0, total, batch_size):
        batch = smiles_values[start : start + batch_size]
        encoded = DrfpEncoder.encode(
            batch,
            n_folded_length=dim,
            radius=radius,
            rings=rings,
        )
        chunks.append(np.asarray(encoded, dtype=np.float32))
        print(f"Encoded {min(start + batch_size, total)}/{total} reactions")
    return np.concatenate(chunks, axis=0)


def write_hdf5(
    output: str,
    ids: list[str],
    vectors: np.ndarray,
    dtype_name: str,
    args: argparse.Namespace,
) -> None:
    output_path = Path(output)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    dtype = np.float16 if dtype_name == "float16" else np.float32
    text_dtype = h5py.string_dtype("utf-8")
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("ids", data=np.asarray(ids, dtype=object), dtype=text_dtype)
        handle.create_dataset("vectors", data=vectors.astype(dtype))
        handle.attrs["embedding_dim"] = int(vectors.shape[1])
        handle.attrs["reaction_representation"] = "drfp"
        handle.attrs["radius"] = int(args.radius)
        handle.attrs["rings"] = bool(args.rings)


def main() -> None:
    args = parse_args()
    ids, smiles_values = read_reactions(args)
    print(f"Loaded {len(ids)} reaction directions")
    vectors = encode_reactions(
        smiles_values,
        batch_size=args.batch_size,
        dim=args.dim,
        radius=args.radius,
        rings=args.rings,
    )
    write_hdf5(args.output, ids, vectors, args.dtype, args)
    print(f"Wrote DRFP reaction vectors to {args.output}")


if __name__ == "__main__":
    main()
