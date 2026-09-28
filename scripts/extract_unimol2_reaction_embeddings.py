#!/usr/bin/env python3
"""
Extract frozen Uni-Mol2 molecule embeddings grouped by reaction side.

Output HDF5 schema:
    /ids
    /reactant_vectors, /reactant_offsets
    /product_vectors, /product_offsets
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from horizyn.chemistry.standardizer import Standardizer


@dataclass(frozen=True)
class ReactionSetRecord:
    reaction_id: str
    reactants: tuple[str, ...]
    products: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reactions",
        nargs="+",
        default=["data/sota/train_rxns.csv", "data/sota/test_rxns.csv"],
        help="Reaction CSV files to encode",
    )
    parser.add_argument(
        "--output",
        default="data/sota/rxns_unimol2_84m_reaction_sets.h5",
        help="Output ragged HDF5 path",
    )
    parser.add_argument("--key-column", default="reaction_id", help="Reaction ID CSV column")
    parser.add_argument(
        "--smiles-column",
        default="reaction_smiles",
        help="Reaction SMILES CSV column",
    )
    parser.add_argument("--model-name", default="unimolv2", help="Uni-Mol model name")
    parser.add_argument("--model-size", default="84m", help="Uni-Mol2 model size")
    parser.add_argument("--batch-size", type=int, default=64, help="Molecule encoding batch size")
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Stored embedding dtype",
    )
    parser.add_argument(
        "--compression",
        choices=("none", "gzip", "lzf"),
        default="none",
        help="HDF5 compression for vector datasets",
    )
    parser.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply Horizyn reaction standardization before splitting",
    )
    parser.add_argument(
        "--standardize-hypervalent",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--standardize-remove-hs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--standardize-kekulize",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--standardize-uncharge",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--standardize-metals",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-invalid-molecules",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip molecules that Uni-Mol2 cannot represent robustly, such as "
            "single-atom ions/protons."
        ),
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="Skip molecules with more than this many atoms before Uni-Mol2 encoding.",
    )
    parser.add_argument(
        "--skip-metals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip molecules containing metal atoms before Uni-Mol2 conformer generation.",
    )
    parser.add_argument(
        "--skip-invalid-reactions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip reaction directions left with an empty reactant or product side.",
    )
    parser.add_argument(
        "--bidirectional",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write forward/reverse reaction IDs with _f/_r suffixes.",
    )
    parser.add_argument(
        "--allow-pseudo-reactions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Treat one-sided molecule-set SMILES as pseudo reactions by using the "
            "same molecule set on both sides. This is useful for ReactZyme paper "
            "splits, which store reaction-level molecule sets rather than full "
            "reactant>>product SMILES."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    return parser.parse_args()


def split_reaction_smiles(
    reaction_id: str,
    smiles: str,
    allow_pseudo_reactions: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    parts = smiles.split(">>")
    if len(parts) != 2:
        if not allow_pseudo_reactions or ">" in smiles:
            raise ValueError(f"Reaction {reaction_id} is not a two-sided reaction SMILES: {smiles}")
        molecules = tuple(mol for mol in smiles.split(".") if mol)
        if not molecules:
            raise ValueError(f"Reaction {reaction_id} has no molecules: {smiles}")
        return molecules, molecules
    reactants = tuple(mol for mol in parts[0].split(".") if mol)
    products = tuple(mol for mol in parts[1].split(".") if mol)
    if not reactants:
        raise ValueError(f"Reaction {reaction_id} has no reactant molecules: {smiles}")
    if not products:
        raise ValueError(f"Reaction {reaction_id} has no product molecules: {smiles}")
    return reactants, products


_METAL_ATOMIC_NUMBERS = {
    3,
    4,
    11,
    12,
    13,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
}


def is_unimol2_candidate(
    smiles: str,
    max_atoms: int | None = None,
    skip_metals: bool = False,
) -> bool:
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None or mol.GetNumAtoms() < 2:
        return False
    if sum(atom.GetAtomicNum() > 1 for atom in mol.GetAtoms()) < 2:
        return False
    if max_atoms is not None and mol.GetNumAtoms() > max_atoms:
        return False
    # Dummy/wildcard atoms usually make Uni-Mol2 conformer generation fail and
    # force expensive per-molecule retry of the whole batch.
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return False
    if skip_metals and any(
        atom.GetAtomicNum() in _METAL_ATOMIC_NUMBERS for atom in mol.GetAtoms()
    ):
        return False
    return True


def filter_side_molecules(
    reaction_id: str,
    side_name: str,
    molecules: tuple[str, ...],
    skip_invalid_molecules: bool,
    max_atoms: int | None = None,
    skip_metals: bool = False,
) -> tuple[str, ...]:
    if not skip_invalid_molecules:
        return molecules

    kept = []
    for smiles in molecules:
        if is_unimol2_candidate(smiles, max_atoms=max_atoms, skip_metals=skip_metals):
            kept.append(smiles)
        else:
            print(
                f"Warning: skipping non-encodable {side_name} molecule "
                f"for {reaction_id}: {smiles}"
            )
    return tuple(kept)


def read_reactions(
    paths: list[str],
    key_column: str,
    smiles_column: str,
    standardize: bool,
    skip_invalid_molecules: bool,
    skip_invalid_reactions: bool,
    bidirectional: bool,
    allow_pseudo_reactions: bool,
    max_atoms: int | None = None,
    skip_metals: bool = False,
    standardizer_kwargs: dict | None = None,
):
    standardizer = Standardizer(**(standardizer_kwargs or {})) if standardize else None
    records: list[ReactionSetRecord] = []
    seen_ids: set[str] = set()
    duplicate_ids = 0

    for path_str in paths:
        path = Path(path_str)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if key_column not in reader.fieldnames or smiles_column not in reader.fieldnames:
                raise KeyError(
                    f"{path} must contain columns {key_column!r} and {smiles_column!r}; "
                    f"found {reader.fieldnames}"
                )
            for row in reader:
                base_id = str(row[key_column])
                smiles = str(row[smiles_column])
                if standardizer is not None:
                    if ">>" in smiles:
                        smiles = standardizer.standardize_reaction(smiles)
                    elif not allow_pseudo_reactions:
                        smiles = standardizer.standardize_reaction(smiles)
                try:
                    reactants, products = split_reaction_smiles(
                        base_id,
                        smiles,
                        allow_pseudo_reactions=allow_pseudo_reactions,
                    )
                except ValueError as exc:
                    if skip_invalid_reactions:
                        print(f"Warning: skipping invalid reaction {base_id}: {exc}")
                        continue
                    raise

                directions = (
                    ((base_id, reactants, products),)
                    if not bidirectional
                    else (
                        (f"{base_id}_f", reactants, products),
                        (f"{base_id}_r", products, reactants),
                    )
                )
                for reaction_id, side_a, side_b in directions:
                    side_a = filter_side_molecules(
                        reaction_id,
                        "reactant",
                        side_a,
                        skip_invalid_molecules=skip_invalid_molecules,
                        max_atoms=max_atoms,
                        skip_metals=skip_metals,
                    )
                    side_b = filter_side_molecules(
                        reaction_id,
                        "product",
                        side_b,
                        skip_invalid_molecules=skip_invalid_molecules,
                        max_atoms=max_atoms,
                        skip_metals=skip_metals,
                    )
                    if not side_a or not side_b:
                        message = (
                            f"Reaction {reaction_id} has an empty side after "
                            "Uni-Mol2 molecule filtering"
                        )
                        if skip_invalid_reactions:
                            print(f"Warning: skipping invalid reaction direction: {message}")
                            continue
                        raise ValueError(message)
                    if reaction_id in seen_ids:
                        duplicate_ids += 1
                        continue
                    seen_ids.add(reaction_id)
                    records.append(
                        ReactionSetRecord(
                            reaction_id=reaction_id,
                            reactants=side_a,
                            products=side_b,
                        )
                    )
    if duplicate_ids:
        print(f"Warning: skipped {duplicate_ids} duplicate reaction IDs after augmentation")
    return records


def ordered_unique_molecules(records: list[ReactionSetRecord]) -> list[str]:
    seen: set[str] = set()
    molecules: list[str] = []
    for record in records:
        for smiles in (*record.reactants, *record.products):
            if smiles not in seen:
                seen.add(smiles)
                molecules.append(smiles)
    return molecules


def get_cls_representations(result) -> np.ndarray:
    if isinstance(result, dict):
        for key in ("cls_repr", "cls_reprs", "molecule_repr", "molecule_reprs"):
            if key in result:
                return np.asarray(result[key])
    return np.asarray(result)


def encode_molecules(
    molecules: list[str],
    model_name: str,
    model_size: str,
    batch_size: int,
    skip_invalid_molecules: bool,
) -> dict[str, np.ndarray]:
    try:
        from unimol_tools import UniMolRepr
    except ImportError as exc:
        raise ImportError(
            "unimol_tools is required for Uni-Mol2 extraction. "
            "Install the official Uni-Mol tools package before running this script."
        ) from exc

    unique_molecules = list(dict.fromkeys(molecules))
    repr_model = UniMolRepr(
        model_name=model_name,
        model_size=model_size,
        batch_size=batch_size,
    )
    cache: dict[str, np.ndarray] = {}
    for start in range(0, len(unique_molecules), batch_size):
        batch = unique_molecules[start : start + batch_size]
        try:
            result = repr_model.get_repr(batch, return_atomic_reprs=False)
            cls_reprs = get_cls_representations(result)
            if cls_reprs.ndim != 2 or cls_reprs.shape[0] != len(batch):
                raise ValueError(
                    "UniMolRepr returned an unexpected CLS representation shape: "
                    f"{cls_reprs.shape} for batch size {len(batch)}"
                )
            for smiles, vector in zip(batch, cls_reprs):
                cache[smiles] = np.asarray(vector)
        except Exception:
            if not skip_invalid_molecules:
                raise
            print(
                "Warning: batch Uni-Mol2 encoding failed; retrying molecules "
                "individually and skipping failures"
            )
            for smiles in batch:
                try:
                    result = repr_model.get_repr([smiles], return_atomic_reprs=False)
                    cls_reprs = get_cls_representations(result)
                    if cls_reprs.ndim != 2 or cls_reprs.shape[0] != 1:
                        raise ValueError(
                            "UniMolRepr returned an unexpected single-molecule "
                            f"shape: {cls_reprs.shape}"
                        )
                    cache[smiles] = np.asarray(cls_reprs[0])
                except Exception as exc:
                    print(f"Warning: skipping molecule after Uni-Mol2 failure: {smiles} ({exc})")
        print(
            f"Encoded {min(start + batch_size, len(unique_molecules))}/"
            f"{len(unique_molecules)} molecules"
        )
    return cache


def filter_records_by_encoded_molecules(
    records: list[ReactionSetRecord],
    molecule_cache: dict[str, np.ndarray],
    skip_invalid_molecules: bool,
    skip_invalid_reactions: bool,
) -> list[ReactionSetRecord]:
    filtered = []
    for record in records:
        reactants = tuple(smiles for smiles in record.reactants if smiles in molecule_cache)
        products = tuple(smiles for smiles in record.products if smiles in molecule_cache)
        if len(reactants) != len(record.reactants) or len(products) != len(record.products):
            if not skip_invalid_molecules:
                missing = sorted(
                    set((*record.reactants, *record.products)) - set(molecule_cache)
                )
                raise KeyError(
                    f"Missing Uni-Mol2 embeddings for reaction {record.reaction_id}: {missing}"
                )
        if not reactants or not products:
            message = (
                f"Reaction {record.reaction_id} has an empty side after "
                "dropping failed Uni-Mol2 molecules"
            )
            if skip_invalid_reactions:
                print(f"Warning: skipping invalid reaction direction: {message}")
                continue
            raise ValueError(message)
        filtered.append(
            ReactionSetRecord(
                reaction_id=record.reaction_id,
                reactants=reactants,
                products=products,
            )
        )
    return filtered


def flatten_side(
    records: list[ReactionSetRecord],
    molecule_cache: dict[str, np.ndarray],
    side: str,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    vectors = []
    offsets = [0]
    for record in records:
        molecules = getattr(record, side)
        for smiles in molecules:
            vectors.append(molecule_cache[smiles])
        offsets.append(offsets[-1] + len(molecules))
    return np.asarray(vectors, dtype=dtype), np.asarray(offsets, dtype=np.int64)


def write_hdf5(
    output: str,
    records: list[ReactionSetRecord],
    molecule_cache: dict[str, np.ndarray],
    model_name: str,
    model_size: str,
    dtype_name: str,
    compression: str,
    force: bool,
) -> None:
    output_path = Path(output)
    if output_path.exists() and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    dtype = np.float16 if dtype_name == "float16" else np.float32
    reactant_vectors, reactant_offsets = flatten_side(records, molecule_cache, "reactants", dtype)
    product_vectors, product_offsets = flatten_side(records, molecule_cache, "products", dtype)
    if reactant_vectors.shape[1] != product_vectors.shape[1]:
        raise ValueError("Reactant/product embedding dimensions do not match")

    h5_compression = None if compression == "none" else compression
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset(
            "ids",
            data=np.asarray([record.reaction_id for record in records], dtype=object),
            dtype=string_dtype,
        )
        handle.create_dataset("reactant_vectors", data=reactant_vectors, compression=h5_compression)
        handle.create_dataset("reactant_offsets", data=reactant_offsets)
        handle.create_dataset("product_vectors", data=product_vectors, compression=h5_compression)
        handle.create_dataset("product_offsets", data=product_offsets)
        handle.attrs["model_name"] = model_name
        handle.attrs["model_size"] = model_size
        handle.attrs["embedding_dim"] = int(reactant_vectors.shape[1])
        handle.attrs["dtype"] = dtype_name
        handle.attrs["reaction_representation"] = "unimol2_attention"


def main() -> None:
    args = parse_args()
    records = read_reactions(
        paths=args.reactions,
        key_column=args.key_column,
        smiles_column=args.smiles_column,
        standardize=args.standardize,
        skip_invalid_molecules=args.skip_invalid_molecules,
        skip_invalid_reactions=args.skip_invalid_reactions,
        bidirectional=args.bidirectional,
        allow_pseudo_reactions=args.allow_pseudo_reactions,
        max_atoms=args.max_atoms,
        skip_metals=args.skip_metals,
        standardizer_kwargs={
            "standardize_hypervalent": args.standardize_hypervalent,
            "standardize_remove_hs": args.standardize_remove_hs,
            "standardize_kekulize": args.standardize_kekulize,
            "standardize_uncharge": args.standardize_uncharge,
            "standardize_metals": args.standardize_metals,
        },
    )
    molecules = ordered_unique_molecules(records)
    direction_label = "bidirectional" if args.bidirectional else "paper-style unsuffixed"
    print(f"Loaded {len(records)} {direction_label} reactions")
    print(f"Encoding {len(molecules)} unique molecules with Uni-Mol2")
    molecule_cache = encode_molecules(
        molecules=molecules,
        model_name=args.model_name,
        model_size=args.model_size,
        batch_size=args.batch_size,
        skip_invalid_molecules=args.skip_invalid_molecules,
    )
    records = filter_records_by_encoded_molecules(
        records=records,
        molecule_cache=molecule_cache,
        skip_invalid_molecules=args.skip_invalid_molecules,
        skip_invalid_reactions=args.skip_invalid_reactions,
    )
    write_hdf5(
        output=args.output,
        records=records,
        molecule_cache=molecule_cache,
        model_name=args.model_name,
        model_size=args.model_size,
        dtype_name=args.dtype,
        compression=args.compression,
        force=args.force,
    )
    print(f"Wrote Uni-Mol2 reaction embeddings to {args.output}")


if __name__ == "__main__":
    main()
