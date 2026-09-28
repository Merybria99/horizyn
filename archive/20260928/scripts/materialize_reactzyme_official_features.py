#!/usr/bin/env python3
"""Materialize official ReactZyme reaction features from legacy train/val shards.

The previous derived ReactZyme data root split each official train set into
``train`` and random ``val`` subsets. Official ReactZyme train is the union of
those two subsets, but reaction IDs can appear in both. This script builds a
deduplicated official feature root:

- official ``train`` = old train + old val, deduplicated by directional ID
- official ``test`` = old test
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_ROOT = PROJECT_ROOT / "data/revised_protocols/reactzyme_official"
DEFAULT_FEATURE_ROOT = DEFAULT_PROTOCOL_ROOT / "features"
PROTOCOLS = ("time", "enzyme_smi", "reaction_smi")
MODALITIES = ("reactiont5v2", "unimol2", "chiro")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument(
        "--legacy-feature-root",
        type=Path,
        required=True,
        help=(
            "Source feature root to migrate from. For the July 2026 migration this was "
            "the now-removed random train/val ReactZyme feature root."
        ),
    )
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_reactions(path: Path) -> OrderedDict[str, str]:
    reactions: OrderedDict[str, str] = OrderedDict()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"reaction_id", "reaction_smiles"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            reaction_id = str(row["reaction_id"]).strip()
            reaction_smiles = str(row["reaction_smiles"]).strip()
            if not reaction_id:
                continue
            previous = reactions.get(reaction_id)
            if previous is not None and previous != reaction_smiles:
                raise ValueError(f"{path} has conflicting SMILES for {reaction_id}")
            reactions.setdefault(reaction_id, reaction_smiles)
    return reactions


def expected_directional_ids(reactions: OrderedDict[str, str]) -> list[str]:
    ids: list[str] = []
    for reaction_id in reactions:
        ids.append(f"{reaction_id}_f")
        ids.append(f"{reaction_id}_r")
    return ids


def decode_ids(values: np.ndarray) -> list[str]:
    ids: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            ids.append(value.decode("utf-8"))
        else:
            ids.append(str(value))
    return ids


def copy_attrs(source: h5py.File, dest: h5py.File, *, extra: dict[str, Any] | None = None) -> None:
    for key, value in source.attrs.items():
        dest.attrs[key] = value
    if extra:
        for key, value in extra.items():
            dest.attrs[key] = value


def source_index(paths: list[Path]) -> tuple[dict[str, tuple[Path, int]], dict[str, int]]:
    index: dict[str, tuple[Path, int]] = {}
    duplicates: dict[str, int] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            ids = decode_ids(handle["ids"][:])
        for row_idx, reaction_id in enumerate(ids):
            if reaction_id in index:
                duplicates[reaction_id] = duplicates.get(reaction_id, 1) + 1
                continue
            index[reaction_id] = (path, row_idx)
    return index, duplicates


def materialize_vector_h5(
    *,
    source_paths: list[Path],
    output_path: Path,
    expected_ids: list[str],
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index, duplicates = source_index(source_paths)
    kept_ids = [reaction_id for reaction_id in expected_ids if reaction_id in index]
    missing = [reaction_id for reaction_id in expected_ids if reaction_id not in index]

    source_cache: dict[Path, h5py.File] = {}
    try:
        first_path = source_paths[0]
        first = h5py.File(first_path, "r")
        source_cache[first_path] = first
        vector_dtype = first["vectors"].dtype
        vector_shape = first["vectors"].shape[1:]

        with h5py.File(output_path, "w") as out:
            string_dtype = h5py.string_dtype("utf-8")
            out.create_dataset("ids", data=np.asarray(kept_ids, dtype=object), dtype=string_dtype)
            vectors = out.create_dataset(
                "vectors",
                shape=(len(kept_ids), *vector_shape),
                dtype=vector_dtype,
            )
            for out_idx, reaction_id in enumerate(kept_ids):
                source_path, source_idx = index[reaction_id]
                handle = source_cache.get(source_path)
                if handle is None:
                    handle = h5py.File(source_path, "r")
                    source_cache[source_path] = handle
                vectors[out_idx] = handle["vectors"][source_idx]
            copy_attrs(
                first,
                out,
                extra={
                    "source_feature_files": json.dumps([relpath(path) for path in source_paths]),
                    "deduplicated": True,
                },
            )
    finally:
        for handle in source_cache.values():
            handle.close()

    return {
        "output": relpath(output_path),
        "expected": len(expected_ids),
        "observed": len(kept_ids),
        "missing": len(missing),
        "duplicate_source_ids": len(duplicates),
        "missing_examples": missing[:20],
    }


def materialize_ragged_h5(
    *,
    source_paths: list[Path],
    output_path: Path,
    expected_ids: list[str],
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index, duplicates = source_index(source_paths)
    kept_ids = [reaction_id for reaction_id in expected_ids if reaction_id in index]
    missing = [reaction_id for reaction_id in expected_ids if reaction_id not in index]

    source_cache: dict[Path, h5py.File] = {}
    reactant_chunks: list[np.ndarray] = []
    product_chunks: list[np.ndarray] = []
    reactant_offsets = [0]
    product_offsets = [0]
    try:
        first_path = source_paths[0]
        first = h5py.File(first_path, "r")
        source_cache[first_path] = first
        reactant_dtype = first["reactant_vectors"].dtype
        product_dtype = first["product_vectors"].dtype
        reactant_dim = first["reactant_vectors"].shape[1:]
        product_dim = first["product_vectors"].shape[1:]

        for reaction_id in kept_ids:
            source_path, source_idx = index[reaction_id]
            handle = source_cache.get(source_path)
            if handle is None:
                handle = h5py.File(source_path, "r")
                source_cache[source_path] = handle

            r0 = int(handle["reactant_offsets"][source_idx])
            r1 = int(handle["reactant_offsets"][source_idx + 1])
            p0 = int(handle["product_offsets"][source_idx])
            p1 = int(handle["product_offsets"][source_idx + 1])
            reactants = handle["reactant_vectors"][r0:r1]
            products = handle["product_vectors"][p0:p1]
            reactant_chunks.append(reactants)
            product_chunks.append(products)
            reactant_offsets.append(reactant_offsets[-1] + len(reactants))
            product_offsets.append(product_offsets[-1] + len(products))

        if reactant_chunks:
            reactant_vectors = np.concatenate(reactant_chunks, axis=0)
        else:
            reactant_vectors = np.empty((0, *reactant_dim), dtype=reactant_dtype)
        if product_chunks:
            product_vectors = np.concatenate(product_chunks, axis=0)
        else:
            product_vectors = np.empty((0, *product_dim), dtype=product_dtype)

        with h5py.File(output_path, "w") as out:
            string_dtype = h5py.string_dtype("utf-8")
            out.create_dataset("ids", data=np.asarray(kept_ids, dtype=object), dtype=string_dtype)
            out.create_dataset("reactant_offsets", data=np.asarray(reactant_offsets, dtype=np.int64))
            out.create_dataset("product_offsets", data=np.asarray(product_offsets, dtype=np.int64))
            out.create_dataset("reactant_vectors", data=reactant_vectors, dtype=reactant_dtype)
            out.create_dataset("product_vectors", data=product_vectors, dtype=product_dtype)
            copy_attrs(
                first,
                out,
                extra={
                    "source_feature_files": json.dumps([relpath(path) for path in source_paths]),
                    "deduplicated": True,
                },
            )
    finally:
        for handle in source_cache.values():
            handle.close()

    return {
        "output": relpath(output_path),
        "expected": len(expected_ids),
        "observed": len(kept_ids),
        "missing": len(missing),
        "duplicate_source_ids": len(duplicates),
        "missing_examples": missing[:20],
    }


def materialize_modality(
    *,
    modality: str,
    source_paths: list[Path],
    output_path: Path,
    expected_ids: list[str],
    overwrite: bool,
) -> dict[str, Any]:
    if modality == "reactiont5v2":
        return materialize_vector_h5(
            source_paths=source_paths,
            output_path=output_path,
            expected_ids=expected_ids,
            overwrite=overwrite,
        )
    return materialize_ragged_h5(
        source_paths=source_paths,
        output_path=output_path,
        expected_ids=expected_ids,
        overwrite=overwrite,
    )


def main() -> None:
    args = parse_args()
    if args.features_root.exists() and args.overwrite:
        shutil.rmtree(args.features_root)
    args.features_root.mkdir(parents=True, exist_ok=True)

    manifests: list[dict[str, Any]] = []
    for protocol in args.protocols:
        protocol_dir = args.protocol_root / protocol
        for split in ("train", "test"):
            expected_ids = expected_directional_ids(
                read_reactions(protocol_dir / f"{split}_rxns.csv")
            )
            coverage: dict[str, Any] = {}
            for modality in args.modalities:
                if split == "train":
                    source_paths = [
                        args.legacy_feature_root / protocol / "train" / f"{modality}.h5",
                        args.legacy_feature_root / protocol / "val" / f"{modality}.h5",
                    ]
                else:
                    source_paths = [
                        args.legacy_feature_root / protocol / "test" / f"{modality}.h5"
                    ]
                output_path = args.features_root / protocol / split / f"{modality}.h5"
                coverage[modality] = materialize_modality(
                    modality=modality,
                    source_paths=source_paths,
                    output_path=output_path,
                    expected_ids=expected_ids,
                    overwrite=args.overwrite,
                )

            manifest = {
                "protocol": protocol,
                "split": split,
                "reactions_csv": relpath(protocol_dir / f"{split}_rxns.csv"),
                "expected_directional_ids": len(expected_ids),
                "coverage": coverage,
            }
            out_dir = args.features_root / protocol / split
            (out_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifests.append(manifest)

    summary = {"splits": manifests}
    (args.features_root / "summary_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
