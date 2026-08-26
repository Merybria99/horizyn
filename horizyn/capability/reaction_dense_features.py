"""Train-fitted dense reaction transformation features for E2R retrieval."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from drfp import DrfpEncoder
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from horizyn.capability.reaction_set_features import molecule_descriptors


MORGAN_BITS = 512
DRFP_BITS = 512
CENTER_BITS = 64
SIDE_STAT_DIM = 8
DENSE_REACTION_DIM = (
    MORGAN_BITS
    + DRFP_BITS
    + 16
    + SIDE_STAT_DIM
    + CENTER_BITS
    + 2
)


def _read_reactions(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"reaction_id", "reaction_smiles"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain reaction_id and reaction_smiles")
    return rows


def _read_directional_reactions(
    path: str | Path | None,
) -> dict[str, str]:
    if path is None:
        return {}
    rows = _read_reactions(path)
    return {
        str(row["reaction_id"]): str(row["reaction_smiles"])
        for row in rows
        if ">>" in str(row["reaction_smiles"])
    }


def _labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [str(item) for item in value if str(item)]
    try:
        if bool(pd.isna(value)):
            return []
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return [str(item) for item in parsed if str(item)]
    return [text]


def _center_labels(path: str | Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    frame = pd.read_parquet(path)
    if "reaction_id" not in frame.columns:
        raise ValueError(f"{path} must contain reaction_id")
    columns = [
        name
        for name in ("reaction_center_raw_labels", "reaction_center_coarse_labels")
        if name in frame.columns
    ]
    result: dict[str, list[str]] = {}
    for row in frame.to_dict("records"):
        result[str(row["reaction_id"])] = sorted(
            {
                label
                for column in columns
                for label in _labels(row.get(column))
            }
        )
    return result


def _molecules(side: str) -> list[Chem.Mol]:
    molecules = []
    for component in str(side).split("."):
        component = component.strip()
        if not component:
            continue
        molecule = Chem.MolFromSmiles(component)
        if molecule is not None:
            for atom in molecule.GetAtoms():
                atom.SetAtomMapNum(0)
            molecules.append(molecule)
    return molecules


def _mean_fingerprint(
    molecules: list[Chem.Mol],
    generator,
) -> np.ndarray:
    if not molecules:
        return np.zeros(MORGAN_BITS, dtype=np.float32)
    rows = []
    for molecule in molecules:
        row = np.zeros(MORGAN_BITS, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), row)
        rows.append(row)
    return np.stack(rows).mean(axis=0).astype(np.float32)


def _descriptor_sum(molecules: list[Chem.Mol]) -> np.ndarray:
    if not molecules:
        return np.zeros(16, dtype=np.float32)
    return np.stack([molecule_descriptors(mol) for mol in molecules]).sum(axis=0)


def _side_stats(molecules: list[Chem.Mol]) -> np.ndarray:
    atoms = [atom for molecule in molecules for atom in molecule.GetAtoms()]
    bonds = [bond for molecule in molecules for bond in molecule.GetBonds()]
    return np.asarray(
        [
            len(molecules),
            len(atoms),
            len(bonds),
            sum(atom.GetIsAromatic() for atom in atoms),
            sum(atom.GetFormalCharge() for atom in atoms),
            sum(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in atoms),
            sum(bond.GetIsAromatic() for bond in bonds),
            sum(bond.GetBondTypeAsDouble() for bond in bonds),
        ],
        dtype=np.float32,
    )


def _hashed_center(labels: list[str]) -> np.ndarray:
    vector = np.zeros(CENTER_BITS, dtype=np.float32)
    for label in labels:
        digest = hashlib.sha256(label.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % CENTER_BITS
        sign = 1.0 if digest[8] & 1 else -1.0
        vector[index] += sign
    return vector


def _raw_rows(
    reactions: list[dict[str, str]],
    centers: dict[str, list[str]],
    directional_reactions: dict[str, str],
) -> list[dict[str, Any]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=MORGAN_BITS,
        includeChirality=True,
    )
    rows = []
    for row in reactions:
        reaction_id = str(row["reaction_id"])
        reaction_smiles = directional_reactions.get(
            reaction_id,
            str(row["reaction_smiles"]),
        )
        parts = reaction_smiles.split(">>")
        parse_ok = len(parts) == 2
        reactants = _molecules(parts[0]) if parse_ok else []
        products = _molecules(parts[1]) if parse_ok else []
        parse_ok = parse_ok and bool(reactants) and bool(products)
        center_labels = centers.get(reaction_id, [])
        if parse_ok:
            morgan_delta = _mean_fingerprint(products, generator) - _mean_fingerprint(
                reactants,
                generator,
            )
            descriptor_delta = _descriptor_sum(products) - _descriptor_sum(reactants)
            side_delta = _side_stats(products) - _side_stats(reactants)
            try:
                drfp = np.asarray(
                    DrfpEncoder.encode(
                        [reaction_smiles],
                        n_folded_length=DRFP_BITS,
                        radius=3,
                        rings=True,
                    )[0],
                    dtype=np.float32,
                )
            except Exception:
                drfp = np.zeros(DRFP_BITS, dtype=np.float32)
        else:
            morgan_delta = np.zeros(MORGAN_BITS, dtype=np.float32)
            drfp = np.zeros(DRFP_BITS, dtype=np.float32)
            descriptor_delta = np.zeros(16, dtype=np.float32)
            side_delta = np.zeros(SIDE_STAT_DIM, dtype=np.float32)
        rows.append(
            {
                "reaction_id": reaction_id,
                "morgan_delta": morgan_delta,
                "drfp": drfp,
                "descriptor_delta": descriptor_delta,
                "side_delta": side_delta,
                "center": _hashed_center(center_labels),
                "parse_ok": bool(parse_ok),
                "mapping_ok": bool(center_labels),
            }
        )
    return rows


def _fit_schema(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row["descriptor_delta"] for row in rows if row["parse_ok"]]
    matrix = np.stack(valid) if valid else np.zeros((1, 16), dtype=np.float32)
    mean = matrix.mean(axis=0).astype(np.float32)
    std = matrix.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return {
        "schema_version": "e2r_dense_reaction_features_v2",
        "dimension": DENSE_REACTION_DIM,
        "fit_split": "train",
        "morgan_bits": MORGAN_BITS,
        "drfp_bits": DRFP_BITS,
        "center_bits": CENTER_BITS,
        "descriptor_mean": mean.tolist(),
        "descriptor_std": std.tolist(),
        "blocks": [
            "signed_morgan_delta",
            "drfp",
            "standardized_descriptor_delta",
            "side_stat_delta",
            "hashed_mapped_center",
            "parse_and_mapping_masks",
        ],
    }


def _materialize(
    rows: list[dict[str, Any]],
    schema: dict[str, Any],
) -> dict[str, np.ndarray]:
    mean = np.asarray(schema["descriptor_mean"], dtype=np.float32)
    std = np.asarray(schema["descriptor_std"], dtype=np.float32)
    vectors = []
    for row in rows:
        descriptor = (row["descriptor_delta"] - mean) / std
        if not row["parse_ok"]:
            descriptor = np.zeros_like(descriptor)
        vector = np.concatenate(
            [
                row["morgan_delta"],
                row["drfp"],
                descriptor.astype(np.float32),
                row["side_delta"],
                row["center"],
                np.asarray(
                    [float(row["parse_ok"]), float(row["mapping_ok"])],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        if vector.shape != (DENSE_REACTION_DIM,):
            raise RuntimeError(f"Unexpected dense reaction shape: {vector.shape}")
        vectors.append(vector)
    return {
        "ids": np.asarray([row["reaction_id"] for row in rows], dtype=object),
        "vectors": np.stack(vectors),
        "mask": np.asarray([row["parse_ok"] for row in rows], dtype=bool),
    }


def build_dense_reaction_feature_splits(
    *,
    train_reactions_path: str | Path,
    validation_reactions_path: str | Path,
    test_reactions_path: str | Path,
    out_dir: str | Path,
    train_center_features_path: str | Path | None = None,
    validation_center_features_path: str | Path | None = None,
    test_center_features_path: str | Path | None = None,
    train_directional_reactions_path: str | Path | None = None,
    validation_directional_reactions_path: str | Path | None = None,
    test_directional_reactions_path: str | Path | None = None,
) -> dict[str, Any]:
    split_paths = {
        "train": train_reactions_path,
        "validation": validation_reactions_path,
        "test": test_reactions_path,
    }
    center_paths = {
        "train": train_center_features_path,
        "validation": validation_center_features_path,
        "test": test_center_features_path,
    }
    directional_paths = {
        "train": train_directional_reactions_path,
        "validation": validation_directional_reactions_path,
        "test": test_directional_reactions_path,
    }
    raw = {
        split: _raw_rows(
            _read_reactions(path),
            _center_labels(center_paths[split]),
            _read_directional_reactions(directional_paths[split]),
        )
        for split, path in split_paths.items()
    }
    schema = _fit_schema(raw["train"])
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report: dict[str, Any] = {
        "schema_version": schema["schema_version"],
        "dimension": DENSE_REACTION_DIM,
        "splits": {},
    }
    for split, rows in raw.items():
        payload = _materialize(rows, schema)
        np.savez_compressed(output / f"{split}_dense_reaction_features.npz", **payload)
        report["splits"][split] = {
            "reactions": len(rows),
            "parse_coverage": float(np.mean(payload["mask"])),
            "mapping_coverage": float(np.mean([row["mapping_ok"] for row in rows])),
        }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
