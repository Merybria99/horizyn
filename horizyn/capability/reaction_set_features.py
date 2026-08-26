"""Paper-compatible reaction features derived from unordered molecule sets."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Lipinski, rdFingerprintGenerator, rdMolDescriptors

from horizyn.capability.cofactors import (
    extract_cofactor_labels,
    load_cofactor_aliases,
    split_cofactor_label_tiers,
)
from horizyn.chemistry.isomeric_smiles import NORMALIZER_VERSION, SMILES_MODE, sha256_file


DESCRIPTOR_NAMES = (
    "mol_weight",
    "heavy_atoms",
    "hetero_atoms",
    "formal_charge",
    "hbond_donors",
    "hbond_acceptors",
    "tpsa",
    "logp",
    "rotatable_bonds",
    "rings",
    "aromatic_fraction",
    "stereocenters",
    "wildcard_atoms",
    "nitrogen_atoms",
    "oxygen_atoms",
    "phosphorus_sulfur_atoms",
)
AGGREGATIONS = ("sum", "mean", "max", "std")
COFACTOR_CAPACITY = 37


def _canonical_component(smiles: str) -> tuple[str, Chem.Mol | None]:
    value = str(smiles).strip()
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        return value, None
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True), molecule


def molecule_set_components(reaction_smiles: str) -> list[str]:
    """Return each participant once when a molecule set was stored as ``S>>S``."""

    text = str(reaction_smiles).strip()
    if ">>" not in text:
        return [value for value in text.split(".") if value]
    left, right = text.split(">>", maxsplit=1)
    left_values = [value for value in left.split(".") if value]
    right_values = [value for value in right.split(".") if value]
    left_key = Counter(_canonical_component(value)[0] for value in left_values)
    right_key = Counter(_canonical_component(value)[0] for value in right_values)
    return left_values if left_key == right_key else left_values + right_values


def molecule_descriptors(molecule: Chem.Mol) -> np.ndarray:
    atoms = list(molecule.GetAtoms())
    heavy_atoms = [atom for atom in atoms if atom.GetAtomicNum() > 1]
    aromatic_count = sum(atom.GetIsAromatic() for atom in heavy_atoms)
    counts = Counter(atom.GetSymbol() for atom in atoms)
    return np.asarray(
        [
            rdMolDescriptors.CalcExactMolWt(molecule),
            len(heavy_atoms),
            sum(atom.GetAtomicNum() not in {0, 1, 6} for atom in atoms),
            Chem.GetFormalCharge(molecule),
            Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule),
            rdMolDescriptors.CalcTPSA(molecule),
            Crippen.MolLogP(molecule),
            Lipinski.NumRotatableBonds(molecule),
            Lipinski.RingCount(molecule),
            aromatic_count / max(len(heavy_atoms), 1),
            len(Chem.FindMolChiralCenters(molecule, includeUnassigned=True)),
            counts.get("*", 0),
            counts.get("N", 0),
            counts.get("O", 0),
            counts.get("P", 0) + counts.get("S", 0),
        ],
        dtype=np.float32,
    )


def aggregate_descriptors(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.zeros(len(DESCRIPTOR_NAMES) * len(AGGREGATIONS), dtype=np.float32)
    matrix = np.stack(rows).astype(np.float32)
    return np.concatenate(
        [matrix.sum(axis=0), matrix.mean(axis=0), matrix.max(axis=0), matrix.std(axis=0)]
    ).astype(np.float32)


def _read_reactions(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        fields = set(reader.fieldnames)
        rows = [dict(row) for row in reader]
    required = {"reaction_id", "reaction_smiles"}
    missing = required - fields
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    reaction_ids = [str(row["reaction_id"]) for row in rows]
    if len(reaction_ids) != len(set(reaction_ids)):
        raise ValueError(f"{path} contains duplicate reaction_id values")
    return rows


def _raw_rows(
    frame: list[dict[str, str]],
    *,
    cofactor_aliases: dict[str, Any],
    morgan_bits: int,
) -> list[dict[str, Any]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=morgan_bits)
    rows: list[dict[str, Any]] = []
    for row in frame:
        components = molecule_set_components(row["reaction_smiles"])
        canonical: list[str] = []
        descriptors: list[np.ndarray] = []
        fingerprints: list[np.ndarray] = []
        wildcard_components = 0
        for component in components:
            canonical_smiles, molecule = _canonical_component(component)
            canonical.append(canonical_smiles)
            if molecule is None:
                continue
            descriptor = molecule_descriptors(molecule)
            descriptors.append(descriptor)
            wildcard_components += int(descriptor[12] > 0)
            fingerprint = np.zeros(morgan_bits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), fingerprint)
            fingerprints.append(fingerprint)
        cofactor_labels, _, _ = extract_cofactor_labels(
            [str(row["reaction_smiles"])],
            cofactor_aliases=cofactor_aliases,
        )
        core_cofactors = split_cofactor_label_tiers(cofactor_labels)[
            "core_cofactor_labels"
        ]
        valid_count = len(descriptors)
        component_count = len(components)
        rows.append(
            {
                "reaction_id": str(row["reaction_id"]),
                "descriptor": aggregate_descriptors(descriptors),
                "fingerprint": (
                    np.stack(fingerprints).mean(axis=0).astype(np.float32)
                    if fingerprints
                    else np.zeros(morgan_bits, dtype=np.float32)
                ),
                "set_stats": np.asarray(
                    [
                        component_count,
                        len(set(canonical)) / max(component_count, 1),
                        valid_count / max(component_count, 1),
                        wildcard_components / max(valid_count, 1),
                    ],
                    dtype=np.float32,
                ),
                "core_cofactors": sorted(set(core_cofactors)),
                "valid": valid_count > 0,
            }
        )
    return rows


def _fit_schema(rows: list[dict[str, Any]], morgan_bits: int) -> dict[str, Any]:
    valid_descriptors = [row["descriptor"] for row in rows if row["valid"]]
    descriptor_matrix = (
        np.stack(valid_descriptors)
        if valid_descriptors
        else np.zeros((1, len(DESCRIPTOR_NAMES) * len(AGGREGATIONS)), dtype=np.float32)
    )
    mean = descriptor_matrix.mean(axis=0).astype(np.float32)
    std = descriptor_matrix.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    cofactor_vocab = sorted(
        {label for row in rows for label in row["core_cofactors"]}
    )
    if len(cofactor_vocab) > COFACTOR_CAPACITY:
        raise ValueError(
            f"Training cofactor vocabulary has {len(cofactor_vocab)} labels, "
            f"exceeding capacity {COFACTOR_CAPACITY}"
        )
    return {
        "schema_version": "reactzyme_reaction_set_features_v1",
        "morgan_radius": 2,
        "morgan_bits": int(morgan_bits),
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "descriptor_aggregations": list(AGGREGATIONS),
        "descriptor_mean": mean.tolist(),
        "descriptor_std": std.tolist(),
        "set_stat_names": [
            "component_count",
            "unique_component_fraction",
            "valid_component_fraction",
            "wildcard_component_fraction",
        ],
        "core_cofactor_vocab": cofactor_vocab,
        "core_cofactor_capacity": COFACTOR_CAPACITY,
        "fit_split": "train",
    }


def _materialize(rows: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, np.ndarray]:
    mean = np.asarray(schema["descriptor_mean"], dtype=np.float32)
    std = np.asarray(schema["descriptor_std"], dtype=np.float32)
    vocab = {label: idx for idx, label in enumerate(schema["core_cofactor_vocab"])}
    cofactor_capacity = int(schema["core_cofactor_capacity"])
    vectors: list[np.ndarray] = []
    masks: list[bool] = []
    for row in rows:
        cofactor = np.zeros(cofactor_capacity, dtype=np.float32)
        for label in row["core_cofactors"]:
            if label in vocab:
                cofactor[vocab[label]] = 1.0
        descriptor = (row["descriptor"] - mean) / std if row["valid"] else row["descriptor"]
        vectors.append(
            np.concatenate(
                [row["fingerprint"], descriptor, row["set_stats"], cofactor]
            ).astype(np.float32)
        )
        masks.append(bool(row["valid"]))
    return {
        "ids": np.asarray([row["reaction_id"] for row in rows], dtype=object),
        "vectors": np.stack(vectors).astype(np.float32),
        "mask": np.asarray(masks, dtype=bool),
    }


def build_reaction_set_feature_splits(
    *,
    train_reactions_path: str | Path,
    validation_reactions_path: str | Path,
    test_reactions_path: str | Path,
    cofactor_dictionary_path: str | Path,
    out_dir: str | Path,
    morgan_bits: int = 512,
) -> dict[str, Any]:
    if morgan_bits <= 0:
        raise ValueError("morgan_bits must be positive")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    aliases = load_cofactor_aliases(cofactor_dictionary_path)
    frames = {
        "train": _read_reactions(train_reactions_path),
        "validation": _read_reactions(validation_reactions_path),
        "test": _read_reactions(test_reactions_path),
    }
    raw = {
        split: _raw_rows(frame, cofactor_aliases=aliases, morgan_bits=morgan_bits)
        for split, frame in frames.items()
    }
    schema = _fit_schema(raw["train"], morgan_bits)
    schema.update(
        {
            "smiles_mode": SMILES_MODE,
            "normalizer_version": NORMALIZER_VERSION,
            "source_csv_sha256": {
                "train": sha256_file(train_reactions_path),
                "validation": sha256_file(validation_reactions_path),
                "test": sha256_file(test_reactions_path),
            },
        }
    )
    report: dict[str, Any] = {"schema": schema, "splits": {}}
    for split, rows in raw.items():
        payload = _materialize(rows, schema)
        np.savez_compressed(out / f"{split}_reaction_set_features.npz", **payload)
        report["splits"][split] = {
            "num_reactions": len(rows),
            "num_valid": int(payload["mask"].sum()),
            "dimension": int(payload["vectors"].shape[1]),
            "num_with_core_cofactor": sum(bool(row["core_cofactors"]) for row in rows),
        }
    (out / "schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "coverage.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def materialize_reaction_set_features(
    *,
    reactions_path: str | Path,
    schema_path: str | Path,
    cofactor_dictionary_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Transform reactions with a previously fitted molecule-set feature schema."""

    schema_file = Path(schema_path)
    with schema_file.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if schema.get("schema_version") != "reactzyme_reaction_set_features_v1":
        raise ValueError(f"Unsupported reaction-set feature schema: {schema_file}")

    morgan_bits = int(schema["morgan_bits"])
    aliases = load_cofactor_aliases(cofactor_dictionary_path)
    frame = _read_reactions(reactions_path)
    rows = _raw_rows(
        frame,
        cofactor_aliases=aliases,
        morgan_bits=morgan_bits,
    )
    payload = _materialize(rows, schema)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return {
        "schema": str(schema_file),
        "output": str(destination),
        "num_reactions": len(rows),
        "num_valid": int(payload["mask"].sum()),
        "dimension": int(payload["vectors"].shape[1]),
        "num_with_core_cofactor": sum(bool(row["core_cofactors"]) for row in rows),
    }


__all__ = [
    "DESCRIPTOR_NAMES",
    "COFACTOR_CAPACITY",
    "aggregate_descriptors",
    "build_reaction_set_feature_splits",
    "materialize_reaction_set_features",
    "molecule_descriptors",
    "molecule_set_components",
]
