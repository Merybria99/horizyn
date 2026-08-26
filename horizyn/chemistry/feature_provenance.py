"""Provenance stamping and validation for reaction feature artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from horizyn.chemistry.isomeric_smiles import (
    NORMALIZER_VERSION,
    SMILES_MODE,
    sha256_file,
)


REQUIRED_H5_PROVENANCE_ATTRS = (
    "smiles_mode",
    "source_csv_sha256",
    "normalizer_version",
    "extractor_name",
    "extractor_version",
    "feature_coverage",
)


def _decode_ids(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _base_reaction_id(reaction_id: str) -> str:
    return reaction_id[:-2] if reaction_id.endswith(("_f", "_r")) else reaction_id


def stamp_h5_reaction_feature_provenance(
    artifact_path: str | Path,
    source_csv_path: str | Path,
    *,
    extractor_name: str,
    extractor_version: str,
) -> dict[str, Any]:
    artifact = Path(artifact_path).resolve()
    source = Path(source_csv_path).resolve()
    source_frame = pd.read_csv(source)
    if "reaction_id" not in source_frame.columns:
        raise ValueError(f"{source} is missing reaction_id")
    source_ids = set(source_frame["reaction_id"].astype(str))
    with h5py.File(artifact, "r+") as handle:
        if "ids" not in handle:
            raise ValueError(f"{artifact} is missing /ids")
        artifact_ids = _decode_ids(handle["ids"][:])
        covered_ids = {_base_reaction_id(value) for value in artifact_ids} & source_ids
        coverage = len(covered_ids) / len(source_ids) if source_ids else 0.0
        attrs = {
            "smiles_mode": SMILES_MODE,
            "source_csv": str(source),
            "source_csv_sha256": sha256_file(source),
            "normalizer_version": NORMALIZER_VERSION,
            "extractor_name": str(extractor_name),
            "extractor_version": str(extractor_version),
            "feature_coverage": float(coverage),
            "source_reaction_count": int(len(source_ids)),
            "artifact_id_count": int(len(artifact_ids)),
        }
        for key, value in attrs.items():
            handle.attrs[key] = value
    return {"artifact": str(artifact), **attrs}


def validate_h5_reaction_feature_provenance(
    artifact_path: str | Path,
    source_csv_path: str | Path,
    *,
    minimum_coverage: float = 1.0,
) -> dict[str, Any]:
    artifact = Path(artifact_path).resolve()
    source = Path(source_csv_path).resolve()
    with h5py.File(artifact, "r") as handle:
        missing = [key for key in REQUIRED_H5_PROVENANCE_ATTRS if key not in handle.attrs]
        if missing:
            raise ValueError(f"{artifact} lacks required isomeric provenance attrs: {missing}")
        attrs = {key: handle.attrs[key] for key in REQUIRED_H5_PROVENANCE_ATTRS}
    if str(attrs["smiles_mode"]) != SMILES_MODE:
        raise ValueError(f"{artifact} has smiles_mode={attrs['smiles_mode']!r}, expected {SMILES_MODE!r}")
    expected_sha = sha256_file(source)
    if str(attrs["source_csv_sha256"]) != expected_sha:
        raise ValueError(f"{artifact} was not extracted from {source}")
    if float(attrs["feature_coverage"]) < minimum_coverage:
        raise ValueError(
            f"{artifact} feature coverage {float(attrs['feature_coverage']):.6f} "
            f"is below {minimum_coverage:.6f}"
        )
    return {key: value.item() if hasattr(value, "item") else value for key, value in attrs.items()}


def write_provenance_report(path: str | Path, reports: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n", encoding="utf-8")

