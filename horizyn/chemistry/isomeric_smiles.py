"""Strict canonical-isomeric SMILES materialization and provenance."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase


SMILES_MODE = "canonical_isomeric"
NORMALIZER_VERSION = "horizyn_canonical_isomeric_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize_isomeric_component(smiles: str) -> str:
    """Canonicalize one molecule without removing maps, charges, or stereo."""

    value = str(smiles).strip()
    if not value:
        raise ValueError("empty molecular component")
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise ValueError(f"invalid molecular SMILES: {value}")
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )


def _canonicalize_side(side: str, *, allow_empty: bool) -> str:
    value = side.strip()
    if not value:
        if allow_empty:
            return ""
        raise ValueError("empty reaction side")
    components = value.split(".")
    if any(not component.strip() for component in components):
        raise ValueError(f"empty dot-separated component in: {side}")
    return ".".join(canonicalize_isomeric_component(component) for component in components)


def canonicalize_isomeric_reaction_smiles(reaction_smiles: str) -> str:
    """Canonicalize molecules while preserving reaction-side/component order.

    Plain molecule-set strings and both ``reactants>>products`` and
    ``reactants>agents>products`` forms are accepted. No stereochemistry is
    inferred when it is absent from the input.
    """

    text = str(reaction_smiles).strip()
    if not text:
        raise ValueError("empty reaction SMILES")
    if ">" not in text:
        return _canonicalize_side(text, allow_empty=False)
    sections = text.split(">")
    if len(sections) != 3:
        raise ValueError(f"reaction SMILES must have exactly three >-separated fields: {text}")
    reactants, agents, products = sections
    return ">".join(
        (
            _canonicalize_side(reactants, allow_empty=False),
            _canonicalize_side(agents, allow_empty=True),
            _canonicalize_side(products, allow_empty=False),
        )
    )


def materialize_isomeric_reaction_csv(
    source_path: str | Path,
    output_path: str | Path,
    *,
    provenance_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write a row-preserving canonical-isomeric reaction CSV and manifest."""

    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{source} has no CSV header")
        fieldnames = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    required = {"reaction_id", "reaction_smiles"}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"{source} missing columns: {sorted(missing)}")
    reaction_ids = [str(row["reaction_id"]) for row in rows]
    seen: set[str] = set()
    duplicate = None
    for reaction_id in reaction_ids:
        if reaction_id in seen:
            duplicate = reaction_id
            break
        seen.add(reaction_id)
    if duplicate is not None:
        raise ValueError(f"{source} contains duplicate reaction_id: {duplicate}")

    normalized_rows: list[dict[str, str]] = []
    changed = 0
    for row_idx, row in enumerate(rows):
        reaction_id = str(row["reaction_id"])
        original = str(row["reaction_smiles"]).strip()
        try:
            canonical = canonicalize_isomeric_reaction_smiles(original)
        except ValueError as exc:
            raise ValueError(
                f"Failed to normalize {source} row={row_idx} reaction_id={reaction_id}: {exc}"
            ) from exc
        normalized_rows.append(
            {**row, "reaction_id": reaction_id, "reaction_smiles": canonical}
        )
        changed += int(canonical != original)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)

    manifest_path = (
        Path(provenance_path).resolve()
        if provenance_path is not None
        else output.with_suffix(output.suffix + ".provenance.json")
    )
    report: dict[str, Any] = {
        "smiles_mode": SMILES_MODE,
        "normalizer_version": NORMALIZER_VERSION,
        "rdkit_version": rdBase.rdkitVersion,
        "source_csv": str(source),
        "source_csv_sha256": sha256_file(source),
        "output_csv": str(output),
        "output_csv_sha256": sha256_file(output),
        "row_count": len(rows),
        "changed_row_count": int(changed),
        "unchanged_row_count": len(rows) - int(changed),
        "coverage": 1.0 if rows else 0.0,
        "preserves_component_order": True,
        "preserves_atom_maps": True,
        "infers_missing_stereochemistry": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
