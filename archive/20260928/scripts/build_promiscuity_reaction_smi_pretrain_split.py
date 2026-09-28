#!/usr/bin/env python3
"""Build a ReactZyme reaction-SMILES-safe source-collapse pretraining split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[1]
RDLogger.DisableLog("rdApp.warning")


def canonical_molecule(smiles: str) -> str:
    value = str(smiles).strip()
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        return value
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def participant_multiset(reaction_smiles: str) -> tuple[str, ...]:
    text = str(reaction_smiles).strip()
    if ">>" not in text:
        components = [canonical_molecule(value) for value in text.split(".") if value]
        return tuple(sorted(components))
    left, right = text.split(">>", maxsplit=1)
    left_values = [canonical_molecule(value) for value in left.split(".") if value]
    right_values = [canonical_molecule(value) for value in right.split(".") if value]
    components = (
        left_values
        if Counter(left_values) == Counter(right_values)
        else left_values + right_values
    )
    return tuple(sorted(components))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-pairs",
        type=Path,
        default=ROOT
        / "runs/source_collapse_f3mc_ec_lambda3_95_5_v1/data/split/train_pairs.csv",
    )
    parser.add_argument(
        "--source-reactions",
        type=Path,
        default=ROOT
        / "runs/source_collapse_f3mc_ec_lambda3_95_5_v1/data/split/train_rxns.csv",
    )
    parser.add_argument(
        "--validation-pairs",
        type=Path,
        default=ROOT
        / "data/revised_protocols/reactzyme_paper/reaction_smi/validation_pairs.csv",
    )
    parser.add_argument(
        "--heldout-test-reactions",
        type=Path,
        default=ROOT
        / "data/revised_protocols/reactzyme_official/reaction_smi/test_rxns.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs/promiscuity_k2_reaction_smi/data/source_pretrain",
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_protein_id(value: str) -> str:
    text = str(value).strip()
    return text.rsplit("_", maxsplit=1)[-1]


def main() -> None:
    args = parse_args()
    pair_fields, source_pairs = read_csv(args.source_pairs)
    reaction_fields, source_reactions = read_csv(args.source_reactions)
    _validation_fields, validation_pairs = read_csv(args.validation_pairs)
    _test_fields, test_reactions = read_csv(args.heldout_test_reactions)

    required_pair_fields = {"reaction_id", "protein_id"}
    required_reaction_fields = {"reaction_id", "reaction_smiles"}
    if not required_pair_fields.issubset(pair_fields):
        missing = sorted(required_pair_fields - set(pair_fields))
        raise ValueError(f"Source pairs are missing {missing}")
    if not required_reaction_fields.issubset(reaction_fields):
        raise ValueError(
            "Source reactions are missing "
            f"{sorted(required_reaction_fields - set(reaction_fields))}"
        )

    source_keys = {
        row["reaction_id"]: participant_multiset(row["reaction_smiles"])
        for row in source_reactions
    }
    heldout_test_keys = {
        participant_multiset(row["reaction_smiles"])
        for row in test_reactions
        if row.get("reaction_smiles", "").strip()
    }
    forbidden_validation_associations = {
        (
            canonical_protein_id(row["protein_id"]),
            participant_multiset(row["reaction_smiles"]),
        )
        for row in validation_pairs
        if row.get("protein_id", "").strip() and row.get("reaction_smiles", "").strip()
    }

    kept_pairs: list[dict[str, Any]] = []
    removed_test_reaction_edges = 0
    removed_validation_edges = 0
    for row in source_pairs:
        reaction_id = row["reaction_id"]
        reaction_key = source_keys.get(reaction_id)
        if reaction_key is None:
            raise ValueError(f"Source pair references missing reaction {reaction_id!r}")
        if reaction_key in heldout_test_keys:
            removed_test_reaction_edges += 1
            continue
        association = (canonical_protein_id(row["protein_id"]), reaction_key)
        if association in forbidden_validation_associations:
            removed_validation_edges += 1
            continue
        kept = dict(row)
        if "pr_id" in pair_fields:
            kept["pr_id"] = str(len(kept_pairs))
        kept_pairs.append(kept)

    kept_reaction_ids = {row["reaction_id"] for row in kept_pairs}
    kept_reactions = [
        row for row in source_reactions if row["reaction_id"] in kept_reaction_ids
    ]
    output_pairs = args.output_dir / "train_pairs.csv"
    output_reactions = args.output_dir / "train_rxns.csv"
    output_manifest = args.output_dir / "manifest.json"
    write_csv(output_pairs, pair_fields, kept_pairs)
    write_csv(output_reactions, reaction_fields, kept_reactions)

    manifest = {
        "schema_version": "promiscuity_reaction_smi_pretrain_split_v1",
        "source_pairs": str(args.source_pairs),
        "source_reactions": str(args.source_reactions),
        "validation_pairs_firewall": str(args.validation_pairs),
        "heldout_test_reactions_firewall": str(args.heldout_test_reactions),
        "source_pair_count": len(source_pairs),
        "kept_pair_count": len(kept_pairs),
        "source_reaction_count": len(source_reactions),
        "kept_reaction_count": len(kept_reactions),
        "removed_validation_association_edges": removed_validation_edges,
        "removed_heldout_test_reaction_edges": removed_test_reaction_edges,
        "heldout_test_participant_keys": len(heldout_test_keys),
        "forbidden_validation_associations": len(forbidden_validation_associations),
        "output_pairs_sha256": sha256(output_pairs),
        "output_reactions_sha256": sha256(output_reactions),
    }
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
