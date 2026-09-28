#!/usr/bin/env python3
"""Build a strict ReactZyme-firewalled source graph for residual pretraining."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-pairs",
        type=Path,
        default=ROOT / "runs/source_collapse_f3mc_ec_lambda3_95_5_v1/data/split/train_pairs.csv",
    )
    parser.add_argument(
        "--source-reactions",
        type=Path,
        default=ROOT / "runs/source_collapse_f3mc_ec_lambda3_95_5_v1/data/split/train_rxns.csv",
    )
    parser.add_argument(
        "--heldout-pairs",
        action="append",
        type=Path,
        default=None,
        help="Validation/test pair CSV; repeat for multiple files",
    )
    parser.add_argument(
        "--heldout-reactions",
        action="append",
        type=Path,
        default=None,
        help="Validation/test reaction CSV; repeat for multiple files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs/biological_residual_reaction_smi/data/source_pretrain",
    )
    return parser.parse_args()


def canonical_molecule(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None:
        return str(smiles).strip()
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def participant_multiset(reaction_smiles: str) -> tuple[str, ...]:
    text = str(reaction_smiles).strip()
    if ">>" not in text:
        return tuple(
            sorted(canonical_molecule(value) for value in text.split(".") if value.strip())
        )
    left, right = text.split(">>", maxsplit=1)
    left_values = [canonical_molecule(value) for value in left.split(".") if value.strip()]
    right_values = [canonical_molecule(value) for value in right.split(".") if value.strip()]
    values = left_values if Counter(left_values) == Counter(right_values) else left_values + right_values
    return tuple(sorted(values))


def canonical_protein_id(value: str) -> str:
    text = str(value).strip()
    for prefix in ("prot_", "uprot_", "nr90_"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


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


def build_strict_graph(
    source_pairs: list[dict[str, str]],
    source_reactions: list[dict[str, str]],
    heldout_pairs: list[dict[str, str]],
    heldout_reactions: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, int]]:
    source_keys = {
        row["reaction_id"]: participant_multiset(row["reaction_smiles"])
        for row in source_reactions
    }
    # Pair files repeat the same (occasionally very large) reaction many times.
    # Canonicalize each distinct string once; this is a material speedup and
    # leaves the conservative chemistry firewall unchanged.
    heldout_smiles = {
        str(row.get("reaction_smiles", "")).strip()
        for row in heldout_reactions + heldout_pairs
        if str(row.get("reaction_smiles", "")).strip()
    }
    heldout_reaction_keys = {
        participant_multiset(reaction_smiles) for reaction_smiles in heldout_smiles
    }
    heldout_proteins = {
        canonical_protein_id(row["protein_id"])
        for row in heldout_pairs
        if str(row.get("protein_id", "")).strip()
    }
    heldout_key_by_smiles = {
        reaction_smiles: participant_multiset(reaction_smiles)
        for reaction_smiles in heldout_smiles
    }
    forbidden_associations = {
        (
            canonical_protein_id(row["protein_id"]),
            heldout_key_by_smiles[str(row["reaction_smiles"]).strip()],
        )
        for row in heldout_pairs
        if str(row.get("protein_id", "")).strip()
        and str(row.get("reaction_smiles", "")).strip()
    }
    kept: list[dict[str, str]] = []
    counters = Counter()
    seen_edges: set[tuple[str, str]] = set()
    for row in source_pairs:
        reaction_id = str(row["reaction_id"])
        protein_id = str(row["protein_id"])
        key = source_keys.get(reaction_id)
        if key is None:
            raise ValueError(f"Source pair references missing reaction {reaction_id!r}")
        canonical_protein = canonical_protein_id(protein_id)
        if key in heldout_reaction_keys:
            counters["heldout_reaction"] += 1
            continue
        if canonical_protein in heldout_proteins:
            counters["heldout_protein"] += 1
            continue
        if (canonical_protein, key) in forbidden_associations:
            counters["heldout_association"] += 1
            continue
        edge = (reaction_id, protein_id)
        if edge in seen_edges:
            counters["duplicate_edge"] += 1
            continue
        seen_edges.add(edge)
        kept.append(dict(row))
    kept_reaction_ids = {row["reaction_id"] for row in kept}
    kept_reactions = [
        dict(row) for row in source_reactions if row["reaction_id"] in kept_reaction_ids
    ]
    # Postcondition audit: no forbidden entity or chemistry may survive.
    if any(
        source_keys[row["reaction_id"]] in heldout_reaction_keys
        or canonical_protein_id(row["protein_id"]) in heldout_proteins
        for row in kept
    ):
        raise RuntimeError("Strict leakage firewall postcondition failed")
    counters["heldout_reaction_keys"] = len(heldout_reaction_keys)
    counters["heldout_proteins"] = len(heldout_proteins)
    counters["forbidden_associations"] = len(forbidden_associations)
    return kept, kept_reactions, dict(counters)


def main() -> None:
    args = parse_args()
    heldout_pair_paths = args.heldout_pairs or [
        ROOT / "data/revised_protocols/reactzyme_paper/reaction_smi/validation_pairs.csv",
        ROOT / "data/revised_protocols/reactzyme_official/reaction_smi/test_pairs.csv",
    ]
    heldout_reaction_paths = args.heldout_reactions or [
        ROOT / "data/revised_protocols/reactzyme_paper/reaction_smi/validation_rxns.csv",
        ROOT / "data/revised_protocols/reactzyme_official/reaction_smi/test_rxns.csv",
    ]
    pair_fields, source_pairs = read_csv(args.source_pairs)
    reaction_fields, source_reactions = read_csv(args.source_reactions)
    if not {"reaction_id", "protein_id"}.issubset(pair_fields):
        raise ValueError("Source pairs require reaction_id and protein_id")
    if not {"reaction_id", "reaction_smiles"}.issubset(reaction_fields):
        raise ValueError("Source reactions require reaction_id and reaction_smiles")
    heldout_pairs = [row for path in heldout_pair_paths for row in read_csv(path)[1]]
    heldout_reactions = [row for path in heldout_reaction_paths for row in read_csv(path)[1]]
    kept_pairs, kept_reactions, counters = build_strict_graph(
        source_pairs, source_reactions, heldout_pairs, heldout_reactions
    )
    for index, row in enumerate(kept_pairs):
        if "pr_id" in pair_fields:
            row["pr_id"] = str(index)
    output_pairs = args.output_dir / "train_pairs.csv"
    output_reactions = args.output_dir / "train_rxns.csv"
    write_csv(output_pairs, pair_fields, kept_pairs)
    write_csv(output_reactions, reaction_fields, kept_reactions)
    manifest = {
        "schema_version": "biological_residual_strict_source_graph_v1",
        "source_pairs": str(args.source_pairs.resolve()),
        "source_reactions": str(args.source_reactions.resolve()),
        "heldout_pair_files": [str(path.resolve()) for path in heldout_pair_paths],
        "heldout_reaction_files": [str(path.resolve()) for path in heldout_reaction_paths],
        "source_pair_count": len(source_pairs),
        "kept_pair_count": len(kept_pairs),
        "source_reaction_count": len(source_reactions),
        "kept_reaction_count": len(kept_reactions),
        **counters,
        "output_pairs_sha256": sha256(output_pairs),
        "output_reactions_sha256": sha256(output_reactions),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
