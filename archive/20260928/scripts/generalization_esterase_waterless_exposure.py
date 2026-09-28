#!/usr/bin/env python3
"""Input-only esterase organic-participant overlap sensitivity.

All query reactions contain water and no actual Reaction-Sim training reaction
does. This audit removes only neutral water for comparison and never filters
the assay panel or changes model inputs.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/generalization_20260919_2251"
PANEL = RUN / "esterase_audit"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def participants(value: str) -> tuple[str, ...]:
    sides = value.split(">")
    if len(sides) == 1:
        sides = [sides[0], "", sides[0]]
    if len(sides) != 3 or sides[1]:
        raise ValueError("Expected a participant set or three-part reaction with empty agent field")
    left, right = [], []
    for side, result in ((sides[0], left), (sides[2], right)):
        mol = Chem.MolFromSmiles(side)
        if mol is None:
            raise ValueError("Unparseable reaction")
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        result.extend(Chem.MolToSmiles(fragment, isomericSmiles=True, canonical=True)
                      for fragment in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    return tuple(sorted(left if left == right else left + right))


def without_water(value: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(x for x in value if x != "O")


def contained(query: tuple[str, ...], train: tuple[str, ...]) -> bool:
    a, b = Counter(query), Counter(train)
    return all(b[key] >= count for key, count in a.items())


def main() -> None:
    catalog_path = RUN / "features/catalog.json"
    edges_path = RUN / "features/pairs.npz"
    manifest_path = RUN / "features/manifest.json"
    catalog = json.loads(catalog_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    with np.load(edges_path) as archive:
        train_ids = {catalog["reactions"][int(i)] for i in archive["train"][:, 0]}
    if len(train_ids) != 6977:
        raise ValueError("Unexpected actual training reaction count")
    source = manifest["inputs"]["train_reactions_path"]
    source_path = Path(source["path"])
    if sha(source_path) != source["sha256"]:
        raise ValueError("Training reaction source changed")
    with source_path.open(newline="") as stream:
        all_train = {row["reaction_id"]: row["reaction_smiles"] for row in csv.DictReader(stream)}
    with (PANEL / "inputs/reactions.csv").open(newline="") as stream:
        queries = list(csv.DictReader(stream))
    if len(queries) != 86:
        raise ValueError("Unexpected chemistry-qualified substrate count")
    train = {rid: without_water(participants(all_train[rid])) for rid in sorted(train_ids)}
    exact_index: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for rid, key in train.items():
        exact_index[key].append(rid)
    rows = []
    for row in queries:
        full = participants(row["reaction_smiles"])
        if full.count("O") != 1:
            raise ValueError("Expected exactly one explicit neutral water component per query")
        organic = without_water(full)
        exact = exact_index.get(organic, [])
        supersets = [rid for rid, key in train.items() if contained(organic, key)]
        rows.append({"reaction_id": row["reaction_id"], "organic_participants": list(organic),
                     "waterless_exact_train_reaction_ids": exact,
                     "organic_containing_train_reaction_ids": supersets})
    output = {"schema": "esterase_organic_input_exposure_v1", "scores_read": False,
              "assay_values_read": False, "candidate_filtering": False,
              "method": "Canonical isomeric component multisets; remove only neutral water O from both query and training; exact multiset and multiplicity-respecting organic containment sensitivities. No tautomer/charge/stereo normalization.",
              "counts": {"queries": len(rows), "train_reactions": len(train),
                         "waterless_exact": sum(bool(x["waterless_exact_train_reaction_ids"]) for x in rows),
                         "organic_contained": sum(bool(x["organic_containing_train_reaction_ids"]) for x in rows)},
              "inputs": {str(p.relative_to(ROOT)): sha(p) for p in
                         (catalog_path, edges_path, manifest_path, source_path, PANEL / "inputs/reactions.csv")},
              "source_sha256": sha(Path(__file__)), "rows": rows}
    path = PANEL / "training_exposure/waterless_sensitivity.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["counts"]))


if __name__ == "__main__":
    main()
