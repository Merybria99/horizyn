#!/usr/bin/env python3
"""Reuse the reconstructed 80%-identity graph; hold out paper-style reaction groups.

No downloading, protein clustering/alignment, native annotation extraction or
reaction annotation inference is performed here. The 90/5/5 reaction holdout is
an explicit adaptation, not the paper's full-corpus inference training protocol.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.horizyn1_circe_v2_pipeline import atomic_json, csv_rows, fasta_records, signature, write_csv

PROTOCOL = "horizyn80_clustered_reaction_holdout_v1"
DIRECTIONS = ("forward", "reverse")


def progress(message):
    print(message, flush=True)


def reaction_id(original, direction):
    # Avoid the dataloader's reserved _f/_r aliases. These are already augmented
    # rows; its forward_only setting means 'do not augment them a second time'.
    return f"{original}__horizyn_{direction}"


@lru_cache(maxsize=1)
def chemistry_tools():
    from rdkit.Chem import rdFingerprintGenerator
    from horizyn.chemistry.standardizer import Standardizer

    standardizer = Standardizer(
        standardize_hypervalent=True, standardize_remove_hs=False,
        standardize_kekulize=False, standardize_uncharge=True, standardize_metals=True,
    )
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=3, fpSize=2048, includeChirality=True,
        useBondTypes=True, includeRingMembership=True,
    )
    return standardizer, generator


@lru_cache(maxsize=100000)
def molecule(smiles):
    from rdkit import Chem, DataStructs

    standardizer, generator = chemistry_tools()
    standardized = standardizer.standardize_molecule(smiles)
    mol = Chem.MolFromSmiles(standardized)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError(f"Invalid reaction participant: {smiles!r}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    fingerprint = DataStructs.ExplicitBitVect(2048)
    # Metal disconnection can create multiple participants. OR their molecular
    # fingerprints just like the paper, rather than inventing reaction changes.
    for fragment in Chem.GetMolFrags(mol, asMols=True):
        fingerprint |= generator.GetFingerprint(fragment)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), fingerprint


def describe_reaction(smiles):
    from rdkit import DataStructs

    parts = smiles.split(">")
    if len(parts) != 3 or not parts[0] or not parts[2] or parts[1]:
        raise ValueError("Expected nonempty reactants>>products without an unmodelled agents field")
    sides, fingerprints = [], []
    for side in (parts[0], parts[2]):
        combined = DataStructs.ExplicitBitVect(2048)
        standardized = []
        for component in side.split("."):
            normalized, fingerprint = molecule(component)
            standardized.extend(normalized.split("."))
            combined |= fingerprint
        sides.append(".".join(sorted(standardized)))
        fingerprints.append(DataStructs.BitVectToBinaryText(combined))
    # For binary fingerprints Tanimoto=1 means exact equality, so hashing
    # sorted side bitstrings is equivalent to all four directional comparisons.
    key = b"".join(sorted(fingerprints))
    return ">>".join(sides), key


def split_reactions(reactions, seed, validation_fraction, test_fraction):
    groups = {}
    records = {}
    for i, row in enumerate(reactions, 1):
        rid = row["reaction_id"]
        if not rid or rid in records:
            raise ValueError(f"Empty or duplicate reaction ID: {rid!r}")
        try:
            normalized, key = describe_reaction(row["reaction_smiles"])
        except Exception as exc:
            raise ValueError(f"Cannot apply paper chemistry protocol to {rid}: {exc}") from exc
        groups.setdefault(key, []).append(rid)
        records[rid] = {"reaction_smiles": normalized, "group": hashlib.sha256(key).hexdigest()}
        if i % 1000 == 0:
            progress(f"Paper ECFP6 equality grouping: {i:,} reactions")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    nval, ntest = max(1, round(len(keys) * validation_fraction)), max(1, round(len(keys) * test_fraction))
    if nval + ntest >= len(keys):
        raise ValueError("Not enough independent reaction groups for train/validation/test")
    for i, key in enumerate(keys):
        split = "validation" if i < nval else "test" if i < nval + ntest else "train"
        for rid in groups[key]:
            records[rid]["split"] = split
    return records, len(groups)


def write_ids(path, ids):
    with path.open("w") as out:
        for value in ids:
            out.write(value + "\n")


def paired_reaction_inventory(reactions, pairs):
    """Select supervised nodes before grouping, without dropping any graph edge.

    The chemistry inventory can contain reactions with no observed associations.
    Keep an explicit record of these, but never make them zero-gold queries or
    let their fingerprint groups affect the supervised split. The later raw-pair
    export verifies that an excluded reaction has no raw associations either.
    """
    inventory = {}
    for row in reactions:
        rid = row["reaction_id"]
        if not rid or rid in inventory:
            raise ValueError(f"Empty or duplicate reaction ID: {rid!r}")
        inventory[rid] = row
    paired = set()
    count = 0
    for count, row in enumerate(pairs, 1):
        rid = row["reaction_id"]
        if rid not in inventory:
            raise ValueError(f"Unknown clustered reaction ID: {rid}")
        paired.add(rid)
        if count % 1000000 == 0:
            progress(f"Checking paired reaction inventory: {count:,} clustered pairs")
    selected = [row for rid, row in inventory.items() if rid in paired]
    excluded = [{**inventory[rid], "reason": "no_observed_enzyme_reaction_pairs"}
                for rid in sorted(inventory.keys() - paired)]
    return selected, excluded, count


def prepare(source, output, *, seed=42, validation_fraction=.05, test_fraction=.05,
            panel_reactions=128, panel_proteins=100000):
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1
            and validation_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than one")
    if min(panel_reactions, panel_proteins) <= 0:
        raise ValueError("Panel sizes must be positive")
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Run output and source dataset must be separate, non-nested directories")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing nonempty preparation output; preserve it and use a new directory")
    inputs = [source / name for name in ("clustered/clustered_manifest.json", "clustered/proteins.fasta",
              "clustered/pairs.tsv", "raw/raw_reactions.tsv", "raw/raw_pairs.tsv")]
    before = [signature(path) for path in inputs]
    manifest = json.loads(inputs[0].read_text())
    if manifest.get("min_seq_id") != .8:
        raise ValueError("Expected an existing 80%-identity source dataset")
    output.mkdir(parents=True, exist_ok=True)
    (output / "panels").mkdir()
    selected_reactions, excluded_reactions, source_pairs = paired_reaction_inventory(
        csv_rows(inputs[3]), csv_rows(inputs[2]))
    if source_pairs != manifest["clustered_pairs"]:
        raise ValueError("Clustered pair count differs from manifest")
    excluded_ids = {row["reaction_id"] for row in excluded_reactions}
    progress(f"Reaction inventory: {len(selected_reactions):,} paired reactions; "
             f"{len(excluded_reactions):,} unpaired entries excluded from supervised splits "
             "(original inventory unchanged; raw-pair consistency checked before completion)")
    records, group_count = split_reactions(selected_reactions, seed, validation_fraction, test_fraction)
    splits = ("train", "validation", "test")
    ids = {split: sorted(r for r, value in records.items() if value["split"] == split) for split in splits}
    selected = sorted(ids["validation"], key=lambda r: hashlib.sha256(f"{seed}:{r}".encode()).digest())[:panel_reactions]
    panel_queries = {reaction_id(r, d) for r in selected for d in DIRECTIONS}
    reaction_rows = []
    for original, record in sorted(records.items()):
        left, right = record["reaction_smiles"].split(">>")
        for direction, smiles in zip(DIRECTIONS, (f"{left}>>{right}", f"{right}>>{left}")):
            reaction_rows.append({**record, "reaction_id": reaction_id(original, direction),
                                  "reaction_smiles": smiles, "original_reaction_id": original,
                                  "direction": direction})
    write_csv(output / "reactions.csv", ["reaction_id", "reaction_smiles"], reaction_rows)
    write_csv(output / "reaction_groups.csv", ["reaction_id", "original_reaction_id", "direction", "group", "split"], reaction_rows)
    for split in splits:
        rows = [row for row in reaction_rows if row["split"] == split]
        write_csv(output / f"{split}_rxns.csv", ["reaction_id", "reaction_smiles"], rows)
        write_ids(output / f"{split}_reaction_ids.txt", [row["reaction_id"] for row in rows])
    write_ids(output / "all_reaction_ids.txt", [row["reaction_id"] for row in reaction_rows])
    atomic_json(output / "reaction_split.json", {r: row["split"] for r, row in records.items()})

    progress("Reusing existing protein representatives (no MMseqs operation)")
    proteins, reservoir = set(), []
    rng = random.Random(seed)
    with (output / "all_candidate_ids.txt").open("w") as out:
        for i, (pid, _) in enumerate(fasta_records(inputs[1]), 1):
            if pid in proteins:
                raise ValueError(f"Duplicate protein ID: {pid}")
            proteins.add(pid)
            out.write(pid + "\n")
            if len(reservoir) < panel_proteins:
                reservoir.append(pid)
            else:
                position = rng.randrange(i)
                if position < panel_proteins:
                    reservoir[position] = pid
    if len(proteins) != manifest["clustered_proteins"]:
        raise ValueError("Representative inventory does not match clustered manifest")

    counts, seen_reactions, panel_positive_proteins, test_proteins = Counter(), set(), set(), set()
    with ExitStack() as stack:
        writers = {}
        for split in (*splits, "panel"):
            path = output / (f"{split}_pairs.csv" if split != "panel" else "panels/validation_reaction_cold_pairs.csv")
            writers[split] = csv.writer(stack.enter_context(path.open("w", newline="")))
            writers[split].writerow(["pr_id", "reaction_id", "protein_id"])
        previous = None
        for number, row in enumerate(csv_rows(inputs[2]), 1):
            original, pid = row["reaction_id"], row["protein_id"]
            pair = original, pid
            if previous is not None and pair <= previous:
                raise ValueError("Clustered pairs must be unique and sorted by reaction/protein")
            previous = pair
            if original not in records or pid not in proteins:
                raise ValueError(f"Unknown clustered graph endpoint: {pair}")
            seen_reactions.add(original)
            split = records[original]["split"]
            counts[split] += 1
            if split == "test":
                test_proteins.add(pid)
            for direction in DIRECTIONS:
                rid = reaction_id(original, direction)
                entry = [f"{number}_{direction}", rid, pid]
                writers[split].writerow(entry)
                if rid in panel_queries:
                    writers["panel"].writerow(entry)
                    panel_positive_proteins.add(pid)
            if number % 1000000 == 0:
                progress(f"Reusing clustered positives: {number:,} pairs")
    if sum(counts.values()) != manifest["clustered_pairs"] or any(not counts[s] for s in splits):
        raise ValueError("Clustered pair counts differ from manifest or a split has no positives")
    if seen_reactions != set(records):
        raise ValueError("Selected reaction inventory does not match clustered pairs")
    shutil.copyfile(output / "test_pairs.csv", output / "test_query_gold.csv")
    write_ids(output / "test_reaction_query_ids.txt", [reaction_id(r, d) for r in ids["test"] for d in DIRECTIONS])
    write_ids(output / "test_enzyme_query_ids.txt", sorted(test_proteins))
    write_ids(output / "panels/validation_reaction_cold_protein_ids.txt", sorted(set(reservoir) | panel_positive_proteins))
    enzyme_anchors = sorted(panel_positive_proteins)[:panel_reactions]
    enzyme_anchor_set = set(enzyme_anchors)
    # Include complete validation-reaction gold for each enzyme anchor too;
    # otherwise true reactions outside the selected R2E anchors look negative.
    write_csv(output / "panels/validation_reaction_cold_pairs.csv", ["pr_id", "reaction_id", "protein_id"],
              (row for row in csv_rows(output / "validation_pairs.csv")
               if row["reaction_id"] in panel_queries or row["protein_id"] in enzyme_anchor_set))
    atomic_json(output / "panels/validation_reaction_cold_query_ids.json", {
        "reaction_to_enzyme": sorted(panel_queries),
        "enzyme_to_reaction": enzyme_anchors,
    })
    # All transferred graph edges are positives now, not uncertainty exclusions.
    write_csv(output / "train_transferred_uncertain_pairs.csv", ["reaction_id", "protein_id"], [])
    own = 0
    with (output / "train_own_raw_associations.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["reaction_id", "protein_id"])
        for number, row in enumerate(csv_rows(inputs[4]), 1):
            rid, pid = row["reaction_id"], row["protein_id"]
            if rid in excluded_ids:
                raise ValueError(f"Excluded reaction {rid} has raw pairs but no clustered pairs; "
                                 "inspect the collapsed graph rather than drop observed associations")
            if rid not in records:
                raise ValueError(f"Unknown raw reaction: {rid}")
            if records[rid]["split"] == "train" and pid in proteins:
                writer.writerow([rid, pid])
                own += 1
            if number % 1000000 == 0:
                progress(f"Selecting training-only cached-label associations: {number:,} raw rows")
    if before != [signature(path) for path in inputs]:
        raise ValueError("Source changed during export; no completed preparation manifest")
    write_csv(output / "excluded_unpaired_reactions.csv",
              ["reaction_id", "reaction_smiles", "reason"], excluded_reactions)
    result = {"schema_version": PROTOCOL, "source_inputs": before, "seed": seed,
              "protein_clustering": "reuse_existing_80_percent_representatives",
              "protein_split": "none; overlap between reaction splits is allowed",
              "reaction_fingerprint": "2048-bit ECFP6 per side, OR composition, concatenated 4096 bits",
              "reaction_similarity_threshold": 1.0, "reverse_invariant_groups": True,
              "direction_augmentation": "forward_and_reverse_in_same_split",
              "validation_fraction": validation_fraction, "test_fraction": test_fraction,
              "source_collapse": "all existing clustered pairs are retrieval positives",
              "reaction_inventory": {
                  "source_reactions": len(selected_reactions) + len(excluded_reactions),
                  "paired_reactions": len(selected_reactions),
                  "excluded_unpaired_reactions": len(excluded_reactions),
                  "excluded_ids": sorted(excluded_ids),
                  "excluded_table": "excluded_unpaired_reactions.csv",
                  "policy": "exclude unpaired entries before grouping; preserve source inventory and all pairs",
                  "excluded_raw_pair_count": 0,
              },
              "reaction_groups": group_count, "source_pairs": sum(counts.values()),
              "pairs_before_augmentation": dict(counts), "pairs_after_augmentation": {s: 2 * counts[s] for s in splits},
              "training_own_annotation_associations": own,
              "evaluation": "R2E: heldout reactions versus all proteins; E2R: enzymes versus heldout reactions only",
              "limitation": "Reconstructed corpus differs from authors' counts; reaction holdout is a user-requested adaptation of full-corpus inference training."}
    from rdkit import rdBase
    result["rdkit_version"] = rdBase.rdkitVersion
    result["standardization"] = ["SANITIZE_CLEANUP", "Uncharger", "MetalDisconnector"]
    atomic_json(output / "preparation_manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=.05)
    parser.add_argument("--test-fraction", type=float, default=.05)
    args = parser.parse_args(argv)
    progress(json.dumps(prepare(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
