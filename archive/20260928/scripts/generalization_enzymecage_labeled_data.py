#!/usr/bin/env python3
"""Preserve every EnzymeCAGE labeled row for a supervised F3 comparison.

F3 identifiers encode protein sequence and exact directional reaction string.
Rows, duplicates, and positive/negative conflicts are retained. This script
never reads external P450 labels or scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def identifier(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode()).hexdigest()[:24]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--positive-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    catalog = json.loads(args.positive_catalog.read_text())
    cached_proteins = set(catalog["proteins"])
    cached_reactions = set(catalog["reactions"])
    args.output.mkdir(parents=True)
    missing_proteins: dict[str, str] = {}
    pair_labels: dict[tuple[str, str], set[int]] = {}
    counts = {}
    for split, source in (("train", args.train), ("validation", args.valid)):
        count = {"rows": 0, "positive": 0, "negative": 0,
                 "missing_protein_rows": 0, "missing_reaction_rows": 0,
                 "unique_pairs": 0, "unique_proteins": 0, "unique_reactions": 0}
        proteins, reactions, pairs = set(), set(), set()
        with source.open(newline="", encoding="utf-8") as src, \
                (args.output / f"{split}_labels.csv").open("w", newline="", encoding="utf-8") as dst:
            reader = csv.DictReader(src)
            if not {"Label", "sequence", "CANO_RXN_SMILES"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing EnzymeCAGE labeled columns: {source}")
            writer = csv.writer(dst)
            writer.writerow(("reaction_id", "protein_id", "label"))
            for row in reader:
                label = int(row["Label"])
                if label not in (0, 1):
                    raise ValueError("Nonbinary EnzymeCAGE label")
                sequence = row["sequence"].strip().upper()
                reaction = row["CANO_RXN_SMILES"].strip()
                if not sequence or reaction.count(">>") != 1:
                    raise ValueError("Missing protein sequence or physical reaction")
                protein = identifier("p_", sequence)
                reaction_id = identifier("r_", reaction)
                writer.writerow((reaction_id, protein, label))
                count["rows"] += 1
                count["positive" if label else "negative"] += 1
                proteins.add(protein)
                reactions.add(reaction_id)
                pairs.add((reaction_id, protein))
                if split == "train":
                    pair_labels.setdefault((reaction_id, protein), set()).add(label)
                if protein not in cached_proteins:
                    count["missing_protein_rows"] += 1
                    existing = missing_proteins.setdefault(protein, sequence)
                    if existing != sequence:
                        raise ValueError("Protein ID hash collision")
                if reaction_id not in cached_reactions:
                    count["missing_reaction_rows"] += 1
        count.update(unique_pairs=len(pairs), unique_proteins=len(proteins),
                     unique_reactions=len(reactions))
        counts[split] = count
    with (args.output / "missing_proteins.fasta").open("w") as handle:
        for protein, sequence in sorted(missing_proteins.items()):
            handle.write(f">{protein}\n{sequence}\n")
    output = {
        "schema": "enzymecage_labeled_f3_data_v1",
        "train": counts["train"], "validation": counts["validation"],
        "missing_unique_proteins": len(missing_proteins),
        "train_conflicting_pair_labels": sum(len(labels) == 2 for labels in pair_labels.values()),
        "cached_positive_catalog_sha256": sha256(args.positive_catalog),
        "sources": {"train": sha256(args.train), "validation": sha256(args.valid)},
        "outputs": {name: sha256(args.output / name) for name in
                    ("train_labels.csv", "validation_labels.csv", "missing_proteins.fasta")},
        "p450_labels_or_scores_read": False,
        "source_sha256": sha256(Path(__file__)),
    }
    (args.output / "manifest.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
