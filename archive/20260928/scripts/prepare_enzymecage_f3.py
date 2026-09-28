#!/usr/bin/env python3
"""Convert EnzymeCAGE's labelled RHEA rows to CIRCE F3 positive-pair files.

Only Label=1 is an observed association. Label=0 is never treated as a
verified inactive pair. Original train/valid membership is retained, with
exact positive-pair overlap removed from validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def identifier(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode()).hexdigest()[:24]


def convert(source: Path, split: str, output: Path, proteins: dict[str, str],
            reactions: dict[str, str], train_pairs: set[tuple[str, str]]) -> dict:
    pairs_path = output / f"{split}_pairs.csv"
    seen: set[tuple[str, str]] = set()
    counts = {"rows": 0, "positive_rows": 0, "duplicate_pairs": 0,
              "train_overlap_pairs": 0, "missing_inputs": 0, "positive_pairs": 0}
    with source.open(newline="", encoding="utf-8") as handle, pairs_path.open("w", newline="") as target:
        reader = csv.DictReader(handle)
        required = {"Label", "sequence", "CANO_RXN_SMILES"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{source} lacks {required - set(reader.fieldnames or [])}")
        writer = csv.writer(target)
        writer.writerow(("pr_id", "reaction_id", "protein_id"))
        for row in reader:
            counts["rows"] += 1
            if row["Label"] != "1":
                continue
            counts["positive_rows"] += 1
            sequence = row["sequence"].strip().upper()
            reaction = row["CANO_RXN_SMILES"].strip()
            if not sequence or not reaction or ">>" not in reaction:
                counts["missing_inputs"] += 1
                continue
            protein_id = identifier("p_", sequence)
            reaction_id = identifier("r_", reaction)
            pair = (reaction_id, protein_id)
            if pair in seen:
                counts["duplicate_pairs"] += 1
                continue
            seen.add(pair)
            if split == "validation" and pair in train_pairs:
                counts["train_overlap_pairs"] += 1
                continue
            if protein_id in proteins and proteins[protein_id] != sequence:
                raise ValueError("Protein hash collision")
            if reaction_id in reactions and reactions[reaction_id] != reaction:
                raise ValueError("Reaction hash collision")
            proteins[protein_id] = sequence
            reactions[reaction_id] = reaction
            writer.writerow((f"{split}_{counts['positive_pairs']}", reaction_id, protein_id))
            counts["positive_pairs"] += 1
            if split == "train":
                train_pairs.add(pair)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    proteins: dict[str, str] = {}
    reactions: dict[str, str] = {}
    train_pairs: set[tuple[str, str]] = set()
    stats = {
        "train": convert(args.train, "train", args.output, proteins, reactions, train_pairs),
        "validation": convert(args.valid, "validation", args.output, proteins, reactions, train_pairs),
    }
    if not stats["train"]["positive_pairs"] or not stats["validation"]["positive_pairs"]:
        raise ValueError("Train and validation both need positive pairs")
    for split in ("train", "validation"):
        ids = {row["reaction_id"] for row in csv.DictReader((args.output / f"{split}_pairs.csv").open())}
        with (args.output / f"{split}_rxns.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("reaction_id", "reaction_smiles"))
            writer.writerows((key, reactions[key]) for key in sorted(ids))
    with (args.output / "proteins.fasta").open("w") as handle:
        for key, sequence in sorted(proteins.items()):
            handle.write(f">{key}\n{sequence}\n")
    with (args.output / "validation_pairs.csv").open(newline="") as handle:
        validation_candidates = {row["protein_id"] for row in csv.DictReader(handle)}
    with (args.output / "validation_candidate_ids.txt").open("w") as handle:
        handle.writelines(f"{key}\n" for key in sorted(validation_candidates))
    stats.update(proteins=len(proteins), reactions=len(reactions),
                 validation_candidate_proteins=len(validation_candidates),
                 train_source=str(args.train.resolve()), valid_source=str(args.valid.resolve()),
                 note="Label=0 ignored; original EnzymeCAGE split retained; exact pair overlap removed")
    (args.output / "manifest.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
