#!/usr/bin/env python3
"""Prepare CLIPZyme rule-split F3 catalogs without changing the model architecture."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def identifier(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode()).hexdigest()[:24]


def load_rescue(fasta: Path, receipt_path: Path) -> tuple[dict[str, str], dict]:
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != "clipzyme_unisave_sequence_rescue_v1"
            or receipt.get("unresolved_ids") or receipt.get("missing_ids") != 72
            or receipt.get("rescued_ids") != 72
            or receipt.get("fasta_sha256") != sha256(fasta)):
        raise ValueError("UniSave rescue is incomplete or changed")
    sequences = {}
    key = None
    with fasta.open() as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                key = line[1:]
                if not key or key in sequences:
                    raise ValueError("Duplicate or blank rescue identifier")
                sequences[key] = ""
            elif key:
                sequences[key] += line
            else:
                raise ValueError("Rescue FASTA starts without an identifier")
    if len(sequences) != 72 or any(not value or not value.isalpha()
                                    or value != value.upper() for value in sequences.values()):
        raise ValueError("Incomplete UniSave rescue FASTA")
    if set(sequences) != set(receipt["entries"]):
        raise ValueError("UniSave rescue IDs and receipt disagree")
    return sequences, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--architecture-lock", type=Path, required=True)
    parser.add_argument("--rescue-fasta", type=Path)
    parser.add_argument("--rescue-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    lock = json.loads(args.architecture_lock.read_text())
    if (manifest.get("schema") != "fixed_architecture_clipzyme_manifest_v1"
            or lock.get("schema") != "cross_paper_fixed_architecture_preparation_v3"
            or not lock.get("same_model_and_base_training_in_local_enzymecage_config")):
        raise ValueError("Require source-order release manifest and fixed F3 lock")
    if bool(args.rescue_fasta) != bool(args.rescue_receipt):
        raise ValueError("Provide both UniSave rescue FASTA and receipt")
    rescue, rescue_receipt = (
        load_rescue(args.rescue_fasta, args.rescue_receipt)
        if args.rescue_fasta else ({}, None)
    )
    # The official cached association list contains IDs whose released
    # sequence value is empty. Refuse before writing a partial training catalog:
    # dropping those rows would change the paper split, and an invented sequence
    # would change the F3 input contract.
    missing = {}
    missing_ids = set()
    for split in ("train", "dev", "test"):
        source_record = manifest["associations"][split]
        source = Path(source_record["path"])
        if sha256(source) != source_record["sha256"]:
            raise ValueError(f"Source associations changed: {split}")
        with source.open(newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if not row["sequence"]]
            missing[split] = len(rows)
            missing_ids.update(row["protein_id"] for row in rows)
    if missing_ids - set(rescue):
        raise ValueError(
            f"Official associations have empty sequences {missing}; full paper-matched "
            "F3 catalogs require a source-verified sequence or declared coverage policy"
        )
    target = args.output
    target.mkdir(parents=True)
    proteins: dict[str, str] = {}
    all_reactions: dict[str, str] = {}
    pairs_per_split: dict[str, set[tuple[str, str]]] = {}
    counts = {}
    for source_split, target_split in (("train", "train"), ("dev", "validation"),
                                       ("test", "test")):
        source_record = manifest["associations"][source_split]
        source = Path(source_record["path"])
        if sha256(source) != source_record["sha256"]:
            raise ValueError(f"Source associations changed: {source_split}")
        pair_path = target / f"{target_split}_pairs.csv"
        reaction_path = target / f"{target_split}_rxns.csv"
        exact_pairs = set()
        sequence_pairs = set()
        reaction_ids = set()
        row_count = 0
        with source.open(newline="") as reader_file, pair_path.open("w", newline="") as pair_file:
            reader = csv.DictReader(reader_file)
            writer = csv.writer(pair_file)
            writer.writerow(("pr_id", "reaction_id", "protein_id"))
            for row in reader:
                row_count += 1
                reaction = row["reaction"]
                sequence = row["sequence"] or rescue.get(row["protein_id"], "")
                if not reaction or ">>" not in reaction or not sequence:
                    raise ValueError(f"Missing target training/evaluation input: {target_split}")
                protein_id = identifier("p_", sequence)
                reaction_id = identifier("r_", reaction)
                if protein_id in proteins and proteins[protein_id] != sequence:
                    raise ValueError("Protein hash collision")
                if reaction_id in all_reactions and all_reactions[reaction_id] != reaction:
                    raise ValueError("Reaction hash collision")
                proteins[protein_id] = sequence
                all_reactions[reaction_id] = reaction
                reaction_ids.add(reaction_id)
                exact_pairs.add((row["protein_id"], reaction))
                sequence_pairs.add((protein_id, reaction_id))
                writer.writerow((f"{target_split}_{row['source_index']}", reaction_id, protein_id))
        if row_count != source_record["count"] or len(exact_pairs) != row_count:
            raise ValueError("Association count/uniqueness mismatch")
        with reaction_path.open("w", newline="") as reaction_file:
            writer = csv.writer(reaction_file)
            writer.writerow(("reaction_id", "reaction_smiles"))
            writer.writerows((reaction_id, all_reactions[reaction_id])
                             for reaction_id in sorted(reaction_ids))
        pairs_per_split[target_split] = sequence_pairs
        counts[target_split] = {"association_rows": row_count,
                                "sequence_level_pairs": len(sequence_pairs),
                                "reactions": len(reaction_ids),
                                "pairs_sha256": sha256(pair_path),
                                "reactions_sha256": sha256(reaction_path)}
    all_fasta = target / "association_proteins.fasta"
    with all_fasta.open("w") as handle:
        for protein_id, sequence in sorted(proteins.items()):
            handle.write(f">{protein_id}\n{sequence}\n")
    # Foundation features may be precomputed for all candidates without using
    # test associations. Keep released candidate IDs/order in a separate map.
    candidate_record = manifest["candidate_pool"]
    candidate_source = Path(candidate_record["path"])
    if sha256(candidate_source) != candidate_record["sha256"]:
        raise ValueError("Screening manifest changed")
    candidate_path = target / "screening_candidate_map.csv"
    candidate_proteins = {}
    missing_candidates = []
    with candidate_source.open(newline="") as source, candidate_path.open("w", newline="") as output:
        reader, writer = csv.DictReader(source), csv.writer(output)
        writer.writerow(("candidate_index", "uniprot_id", "protein_id"))
        for row in reader:
            sequence = row["sequence"] or rescue.get(row["protein_id"], "")
            if not sequence:
                protein_id = ""
                missing_candidates.append(row["protein_id"])
            else:
                protein_id = identifier("p_", sequence)
                candidate_proteins[protein_id] = sequence
                if protein_id in proteins and proteins[protein_id] != sequence:
                    raise ValueError("Screening protein hash collision")
            writer.writerow((row["candidate_index"], row["protein_id"], protein_id))
    if missing_candidates:
        raise ValueError(f"Unresolved screening sequences: {len(missing_candidates)}")
    fasta = target / "screening_proteins.fasta"
    with fasta.open("w") as handle:
        for protein_id, sequence in sorted(candidate_proteins.items()):
            handle.write(f">{protein_id}\n{sequence}\n")
    validation_candidates = target / "validation_candidate_ids.txt"
    with validation_candidates.open("w") as handle:
        for protein_id in sorted({protein for protein, _ in pairs_per_split["validation"]}):
            handle.write(protein_id + "\n")
    cross_split = {}
    for other in ("validation", "test"):
        cross_split[f"train_{other}_sequence_level_pair_overlap"] = len(
            pairs_per_split["train"] & pairs_per_split[other]
        )
    report = {
        "schema": "clipzyme_fixed_f3_catalog_preparation_v1",
        "architecture_lock_sha256": sha256(args.architecture_lock),
        "f3_model_sha256": lock["f3_model_sha256"],
        "source_manifest_sha256": sha256(args.manifest),
        "split_files": counts,
        "cross_split": cross_split,
        "candidate_map_sha256": sha256(candidate_path),
        "candidate_ids": candidate_record["count"],
        "unique_screening_sequences": len(candidate_proteins),
        "screening_fasta_sha256": sha256(fasta),
        "association_fasta_sha256": sha256(all_fasta),
        "source_empty_sequence_rows": missing,
        "sequence_rescue": None if rescue_receipt is None else {
            "receipt_sha256": sha256(args.rescue_receipt),
            "fasta_sha256": sha256(args.rescue_fasta),
            "exact_release": rescue_receipt["exact_release_rescued"],
            "bracketed_identical": rescue_receipt["bracketed_identical_rescued"]},
        "validation_candidate_ids_sha256": sha256(validation_candidates),
        "candidate_ids_without_sequence": missing_candidates,
        "feature_extraction_complete": False,
        "training_started": False,
        "test_pairs_for_evaluation_only": str((target / "test_pairs.csv").resolve()),
        "source_code_sha256": sha256(Path(__file__)),
    }
    (target / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"splits": {k: v["association_rows"] for k, v in counts.items()},
                      "screening_ids": candidate_record["count"],
                      "unique_screening_sequences": len(candidate_proteins),
                      "cross_split": cross_split}), flush=True)


if __name__ == "__main__":
    main()
