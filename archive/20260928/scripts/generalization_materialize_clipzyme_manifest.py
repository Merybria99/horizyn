#!/usr/bin/env python3
"""Materialize source-ordered CLIPZyme splits for fixed-architecture retraining."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import pickle


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = json.loads(args.audit.read_text())
    if audit.get("schema") != "official_clipzyme_enzymemap_release_audit_v1":
        raise ValueError("The official release audit is required")
    with (args.release / "reaction_rule_split.p").open("rb") as handle:
        rule_split = pickle.load(handle)
    with (args.release / "cached_enzymemap.p").open("rb") as handle:
        entries = pickle.load(handle)
    with (args.release / "uniprot2sequence.p").open("rb") as handle:
        sequences = pickle.load(handle)
    with (args.release / "clipzyme_screening_set.p").open("rb") as handle:
        screening = pickle.load(handle)
    if len(entries) != audit["cached_association_entries"] or len(
            screening["uniprots"]) != audit["screening_protein_ids"]:
        raise ValueError("Audit/release count mismatch")
    args.output.mkdir(parents=True)
    paths = {k: args.output / f"{k}_associations.csv"
             for k in ("train", "dev", "test")}
    counts = Counter()
    proteins = {k: set() for k in paths}
    reactions = {k: set() for k in paths}
    pairs = {k: set() for k in paths}
    files = {k: p.open("w", newline="") for k, p in paths.items()}
    try:
        writers = {k: csv.writer(f) for k, f in files.items()}
        for writer in writers.values():
            writer.writerow(("source_index", "rule_id", "sample_id", "protein_id",
                             "reaction", "sequence"))
        for index, entry in enumerate(entries):
            split = rule_split[entry["rule_id"]]
            protein = entry["uniprot_id"]
            reaction = entry["reaction_string"]
            writers[split].writerow((index, entry["rule_id"], entry["sample_id"],
                                     protein, reaction, sequences[protein]))
            counts[split] += 1
            proteins[split].add(protein)
            reactions[split].add(reaction)
            pairs[split].add((protein, reaction))
    finally:
        for f in files.values():
            f.close()
    if dict(counts) != audit["associations_per_split"]:
        raise ValueError("Materialized association counts mismatch")
    candidate_path = args.output / "screening_candidates.csv"
    empty_ids, long_ids = [], []
    with candidate_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("candidate_index", "protein_id", "sequence"))
        for index, protein in enumerate(screening["uniprots"]):
            sequence = sequences[protein] or ""
            writer.writerow((index, protein, sequence))
            if not sequence:
                empty_ids.append(protein)
            elif len(sequence) > 650:
                long_ids.append(protein)
    if (len(empty_ids) != audit["screening_empty_sequences"] or
            len(long_ids) != audit["screening_sequences_longer_than_650"]):
        raise ValueError("Candidate feature coverage differs from audit")
    report = {
        "schema": "fixed_architecture_clipzyme_manifest_v1",
        "release_audit": {"path": str(args.audit.resolve()),
                          "sha256": sha256(args.audit)},
        "source_cache_sha256": sha256(args.release / "cached_enzymemap.p"),
        "source_split_sha256": sha256(args.release / "reaction_rule_split.p"),
        "source_sequence_sha256": sha256(args.release / "uniprot2sequence.p"),
        "associations": {k: {"count": counts[k], "unique_proteins": len(proteins[k]),
                             "unique_reactions": len(reactions[k]),
                             "path": str(paths[k].resolve()), "sha256": sha256(paths[k])}
                         for k in paths},
        "train_dev_protein_overlap": len(proteins["train"] & proteins["dev"]),
        "train_test_protein_overlap": len(proteins["train"] & proteins["test"]),
        "train_dev_reaction_string_overlap": len(reactions["train"] & reactions["dev"]),
        "train_test_reaction_string_overlap": len(reactions["train"] & reactions["test"]),
        "train_dev_exact_pair_overlap": len(pairs["train"] & pairs["dev"]),
        "train_test_exact_pair_overlap": len(pairs["train"] & pairs["test"]),
        "distinct_pairs_per_split": {k: len(v) for k, v in pairs.items()},
        "candidate_pool": {"count": len(screening["uniprots"]),
                           "path": str(candidate_path.resolve()),
                           "sha256": sha256(candidate_path),
                           "empty_sequence_ids": empty_ids,
                           "sequence_ids_over_650": long_ids},
        "training_input_files": [str(paths["train"].resolve())],
        "validation_input_files": [str(paths["dev"].resolve())],
        "heldout_labels_for_evaluation_only": [str(paths["test"].resolve())],
        "candidate_order_identical_to_official_release": True,
        "source_code_sha256": sha256(Path(__file__)),
        "features_or_model_trained_by_this_script": False,
    }
    receipt = args.output / "manifest.json"
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"counts": dict(counts), "candidates": len(screening["uniprots"]),
                      "manifest": str(receipt)}), flush=True)


if __name__ == "__main__":
    main()
