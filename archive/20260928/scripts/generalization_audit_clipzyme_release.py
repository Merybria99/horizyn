#!/usr/bin/env python3
"""Audit the exact official CLIPZyme EnzymeMap split and screening IDs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    acquisition = json.loads(args.acquisition.read_text())
    for name, record in acquisition["files"].items():
        if sha256(args.release / name) != record["sha256"]:
            raise ValueError(f"Release file changed: {name}")
    split_path = args.release / "reaction_rule_split.p"
    if sha256(split_path) != acquisition["reaction_rule_split"]["sha256"]:
        raise ValueError("Rule split changed")
    # The split pickle was opcode-checked during acquisition. The remaining
    # pickles are checksum-verified official release artifacts, not arbitrary
    # files supplied through the model evaluation path.
    with split_path.open("rb") as handle:
        rule_split = pickle.load(handle)
    with (args.release / "cached_enzymemap.p").open("rb") as handle:
        entries = pickle.load(handle)
    with (args.release / "uniprot2sequence.p").open("rb") as handle:
        id_to_sequence = pickle.load(handle)
    with (args.release / "clipzyme_screening_set.p").open("rb") as handle:
        screening = pickle.load(handle)
    with (args.release / "enzymemap.json").open() as handle:
        raw = json.load(handle)

    if not isinstance(entries, list) or not isinstance(raw, list):
        raise ValueError("Unexpected EnzymeMap collection type")
    if set(screening) != {"hiddens", "uniprots"}:
        raise ValueError("Unexpected screening object")
    screen_ids = screening["uniprots"]
    if screening["hiddens"].shape[0] != len(screen_ids):
        raise ValueError("Screening ID/embedding count mismatch")
    if len(screen_ids) != len(set(screen_ids)):
        raise ValueError("Duplicate screening IDs")
    if len(id_to_sequence) != len(screen_ids) or set(id_to_sequence) != set(screen_ids):
        raise ValueError("Screening/sequence ID mismatch")
    split_counts = Counter()
    rule_ids = {group: set() for group in ("train", "dev", "test")}
    missing_rules = set()
    missing_sequences = set()
    rows_without_reaction = 0
    for row in entries:
        rule = row.get("rule_id")
        group = rule_split.get(rule)
        if group is None:
            missing_rules.add(rule)
            continue
        split_counts[group] += 1
        rule_ids[group].add(rule)
        protein = row.get("uniprot_id") or row.get("protein_id")
        if protein not in id_to_sequence:
            missing_sequences.add(protein)
        if not row.get("reaction_string"):
            rows_without_reaction += 1
    if missing_rules or missing_sequences or rows_without_reaction:
        raise ValueError("Cached entries fail rule, sequence, or reaction coverage")
    if sum(map(len, rule_ids.values())) != len(set.union(*rule_ids.values())):
        raise ValueError("Rule IDs cross partitions")
    expected = {"train": 34427, "dev": 7287, "test": 4642}
    if dict(split_counts) != expected or len(entries) != 46356 or len(screen_ids) != 261907:
        raise ValueError(f"Released counts differ from paper: {dict(split_counts)}")
    lengths = [len(s) if s else 0 for s in id_to_sequence.values()]
    max_len = max(lengths)
    receipt = {
        "schema": "official_clipzyme_enzymemap_release_audit_v1",
        "acquisition_sha256": sha256(args.acquisition),
        "release_record": acquisition["release_record"],
        "raw_enzymemap_reaction_rows": len(raw),
        "cached_association_entries": len(entries),
        "associations_per_split": dict(split_counts),
        "rules_per_split_with_entries": {k: len(v) for k, v in rule_ids.items()},
        "rule_id_overlap": 0,
        "screening_protein_ids": len(screen_ids),
        "screening_embedding_shape": list(screening["hiddens"].shape),
        "screening_sequence_ids_match": True,
        "screening_max_sequence_length": max_len,
        "screening_sequences_longer_than_650": sum(length > 650 for length in lengths),
        "screening_empty_sequences": sum(length == 0 for length in lengths),
        "screening_length_statement_exactly_matches_release": all(
            0 < length <= 650 for length in lengths
        ),
        "all_cached_entries_have_reaction_and_protein": True,
        "labels_used_to_select_model": False,
        "source_code_sha256": sha256(Path(__file__)),
        "feature_extraction_complete": False,
        "target_model_retrained": False,
        "matched_paper_evaluation_complete": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
