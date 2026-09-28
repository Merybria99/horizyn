#!/usr/bin/env python3
"""Count CLIPZyme proteins already represented in the EnzymeCAGE F3 cache."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def protein_key(sequence: str) -> str:
    return "p_" + hashlib.sha256(sequence.encode()).hexdigest()[:24]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != "fixed_architecture_clipzyme_manifest_v1":
        raise ValueError("Require official source-order manifest")
    with h5py.File(args.cache, "r") as handle:
        cached = set(handle["ids"].asstr()[:])
    result = {}
    for split, record in manifest["associations"].items():
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Association manifest changed: {split}")
        unique = {}
        empty_rows = 0
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                unique[row["protein_id"]] = row["sequence"]
                empty_rows += not bool(row["sequence"])
        result[split] = {
            "association_rows": record["count"],
            "empty_sequence_association_rows": empty_rows,
            "unique_empty_sequence_protein_ids": sum(not bool(seq) for seq in unique.values()),
            "unique_protein_ids": len(unique),
            "protein_ids_with_cached_sequence": sum(
                bool(seq) and protein_key(seq) in cached for seq in unique.values()
            ),
            "protein_ids_missing_cached_sequence": sum(
                not seq or protein_key(seq) not in cached for seq in unique.values()
            ),
        }
    candidates = manifest["candidate_pool"]
    candidate_path = Path(candidates["path"])
    if sha256(candidate_path) != candidates["sha256"]:
        raise ValueError("Candidate manifest changed")
    cached_candidates = 0
    empty_candidates = 0
    distinct_uncached_sequences = set()
    with candidate_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sequence = row["sequence"]
            if not sequence:
                empty_candidates += 1
            elif protein_key(sequence) in cached:
                cached_candidates += 1
            else:
                distinct_uncached_sequences.add(sequence)
    if cached_candidates + empty_candidates + sum(
            1 for _ in distinct_uncached_sequences) > candidates["count"]:
        raise ValueError("Impossible candidate coverage counts")
    report = {
        "schema": "clipzyme_f3_existing_prott5_feature_coverage_v1",
        "manifest_sha256": sha256(args.manifest),
        "cache_path": str(args.cache.resolve()),
        "cache_ids": len(cached),
        "splits": result,
        "screening": {
            "candidate_ids": candidates["count"],
            "candidate_ids_with_cached_sequence": cached_candidates,
            "empty_sequence_ids": empty_candidates,
            "candidate_ids_requiring_new_extraction": candidates["count"] - cached_candidates - empty_candidates,
            "distinct_sequences_requiring_new_extraction": len(distinct_uncached_sequences),
        },
        "key_policy": "p_ + sha256(sequence)[:24], as in prepare_enzymecage_f3.py",
        "source_code_sha256": sha256(Path(__file__)),
        "cache_is_feature_input_only": True,
        "feature_extraction_complete": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["screening"]), flush=True)


if __name__ == "__main__":
    main()
