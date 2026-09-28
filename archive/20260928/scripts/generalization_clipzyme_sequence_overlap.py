#!/usr/bin/env python3
"""Audit exact sequence/reaction repeats behind distinct CLIPZyme protein IDs."""
from __future__ import annotations

import argparse
from collections import defaultdict
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != "fixed_architecture_clipzyme_manifest_v1":
        raise ValueError("Unexpected source manifest")
    split_pairs = {}
    for split, record in manifest["associations"].items():
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Source split changed: {split}")
        pairs = defaultdict(list)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if not row["sequence"]:
                    continue
                key = (hashlib.sha256(row["sequence"].encode()).hexdigest(),
                       hashlib.sha256(row["reaction"].encode()).hexdigest())
                pairs[key].append({"source_index": int(row["source_index"]),
                                   "uniprot_id": row["protein_id"]})
        split_pairs[split] = pairs
    overlaps = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        shared = split_pairs[left].keys() & split_pairs[right].keys()
        overlaps[f"{left}_{right}"] = {
            "unique_exact_sequence_reaction_pairs": len(shared),
            "records": [{"sequence_sha256": key[0], "reaction_sha256": key[1],
                         left: split_pairs[left][key], right: split_pairs[right][key]}
                        for key in sorted(shared)],
        }
    report = {
        "schema": "clipzyme_rule_split_exact_sequence_reaction_exposure_v1",
        "source_manifest_sha256": sha256(args.manifest),
        "overlaps": overlaps,
        "input_only_association_audit": True,
        "no_model_scores_used": True,
        "source_code_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v["unique_exact_sequence_reaction_pairs"]
                      for k, v in overlaps.items()}), flush=True)


if __name__ == "__main__":
    main()
