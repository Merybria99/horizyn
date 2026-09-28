#!/usr/bin/env python3
"""Label-free test input exposure strata for the official EnzymeMap rule split."""
from __future__ import annotations

import argparse
from collections import Counter
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


def read_rows(record: dict) -> list[dict]:
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise ValueError("Source association file changed")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != record["count"]:
        raise ValueError("Source association count changed")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    if manifest["schema"] != "fixed_architecture_clipzyme_manifest_v1":
        raise ValueError("Unexpected official manifest")
    train = read_rows(manifest["associations"]["train"])
    test = read_rows(manifest["associations"]["test"])
    ids = {row["protein_id"] for row in train}
    seqs = {row["sequence"] for row in train if row["sequence"]}
    rxns = {row["reaction"] for row in train}
    exact_id = {(row["protein_id"], row["reaction"]) for row in train}
    exact_seq = {(row["sequence"], row["reaction"]) for row in train if row["sequence"]}
    flags = []
    for row in test:
        sequence = row["sequence"]
        flags.append({
            "source_index": row["source_index"],
            "protein_id_seen": row["protein_id"] in ids,
            "protein_sequence_seen": bool(sequence) and sequence in seqs,
            "reaction_string_seen": row["reaction"] in rxns,
            "id_reaction_pair_seen": (row["protein_id"], row["reaction"]) in exact_id,
            "sequence_reaction_pair_seen": bool(sequence) and
                (sequence, row["reaction"]) in exact_seq,
            "sequence_missing": not bool(sequence),
        })
    counts = {key: sum(row[key] for row in flags)
              for key in flags[0] if key != "source_index"}
    cross = Counter((row["protein_sequence_seen"], row["reaction_string_seen"])
                    for row in flags)
    args.output.mkdir(parents=True)
    csv_path = args.output / "test_association_input_strata.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flags[0].keys())
        writer.writeheader()
        writer.writerows(flags)
    receipt = {
        "schema": "clipzyme_enzymemap_test_input_exposure_v1",
        "source_manifest_sha256": sha256(args.manifest),
        "test_associations": len(test),
        "counts": counts,
        "sequence_seen_x_reaction_seen": {
            f"sequence_{int(sequence)}_reaction_{int(reaction)}": cross[(sequence, reaction)]
            for sequence in (False, True) for reaction in (False, True)
        },
        "per_association_csv_sha256": sha256(csv_path),
        "no_model_scores_or_outcomes_used": True,
        "source_code_sha256": sha256(Path(__file__)),
    }
    (args.output / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["counts"]), flush=True)


if __name__ == "__main__":
    main()
