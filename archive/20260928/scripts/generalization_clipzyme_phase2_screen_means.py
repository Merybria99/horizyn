#!/usr/bin/env python3
"""Compute label-free ProtT5 means for the fixed EnzymeMap screening library."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generalization_export import h5_ids


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-residues", type=int, default=65536)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with (args.catalog / "screening_candidate_map.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 261907:
        raise ValueError("Official screening library changed")
    required = {row["protein_id"] for row in rows}
    if len(required) != 222985:
        raise ValueError("Unique candidate protein count changed")
    source = args.catalog / "features/proteins_prott5_residue.h5"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source) as data, h5py.File(args.output, "w") as target:
        ids = h5_ids(data)
        offsets = data["offsets"][:]
        selected = [(i, key) for i, key in enumerate(ids) if key in required]
        if len(selected) != len(required):
            raise ValueError("Required screening proteins absent from residue cache")
        target.create_dataset("ids", data=np.asarray([key for _, key in selected], dtype=object),
                              dtype=h5py.string_dtype())
        vectors = target.create_dataset("vectors", shape=(len(selected), 1024), dtype="f4")
        done = target.create_dataset("complete", shape=(len(selected),), dtype="?")
        target.attrs["accumulation_dtype"] = "float32"
        cursor = 0
        while cursor < len(selected):
            physical = selected[cursor][0]
            start = int(offsets[physical])
            stop_index = cursor + 1
            while (stop_index < len(selected) and
                   int(offsets[selected[stop_index][0] + 1]) - start <= args.block_residues):
                stop_index += 1
            stop = int(offsets[selected[stop_index - 1][0] + 1])
            block = data["vectors"][start:stop]
            result = np.empty((stop_index - cursor, 1024), np.float32)
            for j, (row, _) in enumerate(selected[cursor:stop_index]):
                lo, hi = offsets[row:row + 2] - start
                result[j] = block[lo:hi].mean(0, dtype=np.float32)
            vectors[cursor:stop_index] = result
            done[cursor:stop_index] = True
            previous = cursor
            cursor = stop_index
            if cursor // 5000 > previous // 5000 or cursor == len(selected):
                print(json.dumps({"completed": cursor, "total": len(selected)}), flush=True)
    receipt = dict(schema="clipzyme_phase2_screen_protein_means_v1", labels_read=False,
                   source_sha256=digest(source), candidate_map_sha256=digest(
                       args.catalog / "screening_candidate_map.csv"),
                   output_sha256=digest(args.output), proteins=len(required))
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
