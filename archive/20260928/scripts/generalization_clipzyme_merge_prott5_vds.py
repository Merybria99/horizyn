#!/usr/bin/env python3
"""Join reusable and newly extracted ProtT5 residues without copying vectors."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True,
                        help="screening_candidate_map.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.catalog.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {row["protein_id"] for row in rows}
    if len(rows) != 261907 or len(required) != 222985:
        raise ValueError("Official screening candidate pool changed")
    files = [h5py.File(path, "r") for path in (args.cached, args.new)]
    try:
        ids = []
        offsets = [0]
        source_shapes = []
        for handle in files:
            source_ids = [value.decode() if isinstance(value, bytes) else str(value)
                          for value in handle["ids"][:]]
            source_offsets = handle["offsets"][:]
            vectors = handle["vectors"]
            if (len(source_offsets) != len(source_ids) + 1
                    or int(source_offsets[0]) != 0
                    or int(source_offsets[-1]) != vectors.shape[0]
                    or vectors.shape[1] != 1024
                    or vectors.dtype != np.float16):
                raise ValueError("Invalid ProtT5 ragged source")
            ids.extend(source_ids)
            offsets.extend((source_offsets[1:] + offsets[-1]).tolist())
            source_shapes.append(vectors.shape)
        if len(set(ids)) != len(ids):
            raise ValueError("Cached and new ProtT5 sources have overlapping protein IDs")
        missing = required - set(ids)
        if missing:
            raise ValueError(f"Missing {len(missing)} official screening sequences")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        layout = h5py.VirtualLayout(shape=(offsets[-1], 1024), dtype=np.float16)
        cursor = 0
        for path, shape in zip((args.cached, args.new), source_shapes, strict=True):
            source = h5py.VirtualSource(str(path.resolve()), "vectors", shape=shape)
            layout[cursor:cursor + shape[0], :] = source
            cursor += shape[0]
        with h5py.File(args.output, "w", libver="latest") as out:
            out.create_dataset("ids", data=np.asarray(ids, dtype=h5py.string_dtype("utf-8")))
            out.create_dataset("offsets", data=np.asarray(offsets, dtype=np.int64))
            out.create_virtual_dataset("vectors", layout)
            out.attrs["embedding_model_type"] = "prott5"
            out.attrs["residue_dim"] = 1024
            out.attrs["merge_storage"] = "virtual"
            out.attrs["source_cached"] = str(args.cached.resolve())
            out.attrs["source_new"] = str(args.new.resolve())
        with h5py.File(args.output, "r") as merged:
            for i in (0, len(files[0]["ids"]) - 1, len(files[0]["ids"]), len(ids) - 1):
                source_idx = 0 if i < len(files[0]["ids"]) else 1
                local = i if source_idx == 0 else i - len(files[0]["ids"])
                source = files[source_idx]
                actual = merged["vectors"][offsets[i]:offsets[i] + 1]
                expected = source["vectors"][source["offsets"][local]:source["offsets"][local] + 1]
                if not np.array_equal(actual, expected):
                    raise ValueError("Nested virtual dataset failed source replay")
        receipt = {"schema": "clipzyme_f3_prott5_combined_vds_v1",
                   "candidate_rows": len(rows), "candidate_unique_sequences": len(required),
                   "combined_protein_ids": len(ids), "combined_residues": offsets[-1],
                   "source_files": [str(p.resolve()) for p in (args.cached, args.new)],
                   "source_shapes": source_shapes,
                   "catalog_sha256": sha256(args.catalog),
                   "source_sha256": sha256(Path(__file__)),
                   "output_sha256": sha256(args.output)}
        (args.output.parent / "combined_prott5_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({key: receipt[key] for key in
                          ("candidate_rows", "candidate_unique_sequences",
                           "combined_protein_ids", "combined_residues")}), flush=True)
    finally:
        for handle in files:
            handle.close()


if __name__ == "__main__":
    main()
