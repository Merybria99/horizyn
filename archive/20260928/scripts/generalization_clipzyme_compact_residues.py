#!/usr/bin/env python3
"""Materialize the exact EnzymeMap train/validation ProtT5 rows for fast reads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def contiguous_blocks(selected, limit):
    """Yield compact row ranges that are contiguous in the source bank."""
    if limit < 1:
        raise ValueError("Block size must be positive")
    start = 0
    while start < len(selected):
        stop = start + 1
        while stop < len(selected) and stop - start < limit and selected[stop] == selected[stop - 1] + 1:
            stop += 1
        yield start, stop
        start = stop


def read_ahead_blocks(selected, offsets, limit, max_rows):
    start = 0
    while start < len(selected):
        stop = start + 1
        while (stop < len(selected) and stop - start < limit and
               offsets[selected[stop] + 1] - offsets[selected[start]] <= max_rows):
            stop += 1
        yield start, stop
        start = stop


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--block-proteins", type=int, default=256)
    p.add_argument("--purpose", choices=("train_validation", "screening"), default="train_validation")
    p.add_argument("--publish-link", type=Path)
    p.add_argument("--read-ahead-mib", type=int, default=0,
                   help="Read contiguous spans including gaps, then retain only requested rows")
    p.add_argument("--source-identity-only", action="store_true",
                   help="Guard source size/mtime instead of re-reading the entire source for SHA256")
    a = p.parse_args()
    if a.block_proteins < 1 or a.read_ahead_mib < 0:
        p.error("Invalid block size or read-ahead size")
    if a.output.exists():
        raise FileExistsError(a.output)
    wanted = set(json.loads(a.catalog.read_text())["proteins"])
    if not wanted:
        raise ValueError("Protein catalog is empty")
    source_stat = a.source.stat()
    temporary = a.output.with_suffix(".partial.h5")
    if temporary.exists():
        raise FileExistsError(temporary)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.source) as source:
        keys = source["ids"].asstr()[:].tolist()
        offsets = source["offsets"][:]
        selected = [i for i, key in enumerate(keys) if key in wanted]
        if len(selected) != len(wanted):
            raise ValueError("Missing requested protein IDs")
        lengths = np.diff(offsets)[selected]
        compact_offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
        with h5py.File(temporary, "w") as out:
            out.create_dataset("ids", data=np.asarray([keys[i] for i in selected], dtype=object),
                               dtype=h5py.string_dtype())
            out.create_dataset("offsets", data=compact_offsets)
            vectors = out.create_dataset("vectors", shape=(int(compact_offsets[-1]), 1024),
                                         dtype=source["vectors"].dtype)
            blocks = (read_ahead_blocks(selected, offsets, a.block_proteins,
                      a.read_ahead_mib * 1024**2 // (1024 * source["vectors"].dtype.itemsize))
                      if a.read_ahead_mib else contiguous_blocks(selected, a.block_proteins))
            for row, stop in blocks:
                lo, hi = int(offsets[selected[row]]), int(offsets[selected[stop - 1] + 1])
                out_lo, out_hi = int(compact_offsets[row]), int(compact_offsets[stop])
                block = source["vectors"][lo:hi]
                if selected[stop - 1] - selected[row] + 1 == stop - row:
                    vectors[out_lo:out_hi] = block
                else:
                    # Gaps are read for I/O locality, never included in output.
                    gathered = np.concatenate([block[int(offsets[i]) - lo:int(offsets[i + 1]) - lo]
                                               for i in selected[row:stop]])
                    vectors[out_lo:out_hi] = gathered
                if stop // 1000 > row // 1000 or stop == len(selected):
                    print(json.dumps({"copied": stop, "total": len(selected)}), flush=True)
        with h5py.File(temporary) as out:
            rng = np.random.default_rng(42)
            sample = sorted(set([0, len(selected) - 1] +
                                rng.choice(len(selected), min(100, len(selected)), replace=False).tolist()))
            for row in sample:
                physical = selected[row]
                original = source["vectors"][int(offsets[physical]):int(offsets[physical + 1])]
                materialized = out["vectors"][int(compact_offsets[row]):int(compact_offsets[row + 1])]
                if not np.array_equal(original, materialized):
                    raise ValueError(f"Physical ProtT5 copy changed protein {keys[physical]}")
    after = a.source.stat()
    if (source_stat.st_size, source_stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed while copying residue vectors")
    os.replace(temporary, a.output)
    receipt = dict(schema="clipzyme_train_validation_compact_residues_v1",
                   purpose=a.purpose, block_proteins=a.block_proteins, read_ahead_mib=a.read_ahead_mib,
                   source_sha256=None if a.source_identity_only else digest(a.source),
                   source_identity=dict(path=str(a.source.resolve()), size=source_stat.st_size,
                                        mtime_ns=source_stat.st_mtime_ns),
                   source_verification="size/mtime stable throughout copy" if a.source_identity_only else "sha256",
                   catalog_sha256=digest(a.catalog),
                   output_sha256=digest(a.output), protein_count=len(selected),
                   output_size_bytes=a.output.stat().st_size,
                   output_mtime_ns=a.output.stat().st_mtime_ns,
                   residue_count=int(compact_offsets[-1]), verified_sequences=len(sample),
                   data_transformation="byte-identical residue vectors, physical subset only",
                   source_code_sha256=digest(Path(__file__)))
    a.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if a.publish_link:
        a.publish_link.with_suffix(".receipt.json").symlink_to(a.output.with_suffix(".receipt.json").resolve())
        a.publish_link.symlink_to(a.output.resolve())
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
