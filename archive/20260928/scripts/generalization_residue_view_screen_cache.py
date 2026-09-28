#!/usr/bin/env python3
"""Copy the frozen screening bank locally, retaining its exact order and values.

Keep the source HDF5 dataset open across reads: reopening this large virtual
dataset repeatedly rebuilds its mapping metadata. Existing GPU exports keep
their current source; subsequent exports discover the atomically published bank.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np

from generalization_clipzyme_compact_residues import digest, read_ahead_blocks


def identity(path):
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def link_state(path):
    if path.is_symlink():
        return dict(kind="symlink", target=os.readlink(path), live=path.exists())
    if path.exists():
        return dict(kind="file")
    return dict(kind="absent")


def publish(output, link, expected):
    """Replace absent/dangling cache pointers only, with the data link last."""
    receipt_link = link.with_suffix(".receipt.json")
    with link.with_suffix(".publication.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        actual = [link_state(receipt_link), link_state(link)]
        if actual != expected:
            raise ValueError("Cache publication pointers changed during copy")
        if any(s["kind"] == "file" or s.get("live") for s in actual):
            raise FileExistsError("Refusing to replace a live screening cache")
        for target, destination in [(output.with_suffix(".receipt.json"), receipt_link), (output, link)]:
            temporary = destination.with_name(destination.name + f".publish.{os.getpid()}")
            temporary.symlink_to(target.resolve())
            os.replace(temporary, destination)


def materialize(source_path, catalog, output, publish_link=None, block_proteins=512, read_ahead_mib=128):
    if block_proteins < 1 or read_ahead_mib < 1:
        raise ValueError("Positive block and read-ahead sizes required")
    partial = output.with_suffix(".partial.h5")
    if output.exists() or partial.exists():
        raise FileExistsError(output)
    prior_links = None
    if publish_link:
        prior_links = [link_state(publish_link.with_suffix(".receipt.json")), link_state(publish_link)]
        if any(s["kind"] == "file" or s.get("live") for s in prior_links):
            raise FileExistsError("A live screening cache already exists")
    wanted = set(json.loads(catalog.read_text())["proteins"])
    if not wanted:
        raise ValueError("Empty protein catalog")
    before = identity(source_path)
    source_hash = digest(source_path)
    catalog_hash = digest(catalog)
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_path, "r") as source:
        keys = source["ids"].asstr()[:].tolist()
        offsets = source["offsets"][:]
        source_vectors = source["vectors"]
        selected = [i for i, key in enumerate(keys) if key in wanted]
        if len(selected) != len(wanted) or len({keys[i] for i in selected}) != len(wanted):
            raise ValueError("Missing or duplicated requested protein IDs")
        if len(offsets) != len(keys) + 1 or np.any(np.diff(offsets) <= 0):
            raise ValueError("Invalid source offsets")
        backing_paths = set()
        if source_vectors.is_virtual:
            for mapping in source_vectors.virtual_sources():
                name = os.fsdecode(mapping.file_name)
                path = Path(name)
                if name == ".":
                    path = source_path
                elif not path.is_absolute():
                    path = source_path.parent / path
                backing_paths.add(path.resolve())
        backing_before = [identity(p) for p in sorted(backing_paths)]
        lengths = np.diff(offsets)[selected]
        compact_offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
        width = source_vectors.shape[1]
        max_rows = read_ahead_mib * 1024**2 // (width * source_vectors.dtype.itemsize)
        print(json.dumps(dict(stage="copying", proteins=len(selected), rows=int(compact_offsets[-1]),
                              backing_sources=backing_before)), flush=True)
        with h5py.File(partial, "w") as out:
            out.create_dataset("ids", data=np.asarray([keys[i] for i in selected], dtype=object),
                               dtype=h5py.string_dtype())
            out.create_dataset("offsets", data=compact_offsets)
            target = out.create_dataset("vectors", shape=(int(compact_offsets[-1]), width), dtype=source_vectors.dtype)
            for row, stop in read_ahead_blocks(selected, offsets, block_proteins, max_rows):
                lo, hi = int(offsets[selected[row]]), int(offsets[selected[stop-1]+1])
                block = source_vectors[lo:hi]
                if selected[stop-1] - selected[row] + 1 != stop - row:
                    block = np.concatenate([block[int(offsets[i])-lo:int(offsets[i+1])-lo] for i in selected[row:stop]])
                target[int(compact_offsets[row]):int(compact_offsets[stop])] = block
                if stop // 4000 > row // 4000 or stop == len(selected):
                    print(json.dumps(dict(copied=stop, total=len(selected), seconds=time.monotonic()-started)), flush=True)
        with h5py.File(partial, "r") as out:
            copied_vectors = out["vectors"]
            sample = sorted(set([0, len(selected)-1] + np.random.default_rng(42).choice(
                len(selected), min(100, len(selected)), replace=False).tolist()))
            for row in sample:
                physical = selected[row]
                original = source_vectors[int(offsets[physical]):int(offsets[physical+1])]
                copied = copied_vectors[int(compact_offsets[row]):int(compact_offsets[row+1])]
                if not np.array_equal(original, copied):
                    raise ValueError(f"Residue copy changed {keys[physical]}")
    if before != identity(source_path) or source_hash != digest(source_path):
        raise ValueError("Source changed during copy")
    if backing_before != [identity(p) for p in sorted(backing_paths)]:
        raise ValueError("Backing source changed during copy")
    if catalog_hash != digest(catalog):
        raise ValueError("Catalog changed during copy")
    os.replace(partial, output)
    print(json.dumps(dict(stage="hashing_output", seconds=time.monotonic()-started)), flush=True)
    receipt = dict(schema="clipzyme_train_validation_compact_residues_v1", purpose="screening",
        source_sha256=source_hash, source_identity=before, source_verification="sha256 and stable backing-source identities",
        backing_sources=backing_before, catalog_sha256=catalog_hash, output_sha256=digest(output),
        protein_count=len(selected), output_size_bytes=output.stat().st_size, output_mtime_ns=output.stat().st_mtime_ns,
        residue_count=int(compact_offsets[-1]), verified_sequences=len(sample),
        data_transformation="byte-identical residue vectors, physical subset only",
        source_code_sha256=digest(Path(__file__)), block_proteins=block_proteins, read_ahead_mib=read_ahead_mib,
        prior_publication_links=prior_links, copy_and_hash_seconds=time.monotonic()-started)
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    if publish_link:
        publish(output, publish_link, prior_links)
    print(json.dumps(receipt), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "catalog", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--publish-link", type=Path)
    parser.add_argument("--block-proteins", type=int, default=512)
    parser.add_argument("--read-ahead-mib", type=int, default=128)
    a = parser.parse_args()
    materialize(a.source, a.catalog, a.output, a.publish_link, a.block_proteins, a.read_ahead_mib)


if __name__ == "__main__":
    main()
