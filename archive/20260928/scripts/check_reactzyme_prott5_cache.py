#!/usr/bin/env python3
"""Read-only ReactZyme cache screening; NOT a full finite-value certificate.

Checks every required ID and every offset, then reads sampled protein vectors.
Does not load torch, use GPUs, write a report, repair files, or start training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data/standardized/retrieval_training_source_collapse/test/horizyn_reactzyme_shared_candidates/proteins_prott5_residue.h5"


def file_identity(path):
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def required_ids(candidate_root, expected_count):
    required, counts, identities = set(), {}, {}
    for protocol in ("reaction_smi", "enzyme_smi", "time"):
        for part in ("train", "validation", "test"):
            path = candidate_root / protocol / f"{part}_candidate_ids.txt"
            identities[path] = file_identity(path)
            with path.open() as handle:
                rows = [line.strip() for line in handle if line.strip()]
            if not rows or len(rows) != len(set(rows)):
                raise ValueError(f"Empty or duplicate candidate IDs: {path}")
            required.update(rows)
            counts[f"{protocol}/{part}"] = len(rows)
    if len(required) != expected_count:
        raise ValueError(f"Expected {expected_count:,} required IDs, found {len(required):,}")
    return required, counts, identities


def check(args):
    started = time.monotonic()
    print("Loading CPU HDF5 libraries (no GPU use)...", flush=True)
    import h5py
    import numpy as np

    print("Reading the nine candidate lists...", flush=True)
    required, counts, identities = required_ids(args.candidate_root, args.expected_proteins)
    identities[args.cache] = file_identity(args.cache)
    print(f"Opening cache; checking coverage of {len(required):,} required proteins...", flush=True)
    with h5py.File(args.cache, "r") as handle:
        if any(key not in handle for key in ("ids", "offsets", "vectors")):
            raise ValueError("Missing ids, offsets, or vectors dataset")
        if handle["ids"].ndim != 1:
            raise ValueError("IDs must be a one-dimensional dataset")
        ids = list(handle["ids"].asstr()[:])
        if not ids or len(set(ids)) != len(ids) or any(not key for key in ids):
            raise ValueError("Empty or duplicate cache IDs")
        missing = required.difference(ids)
        if missing:
            raise ValueError(f"Missing {len(missing):,} required proteins: {sorted(missing)[:10]}")
        offsets = handle["offsets"][:]
        vectors = handle["vectors"]
        if vectors.ndim != 2 or vectors.shape[1] != 1024 or not np.issubdtype(vectors.dtype, np.floating):
            raise ValueError(f"Unexpected vector shape/dtype: {vectors.shape}, {vectors.dtype}")
        if (offsets.ndim != 1 or len(offsets) != len(ids) + 1
                or not np.issubdtype(offsets.dtype, np.integer)
                or offsets[0] != 0 or offsets[-1] != vectors.shape[0]
                or not bool(np.all(offsets[1:] > offsets[:-1]))):
            raise ValueError("Invalid offset shape, bounds, or ordering")
        # Missing virtual sources can otherwise be silently replaced by fill values.
        backing_datasets = []
        if vectors.is_virtual:
            seen = set()
            for source in vectors.virtual_sources():
                name = source.file_name
                if isinstance(name, bytes):
                    name = name.decode()
                path = args.cache if name == "." else Path(name)
                if not path.is_absolute():
                    path = args.cache.parent / path
                path = path.resolve(strict=True)
                dataset = source.dset_name
                if isinstance(dataset, bytes):
                    dataset = dataset.decode()
                if (path, dataset) in seen:
                    continue
                seen.add((path, dataset))
                identities[path] = file_identity(path)
                with h5py.File(path, "r") as backing:
                    if dataset not in backing:
                        raise ValueError(f"Missing virtual source dataset: {path}:{dataset}")
                    backing_datasets.append(dict(path=str(path), dataset=dataset, shape=list(backing[dataset].shape)))
        positions = np.asarray([i for i, key in enumerate(ids) if key in required], dtype=np.int64)
        lengths = offsets[1:] - offsets[:-1]
        selection = np.linspace(0, len(positions) - 1, min(args.samples, len(positions)), dtype=int)
        sampled = set(int(i) for i in positions[selection])
        sampled.update((0, len(ids) - 1, int(positions[np.argmin(lengths[positions])]),
                        int(positions[np.argmax(lengths[positions])])))
        print(f"Coverage and offsets OK; reading {len(sampled)} sampled proteins...", flush=True)
        sample_bytes = 0
        for number, i in enumerate(sorted(sampled), 1):
            start, stop = int(offsets[i]), int(offsets[i + 1])
            nonzero = False
            # Bound memory even if an older cache contains untruncated sequences.
            for chunk_start in range(start, stop, 4096):
                block = vectors[chunk_start:min(stop, chunk_start + 4096)]
                if not bool(np.isfinite(block).all()):
                    raise ValueError(f"Nonfinite sampled vector: {ids[i]}")
                nonzero = nonzero or bool(np.any(block))
                sample_bytes += block.nbytes
            if not nonzero:
                raise ValueError(f"All-zero sampled protein: {ids[i]}")
            if number % 8 == 0 or number == len(sampled):
                print(f"Sampled reads: {number}/{len(sampled)}", flush=True)
        report = dict(
            cache=str(args.cache), cache_proteins=len(ids), required_proteins=len(required),
            missing_required_proteins=0, split_candidate_counts=counts,
            vectors_shape=list(vectors.shape), vectors_dtype=str(vectors.dtype),
            required_length_min=int(lengths[positions].min()), required_length_max=int(lengths[positions].max()),
            metadata={key: handle.attrs[key] for key in handle.attrs},
            virtual=bool(vectors.is_virtual), backing_datasets=backing_datasets,
            sampled_proteins=len(sampled), sampled_bytes=sample_bytes,
            full_vector_scan=False, representation_provenance_verified=False,
            files_modified=False, training_started=False,
        )
    for path, identity in identities.items():
        if file_identity(path) != identity:
            raise ValueError(f"File changed during screening: {path}")
    report["elapsed_seconds"] = time.monotonic() - started
    print(json.dumps(report, indent=2, default=str), flush=True)
    print("SCREENING PASSED: coverage, offsets, and sampled reads only. "
          "Not a full integrity or representation-provenance certificate.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--candidate-root", type=Path, default=ROOT / "data/revised_protocols/reactzyme_paper")
    parser.add_argument("--expected-proteins", type=int, default=178327)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    if args.expected_proteins < 1 or not 1 <= args.samples <= 256:
        parser.error("Require positive expected-proteins and 1..256 samples")
    args.cache = args.cache.absolute()
    try:
        check(args)
    except Exception as exc:
        print(f"SCREENING FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
