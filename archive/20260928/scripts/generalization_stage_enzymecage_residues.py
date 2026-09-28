#!/usr/bin/env python3
"""Stage immutable EnzymeCAGE ProT5 residue shards on local storage for DDP.

Copies and hashes every byte, rebuilds an equivalent VDS with unchanged IDs,
offsets and row order, and records endpoint samples before any model training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/enzymecage_f3_seed42/features/proteins_prott5_residue.h5"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*2**20), b""):
            h.update(block)
    return h.hexdigest()


def copy_hash(source: Path, dest: Path) -> str:
    if dest.exists():
        raise FileExistsError(dest)
    partial = dest.with_suffix(dest.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    h = hashlib.sha256()
    with source.open("rb") as reader, partial.open("xb") as writer:
        for block in iter(lambda: reader.read(8*2**20), b""):
            writer.write(block)
            h.update(block)
    partial.rename(dest)
    if dest.stat().st_size != source.stat().st_size or digest(dest) != h.hexdigest():
        raise ValueError("Staged shard differs from bytes read from source")
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Choose a fresh local staging directory")
    out.mkdir(parents=True)
    records = []
    with h5py.File(SOURCE) as original:
        source_shards = []
        mappings = []
        for item in original["vectors"].virtual_sources():
            source = Path(item.file_name.decode() if isinstance(item.file_name, bytes) else item.file_name)
            source_shards.append(source)
            lower, upper = item.vspace.get_select_bounds()
            mappings.append((int(lower[0]), int(upper[0])+1))
        shape, dtype = original["vectors"].shape, original["vectors"].dtype
        ids = original["ids"][:]
        offsets = original["offsets"][:]
        attrs = dict(original.attrs)
    if len(source_shards) != 4 or mappings[0][0] != 0 or mappings[-1][1] != shape[0]:
        raise ValueError("Unexpected source VDS layout")
    if any(mappings[i][1] != mappings[i+1][0] for i in range(3)):
        raise ValueError("Source VDS has a gap or overlap")
    layout = h5py.VirtualLayout(shape=shape, dtype=dtype)
    for source, (start, stop) in zip(source_shards, mappings):
        dest = out / source.name
        print(f"Copying {source.name}: {(source.stat().st_size/2**30):.2f} GiB", flush=True)
        copied_sha = copy_hash(source, dest)
        with h5py.File(dest) as shard:
            if shard["vectors"].shape != (stop-start, shape[1]) or shard["vectors"].dtype != dtype:
                raise ValueError("Staged source shard shape/dtype mismatch")
        layout[start:stop] = h5py.VirtualSource(str(dest), "vectors", shape=(stop-start, shape[1]))
        records.append({"source": str(source), "dest": str(dest), "sha256": copied_sha,
                        "bytes": dest.stat().st_size, "row_start": start, "row_stop": stop})
        print(f"Verified {source.name} SHA256 {copied_sha}", flush=True)
    vds = out / SOURCE.name
    with h5py.File(vds, "w", libver="latest") as staged:
        staged.create_dataset("ids", data=ids, dtype=h5py.string_dtype("utf-8"))
        staged.create_dataset("offsets", data=offsets)
        staged.create_virtual_dataset("vectors", layout, fillvalue=0)
        for key, value in attrs.items():
            staged.attrs[key] = value
        staged.attrs["local_staging_provenance"] = str(SOURCE)
    with h5py.File(SOURCE) as original, h5py.File(vds) as staged:
        if original["vectors"].shape != staged["vectors"].shape or not np.array_equal(original["offsets"][:], staged["offsets"][:]):
            raise ValueError("VDS shape/offset mismatch")
        original_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in original["ids"][:]]
        staged_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in staged["ids"][:]]
        if original_ids != staged_ids:
            raise ValueError("VDS IDs changed")
        sample = sorted({0, shape[0]//2, shape[0]-1, *[start for start, _ in mappings],
                         *[stop-1 for _, stop in mappings]})
        for row in sample:
            if not np.array_equal(original["vectors"][row], staged["vectors"][row]):
                raise ValueError(f"VDS row {row} differs")
    receipt = {"schema": "enzymecage_local_residue_staging_v1", "source_vds": str(SOURCE),
               "source_vds_sha256": digest(SOURCE), "staged_vds": str(vds),
               "staged_vds_sha256": digest(vds), "shards": records,
               "ids": len(ids), "residue_rows": shape[0], "sample_rows_equal": sample,
               "source_code_sha256": digest(Path(__file__)), "model_input_values_changed": False}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"complete": True, "vds": str(vds), "shards": len(records)}), flush=True)


if __name__ == "__main__":
    main()
