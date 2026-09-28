#!/usr/bin/env python3
"""Expose forward-only extracted features using the legacy F3 ``_f`` ID convention.

Vectors are copied verbatim to separate HDF5 files; source caches stay intact.
No reverse embeddings are fabricated. Chemistry NPZs already support base IDs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def align(source: Path, target: Path, expected: set[str]) -> int:
    checksum = sha256(source)
    with h5py.File(source, "r") as handle:
        ids = handle["ids"].asstr()[:].tolist()
        if not ids or len(set(ids)) != len(ids) or not set(ids).issubset(expected):
            raise ValueError(f"Unexpected or duplicate source reaction IDs: {source}")
        aligned = [key + "_f" for key in ids]
        if target.exists():
            with h5py.File(target, "r") as previous:
                if (previous.attrs.get("source_sha256") != checksum
                        or previous["ids"].asstr()[:].tolist() != aligned):
                    raise ValueError(f"Stale aligned cache; inspect before replacing: {target}")
            return len(ids)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", suffix=".tmp") as tmp:
            with h5py.File(tmp.name, "w") as output:
                for name in handle:
                    if name != "ids":
                        handle.copy(name, output)
                output.create_dataset("ids", data=np.asarray(aligned, dtype=object),
                                      dtype=h5py.string_dtype("utf-8"))
                output.attrs.update(dict(handle.attrs))
                output.attrs["source_sha256"] = checksum
                output.attrs["reaction_id_policy"] = "base_id + _f; forward only"
            # Publish only a closed, fully written file. Keep tempfile cleanup valid.
            import os
            os.link(tmp.name, target)
    return len(ids)


def check_loader(config_path: Path) -> dict:
    from horizyn.config import load_config
    from horizyn.reaction_conditioned_data_module import ReactionConditionedDataModule
    from horizyn.training_options import reaction_data_module_kwargs

    config = load_config(str(config_path))
    if config.data.reaction_direction_mode != "forward_only":
        raise ValueError("These caches contain forward reactions only")
    kwargs = reaction_data_module_kwargs(config)
    kwargs.update(train_batch_size=2, retrieval_batch_size=2, validation_retrieval_batch_size=2,
                  num_workers=0, persistent_workers=False)
    module = ReactionConditionedDataModule(**kwargs)
    module.setup("fit")
    report = {}
    for split, attribute in (("train", "_train_query_data"), ("validation", "_val_query_data_raw")):
        # Check every requested ID, not just nonempty intersection of feature stores.
        with Path(config.data[f"{split}_reactions_path"]).open(newline="") as handle:
            expected = {row["reaction_id"] + "_f" for row in csv.DictReader(handle)}
        dataset = getattr(module, attribute)
        missing = expected - set(dataset.keys)
        if missing:
            raise ValueError(f"{split}: {len(missing)} forward reactions missing, e.g. {sorted(missing)[:3]}")
        report[split + "_reactions"] = len(expected)
        with Path(config.data[f"{split}_pairs_path"]).open(newline="") as handle:
            pair_count = sum(1 for _ in csv.DictReader(handle))
        pairs = module._train_data if split == "train" else module._val_data
        if len(pairs) != pair_count:
            raise ValueError(f"{split}: loader retained {len(pairs)} of {pair_count} pairs")
        report[split + "_pairs"] = pair_count
    next(iter(module.train_dataloader()))
    for loader in module.val_dataloader():
        next(iter(loader))
    report["loader_batches"] = "train and validation passed"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--check-loader", action="store_true")
    args = parser.parse_args()
    expected = set()
    for split in ("train", "validation", "test"):
        path = args.run_root / f"data/normalized/{split}_rxns.csv"
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            expected.update(row["reaction_id"] for row in csv.DictReader(handle))
    features = args.run_root / "features"
    report = {name: align(features / f"{name}.h5", features / f"{name}.forward.h5", expected)
              for name in ("reactiont5v2", "unimol2", "chiro")}
    if report["reactiont5v2"] != len(expected):
        raise ValueError("ReactionT5 must cover every reaction")
    if args.check_loader:
        report.update(check_loader(args.run_root / "configs/train.yaml"))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
