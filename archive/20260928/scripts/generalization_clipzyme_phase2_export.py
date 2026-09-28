#!/usr/bin/env python3
"""Export EnzymeMap-only inputs for the frozen-F3 phase-2 architecture.

The raw stage runs independently of the F3 checkpoint. The F3 stage binds the
same train/validation inputs to a particular target-trained checkpoint. Test
associations are never opened here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from horizyn.benchmarks.retrieval import (BenchmarkTask, build_reaction_inputs,
    encode_reactions, encode_residue_targets)
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from scripts.generalization_clipzyme_f3_screen import model_from_checkpoint
from scripts.generalization_export import (atomic_json, atomic_npz, h5_ids,
    reaction_block, read_pairs)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def inputs(catalog: Path, config: Path) -> dict:
    result = {"config": config, "train_pairs": catalog / "train_pairs.csv",
              "validation_pairs": catalog / "validation_pairs.csv",
              "residues": catalog / "features/proteins_prott5_residue.h5"}
    for name in ("reactiont5v2", "unimol2", "chiro"):
        result[name] = catalog / "features" / f"{name}.forward.h5"
    for split in ("train", "validation"):
        result[f"{split}_chemistry"] = catalog / "chemistry" / f"{split}_reaction_set_features.npz"
        result[f"{split}_reactions"] = catalog / f"{split}_rxns.csv"
    result["validation_candidates"] = catalog / "validation_candidate_ids.txt"
    return result


def raw(args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    source = inputs(args.catalog, args.config)
    train = read_pairs(source["train_pairs"])
    val = read_pairs(source["validation_pairs"])
    if set(train) & set(val):
        raise ValueError("EnzymeMap train/validation edges overlap")
    candidates = sorted(set(source["validation_candidates"].read_text().splitlines()))
    if not {p for _, p in val} <= set(candidates):
        raise ValueError("Validation positive enzyme missing from candidate pool")
    proteins = sorted(set(p for _, p in train) | set(candidates))
    reactions = sorted(set(q for q, _ in train + val))
    train_q = sorted(set(q for q, _ in train))
    val_q = sorted(set(q for q, _ in val))
    catalog = dict(proteins=proteins, reactions=reactions, train_reactions=train_q,
                   validation_reactions=val_q, validation_candidates=candidates,
                   train_proteins=sorted(set(p for _, p in train)),
                   counts=dict(train_pairs=len(train), validation_pairs=len(val),
                               train_reactions=len(train_q), validation_reactions=len(val_q),
                               proteins=len(proteins), validation_candidates=len(candidates)))
    atomic_json(args.output / "catalog.json", catalog)
    pi = {key: i for i, key in enumerate(proteins)}
    qi = {key: i for i, key in enumerate(reactions)}
    atomic_npz(args.output / "pairs.npz",
               train=np.asarray([(qi[q], pi[p]) for q, p in train], np.int64),
               validation=np.asarray([(qi[q], pi[p]) for q, p in val], np.int64))
    blocks = {}
    for name, molecular in (("t5v2", False), ("unimol2", True), ("chiro", True)):
        block, mask = reaction_block(source[{"t5v2": "reactiont5v2"}.get(name, name)],
                                     reactions, molecular)
        blocks[name], blocks[name + "_mask"] = block, mask
    chemistry = None
    chemistry_mask = np.zeros(len(reactions), bool)
    for split, ids in (("train", train_q), ("validation", val_q)):
        with np.load(source[f"{split}_chemistry"], allow_pickle=True) as data:
            index = {str(key): i for i, key in enumerate(data["ids"])}
            values = data["vectors"]
            present = data["mask"] if "mask" in data else np.ones(len(index), bool)
            if chemistry is None:
                chemistry = np.zeros((len(reactions), values.shape[1]), np.float32)
            for key in ids:
                if key in index:
                    i, j = qi[key], index[key]
                    if chemistry_mask[i] and present[j] and not np.allclose(chemistry[i], values[j]):
                        raise ValueError(f"Overlapping reaction chemistry disagrees: {key}")
                    chemistry[i], chemistry_mask[i] = values[j], present[j]
    blocks["chemistry"], blocks["chemistry_mask"] = chemistry, chemistry_mask
    atomic_npz(args.output / "reaction_features.npz", **blocks)
    manifest = dict(schema="clipzyme_enzymemap_phase2_features_v1",
                    sources={key: dict(path=str(path.resolve()), sha256=digest(path))
                             for key, path in source.items()},
                    train_only_fit=True, validation_labels_only_for_selection=True,
                    test_labels_read=False, source_sha256=digest(Path(__file__)))
    atomic_json(args.output / "manifest.json", manifest)
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    with h5py.File(source["residues"], "r") as h, h5py.File(args.output / "protein_mean.h5", "w") as out:
        physical = {key: i for i, key in enumerate(h5_ids(h))}
        if not set(proteins) <= physical.keys():
            raise ValueError("Missing protein residues")
        offsets = h["offsets"][:]
        out.create_dataset("ids", data=np.asarray(proteins, dtype=object), dtype=h5py.string_dtype())
        means = out.create_dataset("vectors", shape=(len(proteins), 1024), dtype="f4")
        done = out.create_dataset("complete", shape=(len(proteins),), dtype="?")
        out.attrs["manifest_signature"] = signature
        out.attrs["accumulation_dtype"] = "float32"
        for n, key in enumerate(proteins):
            row = physical[key]
            lo, hi = int(offsets[row]), int(offsets[row + 1])
            means[n] = h["vectors"][lo:hi].mean(0, dtype=np.float32)
            done[n] = True
            if n % 1000 == 0:
                print(json.dumps({"protein_means": n + 1, "total": len(proteins)}), flush=True)
    print(json.dumps(catalog["counts"]), flush=True)


def f3(args) -> None:
    catalog = json.loads((args.output / "catalog.json").read_text())
    model, config = model_from_checkpoint(args.config, args.checkpoint, args.device)
    torch.set_num_threads(8)
    residue_path = args.residue_cache or args.catalog / "features/proteins_prott5_residue.h5"
    residue = ResidueEmbedDataset(str(residue_path),
                                  max_tokens=config.data.max_protein_tokens,
                                  truncation=config.data.protein_truncation)
    try:
        with torch.inference_mode():
            proteins = encode_residue_targets(model, residue, catalog["proteins"],
                                              args.device, args.batch_size, False).float().cpu().numpy()
    finally:
        residue.close()
    reactions = np.zeros((len(catalog["reactions"]), 512), np.float32)
    ridx = {key: i for i, key in enumerate(catalog["reactions"])}
    for split in ("train", "validation"):
        config.data.reaction_chemistry_vectors_path = str(
            args.catalog / "chemistry" / f"{split}_reaction_set_features.npz")
        task = BenchmarkTask(name=f"clipzyme_phase2_{split}", task_type="retrieval",
            dataset="EnzymeMap", task_label="rule_split", split=split,
            pairs=args.catalog / f"{split}_pairs.csv",
            reactions=args.catalog / f"{split}_rxns.csv",
            reaction_model_embeds_h5=args.catalog / "features/reactiont5v2.h5",
            reaction_unimol2_embeds_h5=args.catalog / "features/unimol2.h5",
            reaction_chiro_embeds_h5=args.catalog / "features/chiro.h5")
        reaction_inputs = build_reaction_inputs(task, config)
        keys = catalog[f"{split}_reactions"]
        if not set(keys) <= set(reaction_inputs.keys):
            raise ValueError(f"Missing {split} reaction inputs")
        with torch.inference_mode():
            vectors = encode_reactions(model, reaction_inputs, keys, args.device,
                                       max(8, min(args.batch_size, 128))).float().cpu().numpy()
        reactions[[ridx[key] for key in keys]] = vectors
    if not np.isfinite(proteins).all() or not np.isfinite(reactions).all():
        raise ValueError("Nonfinite F3 embeddings")
    train_reactions = reactions[[ridx[key] for key in catalog["train_reactions"]]]
    atomic_npz(args.output / "f3_features.npz", proteins=proteins, reactions=reactions,
               train_reactions=train_reactions)
    manifest = json.loads((args.output / "manifest.json").read_text())
    manifest["checkpoint"] = dict(path=str(args.checkpoint.resolve()), sha256=digest(args.checkpoint))
    manifest["f3_export_config"] = dict(path=str(args.config.resolve()), sha256=digest(args.config))
    manifest["f3_residue_cache"] = dict(path=str(residue_path.resolve()), sha256=digest(residue_path))
    atomic_json(args.output / "manifest.json", manifest)
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    with h5py.File(args.output / "protein_mean.h5", "r+") as h:
        h.attrs["manifest_signature"] = signature
    atomic_json(args.output / "complete.json", dict(manifest_signature=signature,
                checkpoint_sha256=manifest["checkpoint"]["sha256"], test_used=False,
                files={name: digest(args.output / name) for name in
                       ("catalog.json", "pairs.npz", "reaction_features.npz",
                        "protein_mean.h5", "f3_features.npz")}))
    print(json.dumps({"f3_proteins": len(proteins),
                      "f3_reactions": len(reactions)}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("raw", "f3"), required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--residue-cache", type=Path)
    args = p.parse_args()
    if args.stage == "raw":
        raw(args)
    else:
        if args.checkpoint is None:
            p.error("--checkpoint is required for the F3 stage")
        f3(args)


if __name__ == "__main__":
    main()
