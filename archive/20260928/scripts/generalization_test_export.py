#!/usr/bin/env python3
"""Export official held-out features only after recording a frozen recipe.

This script never fits a transform or chooses a model. --freeze must name an
existing, nonempty JSON object. Its SHA256 is durably recorded before any test
pair file, feature cache, or test configuration is opened.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import yaml

from scripts.generalization_export import (
    aligned_h5, atomic_json, atomic_npz, digest, export_means, h5_ids,
    identity, reaction_block, read_pairs, resolve, strings,
)

EXPECTED = {"reaction_smi": (386, 14688), "enzyme_smi": (1573, 8734), "time": (2634, 12277)}


def record_freeze(freeze: Path, output: Path, split: str):
    """This must be the first operation capable of reaching experiment inputs."""
    content = freeze.read_bytes()
    recipe = json.loads(content)
    if not isinstance(recipe, dict) or not recipe:
        raise ValueError("The frozen recipe must be a nonempty JSON object")
    receipt = dict(schema="heldout_export_freeze_receipt_v1", split=split,
                   freeze_path=str(freeze.resolve()),
                   freeze_sha256=hashlib.sha256(content).hexdigest(),
                   exporter_sha256=digest(__file__))
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "freeze_receipt.json"
    if destination.exists() and json.loads(destination.read_text()) != receipt:
        raise ValueError("Output belongs to another frozen recipe or exporter")
    atomic_json(destination, receipt)
    # Flush the receipt before opening any test artifact, including on NFS.
    with destination.open("rb") as handle:
        os.fsync(handle.fileno())
    return receipt


def checked_reuse(path, residue_identity):
    path = path.resolve()
    manifest_path = path.parent / "manifest.json"
    receipt_path = path.parent / "complete.json"
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if manifest["inputs"]["protein_residues"] != residue_identity:
        raise ValueError(f"Mean cache uses a different residue source: {path}")
    if receipt["manifest_signature"] != signature:
        raise ValueError(f"Invalid mean completion receipt: {path}")
    with h5py.File(path) as handle:
        if handle.attrs.get("manifest_signature") != signature or not handle["complete"][:].all():
            raise ValueError(f"Incomplete mean cache: {path}")
        if handle.attrs.get("accumulation_dtype") != "float32" or handle["vectors"].shape[1] != 1024:
            raise ValueError(f"Wrong mean precision/dimension: {path}")
    return dict(file=identity(path, True), manifest=identity(manifest_path, True), receipt=identity(receipt_path, True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-mean-cache", type=Path, action="append", default=[])
    parser.add_argument("--block-residues", type=int, default=65536)
    args = parser.parse_args()
    if args.block_residues < 1022:
        parser.error("--block-residues must be at least 1022")
    args.output = resolve(args.output)
    freeze = record_freeze(args.freeze, args.output, args.split)
    # Every access to test information is below this recorded authorization.
    config_path = ROOT / f"runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry/{args.split}/test.yaml"
    data = yaml.safe_load(config_path.read_text())["data"]
    pair_path = resolve(data["test_pairs_path"])
    reaction_path = resolve(data["test_reactions_path"])
    candidates_path = resolve(data["validation_retrieval_candidate_ids_path"])
    cache = ROOT / f"runs/biological_residual_{args.split}/cache/reactzyme_test"
    cached = json.loads((cache / "manifest.json").read_text())
    signature = cached["signature"]["inputs"]
    if cached["dtype"] != "float32" or cached["signature"]["precision"] != "32":
        raise ValueError("Held-out F3 must use exact FP32 inference and storage")
    checkpoint = signature["checkpoint"]
    expected_checkpoint = ROOT / f"runs/reactzyme_reaction_features_v1/checkpoints/{args.split}/F3_set_chemistry/protein-pooling-epoch=29.ckpt"
    if Path(checkpoint["path"]).resolve() != expected_checkpoint.resolve() or identity(expected_checkpoint) != checkpoint:
        raise ValueError("Unexpected or changed frozen F3 checkpoint")
    inputs = dict(config=identity(config_path, True), pairs=identity(pair_path, True),
                  reactions=identity(reaction_path, True), candidates=identity(candidates_path, True),
                  cache_manifest=identity(cache / "manifest.json", True),
                  checkpoint=identity(expected_checkpoint, True),
                  protein_residues=identity(data["protein_residue_embeds_path"]))
    inputs["export_helpers"] = identity(ROOT / "scripts/generalization_export.py", True)
    if signature["validation_pairs"]["sha256"] != inputs["pairs"]["sha256"]:
        raise ValueError("F3 cache was generated from different held-out associations")
    if signature["validation_reactions"]["sha256"] != inputs["reactions"]["sha256"]:
        raise ValueError("F3 cache was generated from a different reaction catalog")
    if signature["protein_residues"] != inputs["protein_residues"]:
        raise ValueError("F3 and raw means use different residue caches")
    paths = {}
    for modality in ("t5v2", "unimol2", "chiro", "chemistry"):
        ending = "vectors_path" if modality == "chemistry" else "embeds_path"
        key = f"reaction_{modality}_{ending}"
        paths[modality] = resolve(data[key])
        inputs[modality] = identity(paths[modality], True)
        recorded = signature["reaction_feature_inputs"][key]
        if any(recorded[field] != inputs[modality][field] for field in ("path", "size", "mtime_ns")):
            raise ValueError(f"Raw/frozen feature source mismatch: {modality}")
    for name in ("enzyme_base.h5", "validation_reaction_base.h5"):
        inputs[name] = identity(cache / name, True)
    schema_path = paths["chemistry"].parent / "schema.json"
    chemistry_schema = json.loads(schema_path.read_text())
    if chemistry_schema.get("fit_split") != "train":
        raise ValueError("Chemistry schema must be fitted on training only")
    inputs["chemistry_schema"] = identity(schema_path, True)
    reuse = [(resolve(path), checked_reuse(resolve(path), inputs["protein_residues"])) for path in args.reuse_mean_cache]
    manifest = dict(schema="generalization_official_test_features_v1", split=args.split,
                    freeze=freeze, inputs=inputs, reuse_mean_caches=[value for _, value in reuse],
                    test_used=True, selection_allowed=False,
                    notes=["Raw features and per-protein means are fit-free.",
                           "Giant residue source uses strict path/size/mtime identity; all compact inputs use SHA256.",
                           "Missing optional molecular modalities retain their rows and explicit masks."])
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Held-out inputs changed; use a new output directory")
    atomic_json(manifest_path, manifest)
    (args.output / "export_source.py").write_text(Path(__file__).read_text())
    (args.output / "export_helpers.py").write_text((ROOT / "scripts/generalization_export.py").read_text())
    pairs = read_pairs(pair_path)
    reactions = sorted({q for q, _ in pairs})
    proteins = sorted({p for _, p in pairs})
    supplied = [line.strip() for line in candidates_path.read_text().splitlines() if line.strip()]
    if len(supplied) != len(set(supplied)) or set(supplied) != set(proteins):
        raise ValueError("Candidate IDs must equal the complete held-out positive enzyme universe")
    with reaction_path.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(reactions) or {row["reaction_id"] for row in rows} != set(reactions):
        raise ValueError("Reaction catalog must equal the complete held-out query universe")
    if (len(reactions), len(proteins)) != EXPECTED[args.split]:
        raise ValueError(f"Official matrix dimensions changed: {len(reactions)} x {len(proteins)}")
    catalog = dict(split=args.split, proteins=proteins, reactions=reactions,
                   test_reactions=reactions, test_candidates=proteins,
                   counts=dict(test_pairs=len(pairs), proteins=len(proteins), reactions=len(reactions)))
    atomic_json(args.output / "catalog.json", catalog)
    qi, pi = ({key: i for i, key in enumerate(keys)} for keys in (reactions, proteins))
    atomic_npz(args.output / "pairs.npz", test=np.asarray([(qi[q], pi[p]) for q, p in pairs], dtype=np.int64))
    base_p = aligned_h5(cache / "enzyme_base.h5", proteins)
    base_q = aligned_h5(cache / "validation_reaction_base.h5", reactions, "_f")
    if base_p.shape != (len(proteins), 512) or base_q.shape != (len(reactions), 512):
        raise ValueError("Unexpected F3 embedding dimensions")
    atomic_npz(args.output / "f3_features.npz", proteins=base_p, reactions=base_q)
    raw = {}
    for modality, dimension in (("t5v2", 768), ("unimol2", 768), ("chiro", 256)):
        vectors, mask = reaction_block(paths[modality], reactions, modality != "t5v2")
        if vectors.shape != (len(reactions), dimension) or (modality == "t5v2" and not mask.all()):
            raise ValueError(f"Invalid mandatory feature coverage/dimensions: {modality}")
        raw[modality], raw[modality + "_mask"] = vectors, mask
    with np.load(paths["chemistry"], allow_pickle=True) as source:
        ids = strings(source["ids"])
        if len(ids) != len(set(ids)) or set(ids) != set(reactions):
            raise ValueError("Chemistry must retain the complete reaction catalog")
        lookup = {key: i for i, key in enumerate(ids)}
        order = [lookup[key] for key in reactions]
        raw["chemistry"] = np.asarray(source["vectors"][order], np.float32)
        raw["chemistry_mask"] = source["mask"][order].astype(bool)
    if raw["chemistry"].shape != (len(reactions), 617) or not np.isfinite(raw["chemistry"]).all():
        raise ValueError("Invalid chemistry dimensions or values")
    atomic_npz(args.output / "reaction_features.npz", **raw)
    manifest_signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    means_path = args.output / "protein_mean.h5"
    with h5py.File(means_path, "a") as out:
        if "ids" not in out:
            out["ids"] = np.asarray(proteins, dtype=h5py.string_dtype())
            out.create_dataset("vectors", shape=(len(proteins), 1024), dtype=np.float32)
            out["complete"] = np.zeros(len(proteins), bool)
            out.attrs["manifest_signature"] = manifest_signature
            out.attrs["accumulation_dtype"] = "float32"
        if h5_ids(out) != proteins or out.attrs["manifest_signature"] != manifest_signature:
            raise ValueError("Protein means resume provenance mismatch")
        completed = out["complete"][:]
        for reuse_path, _ in reuse:
            with h5py.File(reuse_path) as previous:
                index = {key: i for i, key in enumerate(h5_ids(previous))}
                rows = np.asarray([i for i, key in enumerate(proteins) if not completed[i] and key in index], np.int64)
                if not len(rows):
                    continue
                values = previous["vectors"][:][[index[proteins[i]] for i in rows]]
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite reused means")
                for offset in range(0, len(rows), 512):
                    out["vectors"][rows[offset:offset + 512]] = values[offset:offset + 512]
                out.flush()
                completed[rows] = True
                out["complete"][:] = completed
                out.flush()
                print(f"Reused {len(rows)} means from {reuse_path}", flush=True)
    export_means(SimpleNamespace(output=args.output, block_residues=args.block_residues,
                               reuse_mean_cache=None, write_completion_receipt=False), catalog, manifest)
    if digest(args.freeze) != freeze["freeze_sha256"]:
        raise ValueError("Frozen recipe changed during held-out export")
    atomic_json(args.output / "complete.json", dict(manifest_signature=manifest_signature,
        freeze_sha256=freeze["freeze_sha256"], split=args.split, counts=catalog["counts"],
        test_used=True, selection_allowed=False))
    print(json.dumps(catalog["counts"]), flush=True)


if __name__ == "__main__":
    main()
