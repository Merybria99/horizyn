#!/usr/bin/env python3
"""Encode both endpoints independently with a frozen generalization bundle.

This entry point reads feature arrays and IDs, never association/activity
labels. Its score matrix can be evaluated separately on an immutable panel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import h5py
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horizyn.generalization_retrieval import GeneralizationDualEncoder, sha256, canonical_dot, checked_artifact
from generalization_full_graph import atomic_json


def validate_input_receipt(args, specification):
    """Bind every row-aligned inference array to one authenticated export."""
    path = getattr(args, "input_receipt", None) or args.catalog.parent / "feature_bundle_receipt.json"
    if not path.is_file():
        raise ValueError(f"A feature bundle provenance receipt is required: {path}")
    receipt = json.loads(path.read_text())
    if receipt.get("schema") != "generalization_feature_bundle_receipt_v1":
        raise ValueError("Unrecognized feature bundle receipt schema")
    for key in ("base", "catalog", "protein_means", "reaction_features"):
        actual = checked_artifact(receipt["inputs"][key], path.parent)
        if actual.resolve() != getattr(args, key).resolve():
            raise ValueError(f"Feature receipt points to a different {key} artifact")
    checkpoint = checked_artifact(receipt["checkpoint"], path.parent)
    expected = specification["base_checkpoint"]
    if checkpoint.resolve() != Path(expected["path"]).resolve():
        raise ValueError("Feature export and model bundle base checkpoints disagree")
    stat = checkpoint.stat()
    for key, value in (("size", stat.st_size), ("mtime_ns", stat.st_mtime_ns),
                       ("sha256", receipt["checkpoint"]["sha256"])):
        if key in expected and expected[key] != value:
            raise ValueError("Feature export base checkpoint identity changed")
    if receipt.get("freeze_sha256") and receipt["freeze_sha256"] != specification["frozen_recipe"]["sha256"]:
        raise ValueError("Feature receipt belongs to another frozen recipe")
    for record in receipt.get("source_receipts", []):
        checked_artifact(record, path.parent)
    return dict(path=str(path.resolve()), sha256=sha256(path))


def run(args):
    if args.batch_size <= 0:
        raise ValueError("Prediction batch size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "complete.json").exists():
        raise ValueError("Prediction output already complete; choose a fresh directory")
    started = time.monotonic()
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(42)
    device = torch.device(args.device)
    model, specification = GeneralizationDualEncoder.from_bundle(args.bundle, device)
    input_receipt = validate_input_receipt(args, specification)
    catalog = json.loads(args.catalog.read_text())
    reaction_ids = catalog["reactions"] if "reactions" in catalog else catalog["query_ids"]
    if len(set(catalog["proteins"])) != len(catalog["proteins"]) or len(set(reaction_ids)) != len(reaction_ids):
        raise ValueError("Prediction catalog IDs must be unique")
    with np.load(args.base) as source:
        base_e = torch.tensor(source["proteins"], device=device)
        base_r = torch.tensor(source[args.reaction_key], device=device)
    with h5py.File(args.protein_means, "r") as source:
        proteins = list(source["ids"].asstr()[:])
        if proteins != catalog["proteins"]:
            raise ValueError("Protein means and catalog ID order disagree")
        if "complete" in source and not source["complete"][:].all():
            raise ValueError("Incomplete protein mean cache")
        means = torch.tensor(source["vectors"][:], device=device)
    blocks, masks = {}, {}
    with np.load(args.reaction_features) as source:
        for key in model.modalities:
            blocks[key] = torch.tensor(source[key], device=device)
            masks[key] = torch.tensor(source[key + "_mask"], device=device)
    if len(base_e) != len(proteins) or len(base_r) != len(reaction_ids):
        raise ValueError("Base features and catalog dimensions disagree")
    if any(len(value) != len(base_r) for value in blocks.values()) or any(value.shape != (len(base_r),) for value in masks.values()):
        raise ValueError("Reaction features and base embeddings disagree")
    with torch.inference_mode():
        enzymes = model.encode_enzymes(base_e, means, args.batch_size)
        reactions = model.encode_reactions(base_r, blocks, masks)
        scores = canonical_dot(reactions, enzymes)
        reference = canonical_dot(F.normalize(base_r, dim=1), F.normalize(base_e, dim=1))
    if not torch.isfinite(scores).all() or not torch.isfinite(reference).all():
        raise ValueError("Nonfinite retrieval scores")
    np.savez(args.output / "scores.npz", selected=scores.cpu().numpy(), baseline=reference.cpu().numpy())
    if args.save_embeddings:
        np.savez_compressed(args.output / "encoded_endpoints.npz",
                            proteins=enzymes.cpu().numpy(), reactions=reactions.cpu().numpy())
    shutil.copyfile(args.catalog, args.output / "catalog.json")
    manifest = dict(schema="generalization_predictions_v1", labels_used=False,
        bundle=dict(path=str(args.bundle.resolve()), sha256=sha256(args.bundle)),
        inputs={key: dict(path=str(getattr(args, key).resolve()), sha256=sha256(getattr(args, key)))
                for key in ("base", "catalog", "protein_means", "reaction_features")},
        input_receipt=input_receipt, reaction_key=args.reaction_key, shape=list(scores.shape), embedding_dimension=enzymes.shape[1],
        elapsed_seconds=time.monotonic() - started, score="independent endpoint dot product; FP64 accumulation, FP32 output",
        torch_version=torch.__version__, device=str(device), matmul_precision="highest",
        batch_size=args.batch_size, source_sha256=sha256(__file__),
        output_sha256=sha256(args.output / "scores.npz"))
    atomic_json(args.output / "complete.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "base", "catalog", "protein-means", "reaction-features", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--reaction-key", default="reactions")
    p.add_argument("--input-receipt", type=Path, help="Defaults to feature_bundle_receipt.json beside catalog")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--save-embeddings", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
