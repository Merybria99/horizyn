#!/usr/bin/env python3
"""Package a frozen recipe using training associations/features exclusively."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horizyn.generalization_retrieval import (sha256, checked_artifact,
    validate_frozen_residual, validate_anchor_recipe)
from horizyn.semantic_anchors import (fit_center, centered_unit, reaction_features,
                                     make_protein_reaction_map)
from generalization_full_graph import atomic_json


def build(args):
    if not args.freeze.is_file():
        raise ValueError("A frozen recipe manifest is required")
    frozen = json.loads(args.freeze.read_text())
    if not frozen.get("frozen_before_held_out_evaluation"):
        raise ValueError("Recipe manifest has not been frozen")
    if not (args.features / "complete.json").exists():
        raise ValueError("Incomplete source features")
    manifest = json.loads((args.features / "manifest.json").read_text())
    feature_hash = sha256(args.features / "manifest.json")
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    receipt = json.loads((args.features / "complete.json").read_text())
    if receipt.get("manifest_signature") != signature or manifest.get("test_used") is not False:
        raise ValueError("Source feature receipt/provenance does not match training manifest")
    validate_anchor_recipe(vars(args), frozen)
    for checkpoint_path in args.residual:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validate_frozen_residual(checkpoint_path, state, frozen, feature_hash)
    compact_path = checked_artifact(manifest["inputs"]["compact_manifest"])
    cache_manifest = json.loads(compact_path.read_text())
    checkpoint = dict(cache_manifest["signature"]["inputs"]["checkpoint"])
    checkpoint_path = Path(checkpoint["path"])
    if any(checkpoint.get(key) != value for key, value in (
            ("size", checkpoint_path.stat().st_size), ("mtime_ns", checkpoint_path.stat().st_mtime_ns))):
        raise ValueError("Base checkpoint identity changed since feature export")
    checkpoint["sha256"] = sha256(checkpoint_path)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "bundle.json").exists():
        raise ValueError("Bundles are immutable; choose a fresh output directory")
    device = torch.device(args.device)
    torch.set_num_threads(8)
    catalog = json.loads((args.features / "catalog.json").read_text())
    if any(len(catalog[key]) != len(set(catalog[key])) for key in ("proteins", "reactions")):
        raise ValueError("Training catalog IDs must be unique")
    with np.load(args.features / "pairs.npz") as source:
        train = source["train"]
    if train.ndim != 2 or train.shape[1] != 2 or not np.issubdtype(train.dtype, np.integer) or not len(train):
        raise ValueError("Training pairs must be a nonempty integer Nx2 array")
    if (train < 0).any() or (train[:, 0] >= len(catalog["reactions"])).any() or (train[:, 1] >= len(catalog["proteins"])).any():
        raise ValueError("Training pair indices exceed their catalogs")
    tr, te = np.unique(train[:, 0]), np.unique(train[:, 1])
    with h5py.File(args.features / "protein_mean.h5", "r") as source:
        if list(source["ids"].asstr()[:]) != catalog["proteins"]:
            raise ValueError("Protein means and training catalog ID order disagree")
        if source.attrs.get("manifest_signature") != signature:
            raise ValueError("Protein mean cache belongs to another feature manifest")
        if not source["complete"][:].all():
            raise ValueError("Incomplete protein means")
        proteins = torch.tensor(source["vectors"][:], device=device)
    pc = fit_center(proteins, te)
    tp = centered_unit(proteins[te], pc)
    del proteins
    blocks, masks = {}, {}
    with np.load(args.features / "reaction_features.npz") as source:
        for key in args.modalities:
            blocks[key] = torch.tensor(source[key], device=device)
            masks[key] = torch.tensor(source[key + "_mask"], device=device)
            if len(blocks[key]) != len(catalog["reactions"]) or masks[key].shape != (len(catalog["reactions"]),):
                raise ValueError("Reaction feature dimensions disagree with training catalog")
    centers = {key: fit_center(blocks[key], tr, masks[key]) for key in args.modalities}
    rx = reaction_features(blocks, centers, masks, args.modalities)[tr]
    rmap, emap = ({int(g): i for i, g in enumerate(values)} for values in (tr, te))
    er = torch.tensor([rmap[int(x)] for x in train[:, 0]], device=device)
    ee = torch.tensor([emap[int(x)] for x in train[:, 1]], device=device)
    adjacency = make_protein_reaction_map(er, ee, len(te))
    state = dict(protein_center=pc.cpu(), train_proteins=tp.cpu(), train_reactions=rx.cpu(),
                 adjacency=adjacency.cpu(), reaction_centers={k: v.cpu() for k, v in centers.items()},
                 modalities=args.modalities, enzyme_temperature=args.enzyme_temperature,
                 reaction_temperature=args.reaction_temperature, protein_neighbors=args.protein_neighbors,
                 reaction_neighbors=args.reaction_neighbors,
                 train_protein_ids=[catalog["proteins"][i] for i in te],
                 train_reaction_ids=[catalog["reactions"][i] for i in tr],
                 feature_manifest_sha256=feature_hash)
    anchors = args.output / "anchors.pt"
    torch.save(state, anchors)
    result = dict(schema="generalization_dual_encoder_bundle_v1", fit_uses_training_only=True,
        anchor_alpha=args.alpha, anchors=dict(path="anchors.pt", sha256=sha256(anchors)),
        residuals=[dict(path=str(p.resolve()), sha256=sha256(p)) for p in args.residual],
        frozen_recipe=dict(path=str(args.freeze.resolve()), sha256=sha256(args.freeze)),
        base_checkpoint=checkpoint, feature_manifest_sha256=state["feature_manifest_sha256"],
        latent_dimension=(len(tr) if args.alpha else 0) + 512 * max(1, len(args.residual)),
        score="dot product; endpoints independently encoded and already normalized",
        input_policy="F3 participant self-reaction; train-fitted chemistry; mean stored ProtT5 residues",
        source_sha256=sha256(__file__))
    atomic_json(args.output / "bundle.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--residual", type=Path, action="append", default=[])
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--enzyme-temperature", type=float, default=.03)
    p.add_argument("--reaction-temperature", type=float, default=.03)
    p.add_argument("--protein-neighbors", type=int, default=32)
    p.add_argument("--reaction-neighbors", type=int, default=16)
    p.add_argument("--modalities", nargs="+", default=["t5v2", "unimol2", "chiro", "chemistry"])
    p.add_argument("--device", default="cpu")
    build(p.parse_args())


if __name__ == "__main__":
    main()
