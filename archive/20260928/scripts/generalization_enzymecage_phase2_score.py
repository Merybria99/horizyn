#!/usr/bin/env python3
"""Score frozen EnzymeCAGE-data phase 2 on validation or an external panel.

This step reads inputs and identities only. Evaluation labels are read later.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from horizyn.generalization_density import DensityGatedEncoder
from horizyn.generalization_phase2 import ComposedPhase2Encoder
from horizyn.generalization_retrieval import canonical_dot
from horizyn.semantic_anchors import row_unit
from horizyn.semantic_smooth import SmoothAnchorDualEncoder


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def checked(record: dict) -> Path:
    path = Path(record["path"]).resolve()
    if digest(path) != record["sha256"]:
        raise ValueError(f"Frozen artifact changed: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--protein-means", type=Path, required=True)
    parser.add_argument("--reaction-features", type=Path, required=True)
    parser.add_argument("--scope", choices=("validation", "external"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 17, 73), default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    freeze = json.loads(args.freeze.read_text())
    if freeze["schema"] != "enzymecage_phase2_target_freeze_v1" or freeze["alpha"] != 0.25:
        raise ValueError("Unexpected target phase-2 recipe")
    for key in ("feature_manifest", "smooth_dictionary", "base_checkpoint", "base_config",
                "architecture_lock", "historical_phase2_recipe", "source"):
        checked(freeze[key])
    seed = str(args.seed)
    checked(freeze["residuals"][seed])
    density_path = checked(freeze["density_bundles"][seed])
    catalog = json.loads(args.catalog.read_text())
    proteins, reactions = catalog["proteins"], catalog["reactions"]
    if len(proteins) != len(set(proteins)) or len(reactions) != len(set(reactions)):
        raise ValueError("Duplicate catalog ID")
    if args.scope == "validation":
        if args.catalog.resolve() != Path(freeze["feature_manifest"]["path"]).parent / "catalog.json":
            raise ValueError("Validation catalog does not belong to target training export")
        protein_ids = catalog["validation_candidates"]
        reaction_ids = catalog["validation_reactions"]
    else:
        protein_ids, reaction_ids = proteins, reactions
    protein_index = {key: i for i, key in enumerate(proteins)}
    reaction_index = {key: i for i, key in enumerate(reactions)}
    pi = np.asarray([protein_index[key] for key in protein_ids], dtype=np.int64)
    qi = np.asarray([reaction_index[key] for key in reaction_ids], dtype=np.int64)
    with np.load(args.base) as source:
        base_e = np.asarray(source["proteins"][pi], dtype=np.float32)
        base_r = np.asarray(source["reactions"][qi], dtype=np.float32)
    with h5py.File(args.protein_means) as source:
        if source["ids"].asstr()[:].tolist() != proteins:
            raise ValueError("Protein mean ID order differs from catalog")
        if "complete" in source and not bool(source["complete"][:].all()):
            raise ValueError("Protein means are incomplete")
        mean = np.asarray(source["vectors"][:][pi], dtype=np.float32)
    dictionary = torch.load(checked(freeze["smooth_dictionary"]), map_location=args.device, weights_only=False)
    if dictionary["feature_manifest_sha256"] != freeze["feature_manifest"]["sha256"]:
        raise ValueError("Smooth dictionary training source changed")
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    density, state = DensityGatedEncoder.from_bundle(density_path, args.device)
    if state["bundle"]["feature_manifest_sha256"] != freeze["feature_manifest"]["sha256"]:
        raise ValueError("Density training source changed")
    smooth = SmoothAnchorDualEncoder(dictionary, dictionary["train_reactions"],
                                     dict(freeze["smooth"], alpha=1.0)).to(args.device)
    model = ComposedPhase2Encoder(density, smooth, freeze["alpha"]).to(args.device)
    with np.load(args.reaction_features) as source:
        blocks = {key: torch.as_tensor(source[key][qi], device=args.device)
                  for key in model.modalities}
        masks = {key: torch.as_tensor(source[key + "_mask"][qi], device=args.device)
                 for key in model.modalities}
    e = torch.as_tensor(base_e, device=args.device)
    r = torch.as_tensor(base_r, device=args.device)
    m = torch.as_tensor(mean, device=args.device)
    with torch.inference_mode():
        encoded_e = model.encode_enzymes(e, m, args.batch_size)
        encoded_r = model.encode_reactions(r, blocks, masks, args.batch_size)
        selected = canonical_dot(encoded_r, encoded_e).cpu().numpy()
        baseline = canonical_dot(F.normalize(r, dim=1), F.normalize(e, dim=1)).cpu().numpy()
        baseline_fp64 = canonical_dot(row_unit(r), row_unit(e)).cpu().numpy()
    if (selected.shape != (len(reaction_ids), len(protein_ids))
            or not np.isfinite(selected).all()):
        raise ValueError("Invalid target phase-2 score matrix")
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "scores.npz", reaction_ids=np.asarray(reaction_ids),
                        protein_ids=np.asarray(protein_ids), selected=selected,
                        baseline=baseline, baseline_fp64=baseline_fp64)
    receipt = {
        "schema": "enzymecage_phase2_target_score_v1", "scope": args.scope,
        "seed": args.seed, "labels_read": False,
        "freeze": {"path": str(args.freeze.resolve()), "sha256": digest(args.freeze)},
        "inputs": {name: {"path": str(path.resolve()), "sha256": digest(path)}
                   for name, path in (("base", args.base), ("catalog", args.catalog),
                                      ("protein_means", args.protein_means),
                                      ("reaction_features", args.reaction_features))},
        "scores_sha256": digest(args.output / "scores.npz"),
        "score_shape": list(selected.shape), "source_sha256": digest(Path(__file__)),
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"scope": args.scope, "shape": list(selected.shape),
                      "score_path": str(args.output / "scores.npz")}))


if __name__ == "__main__":
    main()
