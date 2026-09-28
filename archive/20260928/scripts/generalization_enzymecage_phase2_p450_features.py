#!/usr/bin/env python3
"""Export label-free physical P450 inputs for EnzymeCAGE-trained phase 2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from horizyn.benchmarks.retrieval import (BenchmarkTask, build_reaction_inputs,
    cosine_scores, encode_reactions, encode_residue_targets, load_repo_checkpoint)
from horizyn.config import load_config
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from scripts.generalization_export import reaction_block


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--base-score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    freeze = json.loads(args.freeze.read_text())
    if freeze["schema"] != "enzymecage_phase2_target_freeze_v1":
        raise ValueError("Target phase-2 recipe is not frozen")
    checkpoint = Path(freeze["base_checkpoint"]["path"])
    config_path = Path(freeze["base_config"]["path"])
    if digest(checkpoint) != freeze["base_checkpoint"]["sha256"] or digest(config_path) != freeze["base_config"]["sha256"]:
        raise ValueError("Target-trained F3 checkpoint or config changed")
    catalog = json.loads(args.catalog.read_text())
    proteins, reactions = catalog["proteins"], catalog["reactions"]
    if len(proteins) != 490 or len(reactions) != 191:
        raise ValueError("The original 191-query/490-ID P450 pool changed")
    root = args.feature_root
    config = load_config(str(config_path))
    if config.data.reaction_direction_mode != "forward_only":
        raise ValueError("Physical forward EnzymeCAGE reaction input required")
    config.data.reaction_chemistry_vectors_path = str(root / "chemistry_f3.npz")
    task = BenchmarkTask(name="enzymecage_phase2_p450_inputs", task_type="retrieval",
        dataset="EnzymeCAGE_P450", task_label="within_family", split="external",
        pairs=root / "encoding_pairs.csv", reactions=root / "reactions.csv",
        reaction_model_embeds_h5=root / "reactiont5.h5",
        reaction_unimol2_embeds_h5=root / "unimol2.h5",
        reaction_chiro_embeds_h5=root / "chiro.h5")
    inputs = build_reaction_inputs(task, config)
    if set(inputs.keys) != set(reactions):
        raise ValueError("Incomplete P450 reaction features")
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(42)
    model, kind = load_repo_checkpoint(checkpoint, config, args.device)
    if kind != "residue":
        raise ValueError("Unexpected F3 protein input kind")
    model.eval().requires_grad_(False)
    residue = ResidueEmbedDataset(str(root / "proteins.h5"),
        max_tokens=config.data.max_protein_tokens,
        truncation=config.data.protein_truncation)
    try:
        if set(residue.keys) != set(proteins):
            raise ValueError("Incomplete P450 protein feature coverage")
        with torch.inference_mode():
            enzyme = encode_residue_targets(model, residue, proteins, args.device, 16, False).float().cpu()
            reaction = encode_reactions(model, inputs, reactions, args.device, 8).float().cpu()
            means = np.stack([residue[key]["residue_embeddings"].float().mean(0).numpy()
                              for key in proteins]).astype(np.float32)
    finally:
        residue.close()
    baseline = cosine_scores(reaction, enzyme).numpy()
    with np.load(args.base_score, allow_pickle=False) as source:
        old = source["scores"]
        if source["reaction_ids"].tolist() != reactions or source["protein_ids"].tolist() != proteins:
            raise ValueError("Earlier target F3 score axes changed")
    maximum = float(np.max(np.abs(baseline - old)))
    if maximum > 1e-5:
        raise ValueError(f"Re-encoded target F3 baseline disagrees with earlier export: {maximum}")
    raw = {}
    for modality, filename, molecule in (("t5v2", "reactiont5.h5", False),
                                         ("unimol2", "unimol2.h5", True),
                                         ("chiro", "chiro.h5", True)):
        raw[modality], raw[modality + "_mask"] = reaction_block(root / filename, reactions, molecule)
        if not raw[modality + "_mask"].all():
            raise ValueError(f"Incomplete P450 {modality} features")
    with np.load(root / "chemistry_f3.npz", allow_pickle=True) as source:
        index = {str(key): i for i, key in enumerate(source["ids"])}
        order = [index[key] for key in reactions]
        raw["chemistry"] = source["vectors"][order].astype(np.float32)
        raw["chemistry_mask"] = source["mask"][order].astype(bool)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "base.npz", proteins=enzyme.numpy(), reactions=reaction.numpy())
    np.savez_compressed(args.output / "reaction_features.npz", **raw)
    with h5py.File(args.output / "protein_mean.h5", "w") as handle:
        handle.create_dataset("ids", data=np.asarray(proteins, dtype=object), dtype=h5py.string_dtype())
        handle.create_dataset("vectors", data=means)
        handle.create_dataset("complete", data=np.ones(len(proteins), bool))
    receipt = {"schema": "enzymecage_phase2_physical_p450_features_v1",
               "labels_read": False, "freeze_sha256": digest(args.freeze),
               "baseline_max_abs_error": maximum,
               "base_score_sha256": digest(args.base_score),
               "catalog_sha256": digest(args.catalog),
               "sources": {name: digest(root / name) for name in
                           ("proteins.h5", "reactiont5.h5", "unimol2.h5", "chiro.h5", "chemistry_f3.npz", "reactions.csv")},
               "outputs": {name: digest(args.output / name) for name in
                           ("base.npz", "reaction_features.npz", "protein_mean.h5")},
               "source_sha256": digest(Path(__file__))}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "baseline_max_abs_error": maximum}))


if __name__ == "__main__":
    main()
