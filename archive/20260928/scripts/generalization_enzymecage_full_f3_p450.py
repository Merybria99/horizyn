#!/usr/bin/env python3
"""Score the unchanged physical P450 pool with validation-selected BCE F3."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from horizyn.benchmarks.retrieval import (BenchmarkTask, build_reaction_inputs,
    cosine_scores, encode_reactions, encode_residue_targets, load_repo_checkpoint)
from horizyn.config import load_config
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = json.loads((args.training / "receipt.json").read_text())
    if receipt["schema"] != "enzymecage_full_f3_labeled_bce_v1":
        raise ValueError("Expected full F3 BCE training receipt")
    metrics = [json.loads(line) for line in (args.training / "metrics.jsonl").read_text().splitlines()]
    selected = max(metrics, key=lambda row: row["validation_auroc"])
    checkpoint = torch.load(args.training / "best.pt", map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != selected["epoch"] or checkpoint["validation_auroc"] != selected["validation_auroc"]:
        raise ValueError("Selected checkpoint does not match validation metrics")
    config_path = Path(receipt["config"])
    base_path = Path(receipt.get("architecture_reference") or receipt["warm_start"])
    base_sha = receipt.get("architecture_reference_sha256") or receipt["warm_start_sha256"]
    if digest(config_path) != receipt["config_sha256"] or digest(base_path) != base_sha:
        raise ValueError("Training configuration or architecture reference changed")
    catalog = json.loads(args.catalog.read_text())
    proteins, reactions = catalog["proteins"], catalog["reactions"]
    if len(proteins) != 490 or len(reactions) != 191:
        raise ValueError("Original P450 candidate pool changed")
    root = args.feature_root
    config = load_config(str(config_path))
    config.data.reaction_chemistry_vectors_path = str(root / "chemistry_f3.npz")
    task = BenchmarkTask(name="enzymecage_full_f3_p450", task_type="retrieval",
        dataset="EnzymeCAGE_P450", task_label="within_family", split="external",
        pairs=root / "encoding_pairs.csv", reactions=root / "reactions.csv",
        reaction_model_embeds_h5=root / "reactiont5.h5",
        reaction_unimol2_embeds_h5=root / "unimol2.h5",
        reaction_chiro_embeds_h5=root / "chiro.h5")
    inputs = build_reaction_inputs(task, config)
    if set(inputs.keys) != set(reactions):
        raise ValueError("Incomplete P450 reaction features")
    torch.set_num_threads(8)
    model, kind = load_repo_checkpoint(base_path, config, args.device)
    if kind != "residue":
        raise ValueError("Expected residue F3 checkpoint")
    state = {key.removeprefix("base."): value for key, value in checkpoint["model"].items()
             if key.startswith("base.")}
    model.model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    residue = ResidueEmbedDataset(str(root / "proteins.h5"),
        max_tokens=config.data.max_protein_tokens,
        truncation=config.data.protein_truncation)
    try:
        if set(residue.keys) != set(proteins):
            raise ValueError("Incomplete P450 protein features")
        with torch.inference_mode():
            enzyme = encode_residue_targets(model, residue, proteins, args.device, 16, False).float().cpu()
            reaction = encode_reactions(model, inputs, reactions, args.device, 8).float().cpu()
    finally:
        residue.close()
    scores = cosine_scores(reaction, enzyme).numpy().astype(np.float32)
    if not np.isfinite(scores).all() or scores.shape != (191, 490):
        raise ValueError("Nonfinite or incomplete external score matrix")
    args.output.mkdir(parents=True)
    path = args.output / "scores.npz"
    np.savez_compressed(path, scores=scores,
                        reaction_ids=np.asarray(reactions), protein_ids=np.asarray(proteins))
    result = {"schema": "enzymecage_full_f3_p450_scores_v1", "labels_read": False,
              "scope": "external", "training_receipt_sha256": digest(args.training / "receipt.json"),
              "training_initialization": receipt.get("initialization", "F3_positive_only_EnzymeCAGE_checkpoint"),
              "checkpoint_sha256": digest(args.training / "best.pt"),
              "checkpoint_epoch": int(checkpoint["epoch"]),
              "validation_auroc": float(checkpoint["validation_auroc"]),
              "catalog_sha256": digest(args.catalog),
              "physical_inputs_sha256": {name: digest(root / name) for name in
                  ("proteins.h5", "reactions.csv", "reactiont5.h5", "unimol2.h5",
                   "chiro.h5", "chemistry_f3.npz")},
              "score_sha256": digest(path), "source_sha256": digest(Path(__file__))}
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"scores": str(path), "shape": list(scores.shape),
                      "validation_auroc": result["validation_auroc"]}))


if __name__ == "__main__":
    main()
