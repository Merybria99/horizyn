#!/usr/bin/env python3
"""Align all EnzymeCAGE labeled rows with frozen target-trained F3 features."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from horizyn.benchmarks.retrieval import encode_residue_targets, load_repo_checkpoint
from horizyn.config import load_config
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def labels(path: Path, reaction_index: dict, protein_index: dict) -> np.ndarray:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append((reaction_index[row["reaction_id"]],
                         protein_index[row["protein_id"]], int(row["label"])))
    return np.asarray(rows, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-data", type=Path, required=True)
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--missing-residues", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data_manifest = json.loads((args.labeled_data / "manifest.json").read_text())
    if data_manifest["schema"] != "enzymecage_labeled_f3_data_v1":
        raise ValueError("Expected preserved EnzymeCAGE labeled rows")
    catalog = json.loads(args.catalog.read_text())
    protein_ids = list(catalog["proteins"])
    reaction_ids = list(catalog["reactions"])
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    config = load_config(str(args.base_config))
    model, kind = load_repo_checkpoint(args.base_checkpoint, config, args.device)
    if kind != "residue":
        raise ValueError("Unexpected target F3 protein input kind")
    model.eval().requires_grad_(False)
    residues = ResidueEmbedDataset(str(args.missing_residues),
        max_tokens=config.data.max_protein_tokens,
        truncation=config.data.protein_truncation)
    try:
        missing_ids = list(residues.keys)
        if len(missing_ids) != data_manifest["missing_unique_proteins"]:
            raise ValueError("Missing-protein feature extraction is incomplete")
        if set(missing_ids) & set(protein_ids):
            raise ValueError("New protein IDs overlap the existing base feature bank")
        with torch.inference_mode():
            extra = encode_residue_targets(model, residues, missing_ids,
                                           args.device, 16, False).float().cpu().numpy()
    finally:
        residues.close()
    with np.load(args.base_features) as source:
        base_e = np.asarray(source["proteins"], dtype=np.float32)
        base_r = np.asarray(source["reactions"], dtype=np.float32)
    if (base_e.shape != (len(protein_ids), 512)
            or base_r.shape != (len(reaction_ids), 512)
            or extra.shape != (len(missing_ids), 512)):
        raise ValueError("Frozen F3 embedding dimensions changed")
    protein_ids += missing_ids
    enzymes = np.concatenate((base_e, extra), axis=0)
    if not np.isfinite(enzymes).all() or not np.isfinite(base_r).all():
        raise ValueError("Nonfinite F3 input feature")
    pi = {key: i for i, key in enumerate(protein_ids)}
    qi = {key: i for i, key in enumerate(reaction_ids)}
    train = labels(args.labeled_data / "train_labels.csv", qi, pi)
    valid = labels(args.labeled_data / "validation_labels.csv", qi, pi)
    for split, values in (("train", train), ("validation", valid)):
        expected = data_manifest[split]
        if (len(values) != expected["rows"] or int(values[:, 2].sum()) != expected["positive"]):
            raise ValueError(f"{split} labels changed")
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "base.npz", enzymes=enzymes,
                        reactions=base_r, protein_ids=np.asarray(protein_ids),
                        reaction_ids=np.asarray(reaction_ids))
    np.savez_compressed(args.output / "labeled_pairs.npz", train=train, validation=valid)
    receipt = {
        "schema": "enzymecage_labeled_f3_features_v1",
        "labeled_data_manifest_sha256": sha256(args.labeled_data / "manifest.json"),
        "sources": {name: sha256(path) for name, path in
                    (("base_features", args.base_features), ("catalog", args.catalog),
                     ("missing_residues", args.missing_residues),
                     ("base_config", args.base_config),
                     ("base_checkpoint", args.base_checkpoint))},
        "outputs": {name: sha256(args.output / name)
                    for name in ("base.npz", "labeled_pairs.npz")},
        "training_rows": len(train), "validation_rows": len(valid),
        "protein_ids": len(protein_ids), "reaction_ids": len(reaction_ids),
        "missing_proteins_encoded": len(missing_ids),
        "external_labels_or_scores_read": False,
        "source_sha256": sha256(Path(__file__)),
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in
                      ("training_rows", "validation_rows", "protein_ids",
                       "reaction_ids", "missing_proteins_encoded")}), flush=True)


if __name__ == "__main__":
    main()
