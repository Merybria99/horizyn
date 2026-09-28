#!/usr/bin/env python3
"""Build the fixed phase-2 training anchor dictionary on EnzymeCAGE positives.

No validation or external labels are read. Reaction features are centered on
the target training graph, then indexed in the same order as its reaction IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from horizyn.semantic_anchors import reaction_features


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--training-dictionary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not (args.features / "complete.json").is_file():
        raise ValueError("Training feature export is incomplete")
    torch.set_num_threads(8)
    catalog = json.loads((args.features / "catalog.json").read_text())
    with np.load(args.features / "pairs.npz") as source:
        train_reaction_indices = np.unique(source["train"][:, 0])
    expected = [catalog["reactions"][int(i)] for i in train_reaction_indices]
    dictionary = torch.load(args.training_dictionary, map_location="cpu", weights_only=False)
    if dictionary["train_reaction_ids"] != expected:
        raise ValueError("Training reaction order changed")
    if dictionary["feature_manifest_sha256"] != digest(args.features / "manifest.json"):
        raise ValueError("Dictionary and feature export do not match")
    modalities = ["t5v2", "unimol2", "chiro", "chemistry"]
    with np.load(args.features / "reaction_features.npz") as source:
        blocks = {key: torch.as_tensor(source[key], device=args.device) for key in modalities}
        masks = {key: torch.as_tensor(source[key + "_mask"], device=args.device) for key in modalities}
    centers = {key: value.to(args.device) for key, value in dictionary["reaction_centers"].items()}
    with torch.inference_mode():
        encoded = reaction_features(blocks, centers, masks, modalities)
        train_reactions = encoded[train_reaction_indices].cpu()
    if not bool(torch.isfinite(train_reactions).all()):
        raise ValueError("Nonfinite training reaction anchors")
    dictionary.update(
        train_reactions=train_reactions,
        modalities=modalities,
        enzyme_temperature=0.03,
        reaction_temperature=0.03,
        protein_neighbors=32,
        reaction_neighbors=None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dictionary, args.output)
    receipt = {
        "schema": "enzymecage_phase2_smooth_dictionary_v1",
        "features_manifest_sha256": digest(args.features / "manifest.json"),
        "training_dictionary_sha256": digest(args.training_dictionary),
        "output_sha256": digest(args.output),
        "source_sha256": digest(Path(__file__)),
        "training_reactions": len(train_reactions),
        "training_proteins": len(dictionary["train_protein_ids"]),
        "validation_or_external_labels_used": False,
    }
    (args.output.parent / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
