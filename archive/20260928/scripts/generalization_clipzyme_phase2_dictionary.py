#!/usr/bin/env python3
"""Fit the EnzymeMap phase-2 smooth dictionary from training edges only."""
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
from horizyn.semantic_anchors import (fit_center, centered_unit,
                                      reaction_features, make_protein_reaction_map)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if not (a.features / "complete.json").is_file():
        raise ValueError("F3 feature export incomplete")
    torch.set_num_threads(8)
    catalog = json.loads((a.features / "catalog.json").read_text())
    with np.load(a.features / "pairs.npz") as source:
        train = source["train"]
    tr, te = np.unique(train[:, 0]), np.unique(train[:, 1])
    if [catalog["reactions"][int(i)] for i in tr] != catalog["train_reactions"]:
        raise ValueError("Training reaction order mismatch")
    with h5py.File(a.features / "protein_mean.h5") as source:
        if source["ids"].asstr()[:].tolist() != catalog["proteins"]:
            raise ValueError("Protein mean axes mismatch")
        if not source["complete"][:].all():
            raise ValueError("Protein mean export incomplete")
        mean = torch.as_tensor(source["vectors"][:], device=a.device)
    pc = fit_center(mean, te)
    tp = centered_unit(mean[te], pc)
    modalities = ["t5v2", "unimol2", "chiro", "chemistry"]
    with np.load(a.features / "reaction_features.npz") as source:
        blocks = {key: torch.as_tensor(source[key], device=a.device) for key in modalities}
        masks = {key: torch.as_tensor(source[key + "_mask"], device=a.device) for key in modalities}
    centers = {key: fit_center(blocks[key], tr, masks[key]) for key in modalities}
    rx = reaction_features(blocks, centers, masks, modalities)[tr]
    rmap, emap = ({int(g): i for i, g in enumerate(values)} for values in (tr, te))
    er = torch.tensor([rmap[int(x)] for x in train[:, 0]], device=a.device)
    ee = torch.tensor([emap[int(x)] for x in train[:, 1]], device=a.device)
    adjacency = make_protein_reaction_map(er, ee, len(te))
    state = dict(protein_center=pc.cpu(), train_proteins=tp.cpu(), train_reactions=rx.cpu(),
                 adjacency=adjacency.cpu(),
                 reaction_centers={key: value.cpu() for key, value in centers.items()},
                 modalities=modalities, enzyme_temperature=.03, reaction_temperature=.03,
                 protein_neighbors=32, reaction_neighbors=None,
                 train_protein_ids=[catalog["proteins"][int(i)] for i in te],
                 train_reaction_ids=[catalog["reactions"][int(i)] for i in tr],
                 feature_manifest_sha256=digest(a.features / "manifest.json"))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, a.output)
    receipt = dict(schema="clipzyme_phase2_enzymemap_dictionary_v1",
                   train_only=True, test_labels_read=False,
                   train_reactions=len(tr), train_proteins=len(te), train_edges=len(train),
                   feature_manifest_sha256=state["feature_manifest_sha256"],
                   dictionary_sha256=digest(a.output), source_sha256=digest(Path(__file__)))
    a.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
