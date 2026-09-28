#!/usr/bin/env python3
"""Score EnzymeMap-trained phase 2 on the fixed CLIPZyme screening library.

Both endpoints are encoded independently. This script reads no screening labels;
official test evaluation is a separate step.
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
from horizyn.generalization_density import DensityGatedEncoder
from horizyn.generalization_phase2 import ComposedPhase2Encoder
from horizyn.semantic_smooth import SmoothAnchorDualEncoder
from scripts.generalization_export import reaction_block


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def library(catalog, screen, means):
    with (catalog / "screening_candidate_map.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 261907:
        raise ValueError("Screening candidate order changed")
    ordered_ids = [row["uniprot_id"] for row in rows]
    protein_ids = [row["protein_id"] for row in rows]
    pieces, ids = [], []
    for rank in range(4):
        stem = f"protein_rank{rank:02d}_of_04"
        ids.extend((screen / f"{stem}.ids.txt").read_text().splitlines())
        pieces.append(np.load(screen / f"{stem}.npy", mmap_mode="r"))
    base = np.concatenate(pieces)
    if len(ids) != len(set(ids)) or set(ids) != set(protein_ids):
        raise ValueError("F3 screening protein shards do not match candidate map")
    with h5py.File(means) as source:
        mean_ids = source["ids"].asstr()[:].tolist()
        if not source["complete"][:].all():
            raise ValueError("Screening protein means incomplete")
        mean = source["vectors"][:]
    if set(mean_ids) != set(ids):
        raise ValueError("Screening protein mean axes mismatch")
    base_index = {key: i for i, key in enumerate(ids)}
    base = base[[base_index[key] for key in mean_ids]]
    candidate_index = {key: i for i, key in enumerate(mean_ids)}
    expanded = np.asarray([candidate_index[key] for key in protein_ids], np.int64)
    return ordered_ids, mean_ids, expanded, base, mean


def queries(args, feature_catalog):
    if args.scope == "validation":
        keys = feature_catalog["validation_reactions"]
        idx = {key: i for i, key in enumerate(feature_catalog["reactions"])}
        rows = [idx[key] for key in keys]
        with np.load(args.features / "f3_features.npz") as source:
            base = source["reactions"][rows].astype(np.float32)
        with np.load(args.features / "reaction_features.npz") as source:
            blocks = {key: source[key][rows] for key in
                      ("t5v2", "unimol2", "chiro", "chemistry")}
            masks = {key: source[key + "_mask"][rows] for key in blocks}
        return keys, base, blocks, masks
    protocol = json.loads((args.protocol / "receipt.json").read_text())
    inputs = args.protocol / "query_inputs.csv"
    if digest(inputs) != protocol["query_inputs_sha256"]:
        raise ValueError("Official query inputs changed")
    with inputs.open(newline="") as stream:
        keys = [row["reaction_id"] for row in csv.DictReader(stream)]
    if keys != (args.screen / "query_ids.txt").read_text().splitlines():
        raise ValueError("F3 test query order differs from official protocol")
    base = np.load(args.screen / "reaction_embeddings.npy")
    blocks, masks = {}, {}
    for name, molecular in (("t5v2", False), ("unimol2", True), ("chiro", True)):
        physical = "reactiont5v2" if name == "t5v2" else name
        blocks[name], masks[name] = reaction_block(
            args.catalog / "features" / f"{physical}.forward.h5", keys, molecular)
    with np.load(args.catalog / "chemistry/test_reaction_set_features.npz", allow_pickle=True) as source:
        index = {str(key): i for i, key in enumerate(source["ids"])}
        blocks["chemistry"] = np.zeros((len(keys), source["vectors"].shape[1]), np.float32)
        masks["chemistry"] = np.zeros(len(keys), bool)
        present = source["mask"] if "mask" in source else np.ones(len(index), bool)
        for i, key in enumerate(keys):
            if key in index:
                j = index[key]
                blocks["chemistry"][i] = source["vectors"][j]
                masks["chemistry"][i] = present[j]
    return keys, base, blocks, masks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", choices=("validation", "test"), required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--screen", type=Path, required=True)
    p.add_argument("--means", type=Path, required=True)
    p.add_argument("--density-bundle", type=Path, required=True)
    p.add_argument("--dictionary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--alpha", type=float, default=.25,
                   help="Validation-tunable density/smooth composition weight")
    a = p.parse_args()
    if not 0 <= a.alpha <= 1:
        p.error("--alpha must lie in [0, 1]")
    if a.output.exists():
        raise FileExistsError(a.output)
    if not (a.features / "complete.json").exists():
        raise ValueError("Target F3/phase-2 feature export incomplete")
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    feature_catalog = json.loads((a.features / "catalog.json").read_text())
    ordered_ids, protein_ids, expanded, base, means = library(a.catalog, a.screen, a.means)
    query_ids, base_r, blocks, masks = queries(a, feature_catalog)
    density, state = DensityGatedEncoder.from_bundle(a.density_bundle, a.device)
    dictionary = torch.load(a.dictionary, map_location=a.device, weights_only=False)
    if dictionary["feature_manifest_sha256"] != state["bundle"]["feature_manifest_sha256"]:
        raise ValueError("Density and smooth training data mismatch")
    smooth = SmoothAnchorDualEncoder(dictionary, dictionary["train_reactions"],
        dict(kernel="exponential", reaction_neighbors=None, reaction_temperature=.03,
             protein_neighbors=32, enzyme_temperature=.03, alpha=1.0)).to(a.device)
    model = ComposedPhase2Encoder(density, smooth, a.alpha).to(a.device)
    with torch.inference_mode():
        index = model.encode_enzyme_index(torch.as_tensor(base, device=a.device),
                                          torch.as_tensor(means, device=a.device),
                                          a.batch_size)
        reaction = model.encode_reactions(torch.as_tensor(base_r, device=a.device),
            {key: torch.as_tensor(value, device=a.device) for key, value in blocks.items()},
            {key: torch.as_tensor(value, device=a.device) for key, value in masks.items()},
            a.batch_size)
        a.output.mkdir(parents=True)
        scores = np.lib.format.open_memmap(a.output / "scores.npy", mode="w+", dtype=np.float32,
                                           shape=(len(query_ids), len(ordered_ids)))
        for start in range(0, len(query_ids), 32):
            values = model.score_index(reaction[start:start + 32], index).cpu().numpy()
            scores[start:start + len(values)] = values[:, expanded]
            scores.flush()
            print(json.dumps({"scored_queries": start + len(values),
                              "total": len(query_ids)}), flush=True)
        del scores
    (a.output / "query_ids.txt").write_text("\n".join(query_ids) + "\n")
    (a.output / "candidate_ids.txt").write_text("\n".join(ordered_ids) + "\n")
    receipt = dict(schema="clipzyme_enzymemap_phase2_full_screen_v1", scope=a.scope,
        alpha=a.alpha,
        test_labels_read=False, train_feature_manifest_sha256=digest(a.features / "manifest.json"),
        density_bundle_sha256=digest(a.density_bundle), dictionary_sha256=digest(a.dictionary),
        candidate_map_sha256=digest(a.catalog / "screening_candidate_map.csv"),
        screen_reaction_receipt_sha256=digest(a.screen / "reaction_receipt.json"),
        screen_protein_mean_sha256=digest(a.means), score_shape=[len(query_ids), len(ordered_ids)],
        scores_sha256=digest(a.output / "scores.npy"), source_sha256=digest(Path(__file__)))
    (a.output / "score_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
