#!/usr/bin/env python3
"""Validation-only screen for a training-anchor semantic residual to F3."""
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
from horizyn.semantic_anchors import (fit_center, centered_unit, reaction_features,
    make_protein_reaction_map, nearest_training_proteins, enzyme_anchor_features,
    reaction_anchor_features)
from generalization_metrics import evaluate_scores
from generalization_full_graph import atomic_json, sha, validation_data, selection_value, eligible


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    return value


def run(args):
    if not (args.features / "complete.json").exists():
        raise ValueError("Raw feature export is incomplete")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "registry.json").exists():
        raise ValueError("Use a new output directory")
    shutil.copyfile(__file__, args.output / "source.py")
    shutil.copyfile(ROOT / "horizyn/semantic_anchors.py", args.output / "model_source.py")
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    started = time.monotonic()
    catalog = json.loads((args.features / "catalog.json").read_text())
    with np.load(args.features / "pairs.npz") as source:
        pairs = {key: source[key] for key in source.files}
    tr, te = np.unique(pairs["train"][:, 0]), np.unique(pairs["train"][:, 1])
    vr, ve, truth = validation_data(catalog, pairs)
    with h5py.File(args.features / "protein_mean.h5", "r") as source:
        if not source["complete"][:].all():
            raise ValueError("Protein export is incomplete")
        mean_protein = torch.tensor(source["vectors"][:], device=device)
    with np.load(args.features / "f3_features.npz") as source:
        f3e = F.normalize(torch.tensor(source["proteins"][ve], device=device), dim=-1)
        f3r = F.normalize(torch.tensor(source["reactions"][vr], device=device), dim=-1)
    base_scores = f3r @ f3e.T
    del f3r, f3e
    blocks, masks = {}, {}
    with np.load(args.features / "reaction_features.npz") as source:
        for key in ("t5v2", "unimol2", "chiro", "chemistry"):
            blocks[key] = torch.tensor(source[key], device=device)
            masks[key] = torch.tensor(source[key + "_mask"], device=device)
    centers = {key: fit_center(blocks[key], tr, masks[key]) for key in blocks}
    protein_center = fit_center(mean_protein, te)
    encoded_protein = centered_unit(mean_protein, protein_center)
    del mean_protein
    train_protein = encoded_protein[te]
    neighbor_values, neighbor_indices = nearest_training_proteins(
        encoded_protein[ve], train_protein, args.protein_neighbors, args.batch_size)
    del encoded_protein
    r_lookup, e_lookup = ({int(g): i for i, g in enumerate(ids)} for ids in (tr, te))
    er = torch.tensor([r_lookup[int(x)] for x in pairs["train"][:, 0]], device=device)
    ee = torch.tensor([e_lookup[int(x)] for x in pairs["train"][:, 1]], device=device)
    adjacency = make_protein_reaction_map(er, ee, len(te))
    registry = dict(schema="generalization_anchors_v2_fp64_reductions", test_used=False,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        feature_manifest_sha256=sha(args.features / "manifest.json"),
        source_sha256=sha(__file__), model_sha256=sha(ROOT / "horizyn/semantic_anchors.py"),
        train_reactions=len(tr), train_proteins=len(te),
        selection="Mean all-positive validation MRR with both aggregate and unseen-reaction direction guards",
        enzyme_temperatures=[.03, .1], reaction_temperatures=[.03, .1],
        blend_weights=[.1, .25, .5, 1.])
    atomic_json(args.output / "registry.json", registry)
    baseline_eval = evaluate_scores(base_scores, truth)
    baseline = baseline_eval["summary"]
    records = [dict(method="F3", alpha=0., eligible=True, selection_value=selection_value(baseline), validation=baseline)]
    best = records[0]
    selected_state = None
    # Save the training dictionary once; it is independent of every test pool.
    shared = dict(protein_center=protein_center, reaction_centers=centers,
                  train_proteins=train_protein, adjacency=adjacency,
                  train_protein_ids=[catalog["proteins"][i] for i in te],
                  train_reaction_ids=[catalog["reactions"][i] for i in tr],
                  protein_neighbors=args.protein_neighbors, reaction_neighbors=args.reaction_neighbors,
                  feature_manifest_sha256=registry["feature_manifest_sha256"])
    torch.save(cpu_state(shared), args.output / "training_dictionary.pt")
    for mode in ("all_modalities", "without_set_chemistry"):
        modalities = ["t5v2", "unimol2", "chiro"] + (["chemistry"] if mode == "all_modalities" else [])
        rx = reaction_features(blocks, centers, masks, modalities)
        for et in registry["enzyme_temperatures"]:
            efeat = enzyme_anchor_features(neighbor_values, neighbor_indices, adjacency, len(tr), et)
            for rt in registry["reaction_temperatures"]:
                rfeat = reaction_anchor_features(rx[vr], rx[tr], rt, args.reaction_neighbors)
                semantic_scores = rfeat @ efeat.T
                for alpha in registry["blend_weights"]:
                    scores = (1 - alpha) * base_scores + alpha * semantic_scores
                    evaluation = evaluate_scores(scores, truth)
                    summary = evaluation["summary"]
                    record = dict(method="semantic_anchors", mode=mode, enzyme_temperature=et,
                        reaction_temperature=rt, alpha=alpha, validation=summary,
                        selection_value=selection_value(summary),
                        eligible=eligible(summary, baseline, args.max_drop))
                    records.append(record)
                    if record["eligible"] and record["selection_value"] > best["selection_value"]:
                        best = record
                        selected_state = dict(modalities=modalities, train_reactions=rx[tr],
                            enzyme_temperature=et, reaction_temperature=rt, alpha=alpha,
                            selected_validation=record, dictionary_path=str((args.output / "training_dictionary.pt").resolve()))
                        torch.save(cpu_state(selected_state), args.output / "selected.pt")
                        np.savez(args.output / "selected_validation_ranks.npz", **{
                            f"{d}_{k}": v for d, block in evaluation["per_positive"].items() for k, v in block.items()})
                    print(json.dumps({k: v for k, v in record.items() if k != "validation"}), flush=True)
                del semantic_scores, rfeat
            del efeat
        del rx
    output = dict(records=records, selected=best,
                  unrestricted=max(records, key=lambda x: x["selection_value"]),
                  elapsed_seconds=time.monotonic() - started,
                  peak_vram_gib=torch.cuda.max_memory_allocated(device) / 2**30)
    atomic_json(args.output / "validation.json", output)
    atomic_json(args.output / "complete.json", dict(selected=best, test_used=False,
                elapsed_seconds=output["elapsed_seconds"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--protein-neighbors", type=int, default=32)
    parser.add_argument("--reaction-neighbors", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-drop", type=float, default=.005)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
