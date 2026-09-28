#!/usr/bin/env python3
"""Screen train-only ridge/CCA semantic bridges on standard validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horizyn.semantic_linear import apply_normalizer, encode_bridge, fit_bridges, fit_normalizer, paired_statistics
from scripts.generalization_hubness import atomic_json, identity, save_evaluation, selection_summary
from scripts.generalization_metrics import evaluate_scores


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    return value


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "registry.json").exists():
        raise ValueError("Completed registry exists; use a fresh output directory")
    if not (args.features / "complete.json").exists():
        raise ValueError("Protein means export is incomplete")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    started = time.time()
    inputs = ["catalog.json", "pairs.npz", "reaction_features.npz", "f3_features.npz", "manifest.json", "complete.json", "protein_mean.h5"]
    config = dict(schema="semantic_linear_v1", inputs={name: identity(args.features / name) for name in inputs},
                  script=identity(Path(__file__)), module=identity(ROOT / "horizyn/semantic_linear.py"),
                  metrics=identity(ROOT / "scripts/generalization_metrics.py"),
                  ridge_relative=args.ridge, blend=args.blend, modes=["all_four", "without_chemistry"],
                  cca_rank=args.cca_rank, test_used=False, case1_used=False, device=str(device),
                  preprocessing="Row-normalize raw block, subtract unique-training-ID mean, row-normalize; missing modalities zero; concatenate equal modality weights then normalize",
                  fit_weighting="Each training reaction has unit mass, split uniformly among its positive edges; centered weighted covariance and ridge intercepts",
                  regularization="ridge_relative times covariance trace/dimension, separately in each feature space",
                  blend_formula="(1-alpha)*cos(F3_query,F3_enzyme)+alpha*cos(bridge_query,bridge_enzyme); independent weighted-concatenation encoders",
                  selection=dict(objective="mean both all-positive query-macro MRR", tolerance=args.tolerance,
                                 guards="each direction aggregate and unseen reaction within tolerance of baseline"))
    atomic_json(args.output / "config.json", config)
    catalog = json.loads((args.features / "catalog.json").read_text())
    with np.load(args.features / "pairs.npz") as source:
        train, val = source["train"], source["validation"]
    tr, te = np.unique(train[:, 0]), np.unique(train[:, 1])
    qr = {key: i for i, key in enumerate(catalog["reactions"])}
    er = {key: i for i, key in enumerate(catalog["proteins"])}
    vr = np.array([qr[key] for key in catalog["validation_reactions"]])
    ve = np.array([er[key] for key in catalog["validation_candidates"]])
    remap_q, remap_e = np.full(len(qr), -1), np.full(len(er), -1)
    remap_q[vr], remap_e[ve] = np.arange(len(vr)), np.arange(len(ve))
    truth = dict(reaction_index=remap_q[val[:, 0]], enzyme_index=remap_e[val[:, 1]],
                 reaction_seen=np.isin(vr, tr), enzyme_seen=np.isin(ve, te))
    remap_q[tr], remap_e[te] = np.arange(len(tr)), np.arange(len(te))
    edge_r = torch.as_tensor(remap_q[train[:, 0]], device=device)
    edge_e = torch.as_tensor(remap_e[train[:, 1]], device=device)
    tr_tensor, te_tensor = torch.as_tensor(tr, device=device), torch.as_tensor(te, device=device)
    with h5py.File(args.features / "protein_mean.h5", "r") as source:
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in source["ids"][:]]
        if ids != catalog["proteins"] or not source["complete"][:].all():
            raise ValueError("Protein mean IDs/order/completion differ from catalog")
        raw_e = torch.as_tensor(source["vectors"][:], device=device)
    emean = fit_normalizer(raw_e, te_tensor)
    all_e = apply_normalizer(raw_e, emean)
    del raw_e
    blocks, block_means = {}, {}
    with np.load(args.features / "reaction_features.npz") as source:
        for key in ("t5v2", "unimol2", "chiro", "chemistry"):
            raw = torch.as_tensor(source[key], device=device)
            mask = torch.as_tensor(source[key + "_mask"], device=device)
            block_means[key] = fit_normalizer(raw, tr_tensor, mask)
            blocks[key] = apply_normalizer(raw, block_means[key], mask)
    with np.load(args.features / "f3_features.npz") as source:
        f3_q = F.normalize(torch.as_tensor(source["reactions"][vr], device=device), dim=1)
        f3_e = F.normalize(torch.as_tensor(source["proteins"][ve], device=device), dim=1)
    f3_scores = f3_q @ f3_e.T
    baseline = evaluate_scores(f3_scores, truth)
    artifact = save_evaluation(args.output, "baseline", baseline)
    registry = [dict(id="baseline", summary=baseline["summary"], artifacts=[artifact],
                     **selection_summary(baseline["summary"], baseline["summary"], args.tolerance))]
    torch.save(cpu_state(dict(enzyme_mean=emean, reaction_block_means=block_means,
                              train_reaction_ids=catalog["train_reactions"], train_protein_ids=catalog["train_proteins"],
                              preprocessing=config["preprocessing"])), args.output / "scalers.pt")
    for mode, keys in (("all_four", ("t5v2", "unimol2", "chiro", "chemistry")),
                       ("without_chemistry", ("t5v2", "unimol2", "chiro"))):
        all_q = F.normalize(torch.cat([blocks[key] for key in keys], dim=1), dim=1)
        stats = paired_statistics(all_q[tr], all_e[te], edge_r, edge_e)
        print(f"Fit {mode} statistics: train reactions={len(tr)}, enzymes={len(te)}, edges={len(train)}", flush=True)
        for ridge in args.ridge:
            states = fit_bridges(stats, ridge, args.cca_rank)
            for bridge, state in states.items():
                fit_id = f"{mode}_{bridge}_lambda{ridge:g}"
                state_path = args.output / f"{fit_id}.pt"
                torch.save(cpu_state(dict(state=state, modality_keys=list(keys), scaler_path="scalers.pt",
                                          input_manifest_sha256=config["inputs"]["manifest.json"]["sha256"])), state_path)
                q, e = encode_bridge(state, all_q[vr], all_e[ve])
                semantic_scores = q @ e.T
                for alpha in args.blend:
                    name = f"{fit_id}_alpha{alpha:g}"
                    evaluation = evaluate_scores((1-alpha) * f3_scores + alpha * semantic_scores, truth)
                    artifact = save_evaluation(args.output, name, evaluation)
                    entry = dict(id=name, mode=mode, bridge=bridge, ridge_relative=ridge, alpha=alpha,
                                 fit_state=str(state_path.resolve()), summary=evaluation["summary"], artifacts=[artifact],
                                 **selection_summary(evaluation["summary"], baseline["summary"], args.tolerance))
                    registry.append(entry)
                    print(json.dumps({key:entry[key] for key in ("id", "mean_all_positive_mrr", "eligible")}), flush=True)
            atomic_json(args.output / "registry.partial.json", registry)
        del stats
    selected = max((r for r in registry if r["eligible"]), key=lambda x:x["mean_all_positive_mrr"])
    unrestricted = max(registry, key=lambda x:x["mean_all_positive_mrr"])
    atomic_json(args.output / "registry.json", registry)
    atomic_json(args.output / "selection.json", dict(selected=selected, unrestricted=unrestricted,
                baseline=registry[0], complete=True, test_used=False, case1_used=False, elapsed_seconds=time.time()-started))
    print(json.dumps(dict(selected=selected["id"], unrestricted=unrestricted["id"], configurations=len(registry))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT/"runs/generalization_20260919_2251/features")
    parser.add_argument("--output", type=Path, default=ROOT/"runs/generalization_20260919_2251/semantic_linear")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--ridge", type=float, nargs="+", default=[.01, .1, 1., 10.])
    parser.add_argument("--blend", type=float, nargs="+", default=[.1, .25, .5, 1.])
    parser.add_argument("--cca-rank", type=int, default=256)
    parser.add_argument("--tolerance", type=float, default=.005)
    args = parser.parse_args()
    if any(not np.isfinite(x) or x <= 0 for x in args.ridge) or any(not 0 <= x <= 1 for x in args.blend):
        parser.error("Invalid ridge or blend strengths")
    run(args)
