#!/usr/bin/env python3
"""Train-only mean-affinity calibration of frozen F3 dual-encoder embeddings.

No test or Case1 labels are read. Additive scores are
q.e - alpha * mean(train_q).e - beta * q.mean(train_e), which remain
independently encodable. Means use only the original training association graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generalization_metrics import evaluate_scores

DIRECTIONS = ("reaction_to_enzyme", "enzyme_to_reaction")


def atomic_json(path, value):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(block)
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, sha256=digest.hexdigest())


def weighted_mean(vectors, weights):
    weights = weights.to(vectors.dtype)
    return (weights @ vectors) / weights.sum()


def fit_means(train_queries, proteins, train_edges, train_query_global, train_protein_global):
    """Fit unique-ID, pair-frequency, and opposite-anchor-balanced means."""
    device = proteins.device
    edges = torch.as_tensor(train_edges, dtype=torch.long, device=device)
    global_to_train = torch.full((int(edges[:, 0].max()) + 1,), -1, device=device, dtype=torch.long)
    global_to_train[train_query_global] = torch.arange(len(train_queries), device=device)
    qi, ei = global_to_train[edges[:, 0]], edges[:, 1]
    if bool((qi < 0).any()):
        raise ValueError("Training edges include a reaction outside the training cache")
    q_degree = torch.bincount(qi, minlength=len(train_queries)).float()
    e_degree = torch.bincount(ei, minlength=len(proteins)).float()
    q_balanced = torch.zeros_like(q_degree).scatter_add_(0, qi, e_degree[ei].reciprocal())
    e_balanced = torch.zeros_like(e_degree).scatter_add_(0, ei, q_degree[qi].reciprocal())
    return {
        "unique_id": (train_queries.mean(0), proteins[train_protein_global].mean(0)),
        "pair_frequency": (weighted_mean(train_queries, q_degree), weighted_mean(proteins, e_degree)),
        "opposite_anchor_balanced": (weighted_mean(train_queries, q_balanced), weighted_mean(proteins, e_balanced)),
    }


def save_evaluation(output, name, result):
    arrays = {}
    for direction, strata in result["per_query"].items():
        for stratum, values in strata.items():
            for metric, value in values.items():
                arrays[f"query__{direction}__{stratum}__{metric}"] = value
    for direction, values in result["per_positive"].items():
        for metric, value in values.items():
            arrays[f"positive__{direction}__{metric}"] = value
    path = output / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    atomic_json(output / f"{name}.json", result["summary"])
    return str(path.resolve())


def selection_summary(summary, baseline, tolerance):
    values = [summary[d]["all"]["reactzyme_mrr"] for d in DIRECTIONS]
    checks = {}
    for direction in DIRECTIONS:
        for stratum in ("all", "unseen_reaction"):
            current = summary[direction][stratum]["reactzyme_mrr"]
            reference = baseline[direction][stratum]["reactzyme_mrr"]
            checks[f"{direction}/{stratum}"] = current is not None and reference is not None and current >= reference - tolerance
    return dict(mean_all_positive_mrr=sum(values) / 2, eligible=all(checks.values()), guardrails=checks)


def run(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "registry.json").exists():
        raise ValueError("Registry already exists; choose a fresh output directory")
    paths = {name: args.features / name for name in ("catalog.json", "pairs.npz", "f3_features.npz", "manifest.json")}
    config = dict(schema="train_only_hubness_v1", inputs={k: identity(v) for k, v in paths.items()},
                  script=identity(Path(__file__)), metrics=identity(ROOT / "scripts/generalization_metrics.py"),
                  grid=args.grid, centered_cosine=args.centered_cosine, device=str(device),
                  candidate_order="sorted validation IDs in feature catalog", ties="stable ascending candidate index",
                  train_only_means=True, test_used=False, case1_used=False,
                  selection=dict(objective="mean of both all-positive query-macro MRRs", tolerance=args.tolerance,
                                 guardrails="each direction: aggregate and unseen-reaction MRR within tolerance of unmodified F3"),
                  mean_definitions={"unique_id": "equal weight per unique training reaction/enzyme",
                                    "pair_frequency": "uniform unique training association edges",
                                    "opposite_anchor_balanced": "reaction mean: uniform enzymes then their reactions; enzyme mean: uniform reactions then their enzymes"},
                  precision_note="BF16-inferred float16 training cache; same-checkpoint FP32 validation overrides; all scoring and mean accumulation FP32",
                  started_unix=time.time())
    atomic_json(args.output / "config.json", config)
    catalog = json.loads(paths["catalog.json"].read_text())
    with np.load(paths["pairs.npz"]) as source:
        train_edges, val_edges = source["train"].copy(), source["validation"].copy()
    qi = {key: i for i, key in enumerate(catalog["reactions"])}
    ei = {key: i for i, key in enumerate(catalog["proteins"])}
    qrows = np.asarray([qi[key] for key in catalog["validation_reactions"]])
    erows = np.asarray([ei[key] for key in catalog["validation_candidates"]])
    qmap, emap = np.full(len(qi), -1), np.full(len(ei), -1)
    qmap[qrows], emap[erows] = np.arange(len(qrows)), np.arange(len(erows))
    if np.any(qmap[val_edges[:, 0]] < 0) or np.any(emap[val_edges[:, 1]] < 0):
        raise ValueError("Validation edge absent from configured candidate universe")
    train_q_set, train_e_set = set(catalog["train_reactions"]), set(catalog["train_proteins"])
    truth = dict(reaction_index=qmap[val_edges[:, 0]], enzyme_index=emap[val_edges[:, 1]],
                 reaction_seen=[key in train_q_set for key in catalog["validation_reactions"]],
                 enzyme_seen=[key in train_e_set for key in catalog["validation_candidates"]])
    with np.load(paths["f3_features.npz"]) as source:
        proteins = F.normalize(torch.as_tensor(source["proteins"], device=device), dim=1)
        queries = F.normalize(torch.as_tensor(source["reactions"][qrows], device=device), dim=1)
        train_queries = F.normalize(torch.as_tensor(source["train_reactions"], device=device), dim=1)
    enzymes = proteins[torch.as_tensor(erows, device=device)]
    means = fit_means(train_queries, proteins, train_edges,
                     torch.tensor([qi[key] for key in catalog["train_reactions"]], device=device),
                     torch.tensor([ei[key] for key in catalog["train_proteins"]], device=device))
    np.savez(args.output / "train_means.npz", **{f"{key}_{side}": value.cpu().numpy()
             for key, pair in means.items() for side, value in zip(("reaction", "enzyme"), pair)})
    atomic_json(args.output / "evaluation_ids.json", dict(reactions=catalog["validation_reactions"], enzymes=catalog["validation_candidates"]))
    base = queries @ enzymes.T
    baseline = evaluate_scores(base, truth, batch_size=args.batch_size)
    base_artifact = save_evaluation(args.output, "baseline", baseline)
    registry = [dict(id="baseline", mode="cosine", alpha=0., beta=0., mean="none", summary=baseline["summary"],
                     artifacts=[base_artifact], **selection_summary(baseline["summary"], baseline["summary"], args.tolerance))]
    print(json.dumps(registry[0]), flush=True)
    for scheme, (qmean, emean) in means.items():
        directional = {}
        for strength in args.grid:
            for direction in DIRECTIONS:
                if strength == 0:
                    result, artifact = baseline, base_artifact
                else:
                    adjustment = ((qmean @ enzymes.T).unsqueeze(0) if direction == "reaction_to_enzyme"
                                  else (queries @ emean).unsqueeze(1))
                    result = evaluate_scores(base - strength * adjustment, truth, args.batch_size, directions=[direction])
                    artifact = save_evaluation(args.output, f"additive_{scheme}_{direction}_{strength:g}", result)
                directional[(direction, strength)] = (result["summary"][direction], artifact)
        for alpha in args.grid:
            for beta in args.grid:
                if alpha == beta == 0:
                    continue
                summary = {d: directional[d, strength][0] for d, strength in zip(DIRECTIONS, (alpha, beta))}
                entry = dict(id=f"additive_{scheme}_a{alpha:g}_b{beta:g}", mode="additive", mean=scheme,
                             alpha=alpha, beta=beta, summary=summary,
                             artifacts=[directional[d, strength][1] for d, strength in zip(DIRECTIONS, (alpha, beta))],
                             **selection_summary(summary, baseline["summary"], args.tolerance))
                registry.append(entry)
        if args.centered_cosine:
            for alpha in args.grid:
                for beta in args.grid:
                    if alpha == beta == 0:
                        continue
                    scores = F.normalize(queries - alpha * qmean, dim=1) @ F.normalize(enzymes - beta * emean, dim=1).T
                    result = evaluate_scores(scores, truth, args.batch_size)
                    name = f"centered_cosine_{scheme}_a{alpha:g}_b{beta:g}"
                    artifact = save_evaluation(args.output, name, result)
                    entry = dict(id=name, mode="centered_cosine", mean=scheme, alpha=alpha, beta=beta,
                                 summary=result["summary"], artifacts=[artifact],
                                 **selection_summary(result["summary"], baseline["summary"], args.tolerance))
                    registry.append(entry)
                    print(json.dumps({k: entry[k] for k in ("id", "mean_all_positive_mrr", "eligible")}), flush=True)
        atomic_json(args.output / "registry.partial.json", registry)
        print(f"Finished {scheme}: {len(registry)} configurations", flush=True)
    eligible = [entry for entry in registry if entry["eligible"]]
    selected = max(eligible, key=lambda x: x["mean_all_positive_mrr"])
    unrestricted = max(registry, key=lambda x: x["mean_all_positive_mrr"])
    atomic_json(args.output / "registry.json", registry)
    atomic_json(args.output / "selection.json", dict(selected=selected, unrestricted=unrestricted,
                baseline=registry[0], complete=True, test_used=False, case1_used=False, elapsed_seconds=time.time() - config["started_unix"]))
    print(json.dumps(dict(selected=selected["id"], unrestricted=unrestricted["id"], configurations=len(registry))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "runs/generalization_20260919_2251/features")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/generalization_20260919_2251/hubness")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--grid", type=float, nargs="+", default=[0., 0.25, 0.5, 0.75, 1.])
    parser.add_argument("--centered-cosine", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if 0. not in args.grid or any(not np.isfinite(x) or x < 0 for x in args.grid):
        parser.error("Grid must include zero and finite nonnegative strengths")
    run(args)


if __name__ == "__main__":
    main()
