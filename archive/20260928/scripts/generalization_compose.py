#!/usr/bin/env python3
"""Freeze a three-seed graph/anchor recipe using validation only.

Anchor vectors are the raw training-dictionary embeddings, without the earlier
F3 blend. The two endpoints remain independently encodable by weighted concat.
"""
from __future__ import annotations

import argparse
import json
import math
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
from horizyn.generalization_residual import FrozenGeometryResidual
from horizyn.generalization_retrieval import canonical_dot
from horizyn.semantic_anchors import centered_unit, reaction_features, nearest_training_proteins, enzyme_anchor_features, reaction_anchor_features
from scripts.generalization_hubness import atomic_json, identity, save_evaluation, selection_summary
from scripts.generalization_metrics import evaluate_scores

DIRECTIONS = ("reaction_to_enzyme", "enzyme_to_reaction")


def compose_embeddings(graph, anchor, alpha):
    """Apply separately to either endpoint; dot products equal score mixtures."""
    if not 0 <= alpha <= 1 or len(graph) != len(anchor):
        raise ValueError("Invalid mixture weight or endpoint rows")
    return torch.cat((graph * math.sqrt(1-alpha), anchor * math.sqrt(alpha)), dim=1)


def ensemble_embeddings(graph_heads, anchor, alpha):
    """Fixed equal-weight score ensemble, still an independent endpoint map."""
    if not graph_heads:
        raise ValueError("At least one graph head required")
    graph = torch.cat(graph_heads, dim=1) / math.sqrt(len(graph_heads))
    return compose_embeddings(graph, anchor, alpha)


def mean_summaries(summaries):
    return {direction: {stratum: {metric: (None if any(s[direction][stratum][metric] is None for s in summaries)
                        else float(np.mean([s[direction][stratum][metric] for s in summaries])))
                        for metric in values} for stratum, values in summary.items()}
            for direction, summary in summaries[0].items()}


def bootstrap_selected(baseline, selected, draws=10000, seed=20260919):
    rng = np.random.default_rng(seed)
    results = {}
    for direction in DIRECTIONS:
        for stratum in ("all", "seen_reaction", "unseen_reaction"):
            prefix = f"query__{direction}__{stratum}__"
            for row in selected:
                if not np.array_equal(row[prefix+"query_index"], baseline[prefix+"query_index"]):
                    raise ValueError("Paired bootstrap query order mismatch")
            key = direction + "/" + stratum
            results[key] = {}
            for metric in ("reactzyme_mrr", "first_positive_mrr", "top_1", "top_5", "top_10"):
                if metric == "first_positive_mrr":
                    base = 1 / baseline[prefix+"first_rank"]
                    values = np.stack([1 / row[prefix+"first_rank"] for row in selected])
                else:
                    base = baseline[prefix+metric]
                    values = np.stack([row[prefix+metric] for row in selected])
                difference = values.astype(np.float64).mean(0) - base.astype(np.float64)
                distribution = []
                for start in range(0, draws, 128):
                    index = rng.integers(len(difference), size=(min(128, draws-start), len(difference)))
                    distribution.extend(difference[index].mean(1).tolist())
                results[key][metric] = dict(num_queries=len(base), baseline=float(base.mean()),
                    selected_seed_means=values.mean(1).tolist(), selected_seed_std=float(values.mean(1).std(ddof=1)),
                    mean_difference=float(difference.mean()), ci95=np.quantile(distribution, [.025, .975]).tolist())
    return dict(draws=draws, seed=seed, results=results,
                method="Paired query bootstrap of three-seed-mean metrics, keeping seeds paired",
                selection_adjusted=False,
                limitations=["Validation was used to select the recipe; intervals are exploratory, not independent confirmation.",
                             "Queries may share reaction chemistry/protein homology; query resampling does not account for that dependence."])


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "registry.json").exists():
        raise ValueError("Completed registry exists; use a fresh output directory")
    shutil.copy2(__file__, args.output / "source.py")
    graph_paths = {(weighting, seed): args.campaign / f"graph_v2_{weighting}_s{seed}" for weighting in args.weightings for seed in args.seeds}
    for path in [*graph_paths.values(), args.anchor]:
        if not (path / "complete.json").exists():
            raise ValueError(f"Training/screen incomplete: {path}")
    started = time.time()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    sources = {"features_manifest": identity(args.features / "manifest.json"),
               "anchor_selected": identity(args.anchor / "selected.pt"),
               "anchor_dictionary": identity(args.anchor / "training_dictionary.pt"),
               "script": identity(Path(__file__)), "metrics": identity(ROOT / "scripts/generalization_metrics.py"),
               "graph_module": identity(ROOT / "horizyn/generalization_residual.py"),
               "canonical_scoring_module": identity(ROOT / "horizyn/generalization_retrieval.py"),
               "anchor_module": identity(ROOT / "horizyn/semantic_anchors.py")}
    for (weighting, seed), path in graph_paths.items():
        sources[f"graph_{weighting}_{seed}"] = identity(path / "selected.pt")
    atomic_json(args.output / "config.json", dict(schema="graph_anchor_compose_v2_canonical_dot", sources=sources,
        seeds=args.seeds, weightings=args.weightings, anchor_weights=args.alpha, max_drop=args.tolerance,
        test_used=False, case1_used=False, baseline="Frozen F3 cache normalized once",
        graph_input="Native frozen F3 cache outputs; residual encoder performs its own single output normalization",
        score="Actual sqrt-weight-concatenated independent endpoints; canonical_dot accumulates FP64 then returns FP32; no prior F3-anchor blend",
        selection="Three-seed mean all-positive MRR; each direction all/unseen mean within tolerance of F3 baseline",
        float32_matmul_precision="highest", candidate_order="feature catalog sorted validation IDs"))
    state_dir = args.output / "states"
    state_dir.mkdir(exist_ok=True)
    for label, source in sources.items():
        if label.startswith("graph_") and label != "graph_module" or label in {"anchor_selected", "anchor_dictionary"}:
            shutil.copy2(source["path"], state_dir / f"{label}.pt")
    catalog = json.loads((args.features / "catalog.json").read_text())
    with np.load(args.features / "pairs.npz") as source:
        train, val = source["train"], source["validation"]
    qindex = {key:i for i,key in enumerate(catalog["reactions"])}
    eindex = {key:i for i,key in enumerate(catalog["proteins"])}
    vr = np.array([qindex[key] for key in catalog["validation_reactions"]])
    ve = np.array([eindex[key] for key in catalog["validation_candidates"]])
    reverse_r, reverse_e = {g:i for i,g in enumerate(vr)}, {g:i for i,g in enumerate(ve)}
    truth = dict(reaction_index=[reverse_r[x] for x in val[:,0]], enzyme_index=[reverse_e[x] for x in val[:,1]],
                 reaction_seen=np.isin(vr, np.unique(train[:,0])), enzyme_seen=np.isin(ve, np.unique(train[:,1])))
    with np.load(args.features / "f3_features.npz") as source:
        raw_fq = torch.tensor(source["reactions"][vr], device=device)
        raw_fe = torch.tensor(source["proteins"][ve], device=device)
        fq, fe = F.normalize(raw_fq, dim=1), F.normalize(raw_fe, dim=1)
    baseline_fp32 = evaluate_scores(fq @ fe.T, truth)
    save_evaluation(args.output, "baseline_fp32_sensitivity", baseline_fp32)
    baseline = evaluate_scores(canonical_dot(fq, fe), truth)
    baseline_artifact = save_evaluation(args.output, "baseline", baseline)
    anchor = torch.load(state_dir / "anchor_selected.pt", map_location=device, weights_only=False)
    dictionary = torch.load(state_dir / "anchor_dictionary.pt", map_location=device, weights_only=False)
    if dictionary["feature_manifest_sha256"] != sources["features_manifest"]["sha256"]:
        raise ValueError("Anchor and evaluation feature manifests differ")
    if dictionary["train_protein_ids"] != catalog["train_proteins"] or dictionary["train_reaction_ids"] != catalog["train_reactions"]:
        raise ValueError("Training-anchor IDs differ from original training catalog")
    with h5py.File(args.features / "protein_mean.h5") as source:
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in source["ids"][:]]
        if ids != catalog["proteins"] or not source["complete"][:].all():
            raise ValueError("Protein means are incomplete or misaligned")
        raw = torch.tensor(source["vectors"][:], device=device)
    full_encoded = centered_unit(raw, dictionary["protein_center"])[ve]
    encoded = centered_unit(raw[ve], dictionary["protein_center"])
    batch_checks = {"protein_raw_equal": bool(torch.equal(full_encoded, encoded))}
    if not batch_checks["protein_raw_equal"]:
        raise ValueError("Protein endpoint normalization depends on row subset")
    neighbor_values, neighbor_indices = nearest_training_proteins(encoded, dictionary["train_proteins"], dictionary["protein_neighbors"])
    ae = enzyme_anchor_features(neighbor_values, neighbor_indices, dictionary["adjacency"], len(dictionary["train_reaction_ids"]), anchor["enzyme_temperature"])
    alt_values, alt_indices = nearest_training_proteins(encoded, dictionary["train_proteins"], dictionary["protein_neighbors"], batch_size=513)
    alt_ae = enzyme_anchor_features(alt_values, alt_indices, dictionary["adjacency"], len(dictionary["train_reaction_ids"]), anchor["enzyme_temperature"])
    batch_checks["protein_anchor_equal"] = bool(torch.equal(ae, alt_ae))
    with np.load(args.features / "reaction_features.npz") as source:
        blocks = {key:torch.tensor(source[key], device=device) for key in anchor["modalities"]}
        masks = {key:torch.tensor(source[key+"_mask"], device=device) for key in anchor["modalities"]}
    full_encoded = reaction_features(blocks, dictionary["reaction_centers"], masks, anchor["modalities"])[vr]
    encoded = reaction_features({k:v[vr] for k,v in blocks.items()}, dictionary["reaction_centers"],
                                {k:v[vr] for k,v in masks.items()}, anchor["modalities"])
    batch_checks["reaction_raw_equal"] = bool(torch.equal(full_encoded, encoded))
    aq = reaction_anchor_features(encoded, anchor["train_reactions"], anchor["reaction_temperature"], dictionary["reaction_neighbors"])
    alt_aq = torch.cat([reaction_anchor_features(encoded[start:start+193], anchor["train_reactions"],
                       anchor["reaction_temperature"], dictionary["reaction_neighbors"])
                       for start in range(0,len(encoded),193)])
    batch_checks["reaction_anchor_equal"] = bool(torch.equal(aq, alt_aq))
    atomic_json(args.output / "batch_invariance.json", batch_checks)
    if not all(batch_checks.values()):
        raise ValueError(f"Semantic endpoint batch invariance failed: {batch_checks}")
    anchor_scores = aq @ ae.T
    # Check that the saved anchor reproduces its previously frozen validation score.
    previous_alpha = anchor["alpha"]
    previous_eval = evaluate_scores((1-previous_alpha)*(fq@fe.T)+previous_alpha*anchor_scores, truth)
    previous_summary = anchor["selected_validation"]["validation"]
    anchor_max_difference = max(abs(previous_eval["summary"][d][s][k]-value)
        for d, strata in previous_summary.items() for s, values in strata.items()
        for k,value in values.items() if isinstance(value,(int,float)))
    atomic_json(args.output / "anchor_reconstruction.json", dict(max_metric_difference=anchor_max_difference,
        summary=previous_eval["summary"], reference=previous_summary))
    if anchor_max_difference > 1e-6:
        raise ValueError(f"Saved anchor reconstruction differs: {anchor_max_difference}")
    per_seed, graph_features, graph_reconstruction = [], {}, {}
    for (weighting, seed), path in graph_paths.items():
        checkpoint = torch.load(state_dir/f"graph_{weighting}_{seed}.pt", map_location=device, weights_only=False)
        if checkpoint["registry"]["feature_manifest_sha256"] != sources["features_manifest"]["sha256"]:
            raise ValueError("Graph feature manifest differs from evaluation")
        model = FrozenGeometryResidual(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.inference_mode():
            gq, ge = model.encode_reactions(raw_fq), model.encode_enzymes(raw_fe)
            graph_features[weighting, seed] = (gq.cpu(), ge.cpu())
            graph_scores = gq @ ge.T
            reconstructed = evaluate_scores(graph_scores, truth)["summary"]
            reference = checkpoint["selected_validation"]["validation"]
            difference = max(abs(reconstructed[d][s][k]-value)
                for d,strata in reference.items() for s,values in strata.items()
                for k,value in values.items() if isinstance(value,(int,float)))
            graph_reconstruction[f"{weighting}_{seed}"] = difference
            if difference > 1e-6:
                raise ValueError(f"Saved graph reconstruction differs: {weighting}/{seed}: {difference}")
            for alpha in args.alpha:
                name = f"{weighting}_seed{seed}_anchor{alpha:g}"
                result = evaluate_scores(canonical_dot(compose_embeddings(gq, aq, alpha),
                                                       compose_embeddings(ge, ae, alpha)), truth)
                artifact = save_evaluation(args.output, name, result)
                row = dict(id=name, weighting=weighting, seed=seed, anchor_weight=alpha,
                           graph_selected_step=checkpoint["selected_validation"]["step"], summary=result["summary"], artifacts=[artifact],
                           **selection_summary(result["summary"], baseline["summary"], args.tolerance))
                per_seed.append(row)
                print(json.dumps({k:row[k] for k in ("id", "mean_all_positive_mrr", "eligible")}), flush=True)
    atomic_json(args.output / "graph_reconstruction.json", graph_reconstruction)
    recipes = []
    for weighting in args.weightings:
        for alpha in args.alpha:
            rows = [r for r in per_seed if r["weighting"] == weighting and r["anchor_weight"] == alpha]
            if len(rows) != len(args.seeds):
                raise ValueError("Incomplete seed recipe")
            summary = mean_summaries([row["summary"] for row in rows])
            recipes.append(dict(id=f"{weighting}_anchor{alpha:g}", weighting=weighting, anchor_weight=alpha,
                summary=summary, per_seed=rows, seed_std=float(np.std([r["mean_all_positive_mrr"] for r in rows], ddof=1)),
                **selection_summary(summary, baseline["summary"], args.tolerance)))
    eligible = [r for r in recipes if r["eligible"]]
    if not eligible:
        raise ValueError("No eligible recipe; baseline remains unmodified")
    selected = max(eligible, key=lambda r:r["mean_all_positive_mrr"])
    unrestricted = max(recipes, key=lambda r:r["mean_all_positive_mrr"])
    selected_graphs = [graph_features[selected["weighting"],seed] for seed in args.seeds]
    ensemble_q = ensemble_embeddings([q.to(device) for q,e in selected_graphs], aq, selected["anchor_weight"])
    ensemble_e = ensemble_embeddings([e.to(device) for q,e in selected_graphs], ae, selected["anchor_weight"])
    ensemble_evaluation = evaluate_scores(canonical_dot(ensemble_q, ensemble_e), truth)
    ensemble_artifact = save_evaluation(args.output, "selected_fixed_three_seed_ensemble", ensemble_evaluation)
    ensemble_record = dict(summary=ensemble_evaluation["summary"], artifacts=[ensemble_artifact],
        selection_used=False, weighting="equal score weights across all preregistered seeds",
        **selection_summary(ensemble_evaluation["summary"], baseline["summary"], args.tolerance))
    atomic_json(args.output / "registry.json", dict(per_seed=per_seed, recipes=recipes))
    atomic_json(args.output / "selection.json", dict(selected=selected, unrestricted=unrestricted, baseline=baseline["summary"],
                baseline_fp32_sensitivity=baseline_fp32["summary"],
                fixed_three_seed_ensemble=ensemble_record,
                anchor_state="states/anchor_selected.pt", anchor_dictionary="states/anchor_dictionary.pt",
                complete=True, test_used=False, case1_used=False, elapsed_seconds=time.time()-started))
    selected_features = dict(anchor_reactions=aq.cpu(), anchor_enzymes=ae.cpu(),
        graphs={str(seed):dict(reactions=graph_features[selected["weighting"],seed][0], enzymes=graph_features[selected["weighting"],seed][1]) for seed in args.seeds},
        alpha=selected["anchor_weight"], reaction_ids=catalog["validation_reactions"], enzyme_ids=catalog["validation_candidates"],
        inference="compose_embeddings(graph_endpoint, anchor_endpoint, alpha); fixed score ensemble: ensemble_embeddings([seed endpoints], anchor_endpoint, alpha)")
    torch.save(selected_features, args.output/"selected_validation_features.pt")
    with np.load(baseline_artifact) as source:
        baseline_arrays = {key:source[key] for key in source.files}
    selected_arrays = []
    for row in selected["per_seed"]:
        with np.load(row["artifacts"][0]) as source:
            selected_arrays.append({key:source[key] for key in source.files})
    atomic_json(args.output / "bootstrap.json", bootstrap_selected(baseline_arrays, selected_arrays))
    print(json.dumps(dict(selected=selected["id"], unrestricted=unrestricted["id"], elapsed_seconds=time.time()-started)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=ROOT/"runs/generalization_20260919_2251")
    parser.add_argument("--features", type=Path, default=ROOT/"runs/generalization_20260919_2251/features")
    parser.add_argument("--anchor", type=Path, default=ROOT/"runs/generalization_20260919_2251/semantic_anchors")
    parser.add_argument("--output", type=Path, default=ROOT/"runs/generalization_20260919_2251/composition")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,17,73])
    parser.add_argument("--weightings", nargs="+", default=["uniform", "reaction_balanced"])
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.,.1,.25,.5,.75,1.])
    parser.add_argument("--tolerance", type=float, default=.005)
    args = parser.parse_args()
    if len(set(args.seeds)) < 2 or len(args.seeds) != len(set(args.seeds)) or any(not 0 <= a <= 1 for a in args.alpha):
        parser.error("Unique multiple seeds and valid mixture weights required")
    run(args)
