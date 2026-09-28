#!/usr/bin/env python3
"""Phase2 exploratory train/validation-only support-gated residual screen.

Each gate uses only one endpoint and fixed training anchors, preserving an
independently encodable dual representation. No held-out panel is opened.
"""
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
sys.path.insert(0, str(ROOT / "scripts"))
from horizyn.generalization_residual import FrozenGeometryResidual
from horizyn.semantic_anchors import row_unit, fit_center, centered_unit, reaction_features
from scripts.generalization_export import atomic_json, identity
from scripts.generalization_full_graph import validation_data, eligible, selection_value
from scripts.generalization_metrics import evaluate_scores

DIRECTIONS = ("reaction_to_enzyme", "enzyme_to_reaction")


@torch.inference_mode()
def nearest_support(query, reference, exclude=None, batch_size=512):
    reference = reference.double()
    output = []
    for start in range(0, len(query), batch_size):
        scores = query[start:start + batch_size].double() @ reference.T
        if exclude is not None:
            positions = exclude[start:start + batch_size]
            scores[torch.arange(len(scores), device=scores.device), positions] = float("-inf")
        output.append(scores.amax(1).float().clamp(-1, 1))
    return torch.cat(output)


def support_gate(similarity, low, high, cap):
    if high < low:
        raise ValueError("Training support quantiles are reversed")
    if high == low:
        # Exact duplicate training feature vectors can collapse upper
        # quantiles. In that case only support at that level receives weight.
        return (similarity >= high).to(similarity.dtype) * cap
    return ((similarity - low) / (high - low)).clamp(0, 1) * cap


def balanced_value(summary):
    return float(np.mean([summary[direction][stratum]["reactzyme_mrr"]
                          for direction in DIRECTIONS for stratum in ("seen_reaction", "unseen_reaction")]))


def gate_candidates():
    values = [dict(label=f"constant_{cap:g}", space="constant", cap=cap) for cap in (0., .25, .5, 1.)]
    values += [dict(label=f"pointwise_norm_{bound:g}", space="pointwise_norm", cap=bound)
               for bound in (.03, .1, .2)]
    for space in ("raw", "f3"):
        for endpoints in ("both", "reaction", "enzyme"):
            for lower in (.25, .5):
                for cap in (.5, 1.):
                    values.append(dict(label=f"{space}_{endpoints}_q{int(lower * 100)}_cap{cap:g}",
                        space=space, endpoints=endpoints, lower_quantile=lower, upper_quantile=.95, cap=cap))
    return values


def diagnostic_curves(base, graph, raw_support, displacement, reaction_seen):
    curves = []
    for direction in DIRECTIONS:
        endpoint = "reaction" if direction == "reaction_to_enzyme" else "enzyme"
        density = raw_support[endpoint].cpu().numpy()
        movement = displacement[endpoint].cpu().numpy()
        for stratum in ("all", "seen_reaction", "unseen_reaction"):
            before = base["per_query"][direction][stratum]
            after = graph["per_query"][direction][stratum]
            ids = before["query_index"]
            if not np.array_equal(ids, after["query_index"]):
                raise ValueError("Unpaired validation diagnostic queries")
            if not len(ids):
                continue
            # Fixed cosine bands are descriptive, not selection cutoffs.
            boundaries = np.array([-np.inf, .25, .5, .75, .9, .99, np.inf])
            bins = np.digitize(density[ids], boundaries[1:-1])
            for group in range(len(boundaries) - 1):
                use = bins == group
                if not use.any():
                    continue
                curves.append(dict(direction=direction, stratum=stratum,
                    cosine_interval=[str(boundaries[group]), str(boundaries[group + 1])],
                    queries=int(use.sum()), mean_support=float(density[ids[use]].mean()),
                    mean_graph_movement=float(movement[ids[use]].mean()),
                    baseline_mrr=float(before["reactzyme_mrr"][use].mean()),
                    graph_mrr=float(after["reactzyme_mrr"][use].mean()),
                    paired_mrr_change=float((after["reactzyme_mrr"][use] - before["reactzyme_mrr"][use]).mean())))
    return curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--graph-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration-proteins", type=int, default=4096)
    parser.add_argument("--max-drop", type=float, default=.005)
    parser.add_argument("--inference-precision", choices=("legacy_fp32", "stable_fp64"), default="stable_fp64")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new phase2 output directory")
    if args.calibration_proteins < 2:
        parser.error("At least two calibration proteins required")
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, args.output / "source.py")
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    candidates = gate_candidates()
    atomic_json(args.output / "registry.json", dict(schema="phase2_density_gate_screen_v1", test_used=False,
        status="Exploratory phase2 after phase1 held-out outcomes were observed; this script opens training/validation only",
        candidates=candidates, selection="Equal-weight mean of seen/unseen reaction all-positive MRR across both directions, with four baseline guards",
        guard_tolerance=args.max_drop, training_quantiles="Leave-one-out nearest training neighbor; all training reactions and deterministic 4096-protein sample",
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        sources={name: identity(args.features / name, True) for name in ("catalog.json", "manifest.json", "pairs.npz", "f3_features.npz", "reaction_features.npz", "protein_mean.h5")},
        graph_checkpoint=identity(args.graph_checkpoint, True), script=identity(__file__, True)))
    started = time.monotonic()
    catalog = json.loads((args.features / "catalog.json").read_text())
    with np.load(args.features / "pairs.npz") as source:
        pairs = {key: source[key] for key in ("train", "validation")}
    tr, te = np.unique(pairs["train"][:, 0]), np.unique(pairs["train"][:, 1])
    vr, ve, truth = validation_data(catalog, pairs)
    with np.load(args.features / "f3_features.npz") as source:
        base_e = torch.tensor(source["proteins"], device=device)
        base_r = torch.tensor(source["reactions"], device=device)
        train_r = torch.tensor(source["train_reactions"], device=device)
    with h5py.File(args.features / "protein_mean.h5") as source:
        if source["ids"].asstr()[:].tolist() != catalog["proteins"] or not source["complete"][:].all():
            raise ValueError("Raw protein catalog mismatch or incomplete means")
        raw_e = torch.tensor(source["vectors"][:], device=device)
    e_center = fit_center(raw_e, te)
    raw_e = centered_unit(raw_e, e_center)
    modalities = ("t5v2", "unimol2", "chiro", "chemistry")
    with np.load(args.features / "reaction_features.npz") as source:
        blocks = {key: torch.tensor(source[key], device=device) for key in modalities}
        masks = {key: torch.tensor(source[key + "_mask"], device=device) for key in modalities}
    centers = {key: fit_center(blocks[key], tr, masks[key]) for key in modalities}
    raw_r = reaction_features(blocks, centers, masks, modalities)
    training = dict(raw=dict(reaction=raw_r[tr], enzyme=raw_e[te]),
                    f3=dict(reaction=row_unit(train_r), enzyme=row_unit(base_e[te])))
    validation = dict(raw=dict(reaction=raw_r[vr], enzyme=raw_e[ve]),
                      f3=dict(reaction=row_unit(base_r[vr]), enzyme=row_unit(base_e[ve])))
    rng = np.random.default_rng(20260920)
    calibration_indices = dict(reaction=np.arange(len(tr)),
        enzyme=np.sort(rng.choice(len(te), min(args.calibration_proteins, len(te)), replace=False)))
    support, thresholds, training_support = {}, {}, {}
    for space in ("raw", "f3"):
        support[space], thresholds[space], training_support[space] = {}, {}, {}
        for endpoint in ("reaction", "enzyme"):
            anchors = training[space][endpoint]
            chosen = torch.tensor(calibration_indices[endpoint], device=device)
            loo = nearest_support(anchors[chosen], anchors, exclude=chosen)
            support[space][endpoint] = nearest_support(validation[space][endpoint], anchors)
            training_support[space][endpoint] = loo
            thresholds[space][endpoint] = {str(q): float(torch.quantile(loo, q)) for q in (.25, .5, .95)}
            print(json.dumps(dict(space=space, endpoint=endpoint, training_quantiles=thresholds[space][endpoint],
                elapsed_seconds=time.monotonic() - started)), flush=True)
    snapshot = torch.load(args.graph_checkpoint, map_location=device, weights_only=False)
    model = FrozenGeometryResidual(**snapshot["model_config"]).to(device).eval()
    model.load_state_dict(snapshot["state_dict"])
    if args.inference_precision == "stable_fp64":
        model.double()
    with torch.inference_mode():
        original = dict(enzyme=base_e[ve], reaction=base_r[vr])
        tower_inputs = {key: value.double() if args.inference_precision == "stable_fp64" else value for key,value in original.items()}
        delta = dict(enzyme=model.scale * model.enzyme(tower_inputs["enzyme"]),
                     reaction=model.scale * model.reaction(tower_inputs["reaction"]))
        def normalize(key, gates):
            if args.inference_precision == "stable_fp64":
                return F.normalize(original[key].double() + gates.double()[:, None] * delta[key], dim=-1).float()
            return F.normalize(original[key] + gates[:, None] * delta[key], dim=-1)
        baseline_vectors = {key: F.normalize(value, dim=-1) for key, value in original.items()}
        graph_vectors = {key: normalize(key, torch.ones(len(original[key]),device=device)) for key in original}
        movement = {key: (graph_vectors[key] - baseline_vectors[key]).norm(dim=1) for key in original}
        def evaluation(vectors):
            scores = (vectors["reaction"].double() @ vectors["enzyme"].double().T).float()
            return evaluate_scores(scores, truth)
        baseline_eval = evaluation(baseline_vectors)
        graph_eval = evaluation(graph_vectors)
    atomic_json(args.output / "diagnostics.json", dict(training_quantiles=thresholds,
        curves=diagnostic_curves(baseline_eval, graph_eval, support["raw"], movement, truth["reaction_seen"]),
        feature_geometry="Raw per-modality unit features centered using train only, then unit concatenation; F3 features unit normalized",
        scoring="Independent normalized endpoint vectors; FP64 dot rounded to FP32; stable candidate-index ties"))
    np.savez_compressed(args.output / "validation_support.npz", **{
        f"{space}_{endpoint}": value.cpu().numpy() for space, mapping in support.items() for endpoint, value in mapping.items()},
        **{f"graph_movement_{key}": value.cpu().numpy() for key, value in movement.items()},
        **{f"training_{space}_{endpoint}": value.cpu().numpy() for space, mapping in training_support.items() for endpoint, value in mapping.items()})
    records = []
    best = best_aggregate = None
    with torch.inference_mode():
        for candidate in candidates:
            gates = {}
            for endpoint in original:
                if candidate["space"] == "constant":
                    gates[endpoint] = torch.full((len(original[endpoint]),), candidate["cap"], device=device)
                elif candidate["space"] == "pointwise_norm":
                    # Algebraically clip delta / ||base|| before adding to
                    # unit(base), then normalize. Retaining the raw base here
                    # avoids an unnecessary extra normalization roundoff.
                    gates[endpoint] = (candidate["cap"] * original[endpoint].norm(dim=1)
                                       / delta[endpoint].norm(dim=1).clamp_min(1e-12)).clamp(max=1)
                elif candidate["endpoints"] in ("both", endpoint):
                    bounds = thresholds[candidate["space"]][endpoint]
                    gates[endpoint] = support_gate(support[candidate["space"]][endpoint],
                        bounds[str(candidate["lower_quantile"])], bounds[str(candidate["upper_quantile"])], candidate["cap"])
                else:
                    gates[endpoint] = torch.full((len(original[endpoint]),), candidate["cap"], device=device)
            encoded = {key: normalize(key, gates[key]) for key in original}
            result = (baseline_eval if candidate["label"] == "constant_0" and args.inference_precision == "legacy_fp32"
                      else graph_eval if candidate["label"] == "constant_1" else evaluation(encoded))
            summary = result["summary"]
            record = dict(candidate=candidate, validation=summary, balanced_value=balanced_value(summary),
                aggregate_value=selection_value(summary), eligible=eligible(summary, baseline_eval["summary"], args.max_drop),
                gates={key: dict(mean=float(value.mean()), fraction_zero=float((value == 0).float().mean()),
                                fraction_capped=float((value == (1. if candidate["space"] == "pointwise_norm" else candidate["cap"])).float().mean()),
                                maximum_residual_to_base_norm=float((value * delta[key].norm(dim=1) / original[key].norm(dim=1)).max())) for key, value in gates.items()},
                elapsed_seconds=time.monotonic() - started)
            records.append(record)
            if record["eligible"] and (best is None or record["balanced_value"] > best["balanced_value"]):
                best = record
                np.savez(args.output / "selected_validation_features.npz", proteins=encoded["enzyme"].cpu().numpy(), reactions=encoded["reaction"].cpu().numpy())
            if record["eligible"] and (best_aggregate is None or record["aggregate_value"] > best_aggregate["aggregate_value"]):
                best_aggregate = record
            atomic_json(args.output / "results.json", dict(records=records, selected=best, aggregate_diagnostic=best_aggregate,
                test_used=False, original_f3_baseline=baseline_eval["summary"], inference_precision=args.inference_precision,
                baseline_note="Guards reference original native-FP32 normalized F3. constant_0 separately reports the phase2 normalization-only sensitivity.",
                selection="Balanced seen/unseen all-positive MRR; fixed four guards"))
            print(json.dumps({key: record[key] for key in ("candidate", "balanced_value", "aggregate_value", "eligible", "elapsed_seconds")}), flush=True)
    torch.save(dict(selected_gate=best["candidate"], thresholds=thresholds,
                    inference_precision=args.inference_precision,
                    raw_protein_center=e_center.cpu(), reaction_centers={key: value.cpu() for key, value in centers.items()},
                    calibration_indices=calibration_indices, graph_checkpoint=identity(args.graph_checkpoint, True)),
               args.output / "selected_gate.pt")
    atomic_json(args.output / "complete.json", dict(test_used=False, selected=best["candidate"],
        selected_balanced_value=best["balanced_value"], elapsed_seconds=time.monotonic() - started,
        results=identity(args.output / "results.json", True)))


if __name__ == "__main__":
    main()
