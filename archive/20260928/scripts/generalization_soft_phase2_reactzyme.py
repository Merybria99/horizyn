#!/usr/bin/env python3
"""Select new residual heads inside the frozen ReactZyme phase-2 composition."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horizyn.generalization_phase2 import ComposedPhase2Encoder
from horizyn.generalization_residual import FrozenGeometryResidual
from horizyn.generalization_retrieval import canonical_dot, sha256
from generalization_transport_reactzyme import load_features
from generalization_full_graph import validation_data, atomic_json, eligible
from generalization_smooth_anchors import robust_value
from generalization_metrics import evaluate_scores
from generalization_phase2_official_evaluate import checked_official_truth

RUN = ROOT / "runs/generalization_20260919_2251"


def load_residual(path, manifest_sha, device, precision):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved["registry"]["feature_manifest_sha256"] != manifest_sha or saved["registry"]["test_used"]:
        raise ValueError("Residual training lineage differs from the parent phase-2 model")
    model = FrozenGeometryResidual(**saved["model_config"]).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    if precision == "stable_fp64":
        model.double()
    return model.eval().requires_grad_(False)


def replace_dense(parent, residual, base, composed, gates, endpoint):
    dtype = next(residual.parameters()).dtype
    delta = residual.scale * getattr(residual, endpoint)(base.to(dtype))
    dense = parent.density._normalize(base, delta, gates)
    # Preserve the already weighted semantic coordinates bit for bit.
    return torch.cat((math.sqrt(1 - parent.alpha) * dense, composed[:, base.shape[1]:]), dim=1)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("reaction_smi", "enzyme_smi", "time"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.output
    while not (out / "training/complete.json").exists():
        time.sleep(10)
    if (out / "composition_registry.json").exists():
        raise FileExistsError("Composition already started")
    features = RUN / {"reaction_smi": "features", "enzyme_smi": "features_enzyme_smi", "time": "features_time"}[args.split]
    bundle = RUN / "phase2/models" / args.split / "seed42/bundle.json"
    steps = [5, 20, 50, 100]
    registry = dict(created_utc=datetime.now(timezone.utc).isoformat(), split=args.split,
        parent_bundle=dict(path=str(bundle), sha256=sha256(bundle)),
        feature_manifest_sha256=sha256(features / "manifest.json"), steps=steps,
        selection="Maximum balanced seen/unseen validation all-positive MRR; both directions aggregate and unseen within 0.005 of parent",
        fixed_parent_gate_and_semantic_dictionary=True, retains_sleec=True,
        test_used_for_selection=False, source_sha256=sha256(__file__))
    atomic_json(out / "composition_registry.json", registry)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    model, spec = ComposedPhase2Encoder.from_bundle(bundle, args.device)
    if spec["feature_manifest_sha256"] != registry["feature_manifest_sha256"]:
        raise ValueError("Base feature lineage mismatch")
    catalog, be, br, means, blocks, masks = load_features(features, args.device)
    with np.load(features / "pairs.npz") as data:
        pairs = {k: data[k] for k in ("train", "validation")}
    vr, ve, truth = validation_data(catalog, pairs)
    e, ed = model.encode_enzymes(be[ve], means[ve], batch_size=256, return_diagnostics=True)
    r, rd = model.encode_reactions(br[vr], {k:v[vr] for k,v in blocks.items()},
        {k:v[vr] for k,v in masks.items()}, batch_size=256, return_diagnostics=True)
    baseline = evaluate_scores(canonical_dot(r, e), truth)["summary"]
    records = [dict(step=0, parent_retained=True, value=robust_value(baseline),
                    eligible=True, validation=baseline, checkpoint=None)]
    best = records[0]
    for step in steps:
        path = out / f"training/step{step:04d}.pt"
        residual = load_residual(path, registry["feature_manifest_sha256"], args.device,
                                 model.density.inference_precision)
        ee = replace_dense(model, residual, be[ve], e, ed["gate_scale"], "enzyme")
        rr = replace_dense(model, residual, br[vr], r, rd["gate_scale"], "reaction")
        summary = evaluate_scores(canonical_dot(rr, ee), truth)["summary"]
        record = dict(step=step, parent_retained=False, value=robust_value(summary),
                      eligible=eligible(summary, baseline, .005), validation=summary,
                      checkpoint=dict(path=str(path.resolve()), sha256=sha256(path)))
        records.append(record)
        if record["eligible"] and record["value"] > best["value"]:
            best = record
        print(json.dumps(dict(step=step, value=record["value"], eligible=record["eligible"])), flush=True)
        atomic_json(out / "composition_validation.json", dict(records=records, selected=best))
        del residual, ee, rr
    selection = dict(selected=best, registry_sha256=sha256(out / "composition_registry.json"),
                     selected_utc=datetime.now(timezone.utc).isoformat(), test_used_for_selection=False)
    atomic_json(out / "composition_selected.json", selection)
    del be, br, means, blocks, masks, e, r, ed, rd
    test = RUN / f"features_test_{args.split}"
    tc, be, br, means, blocks, masks = load_features(test, args.device)
    e, ed = model.encode_enzymes(be, means, batch_size=256, return_diagnostics=True)
    r, rd = model.encode_reactions(br, blocks, masks, batch_size=256, return_diagnostics=True)
    if best["checkpoint"] is not None:
        if sha256(best["checkpoint"]["path"]) != best["checkpoint"]["sha256"]:
            raise ValueError("Selected residual changed before test prediction")
        residual = load_residual(best["checkpoint"]["path"], registry["feature_manifest_sha256"],
                                 args.device, model.density.inference_precision)
        ee = replace_dense(model, residual, be, e, ed["gate_scale"], "enzyme")
        rr = replace_dense(model, residual, br, r, rd["gate_scale"], "reaction")
    else:
        ee, rr = e, r
    score, base_score = canonical_dot(rr, ee), canonical_dot(r, e)
    np.savez(out / "test_scores.npz", selected=score.cpu().numpy(), baseline_phase2=base_score.cpu().numpy())
    provenance = {}
    checked, edges = checked_official_truth(test, args.split, spec["frozen_recipe"]["sha256"], provenance)
    if checked != tc:
        raise ValueError("Official test catalog mismatch")
    truth = dict(reaction_index=edges[:, 0], enzyme_index=edges[:, 1])
    summary = dict(split=args.split, selected_step=best["step"], parent_retained=best["parent_retained"],
        selected=evaluate_scores(score, truth)["summary"],
        baseline_phase2=evaluate_scores(base_score, truth)["summary"], truth_provenance=provenance,
        selection_sha256=sha256(out / "composition_selected.json"),
        scores_sha256=sha256(out / "test_scores.npz"), test_used_for_selection=False,
        test_results_exploratory=True)
    atomic_json(out / "test_summary.json", summary)
    print(json.dumps(dict(selected_step=best["step"], test={d:summary["selected"][d]["all"]["reactzyme_mrr"]
                     for d in ("reaction_to_enzyme", "enzyme_to_reaction")})), flush=True)


if __name__ == "__main__":
    main()
