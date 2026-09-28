#!/usr/bin/env python3
"""Prescore-pinned generic training-reaction affinity control for Case 1.

This supplements, and never modifies, the original five-method sampled scan.
Unknown background sequences remain unassayed; the mean query is descriptive,
not an inactivity label or an exchangeable biological null.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from horizyn.generalization_phase2 import ComposedPhase2Encoder
from horizyn.generalization_phase4 import HybridPhase4Encoder
from horizyn.generalization_retrieval import checked_artifact
from generalization_large_case1_evaluate import (
    authenticate, identity, reference_ranks, tier_summary, write_csv, write_json,
)

RUN = ROOT / "runs/generalization_20260919_2251"
METHODS = ("phase2", "phase4")
CUTS = (25, 100, 1000, 10000)


def checked(record):
    return checked_artifact(record)


def require_completed_primary():
    scan = RUN / "large_case1_scan"
    audit_path = scan / "completion_audit.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("all_exact") is not True:
        raise ValueError("Require passing independent numerical completion audit")
    primary_path = RUN / "large_case1_evaluation/complete.json"
    primary = json.loads(primary_path.read_text())
    if primary.get("schema") != "large_case1_evaluation_complete_v1":
        raise ValueError("Require the original successful primary evaluation")
    # Bind the gate to this exact completed scan, rather than just a status flag.
    complete = identity(scan / "complete.json")
    if audit["complete"]["sha256"] != complete["sha256"]:
        raise ValueError("Numerical audit belongs to another completed scan")
    checked(audit["plan"])
    checked(audit["protocol"])
    if primary["scan_receipt"]["sha256"] != complete["sha256"]:
        raise ValueError("Primary evaluation belongs to another scan")
    for value in primary["outputs"].values():
        checked(value)
    return {"primary_evaluation": identity(primary_path), "numerical_audit": identity(audit_path)}


@torch.inference_mode()
def prepare(args):
    out = args.output
    if out.exists():
        raise ValueError("Use a fresh prescore protocol directory")
    if (RUN / "large_case1_scan/complete.json").exists() or (RUN / "large_case1_evaluation").exists():
        raise ValueError("Pin this supplemental control before completed background outcomes")
    out.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    paths = {
        "implementation": Path(__file__),
        "rank_helper": ROOT / "scripts/generalization_large_case1_evaluate.py",
        "scan_protocol": RUN / "large_case1_scan/protocol.json",
        "catalog": RUN / "features/catalog.json",
        "pairs": RUN / "features/pairs.npz",
        "native_training_features": RUN / "features/f3_features.npz",
        "raw_reaction_features": RUN / "features/reaction_features.npz",
        "previous_p2_means": RUN / "post_evaluation_calibration/train_means.pt",
        "previous_mean_protocol": RUN / "post_evaluation_calibration/protocol.json",
        "phase2_bundle": RUN / "phase2/models/reaction_smi/seed42/bundle.json",
        "phase4_bundle": RUN / "phase4/models/reaction_smi/seed42/bundle.json",
    }
    frozen_sources = {}
    for phase in METHODS:
        path = RUN / phase / "frozen_recipe.json"
        paths[phase + "_freeze"] = path
        for item in json.loads(path.read_text())["implementation_sources"]:
            frozen_sources[str(checked(item))] = item
    scan_protocol = json.loads(paths["scan_protocol"].read_text())
    label_sources = scan_protocol["evaluation_label_sources"]
    for item in label_sources:
        checked(item)  # Hash only. No labels are parsed during preparation.
    catalog = json.loads(paths["catalog"].read_text())
    with np.load(paths["pairs"], allow_pickle=False) as pairs:
        training = pairs["train"]
    tr = np.unique(training[:, 0])
    if [catalog["reactions"][i] for i in tr] != catalog["train_reactions"] or len(tr) != 6977:
        raise ValueError("Require actual Reaction-Sim training-edge endpoints")
    with np.load(paths["native_training_features"], allow_pickle=False) as data:
        native = torch.tensor(data["train_reactions"], device=args.device)
    if len(native) != len(tr):
        raise ValueError("Compact native training reaction order mismatch")
    with np.load(paths["raw_reaction_features"], allow_pickle=False) as data:
        names = ("t5v2", "unimol2", "chiro", "chemistry")
        blocks = {k: torch.tensor(data[k][tr], device=args.device) for k in names}
        masks = {k: torch.tensor(data[k + "_mask"][tr], device=args.device) for k in names}
    previous = torch.load(paths["previous_p2_means"], map_location="cpu", weights_only=False)
    if previous["protocol"]["sha256"] != identity(paths["previous_mean_protocol"])["sha256"]:
        raise ValueError("Previous training mean protocol changed")
    if not np.array_equal(previous["train_reaction_indices"], tr):
        raise ValueError("Previous reaction mean used different training selectors")
    means, checks = {}, {}
    for phase, cls in (("phase2", ComposedPhase2Encoder), ("phase4", HybridPhase4Encoder)):
        model, _ = cls.from_bundle(paths[phase + "_bundle"], args.device)
        total = None
        for start in range(0, len(tr), 256):
            stop = min(start + 256, len(tr))
            endpoint = model.encode_reactions(native[start:stop],
                {k: v[start:stop] for k, v in blocks.items()},
                {k: v[start:stop] for k, v in masks.items()}, batch_size=512)
            if not torch.isfinite(endpoint).all() or endpoint.dtype != torch.float32:
                raise ValueError("Invalid training endpoints")
            if total is None:
                total = torch.zeros(endpoint.shape[1], dtype=torch.float64, device=args.device)
            total += endpoint.double().sum(0)
        mean = (total / len(tr)).cpu()
        if phase == "phase2":
            checks["phase2_fp64_mean_exactly_replays_prior_training_only_fit"] = torch.equal(mean, previous["reaction_mean"])
            if not checks["phase2_fp64_mean_exactly_replays_prior_training_only_fit"]:
                raise ValueError("Independent P2 training mean recomputation differs")
        means[phase] = mean
        del model
    torch.save({"means_fp64": means, "queries_fp32": {k: v.float() for k, v in means.items()},
                "training_reaction_indices": tr, "training_reaction_count": len(tr)}, out / "means.pt")
    plan = dict(schema="large_case1_mean_query_control_v1",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        background_scores_or_outcomes_read=False, labels_used_for_control_selection=False,
        supplemental=True, original_primary_protocol_changed=False,
        methods=list(METHODS), seeds=[42], split="reaction_smi", cutoffs=list(CUTS),
        tiers=["primary_papers", "primary_papers_and_patents"],
        query="Uniform arithmetic mean of every actual training-edge reaction endpoint, without normalization; FP64 fixed-order 256-row sums then divide by6977 and cast once toFP32.",
        representation="Each method uses its own unchanged complete composed reaction endpoints. The existing training-native cache is used, including its documented BF16-inference/FP16-storage history; no fresh native extraction.",
        enzyme_index="Unchanged shared P2/P4 dense+CSR index from the fixed completed scan; two mean queries scored together using frozen score_index.",
        numeric="FP64 dense+CSR dot and final FP32 scores; uniform exact-tie expected recovery and reciprocal ranks, plus stable/best/worst ranks.",
        primary_comparison="Compare each fixed method's actual Case1 query and its training-mean query on the identical full sampled catalog; all24 primary known catalysts retained, with12 paper positives separately.",
        interpretation="Descriptive generic-training-reaction-affinity ablation, not a formal biological null, inactivity test, calibrated confidence or independent new panel. No model/threshold/candidate selection follows results.",
        execution_gate="Completed fixed scan, independent numerical completion_audit all_exact, then successful authenticated original primary evaluation; no residue-feature reread.",
        sources={k: identity(v) for k, v in paths.items()}, frozen_sources=list(frozen_sources.values()),
        label_sources=label_sources, means=identity(out / "means.pt"), checks=checks)
    write_json(out / "plan.json", plan)
    print(json.dumps(identity(out / "plan.json")))


@torch.inference_mode()
def run(args):
    out = args.output
    plan = json.loads((out / "plan.json").read_text())
    if (out / "complete.json").exists():
        raise ValueError("Existing mean-query result")
    for item in list(plan["sources"].values()) + plan["frozen_sources"] + plan["label_sources"] + [plan["means"]]:
        checked(item)
    gates = require_completed_primary()
    protocol, catalog, literature, mapping, original_scores, provenance = authenticate(
        RUN / "large_case1_scan", RUN / "case1_audit")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model, _ = ComposedPhase2Encoder.from_bundle(checked(plan["sources"]["phase2_bundle"]), args.device)
    saved = torch.load(checked(plan["means"]), map_location="cpu", weights_only=False)
    queries = torch.stack([saved["queries_fp32"][k] for k in METHODS]).to(args.device)
    scan_complete = json.loads((RUN / "large_case1_scan/complete.json").read_text())
    pieces, index_sources, offset = [], [], 0
    for receipt_record in scan_complete["chunk_receipts"]:
        receipt = json.loads(checked(receipt_record).read_text())
        item = receipt["outputs"]["enzyme_index"]
        index_path = checked(item)
        index = torch.load(index_path, map_location=args.device, weights_only=False)
        values = model.score_index(queries, index).cpu().numpy()
        if values.shape != (2, receipt["rows"]) or not np.isfinite(values).all():
            raise ValueError("Invalid mean-query score chunk")
        pieces.append(values)
        index_sources.append(item)
        offset += receipt["rows"]
    if offset != len(catalog["proteins"]):
        raise ValueError("Incomplete candidate coverage")
    scores = np.concatenate(pieces, axis=1)
    np.savez(out / "scores.npz", **{k: scores[i] for i, k in enumerate(METHODS)})
    from generalization_external_evaluate import case1_metadata
    metadata = case1_metadata(RUN / "case1_audit")
    if metadata["catalog"]["proteins"] != literature["proteins"]:
        raise ValueError("Literature sequence order changed")
    indices = [mapping[p]["candidate_index"] for p in literature["proteins"]]
    summaries, rank_rows = {}, []
    for method_index, method in enumerate(METHODS):
        summaries[method] = {}
        for query_name, score in (("actual_case1", original_scores[method]), ("training_mean", scores[method_index])):
            ranks, _ = reference_ranks(score, indices)
            summaries[method][query_name] = {
                tier: tier_summary(ranks, metadata["indices"][tier], len(score), CUTS)
                for tier in plan["tiers"]}
            for i in sorted(metadata["indices"]["primary_papers_and_patents"]):
                rank_rows.append(dict(method=method, query=query_name, representative_id=literature["proteins"][i],
                    primary_paper=i in metadata["indices"]["primary_papers"],
                    **{k: float(v[i]) if k in ("score", "expected_reciprocal_rank") else int(v[i])
                       for k, v in ranks.items()}))
    write_json(out / "summary.json", dict(candidate_count=offset, metrics=summaries,
        interpretation=plan["interpretation"], full_pool_unchanged=True, unknowns_treated_as_inactive=False))
    write_csv(out / "known_positive_ranks.csv", rank_rows)
    write_json(out / "complete.json", dict(schema="large_case1_mean_query_complete_v1",
        plan=identity(out / "plan.json"), gates=gates, scan=identity(RUN / "large_case1_scan/complete.json"),
        original_score_and_catalog_authentication=provenance, index_sources=index_sources,
        outputs={k: identity(out / k) for k in ("scores.npz", "summary.json", "known_positive_ranks.csv")}))
    print(json.dumps({"status": "complete", "candidate_count": offset, "methods": list(METHODS)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=RUN / "large_case1_mean_query")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.action](args)
