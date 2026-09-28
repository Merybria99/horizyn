#!/usr/bin/env python3
"""Train a frozen-F3 residual against the entire training graph on one GPU.

Selection uses standard validation only, both aggregate and unseen-reaction
all-positive MRR guardrails. Test/Case1 labels and features are never opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horizyn.generalization_residual import FrozenGeometryResidual, full_graph_contrastive_loss
from horizyn.generalization_residual import full_graph_decoupled_loss, sampled_smooth_ap_loss
from generalization_metrics import evaluate_scores


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validation_data(catalog, pairs):
    ri = {key: i for i, key in enumerate(catalog["reactions"])}
    ei = {key: i for i, key in enumerate(catalog["proteins"])}
    vr = np.asarray([ri[x] for x in catalog["validation_reactions"]])
    ve = np.asarray([ei[x] for x in catalog["validation_candidates"]])
    reverse_r = {int(g): i for i, g in enumerate(vr)}
    reverse_e = {int(g): i for i, g in enumerate(ve)}
    edges = pairs["validation"]
    truth = {
        "reaction_index": np.asarray([reverse_r[int(x)] for x in edges[:, 0]]),
        "enzyme_index": np.asarray([reverse_e[int(x)] for x in edges[:, 1]]),
        "reaction_seen": np.isin(vr, np.unique(pairs["train"][:, 0])),
        "enzyme_seen": np.isin(ve, np.unique(pairs["train"][:, 1])),
    }
    return vr, ve, truth


def selection_value(summary):
    return np.mean([summary[d]["all"]["reactzyme_mrr"]
                    for d in ("reaction_to_enzyme", "enzyme_to_reaction")]).item()


def eligible(summary, baseline, tolerance):
    return all(summary[d][s]["reactzyme_mrr"] >= baseline[d][s]["reactzyme_mrr"] - tolerance
               for d in ("reaction_to_enzyme", "enzyme_to_reaction")
               for s in ("all", "unseen_reaction"))


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "registry.json").exists():
        raise ValueError("Choose a fresh output directory; completed runs are immutable")
    shutil.copyfile(__file__, args.output / "source.py")
    shutil.copyfile(ROOT / "horizyn/generalization_residual.py", args.output / "model_source.py")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    catalog = json.loads((args.features / "catalog.json").read_text())
    with np.load(args.features / "pairs.npz") as source:
        pairs = {key: source[key] for key in source.files}
    with np.load(args.features / "f3_features.npz") as source:
        # The residual encoders normalize their outputs. Leave cached inputs
        # untouched here so step zero matches exactly one cosine normalization,
        # as in the canonical frozen-cache evaluator. Repeated normalization
        # can move nearly tied floating-point scores across each other.
        base_e = torch.as_tensor(source["proteins"], device=device)
        base_r = torch.as_tensor(source["reactions"], device=device)
        compact_train_r = torch.as_tensor(source["train_reactions"], device=device)
    tr = np.unique(pairs["train"][:, 0])
    te = np.unique(pairs["train"][:, 1])
    global_ri = {key: i for i, key in enumerate(catalog["reactions"])}
    ordered_train_r = [global_ri[key] for key in catalog["train_reactions"]]
    if not np.array_equal(tr, ordered_train_r):
        raise ValueError("Compact train reaction cache order disagrees with catalog")
    r_lookup = {int(g): i for i, g in enumerate(tr)}
    e_lookup = {int(g): i for i, g in enumerate(te)}
    edge_r = torch.tensor([r_lookup[int(x)] for x in pairs["train"][:, 0]], device=device)
    edge_e = torch.tensor([e_lookup[int(x)] for x in pairs["train"][:, 1]], device=device)
    positive_mask = torch.zeros((len(tr), len(te)), device=device, dtype=torch.bool)
    positive_mask[edge_r, edge_e] = True
    if int(positive_mask.sum()) != len(edge_r):
        raise ValueError("Training graph must contain unique positive edges")
    train_e = base_e[te]
    train_e_identity = F.normalize(train_e, dim=-1)
    train_r_identity = F.normalize(compact_train_r, dim=-1)
    biology = None
    biological_labels = getattr(args, 'biological_labels', None)
    biological_weights = {f: getattr(args, 'biology_' + f, 0.) for f in ('ec', 'cofactor', 'mechanism')}
    if biological_labels is not None:
        from horizyn.biological_geometry import BiologicalGeometryLoss
        payload = json.loads(biological_labels.read_text())
        if payload['feature_manifest_sha256'] != sha(args.features / 'manifest.json'):
            raise ValueError('Biological annotation training manifest mismatch')
        biology = BiologicalGeometryLoss(payload, catalog['train_reactions'],
            [catalog['proteins'][int(i)] for i in te], device,
            getattr(args, 'biology_shuffle_seed', None),
            mode=getattr(args, 'biology_mode', 'attraction'), margin=getattr(args, 'biology_margin', .1))
        shutil.copyfile(ROOT / 'horizyn/biological_geometry.py', args.output / 'biology_source.py')
    external_screening = args.selection_method == "external_screening"
    vr, ve, truth = (None, None, None) if external_screening else validation_data(catalog, pairs)
    model = FrozenGeometryResidual(base_e.shape[1], args.hidden, args.scale).to(device)
    warm_start = getattr(args, "warm_start", None)
    warm_start_record = None
    if warm_start is not None:
        initial = torch.load(warm_start, map_location="cpu", weights_only=False)
        if (initial["registry"].get("test_used") is not False or
                initial["registry"]["feature_manifest_sha256"] != sha(args.features / "manifest.json") or
                initial["model_config"] != dict(dimension=model.dimension, hidden=model.hidden, scale=model.scale)):
            raise ValueError("Warm-start residual must match this training split and architecture")
        model.load_state_dict(initial["state_dict"], strict=True)
        warm_start_record = dict(path=str(Path(warm_start).resolve()), sha256=sha(warm_start))
    if getattr(args, "identity_reference", "base") == "initial":
        with torch.no_grad():
            train_e_identity = model.encode_enzymes(train_e).detach()
            train_r_identity = model.encode_reactions(compact_train_r).detach()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    registry = {
        "schema": "generalization_full_graph_v2_single_normalization", "test_used": False,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "feature_manifest_sha256": sha(args.features / "manifest.json"),
        "warm_start": warm_start_record,
        "script_sha256": sha(__file__),
        "module_sha256": sha(ROOT / "horizyn/generalization_residual.py"),
        "selection": ("External full-library BEDROC85 selection; no MRR computed"
                      if external_screening else
                      "Fixed final optimization step; validation metrics are diagnostics only"
                      if args.selection_method == "fixed_last" else
                      "Max validation mean all-positive MRR; each direction aggregate and unseen-reaction stratum within tolerance of initialization"),
        "train_reactions": len(tr), "train_enzymes": len(te), "train_edges": len(edge_r),
        "score_matrix_gib_float32": len(tr) * len(te) * 4 / 2**30,
        "biological_labels_sha256": sha(biological_labels) if biological_labels else None,
        "biology_weights": biological_weights,
        "biology_mode": getattr(args, 'biology_mode', 'attraction'),
        "biology_margin": getattr(args, 'biology_margin', .1),
        "additional_encoder_parameters": 0,
    }
    atomic_json(args.output / "registry.json", registry)
    records, best_value, baseline, best_step = [], float("-inf"), None, 0
    training_records = []
    started = time.monotonic()
    for step in range(args.steps + 1):
        if args.snapshot_every and step > 0 and step % args.snapshot_every == 0:
            torch.save(dict(state_dict=model.state_dict(), model_config=dict(
                dimension=model.dimension, hidden=model.hidden, scale=model.scale),
                registry=registry, fixed_step=step), args.output / f"step{step:04d}.pt")
        if not external_screening and (step % args.validate_every == 0 or step == args.steps):
            model.eval()
            torch.set_float32_matmul_precision("highest")
            with torch.inference_mode():
                scores = model.encode_reactions(base_r[vr]) @ model.encode_enzymes(base_e[ve]).T
                evaluation = evaluate_scores(scores, truth)
            summary = evaluation["summary"]
            if baseline is None:
                baseline = summary
            value = selection_value(summary)
            is_eligible = eligible(summary, baseline, args.max_drop)
            record = dict(step=step, elapsed_seconds=time.monotonic() - started,
                          selection_value=value, eligible=is_eligible, validation=summary)
            if step > 0:
                record.update(training_loss=float(loss.detach()), reaction_loss=float(r_loss),
                              enzyme_loss=float(e_loss), identity_loss=float(identity.detach()),
                              peak_vram_gib=torch.cuda.max_memory_allocated(device) / 2**30)
                record["smooth_ap_loss"] = float(ranking.detach())
            records.append(record)
            select = (step == args.steps if args.selection_method == "fixed_last"
                      else is_eligible and value > best_value)
            if select:
                best_value, best_step = value, step
                torch.save(dict(state_dict=model.state_dict(), model_config=dict(
                    dimension=model.dimension, hidden=model.hidden, scale=model.scale),
                    registry=registry, selected_validation=record), args.output / "selected.pt")
                np.savez(args.output / "selected_validation_ranks.npz", **{
                    f"{direction}_{key}": values
                    for direction, block in evaluation["per_positive"].items()
                    for key, values in block.items()})
            atomic_json(args.output / "validation.json", dict(records=records, best_step=best_step, best_value=best_value))
            print(json.dumps({k: v for k, v in record.items() if k != "validation"}), flush=True)
            del scores, evaluation
        if step == args.steps:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # The entire graph fits one H200. No minibatch-specific false negatives
        # or candidate sampling are introduced by this experimental objective.
        torch.set_float32_matmul_precision("high")
        enzymes = model.encode_enzymes(train_e)
        reactions = model.encode_reactions(compact_train_r)
        logits = (reactions @ enzymes.T) / args.temperature
        if args.contrastive_objective == "decoupled":
            contrastive, r_loss, e_loss = full_graph_decoupled_loss(logits, positive_mask, edge_r, edge_e)
        else:
            contrastive, r_loss, e_loss = full_graph_contrastive_loss(
                logits, edge_r, edge_e, args.enzyme_weighting)
        ranking = logits.new_zeros(())
        if args.ranking_weight:
            ra = torch.randperm(len(tr), device=device)[:args.ranking_anchors]
            ea = torch.randperm(len(te), device=device)[:args.ranking_anchors]
            similarities = logits * args.temperature
            ranking = (sampled_smooth_ap_loss(similarities, positive_mask, ra, args.ranking_temperature)
                       + sampled_smooth_ap_loss(similarities.T, positive_mask.T, ea, args.ranking_temperature)) / 2
        identity = ((1 - (enzymes * train_e_identity).sum(-1)).mean() +
                    (1 - (reactions * train_r_identity).sum(-1)).mean()) / 2
        biological_loss = logits.new_zeros(())
        biological_components = {}
        if biology is not None:
            biological_loss, biological_components = biology(reactions, enzymes, biological_weights)
        loss = contrastive + args.identity_weight * identity + args.ranking_weight * ranking + biological_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if external_screening and ((step + 1) % 5 == 0 or step + 1 == args.steps):
            record = dict(step=step + 1, elapsed_seconds=time.monotonic() - started,
                          training_loss=float(loss.detach()), reaction_loss=float(r_loss),
                          enzyme_loss=float(e_loss), identity_loss=float(identity.detach()),
                          smooth_ap_loss=float(ranking.detach()))
            record['biological_loss'] = float(biological_loss.detach())
            record['biological_components'] = {k: float(v.detach()) for k, v in biological_components.items()}
            record['peak_vram_gib'] = torch.cuda.max_memory_allocated(device) / 2**30
            training_records.append(record)
            atomic_json(args.output / "training_progress.json", dict(records=training_records,
                        validation_pending=True, selection_metric="full_library_validation.table1.bedroc85"))
            print(json.dumps(record), flush=True)
        del logits, enzymes, reactions, contrastive
    atomic_json(args.output / "complete.json", dict(best_step=None if external_screening else best_step,
                best_value=None if external_screening else best_value,
                validation_selection_pending=external_screening,
                elapsed_seconds=time.monotonic() - started, test_used=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--validate-every", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--scale", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--identity-weight", type=float, default=2.0)
    parser.add_argument("--warm-start", type=Path, help="Train/validation-selected residual from this same training split")
    parser.add_argument("--identity-reference", choices=("base", "initial"), default="base")
    parser.add_argument("--enzyme-weighting", choices=("uniform", "reaction_balanced"), default="uniform")
    parser.add_argument("--max-drop", type=float, default=0.005)
    parser.add_argument("--selection-method", choices=("validation_mrr", "fixed_last", "external_screening"),
                        default="validation_mrr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--contrastive-objective", choices=("positive_ce", "decoupled"), default="positive_ce")
    parser.add_argument("--ranking-weight", type=float, default=0.0)
    parser.add_argument("--ranking-anchors", type=int, default=128)
    parser.add_argument("--ranking-temperature", type=float, default=0.05)
    parser.add_argument("--snapshot-every", type=int, default=0)
    parser.add_argument('--biological-labels', type=Path)
    parser.add_argument('--biology-ec', type=float, default=0.)
    parser.add_argument('--biology-cofactor', type=float, default=0.)
    parser.add_argument('--biology-mechanism', type=float, default=0.)
    parser.add_argument('--biology-shuffle-seed', type=int)
    parser.add_argument('--biology-mode', choices=('attraction', 'relative'), default='attraction')
    parser.add_argument('--biology-margin', type=float, default=.1)
    args = parser.parse_args()
    if not 0 <= args.biology_margin <= 2:
        parser.error('Biological geometry margin must be between zero and two')
    if args.steps < 0 or args.validate_every < 1 or args.temperature <= 0:
        parser.error("Invalid step or temperature settings")
    if args.ranking_weight < 0 or args.ranking_anchors < 1 or args.ranking_temperature <= 0 or args.snapshot_every < 0:
        parser.error("Invalid ranking or snapshot settings")
    if str(args.device).startswith('cuda'):
        # Full-graph score/gradient tensors can exceed 20 GiB on ReactZyme.
        # Share the physical-device export lock so they do not overlap large
        # residue exports or another full-graph fit. Query before CUDA init.
        import os
        import subprocess
        import time
        from generalization_clipzyme_f3_screen import export_device_lock
        from generalization_gpu_budget import free_memory_mib
        logical = int(str(args.device).partition(':')[2] or 0)
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        physical = visible.split(',')[logical].strip() if visible else str(logical)
        required_mib = 14000 if args.selection_method == 'external_screening' else 40000
        with export_device_lock(args.device):
            while True:
                available = free_memory_mib(physical)
                if available >= required_mib:
                    break
                print(json.dumps(dict(waiting_for_full_graph_free_mib=available, required_mib=required_mib)), flush=True)
                time.sleep(5)
            run(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
