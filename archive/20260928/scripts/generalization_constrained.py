#!/usr/bin/env python3
"""Post-failure exploratory training of bounded raw-semantic dual encoders.

Only train/validation assets are opened. No test or external feature path is
accepted. Both endpoint linear maps have a global spectral displacement bound.
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
from horizyn.generalization_residual import full_graph_contrastive_loss
from horizyn.semantic_anchors import row_unit, fit_center, centered_unit, reaction_features
from horizyn.semantic_constrained import BoundedSemanticMap
from horizyn.semantic_linear import paired_statistics, fit_bridges
from horizyn.generalization_retrieval import canonical_dot
from scripts.generalization_full_graph import atomic_json, sha, validation_data, eligible, selection_value
from scripts.generalization_metrics import evaluate_scores


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    return value


def robust_value(summary):
    return float(np.mean([summary[d][s]['reactzyme_mrr']
        for d in ('reaction_to_enzyme', 'enzyme_to_reaction')
        for s in ('seen_reaction', 'unseen_reaction')]))


def prepare(args, device):
    source = args.features
    if not (source / 'complete.json').exists():
        raise ValueError('Incomplete train/validation export')
    catalog = json.loads((source / 'catalog.json').read_text())
    if 'validation_candidates' not in catalog:
        raise ValueError('Only train/validation exports accepted')
    with np.load(source / 'pairs.npz') as handle:
        pairs = {key: handle[key] for key in handle.files}
    tr, te = np.unique(pairs['train'][:, 0]), np.unique(pairs['train'][:, 1])
    vr, ve, truth = validation_data(catalog, pairs)
    rmap, emap = np.full(len(catalog['reactions']), -1), np.full(len(catalog['proteins']), -1)
    rmap[tr], emap[te] = np.arange(len(tr)), np.arange(len(te))
    er = torch.tensor(rmap[pairs['train'][:, 0]], device=device)
    ee = torch.tensor(emap[pairs['train'][:, 1]], device=device)
    with h5py.File(source / 'protein_mean.h5') as handle:
        if list(handle['ids'].asstr()[:]) != catalog['proteins'] or not handle['complete'][:].all():
            raise ValueError('Raw protein IDs/order/completion disagree')
        raw = torch.tensor(handle['vectors'][:], device=device)
    center_e = fit_center(raw, te)
    raw_e = centered_unit(raw, center_e)
    del raw
    blocks, centers, masks = {}, {}, {}
    modalities = ['t5v2', 'unimol2', 'chiro']
    with np.load(source / 'reaction_features.npz') as handle:
        for key in modalities:
            blocks[key] = torch.tensor(handle[key], device=device)
            masks[key] = torch.tensor(handle[key + '_mask'], device=device)
            centers[key] = fit_center(blocks[key], tr, masks[key])
    raw_r = reaction_features(blocks, centers, masks, modalities)
    stats = paired_statistics(raw_r[tr], raw_e[te], er, ee)
    bridge = fit_bridges(stats, args.ridge, args.rank)['cca']
    sem_r = row_unit((raw_r - bridge['x_mean']) @ bridge['query_weight'])
    sem_e = row_unit((raw_e - bridge['y_mean']) @ bridge['enzyme_weight'])
    with np.load(source / 'f3_features.npz') as handle:
        base_r = F.normalize(torch.tensor(handle['reactions'], device=device), dim=1)
        base_e = F.normalize(torch.tensor(handle['proteins'], device=device), dim=1)
        train_r = F.normalize(torch.tensor(handle['train_reactions'], device=device), dim=1)
    if [catalog['reactions'][i] for i in tr] != catalog['train_reactions']:
        raise ValueError('Train reaction cache ordering differs')
    state = dict(schema='bounded_semantic_preparation_v1', exploratory_after_phase1_failure=True,
        test_used=False, external_used=False, feature_manifest_sha256=sha(source / 'manifest.json'),
        preprocessing=dict(protein_center=center_e, reaction_centers=centers, modalities=modalities,
                           bridge=bridge),
        semantic_reactions=sem_r, semantic_enzymes=sem_e, base_reactions=base_r, base_enzymes=base_e,
        train_base_reactions=train_r, tr=tr, te=te, vr=vr, ve=ve, truth=truth, edge_r=er, edge_e=ee,
        train_reaction_ids=catalog['train_reactions'], train_enzyme_ids=catalog['train_proteins'],
        source_sha256=sha(__file__), model_sha256=sha(ROOT / 'horizyn/semantic_constrained.py'))
    args.prepared.parent.mkdir(parents=True, exist_ok=True)
    if args.prepared.exists():
        raise ValueError('Prepared artifact already exists')
    torch.save(cpu(state), args.prepared)
    atomic_json(args.prepared.with_suffix('.json'), dict(schema=state['schema'], sha256=sha(args.prepared),
        feature_manifest_sha256=state['feature_manifest_sha256'], source_sha256=state['source_sha256'],
        rank=args.rank, ridge=args.ridge, test_used=False, external_used=False))
    print(json.dumps(dict(prepared=str(args.prepared), rank=args.rank)), flush=True)


def train(args, device):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'registry.json').exists():
        raise ValueError('Use a new immutable experiment directory')
    receipt = json.loads(args.prepared.with_suffix('.json').read_text())
    if sha(args.prepared) != receipt['sha256']:
        raise ValueError('Prepared state hash mismatch')
    state = torch.load(args.prepared, map_location=device, weights_only=False)
    tr, te, vr, ve = (state[key] for key in ('tr','te','vr','ve'))
    br, be, sr, se = (state[key] for key in ('base_reactions','base_enzymes','semantic_reactions','semantic_enzymes'))
    model = BoundedSemanticMap(sr.shape[1], args.radius).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.001)
    torch.set_float32_matmul_precision('high')
    train_base = state['train_base_reactions'] @ be[te].T
    torch.set_float32_matmul_precision('highest')
    baseline = evaluate_scores(canonical_dot(br[vr], be[ve]), state['truth'])['summary']
    registry = dict(schema='bounded_semantic_training_v1', exploratory_after_phase1_failure=True,
        test_used=False, external_used=False, arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        prepared_sha256=receipt['sha256'], feature_manifest_sha256=state['feature_manifest_sha256'],
        source_sha256=sha(__file__), model_sha256=sha(ROOT / 'horizyn/semantic_constrained.py'),
        primary_selection='Four-cell equal mean seen/unseen reaction MRR in both directions; all/unseen four guards versus F3',
        blend_weights=[.05, .1, .25, .5, 1.], baseline=baseline)
    atomic_json(args.output / 'registry.json', registry)
    shutil.copyfile(__file__, args.output / 'source.py')
    records = []
    best = dict(step=-1, alpha=0., robust_value=robust_value(baseline), aggregate_value=selection_value(baseline), validation=baseline)
    best_aggregate = dict(best)
    started = time.monotonic()
    for step in range(args.steps + 1):
        if step % args.validate_every == 0 or step == args.steps:
            model.eval()
            torch.set_float32_matmul_precision('highest')
            with torch.no_grad():
                qr, qe = model.encode_reactions(sr[vr]), model.encode_enzymes(se[ve])
                for alpha in registry['blend_weights']:
                    qr_all = torch.cat((math.sqrt(1-alpha)*br[vr], math.sqrt(alpha)*qr), dim=1)
                    qe_all = torch.cat((math.sqrt(1-alpha)*be[ve], math.sqrt(alpha)*qe), dim=1)
                    summary = evaluate_scores(canonical_dot(qr_all, qe_all), state['truth'])['summary']
                    record = dict(step=step, alpha=alpha, robust_value=robust_value(summary),
                        aggregate_value=selection_value(summary), eligible=eligible(summary, baseline, .005), validation=summary,
                        elapsed_seconds=time.monotonic()-started)
                    records.append(record)
                    payload = dict(state_dict=cpu(model.state_dict()), model_config=dict(dimension=model.dimension, radius=model.radius),
                        prepared_path=str(args.prepared.resolve()), prepared_sha256=receipt['sha256'], selected=record, registry=registry)
                    if record['eligible'] and record['robust_value'] > best['robust_value']:
                        best = record
                        torch.save(payload, args.output / 'selected.pt')
                    if record['eligible'] and record['aggregate_value'] > best_aggregate['aggregate_value']:
                        best_aggregate = record
                        torch.save(payload, args.output / 'aggregate_diagnostic.pt')
            atomic_json(args.output / 'validation.json', dict(records=records, selected=best, aggregate_diagnostic=best_aggregate))
            print(json.dumps(dict(step=step, best_robust=best['robust_value'], best_alpha=best['alpha'],
                best_step=best['step'], elapsed=time.monotonic()-started)), flush=True)
        if step == args.steps:
            break
        model.train()
        torch.set_float32_matmul_precision('high')
        optim.zero_grad(set_to_none=True)
        re = model.encode_reactions(sr[tr])
        ee = model.encode_enzymes(se[te])
        scores = (1 - args.train_alpha)*train_base + args.train_alpha*(re @ ee.T)
        loss, _, _ = full_graph_contrastive_loss(scores / args.temperature,
            state['edge_r'], state['edge_e'], args.enzyme_weighting)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite contrastive loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optim.step()
        norms = model.project()
        del scores, re, ee, loss
    atomic_json(args.output / 'complete.json', dict(selected=best, aggregate_diagnostic=best_aggregate,
        spectral_norms=norms, elapsed_seconds=time.monotonic()-started,
        peak_vram_gib=torch.cuda.max_memory_allocated(device)/2**30, test_used=False, external_used=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, default=ROOT/'runs/generalization_20260919_2251/features')
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--device', default='cuda:2')
    p.add_argument('--rank', type=int, default=256)
    p.add_argument('--ridge', type=float, default=.01)
    p.add_argument('--radius', type=float, default=.5)
    p.add_argument('--train-alpha', type=float, default=.5)
    p.add_argument('--temperature', type=float, default=.07)
    p.add_argument('--enzyme-weighting', choices=['uniform','reaction_balanced'], default='uniform')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--validate-every', type=int, default=50)
    p.add_argument('--learning-rate', type=float, default=.001)
    args = p.parse_args()
    if args.steps < 1 or args.validate_every < 1 or not 0 < args.train_alpha <= 1 or args.temperature <= 0:
        p.error('Invalid training parameters')
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('highest')
    device = torch.device(args.device)
    if args.prepare_only:
        prepare(args, device)
    elif args.output:
        train(args, device)
    else:
        p.error('--output required for training')


if __name__ == '__main__':
    main()
