#!/usr/bin/env python3
"""Transfer the historical train-fitted F3 support gate to new ReactZyme bases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from generalization_gpu_budget import free_memory_mib

import numpy as np
import torch
from torch.nn import functional as F

from generalization_reactzyme_architecture_phase2 import (
    ROOT, RUN, load_features, load_head, smooth_encoder, validation_data, eligible,
    robust_value, evaluate_scores, canonical_dot, sha256, atomic_json, invoke,
    checked_official_truth)
from generalization_density_gate import nearest_support, support_gate


@torch.inference_mode()
def fit_support(run, proteins, reactions, pairs, device):
    enzyme_ids = np.unique(pairs['train'][:, 1])
    with np.load(run/'features/f3_features.npz') as data:
        training_reactions = torch.tensor(data['train_reactions'], device=device)
    anchors = dict(enzyme=F.normalize(proteins[enzyme_ids], dim=-1),
                   reaction=F.normalize(training_reactions, dim=-1))
    rng = np.random.default_rng(20260920)
    chosen = dict(enzyme=np.sort(rng.choice(len(enzyme_ids), min(4096, len(enzyme_ids)), replace=False)),
                  reaction=np.arange(len(training_reactions)))
    thresholds = {}
    for endpoint, values in anchors.items():
        indices = torch.tensor(chosen[endpoint], device=device)
        support = nearest_support(values[indices], values, exclude=indices, batch_size=256)
        thresholds[endpoint] = {str(q): float(torch.quantile(support, q)) for q in (.25, .95)}
    record = dict(thresholds=thresholds, calibration_indices={k:v.tolist() for k,v in chosen.items()},
                  calibration_seed=20260920, feature_manifest_sha256=sha256(run/'features/manifest.json'),
                  test_used=False, validation_used=False, lower_quantile=.25, upper_quantile=.95)
    atomic_json(run/'support_fit.json', record)
    return anchors, thresholds


@torch.inference_mode()
def endpoint_gates(base, anchors, thresholds):
    return {key: support_gate(nearest_support(F.normalize(values, dim=-1), anchors[key], batch_size=256),
                            thresholds[key]['0.25'], thresholds[key]['0.95'], 1.)
            for key,values in base.items()}


def compose(base, head, semantic, gate, recipe, endpoint):
    if head is None:
        learned = F.normalize(base, dim=-1)
    else:
        learned = F.normalize(base + recipe['cap'] * gate[:, None] * head.scale * getattr(head, endpoint)(base), dim=-1)
    return torch.cat((math.sqrt(1-recipe['alpha'])*learned, math.sqrt(recipe['alpha'])*semantic), dim=1)


@torch.inference_mode()
def validation(run, plan, baseline, device):
    catalog, be, br, means, blocks, masks = load_features(run/'features', device)
    with np.load(run/'features/pairs.npz') as data:
        pairs = {k:data[k] for k in ('train','validation')}
    vr, ve, truth = validation_data(catalog, pairs)
    anchors, thresholds = fit_support(run, be, br, pairs, device)
    base = dict(enzyme=be[ve], reaction=br[vr])
    gates = endpoint_gates(base, anchors, thresholds)
    smooth = smooth_encoder(run, device)
    semantic = dict(enzyme=smooth.encode_semantic_enzymes(means[ve], 256),
        reaction=smooth.encode_semantic_reactions({k:v[vr] for k,v in blocks.items()}, {k:v[vr] for k,v in masks.items()}))
    records = []
    for step in plan['steps']:
        head = load_head(run/f'training/step{step:04d}.pt', sha256(run/'features/manifest.json'), device) if step else None
        for cap in (plan['caps'] if step else (1.,)):
            for alpha in plan['alphas']:
                recipe = dict(step=step, cap=cap, alpha=alpha, gate='f3_both_q25_q95')
                vectors = {key:compose(value,head,semantic[key],gates[key],recipe,key) for key,value in base.items()}
                summary = evaluate_scores(canonical_dot(vectors['reaction'], vectors['enzyme']), truth)['summary']
                row = dict(recipe=recipe,summary=summary,value=robust_value(summary),eligible=eligible(summary,baseline,.005))
                records.append(row)
                atomic_json(run/'validation.json', dict(records=records))
    best = max((row for row in records if row['eligible']), key=lambda row:row['value'], default=None)
    atomic_json(run/'validation_complete.json', dict(selected=best, records=len(records), test_used=False))


@torch.inference_mode()
def test_selected(out, task, selected, device, split):
    run = out/task['arm']; template = RUN/f'features_test_{split}'
    invoke('generalization_reactzyme_f3_features.py', ['--template',template,'--config',task['test_config'],
        '--checkpoint',task['checkpoint'],'--output',run/'test_features','--scope','test',
        '--freeze',out/'selection.json','--batch-size',128,'--split',split],task['gpu'],run/'test_features.log')
    from generalization_clipzyme_f3_screen import export_device_lock
    with export_device_lock(device):
        while free_memory_mib(task['gpu'])<40000:time.sleep(5)
        score_exported_test(out,task,selected,device,split)


@torch.inference_mode()
def score_exported_test(out,task,selected,device,split):
    run=out/task['arm'];template=RUN/f'features_test_{split}'
    catalog, be, br, means, blocks, masks = load_features(run/'test_features',device)
    with np.load(run/'features/pairs.npz') as source:
        te = np.unique(source['train'][:,1])
    with np.load(run/'features/f3_features.npz') as source:
        anchors = dict(enzyme=F.normalize(torch.tensor(source['proteins'][te],device=device),dim=-1),
                       reaction=F.normalize(torch.tensor(source['train_reactions'],device=device),dim=-1))
    fit = json.loads((run/'support_fit.json').read_text())
    if fit['feature_manifest_sha256'] != sha256(run/'features/manifest.json'):
        raise ValueError('Training support feature lineage changed')
    base = dict(enzyme=be, reaction=br); gates = endpoint_gates(base,anchors,fit['thresholds'])
    smooth = smooth_encoder(run,device); recipe = selected['recipe']
    head = load_head(run/f"training/step{recipe['step']:04d}.pt",sha256(run/'features/manifest.json'),device) if recipe['step'] else None
    semantic = dict(enzyme=smooth.encode_semantic_enzymes(means,256), reaction=smooth.encode_semantic_reactions(blocks,masks))
    vectors = {key:compose(value,head,semantic[key],gates[key],recipe,key) for key,value in base.items()}
    scores = canonical_dot(vectors['reaction'],vectors['enzyme'])
    np.save(out/'selected_test_scores.npy',scores.cpu().numpy())
    parent = json.loads((RUN/'phase2/models'/split/'seed42/bundle.json').read_text()); provenance = {}
    checked, edges = checked_official_truth(template,split,parent['frozen_recipe']['sha256'],provenance)
    if catalog != checked:
        raise ValueError('Official test catalog mismatch')
    summary = evaluate_scores(scores,dict(reaction_index=edges[:,0],enzyme_index=edges[:,1]))['summary']
    atomic_json(out/'selected_test_summary.json',dict(split=split,arm=task['arm'],recipe=recipe,summary=summary,
        truth_provenance=provenance,selection_sha256=sha256(out/'selection.json'),scores_sha256=sha256(out/'selected_test_scores.npy'),
        support_fit_sha256=sha256(run/'support_fit.json'),test_used_for_selection=False,test_results_exploratory=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--validate-arm');p.add_argument('--resume',action='store_true')
    a=p.parse_args();out=a.campaign.resolve();plan=json.loads((out/'protocol.json').read_text())
    split=plan.get('split','reaction_smi')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    baseline=json.loads((RUN/'phase2_soft_ce_v1'/split/'composition_validation.json').read_text())['records'][0]['validation']
    if a.validate_arm:
        import os
        import subprocess
        from generalization_clipzyme_f3_screen import export_device_lock
        physical=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
        with export_device_lock('cuda:0'):
            while free_memory_mib(physical)<14000:
                time.sleep(5)
            validation(out/a.validate_arm,plan,baseline,'cuda:0')
        return
    def arm(task):
        run=out/task['arm'];run.mkdir(exist_ok=True);source=Path(plan['source_campaign'])/task['arm']
        if a.resume and (run/'validation_complete.json').exists():
            return json.loads((run/'validation_complete.json').read_text())['selected']
        atomic_json(run/'state.json',dict(stage='waiting_for_source_features_and_heads'))
        while not (source/'validation_complete.json').exists():time.sleep(10)
        try:
            for name in ('features','training','anchors.pt'):
                destination=run/name
                if destination.exists() or destination.is_symlink():
                    if not a.resume or destination.resolve()!=(source/name).resolve():
                        raise ValueError('Unexpected existing source link')
                else:destination.symlink_to(source/name)
            receipt=json.loads((run/'features/complete.json').read_text())
            if receipt['checkpoint_sha256']!=sha256(task['checkpoint']):raise ValueError('Wrong base checkpoint')
            atomic_json(run/'state.json',dict(stage='support_fit_and_validation'))
            invoke(Path(__file__).name,['--campaign',out,'--validate-arm',task['arm']],task['gpu'],run/'validation.log')
            atomic_json(run/'state.json',dict(stage='complete'))
            return json.loads((run/'validation_complete.json').read_text())['selected']
        except Exception as exc:
            atomic_json(run/'failure.json',dict(error=repr(exc)));raise
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(arm,plan['tasks']))
    selection=dict(parent_retained=True,value=robust_value(baseline),baseline=baseline,task=None,selected=None,test_used_for_selection=False)
    for task,result in zip(plan['tasks'],results):
        if result is not None and result['value']>selection['value']:
            selection.update(parent_retained=False,value=result['value'],task=task,selected=result)
    selection.update(selected_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha256(out/'protocol.json'))
    atomic_json(out/'selection.json',selection)
    if not selection['parent_retained']:
        task=selection['task'];test_selected(out,task,selection['selected'],f"cuda:{task['gpu']}",split)
    atomic_json(out/'complete.json',dict(parent_retained=selection['parent_retained'],completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
