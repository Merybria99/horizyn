#!/usr/bin/env python3
"""Validate one internal fusion-strength adjustment while retaining SLEEC."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from generalization_reactzyme_architecture_phase2 import (
    ROOT,RUN,load_features,load_head,smooth_encoder,compose,validation_data,eligible,
    robust_value,evaluate_scores,canonical_dot,sha256,atomic_json,invoke,checked_official_truth)
from generalization_clipzyme_f3_screen import model_from_checkpoint,export_device_lock
from generalization_gpu_budget import free_memory_mib
from horizyn.training_io import StoragePrecisionResidues
from horizyn.benchmarks.retrieval import encode_residue_targets


@torch.inference_mode()
def components(model,config,keys,device):
    encoder=model.model.multiview_encoder
    if encoder is None:raise ValueError('Calibration requires the SLEEC multiview encoder')
    values={'global':[],'fused':[]}
    def capture(name):
        def hook(module,inputs,output):
            values[name].append(F.normalize(output.float(),dim=-1,eps=1e-6).cpu())
        return hook
    handles=[encoder.global_output.register_forward_hook(capture('global')),
             encoder.fused_output.register_forward_hook(capture('fused'))]
    residues=StoragePrecisionResidues(config.data.protein_residue_embeds_path,
        max_tokens=config.data.max_protein_tokens,truncation=config.data.protein_truncation)
    try:reference=encode_residue_targets(model,residues,keys,device,128,False).float()
    finally:
        for h in handles:h.remove()
        residues.close()
    g=torch.cat(values['global']).to(device);f=torch.cat(values['fused']).to(device)
    scale=float(encoder.residual_scale)
    reproduced=F.normalize(g+scale*f,dim=-1,eps=1e-6)
    error=float((reproduced.cpu()-reference).abs().max())
    if error>1e-6:raise ValueError(f'Fusion reconstruction failed: {error}')
    return g,f,scale,reference.to(device),error


def adjusted(g,f,scale,multiplier,original):
    if multiplier==1:return original
    if not 0<scale*multiplier<.5:raise ValueError('Calibration must retain positive bounded SLEEC/view contribution')
    return F.normalize(g+(scale*multiplier)*f,dim=-1,eps=1e-6)


@torch.inference_mode()
def run(out,plan,task,device):
    dest=out/task['name'];dest.mkdir(exist_ok=False)
    source=Path(task['source_campaign'])/task['arm'];split=task['split']
    receipt=json.loads((source/'features/complete.json').read_text())
    if receipt['checkpoint_sha256']!=sha256(task['checkpoint']):raise ValueError('Source checkpoint changed')
    recipe=plan['phase2_recipe']
    model,config=model_from_checkpoint(Path(task['train_config']),Path(task['checkpoint']),device)
    catalog,be,br,means,blocks,masks=load_features(source/'features',device)
    with np.load(source/'features/pairs.npz') as data:pairs={k:data[k] for k in ('train','validation')}
    vr,ve,truth=validation_data(catalog,pairs)
    g,f,scale,exported,reconstruction_error=components(model,config,[catalog['proteins'][i] for i in ve],device)
    cache_error=float((exported-be[ve]).abs().max())
    if cache_error>3e-6:raise ValueError(f'Validation export differs from checkpoint cache: {cache_error}')
    smooth=smooth_encoder(source,device)
    head=load_head(source/f"training/step{recipe['step']:04d}.pt",sha256(source/'features/manifest.json'),device)
    se=smooth.encode_semantic_enzymes(means[ve],256)
    sr=smooth.encode_semantic_reactions({k:v[vr] for k,v in blocks.items()},{k:v[vr] for k,v in masks.items()})
    reactions=compose(br[vr],head,sr,recipe,'reaction')
    records=[];baseline=None
    for multiplier in plan['multipliers']:
        enzymes=compose(adjusted(g,f,scale,multiplier,be[ve]),head,se,recipe,'enzyme')
        summary=evaluate_scores(canonical_dot(reactions,enzymes),truth)['summary']
        if multiplier==1:baseline=summary
        row=dict(multiplier=multiplier,scale=scale*multiplier,summary=summary,value=robust_value(summary))
        records.append(row)
    if baseline is None:raise ValueError('Unmodified control missing')
    # Validate the unchanged arm against its previously recorded validation.
    old=json.loads((source/'validation.json').read_text())['records']
    match=next(r for r in old if r['recipe']==recipe)
    baseline_error=max(abs(baseline[d][s]['reactzyme_mrr']-match['summary'][d][s]['reactzyme_mrr'])
        for d in ('reaction_to_enzyme','enzyme_to_reaction') for s in ('all','unseen_reaction'))
    if baseline_error>1e-6:raise ValueError('Unmodified validation no longer reproduces')
    for row in records:row['eligible']=eligible(row['summary'],baseline,plan['maximum_direction_drop'])
    best=max((row for row in records if row['eligible']),key=lambda x:x['value'])
    if best['value']<=robust_value(baseline)+1e-6:best=next(row for row in records if row['multiplier']==1)
    freeze=dict(selected=best,baseline=baseline,records=records,selected_utc=datetime.now(timezone.utc).isoformat(),
        recipe=recipe,task=task,protocol_sha256=sha256(out/'protocol.json'),checkpoint_sha256=receipt['checkpoint_sha256'],
        feature_manifest_sha256=sha256(source/'features/manifest.json'),test_used_for_selection=False,
        reconstruction_max_abs_error=reconstruction_error,cache_max_abs_error=cache_error,
        unchanged_validation_max_metric_error=baseline_error,retains_sleec=True,ensembles=False)
    atomic_json(dest/'selection.json',freeze)
    print(json.dumps(dict(task=task['name'],selected_multiplier=best['multiplier'],value=best['value'],baseline=robust_value(baseline))),flush=True)
    if best['multiplier']==1:
        atomic_json(dest/'complete.json',dict(unmodified_retained=True,test_run=False));return
    # Test features are opened only after the immutable validation selection.
    test_source=Path(task['fixed_test_features'])
    test_receipt=json.loads((test_source/'complete.json').read_text())
    if test_receipt['checkpoint_sha256']!=receipt['checkpoint_sha256']:raise ValueError('Test feature checkpoint mismatch')
    tc,te,tr,tm,tb,tmask=load_features(test_source,device)
    tg,tf,test_scale,texported,test_reconstruction_error=components(model,config,tc['proteins'],device)
    test_cache_error=float((texported-te).abs().max())
    if test_cache_error>3e-6 or test_scale!=scale:raise ValueError('Test fusion lineage mismatch')
    enzymes=compose(adjusted(tg,tf,scale,best['multiplier'],te),head,smooth.encode_semantic_enzymes(tm,256),recipe,'enzyme')
    reactions=compose(tr,head,smooth.encode_semantic_reactions(tb,tmask),recipe,'reaction')
    score=canonical_dot(reactions,enzymes);np.save(dest/'selected_test_scores.npy',score.cpu().numpy())
    parent=json.loads((RUN/'phase2/models'/split/'seed42/bundle.json').read_text());provenance={}
    checked,edges=checked_official_truth(RUN/f'features_test_{split}',split,parent['frozen_recipe']['sha256'],provenance)
    if tc!=checked:raise ValueError('Official test catalog mismatch')
    summary=evaluate_scores(score,dict(reaction_index=edges[:,0],enzyme_index=edges[:,1]))['summary']
    atomic_json(dest/'selected_test_summary.json',dict(summary=summary,split=split,multiplier=best['multiplier'],
        phase2_recipe=recipe,selection_sha256=sha256(dest/'selection.json'),scores_sha256=sha256(dest/'selected_test_scores.npy'),
        truth_provenance=provenance,test_used_for_selection=False,test_results_exploratory=True,
        test_reconstruction_max_abs_error=test_reconstruction_error,test_cache_max_abs_error=test_cache_error))
    atomic_json(dest/'complete.json',dict(unmodified_retained=False,test_run=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--arm')
    a=p.parse_args();out=a.campaign.resolve();plan=json.loads((out/'protocol.json').read_text())
    if a.arm:
        import os
        task=next(t for t in plan['tasks'] if t['name']==a.arm)
        physical=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
        with export_device_lock('cuda:0'):
            while free_memory_mib(physical)<27000:time.sleep(5)
            run(out,plan,task,'cuda:0')
        return
    for task in plan['tasks']:
        source=Path(task['source_campaign'])/task['arm']
        required=[source/'validation_complete.json',source/'anchors.pt',
            source/f"training/step{plan['phase2_recipe']['step']:04d}.pt",
            Path(task['fixed_test_features'])/'complete.json']
        while not all(path.exists() for path in required):time.sleep(10)
        invoke(Path(__file__).name,['--campaign',out,'--arm',task['name']],task['gpu'],out/(task['name']+'.log'))
    atomic_json(out/'complete.json',dict(completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
