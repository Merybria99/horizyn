#!/usr/bin/env python3
"""Apply the fixed screening phase-2 recipe to fresh ReactZyme architecture pilots."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from horizyn.generalization_residual import FrozenGeometryResidual
from horizyn.semantic_smooth import SmoothAnchorDualEncoder
from horizyn.generalization_retrieval import canonical_dot,sha256
from generalization_screen_phase2_campaign import invoke
from generalization_full_graph import atomic_json,validation_data,eligible
from generalization_transport_reactzyme import load_features,RUN
from generalization_metrics import evaluate_scores
from generalization_smooth_anchors import robust_value
from generalization_phase2_official_evaluate import checked_official_truth


def load_head(path,manifest,device):
    state=torch.load(path,map_location=device,weights_only=False)
    if state['registry']['feature_manifest_sha256']!=manifest or state['registry']['test_used']:
        raise ValueError('Residual training lineage mismatch')
    model=FrozenGeometryResidual(**state['model_config']).to(device).eval().requires_grad_(False)
    model.load_state_dict(state['state_dict'],strict=True)
    return model


def smooth_encoder(run,device):
    dictionary=torch.load(run/'anchors.pt',map_location=device,weights_only=False)
    if dictionary['feature_manifest_sha256']!=sha256(run/'features/manifest.json'):
        raise ValueError('Semantic dictionary training lineage mismatch')
    return SmoothAnchorDualEncoder(dictionary,dictionary['train_reactions'],dict(
        alpha=1.,kernel='exponential',reaction_neighbors=None,reaction_temperature=.03,
        protein_neighbors=32,enzyme_temperature=.03)).to(device)


def compose(base,head,semantic,recipe,endpoint):
    if head is None:learned=F.normalize(base,dim=-1)
    else:learned=F.normalize(base+recipe['cap']*head.scale*getattr(head,endpoint)(base),dim=-1)
    alpha=recipe['alpha']
    return torch.cat((math.sqrt(1-alpha)*learned,math.sqrt(alpha)*semantic),dim=1)


@torch.inference_mode()
def validate(run,baseline,device,steps=(0,5,20,50,100),caps=(1.,.5),alphas=(0.,.1,.25)):
    catalog,be,br,means,blocks,masks=load_features(run/'features',device)
    with np.load(run/'features/pairs.npz') as data:pairs={k:data[k] for k in ('train','validation')}
    vr,ve,truth=validation_data(catalog,pairs)
    smooth=smooth_encoder(run,device)
    se=smooth.encode_semantic_enzymes(means[ve],256)
    sr=smooth.encode_semantic_reactions({k:v[vr] for k,v in blocks.items()},{k:v[vr] for k,v in masks.items()})
    records=[]
    for step in steps:
        head=load_head(run/f'training/step{step:04d}.pt',sha256(run/'features/manifest.json'),device) if step else None
        for cap in (caps if step else (1.,)):
            for alpha in alphas:
                recipe=dict(step=step,cap=cap,alpha=alpha)
                e=compose(be[ve],head,se,recipe,'enzyme');r=compose(br[vr],head,sr,recipe,'reaction')
                summary=evaluate_scores(canonical_dot(r,e),truth)['summary']
                row=dict(recipe=recipe,summary=summary,value=robust_value(summary),eligible=eligible(summary,baseline,.005))
                records.append(row)
                atomic_json(run/'validation.json',dict(records=records))
                print(json.dumps(dict(arm=run.name,recipe=recipe,value=row['value'],eligible=row['eligible'])),flush=True)
    best=max((r for r in records if r['eligible']),key=lambda r:r['value'],default=None)
    atomic_json(run/'validation_complete.json',dict(selected=best,records=len(records),test_used=False))
    return best


@torch.inference_mode()
def evaluate_selected(out,task,best,device,split):
    run=out/task['arm'];template=RUN/f'features_test_{split}'
    invoke('generalization_reactzyme_f3_features.py',['--template',template,'--config',task['test_config'],
        '--checkpoint',task['checkpoint'],'--output',run/'test_features','--scope','test',
        '--freeze',out/'selection.json','--batch-size',128,'--split',split],task['gpu'],run/'test_features.log')
    # The export subprocess releases its lock before this process initializes
    # CUDA. Serialize test scoring too, without nesting an export under a lock.
    from generalization_clipzyme_f3_screen import export_device_lock
    from generalization_gpu_budget import free_memory_mib
    import os
    logical=int(str(device).partition(':')[2] or 0)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    physical=visible.split(',')[logical].strip() if visible else str(logical)
    with export_device_lock(device):
        while free_memory_mib(physical)<14000:time.sleep(5)
        score_exported_test(out,task,best,device,split)


@torch.inference_mode()
def score_exported_test(out,task,best,device,split):
    run=out/task['arm'];template=RUN/f'features_test_{split}'
    tc,be,br,means,blocks,masks=load_features(run/'test_features',device)
    smooth=smooth_encoder(run,device);recipe=best['recipe']
    head=load_head(run/f"training/step{recipe['step']:04d}.pt",sha256(run/'features/manifest.json'),device) if recipe['step'] else None
    e=compose(be,head,smooth.encode_semantic_enzymes(means,256),recipe,'enzyme')
    r=compose(br,head,smooth.encode_semantic_reactions(blocks,masks),recipe,'reaction')
    score=canonical_dot(r,e)
    np.save(out/'selected_test_scores.npy',score.cpu().numpy())
    parent=json.loads((RUN/'phase2/models'/split/'seed42/bundle.json').read_text())
    provenance={};checked,edges=checked_official_truth(template,split,parent['frozen_recipe']['sha256'],provenance)
    if checked!=tc:raise ValueError('Official test catalog mismatch')
    truth=dict(reaction_index=edges[:,0],enzyme_index=edges[:,1])
    result=dict(split=split,arm=task['arm'],recipe=recipe,summary=evaluate_scores(score,truth)['summary'],
        truth_provenance=provenance,selection_sha256=sha256(out/'selection.json'),
        scores_sha256=sha256(out/'selected_test_scores.npy'),test_used_for_selection=False,test_results_exploratory=True)
    atomic_json(out/'selected_test_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='Reuse verified completed stages; preserve and retry interrupted residual fits')
    a=p.parse_args();out=a.campaign.resolve();plan=json.loads((out/'protocol.json').read_text())
    split=plan.get('split','reaction_smi')
    template=RUN/{'reaction_smi':'features','enzyme_smi':'features_enzyme_smi','time':'features_time'}[split]
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    baseline=json.loads((RUN/'phase2_soft_ce_v1'/split/'composition_validation.json').read_text())['records'][0]['validation']
    def arm(task):
        run=out/task['arm'];run.mkdir(exist_ok=True)
        checkpoint=Path(task['checkpoint'])
        while not checkpoint.exists() or time.time()-checkpoint.stat().st_mtime<30:time.sleep(10)
        atomic_json(run/'state.json',dict(stage='f3_feature_export',checkpoint_sha256=sha256(checkpoint)))
        if a.resume and (run/'features/complete.json').exists():
            if json.loads((run/'features/complete.json').read_text())['checkpoint_sha256']!=sha256(checkpoint):
                raise ValueError('Recovered features belong to another checkpoint')
        elif plan.get('feature_source_campaign'):
            source=Path(plan['feature_source_campaign'])/task['arm']
            while not (source/'features/complete.json').exists():time.sleep(10)
            if json.loads((source/'features/complete.json').read_text())['checkpoint_sha256']!=sha256(checkpoint):
                raise ValueError('Reusable learned features belong to another checkpoint')
            (run/'features').symlink_to(source/'features',target_is_directory=True)
        else:
            invoke('generalization_reactzyme_f3_features.py',['--template',template,'--config',task['train_config'],
                '--checkpoint',checkpoint,'--output',run/'features','--scope','train_validation','--batch-size',128,'--split',split],
                task['gpu'],run/'features.log')
        atomic_json(run/'state.json',dict(stage='fixed_phase2_training'))
        training=plan['training_recipe']
        if not (a.resume and (run/'training/complete.json').exists()):
            if a.resume and (run/'training').exists():
                destination=run/('training_failed_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
                (run/'training').rename(destination)
                atomic_json(run/'recovery.json',dict(preserved_attempt=str(destination),
                    unchanged_base_checkpoint_sha256=sha256(checkpoint),residual_retrained_from_initialization=True))
            invoke('generalization_full_graph.py',['--features',run/'features','--output',run/'training',
                '--steps',training['steps'],'--snapshot-every',training.get('snapshot_every',5),
                '--selection-method','fixed_last','--temperature',training['temperature'],
                '--identity-weight',training['identity_weight'],'--contrastive-objective',training['objective'],'--cpu-threads',4],
                task['gpu'],run/'training.log')
        if not (a.resume and (run/'anchors.pt').exists()):
            invoke('generalization_clipzyme_phase2_dictionary.py',['--features',run/'features','--output',run/'anchors.pt'],
                task['gpu'],run/'dictionary.log')
        atomic_json(run/'state.json',dict(stage='composition_validation'))
        # Validation is dispatched to a subprocess so each worker has an
        # independent CUDA device context, allocator, and thread count.
        if not (a.resume and (run/'validation_complete.json').exists()):
            invoke('generalization_reactzyme_architecture_phase2_validate.py',[
                '--run',run],task['gpu'],run/'validation.log')
        atomic_json(run/'state.json',dict(stage='complete'))
        return json.loads((run/'validation_complete.json').read_text())['selected']
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(arm,plan['tasks']))
    selection=dict(parent_retained=True,value=robust_value(baseline),baseline=baseline,
        task=None,selected=None,test_used_for_selection=False)
    for task,result in zip(plan['tasks'],results):
        if result is not None and result['value']>selection['value']:
            selection.update(parent_retained=False,value=result['value'],task=task,selected=result)
    selection['selected_utc']=datetime.now(timezone.utc).isoformat()
    selection['protocol_sha256']=sha256(out/'protocol.json')
    atomic_json(out/'selection.json',selection)
    if not selection['parent_retained']:
        # Coordinator itself has no inherited CUDA restriction.
        task=selection['task'];evaluate_selected(out,task,selection['selected'],f"cuda:{task['gpu']}",split)
    atomic_json(out/'complete.json',dict(parent_retained=selection['parent_retained'],
        completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
