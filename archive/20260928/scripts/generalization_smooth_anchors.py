#!/usr/bin/env python3
"""Exploratory phase-two smooth anchors, using training and validation only.

Started after the phase-one external failure was observed. External panels are
not used here for fitting, parameter selection, or diagnostic computations.
"""
from __future__ import annotations

import argparse
import itertools
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
from horizyn.generalization_retrieval import canonical_dot
from horizyn.generalization_residual import FrozenGeometryResidual
from horizyn.semantic_anchors import centered_unit, reaction_features, nearest_training_proteins, enzyme_anchor_features
from horizyn.semantic_smooth import reaction_responses, stable_neighbor_order
from generalization_full_graph import atomic_json, sha, validation_data, eligible, selection_value
from generalization_metrics import evaluate_scores


def robust_value(summary):
    return float(np.mean([summary[d][s]['reactzyme_mrr']
        for d in ('reaction_to_enzyme', 'enzyme_to_reaction')
        for s in ('seen_reaction', 'unseen_reaction')]))


def quantiles(values):
    values = values.detach().float()
    return dict(n=values.numel(), mean=float(values.mean()),
        quantiles={str(q):float(torch.quantile(values, q)) for q in (0., .01, .05, .5, .95, .99, 1.)})


def support_diagnostic(scores, truth):
    r = torch.tensor(truth['reaction_index'], device=scores.device)
    e = torch.tensor(truth['enzyme_index'], device=scores.device)
    seen = torch.tensor(truth['reaction_seen'], device=scores.device)
    positive = scores[r, e]
    maximum = scores.amax(1)
    return dict(all_scores_zero_fraction=float((scores == 0).float().mean()),
        all_zero_reaction_queries=int((scores == 0).all(1).sum()),
        top_score_tie_count=quantiles((scores == maximum[:,None]).sum(1)),
        positive_score_zero_fraction={name:float((positive[mask] == 0).float().mean())
            for name, mask in [('all',torch.ones_like(r,dtype=torch.bool)), ('seen_reaction',seen[r]), ('unseen_reaction',~seen[r])]},
        positive_scores=quantiles(positive))


def cpu(value):
    if isinstance(value, torch.Tensor): return value.cpu()
    if isinstance(value, dict): return {k:cpu(v) for k,v in value.items()}
    return value


def geometry_diagnostic(args, native_e, native_r, compact_train_r, train_e, vr, ve, truth, device):
    checkpoint = torch.load(args.graph_checkpoint, map_location=device, weights_only=False)
    model = FrozenGeometryResidual(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    generator = np.random.default_rng(20260919)
    sample = np.sort(generator.choice(train_e, min(8192,len(train_e)), replace=False))
    report = dict(checkpoint_sha256=sha(args.graph_checkpoint), features_only=True, external_read=False)
    with torch.inference_mode():
        for direction, values, selections in (
            ('reaction', compact_train_r, {'train':np.arange(len(compact_train_r))}),
            ('enzyme', native_e, {'train_sample8192':sample, 'validation':ve}),
            ('reaction', native_r, {'validation_seen':vr[truth['reaction_seen']], 'validation_unseen':vr[~truth['reaction_seen']]})):
            for name, ids in selections.items():
                raw=values[ids]
                transformed=getattr(model,'encode_'+('reactions' if direction=='reaction' else 'enzymes'))(raw)
                distance=1-(F.normalize(raw,dim=1)*transformed).sum(1)
                report[direction+'_'+name]=quantiles(distance)
    return report


def run(args):
    if (args.output/'registry.json').exists(): raise ValueError('Choose a fresh output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.manual_seed(42)
    device=torch.device(args.device);started=time.monotonic()
    settings=[]
    for k,et in itertools.product([32,128,512],[.03,.1]):
        for nr,rt in itertools.product([None,512,2048],[.03,.1,.3]):
            settings.append(dict(protein_neighbors=k,enzyme_temperature=et,kernel='exponential',reaction_neighbors=nr,reaction_temperature=rt))
        for kernel in ['shifted_cosine','centered_cosine']:
            settings.append(dict(protein_neighbors=k,enzyme_temperature=et,kernel=kernel,reaction_neighbors=None,reaction_temperature=.1))
    registry=dict(schema='phase2_smooth_anchors_v1',exploratory_after_phase1_external_failure=True,
        new_external_data_read=False,selection_data='reaction_smi standard validation only',
        primary_selection='Max equal-weight mean seen/unseen reaction all-positive MRR across both directions; four aggregate/unseen baseline guards; full-support kernels only; baseline eligible',
        aggregate_optimized_results='Diagnostic only',alpha=[.1,.25,.5,1.],settings=settings,
        feature_manifest_sha256=sha(args.features/'manifest.json'),dictionary_sha256=sha(args.dictionary),
        module_sha256=sha(ROOT/'horizyn/semantic_smooth.py'),script_sha256=sha(__file__),
        score='Actual sqrt-weight-concatenated normalized endpoints; FP64 dot accumulation rounded to FP32',
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    atomic_json(args.output/'registry.json',registry)
    shutil.copyfile(__file__,args.output/'source.py');shutil.copyfile(ROOT/'horizyn/semantic_smooth.py',args.output/'model_source.py')
    catalog=json.loads((args.features/'catalog.json').read_text())
    with np.load(args.features/'pairs.npz') as f:pairs={k:f[k] for k in f.files}
    vr,ve,truth=validation_data(catalog,pairs)
    tr,te=np.unique(pairs['train'][:,0]),np.unique(pairs['train'][:,1])
    with np.load(args.features/'f3_features.npz') as f:
        native_e=torch.tensor(f['proteins'],device=device);native_r=torch.tensor(f['reactions'],device=device)
        compact_train_r=torch.tensor(f['train_reactions'],device=device)
    atomic_json(args.output/'graph_geometry_diagnostic.json',geometry_diagnostic(args,native_e,native_r,compact_train_r,te,vr,ve,truth,device))
    fq,fe=F.normalize(native_r[vr],dim=1),F.normalize(native_e[ve],dim=1)
    del native_e,native_r,compact_train_r
    base=evaluate_scores(canonical_dot(fq,fe),truth);baseline=base['summary']
    base_row=dict(id='F3',eligible=True,full_support=True,robust_value=robust_value(baseline),aggregate_value=selection_value(baseline),summary=baseline,config=dict(alpha=0))
    records=[base_row];best_full=best_unrestricted=best_aggregate=base_row
    dictionary=torch.load(args.dictionary,map_location=device,weights_only=False)
    if dictionary['feature_manifest_sha256']!=registry['feature_manifest_sha256']:raise ValueError('Dictionary/source manifest mismatch')
    if dictionary['train_protein_ids']!=[catalog['proteins'][i] for i in te] or dictionary['train_reaction_ids']!=[catalog['reactions'][i] for i in tr]:raise ValueError('Training dictionary ID mismatch')
    with h5py.File(args.features/'protein_mean.h5','r') as f:
        if list(f['ids'].asstr()[:])!=catalog['proteins'] or not f['complete'][:].all():raise ValueError('Protein means incomplete/misaligned')
        protein=torch.tensor(f['vectors'][ve],device=device)
    encoded=centered_unit(protein,dictionary['protein_center']);del protein
    values,indices=nearest_training_proteins(encoded,dictionary['train_proteins'],512,args.batch_size)
    values,indices=stable_neighbor_order(values,indices)
    del encoded
    with np.load(args.features/'reaction_features.npz') as f:
        blocks={k:torch.tensor(f[k],device=device) for k in dictionary['modalities']}
        masks={k:torch.tensor(f[k+'_mask'],device=device) for k in dictionary['modalities']}
    rx=reaction_features(blocks,dictionary['reaction_centers'],masks,dictionary['modalities'])
    train_rx=rx[tr];query_rx=rx[vr];del blocks,masks,rx
    similarity=canonical_dot(query_rx,train_rx)
    training_sim=canonical_dot(train_rx,train_rx);training_sim.fill_diagonal_(-float('inf'))
    train_nn=training_sim.amax(1);del training_sim
    seen=torch.tensor(truth['reaction_seen'],device=device)
    atomic_json(args.output/'training_support_diagnostic.json',dict(
        training_leave_one_out_nearest_reaction=quantiles(train_nn),
        validation_seen_nearest_reaction=quantiles(similarity[seen].amax(1)),
        validation_unseen_nearest_reaction=quantiles(similarity[~seen].amax(1)),
        unseen_below_train_fifth_percentile=float((similarity[~seen].amax(1)<torch.quantile(train_nn,.05)).float().mean()),
        validation_nearest_protein=quantiles(values[:,0])))
    del similarity,train_nn
    original=torch.load(args.frozen_validation_endpoints,map_location=device,weights_only=False)
    hard_scores=canonical_dot(original['anchor_reactions'],original['anchor_enzymes'])
    atomic_json(args.output/'phase1_hard_anchor_validation_support.json',support_diagnostic(hard_scores,truth));del original,hard_scores
    e_cache={};r_cache={};support=[]
    for index,config in enumerate(settings):
        ekey=(config['protein_neighbors'],config['enzyme_temperature'])
        if ekey not in e_cache:
            e_cache.clear()
            k,et=ekey;e_cache[ekey]=enzyme_anchor_features(values[:,:k],indices[:,:k],dictionary['adjacency'],len(tr),et)
        semantic_e=e_cache[ekey]
        rkey=(config['kernel'],config['reaction_temperature'],config['reaction_neighbors'])
        if rkey not in r_cache:r_cache[rkey]=reaction_responses(query_rx,train_rx,*rkey)
        semantic_r=r_cache[rkey]
        semantic_scores=canonical_dot(semantic_r,semantic_e)
        diagnostics=support_diagnostic(semantic_scores,truth);support.append(dict(config=config,diagnostics=diagnostics));del semantic_scores
        for alpha in registry['alpha']:
            enzyme=torch.cat([math.sqrt(1-alpha)*fe,math.sqrt(alpha)*semantic_e],1)
            reaction=torch.cat([math.sqrt(1-alpha)*fq,math.sqrt(alpha)*semantic_r],1)
            evaluation=evaluate_scores(canonical_dot(reaction,enzyme),truth)
            summary=evaluation['summary'];row=dict(id=f'config{index:02d}_alpha{alpha:g}',config=dict(config,alpha=alpha),
                eligible=eligible(summary,baseline,args.tolerance),full_support=config['reaction_neighbors'] is None,
                robust_value=robust_value(summary),aggregate_value=selection_value(summary),summary=summary)
            records.append(row)
            if row['eligible']:
                if row['robust_value']>best_unrestricted['robust_value']:best_unrestricted=row
                if row['aggregate_value']>best_aggregate['aggregate_value']:best_aggregate=row
                if row['full_support'] and row['robust_value']>best_full['robust_value']:
                    best_full=row
                    state=dict(config=row['config'],dictionary_path=str(args.dictionary.resolve()),dictionary_sha256=registry['dictionary_sha256'],
                        train_reactions=train_rx,feature_manifest_sha256=registry['feature_manifest_sha256'],selected_validation=row,
                        fit_data='training only; parameters chosen on standard validation',exploratory_after_phase1_external_failure=True)
                    torch.save(cpu(state),args.output/'selected.pt')
                    np.savez(args.output/'selected_validation_ranks.npz',**{d+'_'+k:v for d,block in evaluation['per_positive'].items() for k,v in block.items()})
            print(json.dumps(dict(id=row['id'],robust=row['robust_value'],aggregate=row['aggregate_value'],eligible=row['eligible'],best=best_full['id'])),flush=True)
        atomic_json(args.output/'validation.json',dict(records=records,selected_full_support=best_full,unrestricted_robust=best_unrestricted,aggregate_optimized_diagnostic=best_aggregate))
    atomic_json(args.output/'support_diagnostics.json',support)
    atomic_json(args.output/'complete.json',dict(selected=best_full,unrestricted_robust=best_unrestricted,
        aggregate_optimized_diagnostic=best_aggregate,elapsed_seconds=time.monotonic()-started,test_used=False,case1_used=False,p450_used=False,
        exploratory_after_phase1_external_failure=True,configurations=len(records)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    campaign=ROOT/'runs/generalization_20260919_2251'
    p.add_argument('--features',type=Path,default=campaign/'features')
    p.add_argument('--dictionary',type=Path,default=campaign/'models/reaction_smi/seed42/anchors.pt')
    p.add_argument('--graph-checkpoint',type=Path,default=campaign/'graph_v2_uniform_s42/selected.pt')
    p.add_argument('--frozen-validation-endpoints',type=Path,default=campaign/'composition_v4/selected_validation_features.pt')
    p.add_argument('--output',type=Path,default=campaign/'phase2/smooth_anchors')
    p.add_argument('--device',default='cuda:1');p.add_argument('--batch-size',type=int,default=512);p.add_argument('--tolerance',type=float,default=.005)
    with torch.inference_mode():run(p.parse_args())


if __name__=='__main__':main()
