#!/usr/bin/env python3
"""Fixed six-weight phase-two composition screen on standard validation only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from horizyn.generalization_retrieval import canonical_dot,checked_artifact,sha256
from generalization_full_graph import atomic_json,validation_data,eligible,selection_value
from generalization_metrics import evaluate_scores
from generalization_smooth_anchors import robust_value


def identity(path):return dict(path=str(path.resolve()),sha256=sha256(path))


def run(args):
    if (args.output/'registry.json').exists():raise ValueError('Choose a fresh output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    device=torch.device(args.device)
    catalog=json.loads((args.features/'catalog.json').read_text())
    with np.load(args.features/'pairs.npz') as source:pairs={k:source[k] for k in source.files}
    vr,ve,truth=validation_data(catalog,pairs)
    receipt=json.loads((args.density/'selected_validation_receipt.json').read_text())
    for key in ('features','catalog','selection','gate','feature_source_catalog'):checked_artifact(receipt[key])
    dc=json.loads(Path(receipt['catalog']['path']).read_text())
    if dc['proteins']!=catalog['validation_candidates'] or dc['reactions']!=catalog['validation_reactions']:raise ValueError('Density validation catalog order mismatch')
    check=json.loads((args.smooth/'saved_inference_check.json').read_text())
    if check['checkpoint_sha256']!=sha256(args.smooth/'selected.pt') or check['all_summary_metrics_max_difference']!=0:raise ValueError('Smooth saved inference is not verified')
    with np.load(receipt['features']['path']) as source:
        dq=torch.tensor(source['reactions'],device=device);de=torch.tensor(source['proteins'],device=device)
    smooth=torch.load(args.smooth/'selected_validation_features.pt',map_location=device,weights_only=False)
    if smooth['reaction_ids']!=catalog['validation_reactions'] or smooth['enzyme_ids']!=catalog['validation_candidates']:raise ValueError('Smooth validation catalog order mismatch')
    aq,ae=smooth['anchor_reactions'],smooth['anchor_enzymes']
    with np.load(args.features/'f3_features.npz') as source:
        fq=F.normalize(torch.tensor(source['reactions'][vr],device=device),dim=1)
        fe=F.normalize(torch.tensor(source['proteins'][ve],device=device),dim=1)
    sources={name:identity(path) for name,path in dict(
        feature_manifest=args.features/'manifest.json',density_receipt=args.density/'selected_validation_receipt.json',
        smooth_checkpoint=args.smooth/'selected.pt',smooth_validation_features=args.smooth/'selected_validation_features.pt',
        smooth_inference_check=args.smooth/'saved_inference_check.json',
        composition_module=ROOT/'horizyn/generalization_phase2.py',smooth_module=ROOT/'horizyn/semantic_smooth.py').items()}
    registry=dict(schema='phase2_density_smooth_composition_v1',exploratory_after_phase1_external_failure=True,
        test_used=False,case1_used=False,p450_used=False,alpha=[0,.1,.25,.5,.75,1],sources=sources,
        selection='Equal-weight seen/unseen reaction all-positive MRR across both directions, aggregate/unseen baseline drop <=.005 in each direction; F3 baseline eligible',
        score='Actual sqrt-weight concatenation of density-gated endpoints and raw smooth-anchor endpoints; canonical FP64 dot accumulated then FP32 output',
        tolerance=args.tolerance,script_sha256=sha256(__file__))
    atomic_json(args.output/'registry.json',registry);shutil.copyfile(__file__,args.output/'source.py')
    baseline_evaluation=evaluate_scores(canonical_dot(fq,fe),truth);baseline=baseline_evaluation['summary']
    selected=dict(id='F3',alpha=None,robust_value=robust_value(baseline),aggregate_value=selection_value(baseline),eligible=True,summary=baseline)
    rows=[selected];selected_features=None
    for alpha in registry['alpha']:
        q=torch.cat((math.sqrt(1-alpha)*dq,math.sqrt(alpha)*aq),dim=1)
        e=torch.cat((math.sqrt(1-alpha)*de,math.sqrt(alpha)*ae),dim=1)
        evaluation=evaluate_scores(canonical_dot(q,e),truth);summary=evaluation['summary']
        row=dict(id=f'density_smooth_alpha{alpha:g}',alpha=alpha,summary=summary,
            robust_value=robust_value(summary),aggregate_value=selection_value(summary),eligible=eligible(summary,baseline,args.tolerance))
        rows.append(row)
        np.savez(args.output/(row['id']+'_ranks.npz'),**{d+'_'+k:v for d,block in evaluation['per_positive'].items() for k,v in block.items()})
        if row['eligible'] and row['robust_value']>selected['robust_value']:
            selected=row;selected_features=dict(reactions=q.cpu(),enzymes=e.cpu(),reaction_ids=catalog['validation_reactions'],enzyme_ids=catalog['validation_candidates'],alpha=alpha)
        print(json.dumps({k:v for k,v in row.items() if k!='summary'}),flush=True)
    result=dict(selected=selected,baseline=rows[0],records=rows,elapsed_seconds=time.monotonic()-started,
        aggregate_optimized_diagnostic=max([r for r in rows if r['eligible']],key=lambda r:r['aggregate_value']),
        exploratory_after_phase1_external_failure=True,new_external_evaluation_used=False)
    atomic_json(args.output/'selection.json',result)
    if selected_features is not None:torch.save(selected_features,args.output/'selected_validation_features.pt')
    atomic_json(args.output/'complete.json',dict(selected=selected,elapsed_seconds=time.monotonic()-started,test_used=False))


def main():
    root=ROOT/'runs/generalization_20260919_2251'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,default=root/'features')
    p.add_argument('--density',type=Path,default=root/'phase2/density_gate')
    p.add_argument('--smooth',type=Path,default=root/'phase2/smooth_anchors')
    p.add_argument('--output',type=Path,default=root/'phase2/composition')
    p.add_argument('--device',default='cuda:1');p.add_argument('--tolerance',type=float,default=.005)
    with torch.inference_mode():run(p.parse_args())


if __name__=='__main__':main()
