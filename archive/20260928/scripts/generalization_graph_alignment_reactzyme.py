#!/usr/bin/env python3
"""Fit a small graph-alignment residual inside the frozen phase-2 dual encoder."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horizyn.generalization_graph_alignment import graph_statistics, fit_alignment, align_composed
from horizyn.generalization_phase2 import ComposedPhase2Encoder
from horizyn.generalization_retrieval import canonical_dot, sha256
from generalization_transport_reactzyme import load_features, RUN
from generalization_full_graph import validation_data, atomic_json, eligible
from generalization_smooth_anchors import robust_value
from generalization_metrics import evaluate_scores
from generalization_phase2_official_evaluate import checked_official_truth


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=('reaction_smi','enzyme_smi','time'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    features=RUN/{'reaction_smi':'features','enzyme_smi':'features_enzyme_smi','time':'features_time'}[a.split]
    bundle=RUN/'phase2/models'/a.split/'seed42/bundle.json'
    bank_path=RUN/'phase2_training_neighbor_v1'/a.split/'training_bank.pt'
    grid=[dict(ridge=reg,strength=strength) for reg in (.1,1.,10.) for strength in (.1,.25,.5,1.)]
    registry=dict(created_utc=datetime.now(timezone.utc).isoformat(),split=a.split,
        parent_bundle=dict(path=str(bundle),sha256=sha256(bundle)),
        training_bank=dict(path=str(bank_path),sha256=sha256(bank_path)),
        feature_manifest_sha256=sha256(features/'manifest.json'),grid=grid,
        objective='Anchor-balanced mean positive squared error, ridge penalty on the residual relative to identity',
        selection='Balanced seen/unseen validation all-positive MRR, both direction aggregate/unseen within .005 of parent',
        source_sha256=sha256(__file__),module_sha256=sha256(ROOT/'horizyn/generalization_graph_alignment.py'),
        test_used_for_selection=False,retains_sleec=True,ensemble=False)
    atomic_json(out/'registry.json',registry)
    model,spec=ComposedPhase2Encoder.from_bundle(bundle,a.device)
    bank=torch.load(bank_path,map_location=a.device,weights_only=False)
    if (bank['registry']['parent_bundle']!=registry['parent_bundle'] or
        bank['registry']['train_manifest_sha256']!=registry['feature_manifest_sha256']):
        raise ValueError('Training bank belongs to a different frozen parent')
    catalog,be,br,means,blocks,masks=load_features(features,a.device)
    with np.load(features/'pairs.npz') as data:pairs={k:data[k] for k in ('train','validation')}
    tr,te=np.unique(pairs['train'][:,0]),np.unique(pairs['train'][:,1])
    if ([catalog['proteins'][i] for i in te]!=bank['training_protein_ids'] or
        [catalog['reactions'][i] for i in tr]!=bank['training_reaction_ids']):
        raise ValueError('Training bank axes differ from the association graph')
    edges=np.stack((np.searchsorted(tr,pairs['train'][:,0]),np.searchsorted(te,pairs['train'][:,1])),axis=1)
    stats=graph_statistics(bank['enzyme_values'],bank['reaction_values'],edges)
    del bank
    fits={reg:fit_alignment(stats,reg) for reg in (.1,1.,10.)}
    torch.save(dict(registry=registry,heads={reg:{k:v.cpu() for k,v in heads.items()} for reg,heads in fits.items()}),out/'heads.pt')
    vr,ve,truth=validation_data(catalog,pairs)
    e=model.encode_enzymes(be[ve],means[ve],batch_size=256)
    r=model.encode_reactions(br[vr],{k:v[vr] for k,v in blocks.items()},{k:v[vr] for k,v in masks.items()},batch_size=256)
    baseline=evaluate_scores(canonical_dot(r,e),truth)['summary']
    best=dict(recipe=dict(ridge=None,strength=0),validation=baseline,value=robust_value(baseline),eligible=True)
    records=[best]
    for recipe in grid:
        heads=fits[recipe['ridge']]
        ee=align_composed(e,heads['enzyme'],recipe['strength'],model.alpha)
        rr=align_composed(r,heads['reaction'],recipe['strength'],model.alpha)
        summary=evaluate_scores(canonical_dot(rr,ee),truth)['summary']
        row=dict(recipe=recipe,validation=summary,value=robust_value(summary),eligible=eligible(summary,baseline,.005))
        records.append(row)
        if row['eligible'] and row['value']>best['value']:best=row
        atomic_json(out/'validation.json',dict(records=records,selected=best))
        print(json.dumps(dict(recipe=recipe,value=row['value'],eligible=row['eligible'])),flush=True)
    atomic_json(out/'selection.json',dict(selected=best,registry_sha256=sha256(out/'registry.json'),
        heads_sha256=sha256(out/'heads.pt'),selected_utc=datetime.now(timezone.utc).isoformat(),test_used_for_selection=False))
    del be,br,means,blocks,masks,e,r,ee,rr
    test=RUN/f'features_test_{a.split}'
    tc,be,br,means,blocks,masks=load_features(test,a.device)
    e=model.encode_enzymes(be,means,batch_size=256)
    r=model.encode_reactions(br,blocks,masks,batch_size=256)
    recipe=best['recipe']
    if recipe['strength']:
        heads=fits[recipe['ridge']]
        ee=align_composed(e,heads['enzyme'],recipe['strength'],model.alpha)
        rr=align_composed(r,heads['reaction'],recipe['strength'],model.alpha)
    else:ee,rr=e,r
    score=canonical_dot(rr,ee);base_score=canonical_dot(r,e)
    np.savez(out/'test_scores.npz',selected=score.cpu().numpy(),baseline_phase2=base_score.cpu().numpy())
    provenance={};checked,edges=checked_official_truth(test,a.split,spec['frozen_recipe']['sha256'],provenance)
    if checked!=tc:raise ValueError('Official test axes changed')
    truth=dict(reaction_index=edges[:,0],enzyme_index=edges[:,1])
    result=dict(split=a.split,recipe=recipe,parent_retained=recipe['strength']==0,
        selected=evaluate_scores(score,truth)['summary'],baseline_phase2=evaluate_scores(base_score,truth)['summary'],
        selection_sha256=sha256(out/'selection.json'),truth_provenance=provenance,
        test_used_for_selection=False,test_results_exploratory=True)
    atomic_json(out/'test_summary.json',result)
    print(json.dumps(dict(selected=recipe,test={d:result['selected'][d]['all']['reactzyme_mrr']
        for d in ('reaction_to_enzyme','enzyme_to_reaction')})),flush=True)


if __name__=='__main__':main()
