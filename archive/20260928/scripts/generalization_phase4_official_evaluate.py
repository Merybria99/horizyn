#!/usr/bin/env python3
"""Report fixed phase4 methods using unchanged official metrics and bootstrap math."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from scripts.generalization_export import atomic_json,identity,digest
from scripts.generalization_metrics import evaluate_scores
from scripts.generalization_phase2_official_evaluate import (SPLITS,DIRECTIONS,METRICS,resolve,
    bootstrap_difference,reaction_cluster_bootstrap,checked_official_truth,write_csv)
from scripts.generalization_phase4_predict import validate_frozen_sources

# Fixed before any phase4 score is read. Old phase2 scores remain immutable controls.
METHODS={
 'phase4_seed42':('phase4',42,'primary','selected'),
 'phase4_seed17':('phase4',17,'seed17','selected'),
 'phase4_seed73':('phase4',73,'seed73','selected'),
 'hybrid_anchor_only':('phase4',42,'hybrid_anchor_only','selected'),
 'phase2_seed42':('phase2',42,'primary','selected'),
 'density_only':('phase2',42,'density_only','selected'),
 'F3_native':('phase4',42,'primary','baseline'),
 'F3_fp64':('phase4',42,'primary','baseline_fp64'),
}


def checked_lineage(source,directory,provenance):
    records={}
    path=resolve(directory,source['freeze']['path']);actual=digest(path)
    if actual!=source['freeze']['sha256']:raise ValueError('Phase4 frozen recipe checksum mismatch')
    freeze=json.loads(path.read_text())
    if freeze.get('schema')!='phase4_hybrid_frozen_recipe_v1' or not freeze.get('frozen_before_phase4_prediction') or not freeze.get('frozen_before_new_external_evaluation'):
        raise ValueError('Phase4 recipe must be frozen before new predictions/evaluation')
    validate_frozen_sources(dict(path=str(path),sha256=actual),[Path(__file__),ROOT/'scripts/generalization_metrics.py',ROOT/'scripts/generalization_phase2_official_evaluate.py',ROOT/'horizyn/generalization_phase4.py'])
    records['phase4']=(path,actual);provenance[str(path)]=identity(path,True)
    for key,field in [('phase2','phase2_frozen_recipe'),('original','original_frozen_recipe')]:
        prior=resolve(path.parent,freeze[field]['path']);checksum=digest(prior)
        if checksum!=freeze[field]['sha256']:raise ValueError('Frozen lineage checksum mismatch: '+key)
        records[key]=(prior,checksum);provenance[str(prior)]=identity(prior,True)
    parent=json.loads(records['phase2'][0].read_text());original=json.loads(records['original'][0].read_text())
    if not parent.get('frozen_before_phase2_prediction') or parent['original_frozen_recipe']['sha256']!=records['original'][1]:
        raise ValueError('Phase2 and original feature-freeze lineage disagree')
    if not original.get('frozen_before_held_out_evaluation'):raise ValueError('Original feature freeze was not pre-evaluation')
    return freeze,records


def checked_scores(entry,directory,catalog_path,shape,lineage,provenance,label,split):
    phase,seed,variant,key=METHODS[label]
    if entry.get('score_key')!=key:raise ValueError('Method score key differs from fixed definition')
    path=resolve(directory,entry['path']);info=identity(path,True)
    if entry.get('sha256') and entry['sha256']!=info['sha256']:raise ValueError('Score checksum mismatch')
    catalog_sha=digest(catalog_path)
    if entry.get('catalog_sha256') and entry['catalog_sha256']!=catalog_sha:raise ValueError('Explicit score catalog mismatch')
    receipt_path=resolve(directory,entry['receipt']) if entry.get('receipt') else path.parent/'complete.json'
    if not receipt_path.is_file():raise ValueError('A native prediction receipt is required')
    receipt=json.loads(receipt_path.read_text())
    if receipt.get('output_sha256')!=info['sha256']:raise ValueError('Prediction output hash changed')
    if receipt.get('labels_used') is not False or receipt.get('activity_labels_used'):raise ValueError('Predictions must exclude held-out labels')
    if receipt.get('inputs',{}).get('catalog',{}).get('sha256')!=catalog_sha:raise ValueError('Prediction catalog order differs')
    bundle=resolve(receipt_path.parent,receipt['bundle']['path'])
    if digest(bundle)!=receipt['bundle']['sha256']:raise ValueError('Prediction bundle changed')
    spec=json.loads(bundle.read_text())
    for field,target in [('frozen_recipe','original'),('phase2_frozen_recipe','phase2')]+([('phase4_frozen_recipe','phase4')] if phase=='phase4' else []):
        if spec.get(field,{}).get('sha256')!=lineage[target][1]:raise ValueError('Prediction freeze lineage differs: '+field)
        if receipt.get(field,{}).get('sha256')!=lineage[target][1]:raise ValueError('Prediction receipt lineage differs: '+field)
    if spec.get('split')!=split or spec.get('seed')!=seed or spec.get('variant')!=variant:
        raise ValueError('Prediction model split/seed/variant differs from fixed method')
    if phase=='phase4':
        from horizyn.generalization_phase4 import validate_phase4_bundle
        validate_phase4_bundle(spec,bundle.parent)
    else:
        from horizyn.generalization_phase2 import validate_phase2_bundle
        validate_phase2_bundle(spec,bundle.parent)
    input_record=receipt.get('input_receipt')
    if not input_record:raise ValueError('Prediction lacks authenticated input receipt')
    input_path=resolve(receipt_path.parent,input_record['path'])
    if digest(input_path)!=input_record['sha256']:raise ValueError('Prediction input receipt changed')
    inputs=json.loads(input_path.read_text())
    if inputs.get('schema')!='generalization_feature_bundle_receipt_v1':raise ValueError('Unknown input receipt schema')
    if inputs.get('freeze_sha256')!=lineage['original'][1]:raise ValueError('Input export freeze differs')
    for field in ('catalog','base','protein_means','reaction_features'):
        if inputs['inputs'][field]['sha256']!=receipt['inputs'][field]['sha256']:
            raise ValueError('Prediction input binding differs: '+field)
    with np.load(path,allow_pickle=False) as source:scores=np.asarray(source[key])
    if scores.dtype!=np.float32 or scores.shape!=shape or not np.isfinite(scores).all():raise ValueError('Require finite full FP32 score matrix')
    for artifact in (path,receipt_path,bundle,input_path):provenance[str(artifact)]=identity(artifact,True)
    return scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--feature-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--bootstrap-replicates',type=int,default=10000);p.add_argument('--seed',type=int,default=20260919)
    p.add_argument('--tiger-reference',type=Path);args=p.parse_args()
    if args.bootstrap_replicates<100:p.error('Use at least100 bootstrap replicates')
    source=json.loads(args.manifest.read_text());directory=args.manifest.resolve().parent
    provenance={str(args.manifest.resolve()):identity(args.manifest.resolve(),True)}
    freeze,lineage=checked_lineage(source,directory,provenance)
    methods=[{**m,'splits':m.get('splits',m.get('panels',{}))} for m in source['methods']]
    labels=[m['label'] for m in methods]
    if len(labels)!=len(set(labels)) or set(labels)!=set(METHODS):raise ValueError('Report the eight fixed phase4 methods exactly once')
    if source.get('baseline_method')!='F3_fp64':raise ValueError('The fixed paired baseline is F3_fp64')
    if [m['label'] for m in methods if m.get('primary')]!=['phase4_seed42'] or freeze.get('primary_seed')!=42:
        raise ValueError('The frozen primary is phase4_seed42')
    if any(set(m['splits'])!=set(SPLITS) or m.get('seed')!=METHODS[m['label']][1] for m in methods):
        raise ValueError('Every method must declare its fixed seed and all official splits')
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Evaluation outputs are immutable; use an empty directory')
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    tiger_path=args.tiger_reference or args.feature_root/'case1_audit/tiger_reference_targets.json';tiger=json.loads(tiger_path.read_text());provenance[str(tiger_path)]=identity(tiger_path,True)
    results={label:{} for label in labels};rows=[];contrasts=[];cluster_contrasts=[];literature=[]
    # Authenticate every declared score artifact before opening any official
    # associations. Only catalog metadata and label-free receipts are used here.
    # The fixed three panels and eight methods need approximately 1.7GB RAM.
    authenticated={}
    for split in SPLITS:
        catalog_path=args.feature_root/f'features_test_{split}'/'catalog.json'
        catalog=json.loads(catalog_path.read_text());shape=(len(catalog['reactions']),len(catalog['proteins']))
        authenticated[split]=dict(catalog=catalog,scores={method['label']:checked_scores(
            method['splits'][split],directory,catalog_path,shape,lineage,provenance,method['label'],split) for method in methods})
    print('Authenticated all fixed method/split scores before opening official associations',flush=True)
    for si,split in enumerate(SPLITS):
        feature_root=args.feature_root/f'features_test_{split}';catalog_path=feature_root/'catalog.json'
        catalog,edges=checked_official_truth(feature_root,split,lineage['original'][1],provenance)
        if catalog!=authenticated[split]['catalog']:raise ValueError('Catalog changed after score authentication')
        shape=(len(catalog['reactions']),len(catalog['proteins']));truth=dict(reaction_index=edges[:,0],enzyme_index=edges[:,1])
        groups=np.full(shape[1],shape[0],np.int64);np.minimum.at(groups,edges[:,1],edges[:,0])
        if (groups==shape[0]).any():raise ValueError('An official enzyme query has no positive reaction')
        per_method={}
        for method in methods:
            label=method['label'];scores=authenticated[split]['scores'].pop(label)
            evaluation=evaluate_scores(torch.from_numpy(scores).to(args.device),truth)
            per_method[label]={d:evaluation['per_query'][d]['all'] for d in DIRECTIONS}
            np.savez_compressed(args.output/f'{label}_{split}_per_query.npz',**{f'{d}__{k}':v for d,b in per_method[label].items() for k,v in b.items()})
            results[label][split]={}
            for direction in DIRECTIONS:
                metrics=evaluation['summary'][direction]['all'];results[label][split][direction]=metrics
                rows.append(dict(method=label,primary=bool(method.get('primary')),seed=method['seed'],split=split,direction=direction,**metrics))
                for metric,reference_key in [('reactzyme_mrr','mrr'),('top_1','hit1'),('top_10','hit10')]:
                    value=tiger[split][direction][reference_key]
                    literature.append(dict(method=label,split=split,direction=direction,metric=metric,observed=metrics[metric],tiger_reference=value,numerical_difference=metrics[metric]-value,source=tiger['source'],comparison='Literature only; comparator candidate assets and MRR implementation not independently verified; no paired inference.'))
            print(f'Evaluated {label} {split}',flush=True)
        for reference in ('F3_fp64','phase2_seed42'):
            for method in methods:
                label=method['label']
                if label==reference:continue
                for di,direction in enumerate(DIRECTIONS):
                    stats=bootstrap_difference(per_method[label][direction],per_method[reference][direction],args.bootstrap_replicates,args.seed+si*2+di)
                    for metric,value in stats.items():contrasts.append(dict(method=label,baseline=reference,primary=bool(method.get('primary')),split=split,direction=direction,metric=metric,**value))
                    if direction=='enzyme_to_reaction':
                        stats=reaction_cluster_bootstrap(per_method[label][direction],per_method[reference][direction],groups,args.bootstrap_replicates,args.seed+100+si)
                        for metric,value in stats.items():cluster_contrasts.append(dict(method=label,baseline=reference,primary=bool(method.get('primary')),split=split,direction=direction,metric=metric,**value))
    summary=dict(schema='phase4_frozen_official_six_cell_evaluation_v1',primary_method='phase4_seed42',baseline_method='F3_fp64',additional_paired_reference='phase2_seed42',
        freeze_sha256=lineage['phase4'][1],phase2_freeze_sha256=lineage['phase2'][1],original_freeze_sha256=lineage['original'][1],methods=results,
        six_cell_macro={label:float(np.mean([results[label][s][d]['reactzyme_mrr'] for s in SPLITS for d in DIRECTIONS])) for label in labels},
        metric='Our MRR averages reciprocal rank across every known positive within query, then across queries; first-positive MRR separate; H1/5/10 use first-positive rank.',
        tie_policy='Stable descending score, sorted candidate-index ties, full official candidate universes.',
        uncertainty='10000 paired percentile query bootstrap by default; supplementary E2R reaction-group bootstrap pools group sums/counts with each enzyme assigned its smallest positive reaction index. Homology dependence and multiple comparisons remain limitations; macro is descriptive.',
        tiger_reference=tiger,model_selection='No model selection on these new scores. Phase4 was selected on reused validation after prior phase1/phase2 held-out outcomes, so results are exploratory.',
        numerical_baselines='F3_fp64 matches the phase4 normalization contract; F3_native preserves the original FP32-normalization control. Phase4 validation guards used F3_fp64.')
    atomic_json(args.output/'summary.json',summary);atomic_json(args.output/'paired_bootstrap.json',contrasts);atomic_json(args.output/'reaction_cluster_bootstrap.json',cluster_contrasts)
    for name,values in [('metrics',rows),('paired_bootstrap',contrasts),('reaction_cluster_bootstrap',cluster_contrasts),('tiger_literature_comparison',literature)]:write_csv(args.output/(name+'.csv'),values)
    lines=['# Exploratory frozen phase4 official evaluation','','Primary: **phase4_seed42**. Paired references: **F3_fp64** and **phase2_seed42**; F3_native is a separate normalization control.','','The recipe was fixed before these predictions, following prior phase1/phase2 held-out exposure. Every declared method is reported without choosing among them.','','| Method | Reaction R→E | Reaction E→R | Enzyme R→E | Enzyme E→R | Time R→E | Time E→R | Macro |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for label in labels:
        values=[results[label][s][d]['reactzyme_mrr'] for s in SPLITS for d in DIRECTIONS]
        lines.append('| '+label+' | '+' | '.join(f'{v:.6f}' for v in [*values,summary['six_cell_macro'][label]])+' |')
    tv=[tiger[s][d]['mrr'] for s in SPLITS for d in DIRECTIONS];lines.append('| TIGER ESM2Text (literature) | '+' | '.join(f'{v:.6f}' for v in [*tv,float(np.mean(tv))])+' |')
    lines+=['','Our rows use all-positive MRR. TIGER values are published reference numbers; exact comparator candidate assets and its MRR implementation were not independently verified. No paired statistical claim is made against literature.','','See paired_bootstrap.csv for paired query intervals and reaction_cluster_bootstrap.csv for E2R reaction-group sensitivity against both fixed references. The six-cell macro is descriptive; dependence and unadjusted multiple comparisons limit uncertainty claims.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    if any(digest(path)!=checksum for path,checksum in lineage.values()):raise ValueError('Frozen lineage changed during evaluation')
    atomic_json(args.output/'complete.json',dict(schema='phase4_frozen_official_evaluation_receipt_v1',lineage={k:identity(v[0],True) for k,v in lineage.items()},inputs=provenance,bootstrap_replicates=args.bootstrap_replicates,bootstrap_seed=args.seed,evaluator=identity(__file__,True),metrics_source=identity(ROOT/'scripts/generalization_metrics.py',True),bootstrap_source=identity(ROOT/'scripts/generalization_phase2_official_evaluate.py',True),output=identity(args.output/'summary.json',True)))


if __name__=='__main__':main()
