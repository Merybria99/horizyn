#!/usr/bin/env python3
"""Report fixed calibration methods using unchanged official metrics and bootstrap math."""
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

# Fixed before any calibration score is read. Old phase2 scores remain immutable controls.
METHODS={
 'calibrated_seed42':('calibration',42,'primary','selected'),
 'calibrated_seed17':('calibration',17,'seed17','selected'),
 'calibrated_seed73':('calibration',73,'seed73','selected'),
 'phase2_seed42':('calibration',42,'primary','parent'),
 'phase2_seed17':('calibration',17,'seed17','parent'),
 'phase2_seed73':('calibration',73,'seed73','parent'),
 'F3_native':('calibration',42,'primary','baseline'),
 'F3_fp64':('calibration',42,'primary','baseline_fp64'),
}


def checked_lineage(source,directory,provenance):
    from generalization_calibration_transfer import validate_freeze
    path=resolve(directory,source['freeze']['path']);actual=digest(path)
    if actual!=source['freeze']['sha256']:raise ValueError('Calibration freeze mismatch')
    freeze=validate_freeze(dict(path=str(path),sha256=actual));records={'calibration':(path,actual)}
    for key,field in [('phase2','phase2_frozen_recipe'),('original','original_frozen_recipe')]:
        parent=resolve(path.parent,freeze[field]['path']);checksum=digest(parent)
        if checksum!=freeze[field]['sha256']:raise ValueError('Parent freeze mismatch')
        records[key]=(parent,checksum)
    for path,checksum in records.values():provenance[str(path)]=identity(path,True)
    return freeze,records


def checked_scores(entry,directory,catalog_path,shape,lineage,provenance,label,split):
    from generalization_calibration_transfer import checked_prediction
    _,seed,_,key=METHODS[label]
    if entry.get('score_key')!=key:raise ValueError('Fixed method score key mismatch')
    path=resolve(directory,entry['path']);freeze_path,freeze_sha=lineage['calibration']
    scores,receipt=checked_prediction(dict(path=str(freeze_path),sha256=freeze_sha),path,catalog_path,split,seed,key)
    if scores.shape!=shape:raise ValueError('Official score shape mismatch')
    for artifact in [path,path.parent/'complete.json',Path(receipt['input_receipt']['path'])]:provenance[str(artifact)]=identity(artifact,True)
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
    if len(labels)!=len(set(labels)) or set(labels)!=set(METHODS):raise ValueError('Report the eight fixed calibration methods exactly once')
    if source.get('baseline_method')!='F3_fp64':raise ValueError('The fixed paired baseline is F3_fp64')
    if [m['label'] for m in methods if m.get('primary')]!=['calibrated_seed42'] or freeze.get('primary_seed')!=42:
        raise ValueError('The frozen primary is calibrated_seed42')
    if any(set(m['splits'])!=set(SPLITS) or m.get('seed')!=METHODS[m['label']][1] for m in methods):
        raise ValueError('Every method must declare its fixed seed and all official splits')
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Evaluation outputs are immutable; use an empty directory')
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    tiger_path=args.tiger_reference or args.feature_root/'case1_audit/tiger_reference_targets.json';tiger=json.loads(tiger_path.read_text());provenance[str(tiger_path)]=identity(tiger_path,True)
    results={label:{} for label in labels};rows=[];contrasts=[];cluster_contrasts=[];literature=[];invariance=[]
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
        per_method={};positive_ranks={}
        for method in methods:
            label=method['label'];scores=authenticated[split]['scores'].pop(label)
            evaluation=evaluate_scores(torch.from_numpy(scores).to(args.device),truth)
            per_method[label]={d:evaluation['per_query'][d]['all'] for d in DIRECTIONS}
            positive_ranks[label]={d:evaluation['per_positive'][d]['rank'] for d in DIRECTIONS}
            np.savez_compressed(args.output/f'{label}_{split}_per_query.npz',**{f'{d}__{k}':v for d,b in per_method[label].items() for k,v in b.items()})
            results[label][split]={}
            for direction in DIRECTIONS:
                metrics=evaluation['summary'][direction]['all'];results[label][split][direction]=metrics
                rows.append(dict(method=label,primary=bool(method.get('primary')),seed=method['seed'],split=split,direction=direction,**metrics))
                for metric,reference_key in [('reactzyme_mrr','mrr'),('top_1','hit1'),('top_10','hit10')]:
                    value=tiger[split][direction][reference_key]
                    literature.append(dict(method=label,split=split,direction=direction,metric=metric,observed=metrics[metric],tiger_reference=value,numerical_difference=metrics[metric]-value,source=tiger['source'],comparison='Literature only; comparator candidate assets and MRR implementation not independently verified; no paired inference.'))
            print(f'Evaluated {label} {split}',flush=True)
        for seed in [42,17,73]:
            a,b=positive_ranks[f'calibrated_seed{seed}']['reaction_to_enzyme'],positive_ranks[f'phase2_seed{seed}']['reaction_to_enzyme']
            invariance.append(dict(split=split,seed=seed,positive_ranks_changed=int(np.count_nonzero(a!=b)),total_positive_edges=len(a),max_abs_rank_change=int(np.max(np.abs(a-b))),interpretation='Reaction-only bias is query-constant for R2E; any differences are final FP32 rounding/tie effects, not new discrimination.'))
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
    summary=dict(schema='post_evaluation_calibration_official_six_cell_v1',primary_method='calibrated_seed42',baseline_method='F3_fp64',additional_paired_reference='phase2_seed42',
        freeze_sha256=lineage['calibration'][1],phase2_freeze_sha256=lineage['phase2'][1],original_freeze_sha256=lineage['original'][1],methods=results,
        six_cell_macro={label:float(np.mean([results[label][s][d]['reactzyme_mrr'] for s in SPLITS for d in DIRECTIONS])) for label in labels},
        metric='Our MRR averages reciprocal rank across every known positive within query, then across queries; first-positive MRR separate; H1/5/10 use first-positive rank.',
        tie_policy='Stable descending score, sorted candidate-index ties, full official candidate universes.',
        uncertainty='10000 paired percentile query bootstrap by default; supplementary E2R reaction-group bootstrap pools group sums/counts with each enzyme assigned its smallest positive reaction index. Homology dependence and multiple comparisons remain limitations; macro is descriptive.',
        r2e_constant_shift_diagnostics=invariance,tiger_reference=tiger,model_selection='Fixed calibration transferred without retuning after all panels were previously evaluated. These outcomes are development-exposed exploratory results, not independent confirmation; primary scientific comparison is E2R.',
        numerical_baselines='F3_fp64 matches the frozen P2 normalization contract; F3_native preserves the original FP32-normalization control. Calibration validation guards used both F3_fp64 and P2.')
    atomic_json(args.output/'summary.json',summary);atomic_json(args.output/'paired_bootstrap.json',contrasts);atomic_json(args.output/'reaction_cluster_bootstrap.json',cluster_contrasts)
    for name,values in [('metrics',rows),('paired_bootstrap',contrasts),('reaction_cluster_bootstrap',cluster_contrasts),('tiger_literature_comparison',literature)]:write_csv(args.output/(name+'.csv'),values)
    lines=['# Post-evaluation calibration official evaluation','','Primary: **calibrated_seed42**. Paired references: **F3_fp64** and **phase2_seed42**; F3_native is a separate normalization control.','','The calibration was fixed before these predictions, after these panels were already evaluated. Every declared method is reported; results are development-exposed and provide no independent confirmation. E2R is the substantive comparison. R2E changes are constant-offset rounding effects.','','| Method | Reaction R→E | Reaction E→R | Enzyme R→E | Enzyme E→R | Time R→E | Time E→R | Macro |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for label in labels:
        values=[results[label][s][d]['reactzyme_mrr'] for s in SPLITS for d in DIRECTIONS]
        lines.append('| '+label+' | '+' | '.join(f'{v:.6f}' for v in [*values,summary['six_cell_macro'][label]])+' |')
    tv=[tiger[s][d]['mrr'] for s in SPLITS for d in DIRECTIONS];lines.append('| TIGER ESM2Text (literature) | '+' | '.join(f'{v:.6f}' for v in [*tv,float(np.mean(tv))])+' |')
    lines+=['','Our rows use all-positive MRR. TIGER values are published reference numbers; exact comparator candidate assets and its MRR implementation were not independently verified. No paired statistical claim is made against literature.','','See paired_bootstrap.csv for paired query intervals and reaction_cluster_bootstrap.csv for E2R reaction-group sensitivity against both fixed references. The six-cell macro is descriptive; dependence and unadjusted multiple comparisons limit uncertainty claims.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    if any(digest(path)!=checksum for path,checksum in lineage.values()):raise ValueError('Frozen lineage changed during evaluation')
    atomic_json(args.output/'complete.json',dict(schema='post_evaluation_calibration_official_receipt_v1',lineage={k:identity(v[0],True) for k,v in lineage.items()},inputs=provenance,bootstrap_replicates=args.bootstrap_replicates,bootstrap_seed=args.seed,evaluator=identity(__file__,True),metrics_source=identity(ROOT/'scripts/generalization_metrics.py',True),bootstrap_source=identity(ROOT/'scripts/generalization_phase2_official_evaluate.py',True),output=identity(args.output/'summary.json',True)))


if __name__=='__main__':main()
