#!/usr/bin/env python3
"""Guard-only compatibility amendment; original scientific evaluation unchanged.

Existing assay definitions and evaluators remain immutable. Every prediction,
source and plan authenticates before assay parsing. No model selection occurs.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/generalization_20260919_2251'
METHODS={'calibrated_seed42':(42,'selected'),'calibrated_seed17':(17,'selected'),
    'calibrated_seed73':(73,'selected'),'phase2_seed42':(42,'parent'),
    'phase2_seed17':(17,'parent'),'phase2_seed73':(73,'parent'),
    'F3_native':(42,'baseline'),'F3_fp64':(42,'baseline_fp64')}
SHAPES={'nitrilase':(38,18),'aminotransferase':(18,25),'case1':(1,123)}
REQUIRED=[Path(__file__),ROOT/'scripts/generalization_calibration_transfer.py',
    ROOT/'scripts/generalization_calibration_evaluation_guard.py',
    ROOT/'scripts/generalization_external_evaluate.py',ROOT/'scripts/generalization_nitrilase_evaluate.py',
    ROOT/'scripts/generalization_phase4_external_evaluate.py',
    RUN/'aminotransferase_audit/evaluate_panel.py']

def identity(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def checked(record):
    result=identity(record['path'])
    if result['sha256']!=record['sha256']:raise ValueError('Source identity changed: '+result['path'])
    return Path(result['path'])

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def validate_amendment(amendment,freeze_record):
    if amendment.get('schema')!='calibration_evaluation_guard_amendment_v1':
        raise ValueError('Require an explicit evaluation-only guard amendment')
    record=amendment.get('original_freeze',{})
    if record.get('sha256')!=freeze_record['sha256'] or Path(record.get('path','')).resolve()!=Path(freeze_record['path']).resolve():
        raise ValueError('Guard amendment belongs to another original freeze')
    if not amendment.get('reason') or not amendment.get('implementation_sources'):
        raise ValueError('Guard amendment must record reason and implementation closure')
    expected={str((RUN/(name+'_audit/features/feature_bundle_receipt.json')).resolve()) for name in ('nitrilase','aminotransferase')}
    records=amendment.get('approved_missing_freeze_receipts',[])
    if len(records)!=2 or {str(Path(r['path']).resolve()) for r in records}!=expected:
        raise ValueError('Only the two exact prior external receipts may omit freeze_sha256')
    for record in records:checked(record)

def validate_contract(freeze,plan):
    if freeze.get('schema')!='post_evaluation_calibration_transfer_freeze_v1' or freeze.get('frozen_before_new_predictions') is not True or freeze.get('development_exposed') is not True:
        raise ValueError('Require the development-exposed calibration freeze')
    rule=freeze.get('rule',{})
    if (rule.get('gamma_enzyme'),rule.get('gamma_reaction'),rule.get('enzyme_mean_scheme'))!=(0.,1.,'uniform_proteins'):
        raise ValueError('Only the fixed training-enzyme mean correction is authorized')
    if plan.get('schema')!='calibration_external_evaluation_plan_v1' or plan.get('methods')!=list(METHODS):
        raise ValueError('Retain all eight bound methods in the fixed order')
    if plan.get('panels')!=list(SHAPES) or plan.get('primary')!='calibrated_seed42':
        raise ValueError('Fixed panel and primary definitions changed')
    if plan.get('training_split')!='reaction_smi' or plan.get('development_exposed') is not True:
        raise ValueError('External training split/exposure statement changed')
    if (plan.get('bootstrap_replicates'),plan.get('permutation_replicates'),plan.get('seed'))!=(10000,1000,20260920):
        raise ValueError('Use the original fixed uncertainty protocol')

def rank_invariance(parent,selected):
    """Record ordering and exact tie changes; numeric shifts need not be zero."""
    a,b=np.asarray(parent),np.asarray(selected)
    if a.shape!=b.shape or a.ndim!=2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid invariance score matrices')
    rows=[]
    for i,(x,y) in enumerate(zip(a,b)):
        order=np.argsort(-x,kind='stable');other=np.argsort(-y,kind='stable')
        same_order=bool(np.array_equal(order,other))
        # Adjacent original-order inequalities capture all weak-order changes.
        before=np.sign(np.diff(x[order]));after=np.sign(np.diff(y[order]))
        same_ties=bool(np.array_equal(before,after))
        delta=y.astype(np.float64)-x.astype(np.float64)
        rows.append(dict(query_index=i,stable_order_identical=same_order,
            weak_order_and_ties_identical=same_ties,
            score_shift_min=float(delta.min()),score_shift_max=float(delta.max()),
            score_shift_spread=float(np.ptp(delta))))
    return dict(all_stable_orders_identical=all(r['stable_order_identical'] for r in rows),
        all_weak_orders_and_ties_identical=all(r['weak_order_and_ties_identical'] for r in rows),
        rows=rows,policy='A per-reaction constant is mathematically R2E rank-invariant. FP32 serialization can create or remove ties; any discrepancy is reported, never silently asserted invariant.')

def authenticate(freeze_path,amendment_path):
    freeze_record=identity(freeze_path);freeze=json.loads(Path(freeze_path).read_text())
    amendment=json.loads(Path(amendment_path).read_text())
    validate_amendment(amendment,freeze_record)
    plan_path=checked(freeze['external_evaluation_plan']);plan=json.loads(plan_path.read_text())
    validate_contract(freeze,plan)
    pinned={str(checked(r)):r['sha256'] for r in freeze['implementation_sources']}
    for record in amendment['implementation_sources']:
        path=str(checked(record))
        if path in pinned and pinned[path]!=record['sha256']:raise ValueError('Amendment must not replace an original frozen source')
        pinned[path]=record['sha256']
    for p in REQUIRED:
        if pinned.get(str(p.resolve()))!=identity(p)['sha256']:raise ValueError('Missing frozen evaluator source: '+str(p))
    for r in plan['locked_artifacts']:checked(r)
    # Imports follow full source authentication, before any assay labels are read.
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
    import generalization_calibration_evaluation_guard as transfer
    scores={};catalogs={};provenance={'freeze':freeze_record,'amendment':identity(amendment_path),'plan':identity(plan_path),'predictions':{}}
    for name,shape in SHAPES.items():
        catpath=RUN/(name+'_audit/features/catalog.json')
        catalogs[name]=json.loads(catpath.read_text());scores[name]={}
        for label,(seed,key) in METHODS.items():
            path=freeze_path.parent/f'predictions/{name}/seed{seed}/scores.npz'
            values,receipt=transfer.checked_prediction(freeze_record,path,catpath,'reaction_smi',seed,key,amendment=amendment)
            if values.shape!=shape or not np.isfinite(values).all():raise ValueError('Score shape/nonfinite value: '+name+'/'+label)
            scores[name][label]=values
            provenance['predictions'][name+'/'+label]={'scores':identity(path),'receipt':identity(path.parent/'complete.json'),'score_key':key,'seed':seed}
    return freeze_record,plan,catalogs,scores,provenance

def rectangles(catalog,keep,overlap):
    absent_r=np.array([q not in overlap['matches']['contains_pair_neutralized'] for q in catalog['query_ids']])
    absent_p=np.array([p not in overlap['exact_sequence_matches'] for p in catalog['proteins']])
    if (len(keep),int(keep.sum()),int(absent_r.sum()),int(absent_p.sum()))!=(25,24,3,20):
        raise ValueError('Original input-only AT rectangles changed')
    return [('full_25_primary',np.ones(18,bool),np.ones(25,bool)),
        ('complete_sequence_24_sensitivity',np.ones(18,bool),keep),
        ('pair_absent_3_reactions_diagnostic',absent_r,np.ones(25,bool)),
        ('exact_sequence_absent_20_enzymes_diagnostic',np.ones(18,bool),absent_p),
        ('joint_3_reactions_20_enzymes_diagnostic',absent_r,absent_p)]

def evaluate_universe(scores,labels,plan,metrics,rates=None,censored=None,at=None,phase4=None):
    results={k:metrics.evaluate_matrix(v,labels) for k,v in scores.items()}
    comparisons={}
    refs=['F3_fp64','phase2_seed42']
    for ref in refs:
        comparisons[ref]={k:metrics.paired_bootstrap(v,results[ref],plan['bootstrap_replicates'],plan['seed']) for k,v in results.items() if k!=ref}
    paired_seeds={f'calibrated_seed{s}':metrics.paired_bootstrap(results[f'calibrated_seed{s}'],results[f'phase2_seed{s}'],plan['bootstrap_replicates'],plan['seed']) for s in (42,17,73)}
    permutations={}
    for label,value in scores.items():
        print('permutation',labels.shape,label,flush=True)
        permutations[label]=metrics.permutation_control(value,labels,plan['permutation_replicates'],plan['seed'])
    out=dict(panel={'reactions':labels.shape[0],'enzymes':labels.shape[1],'measured_cells':labels.size,
        'positive':int(labels.sum()),'conditional_non_detect':int(labels.size-labels.sum())},
        primary_direction='enzyme_to_reaction',
        metrics={k:{d:r['summary'] for d,r in v.items()} for k,v in results.items()},
        paired_bootstrap=comparisons,paired_seed_parent_bootstrap=paired_seeds,reaction_permutation=permutations,
        candidate_pool_policy='Every measured cell in the named fixed rectangle is retained. Diagnostic rectangles change candidate pools and are not comparable to the full-panel denominator.',
        per_query=results)
    if rates is not None:
        association={k:phase4.label_defined_rank_association(v,rates,censored,at.censored_spearman) for k,v in scores.items()}
        out['censored_rank_association']=association
        out['rank_association_bootstrap']={ref:{k:at.association_bootstrap(v,association[ref],plan['bootstrap_replicates'],plan['seed']) for k,v in association.items() if k!=ref} for ref in refs}
    return out

def run(args):
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Use a fresh output directory')
    freeze,plan,catalogs,scores,provenance=authenticate(args.freeze.resolve(),args.amendment.resolve())
    args.output.mkdir(parents=True,exist_ok=True)
    write(args.output/'scores_authenticated.json',dict(authenticated_utc=datetime.now(timezone.utc).isoformat(),
        all_scores_authenticated=True,labels_parsed=False,development_exposed=True,provenance=provenance))
    if not args.evaluate:return
    if not args.release_note.strip():raise ValueError('Record the parent release before evaluation')
    write(args.output/'label_open_receipt.json',dict(opened_utc=datetime.now(timezone.utc).isoformat(),
        all_scores_authenticated_before_labels=True,release_note=args.release_note,
        development_exposed=True,independent_confirmation=False))
    import generalization_nitrilase_evaluate as metrics
    import generalization_phase4_external_evaluate as phase4
    p=RUN/'aminotransferase_audit/evaluate_panel.py';spec=importlib.util.spec_from_file_location('calibration_fixed_at',p)
    at=importlib.util.module_from_spec(spec);spec.loader.exec_module(at)
    outputs={};invariance={}
    for panel in SHAPES:
        invariance[panel]={f'seed{s}':rank_invariance(scores[panel][f'phase2_seed{s}'],scores[panel][f'calibrated_seed{s}']) for s in (42,17,73)}
    write(args.output/'r2e_invariance.json',invariance)
    # Case1 is an invariant-ranking check only; its activity labels are not read.
    nr=RUN/'nitrilase_audit';lock=json.loads((nr/'selection_lock.json').read_text())
    y=metrics.load_labels(catalogs['nitrilase'],ROOT/lock['labels_locked']['path'],684)
    if (int(y.sum()),int((y.sum(1)==0).sum()),int((y.sum(0)==0).sum()))!=(85,3,8):raise ValueError('Nitrilase label counts changed')
    source_validation=metrics.validate_underlying_release(ROOT,catalogs['nitrilase'],y,lock)
    outputs['nitrilase']={'full_18x38':evaluate_universe(scores['nitrilase'],y,plan,metrics)}
    write(args.output/'nitrilase.json',outputs['nitrilase'])
    ar=RUN/'aminotransferase_audit';cat=catalogs['aminotransferase']
    y=metrics.load_labels(cat,ar/'sequestered/candidates.csv',450);rates,censored=at.load_assays(cat,ar/'sequestered/primary_assays.csv',y)
    if int(y.sum())!=186:raise ValueError('AT label counts changed')
    with (ar/'inputs/proteins.csv').open() as f:complete={r['protein_id'] for r in csv.DictReader(f) if int(r['unknown_residue_count'])==0}
    keep=np.array([p in complete for p in cat['proteins']])
    overlap=json.loads((ar/'f3_train_overlap.json').read_text())['splits']['reaction_smi']
    outputs['aminotransferase']={}
    for name,rows,columns in rectangles(cat,keep,overlap):
        ss={k:v[rows][:,columns] for k,v in scores['aminotransferase'].items()}
        outputs['aminotransferase'][name]=evaluate_universe(ss,y[rows][:,columns],plan,metrics,rates[rows][:,columns],censored[rows][:,columns],at,phase4)
        write(args.output/f'aminotransferase_{name}.json',outputs['aminotransferase'][name])
    summary=dict(schema='post_evaluation_calibration_external_v1',freeze=freeze,primary='calibrated_seed42',
        methods=list(METHODS),development_exposed=True,independent_confirmation=False,
        primary_direction='enzyme_to_reaction',primary_references=['phase2_seed42','F3_fp64'],
        unchanged_primary_campaign='Frozen phase2/phase4 reports and five-method large-pool scan remain unchanged.',
        nitrilase_source_validation=source_validation,
        limitations=plan['limitations'],provenance=provenance,panels=outputs,r2e_invariance=invariance)
    write(args.output/'summary.json',summary)
    write(args.output/'complete.json',dict(schema='calibration_external_evaluation_receipt_v1',freeze=freeze,
        development_exposed=True,outputs={p.name:identity(p) for p in sorted(args.output.iterdir()) if p.is_file()}))
    print('complete',args.output,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--amendment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--evaluate',action='store_true')
    p.add_argument('--release-note',default='');args=p.parse_args()
    if args.evaluate and not args.release_note.strip():p.error('--evaluate requires a recorded parent release')
    run(args)
if __name__=='__main__':main()
