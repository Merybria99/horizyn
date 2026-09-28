#!/usr/bin/env python3
"""Fixed post-evaluation Morgan transfer; original campaign remains unchanged.

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
METHODS={'morgan_seed42':(42,'selected'),'morgan_seed17':(17,'selected'),
    'morgan_seed73':(73,'selected'),'phase2_seed42':(42,'parent'),
    'phase2_seed17':(17,'parent'),'phase2_seed73':(73,'parent'),
    'F3_native':(42,'baseline'),'F3_fp64':(42,'baseline_fp64')}
SHAPES={'nitrilase':(38,18),'aminotransferase':(18,25),'case1':(1,123),'p450':(191,490)}
REQUIRED=[Path(__file__),ROOT/'scripts/generalization_morgan_transfer.py',
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

def validate_contract(freeze,plan):
    if freeze.get('schema')!='post_evaluation_morgan_transfer_freeze_v1' or freeze.get('frozen_before_new_predictions') is not True or freeze.get('development_exposed') is not True:
        raise ValueError('Require the development-exposed frozen Morgan transfer')
    expected=dict(radius=3,eta_reaction=1.,fp_size=4096,include_chirality=True,alpha=.25,temperature=.03)
    if freeze.get('rule')!=expected:raise ValueError('Only the fixed selected Morgan rule is authorized')
    if plan.get('schema')!='morgan_external_evaluation_plan_v1' or plan.get('methods')!=list(METHODS):
        raise ValueError('Retain all eight bound methods in the fixed order')
    if plan.get('panels')!=list(SHAPES) or plan.get('primary')!='morgan_seed42':
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

def authenticate(freeze_path):
    freeze_record=identity(freeze_path);freeze=json.loads(Path(freeze_path).read_text())
    plan_path=checked(freeze['external_evaluation_plan']);plan=json.loads(plan_path.read_text())
    validate_contract(freeze,plan)
    pinned={str(checked(r)):r['sha256'] for r in freeze['implementation_sources']}
    for p in REQUIRED:
        if pinned.get(str(p.resolve()))!=identity(p)['sha256']:raise ValueError('Missing frozen evaluator source: '+str(p))
    for r in plan['locked_artifacts']:checked(r)
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
    import generalization_morgan_transfer as transfer
    scores={};catalogs={};provenance={'freeze':freeze_record,'plan':identity(plan_path),'predictions':{}}
    for name,shape in SHAPES.items():
        catpath=RUN/(name+'_audit/features/catalog.json')
        catalogs[name]=json.loads(catpath.read_text());scores[name]={}
        for label,(seed,key) in METHODS.items():
            path=freeze_path.parent/f'predictions/{name}/seed{seed}/scores.npz'
            values,receipt=transfer.checked_prediction(freeze_record,path,catpath,'reaction_smi',seed,key)
            if receipt.get('panel')!=name:raise ValueError('Prediction panel role changed')
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
    paired_seeds={f'morgan_seed{s}':metrics.paired_bootstrap(results[f'morgan_seed{s}'],results[f'phase2_seed{s}'],plan['bootstrap_replicates'],plan['seed']) for s in (42,17,73)}
    permutations={}
    for label,value in scores.items():
        print('permutation',labels.shape,label,flush=True)
        permutations[label]=metrics.permutation_control(value,labels,plan['permutation_replicates'],plan['seed'])
    out=dict(panel={'reactions':labels.shape[0],'enzymes':labels.shape[1],'measured_cells':labels.size,
        'positive':int(labels.sum()),'conditional_non_detect':int(labels.size-labels.sum())},
        primary_direction='both_directions_fixed',
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
    freeze,plan,catalogs,scores,provenance=authenticate(args.freeze.resolve())
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
    p=RUN/'aminotransferase_audit/evaluate_panel.py';spec=importlib.util.spec_from_file_location('morgan_fixed_at',p)
    at=importlib.util.module_from_spec(spec);spec.loader.exec_module(at)
    import generalization_external_evaluate as ext
    outputs={}
    case=ext.case1_metadata(RUN/'case1_audit');case_results={};rank_rows=[];group_rows=[]
    for label,value in scores['case1'].items():
        result,rows,groups=ext.evaluate_case1(value,case,label)
        case_results[label]=result;rank_rows.extend(rows);group_rows.extend(groups)
    outputs['case1']=case_results
    ext.write_csv(args.output/'case1_all_candidates.csv',rank_rows)
    ext.write_csv(args.output/'case1_study_and_lineage_summary.csv',group_rows)
    write(args.output/'case1.json',case_results)
    p450=ext.p450_metadata(RUN/'p450_audit','reaction_smi')
    raw_p450={label:ext.evaluate_p450(value,p450)[0] for label,value in scores['p450'].items()}
    comparisons={}
    for reference in ('F3_fp64','phase2_seed42'):
        comparisons[reference]={label:{stratum:{direction:ext.paired_bootstrap(v['per_query'],raw_p450[reference][stratum][direction]['per_query'],plan['bootstrap_replicates'],plan['seed']) for direction,v in strata.items()} for stratum,strata in value.items()} for label,value in raw_p450.items() if label!=reference}
    outputs['p450']=dict(metrics={label:{stratum:{direction:v['summary'] for direction,v in strata.items()} for stratum,strata in value.items()} for label,value in raw_p450.items()},
        paired_bootstrap=comparisons,reaction_permutation={label:ext.reaction_permutation_control(value,p450,plan['permutation_replicates'],plan['seed']) for label,value in scores['p450'].items()})
    write(args.output/'p450.json',outputs['p450'])
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
    summary=dict(schema='post_evaluation_morgan_external_v1',freeze=freeze,primary='morgan_seed42',
        methods=list(METHODS),development_exposed=True,independent_confirmation=False,
        primary_direction='both_directions_fixed',primary_references=['phase2_seed42','F3_fp64'],
        unchanged_primary_campaign='Frozen phase2/phase4 reports and five-method large-pool scan remain unchanged.',
        nitrilase_source_validation=source_validation,
        limitations=plan['limitations'],provenance=provenance,panels=outputs)
    write(args.output/'summary.json',summary)
    write(args.output/'complete.json',dict(schema='morgan_external_evaluation_receipt_v1',freeze=freeze,
        development_exposed=True,outputs={p.name:identity(p) for p in sorted(args.output.iterdir()) if p.is_file()}))
    print('complete',args.output,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--evaluate',action='store_true')
    p.add_argument('--release-note',default='');args=p.parse_args()
    if args.evaluate and not args.release_note.strip():p.error('--evaluate requires a recorded parent release')
    run(args)
if __name__=='__main__':main()
