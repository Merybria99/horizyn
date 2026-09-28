#!/usr/bin/env python3
"""Source-closure-only amendment for the fixed sixth-method Morgan supplement.

The original five-method report is immutable. Authenticate new model scores,
old scan/audit/results, all sources and evidence before computing this supplement.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/generalization_20260919_2251'
METHODS=['phase2','phase4','f3_native','f3_fp64','circev2','morgan_seed42']
CUTS=[1,10,25,100,1000,10000]
REQUIRED=[Path(__file__),ROOT/'scripts/generalization_morgan_transfer.py',
    ROOT/'scripts/generalization_large_case1_evaluate.py',ROOT/'scripts/generalization_external_evaluate.py']

def identity(path):
    p=Path(path).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest())

def checked(record):
    if identity(record['path'])['sha256']!=record['sha256']:raise ValueError('Changed artifact: '+record['path'])
    return Path(record['path']).resolve()

def same(a,b):return Path(a['path']).resolve()==Path(b['path']).resolve() and a['sha256']==b['sha256']

def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def validate_plan(plan):
    if plan.get('schema')!='morgan_large_case1_evaluation_plan_v1' or plan.get('methods')!=METHODS or plan.get('cutoffs')!=CUTS:
        raise ValueError('Require original five methods plus one fixed Morgan supplement and original cutoffs')
    if plan.get('primary_supplement')!='morgan_seed42' or plan.get('training_split')!='reaction_smi' or plan.get('seed')!=42:
        raise ValueError('Supplement method role changed')
    if plan.get('original_five_method_primary_unchanged') is not True or plan.get('development_exposed') is not True or plan.get('unknown_candidates_are_unlabeled') is not True:
        raise ValueError('Exploratory scope or unknown-label policy changed')

def validate_original_gate(audit,complete_record,protocol_record,primary):
    if audit.get('schema')!='large_case1_completion_integrity_audit_v1' or audit.get('all_exact') is not True:
        raise ValueError('Original numerical integrity audit is required')
    if not same(audit['complete'],complete_record) or not same(audit['protocol'],protocol_record):raise ValueError('Original integrity audit lineage mismatch')
    if primary.get('schema')!='large_case1_evaluation_complete_v1' or not same(primary['scan_receipt'],complete_record) or not same(primary['protocol'],protocol_record):
        raise ValueError('Original primary evaluation must succeed before supplement')

def validate_amendment(amendment,freeze_record,plan):
    if amendment.get('schema')!='morgan_large_evaluation_source_amendment_v1' or not same(amendment['original_freeze'],freeze_record):
        raise ValueError('Require the exact evaluation-only source closure amendment')
    protocol_record=plan['original_scan_protocol']
    if not same(amendment['original_scan_protocol'],protocol_record):raise ValueError('Wrong original scan protocol in amendment')
    protocol=json.loads(checked(protocol_record).read_text());expected=protocol['evaluation_source']
    if Path(expected['path']).resolve()!=ROOT/'scripts/generalization_large_case1_evaluate.py':raise ValueError('Unexpected original helper path')
    if not same(amendment['approved_original_evaluation_source'],expected):raise ValueError('Original helper identity changed')
    checked(expected)
    if not amendment.get('reason') or not amendment.get('implementation_sources'):raise ValueError('Incomplete amendment source closure')
    return expected

def authenticate(args):
    freeze_record=identity(args.freeze);freeze=json.loads(Path(args.freeze).read_text())
    if freeze.get('schema')!='post_evaluation_morgan_transfer_freeze_v1' or freeze.get('frozen_before_new_predictions') is not True:raise ValueError('Require immutable Morgan transfer freeze')
    planpath=checked(freeze['large_case1_evaluation_plan']);plan=json.loads(planpath.read_text());validate_plan(plan)
    amendment=json.loads(Path(args.amendment).read_text());expected=validate_amendment(amendment,freeze_record,plan)
    sources={str(checked(r)):r['sha256'] for r in freeze['implementation_sources']}
    for record in amendment['implementation_sources']:
        path=str(checked(record))
        if path in sources and sources[path]!=record['sha256']:raise ValueError('Amendment must not replace original frozen sources')
        sources[path]=record['sha256']
    if sources.get(str(Path(expected['path']).resolve()))!=expected['sha256']:raise ValueError('Missing original pinned helper in source union')
    for path in REQUIRED:
        if sources.get(str(path.resolve()))!=identity(path)['sha256']:raise ValueError('Missing frozen source: '+str(path))
    for record in plan['locked_artifacts']:checked(record)
    if not same(plan['original_scan_protocol'],identity(args.scan/'protocol.json')):raise ValueError('Wrong original fixed pool')
    sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    import generalization_morgan_transfer as transfer
    import generalization_large_case1_evaluate as original
    selected,receipt=transfer.checked_large_prediction(freeze_record,args.scores,args.scan)
    protocol,catalog,literature,mapping,scores,provenance=original.authenticate(args.scan,args.case1_audit)
    audit=json.loads((args.scan/'completion_audit.json').read_text())
    primary_path=RUN/'large_case1_evaluation/complete.json';primary=json.loads(primary_path.read_text())
    validate_original_gate(audit,identity(args.scan/'complete.json'),identity(args.scan/'protocol.json'),primary)
    for record in primary['outputs'].values():checked(record)
    if selected.shape!=scores['phase2'].shape or selected.dtype!=np.float32 or not np.isfinite(selected).all():raise ValueError('Supplement score dimension/dtype/finiteness mismatch')
    scores['morgan_seed42']=selected
    provenance.update(morgan_freeze=freeze_record,morgan_scores=identity(args.scores),morgan_receipt=identity(args.scores.parent/'complete.json'),
        plan=identity(planpath),amendment=identity(args.amendment),primary=identity(primary_path),integrity_audit=identity(args.scan/'completion_audit.json'))
    return original,plan,protocol,catalog,literature,mapping,scores,provenance

def run(args):
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Use a fresh supplemental output directory')
    original,plan,protocol,catalog,literature,mapping,scores,provenance=authenticate(args)
    args.output.mkdir(parents=True,exist_ok=True)
    write(args.output/'scores_authenticated.json',dict(authenticated_utc=datetime.now(timezone.utc).isoformat(),all_scores_authenticated=True,
        no_new_labels_opened=True,original_primary_already_completed=True,provenance=provenance))
    if not args.evaluate:return
    if not args.release_note.strip():raise ValueError('Parent outcome release note is required')
    write(args.output/'label_open_receipt.json',dict(opened_utc=datetime.now(timezone.utc).isoformat(),release_note=args.release_note,
        all_scores_authenticated_before_labels=True,development_exposed=True,independent_confirmation=False))
    from generalization_external_evaluate import case1_metadata
    metadata=case1_metadata(args.case1_audit)
    if metadata['catalog']['proteins']!=literature['proteins']:raise ValueError('Literature evidence order changed')
    old=json.loads((RUN/'large_case1_evaluation/summary.json').read_text())
    pids=literature['proteins'];indices=[mapping[p]['candidate_index'] for p in pids];n=len(catalog['proteins'])
    tiers={'primary_papers':'primary_papers_12','primary_papers_and_patents':'secondary_papers_and_patents_24','broad_workbook_active':'descriptive_workbook_reported_81'}
    results={};rows=[]
    for method in METHODS:
        ranks,_=original.reference_ranks(scores[method],indices)
        results[method]={tiers[name]:original.tier_summary(ranks,positions,n,CUTS) for name,positions in metadata['indices'].items()}
        if method!='morgan_seed42' and results[method]!=old['metrics'][method]:raise ValueError('Original immutable method metrics do not replay exactly: '+method)
        for i,p in enumerate(pids):
            row=dict(method=method,representative_id=p,candidate_id=catalog['proteins'][indices[i]],sequence_sha256=mapping[p]['sequence_sha256'],
                primary_paper=i in metadata['indices']['primary_papers'],paper_or_patent=i in metadata['indices']['primary_papers_and_patents'],
                broad_workbook_reported=i in metadata['indices']['broad_workbook_active'])
            row.update({k:float(v[i]) if k in ('score','expected_reciprocal_rank') else int(v[i]) for k,v in ranks.items()});rows.append(row)
    summary=dict(schema='morgan_large_case1_supplement_v1',primary_supplement='morgan_seed42',methods=METHODS,
        background_count=protocol['background_count'],candidate_count=n,cutoffs=CUTS,metrics=results,
        original_five_metric_replay_exact=True,original_primary_unchanged=True,development_exposed=True,independent_confirmation=False,
        limitations=plan['limitations'],provenance=provenance)
    write(args.output/'summary.json',summary);original.write_csv(args.output/'all_literature_reference_ranks.csv',rows)
    original.write_csv(args.output/'known_positive_ranks.csv',[r for r in rows if r['paper_or_patent']])
    lines=['# Fixed Morgan large-background supplement','',
        'Original five-method primary remains unchanged. This is a late exploratory sixth method, selected from validation before its transfer outcomes. Every unlisted background candidate remains unlabeled.','',
        '| Method | Evidence tier | Expected recovered @1 /10 /25 /100 /1000 /10000 | Expected all-positive MRR |','|---|---|---|---:|']
    for method,values in results.items():
        for tier,value in values.items():
            lines.append('| '+method+' | '+tier+' | '+' / '.join(f"{value['cutoffs'][str(k)]['uniform_tie_expected_recovered']:.3f}" for k in CUTS)+f" | {value['uniform_tie_expected_all_positive_mrr']:.8g} |")
    lines+=['']+plan['limitations'];(args.output/'readout.md').write_text('\n'.join(lines)+'\n')
    write(args.output/'complete.json',dict(schema='morgan_large_case1_evaluation_receipt_v1',freeze=provenance['morgan_freeze'],
        original_primary=provenance['primary'],outputs={p.name:identity(p) for p in sorted(args.output.iterdir()) if p.is_file()}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('freeze','scores','scan','case1-audit','output','amendment'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--evaluate',action='store_true');p.add_argument('--release-note',default='');args=p.parse_args()
    if args.evaluate and not args.release_note.strip():p.error('--evaluate requires --release-note')
    run(args)
if __name__=='__main__':main()
