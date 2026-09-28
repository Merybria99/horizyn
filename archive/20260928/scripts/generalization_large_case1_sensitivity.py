#!/usr/bin/env python3
"""Prespecified supplemental nested-background sensitivity; fixed scores only.

Preparation reads candidate metadata only. Evaluation requires successful complete
primary evaluation and the independent postcompletion numerical audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

METHODS = ('phase2', 'phase4', 'f3_native', 'f3_fp64', 'circev2')
SIZES = (1000, 10000, 100000, 1044645)
CUTS = (25, 100, 1000)
TIERS = {'primary_papers_12': ('primary_paper', 12),
         'secondary_papers_and_patents_24': ('paper_or_patent', 24)}
ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'runs/generalization_20260919_2251'


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def checked(record):
    path = Path(record['path'])
    if identity(path)['sha256'] != record['sha256']:
        raise ValueError('Artifact changed: ' + str(path))
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def priority_order(ids):
    """Fixed score-independent SHA256 priority; IDs unique, original row index retained."""
    if len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError('Require unique nonempty string IDs')
    records = [(hashlib.sha256(b'20260920\0' + x.encode('utf-8')).digest(), x, i)
               for i, x in enumerate(ids)]
    return np.asarray([r[2] for r in sorted(records)], dtype=np.int64)


def subset_indices(order, count, background_count, candidate_count):
    if not 0 < count <= background_count or len(order) != background_count:
        raise ValueError('Invalid nested background size')
    # Ranking always retains the primary source order, never the hash priority order.
    return np.r_[np.sort(order[:count]), np.arange(background_count, candidate_count, dtype=np.int64)]


def prepare(args):
    scan = args.scan.resolve()
    if (scan / 'complete.json').exists() or (args.primary / 'complete.json').exists():
        raise ValueError('Pin this supplemental plan before scan completion/outcome access')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a fresh immutable supplemental directory')
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = scan / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if protocol['background_count'] != SIZES[-1] or protocol['candidate_count'] != SIZES[-1] + 123:
        raise ValueError('This fixed sensitivity requires the authorized pool and all123 appended references')
    if protocol['methods'] != list(METHODS):
        raise ValueError('Unexpected primary methods')
    catalog_path = checked(protocol['artifacts']['catalog'])
    aliases_path = checked(protocol['artifacts']['aliases'])
    checked(protocol['artifacts']['selected_metadata'])
    catalog = json.loads(catalog_path.read_text())
    aliases = json.loads(aliases_path.read_text())['groups']
    ids = catalog['proteins']
    if len(ids) != protocol['candidate_count'] or len(set(ids)) != len(ids):
        raise ValueError('Invalid complete candidate catalog')
    if len(aliases) != 123 or any(r['already_in_selected_background'] for r in aliases):
        raise ValueError('Expected all123 literature sequences in the fixed appended tail')
    if sorted(r['candidate_index'] for r in aliases) != list(range(SIZES[-1], len(ids))):
        raise ValueError('Literature tail coverage differs')
    order = priority_order(ids[:SIZES[-1]])
    state_path = args.output / 'background_priority.npz'
    np.savez_compressed(state_path, priority_original_index=order)
    selections = []
    for size in SIZES:
        selected = subset_indices(order, size, SIZES[-1], len(ids))
        selections.append(dict(background_count=size, candidate_count=len(selected),
                               source_order_indices_sha256=hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest(),
                               source_order_ids_sha256=hashlib.sha256('\n'.join(ids[i] for i in selected).encode()).hexdigest()))
    plan = dict(schema='large_case1_nested_background_sensitivity_plan_v1',
                created_utc=datetime.now(timezone.utc).isoformat(), supplemental=True,
                primary_protocol_unchanged=True, scores_read_for_selection=False, labels_read_for_selection=False,
                created_before_scan_completion=True, development_exposed_case1=True,
                primary_protocol=identity(protocol_path), input_artifacts={k:protocol['artifacts'][k] for k in ('catalog','aliases','selected_metadata')},
                analysis_source=identity(__file__), test_source=identity(ROOT/'tests/unit/test_generalization_large_case1_sensitivity.py'),
                metric_dependency=protocol['evaluation_source'], selection_state=identity(state_path),
                seed=20260920, priority_rule='Sort SHA256(b"20260920\\0" + actual_background_id.encode("utf-8")) bytes ascending; ties by ID then original index.',
                nested_background_sizes=list(SIZES), selections=selections,
                candidate_order='Selected background rows in primary original VDS order, followed by all123 fixed canonical literature rows.',
                methods=list(METHODS), evidence_tiers={k:v[1] for k,v in TIERS.items()}, cutoffs=list(CUTS),
                primary_metric='Uniform-exact-tie expected known-positive recovery and recall at25/100/1000, only when cutoff <= subset candidate count.',
                secondary_metric='Stable original-order recovery/recall; all positive best/worst tied ranks retained.',
                completion_audit_required=str(scan/'completion_audit.json'), primary_evaluation_required=str(args.primary.resolve()/'complete.json'),
                completion_audit_plan=identity(scan/'audit_plan.json'),
                execution_gate='Run only after complete original pool, completion_audit all_exact true, and successful authenticated original primary evaluation.',
                limitations=['All123 literature sequences are forcibly retained; this is conditional known-catalyst recovery, not natural prevalence.',
                             'The outer background is a fixed256-tile cluster sample; hash subsampling does not make it a full-RefSeq or independent candidate sample.',
                             'Nested sets are strongly dependent; one query and one hash seed provide no replication uncertainty.',
                             'Unassayed background is unlabeled: no AUC, specificity, activity precision, global AP, candidate-IID confidence intervals or full4M extrapolation.',
                             'No subset size, method, model or threshold is selected from these results; the full original pool remains primary.'])
    write_json(args.output/'plan.json', plan)
    (args.output/'README.md').write_text('# Supplemental nested-background sensitivity\n\n'
        'The original five-method1,044,768-candidate analysis remains primary. This prespecified diagnostic retains all123 literature sequences and uses nested SHA256-ID-selected backgrounds of1,000,10,000,100,000 and1,044,645 proteins. It reads no scores until original completion, numerical audit and primary evaluation succeed.\n\n'
        'Candidate order remains original VDS order then the fixed literature tail. Both12-paper and24-paper/patent tiers report uniform-tie expected recovery/recall at25/100/1000, with stable-order sensitivity and positive tie intervals. All sizes and methods are reported; nothing is tuned. Forced reference inclusion, the original block-cluster sample, correlated nested sets and unknown background activities limit interpretation.\n\n'
        'See plan.json for exact byte framing, immutable metadata/source hashes, subset identity hashes and execution gates. background_priority.npz records the full input-only priority permutation.\n')
    print(json.dumps(identity(args.output/'plan.json')))


def check_prerequisites(plan):
    """Fail before reading scores or any activity-bearing primary output."""
    if plan.get('schema') != 'large_case1_nested_background_sensitivity_plan_v1' or not plan.get('primary_protocol_unchanged'):
        raise ValueError('Wrong supplemental plan')
    if plan.get('scores_read_for_selection') is not False or plan.get('labels_read_for_selection') is not False:
        raise ValueError('Require input-only prespecification')
    if plan['methods'] != list(METHODS) or plan['nested_background_sizes'] != list(SIZES) or plan['cutoffs'] != list(CUTS):
        raise ValueError('Prespecified method/size/cutoff grid differs')
    for record in [plan['analysis_source'],plan['test_source'],plan['metric_dependency'],plan['selection_state'],plan['primary_protocol'],plan['completion_audit_plan'],*plan['input_artifacts'].values()]:
        checked(record)
    if Path(plan['analysis_source']['path']).resolve() != Path(__file__).resolve():
        raise ValueError('Wrong analysis implementation')
    protocol_path = Path(plan['primary_protocol']['path'])
    scan = protocol_path.parent
    complete_path = scan/'complete.json'
    complete = json.loads(complete_path.read_text())
    audit_path = Path(plan['completion_audit_required'])
    audit = json.loads(audit_path.read_text())
    primary_path = Path(plan['primary_evaluation_required'])
    primary = json.loads(primary_path.read_text())
    if complete.get('complete_fixed_candidate_coverage') is not True or complete.get('labels_used') is not False:
        raise ValueError('Incomplete primary pool')
    if audit.get('schema')!='large_case1_completion_integrity_audit_v1' or audit.get('all_exact') is not True or audit.get('labels_read') is not False:
        raise ValueError('Require successful independent numerical completion audit')
    if checked(audit['complete']).resolve()!=complete_path.resolve() or audit['protocol']['sha256']!=plan['primary_protocol']['sha256']:
        raise ValueError('Numerical audit applies to another scan')
    if audit['plan']['sha256']!=plan['completion_audit_plan']['sha256'] or checked(audit['plan']).resolve()!=Path(plan['completion_audit_plan']['path']).resolve():
        raise ValueError('Numerical audit plan changed')
    if primary.get('schema') != 'large_case1_evaluation_complete_v1':
        raise ValueError('Require completed original primary evaluation')
    for artifact in [complete['protocol'],primary['protocol']]:
        if artifact['sha256'] != plan['primary_protocol']['sha256'] or checked(artifact).resolve() != protocol_path.resolve():
            raise ValueError('Primary protocol changed')
    if checked(primary['scan_receipt']).resolve() != complete_path.resolve():
        raise ValueError('Primary evaluation used a different scan')
    if primary['evaluator']['sha256'] != plan['metric_dependency']['sha256']:
        raise ValueError('Different primary evaluator')
    # Check all small primary outputs before using their independently verified evidence flags.
    for record in primary['outputs'].values():checked(record)
    opened=json.loads(checked(primary['outputs']['label_open_receipt.json']).read_text())
    if datetime.fromisoformat(plan['created_utc'])>=datetime.fromisoformat(opened['opened_at_utc']):
        raise ValueError('Supplemental plan was not pinned before primary outcome access')
    for name in ('catalog','aliases','selected_metadata'):
        if complete['outputs'][name]['sha256'] != plan['input_artifacts'][name]['sha256']:
            raise ValueError('Primary candidate metadata changed')
    checked(complete['outputs']['scores'])
    return complete,primary,primary_path.parent,audit_path


def run(args):
    plan_path = args.output/'plan.json'
    plan = json.loads(plan_path.read_text())
    if (args.output/'complete.json').exists():raise ValueError('Supplemental output is immutable')
    complete,primary,primary_dir,audit_path = check_prerequisites(plan)
    dependency = Path(plan['metric_dependency']['path'])
    spec = importlib.util.spec_from_file_location('locked_primary_case1_evaluator', dependency)
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    catalog = json.loads(checked(plan['input_artifacts']['catalog']).read_text())
    ids = catalog['proteins'];background_count = SIZES[-1]
    with np.load(checked(plan['selection_state']),allow_pickle=False) as data:order = data['priority_original_index']
    if not np.array_equal(order,priority_order(ids[:background_count])):raise ValueError('Nested priority permutation changed')
    aliases = json.loads(checked(plan['input_artifacts']['aliases']).read_text())['groups']
    aliases = sorted(aliases,key=lambda r:r['candidate_index'])
    rows = list(csv.DictReader(checked(primary['outputs']['all_literature_reference_ranks.csv']).open()))
    summary = json.loads(checked(primary['outputs']['summary.json']).read_text())
    by_method = {method:[r for r in rows if r['method']==method] for method in METHODS}
    reference = by_method[METHODS[0]]
    if len(reference)!=123 or [int(r['candidate_index']) for r in reference] != [r['candidate_index'] for r in aliases]:
        raise ValueError('Primary literature order changed')
    for method,values in by_method.items():
        if len(values)!=123:raise ValueError('Incomplete primary literature rows')
        for row,base,alias in zip(values,reference,aliases):
            for key in ('candidate_index','candidate_id','representative_id','sequence_sha256','primary_paper','paper_or_patent'):
                if row[key]!=base[key]:raise ValueError('Primary evidence mapping varies by method')
            if row['candidate_id']!=alias['candidate_id'] or row['sequence_sha256']!=alias['sequence_sha256']:
                raise ValueError('Primary evidence aliases differ from prespecified catalog')
    positions = {name:[i for i,r in enumerate(reference) if r[field]=='True'] for name,(field,_) in TIERS.items()}
    if any(len(positions[name]) != count for name,(_,count) in TIERS.items()):raise ValueError('Unexpected evidence tier counts')
    with np.load(checked(complete['outputs']['scores']),allow_pickle=False) as data:
        if data['ids'].astype(str).tolist()!=ids:raise ValueError('Full score/catalog order mismatch')
        scores = {method:data[method] for method in METHODS}
    metrics,positive_rows = [],[]
    all_pool_exact = True
    for size,selection in zip(SIZES,plan['selections']):
        subset = subset_indices(order,size,background_count,len(ids))
        if hashlib.sha256(subset.astype('<i8').tobytes()).hexdigest()!=selection['source_order_indices_sha256']:
            raise ValueError('Subset membership changed')
        if hashlib.sha256('\n'.join(ids[i] for i in subset).encode()).hexdigest()!=selection['source_order_ids_sha256']:
            raise ValueError('Subset candidate identities changed')
        refs = np.arange(size,len(subset),dtype=np.int64)
        for method,full in scores.items():
            if full.dtype!=np.float32 or full.shape!=(len(ids),) or not np.isfinite(full).all():raise ValueError('Invalid primary scores')
            ranks,_ = helper.reference_ranks(full[subset],refs)
            for tier,positive_positions in positions.items():
                result=helper.tier_summary(ranks,positive_positions,len(subset),cuts=CUTS)
                for cutoff in CUTS:
                    if cutoff>len(subset):continue
                    value=result['cutoffs'][str(cutoff)]
                    metrics.append(dict(background_count=size,candidate_count=len(subset),method=method,tier=tier,
                                        known_positive_count=len(positive_positions),cutoff=cutoff,
                                        stable_recovered=value['stable_recovered'],stable_recall=value['stable_recovered']/len(positive_positions),
                                        guaranteed_recovered=value['guaranteed_recovered'],possible_recovered=value['possible_recovered'],
                                        uniform_tie_expected_recovered=value['uniform_tie_expected_recovered'],
                                        uniform_tie_expected_recall=value['uniform_tie_expected_recall']))
                    if size==background_count:
                        target=summary['metrics'][method][tier]['cutoffs'][str(cutoff)]
                        if any(value[key]!=target[key] for key in value):raise ValueError('Full background does not replay the primary result exactly')
                for i in positive_positions:
                    positive_rows.append(dict(background_count=size,candidate_count=len(subset),method=method,tier=tier,
                        representative_id=reference[i]['representative_id'],candidate_id=reference[i]['candidate_id'],
                        stable_rank=int(ranks['stable_rank'][i]),best_rank=int(ranks['best_rank'][i]),worst_rank=int(ranks['worst_rank'][i]),tie_size=int(ranks['tie_size'][i])))
    helper.write_csv(args.output/'metrics.csv',metrics)
    helper.write_csv(args.output/'positive_rank_intervals.csv',positive_rows)
    result=dict(schema='large_case1_nested_background_sensitivity_v1',supplemental=True,primary_protocol_unchanged=True,
                all_full_pool_metrics_exactly_replay_primary=all_pool_exact,metrics=metrics,limitations=plan['limitations'],
                no_model_or_pool_size_selected=True,selection_was_input_only=True)
    write_json(args.output/'summary.json',result)
    write_json(args.output/'complete.json',dict(schema='large_case1_nested_background_sensitivity_complete_v1',
               plan=identity(plan_path),primary_evaluation=identity(Path(plan['primary_evaluation_required'])),completion_audit=identity(audit_path),
               source=identity(__file__),outputs={p.name:identity(p) for p in (args.output/'metrics.csv',args.output/'positive_rank_intervals.csv',args.output/'summary.json')},
               primary_protocol_unchanged=True,all_full_pool_metrics_exactly_replay_primary=True))
    print(json.dumps(identity(args.output/'complete.json')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--scan',type=Path,default=CAMPAIGN/'large_case1_scan')
    parser.add_argument('--primary',type=Path,default=CAMPAIGN/'large_case1_evaluation')
    parser.add_argument('--output',type=Path,default=CAMPAIGN/'large_case1_background_sensitivity')
    args=parser.parse_args()
    (prepare if args.action=='prepare' else run)(args)


if __name__=='__main__':main()
