#!/usr/bin/env python3
"""Known-catalyst recovery against a locked, unassayed RefSeq background.

Authenticate completed score/candidate artifacts before opening literature
evidence. Unknown background proteins are never assigned inactive labels.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

METHODS=('phase2','phase4','f3_native','f3_fp64','circev2')
CUTS=(1,10,25,100,1000,10000)
LABEL_SOURCES=('candidate_evidence.json','positive_sets.json','source_verification.json','features/catalog.json')


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024**2),b''):value.update(block)
    return value.hexdigest()


def identity(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size)


def checked_identity(path,expected):
    value=identity(path)
    if value['sha256']!=expected:raise ValueError('Checksum mismatch: '+str(path))
    return value


def resolve(base,value):
    path=Path(value)
    return path.resolve() if path.is_absolute() else (base/path).resolve()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def selected_indices(protocol):
    tiles=np.asarray([r['tile_index'] if isinstance(r,dict) else r for r in protocol['tiles']],dtype=np.int64)
    expected=np.sort(np.random.default_rng(20260920).choice((3944613+4095)//4096,256,replace=False))
    if not np.array_equal(tiles,expected):raise ValueError('Tile sample differs from the fixed 256-of-964 seed20260920 selection')
    for tile in protocol['tiles']:
        if isinstance(tile,dict) and (tile['start']!=tile['tile_index']*4096 or tile['stop']!=min((tile['tile_index']+1)*4096,3944613)):
            raise ValueError('Tile boundaries differ from the global 4096-row frame')
    indices=np.concatenate([np.arange(t*4096,min((t+1)*4096,3944613),dtype=np.int64) for t in tiles])
    if len(indices)!=protocol['background_count']:raise ValueError('Background count differs from the selected tile population')
    return tiles,indices


def reference_ranks(scores,indices):
    """Stable source-order ranks and unbiased expectations inside exact ties."""
    scores=np.asarray(scores);indices=np.asarray(indices,dtype=np.int64)
    if scores.ndim!=1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Require a finite one-dimensional candidate score vector')
    if indices.ndim!=1 or len(set(indices.tolist()))!=len(indices) or (indices<0).any() or (indices>=len(scores)).any():
        raise ValueError('Reference candidate indices must be unique and in range')
    order=np.argsort(-scores,kind='stable');inverse=np.empty(len(scores),dtype=np.int64)
    inverse[order]=np.arange(1,len(scores)+1)
    ascending=np.sort(scores);values=scores[indices]
    lower=np.searchsorted(ascending,values,side='left');upper=np.searchsorted(ascending,values,side='right')
    best=len(scores)-upper+1;worst=len(scores)-lower;size=upper-lower
    harmonic=np.r_[0.,np.cumsum(1./np.arange(1,len(scores)+1,dtype=np.float64))]
    expected_rr=(harmonic[worst]-harmonic[best-1])/size
    return dict(candidate_index=indices,score=values,stable_rank=inverse[indices],best_rank=best,worst_rank=worst,
        tie_size=size,expected_reciprocal_rank=expected_rr),order


def tier_summary(ranks,positions,n,cuts=CUTS):
    positions=np.asarray(sorted(positions),dtype=np.int64)
    if not len(positions):raise ValueError('A reported evidence tier must contain positives')
    first=ranks['best_rank'][positions];last=ranks['worst_rank'][positions];ties=ranks['tie_size'][positions]
    out=dict(known_positive_unique_sequences=len(positions),candidate_count=n,
        stable_order_all_positive_mrr=float(np.mean(1./ranks['stable_rank'][positions])),
        uniform_tie_expected_all_positive_mrr=float(np.mean(ranks['expected_reciprocal_rank'][positions])),
        best_stable_positive_rank=int(ranks['stable_rank'][positions].min()),
        positives_in_exact_score_ties=int(np.count_nonzero(ties>1)),cutoffs={})
    for cutoff in cuts:
        k=min(cutoff,n)
        expected=np.clip(k-first+1,0,ties)/ties
        out['cutoffs'][str(cutoff)]=dict(stable_recovered=int(np.count_nonzero(ranks['stable_rank'][positions]<=k)),
            guaranteed_recovered=int(np.count_nonzero(last<=k)),possible_recovered=int(np.count_nonzero(first<=k)),
            uniform_tie_expected_recovered=float(expected.sum()),uniform_tie_expected_recall=float(expected.mean()),
            uniform_random_order_expected_recovered=k*len(positions)/n,uniform_random_order_expected_recall=k/n)
    return out


def validate_aliases(aliases,catalog,literature,background_count,full_match):
    ids=catalog['proteins'];groups=aliases['groups'] if isinstance(aliases,dict) else aliases
    source={r['representative_id']:r for r in literature['groups']}
    if len(groups)!=123 or len(source)!=123 or len(literature['proteins'])!=123:
        raise ValueError('All 123 unique literature reference sequences are required')
    mapping={r['representative_id']:r for r in groups}
    if len(mapping)!=123 or set(mapping)!=set(source):raise ValueError('Duplicate or missing literature representatives')
    if len({r['sequence_sha256'] for r in groups})!=123:raise ValueError('Literature sequences must be exactly deduplicated')
    occupied=set();appended=[];background=set(ids[:background_count])
    for p in literature['proteins']:
        r=mapping[p];original=source[p];index=r['candidate_index']
        if type(index) is not int or not 0<=index<len(ids) or index in occupied:
            raise ValueError('Literature aliases must map once to distinct valid candidate positions')
        occupied.add(index)
        if r['candidate_id']!=ids[index] or r['sequence_sha256']!=original['sequence_sha256'] or sorted(r['all_entry_ids'])!=sorted(original['all_entry_ids']):
            raise ValueError('Literature alias and canonical sequence/catalog identity disagree')
        if type(r['already_in_selected_background']) is not bool or r['already_in_selected_background']!=(index<background_count):
            raise ValueError('Selected-background alias flag disagrees with candidate position')
        match=full_match.get(p,{})
        if match and match['sequence_sha256']!=original['sequence_sha256']:
            raise ValueError('Full RefSeq sequence audit and literature sequence disagree')
        selected=[key for key in match.get('refseq_ids',[]) if key in background]
        if len(selected)>1 or r['candidate_id']!=(selected[0] if selected else 'LIT_'+p):
            raise ValueError('Literature candidate mapping differs from authenticated full-sequence matches')
        if not r['already_in_selected_background']:appended.append(index)
    if appended!=list(range(background_count,len(ids))):
        raise ValueError('Every appended reference must follow canonical literature order, without extra candidates')
    return mapping


def validate_chunk_partition(receipts,protocol):
    """Require ordered full coverage and a separate canonical literature tail."""
    expected=protocol['tiles']
    if len(receipts)!=len(expected)+1:raise ValueError('Require every selected tile plus one literature receipt')
    for receipt,tile in zip(receipts[:-1],expected):
        if type(receipt.get('tile_index')) is not int or receipt['tile_index']!=tile['tile_index'] or receipt['rows']!=tile['stop']-tile['start']:
            raise ValueError('Background chunk order or row count differs from the locked sample')
    if receipts[-1].get('tile_index')!='literature' or receipts[-1]['rows']!=protocol['candidate_count']-protocol['background_count']:
        raise ValueError('Literature chunk must contain every appended reference exactly once')
    if sum(r['rows'] for r in receipts)!=protocol['candidate_count']:
        raise ValueError('Chunk rows do not cover the complete candidate catalog')


def authenticate(scan,case_audit):
    complete_path=scan/'complete.json';complete=json.loads(complete_path.read_text())
    if complete.get('schema')!='large_case1_scan_complete_v1' or complete.get('labels_used') is not False:
        raise ValueError('Require a complete label-free scan')
    protocol_record=complete['protocol'];protocol_path=resolve(scan,protocol_record['path'])
    protocol_identity=checked_identity(protocol_path,protocol_record['sha256']);protocol=json.loads(protocol_path.read_text())
    if (protocol.get('schema')!='large_case1_input_protocol_v1' or protocol.get('frozen_before_scores') is not True
        or protocol.get('labels_read_for_selection') is not False or protocol.get('sampled_retrieval_authorized') is not True
        or protocol.get('full_pool_scan') is not False or protocol.get('candidate_expansion_after_scores') is not False):
        raise ValueError('Require the authorized immutable label-independent sampled protocol')
    if complete.get('complete_fixed_candidate_coverage') is not True or complete.get('candidate_expansion_after_scores') is not False or complete.get('methods')!=list(METHODS):
        raise ValueError('Incomplete coverage or changed methods cannot be evaluated')
    evaluation_source=protocol['evaluation_source']
    if str(Path(evaluation_source['path']).resolve())!=str(Path(__file__).resolve()) or evaluation_source['sha256']!=sha256(__file__):
        raise ValueError('Evaluation source must have been pinned by the prescore protocol')
    if protocol['methods']!=list(METHODS):raise ValueError('Retain exactly the five fixed methods')
    if protocol.get('cutoffs')!=list(CUTS):raise ValueError('Retain the prespecified retrieval cutoffs')
    fixed_sampling=dict(seed=20260920,universe=3944613,tile_size=4096,universe_tiles=964,tile_count=256,without_replacement=True,sorted_for_io=True)
    if any(protocol['sampling'].get(key)!=value for key,value in fixed_sampling.items()):
        raise ValueError('Sampling metadata differs from the fixed design')
    tiles,expected_indices=selected_indices(protocol)
    provenance={str(complete_path):identity(complete_path),str(protocol_path):protocol_identity,str(Path(__file__).resolve()):identity(__file__)}
    dependencies=protocol['evaluation_dependencies']
    required=Path(__file__).with_name('generalization_external_evaluate.py').resolve()
    if str(required) not in {str(resolve(protocol_path.parent,r['path'])) for r in dependencies}:
        raise ValueError('Prescore protocol must authenticate the literature metadata helper before importing it')
    for record in protocol['implementation_sources']+dependencies:
        path=resolve(protocol_path.parent,record['path']);provenance[str(path)]=checked_identity(path,record['sha256'])
    audit_record=protocol['candidate_input_audit'];audit_path=resolve(protocol_path.parent,audit_record['path'])
    provenance[str(audit_path)]=checked_identity(audit_path,audit_record['sha256']);audit=json.loads(audit_path.read_text())
    if audit['refseq_unique_exact_sequences']!=3944613 or audit['refseq_duplicate_sequence_records']!=0 or not audit['all_fasta_ids_match_shard_order']:
        raise ValueError('Require the authenticated unique full-sequence RefSeq input audit')
    for name,record in complete['outputs'].items():
        p=resolve(scan,record['path']);provenance[str(p)]=checked_identity(p,record['sha256'])
    for name in ['catalog','aliases','selected_metadata']:
        before=protocol['artifacts'][name];after=complete['outputs'][name]
        if before['sha256']!=after['sha256'] or resolve(protocol_path.parent,before['path'])!=resolve(scan,after['path']):
            raise ValueError('Candidate or alias metadata changed after score selection was locked')
    receipt_paths=[];receipts=[]
    for record in complete['chunk_receipts']:
        p=resolve(scan,record['path']);provenance[str(p)]=checked_identity(p,record['sha256'])
        receipt=json.loads(p.read_text());receipts.append(receipt);receipt_paths.append(str(p))
        if receipt.get('schema')!='large_case1_chunk_v1' or receipt.get('labels_used') is not False or receipt['protocol']['sha256']!=protocol_identity['sha256'] or resolve(p.parent,receipt['protocol']['path'])!=protocol_path:
            raise ValueError('Every chunk must belong to this label-free protocol')
    if len(set(receipt_paths))!=len(receipt_paths):raise ValueError('Duplicate chunk receipts')
    validate_chunk_partition(receipts,protocol)
    for name in ['candidate_count','background_count']:
        if complete['counts'][name]!=protocol[name]:raise ValueError('Completed candidate counts differ from the protocol')
    if complete['counts']['tile_count']!=len(protocol['tiles']):raise ValueError('Completed tile count differs from the locked sample')
    catalog_path=resolve(scan,complete['outputs']['catalog']['path']);catalog=json.loads(catalog_path.read_text())
    ids=catalog['proteins'];n=len(ids)
    if n!=protocol['candidate_count'] or len(set(ids))!=n or len(catalog['query_ids'])!=1:
        raise ValueError('Require one query and unique candidate IDs matching the complete protocol')
    selected_path=resolve(scan,complete['outputs']['selected_metadata']['path'])
    with np.load(selected_path,allow_pickle=False) as selected:
        if not np.array_equal(selected['indices'],expected_indices) or selected['ids'].astype(str).tolist()!=ids[:protocol['background_count']]:
            raise ValueError('Selected global indices/IDs do not match the locked tile frame and candidate order')
    score_path=resolve(scan,complete['outputs']['scores']['path'])
    with np.load(score_path,allow_pickle=False) as data:
        if data['ids'].astype(str).tolist()!=ids:raise ValueError('Score IDs differ from the locked candidate order')
        scores={key:np.asarray(data[key]) for key in METHODS}
    if any(v.shape!=(n,) or v.dtype!=np.float32 or not np.isfinite(v).all() for v in scores.values()):
        raise ValueError('Each fixed method requires a complete finite FP32 score vector')
    offset=0
    for receipt,receipt_path in zip(receipts,receipt_paths):
        record=receipt['outputs']['scores'];path=resolve(Path(receipt_path).parent,record['path'])
        provenance[str(path)]=checked_identity(path,record['sha256']);stop=offset+receipt['rows']
        with np.load(path,allow_pickle=False) as chunk:
            chunk_ids=chunk['ids'].astype(str).tolist()
            if chunk_ids!=ids[offset:stop] or hashlib.sha256('\n'.join(chunk_ids).encode()).hexdigest()!=receipt['ids_sha256']:
                raise ValueError('Chunk score identities differ from the complete locked catalog')
            for method in METHODS:
                if chunk[method].dtype!=np.float32 or not np.array_equal(chunk[method],scores[method][offset:stop]):
                    raise ValueError('Merged scores differ from their authenticated method-specific chunks')
        offset=stop
    reference=complete['reference_check'];reference_path=resolve(scan,reference['path'])
    provenance[str(reference_path)]=checked_identity(reference_path,reference['sha256']);check=json.loads(reference_path.read_text())
    expected_checks={phase+'_'+method for phase in ('phase2','phase4') for method in (phase,'f3_native','f3_fp64')}
    if (check.get('all_exact') is not True or check.get('labels_used') is not False
        or check['protocol']['sha256']!=protocol_identity['sha256'] or set(check['checks'])!=expected_checks
        or not all(value is True for value in check['checks'].values())
        or check['circev2_shared_native_parity']['sha256']!=protocol['native_reference_receipt']['sha256']):
        raise ValueError('Require exact prior 123-reference score parity for every declared control')
    pinned={str(resolve(protocol_path.parent,r['path'])):r for r in protocol['evaluation_label_sources']}
    for name in LABEL_SOURCES:
        path=(case_audit/name).resolve()
        if str(path) not in pinned:raise ValueError('Every literature evidence source must be pinned before the scan')
        provenance[str(path)]=checked_identity(path,pinned[str(path)]['sha256'])
    # Catalog contains sequence aliases, not activities. All labels are still unopened.
    literature=json.loads((case_audit/'features/catalog.json').read_text())
    aliases=json.loads(resolve(scan,complete['outputs']['aliases']['path']).read_text())
    mapping=validate_aliases(aliases,catalog,literature,protocol['background_count'],audit['matching_literature_aliases'])
    return protocol,catalog,literature,mapping,scores,provenance


def run(args):
    protocol,catalog,literature,mapping,scores,provenance=authenticate(args.scan.resolve(),args.case1_audit.resolve())
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Use a fresh evaluation directory')
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'label_open_receipt.json',dict(opened_at_utc=datetime.now(timezone.utc).isoformat(),
        all_five_scores_and_prescore_candidate_metadata_authenticated=True,protocol=provenance[str((args.scan/'protocol.json').resolve())],
        evaluator=identity(__file__),activity_negatives_inferred=False))
    from generalization_external_evaluate import case1_metadata
    metadata=case1_metadata(args.case1_audit)
    if metadata['catalog']['proteins']!=literature['proteins']:raise ValueError('Literature evidence/catalog order changed')
    pids=literature['proteins'];indices=[mapping[p]['candidate_index'] for p in pids];ids=catalog['proteins'];n=len(ids)
    tier_names={'primary_papers':'primary_papers_12','primary_papers_and_patents':'secondary_papers_and_patents_24','broad_workbook_active':'descriptive_workbook_reported_81'}
    results={};reference_rows=[];tops=[]
    for method,score in scores.items():
        ranks,order=reference_ranks(score,indices)
        results[method]={tier_names[name]:tier_summary(ranks,positions,n) for name,positions in metadata['indices'].items()}
        for i,p in enumerate(pids):
            row=dict(method=method,representative_id=p,candidate_id=ids[indices[i]],sequence_sha256=mapping[p]['sequence_sha256'],
                all_entry_ids=';'.join(mapping[p]['all_entry_ids']),already_in_selected_background=mapping[p]['already_in_selected_background'],
                primary_paper=i in metadata['indices']['primary_papers'],paper_or_patent=i in metadata['indices']['primary_papers_and_patents'],
                broad_workbook_reported=i in metadata['indices']['broad_workbook_active'])
            row.update({key:float(value[i]) if key in ['score','expected_reciprocal_rank'] else int(value[i]) for key,value in ranks.items()})
            reference_rows.append(row)
        for rank,index in enumerate(order[:100],start=1):
            tops.append(dict(method=method,stable_rank=rank,candidate_id=ids[index],candidate_index=int(index),score=float(score[index]),
                candidate_source='sampled_refseq' if index<protocol['background_count'] else 'appended_literature',
                activity_status='Not inferred from rank; see separately verified literature evidence where available.'))
    summary=dict(schema='large_case1_known_catalyst_recovery_v1',methods=list(METHODS),background_count=protocol['background_count'],
        candidate_count=n,literature_unique_sequences=123,literature_original_rows=144,
        literature_mapped_to_background=sum(r['already_in_selected_background'] for r in mapping.values()),
        literature_appended=sum(not r['already_in_selected_background'] for r in mapping.values()),cutoffs=list(CUTS),metrics=results,
        scope='Exploratory large-background stress test of the already examined Case1 reaction. Fixed sampled RefSeq background plus forced inclusion of all 123 literature sequences; no natural-prevalence or full-RefSeq extrapolation.',
        interpretation='Known-catalyst recovery only. Unlisted RefSeq proteins are unlabeled, not inactive. No global AUROC/AP, precision, specificity, candidate-IID confidence intervals, or novel biochemical validation.',
        tie_policy='Stable sampled-RefSeq/VDS then appended-literature order is shown for reproducibility. Primary recovery and mean reciprocal rank use uniform ordering within exact-score ties; best/worst possible rank intervals retained.',
        evidence_tiers='Twelve independently rechecked paper-positive unique sequences primary; 24 paper+patent positives secondary; 81 workbook-reported active sequences descriptive. Sequence groups counted once, aliases never duplicate positives.',
        sampling='Cluster sample of 256 of 964 global 4096-row VDS tiles using fixed seed20260920, without replacement. Candidate selection is independent of score/activity but adjacent source rows may be taxonomically related. One sampled catalog and one query supply no sample-replication uncertainty.',
        provenance=provenance)
    write_json(args.output/'summary.json',summary)
    write_csv(args.output/'all_literature_reference_ranks.csv',reference_rows)
    write_csv(args.output/'known_positive_ranks.csv',[r for r in reference_rows if r['paper_or_patent']])
    write_csv(args.output/'top100_per_method.csv',tops)
    text=['# Locked large-background Case1 recovery','',summary['scope'],'',summary['interpretation'],'',
        f"{summary['background_count']:,} sampled RefSeq sequences + {summary['literature_appended']} appended literature sequences = {n:,} distinct candidates. All123 literature sequence groups are represented.",'',
        '| Method | Evidence tier | Expected known positives @1 /10 /25 /100 /1000 /10000 | Uniform-tie mean reciprocal rank |',
        '|---|---|---|---:|']
    for method,tiers in results.items():
        for tier,value in tiers.items():
            text.append('| '+method+' | '+tier+' | '+' / '.join(f"{value['cutoffs'][str(k)]['uniform_tie_expected_recovered']:.3f}" for k in CUTS)+f" | {value['uniform_tie_expected_all_positive_mrr']:.6g} |")
    text.extend(['',summary['tie_policy'],'',summary['sampling'],'',summary['evidence_tiers'],'',
        'Uniform-random expectations condition on this fixed mixed candidate catalog; they do not estimate natural catalytic prevalence. Every reference rank, tie interval, method result, and source hash is retained. No method was selected from this background test.',''])
    (args.output/'readout.md').write_text('\n'.join(text))
    write_json(args.output/'complete.json',dict(schema='large_case1_evaluation_complete_v1',protocol=identity(args.scan/'protocol.json'),
        scan_receipt=identity(args.scan/'complete.json'),evaluator=identity(__file__),
        outputs={p.name:identity(p) for p in sorted(args.output.iterdir()) if p.is_file()}))
    print(json.dumps({'complete':str(args.output/'complete.json'),'candidate_count':n,'methods':list(METHODS)}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan',type=Path,required=True);p.add_argument('--case1-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
