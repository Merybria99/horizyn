#!/usr/bin/env python3
"""Prespecified positive-training-endpoint exposure; no candidate filtering."""
import argparse,csv,hashlib,json
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

METHODS=['phase2','phase4','f3_native','f3_fp64','circev2']
CUTS=[25,100,1000]
def identity(path):
    p=Path(path).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest())
def checked(r):
    if identity(r['path'])['sha256']!=r['sha256']:raise ValueError('Changed source: '+r['path'])
    return Path(r['path'])
def js(p):return json.loads(Path(p).read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def same(a,b):return a['sha256']==b['sha256'] and Path(a['path']).resolve()==Path(b['path']).resolve()

def fasta_records(record):
    """One streaming read; full-file identity checked before iterator completion."""
    h=hashlib.sha256();name=None;parts=[]
    with Path(record['path']).open('rb') as f:
        for line in f:
            h.update(line)
            if line.startswith(b'>'):
                if name is not None:yield name,b''.join(parts).upper()
                name=line[1:].split()[0].decode();parts=[]
            else:parts.append(b''.join(line.split()))
        if name is not None:yield name,b''.join(parts).upper()
    if h.hexdigest()!=record['sha256']:raise ValueError('Full FASTA checksum mismatch')

def exact_matches(sequence,buckets):
    # A digest hit alone never establishes equality; X and other symbols are literal.
    return sorted(pid for original,pid in buckets.get(hashlib.sha256(sequence).digest(),[]) if sequence==original)

def top_exposure(scores,seen,k):
    scores=np.asarray(scores);seen=np.asarray(seen,dtype=bool)
    if scores.ndim!=1 or seen.shape!=scores.shape or not np.isfinite(scores).all() or not 0<k<=len(scores):raise ValueError('Invalid fixed cutoff inputs')
    threshold=np.partition(scores,len(scores)-k)[len(scores)-k]
    above=scores>threshold;tied=scores==threshold;take=k-int(above.sum())
    expected=float(seen[above].sum()+take*seen[tied].mean())
    stable=int(seen[np.argsort(-scores,kind='stable')[:k]].sum())
    return dict(k=k,uniform_tie_expected_seen_count=expected,uniform_tie_expected_seen_fraction=expected/k,
        stable_order_seen_count=stable,stable_order_seen_fraction=stable/k,
        boundary_score=float(threshold),strictly_above=int(above.sum()),boundary_tie_size=int(tied.sum()),boundary_fraction_included=take/int(tied.sum()))

def gate(plan):
    if plan.get('schema')!='large_case1_training_exposure_plan_v1' or plan.get('methods')!=METHODS or plan.get('cutoffs')!=CUTS or plan.get('scores_read_for_selection') is not False:raise ValueError('Fixed exposure plan changed')
    protocol=checked(plan['scan_protocol']);scan=protocol.parent
    complete_path=scan/'complete.json';complete=js(complete_path);audit=js(scan/'completion_audit.json')
    primary_path=Path(plan['primary_evaluation_required']);primary=js(primary_path)
    if complete.get('schema')!='large_case1_scan_complete_v1' or complete.get('complete_fixed_candidate_coverage') is not True or complete.get('labels_used') is not False:raise ValueError('Incomplete original scan')
    if audit.get('schema')!='large_case1_completion_integrity_audit_v1' or audit.get('all_exact') is not True or audit.get('labels_read') is not False:raise ValueError('Require passing numerical integrity audit')
    if not same(audit['complete'],identity(complete_path)) or not same(audit['protocol'],plan['scan_protocol']):raise ValueError('Audit belongs to another scan')
    if not same(audit['plan'],plan['small_sources']['numerical_audit_plan']):raise ValueError('Numerical audit plan changed')
    if primary.get('schema')!='large_case1_evaluation_complete_v1' or not same(primary['scan_receipt'],identity(complete_path)) or not same(primary['protocol'],plan['scan_protocol']):raise ValueError('Primary evaluation belongs to another scan')
    for r in primary['outputs'].values():checked(r)
    opened=js(primary['outputs']['label_open_receipt.json']['path'])
    if datetime.fromisoformat(plan['created_utc'])>=datetime.fromisoformat(opened['opened_at_utc']):raise ValueError('Exposure plan must predate original outcome access')
    for key in ['source','test_source']:checked(plan[key])
    for r in plan['small_sources'].values():checked(r)
    return scan,complete,primary

def run(args):
    if identity(args.plan)['sha256']!=args.expected_plan_sha:raise ValueError('Changed locked plan')
    plan=js(args.plan);scan,complete,primary=gate(plan)
    if not same(plan['source'],identity(__file__)):raise ValueError('Wrong exposure implementation')
    if (args.output/'complete.json').exists():raise ValueError('Preserve completed exposure output')
    cat=js(plan['small_sources']['training_catalog']['path'])
    with np.load(plan['small_sources']['training_edges']['path']) as f:edges=f['train']
    trainids={cat['proteins'][int(i)] for i in edges[:,1]}
    if len(trainids)!=147299 or sorted(trainids)!=cat['train_proteins']:raise ValueError('Actual positive training endpoints changed')
    buckets=defaultdict(list);found=set()
    for pid,seq in fasta_records(plan['training_fasta']):
        if pid in trainids:
            if pid in found:raise ValueError('Duplicate training FASTA ID')
            found.add(pid);buckets[hashlib.sha256(seq).digest()].append((seq,pid))
    if found!=trainids or len({seq for values in buckets.values() for seq,p in values})!=147299:raise ValueError('Training sequence coverage/uniqueness changed')
    catalog=js(plan['small_sources']['candidate_catalog']['path']);ids=catalog['proteins'];nb=plan['background_count'];lookup={p:i for i,p in enumerate(ids[:nb])}
    if len(lookup)!=nb or len(ids)!=plan['candidate_count']:raise ValueError('Fixed candidate pool changed')
    exposure=np.zeros(len(ids),bool);lengths=np.zeros(len(ids),np.int64);matched={};selected_seen=np.zeros(nb,bool);count=0
    for pid,seq in fasta_records(plan['refseq_fasta']):
        count+=1;index=lookup.get(pid)
        if index is None:continue
        if selected_seen[index]:raise ValueError('Duplicate selected FASTA ID')
        selected_seen[index]=True;lengths[index]=len(seq);hits=exact_matches(seq,buckets)
        exposure[index]=bool(hits)
        if hits:matched[index]=hits
    if count!=3944613 or not selected_seen.all():raise ValueError('RefSeq source or selected coverage changed')
    aliases=js(plan['small_sources']['aliases']['path'])['groups'];source={}
    for pid,seq in fasta_records(plan['literature_fasta']):
        if pid in source:raise ValueError('Duplicate literature FASTA ID')
        source[pid]=seq
    refs=[]
    for group in aliases:
        pid=group['representative_id'];seq=source[pid];index=group['candidate_index']
        if hashlib.sha256(seq).hexdigest()!=group['sequence_sha256'] or ids[index]!=group['candidate_id']:raise ValueError('Literature sequence/alias mismatch')
        hits=exact_matches(seq,buckets);exposure[index]=bool(hits);lengths[index]=len(seq)
        if hits:matched[index]=hits
        refs.append(dict(representative_id=pid,candidate_id=ids[index],candidate_index=index,sequence_sha256=group['sequence_sha256'],full_aa_length=len(seq),positive_training_endpoint_seen=bool(hits),matching_training_ids=hits))
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez(args.output/'candidate_exposure.npz',ids=np.asarray(ids),positive_training_endpoint_seen=exposure,full_aa_length=lengths)
    with (args.output/'exact_training_matches.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['candidate_index','candidate_id','matching_training_ids']);w.writeheader()
        for i in sorted(matched):w.writerow(dict(candidate_index=i,candidate_id=ids[i],matching_training_ids=';'.join(matched[i])))
    with np.load(checked(complete['outputs']['scores'])) as data:
        if data['ids'].tolist()!=ids:raise ValueError('Final scores and candidate order disagree')
        results={method:{str(k):top_exposure(data[method],exposure,k) for k in CUTS} for method in METHODS}
    masks={'sampled_background':np.arange(len(ids))<nb,'full_fixed_pool':np.ones(len(ids),bool),'all123_literature':np.isin(np.arange(len(ids)),[r['candidate_index'] for r in refs])}
    reference={name:dict(candidate_count=int(mask.sum()),positive_training_endpoint_seen=int(exposure[mask].sum()),seen_fraction=float(exposure[mask].mean())) for name,mask in masks.items()}
    for method,values in results.items():
        for value in values.values():value['descriptive_fraction_difference_from_background']=value['uniform_tie_expected_seen_fraction']-reference['sampled_background']['seen_fraction']
    write(args.output/'summary.json',dict(schema='large_case1_training_exposure_v1',supplemental=True,plan=identity(args.plan),
        positive_training_endpoint_count=len(trainids),exact_full_string_equality_verified_on_every_hash_hit=True,
        references=reference,methods=results,literature=refs,limitations=plan['limitations'],no_candidate_filtering=True,
        source_passes=dict(training_fasta=1,refseq_fasta=1,literature_fasta=1,residue_vectors=0)))
    write(args.output/'complete.json',dict(schema='large_case1_training_exposure_complete_v1',plan=identity(args.plan),source=identity(__file__),
        primary_evaluation=identity(plan['primary_evaluation_required']),numerical_audit=identity(scan/'completion_audit.json'),
        outputs={p.name:identity(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name not in ['complete.json','plan.json']}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--expected-plan-sha',required=True);run(p.parse_args())
