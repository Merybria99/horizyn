#!/usr/bin/env python3
"""Execute the separately pinned, descriptive fixed-pool input-length audit."""
import argparse,csv,hashlib,json
from pathlib import Path
import h5py,numpy as np

ROOT=Path(__file__).resolve().parents[1]
PLAN_SHA='68be4ce3764969497bda707470b3cf27713309e7e41c6c054d021b220ab2f7a7'
def ident(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return dict(path=str(p),sha256=h.hexdigest())
def check(r):
    if ident(r['path'])['sha256']!=r['sha256']:raise ValueError('Source hash mismatch: '+r['path'])
    return Path(r['path'])
def js(p):return json.loads(Path(p).read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def weights(scores,k):
    if k>len(scores) or k<=0 or not np.isfinite(scores).all():raise ValueError('Invalid scores/cutoff')
    threshold=np.partition(scores,len(scores)-k)[len(scores)-k]
    higher=scores>threshold;ties=scores==threshold
    remaining=k-int(higher.sum());fraction=remaining/int(ties.sum())
    out=higher.astype(float)+ties*fraction
    stable=np.zeros(len(scores));stable[np.argsort(-scores,kind='stable')[:k]]=1.
    return out,stable,dict(kth_score=float(threshold),number_strictly_above=int(higher.sum()),boundary_tie_size=int(ties.sum()),boundary_fraction_included=fraction)

def describe(full,stored,appended,w):
    keep=w>0;full,stored,appended,w=full[keep],stored[keep],appended[keep],w[keep]
    n=float(w.sum());out={'n_or_weight_sum':n,'lengths':{},'counts':{}}
    for name,v in [('full_aa',full),('stored_residue',stored),('native_effective',np.minimum(stored,1022))]:
        order=np.argsort(v,kind='stable');v=v[order];ww=w[order];cdf=np.cumsum(ww)/n
        out['lengths'][name]=dict(weighted_mean=float(np.dot(v,ww)/n),minimum_positive_weight=int(v[0]),maximum_positive_weight=int(v[-1]),
            inverse_empirical_cdf_quantiles={str(q):int(v[min(int(np.searchsorted(cdf,q,side='left')),len(v)-1)]) for q in [.05,.25,.5,.75,.95]})
    tests={'full_aa_length==50':full==50,'full_aa_length<50':full<50,'full_aa_length==1024':full==1024,
        'full_aa_length>1024':full>1024,'stored_residue_length==1024':stored==1024,
        'stored_residue_length>1022':stored>1022,'full_aa_length<100':full<100,'is_appended_literature':appended}
    bins={'<50':full<50,'50':full==50,'51-99':(full>=51)&(full<=99),'100-249':(full>=100)&(full<=249),
        '250-499':(full>=250)&(full<=499),'500-1021':(full>=500)&(full<=1021),
        '1022-1023':(full>=1022)&(full<=1023),'1024':full==1024,'>1024':full>1024}
    for name,mask in tests.items():out['counts'][name]={'count':float(w[mask].sum()),'fraction':float(w[mask].sum()/n)}
    out['full_aa_bins']={name:{'count':float(w[mask].sum()),'fraction':float(w[mask].sum()/n)} for name,mask in bins.items()}
    return out

def acquire_lengths(plan,scan):
    selected=np.load(check(plan['metadata_sources']['selected_metadata']));indices=selected['indices'];ids=selected['ids'].astype(str)
    offsets=selected['offsets'];stored=offsets[indices+1]-offsets[indices]
    candidates=js(check(plan['metadata_sources']['candidate_catalog']))['proteins'];n=len(candidates);nb=len(ids)
    if candidates[:nb]!=ids.tolist():raise ValueError('Fixed sample ID order changed')
    full=np.zeros(n,dtype=np.int64);allstored=np.zeros(n,dtype=np.int64);allstored[:nb]=stored
    lookup={key:i for i,key in enumerate(ids)};source=plan['metadata_sources']['full_aa_metadata_table']
    # Hash while streaming CSV, avoiding a second read of the 509 MB table.
    digest=hashlib.sha256();found=np.zeros(nb,bool)
    def decoded_lines():
        with Path(source['path']).open('rb') as f:
            for line in f:digest.update(line);yield line.decode('utf-8')
    for row in csv.DictReader(decoded_lines()):
        pos=lookup.get(row['protein_id'])
        if pos is not None:
            if found[pos]:raise ValueError('Duplicate RefSeq metadata row')
            full[pos]=int(row['sequence_length']);found[pos]=True
    if digest.hexdigest()!=source['sha256'] or not found.all():raise ValueError('Full-AA metadata identity/coverage failed')
    if not np.array_equal(stored,np.minimum(full[:nb],1024)):raise ValueError('Stored/full RefSeq length mismatch')
    literature=js(check(plan['metadata_sources']['literature_catalog']));metadata=ROOT/literature['source_metadata'];h5=ROOT/literature['source_residues']
    with metadata.open() as f:rows={r['entry_id']:r for r in csv.DictReader(f)}
    with h5py.File(h5,'r') as f:
        keys=f['ids'].asstr()[:].tolist();off=f['offsets'][:]
    spans={key:int(b-a) for key,a,b in zip(keys,off[:-1],off[1:])}
    aliases=js(check(plan['metadata_sources']['aliases']))['groups'];litmask=np.zeros(n,bool)
    for group in aliases:
        rid=group['representative_id'];r=rows[rid];pos=group['candidate_index']
        if r['sha256']!=group['sequence_sha256']:raise ValueError('Literature sequence identity mismatch')
        length=int(r['length']);span=spans[rid]
        if full[pos] and (full[pos]!=length or allstored[pos]!=span):raise ValueError('Reused literature metadata mismatch')
        full[pos]=length;allstored[pos]=span;litmask[pos]=True
    if (full<=0).any() or (allstored<=0).any() or (allstored>full).any():raise ValueError('Missing/invalid candidate lengths')
    return full,allstored,litmask,dict(full_aa_table=source,literature_metadata=ident(metadata),literature_residue_metadata_source=str(h5),residue_vectors_read=False)

def run(args):
    plan=js(args.plan)
    if ident(args.plan)['sha256']!=PLAN_SHA:raise ValueError('Length plan changed')
    scan=args.scan;completion=js(scan/'complete.json');audit=js(scan/'completion_audit.json');primary=js(args.primary_evaluation/'complete.json')
    if audit.get('all_exact') is not True or primary.get('schema')!='large_case1_evaluation_complete_v1':raise ValueError('Require completed integrity and primary evaluation')
    check(primary['scan_receipt']);check(primary['protocol'])
    if primary['scan_receipt']['sha256']!=ident(scan/'complete.json')['sha256'] or primary['protocol']['sha256']!=plan['scan_protocol']['sha256']:raise ValueError('Primary evaluation belongs to another scan')
    if completion.get('schema')!='large_case1_scan_complete_v1':raise ValueError('Incomplete scan')
    if (args.output/'complete.json').exists():raise ValueError('Preserve the completed length audit')
    args.output.mkdir(parents=True,exist_ok=True)
    full,stored,litmask,metadata=acquire_lengths(plan,scan)
    n=len(full);nb=plan['background_count'];appended=np.arange(n)>=nb
    scores=np.load(check(completion['outputs']['scores']))
    if scores['ids'].tolist()!=js(check(plan['metadata_sources']['candidate_catalog']))['proteins']:raise ValueError('Score IDs mismatch')
    np.savez(args.output/'candidate_lengths.npz',ids=scores['ids'],full_aa=full,stored_residue=stored,is_literature=litmask,is_appended=appended)
    references={'sampled_background':describe(full,stored,appended,(np.arange(n)<nb).astype(float)),
        'full_fixed_candidate_pool':describe(full,stored,appended,np.ones(n)),
        'all_literature_sequences':describe(full,stored,appended,litmask.astype(float))}
    results={}
    for method in plan['methods']:
        values=scores[method];results[method]={}
        for k in plan['top_k']:
            w,stable,boundary=weights(values,k)
            result=dict(boundary=boundary,uniform_tie_expected=describe(full,stored,appended,w),stable_order_secondary=describe(full,stored,appended,stable))
            result['count_fraction_difference_from_background']={key:entry['fraction']-references['sampled_background']['counts'][key]['fraction'] for key,entry in result['uniform_tie_expected']['counts'].items()}
            results[method][str(k)]=result
    write(args.output/'summary.json',dict(schema='large_case1_descriptive_length_audit_v1',plan=ident(args.plan),primary_evaluation=ident(args.primary_evaluation/'complete.json'),
        completion_integrity_audit=ident(scan/'completion_audit.json'),methods=results,references=references,metadata=metadata,
        interpretation=plan['interpretation'],unknown_candidates_remain_unlabeled=True,candidate_filtering=False))
    write(args.output/'complete.json',dict(schema='large_case1_length_audit_complete_v1',source=ident(__file__),plan=ident(args.plan),
        outputs={p.name:ident(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name!='complete.json'}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','scan','primary-evaluation','output']:p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
