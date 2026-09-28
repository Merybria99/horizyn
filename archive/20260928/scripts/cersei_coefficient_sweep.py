#!/usr/bin/env python3
"""Fixed-checkpoint CERSEI inference sensitivity, full official candidate pools.

Freeze a protocol with --initialize, then run one --task per GPU. This script
never trains, selects a checkpoint, or changes the deployed default recipe.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from generalization_reactzyme_architecture_phase2 import load_head, compose, canonical_dot, evaluate_scores
from generalization_clipzyme_f3_screen import model_from_checkpoint, export_device_lock
from generalization_clipzyme_screening_evaluate import evaluate_query
from horizyn.training_io import StoragePrecisionResidues
from horizyn.benchmarks.retrieval import encode_residue_targets

CROSS = ROOT / 'runs/generalization_20260919_2251/cross_paper_retraining'
OLD = CROSS / 'v4_biological_geometry_20260921_v1'
DEFAULT = dict(alpha=.4, cap=.5, nu=3.)
METRICS = ('bedroc85', 'bedroc20', 'ef0.05', 'ef0.1')

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()
def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(obj, indent=2)+'\n'); tmp.replace(p)
def stamp(): return datetime.now(timezone.utc).isoformat()
def tag(r): return 'a%03d_c%03d_n%03d' % tuple(round(r[k]*100) for k in ('alpha','cap','nu'))
def log(**kw): print(json.dumps(dict(utc=stamp(), **kw)), flush=True)

def initialize(out):
    if (out/'protocol.json').exists(): raise FileExistsError(out/'protocol.json')
    grids = dict(alpha=[i/20 for i in range(21)], cap=[i/4 for i in range(7)],
                 nu=[0.,.5,1.,1.5,2.,2.5,3.,4.,5.,6.],
                 interaction_alpha=[0.,.2,.4,.6,.8,1.], interaction_cap=[0.,.5,1.,1.5])
    arms = {}
    def add(r, group):
        key = tag(r)
        if key not in arms: arms[key] = dict(id=key, **r, groups=[])
        if group not in arms[key]['groups']: arms[key]['groups'].append(group)
    add(DEFAULT.copy(), 'default')
    for k in ('alpha','cap','nu'):
        for v in grids[k]: add(dict(DEFAULT, **{k:v}), k)
    for a in grids['interaction_alpha']:
        for c in grids['interaction_cap']: add(dict(alpha=a,cap=c,nu=3.), 'interaction')
    add(dict(alpha=0.,cap=0.,nu=1.), 'native_neural_only')
    tasks = read(OLD/'protocol.json')['tasks']
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(created_utc=stamp(), default=DEFAULT, grids=grids, arms=list(arms.values()),
        tasks=tasks, source_cache_root=str(OLD), source_protocol_sha256=sha(OLD/'protocol.json'),
        source_sha256=sha(__file__), seed=42, trainable_weights_changed=False,
        scope='Exploratory repeated-test sensitivity; all fixed grid arms reported. No default or checkpoint selection.',
        enzyme_map_scope='1521 test queries; 261907 accession candidates; 222985 unique protein vectors; also training-ID-excluded library.',
        scoring='ReactZyme: original FP32 weighted concatenation then FP64 dot rounded once to FP32. EnzymeMap: original FP32 dense + sparse sum, 64-query batches.',
        tie_policy='Original stable candidate-index ReactZyme ranks, plus expected reciprocal rank under uniformly random exact-score ties. Original NumPy argsort screening ranks.',
        coefficients='alpha dictionary mixture; cap times kappa=.2 residual; nu times native learned protein-fusion scale. nu=0 removes all fused contributions at inference, not learned-view-only training ablation.',
        uncertainty='Single trained seed, no training-variance confidence intervals; curves connect evaluated points without smoothing.')
    shutil.copy2(__file__, out/Path(__file__).name)
    write(out/'protocol.json', plan)
    log(initialized=str(out), arms_per_task=len(arms), total_evaluations=len(arms)*len(tasks))

def checked_cache(p):
    expected = read(p.with_suffix('.receipt.json'))['sha256']
    if sha(p) != expected: raise ValueError(f'Cache checksum changed: {p}')
    return torch.load(p, map_location='cpu', weights_only=False, mmap=True)

@torch.inference_mode()
def tie_diagnostics(scores, truth):
    """Expected all-positive MRR under uniform permutations within exact ties."""
    result={}; arrays={}
    nr, ne = scores.shape
    edges=torch.unique(torch.as_tensor(truth['reaction_index'],device=scores.device).long()*ne+
                       torch.as_tensor(truth['enzyme_index'],device=scores.device).long())
    r,e=edges//ne,edges%ne
    for direction, matrix, anchors, candidates in [('reaction_to_enzyme',scores,r,e),('enzyme_to_reaction',scores.T,e,r)]:
        harmonic=torch.cat([torch.zeros(1,device=scores.device,dtype=torch.float64),
                            torch.arange(1,matrix.shape[1]+1,device=scores.device,dtype=torch.float64).reciprocal().cumsum(0)])
        expected=torch.empty(len(anchors),device=scores.device,dtype=torch.float64)
        ties=torch.empty(len(anchors),device=scores.device,dtype=torch.long)
        # Binary search each positive in ascending score rows; avoids pairwise
        # comparisons and remains invariant to candidate-index permutations.
        for start in range(0,len(matrix),256):
            chosen=(anchors>=start)&(anchors<start+256)
            idx=chosen.nonzero().flatten()
            if not len(idx): continue
            idx=idx[torch.argsort(anchors[idx],stable=True)]
            rows=torch.sort(matrix[start:start+256],dim=1).values
            local=anchors[idx]-start
            val=matrix[anchors[idx],candidates[idx]]
            # One padded query list per row handles arbitrary positive counts.
            counts=torch.bincount(local,minlength=len(rows)); width=int(counts.max())
            slots=torch.arange(len(idx),device=scores.device)-torch.repeat_interleave(counts.cumsum(0)-counts,counts)
            vals=torch.zeros((len(rows),width),device=scores.device,dtype=scores.dtype)
            vals[local,slots]=val
            lo=torch.searchsorted(rows,vals,side='left')[local,slots]
            hi=torch.searchsorted(rows,vals,side='right')[local,slots]
            before=matrix.shape[1]-hi; equal=hi-lo
            expected[idx]=(harmonic[before+equal]-harmonic[before])/equal
            ties[idx]=equal
        sums=torch.zeros(len(matrix),device=scores.device,dtype=torch.float64)
        counts=torch.zeros_like(sums)
        sums.scatter_add_(0,anchors,expected);counts.scatter_add_(0,anchors,torch.ones_like(expected))
        q=(counts>0).nonzero().flatten(); means=sums[q]/counts[q]
        result[direction]=dict(expected_tie_mrr=float(means.mean()),positive_edges_with_ties=int((ties>1).sum()),
                               positive_edges=len(ties),max_tie_size=int(ties.max()))
        arrays[direction+'_tie_expected_mrr']=means.cpu().numpy()
        arrays[direction+'_tie_query_index']=q.cpu().numpy()
    return result,arrays

@torch.inference_mode()
def export_components(task, payload, lib, dest, device):
    path=dest/'fusion_components.pt'
    if path.exists(): return checked_cache(path)
    m=task['model']; name=task['name']
    model,config=model_from_checkpoint(Path(m.get('test_config',m['config'])),Path(m['checkpoint']),device)
    if name=='enzymemap':
        keys=lib['protein_ids']; residue_path=CROSS/'clipzyme_f3_catalog_v1/features/proteins_prott5_residue.local.h5'
        receipt=read(residue_path.with_suffix('.receipt.json')); st=residue_path.stat()
        if (receipt['purpose']!='screening' or receipt['protein_count']!=len(keys) or
            st.st_size!=receipt['output_size_bytes'] or st.st_mtime_ns!=receipt['output_mtime_ns'] or
            sha(residue_path.with_name('proteins_prott5_residue.h5'))!=receipt['source_sha256']):
            raise ValueError('Screening residue cache provenance failed')
    else:
        old=read(CROSS/'shared_semantic_strength_variants_v1/protocol.json')
        t=next(x['task'] for x in old['reactzyme'] if x['split']==name)
        keys=read(Path(t['fixed_test_features'])/'catalog.json')['proteins']
        residue_path=Path(config.data.protein_residue_embeds_path)
    enc=model.model.multiview_encoder; captured={'global':[],'fused':[]}
    def hook(key):
        def f(_module,_args,output): captured[key].append(F.normalize(output.float(),dim=-1,eps=1e-6).cpu())
        return f
    hooks=[enc.global_output.register_forward_hook(hook('global')),enc.fused_output.register_forward_hook(hook('fused'))]
    residue=StoragePrecisionResidues(str(residue_path),max_tokens=config.data.max_protein_tokens,truncation=config.data.protein_truncation)
    log(task=name,stage='export_fusion_components',proteins=len(keys),residues=str(residue_path))
    try:
        reference=encode_residue_targets(model,residue,keys,device,128,False,progress_every_batches=25).float()
    finally:
        for h in hooks: h.remove()
        residue.close()
    g=torch.cat(captured['global']);f=torch.cat(captured['fused']);scale=float(enc.residual_scale)
    reconstructed=F.normalize(g+scale*f,dim=-1,eps=1e-6)
    native_scale=scale/3 if m['checkpoint_already_calibrated'] else scale
    default=F.normalize(g+3*native_scale*f,dim=-1,eps=1e-6)
    original=lib['base_e'] if lib is not None else payload['base_e']
    error=float((reconstructed-reference).abs().max());default_error=float((default-original).abs().max())
    if error>1e-6 or default_error>3e-6: raise ValueError(f'Fusion component parity failed {error}, {default_error}')
    obj=dict(global_e=g,fused_e=f,native_scale=native_scale,keys=keys,
             checkpoint_sha256=m['checkpoint_sha256'],reconstruction_error=error,default_cache_error=default_error,
             residue_path=str(residue_path),receipt=receipt if name=='enzymemap' else None)
    tmp=path.with_suffix('.partial.pt');torch.save(obj,tmp);tmp.replace(path)
    write(path.with_suffix('.receipt.json'),dict(sha256=sha(path),created_utc=stamp(),default_cache_error=default_error,
          native_scale=native_scale,checkpoint_sha256=m['checkpoint_sha256']))
    log(task=name,stage='fusion_components_complete',native_scale=native_scale,parity_error=default_error)
    del model,reference,reconstructed,default;torch.cuda.empty_cache()
    return obj

@torch.inference_mode()
def run(args):
    out=args.output; plan=read(out/'protocol.json'); task=next(t for t in plan['tasks'] if t['name']==args.task)
    if sha(__file__)!=plan['source_sha256']: raise ValueError('Run the frozen protocol source; do not change a live experiment')
    dest=out/args.task;dest.mkdir(exist_ok=True);m=task['model'];source=Path(task['source_phase2']);device=args.device
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    for path,key in [(m['checkpoint'],'checkpoint_sha256'),(m['phase2_checkpoint'],'phase2_checkpoint_sha256'),
                     (source/'features/manifest.json','feature_manifest_sha256'),(source/'anchors.pt','dictionary_sha256')]:
        if sha(path)!=m[key]:raise ValueError(f'Model lineage mismatch: {path}')
    payload=checked_cache(OLD/args.task/'test_cache.pt')
    for k in ('dictionary_sha256','feature_manifest_sha256'):
        if payload[k]!=m[k]:raise ValueError(f'Feature lineage mismatch: {k}')
    lib=checked_cache(OLD/args.task/'screening_library.pt') if args.task=='enzymemap' else None
    proteins=lib if lib is not None else payload
    if proteins['checkpoint_sha256']!=m['checkpoint_sha256']: raise ValueError('Base checkpoint mismatch')
    head=load_head(Path(m['phase2_checkpoint']),m['feature_manifest_sha256'],device)
    be=proteins['base_e'].to(device);se=proteins['semantic_e'].to(device)
    br=payload['base_r'].to(device);sr=payload['semantic_r'].to(device)
    write(dest/'input_receipt.json',dict(model=m,cache_receipt=read((OLD/args.task/'test_cache.pt').with_suffix('.receipt.json')),
                                       protocol_sha256=sha(out/'protocol.json'),started_utc=stamp(),residual_scale=float(head.scale)))
    if lib is not None:
        protocol=CROSS/'clipzyme_screening_evaluation_protocol_v2';pr=read(protocol/'receipt.json')
        for fn,k in [('queries.csv','queries_sha256'),('train_uniprot_ids.txt','train_uniprot_ids_sha256')]:
            if sha(protocol/fn)!=pr[k]:raise ValueError('Screening evaluation protocol changed')
        rows=list(csv.DictReader((protocol/'queries.csv').open()))
        if payload['ids']!=[r['reaction_id'] for r in rows] or len(lib['candidate_ids'])!=261907:raise ValueError('Screening axes changed')
        index={k:i for i,k in enumerate(lib['candidate_ids'])}; train=set((protocol/'train_uniprot_ids.txt').read_text().splitlines())
        kept=np.array([i for i,k in enumerate(lib['candidate_ids']) if k not in train],dtype=np.int32)
        positives=[np.array([index[k] for k in json.loads(r['positive_uniprot_ids_json'])],dtype=np.int64) for r in rows]
    components=None
    # Current-recipe reproduction always precedes new coefficient evaluations.
    arms=plan['arms']
    if args.default_only: arms=[r for r in arms if 'default' in r['groups']]
    for arm in arms:
        result_path=dest/(arm['id']+'.json')
        if result_path.exists(): continue
        started=time.monotonic(); recipe=dict(alpha=arm['alpha'],cap=arm['cap']);a=arm['alpha'];c=arm['cap']
        if arm['nu']==3.: b=be
        else:
            if components is None: components=export_components(task,payload,lib,dest,device)
            b=F.normalize(components['global_e'].to(device)+arm['nu']*components['native_scale']*components['fused_e'].to(device),dim=-1,eps=1e-6)
        log(task=args.task,stage='evaluate',arm=arm['id'])
        if lib is None:
            scores=canonical_dot(compose(br,head,sr,recipe,'reaction'),compose(b,head,se,recipe,'enzyme'))
            evaluation=evaluate_scores(scores,payload['truth'])
            summary=evaluation['summary']; arrays={f'{d}_{k}':v for d,x in evaluation['per_query'].items() for k,v in x['all'].items()}
            ties,tie_arrays=tie_diagnostics(scores,payload['truth']);arrays.update(tie_arrays)
            np.savez_compressed(dest/(arm['id']+'_per_query.npz'),**arrays)
            result=dict(summary=summary,tie_diagnostics=ties)
            if 'default' in arm['groups']:
                prior=np.load(Path(m['benchmark_result']).with_name('scores.npy'),mmap_mode='r')
                error=float(np.max(np.abs(scores.cpu().numpy()-prior)))
                result['default_max_score_error']=error
                if error>1e-6:raise ValueError(f'Default score parity failed {error}')
            del scores,evaluation
        else:
            de=torch.cat([F.normalize(x+c*head.scale*head.enzyme(x),dim=-1) for x in b.split(8192)])
            dr=F.normalize(br+c*head.scale*head.reaction(br),dim=-1)
            records=[]
            with ThreadPoolExecutor(max_workers=12) as pool:
                for start in range(0,len(br),64):
                    values=(1-a)*(dr[start:start+64]@de.T)
                    if a:values+=a*torch.sparse.mm(se,sr[start:start+64].T.contiguous()).T
                    values=values.cpu().numpy()[:,lib['expanded']]
                    arguments=[(start+i,payload['ids'][start+i],v,positives[start+i],kept) for i,v in enumerate(values)]
                    records.extend(pool.map(evaluate_query,arguments))
            summary={}
            for table,n in [('table1',len(index)),('table2',len(kept))]:
                selected=[r[table] for r in records if r[table] is not None]
                summary[table]=dict(queries=len(selected),candidate_ids=n,**{k:float(np.mean([r[k] for r in selected])) for k in METRICS})
            if (summary['table1']['queries']!=1521 or summary['table2']['queries']!=1337 or len(kept)!=252113 or
                sum(r['positives_table1'] for r in records)!=pr['table1_candidate_positive_labels']):raise ValueError('Screening denominator changed')
            (dest/(arm['id']+'_per_query.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
            result=dict(summary=summary)
            del de,dr,values
        if 'default' in arm['groups']:
            prior=read(m['benchmark_result'])['summary']
            if lib is None: errors=[abs(summary[d]['all']['reactzyme_mrr']-prior[d]['all']['reactzyme_mrr']) for d in prior]
            else:errors=[abs(summary[t][k]-prior[t][k]) for t in ('table1','table2') for k in METRICS]
            result['default_max_metric_error']=max(errors)
            if max(errors)>1e-6:raise ValueError(f'Default metric parity failed {errors}')
        result.update(arm=arm,task=args.task,elapsed_seconds=time.monotonic()-started,completed_utc=stamp(),protocol_sha256=sha(out/'protocol.json'))
        write(result_path,result)
        log(task=args.task,stage='complete_arm',arm=arm['id'],seconds=result['elapsed_seconds'],summary=summary)
    write(dest/('default_verified.json' if args.default_only else 'complete.json'),dict(completed_utc=stamp(),arms=len(arms),protocol_sha256=sha(out/'protocol.json')))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--initialize',action='store_true');p.add_argument('--task',choices=['reaction_smi','enzyme_smi','time','enzymemap'])
    p.add_argument('--device',default='cuda:0');p.add_argument('--default-only',action='store_true');args=p.parse_args()
    if args.initialize:initialize(args.output.resolve())
    else:
        if not args.task:p.error('--task is required')
        with export_device_lock(args.device):run(args)

if __name__=='__main__':main()
