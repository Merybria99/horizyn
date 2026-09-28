#!/usr/bin/env python3
"""Quantitative analyses 1--3, family bootstrap and per-query audit tables."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
import torch

from cersei_embedding_organization import (OUT, SPLITS, SEED, read, dump, sha, prefix, norm)


def labels_matrix(labels):
    classes=sorted({v for row in labels for v in row}); index={v:i for i,v in enumerate(classes)}
    rr,cc=zip(*[(i,index[v]) for i,row in enumerate(labels) for v in set(row)]) if classes else ([],[])
    return sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(labels),len(classes))),classes


def summarize(values, labels, families, replicates=1000):
    """Exact macro-over-class statistic in every whole-family bootstrap draw.

    Multilabel queries contribute to each of their classes. Empty classes in a
    bootstrap replicate are omitted, not filled with zero. Bank stays fixed.
    """
    values=np.asarray(values,dtype=float); valid=np.isfinite(values)
    annotated=np.asarray([bool(x) for x in labels]); keep=valid & annotated
    result=dict(total_queries=len(values),finite_queries=int(valid.sum()),annotated_queries=int(annotated.sum()),
                evaluated_queries=int(keep.sum()),micro_mean=float(values[valid].mean()) if valid.any() else None)
    if not keep.any():return dict(result,macro=None,ci95=None,classes=0,families=0)
    y=values[keep]; labs=[labels[i] for i in np.flatnonzero(keep)]
    membership,classes=labels_matrix(labs)
    family=np.asarray(families)[keep]; unique,fi=np.unique(family,return_inverse=True)
    nf=len(unique); nclass=len(classes)
    row,col=membership.nonzero()
    totals=sparse.csr_matrix((y[row],(fi[row],col)),shape=(nf,nclass))
    counts=sparse.csr_matrix((np.ones(len(row)),(fi[row],col)),shape=(nf,nclass))
    mean=np.asarray(totals.sum(0)).ravel()/np.asarray(counts.sum(0)).ravel()
    rng=np.random.default_rng(SEED)
    samples=rng.multinomial(nf,np.full(nf,1/nf),size=replicates).astype(np.float32)
    boot_sum=np.asarray(totals.T @ samples.T).T; boot_count=np.asarray(counts.T @ samples.T).T
    boot_mean=np.divide(boot_sum,boot_count,out=np.full_like(boot_sum,np.nan),where=boot_count>0)
    stats=np.nanmean(boot_mean,axis=1)
    return dict(result,macro=float(mean.mean()),ci95=np.quantile(stats,[.025,.975]).tolist() if nf>1 else None,
                classes=nclass,families=nf,largest_family=int(np.bincount(fi).max()),
                per_class={c:float(v) for c,v in zip(classes,mean)})


def write_table(path, records):
    if not records:return
    fields=list(dict.fromkeys(k for row in records for k in row))
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(records)


def families_for(meta, indices=None):
    fam=read(OUT/'homology/families.json')
    ids=meta['protein_ids'] if indices is None else [meta['protein_ids'][i] for i in indices]
    return np.asarray([fam[p][0] for p in ids]),np.asarray([fam[p][1] for p in ids])


def nearest(embeddings, permitted=None, family50=None, device='cpu', k=50, queries=None):
    x=torch.as_tensor(norm(embeddings),device=device)
    n=len(x); permitted=np.ones(n,bool) if permitted is None else np.asarray(permitted)
    result=np.full((n,k),-1,np.int64)
    queries=np.arange(n) if queries is None else np.asarray(queries)
    for start in range(0,len(queries),256):
        qq=queries[start:start+256]; sim=x[qq] @ x.T
        sim[:,torch.as_tensor(~permitted,device=device)]=-torch.inf
        sim[torch.arange(len(qq),device=device),torch.as_tensor(qq,device=device)]=-torch.inf
        if family50 is not None:
            equal=torch.as_tensor(family50[qq,None]==family50[None,:],device=device)
            sim[equal]=-torch.inf
        val,idx=torch.topk(sim,k=min(k,n),dim=1,sorted=True)
        ii=idx.cpu().numpy();ii[~np.isfinite(val.cpu().numpy())]=-1
        result[qq,:ii.shape[1]]=ii
    return result


def neighbor_agreement(neighbors, labels, k):
    sets=[set(x) for x in labels]; y=np.full(len(labels),np.nan)
    for i,lab in enumerate(sets):
        nn=neighbors[i,:k]
        if lab and np.all(nn>=0):y[i]=np.mean([bool(lab & sets[j]) for j in nn])
    return y


def analysis1(split,device):
    d=OUT/split; dest=d/'analysis1';dest.mkdir(exist_ok=True)
    meta=read(d/'metadata.json'); z=np.load(d/'embeddings.npz'); ids=np.asarray(meta['neighborhood_indices'])
    assert np.array_equal(z['prott5_indices'],ids)
    stages=dict(prott5=z['prott5'],phase1=z['phase1_e'][ids],refined=z['refined_e'][ids])
    if 'native_e' in z: stages['native']=z['native_e'][ids]
    fam,fam50=families_for(meta,ids); ecs=[meta['enzyme_ec'][i] for i in ids]
    subset=read(d/'subsets.json');lookup={int(g):i for i,g in enumerate(ids)}
    selected=np.array([lookup[g] for g in subset['protein_indices']])
    records=[]; summaries=[]; all_y={}
    for level in (1,2,3,4):
        labels=[prefix(x,level) for x in ecs]; annotated=np.array([bool(x) for x in labels])
        for exclude in (False,True):
            for stage,x in stages.items():
                nn=nearest(x,annotated,fam50 if exclude else None,device,queries=selected)
                for k in (10,50):
                    y=neighbor_agreement(nn,labels,k); all_y[(level,exclude,k,stage)]=y
                    row=dict(split=split,stage=stage,ec_level=level,k=k,exclude_homologs=exclude,
                             **summarize(y[selected],[labels[i] for i in selected],fam[selected]));summaries.append(row)
                    for i in np.flatnonzero(np.isfinite(y)):
                        records.append(dict(protein_id=meta['protein_ids'][ids[i]],family=int(fam[i]),stage=stage,
                            ec_level=level,k=k,exclude_homologs=exclude,agreement=float(y[i])))
                np.save(dest/f'neighbors_ec{level}_{stage}_exclude{int(exclude)}.npy',nn)
            for k in (10,50):
                for stage in stages:
                    if stage=='prott5':continue
                    delta=all_y[(level,exclude,k,stage)]-all_y[(level,exclude,k,'prott5')]
                    summaries.append(dict(split=split,stage=stage+'-prott5',ec_level=level,k=k,
                        exclude_homologs=exclude,**summarize(delta[selected],[labels[i] for i in selected],fam[selected])))
            print('NEIGHBORS',split,'EC',level,'exclude',exclude,flush=True)
    write_table(dest/'per_protein.csv',records);dump(dest/'summary.json',summaries)
    return summaries


def query_metrics(scores,positive,random_idx=None,coarse_idx=None,chemical_idx=None):
    """Known-positive retrieval coverage uses uniform random resolution of exact ties.

    Unannotated comparisons are never treated as confirmed inactivity. Near-tie
    coverage bounds use a fixed 1e-6 band on the same FP32 score matrix.
    """
    scores=np.asarray(scores);positive=np.asarray(positive,np.int64)
    pos=scores[positive];unknown=np.ones(len(scores),bool);unknown[positive]=False
    result=dict(degree=len(positive),positive_cosine=float(pos.mean()),
        hardest_margin=float(pos.mean()-scores[unknown].max()) if unknown.any() else np.nan)
    for name,indices in [('random',random_idx),('coarse_ec',coarse_idx),('chemical',chemical_idx)]:
        idx=np.asarray(indices if indices is not None else [],np.int64)
        idx=idx[unknown[idx]]
        result[name+'_margin']=float(pos.mean()-(scores[idx].mean() if name=='random' else scores[idx].max())) if len(idx) else np.nan
    for k in (10,50):
        kk=min(k,len(scores));threshold=np.partition(scores,len(scores)-kk)[len(scores)-kk]
        gt=int((scores>threshold).sum());eq=int((scores==threshold).sum())
        result[f'coverage{k}']=float(((pos>threshold)+(pos==threshold)*((kk-gt)/eq)).mean())
        result[f'coverage{k}_lower_1e6']=float(np.mean(pos>threshold+1e-6))
        result[f'coverage{k}_upper_1e6']=float(np.mean(pos>=threshold-1e-6))
        result[f'all_positive{k}_certain']=float(np.all(pos>threshold) or (np.all(pos>=threshold) and gt+eq<=kk))
    return result


def direction_context(meta,truth,direction):
    if direction=='R2E':
        ids=meta['reaction_ids'];candidate_labels=meta['candidate_ec']
        labels=meta['reaction_rules'] if meta['reaction_rules'] and any(meta['reaction_rules']) else [prefix(x,3) for x in meta['reaction_ec']]
        family=np.array(meta['reaction_families']);edge=truth
        query_ec=meta['reaction_ec']
    else:
        ids=meta['candidate_ids'];candidate_labels=meta['reaction_ec'];edge=truth[:,::-1]
        labels=[prefix(x,3) for x in meta['candidate_ec']];query_ec=meta['candidate_ec']
        fam=read(OUT/'homology/families.json');expanded=np.load(OUT/meta['_split']/'expanded.npy')
        family=np.array([fam.get(meta['protein_ids'][i],[-1,-1])[0] for i in expanded])
    positive=defaultdict(list)
    for a,b in edge:positive[int(a)].append(int(b))
    coarse=defaultdict(set)
    for j,ecs in enumerate(candidate_labels):
        for ec in prefix(ecs,1):coarse[ec].add(j)
    return ids,labels,family,positive,coarse,query_ec


def alignment_summary(records,meta,truth,direction):
    ids,labels,fam,positive,_,_=direction_context(meta,truth,direction)
    metrics=[k for k in records[0] if k not in ('query_index','query_id','stage','direction','degree')]
    result=[]; stages=sorted({r['stage'] for r in records});by_stage={}
    for stage in stages:
        rr=[r for r in records if r['stage']==stage]; by_stage[stage]=rr
        qi=np.asarray([r['query_index'] for r in rr]);degree=np.asarray([r['degree'] for r in rr])
        for group,mask in [('all',np.ones(len(rr),bool)),('1',degree==1),('2-5',(degree>=2)&(degree<=5)),('6+',degree>=6)]:
            if not mask.any():continue
            for metric in metrics:
                val=np.asarray([r[metric] for r in rr])[mask];ii=qi[mask]
                result.append(dict(stage=stage,direction=direction,degree_group=group,metric=metric,
                    **summarize(val,[labels[i] for i in ii],fam[ii])))
    if 'phase1' in by_stage and 'refined' in by_stage:
        a,b=by_stage['phase1'],by_stage['refined'];assert [r['query_index'] for r in a]==[r['query_index'] for r in b]
        ii=np.array([r['query_index'] for r in a])
        for metric in metrics:
            y=np.array([rb[metric]-ra[metric] for ra,rb in zip(a,b)])
            result.append(dict(stage='refined-phase1',direction=direction,degree_group='all',metric=metric,
                **summarize(y,[labels[i] for i in ii],fam[ii])))
    return result


def analysis2(split,device):
    d=OUT/split; dest=d/'analysis2';dest.mkdir(exist_ok=True)
    meta=read(d/'metadata.json');meta['_split']=split
    z=np.load(d/'embeddings.npz');truth=np.load(d/'truth.npy');expanded=np.load(d/'expanded.npy')
    chem=np.load(d/'chemistry.npz')['tanimoto']; stages={'phase1':('phase1_r','phase1_e'),'refined':('refined_r','refined_e')}
    if 'native_e' in z:stages['native']=('phase1_r','native_e')
    all_records=[];summaries=[]
    for direction in ('R2E','E2R'):
        ids,labels,fam,pos,coarse,query_ec=direction_context(meta,truth,direction)
        selected=read(d/'subsets.json')['reaction_indices' if direction=='R2E' else 'enzyme_query_indices']
        queries=sorted(set(selected)&set(pos)); assert np.all(fam[queries]>=0)
        nc=len(meta['candidate_ids']) if direction=='R2E' else len(meta['reaction_ids'])
        fixed={}
        for q in queries:
            rng=np.random.default_rng(SEED+q);unknown=np.setdiff1d(np.arange(nc),pos[q],assume_unique=True)
            random=rng.choice(unknown,min(64,len(unknown)),replace=False)
            hard=sorted(set.union(set(),*(coarse[v] for v in prefix(query_ec[q],1))))
            chemical=np.flatnonzero(chem[pos[q]].max(0)>=.5) if direction=='E2R' else None
            fixed[q]=(random,hard,chemical)
        records=[]
        for stage in [*stages,'final_score']:
            if stage=='final_score':scoremap=np.load(meta['final_scores'],mmap_mode='r');assert scoremap.shape==(len(meta['reaction_ids']),len(meta['candidate_ids']))
            else:
                rk,ek=stages[stage];r=torch.as_tensor(norm(z[rk]),device=device);e=torch.as_tensor(norm(z[ek]),device=device)
            for start in range(0,len(queries),16):
                qq=queries[start:start+16]
                if stage=='final_score':values=np.asarray(scoremap[qq] if direction=='R2E' else scoremap[:,qq].T)
                elif direction=='R2E':values=(r[qq] @ e.T).cpu().numpy()[:,expanded]
                else:values=(e[expanded[qq]] @ r.T).cpu().numpy()
                for q,scores in zip(qq,values):
                    result=query_metrics(scores,pos[q],*fixed[q])
                    records.append(dict(query_index=q,query_id=ids[q],stage=stage,direction=direction,**result))
            print('ALIGNMENT',split,direction,stage,'queries',len(queries),flush=True)
        write_table(dest/f'per_query_{direction}.csv',records)
        summaries.extend(alignment_summary(records,meta,truth,direction));all_records.extend(records)
    dump(dest/'summary.json',summaries)
    dump(dest/'coverage.json',dict(known_edges=len(truth),reactions_with_positive=len(set(truth[:,0])),
        enzymes_with_positive=len(set(truth[:,1])),candidate_count=len(meta['candidate_ids']),
        annotation_coverage=read(d/'prepared.json'),score_source=meta['final_scores'],score_sha256=sha(meta['final_scores']),
        near_tie_band=1e-6,final_score_geometry='Dictionary-augmented score, not a 512-dimensional neural embedding'))
    return summaries


def analysis3(device):
    d=OUT/'enzymemap';dest=d/'analysis3';dest.mkdir(exist_ok=True)
    meta=read(d/'metadata.json');z=np.load(d/'embeddings.npz');ch=np.load(d/'chemistry.npz')
    chem=ch['tanimoto'];labels=meta['reaction_rules'];family=np.array(meta['reaction_families']);n=len(labels)
    selected=np.array(read(d/'subsets.json')['reaction_indices']);sample_labels=[labels[i] for i in selected]
    membership,classes=labels_matrix(labels);same=(membership @ membership.T).toarray()>0
    valid=np.asarray([bool(x) for x in labels]) & ch['valid'];eye=np.eye(n,dtype=bool)
    pair_valid=valid[:,None]&valid[None,:]&~eye
    low=same & (chem<=.3) & pair_valid;high=(~same)&(chem>=.7)&pair_valid
    stages={'participant_fingerprint':chem,'ReactionT5v2':norm(z['raw_reaction']) @ norm(z['raw_reaction']).T,
            'phase1':norm(z['phase1_r']) @ norm(z['phase1_r']).T,'refined':norm(z['refined_r']) @ norm(z['refined_r']).T}
    summaries=[];records=[];deltas={}
    for stage,sim in stages.items():
        for exclude in (False,True):
            work=sim.copy();work[~pair_valid]=-np.inf
            if exclude:work[chem>=.5]=-np.inf
            order=np.argsort(-work,axis=1,kind='stable')[:,:50]
            order[~np.isfinite(np.take_along_axis(work,order,axis=1))]=-1
            for k in (10,50):
                y=neighbor_agreement(order,labels,k)
                summaries.append(dict(stage=stage,metric=f'rule_agreement{k}',exclude_similar_participants=exclude,**summarize(y[selected],sample_labels,family[selected])))
        y=np.full(n,np.nan)
        for i in selected:
            if low[i].any() and high[i].any():y[i]=sim[i,low[i]].mean()-sim[i,high[i]].mean()
        summaries.append(dict(stage=stage,metric='dissimilar_same_rule_minus_similar_different_rule',**summarize(y[selected],sample_labels,family[selected])))
        for b in range(10):
            mask=pair_valid & (chem>=b/10) & (chem<(b+1)/10 if b<9 else chem<=1)
            val=np.full(n,np.nan)
            for i in selected:
                yes=mask[i]&same[i];no=mask[i]&~same[i]
                if yes.any() and no.any():val[i]=sim[i,yes].mean()-sim[i,no].mean()
                if yes.any() or no.any():
                    records.append(dict(query_id=meta['reaction_ids'][i],stage=stage,bin=b,
                        same_rule_pairs=int(yes.sum()),different_rule_pairs=int(no.sum()),
                        same_mean=float(sim[i,yes].mean()) if yes.any() else '',
                        different_mean=float(sim[i,no].mean()) if no.any() else '',
                        delta=float(val[i]) if np.isfinite(val[i]) else ''))
            deltas[(stage,b)]=val
            summaries.append(dict(stage=stage,metric='matched_similarity_delta',bin_low=b/10,bin_high=(b+1)/10,
                **summarize(val[selected],sample_labels,family[selected])))
        print('REACTION ORGANIZATION',stage,flush=True)
    for stage in ('phase1','refined'):
        for b in range(10):
            summaries.append(dict(stage=stage+'-ReactionT5v2',metric='matched_similarity_delta',bin_low=b/10,bin_high=(b+1)/10,
                **summarize((deltas[(stage,b)]-deltas[('ReactionT5v2',b)])[selected],sample_labels,family[selected])))
    dump(dest/'summary.json',summaries);write_table(dest/'per_query_similarity_bins.csv',records)
    dump(dest/'coverage.json',dict(reactions=n,query_subset=len(selected),annotated_valid=int(valid.sum()),rule_classes=len(classes),
        reaction_families=len(set(family)),same_rule_low_similarity_pairs=int(np.triu(low,1).sum()),
        different_rule_high_similarity_pairs=int(np.triu(high,1).sum()),
        queries_with_both_contrast_types=int(((low.sum(1)>0)&(high.sum(1)>0))[selected].sum()),
        units='Directed query-to-candidate comparisons, averaged within query before class balancing. No independent-pair confidence intervals.'))


def sanity():
    # Unequal class sizes must not turn the requested macro into a micro mean.
    r=summarize([1,1,1,0],[['a'],['a'],['a'],['b']],[0,0,0,1],100)
    assert r['macro']==.5 and r['micro_mean']==.75 and r['families']==2
    # Multi-positive ties: each of 100 tied candidates has exactly 10/100
    # probability of appearing in top 10, regardless of candidate ordering.
    q=query_metrics(np.ones(100),[0,99],[1,2],[3,4])
    assert np.isclose(q['coverage10'],.1) and q['hardest_margin']==0 and q['all_positive50_certain']==0
    q=query_metrics(np.arange(100),[99,98],[1,2],[3,4])
    assert q['coverage10']==1 and q['all_positive50_certain']==1
    a=np.eye(3,dtype=np.float32);nn=nearest(a,device='cpu',k=2)
    assert all(i not in nn[i] for i in range(3))
    dump(OUT/'metric_sanity.json',dict(passed=True,checks=['class balance','family count','multi-positive ties','perfect top-k','self exclusion'],code_sha256=sha(__file__)))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['sanity','analysis1','analysis2','analysis3']);p.add_argument('--split',choices=SPLITS);p.add_argument('--device',default='cpu');a=p.parse_args()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    if a.action=='sanity':sanity()
    elif a.action=='analysis1':analysis1(a.split,a.device)
    elif a.action=='analysis2':analysis2(a.split,a.device)
    elif a.action=='analysis3':analysis3(a.device)


if __name__=='__main__':main()
