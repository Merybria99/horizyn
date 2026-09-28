"""Fixed-checkpoint geometry diagnostics, shared IDs, paired family bootstrap."""
import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import numpy as np
import torch
from cersei_embedding_organization import ROOT,OUT as OLD,read,dump,sha,norm,prefix,SEED
from cersei_embedding_metrics import nearest,neighbor_agreement,summarize,write_table,query_metrics,direction_context,labels_matrix
from cersei_public_embedding_stats import summarize
OUT=ROOT/'runs/public_embedding_comparison_20260924'
CURRENT=ROOT/'runs/cersei_horizyn_challenge_20260923/no_dictionary_validation_20260923/reference_shared_test'

def prepare_map():
    dest=OUT/'enzymemap';dest.mkdir(exist_ok=True)
    meta=read(OLD/'enzymemap/metadata.json');bank=np.array(meta['neighborhood_indices'])
    source=ROOT/'runs/enzymemap_public_horizyn_20260923_seed42';cat=read(source/'catalog.json')
    assert cat['protein_ids']==meta['protein_ids'] and cat['test_ids']==meta['reaction_ids'] and cat['candidate_ids']==meta['candidate_ids']
    z=np.load(source/'test/embeddings.npz');r=z['reactions'];e=z['unique_proteins'][bank]
    expanded=np.load(OLD/'enzymemap/expanded.npy');representative=np.array([np.flatnonzero(expanded==i)[0] for i in bank])
    scores=np.load(source/'test/scores.npy',mmap_mode='r');err=float(np.max(np.abs(r[::100]@e.T-scores[::100][:,representative])));assert err<3e-5
    np.savez(dest/'horizyn_embeddings.npz',enzyme=e,reaction=r,bank=bank)
    dump(dest/'horizyn_receipt.json',dict(selection=read(source/'selection.json'),score_receipt=read(source/'test/score_receipt.json'),
        parity_max_error=err,embeddings_sha256=sha(dest/'horizyn_embeddings.npz'),scores=str(source/'test/scores.npy')))
    for model in ['cersei','clipzyme']:
        src=ROOT/f'runs/clipzyme_embedding_comparison_20260924/{model}_embeddings.npz'
        shutil.copy2(src,dest/f'{model}_embeddings.npz')

def neighborhood(split,device,control=None):
    dest=OUT/split;meta=read(OLD/split/'metadata.json');sub=read(OLD/split/'subsets.json')
    bank=np.array(meta['neighborhood_indices']);qglobal=sub['protein_indices']
    models=['cersei','horizyn']+(['clipzyme'] if split=='enzymemap' else ['creep'])
    stages={m:np.load(dest/f'{m}_embeddings.npz')['enzyme'] for m in models}
    zz=np.load(OLD/split/'embeddings.npz');assert np.array_equal(zz['prott5_indices'],bank);stages['prott5']=zz['prott5']
    overlaps=np.zeros(len(bank),bool)
    if control:
        ctrl=np.load(dest/f'{control}_embeddings.npz');newbank=ctrl['bank'];idx=np.searchsorted(bank,newbank);assert np.array_equal(bank[idx],newbank)
        stages={m:z[idx] for m,z in stages.items()};bank=newbank
        stages[control]=ctrl['enzyme'];stages['esm1b' if control=='clean' else 'esm_pocket']=ctrl['esm1b' if control=='clean' else 'esm_pocket']
        overlaps=ctrl['training_sequence_overlap'];qglobal=bank.tolist()
    if len(bank)==0:
        dump(dest/f'{control}_neighborhoods_summary.json',[])
        dump(dest/f'{control}_coverage_status.json',dict(supported=False,reason='No exact-sequence pocket-feature coverage',bank_size=0))
        return
    where={v:i for i,v in enumerate(bank)};q=np.array([where[i] for i in qglobal if i in where],dtype=int)
    families=read(OLD/'homology/families.json');f=np.array([families[meta['protein_ids'][i]][0] for i in bank]);f50=np.array([families[meta['protein_ids'][i]][1] for i in bank])
    summaries=[];records=[]
    for unseen in ([False,True] if control else [False]):
        for level in [1,2,3,4]:
            labs=[prefix(meta['enzyme_ec'][i],level) for i in bank];valid=np.array([bool(x) for x in labs])&(~overlaps if unseen else True)
            qq=q[valid[q]]
            for exclude in [False,True]:
                yy={}
                for model,z in stages.items():
                    nn=nearest(z,valid,f50 if exclude else None,device,k=50,queries=qq)
                    for k in [10,50]:
                        vals=neighbor_agreement(nn,labs,k)[qq];yy[(model,k)]=vals
                        attrs=dict(model=model,ec_level=level,k=k,exclude_homologs=exclude,exclude_training_overlap=unseen,bank_size=len(bank),eligible_bank=int(valid.sum()))
                        summaries.append(dict(**attrs,**summarize(vals,[labs[i] for i in qq],f[qq])))
                        records.extend(dict(**attrs,protein_id=meta['protein_ids'][bank[i]],value=float(v)) for i,v in zip(qq,vals) if np.isfinite(v))
                for model in stages:
                    if model=='cersei':continue
                    for k in [10,50]:
                        attrs=dict(model='cersei-'+model,ec_level=level,k=k,exclude_homologs=exclude,exclude_training_overlap=unseen,bank_size=len(bank),eligible_bank=int(valid.sum()))
                        summaries.append(dict(**attrs,**summarize(yy[('cersei',k)]-yy[(model,k)],[labs[i] for i in qq],f[qq])))
    if control=='clean':
        # CLEAN is trained in unnormalized Euclidean geometry, so audit that too.
        for unseen in [False,True]:
            labs=[prefix(meta['enzyme_ec'][i],3) for i in bank];valid=np.array([bool(x) for x in labs])&(~overlaps if unseen else True);qq=q[valid[q]]
            for model in ['clean','esm1b']:
                z=torch.as_tensor(stages[model],device=device);d=torch.cdist(z,z).cpu().numpy();d[:,~valid]=np.inf;d[f50[:,None]==f50[None,:]]=np.inf
                nn=np.argsort(d,axis=1,kind='stable')[:,:50];nn[~np.isfinite(np.take_along_axis(d,nn,axis=1))]=-1
                vals=neighbor_agreement(nn,labs,10)[qq]
                summaries.append(dict(model=model+'-native-l2',ec_level=3,k=10,exclude_homologs=True,exclude_training_overlap=unseen,
                    bank_size=len(bank),eligible_bank=int(valid.sum()),**summarize(vals,[labs[i] for i in qq],f[qq])))
    name='neighborhoods' if not control else control+'_neighborhoods'
    dump(dest/f'{name}_summary.json',summaries);write_table(dest/f'{name}_per_protein.csv',records)
    print(split,name,'COMPLETE',flush=True)

def alignment(split):
    dest=OUT/split;meta=read(OLD/split/'metadata.json');meta['_split']=split;sub=read(OLD/split/'subsets.json');truth=np.load(OLD/split/'truth.npy')
    chem=np.load(OLD/split/'chemistry.npz')['tanimoto']
    sources={'cersei':CURRENT/split/'scores.npy','horizyn':dest/'horizyn_scores.npy'}
    if split=='enzymemap':
        sources['horizyn']=ROOT/'runs/enzymemap_public_horizyn_20260923_seed42/test/scores.npy'
        sources['clipzyme']=ROOT/'runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_released_screen_v1/scores.npy'
    else:sources['creep']=dest/'creep_scores.npy'
    summaries=[]
    for direction in ['R2E','E2R']:
        ids,labs,fam,pos,coarse,query_ec=direction_context(meta,truth,direction)
        qq=sorted(set(sub['reaction_indices' if direction=='R2E' else 'enzyme_query_indices'])&set(pos));nc=len(meta['candidate_ids']) if direction=='R2E' else len(meta['reaction_ids']);fixed={}
        for q in qq:
            unknown=np.setdiff1d(np.arange(nc),pos[q]);rng=np.random.default_rng(SEED+q)
            fixed[q]=(rng.choice(unknown,min(64,len(unknown)),replace=False),sorted(set.union(set(),*(coarse[v] for v in prefix(query_ec[q],1)))),np.flatnonzero(chem[pos[q]].max(0)>=.5) if direction=='E2R' else None)
        records=[];yy={}
        for model,path in sources.items():
            s=np.load(path,mmap_mode='r');assert s.shape==(len(meta['reaction_ids']),len(meta['candidate_ids']))
            rr=[dict(query_index=q,query_id=ids[q],model=model,direction=direction,**query_metrics(s[q] if direction=='R2E' else s[:,q],pos[q],*fixed[q])) for q in qq]
            records.extend(rr);yy[model]=rr
            print(split,direction,model,'scored',len(qq),flush=True)
        for metric in [k for k in rr[0] if k not in ['query_index','query_id','model','direction','degree']]:
            values={m:np.array([r[metric] for r in rows]) for m,rows in yy.items()}
            for model in list(values):
                if model!='cersei':values['cersei-'+model]=values['cersei']-values[model]
            for model,v in values.items():summaries.append(dict(model=model,direction=direction,metric=metric,**summarize(v,[labs[i] for i in qq],fam[qq])))
        write_table(dest/f'alignment_{direction}_per_query.csv',records)
    dump(dest/'alignment_summary.json',summaries)

def reactions(split):
    dest=OUT/split;meta=read(OLD/split/'metadata.json');sub=read(OLD/split/'subsets.json');ch=np.load(OLD/split/'chemistry.npz');chem=ch['tanimoto']
    labels=meta['reaction_rules'] if split=='enzymemap' else [prefix(x,3) for x in meta['reaction_ec']]
    fam=np.array(meta['reaction_families']);q=np.array(sub['reaction_indices']);n=len(labels)
    models=['cersei','horizyn']+(['clipzyme'] if split=='enzymemap' else ['creep'])
    stages={m:norm(np.load(dest/f'{m}_embeddings.npz')['reaction']) for m in models}
    stages['reactiont5']=norm(np.load(OLD/split/'embeddings.npz')['raw_reaction'])
    sims={m:z@z.T for m,z in stages.items()};sims['participant_fingerprint']=chem
    valid=np.array([bool(x) for x in labels])&ch['valid'];mask=valid[:,None]&valid[None,:]&~np.eye(n,dtype=bool)
    summaries=[];records=[];yy={}
    for exclude in [False,True]:
        for model,sim in sims.items():
            work=sim.copy();work[~mask]=-np.inf
            if exclude:work[chem>=.5]=-np.inf
            order=np.argsort(-work,axis=1,kind='stable')[:,:50];order[~np.isfinite(np.take_along_axis(work,order,axis=1))]=-1
            for k in [10,50]:
                y=neighbor_agreement(order,labels,k)[q];yy[(model,exclude,k)]=y
                summaries.append(dict(model=model,k=k,exclude_similar_participants=exclude,label='native rule' if split=='enzymemap' else 'EC3 inherited from test associations',**summarize(y,[labels[i] for i in q],fam[q])))
                records.extend(dict(model=model,k=k,exclude_similar_participants=exclude,query_id=meta['reaction_ids'][i],value=float(v)) for i,v in zip(q,y) if np.isfinite(v))
        for model in sims:
            if model=='cersei':continue
            for k in [10,50]:summaries.append(dict(model='cersei-'+model,k=k,exclude_similar_participants=exclude,**summarize(yy[('cersei',exclude,k)]-yy[(model,exclude,k)],[labels[i] for i in q],fam[q])))
    if split=='enzymemap':
        quality=np.array(read(OLD/split/'mapping_quality.json')['min_native_quality'])
        eligible=(quality>=.5)&np.array([len(x)==1 for x in labels]);rules=np.array([x[0] if len(x)==1 else '' for x in labels])
        matched={model:np.full(len(q),np.nan) for model in sims};counts=[];gaps=[]
        for j,i in enumerate(q):
            if not eligible[i]:continue
            same=np.flatnonzero(eligible&(rules==rules[i])&(np.arange(n)!=i));different=np.flatnonzero(eligible&(rules!=rules[i]))
            if not len(same) or not len(different):continue
            order=different[np.argsort(chem[i,different],kind='stable')];vv=chem[i,order];at=np.searchsorted(vv,chem[i,same])
            left=np.clip(at-1,0,len(vv)-1);right=np.clip(at,0,len(vv)-1)
            chosen=np.where(abs(vv[left]-chem[i,same])<=abs(vv[right]-chem[i,same]),left,right)
            gap=abs(vv[chosen]-chem[i,same]);keep=gap<=.02;a=same[keep];b=order[chosen[keep]]
            if not len(a):continue
            counts.append(len(a));gaps.extend(gap[keep].tolist())
            for model,sim in sims.items():matched[model][j]=np.mean(sim[i,a]-sim[i,b])
        assert len(counts)==427 and sum(counts)==55704
        for model in list(matched):
            if model!='cersei':matched['cersei-'+model]=matched['cersei']-matched[model]
        for model,y in matched.items():
            summaries.append(dict(model=model,metric='quality_matched_gap',**summarize(y,[labels[i] for i in q],fam[q])))
            records.extend(dict(model=model,metric='quality_matched_gap',query_id=meta['reaction_ids'][i],value=float(v)) for i,v in zip(q,y) if np.isfinite(v))
        dump(dest/'reaction_matching_coverage.json',dict(matched_queries=len(counts),comparisons=sum(counts),mean_tanimoto_gap=float(np.mean(gaps)),max_tanimoto_gap=float(np.max(gaps)),protocol='Native mapping quality>=.5, one rule per reaction, matched participant similarity within .02'))
    dump(dest/'reaction_summary.json',summaries);write_table(dest/'reaction_per_query.csv',records)
    print(split,'REACTIONS COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--device',default='cuda:2');p.add_argument('--control',choices=['clean','enzymecage']);a=p.parse_args()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    if a.split=='enzymemap' and not (OUT/a.split/'horizyn_embeddings.npz').exists():prepare_map()
    neighborhood(a.split,a.device,a.control)
    if not a.control:alignment(a.split);reactions(a.split)
