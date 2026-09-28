#!/usr/bin/env python3
"""Frozen-checkpoint attention diagnostics and view suppression, no fitting."""
from __future__ import annotations
import argparse
from collections import defaultdict
import gc
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from cersei_embedding_organization import (ROOT,OUT,KROOT,SPLITS,read,dump,sha,checked_cache,get_head,refine,prefix,norm)
from cersei_embedding_metrics import summarize,write_table,families_for,query_metrics


def cka(a,b,weights=None):
    """Weighted centered linear CKA, descriptive (not assigned a pairwise CI)."""
    a=a.double();b=b.double()
    if weights is None:w=torch.ones(len(a),device=a.device,dtype=a.dtype)/len(a)
    else:w=torch.as_tensor(weights,device=a.device,dtype=a.dtype);w=w/w.sum()
    a=(a-(w[:,None]*a).sum(0))*w.sqrt()[:,None]
    b=(b-(w[:,None]*b).sum(0))*w.sqrt()[:,None]
    return float((a.T@b).square().sum() / ((a.T@a).square().sum()*(b.T@b).square().sum()).sqrt().clamp_min(1e-20))


def macro_weights(labels):
    counts=defaultdict(int)
    for row in labels:
        for label in row:counts[label]+=1
    return np.array([sum(1/counts[label] for label in row) for row in labels])


def source_task(split,k):
    root=KROOT/f'k{k}'/split
    task=read(root/'evaluation/protocol.json')['tasks'][0]
    assert sha(task['model']['checkpoint'])==task['model']['checkpoint_sha256']
    task['head_path']=str(root/f'evaluation/{split}/v4/training/step0100.pt')
    summary=read(root/f'evaluation/{split}/v4/test_summary.json')
    assert sha(task['head_path'])==summary['freeze']['head_sha256']
    return root,task


@torch.inference_mode()
def export(split,k,device):
    from generalization_clipzyme_f3_screen import model_from_checkpoint
    from horizyn.training_io import StoragePrecisionResidues
    from horizyn.utils import residue_collate_fn
    root,task=source_task(split,k);m=task['model']
    d=OUT/split;dest=d/f'learned_views/k{k}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'export_receipt.json').exists():return
    meta=read(d/'metadata.json');subset=read(d/'subsets.json');selected=np.array(subset['protein_indices'])
    keys=[meta['protein_ids'][i] for i in selected]
    cache=checked_cache(root/f'evaluation/{split}/test_cache.pt')
    if split=='enzymemap':
        lib=checked_cache(root/f'evaluation/{split}/screening_library.pt')
        assert lib['protein_ids']==meta['protein_ids'];base_e=lib['base_e'];assert cache['ids']==meta['reaction_ids']
    else:
        cat=read(root/'phase2/test_features/catalog.json')
        assert cat['proteins']==meta['protein_ids'] and cat['reactions']==meta['reaction_ids'];base_e=cache['base_e']
    bank=np.array(meta['neighborhood_indices']);head=get_head(task,device)
    refined_bank=refine(base_e[bank],head,'enzyme',device);refined_reactions=refine(cache['base_r'],head,'reaction',device)
    # The model sees exactly the cached residues, including the existing
    # max-token/truncation rule. No annotations are provided to the encoder.
    model,config=model_from_checkpoint(Path(m.get('test_config',m['config'])),Path(m['checkpoint']),device)
    encoder=model.model.multiview_encoder;assert encoder.num_slots==k
    path=config.data.protein_residue_embeds_path
    if split=='enzymemap':path=ROOT/'runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_catalog_v1/features/proteins_prott5_residue.local.h5'
    residues=StoragePrecisionResidues(str(path),max_tokens=config.data.max_protein_tokens,truncation=config.data.protein_truncation)
    stored={};handles=[]
    def capture(key):
        def hook(module,inp,out):stored[key]=F.normalize(out.float(),dim=-1,eps=1e-6)
        return hook
    handles.append(encoder.global_output.register_forward_hook(capture('global')))
    for j,p in enumerate(encoder.view_projections):handles.append(p.register_forward_hook(capture(f'view{j}')))
    scale=float(encoder.residual_scale)*m['fusion_multiplier']
    metrics=[];all_embeddings=[];projections=[];max_error=0.;reconstruction_error=0.
    variants=['intact']+[f'without_slot{j+1}' for j in range(k)]+['without_all_slots']
    try:
        for start in range(0,len(keys),32):
            ids=keys[start:start+32]
            batch=residue_collate_fn([dict(residues[p],target_id=p) for p in ids])
            x=batch['residue_embeddings'].to(device=device,dtype=next(model.model.parameters()).dtype)
            mask=batch['residue_padding_mask'].to(device)
            native,details=model.model.encode_targets(x,residue_padding_mask=mask,return_pooling_details=True)
            views=torch.stack([stored[f'view{j}'] for j in range(k+2)],1)
            gates=details['enzyme_multiview_gate_weights'];g=stored['global']
            f=F.normalize(encoder.fused_output((views*gates).sum(1)).float(),dim=-1,eps=1e-6)
            reconstructed=F.normalize(g+float(encoder.residual_scale)*f,dim=-1,eps=1e-6)
            reconstruction_error=max(reconstruction_error,float((native-reconstructed).abs().max()))
            deployed=F.normalize(g+scale*f,dim=-1,eps=1e-6)
            err=float((deployed.cpu()-base_e[selected[start:start+len(ids)]]).abs().max());max_error=max(max_error,err)
            if err>3e-6 or reconstruction_error>2e-6:
                raise ValueError(f'View export did not reproduce checkpoint cache: {err}, reconstruction {reconstruction_error}')
            embeddings=[deployed]
            for removed in [[j+2] for j in range(k)]+[list(range(2,k+2))]:
                weights=gates.clone();weights[:,removed]=0;weights=weights/weights.sum(1,keepdim=True)
                ff=F.normalize(encoder.fused_output((views*weights).sum(1)).float(),dim=-1,eps=1e-6)
                embeddings.append(F.normalize(g+scale*ff,dim=-1,eps=1e-6))
            all_embeddings.append(torch.stack([F.normalize(e+.5*head.scale*head.enzyme(e),dim=-1) for e in embeddings],1).cpu().numpy())
            projections.append(views[:,2:].cpu().numpy())
            attention=details['enzyme_multiview_attention'];raw=details['enzyme_multiview_raw_attention'];lengths=(~mask).sum(1)
            unit=F.normalize(attention,dim=-1);attcos=unit@unit.transpose(1,2)
            vcos=views[:,2:]@views[:,2:].transpose(1,2)
            # Top 10% of each actual sequence, never padded positions.
            order=torch.argsort(attention,dim=-1,descending=True,stable=True)
            ranks=torch.empty_like(order);ranks.scatter_(-1,order,torch.arange(order.shape[-1],device=device).expand_as(order))
            top=(ranks<torch.ceil(lengths[:,None,None]*.1)) & (~mask[:,None])
            inter=top.float()@top.float().transpose(1,2);count=top.sum(-1)
            jac=inter/(count[:,:,None]+count[:,None,:]-inter).clamp_min(1)
            triangle=torch.triu(torch.ones(k,k,device=device,dtype=torch.bool),1)
            for i,p in enumerate(ids):
                row=dict(protein_id=p,length=int(lengths[i]),
                    attention_cosine=float(attcos[i][triangle].mean()) if k>1 else np.nan,
                    centered_attention_cosine=float(details['enzyme_multiview_head_similarity'][i]) if k>1 else np.nan,
                    top10pct_jaccard=float(jac[i][triangle].mean()) if k>1 else np.nan,
                    projected_view_cosine=float(vcos[i][triangle].mean()) if k>1 else np.nan,
                    support_fraction=float(details['enzyme_multiview_effective_support'][i]/lengths[i]),
                    normalized_entropy=float(details['enzyme_multiview_effective_normalized_entropy'][i]),
                    learned_gate_mass=float(gates[i,2:].sum(0).mean()),
                    sleec_gate_mass=float(gates[i,1].mean()),global_gate_mass=float(gates[i,0].mean()))
                for j in range(k):row[f'slot{j+1}_gate_mass']=float(gates[i,j+2].mean())
                metrics.append(row)
            if start%320==0:print('VIEW EXPORT',split,k,start+len(ids),'/',len(keys),flush=True)
    finally:
        for handle in handles:handle.remove()
        residues.close()
    projected=np.concatenate(projections);embeddings=np.concatenate(all_embeddings)
    assert np.max(np.abs(embeddings[:,0]-refine(base_e[selected],head,'enzyme',device)))<3e-6
    labels=[prefix(meta['enzyme_ec'][i],3) for i in selected];weights=macro_weights(labels)
    ckamat=np.eye(k)
    for i in range(k):
        for j in range(i):
            ckamat[i,j]=ckamat[j,i]=cka(torch.as_tensor(projected[:,i],device=device),torch.as_tensor(projected[:,j],device=device),weights)
    np.savez(dest/'export.npz',embeddings=embeddings,projected_views=projected,
        bank=refined_bank,reactions=refined_reactions,selected=selected,bank_indices=bank,cka=ckamat)
    write_table(dest/'per_protein_attention.csv',metrics)
    dump(dest/'variants.json',variants)
    dump(dest/'export_receipt.json',dict(split=split,K=k,proteins=len(keys),task=task,
        checkpoint_sha256=sha(m['checkpoint']),head_sha256=sha(task['head_path']),
        cache_reproduction_max_abs_error=max_error,reconstruction_max_abs_error=reconstruction_error,
        source_residues=str(path),export_sha256=sha(dest/'export.npz'),code_sha256=sha(__file__),
        intervention='Zero chosen learned-view gates after computing all original gates; renormalize remaining gates per feature; freeze all parameters and residual head.',
        cka='EC3-class-balanced weighted linear CKA, descriptive matrix; family-bootstrap intervals are provided for per-protein diagnostics and suppression effects.'))
    del cache,base_e,model,head
    if split=='enzymemap':del lib
    gc.collect();torch.cuda.empty_cache()


@torch.inference_mode()
def evaluate(split,k,device):
    d=OUT/split;dest=d/f'learned_views/k{k}';meta=read(d/'metadata.json');z=np.load(dest/'export.npz')
    variants=read(dest/'variants.json');ids=z['selected'];bank_ids=z['bank_indices'];keys=[meta['protein_ids'][i] for i in ids]
    fam,fam50=families_for(meta,ids);_,bank50=families_for(meta,bank_ids)
    labels=[prefix(meta['enzyme_ec'][i],3) for i in ids];bank_labels=[set(prefix(meta['enzyme_ec'][i],3)) for i in bank_ids]
    annotated=np.array([bool(x) for x in bank_labels]);qsets=[set(x) for x in labels]
    truth=np.load(d/'truth.npy');expanded=np.load(d/'expanded.npy');positive=defaultdict(set)
    for r,e in truth:positive[int(expanded[e])].add(int(r))
    lookup={int(g):i for i,g in enumerate(bank_ids)};self_indices=np.array([lookup[int(g)] for g in ids])
    bank=torch.as_tensor(z['bank'],device=device);reactions=torch.as_tensor(z['reactions'],device=device)
    embeddings=z['embeddings'];records=[];arrays={}
    for v,variant in enumerate(variants):
        scores=(torch.as_tensor(embeddings[:,v],device=device)@reactions.T).cpu().numpy()
        y=defaultdict(list)
        for i,s in enumerate(scores):
            r=query_metrics(s,sorted(positive[int(ids[i])]))
            y['coverage10'].append(r['coverage10']);y['coverage50'].append(r['coverage50']);y['hardest_margin'].append(r['hardest_margin'])
            y['cosine_shift'].append(1-float(np.dot(embeddings[i,0],embeddings[i,v])))
        for start in range(0,len(ids),128):
            qq=np.arange(start,min(start+128,len(ids)));x=torch.as_tensor(embeddings[qq,v],device=device)
            sim=x@bank.T;sim[:,torch.as_tensor(~annotated,device=device)]=-torch.inf
            same=torch.as_tensor(fam50[qq,None]==bank50[None,:],device=device);sim[same]=-torch.inf
            val,nn=torch.topk(sim,k=10,dim=1);nn=nn.cpu().numpy();val=val.cpu().numpy()
            for qi,indices,values in zip(qq,nn,val):
                y['ec3_agreement10_no_homologs'].append(float(np.mean([bool(qsets[qi]&bank_labels[j]) for j in indices])) if qsets[qi] and np.isfinite(values).all() else np.nan)
        for metric,values in y.items():
            arrays[(variant,metric)]=np.array(values)
            for i,value in enumerate(values):records.append(dict(protein_id=keys[i],K=k,variant=variant,metric=metric,value=value))
        print('VIEW SUPPRESSION',split,k,variant,flush=True)
    summaries=[]
    for (variant,metric),values in arrays.items():
        summaries.append(dict(K=k,variant=variant,metric=metric,**summarize(values,labels,fam)))
        if variant!='intact':
            summaries.append(dict(K=k,variant=variant+'-intact',metric=metric,
                **summarize(values-arrays[('intact',metric)],labels,fam)))
    import csv
    with (dest/'per_protein_attention.csv').open() as f:attention=list(csv.DictReader(f))
    assert [r['protein_id'] for r in attention]==keys
    for metric in attention[0]:
        if metric in ('protein_id','length'):continue
        values=np.array([float(r[metric]) for r in attention])
        summaries.append(dict(K=k,variant='attention',metric=metric,**summarize(values,labels,fam)))
    write_table(dest/'per_protein_suppression.csv',records);dump(dest/'summary.json',summaries)
    dump(dest/'complete.json',dict(K=k,split=split,queries=len(ids),annotated_ec3=sum(bool(x) for x in labels),
        family_count=len(set(fam)),reaction_bank=len(reactions),protein_neighborhood_bank=len(bank),
        retrieval='Neural enzyme-to-reaction coverage, all known positives unioned for identical EnzymeMap sequences; full test reaction bank.',
        uncertainty='Whole protein-family cluster bootstrap; macro EC3; fixed candidate banks; conditional on one fit.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',required=True,choices=SPLITS);p.add_argument('--k',required=True,type=int,choices=[1,2,4,8]);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    export(a.split,a.k,a.device);evaluate(a.split,a.k,a.device)


if __name__=='__main__':main()
