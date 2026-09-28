#!/usr/bin/env python3
"""Matched-data training of the released ReactZyme classifier families.

Model classes are loaded verbatim by AST from the pinned upstream scripts,
without executing their top-level argument parsing or training side effects.
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from reactzyme_public_features import ROOT, RUN, SPLITS, save_json, sha
from generalization_metrics import evaluate_scores

SOURCES={'mlp':'train.py','contrastive':'train_contra.py',
         'contrastive_corrected':'train_contra.py','transformer':'train_tfmr.py','birnn':'train_rnn.py'}


def upstream_classes(family):
    path=ROOT/'.deps/ReactZyme'/SOURCES[family]
    tree=ast.parse(path.read_text())
    classes=ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef)],type_ignores=[])
    scope={'torch':torch,'nn':torch.nn,'F':F}
    exec(compile(classes,str(path),'exec'),scope)
    return scope,path


def classification_pairs(positives,known,seed,negatives=4):
    rng=np.random.default_rng(seed)
    forbidden=set((int(r),int(p)) for r,p in known)
    rpool=np.unique(positives[:,0]);ppool=np.unique(positives[:,1])
    # A fully positive row/column has no valid one-sided corruption.
    rset,pset=set(rpool.tolist()),set(ppool.tolist())
    rows={};cols={}
    for r,p in forbidden:
        if r in rset and p in pset:
            rows[r]=rows.get(r,0)+1;cols[p]=cols.get(p,0)+1
    if any(rows.get(int(r),0)==len(ppool) or cols.get(int(p),0)==len(rpool) for r,p in positives):
        raise ValueError('No valid one-sided negative for a fully positive row or column')
    neg=np.repeat(positives,negatives,axis=0).copy()
    swap=np.arange(len(neg))%2==0
    neg[swap,0]=rng.choice(rpool,int(swap.sum()))
    neg[~swap,1]=rng.choice(ppool,int((~swap).sum()))
    bad=np.array([(int(r),int(p)) in forbidden for r,p in neg])
    while bad.any():
        for i in np.flatnonzero(bad):
            axis=0 if swap[i] else 1;neg[i,axis]=rng.choice(rpool if axis==0 else ppool)
        bad=np.array([(int(r),int(p)) in forbidden for r,p in neg])
    x=np.concatenate([positives,neg]);y=np.concatenate([np.ones(len(positives)),np.zeros(len(neg))]).astype(np.float32)
    return x,y


@torch.inference_mode()
def fast_scores(model,reactions,proteins,family,batch=32768):
    """Exact eval-mode cache: singleton cross-attention reduces to its value map."""
    model.eval();device=reactions.device
    def tower(module,x):return torch.cat([module(c) for c in x.split(4096)])
    re=model.cross_attn_seq.W_V(tower(model.lin_mol_embed,reactions))
    pe=model.cross_attn_mol.W_V(tower(model.lin_seq_embed,proteins))
    nr,np_=len(re),len(pe);scores=torch.empty((nr,np_),device=device)
    for start in range(0,nr*np_,batch):
        idx=torch.arange(start,min(start+batch,nr*np_),device=device)
        # The upstream concat order is protein-conditioned then reaction-conditioned.
        protein,reaction=pe[idx%np_],re[idx//np_]
        if family in ('mlp','contrastive','contrastive_corrected'):
            out=model.lin_out(torch.cat([protein,reaction],dim=-1))
        else:
            embedded=torch.stack([protein,reaction],dim=1)
            if family=='birnn':
                z,_=model.gru(embedded)
                z=z[:,:,:model.hidden_dim]+z[:,:,model.hidden_dim:]
            else:z=model.transformer(embedded,embedded)
            out=model.lin_out(z.sum(1))
        scores.view(-1)[start:start+len(idx)]=out.view(-1)
    return scores


@torch.inference_mode()
def original_scores(model,reactions,proteins,batch=4096):
    """Numerical fallback through the unmodified upstream pair forward."""
    model.eval();nr,np_=len(reactions),len(proteins)
    scores=torch.empty((nr,np_),device=reactions.device)
    for start in range(0,nr*np_,batch):
        idx=torch.arange(start,min(start+batch,nr*np_),device=reactions.device)
        out=model(reactions[idx//np_],proteins[idx%np_])
        if isinstance(out,tuple):out=out[0]
        scores.view(-1)[start:start+len(idx)]=out.view(-1)
    return scores


def validate_fast_path(model,rxn,prot,family,strict=True):
    model.eval()
    nr,np_=min(len(rxn),5),min(len(prot),7)
    with torch.no_grad():
        r=rxn[:nr].repeat_interleave(np_,0);p=prot[:np_].repeat(nr,1)
        raw=model(r,p);raw=raw[0] if isinstance(raw,tuple) else raw
        fast=fast_scores(model,rxn[:nr],prot[:np_],family,batch=11)
    error=float((raw.reshape(nr,np_)-fast).abs().max())
    if strict and not torch.allclose(raw.reshape(nr,np_),fast,atol=3e-5,rtol=3e-5):
        raise AssertionError(f'Cached inference is not equivalent: {family} error={error}')
    return error


def run(a):
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    device=f'cuda:{a.gpu}';feat=a.features
    out=a.output or RUN/'models'/f'{a.family}_{a.protein}_{a.reaction}_{a.split}_seed{a.seed}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():print('Already complete:',out);return
    if (out/'protocol.json').exists() and not (a.resume or a.evaluate_only):
        raise FileExistsError(f'Existing experiment; use --resume: {out}')
    for name in (a.reaction,a.protein):
        receipt=json.loads((feat/f'{name}.complete.json').read_text())
        if receipt['catalog_sha256']!=sha(feat/'catalog.json'):raise ValueError('Feature catalog mismatch')
    rxn=torch.from_numpy(np.load(feat/f'{a.reaction}.npy')).to(device)
    prot=torch.from_numpy(np.load(feat/f'{a.protein}.npy')).to(device)
    data=np.load(feat/'pairs.npz');train=data[f'{a.split}_train'];valid=data[f'{a.split}_validation']
    cls,path=upstream_classes(a.family)
    model=cls['PretrainedNetwork'](mol_input_dim=rxn.shape[1],seq_input_dim=prot.shape[1],
                                   hidden_dim=128,output_dim=64,dropout=0.).to(device)
    criterion=cls.get('ContrastiveLoss',lambda:None)()
    fast_error=validate_fast_path(model,rxn,prot,a.family)
    protocol=dict(family=a.family,reaction=a.reaction,protein=a.protein,split=a.split,seed=a.seed,
        max_epochs=a.epochs,early_stopping_patience=a.patience,batch_size=a.batch,lr=a.lr,
        weight_decay=5e-10,negative_ratio=4,negative_sampling='Fixed uniform reaction/protein corruption within each part; exclude known training positives; validation also excludes validation positives',
        selection='Minimum validation BCE; test evaluated once after selection',
        loss='BCE plus upstream contrastive loss' if 'contrastive' in a.family else 'BCE',
        contrastive_labels='Corrected: positive attracts' if a.family=='contrastive_corrected' else 'Unmodified upstream label convention',
        source=str(path),source_sha256=sha(path),catalog_sha256=sha(feat/'catalog.json'),
        split_data_sha256=sha(feat/'pairs.npz'),source_audit=json.loads((feat/'data_audit.json').read_text()),
        upstream_checkpoint_selection_bug_fixed=True,fast_inference_max_error=fast_error,
        exact_published_score_reproduction=False,test_used_for_training_or_selection=False)
    if not (out/'protocol.json').exists():save_json(out/'protocol.json',protocol)
    tx,ty=classification_pairs(train,train,a.seed)
    vx,vy=classification_pairs(valid,np.concatenate([train,valid]),a.seed+10000)
    np.savez(out/'negative_samples.npz',train=tx,validation=vx)
    tx=torch.as_tensor(tx,device=device);ty=torch.as_tensor(ty,device=device)
    vx=torch.as_tensor(vx,device=device);vy=torch.as_tensor(vy,device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=5e-10)
    best=float('inf');best_epoch=0;start_epoch=1
    if a.resume and (out/'last.pt').exists():
        ck=torch.load(out/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        best=ck['best'];best_epoch=ck['best_epoch'];start_epoch=ck['epoch']+1
        torch.set_rng_state(ck['cpu_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu(),a.gpu)
        if ck['epoch']-best_epoch>=a.patience:start_epoch=a.epochs+1
    if a.evaluate_only:start_epoch=a.epochs+1
    for epoch in range(start_epoch,a.epochs+1):
        begin=time.time();model.train();order=torch.randperm(len(tx),device=device);total=torch.zeros((),device=device)
        for idx in order.split(a.batch):
            # BatchNorm requires at least two examples; never discard an association.
            if len(idx)==1:idx=torch.cat([idx,order[:1]])
            pair=tx[idx];labels=ty[idx];weights=torch.where(labels>0,1.,.25)
            opt.zero_grad(set_to_none=True)
            pred=model(rxn[pair[:,0]],prot[pair[:,1]])
            if isinstance(pred,tuple):scores,zr,ze=pred
            else:scores=pred
            loss=F.binary_cross_entropy_with_logits(scores.view(-1),labels,weight=weights)
            if criterion is not None:
                loss=loss+criterion(zr,ze,1-labels if a.family=='contrastive_corrected' else labels,weights)
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite training loss')
            loss.backward();opt.step();total+=loss.detach()*len(idx)
        model.eval();val=torch.zeros((),device=device)
        with torch.inference_mode():
            for pair,labels in zip(vx.split(a.batch*4),vy.split(a.batch*4)):
                pred=model(rxn[pair[:,0]],prot[pair[:,1]])
                if isinstance(pred,tuple):pred=pred[0]
                val+=F.binary_cross_entropy_with_logits(pred.view(-1),labels,
                        weight=torch.where(labels>0,1.,.25),reduction='sum')
        value=float(val/len(vx));improved=value<best
        if improved:
            best=value;best_epoch=epoch
            torch.save(dict(model=model.state_dict(),epoch=epoch,validation_bce=value),out/'best.pt')
        record=dict(epoch=epoch,training_loss=float(total/len(tx)),validation_bce=value,
                    best_epoch=best_epoch,best_validation_bce=best,seconds=time.time()-begin)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        save_json(out/'status.json',dict(stage='training',**record));print(json.dumps(record),flush=True)
        if epoch%5==0 or epoch==a.epochs or epoch-best_epoch>=a.patience:
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
                best=best,best_epoch=best_epoch,cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(a.gpu)),out/'last.pt')
        if epoch-best_epoch>=a.patience:break
    selected=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(selected['model'])
    save_json(out/'selection.json',dict(epoch=selected['epoch'],validation_bce=selected['validation_bce'],
        checkpoint_sha256=sha(out/'best.pt'),test_used=False))
    save_json(out/'status.json',dict(stage='test_evaluation',selected_epoch=selected['epoch']))
    test=data[f'{a.split}_test'];rids=np.unique(test[:,0]);pids=np.unique(test[:,1])
    fast_error=validate_fast_path(model,rxn[rids],prot[pids],a.family,strict=False)
    # At large trained logits, regrouped GEMMs can exceed the tight parity
    # tolerance even though the singleton attention identity is exact.
    # Fall back to the original forward rather than loosening that tolerance.
    score_path='cached_singleton_attention' if fast_error<=3e-5 else 'original_pair_forward'
    if score_path=='original_pair_forward':
        scores=original_scores(model,rxn[rids],prot[pids])
    else:scores=fast_scores(model,rxn[rids],prot[pids],a.family,a.score_batch)
    truth=dict(reaction_index=np.searchsorted(rids,test[:,0]),enzyme_index=np.searchsorted(pids,test[:,1]))
    evaluated=evaluate_scores(scores,truth)
    np.savez(out/'test_scores.npz',scores=scores.cpu().numpy(),reaction_index=rids,protein_index=pids)
    np.savez(out/'test_per_query.npz',**{f'{d}_{k}':v for d,sections in evaluated['per_query'].items() for k,v in sections['all'].items()})
    result=dict(family=a.family,protein=a.protein,reaction=a.reaction,split=a.split,seed=a.seed,
        selected_epoch=selected['epoch'],validation_bce=selected['validation_bce'],
        test=evaluated['summary'],test_candidate_shape=list(scores.shape),
        fast_inference_max_error=fast_error,scoring_path=score_path,test_used_for_selection=False)
    save_json(out/'complete.json',result);save_json(out/'status.json',dict(stage='complete'))
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=list(SOURCES),required=True);p.add_argument('--split',choices=SPLITS,required=True)
    p.add_argument('--protein',default='esm2');p.add_argument('--reaction',default='mat_2d')
    p.add_argument('--features',type=Path,default=RUN/'features');p.add_argument('--output',type=Path)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch',type=int,default=1000);p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--patience',type=int,default=30);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--score-batch',type=int,default=32768);p.add_argument('--resume',action='store_true')
    p.add_argument('--evaluate-only',action='store_true',help='Evaluate the already validation-selected best.pt')
    run(p.parse_args())


if __name__=='__main__':main()
