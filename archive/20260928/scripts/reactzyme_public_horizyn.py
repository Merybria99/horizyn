#!/usr/bin/env python3
"""Official Horizyn towers/MLNCE with a disclosed ReactZyme input adapter."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

from reactzyme_public_features import ROOT,RUN,SPLITS,save_json,sha
from generalization_metrics import evaluate_scores

NATIVE=ROOT/'data/external/cyp_specificity_2026/horizyn_official'


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def features():
    # Import the original feature generators, not our modified feature classes.
    sys.path.insert(0,str(NATIVE))
    from horizyn.datasets.csv import CSVDataset
    from horizyn.datasets.fingerprints.rdkit_plus import RDKitPlusFingerprintDataset
    from horizyn.datasets.fingerprints.drfp import DRFPFingerprintDataset
    import csv
    root=RUN/'features';cat=json.loads((root/'catalog.json').read_text())
    path=root/'horizyn_participant_sets.csv'
    with path.open('w') as f:
        w=csv.writer(f);w.writerow(['reaction_id','reaction_smiles'])
        for rid,smi in zip(cat['reaction_ids'],cat['reaction_smiles']):
            # The benchmark omits substrate/product roles. No Rhea direction is recovered.
            w.writerow([rid,smi.replace('*','C')+'>>'])
    ds=CSVDataset(file_path=str(path),key_column='reaction_id',columns=['reaction_smiles'])
    rd=RDKitPlusFingerprintDataset(ds,vec_dim=1024,standardize=True)
    dr=DRFPFingerprintDataset(ds,vec_dim=1024,standardize=True)
    x=[]
    for i,rid in enumerate(cat['reaction_ids']):
        x.append(torch.cat([rd[rid],dr[rid]]).numpy())
        if i%500==0:print('fingerprints',i,len(cat['reaction_ids']),flush=True)
    x=np.stack(x);assert np.isfinite(x).all();np.save(root/'horizyn_fp.npy',x)
    save_json(root/'horizyn_fp.complete.json',dict(count=len(x),catalog_sha256=sha(root/'catalog.json'),
        sha256=sha(root/'horizyn_fp.npy'),source=str(NATIVE),
        adaptation='All released participants on input side, empty output side. No reaction direction supplied. RDKit+/DRFP dimensions and generators unchanged.',
        exact_native_reaction_representation=False))


def train(a):
    torch.set_num_threads(4);torch.manual_seed(42);torch.set_float32_matmul_precision('highest')
    root=RUN/'features';out=RUN/'models'/f'horizyn_participant_set_{a.split}_seed42';out.mkdir(exist_ok=True,parents=True)
    if (out/'complete.json').exists():return
    if (out/'protocol.json').exists():raise FileExistsError(out)
    device=f'cuda:{a.gpu}'
    for n in ('prott5','horizyn_fp'):
        receipt=json.loads((root/f'{n}.complete.json').read_text());assert receipt['catalog_sha256']==sha(root/'catalog.json')
    rxn=torch.tensor(np.load(root/'horizyn_fp.npy'),device=device);prot=torch.tensor(np.load(root/'prott5.npy'),device=device)
    model_source=NATIVE/'horizyn/model.py';loss_source=NATIVE/'horizyn/losses.py'
    m=load_file('original_horizyn_model',model_source);lossmod=load_file('original_horizyn_loss',loss_source)
    model=m.DualContrastiveModel(
        query_encoder_kwargs=dict(input_dim=2048,output_dim=512,num_layers=2,widths=[4096,4096],normalise_output=True),
        target_encoder_kwargs=dict(input_dim=1024,output_dim=512,num_layers=2,widths=[4096,4096],normalise_output=True)).to(device)
    criterion=lossmod.FullBatchMLNCELoss(beta=10.,learn_beta=False).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    data=np.load(root/'pairs.npz');pairs=torch.tensor(data[f'{a.split}_train'],device=device)
    valid=data[f'{a.split}_validation'];rids=np.unique(valid[:,0]);pids=np.unique(valid[:,1])
    truth=dict(reaction_index=np.searchsorted(rids,valid[:,0]),enzyme_index=np.searchsorted(pids,valid[:,1]))
    protocol=dict(method='Original Horizyn architecture and MLNCE, participant-set reaction adapter',
        split=a.split,seed=42,batch=16384,epochs=100,lr=1e-4,weight_decay=.01,beta=10,
        query_dims=[2048,4096,4096,512],protein_dims=[1024,4096,4096,512],
        source_model_sha256=sha(model_source),source_loss_sha256=sha(loss_source),
        data_sha256=sha(root/'pairs.npz'),selection='Maximum mean bidirectional validation all-positive MRR, every 5 epochs',
        precision='BF16 autocast training with FP32 MLNCE; FP32 scoring',test_used_for_training_or_selection=False)
    save_json(out/'protocol.json',protocol);best=-1
    def embeddings(tower,x):return torch.cat([tower(z) for z in x.split(4096)])
    for epoch in range(1,101):
        begin=time.time();model.train();total=0.
        for ids in torch.randperm(len(pairs),device=device).split(16384):
            edges=pairs[ids];r,ri=torch.unique(edges[:,0],return_inverse=True);p,pi=torch.unique(edges[:,1],return_inverse=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):zr,zp=model(rxn[r],prot[p])
            distances=1-zr.float()@zp.float().T;loss=criterion(distances,ri,pi)
            loss.backward();opt.step();total+=float(loss.detach())*len(ids)
        record=dict(epoch=epoch,loss=total/len(pairs),seconds=time.time()-begin)
        if epoch%5==0:
            model.eval()
            with torch.inference_mode():
                z=embeddings(model.query_encoder,rxn[rids])@embeddings(model.target_encoder,prot[pids]).T
                val=evaluate_scores(z,truth)['summary']
            value=np.mean([v['all']['reactzyme_mrr'] for v in val.values()])
            record.update(validation=val,selection_value=float(value))
            if value>best:
                best=value;torch.save(dict(model=model.state_dict(),epoch=epoch,value=float(value)),out/'best.pt')
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        save_json(out/'status.json',dict(stage='training',**record));print(json.dumps(record),flush=True)
    selected=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(selected['model']);model.eval()
    save_json(out/'selection.json',dict(epoch=selected['epoch'],validation_value=selected['value'],test_used=False))
    test=data[f'{a.split}_test'];rids=np.unique(test[:,0]);pids=np.unique(test[:,1])
    truth=dict(reaction_index=np.searchsorted(rids,test[:,0]),enzyme_index=np.searchsorted(pids,test[:,1]))
    with torch.inference_mode():
        scores=embeddings(model.query_encoder,rxn[rids])@embeddings(model.target_encoder,prot[pids]).T
        result=evaluate_scores(scores,truth)
    np.savez(out/'test_scores.npz',scores=scores.cpu().numpy(),reaction_index=rids,protein_index=pids)
    save_json(out/'complete.json',dict(method=protocol['method'],split=a.split,selected_epoch=selected['epoch'],test=result['summary']))
    save_json(out/'status.json',dict(stage='complete'))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['features','train']);p.add_argument('--split',choices=SPLITS,default='reaction_smi');p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args();features() if a.action=='features' else train(a)


if __name__=='__main__':main()
