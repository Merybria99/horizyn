#!/usr/bin/env python3
"""Frozen MAT feature extraction using the released ReactZyme implementation."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from reactzyme_public_features import ROOT, RUN, save_json, sha


def molecule_features(item):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    smiles, mode = item
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid released molecule: {smiles}')
    fallback=False
    if mode=='3d':
        hmol=Chem.AddHs(mol)
        params=AllChem.ETKDGv3();params.randomSeed=42;params.maxIterations=100
        result=AllChem.EmbedMolecule(hmol,params)
        if result==0:
            try: AllChem.UFFOptimizeMolecule(hmol,maxIters=200)
            except (ValueError,RuntimeError): pass
            mol=Chem.RemoveHs(hmol)
        else:
            AllChem.Compute2DCoords(mol);fallback=True
    else:AllChem.Compute2DCoords(mol)
    def onehot(v, options):return [v==x for x in options] if v in options else [x==options[-1] for x in options]
    rows=[]
    for a in mol.GetAtoms():
        rows.append(onehot(a.GetAtomicNum(),[5,6,7,8,9,15,16,17,35,53,999])+
                    onehot(len(a.GetNeighbors()),[0,1,2,3,4,5])+
                    onehot(a.GetTotalNumHs(),[0,1,2,3,4])+
                    onehot(a.GetFormalCharge(),[-1,0,1])+[a.IsInRing(),a.GetIsAromatic()])
    n=len(rows);nodes=np.zeros((n+1,28),np.float32);nodes[0,0]=1;nodes[1:,1:]=rows
    adjacency=np.zeros((n+1,n+1),np.float32);adjacency[1:,1:]=np.eye(n)
    for b in mol.GetBonds():
        i,j=b.GetBeginAtomIdx()+1,b.GetEndAtomIdx()+1;adjacency[i,j]=adjacency[j,i]=1
    pos=mol.GetConformer().GetPositions()
    distance=np.full((n+1,n+1),1e6,np.float32)
    distance[1:,1:]=np.linalg.norm(pos[:,None]-pos[None,:],axis=-1)
    return nodes,adjacency,distance,fallback


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['2d','3d'],default='2d')
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--workers',type=int,default=16)
    p.add_argument('--output',type=Path,default=RUN/'features');a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    spec=importlib.util.spec_from_file_location('official_mat',ROOT/'.deps/ReactZyme/mat.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.make_model(d_atom=28,d_model=1024,N=8,h=16,N_dense=1,
        lambda_attention=.33,lambda_distance=.33,leaky_relu_slope=.1,
        dense_output_nonlinearity='relu',distance_matrix_kernel='exp',dropout=0.,aggregation_type='mean')
    weight=RUN/'assets/mat.pt';state=torch.load(weight,map_location='cpu',weights_only=False)
    own=model.state_dict()
    for name,value in state.items():
        if 'generator' not in name:own[name].copy_(value)
    model.to(f'cuda:{a.gpu}').eval()
    catalog=json.loads((a.output/'catalog.json').read_text())
    fragments=[r.replace('*','C').split('.') for r in catalog['reaction_smiles']]
    unique=sorted(set(s for row in fragments for s in row))
    # Sort by string length to reduce padding; actual graph size governs batching.
    unique.sort(key=lambda s:(len(s),s)); lookup={s:i for i,s in enumerate(unique)}
    name='mat_'+a.mode; path=a.output/f'{name}_molecules.npy'
    feats=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float32,shape=(len(unique),1024))
    done_path=a.output/f'{name}_molecules.done.npy'
    done=np.load(done_path) if done_path.exists() else np.zeros(len(unique),bool)
    todo=np.flatnonzero(~done).tolist();fallbacks=[];start=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        # Submit bounded chunks: large glycan distance matrices must not fill host RAM.
        for offset in range(0,len(todo),32):
            batch_idx=todo[offset:offset+32]
            records=list(pool.map(molecule_features,[(unique[i],a.mode) for i in batch_idx]))
            for i,(x,adj,distance,fallback) in zip(batch_idx,records):
                args=[torch.from_numpy(v).unsqueeze(0).to(f'cuda:{a.gpu}') for v in (x,adj,distance)]
                with torch.inference_mode():
                    encoded=model.encode(args[0],args[0].abs().sum(-1)!=0,args[1],args[2],None)
                    value=encoded.squeeze(0).mean(0).cpu().numpy()
                if not np.isfinite(value).all():raise ValueError('Nonfinite MAT features')
                feats[i]=value;done[i]=True
                if fallback:fallbacks.append(unique[i])
            feats.flush();np.save(done_path,done)
            progress=dict(stage='MAT feature extraction',mode=a.mode,completed=int(done.sum()),
                          total=len(done),seconds=time.time()-start)
            save_json(a.output/f'{name}.status.json',progress);print(json.dumps(progress),flush=True)
    reaction=np.stack([np.asarray(feats[[lookup[s] for s in row]]).mean(0) for row in fragments])
    np.save(a.output/f'{name}.npy',reaction)
    save_json(a.output/f'{name}.complete.json',dict(count=len(reaction),molecules=len(unique),
        pooling='Mean of atom encodings including dummy node, then mean over molecular occurrences; official process_mat.py',
        mode=a.mode,coordinate_fallbacks=fallbacks,weight_sha256=sha(weight),
        catalog_sha256=sha(a.output/'catalog.json'),sha256=sha(a.output/f'{name}.npy')))


if __name__=='__main__':main()
