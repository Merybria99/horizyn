#!/usr/bin/env python3
"""Uni-Mol v1 molecular features; atomic mean then component sum as upstream."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import time
import numpy as np
from reactzyme_public_features import ROOT,RUN,save_json,sha


def coords(item):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    smi,mode=item;mol=Chem.MolFromSmiles(smi)
    if mol is None:raise ValueError(smi)
    mol=Chem.AddHs(mol);fallback=False
    if mode=='3d':
        p=AllChem.ETKDGv3();p.randomSeed=42;p.maxIterations=100
        if AllChem.EmbedMolecule(mol,p)==0:
            try:AllChem.UFFOptimizeMolecule(mol,maxIters=200)
            except (ValueError,RuntimeError):pass
        else:AllChem.Compute2DCoords(mol);fallback=True
    else:AllChem.Compute2DCoords(mol)
    return [a.GetSymbol() for a in mol.GetAtoms()],mol.GetConformer().GetPositions(),fallback


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['2d','3d'],default='2d');p.add_argument('--gpu',type=int,default=3);a=p.parse_args()
    import torch
    from unimol_tools.models.unimol import UniMolModel
    from unimol_tools.data.conformer import coords2unimol
    torch.set_num_threads(4);root=RUN/'features';assets=RUN/'assets/unimol'
    while not all((assets/n).exists() for n in ('mol_pre_all_h_220816.pt','mol.dict.txt')):time.sleep(15)
    model=UniMolModel(pretrained_model_path=str(assets/'mol_pre_all_h_220816.pt'),
        pretrained_dict_path=str(assets/'mol.dict.txt'),remove_hs=False).to(f'cuda:{a.gpu}').eval()
    cat=json.loads((root/'catalog.json').read_text());parts=[s.replace('*','C').split('.') for s in cat['reaction_smiles']]
    molecules=sorted(set(s for row in parts for s in row),key=lambda s:(len(s),s));lookup={s:i for i,s in enumerate(molecules)}
    name='unimol_'+a.mode;path=root/f'{name}_molecules.npy';donepath=root/f'{name}_molecules.done.npy'
    vectors=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float32,shape=(len(molecules),512))
    done=np.load(donepath) if donepath.exists() else np.zeros(len(molecules),bool)
    todo=np.flatnonzero(~done);start=time.time();fallbacks=[];cropped=[]
    with ProcessPoolExecutor(max_workers=12) as pool:
        for start_idx in range(0,len(todo),32):
            indices=todo[start_idx:start_idx+32]
            records=list(pool.map(coords,[(molecules[i],a.mode) for i in indices]))
            batch=[]
            for idx,(atoms,positions,fallback) in zip(indices,records):
                # Official UniMolRepr uses max_atoms=256 and random atom cropping.
                # Per-molecule RNG makes resumed/parallel extraction reproducible.
                np.random.seed(int(hashlib.sha256(molecules[idx].encode()).hexdigest()[:8],16))
                batch.append((coords2unimol(atoms,positions,model.dictionary,256,remove_hs=False),0))
                if fallback:fallbacks.append(molecules[idx])
                if len(atoms)>256:cropped.append(molecules[idx])
            inputs,_=model.batch_collate_fn(batch);inputs={k:v.to(f'cuda:{a.gpu}') for k,v in inputs.items()}
            with torch.inference_mode():
                result=model(**inputs,return_repr=True,return_atomic_reprs=True)
                values=torch.stack([z.mean(0) for z in result['atomic_reprs']]).cpu().numpy()
            if not np.isfinite(values).all():raise ValueError('Nonfinite UniMol features')
            vectors[indices]=values;done[indices]=True;vectors.flush();np.save(donepath,done)
            report=dict(stage='feature_extraction',mode=a.mode,completed=int(done.sum()),total=len(done),seconds=time.time()-start)
            save_json(root/f'{name}.status.json',report);print(json.dumps(report),flush=True)
    reaction=np.stack([vectors[[lookup[s] for s in row]].sum(0) for row in parts]);np.save(root/f'{name}.npy',reaction)
    save_json(root/f'{name}.complete.json',dict(count=len(reaction),catalog_sha256=sha(root/'catalog.json'),
        sha256=sha(root/f'{name}.npy'),backbone='Uni-Mol v1 molecule_all_h',pooling='Atomic mean; sum over molecular occurrences',
        max_atoms=256,coordinate_fallbacks=fallbacks,cropped_molecules=cropped,
        weight_sha256=sha(assets/'mol_pre_all_h_220816.pt')))


if __name__=='__main__':main()
