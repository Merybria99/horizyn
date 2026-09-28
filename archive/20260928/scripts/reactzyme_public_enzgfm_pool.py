#!/usr/bin/env python3
"""Pool existing frozen official EnzGFM residue features without refitting."""
import argparse
import json
import time
import shutil
from pathlib import Path
import h5py
import numpy as np
from reactzyme_public_features import ROOT,RUN,save_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1);p.add_argument('--merge',action='store_true');p.add_argument('--local-cache',type=Path);a=p.parse_args()
    root=RUN/'features';cat=json.loads((root/'catalog.json').read_text());lookup={p:i for i,p in enumerate(cat['protein_ids'])}
    path=root/'enzgfm650.npy';vectors=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float32,shape=(len(lookup),2048))
    base_done=root/'enzgfm650.done.npy'
    donepath=root/f'enzgfm650.rank{a.rank}.done.npy' if a.world_size>1 else base_done
    done=np.load(donepath) if donepath.exists() else (np.load(base_done) if base_done.exists() else np.zeros(len(lookup),bool))
    sources=[ROOT/'data/standardized/retrieval_training_source_collapse/test/horizyn_reactzyme_shared_candidates/reaction_smi_train_validation_proteins_enzgfm_650m_residue.h5',
             ROOT/'runs/reactzyme_reaction_smi_enzgfm650m_hybrid/intermediate_tests/epoch10/inputs/test_proteins_enzgfm_650m_residue.h5']
    start=time.time();receipts=[]
    if a.merge:
        for rank in range(a.world_size):
            meta=json.loads((root/f'enzgfm650.rank{rank}.complete.json').read_text())
            assert meta['catalog_sha256']==sha(root/'catalog.json')
            done |= np.load(root/f'enzgfm650.rank{rank}.done.npy')
            if not receipts:receipts=meta['sources']
    seen_source_ids=set()
    for source in ([] if a.merge else sources):
        read_source=source
        if a.local_cache:
            a.local_cache.mkdir(parents=True,exist_ok=True)
            read_source=a.local_cache/source.name
            if not read_source.exists() or read_source.stat().st_size!=source.stat().st_size:
                if a.world_size!=1:raise ValueError('Pre-stage local cache with one process before parallel pooling')
                print('Staging sequential local copy:',source,flush=True)
                partial=read_source.with_suffix('.partial');shutil.copyfile(source,partial);partial.replace(read_source)
        with h5py.File(read_source) as f:
            ids=f['ids'].asstr()[:];offset=f['offsets'][:]
            receipts.append(dict(path=str(source),bytes=source.stat().st_size,
                attributes={k:(v.item() if isinstance(v,np.generic) else v) for k,v in f.attrs.items()}))
            for pos in range(a.rank*256,len(ids),a.world_size*256):
                stop=min(pos+256,len(ids));target=[lookup.get(s,-1) if s not in seen_source_ids else -1 for s in ids[pos:stop]]
                if all(i<0 or done[i] for i in target):continue
                lengths=np.diff(offset[pos:stop+1]);assert np.all(lengths>0)
                values=f['vectors'][offset[pos]:offset[stop]]
                means=np.add.reduceat(values,offset[pos:stop]-offset[pos],axis=0,dtype=np.float32)/lengths[:,None]
                for i,row in zip(target,means):
                    if i>=0 and not done[i]:vectors[i]=row;done[i]=True
                vectors.flush();np.save(donepath,done)
                progress=dict(completed=int(done.sum()),total=len(done),seconds=time.time()-start)
                save_json(root/f'enzgfm650.rank{a.rank}.status.json',progress)
                if pos%4096==0:print(json.dumps(progress),flush=True)
            seen_source_ids.update(ids)
    if a.world_size>1 and not a.merge:
        save_json(root/f'enzgfm650.rank{a.rank}.complete.json',dict(sources=receipts,catalog_sha256=sha(root/'catalog.json')))
        return
    if not done.all() or not np.isfinite(vectors).all():raise ValueError('Incomplete EnzGFM coverage')
    save_json(root/'enzgfm650.complete.json',dict(count=len(done),catalog_sha256=sha(root/'catalog.json'),sha256=sha(path),sources=receipts,
        feature='Mean of frozen EnzGFM-650M residue vectors; excludes special tokens; max1022 ends_center from existing documented cache',
        exact_paper_feature_preprocessing=False,note='Backbone control: published extraction averages special tokens too and defaults to max1000. Do not label as exact EnzGFM paper-score reproduction.'))


if __name__=='__main__':main()
