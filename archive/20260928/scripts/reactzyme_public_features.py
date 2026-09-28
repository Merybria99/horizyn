#!/usr/bin/env python3
"""Shared, resumable frozen features for matched ReactZyme competitors.

Feature extraction may see unlabelled test inputs; fitting never uses test edges.
The catalog retains distinct released IDs, including chemically equivalent IDs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/reactzyme_public_baselines_20260921'
SPLITS = ('reaction_smi', 'enzyme_smi', 'time')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'catalog.json').exists():
        raise FileExistsError('Catalog is immutable; use the existing catalog or a new output directory')
    proteins, reactions, tables, provenance = {}, {}, {}, {}
    for split in SPLITS:
        tables[split] = {}
        for part in ('train', 'validation', 'test'):
            p = ROOT / f'data/revised_protocols/reactzyme_paper/{split}/{part}_pairs.csv'
            provenance[str(p.relative_to(ROOT))] = sha(p)
            with p.open() as f:
                rows = list(csv.DictReader(f))
            tables[split][part] = [(r['reaction_id'], r['protein_id']) for r in rows]
            for r in rows:
                for target, key, val in ((proteins, 'protein_id', 'protein_sequence'),
                                         (reactions, 'reaction_id', 'reaction_smiles')):
                    if r[key] in target and target[r[key]] != r[val]:
                        raise ValueError(f'Conflicting {key}: {r[key]}')
                    target[r[key]] = r[val]
    pids, rids = sorted(proteins), sorted(reactions)
    pi, ri = {p:i for i,p in enumerate(pids)}, {r:i for i,r in enumerate(rids)}
    catalog = dict(protein_ids=pids, protein_sequences=[proteins[p] for p in pids],
                   reaction_ids=rids, reaction_smiles=[reactions[r] for r in rids])
    arrays, counts = {}, {}
    for split, parts in tables.items():
        counts[split] = {}
        assert not (set(parts['train']) & set(parts['validation']))
        for part, pairs in parts.items():
            x = np.asarray([(ri[r],pi[p]) for r,p in pairs], dtype=np.int64)
            arrays[f'{split}_{part}'] = x
            counts[split][part] = dict(pairs=len(x), reactions=len(np.unique(x[:,0])),
                                      proteins=len(np.unique(x[:,1])))
    np.savez(out / 'pairs.npz', **arrays)
    save_json(out / 'catalog.json', catalog)
    save_json(out / 'data_audit.json', dict(source='Existing V4 reactzyme_paper split, not v2',
        sources=provenance, counts=counts, proteins=len(pids), reactions=len(rids),
        catalog_sha256=sha(out/'catalog.json'), pairs_sha256=sha(out/'pairs.npz'),
        test_used_for_training=False))
    print(json.dumps(counts), flush=True)


def pool_existing(out):
    import h5py
    catalog = json.loads((out/'catalog.json').read_text())
    ids = catalog['protein_ids']; lookup = {p:i for i,p in enumerate(ids)}
    vectors = np.zeros((len(ids),1024), np.float32); done = np.zeros(len(ids), bool)
    sources = []
    parent = ROOT/'runs/generalization_20260919_2251'
    for folder in ['features','features_enzyme_smi','features_time',
                   'features_test_reaction_smi','features_test_enzyme_smi','features_test_time']:
        path = parent/folder/'protein_mean.h5'
        if not path.exists():
            continue
        with h5py.File(path) as f:
            src_ids = f['ids'].asstr()[:]
            complete = f['complete'][:] if 'complete' in f else np.ones(len(src_ids),bool)
            missing = [(j,lookup[p]) for j,p in enumerate(src_ids) if p in lookup and not done[lookup[p]] and complete[j]]
            for start in range(0,len(missing),2048):
                chunk = missing[start:start+2048]
                src, dst = np.array(chunk).T
                vectors[dst] = f['vectors'][src]
                done[dst] = True
        sources.append(dict(path=str(path),new_records=len(missing)))
        print(folder, 'covered',int(done.sum()),'/',len(done), flush=True)
    if not done.all():
        raise ValueError(f'Missing {int((~done).sum())} ProtT5 means; no zero-filled substitutions allowed')
    assert np.isfinite(vectors).all()
    np.save(out/'prott5.npy',vectors)
    save_json(out/'prott5.complete.json',dict(sources=sources, count=len(ids),
        feature='Previously extracted frozen ProtT5 mean, same as V4 global view',
        catalog_sha256=sha(out/'catalog.json'),sha256=sha(out/'prott5.npy')))


def extract_esm(out, gpu, shard, shards, model_path, name, token_budget, max_length, limit):
    import torch
    from transformers import EsmModel, EsmTokenizer
    torch.set_num_threads(4)
    catalog=json.loads((out/'catalog.json').read_text())
    seqs=catalog['protein_sequences']
    # Interleaved length-sorted records balance GPU work and minimize padding.
    indices=sorted(range(len(seqs)),key=lambda i:(min(len(seqs[i]),max_length),i))[shard::shards]
    if limit:
        indices=indices[:limit]
    feat=out/f'{name}_shard{shard}.npy'; status=out/f'{name}_shard{shard}.done.npy'
    vectors=np.lib.format.open_memmap(feat,mode='r+' if feat.exists() else 'w+',dtype=np.float32,shape=(len(indices),1280))
    done=np.load(status) if status.exists() else np.zeros(len(indices),bool)
    np.save(out/f'{name}_shard{shard}.indices.npy',np.array(indices))
    device=f'cuda:{gpu}'
    model=EsmModel.from_pretrained(model_path,add_pooling_layer=False,
        dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    tokenizer=EsmTokenizer.from_pretrained(model_path)
    todo=np.flatnonzero(~done).tolist(); started=time.time(); completed=0; position=0
    while position<len(todo):
        length=min(len(seqs[indices[todo[position]]]),max_length)
        batch_n=min(256,max(1,token_budget//max(length,1)))
        selected=todo[position:position+batch_n]
        # Bound by the longest record in this proposed batch.
        while len(selected)>1 and len(selected)*min(len(seqs[indices[selected[-1]]]),max_length)>token_budget:
            selected=selected[:max(1,len(selected)//2)]
        strings=[seqs[indices[i]][:max_length] for i in selected]
        try:
            inputs=tokenizer(strings,add_special_tokens=False,padding=True,return_tensors='pt').to(device)
            with torch.inference_mode():
                hidden=model(**inputs).last_hidden_state
                mask=inputs['attention_mask'].bool()
                pooled=(hidden.float()*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
            values=pooled.cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite protein features')
        except torch.cuda.OutOfMemoryError:
            if len(selected)==1:
                raise
            token_budget=max(length,token_budget//2)
            torch.cuda.empty_cache()
            continue
        vectors[selected]=values; done[selected]=True
        position+=len(selected);completed+=len(selected)
        vectors.flush();np.save(status,done)
        progress=dict(stage='feature_extraction',model=name,gpu=gpu,shard=shard,
            completed=int(done.sum()),total=len(done),seconds=time.time()-started,
            records_per_second=completed/max(time.time()-started,1),token_budget=token_budget,
            gpu_peak_gib=torch.cuda.max_memory_allocated(gpu)/2**30)
        save_json(out/f'{name}_shard{shard}.status.json',progress)
        print(json.dumps(progress),flush=True)
    save_json(out/f'{name}_shard{shard}.complete.json',dict(
        count=len(indices),model_path=str(model_path),max_length=max_length,
        special_tokens=False,dtype='bfloat16 inference; float32 masked mean',
        catalog_sha256=sha(out/'catalog.json'),feature_sha256=sha(feat)))


def merge(out,name,shards):
    n=len(json.loads((out/'catalog.json').read_text())['protein_ids'])
    dest=np.lib.format.open_memmap(out/f'{name}.npy',mode='w+',dtype=np.float32,shape=(n,1280))
    covered=np.zeros(n,bool)
    for s in range(shards):
        receipt=json.loads((out/f'{name}_shard{s}.complete.json').read_text())
        assert receipt['catalog_sha256']==sha(out/'catalog.json')
        idx=np.load(out/f'{name}_shard{s}.indices.npy')
        assert not covered[idx].any()
        dest[idx]=np.load(out/f'{name}_shard{s}.npy',mmap_mode='r');covered[idx]=True
    if not covered.all() or not np.isfinite(dest).all():
        raise ValueError('Incomplete feature coverage')
    dest.flush()
    save_json(out/f'{name}.complete.json',dict(count=n,shards=shards,sha256=sha(out/f'{name}.npy'),
        catalog_sha256=sha(out/'catalog.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','pool-prott5','esm','merge'])
    p.add_argument('--output',type=Path,default=RUN/'features')
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=4);p.add_argument('--name',default='esm2')
    p.add_argument('--model',default=str(ROOT.parent/'hf_cache/hub/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c'))
    p.add_argument('--token-budget',type=int,default=16384)
    p.add_argument('--max-length',type=int,default=5000)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.output)
    elif a.action=='pool-prott5':pool_existing(a.output)
    elif a.action=='merge':merge(a.output,a.name,a.shards)
    else:extract_esm(a.output,a.gpu,a.shard,a.shards,a.model,a.name,a.token_budget,a.max_length,a.limit)


if __name__=='__main__':main()
