#!/usr/bin/env python3
"""Original Horizyn on the exact EnzymeMap split and full screening libraries.

Native directed RDKit+/DRFP inputs, native towers/MLNCE and SOTA optimizer.
Validation BEDROC85 alone selects a checkpoint; test labels are loaded only
after selection has been saved. Existing CERSEI assets are read-only inputs.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT/'data/external/cyp_specificity_2026/horizyn_official'
CROSS=ROOT/'runs/generalization_20260919_2251/cross_paper_retraining'
CAT=CROSS/'clipzyme_f3_catalog_v1'
MANIFEST=CROSS/'clipzyme_manifests_v2/manifest.json'
SCREEN=CROSS/'clipzyme_screening_evaluation_protocol_v2'
MEANS=CROSS/'clipzyme_phase2_enzymemap_v1/screen_protein_mean.h5'
METRICS=['bedroc85','bedroc20','ef0.05','ef0.1']
sys.path.insert(0,str(NATIVE))
from generalization_clipzyme_screening_evaluate import evaluate_query

def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.partial')
    temp.write_text(json.dumps(v,indent=2)+'\n');temp.replace(p)
def log(**v):print(json.dumps(dict(utc=now(),**v)),flush=True)
def load_module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def save_checkpoint(p,v):
    temp=p.with_suffix('.partial.pt');torch.save(v,temp);temp.replace(p)

@contextmanager
def gpu_lock(device):
    import os
    logical=int(device.split(':')[-1]);visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    physical=visible.split(',')[logical] if visible else str(logical)
    with (Path('/tmp')/f'enzymediscovery_screen_export_gpu_{physical}.lock').open('a+') as f:
        log(waiting_for_gpu=physical);fcntl.flock(f,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def fingerprint_init(path,settings):
    global FP_RD,FP_DR
    torch.set_num_threads(1)
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.warning')
    from horizyn.datasets.csv import CSVDataset
    from horizyn.datasets.fingerprints.rdkit_plus import RDKitPlusFingerprintDataset
    from horizyn.datasets.fingerprints.drfp import DRFPFingerprintDataset
    data=CSVDataset(file_path=path,key_column='reaction_id',columns=['reaction_smiles'])
    FP_RD=RDKitPlusFingerprintDataset(data,vec_dim=1024,**settings)
    FP_DR=DRFPFingerprintDataset(data,vec_dim=1024,**settings)

def fingerprint(key):
    v=torch.cat([FP_RD[key],FP_DR[key]]).numpy()
    if v.shape!=(2048,) or not np.isfinite(v).all():raise ValueError(f'Invalid fingerprint {key}')
    return v

def prepare(out,workers):
    if (out/'features.complete.json').exists():return
    out.mkdir(parents=True,exist_ok=True)
    native_config=yaml.safe_load((NATIVE/'configs/sota.yaml').read_text())
    setting={k:native_config['data'][k] for k in ['standardize_hypervalent','standardize_remove_hs','standardize_kekulize','standardize_uncharge','standardize_metals']}
    setting['standardize']=native_config['data']['standardize_reactions']
    release=read(MANIFEST);screen=read(SCREEN/'receipt.json');mean_receipt=read(MEANS.with_suffix('.receipt.json'))
    inputs=[MANIFEST,MEANS,MEANS.with_suffix('.receipt.json'),CAT/'screening_candidate_map.csv',
            CAT/'train_pairs.csv',CAT/'train_rxns.csv',CAT/'validation_pairs.csv',CAT/'validation_rxns.csv',
            SCREEN/'query_inputs.csv',SCREEN/'receipt.json',SCREEN/'train_uniprot_ids.txt']
    for split in ['train','dev']:
        p=Path(release['associations'][split]['path']);assert sha(p)==release['associations'][split]['sha256'];inputs.append(p)
    assert sha(MEANS)==mean_receipt['output_sha256']
    assert sha(CAT/'screening_candidate_map.csv')==screen['candidate_map_sha256']==mean_receipt['candidate_map_sha256']
    assert sha(SCREEN/'query_inputs.csv')==screen['query_inputs_sha256']
    native_files=['configs/sota.yaml','horizyn/model.py','horizyn/losses.py','horizyn/datasets/fingerprints/base.py',
                  'horizyn/datasets/fingerprints/rdkit_plus.py','horizyn/datasets/fingerprints/drfp.py','horizyn/chemistry/standardizer.py']
    protocol=dict(created_utc=now(),method='Original Horizyn, native directed reaction features, EnzymeMap retraining',seed=42,
        architecture=native_config['model'],optimizer=dict(name='AdamW',lr=1e-4,weight_decay=.01),epochs=100,batch_size=16384,
        loss=dict(name='FullBatchMLNCELoss',beta=10,learn_beta=False),validation_epochs=list(range(10,101,10)),
        selection='Maximum full-library validation BEDROC85; earliest epoch wins exact ties; full 100 epochs retained.',
        protein_features='Frozen cached ProtT5 residue means, all stored residues averaged in FP32, no learned pooling or supervised feature adaptation.',
        reaction_features=dict(rdkit_plus_dim=1024,drfp_dim=1024,physical_reactant_product_sides=True,standardizer=setting),
        precision='BF16 tower training, FP32 MLNCE, full FP32 inference; TF32 disabled.',
        data=dict(train_association_rows=34427,train_unique_sequence_edges=34180,train_unique_reactions=12603,
                  validation_queries=2652,validation_id_excluded_queries=2216,test_queries=1521,test_id_excluded_queries=1337,
                  full_candidates=261907,id_excluded_candidates=252113,unique_sequence_vectors=222985),
        training_policy='All original train association rows preserved after accession-to-sequence mapping; negatives from current training batch only.',
        validation_policy='Only dev reaction strings absent from training; same full-library validation labels and candidate pools as CERSEI.',
        test_policy='Write validation selection and selected-checkpoint hash before loading any test labels. Evaluate both libraries once; no tuning on test.',
        model_weights_initialized_afresh=True,test_used_for_training_or_selection=False,
        inputs={str(p):sha(p) for p in inputs},native_sources={name:sha(NATIVE/name) for name in native_files},
        code_sha256=sha(__file__))
    if (out/'protocol.json').exists():
        old=read(out/'protocol.json');protocol['created_utc']=old['created_utc'];assert protocol==old,'Protocol changed on resume'
    else:
        write(out/'protocol.json',protocol);shutil.copy2(__file__,out/'evaluation_source.py')
    train_rxns=rows(CAT/'train_rxns.csv');dev_rxns=rows(CAT/'validation_rxns.csv');test_rxns=rows(SCREEN/'query_inputs.csv')
    reaction={}
    for row in train_rxns+dev_rxns+test_rxns:
        rid=row['reaction_id'];smiles=row.get('reaction_smiles',row.get('reaction'))
        assert rid=='r_'+hashlib.sha256(smiles.encode()).hexdigest()[:24]
        if rid in reaction:assert reaction[rid]==smiles
        reaction[rid]=smiles
    keys=list(reaction);rxn_index={k:i for i,k in enumerate(keys)}
    with (out/'reaction_inputs.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['reaction_id','reaction_smiles']);w.writerows(reaction.items())
    with h5py.File(MEANS) as f:
        protein_ids=f['ids'].asstr()[:].tolist();assert f['complete'][:].all()
        assert len(protein_ids)==222985 and len(set(protein_ids))==222985
    protein_index={k:i for i,k in enumerate(protein_ids)}
    candidates=rows(CAT/'screening_candidate_map.csv');candidate_ids=[r['uniprot_id'] for r in candidates]
    assert len(set(candidate_ids))==261907
    train=rows(CAT/'train_pairs.csv');rawtrain=rows(release['associations']['train']['path'])
    assert len(train)==len(rawtrain)==34427
    for pair,original in zip(train,rawtrain):
        assert pair['reaction_id']=='r_'+hashlib.sha256(original['reaction'].encode()).hexdigest()[:24]
        if original['sequence']:assert pair['protein_id']=='p_'+hashlib.sha256(original['sequence'].encode()).hexdigest()[:24]
    train_edges=np.array([[rxn_index[r['reaction_id']],protein_index[r['protein_id']]] for r in train],dtype=np.int64)
    assert len(np.unique(train_edges,axis=0))==34180
    train_reactions={r['reaction'] for r in rawtrain};train_ids={r['protein_id'] for r in rawtrain}
    candidate_index={k:i for i,k in enumerate(candidate_ids)}
    validation_truth=defaultdict(set)
    for row in rows(release['associations']['dev']['path']):
        if row['reaction'] not in train_reactions:validation_truth['r_'+hashlib.sha256(row['reaction'].encode()).hexdigest()[:24]].add(row['protein_id'])
    valid_keys=[r['reaction_id'] for r in dev_rxns]
    positive_indices=[[candidate_index[k] for k in sorted(validation_truth[k]) if k in candidate_index] for k in valid_keys]
    kept=np.array([i for i,k in enumerate(candidate_ids) if k not in train_ids],dtype=np.int32)
    kept_set=set(kept)
    assert len(kept)==252113
    assert sum(bool(p) for p in positive_indices)==2652
    assert sum(bool(set(p)&kept_set) for p in positive_indices)==2216
    catalog=dict(reaction_ids=keys,protein_ids=protein_ids,candidate_ids=candidate_ids,
        validation_ids=valid_keys,validation_positive_indices=positive_indices,test_ids=[r['reaction_id'] for r in test_rxns])
    write(out/'catalog.json',catalog)
    np.savez(out/'axes.npz',train=train_edges,expanded=np.array([protein_index[r['protein_id']] for r in candidates],np.int64),kept=kept,
             validation=np.array([rxn_index[k] for k in valid_keys]),test=np.array([rxn_index[r['reaction_id']] for r in test_rxns]))
    log(stage='fingerprints',reactions=len(keys),workers=workers)
    values=np.empty((len(keys),2048),np.float32);started=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=fingerprint_init,initargs=(str(out/'reaction_inputs.csv'),setting)) as pool:
        for i,v in enumerate(pool.map(fingerprint,keys,chunksize=32)):
            values[i]=v
            if (i+1)%500==0:log(stage='fingerprints',completed=i+1,total=len(keys),seconds=time.monotonic()-started)
    np.save(out/'reaction_fingerprints.npy',values)
    # Independently regenerate fixed rows through the same public classes in
    # the parent process to catch multiprocessing/order/serialization errors.
    fingerprint_init(str(out/'reaction_inputs.csv'),setting)
    for idx in np.linspace(0,len(keys)-1,24,dtype=int):assert np.array_equal(values[idx],fingerprint(keys[idx]))
    write(out/'features.complete.json',dict(completed_utc=now(),reactions=len(keys),native_feature_parity_rows=24,
        sha256={n:sha(out/n) for n in ['catalog.json','axes.npz','reaction_fingerprints.npy','reaction_inputs.csv']},protocol_sha256=sha(out/'protocol.json')))
    log(stage='features_complete',seconds=time.monotonic()-started)

def model_and_loss(device):
    m=load_module('native_horizyn_model',NATIVE/'horizyn/model.py');l=load_module('native_horizyn_loss',NATIVE/'horizyn/losses.py')
    model=m.DualContrastiveModel(query_encoder_kwargs=dict(input_dim=2048,output_dim=512,num_layers=2,widths=[4096,4096],normalise_output=True),
                                target_encoder_kwargs=dict(input_dim=1024,output_dim=512,num_layers=2,widths=[4096,4096],normalise_output=True)).to(device)
    return model,l.FullBatchMLNCELoss(beta=10.,learn_beta=False).to(device)

@torch.inference_mode()
def encode(tower,x):return torch.cat([tower(z) for z in x.split(4096)])

@torch.inference_mode()
def validation(model,rxn,protein,axes,cat,out,epoch,checkpoint_hash):
    started=time.monotonic();model.eval();er=encode(model.target_encoder,protein);rr=encode(model.query_encoder,rxn[axes['validation']])
    positives=[np.array(p,dtype=np.int64) for p in cat['validation_positive_indices']];records=[]
    with ThreadPoolExecutor(max_workers=12) as pool:
        for start in range(0,len(rr),64):
            scores=(rr[start:start+64]@er.T).cpu().numpy()[:,axes['expanded']]
            records.extend(r for r in pool.map(evaluate_query,[(start+i,cat['validation_ids'][start+i],v,positives[start+i],axes['kept']) for i,v in enumerate(scores)]) if r is not None)
    summary={}
    for table,count in [('table1',261907),('table2',252113)]:
        selected=[r[table] for r in records if r[table] is not None]
        summary[table]=dict(queries=len(selected),candidate_ids=count,**{m:float(np.mean([r[m] for r in selected])) for m in METRICS})
    assert summary['table1']['queries']==2652 and summary['table2']['queries']==2216
    dest=out/'validation'/f'epoch{epoch:03d}';dest.mkdir(parents=True,exist_ok=True)
    (dest/'per_query.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    result=dict(epoch=epoch,summary=summary,selection_value=summary['table1']['bedroc85'],checkpoint_sha256=checkpoint_hash,
        validation_only=True,test_labels_read=False,elapsed_seconds=time.monotonic()-started,per_query_sha256=sha(dest/'per_query.jsonl'))
    write(dest/'summary.json',result);log(stage='validation',**result)
    return result

def train_and_test(out,device):
    if (out/'complete.json').exists():return
    torch.set_num_threads(4);torch.manual_seed(42);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    protocol=read(out/'protocol.json');assert sha(__file__)==protocol['code_sha256']
    for name,digest in protocol['native_sources'].items():assert sha(NATIVE/name)==digest
    features=read(out/'features.complete.json');assert features['protocol_sha256']==sha(out/'protocol.json')
    for name,digest in features['sha256'].items():assert sha(out/name)==digest
    cat=read(out/'catalog.json');axes=dict(np.load(out/'axes.npz'))
    rxn=torch.from_numpy(np.load(out/'reaction_fingerprints.npy')).to(device)
    with h5py.File(MEANS) as f:
        assert cat['protein_ids']==f['ids'].asstr()[:].tolist();protein=torch.from_numpy(f['vectors'][:]).to(device)
    model,criterion=model_and_loss(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    pairs=torch.from_numpy(axes['train']).to(device);start_epoch=1;best=-float('inf');best_epoch=None
    (out/'checkpoints').mkdir(exist_ok=True)
    if (out/'last.pt').exists():
        last=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert last['protocol_sha256']==sha(out/'protocol.json')
        model.load_state_dict(last['model']);opt.load_state_dict(last['optimizer']);start_epoch=last['epoch']+1;best=last['best_value'];best_epoch=last['best_epoch']
        torch.set_rng_state(last['cpu_rng']);torch.cuda.set_rng_state(last['cuda_rng'],device)
        log(stage='resume',epoch=start_epoch,best_epoch=best_epoch)
    for epoch in range(start_epoch,101):
        begin=time.monotonic();model.train();total=0.;steps=0
        for idx in torch.randperm(len(pairs),device=device).split(16384):
            edges=pairs[idx];r,ri=torch.unique(edges[:,0],return_inverse=True);p,pi=torch.unique(edges[:,1],return_inverse=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):zr,zp=model(rxn[r],protein[p])
            distances=1-zr.float()@zp.float().T;loss=criterion(distances,ri,pi)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();opt.step();total+=float(loss.detach())*len(idx);steps+=1
        record=dict(epoch=epoch,loss=total/len(pairs),updates=steps,train_seconds=time.monotonic()-begin)
        if epoch in protocol['validation_epochs']:
            cp=out/'checkpoints'/f'epoch{epoch:03d}.pt'
            save_checkpoint(cp,dict(model=model.state_dict(),epoch=epoch,protocol_sha256=sha(out/'protocol.json')))
            val=validation(model,rxn,protein,axes,cat,out,epoch,sha(cp));record['validation']=val['summary']
            if val['selection_value']>best:best=val['selection_value'];best_epoch=epoch
            save_checkpoint(out/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best_epoch=best_epoch,best_value=best,
                 cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device),protocol_sha256=sha(out/'protocol.json')))
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        write(out/'status.json',dict(stage='training',utc=now(),best_epoch=best_epoch,best_validation_bedroc85=best if best_epoch else None,**record))
        log(stage='training',**record)
    cp=out/'checkpoints'/f'epoch{best_epoch:03d}.pt'
    selection=dict(selected_epoch=best_epoch,validation_bedroc85=best,checkpoint=str(cp.resolve()),checkpoint_sha256=sha(cp),
        selection_metric='Full-library validation BEDROC85',test_used=False,selection_saved_utc=now(),protocol_sha256=sha(out/'protocol.json'))
    if (out/'selection.json').exists():
        previous=read(out/'selection.json');assert previous['checkpoint_sha256']==selection['checkpoint_sha256'];selection=previous
    else:write(out/'selection.json',selection)
    selected=torch.load(cp,map_location=device,weights_only=False);model.load_state_dict(selected['model']);model.eval()
    log(stage='test_score_export',selected_epoch=best_epoch,validation_bedroc85=best)
    dest=out/'test';dest.mkdir(exist_ok=True)
    if not (dest/'score_receipt.json').exists():
        with torch.inference_mode():
            er=encode(model.target_encoder,protein);rr=encode(model.query_encoder,rxn[axes['test']])
            scores=np.lib.format.open_memmap(dest/'scores.npy',mode='w+',dtype=np.float32,shape=(1521,261907))
            for start in range(0,len(rr),64):scores[start:start+64]=(rr[start:start+64]@er.T).cpu().numpy()[:,axes['expanded']]
            scores.flush();del scores
            np.savez(dest/'embeddings.npz',reactions=rr.cpu().numpy(),unique_proteins=er.cpu().numpy())
        (dest/'query_ids.txt').write_text('\n'.join(cat['test_ids'])+'\n');(dest/'candidate_ids.txt').write_text('\n'.join(cat['candidate_ids'])+'\n')
        write(dest/'score_receipt.json',dict(created_utc=now(),checkpoint_sha256=sha(cp),selection_sha256=sha(out/'selection.json'),
             shape=[1521,261907],scores_sha256=sha(dest/'scores.npy'),test_labels_read=False,feature_receipt_sha256=sha(out/'features.complete.json')))
    # This separate official evaluator is the first process in this campaign
    # allowed to read screening test labels.
    if not (dest/'evaluation/summary.json').exists():
        subprocess.run([sys.executable,str(ROOT/'scripts/generalization_clipzyme_screening_evaluate.py'),
            '--scores',str(dest/'scores.npy'),'--query-ids',str(dest/'query_ids.txt'),'--candidate-ids',str(dest/'candidate_ids.txt'),
            '--protocol',str(SCREEN),'--output',str(dest/'evaluation'),'--metric-workers','12'],check=True)
    result=read(dest/'evaluation/summary.json');write(out/'complete.json',dict(completed_utc=now(),method=protocol['method'],selection=selection,
        summary=result['summary'],test_evaluation_sha256=sha(dest/'evaluation/summary.json'),protocol_sha256=sha(out/'protocol.json')))
    write(out/'status.json',dict(stage='complete',utc=now(),selected_epoch=best_epoch,summary=result['summary']))
    log(stage='complete',summary=result['summary'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--feature-workers',type=int,default=6);a=p.parse_args();out=a.output.resolve()
    prepare(out,a.feature_workers)
    with gpu_lock(a.device):train_and_test(out,a.device)

if __name__=='__main__':main()
