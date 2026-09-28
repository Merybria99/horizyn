"""Released CARE/CLEAN protein-only control on fixed diagnostic subsets."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from cersei_embedding_organization import ROOT,OUT as OLD,read,dump,sha
OUT=ROOT/'runs/public_embedding_comparison_20260924';ASSETS=OUT/'assets'
sys.path.insert(0,str(ASSETS/'fair_esm'))
import esm

@torch.inference_mode()
def main():
    torch.set_num_threads(4);device='cuda:0'
    seqs=[];where={};splits={};repairs=[]
    for split in ['time','enzyme_smi','reaction_smi','enzymemap']:
        meta=read(OLD/split/'metadata.json');sub=read(OLD/split/'subsets.json');indices=sub['protein_indices']
        ss=[meta['sequences'][meta['protein_ids'][i]] for i in indices]
        missing={meta['protein_ids'][i] for i,s in zip(indices,ss) if not s}
        if missing:
            assert split=='enzymemap'
            fasta=ROOT/'runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_catalog_v1/screening_proteins.fasta'
            source={};key=''
            for line in fasta.open():
                if line.startswith('>'):key=line[1:].strip()
                elif key in missing:source[key]=source.get(key,'')+line.strip()
            for p,s in source.items():assert p=='p_'+hashlib.sha256(s.encode()).hexdigest()[:24]
            assert set(source)==missing
            ss=[s or source[meta['protein_ids'][i]] for i,s in zip(indices,ss)]
            repairs=[dict(protein_id=p,length=len(s),sequence_sha256=hashlib.sha256(s.encode()).hexdigest()) for p,s in sorted(source.items())]
            dump(OUT/'sequence_metadata_repairs.json',dict(source=str(fasta),source_sha256=sha(fasta),
                explanation='Three legacy analysis metadata sequences were empty; recover the exact hash-identified sequences used by the fitted model. All three lack EC annotations and are ineligible for functional-neighborhood metrics.',proteins=repairs))
        assert all(ss)
        for s in ss:
            if s not in where:where[s]=len(seqs);seqs.append(s)
        splits[split]=(indices,[where[s] for s in ss],ss)
    trainpath=ROOT/'.deps/CARE/splits/task1/protein_train50.csv'
    with trainpath.open() as f:
        rr=list(csv.DictReader(f));print('Training columns',rr[0].keys(),flush=True)
    trainseq={r['Sequence'] for r in rr};trainids={r['Entry'] for r in rr}
    print('Unique sequences',len(seqs),flush=True)
    rawpath=OUT/'clean_esm1b_raw.npy'
    ordered_digest=hashlib.sha256('\n'.join(seqs).encode()).hexdigest();cache_receipt=OUT/'clean_esm1b_raw.receipt.json'
    if not (rawpath.exists() and cache_receipt.exists() and read(cache_receipt)['ordered_sequences_sha256']==ordered_digest):
        state=torch.load(ASSETS/'esm1b_t33_650M_UR50S.pt',map_location='cpu',weights_only=False)
        model,alphabet=esm.pretrained.load_model_and_alphabet_core('esm1b_t33_650M_UR50S',state,None)
        del state
        model.to(device).eval().requires_grad_(False);converter=alphabet.get_batch_converter(truncation_seq_length=1022)
        order=sorted(range(len(seqs)),key=lambda i:len(seqs[i]));emb=np.zeros((len(seqs),1280),np.float32);start=time.monotonic()
        for n in range(0,len(order),32):
            ii=order[n:n+32];_,_,tokens=converter([(str(i),seqs[i]) for i in ii]);tokens=tokens.to(device)
            hidden=model(tokens,repr_layers=[33],return_contacts=False)['representations'][33]
            for j,i in enumerate(ii):emb[i]=hidden[j,1:1+min(1022,len(seqs[i]))].mean(0).cpu().numpy()
            if n%256==0:print('ESM1b',n,len(order),round(time.monotonic()-start,1),flush=True)
        assert np.isfinite(emb).all()
        np.save(rawpath,emb);dump(cache_receipt,dict(ordered_sequences_sha256=ordered_digest,shape=list(emb.shape),sha256=sha(rawpath)))
        del model;torch.cuda.empty_cache()
    else:emb=np.load(rawpath)
    assert np.isfinite(emb).all() and len(emb)==len(seqs)
    spec=importlib.util.spec_from_file_location('released_clean_model',ASSETS/'clean_model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    ck=torch.load(ASSETS/'clean_care.pth',map_location='cpu',weights_only=False)
    print('CLEAN weights',[(k,tuple(v.shape)) for k,v in ck.items()],flush=True)
    model=mod.LayerNormNet(ck['fc1.weight'].shape[0],ck['fc3.weight'].shape[0],device,torch.float32).eval()
    model.load_state_dict(ck,strict=True);z=model(torch.tensor(emb,device=device)).cpu().numpy()
    for split,(indices,ids,ss) in splits.items():
        dest=OUT/split;dest.mkdir(exist_ok=True);meta=read(OLD/split/'metadata.json')
        overlap=np.array([s in trainseq for s in ss]);id_overlap=np.array([meta['protein_ids'][i] in trainids for i in indices])
        np.savez(dest/'clean_embeddings.npz',enzyme=z[ids],esm1b=emb[ids],bank=np.array(indices),training_sequence_overlap=overlap,training_id_overlap=id_overlap)
        dump(dest/'clean_receipt.json',dict(model='CLEAN CARE protein_train50 checkpoint',training='CARE EC-supervised; unmatched to CERSEI downstream training',
            scope='Frozen transfer protein-only control',proteins=len(ids),exact_training_sequence_overlap=int(overlap.sum()),exact_training_id_overlap=int(id_overlap.sum()),
            truncation_count=sum(len(s)>1022 for s in ss),pooling='ESM1b layer33 mean over residues, first1022 as native extract.py',
            checkpoint_sha256=sha(ASSETS/'clean_care.pth'),source_sha256=sha(ASSETS/'clean_model.py'),train_table_sha256=sha(trainpath),
            recovered_empty_metadata_sequences=len(repairs) if split=='enzymemap' else 0,
            native_distance='Euclidean; also cosine for common diagnostic',embeddings_sha256=sha(dest/'clean_embeddings.npz')))
    dump(OUT/'clean_complete.json',dict(unique_sequences=len(seqs),esm_checkpoint_sha256=sha(ASSETS/'esm1b_t33_650M_UR50S.pt'),complete=True))
    print('CLEAN COMPLETE',flush=True)

if __name__=='__main__':main()
