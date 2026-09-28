"""Released EnzymeCAGE independent pocket-feature diagnostic, not pair scoring."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/public_embedding_comparison_20260924'
OLD=ROOT/'runs/cersei_embedding_organization_20260923_v1';NATIVE=ROOT.parent/'EnzymeCAGE'
sys.path.insert(0,str(NATIVE))
from enzymecage.model import EnzymeCAGE,get_dis_pair

def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

@torch.inference_mode()
def main():
    torch.set_num_threads(4);device='cuda:1'
    needed={};metas={}
    for split in ['time','enzyme_smi','reaction_smi','enzymemap']:
        m=read(OLD/split/'metadata.json');metas[split]=m
        for i in m['neighborhood_indices']:
            s=m['sequences'][m['protein_ids'][i]];needed.setdefault(s,[]).append((split,i))
    # Inventory is determined only by available inputs and exact sequence identity.
    case_root=ROOT/'runs/cyp_external_v1/assets/enzymecage_data/dataset/case-study/liz'
    paths=sorted(case_root.glob('*/r_*/liz_p450.csv'))
    jobs=[];chosen={};missing=[]
    for p in paths:
        with p.open() as f:rows=list(csv.DictReader(f))
        rows=[r for r in rows if r.get('sequence') in needed and r['sequence'] not in chosen]
        if not rows:continue
        base=p.parent/'feature/protein';gv=base/'gvp_feature/gvp_protein_feature.pt';es=base/'ESM-C_600M/pocket_node_feature/esm_node_feature.pt'
        if not gv.exists() or not es.exists():continue
        g=torch.load(gv,map_location='cpu');e=torch.load(es,map_location='cpu');items=[]
        for r in rows:
            s=r['sequence'];uid=r.get('uniprotID',r.get('UniprotID',r.get('enzyme','')))
            if not uid:uid=r.get('UniProt_ID',r.get('uniprot_id',''))
            if uid not in g or uid not in e or len(g[uid][0])!=len(e[uid]):missing.append((str(p),uid));continue
            if s in chosen:continue
            chosen[s]=len(chosen);items.append((s,uid))
        if items:jobs.append((p,gv,es,items))
        del g,e
        print('inventory',p.parent.name,'selected',len(chosen),flush=True)
    if not chosen:raise RuntimeError('No exact-sequence matches: inspect source columns before proceeding')
    cp=NATIVE/'checkpoints/pretrain/seed_42/epoch_19.pth'
    model=EnzymeCAGE(use_esm=True,use_structure=True,use_drfp=True,use_prods_info=False,esm_model='ESM-C_600M',device=device).to(device).eval()
    model.load_state_dict(torch.load(cp,map_location=device),strict=True)
    values=np.zeros((len(chosen),512),np.float32);raw=np.zeros((len(chosen),1152),np.float32);sources=[]
    for p,gp,ep,items in jobs:
        g=torch.load(gp,map_location='cpu');e=torch.load(ep,map_location='cpu')
        for s,uid in items:
            xyz,seq,ns,nv,ei,es,ev=[x.to(device) for x in g[uid]]
            esm=torch.as_tensor(e[uid],device=device,dtype=torch.float32);gvp=model.gvp_encoder((ns,nv),ei,(es,ev))
            z=torch.cat([gvp,esm],dim=-1)[None];mask=torch.ones((1,len(esm)),device=device,dtype=torch.bool)
            _,distance=get_dis_pair(xyz[None]);z,_=model.enzyme_attention(z,z,mask,attn_bias=1-distance/30,return_weights=True)
            values[chosen[s]]=z[0].mean(0).cpu().numpy();raw[chosen[s]]=esm.mean(0).cpu().numpy()
        sources.append(dict(csv=str(p),csv_sha256=sha(p),gvp=str(gp),gvp_sha256=sha(gp),esm=str(ep),esm_sha256=sha(ep),proteins=len(items)))
        print('encoded',p.parent.name,len(items),flush=True);del g,e
    with (NATIVE/'dataset/training/train.csv').open() as f:train=list(csv.DictReader(f))
    trainseq={r['sequence'] for r in train}
    for split,m in metas.items():
        bank=np.array([i for i in m['neighborhood_indices'] if m['sequences'][m['protein_ids'][i]] in chosen])
        seq=[m['sequences'][m['protein_ids'][i]] for i in bank];ii=[chosen[s] for s in seq]
        dest=OUT/split;dest.mkdir(exist_ok=True)
        overlap=np.array([s in trainseq for s in seq],dtype=bool)
        np.savez(dest/'enzymecage_embeddings.npz',enzyme=values[ii],esm_pocket=raw[ii],bank=bank,training_sequence_overlap=overlap)
        dump(dest/'enzymecage_receipt.json',dict(proteins=len(bank),available_bank=len(m['neighborhood_indices']),exact_training_sequence_overlap=int(overlap.sum()),
            diagnostic='Mean of valid residues after released GVP+ESMC pocket self-attention, before enzyme-molecule interaction',
            checkpoint=str(cp),checkpoint_sha256=sha(cp),training='Released EnzymeCAGE native training; not matched to CERSEI',
            source='Existing released case-study pocket features; exact full-sequence matches only; coverage-restricted exploratory control',
            embeddings_sha256=sha(dest/'enzymecage_embeddings.npz')))
    dump(OUT/'enzymecage_complete.json',dict(complete=True,unique_sequences=len(chosen),sources=sources,missing_inputs=missing,
        training_sha256=sha(NATIVE/'dataset/training/train.csv'),checkpoint_sha256=sha(cp),model_source_sha256=sha(NATIVE/'enzymecage/model.py')))
    print('ENZYMECAGE COMPLETE',flush=True)

if __name__=='__main__':main()
