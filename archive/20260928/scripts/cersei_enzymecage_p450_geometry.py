"""Frozen native EnzymeCAGE panel diagnostic on all 490 enzyme candidates."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/enzymecage_p450_reproduction_20260918'
OUT=ROOT/'runs/public_embedding_comparison_20260924/enzymecage_p450';OUT.mkdir(exist_ok=True)
NATIVE=ROOT.parent/'EnzymeCAGE';sys.path.insert(0,str(NATIVE))
from enzymecage.model import EnzymeCAGE,get_dis_pair

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

@torch.inference_mode()
def main():
    torch.set_num_threads(4);device='cuda:1'
    src=RUN/'scores/pretrained/seed_42/test_P450_epoch_19.csv';rec=json.loads(src.with_suffix('.receipt.json').read_text());assert sha(src)==rec['scores_sha256']
    with src.open() as f:rows=list(csv.DictReader(f))
    pids=sorted({r['UniprotID'] for r in rows});rxns=sorted({r['CANO_RXN_SMILES'] for r in rows});pi={p:i for i,p in enumerate(pids)};ri={r:i for i,r in enumerate(rxns)}
    seq={r['UniprotID']:r['sequence'] for r in rows};assert len(pids)==490 and len(rxns)==191 and len(rows)==len(pids)*len(rxns)
    scores=np.full((191,490),np.nan,np.float32);truth=[]
    for r in rows:
        i,j=ri[r['CANO_RXN_SMILES']],pi[r['UniprotID']];assert np.isnan(scores[i,j]);scores[i,j]=float(r['pred'])
        if int(r['Label']):truth.append((i,j))
    assert len(truth)==318 and np.isfinite(scores).all()
    np.save(OUT/'scores.npy',scores);np.save(OUT/'truth.npy',np.array(truth));
    cp=NATIVE/'checkpoints/pretrain/seed_42/epoch_19.pth'
    manifest=json.loads((RUN/'manifest.json').read_text());assert sha(cp)==manifest['configurations']['pretrained_seed_42']['checkpoint_sha256']
    model=EnzymeCAGE(use_esm=True,use_structure=True,use_drfp=True,use_prods_info=False,esm_model='ESM-C_600M',device=device).to(device).eval();model.load_state_dict(torch.load(cp,map_location=device),strict=True)
    gp=RUN/'data/feature/protein/gvp_feature/gvp_protein_feature.pt';ep=RUN/'data/feature/protein/ESM-C_600M/pocket_node_feature/esm_node_feature.pt'
    g=torch.load(gp,map_location='cpu');e=torch.load(ep,map_location='cpu');values=[];raw=[]
    for i,p in enumerate(pids):
        xyz,_,ns,nv,ei,es,ev=[x.to(device) for x in g[p]];esm=torch.as_tensor(e[p],device=device,dtype=torch.float32);assert len(esm)==len(xyz)
        v=model.gvp_encoder((ns,nv),ei,(es,ev));z=torch.cat([v,esm],-1)[None];mask=torch.ones((1,len(esm)),device=device,dtype=torch.bool)
        _,dist=get_dis_pair(xyz[None]);z,_=model.enzyme_attention(z,z,mask,attn_bias=1-dist/30,return_weights=True)
        values.append(z[0].mean(0).cpu().numpy());raw.append(esm.mean(0).cpu().numpy())
        if i%100==0:print('EnzymeCAGE native panel',i,len(pids),flush=True)
    np.savez(OUT/'embeddings.npz',enzyme=np.array(values),esm_pocket=np.array(raw))
    fasta=OUT/'proteins.fasta';fasta.write_text(''.join(f'>{p}\n{seq[p]}\n' for p in pids))
    command=[str(ROOT/'.deps/mmseqs/bin/mmseqs'),'easy-search',str(fasta),str(fasta),str(OUT/'homology.tsv'),str(OUT/'mmseqs_tmp'),
        '--min-seq-id','0.3','-c','0.8','--cov-mode','0','--max-seqs','10000','--threads','8','--format-output','query,target,fident,qcov,tcov,evalue','-e','0.001']
    if not (OUT/'homology.tsv').exists():
        with (OUT/'homology.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    dump(OUT/'metadata.json',dict(protein_ids=pids,reaction_smiles=rxns,protein_sequences=[seq[p] for p in pids],
        score_source=str(src),score_receipt=rec,checkpoint_sha256=sha(cp),native_manifest_sha256=sha(RUN/'manifest.json'),
        score='Released pretrained seed42 sigmoid pair score, no external prior, no fine-tuning',
        label='Listed positive associations; other pairs are unannotated, not measured inactive',
        protein_embedding='512D mean of valid pocket self-attention outputs before pair interaction',
        source_sha256={str(p):sha(p) for p in [gp,ep,NATIVE/'enzymecage/model.py']},homology_command=command))
    print('ENZYMECAGE P450 EXPORT COMPLETE',flush=True)

if __name__=='__main__':main()
