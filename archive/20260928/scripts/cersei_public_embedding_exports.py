"""Evaluation-only exports of existing, validation-selected public baselines."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from cersei_embedding_organization import ROOT, OUT as OLD, read, dump, sha
from reactzyme_public_features import RUN

OUT=ROOT/'runs/public_embedding_comparison_20260924'

@torch.inference_mode()
def cersei(split,device):
    from cersei_embedding_organization import CROSS
    from generalization_clipzyme_f3_screen import model_from_checkpoint
    from generalization_multiview_calibration import components
    from generalization_reactzyme_architecture_phase2 import load_head,load_features
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    dest=OUT/split;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'cersei_embeddings.npz').exists():return
    base=CROSS/'shared_fusion03_beta5_b1024_v1'/split;run=base/'phase2_followup/epoch20'
    cp=base/'training/checkpoints/screen_selection/screen-epoch=19.ckpt'
    current=ROOT/'runs/cersei_horizyn_challenge_20260923/no_dictionary_validation_20260923/reference_shared_test'/split
    receipt=read(current/'summary.json');assert sha(cp)==receipt['checkpoint_sha256']
    assert sha(run/'training/step0100.pt')==receipt['residual_head_sha256']
    meta=read(OLD/split/'metadata.json')
    cat,be,br,*_=load_features(run/'test_features',device)
    assert cat['proteins']==meta['protein_ids'] and cat['reactions']==meta['reaction_ids']
    model,config=model_from_checkpoint(base/'configs/train.yaml',cp,device)
    g,f,scale,reconstructed,err=components(model,config,cat['proteins'],device)
    assert float((reconstructed-be).abs().max())<3e-6
    h=load_head(run/'training/step0100.pt',sha(run/'features/manifest.json'),device)
    b=F.normalize(g+2*scale*f,dim=-1)
    e=F.normalize(b+.5*h.scale*h.enzyme(b),dim=-1).cpu().numpy()
    r=F.normalize(br+.5*h.scale*h.reaction(br),dim=-1).cpu().numpy()
    score=np.load(current/'scores.npy',mmap_mode='r');idx=np.arange(0,len(r),max(1,len(r)//20))
    parity=float(np.max(np.abs(r[idx]@e.T-score[idx])));assert parity<3e-5,parity
    np.savez(dest/'cersei_embeddings.npz',enzyme=e,reaction=r)
    dump(dest/'cersei_receipt.json',dict(source=receipt,checkpoint=str(cp),parity_max_error=parity,
        embeddings_sha256=sha(dest/'cersei_embeddings.npz'),scores=str(current/'scores.npy')))
    print(split,'CURRENT CERSEI COMPLETE',parity,flush=True)

@torch.inference_mode()
def export(split,device,refresh=False):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    dest=OUT/split;dest.mkdir(parents=True,exist_ok=True)
    meta=read(OLD/split/'metadata.json');cat=read(RUN/'features/catalog.json')
    pi={k:i for i,k in enumerate(cat['protein_ids'])};ri={k:i for i,k in enumerate(cat['reaction_ids'])}
    pp=np.array([pi[k] for k in meta['protein_ids']]);rr=np.array([ri[k.removesuffix('_f')] for k in meta['reaction_ids']])
    root=RUN/'models'/f'creep_two_modality_{split}_seed42'
    if refresh or not (dest/'creep_embeddings.npz').exists():
        from reactzyme_public_creep import load_backbones
        protocol=read(root/'protocol.json');batch=protocol['batch'];attention=protocol.get('attention_kernel','eager')
        p,r=load_backbones(device,attention,'none');ph=torch.nn.Linear(1024,256).to(device);rh=torch.nn.Linear(256,256).to(device)
        state=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
        for key,mod in [('protein',p),('reaction',r),('ph',ph),('rh',rh)]:mod.load_state_dict(state[key]);mod.eval().requires_grad_(False)
        del state
        receipt=read(RUN/'features/creep_tokens.complete.json')
        def encode(ids,kind,backbone,head,pad):
            tokens=np.load(RUN/f'features/creep_{kind}_tokens.npy',mmap_mode='r');result=[];start=time.monotonic()
            for i in range(0,len(ids),batch):
                t=torch.tensor(tokens[ids[i:i+batch]],device=device,dtype=torch.long)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    h=backbone(input_ids=t,attention_mask=t.ne(pad)).last_hidden_state
                    z=head(h.mean(1) if kind=='protein' else h[:,0])
                result.append(F.normalize(z.float(),dim=-1).cpu().numpy())
                if i%1024==0: print(split,kind,i,len(ids),round(time.monotonic()-start,1),flush=True)
            return np.concatenate(result)
        native=np.load(root/'test_scores.npz');ep=np.searchsorted(native['protein_index'],pp);rp=np.searchsorted(native['reaction_index'],rr)
        assert np.array_equal(native['protein_index'][ep],pp) and np.array_equal(native['reaction_index'][rp],rr)
        e=encode(native['protein_index'],'protein',p,ph,receipt['protein_pad'])[ep]
        rx=encode(native['reaction_index'],'reaction',r,rh,receipt['reaction_pad'])[rp]
        scores=native['scores'][rp][:,ep];err=float(np.max(np.abs(rx@e.T-scores)))
        assert err<3e-5,err
        np.savez(dest/'creep_embeddings.npz',enzyme=e,reaction=rx)
        np.save(dest/'creep_scores.npy',scores)
        dump(dest/'creep_receipt.json',dict(checkpoint=str(root/'best.pt'),checkpoint_sha256=sha(root/'best.pt'),
            selection=read(root/'selection.json'),protocol=read(root/'protocol.json'),parity_max_error=err,
            inference_batch=batch,attention_kernel=attention,score_source='Original complete evaluation; native order, batch and attention kernel replayed',
            embeddings_sha256=sha(dest/'creep_embeddings.npz')))
        del p,r,ph,rh;torch.cuda.empty_cache()
    from enzymemap_public_horizyn import model_and_loss,encode
    root=RUN/'models'/f'horizyn_participant_set_{split}_seed42'
    model,_=model_and_loss(device);ck=torch.load(root/'best.pt',map_location=device,weights_only=False);model.load_state_dict(ck['model']);model.eval()
    e=encode(model.target_encoder,torch.tensor(np.load(RUN/'features/prott5.npy',mmap_mode='r')[pp],device=device)).cpu().numpy()
    rx=encode(model.query_encoder,torch.tensor(np.load(RUN/'features/horizyn_fp.npy')[rr],device=device)).cpu().numpy()
    z=np.load(root/'test_scores.npz');ep=np.searchsorted(z['protein_index'],pp);rp=np.searchsorted(z['reaction_index'],rr)
    assert np.array_equal(z['protein_index'][ep],pp) and np.array_equal(z['reaction_index'][rp],rr)
    scores=z['scores'][rp][:,ep];err=float(np.max(np.abs(rx@e.T-scores)));assert err<3e-5,err
    np.savez(dest/'horizyn_embeddings.npz',enzyme=e,reaction=rx);np.save(dest/'horizyn_scores.npy',scores)
    dump(dest/'horizyn_receipt.json',dict(checkpoint=str(root/'best.pt'),checkpoint_sha256=sha(root/'best.pt'),
        selection=read(root/'selection.json'),protocol=read(root/'protocol.json'),parity_max_error=err,
        embeddings_sha256=sha(dest/'horizyn_embeddings.npz')))
    print(split,'EXPORT COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--device',required=True);p.add_argument('--model',default='baselines');p.add_argument('--refresh',action='store_true');a=p.parse_args()
    if a.model=='cersei':cersei(a.split,a.device)
    else:export(a.split,a.device,a.refresh)
