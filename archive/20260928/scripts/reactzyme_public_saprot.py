#!/usr/bin/env python3
"""Released SaProt structural strings, with the upstream ESM fallback policy."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from reactzyme_public_features import ROOT,RUN,save_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=2);a=p.parse_args()
    import torch
    from transformers import EsmModel,EsmTokenizer
    from safetensors.torch import save_file
    torch.set_num_threads(4)
    root=RUN/'features';assets=RUN/'assets';model_dir=assets/'saprot'
    while not (assets/'saprot_seq.pt').exists():time.sleep(15)
    if not (model_dir/'model.safetensors').exists():
        state=torch.load(model_dir/'pytorch_model.bin',map_location='cpu',weights_only=True)
        save_file({k:v.clone().contiguous() for k,v in state.items()},str(model_dir/'model.safetensors'))
        del state
    catalog=json.loads((root/'catalog.json').read_text());seqs=catalog['protein_sequences']
    structural=torch.load(assets/'saprot_seq.pt',map_location='cpu',weights_only=False)
    by_sequence={}
    with (ROOT/'data/paper/reactzyme/raw/uniprot_rhea.tsv').open() as f:
        for row in csv.DictReader(f,delimiter='\t'):
            if row['Entry'] in structural:
                value=structural[row['Entry']]
                if not isinstance(value,str):raise TypeError('Unexpected SaProt release format')
                # Avoid mapping a structure with a different amino-acid sequence.
                if value[::2]==row['Sequence']:
                    by_sequence.setdefault(row['Sequence'],value)
    indices=sorted([i for i,s in enumerate(seqs) if s in by_sequence],key=lambda i:len(seqs[i]))
    path=root/'saprot.npy';status=root/'saprot.done.npy'
    vectors=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float32,shape=(len(seqs),1280))
    done=np.load(status) if status.exists() else np.zeros(len(seqs),bool)
    fallback=np.array([s not in by_sequence for s in seqs])
    vectors[fallback]=np.load(root/'esm2.npy',mmap_mode='r')[fallback];done[fallback]=True
    tokenizer=EsmTokenizer.from_pretrained(model_dir)
    model=EsmModel.from_pretrained(model_dir,add_pooling_layer=False,dtype=torch.bfloat16,attn_implementation='sdpa').to(f'cuda:{a.gpu}').eval()
    todo=[i for i in indices if not done[i]];pos=0;start=time.time()
    while pos<len(todo):
        count=min(128,max(1,32768//min(len(seqs[todo[pos]])+2,5002)))
        selected=todo[pos:pos+count]
        inputs=tokenizer([by_sequence[seqs[i]][:10000] for i in selected],padding=True,return_tensors='pt').to(f'cuda:{a.gpu}')
        with torch.inference_mode():
            h=model(**inputs).last_hidden_state.float();mask=inputs['attention_mask'].bool()
            mask[:,0]=False;mask.scatter_(1,(inputs['attention_mask'].sum(1)-1)[:,None],False)
            pooled=(h*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
        value=pooled.cpu().numpy()
        if not np.isfinite(value).all():raise ValueError('Nonfinite SaProt feature')
        vectors[selected]=value;done[selected]=True;pos+=len(selected)
        vectors.flush();np.save(status,done)
        save_json(root/'saprot.status.json',dict(stage='feature_extraction',completed=int(done.sum()),total=len(done),
            structural_done=pos,structural_total=len(todo),fallback_to_esm=int(fallback.sum()),seconds=time.time()-start))
        print(pos,len(todo),flush=True)
    assert done.all()
    save_json(root/'saprot.complete.json',dict(count=len(seqs),fallback_to_esm=int(fallback.sum()),
        policy='Released process_saprot.py uses ESM where structural sequence is unavailable; sequence identity checked',
        max_residues=5000,dtype='bfloat16 extraction, float32 masked mean excluding special tokens',
        catalog_sha256=sha(root/'catalog.json'),sha256=sha(path),structural_release_sha256=sha(assets/'saprot_seq.pt')))


if __name__=='__main__':main()
