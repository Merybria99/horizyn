#!/usr/bin/env python3
"""CREEP's supported two-modality objective on exact ReactZyme associations.

Preserves ProtT5/rxnfp fine-tuning, 256-dimensional linear heads, native padded
mean/CLS pooling, and symmetric EBM-NCE. Replaces EC Cartesian-product mining
with the benchmark's actual training edges, avoiding additional positive labels.
"""
import argparse
import ast
import json
import re
import time
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from reactzyme_public_features import ROOT,RUN,SPLITS,save_json,sha
from generalization_metrics import evaluate_scores


def native_loss():
    p=ROOT/'.deps/CARE/task2_baselines/CREEP/step_01_train_CREEP.py'
    tree=ast.parse(p.read_text());scope=dict(torch=torch,nn=torch.nn,F=F)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('cycle_index','do_CL')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),scope)
    return scope['do_CL'],p


def tokenize(root):
    from transformers import T5Tokenizer
    cat=json.loads((root/'catalog.json').read_text())
    t5=ROOT.parent/'hf_cache/hub/models--Rostlab--prot_t5_xl_half_uniref50-enc/snapshots/94a6abc029ae13029317b140b7424e012bf8dfbf'
    tok=T5Tokenizer.from_pretrained(t5,do_lower_case=False)
    ids=np.lib.format.open_memmap(root/'creep_protein_tokens.npy',mode='w+',dtype=np.int32,shape=(len(cat['protein_ids']),512))
    for i in range(0,len(ids),512):
        seqs=[' '.join(s) for s in cat['protein_sequences'][i:i+512]]
        batch=tok(seqs,truncation=True,max_length=512,padding='max_length',return_tensors='np')
        ids[i:i+len(seqs)]=batch['input_ids']
        if i%8192==0:print('protein tokens',i,len(ids),flush=True)
    ids.flush()
    native=ROOT/'.deps/CARE/CREEP/data/pretrained_rxnfp'
    vocab={s:i for i,s in enumerate((native/'vocab.txt').read_text().splitlines())}
    token_source=ROOT/'.deps/CARE/CREEP/CREEP/utils/tokenization.py'
    scope={};tree=ast.parse(token_source.read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SMI_REGEX_PATTERN' for t in node.targets):
            scope['pattern']=ast.literal_eval(node.value)
    regex=re.compile(scope['pattern']);rxns=[];unknown=0
    for s in cat['reaction_smiles']:
        tokens=regex.findall(s.replace('*','C')+'>>')[:510]
        unknown+=sum(t not in vocab for t in tokens)
        tokens=[vocab['[CLS]']]+[vocab.get(t,vocab['[UNK]']) for t in tokens]+[vocab['[SEP]']]
        rxns.append(tokens+[vocab['[PAD]']]*(512-len(tokens)))
    np.save(root/'creep_reaction_tokens.npy',np.asarray(rxns,dtype=np.int32))
    save_json(root/'creep_tokens.complete.json',dict(protein_pad=tok.pad_token_id,reaction_pad=vocab['[PAD]'],
        catalog_sha256=sha(root/'catalog.json'),unknown_reaction_tokens=unknown,
        reaction_adapter='Released participant set followed by >>; no Rhea reaction direction or EC labels added',
        max_length=512,padding='max_length, matching native CREEP mean pooling'))


def load_backbones(device,attention='eager',checkpointing='block'):
    from transformers import T5Config,T5EncoderModel,BertConfig,BertModel
    t5=ROOT.parent/'hf_cache/hub/models--Rostlab--prot_t5_xl_half_uniref50-enc/snapshots/94a6abc029ae13029317b140b7424e012bf8dfbf'
    rx=ROOT/'.deps/CARE/CREEP/data/pretrained_rxnfp'
    protein=T5EncoderModel(T5Config.from_pretrained(t5))
    state=torch.load(t5/'pytorch_model.bin',map_location='cpu',weights_only=True)
    missing,unexpected=protein.load_state_dict(state,strict=False)
    if missing or unexpected:raise ValueError(f'ProtT5 state mismatch: {missing} {unexpected}')
    del state
    reaction=BertModel(BertConfig.from_pretrained(rx))
    state=torch.load(rx/'pytorch_model.bin',map_location='cpu',weights_only=True)
    state={k.removeprefix('bert.'):v for k,v in state.items() if not k.startswith('cls.')}
    missing,unexpected=reaction.load_state_dict(state,strict=False)
    # Some MLM checkpoints omit the unused BERT pooler. CREEP uses CLS tokens.
    if unexpected or any(not k.startswith('pooler.') for k in missing):raise ValueError((missing,unexpected))
    if attention=='sdpa':
        from reactzyme_public_t5_sdpa import enable_t5_sdpa
        enable_t5_sdpa(protein)
    if checkpointing=='block':protein.gradient_checkpointing_enable()
    elif checkpointing=='ffn':
        from reactzyme_public_t5_sdpa import enable_t5_ffn_checkpointing
        enable_t5_ffn_checkpointing(protein)
    protein.config.use_cache=False
    return protein.to(device),reaction.to(device)


def train(a):
    torch.set_num_threads(4);torch.manual_seed(42)
    root=RUN/'features';out=RUN/'models'/f'creep_two_modality_{a.split}_seed42';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    if (out/'protocol.json').exists() and not a.resume:raise FileExistsError('Use --resume for an existing CREEP run')
    device=f'cuda:{a.gpu}';receipt=json.loads((root/'creep_tokens.complete.json').read_text())
    assert receipt['catalog_sha256']==sha(root/'catalog.json')
    pt=torch.tensor(np.load(root/'creep_protein_tokens.npy'),device=device,dtype=torch.long)
    rt=torch.tensor(np.load(root/'creep_reaction_tokens.npy'),device=device,dtype=torch.long)
    protein,reaction=load_backbones(device,a.attention,a.checkpoint_policy)
    ph=torch.nn.Linear(1024,256).to(device);rh=torch.nn.Linear(256,256).to(device)
    params=list(protein.parameters())+list(reaction.parameters())+list(ph.parameters())+list(rh.parameters())
    opt=torch.optim.Adam(params,lr=1e-5,weight_decay=0,foreach=True)
    lossfn,source=native_loss();args=SimpleNamespace(normalize=False,CL_loss='EBM_NCE',CL_neg_samples=1,T=.1)
    data=np.load(root/'pairs.npz');pairs=torch.tensor(data[f'{a.split}_train'],device=device)
    val=data[f'{a.split}_validation'];vr=np.unique(val[:,0]);vp=np.unique(val[:,1])
    truth=dict(reaction_index=np.searchsorted(vr,val[:,0]),enzyme_index=np.searchsorted(vp,val[:,1]))
    protocol=dict(method='CREEP two-modality, exact-edge ReactZyme adapter',split=a.split,seed=42,
        epochs=40,batch=a.batch,lr=1e-5,loss='Symmetric EBM-NCE, T=0.1, one cyclic negative, no embedding normalization in training',
        native_source_sha256=sha(source),data_sha256=sha(root/'pairs.npz'),
        encoders_finetuned=True,text_used=False,extra_EC_positive_pairs_used=False,
        alpha_generative=0,pooling='Protein mean over padded 512 tokens; reaction CLS',
        selection='Maximum mean bidirectional validation all-positive MRR every epoch',
        precision='BF16 autocast, FP32 parameters and optimizer',test_used_for_training_or_selection=False)
    protocol.update(attention_kernel=a.attention,gradient_checkpointing=a.checkpoint_policy,
        reaction_adapter=receipt['reaction_adapter'],native_default_batch=16)
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('Resume protocol differs')
    else:save_json(out/'protocol.json',protocol)
    def encode(tokens,backbone,head,pad,kind):
        hidden=backbone(input_ids=tokens,attention_mask=tokens.ne(pad)).last_hidden_state
        return head(hidden.mean(1) if kind=='protein' else hidden[:,0])
    def export(indices,tokens,backbone,head,pad,kind):
        result=[]
        for ids in torch.as_tensor(indices,device=device).split(a.batch):
            with torch.autocast('cuda',dtype=torch.bfloat16):z=encode(tokens[ids],backbone,head,pad,kind)
            result.append(F.normalize(z.float(),dim=-1))
        return torch.cat(result)
    best=-1;start_epoch=1
    modules=dict(protein=protein,reaction=reaction,ph=ph,rh=rh)
    if a.resume:
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        for name,mod in modules.items():mod.load_state_dict(ck[name])
        opt.load_state_dict(ck['optimizer']);best=ck['best'];start_epoch=ck['epoch']+1
        torch.set_rng_state(ck['cpu_rng']);torch.cuda.set_rng_state(ck['cuda_rng'],a.gpu);del ck
    for epoch in range(start_epoch,41):
        begin=time.time();protein.train();reaction.train();ph.train();rh.train();total=0.;seen=0
        order=torch.randperm(len(pairs),device=device)
        for step,idx in enumerate(order.split(a.batch)):
            if len(idx)<2:idx=torch.cat([idx,order[:1]])
            e=pairs[idx];opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pz=encode(pt[e[:,1]],protein,ph,receipt['protein_pad'],'protein')
                rz=encode(rt[e[:,0]],reaction,rh,receipt['reaction_pad'],'reaction')
            lp,_=lossfn(pz.float(),rz.float(),args);lr,_=lossfn(rz.float(),pz.float(),args);loss=(lp+lr)/2
            if not torch.isfinite(loss):raise ValueError('Nonfinite CREEP loss')
            loss.backward();opt.step();total+=float(loss.detach())*len(idx);seen+=len(idx)
            if step%20==0:
                rec=dict(stage='training',epoch=epoch,step=step,examples=seen,epoch_examples=len(pairs),
                    loss=total/seen,seconds=time.time()-begin,peak_gpu_gib=torch.cuda.max_memory_allocated(a.gpu)/2**30)
                save_json(out/'status.json',rec);print(json.dumps(rec),flush=True)
        protein.eval();reaction.eval();ph.eval();rh.eval()
        with torch.inference_mode():
            zp=export(vp,pt,protein,ph,receipt['protein_pad'],'protein');zr=export(vr,rt,reaction,rh,receipt['reaction_pad'],'reaction')
            validation=evaluate_scores(zr@zp.T,truth)['summary']
        value=float(np.mean([d['all']['reactzyme_mrr'] for d in validation.values()]))
        record=dict(epoch=epoch,training_loss=total/seen,validation=validation,selection_value=value,seconds=time.time()-begin)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if value>best:
            best=value;torch.save(dict(protein=protein.state_dict(),reaction=reaction.state_dict(),
                ph=ph.state_dict(),rh=rh.state_dict(),epoch=epoch,value=value),out/'best.pt')
        save_json(out/'status.json',dict(stage='validation',**record));print(json.dumps(record),flush=True)
        last={name:mod.state_dict() for name,mod in modules.items()}
        last.update(optimizer=opt.state_dict(),epoch=epoch,best=best,
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(a.gpu))
        torch.save(last,out/'last.partial.pt');os.replace(out/'last.partial.pt',out/'last.pt');del last
    selected=torch.load(out/'best.pt',map_location=device,weights_only=False)
    for name,mod in [('protein',protein),('reaction',reaction),('ph',ph),('rh',rh)]:mod.load_state_dict(selected[name]);mod.eval()
    save_json(out/'selection.json',dict(epoch=selected['epoch'],validation_value=selected['value'],test_used=False))
    test=data[f'{a.split}_test'];tr=np.unique(test[:,0]);tp=np.unique(test[:,1])
    truth=dict(reaction_index=np.searchsorted(tr,test[:,0]),enzyme_index=np.searchsorted(tp,test[:,1]))
    with torch.inference_mode():
        scores=export(tr,rt,reaction,rh,receipt['reaction_pad'],'reaction')@export(tp,pt,protein,ph,receipt['protein_pad'],'protein').T
        metrics=evaluate_scores(scores,truth)['summary']
    np.savez(out/'test_scores.npz',scores=scores.cpu().numpy(),reaction_index=tr,protein_index=tp)
    save_json(out/'complete.json',dict(method=protocol['method'],split=a.split,selected_epoch=selected['epoch'],test=metrics))
    save_json(out/'status.json',dict(stage='complete'))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['tokens','train']);p.add_argument('--split',choices=SPLITS,default='reaction_smi');p.add_argument('--gpu',type=int,default=3);p.add_argument('--batch',type=int,default=64)
    p.add_argument('--attention',choices=['eager','sdpa'],default='sdpa')
    p.add_argument('--checkpoint-policy',choices=['none','block','ffn'],default='block')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();tokenize(RUN/'features') if a.action=='tokens' else train(a)


if __name__=='__main__':main()
