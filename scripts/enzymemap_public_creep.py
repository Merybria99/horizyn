#!/usr/bin/env python3
"""CREEP two-modality refit on EnzymeMap, with full-library screening selection.

prepare is CPU-only. probe and train run with torchrun --nproc-per-node=4.
Native ProtT5/rxnfp backbones, padded pooling, heads and symmetric EBM-NCE
are retained. Only exact training associations are used, with no EC mining.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from reactzyme_public_creep import load_backbones, native_loss
from generalization_clipzyme_screening_evaluate import evaluate_query

ROOT = Path(__file__).resolve().parents[1]
CROSS = ROOT / 'runs/generalization_20260919_2251/cross_paper_retraining'
CAT = CROSS / 'clipzyme_f3_catalog_v1'
SCREEN = CROSS / 'clipzyme_screening_evaluation_protocol_v2'
AUDITED = ROOT / 'runs/enzymemap_public_horizyn_20260923_seed42'
PT5 = ROOT.parent / 'hf_cache/hub/models--Rostlab--prot_t5_xl_half_uniref50-enc/snapshots/94a6abc029ae13029317b140b7424e012bf8dfbf'
RXNFP = ROOT / '.deps/CARE/CREEP/data/pretrained_rxnfp'
METRICS = ['bedroc85', 'bedroc20', 'ef0.05', 'ef0.1']


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(p)


def now():
    return datetime.now(timezone.utc).isoformat()


def log(**kwargs):
    print(json.dumps(dict(utc=now(), **kwargs)), flush=True)


def fasta(path):
    result = {}
    key, parts = None, []
    with Path(path).open() as f:
        for line in f:
            if line.startswith('>'):
                if key is not None:
                    result[key] = ''.join(parts)
                key, parts = line[1:].strip().split()[0], []
            else:
                parts.append(line.strip())
    if key is not None:
        result[key] = ''.join(parts)
    return result


def prepare(out):
    if (out / 'features.complete.json').exists():
        log(stage='already_prepared', output=str(out))
        return
    out.mkdir(parents=True, exist_ok=True)
    receipt = read(AUDITED / 'features.complete.json')
    for name in ['catalog.json', 'axes.npz', 'reaction_inputs.csv']:
        assert sha(AUDITED / name) == receipt['sha256'][name]
        shutil.copyfile(AUDITED / name, out / name)
    cat = read(out / 'catalog.json')
    axes = dict(np.load(out / 'axes.npz'))
    # These assets contain training edges and dev labels, not test labels.
    assert axes['train'].shape == (34427, 2)
    assert len(np.unique(axes['train'], axis=0)) == 34180
    assert len(cat['protein_ids']) == 222985 and len(cat['candidate_ids']) == 261907
    assert len(axes['kept']) == 252113 and len(axes['test']) == 1521
    assert sum(bool(p) for p in cat['validation_positive_indices']) == 2652
    with (CAT / 'train_pairs.csv').open() as f:
        raw_pairs = list(csv.DictReader(f))
    ri = {k: i for i, k in enumerate(cat['reaction_ids'])}
    pi = {k: i for i, k in enumerate(cat['protein_ids'])}
    expected = np.array([[ri[r['reaction_id']], pi[r['protein_id']]] for r in raw_pairs])
    np.testing.assert_array_equal(expected, axes['train'])
    proteins = fasta(CAT / 'screening_proteins.fasta')
    assert set(proteins) == set(cat['protein_ids'])
    for key, seq in proteins.items():
        assert seq and key == 'p_' + hashlib.sha256(seq.encode()).hexdigest()[:24]
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(str(PT5), do_lower_case=False, local_files_only=True)
    tokens = np.lib.format.open_memmap(out / 'protein_tokens.npy', mode='w+', dtype=np.int32,
                                     shape=(len(cat['protein_ids']), 512))
    lengths = []
    for start in range(0, len(tokens), 2048):
        sequences = [proteins[k] for k in cat['protein_ids'][start:start+2048]]
        batch = tokenizer([' '.join(s) for s in sequences], truncation=True, max_length=512,
                          padding='max_length', return_tensors='np')
        tokens[start:start+len(sequences)] = batch['input_ids']
        lengths.extend(len(s) for s in sequences)
        if start % 16384 == 0:
            log(stage='protein_tokenization', done=start, total=len(tokens))
    tokens.flush()
    vocab = {s: i for i, s in enumerate((RXNFP / 'vocab.txt').read_text().splitlines())}
    source = ROOT / '.deps/CARE/CREEP/CREEP/utils/tokenization.py'
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SMI_REGEX_PATTERN' for t in node.targets):
            regex = re.compile(ast.literal_eval(node.value))
    with (out / 'reaction_inputs.csv').open() as f:
        reactions = list(csv.DictReader(f))
    assert [r['reaction_id'] for r in reactions] == cat['reaction_ids']
    reaction_tokens = np.full((len(reactions), 512), vocab['[PAD]'], np.int32)
    unknown, truncated, incomplete_tokenization = 0, 0, []
    for i, row in enumerate(reactions):
        smiles = row['reaction_smiles']
        assert smiles.count('>>') == 1
        parsed = regex.findall(smiles)
        if ''.join(parsed) != smiles:
            incomplete_tokenization.append(row['reaction_id'])
        truncated += len(parsed) > 510
        parsed = parsed[:510]
        unknown += sum(t not in vocab for t in parsed)
        ids = [vocab['[CLS]']] + [vocab.get(t, vocab['[UNK]']) for t in parsed] + [vocab['[SEP]']]
        reaction_tokens[i, :len(ids)] = ids
    assert not incomplete_tokenization, incomplete_tokenization[:10]
    np.save(out / 'reaction_tokens.npy', reaction_tokens)
    source_files = [Path(__file__), ROOT/'scripts/reactzyme_public_creep.py',
                    ROOT/'scripts/reactzyme_public_t5_sdpa.py',
                    ROOT/'.deps/CARE/task2_baselines/CREEP/step_01_train_CREEP.py', source,
                    PT5/'pytorch_model.bin', PT5/'config.json', RXNFP/'pytorch_model.bin', RXNFP/'config.json', RXNFP/'vocab.txt']
    write(out/'features.complete.json', dict(created_utc=now(),
        axes_source=str(AUDITED), exact_training_row_order_verified=True,
        native_sequence_limit=512, protein_pad=tokenizer.pad_token_id, reaction_pad=vocab['[PAD]'],
        proteins=len(tokens), reactions=len(reactions), training_rows=34427,
        training_unique_sequence_edges=34180, candidate_ids=261907,
        truncated_proteins=int(sum(n > 511 for n in lengths)), truncated_reactions=int(truncated),
        unknown_reaction_tokens=int(unknown),
        reaction_adapter='Original directed EnzymeMap reaction SMILES; native rxnfp regex/vocabulary; no fabricated sides or EC positives.',
        source_hashes={str(p):sha(p) for p in source_files if p.name != Path(__file__).name},
        input_hashes={str(p):sha(p) for p in [CAT/'screening_proteins.fasta',CAT/'train_pairs.csv',SCREEN/'receipt.json',AUDITED/'features.complete.json']},
        output_hashes={n:sha(out/n) for n in ['catalog.json','axes.npz','reaction_inputs.csv','protein_tokens.npy','reaction_tokens.npy']}))
    log(stage='prepared', proteins=len(tokens), reactions=len(reactions), unknown_reaction_tokens=unknown)


def symmetric_loss(p, r):
    """Algebraically identical to the released loss, without per-step .item()."""
    positive = (p * r).sum(-1) / .1
    negative_pr = (p * r.roll(-1, 0)).sum(-1) / .1
    negative_rp = (r * p.roll(-1, 0)).sum(-1) / .1
    return F.softplus(-positive).mean()/2 + (F.softplus(negative_pr).mean()+F.softplus(negative_rp).mean())/4


def gather_variable(x, sizes):
    padded = F.pad(x, (0, 0, 0, max(sizes)-len(x)))
    return torch.cat([v[:size] for v, size in zip(all_gather(padded), sizes)])


def check_distributed_loss(device, rank, world, out):
    """Compare native loss and parameter gradients, including uneven batches."""
    torch.manual_seed(123)
    network = torch.nn.Linear(8, 4, bias=True).to(device)
    ddp = DDP(network, device_ids=[device.index])
    native, _ = native_loss()
    args = SimpleNamespace(normalize=False,CL_loss='EBM_NCE',CL_neg_samples=1,T=.1)
    checks = []
    for n in [16, 19]:
        torch.manual_seed(123+n)
        inputs = torch.randn(n, 2, 8, device=device) * .2
        reference = torch.nn.Linear(8,4,bias=True).to(device)
        reference.load_state_dict(network.state_dict())
        p, r = reference(inputs).unbind(1)
        l1, _ = native(p,r,args)
        l2, _ = native(r,p,args)
        expected = (l1+l2)/2
        expected.backward()
        chunks = inputs.tensor_split(world)
        sizes = [len(c) for c in chunks]
        ddp.zero_grad(set_to_none=True)
        p, r = ddp(chunks[rank]).unbind(1)
        actual = symmetric_loss(gather_variable(p,sizes),gather_variable(r,sizes))
        actual.backward()
        torch.testing.assert_close(actual,expected,atol=2e-7,rtol=2e-6)
        errors=[]
        for a,b in zip(network.parameters(),reference.parameters()):
            torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5)
            errors.append(float((a.grad-b.grad).abs().max()))
        checks.append(dict(global_batch=n,loss_error=float(abs(actual-expected)),max_gradient_error=max(errors)))
    if rank==0:
        write(out/'distributed_loss_checks.json',dict(status='passed',world_size=world,checks=checks))
    del ddp,network,reference
    torch.cuda.empty_cache()


class Model(torch.nn.Module):
    def __init__(self,device,features):
        super().__init__()
        self.protein,self.reaction=load_backbones(device,'sdpa','ffn')
        # rxnfp uses last_hidden_state[:,0]; its unused pooler has no gradients.
        if self.reaction.pooler is not None:
            self.reaction.pooler.requires_grad_(False)
        self.ph=torch.nn.Linear(1024,256).to(device)
        self.rh=torch.nn.Linear(256,256).to(device)
        self.protein_pad=features['protein_pad'];self.reaction_pad=features['reaction_pad']

    def encode(self,tokens,kind):
        backbone,head,pad = (self.protein,self.ph,self.protein_pad) if kind=='protein' else (self.reaction,self.rh,self.reaction_pad)
        hidden=backbone(input_ids=tokens,attention_mask=tokens.ne(pad)).last_hidden_state
        return head(hidden.mean(1) if kind=='protein' else hidden[:,0])

    def forward(self,p,r):
        return self.encode(p,'protein'),self.encode(r,'reaction')


@torch.inference_mode()
def export(model,pt,rt,indices,dest,rank,world,batch):
    model.eval();dest.mkdir(parents=True,exist_ok=True)
    for kind,tokens,ids in [('protein',pt,np.arange(len(pt))),('reaction',rt,indices)]:
        local=np.array_split(ids,world)[rank]
        values=np.empty((len(local),256),np.float32)
        started=time.monotonic()
        for start in range(0,len(local),batch):
            selected=torch.as_tensor(local[start:start+batch],device=pt.device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=model.encode(tokens[selected],kind)
            normalized=F.normalize(z.float(),dim=-1)
            if not torch.isfinite(normalized).all():
                raise ValueError('Nonfinite exported embeddings')
            values[start:start+len(selected)]=normalized.cpu().numpy()
            if start%(batch*50)==0:
                log(stage='embedding_export',rank=rank,kind=kind,done=start,total=len(local),seconds=time.monotonic()-started)
        np.save(dest/f'{kind}_rank{rank}.npy',values)
    dist.barrier()
    if rank==0:
        for kind in ['protein','reaction']:
            arrays=[np.load(dest/f'{kind}_rank{i}.npy') for i in range(world)]
            np.save(dest/f'{kind}.npy',np.concatenate(arrays))
            for i in range(world):
                (dest/f'{kind}_rank{i}.npy').unlink()
    dist.barrier()


def validation(dest,axes,cat,device,epoch):
    proteins=torch.from_numpy(np.load(dest/'protein.npy')).to(device)
    reactions=torch.from_numpy(np.load(dest/'reaction.npy')).to(device)
    positives=[np.array(p,dtype=np.int64) for p in cat['validation_positive_indices']]
    records=[]
    with ThreadPoolExecutor(max_workers=12) as pool:
        for start in range(0,len(reactions),64):
            scores=(reactions[start:start+64]@proteins.T).cpu().numpy()[:,axes['expanded']]
            records.extend(r for r in pool.map(evaluate_query,[(start+i,cat['validation_ids'][start+i],v,positives[start+i],axes['kept']) for i,v in enumerate(scores)]) if r is not None)
    summary={}
    for table,count in [('table1',261907),('table2',252113)]:
        selected=[r[table] for r in records if r[table] is not None]
        summary[table]=dict(queries=len(selected),candidate_ids=count,**{m:float(np.mean([r[m] for r in selected])) for m in METRICS})
    assert summary['table1']['queries']==2652 and summary['table2']['queries']==2216
    (dest/'per_query.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    result=dict(epoch=epoch,summary=summary,selection_value=summary['table1']['bedroc85'],validation_only=True,test_labels_read=False)
    write(dest/'summary.json',result)
    return result


def save_checkpoint(path,state):
    temporary=path.with_suffix('.partial.pt')
    torch.save(state,temporary);temporary.replace(path)


def train(a):
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    assert world==4,'This protocol is fixed to four GPUs'
    device=torch.device('cuda',local);torch.cuda.set_device(device)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(hours=2))
    out=a.output.resolve();features=read(out/'features.complete.json')
    if rank==0:
        for name,digest in features['output_hashes'].items():
            assert sha(out/name)==digest
        for path,digest in features['source_hashes'].items():
            assert sha(path)==digest
    dist.barrier()
    check_distributed_loss(device,rank,world,out)
    export_batch = 256
    if a.action == 'train':
        probe = read(out/'runtime_probe.json')
        assert probe['status'] == 'passed' and probe['code_sha256'] == sha(__file__)
        export_batch = probe['selected_export_batch']
    protocol=dict(method='CREEP two-modality, exact-edge EnzymeMap refit',seed=42,epochs=40,
        architecture='Fine-tuned ProtT5 1024 -> linear 256; fine-tuned rxnfp 256 -> linear 256',
        initialization='Original pretrained language-model backbones; fresh linear heads; no supervised CREEP/ReactZyme checkpoint',
        pooling='Protein mean over all 512 padded positions, reaction CLS',
        reaction_adapter=features['reaction_adapter'],loss='Native symmetric EBM-NCE, T=0.1, one cyclic negative, unnormalized training vectors',
        inference='L2-normalized cosine similarity; no ensemble, dictionary, biological-label losses, or reranking',
        optimizer=dict(name='Adam',lr=1e-5,weight_decay=0,fused=True),
        global_batch=256,local_batch=64,world_size=4,native_default_batch=16,prior_reactzyme_batch=64,
        distributed='Autograd all-gather preserves native global cyclic negatives; uneven final batch retained without duplicating rows',
        precision='BF16 autocast; FP32 parameters, optimizer, loss and cosine scores; TF32 disabled',
        memory='SDPA encoder attention; FFN-only activation checkpointing; tokens cached on each GPU',
        validation_epochs=list(range(3,40,3))+[40],
        selection='Maximum full-library validation BEDROC85; earliest exact tie; all 40 epochs retained',
        export_batch=export_batch,train_rows=34427,unique_sequence_edges=34180,
        validation_queries=2652,validation_id_excluded_queries=2216,
        full_candidates=261907,id_excluded_candidates=252113,
        test_queries=1521,test_id_excluded_queries=1337,
        test_policy='Test evaluator reads labels only after frozen selection; both candidate pools evaluated from the same selected checkpoint.',
        text_used=False,EC_positive_mining=False,alpha_generative=0,
        test_used_for_training_or_selection=False,features_sha256=sha(out/'features.complete.json'),
        metric_source_sha256=sha(ROOT/'scripts/generalization_clipzyme_screening_evaluate.py'),
        screening_protocol_sha256=sha(SCREEN/'receipt.json'),code_sha256=sha(__file__))
    if a.action=='train' and rank==0:
        if (out/'protocol.json').exists():
            assert read(out/'protocol.json')==protocol,'Resume protocol changed'
        else:
            write(out/'protocol.json',protocol);shutil.copyfile(__file__,out/'training_source.py')
    dist.barrier()
    torch.manual_seed(42)
    model=Model(device,features)
    ddp=DDP(model,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True,static_graph=True,bucket_cap_mb=64)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=0,fused=True)
    pt=torch.as_tensor(np.load(out/'protein_tokens.npy'),dtype=torch.long,device=device)
    rt=torch.as_tensor(np.load(out/'reaction_tokens.npy'),dtype=torch.long,device=device)
    axes=dict(np.load(out/'axes.npz'));cat=read(out/'catalog.json')
    pairs=torch.as_tensor(axes['train'],device=device)
    best,best_epoch,start_epoch=-1.,None,1
    if a.action=='train' and (out/'last.pt').exists():
        last=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert last['protocol_sha256']==sha(out/'protocol.json')
        model.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizer'])
        best,best_epoch,start_epoch=last['best'],last['best_epoch'],last['epoch']+1
        torch.set_rng_state(last['rng'][rank]['cpu']);torch.cuda.set_rng_state(last['rng'][rank]['cuda'],device)
        del last
    else:
        torch.manual_seed(42+rank)  # independent dropout, common epoch permutation below
    for epoch in range(start_epoch,41):
        ddp.train();torch.cuda.reset_peak_memory_stats(device)
        begin=time.monotonic();total=torch.zeros((),device=device);seen=0
        order=torch.randperm(len(pairs),generator=torch.Generator().manual_seed(42+epoch)).to(device)
        steps=[]
        for step,indices in enumerate(order.split(256)):
            pieces=indices.tensor_split(world);sizes=[len(p) for p in pieces]
            edges=pairs[pieces[rank]];optimizer.zero_grad(set_to_none=True)
            started=time.monotonic()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                p,r=ddp(pt[edges[:,1]],rt[edges[:,0]])
            loss=symmetric_loss(gather_variable(p.float(),sizes),gather_variable(r.float(),sizes))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite CREEP loss')
            loss.backward();optimizer.step();total+=loss.detach()*len(indices);seen+=len(indices)
            if a.action=='probe':
                torch.cuda.synchronize();steps.append(time.monotonic()-started)
            if step%10==0 and rank==0:
                status=dict(stage='training' if a.action=='train' else 'probe',epoch=epoch,step=step,
                    examples=seen,epoch_examples=len(pairs),training_loss=float(total/seen),
                    seconds=time.monotonic()-begin,peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30)
                write(out/('status.json' if a.action=='train' else 'probe_status.json'),status);log(**status)
            if a.action=='probe' and step==2:
                stats=[None]*world
                dist.all_gather_object(stats,dict(rank=rank,step_seconds=steps,peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30))
                ddp.eval();optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                inference=[]
                for batch in [256,512,768]:
                    torch.cuda.reset_peak_memory_stats(device)
                    times=[]
                    for trial in range(2):
                        torch.cuda.synchronize();started=time.monotonic()
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                            z=model.encode(pt[:batch],'protein')
                        assert torch.isfinite(z).all()
                        torch.cuda.synchronize();times.append(time.monotonic()-started)
                        del z
                    values=[None]*world
                    dist.all_gather_object(values,dict(rank=rank,batch=batch,step_seconds=times,
                        peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30))
                    inference.append(dict(batch=batch,ranks=values,sequences_per_second=world*batch/max(v['step_seconds'][-1] for v in values)))
                selected_batch=max(inference,key=lambda x:x['sequences_per_second'])['batch']
                if rank==0:
                    write(out/'runtime_probe.json',dict(status='passed',training=stats,inference=inference,
                        selected_export_batch=selected_batch,
                        code_sha256=sha(__file__),note='Three real training updates on fresh initialization, discarded; main training restarts from original backbones.'))
                dist.destroy_process_group();return
        record=dict(epoch=epoch,training_loss=float(total/seen),train_seconds=time.monotonic()-begin,updates=step+1,training_rows=seen)
        assert seen==34427
        if epoch in protocol['validation_epochs']:
            dest=out/'validation'/f'epoch{epoch:03d}'
            optimizer.zero_grad(set_to_none=True)
            if rank==0:
                write(out/'status.json',dict(stage='full_library_validation_export',epoch=epoch,utc=now()))
            export(model,pt,rt,axes['validation'],dest,rank,world,export_batch)
            if rank==0:
                val=validation(dest,axes,cat,device,epoch);record['validation']=val['summary']
                if val['selection_value']>best:
                    best,best_epoch=val['selection_value'],epoch
                    save_checkpoint(out/'best.pt',dict(model=model.state_dict(),epoch=epoch,validation_value=best,protocol_sha256=sha(out/'protocol.json')))
                log(stage='validation',**record,best_epoch=best_epoch)
            dist.barrier()
            rng=[None]*world
            dist.all_gather_object(rng,dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device)))
            if rank==0:
                save_checkpoint(out/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                    best=best,best_epoch=best_epoch,rng=rng,protocol_sha256=sha(out/'protocol.json')))
            dist.barrier()
        if rank==0:
            with (out/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            write(out/'status.json',dict(stage='epoch_complete',utc=now(),best_epoch=best_epoch,best_validation_bedroc85=best,**record))
    if rank==0:
        selected=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        write(out/'selection.json',dict(selected_epoch=selected['epoch'],validation_bedroc85=selected['validation_value'],
            checkpoint=str(out/'best.pt'),checkpoint_sha256=sha(out/'best.pt'),selection_saved_utc=now(),test_used=False,
            selection_metric='Full-library validation BEDROC85',protocol_sha256=sha(out/'protocol.json')))
        del selected
    dist.barrier()
    selected=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(selected['model']);del selected
    dest=out/'test';export(model,pt,rt,axes['test'],dest,rank,world,export_batch)
    if rank==0:
        protein=torch.from_numpy(np.load(dest/'protein.npy')).to(device)
        reaction=torch.from_numpy(np.load(dest/'reaction.npy')).to(device)
        scores=np.lib.format.open_memmap(dest/'scores.npy',mode='w+',dtype=np.float32,shape=(1521,261907))
        for start in range(0,len(reaction),64):
            scores[start:start+64]=(reaction[start:start+64]@protein.T).cpu().numpy()[:,axes['expanded']]
        scores.flush();del scores
        (dest/'query_ids.txt').write_text('\n'.join(cat['test_ids'])+'\n')
        (dest/'candidate_ids.txt').write_text('\n'.join(cat['candidate_ids'])+'\n')
        write(dest/'score_receipt.json',dict(created_utc=now(),scores_sha256=sha(dest/'scores.npy'),shape=[1521,261907],
            selection_sha256=sha(out/'selection.json'),test_labels_read=False,
            protein_embeddings_sha256=sha(dest/'protein.npy'),reaction_embeddings_sha256=sha(dest/'reaction.npy')))
        subprocess.run([sys.executable,str(ROOT/'scripts/generalization_clipzyme_screening_evaluate.py'),
            '--scores',str(dest/'scores.npy'),'--query-ids',str(dest/'query_ids.txt'),'--candidate-ids',str(dest/'candidate_ids.txt'),
            '--protocol',str(SCREEN),'--output',str(dest/'evaluation'),'--metric-workers','12'],check=True)
        result=read(dest/'evaluation/summary.json')
        write(out/'complete.json',dict(completed_utc=now(),method=protocol['method'],selection=read(out/'selection.json'),
            summary=result['summary'],evaluation_sha256=sha(dest/'evaluation/summary.json')))
        write(out/'status.json',dict(stage='complete',summary=result['summary'],utc=now()))
    dist.barrier();dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','probe','train'])
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.action=='prepare':
        prepare(a.output.resolve())
    else:
        train(a)


if __name__=='__main__':
    main()
