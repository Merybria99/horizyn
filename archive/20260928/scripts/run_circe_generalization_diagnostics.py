#!/usr/bin/env python3
"""Cached, validation-only CIRCE diagnostics. No training or test-set tuning."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horizyn.generalization_diagnostics import (
    METRICS, associations, atomic_json, chemistry_similarity, decorate_rows,
    read_pairs, read_sequence_hits, score_matrix_rows, summarize, transfer_baselines,
)

BASE = ROOT / "runs/circe_v3_reactzyme_3gpu/reactzyme/reaction_smi/seed42"
DEFAULT = ROOT / "runs/circe_generalization_diagnostics_v1"
LABELS = ROOT / "runs/circe_v3_reaction_smi_cls002.DfNq1C/biofp_targets.npz"
METHODS = ("none", "cls002", "cls005")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024**2), b''): result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def run(command, **kwargs):
    print('Running: ' + ' '.join(map(str,command)), flush=True)
    subprocess.run(list(map(str,command)), check=True, **kwargs)


def identity(path):
    path = Path(path).resolve()
    s = path.stat()
    return dict(path=str(path), size=s.st_size, mtime_ns=s.st_mtime_ns)


def validate_manifest(root):
    """Cheap guards even for independently resumed stages; no hashing giant HDF5s."""
    manifest = read_json(root/'manifest.json')
    for source in list(manifest['sources'].values()) + list(manifest['checkpoints'].values()):
        current = identity(source['path'])
        if any(current[k] != source[k] for k in current):
            raise ValueError(f"Input changed: {source['path']}; use a new output directory")
    if sha(root/'validation.yaml') != manifest['evaluation_config_sha256']:
        raise ValueError('Generated evaluation config changed')
    return manifest


def completed(receipt, root, outputs):
    if not receipt.exists():
        return False
    if read_json(receipt).get('manifest_sha256') != sha(root/'manifest.json'):
        raise ValueError(f'Stale completion receipt: {receipt}')
    if not all(path.is_file() for path in outputs):
        raise ValueError(f'Completed stage has missing outputs: {receipt}')
    return True


def prepare(args):
    import torch
    import yaml
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    config_path = BASE / 'configs/validation.yaml'
    config = yaml.safe_load(config_path.read_text())
    data = config['data']
    if (data['test_pairs_path'] != data['validation_pairs_path'] or
        data['test_reactions_path'] != data['validation_reactions_path']):
        raise ValueError('This diagnostic must evaluate validation, never test')
    inputs = {key: data[key] for key in ('train_pairs_path','validation_pairs_path',
              'train_reactions_path','validation_reactions_path','validation_retrieval_candidate_ids_path')}
    inputs.update(fasta=str(args.fasta.resolve()), labels=str(LABELS),
                  label_vocab=str(LABELS.with_name('biofp_vocab.json')), train_config=str(BASE/'configs/train.yaml'))
    sources = {key:dict(identity(path), sha256=sha(path)) for key,path in inputs.items()}
    sources['config'] = dict(identity(config_path),sha256=sha(config_path))
    train_config = yaml.safe_load((BASE/'configs/train.yaml').read_text())
    if Path(train_config['data']['train_pairs_path']).resolve() != Path(data['train_pairs_path']).resolve():
        raise ValueError('Training and diagnostic associations differ')
    if read_json(inputs['label_vocab'])['source_sha256']['train_pairs'] != sources['train_pairs_path']['sha256']:
        raise ValueError('Auxiliary labels were not built from the retained training pairs')
    provenance = {}
    for method, folder in (('cls002','circe_v3_reaction_smi_cls002.DfNq1C'),
                           ('cls005','circe_v3_reaction_smi_cls005_fresh.piabCZ')):
        path = ROOT/'runs'/folder/'pipeline.log'
        with path.open() as handle:
            header = handle.read(16384)
        expected_config = str((BASE/'configs/train.yaml').relative_to(ROOT))
        if f'Loading config from: {expected_config}' not in header:
            raise ValueError(f'Unexpected training configuration: {path}')
        provenance[method] = dict(log=str(path), prefix_sha256=hashlib.sha256(header.encode()).hexdigest())
    for path in (Path(__file__),ROOT/'horizyn/generalization_diagnostics.py',ROOT/'scripts/evaluate_protein_pooling.py'):
        sources[path.name] = dict(identity(path),sha256=sha(path))
    for key in ('protein_residue_embeds_path', 'reaction_t5v2_embeds_path',
                'reaction_unimol2_embeds_path', 'reaction_chiro_embeds_path',
                'reaction_chemistry_vectors_path'):
        sources[key] = identity(data[key])
    sources['mmseqs'] = dict(identity(args.mmseqs), sha256=sha(args.mmseqs))
    checkpoints = {}
    reference_recipe = None
    for method, expected_weight in zip(METHODS,(0,.02,.05)):
        source = DEFAULT / f'checkpoints/{method}_epoch09.ckpt'
        destination = root / f'checkpoints/{method}_epoch09.ckpt'
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            # Immutable named epoch snapshots: hard links survive training's pruning.
            # Fall back to a copy across filesystems. Never link mutable last.ckpt.
            try: os.link(source,destination)
            except OSError: shutil.copy2(source,destination)
        checkpoint = torch.load(destination,map_location='cpu',weights_only=False,mmap=True)
        if checkpoint['epoch'] != 8 or checkpoint['global_step'] != 864:
            raise ValueError(f'Not the matched epoch-9 checkpoint: {destination}')
        hp = checkpoint['hyper_parameters']
        if abs(float(hp.get('biofp_aux_weight',0))-expected_weight)>1e-8:
            raise ValueError(f'Wrong auxiliary weight: {method}')
        recipe = {k: repr(v) for k,v in hp.items() if k not in ('biofp_aux_weight','biofp_family_weights')}
        if reference_recipe is not None and recipe != reference_recipe:
            raise ValueError('Checkpoints differ in training/model settings beyond auxiliary supervision')
        reference_recipe = recipe
        checkpoints[method] = dict(identity(destination),epoch=9,step=864,lambda_aux=expected_weight)
        del checkpoint
    train, valid = read_pairs(data['train_pairs_path']), read_pairs(data['validation_pairs_path'])
    if set(train)&set(valid):raise ValueError('Train/validation edge overlap')
    r2e,e2r=associations(valid)
    candidates=Path(data['validation_retrieval_candidate_ids_path']).read_text().splitlines()
    candidates=[x.strip() for x in candidates if x.strip()]
    if len(candidates)!=len(set(candidates)) or set(candidates)!=set(e2r):
        raise ValueError('Expected the original full validation enzyme pool')
    config_text = yaml.safe_dump(config,sort_keys=False)
    manifest=dict(version=1, sources=sources, checkpoints=checkpoints, training_provenance=provenance,
                  evaluation_config_sha256=hashlib.sha256(config_text.encode()).hexdigest(),
                  chemistry=dict(kind='unordered-molecule-set Morgan Tanimoto',radius=2,bits=2048,chirality=True),
                  sequence=dict(max_seqs=args.max_seqs,min_coverage=.8,sensitivity=7.5,evalue=.001),
                  validation=dict(pairs=len(valid),reactions=len(r2e),enzymes=len(e2r)),
                  test_used=False)
    path=root/'manifest.json'
    if path.exists() and read_json(path)!=manifest:
        raise ValueError('Diagnostic inputs/settings changed; use a new --output directory')
    temporary = root/'validation.yaml.tmp'
    temporary.write_text(config_text)
    os.replace(temporary,root/'validation.yaml')
    atomic_json(path,manifest)
    print('Prepared matched epoch-9 snapshots; no model weights changed.',flush=True)


def context(root):
    manifest=read_json(root/'manifest.json')
    paths={k:v['path'] for k,v in manifest['sources'].items()}
    train,valid=read_pairs(paths['train_pairs_path']),read_pairs(paths['validation_pairs_path'])
    tr,te=associations(train);vr,ve=associations(valid)
    return manifest,paths,train,valid,sorted(tr),sorted(te),sorted(vr),sorted(ve)


def write_fastas(source, output, train_ids, valid_ids):
    wanted=set(train_ids)|set(valid_ids);found=set();seen=set()
    with (output/'train.fasta').open('w') as train, (output/'validation.fasta').open('w') as valid:
        def emit(identifier, sequence):
            if identifier not in wanted:return
            if identifier in seen:raise ValueError(f'Duplicate sequence ID: {identifier}')
            seen.add(identifier)
            if not sequence:raise ValueError(f'Empty sequence: {identifier}')
            found.add(identifier)
            for ids,handle in ((train_ids,train),(valid_ids,valid)):
                if identifier in ids:handle.write(f'>{identifier}\n{sequence}\n')
        name=None;sequence=[]
        with Path(source).open() as handle:
            for line in handle:
                if line.startswith('>'):
                    if name is not None:emit(name,''.join(sequence))
                    name=line[1:].split()[0];sequence=[]
                else:sequence.append(line.strip())
        if name is not None:emit(name,''.join(sequence))
    if wanted-found:raise ValueError(f'Missing {len(wanted-found)} sequences; benchmark not reduced')


def chemistry(root, paths, train_reactions, reactions):
    import numpy as np
    cache=root/'chemistry.npz'
    if cache.exists():
        with np.load(cache) as z:return z['similarity'],z['valid_train'],z['valid_queries']
    def smiles(path):
        with Path(path).open() as h:return {r['reaction_id']:r['reaction_smiles'] for r in csv.DictReader(h)}
    train_smiles,valid_smiles=smiles(paths['train_reactions_path']),smiles(paths['validation_reactions_path'])
    sim,vt,vq=chemistry_similarity([train_smiles[r] for r in train_reactions], [valid_smiles[r] for r in reactions])
    temporary=root/'chemistry.tmp.npz'
    np.savez_compressed(temporary,similarity=sim,valid_train=vt,valid_queries=vq)
    os.replace(temporary,cache)
    return sim,vt,vq


def cpu(args):
    import numpy as np
    import h5py
    import yaml
    root=args.output
    manifest,paths,train,valid,tr,te,vr,ve=context(root)
    if completed(root/'cpu.complete.json',root,[root/'novelty.json',
                 root/'reaction_neighbor_transfer.json',root/'enzyme_neighbor_transfer.json']):
        print('Reusing completed CPU diagnostics.');return
    sim,vt,vq=chemistry(root,paths,tr,vr)
    sequence=root/'sequence';sequence.mkdir(exist_ok=True)
    if not (sequence/'sequences.complete.json').exists():
        write_fastas(paths['fasta'],sequence,set(te),set(ve))
        atomic_json(sequence/'sequences.complete.json',dict(train=len(te),validation=len(ve)))
    hits_path=sequence/'hits.tsv'
    if not hits_path.exists():
        seq=manifest['sequence']
        run([paths['mmseqs'],'easy-search',sequence/'validation.fasta',sequence/'train.fasta',
             sequence/'hits.partial.tsv',sequence/'tmp','--threads',args.threads,
             '--max-seqs',seq['max_seqs'],'-s',seq['sensitivity'],'-e',seq['evalue'],
             '-c',seq['min_coverage'],'--cov-mode',0,
             '--format-output','query,target,fident,qcov,tcov,bits'])
        os.replace(sequence/'hits.partial.tsv',hits_path)
    hits=read_sequence_hits(hits_path,set(te),set(ve))
    annotated=set()
    with np.load(paths['labels'],allow_pickle=True) as labels:
        available=labels['mechanism_mask'].any(1)|labels['cofactor_mask'].any(1)
        annotated=set(map(str,labels['ids'][available]))
    protein_metadata={p:dict(identity=max([h[1] for h in hits.get(p,[])],default=None),
                             auxiliary=p in annotated, hit_count=len(hits.get(p,[]))) for p in ve}
    cfg=yaml.safe_load((root/'validation.yaml').read_text())['data']
    feature_ids={}
    for modality in ('t5v2','unimol2','chiro'):
        with h5py.File(cfg[f'reaction_{modality}_embeds_path'],'r') as store:
            feature_ids[modality]={x.decode() if isinstance(x,bytes) else str(x) for x in store['ids'][:]}
    with np.load(cfg['reaction_chemistry_vectors_path'],allow_pickle=True) as z:
        ids=list(map(str,z['ids'])); masks=z['mask'] if 'mask' in z else np.ones(len(ids),bool)
        chemistry_present={r for r,m in zip(ids,masks) if m}
    reaction_metadata={}
    for i,q in enumerate(vr):
        missing=[m for m,ids in feature_ids.items() if q+'_f' not in ids]
        if q not in chemistry_present:missing.append('chemistry')
        reaction_metadata[q]=dict(similarity=float(sim[i].max()) if vq[i] else None,
                                  missing_modality=bool(missing),missing_modalities=missing,
                                  parseable=bool(vq[i]))
    metadata=dict(reactions=reaction_metadata,proteins=protein_metadata,
                  sequence_no_hit=len(ve)-len(hits),sequence_identity_definition='Maximum identity among retrieved coverage-filtered hits; no hit is unknown',
                  reaction_descriptor='Unordered molecular composition, NOT a reaction-center fingerprint',
                  r2e_sequence_descriptor='Minimum best-hit identity across ground-truth positive enzymes with hits; separate completeness flag',
                  invalid_training_chemistry=int((~vt).sum()))
    atomic_json(root/'novelty.json',metadata)
    for name,matrix in transfer_baselines(sim,tr,vr,ve,train,hits,vq).items():
        rows=score_matrix_rows(matrix,vr,ve,valid)
        atomic_json(root/f'{name}.json',dict(queries=rows,epoch=None,
             definition='Sequence-hit-limited training-association transfer; see manifest for search settings',
             tie_policy='stable sorted candidate IDs; all-zero queries retained and counted'))
    atomic_json(root/'cpu.complete.json',dict(manifest_sha256=sha(root/'manifest.json')))


def model(args):
    from scripts.evaluate_protein_pooling import evaluate_checkpoint, CONFIGURED_FORWARD_CANDIDATES
    root=args.output;manifest=read_json(root/'manifest.json');method=args.method
    out=root/'models'/method;out.mkdir(parents=True,exist_ok=True)
    receipt=out/'complete.json';signature=sha(root/'manifest.json')
    if completed(receipt,root,[out/'queries.json',out/'metrics.json']):
        print(f'Reusing completed {method} ranks.');return
    checkpoint=manifest['checkpoints'][method]['path']
    result=evaluate_checkpoint(checkpoint,str(root/'validation.yaml'),args.device,64,
                               args.target_batch_size,True,direction='both',
                               target_embeds_cache=str(out/'targets.pt'),
                               evaluation_protocol=CONFIGURED_FORWARD_CANDIDATES,
                               per_query_output=str(out/'queries.json'))
    if (result['num_enzyme_candidates']!=manifest['validation']['enzymes'] or
        result['num_reaction_candidates']!=manifest['validation']['reactions']):
        raise ValueError('Candidate pool changed')
    atomic_json(out/'metrics.json',result)
    atomic_json(receipt,dict(manifest_sha256=signature,epoch=9))


def report(args):
    root=args.output;manifest,paths,train,valid,*_=context(root)
    if not completed(root/'cpu.complete.json',root,[root/'novelty.json']):
        raise ValueError('CPU diagnostics not complete')
    metadata=read_json(root/'novelty.json');methods={}
    for method in METHODS:
        out=root/'models'/method
        if not completed(out/'complete.json',root,[out/'queries.json',out/'metrics.json']):
            raise ValueError(f'{method} evaluation not complete')
        methods[method]=read_json(out/'queries.json')['queries']
    for name in ('reaction_neighbor_transfer','enzyme_neighbor_transfer'):
        methods[name]=read_json(root/f'{name}.json')['queries']
    decorated={name:decorate_rows(rows,train,valid,metadata['reactions'],metadata['proteins']) for name,rows in methods.items()}
    for name,rows in decorated.items():atomic_json(root/f'{name}_per_query.json',rows)
    result=summarize(decorated,repeats=args.bootstrap)
    result.update(epoch=9,manifest_sha256=sha(root/'manifest.json'),sequence_no_hit=metadata['sequence_no_hit'])
    atomic_json(root/'summary.json',result)
    for name in ('summaries','paired_differences'):
        rows=result[name]
        with (root/f'{name}.csv').open('w',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# CIRCE generalization diagnostics','',
           'Matched epoch 9; validation only; full candidate pools are preserved. No test-set tuning.', '',
           '| Method | Direction | Reaction group | Queries | Components | ReactZyme MRR | 95% CI |',
           '|---|---|---|---:|---:|---:|---|']
    for r in result['summaries']:
        if r['metric']=='reactzyme_mrr' and r['stratum'] in ('all','reaction/seen','reaction/unseen','reaction/mixed'):
            lines.append(f"| {r['method']} | {r['direction']} | {r['stratum']} | {r['queries']} | {r['clusters']} | {r['mean']:.5f} | {r['ci95']} |")
    lines += ['', '## Interpretation limits', '',
              '- Bootstrap intervals condition on the trained checkpoints and candidate pools; they do not measure seed-to-seed variability.',
              '- Sequence identity is the best retrieved hit at >=80% coverage of both sequences. No-hit cases remain unknown; search is not exhaustive.',
              '- The reaction fingerprint measures unordered molecular composition, not the chemical transformation.',
              '- Baselines use training associations only and retain no-hit queries with tied scores; inspect all_tied_queries before interpreting performance.',
              '- Auxiliary coverage refers to actual training targets, not complete biological annotation availability.',
              '- R-to-E sequence strata describe known positive enzymes. Missing-hit completeness and positive-count strata must be inspected alongside them.',
              '- Paired differences and chemistry/sequence/joint/missing-modality strata are in the CSV and JSON outputs.',
              '- Decide any next training change from these diagnostics; do not automatically launch a hyperparameter sweep.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    # Compare full loss curves only at matched epochs; total loss changes with lambda.
    curves=[]
    for method,run_root in zip(METHODS,(BASE,ROOT/'runs/circe_v3_reaction_smi_cls002.DfNq1C',ROOT/'runs/circe_v3_reaction_smi_cls005_fresh.piabCZ')):
        with (run_root/'logs/protein_pooling_training/version_0/metrics.csv').open() as h:
            for row in csv.DictReader(h):
                if row.get('val/mean_bidirectional_reactzyme_mrr') or row.get('train/loss_mlnce_epoch'):
                    curves.append(dict(method=method,epoch=int(float(row['epoch']))+1,
                                       values={k:float(v) for k,v in row.items() if v and
                                               ('reactzyme_mrr' in k or (k.startswith('train/loss_') and k.endswith('_epoch'))
                                                or '/weight_' in k)}))
    atomic_json(root/'training_curves.json',curves)
    print(f'Report: {root / "report.md"}',flush=True)


def check_gpus(gpus, *, allow_shared=False):
    if not gpus or any(not x.isdigit() for x in gpus) or len(set(gpus))!=len(gpus):
        raise ValueError('Pass distinct physical GPU IDs')
    def query(fields):
        out=subprocess.run(['nvidia-smi',fields,'--format=csv,noheader,nounits'],
                           check=True,capture_output=True,text=True,timeout=15)
        return [[x.strip() for x in row] for row in csv.reader(out.stdout.splitlines()) if row]
    available={row[0]:row[1] for row in query('--query-gpu=index,uuid')}
    if not set(gpus)<=available.keys():raise ValueError('Selected GPU IDs do not exist')
    selected={available[g] for g in gpus}
    busy=[pid for uuid,pid in query('--query-compute-apps=gpu_uuid,pid') if uuid in selected]
    if busy and not allow_shared:
        raise RuntimeError(f'Selected GPUs are occupied by PIDs {busy}; no processes stopped')
    if busy:
        print(f'GPU sharing enabled: existing PIDs {busy} are left running. '
              'Free VRAM is not reserved; memory exhaustion remains possible.', flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','cpu','model','report','all'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--gpus',default='0',help='Free physical GPU IDs; one checkpoint job per GPU, no DDP')
    parser.add_argument('--method',choices=METHODS,default='none')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--target-batch-size',type=int,default=64)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--max-seqs',type=int,default=64)
    parser.add_argument('--bootstrap',type=int,default=500)
    parser.add_argument('--fasta',type=Path,default=ROOT/'data/revised_protocols/reactzyme_official/reaction_smi/proteins.fasta')
    parser.add_argument('--mmseqs',type=Path,default=ROOT/'tools/mmseqs/bin/mmseqs')
    args=parser.parse_args();args.output=args.output.resolve()
    if min(args.threads,args.max_seqs,args.target_batch_size,args.bootstrap)<1:parser.error('Sizes must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    # Distinct locks permit CPU and GPU stages in parallel, not duplicate writers.
    lock=(args.output/f'.{args.stage}_{args.method if args.stage=="model" else "stage"}.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('This stage is already running')
    if args.stage not in ('prepare','all'):
        validate_manifest(args.output)
    if args.stage=='prepare':prepare(args)
    elif args.stage=='cpu':cpu(args)
    elif args.stage=='model':model(args)
    elif args.stage=='report':report(args)
    else:
        gpus=args.gpus.split(',');check_gpus(gpus);prepare(args)
        logs=args.output/'logs';logs.mkdir(exist_ok=True)
        common=[sys.executable,str(Path(__file__).resolve()),'--output',str(args.output)]
        with (logs/'cpu.log').open('a') as h:
            cpu_job=subprocess.Popen(common+['cpu','--threads',str(args.threads),'--mmseqs',str(args.mmseqs)],stdout=h,stderr=subprocess.STDOUT)
            def gpu_worker(gpu,methods):
                for method in methods:
                    with (logs/f'{method}.log').open('a') as log:
                        run(common+['model','--method',method,'--target-batch-size',args.target_batch_size],
                            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'PYTHONUNBUFFERED':'1'},stdout=log,stderr=subprocess.STDOUT)
            errors=[]
            with ThreadPoolExecutor(max_workers=min(len(gpus),len(METHODS))) as pool:
                futures=[pool.submit(gpu_worker,gpu,METHODS[i::len(gpus)]) for i,gpu in enumerate(gpus[:len(METHODS)])]
                for future in futures:
                    try:future.result()
                    except Exception as exc:errors.append(str(exc))
            if cpu_job.wait():errors.append('CPU stage failed; inspect logs/cpu.log')
        if errors:raise RuntimeError('; '.join(errors))
        report(args)


if __name__=='__main__':main()
