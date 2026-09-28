#!/usr/bin/env python3
"""Fresh, cached UniMol2-only reaction-smi experiment and matched validation."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horizyn.config import DotDict, validate_config
from horizyn.generalization_diagnostics import atomic_json, read_pairs, decorate_rows, summarize
from scripts.run_circe_generalization_diagnostics import (
    BASE, DEFAULT, check_gpus, sha, identity, validate_manifest, completed,
)


def make_config(base, output, devices, steps):
    if devices not in (1, 2, 3, 4) or steps <= 0:
        raise ValueError('Use 1-4 GPUs and a positive step budget')
    config = copy.deepcopy(base)
    data, model, training = config['data'], config['model'], config['training']
    data['train_batch_size'] = 1536 // devices
    for name in ('model', 'chiro', 'chirality', 'chienn', 'chemistry', 'directional'):
        data[f'reaction_use_{name}'] = False
        if f'reaction_use_{name}' in model:
            model[f'reaction_use_{name}'] = False
    # Remove unused paths too: accidentally loading these should fail, not hide
    # an unwanted feature branch behind an apparently single-modality config.
    for key in list(data):
        if key.endswith('_path') and any(name in key for name in
                ('reaction_t5v2', 'reaction_model', 'reaction_chiro', 'reaction_chirality',
                 'reaction_chienn', 'reaction_chemistry', 'reaction_directional')):
            data[key] = None
    data.update(protein_biofp_targets_path=None, protein_biofp_vocab_path=None,
                reaction_allow_missing_unimol2=True, reaction_embedding_in_memory=True)
    model['reaction_pooling'] = 'interaction'
    model['query_encoder_dims'] = [512, 512]
    model['reaction_attention_pooling']['separate_side_poolers'] = False
    model['reaction_multimodal_attention'] = dict(
        hidden_dim=512, dropout=0.0, modality_dropout=0.0, token_layer_norm=True,
        modality_encoder_num_layers=1, modality_encoder_widths=[512],
        modality_encoder_use_layer_norm=True, modality_encoder_dropout=0.0,
        modality_encoder_normalise_output=False, side_composition='molecule_set',
        fusion='mean', output_projection='mlp')
    training.update(devices=devices, max_steps=steps,
                    strategy='ddp_find_unused_parameters_true' if devices > 1 else 'auto',
                    init_from_checkpoint=None, biofp_pretrain_checkpoint=None,
                    early_stopping={'enabled':False})
    training['loss']['biofp_aux_weight'] = 0.0
    config['logging'].update(log_dir=str(output/'logs'), checkpoint_dir=str(output/'checkpoints'),
                             recovery_every_n_train_steps=50)
    config['logging']['wandb'].update(enabled=False, mode='disabled')
    config['ablation'] = dict(variant='CIRCE-v3-UniMol2-interaction', split='reaction_smi',
                              fresh_towers=True, seed=config['seed'],
                              global_batch_size=1536, frozen_molecular_backbone='cached UniMol2')
    return config


def prepare(output, devices, steps):
    import h5py
    import numpy as np
    config = make_config(yaml.safe_load((BASE/'configs/train.yaml').read_text()),output,devices,steps)
    data = config['data']
    train_pairs = read_pairs(data['train_pairs_path'])
    validation_pairs = read_pairs(data['validation_pairs_path'])
    config['training']['max_epochs'] = math.ceil(steps / math.ceil(len(train_pairs)/1536))
    if set(train_pairs) & set(validation_pairs):
        raise ValueError('Train/validation association overlap')
    coverage = {}
    for split, pairs in (('train',train_pairs),('validation',validation_pairs)):
        feature_path = data[f'{split}_reaction_unimol2_embeds_path']
        with h5py.File(feature_path,'r') as store:
            ids = [x.decode() if isinstance(x,bytes) else str(x) for x in store['ids'][:]]
            if len(ids) != len(set(ids)):
                raise ValueError('Duplicate UniMol2 cache IDs')
            index = {q:i for i,q in enumerate(ids)}
            offsets = store['reactant_offsets'][:]
            if len(offsets) != len(ids)+1 or np.any(np.diff(offsets)<0):
                raise ValueError('Invalid UniMol2 offsets')
            if offsets[-1] != len(store['reactant_vectors']) or store['reactant_vectors'].shape[1]!=768:
                raise ValueError('Incompatible UniMol2 cache')
            requested = sorted({q for q,p in pairs})
            available = [q for q in requested if q+'_f' in index and
                         offsets[index[q+'_f']+1] > offsets[index[q+'_f']]]
            if not available:
                raise ValueError(f'No usable {split} molecule features')
            missing = sorted(set(requested)-set(available))
            # Existing rows with zero tokens must not silently look present.
            if any(q+'_f' in index for q in missing):
                raise ValueError('Empty molecule-feature rows require cache repair')
            i = index[available[0]+'_f']
            if not np.isfinite(store['reactant_vectors'][offsets[i]:offsets[i+1]]).all():
                raise ValueError('Nonfinite molecule features')
            coverage[split] = dict(reactions=len(requested), usable=len(available),
                                   missing=len(missing), missing_ids=missing,
                                   pairs=len(pairs), feature_file=identity(feature_path))
    validate_config(DotDict(config))
    validation = copy.deepcopy(config)
    validation['data'].update(
        test_pairs_path=data['validation_pairs_path'],
        test_reactions_path=data['validation_reactions_path'],
        reaction_unimol2_embeds_path=data['validation_reaction_unimol2_embeds_path'])
    validate_config(DotDict(validation))
    configs = output/'configs'
    configs.mkdir(parents=True,exist_ok=False)
    for name, value in (('train',config),('validation',validation)):
        (configs/f'{name}.yaml').write_text(yaml.safe_dump(value,sort_keys=False))
    atomic_json(output/'experiment.json',dict(
        seed=config['seed'], steps=steps, global_batch_size=1536, devices=devices,
        batch_per_gpu=data['train_batch_size'], coverage=coverage,
        training_pairs_sha256=sha(data['train_pairs_path']),
        validation_pairs_sha256=sha(data['validation_pairs_path']),
        configs={n:sha(configs/f'{n}.yaml') for n in ('train','validation')},
        code={str(p):sha(p) for p in (Path(__file__),ROOT/'horizyn/model.py',
              ROOT/'horizyn/protein_pooling_lightning_module.py',ROOT/'horizyn/training_options.py',
              ROOT/'scripts/evaluate_protein_pooling.py')},
        initialization='Fresh trainable towers; cached UniMol2 and pretrained frozen SLEEC only',
        missing_policy='Keep all queries/candidates; unavailable molecular input is masked'))
    print(json.dumps(dict(output=str(output),steps=steps,batch_per_gpu=data['train_batch_size'],
                          coverage=coverage),indent=2),flush=True)


def evaluate(output):
    import torch
    from scripts.evaluate_protein_pooling import evaluate_checkpoint, CONFIGURED_FORWARD_CANDIDATES
    checkpoint = output/'checkpoints/last.ckpt'
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    step = saved['global_step'];del saved
    expected = json.loads((output/'experiment.json').read_text())
    if step != expected['steps']:
        raise ValueError(f'Checkpoint is at step {step}, expected {expected["steps"]}')
    report_dir=output/'evaluation';report_dir.mkdir(exist_ok=True)
    metrics=evaluate_checkpoint(str(checkpoint),str(output/'configs/validation.yaml'),
              'cuda:0',64,128,True,direction='both',
              evaluation_protocol=CONFIGURED_FORWARD_CANDIDATES,
              target_embeds_cache=str(report_dir/'targets.pt'),
              per_query_output=str(report_dir/'queries.json'))
    atomic_json(report_dir/'metrics.json',metrics)
    # Use the already completed CPU novelty/search diagnostics, never test data.
    if not (DEFAULT/'novelty.json').is_file():
        print('Validation complete. Run generalization diagnostics to obtain the stratified comparison.')
        return
    manifest=validate_manifest(DEFAULT)
    if not completed(DEFAULT/'cpu.complete.json',DEFAULT,[DEFAULT/'novelty.json']):
        raise ValueError('CPU novelty diagnostics are incomplete')
    for key in ('training','validation'):
        source='train_pairs_path' if key=='training' else 'validation_pairs_path'
        if expected[f'{key}_pairs_sha256'] != manifest['sources'][source]['sha256']:
            raise ValueError('Diagnostic association files differ')
    novelty=json.loads((DEFAULT/'novelty.json').read_text())
    config=yaml.safe_load((output/'configs/train.yaml').read_text())['data']
    train,valid=read_pairs(config['train_pairs_path']),read_pairs(config['validation_pairs_path'])
    rows=json.loads((report_dir/'queries.json').read_text())['queries']
    # This encoder needs UniMol2 only; don't inherit missing ChIRo/chemistry flags.
    new_reaction_metadata={q:{**v,'missing_modality':'unimol2' in v['missing_modalities']}
                           for q,v in novelty['reactions'].items()}
    methods={'interaction':decorate_rows(rows,train,valid,new_reaction_metadata,novelty['proteins'])}
    if step == 864 and (DEFAULT/'none_per_query.json').is_file():
        if not completed(DEFAULT/'models/none/complete.json',DEFAULT,[DEFAULT/'models/none/queries.json']):
            raise ValueError('Matched control evaluation is incomplete')
        methods['none']=json.loads((DEFAULT/'none_per_query.json').read_text())
    result=summarize(methods)
    atomic_json(report_dir/'generalization.json',result)
    for name in ('summaries','paired_differences'):
        rows=result[name]
        if rows:
            with (report_dir/f'{name}.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Validation and seen/unseen comparisons complete:',report_dir,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','launch','run','evaluate'))
    parser.add_argument('--gpus',required=True,help='Free physical IDs, e.g. 0,1,3; no existing jobs stopped')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--steps',type=int,default=864,help='Matched-budget screen; default epoch-9 budget')
    args=parser.parse_args();gpus=args.gpus.split(',')
    if len(gpus) not in (1,2,3,4) or len(set(gpus))!=len(gpus) or any(not g.isdigit() for g in gpus):
        parser.error('Pass 1-4 distinct physical GPU IDs')
    if args.stage in ('launch','run','evaluate'):check_gpus(gpus)
    if args.stage in ('prepare','launch'):
        if args.output:
            output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
        else:
            output=Path(tempfile.mkdtemp(prefix='circe_v3_molecule_interaction_',dir=ROOT/'runs'))
        prepare(output,len(gpus),args.steps)
        if args.stage=='prepare':return
        session=output.name.replace('.','_')
        command=shlex.join(['/usr/bin/env','PYTHONUNBUFFERED=1','OMP_NUM_THREADS=4',
                            'OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4',
                            sys.executable,str(Path(__file__).resolve()),'run',
                            '--output',str(output),'--gpus',args.gpus])
        subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),
                        '/usr/bin/env','-u','BASH_ENV','-u','ENV','/bin/bash','--noprofile','--norc',
                        '-c',f'exec {command} >> {shlex.quote(str(output/"pipeline.log"))} 2>&1'],check=True)
        print(f'Detached launch requested: {session}\nWatch: tail -f {output}/pipeline.log',flush=True)
        return
    if args.output is None:parser.error('--output is required for run/evaluate')
    output=args.output.resolve()
    with (output/'.controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        experiment=json.loads((output/'experiment.json').read_text())
        for name,digest in experiment['configs'].items():
            if sha(output/f'configs/{name}.yaml')!=digest:raise ValueError('Configuration changed')
        for path,digest in experiment['code'].items():
            if sha(path)!=digest:raise ValueError('Experiment code changed; prepare a new output')
        config=yaml.safe_load((output/'configs/train.yaml').read_text())['data']
        for name,key in (('training','train_pairs_path'),('validation','validation_pairs_path')):
            if sha(config[key])!=experiment[f'{name}_pairs_sha256']:
                raise ValueError('Training/validation associations changed')
        for coverage in experiment['coverage'].values():
            if identity(coverage['feature_file']['path'])!=coverage['feature_file']:
                raise ValueError('Molecular cache changed')
        os.environ.update(CUDA_VISIBLE_DEVICES=args.gpus,CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONUNBUFFERED='1',
                          OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
        os.chdir(ROOT)
        if args.stage=='run':
            if len(gpus)!=experiment['devices']:raise ValueError('Prepared GPU count differs')
            if (output/'training.started.json').exists():
                raise ValueError('Training already started here; refusing to overwrite or silently restart')
            atomic_json(output/'training.started.json',dict(pid=os.getpid(),hostname=os.uname().nodename))
            subprocess.run([sys.executable,'scripts/train_protein_pooling.py','--config',
                            str(output/'configs/train.yaml'),'--wandb-mode','disabled'],check=True)
            atomic_json(output/'training.complete.json',dict(steps=experiment['steps']))
        evaluate(output)
        atomic_json(output/'complete.json',dict(training_steps=experiment['steps'],validation_only=True))


if __name__=='__main__':main()
