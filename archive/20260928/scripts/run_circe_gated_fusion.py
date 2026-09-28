#!/usr/bin/env python3
"""Matched-budget three-branch concat/gated-attention reaction-smi experiments."""
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
from horizyn.generalization_diagnostics import atomic_json, read_pairs
from scripts.run_circe_generalization_diagnostics import BASE, check_gpus, identity, sha

MODEL = ROOT.parent / 'hf_cache/hub/models--sagawa--ReactionT5v2-forward/snapshots/933114058cb2604dc1bf536dbebdfcefbe83d4fc'
VARIANTS = {'concat': 'concat', 'gated': 'gated_attention_concat'}


def make_config(base, output, t5_cache, devices, steps, variant):
    if devices not in (1, 2, 3, 4) or steps <= 0 or variant not in VARIANTS:
        raise ValueError('Invalid devices, step budget or fusion variant')
    config = copy.deepcopy(base)
    data, model, training = config['data'], config['model'], config['training']
    for section in (data, model):
        section.update(reaction_use_model=True, reaction_use_chiro=True,
                       reaction_use_chirality=False, reaction_use_chienn=False,
                       reaction_use_chemistry=False, reaction_use_directional=False)
    for key in list(data):
        if ('reaction_chemistry' in key or 'reaction_directional' in key) and key.endswith('_path'):
            data[key] = None
    for part in ('train', 'validation'):
        data[f'{part}_reaction_t5v2_embeds_path'] = str(t5_cache)
        data[f'{part}_reaction_model_embeds_path'] = None
    data.update(train_batch_size=1536 // devices, reaction_embedding_in_memory=True,
                reaction_allow_missing_unimol2=True, reaction_allow_missing_chiro=True,
                reaction_load_directional=False)
    # Retain legacy _f cache IDs; only the unordered molecule side is consumed.
    data['normalize_molecule_sets_as_self_reactions'] = True
    model.update(reaction_pooling='interaction', query_encoder_dims=[256, 512, 512])
    model['reaction_attention_pooling']['separate_side_poolers'] = False
    model['reaction_multimodal_attention'] = dict(
        hidden_dim=256, dropout=0.0, modality_dropout=0.1, token_layer_norm=True,
        modality_encoder_num_layers=1, modality_encoder_widths=[256],
        modality_encoder_use_layer_norm=True, modality_encoder_dropout=0.0,
        modality_encoder_normalise_output=False, side_composition='molecule_set',
        fusion=VARIANTS[variant], output_projection='mlp')
    training.update(devices=devices, max_steps=steps, init_from_checkpoint=None,
                    biofp_pretrain_checkpoint=None, early_stopping={'enabled': False},
                    strategy='ddp_find_unused_parameters_true' if devices > 1 else 'auto')
    config['logging'].update(log_dir=str(output/'logs'), checkpoint_dir=str(output/'checkpoints'),
                             recovery_every_n_train_steps=50)
    config['logging']['wandb'].update(enabled=False, mode='disabled')
    config['ablation'] = dict(variant=f'CIRCE-v3-{variant}-three-branch', fresh_towers=True,
        global_batch_size=1536, seed=config['seed'], split='reaction_smi',
        reaction_roles='unknown participant set; no CGR or inferred sides',
        t5_input='canonical sorted participants, no self-reaction duplication',
        frozen_backbones=['UniMol2', 'ChIRo', 'ReactionT5', 'ProT5', 'SLEEC scorer'])
    return config


def prepare(args, output):
    import h5py
    import numpy as np
    from rdkit import RDLogger
    from horizyn.chemistry.reaction_recovery import components, read_reactions
    RDLogger.DisableLog('rdApp.warning')
    base = yaml.safe_load(args.base_config.read_text())
    data = base['data']
    if base.get('ablation', {}).get('split') != 'reaction_smi':
        raise ValueError('This matched launcher requires a reaction_smi base config')
    if not args.reaction_model.is_dir():
        raise FileNotFoundError(args.reaction_model)
    queries = {}; coverage = {}; source_files = [args.base_config]
    pair_sets = {}
    for part in ('train', 'validation'):
        reactions = read_reactions(data[f'{part}_reactions_path'])
        pairs = read_pairs(data[f'{part}_pairs_path']); pair_sets[part] = set(pairs)
        requested = {q for q, _ in pairs}
        if not requested <= reactions.keys():
            raise ValueError(f'{part}: pairs reference missing reaction rows')
        for query, smiles in reactions.items():
            if '>>' in smiles:
                raise ValueError('This experiment expects unordered molecule sets, not directed reactions')
            canonical = '.'.join(sorted(components(smiles)))
            if query in queries and queries[query] != canonical:
                raise ValueError(f'Conflicting chemistry for {query}')
            queries[query] = canonical
        source_files.extend([Path(data[f'{part}_reactions_path']), Path(data[f'{part}_pairs_path'])])
        coverage[part] = dict(pairs=len(pairs), reactions=len(reactions), modalities={})
        for modality, dim in (('unimol2', 768), ('chiro', 256)):
            path = Path(data[f'{part}_reaction_{modality}_embeds_path']); source_files.append(path)
            with h5py.File(path, 'r') as store:
                ids = [x.decode() if isinstance(x, bytes) else str(x) for x in store['ids'][:]]
                offsets = store['reactant_offsets'][:]
                if len(set(ids)) != len(ids) or len(offsets) != len(ids)+1 or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
                    raise ValueError(f'Invalid IDs/offsets: {path}')
                if store['reactant_vectors'].shape != (int(offsets[-1]), dim):
                    raise ValueError(f'Invalid molecule-vector shape: {path}')
                index = {q: i for i, q in enumerate(ids)}
                empty = [q for q in requested if q+'_f' in index and offsets[index[q+'_f']+1] == offsets[index[q+'_f']]]
                if empty:
                    raise ValueError(f'Empty {modality} entries need cache repair: {empty[:3]}')
                available = sum(q+'_f' in index for q in requested)
                if not available:
                    raise ValueError(f'No usable {part}/{modality} features')
                coverage[part]['modalities'][modality] = dict(available=available, missing=len(requested)-available)
    if pair_sets['train'] & pair_sets['validation']:
        raise ValueError('Training/validation association overlap')
    inputs = output/'t5_inputs.csv'
    with inputs.open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['reaction_id', 'reaction_smiles'])
        writer.writerows((q+'_f', smiles) for q, smiles in sorted(queries.items()))
    variants = list(VARIANTS) if args.variant == 'both' else [args.variant]
    for variant in variants:
        folder = output/variant; (folder/'configs').mkdir(parents=True)
        config = make_config(base, folder, output/'reactiont5_participants.h5', len(args.gpus.split(',')), args.steps, variant)
        config['training']['max_epochs'] = math.ceil(args.steps / max(1, math.ceil(len(pair_sets['train'])/1536))) + 1
        validate_config(DotDict(config))
        validation = copy.deepcopy(config)
        validation['data'].update(test_pairs_path=data['validation_pairs_path'],
            test_reactions_path=data['validation_reactions_path'],
            reaction_t5v2_embeds_path=str(output/'reactiont5_participants.h5'),
            reaction_unimol2_embeds_path=data['validation_reaction_unimol2_embeds_path'],
            reaction_chiro_embeds_path=data['validation_reaction_chiro_embeds_path'])
        for name, cfg in (('train',config), ('validation',validation)):
            (folder/f'configs/{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    code = [Path(__file__), ROOT/'horizyn/model.py', ROOT/'horizyn/gated_reaction_fusion.py',
            ROOT/'horizyn/protein_pooling_lightning_module.py', ROOT/'horizyn/training_options.py',
            ROOT/'scripts/extract_reaction_t5v2_embeddings.py', ROOT/'scripts/evaluate_protein_pooling.py',
            ROOT/'horizyn/config_validation.py']
    atomic_json(output/'experiment.json', dict(variants=variants, steps=args.steps, global_batch_size=1536,
        coverage=coverage, sources=[identity(p) for p in dict.fromkeys(source_files)],
        code={str(p):sha(p) for p in code}, inputs_sha256=sha(inputs),
        configs={str(p.relative_to(output)):sha(p) for p in output.glob('*/configs/*.yaml')},
        t5_model=str(args.reaction_model), t5_input_policy='raw canonical participant strings; no pseudo-reactions',
        enzyme_and_loss='Copied unchanged from base config; fresh trainable tower initialization'))
    print(json.dumps(dict(output=str(output), coverage=coverage), indent=2), flush=True)


def run(args, output):
    import h5py
    import numpy as np
    prepare(args, output)
    experiment = json.loads((output/'experiment.json').read_text())
    temporary = output/'reactiont5_participants.partial.h5'
    destination = output/'reactiont5_participants.h5'
    print('Extracting ReactionT5 participant features once for both variants on the first selected GPU.', flush=True)
    subprocess.run([sys.executable, str(ROOT/'scripts/extract_reaction_t5v2_embeddings.py'),
        '--reactions', str(output/'t5_inputs.csv'), '--output', str(temporary),
        '--model-name', str(args.reaction_model), '--batch-size', '32', '--device', 'cuda:0',
        '--max-length', '512', '--dtype', 'float32', '--no-bidirectional',
        '--no-allow-pseudo-reactions', '--no-standardize'], check=True)
    with (output/'t5_inputs.csv').open() as handle:
        expected = {r['reaction_id'] for r in csv.DictReader(handle)}
    with h5py.File(temporary, 'r') as store:
        ids = [v.decode() if isinstance(v, bytes) else str(v) for v in store['ids'][:]]
        vectors = store['vectors'][:]
        if len(ids) != len(set(ids)) or set(ids) != expected or vectors.shape != (len(ids), 768) or not np.isfinite(vectors).all():
            raise ValueError('Incomplete/invalid ReactionT5 cache; training was not started')
    temporary.rename(destination)
    atomic_json(output/'t5.complete.json', dict(cache=identity(destination), sha256=sha(destination),
        inputs_sha256=sha(output/'t5_inputs.csv'), model=str(args.reaction_model), max_length=512,
        truncation=True, note='Long participant strings are truncated; this is not a complete equation encoder'))
    for variant in experiment['variants']:
        for path, digest in experiment['code'].items():
            if sha(path) != digest: raise ValueError(f'Code changed during experiment: {path}')
        for path, digest in experiment['configs'].items():
            if sha(output/path) != digest: raise ValueError(f'Config changed: {path}')
        for source in experiment['sources']:
            if identity(source['path']) != source: raise ValueError(f'Input changed: {source["path"]}')
        folder = output/variant
        print(f'Training {variant}: {args.steps} steps; global batch 1536.', flush=True)
        with (folder/'training.log').open('a') as log:
            subprocess.run([sys.executable, str(ROOT/'scripts/train_protein_pooling.py'),
                '--config', str(folder/'configs/train.yaml'), '--wandb-mode', 'disabled'], stdout=log, stderr=subprocess.STDOUT, check=True)
        # Evaluation runs in a fresh process so it cannot retain GPU memory before
        # the next variant starts. Both directions use the same validation pool.
        subprocess.run([sys.executable, str(Path(__file__).resolve()), 'evaluate', '--output', str(folder)], check=True)
    atomic_json(output/'complete.json', dict(variants=experiment['variants'], steps=args.steps, validation_only=True))
    print(f'Completed: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('launch', 'run', 'prepare', 'evaluate'))
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--variant', choices=(*VARIANTS, 'both'), default='gated')
    parser.add_argument('--steps', type=int, default=864)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--base-config', type=Path, default=BASE/'configs/train.yaml')
    parser.add_argument('--reaction-model', type=Path, default=MODEL)
    args = parser.parse_args(); gpus = args.gpus.split(',')
    if len(gpus) not in (1,2,3,4) or len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus) or args.steps <= 0:
        parser.error('Use 1-4 distinct physical GPU IDs and a positive step budget')
    if args.stage == 'evaluate':
        if args.output is None: parser.error('--output required')
        import torch
        from scripts.evaluate_protein_pooling import evaluate_checkpoint, CONFIGURED_FORWARD_CANDIDATES
        checkpoint = args.output/'checkpoints/last.ckpt'
        expected = json.loads((args.output.parent/'experiment.json').read_text())['steps']
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        if saved['global_step'] != expected: raise ValueError('Checkpoint did not reach matched step budget')
        del saved
        report = args.output/'evaluation'; report.mkdir(exist_ok=True)
        metrics = evaluate_checkpoint(str(checkpoint), str(args.output/'configs/validation.yaml'),
            'cuda:0', 64, 128, True, direction='both', evaluation_protocol=CONFIGURED_FORWARD_CANDIDATES,
            target_embeds_cache=str(report/'targets.pt'), per_query_output=str(report/'queries.json'))
        atomic_json(report/'metrics.json', metrics)
        return
    if args.stage in ('launch','run'): check_gpus(gpus)
    if args.stage in ('launch','prepare'):
        output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix='circe_v3_gated_fusion_', dir=ROOT/'runs'))
        if args.output: output.mkdir(parents=True, exist_ok=False)
        if args.stage == 'prepare': prepare(args, output); return
        command = shlex.join([sys.executable, str(Path(__file__).resolve()), 'run', '--output', str(output),
            '--gpus', args.gpus, '--variant', args.variant, '--steps', str(args.steps),
            '--base-config', str(args.base_config.resolve()), '--reaction-model', str(args.reaction_model.resolve())])
        session = output.name.replace('.', '_')
        subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT), '/usr/bin/env',
            '-u','BASH_ENV','-u','ENV','/bin/bash','--noprofile','--norc','-c',
            f'exec {command} >> {shlex.quote(str(output/"pipeline.log"))} 2>&1'], check=True)
        print(f'Detached session: {session}\nWatch: tail -f {output}/pipeline.log', flush=True)
        return
    if args.output is None: parser.error('--output required')
    output = args.output.resolve()
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'experiment.json').exists(): raise ValueError('Run already prepared; use a fresh output directory (no implicit resume)')
        os.environ.update(CUDA_VISIBLE_DEVICES=args.gpus, CUDA_DEVICE_ORDER='PCI_BUS_ID',
            PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
        os.chdir(ROOT)
        run(args, output)


if __name__ == '__main__':
    main()
