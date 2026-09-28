#!/usr/bin/env python3
"""Gate-first, fingerprint-free CIRCE reaction fusion using existing caches."""
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
import socket
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horizyn.config import DotDict, validate_config
from horizyn.generalization_diagnostics import atomic_json, read_pairs
from scripts.run_circe_generalization_diagnostics import BASE, check_gpus, identity, sha

DEFAULT_CACHE_RUN = ROOT/'runs/circe_v3_gated_fusion_idhkmc0l'
VARIANTS = {'feature': 'feature_gate', 'scalar': 'scalar_gate'}
ANNOTATIONS = ROOT/'runs/bio_aux_minimal_v1/data/reaction_smi'


def make_config(base, output, t5_cache, devices, epochs, variant, enzyme_tower='baseline',
                positive_biology=None, biofp_aux_weight=0.02):
    if devices not in (1, 2, 3, 4) or epochs <= 0 or variant not in VARIANTS:
        raise ValueError('Invalid device count, epochs or variant')
    if enzyme_tower not in ('baseline', 'multiview'):
        raise ValueError('Invalid enzyme tower')
    if positive_biology is not None and (enzyme_tower != 'multiview' or not math.isfinite(biofp_aux_weight) or biofp_aux_weight <= 0):
        raise ValueError('Positive biology requires multiview and a positive finite biofp_aux_weight')
    config = copy.deepcopy(base)
    data, model, training = config['data'], config['model'], config['training']
    # Preserve the F3-sized reaction adapters/projector and CIRCE-v3 loss.
    if enzyme_tower == 'multiview':
        model['enzyme_input_mode'] = 'raw_mean_sleec_multiview'
        model['pooling'] = 'sleec_guided_attention'
        model['sleec_pooling']['freeze_scorer'] = True
        for key in ('enzyme_block_fusion', 'biofp', 'hyperbolic_encoder'):
            model.pop(key, None)
        model['enzyme_multiview'] = dict(hidden_dim=256, num_slots=4, dropout=0.1,
            uniform_mix=0.05, min_effective_residues=4, diversity_margin=0.9,
            gate_floor=0.05, max_logit_scale=10.0)
        training['loss']['biofp_aux_weight'] = 0.0
        training['enzyme_attention_regularization'] = dict(entropy_weight=0.01, diversity_weight=0.001)
        if positive_biology is not None:
            labels = positive_biology['families']
            model['biofp'] = dict(positive_labels=labels)
            data.update(protein_biofp_targets_path=positive_biology['targets'],
                        protein_biofp_vocab_path=positive_biology['vocab'], biofp_missing_policy='zero_with_mask')
            training['loss'].update(biofp_aux_mode='positive_anchor', biofp_aux_weight=biofp_aux_weight,
                biofp_aux_warmup_epochs=3, biofp_normalize_active_families=True,
                biofp_family_weights={family:1.0 for family in labels}, biofp_confidence_cap=1.0)
    else:
        if model.get('enzyme_input_mode') != 'raw_mean_sleec_biological_factorized':
            raise ValueError('Baseline requires the biological-factorized reference tower')
        model.pop('enzyme_multiview', None)
        training.pop('enzyme_attention_regularization', None)
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
    data.update(train_batch_size=1536//devices, reaction_embedding_in_memory=True,
                reaction_allow_missing_unimol2=True, reaction_allow_missing_chiro=True,
                reaction_load_directional=False, normalize_molecule_sets_as_self_reactions=True)
    # The _f convention is for cache IDs only; cached T5 text has no X>>X duplication.
    model['reaction_pooling'] = 'attention'
    model['reaction_multimodal_attention'].update(
        fusion=VARIANTS[variant], side_composition='molecule_set', output_projection='mlp')
    training.update(devices=devices, max_epochs=epochs, max_steps=-1,
                    init_from_checkpoint=None, biofp_pretrain_checkpoint=None,
                    early_stopping={'enabled': False},
                    strategy='ddp_find_unused_parameters_true' if devices > 1 else 'auto',
                    log_attention_stats=True, attention_logging_interval=10)
    config['logging'].update(log_dir=str(output/'logs'), checkpoint_dir=str(output/'checkpoints'),
        checkpoint_on_validation_end=True, save_top_k=-1, recovery_every_n_train_steps=50)
    config['logging']['wandb'].update(enabled=False, mode='disabled')
    config['ablation'] = dict(variant=f'CIRCE-v3-{variant}-gate', split='reaction_smi',
        fresh_towers=True, global_batch_size=1536, seed=config['seed'],
        fusion='jointly conditioned competitive gating before output FFN; no cross-modal attention',
        reaction_inputs=['ReactionT5 participants', 'UniMol2', 'ChIRo'],
        enzyme_tower=enzyme_tower,
        chemistry_features=False, inferred_reaction_sides=False,
        comparison='scalar versus feature gates; not an exact original F3 reproduction')
    return config


def validate_t5_cache(cache_run, data):
    """Check actual input chemistry, encoding policy, IDs, vectors and receipt."""
    import h5py
    import numpy as np
    from rdkit import RDLogger
    from horizyn.chemistry.reaction_recovery import components, read_reactions
    RDLogger.DisableLog('rdApp.warning')
    cache = cache_run/'reactiont5_participants.h5'
    receipt = json.loads((cache_run/'t5.complete.json').read_text())
    inputs = cache_run/'t5_inputs.csv'
    if sha(cache) != receipt['sha256'] or sha(inputs) != receipt['inputs_sha256']:
        raise ValueError('ReactionT5 cache/input receipt mismatch')
    if receipt.get('max_length') != 512:
        raise ValueError('Unexpected ReactionT5 encoding policy')
    queries = {}
    for part in ('train', 'validation'):
        for q, smiles in read_reactions(data[f'{part}_reactions_path']).items():
            canonical = '.'.join(sorted(components(smiles)))
            if q in queries and queries[q] != canonical:
                raise ValueError(f'Conflicting chemistry for {q}')
            queries[q] = canonical
    with inputs.open() as handle:
        rows = list(csv.DictReader(handle))
    cached_inputs = {r['reaction_id']: r['reaction_smiles'] for r in rows}
    expected = {q+'_f': s for q, s in queries.items()}
    if len(cached_inputs) != len(rows) or cached_inputs != expected:
        raise ValueError('Cached participant strings differ from this train/validation split')
    with h5py.File(cache) as store:
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in store['ids'][:]]
        vectors = store['vectors'][:]
        if len(ids) != len(set(ids)) or set(ids) != set(expected) or vectors.shape != (len(ids), 768) or not np.isfinite(vectors).all():
            raise ValueError('Invalid ReactionT5 vectors/IDs')
    return cache


def prepare(args, output):
    import h5py
    import numpy as np
    from horizyn.chemistry.reaction_recovery import read_reactions
    base = yaml.safe_load(args.base_config.read_text())
    if base.get('ablation', {}).get('split') != 'reaction_smi':
        raise ValueError('Expected the reaction_smi reference recipe')
    data = base['data']
    cache = validate_t5_cache(args.cache_run, data)
    coverage, pair_sets, sources = {}, {}, [args.base_config, cache, args.cache_run/'t5.complete.json',
        Path(data['protein_residue_embeds_path']), Path(base['model']['sleec_pooling']['checkpoint_path'])]
    for part in ('train', 'validation'):
        pairs = read_pairs(data[f'{part}_pairs_path']); pair_sets[part] = set(pairs)
        queries = {q+'_f' for q, _ in pairs}
        if not {q for q, _ in pairs} <= read_reactions(data[f'{part}_reactions_path']).keys():
            raise ValueError(f'{part} pairs reference missing reaction rows')
        coverage[part] = dict(pairs=len(pairs), reactions=len(queries), modalities={})
        sources.extend(Path(data[f'{part}_{kind}_path']) for kind in ('pairs', 'reactions'))
        for modality, dim in (('unimol2', 768), ('chiro', 256)):
            path = Path(data[f'{part}_reaction_{modality}_embeds_path']); sources.append(path)
            with h5py.File(path) as store:
                ids = [x.decode() if isinstance(x, bytes) else str(x) for x in store['ids'][:]]
                offsets = store['reactant_offsets'][:]
                if (len(ids) != len(set(ids)) or len(offsets) != len(ids)+1 or offsets[0] != 0
                        or np.any(np.diff(offsets) < 0) or store['reactant_vectors'].shape != (int(offsets[-1]), dim)):
                    raise ValueError(f'Invalid molecule cache: {path}')
                sizes = dict(zip(ids, np.diff(offsets)))
                if any(sizes.get(q, 1) == 0 for q in queries):
                    raise ValueError(f'Empty molecule rows: {path}')
                available = len(queries & sizes.keys())
                if not available: raise ValueError(f'No usable {modality} features')
                coverage[part]['modalities'][modality] = dict(available=available, missing=len(queries)-available)
    if pair_sets['train'] & pair_sets['validation']:
        raise ValueError('Training/validation positive pair overlap')
    sources.append(Path(data['validation_retrieval_candidate_ids_path']))
    positive_biology = None
    if args.positive_biology:
        annotation_sources = dict(train_pairs_path=Path(data['train_pairs_path']),
            matched_members_path=args.matched_members.resolve(),
            directional_features_path=args.directional_features.resolve(),
            ec_labels_path=args.ec_labels.resolve(),
            enzyme_cofactor_labels_path=args.enzyme_cofactors.resolve())
        print('Building sparse positive-only EC/cofactor/transformation targets from training sources...', flush=True)
        annotation_dir = output/'annotations'
        subprocess.run([str(args.annotation_python.absolute()), '-m',
            'horizyn.capability.positive_biological_targets',
            '--train-pairs', str(annotation_sources['train_pairs_path']),
            '--matched-members', str(annotation_sources['matched_members_path']),
            '--directional-features', str(annotation_sources['directional_features_path']),
            '--ec-labels', str(annotation_sources['ec_labels_path']),
            '--enzyme-cofactors', str(annotation_sources['enzyme_cofactor_labels_path']),
            '--output-dir', str(annotation_dir)], cwd=ROOT, check=True)
        target_path = annotation_dir/'positive_targets.npz'
        vocab_path = annotation_dir/'positive_vocab.json'
        metadata = json.loads(vocab_path.read_text())
        if any(metadata['row_coverage'].get(family, 0) == 0 for family in ('ec', 'cofactor', 'mechanism')):
            raise ValueError('No observed positives in a requested biological family')
        sources.extend(annotation_sources.values()); sources.extend([target_path, vocab_path])
        positive_biology = dict(targets=str(target_path), vocab=str(vocab_path), families=metadata['families'])
        coverage['positive_biology'] = metadata['row_coverage']
    variants = list(VARIANTS) if args.variant == 'both' else [args.variant]
    for variant in variants:
        folder = output/variant; (folder/'configs').mkdir(parents=True)
        config = make_config(base, folder, cache, len(args.gpus.split(',')), args.epochs,
                             variant, enzyme_tower=args.enzyme_tower, positive_biology=positive_biology,
                             biofp_aux_weight=args.biofp_aux_weight)
        validate_config(DotDict(config))
        evaluation = copy.deepcopy(config)
        evaluation['data'].update(test_pairs_path=data['validation_pairs_path'],
            test_reactions_path=data['validation_reactions_path'], reaction_t5v2_embeds_path=str(cache),
            reaction_unimol2_embeds_path=data['validation_reaction_unimol2_embeds_path'],
            reaction_chiro_embeds_path=data['validation_reaction_chiro_embeds_path'])
        for name, cfg in [('train', config), ('validation', evaluation)]:
            (folder/f'configs/{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    code = [Path(__file__), ROOT/'horizyn/model.py', ROOT/'horizyn/gated_reaction_fusion.py',
            ROOT/'horizyn/config_validation.py', ROOT/'horizyn/protein_pooling_lightning_module.py',
            ROOT/'horizyn/training_options.py',
            ROOT/'horizyn/enzyme_multiview.py',
            ROOT/'horizyn/positive_bio.py', ROOT/'horizyn/positive_bio_loss.py',
            ROOT/'horizyn/capability/positive_biological_targets.py',
            ROOT/'horizyn/capability/biological_targets.py',
            ROOT/'horizyn/capability/enzyme_capability_dataset.py',
            ROOT/'scripts/train_protein_pooling.py', ROOT/'scripts/evaluate_protein_pooling.py']
    manifest = dict(variants=variants, epochs=args.epochs, enzyme_tower=args.enzyme_tower,
        positive_biology=args.positive_biology,
        biofp_aux_weight=args.biofp_aux_weight if args.positive_biology else 0,
        global_batch_size=1536, coverage=coverage,
        cache_reused=True, validation_only=True, sources=[identity(p) for p in dict.fromkeys(sources)],
        code={str(p):sha(p) for p in code},
        configs={str(p.relative_to(output)):sha(p) for p in output.glob('*/configs/*.yaml')})
    atomic_json(output/'experiment.json', manifest)
    print(json.dumps({'output': str(output), 'variants': variants, 'epochs': args.epochs,
                      'coverage': coverage, 'cache_reused': True}, indent=2), flush=True)
    return manifest


def verify(output, manifest):
    for source in manifest['sources']:
        if identity(source['path']) != source: raise ValueError(f'Input changed: {source["path"]}')
    for path, digest in manifest['code'].items():
        if sha(path) != digest: raise ValueError(f'Code changed: {path}')
    for path, digest in manifest['configs'].items():
        if sha(output/path) != digest: raise ValueError(f'Config changed: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['launch', 'prepare', 'run'])
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--variant', choices=[*VARIANTS, 'both'], default='feature')
    parser.add_argument('--enzyme-tower', choices=['baseline', 'multiview'], default='baseline',
        help='Keep the reference enzyme tower or use the new collapse-regularized multiview tower')
    parser.add_argument('--positive-biology', action='store_true',
        help='Train-only positive anchor supervision: EC, cofactor and coarse transformation; missing labels neutral')
    parser.add_argument('--biofp-aux-weight', type=float, default=0.02)
    parser.add_argument('--annotation-python', type=Path, default=ROOT.parent/'.capability-run-py/bin/python',
        help='CPU preparation interpreter with pandas/pyarrow; training keeps the launching interpreter')
    parser.add_argument('--ec-labels', type=Path, default=ANNOTATIONS/'ec/enzyme_ec_labels_train_only.csv')
    parser.add_argument('--matched-members', type=Path, default=ANNOTATIONS/'rhea_matches/matched_members.csv')
    parser.add_argument('--directional-features', type=Path,
        default=ANNOTATIONS/'rhea_matches/directional_features/reaction_features.parquet')
    parser.add_argument('--enzyme-cofactors', type=Path,
        default=ROOT/'data/processed/capability_features/train_exact_rhea_reconstructed/enzyme_cofactor_labels_enhanced.csv')
    parser.add_argument('--base-config', type=Path, default=BASE/'configs/train.yaml')
    parser.add_argument('--cache-run', type=Path, default=DEFAULT_CACHE_RUN)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.positive_biology and (args.enzyme_tower != 'multiview' or not math.isfinite(args.biofp_aux_weight) or args.biofp_aux_weight <= 0):
        parser.error('--positive-biology requires --enzyme-tower multiview and positive --biofp-aux-weight')
    gpus = args.gpus.split(',')
    if len(gpus) not in (1, 2, 3, 4) or len(gpus) != len(set(gpus)) or any(not x.isdigit() for x in gpus) or args.epochs <= 0:
        parser.error('Use 1-4 distinct GPU IDs and positive epochs')
    args.base_config = args.base_config.resolve(); args.cache_run = args.cache_run.resolve()
    if args.stage in ('launch', 'run'): check_gpus(gpus)
    if args.stage in ('launch', 'prepare'):
        if args.output:
            output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
        else:
            output = Path(tempfile.mkdtemp(prefix='circe_v3_feature_gate_', dir=ROOT/'runs'))
        if args.stage == 'prepare': prepare(args, output); return
        command = shlex.join([sys.executable, str(Path(__file__).resolve()), 'run',
            '--gpus', args.gpus, '--epochs', str(args.epochs), '--variant', args.variant,
            '--base-config', str(args.base_config), '--cache-run', str(args.cache_run), '--output', str(output)]
            + ['--enzyme-tower', args.enzyme_tower, '--biofp-aux-weight', str(args.biofp_aux_weight)]
            + (['--positive-biology', '--ec-labels', str(args.ec_labels.resolve()),
                '--annotation-python', str(args.annotation_python.absolute()),
                '--matched-members', str(args.matched_members.resolve()),
                '--directional-features', str(args.directional_features.resolve()),
                '--enzyme-cofactors', str(args.enzyme_cofactors.resolve())] if args.positive_biology else []))
        subprocess.run(['tmux', 'new-session', '-d', '-s', output.name, '-c', str(ROOT),
            '/usr/bin/env', '-u', 'BASH_ENV', '-u', 'ENV', '/bin/bash', '--noprofile', '--norc', '-c',
            f'exec {command} >> {shlex.quote(str(output/"pipeline.log"))} 2>&1'], check=True)
        print(f'Detached session: {output.name}\nLog: {output}/pipeline.log')
        return
    if args.output is None: parser.error('--output required for run')
    output = args.output.resolve()
    os.chdir(ROOT)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpus, CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/'experiment.json').exists(): raise ValueError('Existing run; no implicit restart/overwrite')
        atomic_json(output/'controller.json', dict(pid=os.getpid(), hostname=socket.gethostname(), run_root=str(output)))
        manifest = prepare(args, output)
        for variant in manifest['variants']:
            verify(output, manifest)
            folder = output/variant
            print(f'Training {variant} gate: {args.epochs} epochs, global batch 1536, fresh towers.', flush=True)
            with (folder/'training.log').open('a') as log:
                subprocess.run([sys.executable, str(ROOT/'scripts/train_protein_pooling.py'),
                    '--config', str(folder/'configs/train.yaml'), '--wandb-mode', 'disabled'],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            # A separate process releases training memory before evaluation/next variant.
            checkpoint = folder/'checkpoints/last.ckpt'
            report = folder/'evaluation'; report.mkdir()
            subprocess.run([sys.executable, str(ROOT/'scripts/evaluate_protein_pooling.py'),
                '--checkpoint', str(checkpoint), '--config', str(folder/'configs/validation.yaml'),
                '--device', 'cuda:0', '--direction', 'both', '--batch-size', '64', '--target-batch-size', '128',
                '--store-targets-on-cpu', '--evaluation-protocol', 'configured_forward_candidates',
                '--target-embeds-cache', str(report/'targets.pt'), '--per-query-output', str(report/'queries.json'),
                '--output', str(report/'metrics.json')], check=True)
        atomic_json(output/'complete.json', dict(variants=manifest['variants'], epochs=args.epochs, validation_only=True))
        print(f'Completed: {output}', flush=True)


if __name__ == '__main__':
    main()
