#!/usr/bin/env python3
"""Detached, fixed-checkpoint held-out test for the concat/gated experiment."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_circe_gated_fusion import MODEL, VARIANTS
from scripts.run_circe_generalization_diagnostics import check_gpus, sha
from horizyn.generalization_diagnostics import atomic_json, read_pairs


def test_config(base, output, pairs, reactions, features):
    config = copy.deepcopy(base)
    data = config['data']
    for prefix in ('test', 'validation'):
        data[f'{prefix}_pairs_path'] = str(pairs)
        data[f'{prefix}_reactions_path'] = str(reactions)
    for modality in ('t5v2', 'unimol2', 'chiro'):
        path = output/'reactiont5_participants.h5' if modality == 't5v2' else features/f'{modality}.h5'
        data[f'reaction_{modality}_embeds_path'] = str(path)
        data[f'validation_reaction_{modality}_embeds_path'] = str(path)
    for section in (data, config['training']):
        section['validation_retrieval_candidate_set'] = 'custom'
        section['validation_retrieval_candidate_ids_path'] = str(pairs.parent/'test_candidate_ids.txt')
    return config


def prepare(run, output):
    from horizyn.chemistry.reaction_recovery import components, read_reactions
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.warning')
    completed = json.loads((run/'complete.json').read_text())
    if completed['variants'] != list(VARIANTS):
        raise ValueError('Both training variants must be complete')
    split = ROOT/'data/revised_protocols/reactzyme_paper/reaction_smi'
    features = ROOT/'data/revised_protocols/reactzyme_official/features/reaction_smi/test'
    pairs, reactions = split/'test_pairs.csv', split/'test_rxns.csv'
    queries = read_reactions(reactions)
    positives = read_pairs(pairs)
    if not {q for q, _ in positives} <= queries.keys():
        raise ValueError('Test pairs reference missing reaction rows')
    inputs = output/'t5_inputs.csv'
    with inputs.open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['reaction_id', 'reaction_smiles'])
        writer.writerows((q+'_f', '.'.join(sorted(components(s)))) for q,s in sorted(queries.items()))
    checkpoints = {}
    for variant in VARIANTS:
        folder = output/variant
        folder.mkdir()
        base = yaml.safe_load((run/variant/'configs/train.yaml').read_text())
        if set(positives) & set(read_pairs(base['data']['train_pairs_path'])):
            raise ValueError('Train/test positive pair overlap')
        config = test_config(base, output, pairs, reactions, features)
        (folder/'test.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        checkpoint = run/variant/'checkpoints/last.ckpt'
        checkpoints[variant] = dict(path=str(checkpoint), sha256=sha(checkpoint))
    manifest = dict(split='reaction_smi', subset='held_out_test', steps=completed['steps'],
        protocol='paper_test_candidates', checkpoints=checkpoints, reactions=len(queries),
        pairs=len(positives), proteins=len({p for _,p in positives}),
        inputs_sha256=sha(inputs), pairs_sha256=sha(pairs),
        model=json.loads((run/'experiment.json').read_text()).get('t5_model', str(MODEL)),
        t5_policy='canonical sorted participants; no pseudo-reactions; max_length=512')
    atomic_json(output/'manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def execute(run, output):
    import h5py
    import numpy as np
    manifest = prepare(run, output)
    temporary = output/'reactiont5_participants.partial.h5'
    subprocess.run([sys.executable, str(ROOT/'scripts/extract_reaction_t5v2_embeddings.py'),
        '--reactions', str(output/'t5_inputs.csv'), '--output', str(temporary),
        '--model-name', manifest['model'], '--batch-size', '32', '--device', 'cuda:0',
        '--max-length', '512', '--dtype', 'float32', '--no-bidirectional',
        '--no-allow-pseudo-reactions', '--no-standardize'], check=True)
    with (output/'t5_inputs.csv').open() as handle:
        expected = {r['reaction_id'] for r in csv.DictReader(handle)}
    with h5py.File(temporary) as store:
        ids = [v.decode() if isinstance(v,bytes) else str(v) for v in store['ids'][:]]
        vectors = store['vectors'][:]
        if len(ids)!=len(set(ids)) or set(ids)!=expected or vectors.shape!=(len(ids),768) or not np.isfinite(vectors).all():
            raise ValueError('Incomplete/nonfinite test ReactionT5 cache')
    temporary.rename(output/'reactiont5_participants.h5')
    summary = {}
    for variant, checkpoint in manifest['checkpoints'].items():
        if sha(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Checkpoint changed during test')
        folder = output/variant
        print(f'Evaluating held-out test: {variant}', flush=True)
        subprocess.run([sys.executable, str(ROOT/'scripts/evaluate_protein_pooling.py'),
            '--checkpoint', checkpoint['path'], '--config', str(folder/'test.yaml'),
            '--device', 'cuda:0', '--batch-size', '64', '--target-batch-size', '128',
            '--store-targets-on-cpu', '--direction', 'both',
            '--evaluation-protocol', 'paper_test_candidates',
            '--target-embeds-cache', str(folder/'targets.pt'),
            '--per-query-output', str(folder/'queries.json'), '--output', str(folder/'metrics.json')], check=True)
        metrics = json.loads((folder/'metrics.json').read_text())
        summary[variant] = {k:v for k,v in metrics.items() if 'mrr' in k or k.endswith('/top_1')}
    atomic_json(output/'complete.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['launch', 'run', 'prepare'])
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run = args.run_root.resolve()
    if args.stage == 'prepare':
        if args.output is None: parser.error('--output required')
        args.output.mkdir(parents=True, exist_ok=False)
        prepare(run, args.output.resolve())
        return
    check_gpus([args.gpu])
    if args.stage == 'launch':
        output = Path(tempfile.mkdtemp(prefix='heldout_test_', dir=run))
        session = 'circe_gated_test_'+output.name.removeprefix('heldout_test_')
        command = shlex.join([sys.executable, str(Path(__file__).resolve()), 'run',
            '--run-root', str(run), '--output', str(output), '--gpu', args.gpu])
        env = os.environ.copy()
        env.pop('BASH_ENV', None); env.pop('ENV', None)
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(ROOT),
            '/usr/bin/env', '-u', 'BASH_ENV', '-u', 'ENV', '/bin/bash', '--noprofile', '--norc', '-c',
            f'exec {command} >> {shlex.quote(str(output/"pipeline.log"))} 2>&1'], check=True, env=env)
        print(f'Detached session: {session}\nLog: {output}/pipeline.log')
        return
    if args.output is None: parser.error('--output required')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID',
        PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    os.chdir(ROOT)
    execute(run, args.output.resolve())


if __name__ == '__main__':
    main()
