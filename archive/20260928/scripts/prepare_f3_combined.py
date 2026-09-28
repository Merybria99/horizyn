#!/usr/bin/env python3
"""Prepare a matched original/combined F3 pilot without editing shared code."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import pandas as pd
import torch
import yaml


def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def build_pools(pairs, cluster_path, seed):
    members, protein_cluster, positives = defaultdict(set), {}, defaultdict(set)
    proteins = set(pairs.protein_id)
    for line in cluster_path.read_text().splitlines():
        cluster, protein = line.split('\t')
        if protein not in proteins:
            raise ValueError('Non-training protein in competitor clustering')
        if protein in protein_cluster:
            raise ValueError('Duplicate cluster membership')
        members[cluster].add(protein); protein_cluster[protein] = cluster
    if set(protein_cluster) != proteins:
        raise ValueError('Incomplete training protein clustering')
    for q, p in pairs[['reaction_id', 'protein_id']].itertuples(index=False, name=None):
        positives[q].add(p)
    rng = random.Random(seed)
    pools = {}
    for query, known in sorted(positives.items()):
        candidates = set()
        for cluster in {protein_cluster[p] for p in known}:
            candidates.update(members[cluster])
        candidates.difference_update(known)
        candidates = sorted(candidates)
        pools[query] = sorted(rng.sample(candidates, min(128, len(candidates))))
        if set(pools[query]) & known:
            raise ValueError('Known positive in competitor pool')
    nonempty = sum(bool(x) for x in pools.values())
    if nonempty == 0:
        raise ValueError('No homolog competition available')
    return pools, dict(training_proteins=len(proteins), training_reactions=len(positives),
        homology_groups=len(members), reactions_with_competitors=nonempty,
        competitor_counts_quantiles=np.quantile([len(v) for v in pools.values()], [0, .25, .5, .75, 1]).tolist(),
        interpretation='Training-only sequence homology proxy for family; nonedges are unlabelled, not assayed inactive')


def adapt_directional_weights(module, source_state):
    """Transfer every learned model tensor; explicitly expand two input matrices.

    [R,P,P-R,abs(P-R)] starts with the old W applied to (R+P)/2.
    Difference blocks start at zero but have trainable, nonzero input gradients.
    This preserves learned projections, not the old reactant-only predictions.
    """
    target = module.state_dict()
    changes = []
    for key, dest in target.items():
        if not key.startswith('model.'):
            continue
        if key not in source_state:
            raise ValueError(f'Unexpected new model tensor: {key}')
        old = source_state[key]
        if old.shape == dest.shape:
            target[key] = old.clone()
        elif ('unimol_projection' in key or 'chienn_projection' in key) and old.ndim == 2 and dest.shape == (old.shape[0], 4*old.shape[1]):
            value = torch.zeros_like(dest)
            width = old.shape[1]
            value[:, :width] = .5 * old
            value[:, width:2*width] = .5 * old
            target[key] = value
            changes.append(dict(key=key, old=list(old.shape), new=list(dest.shape), rule='[W/2,W/2,0,0]'))
        else:
            raise ValueError(f'Unapproved warm-start shape change: {key}, {old.shape}, {dest.shape}')
    if len(changes) != 2:
        raise ValueError(f'Expected exactly two directional projection expansions, found {changes}')
    # Old product poolers received no molecular-branch training signal.
    copied = []
    for key in list(target):
        if key.startswith('model.query_encoder.') and 'product_pooling' in key:
            source_key = key.replace('product_pooling', 'reactant_pooling')
            if source_key not in target or target[source_key].shape != target[key].shape:
                raise ValueError(f'Cannot initialize product pooler: {key}')
            target[key] = target[source_key].clone(); copied.append(key)
    if len(copied) != 4:
        raise ValueError(f'Expected four product-pooling tensors, got {copied}')
    module.load_state_dict(target, strict=True)
    return dict(expanded=changes, product_pooler_tensors_copied=copied,
        note='New directional representation deliberately changes initial predictions; other learned tensors are transferred exactly.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path('runs/enzymecage_f3_combined_20260918'))
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    root=args.root.resolve(); out=(root/args.output).resolve()
    if (out/'manifest.json').exists():
        raise FileExistsError('Prepared run already exists; use it or select a fresh output')
    out.mkdir(parents=True, exist_ok=True)
    base_path=root/'runs/cyp_enzymecage_f3_epoch03/snapshot/train.yaml'
    base=yaml.safe_load(base_path.read_text())
    train_path=Path(base['data']['train_pairs_path'])
    pairs=pd.read_csv(train_path)
    pools, pool_audit=build_pools(pairs, out/'data/train_homology_cluster.tsv', 42)
    pools_path=out/'data/homolog_competitors.json'
    write_json(pools_path, {'anchor_to_negatives': pools})
    write_json(out/'data/sampling_audit.json', pool_audit)
    # A frozen source snapshot protects this pilot from other ongoing work.
    snapshot=out/'code'
    shutil.copytree(root/'horizyn', snapshot/'horizyn', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (snapshot/'scripts').mkdir()
    shutil.copy2(root/'scripts/train_protein_pooling.py',snapshot/'scripts/train_protein_pooling.py')
    shutil.copy2(Path(__file__).with_name('f3_combined_sampler.py'),snapshot/'horizyn/f3_combined_sampler.py')
    data_source=snapshot/'horizyn/reaction_conditioned_data_module.py'
    source=data_source.read_text()
    old='ReactionHardNegativeBatchSampler = DirectionalHardNegativeBatchSampler'
    if source.count(old)!=1:
        raise ValueError('Sampler integration point changed')
    source=source.replace(old, 'from horizyn.f3_combined_sampler import extend_sampler\nDirectionalHardNegativeBatchSampler = extend_sampler(DirectionalHardNegativeBatchSampler)\n'+old)
    data_source.write_text(source)
    configurations={}
    checkpoint=root/'runs/enzymecage_f3_seed42/checkpoints/protein-pooling-epoch=06.ckpt'
    for name in ['control', 'combined']:
        cfg=copy.deepcopy(base)
        run=out/name
        (run/'configs').mkdir(parents=True)
        cfg['data'].update(train_batch_size=args.batch_size, num_workers=0)
        cfg['training'].update(devices=1, strategy='auto', accelerator='gpu', max_steps=args.steps,
            max_epochs=100, cpu_num_threads=4, use_distributed_sampler=False,
            init_from_checkpoint=str(checkpoint), validation_interval_steps=args.steps,
            num_sanity_val_steps=0, learning_rate=1e-4)
        cfg['training']['early_stopping']={'enabled': False}
        cfg['logging'].update(log_dir=str(run/'logs'),checkpoint_dir=str(run/'checkpoints'),
            recovery_every_n_train_steps=50, checkpoint_on_validation_end=True, log_every_n_steps=10)
        cfg['logging']['wandb']={'enabled':False,'mode':'disabled'}
        cfg['ablation']={'variant':name,'parent':'EnzymeCAGE_F3_epoch06','purpose':'fixed-budget combined pilot',
            'external_test_used_for_selection':False,'optimizer_state':'reset identically in both arms'}
        if name=='combined':
            cfg['model']['reaction_multimodal_attention']['side_composition']='directional_delta'
            cfg['training']['loss'].update(name='DecoupledAllPositiveInfoNCELoss',positive_pair_source='all_known_in_batch',
                lambda_r2e=1.0,lambda_e2r=0.0,unknown_negative_weight=0.5)
            cfg['data'].update(hard_negative_pools_path=str(pools_path),hard_negative_direction='reaction_to_enzyme',
                hard_negative_anchor_queries_per_batch=8,hard_negative_positives_per_query=2,
                hard_negative_negatives_per_query=6,hard_negative_seed=42)
        configurations[name]=cfg
    sys.path.insert(0,str(snapshot))
    from horizyn.config import DotDict
    from horizyn.training_options import protein_pooling_model_kwargs
    from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
    torch.set_num_threads(4);torch.manual_seed(42)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']
    combined=ProteinPooledLitModule(**protein_pooling_model_kwargs(DotDict(configurations['combined'])))
    adaptation=adapt_directional_weights(combined,state)
    init_path=out/'combined/initialization.ckpt'
    torch.save({'state_dict':combined.state_dict(),'parent_sha256':digest(checkpoint),'adaptation':adaptation},init_path)
    configurations['combined']['training']['init_from_checkpoint']=str(init_path)
    for name,cfg in configurations.items():
        path=out/name/'configs/train.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        # Validate the persisted configuration, not only the in-memory model.
        from horizyn.config import load_config
        load_config(path)
    manifest=dict(steps=args.steps,batch_size=args.batch_size,seed=42,selection='fixed update budget, no external test tuning',
        common_checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),base_config_sha256=digest(base_path),
        training_pairs_sha256=digest(train_path),clustering_sha256=digest(out/'data/train_homology_cluster.tsv'),
        pool_sha256=digest(pools_path),pool_audit=pool_audit,warm_start=adaptation,initialization_sha256=digest(init_path),
        configs={name:digest(out/name/'configs/train.yaml') for name in configurations},
        code_sha256={str(p.relative_to(snapshot)):digest(p) for p in snapshot.rglob('*.py')},
        limitations=['Combined intervention; this pilot cannot attribute any improvement to one component.',
            'Single GPU and batch128 differs from original 4-GPU global batch2048; matched across pilot arms.',
            'Homology clusters are a sequence-similarity proxy, not proof of identical enzyme family or inactivity.',
            'Unknown-negative weight 0.5 is prespecified, not calibrated or selected on P450.',
            'Original internal validation remains the development set; P450 is not used for selection.'])
    write_json(out/'manifest.json',manifest)
    print(json.dumps({'prepared':str(out),'sampling':pool_audit,'adaptation':adaptation},indent=2))


if __name__=='__main__':
    main()
