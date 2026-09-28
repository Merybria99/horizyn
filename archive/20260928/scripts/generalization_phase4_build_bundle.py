#!/usr/bin/env python3
"""Prepare train-only native reaction dictionaries, then bind frozen phase4 models."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from horizyn.generalization_phase2 import validate_phase2_bundle
from horizyn.generalization_phase4 import validate_phase4_bundle
from horizyn.generalization_retrieval import checked_artifact,sha256
from horizyn.semantic_anchors import row_unit
from generalization_full_graph import atomic_json


def identity(path):return dict(path=str(path.resolve()),sha256=sha256(path))


def fit_state(args):
    if args.output.exists():raise ValueError('Native state exists; use a fresh path')
    parent=json.loads(args.phase2_bundle.read_text());validate_phase2_bundle(parent,args.phase2_bundle.parent)
    manifest=sha256(args.features/'manifest.json')
    if parent['split']!=args.split or parent['feature_manifest_sha256']!=manifest:raise ValueError('Native cache and frozen parent training lineage disagree')
    catalog=json.loads((args.features/'catalog.json').read_text())
    with np.load(args.features/'pairs.npz') as f:train=f['train']
    ids=[catalog['reactions'][i] for i in np.unique(train[:,0])]
    if ids!=catalog['train_reactions']:raise ValueError('Native compact reaction catalog is not the unique training graph order')
    dictionary=torch.load(checked_artifact(parent['smooth_dictionary']),map_location='cpu',weights_only=False)
    if ids!=dictionary['train_reaction_ids']:raise ValueError('Raw/native training reaction dictionary order mismatch')
    with np.load(args.features/'f3_features.npz') as f:raw=torch.tensor(f['train_reactions'])
    if raw.shape!=(len(ids),512) or raw.dtype!=torch.float32 or not torch.isfinite(raw).all() or (raw.double().norm(dim=1)<=1e-8).any():
        raise ValueError('Native training reaction endpoints must be finite nonzero FP32 N by512')
    state=dict(schema='phase4_native_reaction_state_v1',split=args.split,train_reaction_ids=ids,native_train_reactions=row_unit(raw),
        feature_manifest_sha256=manifest,base_checkpoint=parent['base_checkpoint'],fit_data='Only native embeddings of unique training reaction nodes; no validation or external feature rows',
        normalization='FP64 norm then originalFP32; no fitted centering',
        sources={name:identity(args.features/name) for name in ['manifest.json','catalog.json','pairs.npz','f3_features.npz']},
        script=identity(Path(__file__)))
    args.output.parent.mkdir(parents=True,exist_ok=True);torch.save(state,args.output)
    atomic_json(args.output.with_suffix('.json'),dict(schema=state['schema'],split=args.split,train_reactions=len(ids),state=identity(args.output),sources=state['sources'],base_checkpoint=state['base_checkpoint'],feature_manifest_sha256=manifest,external_evaluation_performed=False))
    print(json.dumps(dict(state=str(args.output),sha256=sha256(args.output),train_reactions=len(ids))),flush=True)


def build(args):
    if (args.output/'bundle.json').exists():raise ValueError('Use a fresh immutable bundle path')
    previous=json.loads(args.phase2_bundle.read_text());validate_phase2_bundle(previous,args.phase2_bundle.parent)
    frozen=json.loads(args.freeze.read_text());state=torch.load(args.native_state,map_location='cpu',weights_only=False)
    seed={'primary':42,'seed17':17,'seed73':73,'hybrid_anchor_only':42}[args.variant]
    if previous['seed']!=seed or previous['split']!=args.split or state['split']!=args.split or previous['feature_manifest_sha256']!=state['feature_manifest_sha256']:
        raise ValueError('Variant/native state/parent split-seed-training lineage mismatch')
    alpha=1. if args.variant=='hybrid_anchor_only' else .25
    spec=dict(schema='phase4_hybrid_bundle_v1',split=args.split,seed=seed,variant=args.variant,alpha=alpha,eta_enzyme=0.,eta_reaction=.5,
        phase4_frozen_recipe=identity(args.freeze),phase2_frozen_recipe=previous['phase2_frozen_recipe'],frozen_recipe=previous['frozen_recipe'],
        phase2_bundle=identity(args.phase2_bundle),native_reaction_state=identity(args.native_state),base_checkpoint=previous['base_checkpoint'],
        feature_manifest_sha256=previous['feature_manifest_sha256'],input_policy=previous['input_policy'],latent_dimension=previous['latent_dimension'],
        score='Independent sqrt-weight endpoints; native/raw reaction dictionary cosine at equal weight; FP64 dot thenFP32',
        baselines=previous['baselines'],exploratory_after_phase1_and_phase2_external_failure=True,source_sha256=sha256(__file__))
    validate_phase4_bundle(spec,args.output)
    args.output.mkdir(parents=True,exist_ok=True);atomic_json(args.output/'bundle.json',spec)
    print(json.dumps(dict(bundle=str(args.output/'bundle.json'),variant=args.variant,split=args.split,seed=seed)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    fit=sub.add_parser('fit-state');bundle=sub.add_parser('bundle')
    for parser in [fit,bundle]:
        parser.add_argument('--phase2-bundle',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
        parser.add_argument('--split',choices=['reaction_smi','enzyme_smi','time'],required=True)
    fit.add_argument('--features',type=Path,required=True)
    for name in ['native-state','freeze']:bundle.add_argument('--'+name,type=Path,required=True)
    bundle.add_argument('--variant',choices=['primary','seed17','seed73','hybrid_anchor_only'],default='primary')
    args=p.parse_args();fit_state(args) if args.command=='fit-state' else build(args)


if __name__=='__main__':main()
