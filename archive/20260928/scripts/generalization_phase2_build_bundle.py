#!/usr/bin/env python3
"""Bind a frozen phase-two recipe to independent, train-fitted artifacts."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from horizyn.generalization_retrieval import checked_artifact,sha256
from horizyn.generalization_phase2 import validate_phase2_bundle
from generalization_full_graph import atomic_json


def identity(path):return dict(path=str(path.resolve()),sha256=sha256(path))


def build(args):
    if (args.output/'bundle.json').exists():raise ValueError('Choose a fresh immutable bundle directory')
    frozen=json.loads(args.freeze.read_text());parent=json.loads(args.base_bundle.read_text())
    density=json.loads(args.density_bundle.read_text())
    dictionary=torch.load(args.smooth_dictionary,map_location='cpu',weights_only=False)
    feature_hash=parent['feature_manifest_sha256']
    if dictionary['feature_manifest_sha256']!=feature_hash or density['feature_manifest_sha256']!=feature_hash:
        raise ValueError('Base, density and smooth training feature manifests disagree')
    if checked_artifact(parent['anchors'],args.base_bundle.parent).resolve()!=args.smooth_dictionary.resolve():
        raise ValueError('Smooth dictionary is not the audited parent training dictionary')
    checked_artifact(parent['frozen_recipe'],args.base_bundle.parent)
    alpha={'density_only':0.,'smooth_only':1.}.get(args.variant,frozen['composition']['alpha'])
    checkpoint=dict(parent['base_checkpoint'])
    checkpoint['sha256']=sha256(checkpoint['path'])
    spec=dict(schema='generalization_phase2_bundle_v1',variant=args.variant,alpha=alpha,split=args.split,
        seed={'seed17':17,'seed73':73}.get(args.variant,42),
        phase2_frozen_recipe=identity(args.freeze),frozen_recipe=parent['frozen_recipe'],
        density_bundle=identity(args.density_bundle),smooth_dictionary=identity(args.smooth_dictionary),
        smooth_config=frozen['smooth'],base_checkpoint=checkpoint,
        feature_manifest_sha256=feature_hash,parent_bundle=identity(args.base_bundle),
        input_policy=parent['input_policy'],latent_dimension=512+len(dictionary['train_reactions']),
        score='Independently encoded sqrt-weight concatenation; canonical FP64 dot rounded to FP32',
        baselines=dict(baseline='nativeF3 one FP32 normalization',baseline_fp64='nativeF3 one FP64 normalization rounded to FP32'),
        exploratory_after_phase1_external_failure=True,source_sha256=sha256(__file__))
    validate_phase2_bundle(spec,args.output)
    args.output.mkdir(parents=True,exist_ok=True);atomic_json(args.output/'bundle.json',spec)
    print(json.dumps(dict(bundle=str(args.output/'bundle.json'),variant=args.variant,alpha=alpha)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('density-bundle','smooth-dictionary','base-bundle','freeze','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--variant',choices=['primary','seed17','seed73','density_only','smooth_only'],default='primary')
    p.add_argument('--split',choices=['reaction_smi','enzyme_smi','time'],required=True)
    build(p.parse_args())


if __name__=='__main__':main()
