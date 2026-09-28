#!/usr/bin/env python3
"""Dispatch the frozen exploratory second-phase predictions across GPUs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horizyn.generalization_retrieval import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    args = p.parse_args()
    campaign = args.campaign.resolve()
    phase = campaign / 'phase2'
    freeze_path = phase / 'frozen_recipe.json'
    frozen = json.loads(freeze_path.read_text())
    if not frozen.get('frozen_before_phase2_prediction'):
        raise ValueError('A complete second-phase freeze is required')
    for record in frozen['implementation_sources']:
        if sha256(record['path']) != record['sha256']:
            raise ValueError(f"Implementation changed after freeze: {record['path']}")
    panels = {}
    for split in ('reaction_smi', 'time', 'enzyme_smi'):
        source = campaign / f'features_test_{split}'
        panels[split] = dict(split=split, base=source/'f3_features.npz',
            protein_means=source/'protein_mean.h5', reaction_features=source/'reaction_features.npz',
            catalog=source/'catalog.json', reaction_key='reactions')
    for name in ('case1', 'p450', 'nitrilase'):
        source = campaign / f'{name}_audit/features'
        panels[name] = dict(split='reaction_smi', base=source/'f3_epoch29/features.npz',
            protein_means=source/'raw/protein_mean.h5', reaction_features=source/'raw/reaction_features.npz',
            catalog=source/'catalog.json', reaction_key='query_matched' if name == 'case1' else 'reactions')
    pending = queue.Queue()
    for panel, spec in panels.items():
        for variant in ('seed42', 'seed17', 'seed73', 'density_only', 'smooth_only'):
            pending.put(dict(panel=panel, variant=variant,
                bundle=str(phase/'models'/spec['split']/variant/'bundle.json'),
                output=str(phase/'predictions'/panel/variant),
                **{key:str(value) for key,value in spec.items()}))
    started = time.monotonic()
    results = []
    def worker(gpu):
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            destination = Path(job['output'])
            destination.mkdir(parents=True, exist_ok=True)
            receipt_path = destination/'complete.json'
            try:
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                    if receipt['phase2_frozen_recipe']['sha256'] != sha256(freeze_path):
                        raise ValueError('Completed prediction freeze mismatch')
                    if receipt['bundle']['sha256'] != sha256(job['bundle']):
                        raise ValueError('Completed prediction bundle mismatch')
                    for key in ('base','catalog','protein_means','reaction_features'):
                        if receipt['inputs'][key]['sha256'] != sha256(job[key]):
                            raise ValueError(f'Completed prediction input changed: {key}')
                    if receipt['output_sha256'] != sha256(destination/'scores.npz'):
                        raise ValueError('Completed prediction score hash mismatch')
                    result = dict(**job, status='reused', gpu=gpu)
                else:
                    command = [sys.executable, str(ROOT/'scripts/generalization_phase2_predict.py'),
                        '--bundle',job['bundle'],'--output',str(destination),
                        '--device',f'cuda:{gpu}','--reaction-key',job['reaction_key']]
                    for key in ('base','catalog','protein_means','reaction_features'):
                        command.extend(['--'+key.replace('_','-'),job[key]])
                    if job['variant'] == 'seed42':
                        command.append('--save-embeddings')
                    (destination/'command.json').write_text(json.dumps(command, indent=2)+'\n')
                    with (destination/'run.log').open('w') as log:
                        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                    result = dict(**job, status='complete', gpu=gpu)
            except Exception as error:
                result = dict(**job, status='failed', gpu=gpu, error=str(error))
            results.append(result)
            print(json.dumps({key:result[key] for key in ('panel','variant','status','gpu')}), flush=True)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        list(pool.map(worker,args.gpus))
    index = dict(schema='phase2_prediction_index_v1',
        freeze=dict(path=str(freeze_path),sha256=sha256(freeze_path)),
        jobs=results,elapsed_seconds=time.monotonic()-started,source_sha256=sha256(__file__))
    (phase/'prediction_index.json').write_text(json.dumps(index,indent=2)+'\n')
    if any(row['status']=='failed' for row in results):
        raise SystemExit('Some phase-two jobs failed; inspect per-job logs')


if __name__ == '__main__':
    main()
