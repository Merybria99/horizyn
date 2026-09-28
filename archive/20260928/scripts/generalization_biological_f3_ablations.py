#!/usr/bin/env python3
"""Predeclare and run EnzymeMap biological supervision controls from scratch."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from generalization_full_graph import atomic_json, sha


def prepare(source, out):
    source_plan = json.loads((source / 'protocol.json').read_text())
    source_task = next(t for t in source_plan['tasks'] if t['name'] == 'enzymemap')
    tasks = []
    for gpu, variant in enumerate(('without_ec', 'without_cofactor', 'without_mechanism', 'shuffled')):
        campaign = out / variant; run = campaign / 'enzymemap'
        (run / 'configs').mkdir(parents=True, exist_ok=True)
        annotation_source = Path(source_task['annotations'])
        payload = json.loads(annotation_source.read_text())
        if variant.startswith('without_'):
            family = variant.removeprefix('without_')
            for endpoint in ('reaction', 'enzyme'):
                for key in ('rows', 'groups', 'confidence'):
                    payload['endpoints'][endpoint][family][key] = []
                payload['endpoints'][endpoint][family]['coverage'] = 0
                payload['endpoints'][endpoint][family]['annotated_memberships'] = 0
        else:
            for endpoint, key in (('reaction', 'reaction_ids'), ('enzyme', 'enzyme_ids')):
                permutation = torch.randperm(len(payload[key]), generator=torch.Generator().manual_seed(1701)).tolist()
                for family in ('ec', 'cofactor', 'mechanism'):
                    block = payload['endpoints'][endpoint][family]
                    block['rows'] = [permutation[i] for i in block['rows']]
        payload['ablation'] = dict(variant=variant, source=str(annotation_source.resolve()),
            source_sha256=sha(annotation_source), seed=1701 if variant == 'shuffled' else None,
            fixed_family_denominator=3, train_associations_unchanged=True)
        labels = run / 'annotations.json'; atomic_json(labels, payload)
        config = yaml.safe_load(Path(source_task['config']).read_text())
        config['logging']['log_dir'] = str(run / 'unused/logs')
        config['logging']['checkpoint_dir'] = str(run / 'unused/checkpoints')
        config['training']['biological_geometry']['path'] = str(labels)
        config_path = run / 'configs/train.yaml'; config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        (run / 'checkpoints').symlink_to(run / 'training/checkpoints', target_is_directory=True)
        task = dict(name='enzymemap', gpu=gpu, config=str(config_path), run_root=str(run), annotations=str(labels))
        plan = dict(source_plan, tasks=[task], annotation_variant=variant,
            parent_protocol_sha256=sha(source / 'protocol.json'),
            created_utc=datetime.now(timezone.utc).isoformat(),
            purpose='Matched biological contribution audit, all configurations reported without selection')
        atomic_json(campaign / 'protocol.json', plan)
        tasks.append(dict(campaign=str(campaign), gpu=gpu, variant=variant))
    atomic_json(out / 'protocol.json', dict(tasks=tasks, created_utc=datetime.now(timezone.utc).isoformat(),
        source_campaign=str(source), selection=False, source_sha256=sha(__file__)))


def run(out, variant):
    campaign = out / variant; task = json.loads((campaign / 'protocol.json').read_text())['tasks'][0]
    dest = Path(task['run_root']); gpu = task['gpu']
    from generalization_gpu_budget import free_memory_mib
    # A residue-training fit peaks near 21 GiB here. Leave room for active fits
    # before starting another, rather than allocating unused VRAM artificially.
    while free_memory_mib(gpu) < 35000:
        print(json.dumps(dict(waiting_for_training_memory=True, gpu=gpu)), flush=True)
        time.sleep(10)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    command = [sys.executable, '-u', str(ROOT / 'scripts/train_protein_pooling_fast_io.py'),
        '--config', task['config'], '--io-mode', 'residue', '--io-output-dir', str(dest / 'training'), '--io-prefetch', '1']
    with (dest / 'training.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        atomic_json(dest / 'training_execution.json', dict(pid=process.pid, gpu=gpu, command=command,
            created_utc=datetime.now(timezone.utc).isoformat()))
        code = process.wait()
    if code:
        raise RuntimeError(f'Training failed with exit code {code}; artifacts preserved')
    # Clear the local restriction: the follower exports four disjoint shards on four GPUs.
    env.pop('CUDA_VISIBLE_DEVICES', None)
    with (dest / 'followup.log').open('a') as log:
        subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/generalization_biological_f3_followup.py'),
            '--campaign', str(campaign), '--task', 'enzymemap'], cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant')
    a = p.parse_args(); out = a.output.resolve()
    if a.variant:
        run(out, a.variant)
    else:
        if out.exists():
            raise FileExistsError(out)
        out.mkdir(parents=True)
        prepare(a.source.resolve(), out)


if __name__ == '__main__':
    main()
