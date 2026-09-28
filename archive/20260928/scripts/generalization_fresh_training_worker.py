#!/usr/bin/env python3
"""Run one fresh, hashed benchmark config with a recorded resource admission check."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from generalization_gpu_budget import free_memory_mib
from generalization_screen_replication import ROOT, sha
from generalization_full_graph import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--task', required=True)
    a = p.parse_args()
    plan = json.loads((a.campaign / 'protocol.json').read_text())
    task = next(t for t in plan['tasks'] if t['name'] == a.task)
    run = Path(task['run_root'])
    if sha(task['config']) != task['config_sha256']:
        raise ValueError('Frozen training config changed')
    if (run / 'training').exists():
        raise ValueError('Fresh training requires an unused output directory')
    atomic_json(run / 'state.json', dict(stage='waiting_for_training_memory', gpu=task['gpu']))
    while free_memory_mib(task['gpu']) < 100000:
        time.sleep(5)
    command = [sys.executable, '-u', str(ROOT / 'scripts/train_protein_pooling_fast_io.py'),
               '--config', task['config'], '--io-mode', 'residue', '--io-output-dir', str(run / 'training'),
               '--io-prefetch', '1']
    started = datetime.now(timezone.utc).isoformat()
    atomic_json(run / 'state.json', dict(stage='fresh_f3_training', started_utc=started, gpu=task['gpu']))
    with (run / 'training.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(task['gpu']), OMP_NUM_THREADS='4',
                     MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'), stdout=log, stderr=subprocess.STDOUT)
        atomic_json(run / 'trainer_execution.json', dict(pid=proc.pid, command=command, gpu=task['gpu'],
            started_utc=started, config_sha256=task['config_sha256'],
            protocol_sha256=sha(a.campaign / 'protocol.json'), worker_sha256=sha(__file__)))
        code = proc.wait()
    if code:
        atomic_json(run / 'failure.json', dict(exit_code=code, failed_utc=datetime.now(timezone.utc).isoformat()))
        atomic_json(run / 'state.json', dict(stage='failed', exit_code=code))
        raise SystemExit(code)
    checkpoint = run / f"checkpoints/screen_selection/screen-epoch={plan['epochs']-1:02d}.ckpt"
    atomic_json(run / 'train_complete.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        completed_utc=datetime.now(timezone.utc).isoformat(), exit_code=0))
    atomic_json(run / 'state.json', dict(stage='base_training_complete'))


if __name__ == '__main__':
    main()
