#!/usr/bin/env python3
"""Bounded fresh-training queue with fixed early ReactZyme test readouts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, payload):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def free_memory():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    return {int(i): int(m) for i, m in (line.split(',') for line in out.splitlines())}


def cache_ready(plan):
    path = Path(plan['residue_cache'])
    receipt_path = path.with_suffix('.receipt.json')
    if not path.exists() or not receipt_path.exists():
        return False
    receipt = json.loads(receipt_path.read_text())
    stat = path.stat()
    if (receipt['output_size_bytes'] != stat.st_size or
            receipt['output_mtime_ns'] != stat.st_mtime_ns or
            receipt['catalog_sha256'] != sha(plan['residue_catalog'])):
        raise ValueError('ReactZyme residue cache receipt mismatch')
    source = receipt['source_identity']
    source_stat = Path(source['path']).stat()
    if (source['size'], source['mtime_ns']) != (source_stat.st_size, source_stat.st_mtime_ns):
        raise ValueError('ReactZyme source changed since cache materialization')
    return True


def launch(script, arguments, log_path, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    with log_path.open('a') as log:
        return subprocess.Popen([sys.executable, '-u', str(ROOT / 'scripts' / script),
                                 *map(str, arguments)], cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=8)
    args = parser.parse_args()
    plan = json.loads(args.protocol.read_text())
    out = args.protocol.parent
    # Freeze inputs before training; never resume a partially trained run silently.
    for task in plan['tasks']:
        for kind in ('train', 'test'):
            key = kind + '_config'
            if key in task and sha(task[key]) != task[key + '_sha256']:
                raise ValueError(f'Changed configuration: {task[key]}')
    for path, digest in plan['code_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f'Changed implementation: {path}')
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    active, failed, finished = {}, {}, set()
    cached = False
    started = time.monotonic()
    try:
        while not stopping and time.monotonic() - started < args.hours * 3600:
            for key, (process, receipt) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                if code == 0:
                    write(receipt, dict(completed_utc=utc(), exit_code=0))
                    finished.add(key)
                else:
                    failed[key] = dict(exit_code=code, utc=utc())
                del active[key]
            cached = cached or cache_ready(plan)
            memory = free_memory()
            waiting = []
            for task in plan['tasks']:
                run = Path(task['run_root'])
                key = task['benchmark'] + '/' + task['arm']
                train_key = key + '/train'
                completion = run / 'train_complete.json'
                if completion.exists():
                    finished.add(train_key)
                if train_key in active or train_key in failed or train_key in finished:
                    continue
                if task['benchmark'].startswith('reactzyme') and not cached:
                    waiting.append(dict(task=train_key, reason='local residue cache'))
                    continue
                busy = any(k.endswith('/train') and
                           next(t['gpu'] for t in plan['tasks'] if k.startswith(t['benchmark'] + '/' + t['arm'] + '/')) == task['gpu']
                           for k in active)
                required = 75000 if task['benchmark'] == 'enzymemap' else 30000
                if busy or memory[task['gpu']] < required:
                    waiting.append(dict(task=train_key, reason='GPU capacity'))
                    continue
                start_path = run / 'train_started.json'
                if start_path.exists():
                    failed[train_key] = dict(error='Prior incomplete launch; explicit resume required')
                    continue
                process = launch('train_protein_pooling.py', ['--config', task['train_config']],
                                 run / 'console.log', task['gpu'])
                write(start_path, dict(started_utc=utc(), pid=process.pid, gpu=task['gpu'],
                                      config_sha256=task['train_config_sha256'], fresh=True))
                active[train_key] = (process, completion)
                memory[task['gpu']] -= required
                print(json.dumps(dict(starting=train_key, pid=process.pid)), flush=True)
            # One independent evaluator at a time; test snapshots fixed in the plan.
            evaluating = any('/test_epoch' in key for key in active)
            for task in plan['tasks']:
                if not task['benchmark'].startswith('reactzyme') or evaluating:
                    continue
                run = Path(task['run_root'])
                for epoch in plan['first_test_epochs'][task['benchmark']]:
                    key = task['benchmark'] + '/' + task['arm'] + f'/test_epoch{epoch}'
                    result = run / f'test_epoch{epoch}'
                    if (result / 'complete.json').exists():
                        finished.add(key)
                    if key in finished or key in failed or key in active:
                        continue
                    checkpoint = run / f'checkpoints/screen_selection/screen-epoch={epoch-1:02d}.ckpt'
                    if not checkpoint.exists() or time.time() - checkpoint.stat().st_mtime < 30:
                        continue
                    gpu = max(memory, key=memory.get)
                    if memory[gpu] < 16000:
                        continue
                    result.mkdir(exist_ok=True)
                    write(result / 'test_request.json', dict(requested_utc=plan['created_utc'],
                        started_utc=utc(), checkpoint_sha256=sha(checkpoint),
                        config_sha256=task['test_config_sha256'], epoch_one_based=epoch,
                        fixed_early_test=True, repeated_test_evaluation=True, phase2_applied=False,
                        selection_used_test_scores=False))
                    process = launch('evaluate_protein_pooling.py', [
                        '--checkpoint', checkpoint, '--config', task['test_config'],
                        '--device', 'cuda:0', '--batch-size', 128, '--target-batch-size', 256,
                        '--direction', 'both', '--evaluation-protocol', 'paper_test_candidates',
                        '--target-embeds-cache', result / 'targets.pt', '--output', result / 'metrics.json',
                        '--per-query-output', result / 'per_query.json'], result / 'evaluate.log', gpu)
                    active[key] = (process, result / 'complete.json')
                    evaluating = True
                    print(json.dumps(dict(starting=key, pid=process.pid)), flush=True)
                    break
            write(out / 'status.json', dict(updated_utc=utc(), state='running',
                active={k: p.pid for k, (p, _) in active.items()}, finished=sorted(finished),
                failures=failed, waiting=waiting, reactzyme_cache_ready=cached,
                enzymemap_evaluation='Separate fixed full-library validation and test queues'))
            required_count = len(plan['tasks']) + sum(len(plan['first_test_epochs'][t['benchmark']])
                           for t in plan['tasks'] if t['benchmark'].startswith('reactzyme'))
            if len(finished) + len(failed) == required_count:
                return
            time.sleep(15)
    finally:
        for process, _ in active.values():
            if process.poll() is None:
                process.terminate()
        write(out / 'queue_exit.json', dict(utc=utc(), stopped_by_signal=stopping,
                                           finished=sorted(finished), failures=failed))


if __name__ == '__main__':
    main()
