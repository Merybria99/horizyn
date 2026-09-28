#!/usr/bin/env python3
"""Memory-aware execution of the frozen residue-view ablation matrix.

This is an execution amendment only. The original runner and scientific
configuration hashes remain unchanged. Workers retain its full evaluation path.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import generalization_residue_view_ablation as base


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(record):
    return identity(record['pid']) == record['start_ticks']


def cache_ready(cache):
    path = Path(cache['output'])
    receipt = path.with_suffix('.receipt.json')
    if not path.is_file() or not receipt.is_file():
        return False
    r, stat = base.read(receipt), path.stat()
    if (r['output_size_bytes'], r['output_mtime_ns'], r['catalog_sha256']) != (
            stat.st_size, stat.st_mtime_ns, cache['catalog_sha256']):
        raise ValueError(f"Cache receipt changed: {cache['name']}")
    return True


def devices():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    return {int(a): dict(uuid=b.strip(), free_mib=int(c), used_mib=int(d), utilization=int(e))
            for a,b,c,d,e in (r.split(',') for r in raw.splitlines())}


def process_memory():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    return {int(p): int(m) for p,m in (r.split(',') for r in raw.splitlines()) if m.strip().isdigit()}


def descendants(root):
    parents = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            fields = (p / 'stat').read_text().split(') ',1)[1].split()
            parents[int(p.name)] = int(fields[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    found = {root}
    while True:
        new = {pid for pid,parent in parents.items() if parent in found} - found
        if not new:
            return found
        found.update(new)


def available_budget(gpu, telemetry, active, usage, policy, out=None):
    """Account for workers still initializing or below their eventual peak."""
    workers = [r for r in active.values() if r['gpu'] == gpu]
    if len(workers) >= policy['max_workers_per_gpu'].get(str(gpu), 0):
        return -1
    startup_headroom, locked_export_usage, locked_exports = 0, 0, 0
    for record in workers:
        used = sum(usage.get(pid, 0) for pid in descendants(record['pid']))
        command_path = out / record['task'] / 'current_command.json' if out else None
        command = base.read(command_path) if command_path and command_path.exists() else {}
        script = command.get('script')
        arguments = command.get('arguments', [])
        locked = script == 'generalization_reactzyme_f3_features.py' or (
            script == 'generalization_clipzyme_f3_screen.py' and '--phase' in arguments
            and arguments[arguments.index('--phase') + 1] in {'proteins', 'reactions'})
        if locked and 'serialized_export_reservation_mib' in policy:
            locked_exports += 1
            locked_export_usage += used
        else:
            startup_headroom += max(0, policy['worker_reservation_mib'] - used)
    if locked_exports:
        # These programs acquire the same physical-GPU flock before any CUDA
        # allocation. Reserve the measured peak once for the entire lock queue.
        startup_headroom += max(0, policy['serialized_export_reservation_mib'] - locked_export_usage)
    return telemetry[gpu]['free_mib'] - policy['safety_margin_mib'] - startup_headroom


def configure_local_imports(out, run, policy):
    package = policy.get('local_transformers')
    if not package:
        return
    receipt_path = Path(package['receipt'])
    if base.sha(receipt_path) != package['receipt_sha256']:
        raise ValueError('Local dependency receipt changed')
    receipt = base.read(receipt_path)
    directory = Path(receipt['local_package'])
    for item in receipt['files']:
        if base.sha(directory / item['path']) != item['sha256']:
            raise ValueError('Local dependency copy changed: ' + item['path'])
    parent = str(directory.parent)
    sys.path.insert(0, parent)
    os.environ['PYTHONPATH'] = os.pathsep.join(filter(None, [parent, os.environ.get('PYTHONPATH')]))
    base.write(run / 'worker_runtime_environment.json', dict(
        local_transformers=package, verified_files=len(receipt['files']),
        export_cuda_allocator=policy.get('export_cuda_allocator'),
        scientific_configuration_unchanged=True, utc=base.utc()))


def defer_calibration_import():
    """Load the heavy evaluation stack only when EnzymeMap calibration needs it.

    The frozen runner imports this one helper before starting any training.
    ReactZyme never calls it. Keep the actual implementation, with lazy import.
    """
    name = 'generalization_biological_f3_followup'
    if name in sys.modules:
        return
    proxy = types.ModuleType(name)

    def calibrate(*args, **kwargs):
        sys.modules.pop(name, None)
        return importlib.import_module(name).calibrate(*args, **kwargs)

    proxy.calibrate = calibrate
    sys.modules[name] = proxy


def worker(out, task_id, gpu):
    plan = base.check(out)
    task = copy.deepcopy(next(t for t in plan['tasks'] if t['id'] == task_id))
    task['gpu'] = gpu
    run = Path(task['run_root'])
    with (run / 'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        policy = base.read(out / 'execution_policy.json')
        configure_local_imports(out, run, policy)
        original_invoke = base.invoke

        def invoke(script, args, gpu, log):
            arguments = list(args)
            if script == 'train_protein_pooling_fast_io.py':
                arguments[arguments.index('--io-prefetch') + 1] = 4
                arguments += ['--io-recovery-every-n-train-steps', 500]
                script = 'train_residue_view_runtime.py'
            base.write(run / 'current_command.json', dict(script=script, arguments=list(map(str,arguments)),
                gpu=gpu, started_utc=base.utc()))
            allocator = policy.get('export_cuda_allocator') if script in {
                'generalization_clipzyme_f3_screen.py', 'generalization_reactzyme_f3_features.py',
                'generalization_clipzyme_phase2_export.py'} else None
            if not allocator:
                return original_invoke(script, arguments, gpu, log)
            previous = os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
            try:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = allocator
                return original_invoke(script, arguments, gpu, log)
            finally:
                if previous is None:
                    os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
                else:
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = previous

        base.invoke = invoke
        defer_calibration_import()
        try:
            base.run_task(out, task)
        except Exception as exc:
            base.write(run / 'failure.json', dict(utc=base.utc(), error=repr(exc), gpu=gpu))
            base.stage(task, 'failed', error=repr(exc))
            raise


def scheduler(out):
    with (out / 'queue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = base.check(out)
        base.write(out / 'execution.json', dict(pid=os.getpid(), start_ticks=identity(os.getpid()),
            started_utc=base.utc(), command=sys.argv, execution='parallel'))
        active, launched = {}, {}
        for task in plan['tasks']:
            file = Path(task['run_root']) / 'worker_execution.json'
            if file.exists():
                record = base.read(file)
                if alive(record):
                    active[task['id']] = record
        while True:
            policy = base.read(out / 'execution_policy.json')
            for key, record in list(active.items()):
                process = launched.get(key)
                if process is not None:
                    process.poll()
                if alive(record):
                    continue
                run = Path(next(t['run_root'] for t in plan['tasks'] if t['id'] == key))
                if not (run / 'complete.json').exists() and not (run / 'failure.json').exists():
                    base.write(run / 'failure.json', dict(utc=base.utc(),
                        error='Worker exited without a completion receipt; explicit recovery required.',
                        execution=record))
                del active[key]
            gpu_status, usage = devices(), process_memory()
            ready = {c['name']: cache_ready(c) for c in plan['caches']}
            waiting, failed = [], []
            for task in plan['tasks']:
                run = Path(task['run_root'])
                if (run / 'complete.json').exists() or task['id'] in active:
                    continue
                if (run / 'failure.json').exists() or (run / 'started.json').exists():
                    failed.append(task['id'])
                    continue
                cache = 'enzymemap' if task['name'] == 'enzymemap' else 'reactzyme'
                if not ready[cache]:
                    waiting.append(dict(task=task['id'], reason='residue cache'))
                    continue
                if policy.get('stop_new_launches', False):
                    waiting.append(dict(task=task['id'], reason='launches temporarily held'))
                    continue
                budgets = {gpu: available_budget(gpu, gpu_status, active, usage, policy, out)
                           for gpu in policy['gpus']}
                gpu = max(budgets, key=budgets.get)
                if budgets[gpu] < policy['worker_reservation_mib']:
                    waiting.append(dict(task=task['id'], reason='GPU memory budget'))
                    continue
                base.check(out)
                command = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--output', str(out),
                           '--task', task['id'], '--gpu', str(gpu)]
                with (run / 'worker.log').open('a') as stream:
                    process = subprocess.Popen(command, cwd=base.ROOT, stdout=stream, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, start_new_session=True,
                        env={**os.environ, 'OMP_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'4'})
                record = dict(pid=process.pid, start_ticks=identity(process.pid), gpu=gpu,
                    started_utc=base.utc(), task=task['id'], command=command,
                    execution_policy_sha256=base.sha(out / 'execution_policy.json'))
                base.write(run / 'worker_execution.json', record)
                active[task['id']] = record; launched[task['id']] = process
                print(json.dumps(dict(starting=task['id'], gpu=gpu, pid=process.pid)), flush=True)
            cache_progress = {}
            for name in ready:
                log = out / (name + '_cache.log')
                if log.exists():
                    lines = log.read_text().splitlines()
                    cache_progress[name] = lines[-1] if lines else 'preparing first block'
            base.report(out, plan, stage='running' if active else 'waiting',
                active=list(active.values()), waiting=waiting, failed=failed,
                gpu=gpu_status, caches_ready=ready, cache_progress=cache_progress)
            if len([t for t in plan['tasks'] if (Path(t['run_root'])/'complete.json').exists()]) == 16:
                base.report(out, plan, stage='complete', active=[], failed=[])
                return
            time.sleep(15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run','worker'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task')
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.action == 'run':
        scheduler(args.output.resolve())
    else:
        worker(args.output.resolve(), args.task, args.gpu)
