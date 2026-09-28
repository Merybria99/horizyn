#!/usr/bin/env python3
"""Fresh K=0 extension of the frozen seed-42 learned-view ablation.

Four independent phase-1 fits share the otherwise idle GPU. Once all finish,
the original native phase-2 and full-library evaluations run sequentially.
The original 16-arm campaign and its immutable receipts remain untouched.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

import generalization_residue_view_ablation as base
from generalization_residue_view_parallel import configure_local_imports, defer_calibration_import

ROOT = base.ROOT
PARENT = ROOT / 'runs/v4_residue_view_count_20260922_seed42'
DEFAULT = ROOT / 'runs/cersei_residue_view_k0_20260923_seed42'


def training_arguments(task):
    return ['--config', task['config'], '--io-mode', 'residue', '--io-output-dir',
            str(Path(task['run_root']) / 'training'), '--io-prefetch', '4',
            '--io-recovery-every-n-train-steps', '500']


def prepare(out):
    import yaml
    if (out / 'protocol.json').exists():
        raise FileExistsError(out / 'protocol.json')
    parent = base.read(PARENT / 'protocol.json')
    tasks = []
    for old in parent['tasks']:
        if old['num_slots'] != 4:
            continue
        target = old['name']
        run = out / 'k0' / target
        (run / 'configs').mkdir(parents=True)
        task = dict(id='k0/' + target, name=target, num_slots=0, seed=42, gpu=2,
                    run_root=str(run), source_config=old['config'],
                    source_config_sha256=base.sha(old['config']))
        for key in ('config', 'test_config'):
            if key not in old:
                continue
            config = yaml.safe_load(Path(old[key]).read_text())
            reference = copy.deepcopy(config)
            config['model']['enzyme_multiview']['num_slots'] = 0
            config['ablation']['num_slots'] = 0
            config['ablation']['control'] = 'fresh global + SLEEC; no learned residue views'
            config['logging']['log_dir'] = str(run / 'unused/logs')
            config['logging']['checkpoint_dir'] = str(run / 'unused/checkpoints')
            assert base.comparable(config) == base.comparable(reference)
            path = run / 'configs' / ('train.yaml' if key == 'config' else 'test.yaml')
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            task[key] = str(path)
            task[key + '_sha256'] = base.sha(path)
            if key == 'test_config':
                task['source_test_config'] = old[key]
                task['source_test_config_sha256'] = base.sha(old[key])
        tasks.append(task)
    policy = base.read(PARENT / 'execution_policy.json')
    base.write(out / 'execution_policy.json', policy)
    paths = sorted((ROOT / 'horizyn').rglob('*.py'))
    paths += sorted((ROOT / 'scripts').glob('generalization_*.py'))
    paths += [ROOT / 'scripts' / name for name in ('cersei_residue_view_k0.py',
        'train_residue_view_runtime.py', 'train_protein_pooling.py', 'train_protein_pooling_fast_io.py')]
    code = {str(p.relative_to(ROOT)): base.sha(p) for p in paths if p.is_file()}
    for path in code:
        dest = out / 'source' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, dest)
    plan = dict(created_utc=base.utc(), purpose='Missing trained K=0 control, requested after K>=1 study',
        parent_protocol=str(PARENT / 'protocol.json'), parent_protocol_sha256=base.sha(PARENT / 'protocol.json'),
        counts=[0], tasks=tasks, caches=parent['caches'], gpu=2, code_sha256=code,
        execution_policy_sha256=base.sha(out / 'execution_policy.json'), recipe=parent['recipe'],
        regression_validation_sha256=base.sha(out / 'regression_validation.json'),
        selection=parent['selection'], biology=False, shared_weights=False,
        protocol_change='Only K and induced enzyme capacity change; learned-view regularizers are identically zero when no learned views exist.',
        retained='Global ProtT5 view, frozen SLEEC view, gated fusion, reaction encoder, both training phases, dictionary, full benchmark evaluation',
        runtime='Four independent fresh phase-1 fits on GPU 2, then serial exports, phase-2 fits and evaluations; same certified residue transport as parent.',
        failure_policy='Stop on error; keep partial checkpoints and require explicit recovery.')
    base.write(out / 'protocol.json', plan)


def check(out):
    import yaml
    from horizyn.config import load_config
    plan = base.read(out / 'protocol.json')
    assert len(plan['tasks']) == 4
    assert {t['name'] for t in plan['tasks']} == set(base.TARGETS)
    assert base.sha(plan['parent_protocol']) == plan['parent_protocol_sha256']
    assert base.sha(out / 'execution_policy.json') == plan['execution_policy_sha256']
    assert base.sha(out / 'regression_validation.json') == plan['regression_validation_sha256']
    assert base.read(out / 'regression_validation.json')['passed']
    for path, digest in plan['code_sha256'].items():
        assert base.sha(ROOT / path) == digest, path
    for task in plan['tasks']:
        assert task['num_slots'] == 0 and task['seed'] == 42
        for key in ('config', 'test_config'):
            if key not in task:
                continue
            assert base.sha(task[key]) == task[key + '_sha256']
            source = 'source_config' if key == 'config' else 'source_test_config'
            assert base.sha(task[source]) == task[source + '_sha256']
            value = yaml.safe_load(Path(task[key]).read_text())
            old = yaml.safe_load(Path(task[source]).read_text())
            assert value['model']['enzyme_multiview']['num_slots'] == 0
            assert base.comparable(value) == base.comparable(old), task['id']
            load_config(task[key])
    base.prepare_caches(out, plan)
    return plan


def validate_training(task):
    import torch
    run = Path(task['run_root'])
    checkpoint = run / 'training/checkpoints/screen_selection/screen-epoch=09.ckpt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['epoch'] == 9
    state = saved['state_dict']
    queries = [v for k, v in state.items() if k.endswith('multiview_encoder.queries')]
    assert len(queries) == 1 and queries[0].shape == (0, 256)
    assert not any('multiview_encoder.' + name in k for k in state for name in ('keys.', 'values.', 'residue_adapter.'))
    assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v, torch.Tensor))
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=base.sha(checkpoint), epoch=10,
                global_step=saved['global_step'], num_slots=0, fresh=True,
                config_sha256=task['config_sha256'], arguments=training_arguments(task))


def phase1(task):
    run = Path(task['run_root'])
    receipt = run / 'phase1_ready.json'
    if receipt.exists():
        expected = base.read(receipt)
        actual = validate_training(task)
        assert all(expected[k] == v for k, v in actual.items())
        return
    if (run / 'phase1_started.json').exists():
        raise RuntimeError('Incomplete phase 1 requires explicit recovery: ' + task['id'])
    args = training_arguments(task)
    base.write(run / 'phase1_started.json', dict(utc=base.utc(), config_sha256=task['config_sha256'],
                                              script='train_residue_view_runtime.py', arguments=args))
    base.stage(task, 'training', epoch_target=10)
    base.invoke('train_residue_view_runtime.py', args, task['gpu'], run / 'training.log')
    result = validate_training(task)
    base.write(receipt, dict(result, completed_utc=base.utc()))
    base.stage(task, 'phase1_complete')


def evaluate(out, task, policy):
    run = Path(task['run_root'])
    if (run / 'complete.json').exists():
        done = base.read(run / 'complete.json')
        assert base.sha(done['summary']) == done['summary_sha256']
        return
    original = base.invoke

    def invoke(script, args, gpu, log):
        if script == 'train_protein_pooling_fast_io.py':
            expected = base.read(run / 'phase1_ready.json')
            actual = validate_training(task)
            assert all(expected[k] == v for k, v in actual.items())
            return  # Fresh phase 1 already completed by this same campaign.
        base.write(run / 'current_command.json', dict(script=script, arguments=list(map(str, args)),
                                                     gpu=gpu, started_utc=base.utc()))
        old_allocator = os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
        try:
            if script in {'generalization_clipzyme_f3_screen.py', 'generalization_reactzyme_f3_features.py',
                          'generalization_clipzyme_phase2_export.py'}:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = policy['export_cuda_allocator']
            return original(script, args, gpu, log)
        finally:
            if old_allocator is None:
                os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
            else:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = old_allocator

    base.invoke = invoke
    try:
        base.run_task(out, task)
    finally:
        base.invoke = original


def status(out, plan, stage):
    completed = [base.read(Path(t['run_root']) / 'complete.json') for t in plan['tasks']
                 if (Path(t['run_root']) / 'complete.json').exists()]
    base.write(out / 'status.json', dict(updated_utc=base.utc(), stage=stage, total=4,
                                      completed=len(completed), results=completed))


def run(out):
    with (out / 'campaign.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = check(out)
        available, reason = base.gpu_available(dict(plan, prerequisite_completions=[]))
        if not available:
            raise RuntimeError(reason)
        policy = base.read(out / 'execution_policy.json')
        configure_local_imports(out, out, policy)
        defer_calibration_import()
        status(out, plan, 'training_four_k0_models')
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(phase1, task): task for task in plan['tasks']}
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        base.write(Path(task['run_root']) / 'failure.json', dict(utc=base.utc(), error=repr(exc)))
                        base.stage(task, 'failed', error=repr(exc))
                        raise
            for task in plan['tasks']:
                status(out, plan, 'evaluating_' + task['name'])
                evaluate(out, task, policy)
            status(out, plan, 'complete')
            base.write(out / 'complete.json', dict(completed_utc=base.utc(), tasks=4,
                                                  protocol_sha256=base.sha(out / 'protocol.json')))
        except Exception as exc:
            status(out, plan, 'failed')
            base.write(out / 'failure.json', dict(utc=base.utc(), error=repr(exc)))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'check', 'run'))
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == 'prepare':
        prepare(output)
    elif args.action == 'check':
        check(output)
        print('Four K=0 configs match the frozen K=4 recipe; sources and caches verified.')
    else:
        run(output)
