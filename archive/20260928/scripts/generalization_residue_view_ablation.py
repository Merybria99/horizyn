#!/usr/bin/env python3
"""Fixed K=1/2/4/8, seed-42 V4 ablation, isolated from existing campaigns.

prepare/check are CPU-only. run persists a sequential queue on GPU 2, after
its pre-existing competitor assignments complete. Every arm receives fresh
training, native phase 2, fixed fusion calibration, and both benchmark readouts.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/generalization_20260919_2251'
CROSS = RUN / 'cross_paper_retraining'
PUBLIC = ROOT / 'runs/reactzyme_public_baselines_20260921'
TARGETS = ('reaction_smi', 'enzyme_smi', 'time', 'enzymemap')
COUNTS = (4, 1, 2, 8)  # Fresh reference first; no result-dependent order.


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def template(target, test=False):
    return RUN / (f'features_test_{target}' if test else {
        'reaction_smi': 'features', 'enzyme_smi': 'features_enzyme_smi',
        'time': 'features_time'}[target])


def comparable(config):
    """Ignore output locations, metadata, and the declared experimental factor."""
    value = copy.deepcopy(config)
    value.pop('ablation', None)
    for key in ('log_dir', 'checkpoint_dir'):
        value['logging'].pop(key, None)
    value['model']['enzyme_multiview'].pop('num_slots')
    return value


def prepare(out):
    import yaml
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    frozen_path = CROSS / 'shared_recipe_alpha04_cap05_v1/case1_freeze.json'
    frozen = read(frozen_path)
    old_test = {v['split']: v['task']['test_config'] for v in
                read(CROSS / 'shared_semantic_strength_variants_v1/protocol.json')['reactzyme']}
    cache_dir = Path('/tmp') / ('enzymediscovery_' + out.name)
    react_source = Path(read(template('reaction_smi') / 'manifest.json')['inputs']['protein_residues']['path'])
    enzyme_source = CROSS / 'clipzyme_f3_catalog_v1/features/proteins_prott5_residue.h5'
    catalogs = {}
    proteins = set()
    for target in TARGETS[:-1]:
        for test in (False, True):
            path = template(target, test) / 'catalog.json'
            proteins.update(read(path)['proteins'])
            catalogs[str(path)] = sha(path)
    write(out / 'reactzyme_catalog.json', {'proteins': sorted(proteins), 'source_catalogs': catalogs,
        'scope': 'Frozen embedding transport only; association graphs remain separate by split.'})
    caches = [dict(name='reactzyme', source=str(react_source), catalog=str(out / 'reactzyme_catalog.json'),
                   output=str(cache_dir / 'reactzyme.h5')),
              dict(name='enzymemap', source=str(enzyme_source),
                   catalog=str(CROSS / 'sleec_multiview_phase2_v1/features/catalog.json'),
                   output=str(cache_dir / 'enzymemap.h5'))]
    for cache in caches:
        stat = Path(cache['source']).stat()
        cache.update(source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
                     catalog_sha256=sha(cache['catalog']))
    tasks, sources = [], {str(frozen_path): sha(frozen_path)}
    for k in COUNTS:
        for target in TARGETS:
            source = Path(next(m['config'] for m in frozen['models'] if m['name'] == target))
            sources[str(source)] = sha(source)
            config = yaml.safe_load(source.read_text())
            run = out / f'k{k}' / target
            (run / 'configs').mkdir(parents=True)
            config['seed'] = 42
            config['training']['max_epochs'] = 10
            # Preserve the original validation cadence (and its RNG effects).
            config['model']['enzyme_multiview']['num_slots'] = k
            cache = caches[1 if target == 'enzymemap' else 0]
            config['data']['protein_residue_embeds_path'] = cache['output']
            config['logging']['log_dir'] = str(run / 'unused/logs')
            config['logging']['checkpoint_dir'] = str(run / 'unused/checkpoints')
            config['ablation'] = dict(factor='learned_residue_views', num_slots=k, seed=42,
                benchmark=target, reference_slots=4, fresh_trainable_f3=True,
                fixed_epoch=10, test_used_for_selection=False)
            path = run / 'configs/train.yaml'
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            task = dict(id=f'k{k}/{target}', name=target, num_slots=k, seed=42,
                config=str(path), config_sha256=sha(path), source_config=str(source),
                run_root=str(run), gpu=2)
            if target != 'enzymemap':
                source_test = Path(old_test[target])
                sources[str(source_test)] = sha(source_test)
                test_config = yaml.safe_load(source_test.read_text())
                test_config['model'] = copy.deepcopy(config['model'])
                test_config['data']['protein_residue_embeds_path'] = cache['output']
                test_config['logging'] = copy.deepcopy(config['logging'])
                test_config['ablation'] = dict(config['ablation'], test_config=True)
                test_path = run / 'configs/test.yaml'
                test_path.write_text(yaml.safe_dump(test_config, sort_keys=False))
                task.update(test_config=str(test_path), test_config_sha256=sha(test_path))
            tasks.append(task)
    code_paths = list((ROOT / 'horizyn').rglob('*.py'))
    code_paths += list((ROOT / 'scripts').glob('generalization_*.py'))
    code_paths += [ROOT / 'scripts/train_protein_pooling.py', ROOT / 'scripts/train_protein_pooling_fast_io.py']
    code = {str(p.relative_to(ROOT)): sha(p) for p in code_paths if p.is_file()}
    dependencies = [str(PUBLIC / 'models' / t['name'] / 'complete.json')
                    for t in read(PUBLIC / 'extended_protocol.json')['tasks'] if t['gpu'] == 2]
    plan = dict(created_utc=utc(), seed=42, counts=list(COUNTS), tasks=tasks,
        source_configs=sources, code_sha256=code, caches=caches, gpu=2,
        prerequisite_completions=dependencies, recipe=frozen['recipe'],
        purpose='Matched learned residue-view count ablation; all 16 arms reported.',
        selection='Fixed epoch 10 and phase-2 step 100; no arm or checkpoint selected using test.',
        biology=False, shared_weights=False, baseline='fresh k4 within each target',
        interpretation='Single-seed exploratory sensitivity; parameter count varies with K.',
        runtime_changes=['10-epoch stop rather than continuing past the retained epoch-10 snapshot',
                         'byte-identical compact residue transport; isolated output directories'],
        failure_policy='Stop on failure, preserve partial files; no automatic fresh restart or arm substitution.')
    write(out / 'protocol.json', plan)


def check(out):
    import yaml
    sys.path.insert(0, str(ROOT))
    from horizyn.config import load_config
    plan = read(out / 'protocol.json')
    if len(plan['tasks']) != 16 or {(t['name'], t['num_slots']) for t in plan['tasks']} != {
            (t, k) for t in TARGETS for k in COUNTS}:
        raise ValueError('Incomplete or duplicated ablation matrix')
    for path, expected in {**plan['source_configs'], **{
            str(ROOT / p): digest for p, digest in plan['code_sha256'].items()}}.items():
        if sha(path) != expected:
            raise ValueError(f'Frozen source changed: {path}')
    reference = {}
    for task in plan['tasks']:
        for key in ('config', 'test_config'):
            if key not in task:
                continue
            if sha(task[key]) != task[key + '_sha256']:
                raise ValueError(f'Configuration changed: {task[key]}')
            load_config(task[key])
        config = yaml.safe_load(Path(task['config']).read_text())
        if config['seed'] != 42 or config['model']['enzyme_multiview']['num_slots'] != task['num_slots']:
            raise ValueError('Factor or seed mismatch')
        value = comparable(config)
        if task['name'] in reference and reference[task['name']] != value:
            raise ValueError('Another training setting differs between K arms')
        reference[task['name']] = value
        for key, path in config['data'].items():
            if key.endswith('_path') and path and key != 'protein_residue_embeds_path':
                if not (ROOT / path).exists():
                    raise FileNotFoundError(path)
    for cache in plan['caches']:
        stat = Path(cache['source']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (cache['source_size'], cache['source_mtime_ns']):
            raise ValueError('Frozen residue source changed')
        if sha(cache['catalog']) != cache['catalog_sha256']:
            raise ValueError('Residue catalog changed')
    return plan


def invoke(script, args, gpu, log):
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
    command = [sys.executable, '-u', str(ROOT / 'scripts' / script), *map(str, args)]
    with Path(log).open('a') as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def stage(task, value, **extra):
    record = dict(task=task['id'], stage=value, updated_utc=utc(), **extra)
    write(Path(task['run_root']) / 'state.json', record)
    print(json.dumps(record), flush=True)


def gpu_available(plan):
    if any(not Path(p).is_file() for p in plan['prerequisite_completions']):
        return False, 'waiting for previously assigned GPU-2 competitor runs'
    result = subprocess.check_output(['nvidia-smi', '-i', str(plan['gpu']),
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    if result.strip():
        return False, 'GPU 2 has an existing compute process'
    free = subprocess.check_output(['nvidia-smi', '-i', str(plan['gpu']),
        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
    return int(free.strip()) >= 100000, 'waiting for at least 100000 MiB free on GPU 2'


def prepare_caches(out, plan):
    for cache in plan['caches']:
        path = Path(cache['output'])
        if not path.exists():
            write(out / 'status.json', dict(stage='preparing_frozen_residue_cache', cache=cache['name'],
                updated_utc=utc(), completed=0, total=16, training_started=False))
            invoke('generalization_clipzyme_compact_residues.py', ['--source', cache['source'],
                '--catalog', cache['catalog'], '--output', path, '--source-identity-only',
                '--read-ahead-mib', 128], plan['gpu'], out / (cache['name'] + '_cache.log'))
        receipt = read(path.with_suffix('.receipt.json'))
        stat = path.stat()
        if (receipt['output_size_bytes'], receipt['output_mtime_ns'], receipt['catalog_sha256']) != (
                stat.st_size, stat.st_mtime_ns, cache['catalog_sha256']):
            raise ValueError('Prepared cache receipt mismatch')


def run_task(out, task):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from generalization_biological_f3_followup import calibrate
    run, gpu, name = Path(task['run_root']), task['gpu'], task['name']
    if (run / 'started.json').exists():
        raise RuntimeError(f"Prior incomplete task requires explicit recovery: {task['id']}")
    write(run / 'started.json', dict(utc=utc(), protocol_sha256=sha(out / 'protocol.json'), fresh=True))
    stage(task, 'training', epoch_target=10)
    invoke('train_protein_pooling_fast_io.py', ['--config', task['config'], '--io-mode', 'residue',
        '--io-output-dir', run / 'training', '--io-prefetch', 1], gpu, run / 'training.log')
    import torch
    checkpoint = run / 'training/checkpoints/screen_selection/screen-epoch=09.ckpt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['epoch'] != 9:
        raise ValueError('Wrong fixed checkpoint epoch')
    queries = [v for k, v in saved['state_dict'].items() if k.endswith('multiview_encoder.queries')]
    if len(queries) != 1 or queries[0].shape[0] != task['num_slots']:
        raise ValueError('Trained checkpoint has the wrong K')
    write(run / 'train_complete.json', dict(epoch=10, checkpoint_sha256=sha(checkpoint),
        num_slots=task['num_slots'], fresh=True, global_step=saved['global_step']))
    del saved, queries
    phase = run / 'phase2'; phase.mkdir()
    features = phase / 'features'
    stage(task, 'exporting_native_training_validation_embeddings')
    if name != 'enzymemap':
        invoke('generalization_reactzyme_f3_features.py', ['--template', template(name), '--config', task['config'],
            '--checkpoint', checkpoint, '--output', features, '--scope', 'train_validation', '--split', name,
            '--batch-size', 256], gpu, phase / 'features.log')
    else:
        features.mkdir()
        source = CROSS / 'sleec_multiview_phase2_v1/features'
        for filename in ('catalog.json', 'pairs.npz', 'reaction_features.npz'):
            (features / filename).symlink_to((source / filename).resolve())
        shutil.copyfile(source / 'protein_mean.h5', features / 'protein_mean.h5')
        manifest = read(source / 'manifest.json')
        for key in ('checkpoint', 'f3_export_config', 'f3_residue_cache'):
            manifest.pop(key, None)
        manifest['raw_feature_reuse'] = dict(source_manifest=str(source / 'manifest.json'),
            sha256=sha(source / 'manifest.json'), purpose='Raw training/dev features only')
        manifest['sources']['config'] = dict(path=task['config'], sha256=sha(task['config']))
        write(features / 'manifest.json', manifest)
        cache = next(c['output'] for c in read(out / 'protocol.json')['caches'] if c['name'] == 'enzymemap')
        invoke('generalization_clipzyme_phase2_export.py', ['--stage', 'f3', '--catalog',
            CROSS / 'clipzyme_f3_catalog_v1', '--config', task['config'], '--checkpoint', checkpoint,
            '--output', features, '--batch-size', 256, '--residue-cache', cache], gpu, phase / 'features.log')
    stage(task, 'training_dictionary_and_phase2')
    invoke('generalization_clipzyme_phase2_dictionary.py', ['--features', features, '--output',
        phase / 'anchors.pt'], gpu, phase / 'dictionary.log')
    evaluation = run / 'evaluation'
    result = evaluation / name / 'v4'
    result.mkdir(parents=True)
    invoke('generalization_full_graph.py', ['--features', features, '--output', result / 'training',
        '--steps', 100, '--snapshot-every', 100, '--selection-method', 'external_screening',
        '--temperature', .2, '--identity-weight', 10, '--learning-rate', .0001,
        '--weight-decay', .001, '--seed', 42, '--cpu-threads', 4,
        '--contrastive-objective', 'positive_ce'], gpu, result / 'training.log')
    model = dict(name=name, config=task['config'], config_sha256=sha(task['config']),
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), source_phase2=str(phase),
        fusion_multiplier=3., checkpoint_already_calibrated=False)
    etask = dict(name=name, gpu=gpu, source_phase2=str(phase), model=model)
    if name == 'enzymemap':
        stage(task, 'exporting_full_screening_library')
        adjusted = calibrate(task, checkpoint, run)
        model.update(checkpoint=str(adjusted), checkpoint_sha256=sha(adjusted), fusion_multiplier=1.,
            checkpoint_already_calibrated=True, calibration_receipt=str(run / 'fusion_receipt.json'))
        screen = run / 'screen'; proteins = screen / 'proteins'; proteins.mkdir(parents=True)
        shared = ['--catalog', CROSS / 'clipzyme_f3_catalog_v1', '--protocol',
            CROSS / 'clipzyme_screening_evaluation_protocol_v2', '--config', task['config'],
            '--checkpoint', adjusted]
        # Four logical shards on ONE physical GPU; never seize other campaigns' GPUs.
        for rank in range(4):
            invoke('generalization_clipzyme_f3_screen.py', [*shared, '--phase', 'proteins',
                '--output', proteins, '--rank', rank, '--world-size', 4, '--shard-block-size', 4096,
                '--batch-size', 256], gpu, screen / f'protein_rank{rank}.log')
        for scope in ('validation', 'test'):
            dest = screen / (scope + '_embeddings'); dest.mkdir()
            for path in proteins.glob('protein_*'):
                (dest / path.name).symlink_to(path.resolve())
            invoke('generalization_clipzyme_f3_screen.py', [*shared, '--phase', 'reactions', '--scope', scope,
                '--output', dest, '--batch-size', 256], gpu, screen / (scope + '.log'))
        selected = phase / 'selected_test'; selected.mkdir()
        (selected / 'test_embeddings').symlink_to(screen / 'test_embeddings', target_is_directory=True)
        write(phase / 'protocol.json', dict(base_checkpoint=str(adjusted),
            base_checkpoint_sha256=sha(adjusted),
            validation_summary=str(screen / 'validation_evaluation/summary.json'),
            fixed_recipe=True, test_used_for_selection=False))
    else:
        model['test_config'] = task['test_config']
        etask['test_features'] = str(phase / 'test_features')
    declaration = dict(created_utc=utc(), tasks=[etask], variants=[dict(name='v4')],
        parent_protocol_sha256=sha(out / 'protocol.json'), epoch=10, num_slots=task['num_slots'],
        fixed_recipe=True, test_used_for_selection=False,
        recipe=dict(semantic_alpha=.4, residual_cap=.5, fusion_multiplier=3.))
    write(evaluation / 'protocol.json', declaration)
    if name != 'enzymemap':
        stage(task, 'exporting_test_embeddings')
        invoke('generalization_reactzyme_f3_features.py', ['--template', template(name, True),
            '--config', task['test_config'], '--checkpoint', checkpoint, '--output', phase / 'test_features',
            '--scope', 'test', '--freeze', evaluation / 'protocol.json', '--batch-size', 256,
            '--split', name], gpu, phase / 'test_features.log')
    for scope in ('validation', 'test'):
        stage(task, 'evaluating_' + scope)
        invoke('generalization_biological_geometry_evaluate.py', ['--campaign', evaluation,
            '--task', name, '--scope', scope], gpu, run / (scope + '.log'))
    summary = result / 'test_summary.json'
    write(run / 'complete.json', dict(completed_utc=utc(), task=task['id'], num_slots=task['num_slots'],
        seed=42, summary=str(summary), summary_sha256=sha(summary), metrics=read(summary)['summary']))
    stage(task, 'complete')


def report(out, plan, **extra):
    completed = [read(Path(t['run_root']) / 'complete.json') for t in plan['tasks']
                 if (Path(t['run_root']) / 'complete.json').exists()]
    write(out / 'status.json', dict(updated_utc=utc(), total=16, completed=len(completed),
        results=completed, **extra))
    rows = ['# Learned residue-view count ablation', '', f'Completed fits: {len(completed)}/16.', '',
        'K = 1, 2, 4, 8; seed 42; fresh target-specific training. Fixed V4 phase 2 and scoring.',
        'All arms are reported. K=4 is freshly retrained; this is a single-seed sensitivity study.', '',
        '| K | Target | Setting / direction | Metric | Value |', '|---:|---|---|---|---:|']
    for entry in completed:
        target = entry['task'].split('/')[1]
        for setting, metrics in entry['metrics'].items():
            values = ({'MRR': metrics['all']['reactzyme_mrr']} if target != 'enzymemap' else
                      {k: metrics[k] for k in ('bedroc85', 'bedroc20', 'ef0.05', 'ef0.1')})
            for metric, value in values.items():
                rows.append(f"| {entry['num_slots']} | {target} | {setting} | {metric} | {value:.6f} |")
    (out / 'results.md').write_text('\n'.join(rows) + '\n')


def run(out):
    lock = (out / 'queue.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(out / 'execution.json', dict(pid=os.getpid(), started_utc=utc(), command=sys.argv))
    try:
        plan = check(out)
        prepare_caches(out, plan)
        for task in plan['tasks']:
            if (Path(task['run_root']) / 'complete.json').exists():
                continue
            while True:
                available, reason = gpu_available(plan)
                if available:
                    break
                report(out, plan, stage='waiting_for_gpu', next_task=task['id'], reason=reason)
                time.sleep(30)
            check(out)
            report(out, plan, stage='running', active=task['id'], gpu=plan['gpu'])
            run_task(out, task)
            report(out, plan, stage='between_tasks')
        report(out, plan, stage='complete')
    except Exception as exc:
        write(out / 'failure.json', dict(utc=utc(), error=repr(exc)))
        if 'plan' in locals():
            report(out, plan, stage='failed', error=repr(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.action == 'prepare':
        prepare(out)
        plan = check(out)
        report(out, plan, stage='prepared')
    elif args.action == 'check':
        check(out)
        print('All 16 configurations pass the paired-factor and source checks.')
    else:
        run(out)
