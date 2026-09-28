#!/usr/bin/env python3
"""Evaluate fixed-epoch biological F3 training with the unchanged V4 recipe.

Every target gets new learned embeddings, fresh phase-2 fits, and a training-only
dictionary. Raw fit-free banks are reused with their original axes. The two
paired variants differ only in whether phase 2 also receives biological loss.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generalization_full_graph import atomic_json, sha
from generalization_biological_geometry_evaluate import CROSS, RUN


def invoke(script, args, gpu, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    with log.open('a') as stream:
        subprocess.run([sys.executable, '-u', str(ROOT / 'scripts' / script), *map(str, args)],
                       cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def native_export(task, checkpoint, phase, gpu):
    features = phase / 'features'
    if (features / 'complete.json').exists():
        if json.loads((features / 'complete.json').read_text())['checkpoint_sha256'] != sha(checkpoint):
            raise ValueError('Existing native export belongs to a different checkpoint')
        return
    if task['name'] != 'enzymemap':
        template = RUN / {'reaction_smi': 'features', 'enzyme_smi': 'features_enzyme_smi', 'time': 'features_time'}[task['name']]
        invoke('generalization_reactzyme_f3_features.py', [
            '--template', template, '--config', task['config'], '--checkpoint', checkpoint,
            '--output', features, '--scope', 'train_validation', '--split', task['name'], '--batch-size', 256],
            gpu, phase / 'features.log')
        return
    source = CROSS / 'sleec_multiview_phase2_v1/features'
    features.mkdir(exist_ok=True)
    for name in ('catalog.json', 'pairs.npz', 'reaction_features.npz'):
        if not (features / name).exists():
            (features / name).symlink_to((source / name).resolve())
    if not (features / 'protein_mean.h5').exists():
        shutil.copyfile(source / 'protein_mean.h5', features / 'protein_mean.h5')
    manifest = json.loads((source / 'manifest.json').read_text())
    for key in ('checkpoint', 'f3_export_config', 'f3_residue_cache'):
        manifest.pop(key, None)
    manifest['raw_feature_reuse'] = dict(source_manifest=str(source / 'manifest.json'),
        sha256=sha(source / 'manifest.json'), purpose='Raw training/dev features only')
    manifest['sources']['config'] = dict(path=task['config'], sha256=sha(task['config']))
    atomic_json(features / 'manifest.json', manifest)
    invoke('generalization_clipzyme_phase2_export.py', [
        '--stage', 'f3', '--catalog', CROSS / 'clipzyme_f3_catalog_v1', '--config', task['config'],
        '--checkpoint', checkpoint, '--output', features, '--batch-size', 256,
        '--residue-cache', '/tmp/enzymediscovery_f3_20260920/train_validation_prott5.h5'],
        gpu, phase / 'features.log')


def calibrate(task, checkpoint, run):
    import torch
    output = run / 'calibrated.ckpt'
    if output.exists():
        return output
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    key, = [k for k in saved['state_dict'] if k.endswith('multiview_encoder.raw_residual_scale')]
    old = saved['state_dict'][key].clone()
    scale = float(.5 * old.sigmoid()); desired = scale * 3
    if not 0 < desired < .5:
        raise ValueError('Fixed fusion multiplier is outside its model domain')
    saved['state_dict'][key] = old.new_tensor(math.log(desired / (.5 - desired)))
    receipt = dict(source_checkpoint=str(checkpoint), source_checkpoint_sha256=sha(checkpoint),
        original_checkpoint_sha256=sha(checkpoint), parameter=key, requested_multiplier=3., multiplier=3.,
        original_scale=scale, new_scale=float(.5 * saved['state_dict'][key].sigmoid()),
        phase2_fitted_before_adjustment=True, all_other_model_parameters_unchanged=True,
        fixed_before_enzymemap_training=True)
    saved['inference_fusion_calibration'] = receipt
    for name in ('optimizer_states', 'lr_schedulers', 'loops', 'callbacks'):
        saved.pop(name, None)
    torch.save(saved, output)
    receipt['calibrated_checkpoint_sha256'] = sha(output)
    atomic_json(run / 'fusion_receipt.json', receipt)
    return output


def screening_exports(task, checkpoint, run, phase, gpu):
    screen = run / 'screen'
    proteins = screen / 'proteins'
    proteins.mkdir(parents=True, exist_ok=True)
    shared = ['--catalog', CROSS / 'clipzyme_f3_catalog_v1', '--protocol',
              CROSS / 'clipzyme_screening_evaluation_protocol_v2', '--config', task['config'],
              '--checkpoint', checkpoint]
    def rank_export(rank):
        if not (proteins / f'protein_rank{rank:02d}_of_04.receipt.json').exists():
            invoke('generalization_clipzyme_f3_screen.py', [*shared, '--phase', 'proteins',
                '--output', proteins, '--rank', rank, '--world-size', 4,
                '--shard-block-size', 4096, '--batch-size', 512], rank, screen / f'protein_rank{rank}.log')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(rank_export, range(4)))
    for scope in ('validation', 'test'):
        dest = screen / (scope + '_embeddings'); dest.mkdir(exist_ok=True)
        for path in proteins.glob('protein_*'):
            if not (dest / path.name).exists():
                (dest / path.name).symlink_to(path.resolve())
        if not (dest / 'reaction_receipt.json').exists():
            invoke('generalization_clipzyme_f3_screen.py', [*shared, '--phase', 'reactions',
                '--scope', scope, '--output', dest, '--batch-size', 256], gpu, screen / (scope + '.log'))
    selected = phase / 'selected_test'; selected.mkdir(exist_ok=True)
    if not (selected / 'test_embeddings').exists():
        (selected / 'test_embeddings').symlink_to(screen / 'test_embeddings', target_is_directory=True)
    atomic_json(phase / 'protocol.json', dict(base_checkpoint=str(checkpoint),
        base_checkpoint_sha256=sha(checkpoint), validation_summary=str(screen / 'validation_evaluation/summary.json'),
        fixed_recipe=True, test_used_for_selection=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--task', required=True)
    a = p.parse_args(); out = a.campaign.resolve()
    plan = json.loads((out / 'protocol.json').read_text())
    task = next(t for t in plan['tasks'] if t['name'] == a.task)
    biological_weight = task.get('biological_weight', plan.get('biological_weight', .1))
    biological_mode = task.get('biological_mode', plan.get('biological_mode', 'attraction'))
    biological_margin = task.get('biological_margin', plan.get('biological_margin', .1))
    run = Path(task['run_root']); gpu = task['gpu']
    phase = run / 'phase2'; phase.mkdir(exist_ok=True)
    def state(stage, **extra):
        value = dict(stage=stage, pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        atomic_json(run / 'followup_state.json', value); print(json.dumps(value), flush=True)
    try:
        state('waiting_for_fixed_epoch10')
        checkpoint = run / 'checkpoints/screen_selection/screen-epoch=09.ckpt'
        while not checkpoint.exists() or time.time() - checkpoint.stat().st_mtime < 30:
            execution = json.loads((run / 'training_execution.json').read_text())
            try:
                os.kill(execution['pid'], 0)
            except ProcessLookupError:
                if not checkpoint.exists():
                    raise RuntimeError('Trainer exited without the fixed epoch-10 checkpoint')
            time.sleep(10)
        import torch
        torch.set_num_threads(4)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['epoch'] != 9:
            raise ValueError('Wrong fixed checkpoint epoch')
        atomic_json(run / 'train_complete.json', dict(epoch=saved['epoch'], global_step=saved['global_step'],
            checkpoint_sha256=sha(checkpoint), parameter_count=sum(v.numel() for v in saved['state_dict'].values()),
            biological_weight=biological_weight, fresh_fit=True, protocol_sha256=sha(out / 'protocol.json')))
        del saved
        state('exporting_new_native_f3_vectors')
        native_export(task, checkpoint, phase, gpu)
        features = phase / 'features'
        annotations = phase / 'annotations.json'
        if not annotations.exists():
            parent = Path(task['annotations']); payload = json.loads(parent.read_text())
            payload['annotation_rebinding'] = dict(source=str(parent.resolve()), sha256=sha(parent),
                original_feature_manifest_sha256=payload['feature_manifest_sha256'], labels_unchanged=True)
            payload['feature_manifest_sha256'] = sha(features / 'manifest.json')
            atomic_json(annotations, payload)
        if not (phase / 'anchors.pt').exists():
            state('training_only_dictionary')
            invoke('generalization_clipzyme_phase2_dictionary.py', ['--features', features,
                '--output', phase / 'anchors.pt'], gpu, phase / 'dictionary.log')
        evaluation = out / 'followup' / a.task; evaluation.mkdir(parents=True, exist_ok=True)
        dest = evaluation / a.task; dest.mkdir(exist_ok=True)
        variants = [dict(name='f3_biology', weights={k: 0. for k in ('ec', 'cofactor', 'mechanism')}),
                    dict(name='f3_phase2_biology', weights={k: .1 for k in ('ec', 'cofactor', 'mechanism')})]
        state('paired_fixed_phase2_training')
        for variant in variants:
            d = dest / variant['name']; d.mkdir(exist_ok=True)
            if (d / 'training/complete.json').exists():
                continue
            args = ['--features', features, '--output', d / 'training', '--steps', 100, '--snapshot-every', 100,
                    '--selection-method', 'external_screening', '--temperature', .2, '--identity-weight', 10,
                    '--learning-rate', .0001, '--cpu-threads', 4, '--biological-labels', annotations,
                    '--biology-mode', biological_mode, '--biology-margin', biological_margin]
            for family, weight in variant['weights'].items():
                args += ['--biology-' + family, weight]
            invoke('generalization_full_graph.py', args, gpu, d / 'training.log')
        model = dict(name=a.task, config=task['config'], config_sha256=sha(task['config']),
            checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), source_phase2=str(phase),
            fusion_multiplier=3., checkpoint_already_calibrated=False)
        etask = dict(name=a.task, gpu=gpu, source_phase2=str(phase), model=model)
        if a.task == 'enzymemap':
            state('exporting_calibrated_full_screening_library')
            adjusted = calibrate(task, checkpoint, run)
            model.update(checkpoint=str(adjusted), checkpoint_sha256=sha(adjusted), fusion_multiplier=1.,
                checkpoint_already_calibrated=True, calibration_receipt=str(run / 'fusion_receipt.json'))
            screening_exports(task, adjusted, run, phase, gpu)
        else:
            old = json.loads((CROSS / 'shared_semantic_strength_variants_v1/protocol.json').read_text())
            old_task = next(x['task'] for x in old['reactzyme'] if x['split'] == a.task)
            # Exact old test input configuration, same architecture, new trained weights.
            model['test_config'] = old_task['test_config']
            etask['test_features'] = str(phase / 'test_features')
        declaration = dict(created_utc=datetime.now(timezone.utc).isoformat(),
            tasks=[etask], variants=variants, parent_protocol_sha256=sha(out / 'protocol.json'),
            fixed_recipe=True, test_used_for_selection=False, epoch=10, biology_in_f3=biological_weight,
            biological_mode=biological_mode, biological_margin=biological_margin,
            recipe=dict(semantic_alpha=.4, residual_cap=.5, fusion_multiplier=3.))
        path = evaluation / 'protocol.json'
        if path.exists():
            prior = json.loads(path.read_text()); declaration['created_utc'] = prior['created_utc']
            if prior != declaration:
                raise ValueError('Existing evaluation declaration differs')
        else:
            atomic_json(path, declaration)
        if a.task != 'enzymemap' and not (phase / 'test_features/complete.json').exists():
            state('exporting_fixed_test_vectors')
            invoke('generalization_reactzyme_f3_features.py', ['--template', RUN / ('features_test_' + a.task),
                '--config', model['test_config'], '--checkpoint', checkpoint, '--output', phase / 'test_features',
                '--scope', 'test', '--freeze', path, '--batch-size', 256, '--split', a.task], gpu, phase / 'test_features.log')
        for scope in ('validation', 'test'):
            state('evaluating_' + scope)
            invoke('generalization_biological_geometry_evaluate.py', ['--campaign', evaluation,
                '--task', a.task, '--scope', scope], gpu, run / ('followup_' + scope + '.log'))
        state('complete', evaluation=str(evaluation))
    except Exception as exc:
        state('failed', error=repr(exc)); raise


if __name__ == '__main__':
    main()
