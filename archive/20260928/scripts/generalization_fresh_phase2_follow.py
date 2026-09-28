#!/usr/bin/env python3
"""Fit fixed phase 2 at predeclared fresh-F3 epochs; select only on validation."""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from generalization_screen_replication import ROOT, sha, invoke
from generalization_full_graph import atomic_json
from generalization_clipzyme_f3_screen import export_device_lock
from generalization_clipzyme_ablation_watch import evaluate
from generalization_clipzyme_test_queue import run_job


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--task', required=True)
    a = p.parse_args()
    campaign = a.campaign.resolve()
    plan = json.loads((campaign / 'protocol.json').read_text())
    task = next(t for t in plan['tasks'] if t['name'] == a.task)
    base, cross = Path(task['run_root']), campaign.parent
    gpu = task['gpu']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    if sha(task['config']) != task['config_sha256']:
        raise ValueError('Frozen training configuration changed')
    root = base / 'phase2_followup'
    root.mkdir(exist_ok=True)
    registry = root / 'registry.json'
    if not registry.exists():
        atomic_json(registry, dict(created_utc=datetime.now(timezone.utc).isoformat(),
            training_protocol_sha256=sha(campaign / 'protocol.json'),
            source_sha256=sha(__file__), epochs=plan['checkpoint_epochs'],
            phase2=plan['phase2'], selection=('table1 BEDROC85' if a.task == 'enzymemap'
                else 'mean bidirectional seen/unseen-reaction validation all-positive MRR'),
            test_used_for_selection=False, repeated_test_evaluation_exploratory=True))
    elif json.loads(registry.read_text())['training_protocol_sha256'] != sha(campaign / 'protocol.json'):
        raise ValueError('Training protocol changed after follow-up was frozen')

    def state(stage, **extra):
        record = dict(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        atomic_json(root / 'state.json', record)
        print(json.dumps(record), flush=True)

    def train_phase(run):
        recipe = plan['phase2']
        if not (run / 'training/complete.json').exists():
            invoke('generalization_full_graph.py', ['--features', run / 'features',
                '--output', run / 'training', '--steps', recipe['steps'],
                '--snapshot-every', recipe['steps'], '--selection-method',
                'external_screening' if a.task == 'enzymemap' else 'fixed_last',
                '--temperature', recipe['temperature'], '--identity-weight', recipe['identity_weight'],
                '--contrastive-objective', recipe['objective'], '--learning-rate', .0001,
                '--seed', plan['seed'], '--cpu-threads', 4], gpu, run / 'training.log')
        if not (run / 'anchors.pt').exists():
            invoke('generalization_clipzyme_phase2_dictionary.py', ['--features', run / 'features',
                '--output', run / 'anchors.pt'], gpu, run / 'dictionary.log')

    try:
        records = []
        for epoch in plan['checkpoint_epochs']:
            run = root / f'epoch{epoch:02d}'
            run.mkdir(exist_ok=True)
            checkpoint = base / f'checkpoints/screen_selection/screen-epoch={epoch-1:02d}.ckpt'
            state('waiting_for_checkpoint', epoch=epoch)
            while not checkpoint.exists() or time.time() - checkpoint.stat().st_mtime < 30:
                if (base / 'failure.json').exists():
                    raise RuntimeError('Base trainer failed before the declared snapshot')
                time.sleep(10)
            result = run / 'validation_result.json'
            if result.exists():
                record = json.loads(result.read_text())
                if record['checkpoint_sha256'] != sha(checkpoint):
                    raise ValueError('Completed validation checkpoint changed')
                records.append(record)
                continue
            state('feature_export_and_phase2', epoch=epoch)
            features = run / 'features'
            if a.task != 'enzymemap':
                from generalization_reactzyme_architecture_phase2 import RUN, validate
                template = RUN / {'reaction_smi':'features', 'enzyme_smi':'features_enzyme_smi',
                                  'time':'features_time'}[a.task]
                if not (features / 'complete.json').exists():
                    invoke('generalization_reactzyme_f3_features.py', ['--template', template,
                        '--config', task['config'], '--checkpoint', checkpoint, '--output', features,
                        '--scope', 'train_validation', '--batch-size', 128, '--split', a.task],
                        gpu, run / 'features.log')
                if json.loads((features / 'complete.json').read_text())['checkpoint_sha256'] != sha(checkpoint):
                    raise ValueError('Feature checkpoint mismatch')
                train_phase(run)
                import torch
                torch.set_num_threads(4)
                torch.set_float32_matmul_precision('highest')
                torch.backends.cuda.matmul.allow_tf32 = False
                baseline = json.loads((RUN / 'phase2_soft_ce_v1' / a.task /
                    'composition_validation.json').read_text())['records'][0]['validation']
                state('fixed_phase2_validation', epoch=epoch)
                with export_device_lock('cuda:0'):
                    validate(run, baseline, 'cuda:0', steps=(100,), caps=(1.,), alphas=(.25,))
                val = json.loads((run / 'validation.json').read_text())['records'][0]
                gc.collect(); torch.cuda.empty_cache()
            else:
                if not (run / 'features_complete.json').exists():
                    features.mkdir(exist_ok=True)
                    source = cross / 'sleec_multiview_phase2_v1/features'
                    for name in ('catalog.json', 'pairs.npz', 'reaction_features.npz'):
                        if not (features / name).exists():
                            (features / name).symlink_to((source / name).resolve())
                    if not (features / 'protein_mean.h5').exists():
                        shutil.copyfile(source / 'protein_mean.h5', features / 'protein_mean.h5')
                    manifest = json.loads((source / 'manifest.json').read_text())
                    for key in ('checkpoint', 'f3_export_config', 'f3_residue_cache'):
                        manifest.pop(key, None)
                    manifest['raw_feature_reuse'] = dict(source_manifest=str(source / 'manifest.json'),
                        sha256=sha(source / 'manifest.json'))
                    manifest['sources']['config'] = dict(path=task['config'], sha256=task['config_sha256'])
                    atomic_json(features / 'manifest.json', manifest)
                    with export_device_lock('cuda:0'):
                        invoke('generalization_clipzyme_phase2_export.py', ['--stage', 'f3',
                            '--catalog', cross / 'clipzyme_f3_catalog_v1', '--config', task['config'],
                            '--checkpoint', checkpoint, '--output', features, '--batch-size', 128,
                            '--residue-cache', '/tmp/enzymediscovery_f3_20260920/train_validation_prott5.h5'],
                            gpu, run / 'features.log')
                    atomic_json(run / 'features_complete.json', dict(checkpoint_sha256=sha(checkpoint),
                        manifest_sha256=sha(features / 'manifest.json')))
                train_phase(run)
                state('full_library_validation', epoch=epoch)
                evaluate(base, epoch-1, SimpleNamespace(catalog=cross / 'clipzyme_f3_catalog_v1',
                    protocol=cross / 'clipzyme_screening_evaluation_protocol_v2',
                    manifest=cross / 'clipzyme_manifests_v2/manifest.json',
                    epochs=[v-1 for v in plan['checkpoint_epochs']], gpus=[0,1,2,3]))
                screen = base / f'screen_epoch{epoch-1}'
                if not (run / 'validation/summary.json').exists():
                    invoke('generalization_clipzyme_f3_validation.py', ['--manifest',
                        cross / 'clipzyme_manifests_v2/manifest.json', '--catalog', cross / 'clipzyme_f3_catalog_v1',
                        '--embeddings', screen / 'validation_embeddings', '--output', run / 'validation',
                        '--refiner', run / 'training/step0100.pt', '--batch-size', 64], gpu, run / 'validation.log')
                atomic_json(run / 'protocol.json', dict(base_config=task['config'],
                    base_checkpoint=str(checkpoint), base_checkpoint_sha256=sha(checkpoint),
                    validation_summary=str(screen / 'validation_evaluation/summary.json'),
                    fixed_recipe=True, test_used_for_selection=False))
                atomic_json(run / 'validation_selected.json', dict(selected=dict(
                    evaluation=str(run / 'validation/summary.json')), selection='Fixed phase2 recipe'))
                # Only protein vectors are required to validate composition. Do not export test reactions yet.
                embeddings = run / 'selected_test/test_embeddings'
                embeddings.mkdir(parents=True, exist_ok=True)
                for f in (screen / 'proteins').glob('protein_*'):
                    if not (embeddings / f.name).exists():
                        (embeddings / f.name).symlink_to(f.resolve())
                if not (run / 'composition_validation/complete.json').exists():
                    invoke('generalization_screen_phase2_compose.py', ['--campaign', run, '--cross-root', cross,
                        '--fixed-alpha', .25, '--fixed-cap', 1, '--output-name', 'composition_validation',
                        '--validate-only'], gpu, run / 'composition_validation.log')
                val = json.loads((run / 'composition_validation/selection.json').read_text())['selected']
            record = dict(epoch=epoch, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                run=str(run), validation=val, test_used=False)
            atomic_json(result, record)
            records.append(record)
            atomic_json(root / 'validation_grid.json', dict(records=records))

        best = max(records, key=lambda r: r['validation']['value'])
        selection = dict(selected=best, recipe=dict(step=100, cap=1., alpha=.25),
            registry_sha256=sha(registry), selected_utc=datetime.now(timezone.utc).isoformat(),
            test_used_for_selection=False)
        atomic_json(root / 'selection.json', selection)
        run, checkpoint = Path(best['run']), Path(best['checkpoint'])
        state('selected_checkpoint_test', epoch=best['epoch'])
        if a.task != 'enzymemap':
            from generalization_reactzyme_architecture_phase2 import evaluate_selected
            evaluate_selected(root, dict(arm=run.name, checkpoint=str(checkpoint),
                test_config=task['test_config'], gpu=gpu), dict(recipe=selection['recipe']), 'cuda:0', a.task)
            result = root / 'selected_test_summary.json'
        else:
            screen = base / f"screen_epoch{best['epoch']-1}"
            run_job(dict(label=campaign.name, config=task['config'], checkpoint=str(checkpoint),
                refiner=str(run / 'training/step0100.pt'), output=str(run / 'selected_test'),
                protein_source=str(screen / 'proteins'), gpu=gpu,
                policy='Best predeclared epoch by full-library validation BEDROC85; fixed phase2 and composition'),
                dict(created_utc=selection['selected_utc'], catalog=str(cross / 'clipzyme_f3_catalog_v1'),
                     screening_protocol=str(cross / 'clipzyme_screening_evaluation_protocol_v2')))
            invoke('generalization_screen_phase2_compose.py', ['--campaign', run, '--cross-root', cross,
                '--fixed-alpha', .25, '--fixed-cap', 1], gpu, run / 'composition.log')
            result = run / 'composition_v1/test_evaluation/summary.json'
        atomic_json(root / 'complete.json', dict(test_summary=str(result), test_summary_sha256=sha(result),
            selected_epoch=best['epoch'], completed_utc=datetime.now(timezone.utc).isoformat()))
        state('complete', epoch=best['epoch'])
    except Exception as exc:
        atomic_json(root / 'failure.json', dict(error=repr(exc), failed_utc=datetime.now(timezone.utc).isoformat()))
        state('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
