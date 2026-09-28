#!/usr/bin/env python3
"""Evaluate one frozen cross-benchmark recipe on EnzymeMap without fallback."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

from generalization_screen_replication import ROOT, sha, invoke
from generalization_clipzyme_ablation_watch import evaluate
from generalization_clipzyme_test_queue import run_job, write_json
from generalization_clipzyme_f3_screen import export_device_lock
from generalization_gpu_budget import free_memory_mib
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    out = args.campaign.resolve()
    plan = json.loads((out / 'protocol.json').read_text())
    task, recipe = plan['enzymemap'], plan['recipe']
    run = Path(task['run_root'])
    cross, gpu = out.parent, task['gpu']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    if sha(task['config']) != task['config_sha256']:
        raise ValueError('Frozen config changed')
    for item in plan['reactzyme']:
        if sha(item['result']) != item['result_sha256']:
            raise ValueError('ReactZyme reference changed')
    execution = dict(started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                     protocol_sha256=sha(out / 'protocol.json'), source_sha256=sha(__file__))
    write_json(out / 'execution.json', execution)

    def state(stage):
        value = dict(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(run / 'state.json', value)
        print(json.dumps(value), flush=True)

    epoch = recipe['epochs'] - 1
    original = run / f'checkpoints/screen_selection/screen-epoch={epoch:02d}.ckpt'
    phase = run / 'phase2'
    features = phase / 'features'
    calibrated = run / 'calibrated'
    head = phase / 'training/step0100.pt'
    try:
        if not (run / 'train_complete.json').exists():
            state('waiting_for_training_gpu')
            with export_device_lock('cuda:0'):
                while free_memory_mib(gpu) < 95000:
                    time.sleep(10)
                state('fresh_base_training')
                invoke('train_protein_pooling_fast_io.py', [
                    '--config', task['config'], '--io-mode', 'residue',
                    '--io-output-dir', run / 'training', '--io-prefetch', 1],
                    gpu, run / 'training.log')
            write_json(run / 'train_complete.json', dict(checkpoint_sha256=sha(original)))
        if sha(original) != json.loads((run / 'train_complete.json').read_text())['checkpoint_sha256']:
            raise ValueError('Training checkpoint changed')

        if not (run / 'phase2_export_complete.json').exists():
            state('unmodified_base_feature_export')
            features.mkdir(parents=True, exist_ok=True)
            source = cross / 'sleec_multiview_phase2_v1/features'
            for name in ('catalog.json', 'pairs.npz', 'reaction_features.npz'):
                target = features / name
                if not target.exists():
                    target.symlink_to((source / name).resolve())
            # The exporter updates HDF5 provenance attributes, so this must be
            # a private copy rather than a writable link into another run.
            if not (features / 'protein_mean.h5').exists():
                shutil.copyfile(source / 'protein_mean.h5', features / 'protein_mean.h5')
            manifest = json.loads((source / 'manifest.json').read_text())
            for key in ('checkpoint', 'f3_export_config', 'f3_residue_cache'):
                manifest.pop(key, None)
            manifest['raw_feature_reuse'] = dict(source_manifest=str(source / 'manifest.json'),
                sha256=sha(source / 'manifest.json'), purpose='Raw training/dev features only')
            manifest['sources']['config'] = dict(path=task['config'], sha256=task['config_sha256'])
            write_json(features / 'manifest.json', manifest)
            # This older exporter does not acquire the shared GPU lock itself.
            with export_device_lock('cuda:0'):
                invoke('generalization_clipzyme_phase2_export.py', [
                    '--stage', 'f3', '--catalog', cross / 'clipzyme_f3_catalog_v1',
                    '--config', task['config'], '--checkpoint', original, '--output', features,
                    '--batch-size', 128, '--residue-cache',
                    '/tmp/enzymediscovery_f3_20260920/train_validation_prott5.h5'],
                    gpu, run / 'phase2_export.log')
            write_json(run / 'phase2_export_complete.json', dict(manifest_sha256=sha(features / 'manifest.json')))

        if not (phase / 'training/complete.json').exists():
            state('fixed_phase2_training')
            p2 = recipe['phase2']
            invoke('generalization_full_graph.py', [
                '--features', features, '--output', phase / 'training', '--steps', p2['steps'],
                '--snapshot-every', p2['steps'], '--selection-method', 'external_screening',
                '--identity-weight', p2['identity_weight'], '--temperature', p2['temperature'],
                '--contrastive-objective', p2['objective'], '--learning-rate', p2['learning_rate'],
                '--seed', p2['seed'], '--cpu-threads', 4], gpu, run / 'phase2_training.log')
        if not (phase / 'anchors.pt').exists():
            state('training_only_semantic_dictionary')
            invoke('generalization_clipzyme_phase2_dictionary.py', [
                '--features', features, '--output', phase / 'anchors.pt'], gpu, run / 'dictionary.log')

        new_checkpoint = calibrated / f'checkpoints/screen_selection/screen-epoch={epoch:02d}.ckpt'
        if not (run / 'fusion_receipt.json').exists():
            state('fixed_inference_fusion_adjustment')
            import torch
            torch.set_num_threads(4)
            saved = torch.load(original, map_location='cpu', weights_only=False)
            key, = [k for k in saved['state_dict'] if k.endswith('multiview_encoder.raw_residual_scale')]
            old = saved['state_dict'][key].clone()
            scale = float(.5 * old.sigmoid())
            desired = scale * recipe['inference_fusion_multiplier']
            if not 0 < desired < .5:
                raise ValueError('Fixed fusion multiplier lies outside model domain; no fallback allowed')
            saved['state_dict'][key] = old.new_tensor(math.log(desired / (.5 - desired)))
            receipt = dict(original_checkpoint_sha256=sha(original), source_checkpoint=str(original),
                source_checkpoint_sha256=sha(original), parameter=key, requested_multiplier=recipe['inference_fusion_multiplier'],
                original_scale=scale, multiplier=recipe['inference_fusion_multiplier'],
                new_scale=float(.5 * saved['state_dict'][key].sigmoid()),
                phase2_fitted_before_adjustment=True, all_other_model_parameters_unchanged=True,
                fixed_before_enzymemap_training=True)
            saved['inference_fusion_calibration'] = receipt
            for name in ('optimizer_states', 'lr_schedulers', 'loops', 'callbacks'):
                saved.pop(name, None)
            new_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(saved, new_checkpoint)
            del saved
            (calibrated / 'configs').mkdir(exist_ok=True)
            (calibrated / 'configs/train.yaml').symlink_to(Path(task['config']))
            receipt['calibrated_checkpoint_sha256'] = sha(new_checkpoint)
            write_json(run / 'fusion_receipt.json', receipt)

        state('full_library_validation')
        evaluate(calibrated, epoch, SimpleNamespace(catalog=cross / 'clipzyme_f3_catalog_v1',
            protocol=cross / 'clipzyme_screening_evaluation_protocol_v2',
            manifest=cross / 'clipzyme_manifests_v2/manifest.json', epochs=[epoch], gpus=[2, 3, 2, 3]))
        screen = calibrated / f'screen_epoch{epoch}'
        if not (phase / 'validation/summary.json').exists():
            invoke('generalization_clipzyme_f3_validation.py', [
                '--manifest', cross / 'clipzyme_manifests_v2/manifest.json',
                '--catalog', cross / 'clipzyme_f3_catalog_v1',
                '--embeddings', screen / 'validation_embeddings', '--output', phase / 'validation',
                '--refiner', head, '--batch-size', 64, '--fusion-calibration', run / 'fusion_receipt.json',
                '--calibrated-base-checkpoint', new_checkpoint], gpu, run / 'phase2_validation.log')

        write_json(phase / 'protocol.json', dict(base_config=str(calibrated / 'configs/train.yaml'),
            base_checkpoint=str(new_checkpoint), base_checkpoint_sha256=sha(new_checkpoint),
            validation_summary=str(screen / 'validation_evaluation/summary.json'),
            fixed_recipe=True, seed=recipe['seed'], recipe=recipe,
            original_head_training_checkpoint_sha256=sha(original),
            test_used_for_selection=False, discovery_context=plan['discovery_context']))
        write_json(phase / 'validation_selected.json', dict(
            selected=dict(evaluation=str(phase / 'validation/summary.json')),
            selection='Fixed shared recipe; validation does not choose a checkpoint, seed, or multiplier'))
        state('fixed_recipe_test')
        job = dict(label=out.name, config=str(calibrated / 'configs/train.yaml'),
            checkpoint=str(new_checkpoint), refiner=str(head), output=str(phase / 'test'),
            fusion_calibration=str(run / 'fusion_receipt.json'),
            protein_source=str(screen / 'proteins'), gpu=gpu, policy=plan['fixed_test_policy'])
        run_job(job, dict(created_utc=plan['created_utc'], catalog=str(cross / 'clipzyme_f3_catalog_v1'),
                         screening_protocol=str(cross / 'clipzyme_screening_evaluation_protocol_v2')))
        if not (phase / 'selected_test').exists():
            (phase / 'selected_test').symlink_to(phase / 'test', target_is_directory=True)
        if not (phase / 'composition_v1/test_evaluation/summary.json').exists():
            state('fixed_semantic_composition')
            with export_device_lock('cuda:0'):
                invoke('generalization_screen_phase2_compose.py', [
                    '--campaign', phase, '--cross-root', cross, '--fixed-alpha', recipe['semantic_alpha'],
                    '--fixed-cap', recipe['residual_cap']], gpu, run / 'composition.log')
        result = phase / 'composition_v1/test_evaluation/summary.json'
        write_json(run / 'complete.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
            test_summary=str(result), test_summary_sha256=sha(result), protocol_sha256=sha(out / 'protocol.json')))
        state('complete')
    except Exception as exc:
        write_json(run / 'failure.json', dict(error=repr(exc), failed_utc=datetime.now(timezone.utc).isoformat()))
        raise


if __name__ == '__main__':
    main()
