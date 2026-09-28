#!/usr/bin/env python3
"""Report a predeclared phase-2 recipe across three ReactZyme splits and EnzymeMap."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from generalization_reactzyme_architecture_phase2 import (
    ROOT, sha256, atomic_json, validate, score_exported_test)
from generalization_clipzyme_f3_screen import export_device_lock
from generalization_clipzyme_test_queue import run_job
from generalization_screen_replication import invoke


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--task', required=True)
    args = p.parse_args()
    out = args.campaign.resolve()
    plan = json.loads((out / 'protocol.json').read_text())
    task = next(t for t in plan['tasks'] if t['name'] == args.task)
    run = out / task['name']
    cross = out.parent
    source = Path(task['source'])
    head = run / 'training/step0100.pt'
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False

    def state(stage):
        atomic_json(run / 'state.json', dict(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat()))

    try:
        state('waiting_for_phase2_training')
        execution = json.loads((run / 'execution.json').read_text())
        while not (run / 'training/complete.json').exists():
            if not Path(f"/proc/{execution['pid']}").exists():
                raise RuntimeError('Training exited without a completion receipt')
            time.sleep(5)
        saved = torch.load(head, map_location='cpu', weights_only=False)
        if saved['registry']['test_used'] or saved['registry']['feature_manifest_sha256'] != sha256(run / 'features/manifest.json'):
            raise ValueError('Phase2 training lineage mismatch')
        state('validation')
        recipe = dict(step=100, cap=1., alpha=.25)
        freeze = dict(created_utc=datetime.now(timezone.utc).isoformat(), recipe=recipe,
            protocol_sha256=sha256(out / 'protocol.json'), phase2_checkpoint_sha256=sha256(head),
            feature_manifest_sha256=sha256(run / 'features/manifest.json'),
            test_used_for_selection=False, fixed_recipe=True, repeated_tests_exploratory=True)
        if task['scope'] == 'reactzyme':
            baseline = json.loads((source / 'validation.json').read_text())['records'][0]['summary']
            with export_device_lock('cuda:0'):
                validate(run, baseline, 'cuda:0', steps=(0, 25, 50, 100), caps=(1.,), alphas=(.25,))
            atomic_json(run / 'selection.json', freeze)
            # Test input caches are reused only after this fixed-recipe freeze.
            t = task['source_task']
            test_features = cross / 'reactzyme_fixed_enzymemap_recipe_transfer_v1' / t['name'] / t['arm'] / 'test_features'
            receipt = json.loads((test_features / 'complete.json').read_text())
            if receipt['checkpoint_sha256'] != sha256(t['checkpoint']):
                raise ValueError('Cached test vectors belong to another F3')
            model = run / 'model'
            model.mkdir()
            for name in ('features', 'training', 'anchors.pt'):
                (model / name).symlink_to((run / name).resolve())
            (model / 'test_features').symlink_to(test_features)
            state('fixed_recipe_test')
            with export_device_lock('cuda:0'):
                score_exported_test(run, dict(arm='model'), dict(recipe=recipe), 'cuda:0', t['split'])
            result = run / 'selected_test_summary.json'
        else:
            parent = json.loads((source / 'protocol.json').read_text())
            screen = Path(parent['validation_summary']).parent.parent
            invoke('generalization_clipzyme_f3_validation.py', [
                '--manifest', cross / 'clipzyme_manifests_v2/manifest.json',
                '--catalog', cross / 'clipzyme_f3_catalog_v1',
                '--embeddings', screen / 'validation_embeddings', '--output', run / 'validation',
                '--refiner', head, '--batch-size', 64], task['gpu'], run / 'validation.log')
            atomic_json(run / 'protocol.json', dict(parent, fixed_recipe=True,
                test_used_for_selection=False, shared_recipe_protocol_sha256=freeze['protocol_sha256']))
            atomic_json(run / 'validation_selected.json', dict(
                selected=dict(evaluation=str(run / 'validation/summary.json')),
                selection='Fixed 100-update ranking-augmented phase2; no validation or test selection'))
            atomic_json(run / 'selection.json', freeze)
            state('fixed_recipe_test')
            job = dict(label=out.name, config=parent['base_config'], checkpoint=parent['base_checkpoint'],
                refiner=str(head), output=str(run / 'test'), protein_source=str(screen / 'proteins'),
                reaction_source=str(source / 'test/test_embeddings'), gpu=task['gpu'],
                policy='Predeclared fixed phase2 pilot, reported even when validation declines')
            run_job(job, dict(created_utc=plan['created_utc'], catalog=str(cross / 'clipzyme_f3_catalog_v1'),
                             screening_protocol=str(cross / 'clipzyme_screening_evaluation_protocol_v2')))
            (run / 'selected_test').symlink_to(run / 'test')
            state('full_library_composition')
            invoke('generalization_screen_phase2_compose.py', [
                '--campaign', run, '--cross-root', cross, '--fixed-alpha', .25, '--fixed-cap', 1],
                task['gpu'], run / 'composition.log')
            result = run / 'composition_v1/test_evaluation/summary.json'
        atomic_json(run / 'complete.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
            test_summary=str(result), test_summary_sha256=sha256(result),
            protocol_sha256=freeze['protocol_sha256']))
        state('complete')
    except Exception as exc:
        atomic_json(run / 'failure.json', dict(error=repr(exc), failed_utc=datetime.now(timezone.utc).isoformat()))
        state('failed')
        raise


if __name__ == '__main__':
    main()
