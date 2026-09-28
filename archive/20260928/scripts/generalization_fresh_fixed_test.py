#!/usr/bin/env python3
"""Evaluate an explicitly frozen same-epoch recipe using existing target-specific fits."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from generalization_screen_replication import ROOT, sha, invoke
from generalization_full_graph import atomic_json
from generalization_reactzyme_architecture_phase2 import evaluate_selected
from generalization_clipzyme_test_queue import run_job


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=3)
    a = p.parse_args(); campaign = a.campaign.resolve()
    plan = json.loads((campaign / 'protocol.json').read_text())
    cross = campaign.parent
    for task in plan['tasks']:
        run = Path(task['run_root']) / 'phase2_followup'
        run.mkdir(parents=True, exist_ok=True)
        if (run / 'complete.json').exists():
            continue
        source = Path(task['source_phase2'])
        if sha(task['checkpoint']) != task['checkpoint_sha256'] or sha(task['config']) != task['config_sha256']:
            raise ValueError('Frozen checkpoint/config changed')
        for f, key in [('features/manifest.json', 'feature_manifest_sha256'),
                       ('training/step0100.pt', 'phase2_checkpoint_sha256'), ('anchors.pt', 'dictionary_sha256')]:
            if sha(source / f) != task[key]:
                raise ValueError('Frozen phase2 artifact changed')
        model = run / 'model'
        model.mkdir(exist_ok=True)
        for name in ('features', 'training', 'anchors.pt'):
            if not (model / name).exists():
                (model / name).symlink_to((source / name).resolve())
        recipe = dict(step=100, cap=1., alpha=.25)
        selection = dict(selected=dict(epoch=plan['fixed_epoch'], checkpoint=task['checkpoint'],
            checkpoint_sha256=task['checkpoint_sha256'], run=str(source), test_used=False),
            recipe=recipe, test_used_for_selection=False, fixed_epoch=True,
            protocol_sha256=sha(campaign / 'protocol.json'), selected_utc=plan['created_utc'])
        atomic_json(run / 'selection.json', selection)
        atomic_json(run / 'state.json', dict(stage='fixed_epoch_test', epoch=plan['fixed_epoch']))
        if task['name'] != 'enzymemap':
            evaluate_selected(run, dict(arm='model', checkpoint=task['checkpoint'],
                test_config=task['test_config'], gpu=a.gpu), dict(recipe=recipe), 'cuda:0', task['name'])
            result = run / 'selected_test_summary.json'
        else:
            for name in ('protocol.json', 'validation_selected.json'):
                (model / name).write_text((source / name).read_text())
            parent = json.loads((model / 'protocol.json').read_text())
            screen = Path(parent['validation_summary']).parent.parent
            run_job(dict(label=campaign.name, config=task['config'], checkpoint=task['checkpoint'],
                refiner=str(source / 'training/step0100.pt'), output=str(model / 'selected_test'),
                protein_source=str(screen / 'proteins'), gpu=a.gpu,
                policy='Fixed epoch-10 shared recipe declared before these tests; no checkpoint fallback'),
                dict(created_utc=plan['created_utc'], catalog=str(cross / 'clipzyme_f3_catalog_v1'),
                     screening_protocol=str(cross / 'clipzyme_screening_evaluation_protocol_v2')))
            invoke('generalization_screen_phase2_compose.py', ['--campaign', model,
                '--cross-root', cross, '--fixed-alpha', .25, '--fixed-cap', 1], a.gpu, run / 'composition.log')
            result = model / 'composition_v1/test_evaluation/summary.json'
        atomic_json(run / 'complete.json', dict(test_summary=str(result), test_summary_sha256=sha(result),
            selected_epoch=plan['fixed_epoch'], completed_utc=datetime.now(timezone.utc).isoformat()))
        atomic_json(run / 'state.json', dict(stage='complete', epoch=plan['fixed_epoch']))
        print(json.dumps(dict(completed=task['name'], result=str(result))), flush=True)
    subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/generalization_fresh_goal_finish.py'),
                    '--campaign', str(campaign), '--gpu', str(a.gpu)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
