#!/usr/bin/env python3
"""Test one checkpoint per arm after its fixed full-library validation grid."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from generalization_clipzyme_test_queue import run_job, write_json

ROOT = Path(__file__).resolve().parents[1]


def select_task(task, epochs, destination):
    run = Path(task['run_root'])
    summaries = [run / f'screen_epoch{epoch-1}' / 'validation_evaluation/summary.json'
                 for epoch in epochs]
    if not all(path.exists() for path in summaries):
        return None
    selection = destination / f"{task['arm']}_validation_selection.json"
    if not selection.exists():
        command = [sys.executable, str(ROOT / 'scripts/generalization_clipzyme_checkpoint_select.py')]
        for path in summaries:
            command += ['--evaluation', str(path)]
        subprocess.run([*command, '--output', str(selection)], check=True,
                       stdout=subprocess.DEVNULL)
    selected = json.loads(selection.read_text())
    # Bind an existing selection to this exact fixed grid before using it.
    if [row['evaluation'] for row in selected['predeclared_grid']] != list(map(str, summaries)):
        raise ValueError('Validation selection belongs to a different checkpoint grid')
    index = selected['selected_index']
    epoch = epochs[index]
    checkpoint = run / f'checkpoints/screen_selection/screen-epoch={epoch-1:02d}.ckpt'
    digest = hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest()
    if digest != selected['selected']['checkpoint_sha256']:
        raise ValueError('Selected weights differ from validated weights')
    return dict(label=f"{task['arm']}_validation_selected_epoch{epoch}",
                config=task['train_config'], checkpoint=str(checkpoint),
                output=str(run / f'screen_epoch{epoch-1}/requested_test'),
                protein_source=str(run / f'screen_epoch{epoch-1}/proteins'),
                policy='One checkpoint selected on full-library validation BEDROC85; no test selection',
                validation_selection=str(selection), epoch_one_based=epoch, gpu=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=4)
    args = parser.parse_args()
    plan = json.loads((args.campaign / 'protocol.json').read_text())
    cross = args.campaign.parent
    out = args.campaign / 'selected_tests'
    out.mkdir(exist_ok=True)
    protocol = dict(created_utc=plan['created_utc'],
                    catalog=str(cross / 'clipzyme_f3_catalog_v1'),
                    screening_protocol=str(cross / 'clipzyme_screening_evaluation_protocol_v2'))
    tasks = [t for t in plan['tasks'] if t['benchmark'] == 'enzymemap']
    deadline = time.monotonic() + 3600 * args.hours
    active, finished, failed = {}, {}, {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        while time.monotonic() < deadline:
            for name, (future, job) in list(active.items()):
                if future.done():
                    try:
                        future.result()
                        finished[name] = job
                    except Exception as exc:
                        failed[name] = repr(exc)
                    del active[name]
            waiting = []
            for task in tasks:
                name = task['arm']
                if name in finished or name in active or name in failed:
                    continue
                if active:
                    waiting.append(dict(arm=name, reason='Evaluator busy'))
                    continue
                try:
                    job = select_task(task, plan['validation_epochs'], out)
                    if job is None:
                        waiting.append(dict(arm=name, reason='Fixed full-library validation grid incomplete'))
                        continue
                    if (Path(job['output']) / 'complete.json').exists():
                        finished[name] = job
                    else:
                        write_json(out / f'{name}_test_plan.json', dict(protocol=protocol, job=job))
                        active[name] = (pool.submit(run_job, job, protocol), job)
                        print(json.dumps(dict(starting=name, epoch=job['epoch_one_based'])), flush=True)
                except Exception as exc:
                    failed[name] = repr(exc)
            write_json(out / 'status.json', dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                       active=list(active), finished=finished, failed=failed, waiting=waiting,
                       selection_metric='full_library_validation.table1.bedroc85',
                       mrr_used_for_selection=False, test_used_for_selection=False))
            if len(finished) + len(failed) == len(tasks):
                return
            time.sleep(20)


if __name__ == '__main__':
    main()
