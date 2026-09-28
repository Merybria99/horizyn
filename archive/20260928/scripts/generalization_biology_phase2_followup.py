#!/usr/bin/env python3
"""Run fixed-base phase2 biological variants and finish qualified Case1 checks."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from generalization_full_graph import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def invoke(campaign, script, arguments, label):
    with (campaign / f'{label}.log').open('a') as log:
        subprocess.run([sys.executable, '-u', str(ROOT / 'scripts' / script), *arguments],
                       cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task')
    args = parser.parse_args(); out = args.campaign.resolve()
    plan = json.loads((out / 'protocol.json').read_text())
    if args.task:
        task = next(t for t in plan['tasks'] if t['name'] == args.task)
        common = ['--campaign', str(out), '--task', args.task]
        atomic_json(out / args.task / 'execution.json', dict(pid=os.getpid(), stage='training',
                    updated_utc=datetime.now(timezone.utc).isoformat()))
        invoke(out / args.task, 'generalization_biological_geometry_run.py', common, 'training')
        for scope in ('validation', 'test'):
            atomic_json(out / args.task / 'execution.json', dict(pid=os.getpid(), stage=scope,
                        updated_utc=datetime.now(timezone.utc).isoformat()))
            invoke(out / args.task, 'generalization_biological_geometry_evaluate.py',
                   [*common, '--scope', scope], scope)
        atomic_json(out / args.task / 'execution.json', dict(pid=os.getpid(), stage='complete',
                    updated_utc=datetime.now(timezone.utc).isoformat()))
        return
    while True:
        invoke(out, 'generalization_biological_geometry_report.py', ['--campaign', str(out)], 'report')
        records = json.loads((out / 'comparison.json').read_text())['methods']
        if all(r['all_results_available'] for r in records):
            break
        for task in plan['tasks']:
            path = out / task['name'] / 'execution.json'
            if path.exists():
                status = json.loads(path.read_text())
                if status['stage'] != 'complete':
                    try:
                        os.kill(status['pid'], 0)
                    except ProcessLookupError as error:
                        raise RuntimeError(f'Phase2 task exited before completion: {task["name"]}') from error
        time.sleep(60)
    ties = ['--campaign', str(out)]
    for variant in plan['tie_report_variants']:
        ties += ['--variant', variant]
    invoke(out, 'generalization_biological_geometry_ties.py', ties, 'ties')
    qualified = {r['variant']['name'] for r in records if r['all_primary_targets_exceeded']}
    for variant in plan['case1_variants']:
        method = out / 'methods' / variant
        if variant in qualified and not (method / 'case1/complete.json').exists():
            invoke(method, 'generalization_shared_recipe_case1.py',
                   ['--campaign', str(method), '--device', 'cuda:3'], 'case1')
    invoke(out, 'generalization_biological_geometry_report.py', ['--campaign', str(out)], 'report')
    atomic_json(out / 'report_complete.json', dict(completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
