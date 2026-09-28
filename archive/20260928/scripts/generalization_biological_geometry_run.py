#!/usr/bin/env python3
"""Train the predeclared loss-only ablations without changing V4 inference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generalization_full_graph import atomic_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--task', required=True)
    args = p.parse_args(); out = args.campaign.resolve()
    plan = json.loads((out / 'protocol.json').read_text())
    task = next(t for t in plan['tasks'] if t['name'] == args.task)
    if sha(Path(task['annotations'])) != task['annotations_sha256']:
        raise ValueError('Annotations changed after declaration')
    for variant in plan['variants']:
        dest = out / args.task / variant['name']; dest.mkdir(exist_ok=True)
        if (dest / 'training/complete.json').exists():
            continue
        if variant['name'] == 'control':
            (dest / 'training').symlink_to(out / 'control_training' / args.task / 'training')
            continue
        atomic_json(dest / 'declaration.json', dict(variant=variant,
            protocol_sha256=sha(out / 'protocol.json'), annotations_sha256=task['annotations_sha256'],
            created_utc=datetime.now(timezone.utc).isoformat(), test_used=False))
        cmd = [sys.executable, str(ROOT / 'scripts/generalization_full_graph.py'),
            '--features', str(Path(task['source_phase2']) / 'features'), '--output', str(dest / 'training'),
            '--steps', '100', '--snapshot-every', '100', '--selection-method', 'external_screening',
            '--temperature', '.2', '--identity-weight', '10', '--learning-rate', '.0001',
            '--cpu-threads', '4', '--biological-labels', task['annotations'],
            '--biology-mode', variant.get('mode', plan.get('biological_mode', 'attraction')),
            '--biology-margin', str(variant.get('margin', plan.get('biological_margin', .1)))]
        for f, weight in variant['weights'].items():
            cmd += ['--biology-' + f, str(weight)]
        if 'shuffle_seed' in variant:
            cmd += ['--biology-shuffle-seed', str(variant['shuffle_seed'])]
        with (dest / 'console.log').open('w') as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(json.dumps(dict(task=args.task, completed=variant['name'])), flush=True)
    atomic_json(out / args.task / 'training_complete.json', dict(
        variants=len(plan['variants']), completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
