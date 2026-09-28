#!/usr/bin/env python3
"""Evaluate every qualified phase2 annotation control on the same Case1 panel."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from generalization_full_graph import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
CROSS = ROOT / 'runs/generalization_20260919_2251/cross_paper_retraining'
OUT = CROSS / 'v4_relative_biology_phase2_case1_controls_20260921_v1'
PARENT = CROSS / 'v4_relative_biology_phase2_20260921_v1'


def main():
    plan = json.loads((OUT / 'protocol.json').read_text())
    for task in plan['tasks']:
        campaign = Path(task['campaign'])
        while not (campaign / 'report_complete.json').exists():
            time.sleep(30)
        method = campaign / 'methods' / task['variant']
        qualification = json.loads((method / 'qualification.json').read_text())
        if qualification['all_primary_targets_exceeded'] and not (method / 'case1/complete.json').exists():
            with (method / 'case1.log').open('a') as log:
                subprocess.run([sys.executable, '-u', 'scripts/generalization_shared_recipe_case1.py',
                    '--campaign', str(method), '--device', 'cuda:3'], cwd=ROOT, check=True,
                    stdout=log, stderr=subprocess.STDOUT)
    records = []
    lines = ['# Biological family contributions on retrospective Case1', '',
        'The original V4 F3 is fixed. These comparisons remove one annotation family from the existing '
        'phase2 objective, retaining the other coefficients and fixed divisor3, or shuffle assignments. '
        'All four controls at both weights are evaluated if they exceed all14 benchmark targets. '
        'No control is selected using Case1. This post-outcome diagnostic is exploratory; the panel '
        'contains related constructs and heterogeneous assays.', '',
        'The table uses the Reaction-Sim model, matching the primary V4 Case1 report. All four '
        'target-trained models and both candidate views remain in each source summary.', '',
        '| Phase2 weight / labels | Unique catalysts @25 /12 (123 sequences) | Paper entries @25 /15 (144 entries) | Conditional AUROC |',
        '| --- | ---: | ---: | ---: |']
    for weight in (1, 3):
        sources = [('all', PARENT / f'methods/all_{weight}/case1/summary.json')]
        sources += [(t['variant'], Path(t['campaign']) / 'methods' / t['variant'] / 'case1/summary.json')
                    for t in plan['tasks'] if t['weight'] == weight]
        for variant, path in sources:
            if not path.exists():
                lines.append(f'| {weight} / {variant} | not qualified | — | — |')
                continue
            data = json.loads(path.read_text())
            record = dict(weight=weight, variant=variant, methods=data['methods'], source=str(path), source_sha256=sha(path))
            records.append(record)
            r = data['methods']['reaction_smi/selected']
            lines.append(f'| {weight} / {variant} | {r["primary_papers"]["recovered_at_25"]} | '
                f'{r["entry_level_144"]["primary_papers"]["recovered_at_25"]} | '
                f'{r["broad_assay_conditional_discrimination"]["auc"]:.6f} |')
    lines += ['', 'V4 without added supervision recovers10/12 unique catalysts and10/15 paper entries '
        'at25, AUROC0.612875. The full-label gain to11/12 is TsT4Ease WT moving from26 to25; '
        'it is a cutoff crossing. No candidate-IID confidence interval or prospective generalization '
        'claim is made.', '',
        '[Weight1 benchmark contributions](v4_relative_biology_phase2_weight1_controls_20260921.md) · '
        '[Weight3 benchmark contributions](v4_relative_biology_phase2_weight3_controls_20260921.md)', '',
        f'[All targets and source hashes]({OUT}/comparison.json)', '']
    atomic_json(OUT / 'comparison.json', dict(records=records, updated_utc=datetime.now(timezone.utc).isoformat()))
    (ROOT / 'documents/v4_relative_biology_phase2_case1_controls_20260921.md').write_text('\n'.join(lines))
    if not (OUT / 'report_complete.json').exists():
        with open('/tmp/enzymediscovery_findings_append.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with (ROOT / 'findings.md').open('a') as stream:
                stream.write('\n\n## Phase2 family controls: benchmark and Case1 follow-up complete\n\n'
                    'Matched family-removal and shuffled controls at weights1/3 are complete. '
                    'All qualified controls receive the same retrospective144-entry Case1 follow-up, '
                    'with the123-sequence view kept separate. No architecture changes or additional '
                    'training data were introduced. '
                    '[All Case1 contributions and source hashes](documents/v4_relative_biology_phase2_case1_controls_20260921.md); '
                    'the report links both full benchmark contribution tables.\n')
    atomic_json(OUT / 'report_complete.json', dict(completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
