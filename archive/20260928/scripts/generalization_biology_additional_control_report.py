#!/usr/bin/env python3
"""Report matched current-training and annotation-specificity controls."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from generalization_full_graph import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
CROSS = ROOT / 'runs/generalization_20260919_2251/cross_paper_retraining'
OUT = CROSS / 'v4_relative_biology_additional_controls_20260921_v1'
METRICS = [(t, m) for t in ('table1', 'table2') for m in ('bedroc85', 'bedroc20', 'ef0.05', 'ef0.1')]


def report():
    tasks = json.loads((OUT / 'protocol.json').read_text())['tasks']
    records = []
    lines = ['# Matched biological-loss controls beyond seed42', '',
        'All controls use the unchanged architecture, target-training associations and fixed 10 epochs. '
        'The first phase2 variant uses the original unannotated phase2 objective; the second also applies '
        'biological geometry to phase2. For shuffled and removed-family controls, their changed annotations '
        'apply throughout training. No control or seed is selected by its test result.', '']
    for method in ('f3_biology', 'f3_phase2_biology'):
        lines += [f'## {method}', '',
            'Each difference below is the corresponding correct-label model minus the indicated control. '
            'The fresh unannotated seed42 row therefore compares F3 biological supervision with no F3 '
            'supervision; the other rows test annotation assignment or mechanism removal.', '',
            '| Control | T1 BEDROC85 | T1 BEDROC20 | T1 EF5 | T1 EF10 | T2 BEDROC85 | T2 BEDROC20 | T2 EF5 | T2 EF10 |',
            '| --- | ' + ' | '.join(['---:'] * 8) + ' |']
        for task in tasks:
            suffix = f'followup/enzymemap/enzymemap/{method}/test_summary.json'
            path = Path(task['campaign']) / suffix
            counterpart = Path(task['counterpart']) / suffix
            if not path.exists() or not counterpart.exists():
                lines.append('| ' + task['variant'] + ' | ' + ' | '.join(['pending'] * 8) + ' |')
                continue
            control, correct = (json.loads(p.read_text())['summary'] for p in (path, counterpart))
            delta = {t: {m: correct[t][m] - control[t][m] for m in ('bedroc85', 'bedroc20', 'ef0.05', 'ef0.1')}
                     for t in ('table1', 'table2')}
            records.append(dict(control=task['variant'], method=method, correct=correct, baseline=control,
                correct_minus_control=delta, source=str(path), source_sha256=sha(path),
                counterpart=str(counterpart), counterpart_sha256=sha(counterpart)))
            lines.append('| ' + task['variant'] + ' | ' + ' | '.join(f'{delta[t][m]:+.6f}' for t, m in METRICS) + ' |')
    fresh = next((r for r in records if r['control'] == 'fresh_control_seed42' and r['method'] == 'f3_biology'), None)
    if fresh:
        archived_path = CROSS / 'v4_biological_geometry_20260921_v1/enzymemap/control/test_summary.json'
        archived = json.loads(archived_path.read_text())['summary']
        errors = [abs(fresh['baseline'][t][m] - archived[t][m]) for t, m in METRICS]
        lines += ['', '## Current versus archived unannotated seed42', '',
            f'The largest absolute difference across the eight screening metrics is {max(errors):.9g}. '
            'The two BEDROC85 differences are '
            f'{fresh["baseline"]["table1"]["bedroc85"] - archived["table1"]["bedroc85"]:+.9f} and '
            f'{fresh["baseline"]["table2"]["bedroc85"] - archived["table2"]["bedroc85"]:+.9f}. '
            'The end-to-end pipelines therefore do not reproduce every ranking exactly.', '',
            f'[Archived result]({archived_path}) · [Fresh control]({fresh["source"]})']
        parity_path = OUT / 'fresh_control_seed42/checkpoint_reproduction.json'
        if parity_path.exists():
            parity = json.loads(parity_path.read_text())
            lines += ['', f'A separate parameter audit verifies that all {len(parity["rows"])} F3 state '
                f'tensors are bit-for-bit identical: {parity["all_state_tensors_identical"]}. '
                f'Maximum parameter difference is {parity["maximum_parameter_difference"]:g}. '
                'Thus the small end-to-end differences arise after the identical F3 state, in feature export, '
                'phase2 fitting and/or scoring; the exact downstream source has not been isolated. '
                f'[Checkpoint hashes and tensor comparison]({parity_path}).']
    specificity = []
    for method in ('f3_biology', 'f3_phase2_biology'):
        lines += ['', f'## Annotation specificity across seeds: {method}', '',
            'Correct labels minus shuffled labels, using the same F3 weight 1 and seed within each pair. '
            'The same fixed annotation shuffle is used across seeds; this is not variability over shuffles.', '',
            '| Seed | T1 BEDROC85 | T1 BEDROC20 | T1 EF5 | T1 EF10 | T2 BEDROC85 | T2 BEDROC20 | T2 EF5 | T2 EF10 |',
            '| --- | ' + ' | '.join(['---:'] * 8) + ' |']
        paired = []
        for seed in (42, 43, 44):
            if seed == 42:
                suffix = f'followup/enzymemap/enzymemap/{method}/test_summary.json'
                correct_path = CROSS / 'v4_relative_biology_f3_weight1_20260921_v1' / suffix
                shuffled_path = CROSS / 'v4_relative_biology_f3_weight1_contributions_20260921_v1/shuffled' / suffix
                if not correct_path.exists() or not shuffled_path.exists():
                    continue
                correct = json.loads(correct_path.read_text())['summary']
                shuffled = json.loads(shuffled_path.read_text())['summary']
                delta = {t: {m: correct[t][m] - shuffled[t][m] for m in ('bedroc85', 'bedroc20', 'ef0.05', 'ef0.1')}
                         for t in ('table1', 'table2')}
            else:
                row = next((r for r in records if r['control'] == f'shuffled_seed{seed}' and r['method'] == method), None)
                if row is None:
                    continue
                delta = row['correct_minus_control']
            paired.append(delta)
            specificity.append(dict(method=method, seed=seed, correct_minus_shuffled=delta))
            lines.append(f'| {seed} | ' + ' | '.join(f'{delta[t][m]:+.6f}' for t, m in METRICS) + ' |')
        if len(paired) == 3:
            lines.append('| Mean | ' + ' | '.join(f'{sum(d[t][m] for d in paired) / 3:+.6f}' for t, m in METRICS) + ' |')
    if len(specificity) == 6:
        lines += ['', 'Correct annotation assignment improves BEDROC85 in two of three seeds, but worsens '
            'all eight metrics at seed44. The mechanism-removal seed43 check improves BEDROC85 when mechanism '
            'supervision is retained, while EF5 falls. These are conditional, mixed contributions; they do not '
            'support a universal benefit for all annotation families.']
    lines += ['', 'These controls were declared after earlier outcomes, so they remain exploratory. '
        'A positive full-loss-versus-removal contrast may include a regularization-strength effect. '
        'The shuffled-label control addresses whether the true annotation assignment matters. '
        'No result establishes a named biochemical meaning for an individual residue query.', '',
        f'[All absolute metrics, paired changes and source hashes]({OUT}/comparison.json) · '
        '[Three-seed results](v4_relative_biology_seed_sensitivity_20260921.md)', '']
    atomic_json(OUT / 'comparison.json', dict(records=records, annotation_specificity=specificity, completed=len(records) == 8,
        updated_utc=datetime.now(timezone.utc).isoformat(), exploratory_repeated_tests=True))
    (ROOT / 'documents/v4_biology_additional_controls_20260921.md').write_text('\n'.join(lines))
    return len(records) == 8, tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        complete, tasks = report()
        if complete:
            for task in tasks:
                subprocess.run([sys.executable, 'scripts/generalization_biological_f3_audit.py',
                    '--campaign', task['campaign']], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            atomic_json(OUT / 'report_complete.json', dict(completed_utc=datetime.now(timezone.utc).isoformat()))
        if complete or not args.watch:
            return
        for path in OUT.rglob('followup_state.json'):
            if json.loads(path.read_text())['stage'] == 'failed':
                raise RuntimeError(f'Follow-up failed: {path}')
        time.sleep(60)


if __name__ == '__main__':
    main()
