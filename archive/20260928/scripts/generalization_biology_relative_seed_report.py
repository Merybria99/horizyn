#!/usr/bin/env python3
"""Compare both declared relative-loss weights with fixed matched seed controls."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from generalization_full_graph import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
CROSS = ROOT / 'runs/generalization_20260919_2251/cross_paper_retraining'
OUT = CROSS / 'v4_relative_biology_seed_replication_20260921_v1'
METRICS = ('bedroc85', 'bedroc20', 'ef0.05', 'ef0.1')


def report():
    records = []
    for seed in (42, 43, 44):
        sources = {}
        if seed == 42:
            sources['control'] = CROSS / 'v4_biological_geometry_20260921_v1/enzymemap/control/test_summary.json'
            for name, campaign in [('relative_0p1', 'v4_relative_biology_f3_20260921_v1'),
                                   ('relative_1', 'v4_relative_biology_f3_weight1_20260921_v1')]:
                sources[name] = CROSS / campaign / 'followup/enzymemap/enzymemap/f3_biology/test_summary.json'
        else:
            sources['control'] = CROSS / f'v4_biological_f3_seed_controls_20260921_v1/control_seed{seed}/followup/enzymemap/enzymemap/f3_biology/test_summary.json'
            for suffix in ('0p1', '1'):
                sources['relative_' + suffix] = OUT / f'weight{suffix}_seed{seed}/followup/enzymemap/enzymemap/f3_biology/test_summary.json'
        for name, path in sources.items():
            if path.exists():
                records.append(dict(seed=seed, method=name, source=str(path), source_sha256=sha(path),
                                    summary=json.loads(path.read_text())['summary']))
    values = {(r['seed'], r['method']): r['summary'] for r in records}
    completed = len(records) == 9
    aggregate, differences = {}, {}
    if completed:
        for table in ('table1', 'table2'):
            aggregate[table], differences[table] = {}, {}
            for metric in METRICS:
                aggregate[table][metric], differences[table][metric] = {}, {}
                for method in ('control', 'relative_0p1', 'relative_1'):
                    data = [values[s, method][table][metric] for s in (42, 43, 44)]
                    aggregate[table][metric][method] = dict(mean=statistics.mean(data), sample_sd=statistics.stdev(data))
                    if method != 'control':
                        delta = [values[s, method][table][metric] - values[s, 'control'][table][metric] for s in (42, 43, 44)]
                        differences[table][metric][method] = dict(values=delta, mean=statistics.mean(delta),
                            sample_sd=statistics.stdev(delta), improved_seeds=sum(d > 0 for d in delta))
    payload = dict(records=records, all_results_available=completed, aggregate=aggregate, paired_differences=differences,
        updated_utc=datetime.now(timezone.utc).isoformat(),
        interpretation='Same-seed contrasts using existing unannotated phase2 in every arm. No seed selection or ensemble. '
            'Seed42 controls are archived; seed43/44 controls are reused from matched attraction replication. '
            'Three seeds and repeated held-out testing do not establish confirmatory superiority.')
    atomic_json(OUT / 'seed_comparison.json', payload)
    lines = ['# Relative biological loss: matched EnzymeMap seeds', '',
        'Both previously declared relative-loss weights are replicated at seeds43 and44, alongside seed42. '
        'The unannotated controls use the same target data, architecture, optimization settings and duration. '
        'This comparison holds the existing phase2 objective unannotated in all arms, isolating the change '
        'to F3 supervision. All seeds are reported; their average is not an ensemble. Seeds change initialization, '
        'batch order and dropout.', '', f'Available model/seed results: {len(records)}/9.', '',
        '| Seed / method | T1 BEDROC85 | T1 BEDROC20 | T1 EF5 | T1 EF10 | T2 BEDROC85 | T2 BEDROC20 | T2 EF5 | T2 EF10 |',
        '| --- | ' + ' | '.join(['---:'] * 8) + ' |']
    for seed in (42, 43, 44):
        for name in ('control', 'relative_0p1', 'relative_1'):
            v = values.get((seed, name))
            cells = [f'{v[t][m]:.6f}' for t in ('table1', 'table2') for m in METRICS] if v else ['pending'] * 8
            lines.append(f'| {seed} / {name} | ' + ' | '.join(cells) + ' |')
    if completed:
        lines += ['', '| Table / metric | Control mean ± SD | Relative0.1 mean ± SD | Relative1 mean ± SD |',
            '| --- | ---: | ---: | ---: |']
        for table, metrics in aggregate.items():
            for metric, stats in metrics.items():
                cells = [f"{stats[name]['mean']:.6f} ± {stats[name]['sample_sd']:.6f}"
                         for name in ('control', 'relative_0p1', 'relative_1')]
                lines.append(f'| {table} / {metric} | ' + ' | '.join(cells) + ' |')
        lines += ['', '| Table / metric / loss | Mean paired change | Seeds improved /3 |', '| --- | ---: | ---: |']
        for table, metrics in differences.items():
            for metric, methods in metrics.items():
                for method, record in methods.items():
                    lines.append(f"| {table} / {metric} / {method} | {record['mean']:+.6f} | {record['improved_seeds']} |")
    lines += ['', 'Family-removal and shuffled-label studies remain seed42 experiments. Replicating the full '
        'loss does not replicate every family contribution. Benchmark tests have been inspected repeatedly; '
        'these comparisons are exploratory. The single-seed bootstrap intervals in other reports do not '
        'include this training variability.', '',
        f'[All records, source hashes and paired changes]({OUT}/seed_comparison.json) · '
        '[Relative1 contributions](v4_relative_biology_weight1_contributions_20260921.md) · '
        '[Attraction seed comparison](v4_biology_seed_sensitivity_20260921.md)', '']
    (ROOT / 'documents/v4_relative_biology_seed_sensitivity_20260921.md').write_text('\n'.join(lines))
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        complete = report()
        if complete:
            for task in json.loads((OUT / 'protocol.json').read_text())['tasks']:
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
