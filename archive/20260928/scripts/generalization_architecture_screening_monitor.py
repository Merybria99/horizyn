#!/usr/bin/env python3
"""Report full-library EnzymeMap BEDROC/EF separately for validation and test."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

ARMS = ('small_reaction_f3', 'lean_mean', 'residue_views', 'fingerprint_mean')
METRICS = ('bedroc85', 'bedroc20', 'ef0.05', 'ef0.1')
EXPECTED = {'validation': ((2652, 261907), (2216, 252113)),
            'test': ((1521, 261907), (1337, 252113))}


def atomic_write(path, text):
    temporary = path.with_suffix('.tmp' + path.suffix)
    temporary.write_text(text)
    temporary.replace(path)


def read_result(path, scope):
    raw = json.loads(path.read_text())
    if scope == 'validation' and not (raw.get('validation_only') and raw.get('test_labels_read') is False):
        raise ValueError(f'Validation provenance missing: {path}')
    for table, (queries, candidates) in zip(('table1', 'table2'), EXPECTED[scope]):
        values = raw['summary'][table]
        if (values['queries'], values['candidate_ids']) != (queries, candidates):
            raise ValueError(f'Incomplete or different screening pool: {path}, {table}')
    return dict(source=str(path), summary=raw['summary'],
                checkpoint_sha256=raw.get('checkpoint_sha256'),
                selection_value=raw.get('selection_value'))


def report(campaign):
    protocol_path = campaign / 'protocol.json'
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    arms = tuple(task['arm'] for task in protocol.get('tasks', [])
                 if task.get('benchmark') == 'enzymemap')
    # The original fingerprint follow-up has its own protocol.
    arms = tuple(dict.fromkeys((*arms, *[name for name in ARMS
                  if (campaign / 'enzymemap' / name).is_dir()]))) or ARMS
    first_epoch = min(protocol.get('validation_epochs', [20]))
    rows, errors = [], []
    for arm in arms:
        run = campaign / 'enzymemap' / arm
        entry = dict(arm=arm, validation=[], test=[], latest_logged_epoch=None)
        logs = set(run.glob('logs/train/**/metrics.csv'))
        logs.update(run.glob('optimized_io/logs/**/metrics.csv'))
        epochs = []
        for log in logs:
            with log.open() as stream:
                epochs.extend(int(float(row['epoch'])) + 1
                              for row in csv.DictReader(stream) if row.get('epoch'))
        entry['latest_logged_epoch'] = max(epochs) if epochs else None
        for scope, pattern in [('validation', 'screen_epoch*/validation_evaluation/summary.json'),
                               ('test', 'screen_epoch*/requested_test/test_evaluation/summary.json')]:
            for path in sorted(run.glob(pattern)):
                try:
                    result = read_result(path, scope)
                    result['epoch_one_based'] = int(path.relative_to(run).parts[0].removeprefix('screen_epoch')) + 1
                    entry[scope].append(result)
                except (KeyError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(dict(source=str(path), error=str(exc)))
            entry[scope].sort(key=lambda result: result['epoch_one_based'])
        # Only validation controls the reported selection. Test rows are displayed
        # independently and can never enter this comparison.
        entry['validation_selected'] = max(entry['validation'],
            key=lambda x: x['summary']['table1']['bedroc85'], default=None)
        rows.append(entry)
    payload = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                   benchmark='EnzymeMap full-library screening',
                   selection_metric='validation.table1.bedroc85',
                   additional_metrics=['bedroc20', 'ef0.05', 'ef0.1'],
                   first_saved_epoch_one_based=first_epoch, test_results_exploratory=True,
                   phase2_applied=False, arms=rows, errors=errors)
    text = ['# EnzymeMap architecture pilots: full-library screening', '',
            f"Updated {payload['updated_utc']}", '',
            'Checkpoint selection: **validation BEDROC85**. BEDROC20, EF5 and EF10 are reported alongside it.',
            f'The running pilots first save at epoch {first_epoch}. Small-pool MRR is not a screening result or selection criterion.',
            'These fresh-base pilots precede phase 2. Test readouts are exploratory after repeated test inspection.', '']
    for scope in ('validation', 'test'):
        text.extend([f'## {scope.title()}', '',
                     '| Model | Latest logged epoch | Evaluated epoch | BEDROC85 | BEDROC20 | EF5 | EF10 |',
                     '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for row in rows:
            candidates = row[scope]
            if not candidates:
                text.append(f"| {row['arm']} | {row['latest_logged_epoch'] or 'starting'} | pending | — | — | — | — |")
            for result in candidates:
                values = result['summary']['table1']
                cells = ' | '.join(f'{values[m]:.6f}' for m in METRICS)
                text.append(f"| {row['arm']} | {row['latest_logged_epoch'] or 'starting'} | {result['epoch_one_based']} | {cells} |")
        queries, candidates = EXPECTED[scope][0]
        text.extend(['', f'Table 1 uses {queries:,} queries and {candidates:,} candidate IDs. ',
                     'Table 2 results and exact source paths are retained in `screening_metrics.json`.', ''])
    if errors:
        text.extend(['## Results withheld by protocol checks', '', *[str(x) for x in errors], ''])
    atomic_write(campaign / 'screening_metrics.json', json.dumps(payload, indent=2) + '\n')
    atomic_write(campaign / 'screening_metrics.md', '\n'.join(text))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--hours', type=float, default=8)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    started = time.monotonic()
    while time.monotonic() - started < a.hours * 3600:
        report(a.campaign)
        if a.once:
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
