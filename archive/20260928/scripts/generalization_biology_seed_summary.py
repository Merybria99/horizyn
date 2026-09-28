#!/usr/bin/env python3
"""Rebuild the fixed three-seed comparison from archived screening results."""
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from generalization_full_graph import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
CROSS = ROOT / 'runs/generalization_20260919_2251/cross_paper_retraining'
METRICS = ('bedroc85', 'bedroc20', 'ef0.05', 'ef0.1')


def main():
    out = CROSS / 'v4_biological_f3_seed_controls_20260921_v1'
    records = []
    for seed in (42, 43, 44):
        for kind in ('control', 'biology'):
            if seed == 42:
                relative = ('v4_biological_geometry_20260921_v1/enzymemap/control/test_summary.json'
                            if kind == 'control' else 'v4_biological_f3_20260921_v1/'
                            'followup/enzymemap/enzymemap/f3_biology/test_summary.json')
                path = CROSS / relative
            else:
                path = out / f'{kind}_seed{seed}/followup/enzymemap/enzymemap/f3_biology/test_summary.json'
            summary = json.loads(path.read_text())['summary']
            records.append(dict(seed=seed, supervision=kind, source=str(path), source_sha256=sha(path),
                                summary=summary))
    values = {(r['seed'], r['supervision']): r['summary'] for r in records}
    aggregate, paired = {}, {}
    for table in ('table1', 'table2'):
        aggregate[table], paired[table] = {}, {}
        for metric in METRICS:
            aggregate[table][metric] = {}
            for kind in ('control', 'biology'):
                v = [values[seed, kind][table][metric] for seed in (42, 43, 44)]
                aggregate[table][metric][kind] = dict(mean=statistics.mean(v), std=statistics.stdev(v),
                                                     minimum=min(v), maximum=max(v))
            delta = [values[s, 'biology'][table][metric] - values[s, 'control'][table][metric]
                     for s in (42, 43, 44)]
            paired[table][metric] = dict(seeds=[42, 43, 44], values=delta,
                mean=statistics.mean(delta), std=statistics.stdev(delta), improved_seeds=sum(d > 0 for d in delta))
    atomic_json(out / 'seed_comparison.json', dict(records=records, aggregate=aggregate,
        paired_deltas=paired, updated_utc=datetime.now(timezone.utc).isoformat(),
        interpretation='Three fixed seeds, unannotated phase2 in both arms; no ensembling or seed selection. '
            'Seed42 uses archived V4 control. Descriptive seed variability, not confirmatory inference.'))
    lines = ['# Biological F3: EnzymeMap seed sensitivity', '',
        'Three fixed seeds, no seed selection or score ensembling. All have original unannotated phase2; '
        'this isolates F3 annotation supervision within each pair. Seed42 control is the archived V4 fit, '
        'seeds43/44 are new matched controls. The seed affects initialization, batch order and dropout; '
        'this is not an isolated initialization ablation. Few seeds and repeated tests preclude a confirmatory significance claim.', '',
        '| Table / metric | V4 mean ± sample SD | Biological F3 mean ± sample SD | Paired mean difference | Seeds improved /3 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for table, metrics in aggregate.items():
        for metric, stats in metrics.items():
            formatted = [f"{stats[k]['mean']:.6f} ± {stats[k]['std']:.6f}" for k in ('control', 'biology')]
            d = paired[table][metric]
            lines.append(f'| {table} / {metric} | ' + ' | '.join(formatted) +
                         f" | {d['mean']:+.6f} | {d['improved_seeds']} |")
    lines += ['', 'The three-seed biological mean exceeds the eight primary published screening point estimates, '
        'but individual seeds do not all win. Variation across seeds is much larger than most paired '
        'biological-loss gains. BEDROC85 improves in two seeds and declines in one; BEDROC20 improves in all three. '
        'This does not show that correctly assigned labels are necessary: the shuffled-label audit at seed42 '
        'remains a counterexample.', '',
        '![Paired seed BEDROC85 comparison](figures/v4_biology_seed_bedroc85.png)', '',
        '[Vector figure](figures/v4_biology_seed_bedroc85.svg) · [PDF](figures/v4_biology_seed_bedroc85.pdf)', '',
        f'[Full records and all metrics]({out}/seed_comparison.json) · '
        '[Contribution controls](v4_biology_contributions_20260921.md)', '']
    (ROOT / 'documents/v4_biology_seed_sensitivity_20260921.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
