#!/usr/bin/env python3
"""Report all screening metrics and matched biological-family differences."""
import argparse
import json
from pathlib import Path

from generalization_full_graph import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
CROSS = ROOT / 'runs/generalization_20260919_2251/cross_paper_retraining'
METRICS = [(t, m) for t in ('table1', 'table2') for m in ('bedroc85', 'bedroc20', 'ef0.05', 'ef0.1')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--output-document', required=True, type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    plan = json.loads((campaign / 'protocol.json').read_text())
    default_controls = ('v4_relative_biology_f3_contributions_20260921_v1'
                        if plan.get('biological_mode') == 'relative' else 'v4_biological_f3_contributions_20260921_v1')
    controls = Path(plan.get('contribution_campaign', CROSS / default_controls))
    sources = {'V4': CROSS / 'v4_biological_geometry_20260921_v1/enzymemap/control/test_summary.json',
               'all signals': campaign / 'followup/enzymemap/enzymemap/f3_phase2_biology/test_summary.json'}
    for variant in ('without_ec', 'without_cofactor', 'without_mechanism', 'shuffled'):
        sources[variant] = controls / variant / 'followup/enzymemap/enzymemap/f3_phase2_biology/test_summary.json'
    records = {name: dict(summary=json.loads(path.read_text())['summary'], source=str(path), sha256=sha(path))
               for name, path in sources.items() if path.exists()}
    header = '| Supervision | T1 BEDROC85 | T1 BEDROC20 | T1 EF5 | T1 EF10 | T2 BEDROC85 | T2 BEDROC20 | T2 EF5 | T2 EF10 |'
    rule = '| --- | ' + ' | '.join(['---:'] * 8) + ' |'
    lines = [f"# Biological contribution audit: {plan.get('biological_mode', 'attraction')} loss", '',
        f"The unchanged V4 architecture is trained from scratch for 10 epochs, seed 42, with F3 biological weight "
        f"{plan.get('biological_weight', .1):g}. The existing 100-step phase2 uses weight 0.1 per retained family. "
        'Family removals apply throughout both stages, without renormalizing surviving family coefficients. '
        'The shuffled control permutes annotation profiles across training endpoints. Training associations and '
        'model components are unchanged. All declared controls are reported.', '', header, rule]
    for name in sources:
        cells = [f"{records[name]['summary'][t][m]:.6f}" for t, m in METRICS] if name in records else ['pending'] * 8
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Conditional contribution of each signal', '',
        'Each value is all-signals minus the corresponding removed-family result. Positive values mean '
        'retaining the family helped that metric in this run. These are conditional differences, not an '
        'additive decomposition or evidence across seeds.', '', header.replace('Supervision', 'Retained family'), rule]
    differences = {}
    for family in ('ec', 'cofactor', 'mechanism'):
        removed = 'without_' + family
        if 'all signals' in records and removed in records:
            differences[family] = {t: {m: records['all signals']['summary'][t][m] - records[removed]['summary'][t][m]
                                      for m in ('bedroc85', 'bedroc20', 'ef0.05', 'ef0.1')} for t in ('table1', 'table2')}
            cells = [f'{differences[family][t][m]:+.6f}' for t, m in METRICS]
        else:
            cells = ['pending'] * 8
        lines.append('| ' + family + ' | ' + ' | '.join(cells) + ' |')
    if 'all signals' in records and 'shuffled' in records:
        contrasts = [records['all signals']['summary'][t]['bedroc85'] - records['shuffled']['summary'][t]['bedroc85']
                     for t in ('table1', 'table2')]
        lines += ['', f'Correct-minus-shuffled BEDROC85 is {contrasts[0]:+.6f} in Table1 and '
            f'{contrasts[1]:+.6f} in Table2. ' + ('Shuffling performs better in both, so the BEDROC85 gain '
            'cannot be attributed specifically to correct biological assignments.' if all(x < 0 for x in contrasts)
            else 'These fixed-seed contrasts need replication before attributing an improvement to correct annotations.')]
    paired_path = campaign / 'enzymemap/paired_vs_shuffled.json'
    if paired_path.exists():
        paired = json.loads(paired_path.read_text())
        lines += ['', '## Uncertainty against shuffled labels', '',
            'Paired reaction-rule cluster bootstrap intervals describe these fixed trained models. They do not '
            'include variation across seeds or adjustment for repeated comparisons.', '',
            '| Table / metric | Correct minus shuffled | Rule-cluster bootstrap 95% interval |',
            '| --- | ---: | --- |']
        for table, block in paired['results'].items():
            for metric, record in block['metrics'].items():
                lo, hi = record['rule_cluster_bootstrap_95']
                lines.append(f"| {table} / {metric} | {record['paired_delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
        lines += ['', f'[Bootstrap inputs and hashes]({paired_path})']
    lines += ['', 'EC ancestry, structural cofactor presence and coarse mapped bond changes are imperfect biological '
        'proxies. Missing labels are neutral. Test results have been examined repeatedly, so these comparisons '
        'are exploratory. Three-seed attraction-loss controls are complete and expose substantial variation '
        'across seeds; they are not seed replication of the relative loss.', '',
        '[Architecture](v4_biological_signal_architecture.md) · '
        '[Seed sensitivity](v4_biology_seed_sensitivity_20260921.md)', '']
    for name, record in records.items():
        lines.append(f"- [{name} source]({record['source']})")
    args.output_document.write_text('\n'.join(lines) + '\n')
    atomic_json(campaign / 'biological_contribution_comparison.json', dict(records=records,
        conditional_differences=differences, matched_family_coefficients=True, exploratory_repeated_tests=True))


if __name__ == '__main__':
    main()
