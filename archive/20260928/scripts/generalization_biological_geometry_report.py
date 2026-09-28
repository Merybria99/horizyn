#!/usr/bin/env python3
"""Qualify and report all predeclared biology-only V4 loss ablations."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generalization_full_graph import atomic_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    a = p.parse_args(); out = a.campaign.resolve()
    plan = json.loads((out / 'protocol.json').read_text())
    mode = plan.get('biological_mode', 'attraction')
    targets_path = out.parent / 'goal_primary_comparators_20260921.json'
    targets = json.loads(targets_path.read_text())
    parent = json.loads(Path(plan['parent_freeze']).read_text())
    records = []
    for variant in plan['variants']:
        dest = out / 'methods' / variant['name']; dest.mkdir(parents=True, exist_ok=True)
        rows, models = [], []
        for task in plan['tasks']:
            name = task['name']; result_path = out / name / variant['name'] / 'test_summary.json'
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text()); summary = result['summary']
            if name == 'enzymemap':
                for table, metrics in targets['enzymemap'].items():
                    for metric, original_target in metrics.items():
                        target = max(original_target, 7.81) if table == 'table2' and metric == 'ef0.1' else original_target
                        value = summary[table][metric]
                        rows.append(dict(benchmark='EnzymeMap', setting=table, metric=metric, value=value,
                            target=target, passes=value > target, source=str(result_path), source_sha256=sha(result_path)))
            else:
                for direction, target in targets['reactzyme'][name].items():
                    value = summary[direction]['all']['reactzyme_mrr']
                    rows.append(dict(benchmark='ReactZyme', setting=name, metric=direction, value=value,
                        target=target, passes=value > target, source=str(result_path), source_sha256=sha(result_path)))
            model = copy.deepcopy(task['model'])
            model['phase2_checkpoint'] = str(out / name / variant['name'] / 'training/step0100.pt')
            model['phase2_checkpoint_sha256'] = sha(Path(model['phase2_checkpoint']))
            model['benchmark_result'] = str(result_path)
            models.append(model)
        record = dict(variant=variant, all_results_available=len(rows) == 14,
            all_primary_targets_exceeded=len(rows) == 14 and all(r['passes'] for r in rows),
            passed_cells=sum(r['passes'] for r in rows), required_cells=14, rows=rows,
            protocol_sha256=sha(out / 'protocol.json'), target_registry_sha256=sha(targets_path),
            interpretation='Exploratory point estimates. Biological annotations add supervision; architecture and downstream associations remain matched.')
        records.append(record)
        if not (dest / 'case1_freeze.json').exists():
            atomic_json(dest / 'qualification.json', record)
            if record['all_primary_targets_exceeded']:
                freeze = copy.deepcopy(parent)
                freeze.update(frozen_utc=datetime.now(timezone.utc).isoformat(),
                    benchmark_qualification_sha256=sha(dest / 'qualification.json'), models=models)
                freeze['recipe']['biological_geometry'] = variant
                atomic_json(dest / 'case1_freeze.json', freeze)
        elif sha(dest / 'qualification.json') != json.loads((dest / 'case1_freeze.json').read_text())['benchmark_qualification_sha256']:
            raise ValueError('Frozen method qualification changed')
    atomic_json(out / 'comparison.json', dict(updated_utc=datetime.now(timezone.utc).isoformat(), methods=records))
    lines = ['# V4 with biological supervision and unchanged inference architecture', '',
        'The existing SLEEC F3 encoders, phase-2 heads, semantic dictionary and scoring formula are unchanged. '
        'Only the phase-2 training objective changes. No additional learned parameters or inference inputs are introduced.', '',
        ('The relative biological regularizer compares within-category distances with distances to other annotated '
         'endpoints in the same family, with margin0.1. These references are not asserted activity negatives. '
         if mode == 'relative' else 'The annotation regularizer pulls together training endpoints sharing observed '
         'EC prefixes, cofactor descriptors or coarse mechanism descriptors. ')
        + 'Categories are balanced and uncertain annotation pairs receive lower weight. Missing labels are neutral. '
        'The three family terms retain a fixed divisor of three in all ablations.', '',
        'EC ancestry receives increasing weights 0.125/0.25/0.5/1 at levels 1/2/3/4. Cofactor presence receives confidence 0.4; '
        'it is not proof of cofactor dependence. Mechanism descriptors are derived from atom-mapped bond changes, with mappings '
        'below confidence 0.5 omitted. They are not complete catalytic mechanisms.', '',
        'The 512-dimensional learned vectors are retrieval representations regularized by biological relationships. '
        'No coordinate is assigned a biological label, and there is no positional encoding. In this phase-2 study the F3 residue '
        'attention queries remain frozen. A separate fresh-F3 study tests supervision reaching those queries.', '',
        '| Variant | Primary target cells exceeded |', '| --- | ---: |']
    for r in records:
        lines.append(f"| {r['variant']['name']} | {r['passed_cells']}/14{' (pending)' if not r['all_results_available'] else ''} |")
    lines += ['', '| Benchmark / setting / metric | Target | ' + ' | '.join(r['variant']['name'] for r in records) + ' |',
              '| --- | ---: | ' + ' | '.join('---:' for _ in records) + ' |']
    if records:
        for base in records[0]['rows']:
            key = tuple(base[k] for k in ('benchmark', 'setting', 'metric'))
            cells = []
            for r in records:
                matched = [x for x in r['rows'] if tuple(x[k] for k in ('benchmark', 'setting', 'metric')) == key]
                cells.append(f"{matched[0]['value']:.6f}" if matched else 'pending')
            lines.append('| ' + ' / '.join(key) + f" | {base['target']:.6f} | " + ' | '.join(cells) + ' |')
    lines += ['', '**Contribution interpretation.** Compare `' + plan.get('contribution_reference', 'all_0p1') + '` with each `without_*` row to remove one signal while '
        'holding the other coefficients fixed. The shuffled control permutes annotation rows within each endpoint, retaining '
        'category counts, annotation profiles and confidences. Compare against both the unchanged control and shuffled labels; '
        'a winning benchmark score alone does not demonstrate that the biological labels caused a meaningful improvement.', '',
        '| Target | Reaction EC / cofactor / mechanism coverage | Enzyme EC / cofactor / mechanism coverage |',
        '| --- | --- | --- |']
    for task in plan['tasks']:
        audit = json.loads((out / task['name'] / 'annotation_audit.json').read_text())
        def cell(endpoint, denominator):
            return ' / '.join(f"{audit['coverage'][endpoint][f]}/{denominator}" for f in ('ec', 'cofactor', 'mechanism'))
        lines.append(f"| {task['name']} | {cell('reaction', audit['train_reactions'])} | {cell('enzyme', audit['train_enzymes'])} |")
    if mode == 'relative':
        reference = plan.get('contribution_reference', 'all_10')
        reference_path = out / 'enzymemap' / reference / 'test_summary.json'
        if reference_path.exists():
            full = json.loads(reference_path.read_text())['summary']
            lines += ['', '## Conditional contribution to enzyme screening', '',
                f'Each cell is `{reference}` minus the indicated control. A positive value favors retaining '
                'the complete loss; a negative value favors that control. Removing a family also reduces '
                'the total regularization strength. Shuffling retains the weights but breaks biological assignment.', '',
                '| Comparison / setting | Δ BEDROC85 | Δ BEDROC20 | Δ EF5 | Δ EF10 |',
                '| --- | ---: | ---: | ---: | ---: |']
            shuffled = next((v['name'] for v in plan['variants'] if 'shuffle_seed' in v), None)
            for name in ('control', 'without_ec', 'without_cofactor', 'without_mechanism', shuffled):
                if name is None:
                    continue
                path = out / 'enzymemap' / name / 'test_summary.json'
                if not path.exists():
                    continue
                baseline = json.loads(path.read_text())['summary']
                for table in ('table1', 'table2'):
                    lines.append(f'| {name} / {table} | ' + ' | '.join(
                        f'{full[table][metric] - baseline[table][metric]:+.6f}'
                        for metric in ('bedroc85', 'bedroc20', 'ef0.05', 'ef0.1')) + ' |')
            lines += ['', 'These phase2-only results must be interpreted separately from fresh F3 training. '
                'They hold the original learned queries and all other F3 parameters fixed. Preserving benchmark '
                'wins does not demonstrate that the annotation penalty improves those original representations.', '']
    lines += ['', '**Boundaries.** Same target-specific training associations and frozen V4 inputs; added annotation supervision '
        'is explicitly different from the no-biology control. Exact V4 phase-2 controls reproduce the saved parameters bit for bit. '
        'These are single-seed exploratory experiments with repeated tests. ReactZyme tie sensitivity and Case1 follow-up '
        'must be considered before interpreting small improvements as generalization.', '',
        f'[Machine-readable results]({out}/comparison.json) · [Frozen study protocol]({out}/protocol.json) · '
        f'[Exact control reproduction]({out}/control_reproduction.json)', '']
    ties_path = out / 'tie_sensitivity.json'
    if ties_path.exists():
        ties = json.loads(ties_path.read_text())
        lines += ['## Numerical sensitivity', '',
            'Large changes in Enzyme-Sim/Time E→R MRR must not be interpreted directly as biological gains. '
            'The diagnostic below allows a 1e-6 band around each positive score. Its pessimistic/optimistic bounds '
            'are secondary sensitivity checks, not replacements for the fixed official-style ranking metric.', '',
            '| Target / variant | E→R MRR lower bound | Upper bound | Positive edges with near competitor |',
            '| --- | ---: | ---: | ---: |']
        for r in ties['records']:
            if r['variant'] not in plan.get('tie_report_variants', ('control', 'all_0p3')):
                continue
            b = next(b for b in r['bounds'] if b['tolerance'] == 1e-6)
            lines.append(f"| {r['target']} / {r['variant']} | {b['pessimistic_mrr']:.6f} | {b['optimistic_mrr']:.6f} | {b['positive_edges_with_near_competitor']}/{b['positive_edges']} |")
        if mode == 'attraction':
            lines += ['', 'For Time, the reported control→0.3 E→R MRR rises from about 0.7807 to 0.8154, '
            'but the 1e-6 lower bound moves only from about 0.77158 to 0.77170. Most of the headline change is '
            'therefore numerically fragile. Reaction-Sim is much less affected. The shuffled-label control also '
            'exceeds all primary targets: preserving the V4 wins alone does not establish the biological loss\'s value.', '']
    lines += ['## Frozen Case1 follow-up', '',
        'Methods shown below are qualified and frozen before their predictions. '
        'The panel was already examined during earlier development, so it remains retrospective.', '',
        '| Variant / target | Unique paper catalysts @25 /12 (123 sequences) | Paper entries @25 /15 (144 entries) | Conditional AUROC (123 sequences) |',
        '| --- | ---: | ---: | ---: |']
    case_variants = plan.get('case1_variants', ('all_0p03', 'all_0p1', 'all_0p3'))
    if not case_variants:
        case_variants = sorted(path.parent.parent.name for path in (out / 'methods').glob('*/case1/summary.json'))
    for variant in case_variants:
        case = out / 'methods' / variant / 'case1/summary.json'
        if not case.exists():
            lines.append(f'| {variant} | pending | pending | pending |'); continue
        data = json.loads(case.read_text())
        for name, r in data['methods'].items():
            if name.endswith('/selected'):
                lines.append(f"| {variant} / {name.removesuffix('/selected')} | {r['primary_papers']['recovered_at_25']} | "
                    f"{r['entry_level_144']['primary_papers']['recovered_at_25']} | {r['broad_assay_conditional_discrimination']['auc']:.6f} |")
    lines += ['', 'All 144 entries and the separate 123-sequence analysis are preserved in each evaluated method\'s '
        '`case1` directory. Literature catalyst recovery and conditional activity discrimination are different '
        'outcomes. This one retrospective panel does not establish broad or prospective wet-lab generalization.', '']
    document = ROOT / 'documents' / plan.get('report_filename', 'v4_biological_geometry_20260921.md')
    document.write_text('\n'.join(lines))
    print(json.dumps({r['variant']['name']: r['passed_cells'] for r in records}))


if __name__ == '__main__':
    main()
