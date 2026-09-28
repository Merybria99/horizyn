#!/usr/bin/env python3
"""Qualify matched unannotated controls across both benchmarks before Case1."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from generalization_full_graph import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
CROSS = ROOT / 'runs/generalization_20260919_2251/cross_paper_retraining'
OUT = CROSS / 'v4_matched_reactzyme_control_20260921_v1'
ENZYME = CROSS / 'v4_relative_biology_additional_controls_20260921_v1/fresh_control_seed42'
NAMES = ('reaction_smi', 'enzyme_smi', 'time', 'enzymemap')


def main():
    while not (OUT / 'report_complete.json').exists():
        for path in OUT.glob('*/followup_state.json'):
            if json.loads(path.read_text())['stage'] == 'failed':
                raise RuntimeError(f'Control follow-up failed: {path}')
        time.sleep(60)
    verified = []
    for campaign in (OUT, ENZYME):
        audit = json.loads((campaign / 'architecture_and_data_audit.json').read_text())
        verified.extend(r['target'] for r in audit['rows'] if r['status'] == 'verified')
    if set(verified) != set(NAMES):
        raise ValueError('Matched controls require all four architecture/data audits')
    targets_path = CROSS / 'goal_primary_comparators_20260921.json'
    targets = json.loads(targets_path.read_text())
    parent = json.loads((CROSS / 'shared_recipe_alpha04_cap05_v1/case1_freeze.json').read_text())
    records = []
    for variant in ('f3_biology', 'f3_phase2_biology'):
        rows, models = [], []
        for name in NAMES:
            campaign = ENZYME if name == 'enzymemap' else OUT
            evaluation = campaign / 'followup' / name
            path = evaluation / name / variant / 'test_summary.json'
            result = json.loads(path.read_text())['summary']
            task, = json.loads((evaluation / 'protocol.json').read_text())['tasks']
            comparisons = ([(table, metric, result[table][metric], max(value, 7.81)
                if table == 'table2' and metric == 'ef0.1' else value)
                for table, metrics in targets['enzymemap'].items() for metric, value in metrics.items()]
                if name == 'enzymemap' else [(name, metric, result[metric]['all']['reactzyme_mrr'], value)
                for metric, value in targets['reactzyme'][name].items()])
            for setting, metric, value, target in comparisons:
                rows.append(dict(benchmark='EnzymeMap' if name == 'enzymemap' else 'ReactZyme',
                    setting=setting, metric=metric, value=value, target=target, passes=value > target,
                    source=str(path), source_sha256=sha(path)))
            model = copy.deepcopy(task['model'])
            model['phase2_checkpoint'] = str(evaluation / name / variant / 'training/step0100.pt')
            model['phase2_checkpoint_sha256'] = sha(model['phase2_checkpoint'])
            model['benchmark_result'] = str(path)
            source = Path(task['source_phase2'])
            model['dictionary_sha256'] = sha(source / 'anchors.pt')
            model['feature_manifest_sha256'] = sha(source / 'features/manifest.json')
            models.append(model)
        method = OUT / 'methods' / variant; method.mkdir(parents=True, exist_ok=True)
        record = dict(variant=variant, required_cells=14, available_cells=len(rows),
            passed_cells=sum(r['passes'] for r in rows),
            all_primary_targets_exceeded=len(rows) == 14 and all(r['passes'] for r in rows), rows=rows,
            phase1_biological_weight=0, phase2_biological_weight=.1 if variant == 'f3_phase2_biology' else 0,
            architecture_unchanged=True, target_registry_sha256=sha(targets_path),
            interpretation='Matched control, not a newly selected biological method. Exploratory repeated tests.')
        records.append(record)
        freeze_path = method / 'case1_freeze.json'
        if not freeze_path.exists():
            atomic_json(method / 'qualification.json', record)
            if record['all_primary_targets_exceeded']:
                freeze = copy.deepcopy(parent)
                freeze.update(frozen_utc=datetime.now(timezone.utc).isoformat(), models=models,
                    benchmark_qualification_sha256=sha(method / 'qualification.json'))
                freeze['recipe']['biological_geometry'] = dict(phase1=0, mode='relative', margin=.1,
                    phase2=record['phase2_biological_weight'], additional_learned_parameters=0)
                atomic_json(freeze_path, freeze)
        elif sha(method / 'qualification.json') != json.loads(freeze_path.read_text())['benchmark_qualification_sha256']:
            raise ValueError('Frozen control qualification changed')
        if record['all_primary_targets_exceeded'] and not (method / 'case1/complete.json').exists():
            with (method / 'case1.log').open('a') as log:
                subprocess.run([sys.executable, '-u', 'scripts/generalization_shared_recipe_case1.py',
                    '--campaign', str(method), '--device', 'cuda:3'], cwd=ROOT, check=True,
                    stdout=log, stderr=subprocess.STDOUT)
    atomic_json(OUT / 'case1_control_comparison.json', dict(records=records))
    lines = ['# Case1 with matched fresh unannotated F3 controls', '',
        'These controls use the fast validation cadence of the fresh biological runs. Every F3 biological '
        'coefficient is zero. The secondary variant applies relative weight0.1 only in the existing phase2. '
        'Each must exceed all14 benchmark comparator cells before this retrospective Case1 follow-up. '
        'The same original training associations, fixed epochs and V4 inference settings are retained.', '',
        '| Control / target | Unique paper catalysts @25 /12 (123 sequences) | Paper entries @25 /15 (144 entries) | Conditional AUROC |',
        '| --- | ---: | ---: | ---: |']
    for record in records:
        variant = record['variant']; path = OUT / 'methods' / variant / 'case1/summary.json'
        if not path.exists():
            lines.append(f'| {variant} | not qualified: {record["passed_cells"]}/14 | — | — |')
            continue
        for name, r in json.loads(path.read_text())['methods'].items():
            if name.endswith('/selected'):
                lines.append(f'| {variant} / {name[:-9]} | {r["primary_papers"]["recovered_at_25"]} | '
                    f'{r["entry_level_144"]["primary_papers"]["recovered_at_25"]} | '
                    f'{r["broad_assay_conditional_discrimination"]["auc"]:.6f} |')
    lines += ['', 'These results distinguish the fresh-training trajectory from biological-loss changes. '
        'Case1 contains related constructs and heterogeneous literature assays; this is not prospective validation.', '',
        '[Matched benchmark differences](v4_matched_reactzyme_controls_20260921.md) · '
        f'[All qualification rows and source hashes]({OUT}/case1_control_comparison.json)', '']
    (ROOT / 'documents/v4_matched_control_case1_20260921.md').write_text('\n'.join(lines))
    atomic_json(OUT / 'case1_control_complete.json', dict(completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
