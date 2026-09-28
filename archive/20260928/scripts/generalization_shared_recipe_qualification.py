#!/usr/bin/env python3
"""Check all primary benchmark targets and freeze a successful shared recipe."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from generalization_screen_replication import ROOT, sha
from generalization_full_graph import atomic_json


def qualify(campaign):
    if (campaign / 'case1_freeze.json').exists():
        frozen = json.loads((campaign / 'case1_freeze.json').read_text())
        if sha(campaign / 'qualification.json') != frozen['benchmark_qualification_sha256']:
            raise ValueError('Frozen benchmark qualification changed')
        return json.loads((campaign / 'qualification.json').read_text())
    plan = json.loads((campaign / 'protocol.json').read_text())
    targets_path = campaign.parent / 'goal_primary_comparators_20260921.json'
    targets = json.loads(targets_path.read_text())
    rows, models = [], []
    for item in plan['reactzyme']:
        if sha(item['result']) != item['result_sha256']:
            raise ValueError('ReactZyme result changed after the shared recipe was declared')
        result = json.loads(Path(item['result']).read_text())
        t = item['task']
        if sha(t['train_config']) != item['train_config_sha256'] or sha(t['checkpoint']) != item['checkpoint_sha256']:
            raise ValueError('ReactZyme checkpoint/config lineage changed')
        for direction, target in targets['reactzyme'][item['split']].items():
            value = result['summary'][direction]['all']['reactzyme_mrr']
            rows.append(dict(benchmark='ReactZyme', setting=item['split'], metric=direction,
                value=value, target=target, passes=value > target, source=item['result']))
        source = Path(t['source_campaign']) / t['arm']
        models.append(dict(name=item['split'], config=t['train_config'], checkpoint=t['checkpoint'],
            source_phase2=str(source), fusion_multiplier=plan['recipe']['inference_fusion_multiplier'],
            checkpoint_already_calibrated=False, benchmark_result=item['result']))
    run = Path(plan['enzymemap']['run_root'])
    result_path = Path(plan.get('enzymemap_result', run / 'phase2/composition_v1/test_evaluation/summary.json'))
    completion_path = Path(plan.get('enzymemap_completion', run / 'complete.json'))
    if result_path.exists() and completion_path.exists():
        completed = json.loads(completion_path.read_text())
        if sha(result_path) != completed['test_summary_sha256']:
            raise ValueError('EnzymeMap completed result changed')
        result = json.loads(result_path.read_text())
        for table, metrics in targets['enzymemap'].items():
            for metric, target in metrics.items():
                value = result['summary'][table][metric]
                rows.append(dict(benchmark='EnzymeMap', setting=table, metric=metric,
                    value=value, target=target, passes=value > target, source=str(result_path)))
        phase = run / 'phase2'
        base = json.loads((phase / 'protocol.json').read_text())
        models.append(dict(name='enzymemap', config=base['base_config'], checkpoint=base['base_checkpoint'],
            source_phase2=str(phase), fusion_multiplier=1., checkpoint_already_calibrated=True,
            calibration_receipt=str(run / 'fusion_receipt.json'), benchmark_result=str(result_path)))
    complete = len(rows) == 14
    record = dict(updated_utc=datetime.now(timezone.utc).isoformat(), all_results_available=complete,
        all_primary_targets_exceeded=complete and all(r['passes'] for r in rows),
        passed_cells=sum(r['passes'] for r in rows), required_cells=14, rows=rows,
        protocol_sha256=sha(campaign / 'protocol.json'), target_registry_sha256=sha(targets_path),
        interpretation='Exploratory benchmark point estimates, not a statistical superiority claim; TIGER ablations are secondary.',
        case1_used_for_selection=False)
    atomic_json(campaign / 'qualification.json', record)
    lines = ['# Shared recipe benchmark qualification', '',
        'All 14 primary benchmark cells exceeded.' if record['all_primary_targets_exceeded'] else
        f"{record['passed_cells']}/14 primary targets exceeded; " + ('candidate does not qualify.' if complete else 'EnzymeMap evaluation pending.'), '',
        '| Benchmark | Setting | Metric | Method | Strongest primary target | Exceeds target |',
        '| --- | --- | --- | ---: | ---: | --- |']
    for r in rows:
        lines.append(f"| {r['benchmark']} | {r['setting']} | {r['metric']} | {r['value']:.6f} | {r['target']:.6f} | {'yes' if r['passes'] else 'no'} |")
    lines += ['', 'SLEEC multiview encoder; base beta 5, seed 42, batch 512, epoch 10; phase 2: 100 positive-CE updates, '
        f"temperature 0.2, identity weight 10; residual cap {plan['recipe']['residual_cap']}, "
        f"semantic alpha {plan['recipe']['semantic_alpha']}, inference fusion multiplier 3.", '',
        'The architecture and recipe are shared; weights are independently trained on each target training split. '
        'Reaction-Sim training recovered from an optimizer checkpoint after a memory collision. '
        'Frozen encoder/SLEEC pretraining differs from the competitors. TIGER and FGW comparisons are published point estimates; '
        'only CLIPZyme screening uses a reproduced released checkpoint. Repeated tests make this exploratory.', '',
        '[Exact score sources](qualification.json) · [Frozen experiment protocol](protocol.json)', '']
    (campaign / 'comparison.md').write_text('\n'.join(lines))
    if record['all_primary_targets_exceeded'] and not (campaign / 'case1_freeze.json').exists():
        for model in models:
            phase = Path(model['source_phase2'])
            model.update(config_sha256=sha(model['config']), checkpoint_sha256=sha(model['checkpoint']),
                phase2_checkpoint=str(phase / 'training/step0100.pt'),
                phase2_checkpoint_sha256=sha(phase / 'training/step0100.pt'),
                dictionary_sha256=sha(phase / 'anchors.pt'),
                feature_manifest_sha256=sha(phase / 'features/manifest.json'))
        atomic_json(campaign / 'case1_freeze.json', dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
            benchmark_qualification_sha256=sha(campaign / 'qualification.json'), recipe=plan['recipe'], models=models,
            candidate_rows=144, unique_sequences=123, primary_deployment='reaction_smi',
            report_all_four_target_trained_models=True, case1_selection=False, ensembles=False,
            historical_case1_already_examined=True, interpretation='Retrospective literature-panel evaluation; no new measured activities.'))
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--watch', action='store_true')
    p.add_argument('--case1-after-win', action='store_true')
    a = p.parse_args()
    campaign = a.campaign.resolve()
    while True:
        result = qualify(campaign)
        if not a.watch or result['all_results_available']:
            break
        plan = json.loads((campaign / 'protocol.json').read_text())
        if (Path(plan['enzymemap']['run_root']) / 'failure.json').exists():
            raise RuntimeError('Shared recipe EnzymeMap experiment failed; inspect its failure receipt')
        time.sleep(15)
    print(json.dumps({k: result[k] for k in ('all_results_available', 'all_primary_targets_exceeded', 'passed_cells')}))
    if a.case1_after_win and result['all_primary_targets_exceeded']:
        subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/generalization_shared_recipe_case1.py'),
                        '--campaign', str(campaign)], cwd=ROOT, check=True)
        subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/generalization_winning_method_report.py'),
                        '--campaign', str(campaign)], cwd=ROOT, check=True)
