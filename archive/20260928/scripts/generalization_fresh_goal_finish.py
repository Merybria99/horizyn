#!/usr/bin/env python3
"""Qualify a validation-selected fresh study and run Case1 only after a joint win."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from generalization_screen_replication import ROOT, sha
from generalization_full_graph import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True, type=Path)
    p.add_argument('--gpu', type=int, default=3)
    a = p.parse_args()
    campaign = a.campaign.resolve()
    plan = json.loads((campaign / 'protocol.json').read_text())
    targets_path = campaign.parent / 'goal_primary_comparators_20260921.json'
    targets = json.loads(targets_path.read_text())
    rows, models = [], []
    for task in plan['tasks']:
        follow = Path(task['run_root']) / 'phase2_followup'
        while not (follow / 'complete.json').exists():
            failure = follow / 'failure.json'
            if failure.exists():
                raise RuntimeError(f"Follower failed: {failure.read_text()}")
            execution = json.loads((Path(task['run_root']) / 'phase2_follower_execution.json').read_text())
            proc = Path('/proc') / str(execution['pid']) / 'cmdline'
            try:
                command = proc.read_bytes()
            except FileNotFoundError:
                # Recheck the completion marker after process exit to avoid a race.
                if (follow / 'complete.json').exists():
                    break
                raise RuntimeError(f"Follower {execution['pid']} exited without completion")
            if b'generalization_fresh_phase2_follow.py' not in command:
                raise RuntimeError('Follower process identity changed')
            time.sleep(15)
        done = json.loads((follow / 'complete.json').read_text())
        if sha(done['test_summary']) != done['test_summary_sha256']:
            raise ValueError('Completed test summary changed')
        selection = json.loads((follow / 'selection.json').read_text())
        selected = selection['selected']
        if selection['test_used_for_selection'] or selected['test_used']:
            raise ValueError('Checkpoint selection used test data')
        if sha(selected['checkpoint']) != selected['checkpoint_sha256']:
            raise ValueError('Selected base checkpoint changed')
        if sha(task['config']) != task['config_sha256']:
            raise ValueError('Training configuration changed')
        result = json.loads(Path(done['test_summary']).read_text())['summary']
        if task['name'] == 'enzymemap':
            for table, metrics in targets['enzymemap'].items():
                for metric, threshold in metrics.items():
                    value = result[table][metric]
                    rows.append(dict(benchmark='EnzymeMap', setting=table, metric=metric,
                        value=value, target=threshold, passes=math.isfinite(value) and value > threshold,
                        source=done['test_summary'], source_sha256=done['test_summary_sha256']))
        else:
            for direction, threshold in targets['reactzyme'][task['name']].items():
                value = result[direction]['all']['reactzyme_mrr']
                rows.append(dict(benchmark='ReactZyme', setting=task['name'], metric=direction,
                    value=value, target=threshold, passes=math.isfinite(value) and value > threshold,
                    source=done['test_summary'], source_sha256=done['test_summary_sha256']))
        phase = Path(selected['run'])
        models.append(dict(name=task['name'], config=task['config'], config_sha256=task['config_sha256'],
            checkpoint=selected['checkpoint'], checkpoint_sha256=selected['checkpoint_sha256'],
            source_phase2=str(phase), fusion_multiplier=1.,
            # This flag selects the native protein exporter; no calibration was applied.
            checkpoint_already_calibrated=True, inference_calibration_applied=False,
            benchmark_result=done['test_summary'], selected_epoch=selected['epoch'],
            phase2_checkpoint=str(phase / 'training/step0100.pt'),
            phase2_checkpoint_sha256=sha(phase / 'training/step0100.pt'),
            dictionary_sha256=sha(phase / 'anchors.pt'),
            feature_manifest_sha256=sha(phase / 'features/manifest.json')))
    qualifies = len(rows) == 14 and all(r['passes'] for r in rows)
    qualification = dict(updated_utc=datetime.now(timezone.utc).isoformat(), all_results_available=True,
        all_primary_targets_exceeded=qualifies, passed_cells=sum(r['passes'] for r in rows),
        required_cells=14, rows=rows, protocol_sha256=sha(campaign / 'protocol.json'),
        target_registry_sha256=sha(targets_path), case1_used_for_selection=False,
        interpretation='Exploratory point-estimate comparison; main methods only; pretrained resources differ.')
    atomic_json(campaign / 'qualification.json', qualification)
    recipe = dict(seed=plan['seed'], batch_size=plan['batch_size'], base_beta=plan['base_beta'],
        max_epochs=plan['epochs'], checkpoint_selection='predeclared validation grid',
        initial_residual_scale=plan['initial_residual_scale'], phase2=plan['phase2'],
        residual_cap=1., semantic_alpha=.25, inference_fusion_multiplier=1.)
    lines = [f'### Fresh stronger-fusion study: {campaign.name}', '',
        f"Completed {qualification['updated_utc']}. **{qualification['passed_cells']}/14** primary benchmark targets exceeded. "
        + ('Joint winner.' if qualifies else 'Does not qualify as a joint winner.'), '',
        'Fresh target-specific SLEEC residue-view F3 fits use initial fused contribution 0.3, beta 5, batch 1024 and seed 42. '
        'Checkpoints at 5/10/15/20 epochs receive identical positive-CE phase 2 (100 updates, temperature 0.2, identity weight 10), '
        'semantic alpha 0.25 and residual cap 1. No inference fusion adjustment is applied. '
        'Checkpoint selection uses target validation only; EnzymeMap uses full-library BEDROC85 and reports all BEDROC/EF cells.', '',
        'Selected epochs: ' + ', '.join(f"{m['name']}={m['selected_epoch']}" for m in models) + '.', '',
        '| Benchmark | Setting | Metric | Method | Strongest primary target | Pass |',
        '| --- | --- | --- | ---: | ---: | --- |']
    for row in rows:
        lines.append(f"| {row['benchmark']} | {row['setting']} | {row['metric']} | {row['value']:.6f} | "
            f"{row['target']:.6f} | {'yes' if row['passes'] else 'no'} |")
    lines += ['', f"[Protocol]({campaign / 'protocol.json'}) · [Exact source records]({campaign / 'qualification.json'}).", '']
    if plan.get('fixed_epoch'):
        lines[4] = (f"The fixed epoch-{plan['fixed_epoch']} snapshots of four fresh target-specific F3 fits use initial "
            f"fused contribution {plan['initial_residual_scale']}, beta {plan['base_beta']}, batch {plan['batch_size']} "
            f"and seed {plan['seed']}. Phase 2 uses 100 positive-CE updates, temperature 0.2, identity weight 10, "
            'semantic alpha 0.25 and residual cap 1. All four targets use the same fixed epoch; no checkpoint or multiplier selection. '
            'This intermediate test was declared after the epoch-10 validation and EnzymeMap epoch-15 test had been inspected; '
            'it is exploratory and does not alter the separate predeclared 5/10/15/20-epoch validation selector.')
    text = '\n'.join(lines)
    (campaign / 'comparison.md').write_text(text)
    with open('/tmp/enzymediscovery_findings_append.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        findings = ROOT / 'findings.md'
        if lines[0] not in findings.read_text():
            with findings.open('a') as stream:
                stream.write('\n' + text)
    if qualifies:
        atomic_json(campaign / 'case1_freeze.json', dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
            benchmark_qualification_sha256=sha(campaign / 'qualification.json'), recipe=recipe, models=models,
            candidate_rows=144, unique_sequences=123, primary_deployment='reaction_smi',
            report_all_four_target_trained_models=True, case1_selection=False, ensembles=False,
            historical_case1_already_examined=True, interpretation='Retrospective literature panel; no new activity measurements.'))
        for script in ('generalization_shared_recipe_case1.py', 'generalization_winning_method_report.py'):
            subprocess.run([sys.executable, '-u', str(ROOT / 'scripts' / script), '--campaign', str(campaign)],
                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu)), check=True)
    atomic_json(campaign / 'goal_followup_complete.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        qualifies=qualifies, passed_cells=qualification['passed_cells'], case1_completed=qualifies,
        qualification_sha256=sha(campaign / 'qualification.json')))
    print(json.dumps(dict(qualifies=qualifies, passed_cells=qualification['passed_cells'])), flush=True)


if __name__ == '__main__':
    main()
