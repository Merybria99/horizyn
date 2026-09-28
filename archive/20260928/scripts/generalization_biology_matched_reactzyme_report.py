#!/usr/bin/env python3
"""Compare fresh ReactZyme biological models with matching validation-cadence controls."""
import argparse
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
OUT = CROSS / 'v4_matched_reactzyme_control_20260921_v1'
MODELS = {
    'attraction_0.1': CROSS / 'v4_biological_f3_20260921_v1',
    'relative_0.1': CROSS / 'v4_relative_biology_f3_20260921_v1',
    'relative_1': CROSS / 'v4_relative_biology_f3_weight1_20260921_v1',
}
TARGETS = ('reaction_smi', 'enzyme_smi', 'time')
DIRECTIONS = ('reaction_to_enzyme', 'enzyme_to_reaction')


def report():
    records = []
    lines = ['# Fresh ReactZyme controls with matched validation cadence', '',
        'The unannotated F3 models use the same seed42, ten epochs, batch512, frozen SLEEC and fast '
        'validation cadence as the fresh biological models. Original V4 used an additional retrieval '
        'validation iterator, which can alter random-number consumption. These fresh controls remove '
        'that known protocol confound from the biological-loss comparison. They do not isolate how much '
        'of the difference from original V4 is caused by validation cadence alone. It remains one seed '
        'and exploratory repeated test evaluation.', '',
        'The primary comparison below uses the original phase2 objective for every model. Consequently '
        'the biological contrast is confined to F3 training. Existing inference fusion, dictionary and '
        'score coefficients are unchanged. No test result chooses a checkpoint or seed.', '',
        '| Split / direction | Fresh unannotated | Attraction 0.1 | Relative 0.1 | Relative 1 |',
        '| --- | ---: | ---: | ---: | ---: |']
    completed_controls = 0
    for target in TARGETS:
        suffix = f'followup/{target}/{target}/f3_biology/test_summary.json'
        control_path = OUT / suffix
        if not control_path.exists():
            for direction in DIRECTIONS:
                lines.append(f'| {target} / {direction} | pending | pending | pending | pending |')
            continue
        completed_controls += 1
        baseline = json.loads(control_path.read_text())['summary']
        for direction in DIRECTIONS:
            value = baseline[direction]['all']['reactzyme_mrr']
            cells = [f'{value:.6f}']
            for name, campaign in MODELS.items():
                path = campaign / suffix
                data = json.loads(path.read_text())['summary']
                result = data[direction]['all']['reactzyme_mrr']
                delta = result - value
                records.append(dict(target=target, direction=direction, method=name,
                    baseline=value, value=result, delta=delta, source=str(path), source_sha256=sha(path),
                    control=str(control_path), control_sha256=sha(control_path)))
                cells.append(f'{result:.6f} ({delta:+.6f})')
            lines.append(f'| {target} / {direction} | ' + ' | '.join(cells) + ' |')
    secondary = []
    lines += ['', 'The parentheses show biological minus matched unannotated MRR. Enzyme-Sim and Time '
        'E→R are sensitive to near-tied scores for distinct reaction IDs with chemically equivalent '
        'released inputs. Their raw MRR changes cannot automatically be interpreted as improved chemistry.', '',
        '## Phase2-only secondary control', '',
        'These models share the fresh unannotated F3 weights above and add relative biological weight0.1 '
        'only to the existing phase2 objective. They are reported separately from the primary causal contrast.', '',
        '| Split | R→E | E→R |', '| --- | ---: | ---: |']
    for target in TARGETS:
        path = OUT / f'followup/{target}/{target}/f3_phase2_biology/test_summary.json'
        if not path.exists():
            lines.append(f'| {target} | pending | pending |')
            continue
        data = json.loads(path.read_text())['summary']
        values = {d: data[d]['all']['reactzyme_mrr'] for d in DIRECTIONS}
        secondary.append(dict(target=target, metrics=values, source=str(path), source_sha256=sha(path)))
        lines.append(f'| {target} | ' + ' | '.join(f'{values[d]:.6f}' for d in DIRECTIONS) + ' |')
    complete = completed_controls == 3 and len(secondary) == 3
    lines += ['', 'The labels `f3_biology` in the raw control directories refer to a pipeline slot: '
        'its configured biological weight is zero. It must not be described as biologically supervised F3. '
        'No Case1 comparison is inferred from these benchmark scores.', '',
        '[Architecture](v4_biological_signal_architecture.md) · '
        '[Fresh EnzymeMap control and annotation specificity](v4_biology_additional_controls_20260921.md)', '',
        '[Direct Case1 evaluations of qualified controls](v4_matched_control_case1_20260921.md)', '',
        f'[Source hashes and paired differences]({OUT}/comparison.json)', '']
    atomic_json(OUT / 'comparison.json', dict(records=records, secondary=secondary, completed=complete,
        updated_utc=datetime.now(timezone.utc).isoformat(), exploratory_repeated_tests=True))
    (ROOT / 'documents/v4_matched_reactzyme_controls_20260921.md').write_text('\n'.join(lines))
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        complete = report()
        if complete:
            subprocess.run([sys.executable, 'scripts/generalization_biological_f3_audit.py',
                '--campaign', str(OUT)], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            if not (OUT / 'report_complete.json').exists():
                with open('/tmp/enzymediscovery_findings_append.lock', 'a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    with (ROOT / 'findings.md').open('a') as stream:
                        stream.write('\n\n## Matched fresh ReactZyme controls completed\n\n'
                            'All three unannotated controls now have fixed-epoch test results and architecture/data '
                            'audits. The primary comparison uses unannotated phase2 throughout, isolating the F3 '
                            'loss change under the same validation cadence. Full absolute metrics, signed differences '
                            'and the separate phase2-only controls are in '
                            '[the matched-control report](documents/v4_matched_reactzyme_controls_20260921.md). '
                            'These controls are exploratory and single-seed; no Case1 improvement is inferred.\n')
                atomic_json(OUT / 'report_complete.json', dict(completed_utc=datetime.now(timezone.utc).isoformat()))
        if complete or not args.watch:
            return
        for path in OUT.glob('*/followup_state.json'):
            if json.loads(path.read_text())['stage'] == 'failed':
                raise RuntimeError(f'Follow-up failed: {path}')
        time.sleep(60)


if __name__ == '__main__':
    main()
