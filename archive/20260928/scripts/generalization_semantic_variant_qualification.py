#!/usr/bin/env python3
"""Qualify every declared semantic variant and evaluate Case1 for every winner."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from generalization_screen_replication import ROOT, sha
from generalization_full_graph import atomic_json
from generalization_shared_recipe_qualification import qualify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    a = p.parse_args(); out = a.campaign.resolve()
    plan = json.loads((out / 'protocol.json').read_text())
    parent = Path(plan['shared_candidate'])
    baseline = json.loads((parent / 'protocol.json').read_text())
    for variant in plan['variants']:
        ready = [out / split / variant['name'] / 'test_summary.json' for split in ('reaction_smi', 'enzyme_smi', 'time')]
        ready.append(out / 'enzymemap' / (variant['name'] + '.json'))
        while not all(path.exists() for path in ready):
            for task in ('reaction_smi', 'enzyme_smi', 'time', 'enzymemap'):
                execution = json.loads((out / (task + '_execution.json')).read_text())
                if not Path(f"/proc/{execution['pid']}").exists() and not (out / task / 'complete.json').exists():
                    raise RuntimeError(f'Variant evaluation failed: {task}; inspect its log')
            time.sleep(15)
        dest = out.parent / ('shared_recipe_' + variant['name'] + '_v1')
        dest.mkdir(exist_ok=False)
        derived = deepcopy(baseline)
        derived['created_utc'] = datetime.now(timezone.utc).isoformat()
        derived['recipe'].update(semantic_alpha=variant['alpha'], residual_cap=variant['cap'])
        derived['parent_variant_protocol'] = dict(path=str(out / 'protocol.json'), sha256=sha(out / 'protocol.json'))
        derived['discovery_context'] = 'One of three fixed variants declared together before their new test evaluations; all variants are retained, and qualifying variants receive Case1 evaluation.'
        for item in derived['reactzyme']:
            path = out / item['split'] / variant['name'] / 'test_summary.json'
            item.update(result=str(path), result_sha256=sha(path))
        marker = out / 'enzymemap' / (variant['name'] + '.json')
        evaluation = json.loads(marker.read_text())
        # Older worker processes may have loaded the script before the checksum
        # field was added; preserve their marker and create a separate receipt.
        if 'test_summary_sha256' not in evaluation:
            evaluation['test_summary_sha256'] = sha(evaluation['test_summary'])
            marker = dest / 'enzymemap_evaluation_receipt.json'
            atomic_json(marker, evaluation)
        derived.update(enzymemap_result=evaluation['test_summary'], enzymemap_completion=str(marker))
        atomic_json(dest / 'protocol.json', derived)
        result = qualify(dest)
        print(json.dumps(dict(variant=variant['name'], passed=result['passed_cells'], wins=result['all_primary_targets_exceeded'])), flush=True)
        if result['all_primary_targets_exceeded']:
            subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/generalization_shared_recipe_case1.py'), '--campaign', str(dest)], cwd=ROOT, check=True)
            subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/generalization_winning_method_report.py'), '--campaign', str(dest)], cwd=ROOT, check=True)
    atomic_json(out / 'qualification_complete.json', dict(completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
