#!/usr/bin/env python3
"""Finish score-sensitivity reports after benchmark/Case1 watchers complete."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from generalization_full_graph import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', action='append', required=True, type=Path)
    args = parser.parse_args()
    pending = [p.resolve() for p in args.campaign]
    while pending:
        for campaign in list(pending):
            if not (campaign / 'report_complete.json').exists():
                receipt = json.loads((campaign / 'report_execution.json').read_text())
                try:
                    os.kill(receipt['pid'], 0)
                except ProcessLookupError as error:
                    raise RuntimeError(f'Report watcher exited before completion: {campaign}') from error
                continue
            commands = [
                ['generalization_biological_geometry_ties.py', '--campaign', str(campaign),
                 '--followup-layout', '--variant', 'f3_biology', '--variant', 'f3_phase2_biology'],
                ['generalization_reactzyme_tie_chemistry.py', '--campaign', str(campaign)],
                ['generalization_biological_f3_report.py', '--campaign', str(campaign)],
            ]
            for name, *arguments in commands:
                with (campaign / (name.removesuffix('.py') + '.final.log')).open('a') as log:
                    subprocess.run([sys.executable, '-u', str(ROOT / 'scripts' / name), *arguments],
                                   cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
            atomic_json(campaign / 'diagnostics_complete.json', dict(
                completed_utc=datetime.now(timezone.utc).isoformat(),
                comparison_sha256=sha(campaign / 'comparison.json'),
                ties_sha256=sha(campaign / 'tie_sensitivity.json'),
                chemistry_sha256=sha(campaign / 'tie_chemistry_audit.json')))
            print('Finalized', campaign.name, flush=True)
            pending.remove(campaign)
        if pending:
            time.sleep(30)


if __name__ == '__main__':
    main()
