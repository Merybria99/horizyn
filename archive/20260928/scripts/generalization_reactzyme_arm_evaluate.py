#!/usr/bin/env python3
"""Report a predeclared single architecture without waiting for other arms."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import torch

from generalization_reactzyme_architecture_phase2 import (
    RUN, atomic_json, sha256, robust_value, evaluate_selected)
from generalization_reactzyme_support_phase2 import test_selected


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();out=a.campaign.resolve();plan=json.loads((out/'protocol.json').read_text())
    task,=plan['tasks'];split=plan.get('split','reaction_smi');run=out/task['arm'];run.mkdir(exist_ok=True)
    source=Path(plan['source_campaign'])/task['arm']
    atomic_json(run/'state.json',dict(stage='waiting_for_source_validation'))
    while not (source/'validation_complete.json').exists():time.sleep(10)
    result=json.loads((source/'validation_complete.json').read_text())['selected']
    baseline=json.loads((RUN/'phase2_soft_ce_v1'/split/'composition_validation.json').read_text())['records'][0]['validation']
    selected=result if result is not None and result['value']>robust_value(baseline) else None
    receipt=dict(parent_retained=selected is None,selected=selected,task=task if selected else None,
        value=selected['value'] if selected else robust_value(baseline),baseline=baseline,
        source_validation_sha256=sha256(source/'validation_complete.json'),
        source_protocol_sha256=sha256(Path(plan['source_campaign'])/'protocol.json'),
        protocol_sha256=sha256(out/'protocol.json'),selected_utc=datetime.now(timezone.utc).isoformat(),
        test_used_for_selection=False,scope='Predeclared single architecture; not the winner across the original multi-arm campaign')
    atomic_json(out/'selection.json',receipt)
    if selected:
        for name in ('features','training','anchors.pt'):(run/name).symlink_to(source/name)
        if plan['support_gated']:(run/'support_fit.json').symlink_to(source/'support_fit.json')
        if json.loads((run/'features/complete.json').read_text())['checkpoint_sha256']!=sha256(task['checkpoint']):
            raise ValueError('Source validation and checkpoint differ')
        atomic_json(run/'state.json',dict(stage='selected_test'))
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
        evaluator=test_selected if plan['support_gated'] else evaluate_selected
        evaluator(out,task,selected,f"cuda:{task['gpu']}",split)
    atomic_json(run/'state.json',dict(stage='complete'))
    atomic_json(out/'complete.json',dict(parent_retained=selected is None,completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
