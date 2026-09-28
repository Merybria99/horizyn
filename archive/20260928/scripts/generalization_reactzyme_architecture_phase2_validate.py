#!/usr/bin/env python3
"""Process-isolated validation worker for the fresh ReactZyme phase-2 study."""
import argparse
import json
from pathlib import Path
import torch
from generalization_reactzyme_architecture_phase2 import RUN,validate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    plan=json.loads((a.run.parent/'protocol.json').read_text());split=plan.get('split','reaction_smi')
    baseline=json.loads((RUN/'phase2_soft_ce_v1'/split/'composition_validation.json').read_text())['records'][0]['validation']
    grid=plan['composition_grid']
    validate(a.run,baseline,'cuda:0',grid['steps'],grid['caps'],grid['alpha'])
