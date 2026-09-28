#!/usr/bin/env python3
"""Check parameter, SLEEC and training-input invariants for completed F3 fits."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from generalization_full_graph import atomic_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    a = p.parse_args(); out = a.campaign.resolve()
    plan = json.loads((out / 'protocol.json').read_text())
    cross = ROOT / 'runs/generalization_20260919_2251/cross_paper_retraining'
    old = json.loads((cross / 'shared_recipe_alpha04_cap05_v1/case1_freeze.json').read_text())
    controls = {m['name']: m for m in old['models']}
    rows = []
    torch.set_num_threads(4)
    for task in plan['tasks']:
        checkpoint = Path(task['run_root']) / 'checkpoints/screen_selection/screen-epoch=09.ckpt'
        if not checkpoint.exists() or time.time() - checkpoint.stat().st_mtime < 30:
            rows.append(dict(target=task['name'], status='pending')); continue
        reference = controls[task['name']]
        original = Path(reference['checkpoint'])
        if reference.get('calibration_receipt'):
            original = Path(json.loads(Path(reference['calibration_receipt']).read_text())['source_checkpoint'])
        before = torch.load(original, map_location='cpu', weights_only=False)['state_dict']
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        after = saved['state_dict']
        if set(before) != set(after) or any(before[k].shape != after[k].shape for k in before):
            raise ValueError(f"Architecture changed: {task['name']}")
        scorer = [k for k in before if 'sleec_scorer' in k]
        if not scorer or not all(torch.equal(before[k], after[k]) for k in scorer):
            raise ValueError('Frozen SLEEC parameters changed')
        if any('biological_readouts.' in k for k in after):
            raise ValueError('Unexpected biological model components')
        c0, c1 = (yaml.safe_load(Path(path).read_text()) for path in (reference['config'], task['config']))
        if c0['model'] != c1['model']:
            raise ValueError('Model configuration changed')
        inputs = {}
        for key in ('train_pairs_path', 'train_reactions_path', 'validation_pairs_path', 'validation_reactions_path'):
            paths = [Path(c['data'][key]) for c in (c0, c1)]
            paths = [x if x.is_absolute() else ROOT / x for x in paths]
            hashes = [sha(x) for x in paths]
            if hashes[0] != hashes[1]:
                raise ValueError(f'Target input changed: {key}')
            inputs[key] = hashes[0]
        core_training = ('learning_rate', 'weight_decay', 'precision', 'loss', 'accumulate_grad_batches')
        for key in core_training:
            if c0['training'].get(key) != c1['training'].get(key):
                raise ValueError(f'Core training setting changed: {key}')
        if c0['data']['train_batch_size'] != c1['data']['train_batch_size']:
            raise ValueError('Batch size changed')
        queries = 'model.multiview_encoder.queries'
        rows.append(dict(target=task['name'], status='verified', epoch=saved['epoch'] + 1,
            global_step=saved['global_step'], checkpoint_sha256=sha(checkpoint),
            original_checkpoint_sha256=sha(original), state_tensors=len(after),
            state_values=sum(x.numel() for x in after.values()), additional_parameter_tensors=0,
            frozen_sleec_tensor_count=len(scorer), frozen_sleec_identical=True,
            model_configuration_identical=True, core_training_settings_identical=True,
            training_configuration_differences={k: dict(reference=c0['training'].get(k), current=c1['training'].get(k))
                for k in sorted(set(c0['training']) | set(c1['training']))
                if c0['training'].get(k) != c1['training'].get(k)},
            training_validation_inputs=inputs,
            learned_query_difference_l2=float((after[queries] - before[queries]).norm()),
            biological_loss=saved['hyper_parameters'].get('biological_geometry_weight'),
            inference_annotations_required=False))
    atomic_json(out / 'architecture_and_data_audit.json', dict(rows=rows,
        created_utc=datetime.now(timezone.utc).isoformat(),
        interpretation='Matching retrieval architecture and downstream associations; additional annotation supervision is explicit.'))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
