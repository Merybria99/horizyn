#!/usr/bin/env python3
"""Evaluate predeclared composition variants, with the same recipe on every benchmark."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from generalization_reactzyme_architecture_phase2 import (
    ROOT, RUN, load_features, load_head, smooth_encoder, compose, validation_data,
    evaluate_scores, canonical_dot, sha256, atomic_json, checked_official_truth)
from generalization_multiview_calibration import components, adjusted
from generalization_clipzyme_f3_screen import model_from_checkpoint, export_device_lock
from generalization_screen_replication import invoke


@torch.inference_mode()
def reactzyme(out, plan, task):
    dest = out / task['split']
    dest.mkdir()
    t = task['task']
    source = Path(t['source_campaign']) / t['arm']
    model, config = model_from_checkpoint(Path(t['train_config']), Path(t['checkpoint']), 'cuda:0')
    if sha256(t['checkpoint']) != task['checkpoint_sha256']:
        raise ValueError('Frozen F3 checkpoint changed')
    head = load_head(source / 'training/step0100.pt', sha256(source / 'features/manifest.json'), 'cuda:0')
    smooth = smooth_encoder(source, 'cuda:0')
    catalog, be, br, means, blocks, masks = load_features(source / 'features', 'cuda:0')
    with np.load(source / 'features/pairs.npz') as data:
        pairs = {k: data[k] for k in ('train', 'validation')}
    vr, ve, truth = validation_data(catalog, pairs)
    g, f, scale, reference, _ = components(model, config, [catalog['proteins'][i] for i in ve], 'cuda:0')
    if float((reference - be[ve]).abs().max()) > 3e-6:
        raise ValueError('Validation embedding parity failed')
    enzymes = adjusted(g, f, scale, 3., be[ve])
    se = smooth.encode_semantic_enzymes(means[ve], 256)
    sr = smooth.encode_semantic_reactions({k: v[vr] for k, v in blocks.items()}, {k: v[vr] for k, v in masks.items()})
    validation = []
    for variant in plan['variants']:
        recipe = dict(step=100, cap=variant['cap'], alpha=variant['alpha'])
        e = compose(enzymes, head, se, recipe, 'enzyme')
        r = compose(br[vr], head, sr, recipe, 'reaction')
        validation.append(dict(variant=variant['name'], recipe=recipe,
            summary=evaluate_scores(canonical_dot(r, e), truth)['summary']))
    atomic_json(dest / 'validation.json', dict(records=validation, test_used=False))
    atomic_json(dest / 'freeze.json', dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha256(out / 'protocol.json'), checkpoint_sha256=sha256(t['checkpoint']),
        all_variants_reported=True, selection='Predeclared fixed variants; none selected using test or validation outcomes'))
    tc, te, tr, tm, tb, tmask = load_features(Path(t['fixed_test_features']), 'cuda:0')
    receipt = json.loads((Path(t['fixed_test_features']) / 'complete.json').read_text())
    if receipt['checkpoint_sha256'] != sha256(t['checkpoint']):
        raise ValueError('Test checkpoint mismatch')
    tg, tf, test_scale, reference, _ = components(model, config, tc['proteins'], 'cuda:0')
    if float((reference - te).abs().max()) > 3e-6 or test_scale != scale:
        raise ValueError('Test embedding parity failed')
    enzymes = adjusted(tg, tf, scale, 3., te)
    se = smooth.encode_semantic_enzymes(tm, 256)
    sr = smooth.encode_semantic_reactions(tb, tmask)
    parent = json.loads((RUN / 'phase2/models' / task['split'] / 'seed42/bundle.json').read_text())
    provenance = {}
    checked, edges = checked_official_truth(RUN / f"features_test_{task['split']}", task['split'], parent['frozen_recipe']['sha256'], provenance)
    if tc != checked:
        raise ValueError('Official test axis mismatch')
    truth = dict(reaction_index=edges[:, 0], enzyme_index=edges[:, 1])
    # The unchanged alpha=.25 model must reproduce the prior actual test result.
    recipe = dict(step=100, cap=1., alpha=.25)
    control = evaluate_scores(canonical_dot(compose(tr, head, sr, recipe, 'reaction'),
        compose(enzymes, head, se, recipe, 'enzyme')), truth)['summary']
    prior = json.loads(Path(task['result']).read_text())['summary']
    if any(abs(control[d]['all']['reactzyme_mrr'] - prior[d]['all']['reactzyme_mrr']) > 1e-6 for d in control):
        raise ValueError('Existing shared-recipe test no longer reproduces')
    for variant in plan['variants']:
        run = dest / variant['name']; run.mkdir()
        recipe = dict(step=100, cap=variant['cap'], alpha=variant['alpha'])
        scores = canonical_dot(compose(tr, head, sr, recipe, 'reaction'), compose(enzymes, head, se, recipe, 'enzyme'))
        np.save(run / 'scores.npy', scores.cpu().numpy())
        atomic_json(run / 'test_summary.json', dict(summary=evaluate_scores(scores, truth)['summary'],
            variant=variant, checkpoint_sha256=sha256(t['checkpoint']), freeze_sha256=sha256(dest / 'freeze.json'),
            scores_sha256=sha256(run / 'scores.npy'), truth_provenance=provenance,
            test_used_for_selection=False, exploratory=True))
    atomic_json(dest / 'complete.json', dict(completed_utc=datetime.now(timezone.utc).isoformat()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--task', required=True)
    a = p.parse_args(); out = a.campaign.resolve(); plan = json.loads((out / 'protocol.json').read_text())
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.task != 'enzymemap':
        task = next(t for t in plan['reactzyme'] if t['split'] == a.task)
        with export_device_lock('cuda:0'):
            reactzyme(out, plan, task)
        return
    parent = Path(plan['shared_candidate'])
    run = Path(json.loads((parent / 'protocol.json').read_text())['enzymemap']['run_root'])
    while not (run / 'complete.json').exists():
        if (run / 'failure.json').exists():
            raise RuntimeError('Required EnzymeMap parent run failed')
        time.sleep(15)
    dest = out / 'enzymemap'; dest.mkdir()
    for variant in plan['variants']:
        name = 'shared_variant_' + variant['name']
        with export_device_lock('cuda:0'):
            invoke('generalization_screen_phase2_compose.py', ['--campaign', run / 'phase2', '--cross-root', out.parent,
                '--fixed-alpha', variant['alpha'], '--fixed-cap', variant['cap'], '--output-name', name],
                3, dest / (variant['name'] + '.log'))
        atomic_json(dest / (variant['name'] + '.json'), dict(variant=variant,
            test_summary=str(run / 'phase2' / name / 'test_evaluation/summary.json'),
            test_summary_sha256=sha256(run / 'phase2' / name / 'test_evaluation/summary.json'),
            validation=str(run / 'phase2' / name / 'validation.json')))
    atomic_json(dest / 'complete.json', dict(completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
