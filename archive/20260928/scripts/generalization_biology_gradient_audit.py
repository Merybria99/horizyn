#!/usr/bin/env python3
"""Measure local loss gradients on existing phase2 parameters without updating them."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horizyn.biological_geometry import BiologicalGeometryLoss
from horizyn.generalization_residual import FrozenGeometryResidual, full_graph_contrastive_loss
from generalization_full_graph import atomic_json, sha
from generalization_clipzyme_f3_screen import export_device_lock

CROSS = ROOT / 'runs/generalization_20260919_2251/cross_paper_retraining'
PARENT = CROSS / 'v4_relative_biology_phase2_20260921_v1'
OUT = CROSS / 'v4_biology_phase2_gradient_audit_20260921_v1'


def measure(task, device):
    source = Path(task['source_phase2']) / 'features'
    checkpoint = PARENT / task['name'] / 'all_3/training/step0100.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    catalog = json.loads((source / 'catalog.json').read_text())
    with np.load(source / 'pairs.npz') as archive:
        edges = archive['train']
    tr, te = np.unique(edges[:, 0]), np.unique(edges[:, 1])
    global_ri = {key: i for i, key in enumerate(catalog['reactions'])}
    if not np.array_equal(tr, [global_ri[key] for key in catalog['train_reactions']]):
        raise ValueError('Training reaction axes disagree')
    with np.load(source / 'f3_features.npz') as archive:
        base_e = torch.as_tensor(archive['proteins'][te], device=device)
        base_r = torch.as_tensor(archive['train_reactions'], device=device)
    r = torch.as_tensor(np.searchsorted(tr, edges[:, 0]), device=device)
    e = torch.as_tensor(np.searchsorted(te, edges[:, 1]), device=device)
    payload = json.loads(Path(task['annotations']).read_text())
    if payload['feature_manifest_sha256'] != sha(source / 'manifest.json'):
        raise ValueError('Training annotation manifest differs')
    biology = BiologicalGeometryLoss(payload, catalog['train_reactions'],
        [catalog['proteins'][int(i)] for i in te], device, mode='relative', margin=.1)
    model = FrozenGeometryResidual(**saved['model_config']).to(device)
    model.load_state_dict(saved['state_dict'], strict=True)
    model.train()
    reactions, enzymes = model.encode_reactions(base_r), model.encode_enzymes(base_e)
    retrieval, _, _ = full_graph_contrastive_loss(reactions @ enzymes.T / .2, r, e)
    identity = 10 * ((1 - (reactions * F.normalize(base_r, dim=-1)).sum(-1)).mean()
                     + (1 - (enzymes * F.normalize(base_e, dim=-1)).sum(-1)).mean()) / 2
    total, pieces = biology(reactions, enzymes, {f: 3 for f in biology.families})
    # At coefficient3 the fixed family divisor3 makes each piece its applied term.
    terms = dict(retrieval=retrieval, identity_weighted=identity, **pieces)
    params = tuple(model.parameters())
    gradients = {}
    for name, loss in terms.items():
        values = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=False)
        gradients[name] = torch.cat([g.detach().flatten() for g in values])
    reference = gradients['retrieval']; reference_norm = reference.norm()
    values = {}
    for name, gradient in gradients.items():
        norm = gradient.norm()
        values[name] = dict(loss=float(terms[name].detach()), gradient_l2=float(norm),
            gradient_norm_relative_to_retrieval=float(norm / reference_norm.clamp_min(1e-20)),
            cosine_with_retrieval_gradient=float(gradient @ reference / (norm * reference_norm).clamp_min(1e-20)))
    biological_gradient = sum(gradients[f] for f in biology.families)
    record = dict(target=task['name'], checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        annotations_sha256=sha(task['annotations']), feature_manifest_sha256=sha(source / 'manifest.json'),
        training_reactions=len(tr), training_enzymes=len(te), training_edges=len(edges),
        parameters=sum(p.numel() for p in params), components=values,
        biological_loss=float(total.detach()),
        biological_gradient_l2=float(biological_gradient.norm()),
        biological_gradient_cosine_with_retrieval=float(biological_gradient @ reference /
            (biological_gradient.norm() * reference_norm).clamp_min(1e-20)))
    if not all(torch.isfinite(g).all() for g in gradients.values()):
        raise FloatingPointError('Nonfinite diagnostic gradient')
    return record


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    plan = json.loads((PARENT / 'protocol.json').read_text())
    atomic_json(OUT / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint_variant='all_3', step=100, scope='Original training endpoints only',
        purpose='Descriptive local gradients, no optimization, selection or model changes',
        script_sha256=sha(__file__)))
    records = []
    for task in plan['tasks']:
        with export_device_lock('cuda:3'):
            record = measure(task, 'cuda:3')
        records.append(record)
        torch.cuda.empty_cache()
        atomic_json(OUT / 'gradient_audit.json', dict(records=records,
            interpretation='Local pre-clipping loss gradients at the fitted step100 heads. '
            'Not Adam update magnitudes, causal feature importance, per-query semantic identities, '
            'or evidence of held-out generalization.'))
        print(json.dumps(record), flush=True)
    lines = ['# Biological supervision: local gradient diagnostic', '',
        'This evaluates the fitted phase2 weight3 heads on their original training graph. No parameter '
        'is updated. The same positive CE, identity weight10, relative margin0.1 and fixed family divisor3 '
        'are used as in training. All four target-trained models are reported.', '',
        '| Target / term | Applied loss | Gradient norm / retrieval norm | Gradient cosine with retrieval |',
        '| --- | ---: | ---: | ---: |']
    for record in records:
        for family in ('ec', 'cofactor', 'mechanism', 'identity_weighted'):
            values = record['components'][family]
            lines.append(f'| {record["target"]} / {family} | {values["loss"]:.7g} | '
                f'{values["gradient_norm_relative_to_retrieval"]:.6f} | '
                f'{values["cosine_with_retrieval_gradient"]:+.6f} |')
    lines += ['', 'A negative cosine means the two local gradients oppose each other at this fitted point. '
        'It does not establish that the biological term harms generalization: regularization can deliberately '
        'oppose the fitted retrieval objective. Equal scalar loss coefficients need not yield equal gradient '
        'magnitudes. Neither the loss value nor the gradient norm is a causal importance percentage.', '',
        'These are gradients with respect to the existing phase2 parameters before gradient clipping. '
        'Adam momentum, second-moment scaling and prior training steps are not represented. The original '
        'F3 and its learned residue queries stay frozen in this study. Benchmark/Case1 removal controls '
        'provide outcome evidence separately.', '',
        f'[Raw gradients statistics and source hashes]({OUT}/gradient_audit.json) · '
        '[Architecture and outcome interpretation](v4_biological_signal_architecture.md)', '']
    (ROOT / 'documents/v4_biology_gradient_audit_20260921.md').write_text('\n'.join(lines))
    atomic_json(OUT / 'complete.json', dict(completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
