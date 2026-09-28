#!/usr/bin/env python3
"""Retrospective inference sensitivity of the exact current manuscript fits.

Freeze the grid and hashes before evaluation; reproduce the main-table score
and metrics first. No training, dictionary, biology labels, or model selection.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from generalization_reactzyme_architecture_phase2 import (
    ROOT, RUN, atomic_json, canonical_dot, checked_official_truth,
    evaluate_scores, load_head, sha256,
)
from generalization_clipzyme_f3_screen import model_from_checkpoint, export_device_lock
from generalization_multiview_calibration import components
from generalization_clipzyme_screening_evaluate import evaluate_query
from cersei_coefficient_sweep import tie_diagnostics

CROSS = RUN / 'cross_paper_retraining'
BASE = CROSS / 'shared_fusion03_beta5_b1024_v1'
CURRENT = ROOT / 'runs/cersei_horizyn_challenge_20260923/no_dictionary_validation_20260923'
TASKS = ['reaction_smi', 'enzyme_smi', 'time', 'enzymemap']
METRICS = ['bedroc85', 'bedroc20', 'ef0.05', 'ef0.1']


def read(p):
    return json.loads(Path(p).read_text())


def initialize(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise FileExistsError('Protocol already frozen')
    tuples = [(2., .1)]
    tuples += [(a, .1) for a in (0., .5, 1., 1.5, 2., 2.5, 3., 4.)]
    tuples += [(2., k) for k in (0., .05, .1, .15, .2, .25, .3)]
    tuples += [(1., 0.), (3., 0.), (3., .2)]
    arms = [dict(id=f'a{a:g}_k{k:g}', fusion_multiplier=a, kappa_inf=k)
            for a, k in dict.fromkeys(tuples)]
    tasks = {}
    for task in TASKS:
        base = BASE / task
        phase = base / ('phase2_followup/epoch15' if task == 'enzymemap' else 'phase2_followup/epoch20')
        cp = base / ('checkpoints/screen_selection/screen-epoch=14.ckpt' if task == 'enzymemap'
                     else 'training/checkpoints/screen_selection/screen-epoch=19.ckpt')
        ref = CURRENT / 'reference_shared_test' / task
        m = dict(checkpoint=str(cp), head=str(phase / 'training/step0100.pt'),
                 manifest=str(phase / 'features/manifest.json'), config=str(base / 'configs/train.yaml'),
                 reference_summary=str(ref / 'summary.json'), reference_scores=str(ref / 'scores.npy'),
                 validation=str(CURRENT / 'reference' / task / 'validation.json'), phase=str(phase))
        for key in ['checkpoint', 'head', 'manifest', 'config', 'reference_summary', 'reference_scores', 'validation']:
            m[key + '_sha256'] = sha256(Path(m[key]))
        old = read(ref / ('score_receipt.json' if task == 'enzymemap' else 'summary.json'))
        assert m['checkpoint_sha256'] == old['checkpoint_sha256']
        assert m['head_sha256'] == old['head_sha256' if task == 'enzymemap' else 'residual_head_sha256']
        tasks[task] = m
    atomic_json(out / 'protocol.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha256(Path(__file__)),
        reference=dict(fusion_multiplier=2., kappa_inf=.1, dictionary_weight=0.),
        arms=arms, tasks=tasks, seed=42, dictionary_weight=0., retraining=False,
        purpose='Retrospective test sensitivity; no selection or replacement of headline results',
        scoring=dict(reactzyme='FP64 dot rounded to FP32; stable candidate-index ties',
                     enzymemap='FP32 dot in original 32-query batches; released NumPy argsort'),
        uncertainty='Single fitted model per partition; no training-variance inference',
        reference_selection=read(CURRENT / 'reference_shared_selection.json')))
    print('Frozen', len(arms), 'arms for each of', len(tasks), 'tasks', flush=True)


@torch.inference_mode()
def run(out, task, device):
    plan = read(out / 'protocol.json'); m = plan['tasks'][task]
    assert sha256(Path(__file__)) == plan['source_sha256']
    for key in ['checkpoint', 'head', 'manifest', 'config', 'reference_summary', 'reference_scores']:
        assert sha256(Path(m[key])) == m[key + '_sha256'], key
    dest = out / task; dest.mkdir(exist_ok=True)
    phase = Path(m['phase']); cp = Path(m['checkpoint'])
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    head = load_head(Path(m['head']), m['manifest_sha256'], device)
    assert abs(head.scale - .2) < 1e-8
    if task != 'enzymemap':
        feature = phase / 'test_features'; cat = read(feature / 'catalog.json')
        assert read(feature / 'complete.json')['checkpoint_sha256'] == m['checkpoint_sha256']
        z = np.load(feature / 'f3_features.npz')
        be = torch.tensor(z['proteins'], device=device); br = torch.tensor(z['reactions'], device=device)
        ids = cat['proteins']
    else:
        catalog = CROSS / 'clipzyme_f3_catalog_v1'
        screen = BASE / task / 'screen_epoch14/proteins'
        ids, arrays = [], []
        for rank in range(4):
            stem = f'protein_rank{rank:02d}_of_04'; path = screen / (stem + '.npy')
            rec = read(screen / (stem + '.receipt.json'))
            assert rec['checkpoint_sha256'] == m['checkpoint_sha256']
            assert rec['embedding_sha256'] == sha256(path)
            ids += (screen / (stem + '.ids.txt')).read_text().splitlines()
            arrays.append(np.load(path))
        be = torch.tensor(np.concatenate(arrays), device=device); del arrays
        source = phase / 'selected_test/test_embeddings'
        assert read(source / 'reaction_receipt.json')['checkpoint_sha256'] == m['checkpoint_sha256']
        br = torch.tensor(np.load(source / 'reaction_embeddings.npy'), device=device)
        mapping = list(csv.DictReader((catalog / 'screening_candidate_map.csv').open()))
        candidates = [r['uniprot_id'] for r in mapping]; lookup = {k:i for i,k in enumerate(ids)}
        expanded = np.asarray([lookup[r['protein_id']] for r in mapping], np.int32)
        ref = CURRENT / 'reference_shared_test' / task
        assert candidates == (ref / 'candidate_ids.txt').read_text().splitlines()
        queries = (ref / 'query_ids.txt').read_text().splitlines()
    cache = dest / 'fusion_components.pt'
    if cache.exists():
        rec = read(dest / 'fusion_components.receipt.json'); assert sha256(cache) == rec['sha256']
        data = torch.load(cache, map_location=device, weights_only=False)
        assert data['checkpoint_sha256'] == m['checkpoint_sha256'] and data['ids'] == ids
        g, f, scale = data['g'], data['f'], data['scale']
    else:
        print(task, 'export components', len(ids), flush=True)
        model, config = model_from_checkpoint(Path(m['config']), cp, device)
        if task == 'enzymemap':
            config.data.protein_residue_embeds_path = str(catalog / 'features/proteins_prott5_residue.local.h5')
        g, f, scale, native, error = components(model, config, ids, device)
        export_error = float((native - be).abs().max()); assert export_error < 3e-6, export_error
        torch.save(dict(g=g.cpu(), f=f.cpu(), scale=scale, ids=ids,
                        checkpoint_sha256=m['checkpoint_sha256']), cache)
        atomic_json(dest / 'fusion_components.receipt.json', dict(sha256=sha256(cache),
            checkpoint_sha256=m['checkpoint_sha256'], reconstruction_error=error, export_error=export_error))
        del model, native; torch.cuda.empty_cache()
    assert float((F.normalize(g + scale*f, dim=-1) - be).abs().max()) < 3e-6
    del be
    if task != 'enzymemap':
        parent = read(RUN / 'phase2/models' / task / 'seed42/bundle.json'); prov = {}
        checked, edges = checked_official_truth(RUN / ('features_test_' + task), task,
                                               parent['frozen_recipe']['sha256'], prov)
        assert checked == cat
        truth = dict(reaction_index=edges[:,0], enzyme_index=edges[:,1])
        atomic_json(dest / 'truth_receipt.json', prov)
    else:
        protocol = CROSS / 'clipzyme_screening_evaluation_protocol_v2'; pr = read(protocol / 'receipt.json')
        for fn, key in [('queries.csv','queries_sha256'), ('train_uniprot_ids.txt','train_uniprot_ids_sha256')]:
            assert sha256(protocol / fn) == pr[key]
        rows = list(csv.DictReader((protocol / 'queries.csv').open()))
        assert queries == [r['reaction_id'] for r in rows]
        idx = {k:i for i,k in enumerate(candidates)}
        train = set((protocol / 'train_uniprot_ids.txt').read_text().splitlines())
        kept = np.asarray([i for i,k in enumerate(candidates) if k not in train], np.int32)
        positives = [np.asarray([idx[k] for k in json.loads(r['positive_uniprot_ids_json'])]) for r in rows]
    for arm in plan['arms']:
        path = dest / (arm['id'] + '.json')
        if path.exists():
            assert read(path)['protocol_sha256'] == sha256(out / 'protocol.json')
            continue
        started = time.monotonic(); print(task, arm['id'], 'start', flush=True)
        a, k = arm['fusion_multiplier'], arm['kappa_inf']
        b = F.normalize(g + (scale*a)*f, dim=-1)
        # Same multiply association as the saved reference implementation.
        cap = k / head.scale
        e = torch.cat([F.normalize(v + cap*head.scale*head.enzyme(v), dim=-1) for v in b.split(8192)]) if task == 'enzymemap' else F.normalize(b + cap*head.scale*head.enzyme(b), dim=-1)
        r = F.normalize(br + cap*head.scale*head.reaction(br), dim=-1)
        baseline = arm['id'] == 'a2_k0.1'; extra = {}
        old_scores = np.load(m['reference_scores'], mmap_mode='r') if baseline else None
        if task != 'enzymemap':
            scores = canonical_dot(r, e); ev = evaluate_scores(scores, truth)
            summary = ev['summary']
            ties, tiearrays = tie_diagnostics(scores, truth); extra['tie_diagnostics'] = ties
            arrays = {d+'_'+key:val for d,x in ev['per_query'].items() for key,val in x['all'].items()}
            arrays.update(tiearrays); np.savez_compressed(dest / (arm['id'] + '_per_query.npz'), **arrays)
            if baseline: extra['reference_max_score_error'] = float(np.max(np.abs(scores.cpu().numpy()-old_scores)))
            del scores, ev
        else:
            records = []; max_error = 0.
            with ThreadPoolExecutor(max_workers=24) as pool:
                for start in range(0, len(br), 32):
                    values = (r[start:start+32] @ e.T).float().cpu().numpy()[:,expanded]
                    if baseline: max_error = max(max_error, float(np.max(np.abs(values-old_scores[start:start+len(values)]))))
                    records += list(pool.map(evaluate_query, [(start+i, queries[start+i], v, positives[start+i], kept) for i,v in enumerate(values)]))
            summary = {}
            for table, nc in [('table1',len(candidates)), ('table2',len(kept))]:
                selected = [x[table] for x in records if x[table] is not None]
                summary[table] = dict(queries=len(selected), candidate_ids=nc,
                                     **{metric:float(np.mean([x[metric] for x in selected])) for metric in METRICS})
            assert summary['table1']['queries'] == 1521 and summary['table2']['queries'] == 1337
            assert len(candidates) == 261907 and len(kept) == 252113
            assert sum(x['positives_table1'] for x in records) == pr['table1_candidate_positive_labels']
            (dest / (arm['id']+'_per_query.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in records))
            if baseline: extra['reference_max_score_error'] = max_error
        if baseline:
            previous = read(m['reference_summary'])['summary']
            if task == 'enzymemap':
                previous = previous['summary']
                errors = [abs(summary[t][v]-previous[t][v]) for t in ['table1','table2'] for v in METRICS]
            else:
                errors = [abs(summary[d]['all']['reactzyme_mrr']-previous[d]['all']['reactzyme_mrr']) for d in summary]
            extra['reference_max_metric_error'] = max(errors)
            assert extra['reference_max_score_error'] < 1e-6, extra
            assert max(errors) < 1e-6, extra
        atomic_json(path, dict(task=task, arm=arm, summary=summary, learned_scale=scale, **extra,
            elapsed_seconds=time.monotonic()-started, protocol_sha256=sha256(out / 'protocol.json'),
            completed_utc=datetime.now(timezone.utc).isoformat()))
        print(task, arm['id'], 'complete', round(time.monotonic()-started, 1), flush=True)
    atomic_json(dest / 'complete.json', dict(arms=len(plan['arms']), protocol_sha256=sha256(out / 'protocol.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--initialize', action='store_true'); p.add_argument('--task', choices=TASKS)
    p.add_argument('--device', default='cuda:0'); args = p.parse_args()
    if args.initialize: initialize(args.output.resolve())
    else:
        with export_device_lock(args.device): run(args.output.resolve(), args.task, args.device)
