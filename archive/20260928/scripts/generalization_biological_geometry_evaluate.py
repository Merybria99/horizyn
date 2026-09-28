#!/usr/bin/env python3
"""Evaluate loss-only biological variants with the exact V4 score and pools."""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from generalization_full_graph import atomic_json, sha, validation_data
from generalization_reactzyme_architecture_phase2 import (
    load_features, smooth_encoder, load_head, compose, canonical_dot, evaluate_scores, RUN, checked_official_truth)
from generalization_multiview_calibration import components, adjusted
from generalization_clipzyme_f3_screen import model_from_checkpoint
from generalization_clipzyme_phase2_screen import library, queries
from generalization_clipzyme_f3_validation import read_csv
from generalization_clipzyme_screening_protocol import identifier
from generalization_clipzyme_screening_evaluate import evaluate_query

CROSS = RUN / 'cross_paper_retraining'
METRICS = ('bedroc85', 'bedroc20', 'ef0.05', 'ef0.1')


def dump_cache(path, payload):
    temporary = path.with_suffix('.partial.pt')
    torch.save(payload, temporary); temporary.replace(path)
    atomic_json(path.with_suffix('.receipt.json'), dict(sha256=sha(path), created_utc=datetime.now(timezone.utc).isoformat()))


@torch.inference_mode()
def reactzyme_cache(task, scope, dest, device):
    source = Path(task['source_phase2']); m = task['model']
    if scope == 'validation':
        feature_path = source / 'features'
    elif task.get('test_features'):
        feature_path = Path(task['test_features'])
    else:
        old_plan = json.loads((CROSS / 'shared_semantic_strength_variants_v1/protocol.json').read_text())
        t = next(x['task'] for x in old_plan['reactzyme'] if x['split'] == task['name'])
        feature_path = Path(t['fixed_test_features'])
    catalog, be, br, means, blocks, masks = load_features(feature_path, device)
    if scope == 'validation':
        with np.load(feature_path / 'pairs.npz') as z:
            pairs = {k: z[k] for k in ('train', 'validation')}
        vr, ve, truth = validation_data(catalog, pairs)
        keys = [catalog['proteins'][i] for i in ve]
        be, br, means = be[ve], br[vr], means[ve]
        blocks = {k: v[vr] for k, v in blocks.items()}; masks = {k: v[vr] for k, v in masks.items()}
    else:
        receipt = json.loads((feature_path / 'complete.json').read_text())
        if receipt['checkpoint_sha256'] != m['checkpoint_sha256']:
            raise ValueError('Test F3 lineage mismatch')
        parent = json.loads((RUN / 'phase2/models' / task['name'] / 'seed42/bundle.json').read_text())
        provenance = {}
        checked, edges = checked_official_truth(RUN / f"features_test_{task['name']}", task['name'], parent['frozen_recipe']['sha256'], provenance)
        if checked != catalog:
            raise ValueError('Official ReactZyme axes differ')
        truth = dict(reaction_index=edges[:, 0], enzyme_index=edges[:, 1]); keys = catalog['proteins']
    model_config = m.get('test_config', m['config']) if scope == 'test' else m['config']
    model, config = model_from_checkpoint(Path(model_config), Path(m['checkpoint']), device)
    if sha(Path(m['checkpoint'])) != m['checkpoint_sha256']:
        raise ValueError('Base checkpoint changed')
    g, f, scale, reference, reconstruction_error = components(model, config, keys, device)
    if float((reference - be).abs().max()) > 3e-6:
        raise ValueError('Base representation parity failed')
    be = adjusted(g, f, scale, m['fusion_multiplier'], be)
    smooth = smooth_encoder(source, device)
    se = smooth.encode_semantic_enzymes(means, 256)
    sr = smooth.encode_semantic_reactions(blocks, masks)
    payload = dict(base_e=be.cpu(), base_r=br.cpu(), semantic_e=se.cpu(), semantic_r=sr.cpu(), truth=truth,
        checkpoint_sha256=m['checkpoint_sha256'], scope=scope,
        feature_manifest_sha256=sha(source / 'features/manifest.json'), dictionary_sha256=sha(source / 'anchors.pt'))
    dump_cache(dest, payload)
    return payload


@torch.inference_mode()
def enzyme_cache(task, scope, dest, device):
    source = Path(task['source_phase2']); parent = json.loads((source / 'protocol.json').read_text())
    screen_root = Path(parent['validation_summary']).parent.parent
    screen = source / 'selected_test/test_embeddings'
    if not screen.exists():
        screen = screen_root / 'requested_test/test_embeddings'
    catalog_path = CROSS / 'clipzyme_f3_catalog_v1'
    smooth = smooth_encoder(source, device)
    libcache = dest.parent / 'screening_library.pt'
    if not libcache.exists():
        for rank in range(4):
            stem = f'protein_rank{rank:02d}_of_04'
            receipt = json.loads((screen / (stem + '.receipt.json')).read_text())
            if (receipt['checkpoint_sha256'] != parent['base_checkpoint_sha256'] or
                    receipt['embedding_sha256'] != sha(screen / (stem + '.npy')) or
                    receipt['ids_sha256'] != sha(screen / (stem + '.ids.txt'))):
                raise ValueError('Screening library lineage changed')
        candidate_ids, protein_ids, expanded, base_e, means = library(catalog_path, screen,
            CROSS / 'clipzyme_phase2_enzymemap_v1/screen_protein_mean.h5')
        pointers = [torch.zeros(1, dtype=torch.int64, device=device)]
        columns, values, total = [], [], 0
        for start in range(0, len(means), 512):
            semantic = smooth.encode_semantic_enzymes(torch.as_tensor(means[start:start+512], device=device), 512)
            sparse = semantic.to_sparse_csr()
            pointers.append(sparse.crow_indices()[1:] + total)
            columns.append(sparse.col_indices()); values.append(sparse.values()); total += sparse.values().numel()
        semantic = torch.sparse_csr_tensor(torch.cat(pointers), torch.cat(columns), torch.cat(values),
            size=(len(means), len(smooth.train_reactions)), device=device).cpu()
        dump_cache(libcache, dict(base_e=torch.from_numpy(base_e), semantic_e=semantic, expanded=expanded,
            candidate_ids=candidate_ids, protein_ids=protein_ids, checkpoint_sha256=parent['base_checkpoint_sha256']))
    feature_catalog = json.loads((source / 'features/catalog.json').read_text())
    args = SimpleNamespace(scope=scope, features=source / 'features', screen=screen,
        protocol=CROSS / 'clipzyme_screening_evaluation_protocol_v2', catalog=catalog_path)
    ids, br, blocks, masks = queries(args, feature_catalog)
    if scope == 'validation':
        directory = screen_root / 'validation_embeddings'
        index = {k: i for i, k in enumerate((directory / 'query_ids.txt').read_text().splitlines())}
        br = np.load(directory / 'reaction_embeddings.npy')[[index[k] for k in ids]]
    semantic = smooth.encode_semantic_reactions({k: torch.as_tensor(v, device=device) for k, v in blocks.items()},
        {k: torch.as_tensor(v, device=device) for k, v in masks.items()})
    payload = dict(base_r=torch.from_numpy(br), semantic_r=semantic.cpu(), ids=ids, scope=scope,
        feature_manifest_sha256=sha(source / 'features/manifest.json'), dictionary_sha256=sha(source / 'anchors.pt'))
    dump_cache(dest, payload)
    return payload


def refined(head, base, endpoint):
    return torch.cat([F.normalize(b + .5 * head.scale * getattr(head, endpoint)(b), dim=-1) for b in base.split(8192)])


@torch.inference_mode()
def evaluate(args):
    out = args.campaign.resolve(); plan = json.loads((out / 'protocol.json').read_text())
    task = next(t for t in plan['tasks'] if t['name'] == args.task); dest = out / args.task
    source = Path(task['source_phase2']); device = args.device
    cache_path = dest / (args.scope + '_cache.pt')
    if cache_path.exists():
        payload = torch.load(cache_path, map_location='cpu', weights_only=False)
        if (payload['feature_manifest_sha256'] != sha(source / 'features/manifest.json') or
                payload['dictionary_sha256'] != sha(source / 'anchors.pt')):
            raise ValueError('Cached evaluation features belong to another training run')
        if args.task != 'enzymemap' and payload['checkpoint_sha256'] != task['model']['checkpoint_sha256']:
            raise ValueError('Cached reaction/protein vectors belong to another base model')
    elif args.task == 'enzymemap':
        payload = enzyme_cache(task, args.scope, cache_path, device)
    else:
        payload = reactzyme_cache(task, args.scope, cache_path, device)
    if args.cache_only:
        return
    variants = [v for v in plan['variants'] if not args.variant or v['name'] == args.variant]
    recipe = dict(cap=.5, alpha=.4, step=100)
    br = payload['base_r'].to(device); sr = payload['semantic_r'].to(device)
    if args.task == 'enzymemap':
        lib = torch.load(dest / 'screening_library.pt', map_location='cpu', weights_only=False)
        be, se = lib['base_e'].to(device), lib['semantic_e'].to(device)
        candidate_ids, expanded = lib['candidate_ids'], lib['expanded']
        index = {key: i for i, key in enumerate(candidate_ids)}
        train_rows = read_csv(CROSS / 'clipzyme_manifests_v2/train_associations.csv')
        seen_r = {r['reaction'] for r in train_rows}; seen_e = {r['protein_id'] for r in train_rows}
        kept = np.asarray([i for i, k in enumerate(candidate_ids) if k not in seen_e], dtype=np.int32)
        if args.scope == 'validation':
            truth = defaultdict(set)
            for row in read_csv(CROSS / 'clipzyme_manifests_v2/dev_associations.csv'):
                if row['reaction'] not in seen_r:
                    truth[identifier(row['reaction'])].add(row['protein_id'])
            positives = [np.asarray([index[k] for k in sorted(truth.get(key, set())) if k in index], dtype=np.int64)
                         for key in payload['ids']]
    else:
        be, se = payload['base_e'].to(device), payload['semantic_e'].to(device)
    for variant in variants:
        v = variant['name']; run = dest / v; result_path = run / (args.scope + '_summary.json')
        if result_path.exists():
            continue
        head_path = run / 'training/step0100.pt'
        if not head_path.exists():
            raise FileNotFoundError(head_path)
        head = load_head(head_path, sha(source / 'features/manifest.json'), device)
        freeze = dict(variant=variant, head_sha256=sha(head_path), recipe=recipe,
            protocol_sha256=sha(out / 'protocol.json'), created_utc=datetime.now(timezone.utc).isoformat(),
            test_used_for_selection=False, exploratory_repeated_test=True)
        atomic_json(run / (args.scope + '_freeze.json'), freeze)
        if args.task != 'enzymemap':
            scores = canonical_dot(compose(br, head, sr, recipe, 'reaction'), compose(be, head, se, recipe, 'enzyme'))
            evaluation = evaluate_scores(scores, payload['truth'])
            if args.scope == 'test':
                np.save(run / 'test_scores.npy', scores.cpu().numpy())
            result = dict(summary=evaluation['summary'], freeze=freeze)
            np.savez(run / (args.scope + '_ranks.npz'), **{
                f'{d}_{k}': values for d, b in evaluation['per_positive'].items() for k, values in b.items()})
        else:
            er = refined(head, be, 'enzyme'); rr = refined(head, br, 'reaction')
            if args.scope == 'test':
                scores = np.lib.format.open_memmap(run / 'test_scores.npy', mode='w+', dtype=np.float32,
                    shape=(len(br), len(candidate_ids)))
            records = []
            with ThreadPoolExecutor(max_workers=16) as pool:
                for start in range(0, len(br), 64):
                    values = .6 * (rr[start:start+64] @ er.T)
                    values += .4 * torch.sparse.mm(se, sr[start:start+64].T.contiguous()).T
                    values = values.cpu().numpy()[:, expanded]
                    if args.scope == 'test':
                        scores[start:start+len(values)] = values
                    else:
                        arguments = [(start+i, payload['ids'][start+i], val, positives[start+i], kept) for i, val in enumerate(values)]
                        records.extend(r for r in pool.map(evaluate_query, arguments) if r is not None)
            if args.scope == 'test':
                scores.flush(); del scores
                (run / 'query_ids.txt').write_text('\n'.join(payload['ids']) + '\n')
                (run / 'candidate_ids.txt').write_text('\n'.join(candidate_ids) + '\n')
                subprocess.run([sys.executable, str(ROOT / 'scripts/generalization_clipzyme_screening_evaluate.py'),
                    '--scores', str(run / 'test_scores.npy'), '--query-ids', str(run / 'query_ids.txt'),
                    '--candidate-ids', str(run / 'candidate_ids.txt'), '--protocol', str(CROSS / 'clipzyme_screening_evaluation_protocol_v2'),
                    '--output', str(run / 'test_evaluation'), '--metric-workers', '16'], check=True)
                result = json.loads((run / 'test_evaluation/summary.json').read_text())
                result['freeze'] = freeze
            else:
                summary = {}
                for table, count in [('table1', len(candidate_ids)), ('table2', len(kept))]:
                    rows = [r[table] for r in records if r[table] is not None]
                    summary[table] = dict(queries=len(rows), candidate_ids=count,
                        **{m: float(np.mean([r[m] for r in rows])) for m in METRICS})
                if summary['table1']['queries'] != 2652 or summary['table2']['queries'] != 2216:
                    raise ValueError('Validation denominator changed')
                (run / 'validation_per_query.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
                result = dict(summary=summary, freeze=freeze)
        if args.scope == 'test' and v == 'control':
            prior = json.loads(Path(task['model']['benchmark_result']).read_text())['summary']
            if args.task == 'enzymemap':
                errors = [abs(result['summary'][t][m] - prior[t][m]) for t in ('table1', 'table2') for m in METRICS]
            else:
                errors = [abs(result['summary'][d]['all']['reactzyme_mrr'] - prior[d]['all']['reactzyme_mrr']) for d in prior]
            result['v4_control_max_metric_error'] = max(errors)
            if max(errors) > 1e-6:
                raise ValueError(f'Original V4 test parity failed: {errors}')
        atomic_json(result_path, result)
        print(json.dumps(dict(task=args.task, variant=v, scope=args.scope, summary=result['summary'])), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--scope', choices=('validation', 'test'), required=True)
    p.add_argument('--variant')
    p.add_argument('--cache-only', action='store_true')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    evaluate(args)


if __name__ == '__main__':
    main()
