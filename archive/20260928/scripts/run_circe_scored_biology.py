#!/usr/bin/env python3
"""Matched lambda experiment: one frozen feature export, small scored heads.

No test data are used. Each lambda uses identical initialization and batches.
The original checkpoint and its training/loss implementation remain untouched.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'runs/circe_v3_feature_gate_vybz7n4w'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save_json(path, value):
    from horizyn.generalization_diagnostics import atomic_json
    atomic_json(path, value)


def save_tensor(path, value):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def feature_root(args):
    return getattr(args, 'feature_cache_root', None) or args.run_root


def prepare_cached_sweep(args):
    """Read an immutable completed snapshot; never regenerate its features."""
    import torch
    source = feature_root(args)
    if source == args.run_root:
        raise ValueError('Use a new output directory, separate from the feature cache')
    if not (source/'complete.json').is_file():
        raise ValueError('Feature-cache source must be a completed experiment')
    original = json.loads((source/'manifest.json').read_text())
    for name in ('config', 'checkpoint'):
        path = getattr(args, name)
        if str(path) != original['settings'][name] or sha(path) != original['source_sha256'][str(path)]:
            raise ValueError(f'Feature cache {name} mismatch')
    if args.cache_batch_size != original['settings']['cache_batch_size']:
        raise ValueError('Keep cache-batch-size identical to the source cache')
    for relative, digest in original['code'].items():
        if relative.startswith('horizyn/') and sha(ROOT/relative) != digest:
            raise ValueError(f'Cached model/data code changed: {relative}')
    data = torch.load(source/'data.pt', weights_only=True)
    cache_files = {}
    for kind, ids in (('enzyme', data['proteins']), ('train', data['train_q']), ('validation', data['val_q'])):
        for chunk in range((len(ids)+args.cache_batch_size-1)//args.cache_batch_size):
            path = source/'cache'/kind/f'{chunk:06d}.pt'
            stat = path.stat()  # fail before starting workers if any chunk is missing
            if not stat.st_size:
                raise ValueError(f'Empty cached feature batch: {path}')
            cache_files[str(path.relative_to(source))] = [stat.st_size, stat.st_mtime_ns]
    settings = {k: v for k, v in vars(args).items() if k not in ('stage', 'worker', 'weight', 'device')}
    manifest = dict(settings=json.loads(json.dumps(settings, default=str)),
        frozen_source=str(source), frozen_source_sha256={name: sha(source/name)
            for name in ('manifest.json', 'data.pt', 'annotations.json', 'split_audit.json')},
        cache_files=cache_files, trainer_sha256=sha(__file__))
    target = args.run_root/'manifest.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('Sweep inputs or settings changed: choose a new run root')
    if not target.exists():
        save_json(target, manifest)
    for name in ('annotations.json', 'split_audit.json'):
        save_json(args.run_root/name, json.loads((source/name).read_text()))
    print(f'Reusing completed frozen features from {source}; no extraction.', flush=True)


def prepare(args):
    import yaml
    from horizyn.capability.scored_biology_targets import build_targets, positive_pairs
    config = yaml.safe_load(args.config.read_text())
    data = config['data']
    old_meta = json.loads(Path(data['protein_biofp_vocab_path']).read_text())
    sources = old_meta['sources']
    files = {str(args.config): sha(args.config), str(args.checkpoint): sha(args.checkpoint)}
    for path in [data['train_pairs_path'], data['validation_pairs_path'],
                 data['train_reactions_path'], data['validation_reactions_path'],
                 data['validation_retrieval_candidate_ids_path'],
                 data['protein_biofp_targets_path'], data['protein_biofp_vocab_path'],
                 *[v for v in sources.values() if v]]:
        files[path] = sha(path)
    for role, expected in old_meta.get('source_sha256', {}).items():
        if expected and files[sources[role]] != expected:
            raise ValueError(f'Annotation source changed since targets were built: {role}')
    # Cache identities avoid rehashing terabyte residue stores. Include VDS sources.
    from horizyn.datasets.residue_hdf5 import residue_hdf5_store_identity
    cache_identity = {'protein': residue_hdf5_store_identity(data['protein_residue_embeds_path'])}
    for key, value in data.items():
        if value and ('reaction_' in key and key.endswith('_embeds_path')):
            stat = Path(value).stat()
            cache_identity[key] = [str(Path(value).resolve()), stat.st_size, stat.st_mtime_ns]
    code = {str(path.relative_to(ROOT)): sha(path) for path in sorted((ROOT/'horizyn').rglob('*.py'))}
    code[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    settings = {k: v for k, v in vars(args).items()
                if k not in ('stage', 'worker', 'weight', 'device')}
    settings = json.loads(json.dumps(settings, default=str))
    manifest = dict(settings=settings, source_sha256=files, caches=cache_identity, code=code)
    manifest_path = args.run_root/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Inputs or settings changed: choose a new run root')
        if (args.run_root/'data.pt').exists():
            print('Reusing verified prepared experiment', flush=True)
            return
    else:
        save_json(manifest_path, manifest)
    train = positive_pairs(data['train_pairs_path'])
    val = positive_pairs(data['validation_pairs_path'])
    train_q = sorted({q for q, _ in train})
    val_q = sorted({q for q, _ in val})
    if set(train) & set(val):
        raise ValueError('Positive training/validation pairs overlap')
    save_json(args.run_root/'split_audit.json', dict(train_reactions=len(train_q),
        validation_reactions=len(val_q), shared_reactions=len(set(train_q) & set(val_q)),
        unseen_validation_reactions=len(set(val_q)-set(train_q)), shared_positive_pairs=0))
    candidates = sorted(set(Path(data['validation_retrieval_candidate_ids_path']).read_text().splitlines()))
    if '' in candidates or not {p for _, p in val} <= set(candidates):
        raise ValueError('Invalid validation candidate catalog')
    proteins = sorted({p for _, p in train} | set(candidates))
    arrays, annotations = build_targets(config)
    save_json(args.run_root/'annotations.json', annotations)
    import torch
    annotation_tensors = {k: torch.from_numpy(v.copy()) for k, v in arrays.items() if not k.endswith('_ids')}
    save_tensor(args.run_root/'data.pt', dict(train=train, validation=val, train_q=train_q, val_q=val_q,
        proteins=proteins, candidates=candidates,
        enzyme_annotation_ids=arrays['enzyme_ids'].tolist(), reaction_annotation_ids=arrays['reaction_ids'].tolist(),
        annotations=annotation_tensors))
    print('Prepared:', json.dumps(annotations['coverage']), flush=True)


def export_features(args):
    import torch
    from torch.nn import functional as F
    from horizyn.config import load_config
    from horizyn.benchmarks.retrieval import (BenchmarkTask, build_reaction_inputs,
        load_repo_checkpoint, encode_reactions, encode_residue_targets)
    from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
    data = torch.load(args.run_root/'data.pt', weights_only=True)
    workers = len(args.gpus.split(','))
    pending = [args.run_root/'cache'/kind/f'{chunk:06d}.pt'
        for kind, ids in (('enzyme', data['proteins']), ('train', data['train_q']), ('validation', data['val_q']))
        for chunk in range((len(ids)+args.cache_batch_size-1)//args.cache_batch_size)
        if chunk % workers == args.worker]
    if all(path.exists() for path in pending):
        print(f'worker={args.worker} frozen export already complete', flush=True)
        return
    config = load_config(str(args.config))
    module, _ = load_repo_checkpoint(args.checkpoint, config, args.device)
    module.requires_grad_(False)
    captured = {}
    def enzyme_hook(_module, inputs):
        captured['enzyme'] = inputs[0].detach().float().cpu()
    def reaction_hook(_module, inputs):
        tokens, valid = inputs[:2]
        captured['reaction'] = torch.cat((F.normalize(tokens.float(), dim=-1).flatten(1), valid.float()), 1).cpu()
    handles = [module.model.multiview_encoder.gate.register_forward_pre_hook(enzyme_hook),
               module.model.query_encoder.competitive_fusion.register_forward_pre_hook(reaction_hook)]
    residue = ResidueEmbedDataset(config.data.protein_residue_embeds_path, dtype=torch.float32,
        max_tokens=config.data.max_protein_tokens, truncation=config.data.protein_truncation)
    try:
        for kind, ids in (('enzyme', data['proteins']), ('train', data['train_q']), ('validation', data['val_q'])):
            reaction_inputs = None
            if kind != 'enzyme':
                task = BenchmarkTask(name=kind, task_type='retrieval', dataset='reactzyme',
                    task_label='reaction_smi', split=kind, pairs=Path(config.data[f'{kind}_pairs_path']),
                    # Dataset creates _f/_r cache IDs; only _f is exported below.
                    bidirectional_reactions=True,
                    reactions=Path(config.data[f'{kind}_reactions_path']),
                    reaction_model_embeds_h5=Path(config.data[f'{kind}_reaction_t5v2_embeds_path']),
                    reaction_unimol2_embeds_h5=Path(config.data[f'{kind}_reaction_unimol2_embeds_path']),
                    reaction_chiro_embeds_h5=Path(config.data[f'{kind}_reaction_chiro_embeds_path']))
                reaction_inputs = build_reaction_inputs(task, config)
                available = set(reaction_inputs.keys)
            for chunk, start in enumerate(range(0, len(ids), args.cache_batch_size)):
                if chunk % workers != args.worker:
                    continue
                keys = ids[start:start+args.cache_batch_size]
                path = args.run_root/'cache'/kind/f'{chunk:06d}.pt'
                if path.exists():
                    continue  # only atomically completed batches become visible
                captured.clear()
                with torch.inference_mode():
                    if kind == 'enzyme':
                        base = encode_residue_targets(module, residue, keys, args.device, len(keys), False)
                        features = captured['enzyme']
                    else:
                        # Pair IDs are plain; forward feature keys are normally suffixed.
                        resolved = [key if key in available else key+'_f' for key in keys]
                        if not set(resolved) <= available:
                            raise KeyError('Reaction cache missing requested IDs')
                        base = encode_reactions(module, reaction_inputs, resolved, args.device, len(keys))
                        features = captured['reaction']
                    base = F.normalize(base.float(), dim=-1).cpu()
                if len(features) != len(keys) or not torch.isfinite(features).all() or not torch.isfinite(base).all():
                    raise ValueError('Invalid frozen feature export')
                save_tensor(path, dict(ids=keys, features=features, base=base))
                print(f'worker={args.worker} {kind} {start+len(keys)}/{len(ids)}', flush=True)
    finally:
        for handle in handles:
            handle.remove()


def load_features(args, kind, ids):
    import torch
    records = [torch.load(feature_root(args)/'cache'/kind/f'{i:06d}.pt', weights_only=True)
               for i in range((len(ids)+args.cache_batch_size-1)//args.cache_batch_size)]
    if [key for row in records for key in row['ids']] != ids:
        raise ValueError(f'Frozen cache ID mismatch: {kind}')
    return [torch.cat([row[field] for row in records]).to(args.device) for field in ('features', 'base')]


def train(args):
    import torch
    from horizyn.scored_biology import ScoredBiology, mixed_scores, known_positive_mask, retrieval_metrics
    from horizyn.losses import DecoupledAllPositiveInfoNCELoss
    output = args.run_root/f'lambda_{args.weight:g}'
    if (output/'complete.json').exists():
        print(f'lambda={args.weight:g} already complete', flush=True)
        return
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    data = torch.load(feature_root(args)/'data.pt', weights_only=True)
    metadata = json.loads((feature_root(args)/'annotations.json').read_text())
    ep, up = load_features(args, 'enzyme', data['proteins'])
    rq, uq = load_features(args, 'train', data['train_q'])
    rv, uv = load_features(args, 'validation', data['val_q'])
    model = ScoredBiology(ep.shape[1], rq.shape[1], metadata['families']).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    criterion = DecoupledAllPositiveInfoNCELoss(beta=10., learn_beta=False, unknown_negative_weight=.5).to(args.device)
    p_index = {key: i for i, key in enumerate(data['proteins'])}
    q_index = {key: i for i, key in enumerate(data['train_q'])}
    train_edges = torch.tensor([(q_index[q], p_index[p]) for q, p in data['train']], device=args.device)
    packed_edges = (train_edges[:, 0]*len(ep)+train_edges[:, 1]).sort().values
    targets = {}
    for side, all_ids in (('enzyme', data['proteins']), ('reaction', data['train_q'])):
        positions = {key: i for i, key in enumerate(data[f'{side}_annotation_ids'])}
        targets[side] = {}
        for family in metadata['families']:
            indices = data['annotations'][f'{side}_{family}_indices']
            confidence = data['annotations'][f'{side}_{family}_confidence']
            index = torch.tensor([positions.get(key, len(indices)) for key in all_ids])
            targets[side][family] = (torch.cat((indices, torch.full_like(indices[:1], -1)))[index].to(args.device),
                torch.cat((confidence, torch.zeros_like(confidence[:1])))[index].to(args.device))
    candidates = torch.tensor([p_index[p] for p in data['candidates']], device=args.device)
    candidate_map = {p: i for i, p in enumerate(data['candidates'])}
    val_map = {q: i for i, q in enumerate(data['val_q'])}
    r2e = [[] for _ in data['val_q']]
    e2r = [[] for _ in data['candidates']]
    for q, p in data['validation']:
        r2e[val_map[q]].append(candidate_map[p])
        e2r[candidate_map[p]].append(val_map[q])
    novel_ids = [i for i, q in enumerate(data['val_q']) if q not in q_index]
    novel_index = torch.tensor(novel_ids, dtype=torch.long, device=args.device)
    output.mkdir(exist_ok=True)
    last = output/'last.pt'
    history, start_epoch, best = [], 0, -1.
    if last.exists():
        state = torch.load(last, weights_only=True, map_location=args.device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch, history, best = state['epoch']+1, state['history'], state['best']

    @torch.no_grad()
    def validate():
        model.eval()
        bq = model.encode(rv, 'reaction')
        bp = torch.cat([model.encode(ep[candidates[start:start+2048]], 'enzyme')
                        for start in range(0, len(candidates), 2048)])
        results = {}
        for alpha in (0., .05, .1, .2):
            r = retrieval_metrics(uv, up[candidates], bq, bp, r2e, alpha)
            e = retrieval_metrics(up[candidates], uv, bp, bq, e2r, alpha)
            results[str(alpha)] = dict(r2e=r, e2r=e, mean_mrr=(r['mrr']+e['mrr'])/2)
            if novel_ids:
                results[str(alpha)]['unseen_reaction_r2e'] = retrieval_metrics(
                    uv[novel_index], up[candidates], bq[novel_index], bp,
                    [r2e[i] for i in novel_ids], alpha)
        results['block_variance'] = dict(reaction=bq.var(0, unbiased=False).mean(-1).tolist(),
                                         enzyme=bp.var(0, unbiased=False).mean(-1).tolist())
        return results

    if not (output/'initial_validation.json').exists():
        save_json(output/'initial_validation.json', validate())
    for epoch in range(start_epoch, args.epochs):
        started = time.monotonic()
        model.train()
        order = torch.randperm(len(train_edges), generator=torch.Generator().manual_seed(args.seed+epoch))
        totals, steps = {}, 0
        gradient_ratio = 0.
        weight = args.weight * min(epoch / 3, 1.)
        for start in range(0, len(order), args.batch_size):
            batch = train_edges[order[start:start+args.batch_size].to(args.device)]
            qi, pi = batch[:, 0].unique(sorted=True), batch[:, 1].unique(sorted=True)
            bq, bp = model.encode(rq[qi], 'reaction'), model.encode(ep[pi], 'enzyme')
            score = mixed_scores(uq[qi], up[pi], bq, bp, args.alpha)
            positive = known_positive_mask(qi, pi, packed_edges, len(ep)).nonzero(as_tuple=True)
            retrieval = criterion(1-score, *positive)
            losses = {}
            bio_terms = []
            for side, ids, blocks in (('enzyme', pi, bp), ('reaction', qi, bq)):
                subset = {f: (v[0][ids], v[1][ids]) for f, v in targets[side].items()}
                bio, components = model.supervision(blocks, subset)
                if any((value[0] >= 0).any() for value in subset.values()):
                    bio_terms.append(bio)
                losses.update({f'{side}_{f}': value for f, value in components.items()})
            biological = torch.stack(bio_terms).mean() if bio_terms else bq.sum()*0
            total = retrieval + weight*biological
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            if steps == 0 and weight:
                parameters = tuple(model.parameters())
                rg = torch.autograd.grad(retrieval, parameters, retain_graph=True, allow_unused=True)
                bg = torch.autograd.grad(weight*biological, parameters, retain_graph=True, allow_unused=True)
                def norm(gradients):
                    return torch.stack([g.square().sum() for g in gradients if g is not None]).sum().sqrt()
                gradient_ratio = (norm(bg)/norm(rg).clamp_min(1e-12)).item()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.update(retrieval=retrieval, biology=biological, weighted_biology=weight*biological, total=total)
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.) + value.detach().item()
            steps += 1
        validation = validate()
        chosen_alpha = max((0., .05, .1, .2), key=lambda a: validation[str(a)]['mean_mrr'])
        current = validation[str(chosen_alpha)]['mean_mrr']
        row = dict(epoch=epoch+1, lambda_effective=weight, losses={k:v/steps for k,v in totals.items()},
                   validation=validation, selected_alpha=chosen_alpha, seconds=time.monotonic()-started,
                   first_batch_weighted_biology_to_retrieval_gradient_ratio=gradient_ratio)
        history.append(row)
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
            history=history, best=max(best, current), selected_alpha=chosen_alpha,
            base_checkpoint=str(args.checkpoint), labels=metadata['families'],
            enzyme_dim=ep.shape[1], reaction_dim=rq.shape[1], lambda_weight=args.weight,
            manifest_sha256=sha(args.run_root/'manifest.json'))
        if current > best:
            best = current
            save_tensor(output/'best.pt', state)
        save_tensor(last, state)
        save_json(output/'history.json', history)
        print(f'lambda={args.weight:g} epoch={epoch+1}/{args.epochs} '
              f'loss={row["losses"]["total"]:.5f} mean_MRR={current:.6f} '
              f'alpha={chosen_alpha:g} seconds={row["seconds"]:.1f}', flush=True)
    save_json(output/'complete.json', dict(epochs=args.epochs, best_validation_mean_mrr=best))


def summarize(args, weights):
    rows = []
    for weight in weights:
        history = json.loads((args.run_root/f'lambda_{weight:g}'/'history.json').read_text())
        best = max(history, key=lambda row: row['validation'][str(row['selected_alpha'])]['mean_mrr'])
        for alpha in (0., .05, .1, .2):
            metrics = best['validation'][str(alpha)]
            rows.append(dict(lambda_weight=weight, epoch=best['epoch'], alpha=alpha,
                selected_on_validation=alpha == best['selected_alpha'],
                r2e_mrr=metrics['r2e']['mrr'], e2r_mrr=metrics['e2r']['mrr'], mean_mrr=metrics['mean_mrr'],
                unseen_reaction_r2e_mrr=metrics.get('unseen_reaction_r2e', {}).get('mrr')))
    save_json(args.run_root/'comparison.json', rows)
    with (args.run_root/'comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def children(args, stage, jobs, lock_fd):
    """Bounded own-child cleanup on failure; never stop unrelated GPU users."""
    gpus = args.gpus.split(',')
    active = []
    try:
        for offset in range(0, len(jobs), len(gpus)):
            active = []
            for gpu, job in zip(gpus, jobs[offset:offset+len(gpus)]):
                command = command_for(args, stage) + job
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1',
                    OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
                label = '_'.join(job).replace('--', '')
                with (args.run_root/f'{stage}_{label}.log').open('a') as log:
                    active.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, pass_fds=(lock_fd,)))
            while active:
                for process in list(active):
                    status = process.poll()
                    if status is not None:
                        if status:
                            raise RuntimeError(f'{stage} worker failed, exit={status}; inspect worker log')
                        active.remove(process)
                if active:
                    time.sleep(1)
    finally:
        for process in active:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in active:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


def command_for(args, stage):
    command = [sys.executable, str(Path(__file__).resolve()), stage]
    for name in ('config', 'checkpoint', 'run_root', 'gpus', 'lambdas', 'epochs', 'seed',
                 'cache_batch_size', 'batch_size', 'lr', 'alpha'):
        command += ['--'+name.replace('_', '-'), str(getattr(args, name))]
    if getattr(args, 'allow_shared_gpus', False):
        command.append('--allow-shared-gpus')
    if getattr(args, 'feature_cache_root', None):
        command += ['--feature-cache-root', str(args.feature_cache_root)]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('launch', 'run', 'prepare', 'export', 'train'))
    parser.add_argument('--config', type=Path, default=BASE/'feature/configs/train.yaml')
    parser.add_argument('--checkpoint', type=Path, default=BASE/'feature/checkpoints/protein-pooling-epoch=29.ckpt')
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--feature-cache-root', type=Path,
                        help='Reuse a completed frozen-feature snapshot in a new output directory; skip extraction')
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--allow-shared-gpus', action='store_true',
                        help='Allow occupied selected GPUs; leaves existing jobs running, does not reserve VRAM')
    parser.add_argument('--lambdas', default='0,0.02,0.05')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cache-batch-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=1536)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=.1)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--weight', type=float, default=0.)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.config, args.checkpoint, args.run_root = [path.resolve() for path in (args.config, args.checkpoint, args.run_root)]
    if args.feature_cache_root is not None:
        args.feature_cache_root = args.feature_cache_root.resolve()
        if args.feature_cache_root == args.run_root:
            parser.error('Feature source and output run root must differ')
        if args.stage == 'export':
            parser.error('Cannot export into a reused feature snapshot')
    import math
    weights = [float(value) for value in args.lambdas.split(',')]
    if (not weights or len(set(weights)) != len(weights) or any(not math.isfinite(w) or w < 0 for w in weights)
        or min(args.epochs, args.batch_size, args.cache_batch_size) <= 0
        or not math.isfinite(args.lr) or args.lr <= 0 or not 0 < args.alpha <= 1):
        parser.error('Invalid experiment hyperparameters')
    args.run_root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'export':
        export_features(args)
        return
    if args.stage == 'train':
        train(args)
        return
    from scripts.run_circe_generalization_diagnostics import check_gpus
    if args.stage == 'launch':
        check_gpus(args.gpus.split(','), allow_shared=args.allow_shared_gpus)
        session = 'scored_bio_' + hashlib.sha256(str(args.run_root).encode()).hexdigest()[:10]
        if subprocess.run(['tmux', 'has-session', '-t', '='+session], capture_output=True).returncode == 0:
            raise RuntimeError(f'Session already exists: {session}')
        command = shlex.join(command_for(args, 'run'))
        shell = f'exec {command} >> {shlex.quote(str(args.run_root/"pipeline.log"))} 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(ROOT),
                        'env -u BASH_ENV -u ENV /bin/bash --noprofile --norc -c '+shlex.quote(shell)], check=True)
        print(f'Detached session: {session}\nLog: {args.run_root/"pipeline.log"}', flush=True)
        return
    with (args.run_root/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def interrupted(_signum, _frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        if args.feature_cache_root is not None:
            prepare_cached_sweep(args)
        else:
            prepare(args)
        if args.stage == 'prepare':
            return
        check_gpus(args.gpus.split(','), allow_shared=args.allow_shared_gpus)
        if args.feature_cache_root is None:
            print('Exporting frozen features once; see export_worker logs', flush=True)
            children(args, 'export', [['--worker', str(i)] for i in range(len(args.gpus.split(',')))], lock.fileno())
            check_gpus(args.gpus.split(','), allow_shared=args.allow_shared_gpus)
        print('Training matched lambda heads; see train_weight logs', flush=True)
        children(args, 'train', [['--weight', str(weight)] for weight in weights], lock.fileno())
        summarize(args, weights)
        save_json(args.run_root/'complete.json', dict(lambdas=weights, epochs=args.epochs))
        print('All lambda runs complete. Compare lambda_*/history.json.', flush=True)


if __name__ == '__main__':
    main()
