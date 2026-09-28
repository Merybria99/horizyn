#!/usr/bin/env python3
"""Frozen CYP diagnostics: reproduce epoch 3, test validation-selected epoch, ablate inputs.

Run from horizyn/scripts with the existing CYP feature bundle. CPU only; writes a
new run directory, never changes weights, source features or the training run.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '2')
import numpy as np
import torch
import yaml
from scripts.cyp_specificity import digest, read_rows, write_rows, write_json, verify_bundle, evaluate_scores
from horizyn.benchmarks.retrieval import (BenchmarkTask, build_reaction_inputs,
    build_query_inputs, encode_residue_targets, load_repo_checkpoint, cosine_scores)
from horizyn.config import load_config
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset

GROUPS = ['reaction_model', 'unimol2', 'chiro', 'reaction_chemistry']


def resolve(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def geometry(matrix):
    x = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1)
    unit = x / np.maximum(norms[:, None], 1e-12)
    cos = (unit @ unit.T)[np.triu_indices(len(x), 1)]
    return dict(rows=len(x), mean_norm=float(norms.mean()),
                pairwise_cosine_quantiles=np.quantile(cos, [0, .25, .5, .75, 1]).tolist(),
                mean_coordinate_std=float(x.std(axis=0).mean()))


def paired_difference(current, reference):
    old = {r['query_id']: r for r in reference}
    delta = np.array([r['first_positive_mrr'] - old[r['query_id']]['first_positive_mrr'] for r in current])
    groups = sorted({r['organism'] for r in current})
    indices = [np.array([i for i, r in enumerate(current) if r['organism'] == g]) for g in groups]
    rng = np.random.default_rng(42)
    samples = [float(delta[np.concatenate([indices[i] for i in rng.integers(len(groups), size=len(groups))])].mean()) for _ in range(5000)]
    return dict(delta_mrr=float(delta.mean()), organism_cluster_ci95=np.quantile(samples, [.025, .975]).tolist(),
                better_queries=int((delta > 0).sum()), worse_queries=int((delta < 0).sum()),
                unchanged_queries=int((delta == 0).sum()), clusters=len(groups), bootstrap_samples=5000)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/benchmarks/cyp_enzymecage_f3_epoch03.yaml')
    p.add_argument('--baseline', type=Path, default=ROOT / 'runs/cyp_enzymecage_f3_epoch03')
    p.add_argument('--training', type=Path, default=ROOT / 'runs/enzymecage_f3_seed42')
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    settings = yaml.safe_load(args.config.read_text())
    spec = settings['models']['f3']
    bundle, feat = resolve(settings['benchmark']), args.baseline / 'features'
    manifest = verify_bundle(bundle)
    source = json.loads((args.baseline / 'scores/f3.json').read_text())
    for key in ['checkpoint', 'config']:
        if digest(resolve(spec[key])) != source['signature'][key + '_sha256']:
            raise ValueError(f'Baseline {key} changed')
    if digest(args.baseline / 'scores/f3.csv') != source['scores_sha256']:
        raise ValueError('Baseline scores changed')
    if digest(bundle / 'manifest.json') != source['signature']['benchmark_sha256']:
        raise ValueError('Benchmark changed')
    for value in list(source['signature']['features'].values()) + [source['signature']['chemistry']]:
        st = Path(value['path']).stat()
        if st.st_size != value['size'] or st.st_mtime_ns != value['mtime_ns']:
            raise ValueError(f'Baseline feature file changed: {value["path"]}')
    metrics_path = args.training / 'logs/train/protein_pooling_training/version_1/metrics.csv'
    metric = 'val/mean_bidirectional_mrr'
    validation = [r for r in read_rows(metrics_path) if r.get(metric)]
    chosen = max(validation, key=lambda r: float(r[metric]))
    selected_epoch = int(float(chosen['epoch']))
    selected_checkpoint = args.training / f'checkpoints/protein-pooling-epoch={selected_epoch:02d}.ckpt'
    if not selected_checkpoint.exists():
        raise ValueError('Validation-selected checkpoint unavailable')
    cases = [('baseline', [])]
    cases += [(f'mask_{g}', [g]) for g in GROUPS]
    cases += [(f'only_{g}', [x for x in GROUPS if x != g]) for g in GROUPS]
    signature = dict(script_sha256=digest(__file__), config_sha256=digest(args.config),
        baseline_receipt_sha256=digest(args.baseline / 'scores/f3.json'),
        benchmark_sha256=digest(bundle / 'manifest.json'),
        features={f.name: digest(f) for f in [feat / n for n in ['reactiont5.h5', 'unimol2.h5', 'chiro.h5', 'chemistry_f3.npz', 'reactions.csv']]},
        proteins_identity=source['signature']['features']['proteins.h5'],
        model_code={str(f.relative_to(ROOT)): digest(f) for f in sorted((ROOT / 'horizyn').rglob('*.py'))},
        helper_sha256=digest(ROOT / 'scripts/cyp_specificity.py'),
        selection=dict(metric=metric, epoch=selected_epoch, value=float(chosen[metric]),
                       metrics_sha256=digest(metrics_path), checkpoint=str(selected_checkpoint),
                       checkpoint_sha256=digest(selected_checkpoint)),
        cases=[c for c, _ in cases], device='cpu', threads=2,
        limitations=['Exploratory diagnostic on previously inspected CYP queries; do not tune or choose input variants on these results.',
            'Masking changes the feature distribution; this is a frozen inference intervention, not evidence that retraining without a modality will improve.',
            'Each query has a designated published catalyst; alternatives are unknown rather than assayed inactive.',
            'Internal validation selected the later epoch; no CYP results enter checkpoint selection.'])
    write_json(out / 'started.json', signature)
    query_rows = list(read_rows(bundle / 'queries.csv'))
    qids = [r['reaction_id'] for r in query_rows]
    pids = [r['protein_id'] for r in read_rows(bundle / 'proteins.csv')]
    candidates = list(read_rows(bundle / 'candidates.csv'))
    qidx, pidx = {q: i for i, q in enumerate(qids)}, {p: i for i, p in enumerate(pids)}
    original_records = list(read_rows(args.baseline / 'scores/f3.csv'))
    reference_scores = {(r['query_id'], r['protein_id']): float(r['score']) for r in original_records}
    _, reference_per_query, _ = evaluate_scores(bundle, original_records)
    results = []
    for label, checkpoint in [('epoch03', resolve(spec['checkpoint'])), (f'validation_best_epoch{selected_epoch:02d}', selected_checkpoint)]:
        dest = out / label
        dest.mkdir()
        print(f'Loading {label}: {checkpoint}', flush=True)
        config = load_config(str(resolve(spec['config'])))
        config.data.reaction_chemistry_vectors_path = str(feat / 'chemistry_f3.npz')
        for modality in ['unimol2', 'chiro', 'chemistry']:
            config.data[f'reaction_allow_missing_{modality}'] = False
        task = BenchmarkTask(name='CYP45_failfast', task_type='retrieval', dataset='CYP', task_label='within_family', split='external',
            pairs=bundle / 'encoding_pairs.csv', reactions=feat / 'reactions.csv',
            reaction_model_embeds_h5=feat / 'reactiont5.h5', reaction_unimol2_embeds_h5=feat / 'unimol2.h5', reaction_chiro_embeds_h5=feat / 'chiro.h5')
        reaction_data = build_reaction_inputs(task, config)
        if set(reaction_data.keys) != set(qids):
            raise ValueError('Query coverage differs')
        inputs = build_query_inputs(reaction_data, qids, 'cpu')
        for key, value in inputs.items():
            if value.shape[0] != len(qids) or not torch.isfinite(value).all():
                raise ValueError(f'Invalid input: {key}')
            if key.startswith('has_') and not value.all():
                raise ValueError(f'Missing feature: {key}')
        module, kind = load_repo_checkpoint(checkpoint, config, 'cpu')
        if kind != 'residue':
            raise ValueError('Expected residue model')
        module.eval()
        model, encoder = module.model, module.model.query_encoder
        if any(getattr(model, key, None) is not None for key in ['e2r_adapter', 'r2e_adapter']):
            raise ValueError('Directional adapters require separate scoring')
        if getattr(encoder, 'use_reaction_directional', False):
            raise ValueError('Directional residual requires separate intervention')
        names = list(encoder.modality_names)
        alias = {g: g for g in GROUPS}
        alias['chiro'] = next((g for g in ['chiro', 'chirality', 'chienn'] if g in names), 'chiro')
        if set(alias.values()) != set(names):
            raise ValueError(f'Unexpected modalities {names}')
        proteins = ResidueEmbedDataset(str(feat / 'proteins.h5'), max_tokens=1022, truncation='ends_center')
        if set(proteins.keys) != set(pids):
            raise ValueError('Protein coverage differs')
        print(f'Encoding {len(pids)} cached protein features on CPU', flush=True)
        try:
            with torch.inference_mode():
                targets = encode_residue_targets(module, proteins, pids, 'cpu', 16, False, progress_every_batches=10)
        finally:
            proteins.close()
        np.savez_compressed(dest / 'targets.npz', protein_ids=np.array(pids), embeddings=targets.numpy())
        original_dropout = encoder._apply_modality_dropout
        baseline_top1 = None
        baseline_per_query = None
        for case, exclude in cases:
            masked = [alias[g] for g in exclude]
            def masking(mask):
                mask = original_dropout(mask).clone()
                for name in masked:
                    mask[:, names.index(name)] = False
                if not mask.any(1).all():
                    raise ValueError('All modalities masked')
                return mask
            encoder._apply_modality_dropout = masking
            encoded, attentions = [], []
            try:
                with torch.inference_mode():
                    for offset in range(0, len(qids), 8):
                        embeddings, attention = model.encode_queries({k: v[offset:offset+8] for k, v in inputs.items()}, return_attention=True)
                        if list(attention['modality_names']) != names:
                            raise ValueError('Attention label mismatch')
                        weights = attention['modality'].cpu()
                        if not torch.allclose(weights.sum(1), torch.ones(len(weights)), atol=1e-5):
                            raise ValueError('Attention not normalized')
                        if any(torch.any(weights[:, names.index(name)] != 0) for name in masked):
                            raise ValueError('Mask intervention failed')
                        encoded.append(embeddings.cpu())
                        attentions.append(weights)
                    queries = torch.cat(encoded)
                    matrix = cosine_scores(queries, targets).numpy()
            finally:
                encoder._apply_modality_dropout = original_dropout
            records = [dict(query_id=r['query_id'], protein_id=r['protein_id'], score=float(matrix[qidx[r['query_id']], pidx[r['protein_id']]])) for r in candidates]
            metrics, per_query, rankings = evaluate_scores(bundle, records)
            top1 = {r['query_id']: r['protein_id'] for r in rankings if r['rank'] == 1}
            if case == 'baseline':
                baseline_top1, baseline_per_query = top1, per_query
                if label == 'epoch03':
                    error = max(abs(r['score'] - reference_scores[r['query_id'], r['protein_id']]) for r in records)
                    if error > 1e-5:
                        raise ValueError(f'Baseline mismatch {error}')
                    if [r['first_positive_rank'] for r in per_query] != [r['first_positive_rank'] for r in reference_per_query]:
                        raise ValueError('Baseline ranks differ')
                np.savez_compressed(dest / 'baseline.full_matrix.npz', query_ids=np.array(qids), protein_ids=np.array(pids), scores=matrix)
            weights = torch.cat(attentions).numpy()
            write_rows(dest / f'{case}.csv', records)
            write_rows(dest / f'{case}.per_query.csv', per_query)
            write_rows(dest / f'{case}.attention.csv', [dict(query_id=q, **{m: float(weights[i,j]) for j,m in enumerate(names)}) for i,q in enumerate(qids)])
            result = dict(model=label, case=case, metrics=metrics,
                versus_same_epoch=paired_difference(per_query, baseline_per_query),
                versus_original_epoch03=paired_difference(per_query, reference_per_query),
                changed_top1_queries=sum(top1[q] != baseline_top1[q] for q in qids),
                attention_mean={m: float(weights[:,j].mean()) for j,m in enumerate(names)},
                query_geometry=geometry(queries.numpy()), scores_sha256=digest(dest / f'{case}.csv'))
            if case == 'baseline' and label == 'epoch03':
                result['max_abs_score_error_vs_saved_gpu'] = error
            write_json(dest / f'{case}.json', result)
            results.append(result)
            print(f'{label} {case}: MRR={metrics["first_positive_mrr"]:.5f}, hit1={metrics["hit_at_1"]:.4f}', flush=True)
        write_json(dest / 'complete.json', dict(checkpoint_sha256=digest(checkpoint), files={f.name:digest(f) for f in sorted(dest.iterdir()) if f.is_file()}))
        del module, model, encoder, targets, inputs, reaction_data, original_dropout, masking
        gc.collect()
    write_json(out / 'summary.json', dict(complete=True, signature=signature, results=results, elapsed_seconds=time.monotonic()-start))
    print(f'Completed {len(results)} conditions in {time.monotonic()-start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
