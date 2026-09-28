#!/usr/bin/env python3
"""Evaluate the fixed ReactZyme Horizyn checkpoints on the restricted Case1 panel.

No fitting or model selection occurs here. The Reaction-Sim fit is primary;
the two other fits are reported separately. Predictions precede label loading.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/reactzyme_public_baselines_20260921'
AUDIT = ROOT / 'runs/generalization_20260919_2251/case1_audit'
CASE = ROOT / 'wet_lab/Case1/restricted_setting'
NATIVE = ROOT / 'data/external/cyp_specificity_2026/horizyn_official'
SPLITS = ('reaction_smi', 'enzyme_smi', 'time')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def identity(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / 'evaluation_source.py')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    device = f'cuda:{args.gpu}'
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    physical = physical[args.gpu] if physical != [''] else str(args.gpu)
    if not physical or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in physical):
        raise ValueError('Invalid GPU identifier')
    gpu_lock = open(f'/tmp/enzymediscovery_screen_export_gpu_{physical}.lock', 'a+')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    catalog_path = AUDIT / 'features/catalog.json'
    means_path = AUDIT / 'features/raw/protein_mean.h5'
    raw_manifest = read(AUDIT / 'features/raw/manifest.json')
    catalog = read(catalog_path)
    assert catalog['candidate_rows'] == 144 and len(catalog['proteins']) == 123
    assert sha(catalog_path) == raw_manifest['catalog_sha256']
    assert sha(means_path) == raw_manifest['outputs']['protein_mean.h5']['sha256']
    residue_path = ROOT / raw_manifest['inputs']['protein_residues']['path']
    assert sha(residue_path) == raw_manifest['inputs']['protein_residues']['sha256']
    with h5py.File(means_path) as f:
        assert f['ids'].asstr()[:].tolist() == catalog['proteins']
        assert f['complete'][:].all()
        protein_means = f['vectors'][:]
    assert protein_means.shape == (123, 1024) and np.isfinite(protein_means).all()
    entry_path = CASE / 'candidate_pool/candidate_ids_prott5_order.txt'
    entry_order = entry_path.read_text().splitlines()
    entry_index = {entry: i for i, group in enumerate(catalog['groups']) for entry in group['all_entry_ids']}
    assert [g['representative_id'] for g in catalog['groups']] == catalog['proteins']
    assert len(entry_order) == len(set(entry_order)) == len(entry_index) == 144
    assert set(entry_order) == set(entry_index)

    reaction_path = CASE / 'runs/tagatose_4_epimerase_e07cc5e4ab31/reaction.csv'
    with reaction_path.open() as f:
        reactions = list(csv.DictReader(f))
    assert len(reactions) == 1 and [reactions[0]['reaction_id']] == catalog['query_ids']
    physical_smiles = reactions[0]['reaction_smiles']
    sides = physical_smiles.split('>')
    assert len(sides) == 3
    participant_smiles = '.'.join(m for side in sides for m in side.split('.') if m).replace('*', 'C') + '>>'

    feature_root = BASE / 'features'
    training_catalog = read(feature_root / 'catalog.json')
    feature_records = {}
    for name in ('prott5', 'horizyn_fp'):
        receipt = read(feature_root / f'{name}.complete.json')
        assert receipt['catalog_sha256'] == sha(feature_root / 'catalog.json')
        assert receipt['sha256'] == sha(feature_root / f'{name}.npy')
        feature_records[name] = identity(feature_root / f'{name}.npy')
    frozen_models = {}
    for split in SPLITS:
        directory = BASE / 'models' / f'horizyn_participant_set_{split}_seed42'
        protocol, selection, complete = [read(directory / name) for name in ('protocol.json', 'selection.json', 'complete.json')]
        assert read(directory / 'status.json')['stage'] == 'complete'
        assert protocol['split'] == split and protocol['seed'] == 42
        assert not protocol['test_used_for_training_or_selection'] and not selection['test_used']
        assert selection['epoch'] == complete['selected_epoch']
        assert protocol['source_model_sha256'] == sha(NATIVE / 'horizyn/model.py')
        frozen_models[split] = dict(primary=split == 'reaction_smi', selected_epoch=selection['epoch'],
            artifacts={name: identity(directory / name) for name in ('best.pt', 'protocol.json', 'selection.json', 'complete.json', 'test_scores.npz')})
    freeze = dict(created_utc=utc(), method='Original Horizyn architecture, ReactZyme participant-set adaptation',
        primary_split='reaction_smi', models=frozen_models, candidate_rows=144, unique_sequences=123,
        case1_used_for_model_selection=False, retraining=False,
        catalog=identity(catalog_path), protein_means=identity(means_path), entry_order=identity(entry_path),
        original_reaction=identity(reaction_path), original_reaction_smiles=physical_smiles,
        encoded_reaction_smiles=participant_smiles,
        reaction_policy='All physical participants on input side with empty output, matching the ReactZyme Horizyn training adapter.',
        training_features=feature_records, precision='FP32, highest matmul precision, TF32 disabled',
        tie_policy='Stable descending score in fixed catalog order; aliases expanded into original 144-entry order.',
        evaluation_source=identity(__file__),
        native_sources={str(p.relative_to(NATIVE)): identity(p) for p in [
            NATIVE / 'horizyn/model.py', NATIVE / 'horizyn/datasets/fingerprints/base.py',
            NATIVE / 'horizyn/datasets/fingerprints/rdkit_plus.py', NATIVE / 'horizyn/datasets/fingerprints/drfp.py',
            NATIVE / 'horizyn/chemistry/standardizer.py']})
    write(out / 'freeze.json', freeze)
    print(json.dumps(dict(stage='frozen', output=str(out), checkpoints=3, primary='reaction_smi')), flush=True)

    # Isolate the original feature generators from the local modified package.
    sys.path.insert(0, str(NATIVE))
    import horizyn
    from horizyn.datasets.csv import CSVDataset
    from horizyn.datasets.fingerprints.rdkit_plus import RDKitPlusFingerprintDataset
    from horizyn.datasets.fingerprints.drfp import DRFPFingerprintDataset
    from horizyn.model import DualContrastiveModel
    assert Path(horizyn.__file__).resolve().is_relative_to(NATIVE.resolve())
    sample_indices = np.linspace(0, len(training_catalog['reaction_ids']) - 1, 8, dtype=int)
    query_id = catalog['query_ids'][0]
    fingerprint_csv = out / 'fingerprint_inputs.csv'
    with fingerprint_csv.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['reaction_id', 'reaction_smiles'])
        w.writerow([query_id, participant_smiles])
        for i in sample_indices:
            w.writerow([training_catalog['reaction_ids'][i], training_catalog['reaction_smiles'][i].replace('*', 'C') + '>>'])
    dataset = CSVDataset(file_path=str(fingerprint_csv), key_column='reaction_id', columns=['reaction_smiles'])
    rd = RDKitPlusFingerprintDataset(dataset, vec_dim=1024, standardize=True)
    dr = DRFPFingerprintDataset(dataset, vec_dim=1024, standardize=True)

    def fp(key):
        return torch.cat([rd[key], dr[key]]).float().numpy()

    cached_fp = np.load(feature_root / 'horizyn_fp.npy', mmap_mode='r')
    cached_protein = np.load(feature_root / 'prott5.npy', mmap_mode='r')
    for i in sample_indices:
        assert np.array_equal(fp(training_catalog['reaction_ids'][i]), cached_fp[i]), 'Native fingerprint parity failure'
    query_fp = fp(query_id)[None, :]
    assert query_fp.shape == (1, 2048) and np.isfinite(query_fp).all() and np.any(query_fp)
    np.savez(out / 'inputs.npz', protein_means=protein_means, reaction_fingerprint=query_fp,
        protein_ids=np.asarray(catalog['proteins']), query_ids=np.asarray(catalog['query_ids']))
    parity = {'fingerprint_rows_bit_identical': len(sample_indices), 'checkpoint_score_checks': {}}

    with torch.inference_mode():
        for split in SPLITS:
            record = frozen_models[split]
            checkpoint = Path(record['artifacts']['best.pt']['path'])
            assert sha(checkpoint) == record['artifacts']['best.pt']['sha256']
            ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert ck['epoch'] == record['selected_epoch']
            model = DualContrastiveModel(
                query_encoder_kwargs=dict(input_dim=2048, output_dim=512, num_layers=2, widths=[4096, 4096], normalise_output=True),
                target_encoder_kwargs=dict(input_dim=1024, output_dim=512, num_layers=2, widths=[4096, 4096], normalise_output=True))
            model.load_state_dict(ck['model'], strict=True)
            model = model.to(device).eval().requires_grad_(False)
            # Reproduce a fixed submatrix of the saved benchmark predictions.
            with np.load(record['artifacts']['test_scores.npz']['path']) as saved:
                ridx, pidx = saved['reaction_index'], saved['protein_index']
                ri = np.linspace(0, len(ridx) - 1, 16, dtype=int)
                pi = np.linspace(0, len(pidx) - 1, 32, dtype=int)
                q, p = model(torch.tensor(np.asarray(cached_fp[ridx[ri]]), device=device),
                    torch.tensor(np.asarray(cached_protein[pidx[pi]]), device=device))
                reference = saved['scores'][np.ix_(ri, pi)]
                reproduced = (q @ p.T).cpu().numpy()
                error = float(np.abs(reproduced - reference).max())
                assert np.allclose(reproduced, reference, atol=3e-6, rtol=1e-5), (split, error)
                parity['checkpoint_score_checks'][split] = dict(shape=[16, 32], max_absolute_error=error)
            q, p = model(torch.tensor(query_fp, device=device), torch.tensor(protein_means, device=device))
            scores = (q @ p.T).cpu().numpy()
            assert scores.shape == (1, 123) and np.isfinite(scores).all()
            for embeddings in (q, p):
                assert torch.allclose(embeddings.norm(dim=-1), torch.ones(len(embeddings), device=device), atol=1e-5)
            np.savez(out / f'{split}_predictions.npz', scores=scores, reactions=q.cpu().numpy(), proteins=p.cpu().numpy(),
                protein_ids=np.asarray(catalog['proteins']), query_ids=np.asarray(catalog['query_ids']))
            print(json.dumps(dict(stage='predicted', split=split, selected_epoch=record['selected_epoch'], max_benchmark_score_error=error)), flush=True)
            del model, ck, q, p
            torch.cuda.empty_cache()
    write(out / 'prediction_receipt.json', dict(completed_utc=utc(), freeze_sha256=sha(out / 'freeze.json'),
        case1_labels_loaded=False, parity=parity, inputs_sha256=sha(out / 'inputs.npz'),
        outputs={split: identity(out / f'{split}_predictions.npz') for split in SPLITS}))
    fcntl.flock(gpu_lock, fcntl.LOCK_UN)
    gpu_lock.close()

    # Reuse the paper's fixed evidence definitions only after all scores are saved.
    from generalization_external_evaluate import case1_metadata, evaluate_case1, rank_vector, recall_summary, write_csv
    meta = case1_metadata(AUDIT)
    summaries, unique_rows, entry_rows, group_rows = {}, [], [], []
    for split in SPLITS:
        with np.load(out / f'{split}_predictions.npz') as data:
            scores = data['scores'].reshape(-1)
        summary, unique, groups = evaluate_case1(scores, meta, split)
        expanded = scores[[entry_index[e] for e in entry_order]]
        ranks = rank_vector(expanded)
        summary['entry_level_144'] = {name: recall_summary(ranks, {i for i, e in enumerate(entry_order) if entry_index[e] in ids})
            for name, ids in meta['indices'].items()}
        assert summary['primary_papers']['positive_count'] == 12
        assert summary['entry_level_144']['primary_papers']['positive_count'] == 15
        summary['primary'] = split == 'reaction_smi'
        summary['selected_epoch'] = frozen_models[split]['selected_epoch']
        summaries[split] = summary
        unique_rows.extend(sorted(unique, key=lambda r: r['rank']))
        group_rows.extend(groups)
        for i in np.argsort(-expanded, kind='stable'):
            entry = entry_order[i]
            entry_rows.append(dict(method=split, entry_id=entry, representative_id=catalog['proteins'][entry_index[entry]],
                score=float(expanded[i]), rank=int(ranks[i]), primary_paper=entry_index[entry] in meta['indices']['primary_papers'],
                broad_reported_active=entry_index[entry] in meta['positive']))
    write_csv(out / 'unique_sequence_rankings.csv', unique_rows)
    write_csv(out / 'all_144_entry_rankings.csv', entry_rows)
    write_csv(out / 'study_and_construct_rankings.csv', group_rows)
    write(out / 'summary.json', dict(methods=summaries, primary_split='reaction_smi', candidate_rows=144, unique_sequences=123,
        original_catalog_entries=145, unresolved_entry='H017', model_selection=False, retraining=False,
        evidence='Retrospective single-reaction panel; 12 primary-paper positives among 123 sequences, 15 among 144 entries. Conditional AUROC uses 81 reported-active versus 42 non-detect-only sequences.',
        label_sources={name: identity(AUDIT / name) for name in ('candidate_evidence.json', 'positive_sets.json', 'source_verification.json')},
        metric_source=identity(Path(__file__).parent / 'generalization_external_evaluate.py')))
    lines = ['# Horizyn on Case1', '',
        'Frozen ReactZyme-trained Horizyn models; no retraining or Case1 checkpoint selection. '
        'Reaction-Sim is the primary model, designated before these predictions. '
        'This uses the participant-set adaptation and its selected checkpoints, not the released development checkpoint.', '',
        '| Training split | Selected epoch | Paper positives @5 /12 | @10 /12 | @25 /12 | Entry positives @25 /15 | Conditional AUROC |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for split, s in summaries.items():
        paper = s['primary_papers']
        values = [split + (' (primary)' if s['primary'] else ''), s['selected_epoch'], paper['recovered_at_5'],
            paper['recovered_at_10'], paper['recovered_at_25'], s['entry_level_144']['primary_papers']['recovered_at_25'],
            f"{s['broad_assay_conditional_discrimination']['auc']:.6f}"]
        lines.append('| ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', 'All 144 resolved entries and 123 unique sequences are retained; H017 is unresolved in the original 145-entry panel. '
        'Scores use native RDKit+/DRFP with all reaction participants on the input side, '
        'cached ProtT5 means, and the original Horizyn towers in FP32. '
        'Eight native fingerprint rows match the training cache bit for bit; '
        'a fixed 16-by-32 benchmark score submatrix is reproduced for each checkpoint.', '',
        'The paper-supported positives and broader activity labels are distinct endpoints. '
        'The latter combine heterogeneous assay conditions; non-detection is not universal inactivity. '
        'This is one retrospective reaction with related constructs, so no candidate-IID confidence intervals or broad transfer claim is made.', '',
        '[Full metrics](summary.json) · [Unique-sequence rankings](unique_sequence_rankings.csv) · '
        '[144-entry rankings](all_144_entry_rankings.csv) · [Frozen inputs and models](freeze.json) · [Prediction checks](prediction_receipt.json)', '']
    (out / 'report.md').write_text('\n'.join(lines))
    write(out / 'complete.json', dict(completed_utc=utc(), models=3, primary_split='reaction_smi',
        summary_sha256=sha(out / 'summary.json'), prediction_receipt_sha256=sha(out / 'prediction_receipt.json'),
        outputs={name: identity(out / name) for name in ('unique_sequence_rankings.csv', 'all_144_entry_rankings.csv', 'study_and_construct_rankings.csv', 'report.md')}))
    print(json.dumps(dict(stage='complete', output=str(out))), flush=True)


if __name__ == '__main__':
    main()
