#!/usr/bin/env python3
"""Rerun the frozen manuscript CIRCE recipe on the restricted Case 1 panel.

Freeze the protocol, predict each benchmark-trained checkpoint, then evaluate.
Prediction never reads activity labels. No training or Case 1 model selection.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from generalization_reactzyme_architecture_phase2 import (
    ROOT, RUN, atomic_json, canonical_dot, load_head, sha256,
)
from generalization_clipzyme_f3_screen import model_from_checkpoint
from generalization_multiview_calibration import components
from horizyn.benchmarks.retrieval import BenchmarkTask, build_reaction_inputs, encode_reactions
from horizyn.capability.reaction_set_features import materialize_reaction_set_features

CASE = ROOT / 'wet_lab/Case1/restricted_setting'
AUDIT = RUN / 'case1_audit'
SOURCE = ROOT / 'runs/cersei_dictionary_free_sensitivity_20260924/protocol.json'
TASKS = ('reaction_smi', 'enzyme_smi', 'time', 'enzymemap')


def identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256(path)}


def freeze(out):
    if (out / 'protocol.json').exists():
        raise FileExistsError('Refusing to replace the frozen protocol')
    source = json.loads(SOURCE.read_text())
    out.mkdir(parents=True, exist_ok=True)
    inputs = {
        'catalog': AUDIT / 'features/catalog.json',
        'residues': CASE / 'candidate_pool/proteins_prott5_residue.h5',
        'entry_order': CASE / 'candidate_pool/candidate_ids_prott5_order.txt',
    }
    protocol = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), source=identity(SOURCE),
        source_script=identity(__file__), models=source['tasks'],
        recipe=source['reference'], selection=source['reference_selection'],
        primary_checkpoint='reaction_smi',
        primary_rationale='Retain the historically used Reaction-Sim deployment checkpoint; report all four independently. No Case 1 selection.',
        candidate_entries=144, unique_sequences=123,
        inputs={key: identity(path) for key, path in inputs.items()},
        evaluation_plan={
            'cutoffs': [5, 10, 25], 'ranking': 'descending cosine, stable original-catalog ties',
            'primary_evidence': '12 sequence-deduplicated paper-supported positives; also report their exact-sequence-expanded 144-entry labels',
            'secondary_evidence': '24 paper-or-patent sequences; 81 workbook-active vs 42 conditional non-detect-only sequences',
            'references': 'uniform-random expected recovery; no labels used to change scores or select a checkpoint',
        },
        retraining=False, ensembles=False, biological_supervision=False,
        retrospective=True,
    )
    atomic_json(out / 'protocol.json', protocol)
    print(json.dumps({'protocol': str(out / 'protocol.json'), 'models': list(protocol['models'])}), flush=True)


def read_protocol(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    for value in protocol['inputs'].values():
        if sha256(value['path']) != value['sha256']:
            raise ValueError('Case 1 input checksum changed')
    if sha256(__file__) != protocol['source_script']['sha256']:
        raise ValueError('Runner changed after protocol freeze')
    return protocol


@torch.inference_mode()
def predict(out, name, device):
    start = time.monotonic()
    protocol = read_protocol(out)
    item = protocol['models'][name]
    dest = out / name
    dest.mkdir(exist_ok=False)
    for key in ('checkpoint', 'head', 'manifest', 'config'):
        if sha256(item[key]) != item[key + '_sha256']:
            raise ValueError(f'Model artifact changed: {key}')
    catalog = json.loads(Path(protocol['inputs']['catalog']['path']).read_text())
    assert catalog['candidate_rows'] == 144 and len(catalog['proteins']) == 123
    manifest = json.loads(Path(item['manifest']).read_text())
    assert manifest['checkpoint']['sha256'] == item['checkpoint_sha256']
    print(f'{name}: loading frozen checkpoint on {device}', flush=True)
    model, config = model_from_checkpoint(Path(item['config']), Path(item['checkpoint']), device)
    config.data.protein_residue_embeds_path = protocol['inputs']['residues']['path']
    g, f, scale, native, error = components(model, config, catalog['proteins'], device)
    recipe = protocol['recipe']
    be = F.normalize(g + recipe['fusion_multiplier'] * scale * f, dim=-1)
    participant_set = bool(config.data.normalize_molecule_sets_as_self_reactions)
    train_csv = Path(config.data.train_reactions_path)
    if not train_csv.is_absolute():
        train_csv = ROOT / train_csv
    with train_csv.open() as handle:
        train_rows = list(csv.DictReader(handle))
    equations = [row['reaction_smiles'] for row in train_rows]
    directed = sum('>>' in s and s.split('>>')[0] != s.split('>>')[1] for s in equations)
    if participant_set and directed:
        raise ValueError('Participant policy conflicts with actual training reactions')
    if not participant_set and not directed:
        raise ValueError('Expected directed EnzymeMap training reactions')
    historical = CASE / 'runs/tagatose_4_epimerase_e07cc5e4ab31'
    directory = AUDIT / 'features/f3_epoch29' if participant_set else historical / 'features'
    reaction_csv = directory / 'feature_reaction.csv' if participant_set else historical / 'reaction.csv'
    schema = Path(config.data.train_reaction_chemistry_vectors_path).parent / 'schema.json'
    cofactors = ROOT / 'data/processed/capability_features/train_exact_rhea_reconstructed/cofactor_dictionary.csv'
    chemistry = dest / 'reaction_set_features.npz'
    chemistry_receipt = materialize_reaction_set_features(
        reactions_path=reaction_csv, schema_path=schema,
        cofactor_dictionary_path=cofactors, output_path=chemistry)
    config.data.reaction_chemistry_vectors_path = str(chemistry)
    task = BenchmarkTask(
        name='circe_current_case1', task_type='screening', dataset='wet_lab',
        task_label=catalog['query_ids'][0], split='query', pairs=reaction_csv, reactions=reaction_csv,
        reaction_model_embeds_h5=directory / 'reactiont5v2.h5',
        reaction_unimol2_embeds_h5=directory / 'unimol2.h5',
        reaction_chiro_embeds_h5=directory / 'chiro.h5', directions=('reaction_to_enzyme',))
    br = encode_reactions(model, build_reaction_inputs(task, config), catalog['query_ids'], device, 1).float().to(device)
    head = load_head(Path(item['head']), item['manifest_sha256'], device)
    assert abs(head.scale - .2) < 1e-8 and recipe['dictionary_weight'] == 0
    ze = F.normalize(be + recipe['kappa_inf'] * head.enzyme(be), dim=-1)
    zr = F.normalize(br + recipe['kappa_inf'] * head.reaction(br), dim=-1)
    scores = (zr @ ze.T).float() if name == 'enzymemap' else canonical_dot(zr, ze)
    if scores.shape != (1, 123) or not torch.isfinite(scores).all():
        raise ValueError('Invalid prediction')
    norm_error = max(float((x.norm(dim=-1) - 1).abs().max()) for x in (ze, zr))
    assert norm_error < 1e-6
    entry_order = Path(protocol['inputs']['entry_order']['path']).read_text().splitlines()
    by_rep = {group['representative_id']: group for group in catalog['groups']}
    index = {entry: i for i, rep in enumerate(catalog['proteins']) for entry in by_rep[rep]['all_entry_ids']}
    assert len(entry_order) == 144 and set(entry_order) == set(index)
    unique = scores.cpu().numpy().reshape(-1)
    expanded = unique[[index[e] for e in entry_order]]
    np.savez(dest / 'scores.npz', unique=unique, entries=expanded,
             protein_ids=np.asarray(catalog['proteins']), entry_ids=np.asarray(entry_order))
    np.savez(dest / 'embeddings.npz', enzyme=ze.cpu().numpy(), reaction=zr.cpu().numpy(),
             enzyme_global=g.cpu().numpy(), enzyme_functional=f.cpu().numpy(),
             enzyme_base=be.cpu().numpy(), reaction_base=br.cpu().numpy())
    torch.cuda.synchronize(device)
    receipt = dict(
        completed_utc=datetime.now(timezone.utc).isoformat(), model=name, labels_used=False,
        protocol=identity(out / 'protocol.json'), checkpoint=identity(item['checkpoint']),
        head=identity(item['head']), train_catalog=identity(train_csv),
        train_rows=len(train_rows), directed_train_rows=directed,
        reaction_input_policy='participant_self_reaction' if participant_set else 'physical_reactant_product',
        reaction_csv=identity(reaction_csv), schema=identity(schema), cofactor_dictionary=identity(cofactors),
        raw_reaction_inputs={name: identity(directory / name) for name in ('reactiont5v2.h5', 'unimol2.h5', 'chiro.h5')},
        chemistry=chemistry_receipt, chemistry_file=identity(chemistry),
        scores=identity(dest / 'scores.npz'), embeddings=identity(dest / 'embeddings.npz'),
        learned_fusion_scale=scale, effective_fusion_scale=recipe['fusion_multiplier'] * scale,
        fusion_reconstruction_max_error=error, embedding_unit_norm_max_error=norm_error,
        seconds=time.monotonic() - start, device=device, gpu=torch.cuda.get_device_name(device),
        peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
        python=sys.version, torch=torch.__version__, numpy=np.__version__,
    )
    atomic_json(dest / 'prediction_receipt.json', receipt)
    print(json.dumps({'model': name, 'seconds': receipt['seconds'], 'candidates': len(expanded), 'labels_used': False}), flush=True)


def evaluate(out):
    protocol = read_protocol(out)
    # All four predictions must be persisted and intact before labels are read.
    for name in TASKS:
        receipt = json.loads((out / name / 'prediction_receipt.json').read_text())
        assert receipt['protocol']['sha256'] == sha256(out / 'protocol.json')
        assert receipt['scores']['sha256'] == sha256(out / name / 'scores.npz')
    from generalization_external_evaluate import (
        case1_metadata, evaluate_case1, rank_vector, recall_summary, write_csv)
    meta = case1_metadata(AUDIT)
    index = {e: i for i, p in enumerate(meta['catalog']['proteins']) for e in meta['group'][p]['all_entry_ids']}
    summaries, unique_rows, entry_rows, group_rows = {}, [], [], []
    for name in TASKS:
        with np.load(out / name / 'scores.npz') as data:
            unique, expanded, entries = data['unique'], data['entries'], data['entry_ids'].tolist()
            assert data['protein_ids'].tolist() == meta['catalog']['proteins']
            assert np.array_equal(unique[[index[e] for e in entries]], expanded)
        summary, rows, groups = evaluate_case1(unique, meta, name)
        ranks = rank_vector(expanded)
        summary['entry_level_144'] = {
            key: recall_summary(ranks, {j for j, e in enumerate(entries) if index[e] in ids})
            for key, ids in meta['indices'].items()}
        by_idx = {row['representative_id']: row for row in rows}
        expanded_rows = []
        for j, e in enumerate(entries):
            rep = meta['catalog']['proteins'][index[e]]
            source = by_idx[rep]
            record = meta['records'][e]
            expanded_rows.append(dict(
                method=name, entry_id=e, representative_id=rep, rank=int(ranks[j]),
                unique_sequence_rank=source['rank'], score=float(expanded[j]),
                name=record['name'], organism=record['organism'],
                primary_paper=source['primary_paper'], primary_paper_or_patent=source['primary_paper_or_patent'],
                broad_reported_active=source['broad_reported_active'],
                cross_assay_conflict=source['cross_assay_conflict'],
                primary_source_ids=source['primary_source_ids'], same_sequence_entries=source['entry_ids'],
            ))
        summaries[name] = summary
        unique_rows.extend(sorted(rows, key=lambda x: x['rank']))
        entry_rows.extend(sorted(expanded_rows, key=lambda x: x['rank']))
        group_rows.extend(groups)
        write_csv(out / name / 'ranking_144.csv', sorted(expanded_rows, key=lambda x: x['rank']))
        write_csv(out / name / 'ranking_123_unique.csv', sorted(rows, key=lambda x: x['rank']))
    write_csv(out / 'all_144_entry_rankings.csv', entry_rows)
    write_csv(out / 'unique_sequence_rankings.csv', unique_rows)
    write_csv(out / 'study_and_construct_rankings.csv', group_rows)
    atomic_json(out / 'summary.json', dict(
        completed_utc=datetime.now(timezone.utc).isoformat(), methods=summaries,
        protocol=identity(out / 'protocol.json'), candidate_entries=144, unique_sequences=123,
        evidence={name: identity(AUDIT / name) for name in ('candidate_evidence.json', 'positive_sets.json', 'source_verification.json')},
        model_selection=False, measured_new_activity=False,
    ))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'predict', 'evaluate'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', choices=TASKS)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.mode == 'freeze':
        freeze(args.out.resolve())
    elif args.mode == 'predict':
        if not args.model:
            parser.error('--model is required for prediction')
        predict(args.out.resolve(), args.model, args.device)
    else:
        evaluate(args.out.resolve())
