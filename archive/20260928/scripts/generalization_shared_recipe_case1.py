#!/usr/bin/env python3
"""Evaluate a benchmark-qualified, frozen recipe on all 144 restricted Case1 entries."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from generalization_reactzyme_architecture_phase2 import (
    ROOT, RUN, sha256, atomic_json, smooth_encoder, load_head, compose, canonical_dot)
from generalization_clipzyme_f3_screen import model_from_checkpoint, export_device_lock
from generalization_multiview_calibration import components, adjusted
from generalization_export import reaction_block
from horizyn.benchmarks.retrieval import BenchmarkTask, build_reaction_inputs, encode_reactions, encode_residue_targets
from horizyn.capability.reaction_set_features import materialize_reaction_set_features
from horizyn.training_io import StoragePrecisionResidues
from horizyn.inference_fusion_lineage import verify_refiner_base
from generalization_external_evaluate import (
    case1_metadata, evaluate_case1, write_csv, rank_vector, recall_summary)


@torch.inference_mode()
def predict(item, freeze, output, device):
    out = output / item['name']
    out.mkdir(exist_ok=False)
    phase = Path(item['source_phase2'])
    for path, expected in ((item['config'], item['config_sha256']), (item['checkpoint'], item['checkpoint_sha256']),
            (item['phase2_checkpoint'], item['phase2_checkpoint_sha256']),
            (phase / 'anchors.pt', item['dictionary_sha256']),
            (phase / 'features/manifest.json', item['feature_manifest_sha256'])):
        if sha256(path) != expected:
            raise ValueError(f'Frozen model artifact changed: {path}')
    audit = RUN / 'case1_audit'
    catalog_path = audit / 'features/catalog.json'
    catalog = json.loads(catalog_path.read_text())
    if catalog['candidate_rows'] != 144 or len(catalog['proteins']) != 123:
        raise ValueError('Restricted Case1 panel changed')
    case = ROOT / 'wet_lab/Case1/restricted_setting'
    residues = case / 'candidate_pool/proteins_prott5_residue.h5'
    means_path = audit / 'features/raw/protein_mean.h5'
    raw_receipt = json.loads((audit / 'features/raw/manifest.json').read_text())
    if (sha256(catalog_path) != raw_receipt['catalog_sha256'] or
            sha256(residues) != raw_receipt['inputs']['protein_residues']['sha256'] or
            sha256(means_path) != raw_receipt['outputs']['protein_mean.h5']['sha256']):
        raise ValueError('Case1 cached candidate inputs changed')
    module, config = model_from_checkpoint(Path(item['config']), Path(item['checkpoint']), device)
    train_manifest = json.loads((phase / 'features/manifest.json').read_text())
    lineage = verify_refiner_base(train_manifest['checkpoint']['sha256'], item['checkpoint_sha256'],
        item.get('calibration_receipt'), item['checkpoint'])
    config.data.protein_residue_embeds_path = str(residues)
    if item['checkpoint_already_calibrated']:
        dataset = StoragePrecisionResidues(str(residues), max_tokens=config.data.max_protein_tokens,
                                          truncation=config.data.protein_truncation)
        try:
            be = encode_residue_targets(module, dataset, catalog['proteins'], device, 64, False).float().to(device)
        finally:
            dataset.close()
    else:
        g, f, scale, reference, error = components(module, config, catalog['proteins'], device)
        be = adjusted(g, f, scale, item['fusion_multiplier'], reference)
    participant_set = bool(config.data.normalize_molecule_sets_as_self_reactions)
    historical = case / 'runs/tagatose_4_epimerase_e07cc5e4ab31'
    directory = audit / 'features/f3_epoch29' if participant_set else historical / 'features'
    reaction_csv = directory / 'feature_reaction.csv' if participant_set else historical / 'reaction.csv'
    schema = Path(config.data.train_reaction_chemistry_vectors_path).parent / 'schema.json'
    cofactors = ROOT / 'data/processed/capability_features/train_exact_rhea_reconstructed/cofactor_dictionary.csv'
    chemistry = out / 'reaction_set_features.npz'
    chemistry_receipt = materialize_reaction_set_features(reactions_path=reaction_csv, schema_path=schema,
        cofactor_dictionary_path=cofactors, output_path=chemistry)
    config.data.reaction_chemistry_vectors_path = str(chemistry)
    qids = catalog['query_ids']
    task = BenchmarkTask(name='frozen_shared_case1', task_type='screening', dataset='wet_lab',
        task_label=qids[0], split='query', pairs=reaction_csv, reactions=reaction_csv,
        reaction_model_embeds_h5=directory / 'reactiont5v2.h5', reaction_unimol2_embeds_h5=directory / 'unimol2.h5',
        reaction_chiro_embeds_h5=directory / 'chiro.h5', directions=('reaction_to_enzyme',))
    br = encode_reactions(module, build_reaction_inputs(task, config), qids, device, 1).float().to(device)
    with h5py.File(means_path) as data:
        if data['ids'].asstr()[:].tolist() != catalog['proteins'] or not data['complete'][:].all():
            raise ValueError('Case1 protein mean axis mismatch')
        means = torch.as_tensor(data['vectors'][:], device=device)
    blocks, masks = {}, {}
    for name, physical, molecular in (('t5v2', 'reactiont5v2', False), ('unimol2', 'unimol2', True), ('chiro', 'chiro', True)):
        b, m = reaction_block(directory / f'{physical}.h5', qids, molecular)
        blocks[name], masks[name] = torch.as_tensor(b, device=device), torch.as_tensor(m, device=device)
    with np.load(chemistry, allow_pickle=True) as data:
        if list(data['ids']) != qids:
            raise ValueError('Case1 chemistry query axis mismatch')
        blocks['chemistry'] = torch.as_tensor(data['vectors'], device=device)
        masks['chemistry'] = torch.as_tensor(data['mask'], device=device)
    head = load_head(Path(item['phase2_checkpoint']), item['feature_manifest_sha256'], device)
    smooth = smooth_encoder(phase, device)
    se = smooth.encode_semantic_enzymes(means, 128)
    sr = smooth.encode_semantic_reactions(blocks, masks)
    recipe = dict(step=100, cap=freeze['recipe']['residual_cap'], alpha=freeze['recipe']['semantic_alpha'])
    enzymes = compose(be, head, se, recipe, 'enzyme')
    reactions = compose(br, head, sr, recipe, 'reaction')
    if item['name'] == 'enzymemap':
        # Match screening's FP32 dense + semantic score contract.
        dense_e = F.normalize(be + recipe['cap'] * head.scale * head.enzyme(be), dim=-1)
        dense_r = F.normalize(br + recipe['cap'] * head.scale * head.reaction(br), dim=-1)
        scores = (1 - recipe['alpha']) * (dense_r @ dense_e.T) + recipe['alpha'] * (sr @ se.T)
    else:
        scores = canonical_dot(reactions, enzymes)
    native = canonical_dot(F.normalize(br, dim=-1), F.normalize(be, dim=-1))
    if scores.shape != (1, 123) or not bool(torch.isfinite(scores).all()):
        raise ValueError('Invalid Case1 prediction dimensions or values')
    np.savez(out / 'scores.npz', selected=scores.cpu().numpy(), native_before_phase2=native.cpu().numpy())
    atomic_json(out / 'prediction_receipt.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        labels_used=False, checkpoint_sha256=item['checkpoint_sha256'], phase2_checkpoint_sha256=item['phase2_checkpoint_sha256'],
        catalog_sha256=sha256(catalog_path), means_sha256=sha256(means_path), residue_sha256=sha256(residues),
        schema_sha256=sha256(schema), chemistry=chemistry_receipt, chemistry_sha256=sha256(chemistry),
        reaction_input_policy='participant_self_reaction' if participant_set else 'physical_reactant_product',
        raw_reaction_inputs={name: sha256(directory / name) for name in ('reactiont5v2.h5', 'unimol2.h5', 'chiro.h5')},
        inference_fusion_lineage=lineage, scores_sha256=sha256(out / 'scores.npz'), source_sha256=sha256(__file__)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    campaign = args.campaign.resolve()
    freeze_path = campaign / 'case1_freeze.json'
    freeze = json.loads(freeze_path.read_text())
    if sha256(campaign / 'qualification.json') != freeze['benchmark_qualification_sha256']:
        raise ValueError('Benchmark qualification changed after freeze')
    if not json.loads((campaign / 'qualification.json').read_text())['all_primary_targets_exceeded']:
        raise ValueError('Case1 follow-up requires qualification on both benchmarks')
    out = campaign / 'case1'
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    with export_device_lock(args.device):
        for item in freeze['models']:
            if not (out / item['name'] / 'prediction_receipt.json').exists():
                predict(item, freeze, out, args.device)
                torch.cuda.empty_cache()
    # All models are frozen and scores persisted before loading activity evidence.
    meta = case1_metadata(RUN / 'case1_audit')
    unique_rows, entry_rows, group_rows, summaries = [], [], [], {}
    entry_order = (ROOT / 'wet_lab/Case1/restricted_setting/candidate_pool/candidate_ids_prott5_order.txt').read_text().splitlines()
    index = {entry: i for i, protein in enumerate(meta['catalog']['proteins']) for entry in meta['group'][protein]['all_entry_ids']}
    if len(entry_order) != 144 or set(entry_order) != set(index):
        raise ValueError('144-entry candidate axis mismatch')
    for item in freeze['models']:
        path = out / item['name'] / 'scores.npz'
        receipt = json.loads((path.parent / 'prediction_receipt.json').read_text())
        if sha256(path) != receipt['scores_sha256']:
            raise ValueError('Case1 scores changed before evaluation')
        with np.load(path) as data:
            for key in ('selected', 'native_before_phase2'):
                label = item['name'] + '/' + key
                score = data[key].reshape(-1)
                summary, rows, groups = evaluate_case1(score, meta, label)
                expanded = score[[index[e] for e in entry_order]]
                ranks = rank_vector(expanded)
                summary['entry_level_144'] = {name: recall_summary(ranks, {i for i, e in enumerate(entry_order) if index[e] in ids})
                                            for name, ids in meta['indices'].items()}
                summaries[label] = summary
                unique_rows.extend(rows); group_rows.extend(groups)
                for i, entry in enumerate(entry_order):
                    entry_rows.append(dict(method=label, entry_id=entry, representative_id=meta['catalog']['proteins'][index[entry]],
                        score=float(expanded[i]), rank=int(ranks[i]), primary_paper=index[entry] in meta['indices']['primary_papers'],
                        broad_reported_active=index[entry] in meta['positive']))
    write_csv(out / 'unique_sequence_rankings.csv', unique_rows)
    write_csv(out / 'all_144_entry_rankings.csv', entry_rows)
    write_csv(out / 'study_and_construct_rankings.csv', group_rows)
    atomic_json(out / 'summary.json', dict(methods=summaries, candidate_rows=144, unique_sequences=123,
        freeze_sha256=sha256(freeze_path), model_selection=False, measured_new_activity=False,
        evidence='Retrospective literature panel: 12 primary-paper, 24 paper-or-patent, 81 workbook-active unique sequences; remaining 42 are conditional non-detect-only, not universal negatives.'))
    lines = ['# Case1: frozen shared benchmark recipe', '',
        'All 144 requested entries are retained; primary recovery counts use 123 unique sequences. '
        'Every target-trained checkpoint is reported without Case1 model selection. No new wet-lab measurements.', '',
        '| Target-trained model / score | Primary papers @25 (of 12) | Papers + patents @25 (of 24) | Workbook active @25 (of 81) | Conditional AUROC |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name, summary in summaries.items():
        lines.append('| ' + name + ' | ' + ' | '.join(str(summary[k]['recovered_at_25']) for k in
            ('primary_papers', 'primary_papers_and_patents', 'broad_workbook_active')) + f" | {summary['broad_assay_conditional_discrimination']['auc']:.6f} |")
    lines += ['', 'The panel contains one reaction, related constructs, heterogeneous assay conditions and literature curation. '
        'It supports a limited retrospective assessment, not broad catalytic generalization or prospective activity validation.', '',
        '[Full metrics](summary.json) · [144-entry rankings](all_144_entry_rankings.csv) · [123-sequence rankings](unique_sequence_rankings.csv)', '']
    (out / 'comparison.md').write_text('\n'.join(lines))
    atomic_json(out / 'complete.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        freeze_sha256=sha256(freeze_path), summary_sha256=sha256(out / 'summary.json')))


if __name__ == '__main__':
    main()
