#!/usr/bin/env python3
"""Describe existing residue-query behavior on fixed unseen validation proteins."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from generalization_full_graph import atomic_json, sha

CROSS = ROOT / 'runs/generalization_20260919_2251/cross_paper_retraining'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--proteins', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(2)
    catalog = CROSS / 'clipzyme_f3_catalog_v1'
    def protein_ids(scope):
        with (catalog / f'{scope}_pairs.csv').open() as handle:
            return {r['protein_id'] for r in csv.DictReader(handle)}
    eligible = sorted(protein_ids('validation') - protein_ids('train'))
    ids = eligible[:args.proteins]
    if not ids:
        raise ValueError('No validation proteins absent from training')
    cache = Path('/tmp/enzymediscovery_f3_20260920/train_validation_prott5.h5')
    dataset = ResidueEmbedDataset(str(cache), max_tokens=1022, truncation='ends_center')
    residues = [dataset[key]['residue_embeddings'] for key in ids]
    dataset.close()
    original = next(m for m in json.loads((CROSS / 'shared_recipe_alpha04_cap05_v1/case1_freeze.json').read_text())['models']
                    if m['name'] == 'enzymemap')
    original_checkpoint = Path(json.loads(Path(original['calibration_receipt']).read_text())['source_checkpoint'])
    sources = {'V4': original_checkpoint}
    for name, folder in [('attraction_0p1', 'v4_biological_f3_20260921_v1'),
                         ('relative_0p1', 'v4_relative_biology_f3_20260921_v1'),
                         ('relative_1', 'v4_relative_biology_f3_weight1_20260921_v1')]:
        sources[name] = CROSS / folder / 'enzymemap/training/checkpoints/screen_selection/screen-epoch=09.ckpt'
    out = CROSS / 'v4_biological_query_audit_20260921_v1'; out.mkdir(exist_ok=True)
    atomic_json(out / 'protocol.json', dict(proteins=ids, eligible_validation_proteins=len(eligible),
        sampling='First 128 sorted hashed protein IDs in validation, excluding exact training protein IDs.',
        training_ids_excluded=True, no_test_or_case1_data=True, model_selection=False,
        raw_residue_cache=str(cache), validation_pairs_sha256=sha(catalog / 'validation_pairs.csv'),
        scope='Native F3 before fixed inference fusion adjustment, phase2 or semantic scoring.'))
    records, examples = [], {}
    for name, checkpoint in sources.items():
        module = ProteinPooledLitModule.load_from_checkpoint(str(checkpoint), map_location='cpu').eval()
        module.to(args.device).requires_grad_(False)
        statistics = []
        for key, residue in zip(ids, residues):
            x = residue[None].to(args.device)
            with torch.inference_mode():
                output, details = module.model.encode_targets(x, return_pooling_details=True)
            item = {'protein_id': key, 'length': len(residue)}
            for metric in ['enzyme_multiview_normalized_entropy', 'enzyme_multiview_effective_support',
                           'enzyme_multiview_head_similarity', 'enzyme_multiview_global_cosine']:
                item[metric] = float(details[metric].mean())
            gates = details['enzyme_multiview_gate_weights'][0].mean(-1).cpu().tolist()
            item['gates'] = dict(zip(module.model.multiview_encoder.view_names, gates))
            item['effective_support_fraction'] = item['enzyme_multiview_effective_support'] / len(residue)
            statistics.append(item)
            if key == ids[0]:
                # Reverse precomputed contextual residue vectors, not amino acids.
                with torch.inference_mode():
                    reordered = module.model.encode_targets(x.flip(1))
                error = float((output - reordered).abs().max())
                if error > 1e-5:
                    raise ValueError('Unexpected position dependence in the content pooler')
                weights = details['enzyme_multiview_attention'][0].cpu().numpy()
                examples[name] = dict(protein_id=key, length=len(residue), attention=weights.tolist(),
                    sleec_attention=details['weights'][0].cpu().tolist(), permutation_max_abs_error=error,
                    top10_residue_array_indices=[np.argsort(-row, kind='stable')[:10].tolist() for row in weights])
        summary = {metric: float(np.mean([r[metric] for r in statistics])) for metric in
                   ['enzyme_multiview_normalized_entropy', 'effective_support_fraction',
                    'enzyme_multiview_head_similarity', 'enzyme_multiview_global_cosine']}
        summary['gates'] = {view: float(np.mean([r['gates'][view] for r in statistics]))
                            for view in module.model.multiview_encoder.view_names}
        records.append(dict(model=name, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
            summary=summary, proteins=statistics, annotation_helper_created=module._biological_geometry is not None,
            native_residual_scale=float(module.model.multiview_encoder.residual_scale)))
        print(name, summary, flush=True)
        del module
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    atomic_json(out / 'query_audit.json', dict(records=records, examples=examples,
        interpretation='Attention and gates describe representation use, not causal attribution, residue activity labels, '
            'or semantic meanings for individual queries. Permutation invariance applies after ProtT5 context encoding.'))
    lines = ['# Existing residue-query behavior on unseen validation proteins', '',
        f'This fixed diagnostic uses {len(ids)} validation proteins absent from target training by exact protein ID. '
        'It compares native F3 models before phase2, semantic scoring or the fixed inference fusion adjustment. '
        'Selection uses sorted hashed IDs, not attention or activity outcomes. No test or Case1 data are used.', '',
        '| Model | Mean normalized raw attention entropy | Effective support / length | Centered head cosine | Global-output cosine |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in records:
        s = r['summary']
        lines.append(f"| {r['model']} | {s['enzyme_multiview_normalized_entropy']:.4f} | "
            f"{s['effective_support_fraction']:.4f} | {s['enzyme_multiview_head_similarity']:.4f} | "
            f"{s['enzyme_multiview_global_cosine']:.4f} |")
    lines += ['', 'Entropy is divided by log(sequence length); one means uniform raw attention. Effective support '
        'is exp(entropy) after the uniform mixture. Centered head cosine measures similarity after subtracting '
        'uniform attention; smaller values indicate more distinct residue weighting. Global-output cosine '
        'measures proximity to the existing global branch at the native training fusion scale.', '',
        '| Model | Mean global gate | Mean SLEEC gate | Combined four learned-view gates | Permutation error, example |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in records:
        gates = list(r['summary']['gates'].values())
        lines.append(f"| {r['model']} | {gates[0]:.4f} | {gates[1]:.4f} | {sum(gates[2:]):.4f} | "
                     f"{examples[r['model']]['permutation_max_abs_error']:.2e} |")
    lines += ['', 'Gate magnitudes are descriptive, not an additive decomposition of prediction importance. '
        'The gates act by feature dimension before another projection. The four queries select contextual '
        'residue content and are shared across proteins; they are not position embeddings. Reversing the '
        'already computed residue vectors preserves the pooled output within floating-point tolerance. '
        'Reversing an amino-acid sequence before ProtT5 would be a different operation.', '',
        'Neither diverse attention nor changed attention proves a catalytic-site assignment. These data do '
        'not justify naming individual queries EC, cofactor or mechanism queries. Biological supervision '
        'reaches the final shared embedding and can be distributed across existing views.', '',
        f'[Per-protein values, attention example and checkpoint hashes]({out}/query_audit.json) · '
        '[Architecture explanation](v4_biological_signal_architecture.md)', '']
    (ROOT / 'documents/v4_biological_query_audit_20260921.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
