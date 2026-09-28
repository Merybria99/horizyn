#!/usr/bin/env python3
"""Audit shared architecture, target-only training edges, frozen results and Case1 coverage."""
from datetime import datetime, timezone
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from generalization_screen_replication import ROOT, sha
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, action='append', help='Audit a specified training family instead of the original four inference variants')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cross = ROOT / 'runs/generalization_20260919_2251/cross_paper_retraining'
    names = ['shared_recipe_beta5_b512_e10_fusion3_v2', 'shared_recipe_alpha04_cap1_v1',
             'shared_recipe_alpha05_cap1_v1', 'shared_recipe_alpha04_cap05_v1']
    records, training = [], []
    seen_models = {}
    model_config = loss_config = phase_args = None
    torch.set_num_threads(4)
    campaigns = [p.resolve() for p in args.campaign] if args.campaign else [cross / n for n in names]
    for campaign in campaigns:
        name = campaign.name
        plan = json.loads((campaign / 'protocol.json').read_text())
        qualification = json.loads((campaign / 'qualification.json').read_text())
        freeze = json.loads((campaign / 'case1_freeze.json').read_text())
        done = json.loads((campaign / 'case1/complete.json').read_text())
        assert qualification['all_primary_targets_exceeded'] and qualification['passed_cells'] == 14
        assert qualification['protocol_sha256'] == sha(campaign / 'protocol.json')
        assert qualification['target_registry_sha256'] == sha(cross / 'goal_primary_comparators_20260921.json')
        assert freeze['benchmark_qualification_sha256'] == sha(campaign / 'qualification.json')
        assert done['freeze_sha256'] == sha(campaign / 'case1_freeze.json')
        assert done['summary_sha256'] == sha(campaign / 'case1/summary.json')
        for row in qualification['rows']:
            result = json.loads(Path(row['source']).read_text())['summary']
            actual = (result[row['metric']]['all']['reactzyme_mrr'] if row['benchmark'] == 'ReactZyme'
                      else result[row['setting']][row['metric']])
            assert np.isfinite(actual) and actual == row['value'] and actual > row['target']
            if 'source_sha256' in row:
                assert sha(row['source']) == row['source_sha256']
            if (row['benchmark'], row['setting'], row['metric']) == ('EnzymeMap', 'table2', 'ef0.1'):
                assert actual > 7.81
        for entry in plan.get('reactzyme', []):
            assert sha(entry['result']) == entry['result_sha256']
        for model in freeze['models']:
            phase = Path(model['source_phase2'])
            receipt = json.loads((campaign / 'case1' / model['name'] / 'prediction_receipt.json').read_text())
            assert receipt['created_utc'] >= freeze['frozen_utc'] and not receipt['labels_used']
            assert sha(campaign / 'case1' / model['name'] / 'scores.npz') == receipt['scores_sha256']
            signature = {key: model[key] for key in ('config_sha256', 'checkpoint_sha256',
                         'phase2_checkpoint_sha256', 'dictionary_sha256', 'feature_manifest_sha256')}
            if model['name'] in seen_models:
                assert signature == seen_models[model['name']]
                continue
            seen_models[model['name']] = signature
            config = yaml.safe_load(Path(model['config']).read_text())
            assert sha(model['config']) == model['config_sha256']
            assert sha(model['checkpoint']) == model['checkpoint_sha256']
            assert sha(model['phase2_checkpoint']) == model['phase2_checkpoint_sha256']
            assert sha(phase / 'anchors.pt') == model['dictionary_sha256']
            assert sha(phase / 'features/manifest.json') == model['feature_manifest_sha256']
            if model_config is None:
                model_config, loss_config = config['model'], config['training']['loss']
            assert config['model'] == model_config
            assert config['training']['loss'] == loss_config
            expected_batch = plan['recipe']['batch_size'] if 'recipe' in plan else plan['batch_size']
            assert config['data']['train_batch_size'] == expected_batch and config['seed'] == 42
            assert config['training']['learning_rate'] == .0001 and config['training']['weight_decay'] == .01
            assert config['model']['sleec_pooling']['freeze_scorer']
            checkpoint = torch.load(model['checkpoint'], map_location='cpu', weights_only=False)
            expected_epoch = model.get('selected_epoch', plan.get('recipe', {}).get('epochs'))
            assert checkpoint['epoch'] == expected_epoch - 1
            del checkpoint
            pairs_path = Path(config['data']['train_pairs_path'])
            if not pairs_path.is_absolute():
                pairs_path = ROOT / pairs_path
            with pairs_path.open() as stream:
                official_rows = list(csv.DictReader(stream))
            official_edges = {(r['reaction_id'], r['protein_id']) for r in official_rows}
            catalog = json.loads((phase / 'features/catalog.json').read_text())
            with np.load(phase / 'features/pairs.npz') as data:
                edges = data['train']
            phase_edges = {(catalog['reactions'][int(r)], catalog['proteins'][int(e)]) for r, e in edges}
            assert phase_edges == official_edges and len(edges) == len(phase_edges)
            saved = torch.load(model['phase2_checkpoint'], map_location='cpu', weights_only=False)
            registry = saved['registry']
            assert saved['fixed_step'] == 100 and registry['test_used'] is False and registry['warm_start'] is None
            assert registry['feature_manifest_sha256'] == model['feature_manifest_sha256']
            assert sha(phase / 'training/source.py') == registry['script_sha256']
            assert sha(phase / 'training/model_source.py') == registry['module_sha256']
            parameters = {k: registry['arguments'][k] for k in ('hidden','scale','learning_rate',
                'weight_decay','temperature','identity_weight','contrastive_objective','ranking_weight',
                'enzyme_weighting','identity_reference','seed','steps')}
            if phase_args is None:
                phase_args = parameters
            assert parameters == phase_args
            dictionary = torch.load(phase / 'anchors.pt', map_location='cpu', weights_only=False)
            assert dictionary['feature_manifest_sha256'] == model['feature_manifest_sha256']
            assert set(dictionary['train_protein_ids']) == {e for r, e in official_edges}
            assert set(dictionary['train_reaction_ids']) == {r for r, e in official_edges}
            training.append(dict(target=model['name'], training_pairs=str(pairs_path),
                training_pairs_sha256=sha(pairs_path), original_rows=len(official_rows),
                unique_training_edges=len(official_edges), phase2_edges_equal_target_training_edges=True,
                dictionary_contains_only_target_training_ids=True, config_sha256=model['config_sha256'],
                f3_checkpoint_sha256=model['checkpoint_sha256'], phase2_checkpoint_sha256=model['phase2_checkpoint_sha256'],
                phase2_training_source_sha256=registry['script_sha256'], phase2_parameters=parameters,
                fresh_f3_scope=('Target-specific fresh F3; frozen pretrained feature banks/SLEEC retained.' +
                    (' Reaction-Sim resumed its own optimizer after OOM.' if 'recipe' in plan else ' Uninterrupted base fits.'))))
            del saved, dictionary
        for filename, size in [('all_144_entry_rankings.csv', 144), ('unique_sequence_rankings.csv', 123)]:
            with (campaign / 'case1' / filename).open() as stream:
                rows = list(csv.DictReader(stream))
            groups = {r['method'] for r in rows}
            assert len(groups) == 8
            assert all(sum(r['method'] == group for r in rows) == size for group in groups)
        report = json.loads((campaign / 'method_report.json').read_text())
        assert sha(report['report']) == report['sha256']
        assert Path(report['report']).read_text() in (ROOT / 'findings.md').read_text()
        records.append(dict(campaign=name, benchmark_cells_verified=14, case1_entry_count=144,
            case1_unique_sequence_count=123, case1_score_modes=8, all_model_choices_frozen_before_new_predictions=True,
            report_in_findings=True, qualification_sha256=sha(campaign / 'qualification.json')))
    output = args.output.resolve() if args.output else (campaigns[0] / 'evidence_audit.json' if args.campaign else
                                                      cross / 'shared_winners_evidence_audit_20260921.json')
    output.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
        architecture_identical_across_targets=True, base_loss_and_optimizer_identical_across_targets=True,
        phase2_parameters_identical_across_targets=True, training=training, methods=records,
        frozen_sleec_pretraining_manifest=str(ROOT / 'checkpoints/SLEEC/sleec_stage1_prott5_uniref90_msa_4gpu_20260601_235157/run_manifest.json'),
        limitations=['Shared fitted weights and seed: configurations are not independent replications.',
            'Matched downstream training associations; frozen backbone and SLEEC pretraining differ from competitors.',
            'Test scores inspected repeatedly: exploratory point estimates, not statistical superiority.',
            'Case1 is one retrospective literature panel with homologs and related training chemistry.',
            'Only the listed campaigns are covered; pending studies are excluded.']), indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
