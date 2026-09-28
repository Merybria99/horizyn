import argparse
import json

import pytest
import torch

from horizyn.scored_biology import ScoredBiology, mixed_scores, known_positive_mask, retrieval_metrics
from horizyn.capability.scored_biology_targets import positive_pairs, sparse_rows


LABELS = {'ec': ['1', '1.1'], 'cofactor': ['NAD', 'NADP'], 'mechanism': ['oxidation', 'reduction']}


def test_explicit_gpu_sharing_retains_id_checks(monkeypatch, capsys):
    from scripts.run_circe_generalization_diagnostics import check_gpus
    def query(command, **kwargs):
        return argparse.Namespace(stdout='0, GPU-zero\n1, GPU-one\n'
            if 'index,uuid' in command[1] else 'GPU-zero, 1234\n')
    monkeypatch.setattr('scripts.run_circe_generalization_diagnostics.subprocess.run', query)
    with pytest.raises(RuntimeError, match='occupied'):
        check_gpus(['0'])
    check_gpus(['0', '1'], allow_shared=True)
    assert 'left running' in capsys.readouterr().out
    with pytest.raises(ValueError, match='do not exist'):
        check_gpus(['8'], allow_shared=True)
    with pytest.raises(ValueError, match='distinct'):
        check_gpus(['0', '0'], allow_shared=True)


@pytest.mark.parametrize('stage', ['run', 'export', 'train'])
def test_gpu_sharing_forwarded_to_children(stage):
    from scripts.run_circe_scored_biology import command_for
    args = argparse.Namespace(config='config', checkpoint='checkpoint', run_root='run',
        gpus='0,1', lambdas='0,.02,.05', epochs=10, seed=42, cache_batch_size=128,
        batch_size=1536, lr=1e-4, alpha=.1, allow_shared_gpus=True)
    assert command_for(args, stage).count('--allow-shared-gpus') == 1
    args.allow_shared_gpus = False
    assert '--allow-shared-gpus' not in command_for(args, stage)
    args.feature_cache_root = 'completed_source'
    command = command_for(args, stage)
    assert command[command.index('--feature-cache-root') + 1] == 'completed_source'


def targets(confidence=1.):
    return {f: (torch.tensor([[0], [-1]]), torch.tensor([[confidence], [0.]])) for f in LABELS}


def test_absolute_confidence_and_missing_gradient():
    model = ScoredBiology(4, 5, LABELS)
    x = torch.randn(2, 3, 64, requires_grad=True)
    full = model.supervision(x, targets())[0]
    gradient = torch.autograd.grad(full, x)[0]
    weak = model.supervision(x, targets(.4))[0]
    weak_gradient = torch.autograd.grad(weak, x)[0]
    assert torch.allclose(weak, .4*full)
    assert torch.allclose(weak_gradient, .4*gradient)
    assert gradient[1].count_nonzero() == 0
    empty = {f: (torch.full((2, 1), -1), torch.zeros(2, 1)) for f in LABELS}
    zero = model.supervision(x, empty)[0]
    assert zero.item() == 0
    assert torch.autograd.grad(zero, x)[0].count_nonzero() == 0


def test_unit_blocks_and_exact_baseline():
    model = ScoredBiology(4, 5, LABELS)
    bp, bq = model.encode(torch.randn(3, 4), 'enzyme'), model.encode(torch.randn(2, 5), 'reaction')
    assert torch.allclose(bp.norm(dim=-1), torch.ones(3, 3))
    uq, up = torch.randn(2, 7), torch.randn(3, 7)
    assert torch.equal(mixed_scores(uq, up, bq, bp, 0), uq@up.T)
    embedding_q = torch.cat((uq*(.9**.5), bq.flatten(1)*(.1/3)**.5), 1)
    embedding_p = torch.cat((up*(.9**.5), bp.flatten(1)*(.1/3)**.5), 1)
    assert torch.allclose(mixed_scores(uq, up, bq, bp, .1), embedding_q@embedding_p.T, atol=1e-6)


def test_all_known_edges_not_only_diagonal():
    edges = torch.tensor([0*4+0, 0*4+1, 1*4+1, 2*4+3])
    mask = known_positive_mask(torch.tensor([0, 1]), torch.tensor([0, 1, 2]), edges, 4)
    assert mask.tolist() == [[True, True, False], [False, True, False]]


def test_metrics_use_all_positives_and_stable_ties():
    q = torch.zeros(2, 2)
    p = torch.ones(3, 2)
    bq, bp = torch.zeros(2, 3, 64), torch.zeros(3, 3, 64)
    result = retrieval_metrics(q, p, bq, bp, [[0, 2], []], 0, 1)
    assert result['mrr'] == pytest.approx((1+1/3)/2)
    assert result['first_mrr'] == 1
    assert result['queries'] == 1


def test_positive_pair_filter_and_sparse_alignment(tmp_path):
    path = tmp_path/'pairs.csv'
    path.write_text('reaction_id,protein_id,Label\nr,p,1\ns,q,0\nr,p,1\n')
    assert positive_pairs(path) == [('r', 'p')]
    indices, confidence = sparse_rows(['b', 'a'], {'a': {'NAD': .4}}, ['NAD'])
    assert indices.tolist() == [[-1], [0]]
    assert confidence[1, 0] == pytest.approx(.4)


def test_small_cpu_training_and_resume(tmp_path):
    from scripts.run_circe_scored_biology import train, save_tensor
    args = argparse.Namespace(run_root=tmp_path, seed=42, device='cpu', cache_batch_size=128,
        weight=.05, epochs=2, lr=1e-4, batch_size=4, alpha=.1, checkpoint=tmp_path/'base.ckpt')
    (tmp_path/'manifest.json').write_text('{}')
    (tmp_path/'annotations.json').write_text(json.dumps({'families': LABELS}))
    annotations = {}
    for side in ('enzyme', 'reaction'):
        for f in LABELS:
            annotations[f'{side}_{f}_indices'] = torch.tensor([[0], [1], [-1]])
            annotations[f'{side}_{f}_confidence'] = torch.tensor([[.4], [1.], [0.]])
    data = dict(train=[('a', 'p'), ('a', 'q'), ('b', 'q'), ('c', 'r')],
        validation=[('d', 'p'), ('d', 'q'), ('e', 'r')], train_q=['a', 'b', 'c'], val_q=['d', 'e'],
        proteins=['p', 'q', 'r'], candidates=['p', 'q', 'r'], annotations=annotations,
        enzyme_annotation_ids=['p', 'q', 'r'], reaction_annotation_ids=['a', 'b', 'c'])
    save_tensor(tmp_path/'data.pt', data)
    for kind, ids, dim in [('enzyme', data['proteins'], 4), ('train', data['train_q'], 5), ('validation', data['val_q'], 5)]:
        save_tensor(tmp_path/'cache'/kind/'000000.pt', dict(ids=ids, features=torch.randn(len(ids), dim),
            base=torch.nn.functional.normalize(torch.randn(len(ids), 8), dim=-1)))
    train(args)
    history = json.loads((tmp_path/'lambda_0.05/history.json').read_text())
    assert len(history) == 2
    assert history[0]['lambda_effective'] == 0
    train(args)
    assert json.loads((tmp_path/'lambda_0.05/history.json').read_text()) == history
    state = torch.load(tmp_path/'lambda_0.05/best.pt', weights_only=True)
    assert state['base_checkpoint'] == str(args.checkpoint)
    # A fresh head run can read the original snapshot without copying/exporting.
    args.feature_cache_root = tmp_path
    args.run_root = tmp_path/'fresh_sweep'
    args.run_root.mkdir()
    (args.run_root/'manifest.json').write_text('{}')
    train(args)
    fresh = json.loads((args.run_root/'lambda_0.05/history.json').read_text())
    assert [row['losses'] for row in fresh] == [row['losses'] for row in history]
    assert not (args.run_root/'cache').exists()


def test_cached_sweep_preflight_and_guards(tmp_path):
    from scripts.run_circe_scored_biology import prepare_cached_sweep, save_tensor, sha
    source, output = tmp_path/'source', tmp_path/'output'
    source.mkdir()
    output.mkdir()
    config, checkpoint = tmp_path/'config', tmp_path/'checkpoint'
    config.write_text('config')
    checkpoint.write_text('checkpoint')
    original = dict(settings=dict(config=str(config), checkpoint=str(checkpoint), cache_batch_size=1),
        source_sha256={str(path):sha(path) for path in (config, checkpoint)}, code={})
    for name, value in (('manifest.json', original), ('complete.json', {}),
                        ('annotations.json', {}), ('split_audit.json', {})):
        (source/name).write_text(json.dumps(value))
    save_tensor(source/'data.pt', dict(proteins=['p'], train_q=['q'], val_q=['v']))
    for kind in ('enzyme', 'train', 'validation'):
        save_tensor(source/'cache'/kind/'000000.pt', {'features': torch.zeros(1, 2)})
    before = {str(path.relative_to(source)):sha(path) for path in source.rglob('*') if path.is_file()}
    args = argparse.Namespace(feature_cache_root=source, run_root=output, config=config,
        checkpoint=checkpoint, cache_batch_size=1, lambdas='0.1,0.2,0.5,1', stage='prepare')
    prepare_cached_sweep(args)
    prepare_cached_sweep(args)
    after = {str(path.relative_to(source)):sha(path) for path in source.rglob('*') if path.is_file()}
    assert before == after
    assert not (output/'cache').exists()
    args.cache_batch_size = 2
    with pytest.raises(ValueError, match='cache-batch-size'):
        prepare_cached_sweep(args)
    args.cache_batch_size = 1
    checkpoint.write_text('different checkpoint')
    with pytest.raises(ValueError, match='checkpoint mismatch'):
        prepare_cached_sweep(args)


def test_target_builder_filters_heldout_edges_and_preserves_confidence(tmp_path):
    import numpy as np
    import pandas as pd
    from horizyn.capability.scored_biology_targets import build_targets
    paths = {key: tmp_path/f'{key}.csv' for key in ('train_pairs', 'matched_members', 'enzyme_cofactors', 'ec_labels')}
    paths['train_pairs'].write_text('reaction_id,protein_id\ntrain,p\n')
    paths['matched_members'].write_text('source_protein_id,source_reaction_id,reaction_id\np,heldout,R1\n')
    paths['enzyme_cofactors'].write_text('enzyme_id,enzyme_uniprotkb_cofactor_labels_train,quality_flags\np,NAD,enzyme_cofactor_from_uniprotkb_experimental\nother,NADP,\n')
    paths['directional_features'] = tmp_path/'features.parquet'
    pd.DataFrame([{'reaction_id': 'R1', 'core_cofactor_labels': 'NADP'}]).to_parquet(paths['directional_features'])
    old_arrays = {'ids': np.asarray(['p'])}
    for family in ('ec', 'mechanism'):
        old_arrays[f'{family}_positive_indices'] = np.array([[0]])
        old_arrays[f'{family}_confidence'] = np.array([[.8]], dtype=np.float32)
    np.savez(tmp_path/'old.npz', **old_arrays)
    metadata = dict(sources={k:str(v) for k,v in paths.items()}, families=LABELS, limitations=[])
    (tmp_path/'vocab.json').write_text(json.dumps(metadata))
    arrays, meta = build_targets({'data': dict(train_pairs_path=str(paths['train_pairs']),
        protein_biofp_targets_path=str(tmp_path/'old.npz'), protein_biofp_vocab_path=str(tmp_path/'vocab.json'))})
    assert arrays['enzyme_cofactor_confidence'][0, 0] == pytest.approx(.7)
    label = meta['families']['cofactor'][arrays['enzyme_cofactor_indices'][0, 0]]
    assert label == 'NAD'
    assert arrays['reaction_cofactor_indices'].max() == -1
    assert arrays['reaction_ec_indices'].max() == -1
