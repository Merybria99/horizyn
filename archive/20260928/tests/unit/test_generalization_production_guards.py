"""Production provenance and sparse-index regressions; synthetic data only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from generalization_predict import validate_input_receipt
from generalization_build_bundle import build
from horizyn.generalization_retrieval import (GeneralizationDualEncoder, canonical_dot,
    sha256, validate_frozen_residual)
from horizyn.generalization_residual import FrozenGeometryResidual


def dump(path, data):
    path.write_text(json.dumps(data))
    return dict(path=str(path), sha256=sha256(path))


def anchors():
    torch.manual_seed(120)
    return dict(modalities=['t5v2'], enzyme_temperature=.03, reaction_temperature=.03,
        protein_neighbors=3, reaction_neighbors=2, protein_center=torch.zeros(4),
        reaction_centers={'t5v2': torch.zeros(4)},
        train_proteins=torch.nn.functional.normalize(torch.randn(5, 4), dim=1),
        train_reactions=torch.nn.functional.normalize(torch.randn(4, 4), dim=1),
        adjacency=torch.tensor([[0, 1], [1, -1], [1, 2], [3, -1], [0, 3]]),
        feature_manifest_sha256='training-manifest')


@pytest.mark.parametrize('alpha', [0., .25, 1.])
@pytest.mark.parametrize('batch_size', [1, 3, 32])
def test_sparse_index_matches_dense_and_roundtrips(tmp_path, alpha, batch_size):
    model = GeneralizationDualEncoder(anchors(), alpha=alpha)
    torch.manual_seed(14)
    base_e, mean = torch.randn(7, 4), torch.randn(7, 4)
    base_r, raw = torch.randn(3, 4), torch.randn(3, 4)
    enzymes = model.encode_enzymes(base_e, mean, batch_size)
    reactions = model.encode_reactions(base_r, {'t5v2': raw}, {'t5v2': torch.ones(3, dtype=torch.bool)})
    index = model.encode_enzyme_index(base_e, mean, batch_size)
    path = tmp_path / 'index.pt'
    torch.save(index, path)
    index = torch.load(path, weights_only=False)
    torch.testing.assert_close(model.score_index(reactions, index), canonical_dot(reactions, enzymes), atol=0, rtol=0)
    assert len(index['anchors'].crow_indices()) == len(base_e) + 1
    assert index['anchors'].crow_indices()[-1] == index['anchors']._nnz()


@pytest.mark.parametrize('batch_size', [0, -1, 1.5])
def test_invalid_index_batch_size_rejected(batch_size):
    model = GeneralizationDualEncoder(anchors())
    with pytest.raises(ValueError, match='positive integer'):
        model.encode_enzyme_index(torch.ones(2, 4), torch.ones(2, 4), batch_size)


def test_index_rejects_misaligned_protein_rows():
    model = GeneralizationDualEncoder(anchors())
    with pytest.raises(ValueError, match='identical row counts'):
        model.encode_enzyme_index(torch.ones(2, 4), torch.ones(1, 4))


@pytest.fixture
def bundle(tmp_path):
    a = anchors()
    torch.save(a, tmp_path / 'anchors.pt')
    residual = FrozenGeometryResidual(4, 8)
    state = dict(model_config=dict(dimension=4, hidden=8), state_dict=residual.state_dict(),
                 registry=dict(feature_manifest_sha256=a['feature_manifest_sha256'], test_used=False))
    checkpoint = tmp_path / 'selected.pt'
    torch.save(state, checkpoint)
    record = dict(path=str(checkpoint), sha256=sha256(checkpoint))
    frozen = dict(frozen_before_held_out_evaluation=True, models={'test': {'42': record}},
                  anchors={k: a[k] for k in ('modalities', 'enzyme_temperature', 'reaction_temperature', 'protein_neighbors', 'reaction_neighbors')})
    freeze_record = dump(tmp_path / 'frozen.json', frozen)
    specification = dict(schema='generalization_dual_encoder_bundle_v1',
        anchors=dict(path='anchors.pt', sha256=sha256(tmp_path / 'anchors.pt')),
        residuals=[record], frozen_recipe=freeze_record, anchor_alpha=.25,
        feature_manifest_sha256=a['feature_manifest_sha256'])
    path = tmp_path / 'bundle.json'
    dump(path, specification)
    return path, specification, frozen, state


def test_bundle_valid_and_modified_freeze_rejected(bundle):
    path, spec, frozen, state = bundle
    model, _ = GeneralizationDualEncoder.from_bundle(path)
    assert len(model.residuals) == 1
    dump(Path(spec['frozen_recipe']['path']), dict(frozen, frozen_before_held_out_evaluation=False))
    with pytest.raises(ValueError, match='hash mismatch'):
        GeneralizationDualEncoder.from_bundle(path)


def test_bundle_cross_split_features_rejected(bundle):
    path, spec, frozen, state = bundle
    spec['feature_manifest_sha256'] = 'another-training-split'
    dump(path, spec)
    with pytest.raises(ValueError, match='feature manifests disagree'):
        GeneralizationDualEncoder.from_bundle(path)


def test_same_weights_at_unfrozen_path_rejected(bundle, tmp_path):
    path, spec, frozen, state = bundle
    copy = tmp_path / 'not_selected.pt'
    copy.write_bytes(Path(spec['residuals'][0]['path']).read_bytes())
    with pytest.raises(ValueError, match='not in the frozen recipe'):
        validate_frozen_residual(copy, state, frozen, 'training-manifest')


def test_residual_cross_split_features_rejected(bundle):
    path, spec, frozen, state = bundle
    with pytest.raises(ValueError, match='feature manifests disagree'):
        validate_frozen_residual(spec['residuals'][0]['path'], state, frozen, 'other-split')


@pytest.fixture
def receipt(tmp_path):
    paths = {}
    for key in ('base', 'catalog', 'protein_means', 'reaction_features', 'checkpoint'):
        path = tmp_path / (key + '.bin')
        path.write_bytes(key.encode())
        paths[key] = path
    args = argparse.Namespace(**{k: v for k, v in paths.items() if k != 'checkpoint'}, input_receipt=None)
    value = dict(schema='generalization_feature_bundle_receipt_v1',
                 inputs={k: dict(path=str(v), sha256=sha256(v)) for k, v in paths.items() if k != 'checkpoint'},
                 checkpoint=dict(path=str(paths['checkpoint']), sha256=sha256(paths['checkpoint'])),
                 freeze_sha256='frozen', source_receipts=[])
    dump(tmp_path / 'feature_bundle_receipt.json', value)
    spec = dict(base_checkpoint=value['checkpoint'], frozen_recipe={'sha256': 'frozen'})
    return args, spec, value


def test_valid_feature_receipt(receipt):
    args, spec, value = receipt
    assert validate_input_receipt(args, spec)['sha256']


@pytest.mark.parametrize('key', ['base', 'catalog', 'protein_means', 'reaction_features'])
def test_receipt_rejects_replaced_arrays_and_ids(receipt, key):
    args, spec, value = receipt
    getattr(args, key).write_bytes(b'reordered-or-replaced-same-size')
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_input_receipt(args, spec)


def test_receipt_rejects_wrong_checkpoint(receipt):
    args, spec, value = receipt
    spec['base_checkpoint'] = {'path': str(args.base)}
    with pytest.raises(ValueError, match='base checkpoints disagree'):
        validate_input_receipt(args, spec)


def test_builder_rejects_shuffled_protein_ids_before_fitting(tmp_path):
    feature = tmp_path / 'features'
    feature.mkdir()
    checkpoint = tmp_path / 'parent.ckpt'
    checkpoint.write_bytes(b'checkpoint')
    compact = tmp_path / 'compact.json'
    compact_record = dump(compact, dict(signature=dict(inputs=dict(checkpoint=dict(path=str(checkpoint),
        size=checkpoint.stat().st_size, mtime_ns=checkpoint.stat().st_mtime_ns)))))
    manifest = dict(test_used=False, inputs=dict(compact_manifest=compact_record))
    dump(feature / 'manifest.json', manifest)
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    dump(feature / 'complete.json', dict(manifest_signature=signature))
    dump(feature / 'catalog.json', dict(proteins=['p1', 'p2'], reactions=['r1']))
    np.savez(feature / 'pairs.npz', train=np.array([[0, 0], [0, 1]]))
    with h5py.File(feature / 'protein_mean.h5', 'w') as f:
        f.create_dataset('ids', data=['p2', 'p1'], dtype=h5py.string_dtype())
        f.create_dataset('complete', data=[True, True])
        f.create_dataset('vectors', data=np.ones((2, 4), dtype=np.float32))
        f.attrs['manifest_signature'] = signature
    settings = {k: anchors()[k] for k in ('modalities', 'enzyme_temperature', 'reaction_temperature', 'protein_neighbors', 'reaction_neighbors')}
    freeze = tmp_path / 'freeze.json'
    dump(freeze, dict(frozen_before_held_out_evaluation=True, anchors=settings, models={}))
    args = argparse.Namespace(features=feature, output=tmp_path/'output', freeze=freeze,
                              residual=[], alpha=.25, device='cpu', **settings)
    with pytest.raises(ValueError, match='catalog ID order disagree'):
        build(args)
