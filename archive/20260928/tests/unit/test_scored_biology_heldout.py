import argparse
import json

import h5py
import pytest
import torch
from torch.nn import functional as F

from horizyn.scored_biology import ScoredBiology
from scripts.test_circe_scored_biology import read_f3, score, selection
from scripts.run_circe_scored_biology import save_json, save_tensor, sha


def h5_vectors(path, ids, vectors):
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('ids', data=ids, dtype=h5py.string_dtype())
        handle.create_dataset('vectors', data=vectors.numpy())


def test_f3_explicit_id_alignment(tmp_path):
    path = tmp_path/'vectors.h5'
    h5_vectors(path, ['b_f', 'a_f'], torch.eye(2))
    assert torch.equal(read_f3(path, ['a', 'b'], reaction=True), torch.eye(2).flip(0))
    with pytest.raises(ValueError, match='IDs differ'):
        read_f3(path, ['a', 'missing'], reaction=True)


def test_heldout_scores_three_matched_methods_without_alpha_search(tmp_path):
    torch.manual_seed(42)
    train = tmp_path/'training'
    train.mkdir()
    save_json(train/'manifest.json', {'test': 'synthetic'})
    labels = {'ec': ['1', '1.1'], 'cofactor': ['NAD', 'NADP'], 'mechanism': ['oxidation', 'reduction']}
    model = ScoredBiology(4, 5, labels)
    head = train/'lambda_0.1/best.pt'
    save_tensor(head, dict(model=model.state_dict(), enzyme_dim=4, reaction_dim=5,
        labels=labels, selected_alpha=.05, manifest_sha256=sha(train/'manifest.json')))
    root = tmp_path/'test'
    root.mkdir()
    save_json(root/'manifest.json', {'source_sha256': {str(head): sha(head)}})
    save_tensor(root/'data.pt', dict(proteins=['p0', 'p1', 'p2'], train_q=['q0', 'q1'],
        pairs=[('q0', 'p0'), ('q0', 'p1'), ('q1', 'p2')]))
    p, q = F.normalize(torch.randn(3, 7), dim=-1), F.normalize(torch.randn(2, 7), dim=-1)
    save_tensor(root/'cache/enzyme/000000.pt', dict(ids=['p0', 'p1', 'p2'], features=torch.randn(3, 4), base=p))
    save_tensor(root/'cache/train/000000.pt', dict(ids=['q0', 'q1'], features=torch.randn(2, 5), base=q))
    h5_vectors(root/'enzyme_base.h5', ['p2', 'p0', 'p1'], p[[2, 0, 1]])
    h5_vectors(root/'validation_reaction_base.h5', ['q1_f', 'q0_f'], q[[1, 0]])
    score(argparse.Namespace(heads=head, run_root=root, f3_cache=root, cache_batch_size=128, device='cpu'))
    result = json.loads((root/'metrics.json').read_text())['results']
    assert len(result) == 3
    assert result['scored_biology']['alpha'] == .05
    assert result['original_f3_epoch30']['mean_mrr'] == result['frozen_feature_gate_epoch30']['mean_mrr']
    for row in result.values():
        assert row['r2e']['queries'] == 2
        assert row['e2r']['queries'] == 3
    assert json.loads((root/'complete.json').read_text())['test_selection'] is False
    save_json(train/'manifest.json', {'changed': True})
    with pytest.raises(ValueError, match='manifest disagree'):
        selection(head)
