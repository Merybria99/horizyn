import torch
from horizyn.biological_geometry import category_pair_distance


def test_exact_pairs_and_gradient():
    torch.manual_seed(4)
    a = torch.randn(5, 7, requires_grad=True)
    rows = torch.tensor([0, 1, 2, 1, 3, 4])
    groups = torch.tensor([0, 0, 0, 1, 1, 2])
    confidence = torch.tensor([1., .4, .8, .7, 1., 1.])
    cw = torch.tensor([.5, 1., 1.])
    loss = category_pair_distance(a, rows, groups, confidence, cw)
    unit = torch.nn.functional.normalize(a, dim=-1)
    expected = a.sum() * 0
    for group in (0, 1):
        ix = torch.where(groups == group)[0]
        terms = [confidence[i] * confidence[j] * (1 - unit[rows[i]] @ unit[rows[j]])
                 for i in ix for j in ix if i != j]
        expected = expected + cw[group] * torch.stack(terms).mean() / cw[:2].sum()
    assert torch.allclose(loss, expected, atol=1e-6)
    g1 = torch.autograd.grad(loss, a, retain_graph=True)[0]
    g2 = torch.autograd.grad(expected, a)[0]
    assert torch.allclose(g1, g2, atol=1e-6)
    # Unannotated/singleton-only rows have no gradients.
    assert torch.equal(g1[4], torch.zeros_like(g1[4]))


def test_confidence_not_normalized_away():
    x = torch.eye(2)
    rows = torch.arange(2)
    group = torch.zeros(2, dtype=torch.long)
    w = torch.ones(1)
    high = category_pair_distance(x, rows, group, torch.ones(2), w)
    low = category_pair_distance(x, rows, group, torch.full((2,), .5), w)
    assert torch.allclose(low, high * .25)


def test_missing_is_neutral():
    x = torch.randn(3, 4, requires_grad=True)
    loss = category_pair_distance(x, torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long), torch.empty(0), torch.empty(0))
    loss.backward()
    assert loss.item() == 0 and torch.equal(x.grad, torch.zeros_like(x))


def test_batch_lookup_matches_full_loss_and_rejects_nontraining(tmp_path):
    import json
    import pytest
    from horizyn.biological_geometry import BatchBiologicalGeometryLoss, BiologicalGeometryLoss
    block = dict(rows=[0, 1], groups=[0, 0], confidence=[1., .7], category_weight=[1.])
    payload = dict(reaction_ids=['r1_f', 'r2_f'], enzyme_ids=['e1', 'e2'],
        endpoints={e: {f: block for f in BiologicalGeometryLoss.families} for e in ['reaction', 'enzyme']})
    path = tmp_path / 'labels.json'; path.write_text(json.dumps(payload))
    batch = BatchBiologicalGeometryLoss(path)
    full = BiologicalGeometryLoss(payload, payload['reaction_ids'], payload['enzyme_ids'], 'cpu')
    r, e = torch.randn(2, 4, requires_grad=True), torch.randn(2, 4, requires_grad=True)
    actual, _ = batch(r, e, payload['reaction_ids'], payload['enzyme_ids'])
    expected, _ = full(r, e, dict.fromkeys(BiologicalGeometryLoss.families, 1.))
    assert torch.allclose(actual, expected, atol=1e-6)
    actual.backward()
    assert r.grad.norm() > 0 and e.grad.norm() > 0
    with pytest.raises(ValueError, match='Non-training'):
        batch(r, e, ['heldout', 'r2_f'], payload['enzyme_ids'])


def test_forward_suffix_alias_only(tmp_path):
    import json
    import pytest
    from horizyn.biological_geometry import BatchBiologicalGeometryLoss
    block = dict(rows=[0, 1], groups=[0, 0], confidence=[1., 1.], category_weight=[1.])
    payload = dict(reaction_ids=['r1', 'r2'], enzyme_ids=['e1', 'e2'],
        endpoints={e: {f: block for f in ('ec', 'cofactor', 'mechanism')} for e in ['reaction', 'enzyme']})
    path = tmp_path / 'labels.json'; path.write_text(json.dumps(payload))
    loss = BatchBiologicalGeometryLoss(path)
    r, e = torch.randn(2, 4), torch.randn(2, 4)
    assert torch.allclose(loss(r, e, ['r1_f', 'r2_f'], ['e1', 'e2'])[0],
                          loss(r, e, ['r1', 'r2'], ['e1', 'e2'])[0])
    with pytest.raises(ValueError, match='Non-training'):
        loss(r, e, ['r1_r', 'r2_f'], ['e1', 'e2'])


def test_biology_reaches_existing_queries_without_a_new_head():
    from horizyn.enzyme_multiview import EnzymeMultiviewEncoder
    torch.manual_seed(11)
    model = EnzymeMultiviewEncoder(8, 6, hidden_dim=8, dropout=0.)
    state_keys = set(model.state_dict())
    output = model(torch.randn(3, 15, 8), sleec_prior=torch.randn(3, 15))
    loss = category_pair_distance(output, torch.arange(3), torch.zeros(3, dtype=torch.long),
                                  torch.ones(3), torch.ones(1))
    loss.backward()
    assert torch.isfinite(model.queries.grad).all()
    assert model.queries.grad.norm() > 0
    assert set(model.state_dict()) == state_keys
    assert not model.biological_readouts


def test_slots_pool_content_not_residue_array_positions():
    from horizyn.enzyme_multiview import EnzymeMultiviewEncoder
    torch.manual_seed(12)
    model = EnzymeMultiviewEncoder(8, 6, hidden_dim=8, dropout=0.).eval()
    residues, prior = torch.randn(2, 15, 8), torch.randn(2, 15)
    permutation = torch.randperm(15)
    with torch.no_grad():
        original = model(residues, sleec_prior=prior)
        reordered = model(residues[:, permutation], sleec_prior=prior[:, permutation])
    # ProtT5 context has already been computed. This does not claim invariance
    # to reordering amino acids before the language model processes them.
    assert torch.allclose(original, reordered, atol=1e-6)


def test_relative_geometry_matches_explicit_weighted_pairs_and_gradients():
    from horizyn.biological_geometry import category_relative_distance
    torch.manual_seed(15)
    x = torch.randn(5, 7, requires_grad=True)
    rows = torch.tensor([0, 1, 1, 2, 3])
    groups = torch.tensor([0, 0, 1, 1, 1])
    confidence = torch.tensor([1., .4, .7, .6, 1.])
    weights = torch.tensor([.5, 1.])
    unit = torch.nn.functional.normalize(x, dim=-1)
    row_confidence = {0: 1., 1: .7, 2: .6, 3: 1.}
    expected = x.sum() * 0
    for group in (0, 1):
        members = torch.where(groups == group)[0].tolist()
        outside = set(row_confidence) - {int(rows[i]) for i in members}
        inside_terms = [(confidence[i] * confidence[j], 1 - unit[rows[i]] @ unit[rows[j]])
                        for i in members for j in members if i != j]
        between_terms = [(confidence[i] * row_confidence[j], 1 - unit[rows[i]] @ unit[j])
                         for i in members for j in outside]
        within = sum(w * d for w, d in inside_terms) / sum(w for w, _ in inside_terms)
        between = sum(w * d for w, d in between_terms) / sum(w for w, _ in between_terms)
        evidence = sum(w for w, _ in inside_terms) / len(inside_terms)
        expected = expected + weights[group] * evidence * torch.relu(1.5 + within - between) / weights.sum()
    actual = category_relative_distance(x, rows, groups, confidence, weights, margin=1.5)
    assert torch.allclose(actual, expected, atol=1e-6)
    g1 = torch.autograd.grad(actual, x, retain_graph=True)[0]
    g2 = torch.autograd.grad(expected, x)[0]
    assert torch.allclose(g1, g2, atol=1e-6)
    assert torch.equal(g1[4], torch.zeros_like(g1[4]))


def test_relative_geometry_rewards_peers_and_preserves_confidence():
    from horizyn.biological_geometry import category_relative_distance
    x = torch.tensor([[1., 0.], [.9, .1], [-1., 0.]])
    rows = torch.arange(3); groups = torch.tensor([0, 0, 1]); weights = torch.ones(2)
    aligned = category_relative_distance(x, rows, groups, torch.ones(3), weights)
    wrong = category_relative_distance(x, rows, torch.tensor([0, 1, 0]), torch.ones(3), weights)
    uncertain = category_relative_distance(x, rows, torch.tensor([0, 1, 0]), torch.full((3,), .5), weights)
    assert aligned.item() == 0 and wrong.item() > 0
    assert torch.allclose(uncertain, wrong * .25)
    no_comparison = category_relative_distance(x, rows, torch.zeros(3, dtype=torch.long), torch.ones(3), torch.ones(1))
    assert no_comparison.item() == 0
