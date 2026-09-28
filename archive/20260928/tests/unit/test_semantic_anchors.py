import torch

from horizyn.semantic_anchors import (fit_center, centered_unit,
    make_protein_reaction_map, nearest_training_proteins, enzyme_anchor_features,
    reaction_anchor_features, stable_topk, row_unit)


def test_centers_fit_only_requested_training_rows():
    raw = torch.tensor([[1., 0.], [0., 1.], [-100., -200.]])
    center = fit_center(raw, torch.tensor([0, 1]))
    torch.testing.assert_close(center, torch.tensor([.5, .5]))
    raw[-1] *= -20
    torch.testing.assert_close(fit_center(raw, torch.tensor([0, 1])), center)


def test_independent_encoders_and_duplicate_edges():
    r, e = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 1, 2])
    adjacency = make_protein_reaction_map(r, e, 3)
    proteins = torch.eye(3)
    values, indices = nearest_training_proteins(proteins, proteins, top_k=1)
    features = enzyme_anchor_features(values, indices, adjacency, 2, .1)
    torch.testing.assert_close(features, torch.tensor([[1., 0.], [0., 1.], [0., 1.]]))
    # Encoding a singleton gives exactly its row in a larger batch.
    single = enzyme_anchor_features(values[1:2], indices[1:2], adjacency, 2, .1)
    torch.testing.assert_close(single, features[1:2])
    queries = reaction_anchor_features(torch.eye(2), torch.eye(2), .1, top_k=1)
    torch.testing.assert_close(queries @ features.T, torch.tensor([[1., 0., 0.], [0., 1., 1.]]))


def test_anchor_cutoff_ties_choose_first_training_indices():
    values, indices = stable_topk(torch.tensor([[.5, .9, .5, .5]]), 2)
    assert set(indices[0].tolist()) == {0, 1}
    torch.testing.assert_close(values, torch.tensor([[.9, .5]]))


def test_raw_semantic_map_is_independent_of_batch_shape():
    torch.manual_seed(72)
    raw = torch.randn(731, 128)
    center = fit_center(raw, torch.arange(500))
    whole = centered_unit(raw, center)
    subset = centered_unit(raw[[1, 53, 721]], center)
    torch.testing.assert_close(subset, whole[[1, 53, 721]], rtol=0, atol=0)
    anchors = row_unit(raw[:500])
    first = nearest_training_proteins(whole[500:], anchors, top_k=5, batch_size=231)
    second = nearest_training_proteins(whole[500:], anchors, top_k=5, batch_size=17)
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    assert torch.equal(first[1], second[1])
