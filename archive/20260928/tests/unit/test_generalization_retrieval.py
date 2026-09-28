import torch

from horizyn.generalization_retrieval import GeneralizationDualEncoder, canonical_dot
from horizyn.generalization_residual import FrozenGeometryResidual


def test_independent_concat_encoders_equal_weighted_cosines():
    anchors = dict(modalities=["t5v2"], enzyme_temperature=.03, reaction_temperature=.03,
                   protein_neighbors=1, reaction_neighbors=1,
                   protein_center=torch.zeros(2), reaction_centers={"t5v2": torch.zeros(2)},
                   train_proteins=torch.eye(2), train_reactions=torch.eye(2),
                   adjacency=torch.tensor([[0], [1]]))
    model = GeneralizationDualEncoder(anchors, alpha=.25)
    base_e = torch.tensor([[1., 2.], [2., 1.]])
    base_r = torch.tensor([[2., 3.]])
    e = model.encode_enzymes(base_e, torch.eye(2))
    r = model.encode_reactions(base_r, {"t5v2": torch.tensor([[0., 1.]])}, {"t5v2": torch.tensor([True])})
    cosine = torch.nn.functional.normalize(base_r, dim=-1) @ torch.nn.functional.normalize(base_e, dim=-1).T
    torch.testing.assert_close(r @ e.T, .75 * cosine + .25 * torch.tensor([[0., 1.]]))
    singleton = model.encode_enzymes(base_e[:1], torch.eye(2)[:1])
    torch.testing.assert_close(singleton, e[:1])
    torch.testing.assert_close(e.norm(dim=1), torch.ones(2))
    sparse_index = model.encode_enzyme_index(base_e, torch.eye(2), batch_size=1)
    torch.testing.assert_close(model.score_index(r, sparse_index), canonical_dot(r, e), rtol=0, atol=0)
    assert sparse_index["anchors"]._nnz() == 2


def test_residual_receives_native_base_embedding_without_extra_normalization():
    torch.manual_seed(41)
    residual = FrozenGeometryResidual(2, 8)
    torch.nn.init.normal_(residual.enzyme[-1].weight)
    torch.nn.init.normal_(residual.reaction[-1].weight)
    anchors = dict(modalities=["t5v2"], enzyme_temperature=.03, reaction_temperature=.03,
                   protein_neighbors=1, reaction_neighbors=1,
                   protein_center=torch.zeros(2), reaction_centers={"t5v2": torch.zeros(2)},
                   train_proteins=torch.eye(2), train_reactions=torch.eye(2),
                   adjacency=torch.tensor([[0], [1]]))
    bundle = GeneralizationDualEncoder(anchors, [residual], alpha=0)
    raw = torch.tensor([[7., 2.], [2., 11.]])
    expected = residual.encode_enzymes(raw)
    actual = bundle.encode_enzymes(raw, torch.eye(2))
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, residual.encode_enzymes(torch.nn.functional.normalize(raw, dim=-1)))
