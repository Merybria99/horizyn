import pytest
import torch

from horizyn.semantic_smooth import reaction_responses, stable_neighbor_order
from horizyn.semantic_anchors import stable_topk


def test_nested_neighbor_prefixes_obey_global_index_ties():
    scores = torch.tensor([[.8, .9, .8, .8, .9, .1]])
    values, indices = stable_topk(scores, 5)
    values, indices = stable_neighbor_order(values, indices)
    assert indices.tolist() == [[1, 4, 0, 2, 3]]
    for k in [1, 2, 3, 4, 5]:
        _, expected = stable_topk(scores, k)
        assert set(indices[0, :k].tolist()) == set(expected[0].tolist())


@pytest.mark.parametrize('kernel', ['exponential', 'shifted_cosine', 'centered_cosine'])
def test_response_is_independent_of_other_inference_queries(kernel):
    torch.manual_seed(31)
    anchors = torch.nn.functional.normalize(torch.randn(11, 4), dim=1)
    queries = torch.nn.functional.normalize(torch.randn(7, 4), dim=1)
    whole = reaction_responses(queries, anchors, kernel)
    subsets = torch.cat([reaction_responses(q[None], anchors, kernel) for q in queries])
    torch.testing.assert_close(whole, subsets, rtol=0, atol=0)
    torch.testing.assert_close(whole.norm(dim=1), torch.ones(7))
    if kernel != 'centered_cosine':
        assert (whole > 0).all()


def test_dense_exponential_resolves_disjoint_sparse_support():
    anchors = torch.eye(3)
    query = anchors[0:1]
    enzyme = anchors[2:3]
    sparse = reaction_responses(query, anchors, neighbors=1)
    dense = reaction_responses(query, anchors, neighbors=None)
    assert (sparse @ enzyme.T).item() == 0
    assert (dense @ enzyme.T).item() > 0


def test_centered_cosine_matches_explicit_training_centering():
    torch.manual_seed(14)
    anchors = torch.nn.functional.normalize(torch.randn(9, 4), dim=1)
    queries = torch.nn.functional.normalize(torch.randn(5, 4), dim=1)
    center = anchors.double().mean(0)
    expected = (queries.double() - center) @ (anchors.double() - center).T
    expected = torch.nn.functional.normalize(expected, dim=1).float()
    torch.testing.assert_close(reaction_responses(queries, anchors, 'centered_cosine'), expected, atol=2e-7, rtol=2e-6)


def test_saved_encoder_reconstructs_training_response(tmp_path):
    from horizyn.semantic_smooth import SmoothAnchorDualEncoder
    from horizyn.semantic_anchors import enzyme_anchor_features
    from horizyn.generalization_retrieval import canonical_dot, sha256
    dictionary=dict(modalities=['t5v2'],protein_center=torch.zeros(3),
        train_proteins=torch.eye(3),adjacency=torch.arange(3).reshape(3,1),
        reaction_centers={'t5v2':torch.zeros(3)},feature_manifest_sha256='train-only')
    path=tmp_path/'dictionary.pt';torch.save(dictionary,path)
    config=dict(alpha=.5,protein_neighbors=1,enzyme_temperature=.03,
        kernel='exponential',reaction_neighbors=None,reaction_temperature=.03)
    state=dict(config=config,dictionary_path=str(path),dictionary_sha256=sha256(path),
        feature_manifest_sha256='train-only',train_reactions=torch.eye(3))
    checkpoint=tmp_path/'selected.pt';torch.save(state,checkpoint)
    model,_=SmoothAnchorDualEncoder.from_checkpoint(checkpoint)
    base=torch.eye(3);means=torch.eye(3)
    enzymes=model.encode_enzymes(base,means,batch_size=1)
    queries=model.encode_reactions(base,{'t5v2':base},{'t5v2':torch.ones(3,dtype=torch.bool)})
    semantic=reaction_responses(base,torch.eye(3),temperature=.03)
    expected=.5*torch.eye(3)+.5*semantic
    torch.testing.assert_close(canonical_dot(queries,enzymes),expected,atol=6e-8,rtol=1e-7)
    torch.testing.assert_close(model.encode_enzymes(base[:1],means[:1]),enzymes[:1],atol=0,rtol=0)
