import pytest
import torch

from horizyn.semantic_anchors import row_unit
from horizyn.semantic_constrained import BoundedSemanticMap


def test_bound_holds_for_unseen_vectors_and_each_endpoint_is_independent():
    torch.manual_seed(7)
    model = BoundedSemanticMap(13, .4)
    with torch.no_grad():
        model.enzyme_delta.normal_()
        model.reaction_delta.normal_()
    assert max(model.project()) <= .4
    novel = row_unit(torch.randn(21, 13))
    for matrix, encode in ((model.enzyme_delta, model.encode_enzymes),
                           (model.reaction_delta, model.encode_reactions)):
        assert float((novel @ matrix).norm(dim=1).max()) <= .4
        assert torch.allclose(encode(novel), torch.cat([encode(x[None]) for x in novel]), atol=1e-7)
        assert torch.allclose(encode(novel).norm(dim=1), torch.ones(21))


def test_identity_and_invalid_bounds():
    vectors = torch.randn(5, 8)
    model = BoundedSemanticMap(8, 0)
    assert torch.equal(model.encode_enzymes(vectors), row_unit(vectors))
    for radius in (-.1, 1., 2.):
        with pytest.raises(ValueError):
            BoundedSemanticMap(8, radius)
