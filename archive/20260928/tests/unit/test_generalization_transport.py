import pytest
import torch

from horizyn.generalization_transport import transport_embeddings, refine_embeddings, refine_phase2


def test_training_bank_only_and_batch_invariance():
    generator = torch.Generator().manual_seed(42)
    keys = torch.nn.functional.normalize(torch.randn(17, 7, generator=generator), dim=1)
    values = torch.nn.functional.normalize(torch.randn(17, 5, generator=generator), dim=1)
    query = torch.nn.functional.normalize(torch.randn(8, 7, generator=generator), dim=1)
    full = transport_embeddings(query, keys, values, neighbors=4, batch_size=8)
    singles = transport_embeddings(query, keys, values, neighbors=4, batch_size=1)
    perm = torch.tensor([6, 2, 0, 7])
    subset = transport_embeddings(query[perm], keys, values, neighbors=4, batch_size=2)
    torch.testing.assert_close(full, singles, rtol=0, atol=0)
    torch.testing.assert_close(full[perm], subset, rtol=0, atol=0)
    torch.testing.assert_close(full.norm(dim=1), torch.ones(8))


def test_ties_use_training_index_and_identity_is_exact():
    keys = torch.ones(4, 2)
    values = torch.eye(4)
    result = transport_embeddings(torch.ones(1, 2), keys, values, neighbors=1)
    torch.testing.assert_close(result, values[:1], rtol=0, atol=0)
    base = torch.randn(1, 4)
    assert refine_embeddings(base, result, 0) is base


def test_phase2_preserves_semantic_coordinates():
    base = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
    transport = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
    semantic = torch.randn(3, 7)
    original = torch.cat((base * (.75 ** .5), semantic), dim=1)
    updated = refine_phase2(original, transport, .25, alpha=.25, dense_dimension=4)
    torch.testing.assert_close(updated[:, 4:], semantic, rtol=0, atol=0)
    torch.testing.assert_close(updated[:, :4].norm(dim=1), torch.full((3,), .75 ** .5))
    assert refine_phase2(original, transport, 0, alpha=.25, dense_dimension=4) is original


def test_invalid_bank_rejected():
    with pytest.raises(ValueError):
        transport_embeddings(torch.ones(2, 3), torch.ones(4, 3), torch.ones(3, 5))
    with pytest.raises(ValueError):
        transport_embeddings(torch.ones(2, 3), torch.full((4, 3), float('nan')), torch.ones(4, 5))
