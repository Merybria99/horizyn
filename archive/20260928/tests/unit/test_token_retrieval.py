"""Checks for the new scoring path and compatibility with CIRCE's objective."""
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "VenusRXN")]
from horizyn.losses import SampledMultiPositiveInfoNCELoss
from horizyn.token_retrieval import TokenModelConfig, TokenRetrievalModel, local_pair_scores
from horizyn.token_retrieval_data import MoleculeStore, ReactionSMIProtocol, truncate_sequence


def tiny_config():
    return TokenModelConfig(residue_dim=12, width=16, heads=4, ffn_dim=32,
                            graph_layers=1, latent_layers=1, protein_tokens=4,
                            reaction_tokens=3, local_dim=8, global_dim=12,
                            molecule_chunk=2, protein_chunk=2, dropout=0.0)


def test_b1_b2_identical_initialization():
    torch.manual_seed(42)
    a = TokenRetrievalModel(tiny_config(), "B1")
    torch.manual_seed(42)
    b = TokenRetrievalModel(tiny_config(), "B2")
    assert a.state_dict().keys() == b.state_dict().keys()
    for key in a.state_dict():
        torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["B1", "B2"])
def test_full_grid_and_selected_pair_scores_agree(variant):
    model = TokenRetrievalModel(tiny_config(), variant)
    rg = F.normalize(torch.randn(5, 12), dim=-1)
    eg = F.normalize(torch.randn(7, 12), dim=-1)
    rt = F.normalize(torch.randn(5, 3, 8), dim=-1)
    et = F.normalize(torch.randn(7, 4, 8), dim=-1)
    indices = torch.cartesian_prod(torch.arange(5), torch.arange(7))
    grid = model.score_pairs((rg, rt), (eg, et), query_chunk=2, target_chunk=3)
    pairs = model.score_pairs((rg, rt), (eg, et), indices).reshape(5, 7)
    torch.testing.assert_close(grid, pairs, atol=1e-6, rtol=1e-5)


def test_molecule_enumeration_and_padding_invariance():
    store = MoleculeStore()
    model = TokenRetrievalModel(tiny_config(), "B2").eval()
    first = store.reaction("C[C@H](O)C(=O)O.[Na+]")
    second = store.reaction("[Na+].C[C@H](O)C(=O)O")
    assert len(first) == len(second) == 2
    with torch.no_grad():
        a = model.encode_reactions([first])
        b = model.encode_reactions([list(reversed(second))])
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=2e-6, rtol=1e-5)
        residues = torch.randn(1, 6, 12)
        mask = torch.ones(1, 6, dtype=torch.bool)
        c = model.protein_pool(residues, mask)
        padded = torch.cat([residues, torch.randn(1, 5, 12) * 1000], dim=1)
        d = model.protein_pool(padded, F.pad(mask, (0, 5), value=False))
        for x, y in zip(c, d):
            torch.testing.assert_close(x, y, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("variant", ["B1", "B2"])
def test_checkpointed_encoders_receive_finite_gradients(variant):
    store = MoleculeStore()
    model = TokenRetrievalModel(tiny_config(), variant).train()
    residues = [torch.randn(8, 12), torch.randn(13, 12), torch.randn(5, 12)]
    reactions = [store.reaction("CCO.O"), store.reaction("CC(=O)O")]
    reaction, enzyme = model(residues, reactions)
    pairs = torch.cartesian_prod(torch.arange(2), torch.arange(3))
    scores = model.score_pairs(reaction, enzyme, pairs).reshape(2, 3)
    positives = torch.tensor([[True, False, False], [False, True, False]])
    negative = ~positives
    loss = SampledMultiPositiveInfoNCELoss(beta=10)(1-scores, *positives.nonzero(as_tuple=True),
                                                  negative, torch.zeros_like(negative))
    loss.backward()
    for group in [model.protein_pool, model.reaction_pool, model.graph_encoder]:
        grads = [p.grad for p in group.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0


def test_allowed_pair_optimization_preserves_loss_and_gradients():
    raw = torch.randn(3, 4, requires_grad=True)
    positives = torch.zeros(3, 4, dtype=torch.bool)
    positives[[0, 0, 1, 2], [0, 2, 1, 3]] = True
    bio = torch.zeros_like(positives)
    bio[[0, 1, 2], [1, 2, 0]] = True
    random_neg = torch.zeros_like(positives)
    random_neg[2, 1] = True
    allowed = positives | bio | random_neg
    fn = SampledMultiPositiveInfoNCELoss(beta=10)
    dense_loss = fn(1 - raw, *positives.nonzero(as_tuple=True), bio, random_neg)
    dense_grad, = torch.autograd.grad(dense_loss, raw, retain_graph=True)
    sparse = torch.zeros_like(raw).index_put(tuple(allowed.nonzero().T), raw[allowed])
    sparse_loss = fn(1 - sparse, *positives.nonzero(as_tuple=True), bio, random_neg)
    sparse_grad, = torch.autograd.grad(sparse_loss, raw)
    torch.testing.assert_close(dense_loss, sparse_loss)
    torch.testing.assert_close(dense_grad, sparse_grad)
    assert (sparse_grad[~allowed] == 0).all()


def test_typed_mask_includes_pool_entries_not_only_selected_rows():
    protocol = object.__new__(ReactionSMIProtocol)
    protocol.positives = {"q": {"p1", "p2"}}
    protocol.typed_pools = {"q": {"biological": ["b"], "random": ["r"]}}
    p, b, r = protocol.masks(["q"], ["p1", "p2", "b", "r", "unknown"], "cpu")
    assert p.tolist() == [[True, True, False, False, False]]
    assert b.tolist() == [[False, False, True, False, False]]
    assert r.tolist() == [[False, False, False, True, False]]


def test_truncation_uses_existing_ends_center_positions():
    from horizyn.datasets.residue_hdf5 import truncate_residue_embeddings
    sequence = "ACDEFGHIKLMNPQRSTVWY" * 100
    expected = truncate_residue_embeddings(torch.arange(len(sequence))[:, None], 1022).flatten()
    assert truncate_sequence(sequence) == "".join(sequence[i] for i in expected)
