import torch

from horizyn.model import ResidualEnzymePrototypeHead
from horizyn.screen_scoring import ScreenScorer, export_scoring_head
from types import SimpleNamespace


def test_cached_prototypes_match_direct_trained_head(tmp_path):
    torch.manual_seed(12)
    q, e = torch.randn(7, 16), torch.randn(31, 16)
    head = ResidualEnzymePrototypeHead(16, prototype_count=4, bottleneck_dim=8).eval()
    with torch.no_grad():
        head.prior_projection.weight.normal_()
        head.residual_gate_logits.fill_(0.0)
    spec = export_scoring_head(SimpleNamespace(enzyme_prototype_head=head), tmp_path)
    scorer = ScreenScorer.from_export(e, tmp_path, {"scoring": spec})
    with torch.inference_mode():
        torch.testing.assert_close(scorer(q), head.score(q, e))
        torch.testing.assert_close(ScreenScorer(e, head=head, block_size=5)(q), head.score(q, e))


def test_legacy_cosine_unchanged():
    q, e = torch.randn(7, 16), torch.randn(31, 16)
    expected = torch.nn.functional.normalize(q, dim=1) @ torch.nn.functional.normalize(e, dim=1).T
    torch.testing.assert_close(ScreenScorer(e)(q), expected, atol=0, rtol=0)


def test_refined_scores_match_training_geometry():
    from horizyn.generalization_residual import FrozenGeometryResidual
    torch.manual_seed(24)
    model = FrozenGeometryResidual(16, 32).eval()
    with torch.no_grad():
        model.enzyme[-1].weight.normal_(std=0.1)
        model.reaction[-1].weight.normal_(std=0.1)
        q = model.encode_reactions(torch.randn(7, 16))
        e = model.encode_enzymes(torch.randn(31, 16))
        torch.testing.assert_close(ScreenScorer(e, already_normalized=True)(q), q @ e.T, atol=0, rtol=0)
