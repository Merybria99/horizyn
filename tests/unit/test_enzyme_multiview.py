"""CPU tests for the sequence-only enzyme multi-view encoder."""

import pytest
import torch

from horizyn.enzyme_multiview import (
    EnzymeMultiviewEncoder,
    validate_enzyme_multiview_config,
)


def make_encoder(**kwargs):
    torch.manual_seed(17)
    return EnzymeMultiviewEncoder(12, 16, hidden_dim=8, num_slots=4, dropout=0.0, **kwargs)


def inputs():
    torch.manual_seed(29)
    residues = torch.randn(3, 21, 12)
    mask = torch.arange(21)[None] >= torch.tensor([21, 14, 1])[:, None]
    prior = torch.randn(3, 21)
    return residues, mask, prior


def test_views_weights_output_and_initialization():
    encoder = make_encoder()
    residues, mask, prior = inputs()
    output, details = encoder(residues, mask, prior, return_details=True)
    assert output.shape == (3, 16)
    torch.testing.assert_close(output.norm(dim=-1), torch.ones(3))
    assert details["enzyme_multiview_attention"].shape == (3, 4, 21)
    assert details["enzyme_multiview_gate_weights"].shape == (3, 6, 8)
    torch.testing.assert_close(details["enzyme_multiview_gate_weights"], torch.full((3, 6, 8), 1 / 6))
    torch.testing.assert_close(details["enzyme_multiview_attention"].sum(-1), torch.ones(3, 4))
    torch.testing.assert_close(details["weights"].sum(-1), torch.ones(3))
    torch.testing.assert_close(encoder.queries @ encoder.queries.T, torch.eye(4), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(encoder.residual_scale, torch.tensor(0.1))
    assert torch.all(details["enzyme_multiview_global_cosine"] > 0.99)
    for key, value in details.items():
        assert torch.isfinite(value).all(), key


def test_prior_only_changes_separate_site_view_not_learned_attention():
    encoder = make_encoder().eval()
    residues, mask, prior = inputs()
    _, first = encoder(residues, mask, prior, return_details=True)
    _, second = encoder(residues, mask, -prior * 10, return_details=True)
    torch.testing.assert_close(first["enzyme_multiview_raw_attention"], second["enzyme_multiview_raw_attention"])
    assert not torch.allclose(first["weights"], second["weights"])


def test_stronger_initial_fusion_is_configured_without_changing_other_parameters():
    baseline = make_encoder().eval()
    stronger = make_encoder(initial_residual_scale=0.3).eval()
    torch.testing.assert_close(stronger.residual_scale, torch.tensor(0.3))
    for key, value in baseline.state_dict().items():
        if key != 'raw_residual_scale':
            torch.testing.assert_close(value, stronger.state_dict()[key], rtol=0, atol=0)
    residues, mask, prior = inputs()
    first = baseline(residues, mask, prior)
    second = stronger(residues, mask, prior)
    assert not torch.allclose(first, second)
    second[:, 0].sum().backward()
    assert torch.isfinite(stronger.raw_residual_scale.grad)
    assert stronger.fused_output[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('scale', [0., .5, -1., float('nan')])
def test_invalid_initial_fusion_scale_is_rejected(scale):
    with pytest.raises(ValueError, match='initial_residual_scale'):
        make_encoder(initial_residual_scale=scale)


def test_uniform_heads_are_reported_as_redundant_not_artificially_diverse():
    encoder = make_encoder()
    with torch.no_grad():
        encoder.queries.zero_()
    residues, mask, prior = inputs()
    _, details = encoder(residues, mask, prior, return_details=True)
    torch.testing.assert_close(details["enzyme_multiview_head_similarity_raw"][:2], torch.ones(2))
    assert torch.count_nonzero(details["enzyme_multiview_head_similarity_eligible_fraction"]) == 0
    torch.testing.assert_close(details["enzyme_multiview_effective_support"], (~mask).sum(-1).float())
    torch.testing.assert_close(details["enzyme_multiview_normalized_entropy"][:2], torch.ones(2))
    torch.testing.assert_close(details["enzyme_multiview_effective_normalized_entropy"][:2], torch.ones(2))
    torch.testing.assert_close(details["enzyme_multiview_effective_max_weight"], (~mask).sum(-1).float().reciprocal())


def test_padding_nan_is_masked_before_every_projection_and_has_zero_gradient():
    encoder = make_encoder()
    residues, mask, prior = inputs()
    reference, _ = encoder(residues, mask, prior, return_details=True)
    residues = residues.masked_fill(mask[..., None], torch.nan).requires_grad_()
    prior = prior.masked_fill(mask, torch.nan)
    output, details = encoder(residues, mask, prior, return_details=True)
    torch.testing.assert_close(output, reference)
    loss = output[:, 0].sum() + details["enzyme_multiview_entropy_loss"].mean() + details["enzyme_multiview_diversity_loss"].mean()
    loss.backward()
    assert torch.isfinite(residues.grad).all()
    assert torch.count_nonzero(residues.grad[mask]) == 0
    assert torch.count_nonzero(details["enzyme_multiview_attention"].masked_select(mask[:, None])) == 0
    assert torch.count_nonzero(details["weights"][mask]) == 0
    for parameter in encoder.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_residue_permutation_and_extra_padding_do_not_change_embedding():
    encoder = make_encoder().eval()
    residues, mask, prior = inputs()
    expected = encoder(residues, mask, prior)
    order = torch.randperm(residues.shape[1])
    actual = encoder(residues[:, order], mask[:, order], prior[:, order])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    padded = torch.cat([residues, torch.full((3, 7, 12), torch.nan)], 1)
    padded_mask = torch.cat([mask, torch.ones(3, 7, dtype=torch.bool)], 1)
    padded_prior = torch.cat([prior, torch.full((3, 7), torch.nan)], 1)
    torch.testing.assert_close(encoder(padded, padded_mask, padded_prior), expected, atol=1e-6, rtol=1e-5)


def test_uniform_mixture_and_gate_floor_survive_extreme_preferences():
    encoder = make_encoder(uniform_mix=0.1, gate_floor=0.12)
    with torch.no_grad():
        encoder.gate[-1].bias.fill_(-100)
        encoder.gate[-1].bias[:8].fill_(100)
        encoder.raw_logit_scale.fill_(100)
        encoder.raw_residual_scale.fill_(100)
    residues, mask, prior = inputs()
    prior[:, 0] = 1e4
    _, details = encoder(residues, mask, prior, return_details=True)
    floor = 0.1 / (~mask).sum(-1).float()
    assert torch.all(details["enzyme_multiview_attention"] >= floor[:, None, None] * (~mask[:, None]))
    assert torch.all(details["weights"] >= floor[:, None] * (~mask))
    assert torch.all(details["enzyme_multiview_gate_weights"] >= 0.12 / 6)
    assert encoder.logit_scale <= encoder.max_logit_scale
    assert encoder.residual_scale <= 0.5


def test_collapsed_heads_receive_penalties_without_forcing_uniformity():
    encoder = make_encoder()
    weights = torch.zeros(2, 4, 32)
    weights[:, :, 0] = 1
    weights.requires_grad_()
    valid = torch.ones(2, 32, dtype=torch.bool)
    entropy_loss, diversity_loss, _, similarity = encoder._attention_safeguards(weights, valid)
    torch.testing.assert_close(entropy_loss, torch.full((2,), torch.tensor(4.0).log().square()))
    torch.testing.assert_close(diversity_loss, torch.full((2,), 0.01), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(similarity, torch.ones(2))
    (entropy_loss + diversity_loss).sum().backward()
    assert torch.isfinite(weights.grad).all()
    uniform = torch.full((2, 4, 32), 1 / 32)
    entropy_loss, diversity_loss, _, _ = encoder._attention_safeguards(uniform, valid)
    assert torch.count_nonzero(entropy_loss) == 0
    assert torch.count_nonzero(diversity_loss) == 0


def test_short_and_single_residue_proteins_have_compatible_regularizers():
    encoder = make_encoder()
    weights = torch.full((1, 4, 2), 0.5)
    weights[0, 0] = torch.tensor([0.75, 0.25])
    entropy_loss, _, _, _ = encoder._attention_safeguards(weights, torch.ones(1, 2, dtype=torch.bool))
    assert entropy_loss.item() == 0.0
    residues = torch.randn(1, 1, 12)
    output, details = encoder(residues, return_details=True)
    assert torch.isfinite(output).all()
    assert details["enzyme_multiview_entropy_loss"].item() == 0
    assert details["enzyme_multiview_diversity_loss"].item() == 0
    assert details["enzyme_multiview_normalized_entropy"].item() == 0


def test_distinct_heads_do_not_receive_overlap_penalty():
    encoder = make_encoder()
    weights = torch.zeros(1, 4, 32)
    for head in range(4):
        weights[:, head, head * 4 : head * 4 + 4] = 0.25
    entropy_loss, diversity_loss, _, _ = encoder._attention_safeguards(weights, torch.ones(1, 32, dtype=torch.bool))
    assert torch.count_nonzero(entropy_loss) == 0
    assert torch.count_nonzero(diversity_loss) == 0


def test_one_slot_has_no_head_diversity_penalty():
    encoder = EnzymeMultiviewEncoder(12, 16, hidden_dim=8, num_slots=1, dropout=0.0)
    output, details = encoder(torch.randn(2, 9, 12), return_details=True)
    assert output.shape == (2, 16)
    assert torch.count_nonzero(details["enzyme_multiview_diversity_loss"]) == 0


def test_training_gradients_reach_all_views_and_gate_after_warm_start():
    encoder = make_encoder()
    residues, mask, prior = inputs()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        output, details = encoder(residues, mask, prior, return_details=True)
        loss = -output[:, 0].mean() + 0.01 * details["enzyme_multiview_entropy_loss"].mean() + 0.001 * details["enzyme_multiview_diversity_loss"].mean()
        loss.backward()
        optimizer.step()
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_checkpoint_roundtrip_and_bfloat16_cpu_autocast():
    encoder = make_encoder().eval()
    residues, mask, prior = inputs()
    clone = make_encoder().eval()
    clone.load_state_dict(encoder.state_dict(), strict=True)
    torch.testing.assert_close(clone(residues, mask, prior), encoder(residues, mask, prior))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output, details = clone(residues, mask, prior, return_details=True)
        loss = output.sum() + details["enzyme_multiview_diversity_loss"].mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in clone.parameters())


@pytest.mark.parametrize("bad", [
    {"num_slots": -1}, {"num_slots": True}, {"num_slots": 0.5},
    {"hidden_dim": 0}, {"hidden_dim": True}, {"num_slots": 9, "hidden_dim": 8},
    {"dropout": 1.0}, {"uniform_mix": -0.1}, {"gate_floor": 1.0},
    {"min_effective_residues": 0}, {"diversity_margin": 1.1},
    {"max_logit_scale": 0}, {"max_logit_scale": float("nan")},
    {"unknown": True}, {"dropout": True},
])
def test_invalid_config_is_rejected(bad):
    with pytest.raises(ValueError):
        validate_enzyme_multiview_config(bad)


def test_zero_slots_is_trainable_global_and_sleec_fusion():
    encoder = EnzymeMultiviewEncoder(12, 16, hidden_dim=8, num_slots=0, dropout=0.0)
    residues, mask, prior = inputs()
    residues = residues.masked_fill(mask[..., None], torch.nan)
    output, details = encoder(residues, mask, prior, return_details=True)
    assert encoder.view_names == ("global", "sleec")
    assert encoder.residue_adapter is encoder.keys is encoder.values is None
    assert encoder.queries.shape == (0, 8) and not encoder.queries.requires_grad
    assert details["enzyme_multiview_attention"].shape == (3, 0, 21)
    assert details["enzyme_multiview_gate_weights"].shape == (3, 2, 8)
    for key, value in details.items():
        assert torch.isfinite(value).all(), key
    for name in ("entropy_loss", "diversity_loss"):
        assert torch.count_nonzero(details[f"enzyme_multiview_{name}"]) == 0
    clean = residues.masked_fill(mask[..., None], 0.)
    mean = clean.sum(1) / (~mask).sum(1)[:, None]
    site = (clean * details["weights"][..., None]).sum(1)
    normalize = torch.nn.functional.normalize
    views = torch.stack([normalize(projection(value), dim=-1)
                         for projection, value in zip(encoder.view_projections, (mean, site))], 1)
    fused = (views * details["enzyme_multiview_gate_weights"]).sum(1)
    expected = normalize(normalize(encoder.global_output(mean), dim=-1)
                         + encoder.residual_scale * normalize(encoder.fused_output(fused), dim=-1), dim=-1)
    torch.testing.assert_close(output, expected)
    assert not torch.allclose(output, encoder(residues, mask, -prior * 10))
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = -encoder(residues, mask, prior)[:, 0].mean()
        loss.backward()
        optimizer.step()
    for name, parameter in encoder.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize("case", ["empty", "all_padding", "bad_mask", "bad_prior", "bad_dimension"])
def test_invalid_inputs_rejected(case):
    encoder = make_encoder()
    residues, mask, prior = inputs()
    if case == "empty":
        residues, mask, prior = residues[:, :0], mask[:, :0], prior[:, :0]
    elif case == "all_padding":
        mask[0] = True
    elif case == "bad_mask":
        mask = mask[:, :-1]
    elif case == "bad_prior":
        prior = prior[:, :-1]
    else:
        residues = residues[:, :, :-1]
    with pytest.raises(ValueError):
        encoder(residues, mask, prior)
