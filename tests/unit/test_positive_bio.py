"""Positive-only fixed-target heads: no target labels enter the forward path."""

import io

import pytest
import torch

from horizyn.enzyme_multiview import EnzymeMultiviewEncoder
from horizyn.positive_bio import (
    PositiveBiologicalReadout,
    build_positive_anchors,
    ec_prefixes,
    validate_positive_biological_labels,
)


LABELS = {"ec": ["1.1.1.1", "1.1.1.2", "2.1.-.-"], "cofactor": ["NAD", "FAD"], "mechanism": ["redox", "transfer"]}


def make_encoder(labels=None):
    torch.manual_seed(97)
    return EnzymeMultiviewEncoder(12, 16, hidden_dim=16, num_slots=4, dropout=0, biological_labels=labels)


def test_anchor_hierarchy_only_contains_specified_ancestors():
    assert ec_prefixes("1.2.-.-") == ("1", "1.2")
    labels = ["1.2.3.4", "1.2.3.5", "1.2.-.-", "2.3.4.5"]
    anchors = build_positive_anchors("ec", labels, 32)
    torch.testing.assert_close(anchors.norm(dim=-1), torch.ones(4))
    similarity = anchors @ anchors.T
    assert similarity[0, 1].item() == pytest.approx(0.75)
    assert similarity[0, 2].item() == pytest.approx(2**-0.5)
    assert similarity[0, 3].item() == 0.0
    # Unknown levels add no synthetic descendant vector.
    torch.testing.assert_close(
        build_positive_anchors("ec", ["1.2.-.-"], 16),
        build_positive_anchors("ec", ["1.2"], 16),
    )


@pytest.mark.parametrize("family", ["cofactor", "mechanism"])
def test_small_category_vocabularies_are_orthogonal(family):
    anchors = build_positive_anchors(family, ["b", "c", "a"], 8)
    torch.testing.assert_close(anchors @ anchors.T, torch.eye(3))


def test_category_simplex_when_one_more_label_than_dimensions():
    anchors = build_positive_anchors("cofactor", ["a", "b", "c"], 2)
    expected = torch.full((3, 3), -0.5)
    expected.fill_diagonal_(1.0)
    torch.testing.assert_close(anchors @ anchors.T, expected)


def test_large_vocabulary_deterministic_unit_codes_without_global_rng_changes():
    labels = [f"1.2.3.{index + 1}" for index in range(4500)]
    before = torch.random.get_rng_state().clone()
    first = build_positive_anchors("ec", labels, 32)
    assert torch.equal(before, torch.random.get_rng_state())
    second = build_positive_anchors("ec", labels[::-1], 32)
    torch.testing.assert_close(first, second.flip(0))
    torch.testing.assert_close(first.norm(dim=-1), torch.ones(len(labels)))
    assert torch.isfinite(first).all()


@pytest.mark.parametrize("labels", [
    {"ec": ["1.2.-.4"]}, {"ec": ["8.1.1.1"]}, {"ec": ["1.01.1.1"]},
    {"ec": ["1.2", "1.2.-.-"]}, {"ec": ["-.-.-.-"]}, {"ec": ["1.2.3.4.5"]},
    {"cofactor": ["NAD", "NAD"]}, {"mechanism": []}, {"activity": ["x"]},
    {"cofactor": [" NAD"]}, {"ec": "1.2.3.4"}, {"cofactor": [3]},
])
def test_invalid_biological_labels_rejected(labels):
    with pytest.raises(ValueError):
        validate_positive_biological_labels(labels)


def test_column_order_preserved_and_no_label_modules_have_legacy_state_keys():
    assert validate_positive_biological_labels(LABELS) == LABELS
    plain = make_encoder()
    assert not any("biological" in name for name in plain.state_dict())
    assert len(plain.biological_readouts) == 0
    clone = make_encoder({})
    clone.load_state_dict(plain.state_dict(), strict=True)


def test_supervision_does_not_change_initial_retrieval_or_global_rng():
    plain = make_encoder().eval()
    after_plain = torch.random.get_rng_state().clone()
    supervised = make_encoder(LABELS).eval()
    assert torch.equal(after_plain, torch.random.get_rng_state())
    residues = torch.randn(3, 13, 12)
    torch.testing.assert_close(plain(residues), supervised(residues), rtol=0, atol=0)
    output, details = supervised(residues, return_details=True)
    torch.testing.assert_close(output, supervised(residues), rtol=0, atol=0)
    for family, labels in LABELS.items():
        assert details[f"biofp_alignment_{family}"].shape == (3, len(labels))
        assert details[f"biofp_shared_{family}"].shape == (3, 4, 16)
        assert details[f"biofp_embedding_{family}"].shape == (3, 16)
        assert details[f"biofp_readout_weights_{family}"].shape == (3, 4)
        torch.testing.assert_close(details[f"biofp_embedding_{family}"].norm(dim=-1), torch.ones(3))
    assert all(torch.isfinite(value).all() for value in details.values())


def test_both_losses_have_gradients_on_the_exact_same_shared_slots():
    encoder = make_encoder(LABELS)
    output, details = encoder(torch.randn(3, 13, 12), return_details=True)
    shared = details["biofp_shared_ec"]
    assert shared is details["biofp_shared_cofactor"]
    retrieval_gradient = torch.autograd.grad(output[:, 0].sum(), shared, retain_graph=True)[0]
    alignment_gradient = torch.autograd.grad(details["biofp_alignment_ec"][:, 0].mean(), shared, retain_graph=True)[0]
    assert retrieval_gradient.abs().sum() > 0
    assert alignment_gradient.abs().sum() > 0
    alignment_loss = sum(details[f"biofp_alignment_{family}"][:, 0].mean() for family in LABELS)
    alignment_loss.backward()
    for name in ("queries", "keys.weight", "values.weight", "residue_adapter.1.weight"):
        parameter = dict(encoder.named_parameters())[name]
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    for family, head in encoder.biological_readouts.items():
        assert not head.anchors.requires_grad
        assert head.projection[-1].weight.grad.abs().sum() > 0, family
        assert head.gate[-1].weight.grad.abs().sum() > 0, family


def test_positive_alignment_masking_can_make_missing_rows_exactly_neutral():
    encoder = make_encoder(LABELS)
    residues = torch.randn(3, 13, 12, requires_grad=True)
    _, details = encoder(residues, return_details=True)
    distances = details["biofp_alignment_cofactor"]
    observed = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    (distances * observed).sum().backward()
    assert residues.grad[:2].abs().sum() > 0
    assert torch.count_nonzero(residues.grad[2]) == 0


def test_padding_does_not_change_biological_embedding_and_has_zero_gradient():
    encoder = make_encoder(LABELS).eval()
    residues = torch.randn(3, 13, 12)
    mask = torch.arange(13)[None] >= torch.tensor([13, 8, 1])[:, None]
    _, first = encoder(residues, mask, return_details=True)
    residues = residues.masked_fill(mask[..., None], torch.nan).requires_grad_()
    _, second = encoder(residues, mask, return_details=True)
    loss = residues.new_tensor(0)
    for family in LABELS:
        key = f"biofp_alignment_{family}"
        torch.testing.assert_close(first[key], second[key])
        loss = loss + second[key][:, 0].sum()
    loss.backward()
    assert torch.isfinite(residues.grad).all()
    assert torch.count_nonzero(residues.grad[mask]) == 0


def test_readout_gate_floor_single_row_variance_and_checkpoint_portability():
    encoder = make_encoder(LABELS).eval()
    residues = torch.randn(1, 13, 12)
    with torch.no_grad():
        for readout in encoder.biological_readouts.values():
            readout.gate[-1].weight.fill_(100)
    expected, first = encoder(residues, return_details=True)
    memory = io.BytesIO()
    torch.save(encoder.state_dict(), memory)
    memory.seek(0)
    restored = make_encoder(LABELS).eval()
    restored.load_state_dict(torch.load(memory, weights_only=True))
    actual, second = restored(residues, return_details=True)
    torch.testing.assert_close(expected, actual)
    for family in LABELS:
        torch.testing.assert_close(first[f"biofp_alignment_{family}"], second[f"biofp_alignment_{family}"])
        assert first[f"enzyme_multiview_bio_{family}_feature_variance"].item() == 0
        assert torch.all(first[f"biofp_readout_weights_{family}"] >= 0.05 / 4)
        assert f"biological_readouts.{family}.anchors" in restored.state_dict()


def test_readout_bfloat16_autocast_stays_finite():
    head = PositiveBiologicalReadout(16, "ec", LABELS["ec"])
    slots = torch.randn(2, 4, 16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        distance, embedding, weights = head(slots)
        loss = distance[:, 0].sum()
    loss.backward()
    assert torch.isfinite(distance).all()
    assert torch.isfinite(embedding).all()
    assert torch.isfinite(weights).all()
    assert torch.isfinite(slots.grad).all()
