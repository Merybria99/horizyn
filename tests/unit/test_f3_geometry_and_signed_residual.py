import torch

from horizyn.model import MultimodalReactionAttentionEncoder
from horizyn.reaction_geometry import reaction_geometry_loss


def encoder(mode):
    return MultimodalReactionAttentionEncoder(
        input_dim=8, output_dim=8, num_layers=0, widths=[],
        reaction_model_dim=6, unimol_dim=4, chienn_dim=3,
        reaction_pooling="mean", side_composition=mode,
    )


def inputs():
    return dict(reaction_embedding=torch.randn(5, 6),
                reactant_embeddings=torch.randn(5, 2, 4),
                product_embeddings=torch.randn(5, 3, 4),
                reactant_chirality_embeddings=torch.randn(5, 2, 3),
                product_chirality_embeddings=torch.randn(5, 3, 3))


def test_signed_residual_starts_at_base_and_learns_product_information():
    torch.manual_seed(19)
    base, compact = encoder("molecule_set"), encoder("signed_residual")
    missing, unexpected = compact.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all("signed_residual" in key for key in missing)
    batch = inputs()
    torch.testing.assert_close(base(**batch), compact(**batch), rtol=0, atol=0)
    compact(**batch)[:, 0].sum().backward()
    for branch in (compact.unimol_signed_residual, compact.chienn_signed_residual):
        assert branch[-1].weight.grad.abs().sum() > 0
        with torch.no_grad():
            branch[-1].weight.add_(-0.1 * branch[-1].weight.grad)
    changed = dict(batch, product_embeddings=batch["product_embeddings"] + 1)
    assert not torch.allclose(compact(**batch), compact(**changed))
    assert sum(p.numel() for p in compact.parameters()) < sum(p.numel() for p in encoder("directional_delta").parameters())


def test_signed_residual_checkpoint_roundtrip():
    model = encoder("signed_residual")
    with torch.no_grad():
        model.unimol_signed_residual[-1].weight.normal_()
    restored = encoder("signed_residual")
    restored.load_state_dict(model.state_dict())
    batch = inputs()
    torch.testing.assert_close(model(**batch), restored(**batch), rtol=0, atol=0)


def test_geometry_is_rotation_invariant_and_excludes_missing_descriptors():
    torch.manual_seed(7)
    teacher = torch.randn(6, 8)
    rotation, _ = torch.linalg.qr(torch.randn(8, 8))
    assert reaction_geometry_loss(teacher @ rotation, teacher).abs() < 1e-6
    student = torch.randn(6, 8, requires_grad=True)
    available = torch.tensor([True, True, True, True, False, False])
    teacher[4] = float("nan")
    teacher[5] = 0
    loss = reaction_geometry_loss(student, teacher, available)
    torch.testing.assert_close(loss, reaction_geometry_loss(student[:4], teacher[:4]))
    loss.backward()
    assert torch.isfinite(student.grad).all()
    assert student.grad[:4].abs().sum() > 0
    assert student.grad[4:].abs().sum() == 0


def test_geometry_teacher_detached_and_small_batches_safe():
    student = torch.randn(6, 8, requires_grad=True)
    teacher = torch.randn(6, 8, requires_grad=True)
    reaction_geometry_loss(student, teacher).backward()
    assert teacher.grad is None
    assert reaction_geometry_loss(student[:2], teacher[:2]).item() == 0


def test_geometry_gradient_improves_teacher_neighborhood_match():
    torch.manual_seed(17)
    student = torch.randn(8, 4, requires_grad=True)
    teacher = torch.randn(8, 4)
    loss = reaction_geometry_loss(student, teacher)
    loss.backward()
    assert reaction_geometry_loss(student - .05 * student.grad, teacher) < loss
