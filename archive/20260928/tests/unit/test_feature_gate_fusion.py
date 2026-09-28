"""Competitive fusion semantics, masking, and encoder compatibility."""

import math

import pytest
import torch
from torch import nn

from horizyn.config import DotDict
from horizyn.config_validation import _validate_reaction_attention
from horizyn.gated_reaction_fusion import CompetitiveReactionFusion
from horizyn.model import MultimodalReactionAttentionEncoder
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule


@pytest.fixture(autouse=True)
def limit_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def make_encoder(variant="feature_gate", **overrides):
    options = dict(
        input_dim=10,
        output_dim=12,
        widths=[24],
        num_layers=1,
        reaction_model_dim=10,
        unimol_dim=8,
        chienn_dim=6,
        reaction_pooling="mean",
        side_composition="directional_delta",
        separate_side_poolers=False,
        modality_token_layer_norm=True,
        modality_fusion=variant,
    )
    options.update(overrides)
    return MultimodalReactionAttentionEncoder(**options)


def make_inputs():
    return dict(
        reaction_embedding=torch.randn(4, 10),
        reactant_embeddings=torch.randn(4, 3, 8),
        product_embeddings=torch.randn(4, 2, 8),
        reactant_padding_mask=torch.tensor([[False, False, True]] * 4),
        product_padding_mask=torch.tensor([[False, True]] * 4),
        has_unimol2=torch.tensor([True, True, False, False]),
        reactant_chirality_embeddings=torch.randn(4, 3, 6),
        product_chirality_embeddings=torch.randn(4, 2, 6),
        reactant_chirality_padding_mask=torch.tensor([[False, False, True]] * 4),
        product_chirality_padding_mask=torch.tensor([[False, True]] * 4),
        has_chiro=torch.tensor([True, False, True, False]),
    )


def poison_missing_inputs(batch):
    poisoned = {key: value.clone() for key, value in batch.items()}
    for side in ("reactant", "product"):
        for suffix, present in (("", "has_unimol2"), ("_chirality", "has_chiro")):
            prefix = side + suffix
            values = poisoned[f"{prefix}_embeddings"]
            mask = poisoned[f"{prefix}_padding_mask"]
            values.masked_fill_(mask[..., None], float("nan"))
            values.masked_fill_(~poisoned[present][:, None, None], float("nan"))
    for name in ("chemistry", "directional"):
        vector_key = f"reaction_{name}_vector"
        if vector_key in poisoned:
            poisoned[vector_key].masked_fill_(
                ~poisoned[f"has_reaction_{name}"][:, None], float("nan")
            )
    return poisoned


@pytest.mark.parametrize("featurewise", [True, False])
def test_competitive_fusion_zero_initialization_is_masked_mean(featurewise):
    fusion = CompetitiveReactionFusion(5, 3, featurewise=featurewise)
    assert isinstance(fusion.gate[0], nn.Linear)
    assert (fusion.gate[0].in_features, fusion.gate[0].out_features) == (18, 5)
    assert isinstance(fusion.gate[1], nn.ReLU)
    assert fusion.gate[-1].out_features == (15 if featurewise else 3)
    assert fusion.gate[-1].weight.eq(0).all()
    assert fusion.gate[-1].bias.eq(0).all()
    assert not any(isinstance(module, nn.MultiheadAttention) for module in fusion.modules())

    tokens = torch.randn(4, 3, 5)
    valid = torch.tensor(
        [[True, True, True], [True, False, True], [False, True, False], [False, False, False]]
    )
    fused, details = fusion(tokens, valid, return_details=True)
    expected_weights = valid.float() / valid.sum(dim=1, keepdim=True).clamp_min(1)
    expected_weights = expected_weights[..., None].expand_as(tokens)
    torch.testing.assert_close(details["modality_feature_weights"], expected_weights)
    torch.testing.assert_close(fused, (tokens * expected_weights).sum(dim=1))
    assert details["modality_feature_logits"].shape == tokens.shape
    expected_entropy = torch.tensor([math.log(3), math.log(2), 0.0, 0.0])[:, None]
    torch.testing.assert_close(
        details["modality_feature_entropy"], expected_entropy.expand(4, 5)
    )
    no_details, _ = fusion(tokens, valid)
    torch.testing.assert_close(no_details, fused)


@pytest.mark.parametrize("featurewise", [True, False])
def test_competitive_fusion_masks_poison_and_all_missing_with_finite_gradients(featurewise):
    torch.manual_seed(11)
    fusion = CompetitiveReactionFusion(5, 3, featurewise=featurewise)
    with torch.no_grad():
        fusion.gate[-1].weight.normal_(std=0.2)
        fusion.gate[-1].bias.normal_(std=0.2)
    valid = torch.tensor(
        [[True, True, True], [True, False, True], [False, True, False], [False, False, False]]
    )
    tokens = torch.randn(4, 3, 5, requires_grad=True)
    poisoned = tokens.masked_fill(~valid[..., None], float("nan"))
    fused, details = fusion(poisoned, valid, return_details=True)
    clean_fused, _ = fusion(tokens.masked_fill(~valid[..., None], 0), valid)
    torch.testing.assert_close(fused, clean_fused)
    assert torch.isfinite(fused).all()
    assert fused[-1].eq(0).all()
    weights = details["modality_feature_weights"]
    assert weights[~valid].eq(0).all()
    torch.testing.assert_close(weights.sum(dim=1), valid.any(dim=1)[:, None].float().expand(4, 5))
    assert all(torch.isfinite(value).all() for value in details.values())
    fused.square().sum().backward()
    assert torch.isfinite(tokens.grad).all()
    assert tokens.grad[~valid].eq(0).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in fusion.parameters()
    )


def test_feature_gate_competes_per_channel_and_scalar_gate_broadcasts():
    tokens = torch.tensor([[[2.0, 8.0], [6.0, 4.0]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    feature_gate = CompetitiveReactionFusion(2, 2, featurewise=True)
    scalar_gate = CompetitiveReactionFusion(2, 2, featurewise=False)
    with torch.no_grad():
        feature_gate.gate[-1].bias.copy_(torch.tensor([math.log(3), 0.0, 0.0, math.log(3)]))
        scalar_gate.gate[-1].bias.copy_(torch.tensor([math.log(3), 0.0]))
    feature_output, feature_details = feature_gate(tokens, valid, return_details=True)
    scalar_output, scalar_details = scalar_gate(tokens, valid, return_details=True)
    torch.testing.assert_close(feature_output, torch.tensor([[3.0, 5.0]]))
    torch.testing.assert_close(scalar_output, torch.tensor([[3.0, 7.0]]))
    torch.testing.assert_close(
        feature_details["modality_feature_weights"],
        torch.tensor([[[0.75, 0.25], [0.25, 0.75]]]),
    )
    torch.testing.assert_close(
        scalar_details["modality_feature_weights"],
        torch.tensor([[[0.75, 0.75], [0.25, 0.25]]]),
    )


def test_gate_context_contains_masked_tokens_and_availability_bits():
    fusion = CompetitiveReactionFusion(2, 3)
    tokens = torch.tensor([[[1.0, 2.0], [float("nan"), float("nan")], [3.0, 4.0]]])
    valid = torch.tensor([[True, False, True]])
    contexts = []
    handle = fusion.gate[0].register_forward_pre_hook(
        lambda module, args: contexts.append(args[0].detach().clone())
    )
    try:
        fusion(tokens, valid)
    finally:
        handle.remove()
    torch.testing.assert_close(contexts[0], torch.tensor([[1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 1.0, 0.0, 1.0]]))


@pytest.mark.parametrize("pooling", ["mean", "attention"])
def test_shared_encoder_initialization_rng_and_initial_output_match_mean(pooling):
    models = {}
    following_random = {}
    for variant in ("mean", "scalar_gate", "feature_gate"):
        torch.manual_seed(23)
        models[variant] = make_encoder(variant, reaction_pooling=pooling).eval()
        following_random[variant] = torch.randn(10)
    baseline = models["mean"]
    batch = make_inputs()
    expected = baseline(**batch)
    for variant in ("scalar_gate", "feature_gate"):
        candidate = models[variant]
        torch.testing.assert_close(following_random[variant], following_random["mean"])
        for name, tensor in baseline.state_dict().items():
            torch.testing.assert_close(candidate.state_dict()[name], tensor)
        torch.testing.assert_close(candidate(**batch), expected)
        assert candidate.competitive_fusion is not None
        assert candidate.cross_modal_fusion is None


@pytest.mark.parametrize("variant", ["feature_gate", "scalar_gate"])
@pytest.mark.parametrize("pooling", ["mean", "attention"])
@pytest.mark.parametrize("composition", ["molecule_set", "directional_delta"])
def test_encoder_missing_poison_checkpoint_and_backward(tmp_path, variant, pooling, composition):
    torch.manual_seed(37)
    options = dict(
        reaction_pooling=pooling, side_composition=composition, use_reaction_model=False
    )
    model = make_encoder(variant, **options).eval()
    batch = make_inputs()
    batch.pop("reaction_embedding")
    expected, expected_details = model(**batch, return_attention=True)
    poisoned = poison_missing_inputs(batch)
    actual, details = model(**poisoned, return_attention=True)
    torch.testing.assert_close(actual, expected)
    assert actual[-1].eq(0).all()
    assert torch.isfinite(actual).all()
    assert details["modality_feature_weights"].shape == (4, 2, 10)
    assert details["modality_feature_weights"][-1].eq(0).all()
    assert not details.get("modality_weights_are_availability", False)
    torch.testing.assert_close(
        details["modality"], details["modality_feature_weights"].mean(dim=-1)
    )
    torch.testing.assert_close(
        details["modality_feature_weights"], expected_details["modality_feature_weights"]
    )
    (actual * torch.randn_like(actual)).sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert model.competitive_fusion.gate[-1].weight.grad.abs().sum() > 0
    checkpoint = tmp_path / "encoder.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = make_encoder(variant, **options).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored(**batch), expected)


@pytest.mark.parametrize("variant", ["feature_gate", "scalar_gate"])
def test_encoder_masks_optional_chemistry_and_directional_poison(variant):
    torch.manual_seed(41)
    model = make_encoder(
        variant,
        use_reaction_model=False,
        use_reaction_chemistry=True,
        reaction_chemistry_dim=5,
        use_reaction_directional=True,
        reaction_directional_dim=7,
    ).eval()
    batch = make_inputs()
    batch.pop("reaction_embedding")
    batch.update(
        reaction_chemistry_vector=torch.randn(4, 5),
        has_reaction_chemistry=torch.tensor([True, False, True, False]),
        reaction_directional_vector=torch.randn(4, 7),
        has_reaction_directional=torch.tensor([True, False, True, False]),
    )
    expected = model(**batch)
    actual = model(**poison_missing_inputs(batch))
    torch.testing.assert_close(actual, expected)
    assert actual[-1].eq(0).all()
    (actual * torch.randn_like(actual)).sum().backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


@pytest.mark.parametrize("variant", ["feature_gate", "scalar_gate"])
@pytest.mark.parametrize("pooling", ["mean", "attention"])
@pytest.mark.parametrize("molecule_count", [0, 2])
def test_encoder_empty_or_fully_padded_molecule_sets(variant, pooling, molecule_count):
    model = make_encoder(variant, reaction_pooling=pooling, use_reaction_model=False)
    batch = make_inputs()
    batch.pop("reaction_embedding")
    batch["has_unimol2"] = torch.zeros(4, dtype=torch.bool)
    batch["has_chiro"] = torch.zeros(4, dtype=torch.bool)
    for side in ("reactant", "product"):
        for suffix in ("", "_chirality"):
            prefix = side + suffix
            batch[f"{prefix}_embeddings"] = batch[f"{prefix}_embeddings"][:, :molecule_count].fill_(float("nan"))
            batch[f"{prefix}_padding_mask"] = torch.ones(4, molecule_count, dtype=torch.bool)
    output, details = model(**batch, return_attention=True)
    assert torch.isfinite(output).all() and output.eq(0).all()
    assert details["modality_feature_weights"].eq(0).all()
    for key in ("unimol2_reactant", "unimol2_product", "chiro_reactant", "chiro_product"):
        assert details[key].shape == (4, molecule_count)
        assert details[key].eq(0).all()
    output.sum().backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def config_for(variant, token_dim=10):
    return DotDict(
        dict(
            data={},
            model=dict(
                query_encoder_type="multimodal_reaction_attention",
                query_encoder_dims=[token_dim, 12],
                embedding_dim=12,
                reaction_pooling="mean",
                reaction_multimodal_attention=dict(fusion=variant),
            ),
            training={},
        )
    )


@pytest.mark.parametrize("variant", ["feature_gate", "scalar_gate"])
def test_config_accepts_gate_dimensions_without_head_divisibility(variant):
    _validate_reaction_attention(config_for(variant))
    model = make_encoder(variant)
    assert model(**make_inputs()).shape == (4, 12)
    assert not any(isinstance(module, nn.MultiheadAttention) for module in model.modules())


@pytest.mark.parametrize("variant", ["attention", "mean", "concat", "gated_attention_concat"])
def test_existing_fusion_configs_and_encoders_remain_supported(variant):
    _validate_reaction_attention(config_for(variant, token_dim=16))
    model = make_encoder(variant, input_dim=16)
    assert model(**make_inputs()).shape == (4, 12)


def test_legacy_attention_gate_still_requires_divisible_dimensions():
    with pytest.raises(ValueError, match="divisible"):
        _validate_reaction_attention(config_for("gated_attention_concat"))


def test_feature_diagnostics_use_feature_competition_and_active_denominators():
    weights = torch.tensor([
        [[1., 0.], [0., 1.], [0., 0.]],
        [[.5, .5], [.5, .5], [0., 0.]],
        [[0., 0.], [0., 0.], [1., 1.]],
        [[0., 0.], [0., 0.], [0., 0.]],
    ])
    marginal = weights.mean(dim=-1)
    details = {
        "modality": marginal,
        "modality_entropy": -(marginal * marginal.clamp_min(1e-12).log()).sum(dim=1),
        "modality_names": ("reaction_model", "unimol2", "chiro"),
        "modality_mask": torch.tensor([
            [True, True, True], [True, True, False],
            [False, False, True], [False, False, False],
        ]),
        "modality_feature_weights": weights,
        "modality_feature_entropy": -(weights * weights.clamp_min(1e-12).log()).sum(dim=1),
    }
    stats = ProteinPooledLitModule._reaction_multimodal_attention_stats(details)
    expected = {
        "reaction_multimodal_attention/max_weight": 5 / 6,
        "reaction_multimodal_attention/entropy": math.log(2) / 3,
        "reaction_multimodal_attention/normalized_entropy": 1 / 3,
        "reaction_multimodal_attention/weight_reaction_model": .5,
        "reaction_multimodal_attention/weight_unimol2": .5,
        "reaction_multimodal_attention/weight_chiro": .5,
        "reaction_fusion/gate_reaction_model": .5,
        "reaction_fusion/gate_closed_reaction_model": .25,
        "reaction_fusion/gate_open_reaction_model": .25,
        "reaction_fusion/dominance_reaction_model": .5,
        "reaction_fusion/dominance_unimol2": .5,
        "reaction_fusion/gate_closed_chiro": .5,
        "reaction_fusion/gate_open_chiro": .5,
        "reaction_fusion/dominance_chiro": .5,
    }
    for name, value in expected.items():
        torch.testing.assert_close(stats[name], torch.tensor(value), msg=name)
    assert all(torch.isfinite(value).all() for value in stats.values())


def test_feature_diagnostics_all_missing_are_finite_and_zero():
    details = {
        "modality": torch.zeros(2, 3),
        "modality_entropy": torch.zeros(2),
        "modality_mask": torch.zeros(2, 3, dtype=torch.bool),
        "modality_names": ("reaction_model", "unimol2", "chiro"),
        "modality_feature_weights": torch.zeros(2, 3, 5),
        "modality_feature_entropy": torch.zeros(2, 5),
    }
    stats = ProteinPooledLitModule._reaction_multimodal_attention_stats(details)
    assert stats
    assert all(torch.isfinite(value).all() and value.eq(0).all() for value in stats.values())


@pytest.mark.parametrize("variant", ["feature_gate", "scalar_gate"])
def test_lightning_training_checkpoint_reload_and_evaluation(tmp_path, variant):
    import h5py
    import lightning.pytorch as pl
    import numpy as np
    import yaml

    from horizyn.reaction_conditioned_data_module import ReactionConditionedDataModule
    from scripts.evaluate_protein_pooling import (
        CONFIGURED_FORWARD_CANDIDATES,
        evaluate_checkpoint,
    )

    rng = np.random.default_rng(4)
    (tmp_path / "reactions.csv").write_text(
        "reaction_id,reaction_smiles\nr0,CCO.CCN\nr1,CCC.CCO\n"
    )
    (tmp_path / "pairs.csv").write_text(
        "pr_id,reaction_id,protein_id\n0,r0,p0\n1,r1,p1\n2,r0,p2\n3,r1,p3\n"
    )
    with h5py.File(tmp_path / "proteins.h5", "w") as handle:
        handle["ids"] = np.array(["p0", "p1", "p2", "p3"], dtype="S")
        handle["offsets"] = np.arange(0, 17, 4)
        handle["vectors"] = rng.normal(size=(16, 8)).astype("float32")
    for name, dim in (("unimol", 8), ("chiro", 6)):
        with h5py.File(tmp_path / f"{name}.h5", "w") as handle:
            handle["ids"] = np.array(["r0_f"], dtype="S")
            for side in ("reactant", "product"):
                handle[f"{side}_offsets"] = np.array([0, 2])
                handle[f"{side}_vectors"] = rng.normal(size=(2, dim)).astype("float32")
    with h5py.File(tmp_path / "t5.h5", "w") as handle:
        handle["ids"] = np.array(["r0_f", "r1_f"], dtype="S")
        handle["vectors"] = rng.normal(size=(2, 10)).astype("float32")

    data = dict(
        train_pairs_path=str(tmp_path / "pairs.csv"),
        test_pairs_path=str(tmp_path / "pairs.csv"),
        train_reactions_path=str(tmp_path / "reactions.csv"),
        test_reactions_path=str(tmp_path / "reactions.csv"),
        protein_residue_embeds_path=str(tmp_path / "proteins.h5"),
        residue_dim=8,
        reaction_representation="multimodal_reaction_attention",
        reaction_unimol2_embeds_path=str(tmp_path / "unimol.h5"),
        reaction_unimol_dim=8,
        reaction_chiro_embeds_path=str(tmp_path / "chiro.h5"),
        reaction_chiro_dim=6,
        reaction_t5v2_embeds_path=str(tmp_path / "t5.h5"),
        reaction_model_dim=10,
        reaction_use_model=True,
        reaction_use_chiro=True,
        reaction_use_chemistry=False,
        reaction_allow_missing_unimol2=True,
        reaction_allow_missing_chiro=True,
        normalize_molecule_sets_as_self_reactions=True,
        reaction_direction_mode="forward_only",
        train_batch_size=4,
        num_workers=0,
        standardize_reactions=False,
        validation_enabled=False,
    )
    model = ProteinPooledLitModule(
        query_encoder_dims=[10, 16],
        target_encoder_dims=[8, 16],
        embedding_dim=16,
        residue_dim=8,
        pooling="mean",
        query_encoder_type="multimodal_reaction_attention",
        reaction_unimol_dim=8,
        reaction_model_dim=10,
        reaction_chienn_dim=6,
        reaction_use_model=True,
        reaction_use_chienn=True,
        reaction_use_chemistry=False,
        reaction_side_composition="molecule_set",
        reaction_pooling="mean",
        reaction_separate_side_poolers=False,
        reaction_modality_fusion=variant,
        reaction_modality_encoder_widths=[16],
        loss_name="DecoupledAllPositiveInfoNCELoss",
        positive_pair_source="all_known_in_batch",
        unknown_negative_weight=0.5,
        biofp_aux_weight=0,
        log_attention_stats=True,
        attention_logging_interval=1,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=0,
    )
    trainer.fit(model, datamodule=ReactionConditionedDataModule(**data))
    assert torch.isfinite(trainer.callback_metrics["train/loss"])
    assert any("reaction_fusion/gate_" in key for key in trainer.callback_metrics)
    checkpoint = tmp_path / "last.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = ProteinPooledLitModule.load_from_checkpoint(
        checkpoint, map_location="cpu", weights_only=False
    )
    assert restored.hparams.reaction_modality_fusion == variant
    candidates = tmp_path / "candidates.txt"
    candidates.write_text("p0\np1\np2\np3\n")
    data["validation_retrieval_candidate_ids_path"] = str(candidates)
    config = {
        "data": data,
        "model": {
            "name": "ProteinPooledDualModel",
            "query_encoder_dims": [10, 16],
            "target_encoder_dims": [8, 16],
            "embedding_dim": 16,
            "query_encoder_type": "multimodal_reaction_attention",
            "reaction_use_chiro": True,
        },
        "training": {"max_epochs": 2},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = evaluate_checkpoint(
        str(checkpoint), str(config_path), "cpu", 2, 2, False,
        direction="both",
        evaluation_protocol=CONFIGURED_FORWARD_CANDIDATES,
        per_query_output=str(tmp_path / "queries.json"),
    )
    assert result["num_reaction_candidates"] == 2
    assert result["num_enzyme_candidates"] == 4
