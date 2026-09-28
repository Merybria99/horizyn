"""Configuration, per-step training, and portable multiview tower checkpoints."""

import copy
from unittest.mock import patch

import lightning.pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader

from horizyn.config import DotDict
from horizyn.config_validation import _validate_enzyme_inputs, validate_enzyme_multiview_config
from horizyn.model import FunctionalResidueScorer
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from horizyn.training_options import protein_pooling_model_kwargs


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def scorer_checkpoint(tmp_path):
    path = tmp_path / "sleec.pt"
    torch.save(FunctionalResidueScorer(8, 6).scorer.state_dict(), path)
    return str(path)


def config(checkpoint="sleec.pt"):
    return DotDict(
        data={"residue_dim": 8},
        model={
            "query_encoder_dims": [6, 8], "target_encoder_dims": [8, 8],
            "embedding_dim": 8, "pooling": "sleec_guided_attention",
            "enzyme_input_mode": "raw_mean_sleec_multiview",
            "sleec_pooling": {"checkpoint_path": checkpoint, "freeze_scorer": True,
                              "scorer_hidden_dim": 6},
            "enzyme_multiview": {"hidden_dim": 8, "num_slots": 2, "dropout": 0.0},
        },
        training={"learning_rate": 1e-3, "weight_decay": 0.01,
                  "loss": {"name": "DecoupledAllPositiveInfoNCELoss", "beta": 10.0,
                           "biofp_aux_weight": 0.0}},
    )


def make_module(checkpoint, **overrides):
    options = dict(
        query_encoder_dims=[6, 8], target_encoder_dims=[8, 8], embedding_dim=8,
        residue_dim=8, pooling="sleec_guided_attention", sleec_scorer_hidden_dim=6,
        sleec_checkpoint_path=checkpoint, sleec_freeze_scorer=True,
        enzyme_input_mode="raw_mean_sleec_multiview",
        enzyme_multiview={"hidden_dim": 8, "num_slots": 2, "dropout": 0.0},
        loss_name="DecoupledAllPositiveInfoNCELoss", log_attention_stats=False,
    )
    options.update(overrides)
    return ProteinPooledLitModule(**options)


def batch():
    return {
        "query_vec": torch.randn(4, 6),
        "residue_embeddings": torch.randn(4, 6, 8),
        "residue_padding_mask": torch.tensor([[False] * 5 + [True]] * 4),
        "query_id": [f"q{i}" for i in range(4)],
        "target_id": [f"p{i}" for i in range(4)],
    }


def test_configuration_plumbing_and_defaults(scorer_checkpoint):
    value = config(scorer_checkpoint)
    before = copy.deepcopy(value)
    _validate_enzyme_inputs(value)
    options = protein_pooling_model_kwargs(value)
    assert options["enzyme_multiview"] == value.model.enzyme_multiview
    assert options["enzyme_attention_regularization"] is None
    assert "enzyme_block_ffn" not in options
    model = ProteinPooledLitModule(**options)
    assert model.enzyme_attention_regularization == {"entropy_weight": .01, "diversity_weight": .001}
    assert model.model.multiview_encoder.num_slots == 2
    assert model.hparams.enzyme_multiview == value.model.enzyme_multiview
    assert value == before
    defaults = validate_enzyme_multiview_config(None)
    assert defaults["num_slots"] == 4 and defaults["hidden_dim"] == 256


@pytest.mark.parametrize("section,key,value,match", [
    ("model", "pooling", "mean", "sleec_guided_attention"),
    ("model", "sleec_pooling", {"freeze_scorer": False}, "frozen"),
    ("model", "sleec_pooling", {"freeze_scorer": True}, "checkpoint_path"),
    ("model", "enzyme_multiview", {"diversity_margin": 1.0}, "diversity_margin"),
    ("model", "enzyme_multiview", {"hidden_dim": 2, "num_slots": 4}, "num_slots"),
    ("model", "enzyme_multiview", {"max_logit_scale": float("nan")}, "finite"),
    ("model", "r2e_adapter", {"enabled": True}, "r2e_adapter"),
    ("model", "enzyme_block_fusion", {"weight_kl_weight": .1}, "block KL"),
    ("training", "loss", {"biofp_aux_weight": .1}, "biofp_aux_weight"),
    ("training", "enzyme_attention_regularization", {"entropy_weight": -1}, "non-negative"),
    ("training", "training_stage", "prototype_only", "updates the enzyme tower"),
])
def test_configuration_rejects_incompatible_settings(section, key, value, match):
    options = config()
    options[section][key] = value
    with pytest.raises(ValueError, match=match):
        _validate_enzyme_inputs(options)


def test_regularizers_are_scoped_to_new_mode():
    options = config()
    options.model.enzyme_input_mode = "standard"
    options.model.pop("enzyme_multiview")
    options.training.enzyme_attention_regularization = {"entropy_weight": .01}
    with pytest.raises(ValueError, match="requires raw_mean_sleec_multiview"):
        _validate_enzyme_inputs(options)
    legacy = ProteinPooledLitModule(query_encoder_dims=[6, 8], target_encoder_dims=[8, 8], residue_dim=8, embedding_dim=8)
    assert legacy.enzyme_attention_regularization == {"entropy_weight": 0., "diversity_weight": 0.}


def test_regularization_details_and_gradients_without_attention_logging(scorer_checkpoint):
    model = make_module(scorer_checkpoint, enzyme_attention_regularization={"entropy_weight": .2, "diversity_weight": .3})
    data = batch()
    original = model.model.multiview_encoder.forward
    sentinel = torch.tensor(.4, requires_grad=True)

    def tracked(*args, **kwargs):
        assert kwargs["return_details"] is True
        embedding, details = original(*args, **kwargs)
        details["enzyme_multiview_entropy_loss"] = sentinel.expand(embedding.shape[0])
        details["enzyme_multiview_diversity_loss"] = (2 * sentinel).expand(embedding.shape[0])
        return embedding, details

    with patch.object(model.model.multiview_encoder, "forward", side_effect=tracked):
        loss, count, stats, components = model._compute_full_batch_loss(data, return_attention_stats=False)
    assert count == 4 and stats == {}
    torch.testing.assert_close(components["weighted_enzyme_multiview_entropy"], torch.tensor(.08))
    torch.testing.assert_close(components["weighted_enzyme_multiview_diversity"], torch.tensor(.24))
    loss.backward()
    torch.testing.assert_close(sentinel.grad, torch.tensor(.8))
    assert all(p.grad is None for p in model.model.pooling.parameters())
    learned = [p.grad for p in model.model.multiview_encoder.parameters() if p.grad is not None]
    assert learned and all(torch.isfinite(gradient).all() for gradient in learned)
    assert model.model.multiview_encoder.queries.grad.abs().sum() > 0


def test_actual_penalties_remain_differentiable_and_logging_skips_maps(scorer_checkpoint):
    model = make_module(scorer_checkpoint)
    data = batch()
    with torch.no_grad():
        model.model.multiview_encoder.queries[:] = model.model.multiview_encoder.queries[:1]
    loss, _, stats, components = model._compute_full_batch_loss(data, return_attention_stats=True)
    diversity = components["enzyme_multiview_diversity"]
    assert diversity.requires_grad and diversity > 0
    assert components["enzyme_multiview_entropy"].requires_grad
    assert "enzyme_multiview/entropy" in stats
    assert all(value.ndim == 0 and not value.requires_grad for key, value in stats.items() if key.startswith("enzyme_multiview/"))
    assert not any("raw_attention" in key or "gate_weights" in key for key in stats)
    loss.backward()
    assert torch.isfinite(model.model.multiview_encoder.queries.grad).all()


class TinyData(pl.LightningDataModule):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self._train_query_to_targets = {f"q{i}": [f"p{i}"] for i in range(4)}
        self._train_query_to_targets["q0"].append("p1")

    def train_dataloader(self):
        return DataLoader([self.data, self.data], batch_size=None)


@pytest.mark.parametrize("num_slots", [0, 2])
def test_training_checkpoint_reload_and_chunked_retrieval_parity(tmp_path, scorer_checkpoint, num_slots):
    torch.manual_seed(8)
    model = make_module(scorer_checkpoint, positive_pair_source="all_known_in_batch",
                        enzyme_multiview={"hidden_dim": 8, "num_slots": num_slots, "dropout": 0.0})
    data = batch()
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=2, max_epochs=1,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, limit_val_batches=0)
    trainer.fit(model, datamodule=TinyData(data))
    assert trainer.global_step == 2
    for name in ("entropy", "diversity"):
        assert torch.isfinite(trainer.callback_metrics[f"train/loss_weighted_enzyme_multiview_{name}"])
        if num_slots == 0:
            assert trainer.callback_metrics[f"train/loss_weighted_enzyme_multiview_{name}"] == 0
    checkpoint = tmp_path / "multiview.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = ProteinPooledLitModule.load_from_checkpoint(
        checkpoint, map_location="cpu", weights_only=False, sleec_checkpoint_path=None)
    assert restored.hparams.enzyme_input_mode == "raw_mean_sleec_multiview"
    assert restored.model.multiview_encoder.num_slots == num_slots
    assert restored.model._multiview_sleec_ready
    model.eval()
    restored.eval()
    with torch.no_grad():
        expected = model.model.encode_targets(data["residue_embeddings"], residue_padding_mask=data["residue_padding_mask"])
        actual = restored.model.encode_targets(data["residue_embeddings"], residue_padding_mask=data["residue_padding_mask"])
        chunks = torch.cat([restored.model.encode_targets(values, residue_padding_mask=mask)
            for values, mask in zip(data["residue_embeddings"].split(2), data["residue_padding_mask"].split(2))])
        queries = restored.model.encode_queries(data["query_vec"])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(chunks, expected)
    torch.testing.assert_close(queries @ actual.T, queries @ chunks.T)
    torch.testing.assert_close(actual.norm(dim=-1), torch.ones(4))


def test_legacy_checkpoint_without_new_options_still_loads(tmp_path):
    model = ProteinPooledLitModule(query_encoder_dims=[6, 8], target_encoder_dims=[8, 8], residue_dim=8, embedding_dim=8).eval()
    hparams = dict(model.hparams)
    hparams.pop("enzyme_multiview")
    hparams.pop("enzyme_attention_regularization")
    path = tmp_path / "legacy.ckpt"
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": hparams,
                "pytorch-lightning_version": pl.__version__}, path)
    restored = ProteinPooledLitModule.load_from_checkpoint(path, weights_only=False).eval()
    values = torch.randn(2, 4, 8)
    torch.testing.assert_close(restored.model.encode_targets(values), model.model.encode_targets(values))


def test_incomplete_checkpoint_cannot_mark_random_scorer_ready(scorer_checkpoint):
    source = make_module(scorer_checkpoint)
    state = {key: value for key, value in source.state_dict().items() if "pooling.sleec_scorer" not in key}
    target = make_module(None)
    target.load_state_dict(state, strict=False)
    assert not target.model._multiview_sleec_ready
    with pytest.raises(RuntimeError, match="loaded SLEEC prior"):
        target.model.encode_targets(torch.randn(2, 4, 8))


def test_cached_embeddings_cannot_skip_regularizers(scorer_checkpoint):
    model = make_module(scorer_checkpoint)
    with pytest.raises(RuntimeError, match="requires residue inputs"):
        model._compute_full_batch_loss({"cached_query_embedding": torch.randn(2, 8)})


def test_fp32_hook_tracks_the_scorer_used_by_multiview(scorer_checkpoint):
    model = make_module(scorer_checkpoint, fp32_sensitive_modules=True)
    assert hasattr(model.model.pooling.sleec_scorer, "_circe_fp32_contexts")
