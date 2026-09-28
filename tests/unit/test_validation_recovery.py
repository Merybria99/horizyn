"""Degenerate validation is not a zero loss; recovery must precede validation."""
from unittest.mock import Mock

import lightning.pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from horizyn.losses import BidirectionalSampledMultiPositiveInfoNCELoss, NoContrastiveAnchorsError
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from horizyn.training_checkpoints import recovery_checkpoint


def module_for_loss():
    module = ProteinPooledLitModule(
        query_encoder_dims=[4, 4], target_encoder_dims=[4, 4], embedding_dim=4,
        residue_dim=4, pooling="mean", loss_name="BidirectionalSampledMultiPositiveInfoNCELoss",
        sampled_require_both_directions=True,
    )
    module.log = Mock()
    module.print = Mock()
    return module


def empty_contrast(*args, **kwargs):
    loss = BidirectionalSampledMultiPositiveInfoNCELoss(beta=10)
    return loss(torch.zeros(1, 2), torch.tensor([0, 0]), torch.tensor([0, 1]),
                biological_negative_mask=torch.zeros(1, 2, dtype=torch.bool),
                random_negative_mask=torch.zeros(1, 2, dtype=torch.bool))


def test_empty_training_still_raises():
    with pytest.raises(NoContrastiveAnchorsError):
        empty_contrast()


@pytest.mark.parametrize("transpose", [False, True])
def test_zero_weight_active_direction_raises_recoverable_no_anchors(transpose):
    dists = torch.tensor([[0.1, 0.8]], requires_grad=True)
    negative = torch.tensor([[False, True]])
    if transpose:
        dists, negative = dists.t(), negative.t()
    loss = BidirectionalSampledMultiPositiveInfoNCELoss(
        lambda_r2e=float(transpose), lambda_e2r=float(not transpose),
    )
    with pytest.raises(NoContrastiveAnchorsError, match="nonzero loss weight"):
        loss(dists, torch.tensor([0]), torch.tensor([0]), negative,
             torch.zeros_like(negative))


@pytest.mark.parametrize("transpose", [False, True])
def test_single_active_weighted_direction_has_finite_loss_and_gradients(transpose):
    dists = torch.tensor([[0.1, 0.8]])
    negative = torch.tensor([[False, True]])
    if transpose:
        dists, negative = dists.t(), negative.t()
    dists = dists.clone().requires_grad_()
    loss = BidirectionalSampledMultiPositiveInfoNCELoss(
        beta=2.0, lambda_r2e=float(not transpose), lambda_e2r=float(transpose),
    )
    actual, components = loss(
        dists, torch.tensor([0]), torch.tensor([0]), negative,
        torch.zeros_like(negative), return_components=True,
    )
    expected = torch.logsumexp(torch.tensor([-0.2, -1.6]), dim=0) + 0.2
    assert torch.allclose(actual, expected)
    assert components["e2r_weight" if transpose else "r2e_weight"].item() == 1.0
    actual.backward()
    assert torch.isfinite(dists.grad).all()


def test_degenerate_validation_skips_loss_without_fabricating_zero():
    module = module_for_loss().eval()
    module._compute_full_batch_loss = empty_contrast
    assert module.validation_step({}, 0) is None
    assert module.validation_step({}, 1) is None
    assert [call.args[:2] for call in module.log.call_args_list] == [
        ("val/contrastive_batch_valid_fraction", 0.0),
        ("val/contrastive_batch_valid_fraction", 0.0),
    ]
    module.print.assert_called_once()
    module.on_validation_epoch_start()
    module.validation_step({}, 2)
    assert module.print.call_count == 2


def test_unrelated_validation_errors_are_not_suppressed():
    module = module_for_loss().eval()
    module._compute_full_batch_loss = Mock(side_effect=ValueError("invalid masks"))
    with pytest.raises(ValueError, match="invalid masks"):
        module.validation_step({}, 0)


def test_valid_validation_loss_is_unchanged():
    module = module_for_loss().eval()
    expected = torch.tensor(0.75)
    module._compute_full_batch_loss = Mock(return_value=(expected, 7, None, {}))
    assert module.validation_step({}, 0) is expected
    losses = [call for call in module.log.call_args_list if call.args[0] == "val/loss"]
    assert len(losses) == 1 and losses[0].args[1] is expected
    assert losses[0].kwargs["batch_size"] == 7


@pytest.mark.parametrize("interval", [0, -1, True, 1.5, "100"])
def test_invalid_recovery_interval_is_rejected(tmp_path, interval):
    with pytest.raises(ValueError, match="positive integer"):
        recovery_checkpoint({"checkpoint_dir": str(tmp_path), "recovery_every_n_train_steps": interval})


def test_recovery_separate_from_validation_selection(tmp_path):
    callback = recovery_checkpoint({"checkpoint_dir": str(tmp_path),
                                    "checkpoint_on_validation_end": True,
                                    "save_every_n_train_steps": 2000})
    assert callback.monitor is None and callback._every_n_train_steps == 2000
    assert callback.dirpath == str(tmp_path / "recovery")
    assert callback.save_weights_only is False and callback.save_top_k == 1
    assert recovery_checkpoint({"checkpoint_dir": str(tmp_path)}) is None


class CheckpointFixture(pl.LightningModule):
    def __init__(self, fail_validation=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.fail_validation = fail_validation

    def training_step(self, batch, batch_idx):
        return (self.weight * batch[0]).square().mean()

    def validation_step(self, batch, batch_idx):
        if self.fail_validation:
            raise RuntimeError("deliberate validation failure")
        self.log("val/mean_bidirectional_mrr", 0.5)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.01)


def test_recovery_survives_first_validation_failure_and_resumes(tmp_path):
    config = {"checkpoint_dir": str(tmp_path), "recovery_every_n_train_steps": 1}
    recovery = recovery_checkpoint(config)
    selection = pl.callbacks.ModelCheckpoint(dirpath=tmp_path / "best",
        monitor="val/mean_bidirectional_mrr", save_top_k=1, save_last=True,
        save_on_train_epoch_end=False)
    loader = DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=2)
    kwargs = dict(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
                  enable_model_summary=False, num_sanity_val_steps=0, val_check_interval=1,
                  max_epochs=2, max_steps=2)
    trainer = pl.Trainer(**kwargs, callbacks=[selection, recovery])
    with pytest.raises(RuntimeError, match="deliberate validation failure"):
        trainer.fit(CheckpointFixture(True), loader, loader)
    path = tmp_path / "recovery/last.ckpt"
    assert path.is_file() and not selection.last_model_path
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["global_step"] == 1 and payload["optimizer_states"]
    resumed = pl.Trainer(**kwargs, callbacks=[recovery_checkpoint(config)])
    resumed.fit(CheckpointFixture(), loader, loader, ckpt_path=str(path))
    assert resumed.global_step == 2
    assert torch.load(path, map_location="cpu", weights_only=False)["global_step"] == 2
