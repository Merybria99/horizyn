"""Regression for extending a completed run from a validation-end checkpoint."""
import copy

import lightning.pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from horizyn.training_checkpoints import training_early_stopping


MONITOR = "val/mean_bidirectional_mrr"
SETTINGS = dict(enabled=True, monitor=MONITOR, mode="max", patience=5, min_delta=0.0001)


class ResumeFixture(pl.LightningModule):
    def __init__(self, *, publish_metric=True, flat_metric=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.publish_metric = publish_metric
        self.flat_metric = flat_metric
        self.training_epochs = []
        self.validation_epochs = []
        self.initial_state = None

    def on_train_start(self):
        self.initial_state = dict(
            global_step=self.global_step,
            weight=self.weight.detach().cpu().clone(),
            optimizer=copy.deepcopy(self.trainer.optimizers[0].state_dict()),
            early_stopping=copy.deepcopy(self.trainer.early_stopping_callback.state_dict()),
        )

    def training_step(self, batch, batch_idx):
        self.training_epochs.append(self.current_epoch)
        return (self.weight * batch[0]).square().mean()

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        self.validation_epochs.append(self.current_epoch)
        if self.publish_metric:
            value = 0.5 if self.flat_metric else 0.5 + 0.01 * self.current_epoch
            self.log(MONITOR, value, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.01)


def selection(path):
    return pl.callbacks.ModelCheckpoint(
        dirpath=path, monitor=MONITOR, mode="max", save_top_k=1,
        save_last=True, save_on_train_epoch_end=False,
    )


def trainer(path, max_epochs, early_stopping, checkpoint=None):
    return pl.Trainer(
        default_root_dir=path, accelerator="cpu", devices=1,
        max_epochs=max_epochs, logger=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0,
        check_val_every_n_epoch=1, val_check_interval=1.0,
        callbacks=[checkpoint or selection(path), early_stopping],
    )


def loader():
    return DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=2)


def legacy_checkpoint(path):
    stop = pl.callbacks.EarlyStopping(**{k: v for k, v in SETTINGS.items() if k != "enabled"})
    original = trainer(path, 5, stop)
    original.fit(ResumeFixture(), loader(), loader())
    assert original.global_step == 10
    return path / "last.ckpt"


def test_callback_checks_validation_after_validation_and_keeps_strictness():
    callback = training_early_stopping(SETTINGS, {})
    assert callback._check_on_train_epoch_end is False
    assert callback.strict is True
    assert callback.patience == 5
    legacy = pl.callbacks.EarlyStopping(monitor=MONITOR, mode="max")
    assert callback.state_key == legacy.state_key
    assert training_early_stopping({}, {}) is None
    default = training_early_stopping(dict(enabled=True), {})
    assert default.monitor == "val/loss" and default._check_on_train_epoch_end is False
    train_monitor = training_early_stopping({**SETTINGS, "monitor": "train/loss"}, {})
    assert train_monitor._check_on_train_epoch_end is None


def test_legacy_callback_reproduces_epoch_boundary_resume_failure(tmp_path):
    path = legacy_checkpoint(tmp_path / "old")
    stop = pl.callbacks.EarlyStopping(**{k: v for k, v in SETTINGS.items() if k != "enabled"})
    resumed = trainer(tmp_path / "legacy_resume", 7, stop)
    with pytest.raises(RuntimeError, match="Early stopping conditioned on metric.*not available"):
        resumed.fit(ResumeFixture(), loader(), loader(), ckpt_path=str(path))


def test_completed_five_epoch_checkpoint_resumes_with_full_state(tmp_path):
    path = legacy_checkpoint(tmp_path / "old")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    stop = training_early_stopping(SETTINGS, {})
    model = ResumeFixture()
    resumed = trainer(tmp_path / "resumed", 7, stop)
    resumed.fit(model, loader(), loader(), ckpt_path=str(path))
    assert model.initial_state["global_step"] == payload["global_step"] == 10
    torch.testing.assert_close(model.initial_state["weight"], payload["state_dict"]["weight"])
    initial_optimizer = model.initial_state["optimizer"]
    expected_optimizer = payload["optimizer_states"][0]
    assert initial_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for key, expected in expected_optimizer["state"][0].items():
        torch.testing.assert_close(initial_optimizer["state"][0][key], expected)
    expected_stop = payload["callbacks"][stop.state_key]
    actual_stop = model.initial_state["early_stopping"]
    assert actual_stop["wait_count"] == expected_stop["wait_count"]
    torch.testing.assert_close(actual_stop["best_score"], expected_stop["best_score"])
    assert model.training_epochs == [5, 5, 6, 6]
    assert model.validation_epochs == [5, 6]
    assert resumed.global_step == 14
    assert stop.wait_count == 0
    assert float(stop.best_score) == pytest.approx(0.56)

    uninterrupted_model = ResumeFixture()
    uninterrupted = trainer(tmp_path / "reference", 7, training_early_stopping(SETTINGS, {}))
    uninterrupted.fit(uninterrupted_model, loader(), loader())
    torch.testing.assert_close(model.weight, uninterrupted_model.weight)
    assert torch.load(tmp_path / "resumed/last.ckpt", map_location="cpu", weights_only=False)["global_step"] == 14


def test_real_missing_validation_metric_still_fails(tmp_path):
    # Do not let checkpoint selection throw first: test strict early stopping itself.
    recovery = pl.callbacks.ModelCheckpoint(dirpath=tmp_path, monitor=None, save_top_k=0)
    run = trainer(tmp_path, 1, training_early_stopping(SETTINGS, {}), recovery)
    with pytest.raises(RuntimeError, match="Early stopping conditioned on metric.*not available"):
        run.fit(ResumeFixture(publish_metric=False), loader(), loader())


def test_patience_still_stops_on_nonimproving_validation(tmp_path):
    stop = training_early_stopping({**SETTINGS, "patience": 2}, {})
    run = trainer(tmp_path, 30, stop)
    model = ResumeFixture(flat_metric=True)
    run.fit(model, loader(), loader())
    assert model.validation_epochs == [0, 1, 2]
    assert stop.wait_count == 2 and run.should_stop
