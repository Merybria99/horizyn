"""Recovery and validation callbacks that support resumed training."""
from pathlib import Path

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


def training_early_stopping(early_stopping_config, logging_config):
    """Check validation monitors only after a fresh validation has completed.

    A checkpoint saved at validation end can resume into the preceding epoch's
    unfinished train-epoch-end hooks, with an empty callback-metric cache.
    Lightning's automatic train-epoch-end check would fail at that boundary.
    Keep strict metric checking and the existing state key (monitor/mode), so
    genuine missing metrics still fail and patience/best-score state restores.
    """
    if not early_stopping_config.get("enabled", False):
        return None
    monitor = early_stopping_config.get(
        "monitor", logging_config.get("checkpoint_monitor", "val/loss")
    )
    return EarlyStopping(
        monitor=monitor,
        mode=early_stopping_config.get(
            "mode", logging_config.get("checkpoint_mode", "min")
        ),
        patience=int(early_stopping_config.get("patience", 5)),
        min_delta=float(early_stopping_config.get("min_delta", 0.0)),
        check_on_train_epoch_end=False if monitor.startswith("val/") else None,
    )


def recovery_checkpoint(logging_config):
    interval = logging_config.get("recovery_every_n_train_steps")
    if interval is None and logging_config.get("checkpoint_on_validation_end", False):
        interval = logging_config.get("save_every_n_train_steps")
    if interval is None:
        return None
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ValueError("recovery_every_n_train_steps must be a positive integer")
    return ModelCheckpoint(
        dirpath=Path(logging_config["checkpoint_dir"]) / "recovery",
        filename="recovery-{step:08d}", auto_insert_metric_name=False,
        monitor=None, save_top_k=1, save_last="link", save_weights_only=False,
        every_n_train_steps=interval, every_n_epochs=0,
        save_on_train_epoch_end=False,
    )
