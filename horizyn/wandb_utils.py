"""
Utilities for DDP-safe Weights & Biases logging.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable

import lightning.pytorch as pl


def configure_wandb_env_defaults() -> None:
    """Route W&B data/cache/artifact staging to shared project storage by default."""
    root = Path(
        os.environ.get("WANDB_ROOT", "/datastor2/deep-proteins/EnzymeDiscovery/wandb")
    )
    os.environ.setdefault("WANDB_DIR", str(root))
    os.environ.setdefault("WANDB_DATA_DIR", str(root / "data"))
    os.environ.setdefault("WANDB_CACHE_DIR", str(root / "cache"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(root / "config"))
    for key in ("WANDB_DIR", "WANDB_DATA_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def to_plain_config(value: Any) -> Any:
    """Convert nested DotDict/list/Path values into logger-friendly objects."""
    if isinstance(value, dict):
        return {key: to_plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def is_rank_zero_process() -> bool:
    """Best-effort rank-zero check before Lightning has fully initialized."""
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        return int(rank) == 0
    except ValueError:
        return True


def rank_zero_print(message: str = "") -> None:
    if is_rank_zero_process():
        print(message)


def resolve_wandb_settings(args: argparse.Namespace, config) -> dict[str, Any]:
    """Resolve CLI/config/env W&B settings into one effective dictionary."""
    wandb_config = config.logging.get("wandb", {})
    requested = bool(getattr(args, "wandb", False) or wandb_config.get("enabled", False))
    mode = (
        getattr(args, "wandb_mode", None)
        or wandb_config.get("mode", os.environ.get("WANDB_MODE", "offline"))
    )
    settings = {
        "requested": requested,
        "enabled": requested and mode != "disabled",
        "mode": mode,
        "project": getattr(args, "wandb_project", None)
        or wandb_config.get("project", "horizyn-training"),
        "entity": getattr(args, "wandb_entity", None) or wandb_config.get("entity", None),
        "run_name": getattr(args, "wandb_run_name", None)
        or wandb_config.get("run_name", None),
        "tags": (
            getattr(args, "wandb_tags", None)
            if getattr(args, "wandb_tags", None) is not None
            else wandb_config.get("tags", None)
        ),
        "log_model": bool(
            getattr(args, "wandb_log_model", False) or wandb_config.get("log_model", False)
        ),
    }
    return settings


def build_wandb_logger(args: argparse.Namespace, config) -> pl.loggers.WandbLogger | None:
    """Create a W&B logger without materializing the run before DDP setup."""
    settings = resolve_wandb_settings(args, config)
    if not settings["enabled"]:
        return None
    configure_wandb_env_defaults()

    try:
        import wandb  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging was requested but wandb is not installed. "
            "Install it with: uv pip install wandb"
        ) from exc

    return pl.loggers.WandbLogger(
        project=settings["project"],
        entity=settings["entity"],
        name=settings["run_name"],
        save_dir=config.logging.log_dir,
        offline=settings["mode"] == "offline",
        tags=settings["tags"],
        log_model=settings["log_model"],
    )


def build_logger_hparams(
    args: argparse.Namespace,
    config,
    total_params: int,
    trainable_params: int,
) -> dict[str, Any]:
    """Build the explicit run config logged once on global rank zero."""
    plain_config = to_plain_config(config)
    wandb_settings = resolve_wandb_settings(args, config)

    if isinstance(plain_config, dict):
        logging_config = plain_config.setdefault("logging", {})
        if isinstance(logging_config, dict):
            logging_config["wandb"] = wandb_settings

    return {
        "config_path": args.config,
        "resume": getattr(args, "resume", None),
        "config": plain_config,
        "wandb": wandb_settings,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def _iter_loggers(trainer: pl.Trainer) -> Iterable[Any]:
    loggers = getattr(trainer, "loggers", None)
    if loggers is not None:
        return loggers
    logger = getattr(trainer, "logger", None)
    if logger is None:
        return []
    if isinstance(logger, (list, tuple)):
        return logger
    return [logger]


class DelayedHyperparameterLogger(pl.Callback):
    """Log explicit hparams after DDP rank setup, and only on global rank zero."""

    def __init__(self, hparams: dict[str, Any]):
        super().__init__()
        self.hparams = hparams
        self._logged = False

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._logged or not trainer.is_global_zero:
            return
        for logger in _iter_loggers(trainer):
            if hasattr(logger, "log_hyperparams"):
                logger.log_hyperparams(self.hparams)
        self._logged = True
