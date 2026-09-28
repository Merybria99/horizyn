#!/usr/bin/env python3
"""
Train a Horizyn dual encoder with residue-level protein pooling.
"""

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch


from horizyn.config import load_config, parse_overrides
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from horizyn.reaction_conditioned_data_module import ReactionConditionedDataModule
from horizyn.training_checkpoints import recovery_checkpoint, training_early_stopping
from horizyn.training_options import (
    _config_first,  # Compatibility import for callers of the old entry point.
    protein_pooling_model_kwargs,
    reaction_data_module_kwargs,
    resolve_reaction_chirality,
)

# Compatibility imports: older scripts/tests still import these names here.
from horizyn.training_runtime import _configure_cpu_threads, _performance_ddp_strategy
from horizyn.training_warm_start import (
    _load_biofp_pretrain_warm_start,
    _load_partial_model_warm_start,
)
from horizyn.wandb_utils import (
    DelayedHyperparameterLogger,
    build_logger_hparams,
    build_wandb_logger,
    rank_zero_print,
    resolve_wandb_settings,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Horizyn with residue-level protein pooling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--pipeline", choices=["v4", "f3", "circev2"], default="v4")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument(
        "--model-preflight-only",
        action="store_true",
        help="Build and warm-start the model, then exit before logger/trainer setup",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default=None, help="W&B project name")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team")
    parser.add_argument("--wandb-run-name", default=None, help="Optional W&B run name")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=None,
        help="W&B mode",
    )
    parser.add_argument("--wandb-tags", nargs="*", default=None, help="Optional W&B tags")
    parser.add_argument(
        "--wandb-log-model",
        action="store_true",
        help="Upload checkpoints as W&B artifacts",
    )
    return parser


def _source_replay_kwargs(config) -> tuple[dict[str, Any] | None, float, int]:
    """Resolve an optional replay data module from another training config.

    Reusing the source-pretraining config keeps the cache and feature paths in
    one place. Only arguments accepted by ``ReactionConditionedDataModule``
    are forwarded; sampler/model/trainer keys stay with the primary run.
    """

    replay = config.data.get("source_replay", {})
    if not replay or not replay.get("enabled", False):
        return None, 0.0, int(config.get("seed", 42))
    config_path = replay.get("config_path", None)
    if config_path is None:
        raise ValueError("data.source_replay.enabled=true requires config_path")
    source = load_config(config_path)
    accepted = set(inspect.signature(ReactionConditionedDataModule.__init__).parameters)
    accepted.discard("self")
    replay_kwargs = {key: value for key, value in dict(source.data).items() if key in accepted}
    fraction = float(replay.get("fraction", 0.15))
    seed = int(replay.get("seed", config.get("seed", 42)))
    return replay_kwargs, fraction, seed


def _print_training_config(config, wandb_settings) -> None:
    """Display the supported recipe without dumping inactive experimental options."""
    print(f"Pipeline: {config.model.get('enzyme_input_mode', 'pooled')}")
    print(f"Residues: {config.data.protein_residue_embeds_path}")
    print(f"Training associations: {config.data.train_pairs_path}")
    print(f"Alignment loss: {config.training.loss.get('name', 'FullBatchMLNCELoss')}")
    print(f"Epochs: {config.training.max_epochs}; batch per rank: {config.data.train_batch_size}")
    print(f"Checkpoint directory: {config.logging.checkpoint_dir}")
    print(f"Validation enabled: {config.training.get('validation_enabled', True)}")
    print(f"W&B: {wandb_settings}")


def main() -> None:
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args()
    overrides = parse_overrides(unknown)
    if args.seed is not None:
        overrides["seed"] = args.seed

    print(f"Loading config from: {args.config}")
    try:
        config = load_config(args.config, overrides=overrides)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    from .registry import validate_pipeline
    validate_pipeline(config, args.pipeline)

    validation_enabled = bool(config.training.get("validation_enabled", True))
    if validation_enabled:
        for key in ("validation_pairs_path", "validation_reactions_path"):
            value = config.data.get(key, None)
            if value is not None and any(part.lower() == "test" for part in Path(str(value)).parts):
                raise ValueError(f"{key} must not point at the held-out test subset: {value}")

    if (
        "check_val_every_n_epoch" not in config.training
        and config.training.get("validation_interval_steps", None) is None
    ):
        config.training.check_val_every_n_epoch = 1
    wandb_settings = resolve_wandb_settings(args, config)
    _print_training_config(config, wandb_settings)

    pl.seed_everything(config.get("seed", 42), workers=True)
    _configure_cpu_threads(config.training.get("cpu_num_threads", None))
    if "float32_matmul_precision" in config.training:
        torch.set_float32_matmul_precision(config.training.float32_matmul_precision)

    replay_config, replay_fraction, replay_seed = _source_replay_kwargs(config)

    data_module = ReactionConditionedDataModule(
        **reaction_data_module_kwargs(
            config,
            replay_config=replay_config,
            replay_fraction=replay_fraction,
            replay_seed=replay_seed,
        )
    )

    model = ProteinPooledLitModule(**protein_pooling_model_kwargs(config))
    _load_partial_model_warm_start(
        model,
        config.training.get("init_from_checkpoint", None),
        capability_gate_bias=config.training.get("capability_gate_init_bias", -2.0),
    )
    _load_biofp_pretrain_warm_start(
        model,
        config.training.get("biofp_pretrain_checkpoint", None),
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    if args.model_preflight_only:
        print("Model preflight complete")
        return

    csv_logger = pl.loggers.CSVLogger(
        save_dir=config.logging.log_dir,
        name="protein_pooling_training",
    )
    logger: Any = csv_logger
    try:
        wandb_logger = build_wandb_logger(args, config)
    except RuntimeError as exc:
        print(f"Error initializing W&B logger: {exc}")
        sys.exit(1)

    callbacks: list[pl.Callback] = []
    if wandb_logger is not None:
        callbacks.append(
            DelayedHyperparameterLogger(
                build_logger_hparams(
                    args=args,
                    config=config,
                    total_params=total_params,
                    trainable_params=trainable_params,
                )
            )
        )
        logger = [csv_logger, wandb_logger]
        rank_zero_print("W&B logger enabled")

    checkpoint_kwargs: dict[str, Any] = {
        "dirpath": config.logging.checkpoint_dir,
        "filename": "protein-pooling-{epoch:02d}",
        "save_last": bool(config.logging.get("save_last", True)),
    }
    if validation_enabled:
        checkpoint_kwargs.update(
            {
                "save_top_k": int(config.logging.get("save_top_k", 3)),
                "monitor": config.logging.get("checkpoint_monitor", "val/loss"),
                "mode": config.logging.get("checkpoint_mode", "min"),
            }
        )
    else:
        # A Level-1 final refit is selected by its fixed epoch budget, never by
        # a validation/test metric. ``last.ckpt`` is the only selection target.
        checkpoint_kwargs["save_top_k"] = 0
    if not validation_enabled:
        # ``save_last`` must observe the actual final epoch even when the
        # configured development checkpoint cadence is coarser.
        checkpoint_kwargs["every_n_epochs"] = 1
    elif config.logging.get("checkpoint_on_validation_end", False):
        checkpoint_kwargs["every_n_train_steps"] = None
        checkpoint_kwargs["every_n_epochs"] = 1
        checkpoint_kwargs["save_on_train_epoch_end"] = False
    elif config.logging.get("save_every_n_train_steps", None) is not None:
        checkpoint_kwargs["every_n_train_steps"] = config.logging.save_every_n_train_steps
        checkpoint_kwargs["every_n_epochs"] = None
    else:
        checkpoint_kwargs["every_n_epochs"] = config.logging.get("save_every_n_epochs", 10)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**checkpoint_kwargs)
    callbacks.append(checkpoint_callback)
    if config.logging.get("always_save_last_every_epoch", False):
        callbacks.append(pl.callbacks.ModelCheckpoint(
            dirpath=str(Path(config.logging.checkpoint_dir) / "final_epoch"),
            filename="final-epoch={epoch:02d}",
            save_top_k=0,
            save_last=True,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
        ))
    screen_selection_interval = config.logging.get("screen_selection_every_n_epochs")
    if screen_selection_interval is not None:
        if not validation_enabled or int(screen_selection_interval) < 1:
            raise ValueError("Screening selection checkpoints require validation and a positive epoch interval")
        # Keep a predeclared epoch grid independent of small-pool retrieval
        # metrics. The full-library BEDROC evaluator selects among these later.
        callbacks.append(pl.callbacks.ModelCheckpoint(
            dirpath=str(Path(config.logging.checkpoint_dir) / "screen_selection"),
            filename="screen-epoch={epoch:02d}",
            auto_insert_metric_name=False,
            save_top_k=-1,
            save_weights_only=bool(config.logging.get("screen_selection_save_weights_only", True)),
            every_n_epochs=int(screen_selection_interval),
            save_on_train_epoch_end=True,
        ))
    recovery_callback = recovery_checkpoint(config.logging)
    if recovery_callback is not None:
        callbacks.append(recovery_callback)
        print(
            "Independent recovery checkpoints: "
            f"every {recovery_callback._every_n_train_steps} optimizer steps; "
            f"{recovery_callback.dirpath}/last.ckpt",
            flush=True,
        )
    early_stopping_callback = training_early_stopping(
        config.training.get("early_stopping", {}), config.logging
    )
    if early_stopping_callback is not None:
        callbacks.append(early_stopping_callback)

    devices = config.training.get("devices", 1 if torch.cuda.is_available() else "auto")
    accelerator = config.training.get("accelerator", "auto")
    strategy = config.training.get("strategy", "auto")
    if torch.cuda.is_available() and devices not in (1, "1", "auto") and strategy == "auto":
        strategy = "ddp"
    strategy = _performance_ddp_strategy(
        strategy,
        gradient_as_bucket_view=config.training.get("ddp_gradient_as_bucket_view", False),
    )

    use_distributed_sampler = config.training.get("use_distributed_sampler", True)
    if (
        config.data.get("hard_negative_pools_path", None)
        or config.data.get("typed_negative_pools_path", None)
        or config.data.get("indexed_pairs_dir", None)
        or config.data.get("train_sampler", {}).get("name")
        in {"enzyme_grouped", "hypergraph_grouped", "balanced_anchors"}
    ) and use_distributed_sampler:
        print(
            "Disabling Lightning distributed sampler injection: "
            "the configured negative batch sampler performs DDP sharding."
        )
        use_distributed_sampler = False

    trainer_kwargs = {
        "max_epochs": config.training.max_epochs,
        "logger": logger,
        "callbacks": callbacks,
        "log_every_n_steps": config.logging.get("log_every_n_steps", 1),
        "enable_progress_bar": config.training.get("enable_progress_bar", True),
        "deterministic": True,
        "devices": devices,
        "accelerator": accelerator,
        "strategy": strategy,
        "num_nodes": config.training.get("num_nodes", 1),
        "use_distributed_sampler": use_distributed_sampler,
        "precision": config.training.get("precision", "32-true"),
        "num_sanity_val_steps": config.training.get("num_sanity_val_steps", 0),
        "accumulate_grad_batches": config.training.get("accumulate_grad_batches", 1),
        "limit_val_batches": 1.0 if validation_enabled else 0,
    }
    for key in ("max_steps", "limit_train_batches", "gradient_clip_val"):
        if key in config.training:
            trainer_kwargs[key] = config.training[key]
    if not validation_enabled:
        trainer_kwargs["check_val_every_n_epoch"] = None
    elif config.training.get("validation_interval_steps", None) is not None:
        trainer_kwargs["check_val_every_n_epoch"] = None
        trainer_kwargs["val_check_interval"] = config.training.validation_interval_steps
    else:
        trainer_kwargs["check_val_every_n_epoch"] = config.training.get(
            "check_val_every_n_epoch",
            10,
        )
    trainer = pl.Trainer(**trainer_kwargs)

    print("=" * 80)
    print("STARTING PROTEIN-POOLING TRAINING")
    print("=" * 80 + "\n")
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume)

    print("\n" + "=" * 80)
    print("PROTEIN-POOLING TRAINING COMPLETE")
    print("=" * 80)
    if validation_enabled:
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    else:
        print("Best checkpoint: not applicable (fixed-epoch final refit)")
    print(f"Last checkpoint: {checkpoint_callback.last_model_path}")


if __name__ == "__main__":
    main()
