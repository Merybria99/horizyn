#!/usr/bin/env python3
"""
Horizyn Model Training Script

Trains the Horizyn contrastive learning model for enzyme-reaction matching.

Usage:
    python train.py --config configs/sota.yaml

    # Override config values
    python train.py --config configs/sota.yaml --training.max_epochs 50

    # Set random seed
    python train.py --config configs/sota.yaml --seed 123

Requirements:
    - Data must be downloaded first (see scripts/download_data.py)
    - Requires ~16GB RAM (all data loaded into memory)
    - Requires single GPU with 16GB+ VRAM

Example:
    # Train SOTA model for 100 epochs
    python train.py --config configs/sota.yaml

    # Train with custom batch size
    python train.py --config configs/sota.yaml --data.train_batch_size 8192
"""

import argparse
import sys
from typing import Any

import lightning.pytorch as pl
import torch

from horizyn.config import load_config, parse_overrides
from horizyn.data_module import HorizynDataModule
from horizyn.lightning_module import HorizynLitModule
from horizyn.wandb_utils import (
    DelayedHyperparameterLogger,
    build_logger_hparams,
    build_wandb_logger,
    rank_zero_print,
    resolve_wandb_settings,
)


def main():
    """Main training function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train Horizyn contrastive learning model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., configs/sota.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (overrides config.seed if provided)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project name (default: logging.wandb.project)",
    )
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team")
    parser.add_argument("--wandb-run-name", default=None, help="Optional W&B run name")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=None,
        help="W&B mode (default: logging.wandb.mode or WANDB_MODE or offline)",
    )
    parser.add_argument("--wandb-tags", nargs="*", default=None, help="Optional W&B tags")
    parser.add_argument(
        "--wandb-log-model",
        action="store_true",
        help="Upload checkpoints as W&B model artifacts",
    )

    # Parse known args and capture remaining for overrides
    args, unknown = parser.parse_known_args()

    # Parse config overrides from remaining arguments
    overrides = parse_overrides(unknown)

    # Apply seed override if provided
    if args.seed is not None:
        overrides["seed"] = args.seed

    # Load configuration
    print(f"Loading config from: {args.config}")
    try:
        config = load_config(args.config, overrides=overrides)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you're running from the project root directory.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Config validation failed")
        print(f"{e}")
        sys.exit(1)

    wandb_settings = resolve_wandb_settings(args, config)
    query_encoder_checkpoint_path = config.model.get("query_encoder_checkpoint_path", None)
    # Print configuration summary
    print("\n" + "=" * 80)
    print("HORIZYN TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Seed: {config.seed}")
    print(f"Max Epochs: {config.training.max_epochs}")
    print(f"Train Batch Size: {config.data.train_batch_size}")
    print(f"Retrieval Batch Size: {config.data.retrieval_batch_size}")
    print(f"Learning Rate: {config.training.learning_rate}")
    print(f"Weight Decay: {config.training.weight_decay}")
    print(f"Model: {config.model.name}")
    print(f"Reaction Pooling: {config.model.get('reaction_pooling', 'attention')}")
    print(f"Query Encoder: {config.model.query_encoder_dims}")
    if query_encoder_checkpoint_path:
        print(f"Query Encoder Checkpoint: {query_encoder_checkpoint_path}")
    print(f"Target Encoder: {config.model.target_encoder_dims}")
    print(f"Embedding Dim: {config.model.embedding_dim}")
    print(f"Loss: {config.training.loss.name} (beta={config.training.loss.beta})")
    print(f"Log Dir: {config.logging.log_dir}")
    print(f"Checkpoint Dir: {config.logging.checkpoint_dir}")
    print(f"W&B Enabled: {wandb_settings['enabled']} ({wandb_settings['mode']})")
    print(f"Accelerator: {config.training.get('accelerator', 'auto')}")
    print(f"Devices: {config.training.get('devices', 1 if torch.cuda.is_available() else 'auto')}")
    print(f"Strategy: {config.training.get('strategy', 'auto')}")
    print("=" * 80 + "\n")

    # Set random seed for reproducibility
    seed = config.get("seed", 42)
    pl.seed_everything(seed, workers=True)
    print(f"Set random seed to: {seed}\n")

    # Check for GPU availability
    if not torch.cuda.is_available():
        print("Warning: No GPU detected. Training will be very slow on CPU.")
        print("Consider using a machine with a CUDA-capable GPU.\n")

    # Setup data module
    print("Initializing data module...")
    try:
        data_module = HorizynDataModule(
            train_pairs_path=config.data.train_pairs_path,
            test_pairs_path=config.data.test_pairs_path,
            train_reactions_path=config.data.train_reactions_path,
            test_reactions_path=config.data.test_reactions_path,
            protein_embeds_path=config.data.protein_embeds_path,
            train_batch_size=config.data.train_batch_size,
            retrieval_batch_size=config.data.retrieval_batch_size,
            rdkit_fp_dim=config.data.get("rdkit_fp_dim", 1024),
            drfp_dim=config.data.get("drfp_dim", 1024),
            reaction_representation=config.data.get("reaction_representation", "fingerprint"),
            reaction_embeds_path=config.data.get("reaction_embeds_path", None),
            reaction_unimol_dim=config.data.get("reaction_unimol_dim", 768),
            num_workers=config.data.get("num_workers", 0),
            pin_memory=config.data.get("pin_memory", False),
            standardize_reactions=config.data.get("standardize_reactions", True),
            standardize_hypervalent=config.data.get("standardize_hypervalent", True),
            standardize_remove_hs=config.data.get("standardize_remove_hs", True),
            standardize_kekulize=config.data.get("standardize_kekulize", False),
            standardize_uncharge=config.data.get("standardize_uncharge", True),
            standardize_metals=config.data.get("standardize_metals", True),
        )
    except FileNotFoundError as e:
        print(f"\nError: Data file not found")
        print(f"{e}")
        print("\nPlease download the dataset first:")
        print("    python scripts/download_data.py")
        sys.exit(1)
    except Exception as e:
        print(f"\nError initializing data module: {e}")
        sys.exit(1)

    print("Data module initialized.\n")

    # Setup model
    print("Initializing model...")
    model = HorizynLitModule(
        query_encoder_dims=config.model.query_encoder_dims,
        target_encoder_dims=config.model.target_encoder_dims,
        embedding_dim=config.model.embedding_dim,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        beta=config.training.loss.beta,
        learn_beta=config.training.loss.get("learn_beta", False),
        loss_name=config.training.loss.get("name", "FullBatchMLNCELoss"),
        lambda_r=config.training.loss.get("lambda_r", 0.05),
        lambda_e=config.training.loss.get("lambda_e", 0.05),
        lambda_g=config.training.loss.get("lambda_g", 0.01),
        tau_r=config.training.loss.get("tau_r", 0.1),
        tau_e=config.training.loss.get("tau_e", 0.1),
        tau_t=config.training.loss.get("tau_t", 0.1),
        delta_r=config.training.loss.get("delta_r", 0.5),
        delta_e=config.training.loss.get("delta_e", 0.5),
        symmetric_gw=config.training.loss.get("symmetric_gw", True),
        metric_ks=config.training.metrics.get("top_k", [1, 10, 100, 1000]),
        query_encoder_type=config.model.get("query_encoder_type", "mlp"),
        reaction_unimol_dim=config.data.get("reaction_unimol_dim", 768),
        reaction_pooling=config.model.get("reaction_pooling", "attention"),
        reaction_attention_bias=config.model.get("reaction_attention_pooling", {}).get(
            "attention_bias",
            True,
        ),
        reaction_separate_side_poolers=config.model.get(
            "reaction_attention_pooling",
            {},
        ).get("separate_side_poolers", True),
        query_encoder_checkpoint_path=query_encoder_checkpoint_path,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Setup logging
    csv_logger = pl.loggers.CSVLogger(
        save_dir=config.logging.log_dir,
        name="horizyn_training",
    )
    logger: Any = csv_logger
    try:
        wandb_logger = build_wandb_logger(args, config)
    except RuntimeError as e:
        print(f"\nError initializing W&B logger: {e}")
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

    # Setup callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.logging.checkpoint_dir,
        filename="horizyn-{epoch:02d}",
        every_n_epochs=config.logging.get("save_every_n_epochs", 10),
        save_last=True,
        save_top_k=3,
        monitor="val/loss",
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    # Setup trainer
    print("Setting up Lightning Trainer...")
    devices = config.training.get("devices", 1 if torch.cuda.is_available() else "auto")
    accelerator = config.training.get("accelerator", "auto")
    strategy = config.training.get("strategy", "auto")

    if torch.cuda.is_available() and devices not in (1, "1", "auto") and strategy == "auto":
        strategy = "ddp"

    trainer_kwargs = {
        "max_epochs": config.training.max_epochs,
        "logger": logger,
        "callbacks": callbacks,
        "log_every_n_steps": config.logging.get("log_every_n_steps", 1),
        "check_val_every_n_epoch": config.training.get("check_val_every_n_epoch", 10),
        "enable_progress_bar": config.training.get("enable_progress_bar", True),
        "deterministic": True,
        "devices": devices,
        "accelerator": accelerator,
        "strategy": strategy,
        "num_nodes": config.training.get("num_nodes", 1),
        "use_distributed_sampler": config.training.get("use_distributed_sampler", True),
    }

    trainer = pl.Trainer(
        **trainer_kwargs,
    )

    print(f"Trainer configured for {config.training.max_epochs} epochs\n")

    # Train
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")

    try:
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path=args.resume,  # Resume from checkpoint if provided
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Last checkpoint saved to: {checkpoint_callback.last_model_path}")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Training complete
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Last checkpoint: {checkpoint_callback.last_model_path}")
    print(f"Logs saved to: {config.logging.log_dir}")
    print("\nTo resume training, use:")
    print(
        f"    python train.py --config {args.config} --resume {checkpoint_callback.last_model_path}"
    )


if __name__ == "__main__":
    main()
