#!/usr/bin/env python3
"""
Train the reaction-conditioned ProtT5 residue-pooling Horizyn variant.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from horizyn.config import load_config, parse_overrides
from horizyn.reaction_conditioned_data_module import ReactionConditionedDataModule
from horizyn.reaction_conditioned_lightning_module import ReactionConditionedLitModule
from horizyn.training_options import _config_first, resolve_reaction_chirality
from horizyn.wandb_utils import (
    DelayedHyperparameterLogger,
    build_logger_hparams,
    build_wandb_logger,
    rank_zero_print,
    resolve_wandb_settings,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Horizyn with reaction-conditioned residue pooling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
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

    config.training.check_val_every_n_epoch = 1
    query_encoder_checkpoint_path = config.model.get("query_encoder_checkpoint_path", None)
    reaction_multimodal_attention_config = config.model.get(
        "reaction_multimodal_attention",
        {},
    )
    (
        reaction_chirality_path,
        reaction_chirality_dim,
        reaction_use_chirality,
        reaction_allow_missing_chirality,
    ) = resolve_reaction_chirality(config)
    wandb_settings = resolve_wandb_settings(args, config)
    print("\n" + "=" * 80)
    print("REACTION-CONDITIONED HORIZYN TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Seed: {config.seed}")
    print(f"Max Epochs: {config.training.max_epochs}")
    print(f"Train Batch Size: {config.data.train_batch_size}")
    print(f"Residue HDF5: {config.data.protein_residue_embeds_path}")
    print(f"Reaction Representation: {config.data.get('reaction_representation', 'fingerprint')}")
    if config.data.get("reaction_representation", "fingerprint") != "fingerprint":
        if config.data.get("reaction_representation") == "multimodal_reaction_attention":
            print(
                "Reaction Model HDF5: "
                f"{config.data.get('reaction_t5v2_embeds_path', config.data.get('reaction_model_embeds_path', None))}"
            )
            print(
                "Reaction Uni-Mol2 HDF5: "
                f"{config.data.get('reaction_unimol2_embeds_path', config.data.get('reaction_embeds_path', None))}"
            )
            print(f"Reaction ChIRo HDF5: {reaction_chirality_path}")
        else:
            print(f"Reaction HDF5: {config.data.reaction_embeds_path}")
    print(f"Max Protein Tokens: {config.data.get('max_protein_tokens', None)}")
    print(f"Model: {config.model.name}")
    print(f"Query Encoder Type: {config.model.get('query_encoder_type', 'mlp')}")
    print(f"Reaction Pooling: {config.model.get('reaction_pooling', 'attention')}")
    print(f"Query Encoder: {config.model.query_encoder_dims}")
    if query_encoder_checkpoint_path:
        print(f"Query Encoder Checkpoint: {query_encoder_checkpoint_path}")
    print(f"Target Encoder: {config.model.target_encoder_dims}")
    print(f"Embedding Dim: {config.model.embedding_dim}")
    pooling_config = config.model.get("reaction_conditioned_pooling", {})
    print(f"Score Mode: {config.model.get('score_mode', 'target_mlp')}")
    print(f"Attention Rank: {pooling_config.get('attention_rank', None)}")
    print(f"Normalize Pooled Values: {pooling_config.get('normalize_pooled_values', True)}")
    if config.model.get("query_encoder_type", "mlp") == "multimodal_reaction_attention":
        print("Reaction Multimodal Attention: enabled")
        print(f"Reaction Model Dim: {config.data.get('reaction_model_dim', 1024)}")
        print(f"Reaction Uni-Mol2 Dim: {config.data.get('reaction_unimol_dim', 768)}")
        print(f"Reaction ChIRo Dim: {reaction_chirality_dim}")
        print(f"Reaction Allow Missing ChIRo: {reaction_allow_missing_chirality}")
        print(
            "Reaction Modality Dropout: "
            f"{reaction_multimodal_attention_config.get('modality_dropout', 0.0)}"
        )
        print(
            "Reaction Modality Token LayerNorm: "
            f"{reaction_multimodal_attention_config.get('token_layer_norm', reaction_multimodal_attention_config.get('token_normalization', False))}"
        )
        print(
            "Reaction Modality Encoder Widths: "
            f"{reaction_multimodal_attention_config.get('modality_encoder_widths', None)}"
        )
        print(
            "Reaction Modality Encoder LayerNorm: "
            f"{reaction_multimodal_attention_config.get('modality_encoder_use_layer_norm', reaction_multimodal_attention_config.get('modality_encoder_layer_norm', False))}"
        )
        print(
            "Reaction Modality Encoder Dropout: "
            f"{reaction_multimodal_attention_config.get('modality_encoder_dropout', 0.0)}"
        )
    print(f"Loss: {config.training.loss.get('name', 'FullBatchMLNCELoss')}")
    print(f"Log Dir: {config.logging.log_dir}")
    print(f"Checkpoint Dir: {config.logging.checkpoint_dir}")
    print(f"W&B Enabled: {wandb_settings['enabled']} ({wandb_settings['mode']})")
    print(f"Accelerator: {config.training.get('accelerator', 'auto')}")
    print(f"Devices: {config.training.get('devices', 1 if torch.cuda.is_available() else 'auto')}")
    print(f"Strategy: {config.training.get('strategy', 'auto')}")
    print(f"Precision: {config.training.get('precision', '32-true')}")
    print(f"Validation Frequency: every {config.training.check_val_every_n_epoch} epoch")
    print("=" * 80 + "\n")

    seed = config.get("seed", 42)
    pl.seed_everything(seed, workers=True)
    if "float32_matmul_precision" in config.training:
        torch.set_float32_matmul_precision(config.training.float32_matmul_precision)

    data_module = ReactionConditionedDataModule(
        train_pairs_path=config.data.train_pairs_path,
        test_pairs_path=config.data.test_pairs_path,
        train_reactions_path=config.data.train_reactions_path,
        test_reactions_path=config.data.test_reactions_path,
        protein_residue_embeds_path=config.data.protein_residue_embeds_path,
        train_batch_size=config.data.train_batch_size,
        retrieval_batch_size=config.data.get("retrieval_batch_size", 1),
        num_workers=config.data.get("num_workers", 0),
        pin_memory=config.data.get("pin_memory", False),
        rdkit_fp_dim=config.data.get("rdkit_fp_dim", 1024),
        drfp_dim=config.data.get("drfp_dim", 1024),
        reaction_representation=config.data.get("reaction_representation", "fingerprint"),
        reaction_embeds_path=config.data.get("reaction_embeds_path", None),
        reaction_t5v2_embeds_path=config.data.get("reaction_t5v2_embeds_path", None),
        reaction_model_embeds_path=config.data.get("reaction_model_embeds_path", None),
        reaction_unimol2_embeds_path=config.data.get("reaction_unimol2_embeds_path", None),
        reaction_chiro_embeds_path=config.data.get("reaction_chiro_embeds_path", None),
        reaction_chirality_embeds_path=config.data.get("reaction_chirality_embeds_path", None),
        reaction_chienn_embeds_path=reaction_chirality_path,
        train_reaction_embeds_path=config.data.get("train_reaction_embeds_path", None),
        validation_reaction_embeds_path=config.data.get(
            "validation_reaction_embeds_path",
            None,
        ),
        train_reaction_t5v2_embeds_path=config.data.get(
            "train_reaction_t5v2_embeds_path",
            None,
        ),
        validation_reaction_t5v2_embeds_path=config.data.get(
            "validation_reaction_t5v2_embeds_path",
            None,
        ),
        train_reaction_model_embeds_path=config.data.get(
            "train_reaction_model_embeds_path",
            None,
        ),
        validation_reaction_model_embeds_path=config.data.get(
            "validation_reaction_model_embeds_path",
            None,
        ),
        train_reaction_unimol2_embeds_path=config.data.get(
            "train_reaction_unimol2_embeds_path",
            None,
        ),
        validation_reaction_unimol2_embeds_path=config.data.get(
            "validation_reaction_unimol2_embeds_path",
            None,
        ),
        train_reaction_chiro_embeds_path=config.data.get(
            "train_reaction_chiro_embeds_path",
            None,
        ),
        validation_reaction_chiro_embeds_path=config.data.get(
            "validation_reaction_chiro_embeds_path",
            None,
        ),
        train_reaction_chirality_embeds_path=config.data.get(
            "train_reaction_chirality_embeds_path",
            None,
        ),
        validation_reaction_chirality_embeds_path=config.data.get(
            "validation_reaction_chirality_embeds_path",
            None,
        ),
        train_reaction_chienn_embeds_path=config.data.get(
            "train_reaction_chienn_embeds_path",
            None,
        ),
        validation_reaction_chienn_embeds_path=config.data.get(
            "validation_reaction_chienn_embeds_path",
            None,
        ),
        reaction_model_dim=config.data.get("reaction_model_dim", None),
        reaction_unimol_dim=config.data.get("reaction_unimol_dim", 768),
        reaction_chiro_dim=config.data.get("reaction_chiro_dim", None),
        reaction_chirality_dim=config.data.get("reaction_chirality_dim", None),
        reaction_chienn_dim=reaction_chirality_dim,
        reaction_use_chiro=config.data.get("reaction_use_chiro", None),
        reaction_use_chirality=config.data.get("reaction_use_chirality", None),
        reaction_use_chienn=reaction_use_chirality,
        reaction_allow_missing_unimol2=config.data.get(
            "reaction_allow_missing_unimol2",
            False,
        ),
        reaction_allow_missing_chiro=config.data.get("reaction_allow_missing_chiro", None),
        reaction_allow_missing_chirality=config.data.get(
            "reaction_allow_missing_chirality",
            None,
        ),
        reaction_allow_missing_chienn=config.data.get(
            "reaction_allow_missing_chienn",
            reaction_allow_missing_chirality,
        ),
        reaction_embedding_in_memory=config.data.get("reaction_embedding_in_memory", True),
        residue_dim=config.data.get("residue_dim", 1024),
        max_protein_tokens=config.data.get("max_protein_tokens", 1024),
        protein_truncation=config.data.get("protein_truncation", "ends_center"),
        standardize_reactions=config.data.get("standardize_reactions", True),
        standardize_hypervalent=config.data.get("standardize_hypervalent", True),
        standardize_remove_hs=config.data.get("standardize_remove_hs", True),
        standardize_kekulize=config.data.get("standardize_kekulize", False),
        standardize_uncharge=config.data.get("standardize_uncharge", True),
        standardize_metals=config.data.get("standardize_metals", True),
    )

    model = ReactionConditionedLitModule(
        query_encoder_dims=config.model.query_encoder_dims,
        target_encoder_dims=config.model.target_encoder_dims,
        embedding_dim=config.model.embedding_dim,
        residue_dim=config.data.get("residue_dim", 1024),
        attention_rank=pooling_config.get("attention_rank", None),
        projection_bias=pooling_config.get("projection_bias", False),
        score_mode=config.model.get("score_mode", "target_mlp"),
        value_projection_bias=pooling_config.get("value_projection_bias", False),
        normalize_pooled_values=pooling_config.get("normalize_pooled_values", True),
        return_attention=pooling_config.get("return_attention", False),
        query_encoder_type=config.model.get("query_encoder_type", "mlp"),
        reaction_model_dim=config.data.get("reaction_model_dim", 1024),
        reaction_unimol_dim=config.data.get("reaction_unimol_dim", 768),
        reaction_chienn_dim=reaction_chirality_dim,
        reaction_use_chienn=reaction_use_chirality,
        reaction_pooling=config.model.get("reaction_pooling", "attention"),
        reaction_attention_bias=config.model.get("reaction_attention_pooling", {}).get(
            "attention_bias",
            True,
        ),
        reaction_separate_side_poolers=config.model.get(
            "reaction_attention_pooling",
            {},
        ).get("separate_side_poolers", True),
        reaction_modality_attention_hidden_dim=reaction_multimodal_attention_config.get(
            "hidden_dim",
            None,
        ),
        reaction_modality_attention_dropout=reaction_multimodal_attention_config.get(
            "dropout",
            0.0,
        ),
        reaction_modality_dropout=reaction_multimodal_attention_config.get(
            "modality_dropout",
            0.0,
        ),
        reaction_modality_token_layer_norm=reaction_multimodal_attention_config.get(
            "token_layer_norm",
            reaction_multimodal_attention_config.get("token_normalization", False),
        ),
        reaction_modality_encoder_num_layers=reaction_multimodal_attention_config.get(
            "modality_encoder_num_layers",
            None,
        ),
        reaction_modality_encoder_widths=reaction_multimodal_attention_config.get(
            "modality_encoder_widths",
            None,
        ),
        reaction_modality_encoder_use_layer_norm=reaction_multimodal_attention_config.get(
            "modality_encoder_use_layer_norm",
            reaction_multimodal_attention_config.get("modality_encoder_layer_norm", False),
        ),
        reaction_modality_encoder_dropout=reaction_multimodal_attention_config.get(
            "modality_encoder_dropout",
            0.0,
        ),
        reaction_modality_encoder_normalise_output=reaction_multimodal_attention_config.get(
            "modality_encoder_normalise_output",
            False,
        ),
        query_encoder_checkpoint_path=query_encoder_checkpoint_path,
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
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    csv_logger = pl.loggers.CSVLogger(
        save_dir=config.logging.log_dir,
        name="reaction_conditioned_training",
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

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.logging.checkpoint_dir,
        filename="reaction-conditioned-{epoch:02d}",
        every_n_epochs=config.logging.get("save_every_n_epochs", 10),
        save_last=True,
        save_top_k=3,
        monitor="val/loss",
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    devices = config.training.get("devices", 1 if torch.cuda.is_available() else "auto")
    accelerator = config.training.get("accelerator", "auto")
    strategy = config.training.get("strategy", "auto")
    if torch.cuda.is_available() and devices not in (1, "1", "auto") and strategy == "auto":
        strategy = "ddp"

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config.logging.get("log_every_n_steps", 1),
        check_val_every_n_epoch=config.training.get("check_val_every_n_epoch", 10),
        enable_progress_bar=config.training.get("enable_progress_bar", True),
        deterministic=True,
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        num_nodes=config.training.get("num_nodes", 1),
        use_distributed_sampler=config.training.get("use_distributed_sampler", True),
        precision=config.training.get("precision", "32-true"),
    )

    print("=" * 80)
    print("STARTING REACTION-CONDITIONED TRAINING")
    print("=" * 80 + "\n")

    try:
        trainer.fit(model, datamodule=data_module, ckpt_path=args.resume)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        print(f"Last checkpoint saved to: {checkpoint_callback.last_model_path}")
        sys.exit(0)
    except Exception as exc:
        print(f"\nError during training: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 80)
    print("REACTION-CONDITIONED TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Last checkpoint: {checkpoint_callback.last_model_path}")
    print(f"Logs saved to: {config.logging.log_dir}")


if __name__ == "__main__":
    main()
