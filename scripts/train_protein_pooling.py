#!/usr/bin/env python3
"""
Train a Horizyn dual encoder with residue-level protein pooling.
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
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from horizyn.reaction_conditioned_data_module import ReactionConditionedDataModule
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


def _config_first(section, *names: str, default=None):
    for name in names:
        value = section.get(name, None)
        if value is not None:
            return value
    return default


def _load_partial_model_warm_start(
    module: ProteinPooledLitModule,
    checkpoint_path: str | Path | None,
    *,
    capability_gate_bias: float = -2.0,
) -> None:
    """Warm-start matching model weights without restoring trainer state.

    This is intentionally different from ``--resume``. It lets a capability
    branch model start from a trained three-branch retrieval checkpoint while
    skipping or adapting the new capability-specific parameters.
    """
    if checkpoint_path is None or str(checkpoint_path) == "":
        return

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Warm-start checkpoint must contain a dict: {checkpoint_path}")

    source_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(source_state, dict):
        raise ValueError(f"Warm-start checkpoint has no state_dict: {checkpoint_path}")

    current_state = module.state_dict()
    updated_state = dict(current_state)
    loaded: list[str] = []
    partially_loaded: list[str] = []
    skipped_shape: list[str] = []

    for key, value in source_state.items():
        if not key.startswith("model."):
            continue
        if key not in current_state:
            continue
        current_value = current_state[key]
        if not torch.is_tensor(value) or not torch.is_tensor(current_value):
            continue
        if value.shape == current_value.shape:
            updated_state[key] = value
            loaded.append(key)
            continue

        # Upgrade the old raw/SLEEC/Lorentz gate into the new
        # raw/SLEEC/Lorentz/capability gate without perturbing the old branches.
        if key == "model.enzyme_feature_fusion.gate.0.weight":
            if (
                value.ndim == 2
                and current_value.ndim == 2
                and value.shape[0] == current_value.shape[0]
                and value.shape[1] < current_value.shape[1]
            ):
                new_value = current_value.clone()
                new_value[:, : value.shape[1]] = value
                new_value[:, value.shape[1] :] = 0
                updated_state[key] = new_value
                partially_loaded.append(key)
                continue
        if key == "model.enzyme_feature_fusion.gate.3.weight":
            if (
                value.ndim == 2
                and current_value.ndim == 2
                and value.shape[1] == current_value.shape[1]
                and value.shape[0] < current_value.shape[0]
            ):
                new_value = current_value.clone()
                new_value[: value.shape[0], :] = value
                new_value[value.shape[0] :, :] = 0
                updated_state[key] = new_value
                partially_loaded.append(key)
                continue
        if key == "model.enzyme_feature_fusion.gate.3.bias":
            if (
                value.ndim == 1
                and current_value.ndim == 1
                and value.shape[0] < current_value.shape[0]
            ):
                new_value = current_value.clone()
                new_value[: value.shape[0]] = value
                new_value[value.shape[0] :] = float(capability_gate_bias)
                updated_state[key] = new_value
                partially_loaded.append(key)
                continue

        skipped_shape.append(
            f"{key}: checkpoint={tuple(value.shape)} current={tuple(current_value.shape)}"
        )

    training_stage = getattr(module, "training_stage", None)
    if training_stage in {"e2r_adapter", "r2e_adapter", "bidirectional_adapters"}:
        adapter_prefixes = (
            ("model.e2r_adapter.", "model.r2e_adapter.")
            if training_stage == "bidirectional_adapters"
            else (f"model.{training_stage}.",)
        )
        required_base_keys = {
            key
            for key in current_state
            if key.startswith("model.") and not key.startswith(adapter_prefixes)
        }
        loaded_base_keys = set(loaded) | set(partially_loaded)
        missing_base_keys = sorted(required_base_keys - loaded_base_keys)
        if missing_base_keys:
            preview = "\n".join(f"  {key}" for key in missing_base_keys[:20])
            raise ValueError(
                f"{training_stage} warm start did not exactly restore the frozen parent "
                f"model ({len(missing_base_keys)} missing keys):\n{preview}"
            )
        base_shape_skips = [
            value for value in skipped_shape if not value.startswith(adapter_prefixes)
        ]
        if base_shape_skips:
            preview = "\n".join(f"  {value}" for value in base_shape_skips[:20])
            raise ValueError(
                f"{training_stage} warm start has incompatible frozen-parent shapes:\n" f"{preview}"
            )

    missing, unexpected = module.load_state_dict(updated_state, strict=False)
    rank_zero_print(
        "Warm-started model from "
        f"{checkpoint_path}: loaded={len(loaded)}, partial={len(partially_loaded)}, "
        f"shape_skipped={len(skipped_shape)}, missing={len(missing)}, "
        f"unexpected={len(unexpected)}"
    )
    if partially_loaded:
        rank_zero_print("Partially adapted warm-start keys:")
        for key in partially_loaded:
            rank_zero_print(f"  {key}")
    if skipped_shape:
        rank_zero_print("Shape-skipped warm-start keys:")
        for key in skipped_shape[:20]:
            rank_zero_print(f"  {key}")
        if len(skipped_shape) > 20:
            rank_zero_print(f"  ... {len(skipped_shape) - 20} more")


def _load_biofp_pretrain_warm_start(
    module: ProteinPooledLitModule,
    checkpoint_path: str | Path | None,
) -> None:
    """Warm-start only enzyme BioFP split components from enzyme-only pretraining."""
    if checkpoint_path is None or str(checkpoint_path) == "":
        return

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"BioFP pretrain checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"BioFP pretrain checkpoint must contain a dict: {checkpoint_path}")

    source_state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(source_state, dict):
        raise ValueError(f"BioFP pretrain checkpoint has no state_dict: {checkpoint_path}")

    allowed_prefixes = (
        "model.pooling.",
        "model.raw_mean_pooling.",
        "model.biofp_split_encoder.",
        "model.biological_factorized_encoder.",
        "model.hyperbolic_projector.",
    )
    current_state = module.state_dict()
    updated_state = dict(current_state)
    loaded: list[str] = []
    skipped_shape: list[str] = []

    for key, value in source_state.items():
        if not isinstance(key, str):
            continue
        if key.startswith("model.model."):
            key = "model." + key.removeprefix("model.model.")
        if not key.startswith(allowed_prefixes):
            continue
        if key not in current_state:
            continue
        current_value = current_state[key]
        if not torch.is_tensor(value) or not torch.is_tensor(current_value):
            continue
        if value.shape == current_value.shape:
            updated_state[key] = value
            loaded.append(key)
        else:
            skipped_shape.append(
                f"{key}: checkpoint={tuple(value.shape)} current={tuple(current_value.shape)}"
            )

    missing, unexpected = module.load_state_dict(updated_state, strict=False)
    rank_zero_print(
        "Warm-started BioFP enzyme stack from "
        f"{checkpoint_path}: loaded={len(loaded)}, shape_skipped={len(skipped_shape)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if skipped_shape:
        rank_zero_print("Shape-skipped BioFP warm-start keys:")
        for key in skipped_shape[:20]:
            rank_zero_print(f"  {key}")
        if len(skipped_shape) > 20:
            rank_zero_print(f"  ... {len(skipped_shape) - 20} more")


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

    for key in ("validation_pairs_path", "validation_reactions_path"):
        value = config.data.get(key, None)
        if value is not None and any(part.lower() == "test" for part in Path(str(value)).parts):
            raise ValueError(f"{key} must not point at the held-out test subset: {value}")

    if (
        "check_val_every_n_epoch" not in config.training
        and config.training.get("validation_interval_steps", None) is None
    ):
        config.training.check_val_every_n_epoch = 1
    pooling_config = config.model.get("protein_attention_pooling", {})
    sleec_config = config.model.get("sleec_pooling", {})
    enzyme_fusion_config = config.model.get("enzyme_fusion", {})
    capability_config = config.model.get("capability_vector", {})
    factorized_capability_config = config.model.get("factorized_capability_vector", {})
    biofp_config = config.model.get("biofp", {})
    text_config = config.model.get("text_vector", {})
    hyperbolic_config = config.model.get("hyperbolic_encoder", {})
    enzyme_block_config = config.model.get("enzyme_block_fusion", {})
    reaction_attention_regularization_config = config.training.get(
        "reaction_attention_regularization",
        {},
    )
    reaction_chemistry_consistency_config = config.training.get(
        "reaction_chemistry_consistency",
        {},
    )
    reaction_residual_identity_config = config.training.get(
        "reaction_residual_identity",
        {},
    )
    reaction_hyperbolic_config = config.model.get("reaction_hyperbolic_encoder", {})
    reaction_multimodal_attention_config = config.model.get(
        "reaction_multimodal_attention",
        {},
    )
    reaction_fingerprint_attention_config = config.model.get(
        "reaction_fingerprint_attention",
        {},
    )
    e2r_adapter_config = config.model.get("e2r_adapter", {})
    r2e_adapter_config = config.model.get("r2e_adapter", {})
    reaction_chirality_path = _config_first(
        config.data,
        "reaction_chiro_embeds_path",
        "reaction_chirality_embeds_path",
        "reaction_chienn_embeds_path",
        default=None,
    )
    reaction_chirality_dim = _config_first(
        config.data,
        "reaction_chiro_dim",
        "reaction_chirality_dim",
        "reaction_chienn_dim",
        default=256,
    )
    reaction_use_chirality = _config_first(
        config.model,
        "reaction_use_chiro",
        "reaction_use_chirality",
        "reaction_use_chienn",
        default=_config_first(
            config.data,
            "reaction_use_chiro",
            "reaction_use_chirality",
            "reaction_use_chienn",
            default=True,
        ),
    )
    reaction_allow_missing_chirality = _config_first(
        config.data,
        "reaction_allow_missing_chiro",
        "reaction_allow_missing_chirality",
        "reaction_allow_missing_chienn",
        default=False,
    )
    query_encoder_checkpoint_path = config.model.get("query_encoder_checkpoint_path", None)
    wandb_settings = resolve_wandb_settings(args, config)
    print("\n" + "=" * 80)
    print("PROTEIN-POOLING HORIZYN TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Seed: {config.seed}")
    print(f"Max Epochs: {config.training.max_epochs}")
    print(f"Train Batch Size: {config.data.train_batch_size}")
    print(f"Residue HDF5: {config.data.protein_residue_embeds_path}")
    if config.data.get("protein_score_residue_embeds_path", None):
        print(f"Score Residue HDF5: {config.data.protein_score_residue_embeds_path}")
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
    print(f"Pooling: {config.model.get('pooling', 'mean')}")
    print(f"Enzyme Input Mode: {config.model.get('enzyme_input_mode', 'standard')}")
    if config.model.get("enzyme_input_mode", "standard") in {
        "raw_mean_sleec_hyperbolic_gated",
        "raw_mean_sleec_hyperbolic_capability_gated",
        "raw_mean_sleec_hyperbolic_text_gated",
        "raw_mean_sleec_blockwise",
        "raw_mean_sleec_hyperbolic_blockwise",
        "raw_mean_sleec_hyperbolic_capability_blockwise",
        "raw_mean_sleec_biofp_split",
    }:
        if config.model.get("enzyme_input_mode", "standard") == "raw_mean_sleec_biofp_split":
            print("Enzyme Fusion: raw mean + pooled + inline predicted BioFP split")
            print(f"BioFP Targets NPZ: {config.data.get('protein_biofp_targets_path', None)}")
        elif config.model.get("enzyme_input_mode", "standard").endswith("blockwise"):
            print("Enzyme Fusion: blockwise raw mean + SLEEC site + optional EC/capability")
            print(f"Enzyme Block Dims: {enzyme_block_config.get('dims', {})}")
            print(f"Enzyme Block Weights: {enzyme_block_config.get('weights', {})}")
            if "capability" in enzyme_block_config.get("dims", {}):
                print(
                    f"Capability Vector HDF5/NPZ: {config.data.get('protein_capability_vectors_path', None)}"
                )
        elif config.model.get("enzyme_input_mode", "standard").endswith("capability_gated"):
            print("Enzyme Fusion: gated raw mean + pooled + hyperbolic tangent + capability")
            print(
                f"Capability Vector HDF5/NPZ: {config.data.get('protein_capability_vectors_path', None)}"
            )
            print(f"Capability Adapter: {capability_config.get('adapter', False)}")
        elif config.model.get("enzyme_input_mode", "standard").endswith("text_gated"):
            print("Enzyme Fusion: gated raw mean + pooled + hyperbolic tangent + TIGER text")
            print(f"Text Vector HDF5/NPZ: {config.data.get('protein_text_vectors_path', None)}")
            print(f"Text Vector Dim: {text_config.get('dim', 768)}")
            print(f"Text Fusion Dim: {text_config.get('fusion_dim', 512)}")
            print(f"Text Fusion Heads: {text_config.get('num_heads', 8)}")
            print(f"Text Adapter: {text_config.get('adapter', False)}")
        else:
            print("Enzyme Fusion: gated raw mean + pooled + hyperbolic tangent")
        print(f"Enzyme Fusion Hidden Dim: {enzyme_fusion_config.get('hidden_dim', None)}")
        print(f"Enzyme Fusion Dropout: {enzyme_fusion_config.get('dropout', 0.0)}")
    print(f"Query Encoder Type: {config.model.get('query_encoder_type', 'mlp')}")
    print(f"Reaction Pooling: {config.model.get('reaction_pooling', 'attention')}")
    if query_encoder_checkpoint_path:
        print(f"Query Encoder Checkpoint: {query_encoder_checkpoint_path}")
    print(f"Attention Bias: {pooling_config.get('attention_bias', True)}")
    if config.model.get("pooling", "mean") == "sleec":
        print(f"SLEEC Pooling Mode: {sleec_config.get('mode', 'topk')}")
        print(f"SLEEC Top-K Fraction: {sleec_config.get('topk_fraction', 0.2)}")
        print(f"SLEEC Threshold: {sleec_config.get('threshold', 0.5)}")
        print(f"SLEEC Score Source: {sleec_config.get('score_embedding_source', 'same')}")
        if sleec_config.get("checkpoint_path", None):
            print(f"SLEEC Stage-1 Checkpoint: {sleec_config.checkpoint_path}")
            print(f"SLEEC Freeze Scorer: {sleec_config.get('freeze_scorer', False)}")
        print(f"SLEEC Residue Loss Weight: {config.training.get('lambda_residue', 0.0)}")
    if config.model.get("pooling", "mean") == "sleec_guided_attention":
        print(f"SLEEC-Guided Threshold: {sleec_config.get('threshold', 0.34)}")
        print(f"SLEEC-Guided Freeze Scorer: {sleec_config.get('freeze_scorer', True)}")
    if hyperbolic_config.get("checkpoint_path", None):
        print(f"Hyperbolic Encoder Checkpoint: {hyperbolic_config.checkpoint_path}")
        print(f"Hyperbolic Hyp Dim: {hyperbolic_config.get('hyp_dim', None)}")
        print(f"Hyperbolic Use Tangent: {hyperbolic_config.get('use_tangent', True)}")
        print(f"Hyperbolic Freeze Projector: {hyperbolic_config.get('freeze_projector', False)}")
        print(
            "Hyperbolic Load Attention Pooler: "
            f"{hyperbolic_config.get('load_attention_pooler', False)}"
        )
    if reaction_hyperbolic_config.get("checkpoint_path", None):
        print(f"Reaction Hyperbolic Checkpoint: {reaction_hyperbolic_config.checkpoint_path}")
        print(f"Reaction Hyperbolic Hyp Dim: {reaction_hyperbolic_config.get('hyp_dim', None)}")
        print(
            "Reaction Hyperbolic Use Tangent: "
            f"{reaction_hyperbolic_config.get('use_tangent', True)}"
        )
        print(
            "Reaction Hyperbolic Freeze Encoder: "
            f"{reaction_hyperbolic_config.get('freeze_encoder', True)}"
        )
        print(
            "Reaction Hyperbolic Freeze Projector: "
            f"{reaction_hyperbolic_config.get('freeze_projector', True)}"
        )
    if reaction_fingerprint_attention_config.get("enabled", False):
        print("Reaction Fingerprint Attention: enabled")
        print(
            "Reaction Fingerprint Attention Token Dim: "
            f"{reaction_fingerprint_attention_config.get('token_dim', 512)}"
        )
        print(
            "Reaction Fingerprint Attention Hidden Dim: "
            f"{reaction_fingerprint_attention_config.get('hidden_dim', 512)}"
        )
        print(
            "Reaction Fingerprint Attention Dropout: "
            f"{reaction_fingerprint_attention_config.get('dropout', 0.0)}"
        )
    if config.model.get("query_encoder_type", "mlp") == "multimodal_reaction_attention":
        print("Reaction Multimodal Attention: enabled")
        print(f"Reaction Model Dim: {config.data.get('reaction_model_dim', 1024)}")
        print(f"Reaction Uni-Mol2 Dim: {config.data.get('reaction_unimol_dim', 768)}")
        print(f"Reaction ChIRo Dim: {reaction_chirality_dim}")
        print(f"Reaction Allow Missing ChIRo: {reaction_allow_missing_chirality}")
        print(
            "Reaction Modality Attention Hidden Dim: "
            f"{reaction_multimodal_attention_config.get('hidden_dim', config.model.query_encoder_dims[0])}"
        )
        print(
            "Reaction Modality Attention Dropout: "
            f"{reaction_multimodal_attention_config.get('dropout', 0.0)}"
        )
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
    print(f"Query Encoder: {config.model.query_encoder_dims}")
    print(f"Target Encoder: {config.model.target_encoder_dims}")
    print(f"Embedding Dim: {config.model.embedding_dim}")
    print(f"Loss: {config.training.loss.get('name', 'FullBatchMLNCELoss')}")
    print(
        "Positive Pair Source: "
        f"{config.training.loss.get('positive_pair_source', 'observed_pairs')}"
    )
    if config.training.loss.get("name", "FullBatchMLNCELoss") == "MultiAlignmentRetrievalLoss":
        print(
            "Direction Loss Weights: "
            f"R2E={config.training.loss.get('lambda_r2e', 0.5)}, "
            f"E2R={config.training.loss.get('lambda_e2r', 0.5)}"
        )
        print(
            "R2E Hard Negatives: "
            f"lambda={config.training.loss.get('lambda_r2e_hard_neg', 0.0)}, "
            f"top_k={config.training.loss.get('r2e_hard_neg_top_k', 0)}, "
            f"margin={config.training.loss.get('r2e_hard_neg_margin', 0.0)}"
        )
    print(f"Log Dir: {config.logging.log_dir}")
    print(f"Checkpoint Dir: {config.logging.checkpoint_dir}")
    print(f"Checkpoint Monitor: {config.logging.get('checkpoint_monitor', 'val/loss')}")
    print(f"Checkpoint Mode: {config.logging.get('checkpoint_mode', 'min')}")
    print(f"W&B Enabled: {wandb_settings['enabled']} ({wandb_settings['mode']})")
    print(f"Accelerator: {config.training.get('accelerator', 'auto')}")
    print(f"Devices: {config.training.get('devices', 1 if torch.cuda.is_available() else 'auto')}")
    print(f"Strategy: {config.training.get('strategy', 'auto')}")
    print(f"Precision: {config.training.get('precision', '32-true')}")
    if config.training.get("validation_interval_steps", None) is not None:
        print(
            "Validation Frequency: every "
            f"{config.training.validation_interval_steps} optimizer steps"
        )
    else:
        print(
            "Validation Frequency: every "
            f"{config.training.get('check_val_every_n_epoch', 10)} epoch"
        )
    print(
        "Validation Retrieval Metrics: "
        f"{config.training.get('validation_retrieval_metrics', False)}"
    )
    if config.training.get("validation_retrieval_metrics", False):
        print(
            "Validation Retrieval Directions: "
            f"{config.training.get('validation_retrieval_directions', ['reaction_to_enzyme', 'enzyme_to_reaction'])}"
        )
        print(
            "Validation Retrieval Candidate Set: "
            f"{config.training.get('validation_retrieval_candidate_set', config.data.get('validation_retrieval_candidate_set', 'validation'))}"
        )
        if (
            config.training.get(
                "validation_retrieval_candidate_set",
                config.data.get("validation_retrieval_candidate_set", "validation"),
            )
            == "custom"
        ):
            print(
                "Validation Retrieval Candidate IDs: "
                f"{config.training.get('validation_retrieval_candidate_ids_path', config.data.get('validation_retrieval_candidate_ids_path', None))}"
            )
        print(
            "Validation Retrieval Batch Size: "
            f"{config.training.get('validation_retrieval_batch_size', config.data.get('validation_retrieval_batch_size', config.data.get('retrieval_batch_size', 1)))}"
        )
    if config.data.get("hard_negative_pools_path", None):
        print(f"Hard-Negative Pools: {config.data.hard_negative_pools_path}")
        print(
            "Hard-Negative Batch Sampler: "
            f"anchors={config.data.get('hard_negative_anchor_queries_per_batch', 48)}, "
            f"pos/query={config.data.get('hard_negative_positives_per_query', 2)}, "
            f"neg/query={config.data.get('hard_negative_negatives_per_query', 8)}"
        )
    print("=" * 80 + "\n")

    pl.seed_everything(config.get("seed", 42), workers=True)
    if "float32_matmul_precision" in config.training:
        torch.set_float32_matmul_precision(config.training.float32_matmul_precision)

    data_module = ReactionConditionedDataModule(
        train_pairs_path=config.data.train_pairs_path,
        test_pairs_path=config.data.get(
            "validation_pairs_path",
            config.data.get("test_pairs_path", None),
        ),
        train_reactions_path=config.data.train_reactions_path,
        test_reactions_path=config.data.get(
            "validation_reactions_path",
            config.data.get("test_reactions_path", None),
        ),
        protein_residue_embeds_path=config.data.protein_residue_embeds_path,
        protein_score_residue_embeds_path=config.data.get(
            "protein_score_residue_embeds_path",
            None,
        ),
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
        reaction_chemistry_vectors_path=config.data.get(
            "reaction_chemistry_vectors_path",
            None,
        ),
        train_reaction_chemistry_vectors_path=config.data.get(
            "train_reaction_chemistry_vectors_path",
            None,
        ),
        validation_reaction_chemistry_vectors_path=config.data.get(
            "validation_reaction_chemistry_vectors_path",
            None,
        ),
        reaction_directional_vectors_path=config.data.get(
            "reaction_directional_vectors_path",
            None,
        ),
        train_reaction_directional_vectors_path=config.data.get(
            "train_reaction_directional_vectors_path",
            None,
        ),
        validation_reaction_directional_vectors_path=config.data.get(
            "validation_reaction_directional_vectors_path",
            None,
        ),
        reaction_model_dim=config.data.get("reaction_model_dim", None),
        reaction_unimol_dim=config.data.get("reaction_unimol_dim", 768),
        reaction_chiro_dim=config.data.get("reaction_chiro_dim", None),
        reaction_chirality_dim=config.data.get("reaction_chirality_dim", None),
        reaction_chienn_dim=reaction_chirality_dim,
        reaction_chemistry_dim=config.data.get("reaction_chemistry_dim", None),
        reaction_directional_dim=config.data.get("reaction_directional_dim", None),
        reaction_use_model=config.data.get("reaction_use_model", True),
        reaction_use_chiro=config.data.get("reaction_use_chiro", None),
        reaction_use_chirality=config.data.get("reaction_use_chirality", None),
        reaction_use_chienn=reaction_use_chirality,
        reaction_use_chemistry=config.data.get("reaction_use_chemistry", False),
        reaction_use_directional=config.data.get("reaction_use_directional", False),
        reaction_load_directional=config.data.get("reaction_load_directional", False),
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
        reaction_allow_missing_chemistry=config.data.get(
            "reaction_allow_missing_chemistry",
            True,
        ),
        reaction_allow_missing_directional=config.data.get(
            "reaction_allow_missing_directional",
            True,
        ),
        reaction_embedding_in_memory=config.data.get("reaction_embedding_in_memory", True),
        residue_dim=config.data.get("residue_dim", 1024),
        score_residue_dim=config.data.get("score_residue_dim", None),
        max_protein_tokens=config.data.get("max_protein_tokens", 1024),
        protein_truncation=config.data.get("protein_truncation", "ends_center"),
        standardize_reactions=config.data.get("standardize_reactions", True),
        standardize_hypervalent=config.data.get("standardize_hypervalent", True),
        standardize_remove_hs=config.data.get("standardize_remove_hs", True),
        standardize_kekulize=config.data.get("standardize_kekulize", False),
        standardize_uncharge=config.data.get("standardize_uncharge", True),
        standardize_metals=config.data.get("standardize_metals", True),
        normalize_molecule_sets_as_self_reactions=config.data.get(
            "normalize_molecule_sets_as_self_reactions",
            False,
        ),
        enzyme_ec_labels_path=config.data.get("enzyme_ec_labels_path", None),
        protein_capability_vectors_path=config.data.get("protein_capability_vectors_path", None),
        protein_capability_metadata_path=config.data.get(
            "protein_capability_metadata_path",
            None,
        ),
        capability_vector_in_memory=config.data.get("capability_vector_in_memory", True),
        capability_missing_policy=config.data.get(
            "capability_missing_policy",
            config.model.get("capability_vector", {}).get("missing_policy", "zero_with_mask"),
        ),
        protein_factorized_capability_vectors_path=config.data.get(
            "protein_factorized_capability_vectors_path",
            None,
        ),
        factorized_capability_missing_policy=config.data.get(
            "factorized_capability_missing_policy",
            config.model.get("factorized_capability_vector", {}).get(
                "missing_policy",
                "zero_with_mask",
            ),
        ),
        protein_text_vectors_path=config.data.get("protein_text_vectors_path", None),
        protein_text_metadata_path=config.data.get("protein_text_metadata_path", None),
        text_vector_missing_policy=config.data.get(
            "text_vector_missing_policy",
            config.model.get("text_vector", {}).get("missing_policy", "zero_with_mask"),
        ),
        protein_biofp_targets_path=config.data.get("protein_biofp_targets_path", None),
        protein_biofp_vocab_path=config.data.get("protein_biofp_vocab_path", None),
        biofp_missing_policy=config.data.get(
            "biofp_missing_policy",
            config.model.get("biofp", {}).get("missing_policy", "zero_with_mask"),
        ),
        validation_retrieval_metrics=config.training.get(
            "validation_retrieval_metrics",
            False,
        ),
        validation_retrieval_candidate_set=config.training.get(
            "validation_retrieval_candidate_set",
            config.data.get("validation_retrieval_candidate_set", "validation"),
        ),
        validation_retrieval_candidate_ids_path=config.training.get(
            "validation_retrieval_candidate_ids_path",
            config.data.get("validation_retrieval_candidate_ids_path", None),
        ),
        validation_retrieval_batch_size=config.training.get(
            "validation_retrieval_batch_size",
            config.data.get(
                "validation_retrieval_batch_size", config.data.get("retrieval_batch_size", 1)
            ),
        ),
        validation_retrieval_directions=config.training.get(
            "validation_retrieval_directions",
            config.data.get("validation_retrieval_directions", None),
        ),
        hard_negative_pools_path=config.data.get("hard_negative_pools_path", None),
        hard_negative_direction=config.data.get(
            "hard_negative_direction",
            "reaction_to_enzyme",
        ),
        hard_negative_anchor_queries_per_batch=config.data.get(
            "hard_negative_anchor_queries_per_batch",
            48,
        ),
        hard_negative_positives_per_query=config.data.get(
            "hard_negative_positives_per_query",
            2,
        ),
        hard_negative_negatives_per_query=config.data.get(
            "hard_negative_negatives_per_query",
            8,
        ),
        hard_negative_seed=config.data.get("hard_negative_seed", config.get("seed", 42)),
        reaction_balanced_sampling=(
            config.data.get("train_sampler", {}).get("name", "shuffle")
            == "reaction_degree_balanced"
        ),
        reaction_balanced_degree_exponent=config.data.get("train_sampler", {}).get(
            "degree_exponent",
            0.5,
        ),
        reaction_balanced_seed=config.data.get("train_sampler", {}).get(
            "seed",
            config.get("seed", 42),
        ),
        reaction_direction_mode=config.data.get(
            "reaction_direction_mode",
            "bidirectional",
        ),
    )

    model = ProteinPooledLitModule(
        query_encoder_dims=config.model.query_encoder_dims,
        target_encoder_dims=config.model.target_encoder_dims,
        embedding_dim=config.model.embedding_dim,
        residue_dim=config.data.get("residue_dim", 1024),
        pooling=config.model.get("pooling", "mean"),
        attention_bias=pooling_config.get("attention_bias", True),
        sleec_mode=sleec_config.get("mode", "topk"),
        sleec_topk_fraction=sleec_config.get("topk_fraction", 0.2),
        sleec_threshold=sleec_config.get("threshold", 0.5),
        sleec_scorer_hidden_dim=sleec_config.get("scorer_hidden_dim", 256),
        sleec_score_hidden_dim=sleec_config.get(
            "score_hidden_dim",
            config.data.get("score_residue_dim", None),
        ),
        sleec_checkpoint_path=sleec_config.get("checkpoint_path", None),
        sleec_freeze_scorer=sleec_config.get("freeze_scorer", False),
        sleec_guided_initial_bias_scale=sleec_config.get("initial_bias_scale", 1.0),
        sleec_guided_train_bias_scale=sleec_config.get("train_bias_scale", True),
        hyperbolic_checkpoint_path=hyperbolic_config.get("checkpoint_path", None),
        hyperbolic_freeze_projector=hyperbolic_config.get("freeze_projector", False),
        hyperbolic_use_tangent=hyperbolic_config.get("use_tangent", True),
        hyperbolic_hyp_dim=hyperbolic_config.get("hyp_dim", None),
        hyperbolic_load_attention_pooler=hyperbolic_config.get(
            "load_attention_pooler",
            False,
        ),
        hyperbolic_freeze_attention_pooler=hyperbolic_config.get(
            "freeze_attention_pooler",
            False,
        ),
        enzyme_input_mode=config.model.get("enzyme_input_mode", "standard"),
        enzyme_fusion_hidden_dim=enzyme_fusion_config.get("hidden_dim", None),
        enzyme_fusion_dropout=enzyme_fusion_config.get("dropout", 0.0),
        enzyme_block_dims=enzyme_block_config.get("dims", None),
        enzyme_block_weights=enzyme_block_config.get("weights", None),
        enzyme_block_dropout=enzyme_block_config.get("dropout", 0.0),
        enzyme_block_learned_weights=enzyme_block_config.get(
            "learned_weights",
            False,
        ),
        capability_vector_dim=capability_config.get(
            "dim",
            config.model.get("capability_vector_dim", 256),
        ),
        capability_freeze=capability_config.get(
            "freeze",
            config.model.get("capability_freeze", True),
        ),
        capability_adapter=capability_config.get(
            "adapter",
            config.model.get("capability_adapter", False),
        ),
        capability_dropout=capability_config.get(
            "dropout",
            config.model.get("capability_dropout", 0.1),
        ),
        factorized_capability_dims=factorized_capability_config.get("dims", None),
        factorized_capability_freeze=factorized_capability_config.get(
            "freeze",
            True,
        ),
        factorized_capability_use_masks=factorized_capability_config.get(
            "use_masks",
            True,
        ),
        biofp_center_dim=biofp_config.get("center_dim", 0),
        biofp_cofactor_dim=biofp_config.get("cofactor_dim", 0),
        biofp_transition_dim=biofp_config.get("transition_dim", 0),
        biofp_family_dims=biofp_config.get("family_dims", None),
        biofp_seq_dim=biofp_config.get("seq_dim", 384),
        biofp_dim=biofp_config.get("dim", 128),
        biofp_hidden_dim=biofp_config.get("hidden_dim", 512),
        biofp_seq_weight=biofp_config.get("seq_weight", 0.75),
        biofp_dropout=biofp_config.get("dropout", 0.1),
        biofp_aux_weight=config.training.loss.get("biofp_aux_weight", 0.0),
        biofp_center_weight=config.training.loss.get("biofp_center_weight", 0.45),
        biofp_transition_weight=config.training.loss.get("biofp_transition_weight", 0.35),
        biofp_cofactor_weight=config.training.loss.get("biofp_cofactor_weight", 0.20),
        biofp_family_weights=config.training.loss.get("biofp_family_weights", None),
        biofp_confidence_cap=config.training.loss.get("biofp_confidence_cap", 8.0),
        reaction_attention_entropy_weight=reaction_attention_regularization_config.get(
            "weight",
            0.0,
        ),
        reaction_attention_min_normalized_entropy=(
            reaction_attention_regularization_config.get(
                "min_normalized_entropy",
                0.75,
            )
        ),
        reaction_chemistry_consistency_weight=(
            reaction_chemistry_consistency_config.get("weight", 0.0)
        ),
        reaction_residual_identity_weight=(
            reaction_residual_identity_config.get("weight", 0.0)
        ),
        enzyme_block_weight_kl_weight=enzyme_block_config.get(
            "weight_kl_weight",
            0.0,
        ),
        text_vector_dim=text_config.get(
            "dim",
            config.model.get("text_vector_dim", 768),
        ),
        text_fusion_dim=text_config.get(
            "fusion_dim",
            config.model.get("text_fusion_dim", 512),
        ),
        text_num_heads=text_config.get(
            "num_heads",
            config.model.get("text_num_heads", 8),
        ),
        text_dropout=text_config.get(
            "dropout",
            config.model.get("text_dropout", 0.1),
        ),
        text_freeze=text_config.get(
            "freeze",
            config.model.get("text_freeze", True),
        ),
        text_adapter=text_config.get(
            "adapter",
            config.model.get("text_adapter", False),
        ),
        reaction_hyperbolic_checkpoint_path=reaction_hyperbolic_config.get(
            "checkpoint_path",
            None,
        ),
        reaction_hyperbolic_hyp_dim=reaction_hyperbolic_config.get("hyp_dim", None),
        reaction_hyperbolic_freeze_encoder=reaction_hyperbolic_config.get(
            "freeze_encoder",
            True,
        ),
        reaction_hyperbolic_freeze_projector=reaction_hyperbolic_config.get(
            "freeze_projector",
            True,
        ),
        reaction_hyperbolic_use_tangent=reaction_hyperbolic_config.get("use_tangent", True),
        reaction_fingerprint_attention_enabled=reaction_fingerprint_attention_config.get(
            "enabled",
            False,
        ),
        reaction_fingerprint_attention_input_dim=(
            config.data.get("rdkit_fp_dim", 1024) + config.data.get("drfp_dim", 1024)
        ),
        reaction_fingerprint_attention_rdkit_dim=config.data.get("rdkit_fp_dim", 1024),
        reaction_fingerprint_attention_drfp_dim=config.data.get("drfp_dim", 1024),
        reaction_fingerprint_attention_token_dim=reaction_fingerprint_attention_config.get(
            "token_dim",
            512,
        ),
        reaction_fingerprint_attention_hidden_dim=reaction_fingerprint_attention_config.get(
            "hidden_dim",
            512,
        ),
        reaction_fingerprint_attention_dropout=reaction_fingerprint_attention_config.get(
            "dropout",
            0.0,
        ),
        reaction_fingerprint_attention_bias=reaction_fingerprint_attention_config.get(
            "attention_bias",
            True,
        ),
        e2r_adapter_enabled=e2r_adapter_config.get("enabled", False),
        e2r_adapter_hidden_dim=e2r_adapter_config.get("hidden_dim", 512),
        e2r_adapter_dropout=e2r_adapter_config.get("dropout", 0.1),
        e2r_adapter_gate_init=e2r_adapter_config.get("gate_init", 0.1),
        e2r_adapter_use_factorized_inputs=e2r_adapter_config.get(
            "use_factorized_inputs",
            False,
        ),
        e2r_adapter_use_directional_inputs=e2r_adapter_config.get(
            "use_directional_inputs",
            False,
        ),
        e2r_adapter_directional_dim=e2r_adapter_config.get(
            "directional_dim",
            config.data.get("reaction_directional_dim", None),
        ),
        e2r_adapter_directional_hidden_dim=e2r_adapter_config.get(
            "directional_hidden_dim",
            128,
        ),
        e2r_identity_weight=config.training.loss.get("e2r_identity_weight", 0.0),
        r2e_adapter_enabled=r2e_adapter_config.get("enabled", False),
        r2e_adapter_hidden_dim=r2e_adapter_config.get("hidden_dim", 512),
        r2e_adapter_dropout=r2e_adapter_config.get("dropout", 0.1),
        r2e_adapter_gate_init=r2e_adapter_config.get("gate_init", 0.1),
        r2e_adapter_use_factorized_inputs=r2e_adapter_config.get(
            "use_factorized_inputs",
            False,
        ),
        r2e_adapter_block_dims=r2e_adapter_config.get(
            "block_dims",
            enzyme_block_config.get("dims", None),
        ),
        r2e_adapter_block_weights=r2e_adapter_config.get(
            "block_weights",
            enzyme_block_config.get("weights", None),
        ),
        r2e_identity_weight=config.training.loss.get("r2e_identity_weight", 0.0),
        query_normalise_output=config.model.get(
            "query_normalise_output",
            config.model.get("normalise_output", True),
        ),
        target_normalise_output=config.model.get(
            "target_normalise_output",
            config.model.get("normalise_output", True),
        ),
        enforce_normalisation=config.model.get(
            "enforce_normalisation",
            config.model.get("normalise_output", True),
        ),
        embedding_similarity=config.training.get(
            "embedding_similarity",
            config.training.loss.get("embedding_similarity", "cosine"),
        ),
        validation_similarity=config.training.get(
            "validation_similarity",
            config.training.get(
                "embedding_similarity",
                config.training.loss.get("embedding_similarity", "cosine"),
            ),
        ),
        query_encoder_type=config.model.get("query_encoder_type", "mlp"),
        reaction_model_dim=config.data.get("reaction_model_dim", 1024),
        reaction_unimol_dim=config.data.get("reaction_unimol_dim", 768),
        reaction_chienn_dim=reaction_chirality_dim,
        reaction_chemistry_dim=config.data.get("reaction_chemistry_dim", None),
        reaction_directional_dim=config.data.get("reaction_directional_dim", None),
        reaction_use_model=config.data.get("reaction_use_model", True),
        reaction_use_chienn=reaction_use_chirality,
        reaction_use_chemistry=config.data.get("reaction_use_chemistry", False),
        reaction_use_directional=config.data.get("reaction_use_directional", False),
        reaction_chirality_name=config.model.get("reaction_chirality_name", "chiro"),
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
        reaction_chemistry_dropout=reaction_multimodal_attention_config.get(
            "chemistry_dropout",
            0.0,
        ),
        reaction_modality_token_layer_norm=reaction_multimodal_attention_config.get(
            "token_layer_norm",
            reaction_multimodal_attention_config.get("token_normalization", False),
        ),
        reaction_modality_l2_normalize=reaction_multimodal_attention_config.get(
            "modality_l2_normalize",
            False,
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
        reaction_side_composition=reaction_multimodal_attention_config.get(
            "side_composition",
            "directional_delta",
        ),
        reaction_modality_fusion=reaction_multimodal_attention_config.get(
            "fusion",
            "attention",
        ),
        reaction_factorized_dims=reaction_multimodal_attention_config.get(
            "factorized_dims",
            None,
        ),
        reaction_factorized_weights=reaction_multimodal_attention_config.get(
            "factorized_weights",
            None,
        ),
        reaction_attention_prior_weights=reaction_multimodal_attention_config.get(
            "attention_prior_weights",
            None,
        ),
        reaction_attention_adaptation_strength=reaction_multimodal_attention_config.get(
            "attention_adaptation_strength",
            0.4,
        ),
        reaction_output_projection=reaction_multimodal_attention_config.get(
            "output_projection",
            "mlp",
        ),
        reaction_residual_gate_init=reaction_multimodal_attention_config.get(
            "residual_gate_init",
            0.1,
        ),
        reaction_directional_gate_init=reaction_multimodal_attention_config.get(
            "directional_gate_init",
            0.1,
        ),
        query_encoder_checkpoint_path=query_encoder_checkpoint_path,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        beta=config.training.loss.beta,
        learn_beta=config.training.loss.get("learn_beta", False),
        beta_min=config.training.loss.get("beta_min", -float("inf")),
        beta_max=config.training.loss.get("beta_max", float("inf")),
        loss_name=config.training.loss.get("name", "FullBatchMLNCELoss"),
        positive_pair_source=config.training.loss.get(
            "positive_pair_source",
            "observed_pairs",
        ),
        degree_alpha=config.training.loss.get("degree_alpha", 0.5),
        cardinality_weight=config.training.loss.get("cardinality_weight", 0.3),
        cardinality_warmup_epochs=config.training.loss.get("cardinality_warmup_epochs", 5),
        separate_direction_temperatures=config.training.loss.get(
            "separate_direction_temperatures", False
        ),
        beta_r2e=config.training.loss.get("beta_r2e", config.training.loss.beta),
        beta_e2r=config.training.loss.get("beta_e2r", config.training.loss.beta),
        temperature_regularization_weight=config.training.loss.get(
            "temperature_regularization_weight", 0.0
        ),
        soft_rank_weight=config.training.loss.get("soft_rank_weight", 0.0),
        soft_rank_tau=config.training.loss.get("soft_rank_tau", 0.1),
        soft_rank_top_k=config.training.loss.get("soft_rank_top_k", 128),
        sigmoid_bias_init=config.training.loss.get("sigmoid_bias_init", 0.0),
        sigmoid_learn_bias=config.training.loss.get("sigmoid_learn_bias", True),
        sigmoid_negative_weight=config.training.loss.get("sigmoid_negative_weight", 1.0),
        lambda_r=config.training.loss.get("lambda_r", 0.05),
        lambda_e=config.training.loss.get("lambda_e", 0.05),
        lambda_g=config.training.loss.get("lambda_g", 0.01),
        tau_r=config.training.loss.get("tau_r", 0.1),
        tau_e=config.training.loss.get("tau_e", 0.1),
        tau_t=config.training.loss.get("tau_t", 0.1),
        delta_r=config.training.loss.get("delta_r", 0.5),
        delta_e=config.training.loss.get("delta_e", 0.5),
        symmetric_gw=config.training.loss.get("symmetric_gw", True),
        direction_balance_weight=config.training.loss.get(
            "direction_balance_weight",
            0.0,
        ),
        lambda_rr=config.training.loss.get("lambda_rr", 0.0),
        lambda_ee=config.training.loss.get("lambda_ee", 0.0),
        lambda_gw=config.training.loss.get("lambda_gw", config.training.loss.get("lambda_g", 0.0)),
        lambda_direction_gap=config.training.loss.get(
            "lambda_direction_gap",
            config.training.loss.get("direction_balance_weight", 0.0),
        ),
        lambda_r2e=config.training.loss.get("lambda_r2e", 0.5),
        lambda_e2r=config.training.loss.get("lambda_e2r", 0.5),
        lambda_r2e_hard_neg=config.training.loss.get("lambda_r2e_hard_neg", 0.0),
        r2e_hard_neg_top_k=config.training.loss.get("r2e_hard_neg_top_k", 0),
        r2e_hard_neg_margin=config.training.loss.get("r2e_hard_neg_margin", 0.0),
        lambda_e2r_hard_neg=config.training.loss.get("lambda_e2r_hard_neg", 0.0),
        e2r_hard_neg_top_k=config.training.loss.get("e2r_hard_neg_top_k", 0),
        e2r_hard_neg_margin=config.training.loss.get("e2r_hard_neg_margin", 0.0),
        tau_rr=config.training.loss.get("tau_rr", config.training.loss.get("tau_r", 0.1)),
        tau_ee=config.training.loss.get("tau_ee", config.training.loss.get("tau_e", 0.1)),
        tau_gw=config.training.loss.get("tau_gw", config.training.loss.get("tau_t", 0.1)),
        ec_positive_policy=config.training.loss.get(
            "ec_positive_policy",
            "hierarchical_weighted",
        ),
        ec_min_shared_depth=config.training.loss.get("ec_min_shared_depth", 2),
        gw_max_anchors=config.training.loss.get("gw_max_anchors", 512),
        apply_structure_terms_on_val=config.training.loss.get(
            "apply_structure_terms_on_val",
            False,
        ),
        lambda_residue=config.training.get("lambda_residue", 0.0),
        log_attention_stats=config.training.get("log_attention_stats", True),
        attention_logging_interval=config.training.get("attention_logging_interval", 100),
        validation_retrieval_metrics=config.training.get(
            "validation_retrieval_metrics",
            False,
        ),
        validation_retrieval_directions=config.training.get(
            "validation_retrieval_directions",
            config.data.get("validation_retrieval_directions", None),
        ),
        retrieval_metric_top_k=config.training.get("metrics", {}).get(
            "top_k",
            [1, 10, 100, 1000],
        ),
        training_stage=config.training.get("training_stage", "joint"),
        detach_reaction_embeddings=config.training.loss.get(
            "detach_reaction_embeddings",
            False,
        ),
        anchor_weight=config.training.loss.get("anchor_weight", 0.0),
        capability_consistency_weight=config.training.loss.get(
            "capability_consistency_weight",
            0.0,
        ),
    )
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

    checkpoint_kwargs = {
        "dirpath": config.logging.checkpoint_dir,
        "filename": "protein-pooling-{epoch:02d}",
        "save_last": True,
        "save_top_k": int(config.logging.get("save_top_k", 3)),
        "monitor": config.logging.get("checkpoint_monitor", "val/loss"),
        "mode": config.logging.get("checkpoint_mode", "min"),
    }
    if config.logging.get("checkpoint_on_validation_end", False):
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
    early_stopping = config.training.get("early_stopping", {})
    if early_stopping.get("enabled", False):
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor=early_stopping.get(
                    "monitor",
                    config.logging.get("checkpoint_monitor", "val/loss"),
                ),
                mode=early_stopping.get(
                    "mode",
                    config.logging.get("checkpoint_mode", "min"),
                ),
                patience=int(early_stopping.get("patience", 5)),
                min_delta=float(early_stopping.get("min_delta", 0.0)),
            )
        )

    devices = config.training.get("devices", 1 if torch.cuda.is_available() else "auto")
    accelerator = config.training.get("accelerator", "auto")
    strategy = config.training.get("strategy", "auto")
    if torch.cuda.is_available() and devices not in (1, "1", "auto") and strategy == "auto":
        strategy = "ddp"

    use_distributed_sampler = config.training.get("use_distributed_sampler", True)
    if config.data.get("hard_negative_pools_path", None) and use_distributed_sampler:
        print(
            "Disabling Lightning distributed sampler injection: "
            "DirectionalHardNegativeBatchSampler performs DDP sharding."
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
    }
    if config.training.get("validation_interval_steps", None) is not None:
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
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Last checkpoint: {checkpoint_callback.last_model_path}")


if __name__ == "__main__":
    main()
