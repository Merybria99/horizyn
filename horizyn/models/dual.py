"""Shared V4/F3/CIRCEv2 dual encoder. Inactive legacy options retain checkpoint schemas."""

import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from horizyn.checkpoint_utils import extract_query_encoder_state_dict
from horizyn.biological_residual import PromiscuityAwareBiologicalResidual

from .common import (BaseModel, MLP, _stage1_config, _torch_load_checkpoint)
from .enzyme import (EnzymeBiologicalFactorizedEncoder, ProteinAttentionPooling, ProteinMeanPooling, SLEECFunctionalPool, SLEECGuidedAttentionPool)
from .compat import (BlockwiseEnzymeFeatureFusion, E2RReactionAdapter, EnzymeBioFPSplitEncoder, GatedEnzymeFeatureFusion, R2EEnzymeAdapter, ReactionConditionedDualModel, ReactionFingerprintAttentionPool, ResidualEnzymePrototypeHead, TigerTextGatedFusion)


class ProteinPooledDualModel(BaseModel):
    """
    Dual encoder that pools residue-level protein embeddings before enzyme encoding.

    This model keeps the Horizyn-style reusable enzyme embedding: protein
    residues are pooled independently of the reaction, then the pooled vector is
    optionally passed through a frozen or trainable Stage 1 hyperbolic EC
    projector before the existing target MLP encoder. ``pooling="mean"``
    reproduces masked mean pooling, ``pooling="attention"`` learns residue
    weights, ``pooling="sleec"`` learns SLEEC-style functional-residue weights,
    and ``pooling="sleec_guided_attention"`` uses the SLEEC-prior attention
    pooler from Stage 1. ``enzyme_input_mode="raw_mean_sleec_multiview"`` uses
    only that pooler's frozen scorer and builds a separate multiview tower.
    """

    def __init__(
        self,
        query_encoder_kwargs: dict[str, Any],
        target_encoder_kwargs: dict[str, Any],
        residue_dim: int = 1024,
        pooling: str = "mean",
        attention_bias: bool = True,
        sleec_mode: str = "topk",
        sleec_topk_fraction: float = 0.2,
        sleec_threshold: float = 0.5,
        sleec_scorer_hidden_dim: int = 256,
        sleec_score_hidden_dim: int | None = None,
        sleec_checkpoint_path: str | None = None,
        sleec_freeze_scorer: bool = False,
        sleec_guided_initial_bias_scale: float = 1.0,
        sleec_guided_train_bias_scale: bool = True,
        hyperbolic_checkpoint_path: str | None = None,
        hyperbolic_freeze_projector: bool = False,
        hyperbolic_use_tangent: bool = True,
        hyperbolic_hyp_dim: int | None = None,
        hyperbolic_load_attention_pooler: bool = False,
        hyperbolic_freeze_attention_pooler: bool = False,
        enzyme_input_mode: str = "standard",
        enzyme_fusion_hidden_dim: int | None = None,
        enzyme_fusion_dropout: float = 0.0,
        enzyme_block_dims: dict[str, int] | None = None,
        enzyme_block_weights: dict[str, float] | None = None,
        enzyme_block_dropout: float = 0.0,
        enzyme_block_learned_weights: bool = False,
        enzyme_multiview: dict[str, Any] | None = None,
        capability_vector_dim: int | None = None,
        capability_freeze: bool = True,
        capability_adapter: bool = False,
        capability_dropout: float = 0.1,
        factorized_capability_dims: dict[str, int] | None = None,
        factorized_capability_freeze: bool = True,
        factorized_capability_use_masks: bool = True,
        biofp_center_dim: int = 0,
        biofp_cofactor_dim: int = 0,
        biofp_transition_dim: int = 0,
        biofp_family_dims: dict[str, int] | None = None,
        biofp_positive_labels: dict[str, list[str]] | None = None,
        biofp_seq_dim: int = 384,
        biofp_dim: int = 128,
        biofp_hidden_dim: int = 512,
        biofp_seq_weight: float = 0.75,
        biofp_dropout: float = 0.1,
        text_vector_dim: int | None = None,
        text_fusion_dim: int = 512,
        text_num_heads: int = 8,
        text_dropout: float = 0.1,
        text_freeze: bool = True,
        text_adapter: bool = False,
        reaction_hyperbolic_checkpoint_path: str | None = None,
        reaction_hyperbolic_freeze_encoder: bool = True,
        reaction_hyperbolic_freeze_projector: bool = True,
        reaction_hyperbolic_use_tangent: bool = True,
        reaction_fingerprint_attention_enabled: bool = False,
        reaction_fingerprint_attention_input_dim: int = 2048,
        reaction_fingerprint_attention_rdkit_dim: int = 1024,
        reaction_fingerprint_attention_drfp_dim: int = 1024,
        reaction_fingerprint_attention_token_dim: int = 512,
        reaction_fingerprint_attention_hidden_dim: int = 512,
        reaction_fingerprint_attention_dropout: float = 0.0,
        reaction_fingerprint_attention_bias: bool = True,
        e2r_adapter_enabled: bool = False,
        e2r_adapter_hidden_dim: int = 512,
        e2r_adapter_dropout: float = 0.1,
        e2r_adapter_gate_init: float = 0.1,
        e2r_adapter_use_factorized_inputs: bool = False,
        e2r_adapter_use_directional_inputs: bool = False,
        e2r_adapter_directional_dim: int | None = None,
        e2r_adapter_directional_hidden_dim: int = 128,
        e2r_adapter_reaction_model_dim: int = 768,
        e2r_adapter_unimol_dim: int = 768,
        e2r_adapter_chiro_dim: int = 256,
        e2r_adapter_chemistry_dim: int | None = None,
        r2e_adapter_enabled: bool = False,
        r2e_adapter_hidden_dim: int = 512,
        r2e_adapter_dropout: float = 0.1,
        r2e_adapter_gate_init: float = 0.1,
        r2e_adapter_use_factorized_inputs: bool = False,
        r2e_adapter_block_dims: dict[str, int] | None = None,
        r2e_adapter_block_weights: dict[str, float] | None = None,
        enzyme_prototype_count: int = 1,
        enzyme_prototype_bottleneck_dim: int = 64,
        enzyme_prototype_residual_gate_init: float = 0.05,
        enzyme_prototype_temperature: float = 0.1,
        enzyme_prototype_dropout: float = 0.0,
        biological_residual_enabled: bool = False,
        biological_residual_token_dim: int = 128,
        biological_residual_heads: int = 4,
        biological_residual_layers: int = 1,
        biological_residual_dropout: float = 0.1,
        biological_residual_max_molecules: int = 32,
        biological_residual_sleec_bias: float = 0.25,
        biological_residual_sleec_pool_scale: float = 2.0,
        biological_residual_lse_temperature: float = 0.1,
        biological_residual_target_chunk_size: int = 128,
        biological_residual_fusion_alpha: float = 0.0,
        biological_residual_max_fusion_alpha: float = 0.2,
        biological_residual_chiro_dim: int = 256,
        return_attention: bool = False,
        query_encoder: type[BaseModel] = MLP,
        target_encoder: type[BaseModel] = MLP,
        enforce_normalisation: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if enzyme_multiview is not None and not isinstance(enzyme_multiview, dict):
            raise ValueError("enzyme_multiview must be a mapping or None")
        if enzyme_multiview and enzyme_input_mode != "raw_mean_sleec_multiview":
            raise ValueError("enzyme_multiview settings require raw_mean_sleec_multiview mode")
        if pooling not in {"mean", "avg", "attention", "sleec", "sleec_guided_attention"}:
            raise ValueError(
                "pooling must be one of: mean, avg, attention, sleec, sleec_guided_attention"
            )
        if pooling == "avg":
            pooling = "mean"
        if enzyme_input_mode not in {
            "standard",
            "raw_sleec_hyperbolic_concat",
            "raw_mean_sleec_hyperbolic_gated",
            "raw_mean_sleec_hyperbolic_capability_gated",
            "raw_mean_sleec_hyperbolic_text_gated",
            "raw_mean_sleec_blockwise",
            "raw_mean_sleec_hyperbolic_blockwise",
            "raw_mean_sleec_hyperbolic_capability_blockwise",
            "raw_mean_sleec_hyperbolic_factorized_capability_blockwise",
            "raw_mean_sleec_biofp_split",
            "raw_mean_sleec_biological_factorized",
            "raw_mean_sleec_multiview",
        }:
            raise ValueError(
                "enzyme_input_mode must be one of: standard, raw_sleec_hyperbolic_concat, "
                "raw_mean_sleec_hyperbolic_gated, "
                "raw_mean_sleec_hyperbolic_capability_gated, "
                "raw_mean_sleec_hyperbolic_text_gated, raw_mean_sleec_blockwise, "
                "raw_mean_sleec_hyperbolic_blockwise, "
                "raw_mean_sleec_hyperbolic_capability_blockwise, "
                "raw_mean_sleec_hyperbolic_factorized_capability_blockwise, "
                "raw_mean_sleec_biofp_split, raw_mean_sleec_biological_factorized, "
                "raw_mean_sleec_multiview"
            )
        if enzyme_input_mode == "raw_mean_sleec_multiview":
            if pooling != "sleec_guided_attention" or not sleec_freeze_scorer:
                raise ValueError(
                    "raw_mean_sleec_multiview requires sleec_guided_attention "
                    "pooling and sleec_freeze_scorer=True"
                )
            if hyperbolic_checkpoint_path or r2e_adapter_enabled or biological_residual_enabled:
                raise ValueError(
                    "raw_mean_sleec_multiview cannot use a hyperbolic checkpoint, "
                    "R2E adapter, or biological residual"
                )
        if (
            enzyme_input_mode
            in {
                "raw_sleec_hyperbolic_concat",
                "raw_mean_sleec_hyperbolic_gated",
                "raw_mean_sleec_hyperbolic_capability_gated",
                "raw_mean_sleec_hyperbolic_text_gated",
                "raw_mean_sleec_hyperbolic_blockwise",
                "raw_mean_sleec_hyperbolic_capability_blockwise",
                "raw_mean_sleec_hyperbolic_factorized_capability_blockwise",
            }
            and not hyperbolic_checkpoint_path
        ):
            raise ValueError(
                f"enzyme_input_mode='{enzyme_input_mode}' requires hyperbolic_checkpoint_path"
            )
        if (
            enzyme_input_mode
            in {
                "raw_sleec_hyperbolic_concat",
                "raw_mean_sleec_hyperbolic_gated",
                "raw_mean_sleec_hyperbolic_capability_gated",
                "raw_mean_sleec_hyperbolic_text_gated",
                "raw_mean_sleec_hyperbolic_blockwise",
                "raw_mean_sleec_hyperbolic_capability_blockwise",
                "raw_mean_sleec_hyperbolic_factorized_capability_blockwise",
                "raw_mean_sleec_biological_factorized",
            }
            and not hyperbolic_use_tangent
        ):
            raise ValueError(
                f"enzyme_input_mode='{enzyme_input_mode}' requires hyperbolic_use_tangent=True"
            )

        stage1_checkpoint = (
            _torch_load_checkpoint(hyperbolic_checkpoint_path)
            if hyperbolic_checkpoint_path
            else None
        )
        stage1_config = _stage1_config(stage1_checkpoint) if stage1_checkpoint else {}
        reaction_stage1_checkpoint = (
            _torch_load_checkpoint(reaction_hyperbolic_checkpoint_path)
            if reaction_hyperbolic_checkpoint_path
            else None
        )
        reaction_stage1_config = (
            _stage1_config(reaction_stage1_checkpoint) if reaction_stage1_checkpoint else {}
        )
        if reaction_fingerprint_attention_enabled and reaction_stage1_checkpoint is not None:
            raise ValueError(
                "reaction_fingerprint_attention cannot be enabled together with "
                "reaction_hyperbolic_encoder.checkpoint_path"
            )
        if reaction_fingerprint_attention_enabled:
            expected_query_input_dim = int(reaction_fingerprint_attention_token_dim)
            if int(query_encoder_kwargs["input_dim"]) != expected_query_input_dim:
                raise ValueError(
                    "Query encoder input_dim must match reaction fingerprint attention "
                    f"token_dim: query input_dim={query_encoder_kwargs['input_dim']}, "
                    f"token_dim={expected_query_input_dim}"
                )

        self.query_encoder = query_encoder(**query_encoder_kwargs)
        self.target_encoder = target_encoder(**target_encoder_kwargs)
        self.enzyme_prototype_head = ResidualEnzymePrototypeHead(
            embedding_dim=int(query_encoder_kwargs["output_dim"]),
            prototype_count=int(enzyme_prototype_count),
            bottleneck_dim=int(enzyme_prototype_bottleneck_dim),
            residual_gate_init=float(enzyme_prototype_residual_gate_init),
            aggregation_temperature=float(enzyme_prototype_temperature),
            dropout=float(enzyme_prototype_dropout),
        )
        self.biological_residual = (
            PromiscuityAwareBiologicalResidual(
                residue_dim=int(residue_dim),
                unimol_dim=int(e2r_adapter_unimol_dim),
                chiro_dim=int(biological_residual_chiro_dim),
                token_dim=int(biological_residual_token_dim),
                heads=int(biological_residual_heads),
                layers=int(biological_residual_layers),
                dropout=float(biological_residual_dropout),
                max_molecules=int(biological_residual_max_molecules),
                sleec_threshold=float(sleec_threshold),
                sleec_bias=float(biological_residual_sleec_bias),
                sleec_pool_scale=float(biological_residual_sleec_pool_scale),
                lse_temperature=float(biological_residual_lse_temperature),
                target_chunk_size=int(biological_residual_target_chunk_size),
                fusion_alpha=float(biological_residual_fusion_alpha),
                max_fusion_alpha=float(biological_residual_max_fusion_alpha),
            )
            if biological_residual_enabled
            else None
        )
        self.pooling_name = pooling
        self.enzyme_input_mode = enzyme_input_mode
        self.residue_dim = residue_dim
        self.hyperbolic_use_tangent = bool(hyperbolic_use_tangent)
        self.hyperbolic_projector: nn.Module | None = None
        self.hyperbolic_tangent_dim: int | None = None
        self.reaction_hyperbolic_use_tangent = bool(reaction_hyperbolic_use_tangent)
        self.reaction_hyperbolic_input_dim: int | None = None
        self.reaction_hyperbolic_encoder: nn.Module | None = None
        self.reaction_hyperbolic_projector: nn.Module | None = None
        self.reaction_fingerprint_attention: ReactionFingerprintAttentionPool | None = None
        self.raw_mean_pooling: ProteinMeanPooling | None = None
        self.enzyme_feature_fusion: GatedEnzymeFeatureFusion | None = None
        self.enzyme_block_fusion: BlockwiseEnzymeFeatureFusion | None = None
        self.blockwise_target_is_embedding = False
        self.biofp_split_encoder: EnzymeBioFPSplitEncoder | None = None
        self.biological_factorized_encoder: EnzymeBiologicalFactorizedEncoder | None = None
        self.multiview_encoder: nn.Module | None = None
        self.text_feature_fusion: TigerTextGatedFusion | None = None
        self.capability_vector_dim = (
            None if capability_vector_dim is None else int(capability_vector_dim)
        )
        self.capability_freeze = bool(capability_freeze)
        self.capability_adapter_enabled = bool(capability_adapter)
        self.capability_adapter: nn.Module | None = None
        self.factorized_capability_dims = (
            None
            if factorized_capability_dims is None
            else {str(name): int(dim) for name, dim in factorized_capability_dims.items()}
        )
        self.factorized_capability_freeze = bool(factorized_capability_freeze)
        self.factorized_capability_use_masks = bool(factorized_capability_use_masks)
        self.text_vector_dim = None if text_vector_dim is None else int(text_vector_dim)
        self.text_freeze = bool(text_freeze)
        self.text_adapter_enabled = bool(text_adapter)
        self.text_adapter: nn.Module | None = None
        self.e2r_adapter = (
            E2RReactionAdapter(
                embedding_dim=int(query_encoder_kwargs["output_dim"]),
                hidden_dim=int(e2r_adapter_hidden_dim),
                dropout=float(e2r_adapter_dropout),
                gate_init=float(e2r_adapter_gate_init),
                use_factorized_inputs=bool(e2r_adapter_use_factorized_inputs),
                use_directional_inputs=bool(e2r_adapter_use_directional_inputs),
                reaction_model_dim=int(e2r_adapter_reaction_model_dim),
                unimol_dim=int(e2r_adapter_unimol_dim),
                chiro_dim=int(e2r_adapter_chiro_dim),
                chemistry_dim=e2r_adapter_chemistry_dim,
                directional_dim=e2r_adapter_directional_dim,
                directional_hidden_dim=int(e2r_adapter_directional_hidden_dim),
            )
            if e2r_adapter_enabled
            else None
        )
        self.r2e_adapter = (
            R2EEnzymeAdapter(
                embedding_dim=int(target_encoder_kwargs["output_dim"]),
                hidden_dim=int(r2e_adapter_hidden_dim),
                dropout=float(r2e_adapter_dropout),
                gate_init=float(r2e_adapter_gate_init),
                use_factorized_inputs=bool(r2e_adapter_use_factorized_inputs),
                block_dims=r2e_adapter_block_dims,
                block_weights=r2e_adapter_block_weights,
            )
            if r2e_adapter_enabled
            else None
        )
        if self.enzyme_input_mode in {
            "raw_mean_sleec_hyperbolic_capability_gated",
            "raw_mean_sleec_hyperbolic_capability_blockwise",
        }:
            if self.capability_vector_dim is None or self.capability_vector_dim <= 0:
                raise ValueError(
                    "capability_vector_dim must be positive for " f"{self.enzyme_input_mode}"
                )
            if not (0.0 <= capability_dropout <= 1.0):
                raise ValueError("capability_dropout must be in the range [0, 1]")
            if self.capability_adapter_enabled:
                self.capability_adapter = nn.Sequential(
                    nn.LayerNorm(self.capability_vector_dim),
                    nn.Linear(self.capability_vector_dim, self.capability_vector_dim),
                    nn.GELU(),
                    nn.Dropout(float(capability_dropout)),
                    nn.Linear(self.capability_vector_dim, self.capability_vector_dim),
                )
        if self.enzyme_input_mode == "raw_mean_sleec_hyperbolic_factorized_capability_blockwise":
            expected_families = set(BlockwiseEnzymeFeatureFusion.FACTORIZED_CAPABILITY_BRANCHES)
            if self.factorized_capability_dims is None:
                raise ValueError(
                    "factorized_capability_dims must be configured for " f"{self.enzyme_input_mode}"
                )
            if set(self.factorized_capability_dims) != expected_families:
                raise ValueError(
                    "factorized_capability_dims must contain exactly "
                    f"{sorted(expected_families)}"
                )
        if self.enzyme_input_mode == "raw_mean_sleec_hyperbolic_text_gated":
            if self.text_vector_dim is None or self.text_vector_dim <= 0:
                raise ValueError(
                    "text_vector_dim must be positive for " "raw_mean_sleec_hyperbolic_text_gated"
                )
            if int(text_fusion_dim) <= 0:
                raise ValueError("text_fusion_dim must be positive")
            if int(text_num_heads) <= 0:
                raise ValueError("text_num_heads must be positive")
            if int(text_fusion_dim) % int(text_num_heads) != 0:
                raise ValueError("text_fusion_dim must be divisible by text_num_heads")
            if not (0.0 <= text_dropout <= 1.0):
                raise ValueError("text_dropout must be in the range [0, 1]")
            if self.text_adapter_enabled:
                self.text_adapter = nn.Sequential(
                    nn.LayerNorm(self.text_vector_dim),
                    nn.Linear(self.text_vector_dim, self.text_vector_dim),
                    nn.GELU(),
                    nn.Dropout(float(text_dropout)),
                    nn.Linear(self.text_vector_dim, self.text_vector_dim),
                )
        if self.enzyme_input_mode == "raw_mean_sleec_biofp_split":
            if int(target_encoder_kwargs["input_dim"]) != int(target_encoder_kwargs["output_dim"]):
                raise ValueError(
                    "raw_mean_sleec_biofp_split expects target_encoder input_dim "
                    "to match output_dim because the BioFP split encoder emits the "
                    "final enzyme embedding"
                )
            if int(biofp_seq_dim) + int(biofp_dim) != int(target_encoder_kwargs["output_dim"]):
                raise ValueError(
                    "model.biofp.seq_dim + model.biofp.dim must equal target "
                    f"embedding dim {target_encoder_kwargs['output_dim']}"
                )
            self.raw_mean_pooling = ProteinMeanPooling(return_attention=False)
            self.biofp_split_encoder = EnzymeBioFPSplitEncoder(
                residue_dim=residue_dim,
                output_dim=int(target_encoder_kwargs["output_dim"]),
                seq_dim=int(biofp_seq_dim),
                bio_dim=int(biofp_dim),
                hidden_dim=int(biofp_hidden_dim),
                family_output_dims={
                    "center": int(biofp_center_dim),
                    "cofactor": int(biofp_cofactor_dim),
                    "transition": int(biofp_transition_dim),
                },
                seq_weight=float(biofp_seq_weight),
                dropout=float(biofp_dropout),
            )
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad = False

        if self.enzyme_input_mode == "raw_mean_sleec_biological_factorized":
            if enzyme_block_dims is None or enzyme_block_weights is None:
                raise ValueError(
                    "raw_mean_sleec_biological_factorized requires enzyme block dims and weights"
                )
            if int(target_encoder_kwargs["input_dim"]) != int(target_encoder_kwargs["output_dim"]):
                raise ValueError(
                    "raw_mean_sleec_biological_factorized emits the final enzyme embedding"
                )

        if self.enzyme_input_mode == "raw_mean_sleec_blockwise":
            self.raw_mean_pooling = ProteinMeanPooling(return_attention=False)
            self.enzyme_block_fusion = BlockwiseEnzymeFeatureFusion(
                raw_dim=residue_dim,
                pooled_dim=residue_dim,
                output_dim=int(target_encoder_kwargs["output_dim"]),
                block_dims=enzyme_block_dims,
                block_weights=enzyme_block_weights,
                dropout=float(enzyme_block_dropout),
            )
            self.blockwise_target_is_embedding = True
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad = False

        if reaction_fingerprint_attention_enabled:
            self.reaction_fingerprint_attention = ReactionFingerprintAttentionPool(
                input_dim=int(reaction_fingerprint_attention_input_dim),
                rdkit_dim=int(reaction_fingerprint_attention_rdkit_dim),
                drfp_dim=int(reaction_fingerprint_attention_drfp_dim),
                token_dim=int(reaction_fingerprint_attention_token_dim),
                hidden_dim=int(reaction_fingerprint_attention_hidden_dim),
                dropout=float(reaction_fingerprint_attention_dropout),
                attention_bias=bool(reaction_fingerprint_attention_bias),
            )

        if pooling == "mean":
            self.pooling = ProteinMeanPooling(return_attention=return_attention)
        elif pooling == "attention":
            self.pooling = ProteinAttentionPooling(
                hidden_dim=residue_dim,
                attention_bias=attention_bias,
                return_attention=return_attention,
            )
        elif pooling == "sleec":
            self.pooling = SLEECFunctionalPool(
                hidden_dim=residue_dim,
                scorer_hidden_dim=sleec_scorer_hidden_dim,
                score_hidden_dim=sleec_score_hidden_dim,
                mode=sleec_mode,
                topk_fraction=sleec_topk_fraction,
                threshold=sleec_threshold,
                checkpoint_path=sleec_checkpoint_path,
                freeze_scorer=sleec_freeze_scorer,
            )
        else:
            self.pooling = SLEECGuidedAttentionPool(
                hidden_dim=residue_dim,
                scorer_hidden_dim=sleec_scorer_hidden_dim,
                score_hidden_dim=sleec_score_hidden_dim,
                threshold=sleec_threshold,
                checkpoint_path=(
                    None
                    if hyperbolic_load_attention_pooler and stage1_checkpoint is not None
                    else sleec_checkpoint_path
                ),
                freeze_scorer=sleec_freeze_scorer,
                attention_bias=attention_bias,
                initial_sleec_bias_scale=sleec_guided_initial_bias_scale,
                train_sleec_bias_scale=sleec_guided_train_bias_scale,
            )
            if hyperbolic_load_attention_pooler and stage1_checkpoint is not None:
                pool_state = stage1_checkpoint.get("sleec_guided_attention_pool_state_dict")
                if pool_state is None:
                    raise ValueError(
                        "Requested hyperbolic_load_attention_pooler=True, but "
                        f"{hyperbolic_checkpoint_path} has no "
                        "sleec_guided_attention_pool_state_dict"
                    )
                pool_state = dict(pool_state)
                if "raw_prior_strength" in pool_state and "raw_sleec_bias_scale" not in pool_state:
                    pool_state["raw_sleec_bias_scale"] = pool_state.pop("raw_prior_strength")
                self.pooling.load_state_dict(pool_state)
            if hyperbolic_freeze_attention_pooler:
                for parameter in self.pooling.parameters():
                    parameter.requires_grad = False

        if self.enzyme_input_mode == "raw_mean_sleec_multiview":
            from horizyn.enzyme_multiview import EnzymeMultiviewEncoder

            self.multiview_encoder = EnzymeMultiviewEncoder(
                residue_dim=residue_dim,
                output_dim=int(target_encoder_kwargs["output_dim"]),
                biological_labels=biofp_positive_labels,
                **dict(enzyme_multiview or {}),
            )
            self.blockwise_target_is_embedding = True
            # Only the frozen scorer supplies a prior. The legacy learned
            # attention and target MLP are never used by this tower.
            for module in (self.pooling, self.target_encoder):
                for parameter in module.parameters():
                    parameter.requires_grad = False
            self._multiview_sleec_ready = bool(sleec_checkpoint_path)
            # Torch 2.4 exposes this hook registration only through the
            # underscored API; with_module matches the public-hook signature.
            self._register_load_state_dict_pre_hook(
                self._restore_multiview_sleec_prior, with_module=True
            )

        if stage1_checkpoint is not None:
            projector_input_dim = int(stage1_config.get("input_dim", residue_dim))
            projector_hyp_dim = int(
                stage1_config.get("hyp_dim", target_encoder_kwargs["input_dim"])
            )
            is_lorentz_checkpoint = (
                "hyperbolic_projector_state_dict" in stage1_checkpoint
                or "projector_input_normalization" in stage1_config
            )
            if not is_lorentz_checkpoint:
                raise ValueError(
                    "Legacy non-Lorentz hyperbolic enzyme checkpoints are no longer supported. "
                    "Use a Lorentz hyperbolic enzyme checkpoint produced by "
                    "scripts/pretrain_hyperbolic_enzyme.py."
                )
            if projector_input_dim != residue_dim:
                raise ValueError(
                    "Hyperbolic projector input_dim must match residue pooled dim: "
                    f"checkpoint input_dim={projector_input_dim}, residue_dim={residue_dim}"
                )
            expected_target_input_dim = (
                projector_hyp_dim
                if self.hyperbolic_use_tangent or not is_lorentz_checkpoint
                else projector_hyp_dim + 1
            )
            if self.enzyme_input_mode == "raw_sleec_hyperbolic_concat":
                expected_target_input_dim = residue_dim + projector_hyp_dim
            if (
                self.enzyme_input_mode
                not in {
                    "raw_mean_sleec_hyperbolic_gated",
                    "raw_mean_sleec_hyperbolic_capability_gated",
                    "raw_mean_sleec_hyperbolic_text_gated",
                    "raw_mean_sleec_biological_factorized",
                }
                and int(target_encoder_kwargs["input_dim"]) != expected_target_input_dim
            ):
                raise ValueError(
                    "Target encoder input_dim must match hyperbolic hyp_dim when a "
                    "hyperbolic checkpoint is used: "
                    f"target input_dim={target_encoder_kwargs['input_dim']}, "
                    f"expected={expected_target_input_dim}"
                )
            from horizyn.hyperbolic_enzyme import LorentzEnzymeProjector

            self.hyperbolic_projector = LorentzEnzymeProjector(
                input_dim=projector_input_dim,
                hyp_dim=projector_hyp_dim,
                curvature=float(
                    stage1_config.get("curvature", stage1_checkpoint.get("curvature", 0.25))
                ),
                tangent_clip=stage1_config.get("tangent_clip", 5.0),
                tangent_clip_mode=str(stage1_config.get("tangent_clip_mode", "hard")),
                input_normalization=str(stage1_config.get("projector_input_normalization", "none")),
                eps=float(stage1_config.get("eps", 1e-6)),
            )
            projector_state = stage1_checkpoint.get("hyperbolic_projector_state_dict")
            if projector_state is None:
                projector_state = stage1_checkpoint.get("projector_state_dict")
            if projector_state is None:
                raise ValueError(
                    f"Hyperbolic checkpoint has no projector state dict: {hyperbolic_checkpoint_path}"
                )
            self.hyperbolic_projector.load_state_dict(projector_state)
            self.hyperbolic_tangent_dim = projector_hyp_dim
            if hyperbolic_freeze_projector:
                for parameter in self.hyperbolic_projector.parameters():
                    parameter.requires_grad = False
            if self.enzyme_input_mode in {
                "raw_mean_sleec_hyperbolic_gated",
                "raw_mean_sleec_hyperbolic_capability_gated",
                "raw_mean_sleec_hyperbolic_text_gated",
            }:
                self.raw_mean_pooling = ProteinMeanPooling(return_attention=False)
                self.enzyme_feature_fusion = GatedEnzymeFeatureFusion(
                    raw_dim=residue_dim,
                    pooled_dim=residue_dim,
                    hyperbolic_dim=projector_hyp_dim,
                    capability_dim=(
                        self.capability_vector_dim
                        if self.enzyme_input_mode == "raw_mean_sleec_hyperbolic_capability_gated"
                        else None
                    ),
                    output_dim=int(target_encoder_kwargs["input_dim"]),
                    gate_hidden_dim=enzyme_fusion_hidden_dim,
                    dropout=float(enzyme_fusion_dropout),
                )
                if self.enzyme_input_mode == "raw_mean_sleec_hyperbolic_text_gated":
                    if self.text_vector_dim is None:
                        raise RuntimeError("text_vector_dim was not initialized")
                    self.text_feature_fusion = TigerTextGatedFusion(
                        sequence_dim=int(target_encoder_kwargs["input_dim"]),
                        text_dim=self.text_vector_dim,
                        output_dim=int(target_encoder_kwargs["input_dim"]),
                        fusion_dim=int(text_fusion_dim),
                        num_heads=int(text_num_heads),
                        dropout=float(text_dropout),
                    )
            if self.enzyme_input_mode in {
                "raw_mean_sleec_hyperbolic_blockwise",
                "raw_mean_sleec_hyperbolic_capability_blockwise",
                "raw_mean_sleec_hyperbolic_factorized_capability_blockwise",
            }:
                self.raw_mean_pooling = ProteinMeanPooling(return_attention=False)
                self.enzyme_block_fusion = BlockwiseEnzymeFeatureFusion(
                    raw_dim=residue_dim,
                    pooled_dim=residue_dim,
                    hyperbolic_dim=projector_hyp_dim,
                    capability_dim=(
                        self.capability_vector_dim
                        if self.enzyme_input_mode
                        == "raw_mean_sleec_hyperbolic_capability_blockwise"
                        else None
                    ),
                    factorized_capability_dims=(
                        self.factorized_capability_dims
                        if self.enzyme_input_mode
                        == "raw_mean_sleec_hyperbolic_factorized_capability_blockwise"
                        else None
                    ),
                    factorized_capability_use_masks=self.factorized_capability_use_masks,
                    output_dim=int(target_encoder_kwargs["output_dim"]),
                    block_dims=enzyme_block_dims,
                    block_weights=enzyme_block_weights,
                    dropout=float(enzyme_block_dropout),
                )
                self.blockwise_target_is_embedding = True
                for parameter in self.target_encoder.parameters():
                    parameter.requires_grad = False

        if self.enzyme_input_mode == "raw_mean_sleec_biological_factorized":
            uses_ec_block = enzyme_block_dims is not None and "ec" in enzyme_block_dims
            if uses_ec_block and self.hyperbolic_projector is None:
                from horizyn.hyperbolic_enzyme import LorentzEnzymeProjector

                random_hyp_dim = int(hyperbolic_hyp_dim or 128)
                self.hyperbolic_projector = LorentzEnzymeProjector(
                    input_dim=residue_dim,
                    hyp_dim=random_hyp_dim,
                    curvature=0.25,
                    tangent_clip=5.0,
                    tangent_clip_mode="hard",
                    input_normalization="none",
                    eps=1e-6,
                )
                self.hyperbolic_tangent_dim = random_hyp_dim
            if uses_ec_block and self.hyperbolic_tangent_dim is None:
                raise RuntimeError("Hyperbolic tangent dimension was not initialized")
            family_dims = dict(biofp_family_dims or {})
            family_names = [
                name
                for name in (enzyme_block_dims or {})
                if name not in EnzymeBiologicalFactorizedEncoder.RESERVED_BLOCKS
            ]
            if not family_dims and family_names:
                default_family_dims = {
                    "mechanism": int(biofp_transition_dim),
                    "cofactor": int(biofp_cofactor_dim),
                }
                family_dims = {name: default_family_dims.get(name, 0) for name in family_names}
            self.raw_mean_pooling = ProteinMeanPooling(return_attention=False)
            self.biological_factorized_encoder = EnzymeBiologicalFactorizedEncoder(
                residue_dim=residue_dim,
                hyperbolic_dim=(self.hyperbolic_tangent_dim if uses_ec_block else None),
                output_dim=int(target_encoder_kwargs["output_dim"]),
                family_output_dims=family_dims,
                block_dims=enzyme_block_dims,
                block_weights=enzyme_block_weights,
                hidden_dim=int(biofp_hidden_dim),
                dropout=float(biofp_dropout),
                initial_sleec_bias_scale=float(sleec_guided_initial_bias_scale),
                train_sleec_bias_scale=bool(sleec_guided_train_bias_scale),
                learned_block_weights=bool(enzyme_block_learned_weights),
            )
            self.blockwise_target_is_embedding = True
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad = False

        if reaction_stage1_checkpoint is not None:
            reaction_encoder_dims = reaction_stage1_config.get("reaction_encoder_dims")
            if not isinstance(reaction_encoder_dims, list) or len(reaction_encoder_dims) < 2:
                reaction_encoder_dims = [
                    int(
                        reaction_stage1_config.get(
                            "input_dim", reaction_stage1_checkpoint["input_dim"]
                        )
                    ),
                    int(
                        reaction_stage1_config.get(
                            "reaction_embedding_dim",
                            reaction_stage1_checkpoint.get("reaction_embedding_dim", 512),
                        )
                    ),
                ]
            reaction_encoder_dims = [int(value) for value in reaction_encoder_dims]
            reaction_input_dim = int(
                reaction_stage1_config.get("input_dim", reaction_encoder_dims[0])
            )
            reaction_embedding_dim = int(
                reaction_stage1_config.get(
                    "reaction_embedding_dim",
                    reaction_encoder_dims[-1],
                )
            )
            if reaction_encoder_dims[0] != reaction_input_dim:
                raise ValueError(
                    "Reaction hyperbolic encoder input_dim does not match "
                    f"reaction_encoder_dims: input_dim={reaction_input_dim}, "
                    f"dims[0]={reaction_encoder_dims[0]}"
                )
            if reaction_encoder_dims[-1] != reaction_embedding_dim:
                raise ValueError(
                    "Reaction hyperbolic encoder output dim does not match "
                    f"reaction_embedding_dim={reaction_embedding_dim}, "
                    f"dims[-1]={reaction_encoder_dims[-1]}"
                )
            reaction_hyp_dim = int(
                reaction_stage1_config.get(
                    "hyp_dim",
                    reaction_stage1_checkpoint.get("hyp_dim", query_encoder_kwargs["input_dim"]),
                )
            )
            expected_query_input_dim = (
                reaction_hyp_dim if self.reaction_hyperbolic_use_tangent else reaction_hyp_dim + 1
            )
            if int(query_encoder_kwargs["input_dim"]) != expected_query_input_dim:
                raise ValueError(
                    "Query encoder input_dim must match reaction hyperbolic output dim: "
                    f"query input_dim={query_encoder_kwargs['input_dim']}, "
                    f"expected={expected_query_input_dim}"
                )

            self.reaction_hyperbolic_input_dim = reaction_input_dim
            self.reaction_hyperbolic_encoder = MLP(
                input_dim=reaction_encoder_dims[0],
                output_dim=reaction_encoder_dims[-1],
                num_layers=max(len(reaction_encoder_dims) - 2, 0),
                widths=reaction_encoder_dims[1:-1],
                normalise_output=True,
            )
            self.reaction_hyperbolic_encoder.load_state_dict(
                extract_query_encoder_state_dict(reaction_stage1_checkpoint)
            )

            from horizyn.hyperbolic_enzyme import LorentzEnzymeProjector

            self.reaction_hyperbolic_projector = LorentzEnzymeProjector(
                input_dim=reaction_embedding_dim,
                hyp_dim=reaction_hyp_dim,
                curvature=float(
                    reaction_stage1_config.get(
                        "curvature",
                        reaction_stage1_checkpoint.get("curvature", 0.25),
                    )
                ),
                tangent_clip=reaction_stage1_config.get("tangent_clip", 5.0),
                tangent_clip_mode=str(reaction_stage1_config.get("tangent_clip_mode", "hard")),
                input_normalization=str(
                    reaction_stage1_config.get("projector_input_normalization", "none")
                ),
                eps=float(reaction_stage1_config.get("eps", 1e-6)),
            )
            reaction_projector_state = reaction_stage1_checkpoint.get(
                "hyperbolic_projector_state_dict"
            )
            if reaction_projector_state is None:
                reaction_projector_state = reaction_stage1_checkpoint.get("projector_state_dict")
            if reaction_projector_state is None:
                raise ValueError(
                    "Reaction hyperbolic checkpoint has no projector state dict: "
                    f"{reaction_hyperbolic_checkpoint_path}"
                )
            self.reaction_hyperbolic_projector.load_state_dict(reaction_projector_state)

            if reaction_hyperbolic_freeze_encoder:
                for parameter in self.reaction_hyperbolic_encoder.parameters():
                    parameter.requires_grad = False
            if reaction_hyperbolic_freeze_projector:
                for parameter in self.reaction_hyperbolic_projector.parameters():
                    parameter.requires_grad = False

        if enforce_normalisation:
            ReactionConditionedDualModel._validate_normalized_encoder(
                self.query_encoder,
                "query_encoder",
            )
            ReactionConditionedDualModel._validate_normalized_encoder(
                self.target_encoder,
                "target_encoder",
            )

    @property
    def enzyme_prototype_count(self) -> int:
        return self.enzyme_prototype_head.prototype_count

    def score_reaction_enzyme_embeddings(
        self,
        reaction_embeddings: torch.Tensor,
        enzyme_embeddings: torch.Tensor,
        *,
        similarity: str = "cosine",
        feature_power: float = 1.0,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Score reusable enzyme embeddings through their latent prototypes."""

        return self.enzyme_prototype_head.score(
            reaction_embeddings,
            enzyme_embeddings,
            similarity=similarity,
            feature_power=feature_power,
            return_details=return_details,
        )

    def encode_queries(
        self,
        query_inputs: dict | torch.Tensor,
        return_attention: bool = False,
        retrieval_direction: str = "reaction_to_enzyme",
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if retrieval_direction not in {
            "reaction_to_enzyme",
            "enzyme_to_reaction",
        }:
            raise ValueError("retrieval_direction must be reaction_to_enzyme or enzyme_to_reaction")
        adapter_inputs = query_inputs
        query_attention: dict[str, torch.Tensor] = {}
        if self.reaction_fingerprint_attention is not None:
            if isinstance(query_inputs, dict):
                raise ValueError("Reaction fingerprint attention expects fingerprint tensor inputs")
            pooled_result = self.reaction_fingerprint_attention(
                query_inputs,
                return_details=return_attention,
            )
            if return_attention:
                query_inputs, reaction_attention = pooled_result
                query_attention["reaction_fingerprint"] = reaction_attention["weights"]
                query_attention["reaction_fingerprint_logits"] = reaction_attention["logits"]
                query_attention["reaction_fingerprint_entropy"] = reaction_attention["entropy"]
                query_attention["reaction_fingerprint_weight_rdkit_reactants"] = reaction_attention[
                    "weight_rdkit_reactants"
                ]
                query_attention["reaction_fingerprint_weight_rdkit_products"] = reaction_attention[
                    "weight_rdkit_products"
                ]
                query_attention["reaction_fingerprint_weight_drfp"] = reaction_attention[
                    "weight_drfp"
                ]
            else:
                query_inputs = pooled_result

        if self.reaction_hyperbolic_encoder is not None:
            if isinstance(query_inputs, dict):
                raise ValueError(
                    "Reaction hyperbolic encoding currently expects fingerprint tensor inputs"
                )
            if query_inputs.ndim != 2:
                raise ValueError(
                    "Reaction hyperbolic encoding expects rank-2 query tensors, "
                    f"got {tuple(query_inputs.shape)}"
                )
            if (
                self.reaction_hyperbolic_input_dim is not None
                and query_inputs.shape[-1] != self.reaction_hyperbolic_input_dim
            ):
                raise ValueError(
                    "Reaction hyperbolic input dim mismatch: "
                    f"got {query_inputs.shape[-1]}, expected {self.reaction_hyperbolic_input_dim}"
                )
            reaction_stage1 = self.reaction_hyperbolic_encoder(query_inputs)
            z_hyp, z_tangent = self.reaction_hyperbolic_projector(reaction_stage1)
            query_inputs = z_tangent if self.reaction_hyperbolic_use_tangent else z_hyp

        if isinstance(query_inputs, dict):
            if return_attention and getattr(
                self.query_encoder,
                "supports_attention_return",
                False,
            ):
                query_result = self.query_encoder(
                    **query_inputs,
                    return_attention=True,
                )
            else:
                query_result = self.query_encoder(**query_inputs)
        else:
            query_result = self.query_encoder(query_inputs)

        if return_attention and isinstance(query_result, tuple):
            query_embeddings, attention = query_result
            query_attention.update(attention)
        else:
            query_embeddings = query_result[0] if isinstance(query_result, tuple) else query_result

        if query_embeddings.ndim != 2:
            raise ValueError(
                f"Query encoder must return rank-2 tensors, got {tuple(query_embeddings.shape)}"
            )
        if retrieval_direction == "enzyme_to_reaction" and self.e2r_adapter is not None:
            adapter_result = self.e2r_adapter(
                query_embeddings,
                adapter_inputs,
                return_details=return_attention,
            )
            if return_attention:
                query_embeddings, adapter_details = adapter_result
                query_attention.update(adapter_details)
            else:
                query_embeddings = adapter_result
        if return_attention:
            return query_embeddings, query_attention
        return query_embeddings

    def pool_residues(
        self,
        residue_embeddings: torch.Tensor,
        residue_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        score_residue_embeddings: torch.Tensor | None = None,
        score_residue_padding_mask: torch.Tensor | None = None,
        score_attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
        return_details: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ):
        if attention_mask is not None and residue_padding_mask is not None:
            raise ValueError("Pass either attention_mask or residue_padding_mask, not both")
        if score_attention_mask is not None and score_residue_padding_mask is not None:
            raise ValueError(
                "Pass either score_attention_mask or score_residue_padding_mask, not both"
            )
        if attention_mask is None:
            attention_mask = (
                None if residue_padding_mask is None else ~residue_padding_mask.to(torch.bool)
            )
        if score_attention_mask is None:
            score_attention_mask = (
                None
                if score_residue_padding_mask is None
                else ~score_residue_padding_mask.to(torch.bool)
            )
        if self.pooling_name == "sleec":
            return self.pooling(
                residue_embeddings,
                attention_mask=attention_mask,
                return_attention=return_attention,
                return_details=return_details,
                score_embeddings=score_residue_embeddings,
                score_attention_mask=score_attention_mask,
            )
        if self.pooling_name == "sleec_guided_attention":
            result = self.pooling(
                residue_embeddings,
                attention_mask=attention_mask,
                return_details=return_details or return_attention,
                score_embeddings=score_residue_embeddings,
                score_attention_mask=score_attention_mask,
            )
            if return_details:
                return result
            if return_attention:
                pooled, details = result
                return pooled, details["weights"]
            return result
        if return_details:
            pooled, weights = self.pooling(
                residue_embeddings,
                attention_mask=attention_mask,
                return_attention=True,
            )
            return pooled, {"weights": weights}
        return self.pooling(
            residue_embeddings,
            attention_mask=attention_mask,
            return_attention=return_attention,
        )

    def _apply_r2e_adapter(
        self,
        target_embeddings: torch.Tensor,
        retrieval_direction: str,
        details: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if retrieval_direction not in {
            "reaction_to_enzyme",
            "enzyme_to_reaction",
        }:
            raise ValueError("retrieval_direction must be reaction_to_enzyme or enzyme_to_reaction")
        if retrieval_direction != "reaction_to_enzyme" or self.r2e_adapter is None:
            return target_embeddings, details
        adapter_result = self.r2e_adapter(
            target_embeddings,
            return_details=details is not None,
        )
        if details is None:
            return adapter_result, None
        adapted, adapter_details = adapter_result
        details = dict(details)
        details.update(adapter_details)
        return adapted, details

    def _restore_multiview_sleec_prior(
        self, module, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ) -> None:
        """Allow checkpoint reload without requiring the original SLEEC file."""
        scorer_state = self.pooling.sleec_scorer.state_dict()
        scorer_prefix = prefix + "pooling.sleec_scorer."
        if all(
            scorer_prefix + name in state_dict
            and state_dict[scorer_prefix + name].shape == value.shape
            for name, value in scorer_state.items()
        ):
            self._multiview_sleec_ready = True

    def _encode_multiview_targets(
        self,
        residue_embeddings: torch.Tensor,
        residue_padding_mask: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        score_residue_embeddings: torch.Tensor | None,
        score_residue_padding_mask: torch.Tensor | None,
        score_attention_mask: torch.Tensor | None,
        return_attention: bool,
        return_pooling_details: bool,
        retrieval_direction: str,
    ):
        if retrieval_direction not in {"reaction_to_enzyme", "enzyme_to_reaction"}:
            raise ValueError("retrieval_direction must be reaction_to_enzyme or enzyme_to_reaction")
        if self.multiview_encoder is None:
            raise RuntimeError("Multiview enzyme encoder was not initialized")
        if not self._multiview_sleec_ready:
            raise RuntimeError(
                "Multiview enzyme encoding requires a loaded SLEEC prior; provide "
                "sleec_checkpoint_path or load a complete trained model checkpoint"
            )
        if residue_embeddings.ndim != 3 or residue_embeddings.shape[-1] != self.residue_dim:
            raise ValueError("residue_embeddings must have shape [batch, length, residue_dim]")
        if attention_mask is not None and residue_padding_mask is not None:
            raise ValueError("Pass either attention_mask or residue_padding_mask, not both")
        if score_attention_mask is not None and score_residue_padding_mask is not None:
            raise ValueError(
                "Pass either score_attention_mask or score_residue_padding_mask, not both"
            )
        if attention_mask is None:
            attention_mask = (
                torch.ones(
                    residue_embeddings.shape[:2], dtype=torch.bool,
                    device=residue_embeddings.device,
                )
                if residue_padding_mask is None else ~residue_padding_mask.to(torch.bool)
            )
        valid_mask = attention_mask.to(device=residue_embeddings.device, dtype=torch.bool)
        if valid_mask.shape != residue_embeddings.shape[:2]:
            raise ValueError("Residue mask must match residue_embeddings batch and length")
        if score_residue_embeddings is None:
            score_residue_embeddings = residue_embeddings
        if (
            score_residue_embeddings.ndim != 3
            or score_residue_embeddings.shape[:2] != residue_embeddings.shape[:2]
        ):
            raise ValueError(
                "Scorer embeddings must align with residue embeddings in batch and length"
            )
        if score_attention_mask is None:
            score_attention_mask = (
                valid_mask if score_residue_padding_mask is None
                else ~score_residue_padding_mask.to(torch.bool)
            )
        score_valid_mask = score_attention_mask.to(device=residue_embeddings.device, dtype=torch.bool)
        if not torch.equal(score_valid_mask, valid_mask):
            raise ValueError("Scorer residue mask must match value residue validity")
        # Padding may contain NaN sentinels; exclude it before the scorer MLP.
        score_inputs = score_residue_embeddings.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        self.pooling.sleec_scorer.eval()
        with torch.no_grad():
            sleec_logits, _ = self.pooling.sleec_scorer(score_inputs, attention_mask=valid_mask)
        result = self.multiview_encoder(
            residue_embeddings,
            residue_padding_mask=~valid_mask,
            sleec_prior=sleec_logits,
            return_details=return_pooling_details or return_attention,
        )
        if return_pooling_details or return_attention:
            embeddings, details = result
            weights = details["weights"]
            sleec_positive = (details["scores"] > self.pooling.threshold) & valid_mask
            details["num_sleec_positive"] = sleec_positive.sum(dim=-1).to(weights.dtype)
            details["attention_mass_sleec_positive"] = (
                weights * sleec_positive.to(weights.dtype)
            ).sum(dim=-1)
        if return_pooling_details:
            return embeddings, details
        if return_attention:
            # Preserve the conventional [batch, length] interface using the
            # SLEEC site view; learned-slot attention is exposed in details.
            return embeddings, details["weights"]
        return result

    def encode_targets(
        self,
        residue_embeddings: torch.Tensor,
        residue_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        score_residue_embeddings: torch.Tensor | None = None,
        score_residue_padding_mask: torch.Tensor | None = None,
        score_attention_mask: torch.Tensor | None = None,
        capability_vectors: torch.Tensor | None = None,
        capability_mask: torch.Tensor | None = None,
        factorized_capability_vectors: dict[str, torch.Tensor] | None = None,
        factorized_capability_masks: dict[str, torch.Tensor] | None = None,
        text_vectors: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        return_attention: bool = False,
        return_pooling_details: bool = False,
        retrieval_direction: str = "reaction_to_enzyme",
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ):
        if self.enzyme_input_mode == "raw_mean_sleec_multiview":
            return self._encode_multiview_targets(
                residue_embeddings, residue_padding_mask, attention_mask,
                score_residue_embeddings, score_residue_padding_mask, score_attention_mask,
                return_attention, return_pooling_details, retrieval_direction,
            )
        internal_pooling_details = return_pooling_details or (
            self.enzyme_input_mode == "raw_mean_sleec_biological_factorized"
            and self.pooling_name == "sleec_guided_attention"
        )
        pooled_result = self.pool_residues(
            residue_embeddings,
            residue_padding_mask=residue_padding_mask,
            attention_mask=attention_mask,
            score_residue_embeddings=score_residue_embeddings,
            score_residue_padding_mask=score_residue_padding_mask,
            score_attention_mask=score_attention_mask,
            return_attention=return_attention,
            return_details=internal_pooling_details,
        )
        if internal_pooling_details:
            pooled, details = pooled_result
            attention = details.get("weights") if return_attention else None
        elif return_attention:
            pooled, attention = pooled_result
            details = None
        else:
            pooled = pooled_result
            details = None
            attention = None

        raw_mean = None
        if self.enzyme_input_mode in {
            "raw_mean_sleec_hyperbolic_gated",
            "raw_mean_sleec_hyperbolic_capability_gated",
            "raw_mean_sleec_hyperbolic_text_gated",
            "raw_mean_sleec_blockwise",
            "raw_mean_sleec_hyperbolic_blockwise",
            "raw_mean_sleec_hyperbolic_capability_blockwise",
            "raw_mean_sleec_hyperbolic_factorized_capability_blockwise",
            "raw_mean_sleec_biofp_split",
            "raw_mean_sleec_biological_factorized",
        }:
            if self.raw_mean_pooling is None:
                raise RuntimeError("raw_mean_pooling was not initialized")
            raw_attention_mask = attention_mask
            if raw_attention_mask is None and residue_padding_mask is not None:
                raw_attention_mask = ~residue_padding_mask.to(torch.bool)
            raw_mean = self.raw_mean_pooling(
                residue_embeddings,
                attention_mask=raw_attention_mask,
            )

        if self.enzyme_input_mode == "raw_mean_sleec_biofp_split":
            if raw_mean is None or self.biofp_split_encoder is None:
                raise RuntimeError("BioFP split encoder was not initialized")
            biofp_result = self.biofp_split_encoder(
                raw_mean,
                pooled,
                residue_embeddings,
                residue_padding_mask=residue_padding_mask,
                return_details=return_pooling_details,
            )
            if return_pooling_details:
                target_embeddings, biofp_details = biofp_result
                details = dict(details or {})
                details.update(biofp_details)
            else:
                target_embeddings = biofp_result
            target_embeddings, details = self._apply_r2e_adapter(
                target_embeddings,
                retrieval_direction,
                details if return_pooling_details else None,
            )
            if return_pooling_details:
                return target_embeddings, details
            if return_attention:
                return target_embeddings, attention
            return target_embeddings

        if self.enzyme_input_mode == "raw_mean_sleec_biological_factorized":
            if raw_mean is None or self.biological_factorized_encoder is None:
                raise RuntimeError("Biological factorized encoder was not initialized")
            z_tangent = None
            if "ec" in self.biological_factorized_encoder.block_dims:
                if self.hyperbolic_projector is None:
                    raise RuntimeError("Biological ec block has no hyperbolic projector")
                _z_hyp, z_tangent = self.hyperbolic_projector(pooled)
            sleec_prior = None
            if details is not None:
                candidate_prior = details.get("logits")
                if candidate_prior is not None:
                    sleec_prior = candidate_prior.to(
                        device=residue_embeddings.device,
                        dtype=residue_embeddings.dtype,
                    )
            biological_result = self.biological_factorized_encoder(
                raw_mean,
                pooled,
                z_tangent,
                residue_embeddings,
                residue_padding_mask=residue_padding_mask,
                sleec_prior=sleec_prior,
                return_details=return_pooling_details,
            )
            if return_pooling_details:
                target_embeddings, biological_details = biological_result
                details = dict(details or {})
                details.update(biological_details)
            else:
                target_embeddings = biological_result
            target_embeddings, details = self._apply_r2e_adapter(
                target_embeddings,
                retrieval_direction,
                details if return_pooling_details else None,
            )
            if return_pooling_details:
                return target_embeddings, details
            if return_attention:
                return target_embeddings, attention
            return target_embeddings

        target_input = pooled
        if self.hyperbolic_projector is not None:
            z_hyp, z_tangent = self.hyperbolic_projector(pooled)
            hyperbolic_input = z_tangent if self.hyperbolic_use_tangent else z_hyp
            if self.enzyme_input_mode == "raw_sleec_hyperbolic_concat":
                target_input = torch.cat([pooled, z_tangent], dim=-1)
            elif self.enzyme_input_mode in {
                "raw_mean_sleec_hyperbolic_gated",
                "raw_mean_sleec_hyperbolic_capability_gated",
                "raw_mean_sleec_hyperbolic_text_gated",
                "raw_mean_sleec_hyperbolic_blockwise",
                "raw_mean_sleec_hyperbolic_capability_blockwise",
                "raw_mean_sleec_hyperbolic_factorized_capability_blockwise",
            }:
                if raw_mean is None or self.enzyme_feature_fusion is None:
                    if self.enzyme_block_fusion is None:
                        raise RuntimeError("enzyme feature fusion was not initialized")
                capability_input = None
                capability_branch_mask = None
                factorized_inputs = None
                factorized_masks = None
                if self.enzyme_input_mode in {
                    "raw_mean_sleec_hyperbolic_capability_gated",
                    "raw_mean_sleec_hyperbolic_capability_blockwise",
                }:
                    if capability_vectors is None:
                        raise ValueError(
                            "capability_vectors are required for " f"{self.enzyme_input_mode}"
                        )
                    capability_input = capability_vectors.to(
                        device=pooled.device,
                        dtype=pooled.dtype,
                    )
                    if capability_mask is not None:
                        capability_branch_mask = capability_mask.to(
                            device=pooled.device,
                            dtype=torch.bool,
                        )
                    if self.capability_freeze:
                        capability_input = capability_input.detach()
                    if self.capability_adapter is not None:
                        capability_input = capability_input + self.capability_adapter(
                            capability_input
                        )
                if (
                    self.enzyme_input_mode
                    == "raw_mean_sleec_hyperbolic_factorized_capability_blockwise"
                ):
                    if factorized_capability_vectors is None:
                        raise ValueError(
                            "factorized_capability_vectors are required for "
                            f"{self.enzyme_input_mode}"
                        )
                    factorized_inputs = {
                        name: value.to(device=pooled.device, dtype=pooled.dtype)
                        for name, value in factorized_capability_vectors.items()
                    }
                    if self.factorized_capability_freeze:
                        factorized_inputs = {
                            name: value.detach() for name, value in factorized_inputs.items()
                        }
                    factorized_masks = (
                        None
                        if factorized_capability_masks is None
                        else {
                            name: value.to(device=pooled.device, dtype=torch.bool)
                            for name, value in factorized_capability_masks.items()
                        }
                    )
                if self.enzyme_input_mode in {
                    "raw_mean_sleec_hyperbolic_blockwise",
                    "raw_mean_sleec_hyperbolic_capability_blockwise",
                    "raw_mean_sleec_hyperbolic_factorized_capability_blockwise",
                }:
                    if self.enzyme_block_fusion is None:
                        raise RuntimeError("enzyme block fusion was not initialized")
                    block_result = self.enzyme_block_fusion(
                        raw_mean,
                        pooled,
                        z_tangent,
                        capability_vector=capability_input,
                        capability_mask=capability_branch_mask,
                        factorized_capability_vectors=factorized_inputs,
                        factorized_capability_masks=factorized_masks,
                        return_details=details is not None,
                    )
                    if details is not None:
                        target_input, block_details = block_result
                        details = dict(details)
                        details.update(block_details)
                    else:
                        target_input = block_result
                else:
                    if self.enzyme_feature_fusion is None:
                        raise RuntimeError("enzyme feature fusion was not initialized")
                    fusion_result = self.enzyme_feature_fusion(
                        raw_mean,
                        pooled,
                        z_tangent,
                        capability_vector=capability_input,
                        capability_mask=capability_branch_mask,
                        return_gates=details is not None,
                    )
                    if details is not None:
                        target_input, gate_weights = fusion_result
                        details = dict(details)
                        details["enzyme_fusion_gate_raw_mean"] = gate_weights[:, 0]
                        details["enzyme_fusion_gate_pooled"] = gate_weights[:, 1]
                        details["enzyme_fusion_gate_hyperbolic"] = gate_weights[:, 2]
                        if gate_weights.shape[1] > 3:
                            details["enzyme_fusion_gate_capability"] = gate_weights[:, 3]
                    else:
                        target_input = fusion_result
                if self.enzyme_input_mode == "raw_mean_sleec_hyperbolic_text_gated":
                    if self.text_feature_fusion is None:
                        raise RuntimeError("text_feature_fusion was not initialized")
                    if text_vectors is None:
                        raise ValueError(
                            "text_vectors are required for " "raw_mean_sleec_hyperbolic_text_gated"
                        )
                    text_input = text_vectors.to(device=pooled.device, dtype=pooled.dtype)
                    text_branch_mask = (
                        None
                        if text_mask is None
                        else text_mask.to(device=pooled.device, dtype=torch.bool)
                    )
                    if self.text_freeze:
                        text_input = text_input.detach()
                    if self.text_adapter is not None:
                        text_input = text_input + self.text_adapter(text_input)
                    text_fusion_result = self.text_feature_fusion(
                        target_input,
                        text_input,
                        text_mask=text_branch_mask,
                        return_details=details is not None,
                    )
                    if details is not None:
                        target_input, text_details = text_fusion_result
                        details = dict(details)
                        details.update(text_details)
                    else:
                        target_input = text_fusion_result
            else:
                target_input = hyperbolic_input
            if details is not None:
                details = dict(details)
                details["hyperbolic_tangent_norm"] = z_tangent.norm(dim=-1)
                details["hyperbolic_ball_norm"] = z_hyp.norm(dim=-1)

        if self.enzyme_input_mode == "raw_mean_sleec_blockwise":
            if raw_mean is None or self.enzyme_block_fusion is None:
                raise RuntimeError("enzyme block fusion was not initialized")
            block_result = self.enzyme_block_fusion(
                raw_mean,
                pooled,
                return_details=details is not None,
            )
            if details is not None:
                target_input, block_details = block_result
                details = dict(details)
                details.update(block_details)
            else:
                target_input = block_result

        target_embeddings = (
            target_input
            if self.blockwise_target_is_embedding
            else self.target_encoder(target_input)
        )
        target_embeddings, details = self._apply_r2e_adapter(
            target_embeddings,
            retrieval_direction,
            details if return_pooling_details else None,
        )
        if return_pooling_details:
            return target_embeddings, details
        if return_attention:
            return target_embeddings, attention
        return target_embeddings

    def forward(
        self,
        query_inputs: dict | torch.Tensor,
        residue_embeddings: torch.Tensor,
        residue_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        score_residue_embeddings: torch.Tensor | None = None,
        score_residue_padding_mask: torch.Tensor | None = None,
        score_attention_mask: torch.Tensor | None = None,
        capability_vectors: torch.Tensor | None = None,
        capability_mask: torch.Tensor | None = None,
        factorized_capability_vectors: dict[str, torch.Tensor] | None = None,
        factorized_capability_masks: dict[str, torch.Tensor] | None = None,
        text_vectors: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        return_attention: bool = False,
        return_pooling_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_embeddings = self.encode_queries(query_inputs)
        target_result = self.encode_targets(
            residue_embeddings,
            residue_padding_mask=residue_padding_mask,
            attention_mask=attention_mask,
            score_residue_embeddings=score_residue_embeddings,
            score_residue_padding_mask=score_residue_padding_mask,
            score_attention_mask=score_attention_mask,
            capability_vectors=capability_vectors,
            capability_mask=capability_mask,
            factorized_capability_vectors=factorized_capability_vectors,
            factorized_capability_masks=factorized_capability_masks,
            text_vectors=text_vectors,
            text_mask=text_mask,
            return_attention=return_attention,
            return_pooling_details=return_pooling_details,
        )
        if return_pooling_details:
            target_embeddings, details = target_result
            return query_embeddings, target_embeddings, details
        if return_attention:
            target_embeddings, attention = target_result
            return query_embeddings, target_embeddings, attention
        return query_embeddings, target_result
