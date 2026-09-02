"""
Lightning module for residue-level protein pooling dual encoders.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F

from horizyn.checkpoint_utils import load_query_encoder_checkpoint
from horizyn.losses import (
    BalancedSigmoidEBMLoss,
    BidirectionalAnchorBalancedSupConLoss,
    DecoupledAllPositiveInfoNCELoss,
    DegreeTemperedFullBatchMLNCELoss,
    HorizynFGWLoss,
    HybridCardinalityRetrievalLoss,
    MultiAlignmentRetrievalLoss,
    build_horizyn_loss,
)
from horizyn.metrics import create_retrieval_metrics
from horizyn.model import (
    HybridReactionEncoder,
    MultimodalReactionAttentionEncoder,
    ProteinPooledDualModel,
    UniMol2ReactionAttentionEncoder,
)
from horizyn.structural_similarity import (
    masked_mean_features,
    pairwise_cosine_similarity,
    reaction_feature_matrix,
)


class ProteinPooledLitModule(pl.LightningModule):
    """
    Train a Horizyn-style dual encoder with learned or mean residue pooling.
    """

    _VALID_POSITIVE_PAIR_SOURCES = {"observed_pairs", "all_known_in_batch"}
    _BALANCED_DIRECTIONS = ("reaction_to_enzyme", "enzyme_to_reaction")
    _BALANCED_METRICS = ("mrr", "reactzyme_mrr", "top_1")

    def __init__(
        self,
        query_encoder_dims: List[int],
        target_encoder_dims: List[int],
        embedding_dim: int = 512,
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
        biofp_seq_dim: int = 384,
        biofp_dim: int = 128,
        biofp_hidden_dim: int = 512,
        biofp_seq_weight: float = 0.75,
        biofp_dropout: float = 0.1,
        biofp_aux_weight: float = 0.0,
        biofp_center_weight: float = 0.45,
        biofp_transition_weight: float = 0.35,
        biofp_cofactor_weight: float = 0.20,
        biofp_family_weights: dict[str, float] | None = None,
        biofp_confidence_cap: float = 8.0,
        cross_tower_alignment_weight: float = 0.0,
        cross_tower_alignment_family_weights: dict[str, float] | None = None,
        reaction_attention_entropy_weight: float = 0.0,
        reaction_attention_min_normalized_entropy: float = 0.75,
        reaction_chemistry_consistency_weight: float = 0.0,
        reaction_residual_identity_weight: float = 0.0,
        enzyme_block_weight_kl_weight: float = 0.0,
        text_vector_dim: int | None = None,
        text_fusion_dim: int = 512,
        text_num_heads: int = 8,
        text_dropout: float = 0.1,
        text_freeze: bool = True,
        text_adapter: bool = False,
        reaction_hyperbolic_checkpoint_path: str | None = None,
        reaction_hyperbolic_hyp_dim: int | None = None,
        reaction_hyperbolic_freeze_encoder: bool = True,
        reaction_hyperbolic_freeze_projector: bool = True,
        reaction_hyperbolic_use_tangent: bool = True,
        beta: float = 10.0,
        learn_beta: bool = False,
        beta_min: float = -float("inf"),
        beta_max: float = float("inf"),
        loss_name: str = "FullBatchMLNCELoss",
        positive_pair_source: str = "observed_pairs",
        degree_alpha: float = 0.5,
        cardinality_weight: float = 0.3,
        cardinality_warmup_epochs: int = 5,
        separate_direction_temperatures: bool = False,
        beta_r2e: float = 10.0,
        beta_e2r: float = 10.0,
        temperature_regularization_weight: float = 0.0,
        soft_rank_weight: float = 0.0,
        soft_rank_tau: float = 0.1,
        soft_rank_top_k: int = 128,
        sigmoid_bias_init: float = 0.0,
        sigmoid_learn_bias: bool = True,
        sigmoid_negative_weight: float = 1.0,
        lambda_r: float = 0.05,
        lambda_e: float = 0.05,
        lambda_g: float = 0.01,
        tau_r: float = 0.1,
        tau_e: float = 0.1,
        tau_t: float = 0.1,
        delta_r: float = 0.5,
        delta_e: float = 0.5,
        symmetric_gw: bool = True,
        direction_balance_weight: float = 0.0,
        lambda_rr: float = 0.0,
        lambda_ee: float = 0.0,
        lambda_gw: float = 0.0,
        lambda_direction_gap: float = 0.0,
        lambda_r2e: float = 0.5,
        lambda_e2r: float = 0.5,
        lambda_r2e_hard_neg: float = 0.0,
        r2e_hard_neg_top_k: int = 0,
        r2e_hard_neg_margin: float = 0.0,
        lambda_e2r_hard_neg: float = 0.0,
        e2r_hard_neg_top_k: int = 0,
        e2r_hard_neg_margin: float = 0.0,
        tau_rr: float = 0.1,
        tau_ee: float = 0.1,
        tau_gw: float = 0.1,
        ec_positive_policy: str = "hierarchical_weighted",
        ec_min_shared_depth: int = 2,
        gw_max_anchors: int | None = 512,
        apply_structure_terms_on_val: bool = False,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        lambda_residue: float = 0.0,
        log_attention_stats: bool = True,
        attention_logging_interval: int = 100,
        query_encoder_type: str = "mlp",
        reaction_model_dim: int = 1024,
        reaction_unimol_dim: int = 768,
        reaction_chienn_dim: int = 256,
        reaction_chemistry_dim: int | None = None,
        reaction_directional_dim: int | None = None,
        reaction_use_model: bool = True,
        reaction_use_chienn: bool = True,
        reaction_use_chemistry: bool = False,
        reaction_use_directional: bool = False,
        reaction_chirality_name: str = "chiro",
        reaction_pooling: str = "attention",
        reaction_attention_bias: bool = True,
        reaction_separate_side_poolers: bool = True,
        reaction_modality_attention_hidden_dim: int | None = None,
        reaction_modality_attention_dropout: float = 0.0,
        reaction_modality_dropout: float = 0.0,
        reaction_chemistry_dropout: float = 0.0,
        reaction_modality_token_layer_norm: bool = False,
        reaction_modality_l2_normalize: bool = False,
        reaction_modality_encoder_num_layers: int | None = None,
        reaction_modality_encoder_widths: int | list[int] | None = None,
        reaction_modality_encoder_use_layer_norm: bool = False,
        reaction_modality_encoder_dropout: float = 0.0,
        reaction_modality_encoder_normalise_output: bool = False,
        reaction_side_composition: str = "directional_delta",
        reaction_modality_fusion: str = "attention",
        reaction_factorized_dims: dict[str, int] | None = None,
        reaction_factorized_weights: dict[str, float] | None = None,
        reaction_attention_prior_weights: dict[str, float] | None = None,
        reaction_attention_adaptation_strength: float = 0.4,
        reaction_output_projection: str = "mlp",
        reaction_residual_gate_init: float = 0.1,
        reaction_directional_gate_init: float = 0.1,
        query_encoder_checkpoint_path: str | None = None,
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
        e2r_identity_weight: float = 0.0,
        r2e_adapter_enabled: bool = False,
        r2e_adapter_hidden_dim: int = 512,
        r2e_adapter_dropout: float = 0.1,
        r2e_adapter_gate_init: float = 0.1,
        r2e_adapter_use_factorized_inputs: bool = False,
        r2e_adapter_block_dims: dict[str, int] | None = None,
        r2e_adapter_block_weights: dict[str, float] | None = None,
        r2e_identity_weight: float = 0.0,
        query_normalise_output: bool = True,
        target_normalise_output: bool = True,
        enforce_normalisation: bool = True,
        embedding_similarity: str = "cosine",
        validation_similarity: str | None = None,
        validation_retrieval_metrics: bool = False,
        validation_retrieval_directions: List[str] | None = None,
        retrieval_metric_top_k: List[int] | None = None,
        training_stage: str = "joint",
        detach_reaction_embeddings: bool = False,
        anchor_weight: float = 0.0,
        capability_consistency_weight: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        if enzyme_input_mode == "raw_mean_sleec_biological_factorized":
            if enzyme_block_dims is None and enzyme_block_weights is None:
                enzyme_block_dims = {
                    "core": 288,
                    "site": 96,
                    "mechanism": 64,
                    "cofactor": 32,
                    "ec": 32,
                }
                enzyme_block_weights = {
                    "core": 0.55,
                    "site": 0.20,
                    "mechanism": 0.12,
                    "cofactor": 0.08,
                    "ec": 0.05,
                }
            elif enzyme_block_dims is None or enzyme_block_weights is None:
                raise ValueError(
                    "Biological factorized enzyme blocks require both dims and weights"
                )

        if positive_pair_source not in self._VALID_POSITIVE_PAIR_SOURCES:
            raise ValueError(
                "positive_pair_source must be one of: "
                f"{sorted(self._VALID_POSITIVE_PAIR_SOURCES)}"
            )
        if ec_positive_policy != "hierarchical_weighted":
            raise ValueError("ec_positive_policy must be 'hierarchical_weighted'")
        if ec_min_shared_depth not in {2, 3, 4}:
            raise ValueError("ec_min_shared_depth must be one of: 2, 3, 4")
        if gw_max_anchors is not None and gw_max_anchors <= 0:
            raise ValueError("gw_max_anchors must be positive or None")
        if query_encoder_dims[-1] != embedding_dim:
            raise ValueError("query_encoder_dims final element must equal embedding_dim")
        if target_encoder_dims[-1] != embedding_dim:
            raise ValueError("target_encoder_dims final element must equal embedding_dim")
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
        }:
            raise ValueError(
                "enzyme_input_mode must be one of: standard, raw_sleec_hyperbolic_concat, "
                "raw_mean_sleec_hyperbolic_gated, "
                "raw_mean_sleec_hyperbolic_capability_gated, "
                "raw_mean_sleec_hyperbolic_text_gated, raw_mean_sleec_blockwise, "
                "raw_mean_sleec_hyperbolic_blockwise, "
                "raw_mean_sleec_hyperbolic_capability_blockwise, "
                "raw_mean_sleec_hyperbolic_factorized_capability_blockwise, "
                "raw_mean_sleec_biofp_split, raw_mean_sleec_biological_factorized"
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
        if enzyme_fusion_hidden_dim is not None and enzyme_fusion_hidden_dim <= 0:
            raise ValueError("enzyme_fusion_hidden_dim must be positive or None")
        if not (0.0 <= enzyme_fusion_dropout <= 1.0):
            raise ValueError("enzyme_fusion_dropout must be in the range [0, 1]")
        if enzyme_input_mode in {
            "raw_mean_sleec_hyperbolic_capability_gated",
            "raw_mean_sleec_hyperbolic_capability_blockwise",
        }:
            if capability_vector_dim is None or capability_vector_dim <= 0:
                raise ValueError(
                    "capability_vector_dim must be positive for " f"{enzyme_input_mode}"
                )
            if not (0.0 <= capability_dropout <= 1.0):
                raise ValueError("capability_dropout must be in the range [0, 1]")
        if enzyme_input_mode in {
            "raw_mean_sleec_blockwise",
            "raw_mean_sleec_hyperbolic_blockwise",
            "raw_mean_sleec_hyperbolic_capability_blockwise",
            "raw_mean_sleec_hyperbolic_factorized_capability_blockwise",
            "raw_mean_sleec_biological_factorized",
        }:
            if not (0.0 <= enzyme_block_dropout <= 1.0):
                raise ValueError("enzyme_block_dropout must be in the range [0, 1]")
            if target_encoder_dims[-1] != embedding_dim:
                raise ValueError("target_encoder_dims final element must equal embedding_dim")
        if enzyme_input_mode == "raw_mean_sleec_hyperbolic_text_gated":
            if text_vector_dim is None or text_vector_dim <= 0:
                raise ValueError(
                    "text_vector_dim must be positive for " "raw_mean_sleec_hyperbolic_text_gated"
                )
            if text_fusion_dim <= 0:
                raise ValueError("text_fusion_dim must be positive")
            if text_num_heads <= 0:
                raise ValueError("text_num_heads must be positive")
            if text_fusion_dim % text_num_heads != 0:
                raise ValueError("text_fusion_dim must be divisible by text_num_heads")
            if not (0.0 <= text_dropout <= 1.0):
                raise ValueError("text_dropout must be in the range [0, 1]")
        if enzyme_input_mode == "raw_mean_sleec_biofp_split":
            if biofp_seq_dim <= 0 or biofp_dim <= 0 or biofp_hidden_dim <= 0:
                raise ValueError("BioFP split dimensions must be positive")
            if biofp_seq_dim + biofp_dim != embedding_dim:
                raise ValueError("biofp_seq_dim + biofp_dim must equal embedding_dim")
            if not (0.0 <= biofp_seq_weight <= 1.0):
                raise ValueError("biofp_seq_weight must be in [0, 1]")
            if not (0.0 <= biofp_dropout <= 1.0):
                raise ValueError("biofp_dropout must be in [0, 1]")
            for name, value in {
                "biofp_center_dim": biofp_center_dim,
                "biofp_cofactor_dim": biofp_cofactor_dim,
                "biofp_transition_dim": biofp_transition_dim,
            }.items():
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")
            if biofp_aux_weight > 0 and not any(
                value > 0 for value in (biofp_center_dim, biofp_cofactor_dim, biofp_transition_dim)
            ):
                raise ValueError("biofp_aux_weight > 0 requires at least one BioFP head")
        if enzyme_input_mode == "raw_mean_sleec_biological_factorized":
            biological_families = {
                name for name in (enzyme_block_dims or {}) if name not in {"core", "site", "ec"}
            }
            if biological_families and not biofp_family_dims:
                raise ValueError("Biological family blocks require biofp_family_dims")
            family_dims = biofp_family_dims or {}
            if set(family_dims) - biological_families:
                raise ValueError("biofp_family_dims contains an unconfigured family block")
            if any(int(value) < 0 for value in family_dims.values()):
                raise ValueError("biofp_family_dims values must be non-negative")
            if biofp_aux_weight > 0 and not any(int(value) > 0 for value in family_dims.values()):
                raise ValueError("biofp_aux_weight > 0 requires at least one family head")
        if biofp_aux_weight < 0:
            raise ValueError("biofp_aux_weight must be non-negative")
        if biofp_confidence_cap <= 0:
            raise ValueError("biofp_confidence_cap must be positive")
        if cross_tower_alignment_weight < 0:
            raise ValueError("cross_tower_alignment_weight must be non-negative")
        alignment_family_weights = {
            str(name): float(weight)
            for name, weight in (cross_tower_alignment_family_weights or {}).items()
        }
        if any(weight < 0 for weight in alignment_family_weights.values()):
            raise ValueError("cross_tower_alignment_family_weights must be non-negative")
        if cross_tower_alignment_weight > 0:
            if enzyme_input_mode != "raw_mean_sleec_biological_factorized":
                raise ValueError(
                    "cross-tower factor alignment requires "
                    "enzyme_input_mode='raw_mean_sleec_biological_factorized'"
                )
            biological_families = {
                name for name in (enzyme_block_dims or {}) if name not in {"core", "site", "ec"}
            }
            unknown_families = set(alignment_family_weights) - biological_families
            if unknown_families:
                raise ValueError(
                    "cross_tower_alignment_family_weights contains unconfigured family blocks: "
                    f"{sorted(unknown_families)}"
                )
            family_dims = biofp_family_dims or {}
            supervised_weight = sum(
                weight
                for family, weight in alignment_family_weights.items()
                if int(family_dims.get(family, 0)) > 0
            )
            if supervised_weight <= 0:
                raise ValueError(
                    "cross-tower factor alignment requires at least one positive-weight "
                    "family with annotation targets"
                )
        if reaction_attention_entropy_weight < 0:
            raise ValueError("reaction_attention_entropy_weight must be non-negative")
        if reaction_chemistry_consistency_weight < 0:
            raise ValueError("reaction_chemistry_consistency_weight must be non-negative")
        if reaction_residual_identity_weight < 0:
            raise ValueError("reaction_residual_identity_weight must be non-negative")
        if not 0.0 <= reaction_attention_min_normalized_entropy <= 1.0:
            raise ValueError("reaction_attention_min_normalized_entropy must be in [0, 1]")
        if enzyme_block_weight_kl_weight < 0:
            raise ValueError("enzyme_block_weight_kl_weight must be non-negative")
        if hyperbolic_checkpoint_path:
            expected_target_input_dim = hyperbolic_hyp_dim
            if (
                enzyme_input_mode == "raw_sleec_hyperbolic_concat"
                and hyperbolic_hyp_dim is not None
            ):
                expected_target_input_dim = residue_dim + hyperbolic_hyp_dim
        else:
            expected_target_input_dim = residue_dim
        if (
            enzyme_input_mode
            not in {
                "raw_mean_sleec_hyperbolic_gated",
                "raw_mean_sleec_hyperbolic_capability_gated",
                "raw_mean_sleec_hyperbolic_text_gated",
                "raw_mean_sleec_blockwise",
                "raw_mean_sleec_hyperbolic_blockwise",
                "raw_mean_sleec_hyperbolic_capability_blockwise",
                "raw_mean_sleec_hyperbolic_factorized_capability_blockwise",
                "raw_mean_sleec_biofp_split",
                "raw_mean_sleec_biological_factorized",
            }
            and expected_target_input_dim is not None
            and target_encoder_dims[0] != expected_target_input_dim
        ):
            raise ValueError(
                "target_encoder_dims first element must equal the enzyme representation dim "
                f"({expected_target_input_dim}); got {target_encoder_dims[0]}"
            )
        expected_query_input_dim = None
        if reaction_hyperbolic_checkpoint_path and reaction_hyperbolic_hyp_dim is not None:
            expected_query_input_dim = (
                reaction_hyperbolic_hyp_dim
                if reaction_hyperbolic_use_tangent
                else reaction_hyperbolic_hyp_dim + 1
            )
        if (
            expected_query_input_dim is not None
            and query_encoder_dims[0] != expected_query_input_dim
        ):
            raise ValueError(
                "query_encoder_dims first element must equal the reaction hyperbolic "
                f"representation dim ({expected_query_input_dim}); got {query_encoder_dims[0]}"
            )
        if reaction_fingerprint_attention_enabled:
            if reaction_hyperbolic_checkpoint_path:
                raise ValueError(
                    "reaction_fingerprint_attention cannot be enabled together with "
                    "reaction_hyperbolic_checkpoint_path"
                )
            if query_encoder_dims[0] != reaction_fingerprint_attention_token_dim:
                raise ValueError(
                    "query_encoder_dims first element must equal the reaction fingerprint "
                    f"attention token dim ({reaction_fingerprint_attention_token_dim}); "
                    f"got {query_encoder_dims[0]}"
                )
        if query_encoder_type not in {
            "mlp",
            "unimol2_reaction_attention",
            "hybrid_reaction",
            "multimodal_reaction_attention",
        }:
            raise ValueError(
                "query_encoder_type must be one of: mlp, unimol2_reaction_attention, "
                "hybrid_reaction, multimodal_reaction_attention"
            )
        if reaction_pooling not in {"attention", "mean"}:
            raise ValueError("reaction_pooling must be one of: attention, mean")
        if embedding_similarity not in {"cosine", "dot"}:
            raise ValueError("embedding_similarity must be one of: cosine, dot")
        if validation_similarity is None:
            validation_similarity = embedding_similarity
        if validation_similarity not in {"cosine", "dot"}:
            raise ValueError("validation_similarity must be one of: cosine, dot")
        if (
            query_encoder_type == "unimol2_reaction_attention"
            and query_encoder_dims[0] != 4 * reaction_unimol_dim
        ):
            raise ValueError("Uni-Mol2 reaction query input dim must equal 4 * reaction_unimol_dim")

        query_hidden_widths = query_encoder_dims[1:-1]
        target_hidden_widths = target_encoder_dims[1:-1]
        query_encoder_class = None
        if query_encoder_type == "unimol2_reaction_attention":
            query_encoder_class = UniMol2ReactionAttentionEncoder
        elif query_encoder_type == "hybrid_reaction":
            query_encoder_class = HybridReactionEncoder
        elif query_encoder_type == "multimodal_reaction_attention":
            query_encoder_class = MultimodalReactionAttentionEncoder
        query_encoder_kwargs = {
            "input_dim": query_encoder_dims[0],
            "output_dim": embedding_dim,
            "num_layers": len(query_hidden_widths),
            "widths": query_hidden_widths,
            "normalise_output": query_normalise_output,
        }
        if query_encoder_type in {"unimol2_reaction_attention", "hybrid_reaction"}:
            query_encoder_kwargs.update(
                {
                    "unimol_dim": reaction_unimol_dim,
                    "reaction_pooling": reaction_pooling,
                    "attention_bias": reaction_attention_bias,
                    "separate_side_poolers": reaction_separate_side_poolers,
                }
            )
        elif query_encoder_type == "multimodal_reaction_attention":
            query_encoder_kwargs.update(
                {
                    "reaction_model_dim": reaction_model_dim,
                    "unimol_dim": reaction_unimol_dim,
                    "chienn_dim": reaction_chienn_dim,
                    "reaction_chemistry_dim": reaction_chemistry_dim,
                    "reaction_directional_dim": reaction_directional_dim,
                    "use_reaction_model": reaction_use_model,
                    "use_chienn": reaction_use_chienn,
                    "use_reaction_chemistry": reaction_use_chemistry,
                    "use_reaction_directional": reaction_use_directional,
                    "chirality_modality_name": reaction_chirality_name,
                    "reaction_pooling": reaction_pooling,
                    "attention_bias": reaction_attention_bias,
                    "separate_side_poolers": reaction_separate_side_poolers,
                    "modality_attention_hidden_dim": reaction_modality_attention_hidden_dim,
                    "modality_attention_dropout": reaction_modality_attention_dropout,
                    "modality_dropout": reaction_modality_dropout,
                    "chemistry_dropout": reaction_chemistry_dropout,
                    "modality_token_layer_norm": reaction_modality_token_layer_norm,
                    "modality_l2_normalize": reaction_modality_l2_normalize,
                    "modality_encoder_num_layers": reaction_modality_encoder_num_layers,
                    "modality_encoder_widths": reaction_modality_encoder_widths,
                    "modality_encoder_use_layer_norm": reaction_modality_encoder_use_layer_norm,
                    "modality_encoder_dropout": reaction_modality_encoder_dropout,
                    "modality_encoder_normalise_output": (
                        reaction_modality_encoder_normalise_output
                    ),
                    "side_composition": reaction_side_composition,
                    "modality_fusion": reaction_modality_fusion,
                    "factorized_dims": reaction_factorized_dims,
                    "factorized_weights": reaction_factorized_weights,
                    "attention_prior_weights": reaction_attention_prior_weights,
                    "attention_adaptation_strength": reaction_attention_adaptation_strength,
                    "output_projection": reaction_output_projection,
                    "residual_gate_init": reaction_residual_gate_init,
                    "directional_gate_init": reaction_directional_gate_init,
                }
            )

        model_kwargs = {
            "query_encoder_kwargs": query_encoder_kwargs,
            "target_encoder_kwargs": {
                "input_dim": target_encoder_dims[0],
                "output_dim": embedding_dim,
                "num_layers": len(target_hidden_widths),
                "widths": target_hidden_widths,
                "normalise_output": target_normalise_output,
            },
            "residue_dim": residue_dim,
            "pooling": pooling,
            "attention_bias": attention_bias,
            "sleec_mode": sleec_mode,
            "sleec_topk_fraction": sleec_topk_fraction,
            "sleec_threshold": sleec_threshold,
            "sleec_scorer_hidden_dim": sleec_scorer_hidden_dim,
            "sleec_score_hidden_dim": sleec_score_hidden_dim,
            "sleec_checkpoint_path": sleec_checkpoint_path,
            "sleec_freeze_scorer": sleec_freeze_scorer,
            "sleec_guided_initial_bias_scale": sleec_guided_initial_bias_scale,
            "sleec_guided_train_bias_scale": sleec_guided_train_bias_scale,
            "hyperbolic_checkpoint_path": hyperbolic_checkpoint_path,
            "hyperbolic_freeze_projector": hyperbolic_freeze_projector,
            "hyperbolic_use_tangent": hyperbolic_use_tangent,
            "hyperbolic_hyp_dim": hyperbolic_hyp_dim,
            "hyperbolic_load_attention_pooler": hyperbolic_load_attention_pooler,
            "hyperbolic_freeze_attention_pooler": hyperbolic_freeze_attention_pooler,
            "enzyme_input_mode": enzyme_input_mode,
            "enzyme_fusion_hidden_dim": enzyme_fusion_hidden_dim,
            "enzyme_fusion_dropout": enzyme_fusion_dropout,
            "enzyme_block_dims": enzyme_block_dims,
            "enzyme_block_weights": enzyme_block_weights,
            "enzyme_block_dropout": enzyme_block_dropout,
            "enzyme_block_learned_weights": enzyme_block_learned_weights,
            "capability_vector_dim": capability_vector_dim,
            "capability_freeze": capability_freeze,
            "capability_adapter": capability_adapter,
            "capability_dropout": capability_dropout,
            "factorized_capability_dims": factorized_capability_dims,
            "factorized_capability_freeze": factorized_capability_freeze,
            "factorized_capability_use_masks": factorized_capability_use_masks,
            "biofp_center_dim": biofp_center_dim,
            "biofp_cofactor_dim": biofp_cofactor_dim,
            "biofp_transition_dim": biofp_transition_dim,
            "biofp_family_dims": biofp_family_dims,
            "biofp_seq_dim": biofp_seq_dim,
            "biofp_dim": biofp_dim,
            "biofp_hidden_dim": biofp_hidden_dim,
            "biofp_seq_weight": biofp_seq_weight,
            "biofp_dropout": biofp_dropout,
            "text_vector_dim": text_vector_dim,
            "text_fusion_dim": text_fusion_dim,
            "text_num_heads": text_num_heads,
            "text_dropout": text_dropout,
            "text_freeze": text_freeze,
            "text_adapter": text_adapter,
            "reaction_hyperbolic_checkpoint_path": reaction_hyperbolic_checkpoint_path,
            "reaction_hyperbolic_freeze_encoder": reaction_hyperbolic_freeze_encoder,
            "reaction_hyperbolic_freeze_projector": reaction_hyperbolic_freeze_projector,
            "reaction_hyperbolic_use_tangent": reaction_hyperbolic_use_tangent,
            "reaction_fingerprint_attention_enabled": reaction_fingerprint_attention_enabled,
            "reaction_fingerprint_attention_input_dim": reaction_fingerprint_attention_input_dim,
            "reaction_fingerprint_attention_rdkit_dim": reaction_fingerprint_attention_rdkit_dim,
            "reaction_fingerprint_attention_drfp_dim": reaction_fingerprint_attention_drfp_dim,
            "reaction_fingerprint_attention_token_dim": reaction_fingerprint_attention_token_dim,
            "reaction_fingerprint_attention_hidden_dim": reaction_fingerprint_attention_hidden_dim,
            "reaction_fingerprint_attention_dropout": reaction_fingerprint_attention_dropout,
            "reaction_fingerprint_attention_bias": reaction_fingerprint_attention_bias,
            "e2r_adapter_enabled": e2r_adapter_enabled,
            "e2r_adapter_hidden_dim": e2r_adapter_hidden_dim,
            "e2r_adapter_dropout": e2r_adapter_dropout,
            "e2r_adapter_gate_init": e2r_adapter_gate_init,
            "e2r_adapter_use_factorized_inputs": e2r_adapter_use_factorized_inputs,
            "e2r_adapter_use_directional_inputs": e2r_adapter_use_directional_inputs,
            "e2r_adapter_directional_dim": e2r_adapter_directional_dim,
            "e2r_adapter_directional_hidden_dim": e2r_adapter_directional_hidden_dim,
            "e2r_adapter_reaction_model_dim": reaction_model_dim,
            "e2r_adapter_unimol_dim": reaction_unimol_dim,
            "e2r_adapter_chiro_dim": reaction_chienn_dim,
            "e2r_adapter_chemistry_dim": reaction_chemistry_dim,
            "r2e_adapter_enabled": r2e_adapter_enabled,
            "r2e_adapter_hidden_dim": r2e_adapter_hidden_dim,
            "r2e_adapter_dropout": r2e_adapter_dropout,
            "r2e_adapter_gate_init": r2e_adapter_gate_init,
            "r2e_adapter_use_factorized_inputs": r2e_adapter_use_factorized_inputs,
            "r2e_adapter_block_dims": r2e_adapter_block_dims,
            "r2e_adapter_block_weights": r2e_adapter_block_weights,
            "enforce_normalisation": enforce_normalisation,
        }
        if query_encoder_class is not None:
            model_kwargs["query_encoder"] = query_encoder_class
        self.model = ProteinPooledDualModel(**model_kwargs)
        load_query_encoder_checkpoint(
            self.model.query_encoder,
            query_encoder_checkpoint_path,
        )
        self.loss_fn = build_horizyn_loss(
            name=loss_name,
            beta=beta,
            learn_beta=learn_beta,
            beta_min=beta_min,
            beta_max=beta_max,
            degree_alpha=degree_alpha,
            cardinality_weight=cardinality_weight,
            cardinality_warmup_epochs=cardinality_warmup_epochs,
            separate_direction_temperatures=separate_direction_temperatures,
            beta_r2e=beta_r2e,
            beta_e2r=beta_e2r,
            temperature_regularization_weight=temperature_regularization_weight,
            soft_rank_weight=soft_rank_weight,
            soft_rank_tau=soft_rank_tau,
            soft_rank_top_k=soft_rank_top_k,
            sigmoid_bias_init=sigmoid_bias_init,
            sigmoid_learn_bias=sigmoid_learn_bias,
            sigmoid_negative_weight=sigmoid_negative_weight,
            lambda_r=lambda_r,
            lambda_e=lambda_e,
            lambda_g=lambda_g,
            tau_r=tau_r,
            tau_e=tau_e,
            tau_t=tau_t,
            delta_r=delta_r,
            delta_e=delta_e,
            symmetric_gw=symmetric_gw,
            direction_balance_weight=direction_balance_weight,
            lambda_rr=lambda_rr,
            lambda_ee=lambda_ee,
            lambda_gw=lambda_gw,
            lambda_direction_gap=lambda_direction_gap,
            lambda_r2e=lambda_r2e,
            lambda_e2r=lambda_e2r,
            lambda_r2e_hard_neg=lambda_r2e_hard_neg,
            r2e_hard_neg_top_k=r2e_hard_neg_top_k,
            r2e_hard_neg_margin=r2e_hard_neg_margin,
            lambda_e2r_hard_neg=lambda_e2r_hard_neg,
            e2r_hard_neg_top_k=e2r_hard_neg_top_k,
            e2r_hard_neg_margin=e2r_hard_neg_margin,
            tau_rr=tau_rr,
            tau_ee=tau_ee,
            tau_gw=tau_gw,
            gw_max_anchors=gw_max_anchors,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.positive_pair_source = positive_pair_source
        self.ec_positive_policy = ec_positive_policy
        self.ec_min_shared_depth = int(ec_min_shared_depth)
        self.apply_structure_terms_on_val = bool(apply_structure_terms_on_val)
        self.lambda_residue = float(lambda_residue)
        self.log_attention_stats = log_attention_stats
        self.attention_logging_interval = attention_logging_interval
        self.validation_retrieval_metrics = bool(validation_retrieval_metrics)
        directions = (
            ["reaction_to_enzyme", "enzyme_to_reaction"]
            if validation_retrieval_directions is None
            else list(validation_retrieval_directions)
        )
        valid_directions = set(self._BALANCED_DIRECTIONS)
        invalid_directions = sorted(set(directions) - valid_directions)
        if invalid_directions:
            raise ValueError(
                "validation_retrieval_directions entries must be drawn from "
                f"{sorted(valid_directions)}; got {invalid_directions}"
            )
        if not directions:
            raise ValueError("validation_retrieval_directions must not be empty")
        self.validation_retrieval_directions = tuple(dict.fromkeys(directions))
        self.embedding_similarity = embedding_similarity
        self.validation_similarity = validation_similarity
        self.training_stage = str(training_stage or "joint")
        if self.training_stage not in {
            "joint",
            "enzyme_only_tuning",
            "joint_capability_retrieval",
            "e2r_adapter",
            "r2e_adapter",
            "bidirectional_adapters",
        }:
            raise ValueError(
                "training_stage must be one of: joint, enzyme_only_tuning, "
                "joint_capability_retrieval, e2r_adapter, r2e_adapter, "
                "bidirectional_adapters"
            )
        self.detach_reaction_embeddings = bool(detach_reaction_embeddings)
        if self.training_stage == "enzyme_only_tuning":
            self.detach_reaction_embeddings = True
        self.anchor_weight = float(anchor_weight)
        self.e2r_identity_weight = float(e2r_identity_weight)
        if self.e2r_identity_weight < 0:
            raise ValueError("e2r_identity_weight must be non-negative")
        if self.training_stage == "e2r_adapter" and self.model.e2r_adapter is None:
            raise ValueError("training_stage=e2r_adapter requires an enabled E2R adapter")
        self.r2e_identity_weight = float(r2e_identity_weight)
        if self.r2e_identity_weight < 0:
            raise ValueError("r2e_identity_weight must be non-negative")
        if self.training_stage == "r2e_adapter" and self.model.r2e_adapter is None:
            raise ValueError("training_stage=r2e_adapter requires an enabled R2E adapter")
        if self.training_stage == "bidirectional_adapters" and (
            self.model.e2r_adapter is None or self.model.r2e_adapter is None
        ):
            raise ValueError(
                "training_stage=bidirectional_adapters requires enabled E2R and R2E adapters"
            )
        self.capability_consistency_weight = float(capability_consistency_weight)
        self.biofp_aux_weight = float(biofp_aux_weight)
        self.biofp_family_weights = (
            {str(name): float(weight) for name, weight in biofp_family_weights.items()}
            if biofp_family_weights is not None
            else {
                "center": float(biofp_center_weight),
                "cofactor": float(biofp_cofactor_weight),
                "transition": float(biofp_transition_weight),
            }
        )
        self.biofp_confidence_cap = float(biofp_confidence_cap)
        self.cross_tower_alignment_weight = float(cross_tower_alignment_weight)
        alignment_weight_sum = sum(alignment_family_weights.values())
        self.cross_tower_alignment_family_weights = (
            {
                family: weight / alignment_weight_sum
                for family, weight in alignment_family_weights.items()
                if weight > 0
            }
            if alignment_weight_sum > 0
            else {}
        )
        self.reaction_attention_entropy_weight = float(reaction_attention_entropy_weight)
        self.reaction_attention_min_normalized_entropy = float(
            reaction_attention_min_normalized_entropy
        )
        self.reaction_chemistry_consistency_weight = float(reaction_chemistry_consistency_weight)
        self.reaction_residual_identity_weight = float(reaction_residual_identity_weight)
        self.enzyme_block_weight_kl_weight = float(enzyme_block_weight_kl_weight)
        self.capability_consistency_projection: torch.nn.Module | None = None
        if self.capability_consistency_weight > 0:
            if capability_vector_dim is None:
                raise ValueError("capability_consistency_weight > 0 requires capability_vector_dim")
            self.capability_consistency_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(int(capability_vector_dim)),
                torch.nn.Linear(int(capability_vector_dim), int(embedding_dim)),
            )
        self._apply_training_stage_freezes()
        self.retrieval_metric_top_k = tuple(
            int(k) for k in (retrieval_metric_top_k or [1, 10, 100, 1000])
        )
        self.metric_functionals = create_retrieval_metrics(
            top_k=list(self.retrieval_metric_top_k),
            include_r_precision=True,
            include_avg_precision=True,
        )
        self.val_target_lookup_table: torch.Tensor | None = None
        self.val_query_lookup_table: torch.Tensor | None = None
        self.val_target_id_to_idx: dict[str, int] = {}
        self.val_query_id_to_idx: dict[str, int] = {}
        self._balanced_metric_sums: dict[str, dict[str, torch.Tensor]] = {}
        self._balanced_metric_counts: dict[str, dict[str, torch.Tensor]] = {}

    @staticmethod
    def _set_requires_grad(module: torch.nn.Module | None, requires_grad: bool) -> None:
        if module is None:
            return
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad

    def _freeze_sleec_scorer_if_present(self) -> None:
        pooling = getattr(self.model, "pooling", None)
        scorer = getattr(pooling, "scorer", None)
        self._set_requires_grad(scorer, False)

    def _apply_training_stage_freezes(self) -> None:
        if self.training_stage == "joint":
            return
        if self.training_stage == "e2r_adapter":
            self._set_requires_grad(self.model, False)
            self._set_requires_grad(getattr(self.model, "e2r_adapter", None), True)
            return
        if self.training_stage == "r2e_adapter":
            self._set_requires_grad(self.model, False)
            self._set_requires_grad(getattr(self.model, "r2e_adapter", None), True)
            return
        if self.training_stage == "bidirectional_adapters":
            self._set_requires_grad(self.model, False)
            self._set_requires_grad(getattr(self.model, "e2r_adapter", None), True)
            self._set_requires_grad(getattr(self.model, "r2e_adapter", None), True)
            return
        self._set_requires_grad(getattr(self.model, "hyperbolic_projector", None), False)
        self._set_requires_grad(getattr(self.model, "reaction_hyperbolic_encoder", None), False)
        self._set_requires_grad(getattr(self.model, "reaction_hyperbolic_projector", None), False)
        self._freeze_sleec_scorer_if_present()
        if self.training_stage == "enzyme_only_tuning":
            self._set_requires_grad(getattr(self.model, "query_encoder", None), False)
            self._set_requires_grad(
                getattr(self.model, "reaction_fingerprint_attention", None), False
            )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and getattr(self, "training_stage", None) in {
            "e2r_adapter",
            "r2e_adapter",
            "bidirectional_adapters",
        }:
            self.model.eval()
            adapter_names = (
                ("e2r_adapter", "r2e_adapter")
                if self.training_stage == "bidirectional_adapters"
                else (self.training_stage,)
            )
            for adapter_name in adapter_names:
                adapter = getattr(self.model, adapter_name, None)
                if adapter is not None:
                    adapter.train(True)
        return self

    def _distributed_enabled(self) -> bool:
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    def _gather_objects(self, obj: Any) -> List[Any]:
        if not self._distributed_enabled():
            return [obj]
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, obj)
        return gathered

    def _gather_variable_tensor(
        self,
        tensor: torch.Tensor,
        sync_grads: bool = False,
    ) -> List[torch.Tensor]:
        if not self._distributed_enabled():
            return [tensor]

        local_size = torch.tensor([tensor.shape[0]], dtype=torch.long, device=self.device)
        gathered_sizes = self.all_gather(local_size)
        if gathered_sizes.dim() == local_size.dim():
            gathered_sizes = gathered_sizes.unsqueeze(0)
        gathered_sizes = gathered_sizes.reshape(-1)
        max_size = int(gathered_sizes.max().item())

        if tensor.shape[0] < max_size:
            pad_shape = (max_size - tensor.shape[0], *tensor.shape[1:])
            tensor = torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=0)

        gathered = self.all_gather(tensor, sync_grads=sync_grads)
        if gathered.dim() == tensor.dim():
            gathered = gathered.unsqueeze(0)

        return [
            gathered[rank, : int(gathered_sizes[rank].item())]
            for rank in range(gathered_sizes.numel())
        ]

    @staticmethod
    def _deduplicate_query_vectors(
        query_vecs: torch.Tensor | dict[str, torch.Tensor],
        query_ids: List[str],
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor, List[str]]:
        id_to_idx: dict[str, int] = {}
        keep_indices = []
        inverse_indices = []
        unique_ids = []
        for row_idx, query_id in enumerate(query_ids):
            if query_id not in id_to_idx:
                id_to_idx[query_id] = len(keep_indices)
                keep_indices.append(row_idx)
                unique_ids.append(query_id)
            inverse_indices.append(id_to_idx[query_id])

        first_tensor = (
            next(value for value in query_vecs.values() if torch.is_tensor(value))
            if isinstance(query_vecs, dict)
            else query_vecs
        )
        device = first_tensor.device
        keep = torch.tensor(keep_indices, dtype=torch.long, device=device)
        inverse = torch.tensor(inverse_indices, dtype=torch.long, device=device)
        if isinstance(query_vecs, dict):
            unique_query_vecs = {
                key: (
                    value[keep]
                    if torch.is_tensor(value) and value.shape[0] == len(query_ids)
                    else value
                )
                for key, value in query_vecs.items()
            }
        else:
            unique_query_vecs = query_vecs[keep]
        return unique_query_vecs, inverse, unique_ids

    @staticmethod
    def _deduplicate_residue_tensors(
        residue_embeddings: torch.Tensor,
        residue_padding_mask: torch.Tensor,
        target_ids: List[str],
        score_residue_embeddings: torch.Tensor | None = None,
        score_residue_padding_mask: torch.Tensor | None = None,
        residue_labels: torch.Tensor | None = None,
        residue_label_mask: torch.Tensor | None = None,
        capability_vectors: torch.Tensor | None = None,
        capability_mask: torch.Tensor | None = None,
        text_vectors: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        List[str],
    ]:
        id_to_idx: dict[str, int] = {}
        keep_indices = []
        inverse_indices = []
        unique_ids = []
        for row_idx, target_id in enumerate(target_ids):
            if target_id not in id_to_idx:
                id_to_idx[target_id] = len(keep_indices)
                keep_indices.append(row_idx)
                unique_ids.append(target_id)
            inverse_indices.append(id_to_idx[target_id])

        device = residue_embeddings.device
        keep = torch.tensor(keep_indices, dtype=torch.long, device=device)
        inverse = torch.tensor(inverse_indices, dtype=torch.long, device=device)
        unique_score_residues = (
            score_residue_embeddings[keep] if score_residue_embeddings is not None else None
        )
        unique_score_mask = (
            score_residue_padding_mask[keep] if score_residue_padding_mask is not None else None
        )
        unique_labels = residue_labels[keep] if residue_labels is not None else None
        unique_label_mask = residue_label_mask[keep] if residue_label_mask is not None else None
        unique_capability_vectors = (
            capability_vectors[keep] if capability_vectors is not None else None
        )
        unique_capability_mask = capability_mask[keep] if capability_mask is not None else None
        unique_text_vectors = text_vectors[keep] if text_vectors is not None else None
        unique_text_mask = text_mask[keep] if text_mask is not None else None
        return (
            residue_embeddings[keep],
            residue_padding_mask[keep],
            unique_score_residues,
            unique_score_mask,
            unique_labels,
            unique_label_mask,
            unique_capability_vectors,
            unique_capability_mask,
            unique_text_vectors,
            unique_text_mask,
            inverse,
            unique_ids,
        )

    @staticmethod
    def _deduplicate_target_tensor_dict(
        target_ids: List[str],
        tensors: dict[str, torch.Tensor],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if not tensors:
            return {}
        id_to_idx: dict[str, int] = {}
        keep_indices = []
        for row_idx, target_id in enumerate(target_ids):
            if target_id not in id_to_idx:
                id_to_idx[target_id] = len(keep_indices)
                keep_indices.append(row_idx)
        keep = torch.tensor(keep_indices, dtype=torch.long, device=device)
        return {
            name: tensor[keep] if tensor.shape[0] == len(target_ids) else tensor
            for name, tensor in tensors.items()
        }

    def _compute_cosine_distances(
        self,
        query_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
    ) -> torch.Tensor:
        query_embeds = F.normalize(query_embeds, p=2, dim=-1, eps=1e-12)
        target_embeds = F.normalize(target_embeds, p=2, dim=-1, eps=1e-12)
        return 1.0 - torch.matmul(query_embeds, target_embeds.T)

    def _compute_dot_distances(
        self,
        query_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
    ) -> torch.Tensor:
        return -torch.matmul(query_embeds.float(), target_embeds.float().T)

    def _compute_embedding_distances(
        self,
        query_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if self.embedding_similarity == "dot":
            return self._compute_dot_distances(query_embeds, target_embeds)
        return self._compute_cosine_distances(query_embeds, target_embeds)

    def _compute_cosine_scores(
        self,
        query_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
    ) -> torch.Tensor:
        query_embeds = F.normalize(query_embeds, p=2, dim=-1, eps=1e-12)
        target_embeds = F.normalize(target_embeds, p=2, dim=-1, eps=1e-12)
        return torch.matmul(query_embeds, target_embeds.T)

    def _compute_dot_scores(
        self,
        query_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
    ) -> torch.Tensor:
        return torch.matmul(query_embeds.float(), target_embeds.float().T)

    def _compute_retrieval_scores(
        self,
        query_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if self.validation_similarity == "dot":
            return self._compute_dot_scores(query_embeds, target_embeds)
        return self._compute_cosine_scores(query_embeds, target_embeds)

    def _loss_with_components(
        self,
        dists: torch.Tensor,
        query_idx: torch.Tensor,
        target_idx: torch.Tensor,
        query_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
        reaction_similarity: Optional[torch.Tensor] = None,
        enzyme_similarity: Optional[torch.Tensor] = None,
        reaction_ec_weights: Optional[torch.Tensor] = None,
        enzyme_ec_weights: Optional[torch.Tensor] = None,
        reaction_degrees: Optional[torch.Tensor] = None,
        e2r_dists: Optional[torch.Tensor] = None,
        structure_terms_enabled: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        query_embeds = F.normalize(query_embeds, p=2, dim=-1, eps=1e-12)
        target_embeds = F.normalize(target_embeds, p=2, dim=-1, eps=1e-12)
        if isinstance(self.loss_fn, MultiAlignmentRetrievalLoss):
            loss, components = self.loss_fn(
                dists,
                query_idx,
                target_idx,
                query_embeds=query_embeds,
                target_embeds=target_embeds,
                reaction_similarity=reaction_similarity,
                enzyme_similarity=enzyme_similarity,
                reaction_ec_weights=reaction_ec_weights,
                enzyme_ec_weights=enzyme_ec_weights,
                e2r_dists=e2r_dists,
                structure_terms_enabled=structure_terms_enabled,
                return_components=True,
            )
            return loss, components
        if isinstance(self.loss_fn, (HorizynFGWLoss, BidirectionalAnchorBalancedSupConLoss)):
            loss, components = self.loss_fn(
                dists,
                query_idx,
                target_idx,
                query_embeds=query_embeds,
                target_embeds=target_embeds,
                reaction_similarity=reaction_similarity,
                enzyme_similarity=enzyme_similarity,
                return_components=True,
            )
            return loss, components
        if isinstance(self.loss_fn, DegreeTemperedFullBatchMLNCELoss):
            return self.loss_fn(
                dists,
                query_idx,
                target_idx,
                query_degrees=reaction_degrees,
                return_components=True,
            )
        if isinstance(
            self.loss_fn,
            (
                DecoupledAllPositiveInfoNCELoss,
                HybridCardinalityRetrievalLoss,
                BalancedSigmoidEBMLoss,
            ),
        ):
            if isinstance(self.loss_fn, HybridCardinalityRetrievalLoss):
                self.loss_fn.set_training_progress(int(self.current_epoch))
            return self.loss_fn(
                dists,
                query_idx,
                target_idx,
                return_components=True,
            )
        return self.loss_fn(dists, query_idx, target_idx), {}

    def _positive_pair_indices(
        self,
        unique_query_ids: list[str],
        unique_target_ids: list[str],
        pair_query_ids: list[str],
        pair_target_ids: list[str],
        positive_pair_source: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = positive_pair_source or self.positive_pair_source
        if source not in self._VALID_POSITIVE_PAIR_SOURCES:
            raise ValueError(
                "positive_pair_source must be one of: "
                f"{sorted(self._VALID_POSITIVE_PAIR_SOURCES)}"
            )

        query_id_to_idx = {query_id: idx for idx, query_id in enumerate(unique_query_ids)}
        target_id_to_idx = {target_id: idx for idx, target_id in enumerate(unique_target_ids)}
        positive_query_indices: list[int] = []
        positive_target_indices: list[int] = []

        if source == "observed_pairs":
            for query_id, target_id in zip(pair_query_ids, pair_target_ids):
                positive_query_indices.append(query_id_to_idx[query_id])
                positive_target_indices.append(target_id_to_idx[target_id])
        else:
            datamodule = self.trainer.datamodule
            train_query_to_targets = getattr(datamodule, "_train_query_to_targets", None)
            if train_query_to_targets is None:
                raise RuntimeError(
                    "positive_pair_source='all_known_in_batch' requires the datamodule "
                    "to expose _train_query_to_targets"
                )
            for query_id in unique_query_ids:
                query_idx = query_id_to_idx[query_id]
                for target_id in train_query_to_targets.get(query_id, []):
                    target_idx = target_id_to_idx.get(target_id)
                    if target_idx is None:
                        continue
                    positive_query_indices.append(query_idx)
                    positive_target_indices.append(target_idx)

        if not positive_query_indices:
            raise RuntimeError(
                "No positive pairs were available for the current batch using "
                f"positive_pair_source='{source}'"
            )

        return (
            torch.tensor(positive_query_indices, dtype=torch.long, device=self.device),
            torch.tensor(positive_target_indices, dtype=torch.long, device=self.device),
        )

    def _full_train_query_degrees(
        self,
        query_ids: list[str],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        datamodule = self.trainer.datamodule
        train_query_to_targets = getattr(datamodule, "_train_query_to_targets", None)
        if train_query_to_targets is None:
            raise RuntimeError(
                "DegreeTemperedFullBatchMLNCELoss requires the datamodule to expose "
                "_train_query_to_targets"
            )
        return reference.new_tensor(
            [max(1, len(set(train_query_to_targets.get(query_id, ())))) for query_id in query_ids]
        )

    def _ec_weight_matrix(
        self,
        item_ids: list[str],
        ec_sets: dict[str, tuple[str, ...]] | None,
    ) -> torch.Tensor | None:
        if not ec_sets:
            return None
        num_items = len(item_ids)
        weights = torch.zeros(num_items, num_items, dtype=torch.float32, device=self.device)

        prefix_groups: dict[int, dict[str, list[int]]] = {
            depth: defaultdict(list) for depth in range(self.ec_min_shared_depth, 5)
        }
        for item_idx, item_id in enumerate(item_ids):
            ecs = ec_sets.get(item_id)
            if ecs is None and item_id.endswith(("_f", "_r")):
                ecs = ec_sets.get(item_id[:-2])
            if not ecs:
                continue
            for depth, groups in prefix_groups.items():
                prefixes = set()
                for ec_number in ecs:
                    parts = ec_number.split(".")
                    if len(parts) >= depth:
                        prefixes.add(".".join(parts[:depth]))
                for prefix in prefixes:
                    groups[prefix].append(item_idx)

        depth_to_weight = {2: 0.25, 3: 0.5, 4: 1.0}
        pair_to_weight: dict[tuple[int, int], float] = {}
        for depth, groups in prefix_groups.items():
            weight = depth_to_weight[depth]
            for member_indices in groups.values():
                if len(member_indices) < 2:
                    continue
                members = sorted(set(member_indices))
                for left_pos, row_idx in enumerate(members[:-1]):
                    for col_idx in members[left_pos + 1 :]:
                        key = (row_idx, col_idx)
                        pair_to_weight[key] = max(pair_to_weight.get(key, 0.0), weight)

        if pair_to_weight:
            rows = []
            cols = []
            values = []
            for (row_idx, col_idx), weight in pair_to_weight.items():
                rows.extend((row_idx, col_idx))
                cols.extend((col_idx, row_idx))
                values.extend((weight, weight))
            row_tensor = torch.tensor(rows, dtype=torch.long, device=self.device)
            col_tensor = torch.tensor(cols, dtype=torch.long, device=self.device)
            value_tensor = torch.tensor(values, dtype=weights.dtype, device=self.device)
            weights[row_tensor, col_tensor] = value_tensor
        return weights

    def _needs_reaction_ec_weights(self) -> bool:
        return (
            isinstance(self.loss_fn, MultiAlignmentRetrievalLoss) and self.loss_fn.lambda_rr > 0.0
        )

    def _needs_enzyme_ec_weights(self) -> bool:
        return (
            isinstance(self.loss_fn, MultiAlignmentRetrievalLoss) and self.loss_fn.lambda_ee > 0.0
        )

    def _log_loss_components(
        self,
        prefix: str,
        components: dict[str, torch.Tensor],
        batch_size: int,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        for name, value in components.items():
            self.log(
                f"{prefix}/loss_{name}",
                value,
                on_epoch=on_epoch,
                on_step=on_step,
                batch_size=batch_size,
                sync_dist=True,
            )

    def _global_full_batch_loss(
        self,
        query_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
        query_similarity_features: Optional[torch.Tensor],
        target_similarity_features: Optional[torch.Tensor],
        unique_query_ids: List[str],
        unique_target_ids: List[str],
        pair_query_ids: List[str],
        pair_target_ids: List[str],
        positive_pair_source: str | None = None,
        structure_terms_enabled: bool = True,
        e2r_query_embeds: torch.Tensor | None = None,
        e2r_target_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if (e2r_query_embeds is None) != (e2r_target_embeds is None):
            raise ValueError(
                "e2r_query_embeds and e2r_target_embeds must either both be provided or both be None"
            )
        gathered_query_embeds = self._gather_variable_tensor(query_embeds, sync_grads=True)
        gathered_target_embeds = self._gather_variable_tensor(target_embeds, sync_grads=True)
        gathered_e2r_query_embeds = (
            self._gather_variable_tensor(e2r_query_embeds, sync_grads=True)
            if e2r_query_embeds is not None
            else None
        )
        gathered_e2r_target_embeds = (
            self._gather_variable_tensor(e2r_target_embeds, sync_grads=True)
            if e2r_target_embeds is not None
            else None
        )
        gathered_query_features = (
            self._gather_variable_tensor(query_similarity_features, sync_grads=False)
            if query_similarity_features is not None
            else None
        )
        gathered_target_features = (
            self._gather_variable_tensor(target_similarity_features, sync_grads=False)
            if target_similarity_features is not None
            else None
        )
        gathered_unique_query_ids = self._gather_objects(unique_query_ids)
        gathered_unique_target_ids = self._gather_objects(unique_target_ids)
        gathered_pair_query_ids = self._gather_objects(pair_query_ids)
        gathered_pair_target_ids = self._gather_objects(pair_target_ids)

        all_query_embeds = torch.cat(gathered_query_embeds, dim=0)
        all_target_embeds = torch.cat(gathered_target_embeds, dim=0)
        all_e2r_query_embeds = (
            torch.cat(gathered_e2r_query_embeds, dim=0)
            if gathered_e2r_query_embeds is not None
            else None
        )
        all_e2r_target_embeds = (
            torch.cat(gathered_e2r_target_embeds, dim=0)
            if gathered_e2r_target_embeds is not None
            else None
        )
        all_query_features = (
            torch.cat(gathered_query_features, dim=0)
            if gathered_query_features is not None
            else None
        )
        all_target_features = (
            torch.cat(gathered_target_features, dim=0)
            if gathered_target_features is not None
            else None
        )
        all_unique_query_ids = [
            query_id for rank_ids in gathered_unique_query_ids for query_id in rank_ids
        ]
        all_unique_target_ids = [
            target_id for rank_ids in gathered_unique_target_ids for target_id in rank_ids
        ]

        query_id_to_global_idx: dict[str, int] = {}
        query_keep_indices = []
        for idx, query_id in enumerate(all_unique_query_ids):
            if query_id not in query_id_to_global_idx:
                query_id_to_global_idx[query_id] = len(query_keep_indices)
                query_keep_indices.append(idx)

        target_id_to_global_idx: dict[str, int] = {}
        target_keep_indices = []
        for idx, target_id in enumerate(all_unique_target_ids):
            if target_id not in target_id_to_global_idx:
                target_id_to_global_idx[target_id] = len(target_keep_indices)
                target_keep_indices.append(idx)

        query_keep = torch.tensor(query_keep_indices, dtype=torch.long, device=self.device)
        target_keep = torch.tensor(target_keep_indices, dtype=torch.long, device=self.device)
        global_query_embeds = all_query_embeds[query_keep]
        global_target_embeds = all_target_embeds[target_keep]
        global_e2r_query_embeds = (
            all_e2r_query_embeds[query_keep] if all_e2r_query_embeds is not None else None
        )
        global_e2r_target_embeds = (
            all_e2r_target_embeds[target_keep] if all_e2r_target_embeds is not None else None
        )
        global_query_features = (
            all_query_features[query_keep] if all_query_features is not None else None
        )
        global_target_features = (
            all_target_features[target_keep] if all_target_features is not None else None
        )
        global_unique_query_ids = [all_unique_query_ids[idx] for idx in query_keep_indices]
        global_unique_target_ids = [all_unique_target_ids[idx] for idx in target_keep_indices]

        all_pair_query_ids = [
            query_id for rank_ids in gathered_pair_query_ids for query_id in rank_ids
        ]
        all_pair_target_ids = [
            target_id for rank_ids in gathered_pair_target_ids for target_id in rank_ids
        ]
        global_query_idx, global_target_idx = self._positive_pair_indices(
            global_unique_query_ids,
            global_unique_target_ids,
            all_pair_query_ids,
            all_pair_target_ids,
            positive_pair_source=positive_pair_source,
        )

        reaction_ec_weights = None
        enzyme_ec_weights = None
        if structure_terms_enabled and isinstance(self.loss_fn, MultiAlignmentRetrievalLoss):
            datamodule = self.trainer.datamodule
            if self._needs_reaction_ec_weights():
                reaction_ec_weights = self._ec_weight_matrix(
                    global_unique_query_ids,
                    getattr(datamodule, "_train_reaction_ec_sets", None),
                )
            if self._needs_enzyme_ec_weights():
                enzyme_ec_weights = self._ec_weight_matrix(
                    global_unique_target_ids,
                    getattr(datamodule, "_train_enzyme_ec_sets", None),
                )

        reaction_degrees = None
        if isinstance(self.loss_fn, DegreeTemperedFullBatchMLNCELoss):
            reaction_degrees = self._full_train_query_degrees(
                global_unique_query_ids,
                global_query_embeds,
            )

        dists = self._compute_embedding_distances(global_query_embeds, global_target_embeds)
        e2r_dists = (
            self._compute_embedding_distances(
                global_e2r_query_embeds,
                global_e2r_target_embeds,
            )
            if global_e2r_query_embeds is not None and global_e2r_target_embeds is not None
            else None
        )
        return self._loss_with_components(
            dists,
            global_query_idx,
            global_target_idx,
            global_query_embeds,
            global_target_embeds,
            reaction_similarity=pairwise_cosine_similarity(global_query_features),
            enzyme_similarity=pairwise_cosine_similarity(global_target_features),
            reaction_ec_weights=reaction_ec_weights,
            enzyme_ec_weights=enzyme_ec_weights,
            reaction_degrees=reaction_degrees,
            e2r_dists=e2r_dists,
            structure_terms_enabled=structure_terms_enabled,
        )

    def _reset_balanced_metric_state(self) -> None:
        self._balanced_metric_sums = {
            direction: {
                metric_name: torch.zeros((), dtype=torch.float32, device=self.device)
                for metric_name in self._BALANCED_METRICS
            }
            for direction in self._BALANCED_DIRECTIONS
        }
        self._balanced_metric_counts = {
            direction: {
                metric_name: torch.zeros((), dtype=torch.float32, device=self.device)
                for metric_name in self._BALANCED_METRICS
            }
            for direction in self._BALANCED_DIRECTIONS
        }

    def _accumulate_balanced_metric_batch(
        self,
        direction: str,
        metric_name: str,
        values: list[torch.Tensor],
        device: torch.device,
    ) -> None:
        if direction not in self._BALANCED_DIRECTIONS or metric_name not in self._BALANCED_METRICS:
            return
        if not values:
            return
        if not self._balanced_metric_sums:
            self._reset_balanced_metric_state()
        metric_sum = torch.stack([value.detach().to(device=device) for value in values]).sum()
        metric_count = torch.tensor(float(len(values)), dtype=torch.float32, device=device)
        self._balanced_metric_sums[direction][metric_name] = (
            self._balanced_metric_sums[direction][metric_name] + metric_sum
        )
        self._balanced_metric_counts[direction][metric_name] = (
            self._balanced_metric_counts[direction][metric_name] + metric_count
        )

    @staticmethod
    def _harmonic_mean(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if bool((first <= 0).item()) or bool((second <= 0).item()):
            return first.new_tensor(0.0)
        return (2.0 * first * second) / (first + second)

    def on_validation_epoch_start(self) -> None:
        if not self.validation_retrieval_metrics:
            return
        self._reset_balanced_metric_state()
        datamodule = self.trainer.datamodule
        target_ids = getattr(datamodule, "_val_retrieval_target_candidate_ids", None)
        query_ids = getattr(datamodule, "_val_retrieval_query_candidate_ids", None)
        if target_ids is None or query_ids is None:
            raise RuntimeError(
                "validation_retrieval_metrics=True requires the datamodule to expose "
                "_val_retrieval_target_candidate_ids and _val_retrieval_query_candidate_ids"
            )
        embedding_dim = int(self.hparams.embedding_dim)
        self.val_target_lookup_table = torch.zeros(
            (len(target_ids), embedding_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.val_query_lookup_table = torch.zeros(
            (len(query_ids), embedding_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.val_target_id_to_idx = {target_id: idx for idx, target_id in enumerate(target_ids)}
        self.val_query_id_to_idx = {query_id: idx for idx, query_id in enumerate(query_ids)}

    def on_validation_epoch_end(self) -> None:
        if not self.validation_retrieval_metrics or not self._balanced_metric_sums:
            return
        if set(self.validation_retrieval_directions) != set(self._BALANCED_DIRECTIONS):
            return
        for metric_name in self._BALANCED_METRICS:
            direction_means: dict[str, torch.Tensor] = {}
            for direction in self._BALANCED_DIRECTIONS:
                metric_sum = self._balanced_metric_sums[direction][metric_name].clone()
                metric_count = self._balanced_metric_counts[direction][metric_name].clone()
                if self._distributed_enabled():
                    dist.all_reduce(metric_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(metric_count, op=dist.ReduceOp.SUM)
                direction_means[direction] = torch.where(
                    metric_count > 0,
                    metric_sum / metric_count.clamp_min(1.0),
                    metric_sum.new_tensor(0.0),
                )

            balanced_value = self._harmonic_mean(
                direction_means["reaction_to_enzyme"],
                direction_means["enzyme_to_reaction"],
            )
            arithmetic_value = 0.5 * (
                direction_means["reaction_to_enzyme"] + direction_means["enzyme_to_reaction"]
            )
            metric_suffix = "mrr" if metric_name == "mrr" else metric_name
            self.log(
                f"val/balanced_{metric_suffix}",
                balanced_value,
                on_epoch=True,
                on_step=False,
                prog_bar=(metric_name == "mrr"),
                add_dataloader_idx=False,
                batch_size=1,
                sync_dist=True,
            )
            self.log(
                f"val/mean_bidirectional_{metric_suffix}",
                arithmetic_value,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                add_dataloader_idx=False,
                batch_size=1,
                sync_dist=True,
            )

    def _encode_target_batch(
        self,
        batch: Dict[str, Any],
        retrieval_direction: str = "reaction_to_enzyme",
    ) -> torch.Tensor:
        (
            residue_embeddings,
            residue_padding_mask,
            score_residue_embeddings,
            score_residue_padding_mask,
        ) = self._residue_inputs_from_batch(batch)
        return self.model.encode_targets(
            residue_embeddings,
            residue_padding_mask=residue_padding_mask,
            score_residue_embeddings=score_residue_embeddings,
            score_residue_padding_mask=score_residue_padding_mask,
            capability_vectors=self._capability_vectors_from_batch(batch),
            capability_mask=self._capability_mask_from_batch(batch),
            factorized_capability_vectors=self._factorized_capability_vectors_from_batch(batch),
            factorized_capability_masks=self._factorized_capability_masks_from_batch(batch),
            text_vectors=self._text_vectors_from_batch(batch),
            text_mask=self._text_mask_from_batch(batch),
            retrieval_direction=retrieval_direction,
        )

    def _encode_query_batch(
        self,
        batch: Dict[str, Any],
        retrieval_direction: str = "reaction_to_enzyme",
    ) -> torch.Tensor:
        if "query_vec" in batch:
            query_inputs = batch["query_vec"]
        else:
            query_feature_keys = {
                "reaction_embedding",
                "reactant_embeddings",
                "reactant_padding_mask",
                "product_embeddings",
                "product_padding_mask",
                "has_unimol2",
                "reactant_chirality_embeddings",
                "reactant_chirality_padding_mask",
                "product_chirality_embeddings",
                "product_chirality_padding_mask",
                "has_chirality",
                "has_chiro",
                "has_chienn",
                "reaction_chemistry_vector",
                "has_reaction_chemistry",
                "reaction_directional_vector",
                "has_reaction_directional",
            }
            query_inputs = {key: value for key, value in batch.items() if key in query_feature_keys}
        if getattr(self.model, "e2r_adapter", None) is None:
            return self.model.encode_queries(query_inputs)
        return self.model.encode_queries(
            query_inputs,
            retrieval_direction=retrieval_direction,
        )

    def _update_lookup_table(
        self,
        lookup_table: torch.Tensor,
        embeddings: torch.Tensor,
        row_indices: torch.Tensor,
    ) -> None:
        gathered_embeddings = self._gather_variable_tensor(
            F.normalize(embeddings.detach().float(), p=2, dim=-1, eps=1e-12),
            sync_grads=False,
        )
        gathered_indices = self._gather_variable_tensor(
            row_indices.to(device=self.device, dtype=torch.long),
            sync_grads=False,
        )
        for rank_embeddings, rank_indices in zip(gathered_embeddings, gathered_indices):
            if rank_indices.numel() == 0:
                continue
            lookup_table[rank_indices.long()] = rank_embeddings
        if self._distributed_enabled():
            self.trainer.strategy.barrier()

    def _positive_indices_for_anchor(
        self,
        anchor_id: str,
        positive_id_lookup: dict[str, list[str]],
        candidate_id_to_idx: dict[str, int],
    ) -> torch.Tensor | None:
        positive_ids = positive_id_lookup.get(anchor_id, [])
        indices = [
            candidate_id_to_idx[positive_id]
            for positive_id in positive_ids
            if positive_id in candidate_id_to_idx
        ]
        if not indices:
            return None
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def _reciprocal_rank(self, scores: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        valid_targets = target_idx[target_idx >= 0]
        if valid_targets.numel() == 0:
            return scores.new_tensor(0.0)
        order = torch.argsort(scores, descending=True)
        relevance = torch.isin(order, valid_targets)
        hit_positions = torch.nonzero(relevance, as_tuple=False)
        if hit_positions.numel() == 0:
            return scores.new_tensor(0.0)
        rank = hit_positions[0, 0].to(dtype=scores.dtype) + 1.0
        return 1.0 / rank

    def _mean_rank(self, scores: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        valid_targets = target_idx[target_idx >= 0]
        if valid_targets.numel() == 0:
            return scores.new_tensor(float(scores.numel()))
        order = torch.argsort(scores, descending=True)
        relevance = torch.isin(order, valid_targets)
        hit_positions = torch.nonzero(relevance, as_tuple=False)
        if hit_positions.numel() == 0:
            return scores.new_tensor(float(scores.numel()))
        return hit_positions[0, 0].to(dtype=scores.dtype) + 1.0

    def _batched_retrieval_metric_values(
        self,
        scores: torch.Tensor,
        anchor_ids: list[str],
        positive_id_lookup: dict[str, list[str]],
        candidate_id_to_idx: dict[str, int],
    ) -> tuple[int, dict[str, torch.Tensor]]:
        row_indices: list[int] = []
        positive_rows: list[list[int]] = []
        max_positives = 0
        for row_idx, anchor_id in enumerate(anchor_ids):
            positive_ids = positive_id_lookup.get(anchor_id, [])
            positive_indices = [
                candidate_id_to_idx[positive_id]
                for positive_id in positive_ids
                if positive_id in candidate_id_to_idx
            ]
            if not positive_indices:
                continue
            row_indices.append(row_idx)
            positive_rows.append(positive_indices)
            max_positives = max(max_positives, len(positive_indices))

        num_valid = len(row_indices)
        if num_valid == 0:
            return 0, {}

        device = scores.device
        valid_row_indices = torch.tensor(row_indices, dtype=torch.long, device=device)
        valid_scores = scores.index_select(0, valid_row_indices).detach().float()
        positive_idx = torch.full(
            (num_valid, max_positives),
            -1,
            dtype=torch.long,
            device=device,
        )
        for row_idx, indices in enumerate(positive_rows):
            positive_idx[row_idx, : len(indices)] = torch.tensor(
                indices,
                dtype=torch.long,
                device=device,
            )

        positive_mask = positive_idx >= 0
        safe_positive_idx = positive_idx.clamp_min(0)
        positive_scores = valid_scores.gather(1, safe_positive_idx).masked_fill(
            ~positive_mask,
            float("-inf"),
        )
        positive_counts = positive_mask.sum(dim=1).to(dtype=torch.float32)

        best_positive_scores = positive_scores.max(dim=1).values
        best_positive_ranks = (valid_scores > best_positive_scores.unsqueeze(1)).sum(dim=1).to(
            torch.float32
        ) + 1.0

        metric_values: dict[str, torch.Tensor] = {
            "mrr": best_positive_ranks.reciprocal(),
            "mean_rank": best_positive_ranks,
        }
        for k in self.retrieval_metric_top_k:
            metric_values[f"top_{k}"] = (best_positive_ranks <= float(k)).to(torch.float32)

        positive_ranks = torch.full(
            (num_valid, max_positives),
            float("inf"),
            dtype=torch.float32,
            device=device,
        )
        for col_idx in range(max_positives):
            col_mask = positive_mask[:, col_idx]
            if not bool(col_mask.any().item()):
                continue
            col_scores = valid_scores[col_mask]
            col_positive_scores = positive_scores[col_mask, col_idx]
            positive_ranks[col_mask, col_idx] = (col_scores > col_positive_scores.unsqueeze(1)).sum(
                dim=1
            ).to(torch.float32) + 1.0
        metric_values["reactzyme_mrr"] = positive_ranks.reciprocal().masked_fill(
            ~positive_mask, 0.0
        ).sum(dim=1) / positive_counts.clamp_min(1.0)

        if "r_precision" in self.metric_functionals or "avg_precision" in self.metric_functionals:
            if "r_precision" in self.metric_functionals:
                metric_values["r_precision"] = (
                    (positive_ranks <= positive_counts.unsqueeze(1)) & positive_mask
                ).sum(dim=1).to(torch.float32) / positive_counts.clamp_min(1.0)

            if "avg_precision" in self.metric_functionals:
                sorted_positive_ranks = (
                    positive_ranks.masked_fill(
                        ~positive_mask,
                        float("inf"),
                    )
                    .sort(dim=1)
                    .values
                )
                precision_order = torch.arange(
                    1,
                    max_positives + 1,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                precision_terms = precision_order / sorted_positive_ranks
                precision_terms = precision_terms.masked_fill(
                    ~torch.isfinite(sorted_positive_ranks),
                    0.0,
                )
                metric_values["avg_precision"] = precision_terms.sum(
                    dim=1
                ) / positive_counts.clamp_min(1.0)

        return num_valid, metric_values

    def _log_retrieval_direction_metrics(
        self,
        prefix: str,
        scores: torch.Tensor,
        anchor_ids: list[str],
        positive_id_lookup: dict[str, list[str]],
        candidate_id_to_idx: dict[str, int],
    ) -> None:
        batch_size = len(anchor_ids)
        num_valid, metric_values_by_name = self._batched_retrieval_metric_values(
            scores,
            anchor_ids,
            positive_id_lookup,
            candidate_id_to_idx,
        )

        self.log(
            f"val/{prefix}/num_queries",
            scores.new_tensor(float(num_valid)),
            on_epoch=True,
            on_step=False,
            add_dataloader_idx=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        for metric_name, metric_values in metric_values_by_name.items():
            if metric_values.numel() == 0:
                continue
            values = list(metric_values.detach().unbind(0))
            self._accumulate_balanced_metric_batch(
                prefix,
                metric_name,
                values,
                scores.device,
            )
            metric_value = metric_values.mean().to(device=scores.device)
            self.log(
                f"val/{prefix}/{metric_name}",
                metric_value,
                on_epoch=True,
                on_step=False,
                prog_bar=(metric_name == "top_1"),
                add_dataloader_idx=False,
                batch_size=batch_size,
                sync_dist=True,
            )

    def _validation_target_lookup_step(self, batch: Dict[str, Any]) -> None:
        if self.val_target_lookup_table is None:
            raise RuntimeError("Target lookup table was not initialized")
        target_embeds = self._encode_target_batch(
            batch,
            retrieval_direction="reaction_to_enzyme",
        )
        self._update_lookup_table(
            self.val_target_lookup_table,
            target_embeds,
            batch["target_lookup_row_idx"],
        )

    def _validation_query_lookup_step(self, batch: Dict[str, Any]) -> None:
        if self.val_query_lookup_table is None:
            raise RuntimeError("Query lookup table was not initialized")
        query_embeds = self._encode_query_batch(
            batch,
            retrieval_direction="enzyme_to_reaction",
        )
        self._update_lookup_table(
            self.val_query_lookup_table,
            query_embeds,
            batch["query_lookup_row_idx"],
        )

    def _validation_reaction_to_enzyme_step(self, batch: Dict[str, Any]) -> None:
        if self.val_target_lookup_table is None:
            raise RuntimeError("Target lookup table was not initialized")
        datamodule = self.trainer.datamodule
        query_embeds = self._encode_query_batch(batch)
        scores = self._compute_retrieval_scores(query_embeds, self.val_target_lookup_table)
        self._log_retrieval_direction_metrics(
            "reaction_to_enzyme",
            scores,
            list(batch["query_id"]),
            datamodule._query_to_targets,
            self.val_target_id_to_idx,
        )

    def _validation_enzyme_to_reaction_step(self, batch: Dict[str, Any]) -> None:
        if self.val_query_lookup_table is None:
            raise RuntimeError("Query lookup table was not initialized")
        datamodule = self.trainer.datamodule
        target_embeds = self._encode_target_batch(
            batch,
            retrieval_direction="enzyme_to_reaction",
        )
        scores = self._compute_retrieval_scores(target_embeds, self.val_query_lookup_table)
        self._log_retrieval_direction_metrics(
            "enzyme_to_reaction",
            scores,
            list(batch["target_id"]),
            datamodule._target_to_queries,
            self.val_query_id_to_idx,
        )

    @staticmethod
    def _attention_stats(
        attention_weights: torch.Tensor,
        residue_padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        valid_mask = ~residue_padding_mask.to(torch.bool)
        safe_weights = attention_weights.masked_fill(~valid_mask, 0.0)
        entropy = -(safe_weights.clamp_min(1e-12).log() * safe_weights).sum(dim=1)
        lengths = valid_mask.sum(dim=1).clamp_min(2).to(dtype=attention_weights.dtype)
        normalized_entropy = entropy / torch.log(lengths)
        return {
            "attention/max_weight": safe_weights.max(dim=1).values.mean(),
            "attention/entropy": entropy.mean(),
            "attention/normalized_entropy": normalized_entropy.mean(),
        }

    @staticmethod
    def _query_attention_stats(
        attention: dict[str, torch.Tensor],
        query_inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        stats: dict[str, torch.Tensor] = {}
        for side in ("reactant", "product"):
            weights = attention.get(side)
            padding_mask = query_inputs.get(f"{side}_padding_mask")
            if weights is None or padding_mask is None:
                continue
            valid_mask = ~padding_mask.to(torch.bool)
            safe_weights = weights.masked_fill(~valid_mask, 0.0)
            entropy = -(safe_weights.clamp_min(1e-12).log() * safe_weights).sum(dim=1)
            lengths = valid_mask.sum(dim=1).clamp_min(2).to(dtype=weights.dtype)
            normalized_entropy = entropy / torch.log(lengths)
            stats[f"reaction_attention/{side}/max_weight"] = safe_weights.max(dim=1).values.mean()
            stats[f"reaction_attention/{side}/entropy"] = entropy.mean()
            stats[f"reaction_attention/{side}/normalized_entropy"] = normalized_entropy.mean()
        return stats

    @staticmethod
    def _reaction_fingerprint_attention_stats(
        attention: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        weights = attention.get("reaction_fingerprint")
        entropy = attention.get("reaction_fingerprint_entropy")
        if weights is None:
            return {}
        stats = {
            "reaction_fingerprint_attention/max_weight": weights.max(dim=1).values.mean(),
            "reaction_fingerprint_attention/weight_rdkit_reactants": weights[:, 0].mean(),
            "reaction_fingerprint_attention/weight_rdkit_products": weights[:, 1].mean(),
            "reaction_fingerprint_attention/weight_drfp": weights[:, 2].mean(),
        }
        if entropy is not None:
            stats["reaction_fingerprint_attention/entropy"] = entropy.mean()
            stats["reaction_fingerprint_attention/normalized_entropy"] = (
                entropy / torch.log(entropy.new_tensor(3.0))
            ).mean()
        return stats

    @staticmethod
    def _reaction_multimodal_attention_stats(
        attention: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        weights = attention.get("modality")
        entropy = attention.get("modality_entropy")
        modality_mask = attention.get("modality_mask")
        block_norms = attention.get("modality_block_norms")
        directional_gate = attention.get("directional_gate")
        directional_mask = attention.get("directional_mask")
        residual_gate = attention.get("residual_gate")
        residual_base_cosine = attention.get("residual_base_cosine")
        residual_base_norm_ratio = attention.get("residual_base_norm_ratio")
        if weights is None:
            return {}
        modality_names = attention.get("modality_names")
        if modality_names is None:
            modality_names = (
                "reaction_model",
                "unimol2",
                "chiro",
                "reaction_chemistry",
            )[: weights.shape[1]]
        stats = {
            "reaction_multimodal_attention/max_weight": weights.max(dim=1).values.mean(),
        }
        if modality_mask is not None:
            mask_float = modality_mask.float()
        for index, name in enumerate(modality_names):
            stats[f"reaction_multimodal_attention/weight_{name}"] = weights[:, index].mean()
            if modality_mask is not None:
                stats[f"reaction_multimodal_attention/active_{name}"] = mask_float[:, index].mean()
            if name in {"chiro", "chirality", "chienn"}:
                stats["reaction_multimodal_attention/weight_chirality"] = weights[:, index].mean()
                if modality_mask is not None:
                    stats["reaction_multimodal_attention/active_chirality"] = mask_float[
                        :, index
                    ].mean()
            if block_norms is not None:
                stats[f"reaction_factorized/block_norm_{name}"] = block_norms[:, index].mean()
        if entropy is not None:
            stats["reaction_multimodal_attention/entropy"] = entropy.mean()
            stats["reaction_multimodal_attention/normalized_entropy"] = (
                entropy / torch.log(entropy.new_tensor(float(weights.shape[1])))
            ).mean()
        if directional_gate is not None:
            stats["reaction_directional/gate"] = directional_gate.float().mean()
        if directional_mask is not None:
            stats["reaction_directional/active_fraction"] = directional_mask.float().mean()
        if residual_gate is not None:
            stats["reaction_residual/gate"] = residual_gate.float().mean()
        if residual_base_cosine is not None:
            stats["reaction_residual/base_cosine"] = residual_base_cosine.float().mean()
        if residual_base_norm_ratio is not None:
            stats["reaction_residual/base_norm_ratio"] = (
                residual_base_norm_ratio.float().mean()
            )
        return stats

    @staticmethod
    def _residue_inputs_from_batch(
        batch: Dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if "residue_embeddings" in batch:
            residue_embeddings = batch["residue_embeddings"]
            residue_padding_mask = batch["residue_padding_mask"]
        elif "protein_residue_embeddings" in batch:
            if "protein_residue_mask" not in batch:
                raise KeyError("'protein_residue_mask' is required with protein_residue_embeddings")
            residue_embeddings = batch["protein_residue_embeddings"]
            residue_padding_mask = ~batch["protein_residue_mask"].to(dtype=torch.bool)
        else:
            raise KeyError("Batch must contain residue_embeddings or protein_residue_embeddings")

        score_embeddings = None
        score_padding_mask = None
        if "score_residue_embeddings" in batch:
            score_embeddings = batch["score_residue_embeddings"]
            if "score_residue_padding_mask" in batch:
                score_padding_mask = batch["score_residue_padding_mask"]
            elif "score_residue_mask" in batch:
                score_padding_mask = ~batch["score_residue_mask"].to(dtype=torch.bool)
        elif "protein_score_residue_embeddings" in batch:
            score_embeddings = batch["protein_score_residue_embeddings"]
            if "protein_score_residue_padding_mask" in batch:
                score_padding_mask = batch["protein_score_residue_padding_mask"]
            elif "protein_score_residue_mask" in batch:
                score_padding_mask = ~batch["protein_score_residue_mask"].to(dtype=torch.bool)
            else:
                raise KeyError(
                    "'protein_score_residue_mask' or 'protein_score_residue_padding_mask' "
                    "is required with protein_score_residue_embeddings"
                )
        if score_embeddings is not None and score_padding_mask is None:
            score_padding_mask = residue_padding_mask

        return residue_embeddings, residue_padding_mask, score_embeddings, score_padding_mask

    @staticmethod
    def _residue_labels_from_batch(
        batch: Dict[str, Any],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        labels = batch.get("residue_labels", batch.get("protein_residue_labels", None))
        label_mask = batch.get(
            "residue_label_mask",
            batch.get("protein_residue_label_mask", None),
        )
        if labels is None or label_mask is None:
            return None, None
        return labels, label_mask

    @staticmethod
    def _capability_vectors_from_batch(batch: Dict[str, Any]) -> torch.Tensor | None:
        value = batch.get("capability_vec", batch.get("protein_capability_vec", None))
        return value if torch.is_tensor(value) else None

    @staticmethod
    def _capability_mask_from_batch(batch: Dict[str, Any]) -> torch.Tensor | None:
        value = batch.get("capability_mask", batch.get("protein_capability_mask", None))
        if torch.is_tensor(value):
            return value.to(dtype=torch.bool)
        return None

    @staticmethod
    def _factorized_capability_vectors_from_batch(
        batch: Dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for family in ("cofactor", "center", "transition"):
            key = f"capability_{family}_vec"
            value = batch.get(key, batch.get(f"protein_{key}", None))
            if torch.is_tensor(value):
                out[family] = value
        return out

    @staticmethod
    def _factorized_capability_masks_from_batch(
        batch: Dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for family in ("cofactor", "center", "transition"):
            key = f"capability_{family}_mask"
            value = batch.get(key, batch.get(f"protein_{key}", None))
            if torch.is_tensor(value):
                out[family] = value.to(dtype=torch.bool)
        return out

    @staticmethod
    def _text_vectors_from_batch(batch: Dict[str, Any]) -> torch.Tensor | None:
        value = batch.get("text_vec", batch.get("protein_text_vec", None))
        return value if torch.is_tensor(value) else None

    @staticmethod
    def _text_mask_from_batch(batch: Dict[str, Any]) -> torch.Tensor | None:
        value = batch.get("text_mask", batch.get("protein_text_mask", None))
        if torch.is_tensor(value):
            return value.to(dtype=torch.bool)
        return None

    @staticmethod
    def _biofp_targets_from_batch(batch: Dict[str, Any]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        valid_suffixes = ("_targets", "_mask", "_denominator", "_confidence")
        for raw_key, value in batch.items():
            key = raw_key[len("protein_") :] if raw_key.startswith("protein_biofp_") else raw_key
            if key.startswith("biofp_") and key.endswith(valid_suffixes) and torch.is_tensor(value):
                out[key] = value
        return out

    @staticmethod
    def _sleec_stats(
        pooling_details: dict[str, torch.Tensor],
        residue_padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        valid_mask = ~residue_padding_mask.to(dtype=torch.bool)
        scores = pooling_details["scores"].masked_fill(~valid_mask, 0.0)
        valid_count = valid_mask.sum().clamp_min(1)
        return {
            "sleec/mean_score": scores.sum() / valid_count,
            "sleec/mean_selected_residues": pooling_details["num_selected"].mean(),
            "sleec/pooling_entropy": pooling_details["pooling_entropy"].mean(),
        }

    @staticmethod
    def _sleec_guided_attention_stats(
        pooling_details: dict[str, torch.Tensor],
        residue_padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        valid_mask = ~residue_padding_mask.to(torch.bool)
        weights = pooling_details["weights"].masked_fill(~valid_mask, 0.0)
        scores = pooling_details["scores"].masked_fill(~valid_mask, 0.0)
        valid_count = valid_mask.sum().clamp_min(1)
        stats = {
            "sleec_guided_attention/mean_score": scores.sum() / valid_count,
            "sleec_guided_attention/pooling_entropy": pooling_details["pooling_entropy"].mean(),
            "sleec_guided_attention/attention_mass_sleec_positive": pooling_details[
                "attention_mass_sleec_positive"
            ].mean(),
            "sleec_guided_attention/num_sleec_positive": pooling_details[
                "num_sleec_positive"
            ].mean(),
            "sleec_guided_attention/max_weight": weights.max(dim=1).values.mean(),
        }
        if "sleec_bias_scale" in pooling_details:
            stats["sleec_guided_attention/bias_scale"] = pooling_details["sleec_bias_scale"]
        if "hyperbolic_tangent_norm" in pooling_details:
            stats["hyperbolic/tangent_norm"] = pooling_details["hyperbolic_tangent_norm"].mean()
        if "hyperbolic_ball_norm" in pooling_details:
            stats["hyperbolic/ball_norm"] = pooling_details["hyperbolic_ball_norm"].mean()
        if "enzyme_fusion_gate_raw_mean" in pooling_details:
            stats["enzyme_fusion/gate_raw_mean"] = pooling_details[
                "enzyme_fusion_gate_raw_mean"
            ].mean()
            stats["enzyme_fusion/gate_pooled"] = pooling_details["enzyme_fusion_gate_pooled"].mean()
            stats["enzyme_fusion/gate_hyperbolic"] = pooling_details[
                "enzyme_fusion_gate_hyperbolic"
            ].mean()
            if "enzyme_fusion_gate_capability" in pooling_details:
                stats["enzyme_fusion/gate_capability"] = pooling_details[
                    "enzyme_fusion_gate_capability"
                ].mean()
        if "text_fusion_alpha" in pooling_details:
            stats["text_fusion/alpha"] = pooling_details["text_fusion_alpha"].mean()
            stats["text_fusion/has_text"] = pooling_details["text_fusion_has_text"].mean()
        if "biofp_sequence_gate_raw_mean" in pooling_details:
            stats["biofp/gate_raw_mean"] = pooling_details["biofp_sequence_gate_raw_mean"].mean()
            stats["biofp/gate_pooled"] = pooling_details["biofp_sequence_gate_pooled"].mean()
        for family in ("mechanism", "center", "cofactor", "transition"):
            key = f"biofp_attention_entropy_{family}"
            if key in pooling_details:
                stats[f"biofp/attention_entropy_{family}"] = pooling_details[key].mean()
        for key, value in pooling_details.items():
            if key.startswith("enzyme_block_weight_") and key != "enzyme_block_weight_kl":
                block_name = key.removeprefix("enzyme_block_weight_")
                stats[f"enzyme_block/weight_{block_name}"] = value.mean()
        if "enzyme_block_weight_kl" in pooling_details:
            stats["enzyme_block/weight_kl"] = pooling_details["enzyme_block_weight_kl"].mean()
        return stats

    def _residue_supervision_loss(
        self,
        pooling_details: dict[str, torch.Tensor] | None,
        residue_labels: torch.Tensor | None,
        residue_label_mask: torch.Tensor | None,
        residue_padding_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.lambda_residue <= 0 or pooling_details is None:
            return None
        if residue_labels is None or residue_label_mask is None:
            return None

        logits = pooling_details["logits"]
        if residue_labels.shape != logits.shape or residue_label_mask.shape != logits.shape:
            raise ValueError(
                "Residue labels and label mask must match SLEEC logits shape: "
                f"labels={tuple(residue_labels.shape)}, "
                f"label_mask={tuple(residue_label_mask.shape)}, logits={tuple(logits.shape)}"
            )
        valid_label_mask = residue_label_mask.to(dtype=torch.bool, device=logits.device)
        valid_label_mask = valid_label_mask & ~residue_padding_mask.to(
            dtype=torch.bool,
            device=logits.device,
        )
        if not bool(valid_label_mask.any()):
            return None

        labels = residue_labels.to(device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(
            logits[valid_label_mask],
            labels[valid_label_mask],
        )

    def _target_anchor_loss(
        self,
        *,
        target_embeds: torch.Tensor,
        residue_embeddings: torch.Tensor,
        residue_padding_mask: torch.Tensor,
        score_residue_embeddings: torch.Tensor | None,
        score_residue_padding_mask: torch.Tensor | None,
        capability_vectors: torch.Tensor | None,
        capability_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.anchor_weight <= 0:
            return None
        if self.model.enzyme_input_mode not in {
            "raw_mean_sleec_hyperbolic_capability_gated",
            "raw_mean_sleec_hyperbolic_capability_blockwise",
        }:
            return None
        if capability_vectors is None:
            return None
        if capability_mask is None:
            capability_mask = torch.ones(
                capability_vectors.shape[0],
                dtype=torch.bool,
                device=capability_vectors.device,
            )
        no_capability_mask = torch.zeros_like(capability_mask, dtype=torch.bool)
        with torch.no_grad():
            anchor = self.model.encode_targets(
                residue_embeddings,
                residue_padding_mask=residue_padding_mask,
                score_residue_embeddings=score_residue_embeddings,
                score_residue_padding_mask=score_residue_padding_mask,
                capability_vectors=capability_vectors,
                capability_mask=no_capability_mask,
            )
        valid_mask = capability_mask.to(device=target_embeds.device, dtype=torch.bool)
        if not bool(valid_mask.any()):
            return None
        return F.mse_loss(target_embeds[valid_mask], anchor.detach()[valid_mask])

    def _capability_consistency_loss(
        self,
        *,
        target_embeds: torch.Tensor,
        capability_vectors: torch.Tensor | None,
        capability_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.capability_consistency_weight <= 0:
            return None
        if self.capability_consistency_projection is None or capability_vectors is None:
            return None
        if capability_mask is None:
            capability_mask = torch.ones(
                capability_vectors.shape[0],
                dtype=torch.bool,
                device=capability_vectors.device,
            )
        valid_mask = capability_mask.to(device=target_embeds.device, dtype=torch.bool)
        if not bool(valid_mask.any()):
            zero_reference = self.capability_consistency_projection(
                capability_vectors[:1].to(
                    device=target_embeds.device,
                    dtype=target_embeds.dtype,
                )
            )
            return zero_reference.sum() * 0.0
        capability_reference = self.capability_consistency_projection(
            capability_vectors.to(device=target_embeds.device, dtype=target_embeds.dtype)
        )
        capability_reference = F.normalize(capability_reference, p=2, dim=-1, eps=1e-12)
        target_normalized = F.normalize(target_embeds, p=2, dim=-1, eps=1e-12)
        cosine = (target_normalized[valid_mask] * capability_reference[valid_mask]).sum(dim=-1)
        return (1.0 - cosine).mean()

    def _biofp_auxiliary_loss(
        self,
        *,
        pooling_details: dict[str, torch.Tensor] | None,
        biofp_targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        if self.biofp_aux_weight <= 0:
            return None, {}
        if pooling_details is None:
            return None, {}
        total = None
        components: dict[str, torch.Tensor] = {}
        for family, family_weight in self.biofp_family_weights.items():
            if family_weight <= 0:
                continue
            logits = pooling_details.get(f"biofp_logits_{family}")
            if logits is None:
                continue

            # Keep the component schema identical on every DDP rank. Some
            # batches contain no annotated proteins on one rank while another
            # rank has valid labels; rank-dependent self.log calls would then
            # enter different sync_dist collectives and deadlock validation.
            loss = logits.sum() * 0.0
            active = loss.detach()
            active_labels = loss.detach()
            targets = biofp_targets.get(f"biofp_{family}_targets")
            mask = biofp_targets.get(f"biofp_{family}_mask")
            denominator = biofp_targets.get(f"biofp_{family}_denominator")
            explicit_confidence = biofp_targets.get(f"biofp_{family}_confidence")
            if targets is not None and mask is not None and denominator is not None:
                targets = targets.to(device=logits.device, dtype=logits.dtype)
                mask = mask.to(device=logits.device, dtype=torch.bool)
                denominator = denominator.to(device=logits.device, dtype=logits.dtype)
                if targets.shape != logits.shape:
                    raise ValueError(
                        f"BioFP {family} targets must match logits shape: "
                        f"targets={tuple(targets.shape)}, logits={tuple(logits.shape)}"
                    )
                if mask.ndim == 1:
                    mask = mask.unsqueeze(1).expand_as(logits)
                if denominator.ndim == 1:
                    denominator = denominator.unsqueeze(1).expand_as(logits)
                if mask.shape != logits.shape:
                    raise ValueError(
                        f"BioFP {family} mask must have shape ({logits.shape[0]},) or "
                        f"{tuple(logits.shape)}, "
                        f"got {tuple(mask.shape)}"
                    )
                if denominator.shape != logits.shape:
                    raise ValueError(f"BioFP {family} denominator must be row-wise or match logits")
                if explicit_confidence is None:
                    confidence = (
                        denominator.clamp(min=1.0, max=self.biofp_confidence_cap)
                        / self.biofp_confidence_cap
                    )
                else:
                    confidence = explicit_confidence.to(
                        device=logits.device,
                        dtype=logits.dtype,
                    )
                    if confidence.ndim == 1:
                        confidence = confidence.unsqueeze(1).expand_as(logits)
                    if confidence.shape != logits.shape:
                        raise ValueError(
                            f"BioFP {family} confidence must be row-wise or match logits"
                        )
                    confidence = confidence.clamp_min(0.0)
                valid = mask & denominator.gt(0) & confidence.gt(0)
                if bool(valid.any()):
                    per_label = F.binary_cross_entropy_with_logits(
                        logits,
                        targets,
                        reduction="none",
                    )
                    cell_weights = confidence * valid.to(dtype=confidence.dtype)
                    loss = (per_label * cell_weights).sum() / cell_weights.sum().clamp_min(1e-12)
                    active = valid.any(dim=1).sum().to(dtype=logits.dtype).detach()
                    active_labels = valid.sum().to(dtype=logits.dtype).detach()

            weighted = loss * float(family_weight)
            total = weighted if total is None else total + weighted
            components[f"biofp_{family}"] = loss
            components[f"weighted_biofp_{family}"] = weighted
            components[f"biofp_{family}_active"] = active
            components[f"biofp_{family}_active_labels"] = active_labels
        if total is None:
            return None, components
        return total, components

    def _cross_tower_factor_alignment_loss(
        self,
        *,
        query_embeds: torch.Tensor,
        pooling_details: dict[str, torch.Tensor] | None,
        biofp_targets: dict[str, torch.Tensor],
        query_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        """Align annotation-gated reaction slices with enzyme factor blocks.

        The reaction tower already retrieves against the enzyme tower's fixed
        block layout. Consequently, its coordinates in each family range are
        the corresponding implicit reaction factor. This loss makes that
        correspondence explicit only for observed pairs with positive family
        annotations; labels remain supervision and are never encoder inputs.
        """
        if self.cross_tower_alignment_weight <= 0:
            return None, {}
        if pooling_details is None:
            raise RuntimeError("Cross-tower factor alignment requires enzyme block details")
        encoder = self.model.biological_factorized_encoder
        if encoder is None:
            raise RuntimeError("Cross-tower factor alignment requires the factorized encoder")
        if query_embeds.ndim != 2 or query_embeds.shape[1] != encoder.output_dim:
            raise ValueError(
                "Cross-tower factor alignment requires reaction embeddings to match the "
                f"factorized layout ({encoder.output_dim} dimensions)"
            )
        if query_indices.shape != target_indices.shape:
            raise ValueError("Cross-tower alignment pair indices must have the same shape")

        block_slices: dict[str, slice] = {}
        offset = 0
        for name in encoder.block_names:
            block_dim = encoder.block_dims[name]
            block_slices[name] = slice(offset, offset + block_dim)
            offset += block_dim

        total = None
        components: dict[str, torch.Tensor] = {}
        for family, family_weight in self.cross_tower_alignment_family_weights.items():
            enzyme_block = pooling_details.get(f"enzyme_block_{family}")
            if enzyme_block is None:
                raise RuntimeError(f"Cross-tower factor alignment requires enzyme_block_{family}")
            reaction_block = F.normalize(
                query_embeds[:, block_slices[family]],
                p=2,
                dim=-1,
                eps=1e-12,
            )
            if enzyme_block.shape != (
                len(enzyme_block),
                encoder.block_dims[family],
            ):
                raise ValueError(f"enzyme_block_{family} has an invalid shape")

            # Initialize every component from both towers so ranks without
            # annotated pairs retain an identical logging and gradient schema.
            loss = reaction_block.sum() * 0.0 + enzyme_block.sum() * 0.0
            active_pairs = loss.detach()
            mean_gate = loss.detach()
            targets = biofp_targets.get(f"biofp_{family}_targets")
            mask = biofp_targets.get(f"biofp_{family}_mask")
            denominator = biofp_targets.get(f"biofp_{family}_denominator")
            explicit_confidence = biofp_targets.get(f"biofp_{family}_confidence")
            if targets is not None and mask is not None and denominator is not None:
                targets = targets.to(device=enzyme_block.device, dtype=enzyme_block.dtype)
                mask = mask.to(device=enzyme_block.device, dtype=torch.bool)
                denominator = denominator.to(
                    device=enzyme_block.device,
                    dtype=enzyme_block.dtype,
                )
                expected_rows = enzyme_block.shape[0]
                if targets.ndim != 2 or targets.shape[0] != expected_rows:
                    raise ValueError(
                        f"BioFP {family} targets must have {expected_rows} rows"
                    )
                if mask.ndim == 1:
                    mask = mask.unsqueeze(1).expand_as(targets)
                if denominator.ndim == 1:
                    denominator = denominator.unsqueeze(1).expand_as(targets)
                if mask.shape != targets.shape:
                    raise ValueError(
                        f"BioFP {family} mask must be row-wise or match targets"
                    )
                if denominator.shape != targets.shape:
                    raise ValueError(
                        f"BioFP {family} denominator must be row-wise or match targets"
                    )
                if explicit_confidence is None:
                    confidence = (
                        denominator.clamp(min=1.0, max=self.biofp_confidence_cap)
                        / self.biofp_confidence_cap
                    )
                else:
                    confidence = explicit_confidence.to(
                        device=enzyme_block.device,
                        dtype=enzyme_block.dtype,
                    )
                    if confidence.ndim == 1:
                        confidence = confidence.unsqueeze(1).expand_as(targets)
                    if confidence.shape != targets.shape:
                        raise ValueError(
                            f"BioFP {family} confidence must be row-wise or match targets"
                        )
                    confidence = confidence.clamp_min(0.0)

                positive = targets.gt(0) & mask & denominator.gt(0) & confidence.gt(0)
                positive_strength = targets.clamp_min(0.0) * positive.to(targets.dtype)
                positive_mass = positive_strength.sum(dim=1)
                row_gate = (confidence * positive_strength).sum(dim=1) / positive_mass.clamp_min(
                    1e-12
                )
                row_valid = positive_mass.gt(0)
                pair_valid = row_valid[target_indices]
                pair_gates = row_gate[target_indices] * pair_valid.to(row_gate.dtype)
                if bool(pair_valid.any()):
                    pair_cosine = F.cosine_similarity(
                        reaction_block[query_indices],
                        enzyme_block[target_indices],
                        dim=-1,
                        eps=1e-12,
                    )
                    loss = ((1.0 - pair_cosine) * pair_gates).sum() / pair_gates.sum().clamp_min(
                        1e-12
                    )
                    active_pairs = pair_valid.sum().to(dtype=enzyme_block.dtype).detach()
                    mean_gate = pair_gates[pair_valid].mean().detach()

            weighted = loss * float(family_weight)
            total = weighted if total is None else total + weighted
            components[f"cross_tower_alignment_{family}"] = loss
            components[f"weighted_cross_tower_alignment_{family}"] = weighted
            components[f"cross_tower_alignment_{family}_active_pairs"] = active_pairs
            components[f"cross_tower_alignment_{family}_mean_gate"] = mean_gate

        if total is None:
            return None, components
        return total, components

    def _reaction_chemistry_consistency_loss(
        self,
        *,
        query_embeds: torch.Tensor,
        query_inputs: dict[str, torch.Tensor] | torch.Tensor,
        query_attention_details: dict[str, Any] | None,
        retrieval_direction: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match full reaction embeddings to a detached chemistry-free view."""
        zero = query_embeds.sum() * 0.0
        zero_count = zero.detach()
        if self.reaction_chemistry_consistency_weight <= 0:
            return zero, zero_count
        if not isinstance(query_inputs, dict):
            raise RuntimeError("Reaction chemistry consistency requires multimodal query inputs")
        if not isinstance(query_attention_details, dict):
            raise RuntimeError("Reaction chemistry consistency requires modality attention details")

        modality_names = query_attention_details.get("modality_names")
        modality_mask = query_attention_details.get("modality_mask")
        if modality_names is None or modality_mask is None:
            raise RuntimeError("Reaction chemistry consistency requires modality names and mask")
        try:
            chemistry_index = tuple(modality_names).index("reaction_chemistry")
        except ValueError as error:
            raise RuntimeError(
                "Reaction chemistry consistency requires an enabled chemistry modality"
            ) from error

        active = modality_mask[:, chemistry_index].to(
            device=query_embeds.device,
            dtype=torch.bool,
        )
        active_count = active.sum().to(dtype=query_embeds.dtype).detach()
        if not bool(active.any()):
            return zero, active_count

        chemistry_free_inputs = dict(query_inputs)
        chemistry_free_inputs["has_reaction_chemistry"] = torch.zeros(
            query_embeds.shape[0],
            dtype=torch.bool,
            device=query_embeds.device,
        )
        with torch.no_grad():
            chemistry_free_embeds = self.model.encode_queries(
                chemistry_free_inputs,
                retrieval_direction=retrieval_direction,
            )
            if isinstance(chemistry_free_embeds, tuple):
                chemistry_free_embeds = chemistry_free_embeds[0]

        loss = (
            1.0
            - F.cosine_similarity(
                query_embeds[active],
                chemistry_free_embeds[active].detach(),
                dim=-1,
            )
        ).mean()
        return loss, active_count

    def _reaction_residual_identity_loss(
        self,
        *,
        query_embeds: torch.Tensor,
        query_attention_details: dict[str, Any] | None,
    ) -> torch.Tensor:
        """Penalize angular drift from the factorized base embedding."""
        zero = query_embeds.sum() * 0.0
        if self.reaction_residual_identity_weight <= 0:
            return zero
        if not isinstance(query_attention_details, dict):
            raise RuntimeError("Reaction residual identity requires projection details")
        base_cosine = query_attention_details.get("residual_base_cosine")
        if base_cosine is None:
            raise RuntimeError(
                "Reaction residual identity requires output_projection=residual_mlp"
            )
        return (1.0 - base_cosine).mean()

    def _compute_full_batch_loss(
        self,
        batch: Dict[str, Any],
        return_attention_stats: bool = False,
        positive_pair_source: str | None = None,
        structure_terms_enabled: bool = True,
    ) -> tuple[torch.Tensor, int, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        query_vecs = batch["query_vec"]
        (
            residue_embeddings,
            residue_padding_mask,
            score_residue_embeddings,
            score_residue_padding_mask,
        ) = self._residue_inputs_from_batch(batch)
        residue_labels, residue_label_mask = self._residue_labels_from_batch(batch)
        capability_vectors = self._capability_vectors_from_batch(batch)
        capability_mask = self._capability_mask_from_batch(batch)
        factorized_capability_vectors = self._factorized_capability_vectors_from_batch(batch)
        factorized_capability_masks = self._factorized_capability_masks_from_batch(batch)
        text_vectors = self._text_vectors_from_batch(batch)
        text_mask = self._text_mask_from_batch(batch)
        biofp_targets = self._biofp_targets_from_batch(batch)
        query_ids = list(batch["query_id"])
        target_ids = list(batch["target_id"])

        unique_query_vecs, _, unique_query_ids = self._deduplicate_query_vectors(
            query_vecs,
            query_ids,
        )
        (
            unique_residues,
            unique_masks,
            unique_score_residues,
            unique_score_masks,
            unique_labels,
            unique_label_mask,
            unique_capability_vectors,
            unique_capability_mask,
            unique_text_vectors,
            unique_text_mask,
            _,
            unique_target_ids,
        ) = self._deduplicate_residue_tensors(
            residue_embeddings,
            residue_padding_mask,
            target_ids,
            score_residue_embeddings=score_residue_embeddings,
            score_residue_padding_mask=score_residue_padding_mask,
            residue_labels=residue_labels,
            residue_label_mask=residue_label_mask,
            capability_vectors=capability_vectors,
            capability_mask=capability_mask,
            text_vectors=text_vectors,
            text_mask=text_mask,
        )
        unique_biofp_targets = self._deduplicate_target_tensor_dict(
            target_ids,
            biofp_targets,
            device=unique_residues.device,
        )
        unique_factorized_capability_vectors = self._deduplicate_target_tensor_dict(
            target_ids,
            factorized_capability_vectors,
            device=unique_residues.device,
        )
        unique_factorized_capability_masks = self._deduplicate_target_tensor_dict(
            target_ids,
            factorized_capability_masks,
            device=unique_residues.device,
        )
        query_similarity_features = reaction_feature_matrix(unique_query_vecs)
        target_similarity_features = masked_mean_features(unique_residues, unique_masks)

        attention_stats = {}
        query_attention_details = None
        pooling_details = None
        needs_r2e_adapter_details = (
            self.training_stage in {"r2e_adapter", "bidirectional_adapters"}
            and self.r2e_identity_weight > 0
        )
        needs_biofp_details = self.biofp_aux_weight > 0 and self.model.enzyme_input_mode in {
            "raw_mean_sleec_biofp_split",
            "raw_mean_sleec_biological_factorized",
        }
        needs_cross_tower_details = self.cross_tower_alignment_weight > 0
        needs_sleec_details = self.model.pooling_name == "sleec" and (
            return_attention_stats or self.lambda_residue > 0 or needs_r2e_adapter_details
        )
        needs_guided_details = self.model.pooling_name == "sleec_guided_attention" and (
            return_attention_stats
            or needs_biofp_details
            or needs_cross_tower_details
            or self.enzyme_block_weight_kl_weight > 0
            or needs_r2e_adapter_details
        )
        needs_reaction_attention_details = (
            self.reaction_attention_entropy_weight > 0
            or self.reaction_chemistry_consistency_weight > 0
            or self.reaction_residual_identity_weight > 0
        )
        query_retrieval_direction = (
            "enzyme_to_reaction" if self.training_stage == "e2r_adapter" else "reaction_to_enzyme"
        )
        target_retrieval_direction = (
            "enzyme_to_reaction"
            if self.training_stage in {"e2r_adapter", "bidirectional_adapters"}
            else "reaction_to_enzyme"
        )
        if return_attention_stats or needs_reaction_attention_details:
            query_result = self.model.encode_queries(
                unique_query_vecs,
                return_attention=True,
                retrieval_direction=query_retrieval_direction,
            )
            if isinstance(query_result, tuple):
                query_embeds, query_attention = query_result
                query_attention_details = query_attention
                if return_attention_stats:
                    attention_stats.update(
                        self._reaction_fingerprint_attention_stats(query_attention)
                    )
                    attention_stats.update(
                        self._reaction_multimodal_attention_stats(query_attention)
                    )
                    if isinstance(unique_query_vecs, dict):
                        attention_stats.update(
                            self._query_attention_stats(query_attention, unique_query_vecs)
                        )
            else:
                query_embeds = query_result

            if self.model.pooling_name == "sleec":
                target_embeds, pooling_details = self.model.encode_targets(
                    unique_residues,
                    residue_padding_mask=unique_masks,
                    score_residue_embeddings=unique_score_residues,
                    score_residue_padding_mask=unique_score_masks,
                    capability_vectors=unique_capability_vectors,
                    capability_mask=unique_capability_mask,
                    factorized_capability_vectors=unique_factorized_capability_vectors,
                    factorized_capability_masks=unique_factorized_capability_masks,
                    text_vectors=unique_text_vectors,
                    text_mask=unique_text_mask,
                    return_pooling_details=True,
                    retrieval_direction=target_retrieval_direction,
                )
                attention_stats.update(self._sleec_stats(pooling_details, unique_masks))
            elif self.model.pooling_name == "sleec_guided_attention":
                target_embeds, pooling_details = self.model.encode_targets(
                    unique_residues,
                    residue_padding_mask=unique_masks,
                    score_residue_embeddings=unique_score_residues,
                    score_residue_padding_mask=unique_score_masks,
                    capability_vectors=unique_capability_vectors,
                    capability_mask=unique_capability_mask,
                    factorized_capability_vectors=unique_factorized_capability_vectors,
                    factorized_capability_masks=unique_factorized_capability_masks,
                    text_vectors=unique_text_vectors,
                    text_mask=unique_text_mask,
                    return_pooling_details=True,
                    retrieval_direction=target_retrieval_direction,
                )
                attention_stats.update(
                    self._sleec_guided_attention_stats(pooling_details, unique_masks)
                )
            elif self.model.pooling_name == "attention":
                target_embeds, attention = self.model.encode_targets(
                    unique_residues,
                    residue_padding_mask=unique_masks,
                    capability_vectors=unique_capability_vectors,
                    capability_mask=unique_capability_mask,
                    factorized_capability_vectors=unique_factorized_capability_vectors,
                    factorized_capability_masks=unique_factorized_capability_masks,
                    text_vectors=unique_text_vectors,
                    text_mask=unique_text_mask,
                    return_attention=True,
                    retrieval_direction=target_retrieval_direction,
                )
                attention_stats.update(self._attention_stats(attention, unique_masks))
            else:
                target_embeds = self.model.encode_targets(
                    unique_residues,
                    residue_padding_mask=unique_masks,
                    capability_vectors=unique_capability_vectors,
                    capability_mask=unique_capability_mask,
                    factorized_capability_vectors=unique_factorized_capability_vectors,
                    factorized_capability_masks=unique_factorized_capability_masks,
                    text_vectors=unique_text_vectors,
                    text_mask=unique_text_mask,
                    retrieval_direction=target_retrieval_direction,
                )
        elif needs_sleec_details:
            query_embeds = self.model.encode_queries(
                unique_query_vecs,
                retrieval_direction=query_retrieval_direction,
            )
            target_embeds, pooling_details = self.model.encode_targets(
                unique_residues,
                residue_padding_mask=unique_masks,
                score_residue_embeddings=unique_score_residues,
                score_residue_padding_mask=unique_score_masks,
                capability_vectors=unique_capability_vectors,
                capability_mask=unique_capability_mask,
                factorized_capability_vectors=unique_factorized_capability_vectors,
                factorized_capability_masks=unique_factorized_capability_masks,
                text_vectors=unique_text_vectors,
                text_mask=unique_text_mask,
                return_pooling_details=True,
                retrieval_direction=target_retrieval_direction,
            )
        elif needs_guided_details:
            query_embeds = self.model.encode_queries(
                unique_query_vecs,
                retrieval_direction=query_retrieval_direction,
            )
            target_embeds, pooling_details = self.model.encode_targets(
                unique_residues,
                residue_padding_mask=unique_masks,
                score_residue_embeddings=unique_score_residues,
                score_residue_padding_mask=unique_score_masks,
                capability_vectors=unique_capability_vectors,
                capability_mask=unique_capability_mask,
                factorized_capability_vectors=unique_factorized_capability_vectors,
                factorized_capability_masks=unique_factorized_capability_masks,
                text_vectors=unique_text_vectors,
                text_mask=unique_text_mask,
                return_pooling_details=True,
                retrieval_direction=target_retrieval_direction,
            )
        elif needs_biofp_details:
            query_embeds = self.model.encode_queries(
                unique_query_vecs,
                retrieval_direction=query_retrieval_direction,
            )
            target_embeds, pooling_details = self.model.encode_targets(
                unique_residues,
                residue_padding_mask=unique_masks,
                score_residue_embeddings=unique_score_residues,
                score_residue_padding_mask=unique_score_masks,
                capability_vectors=unique_capability_vectors,
                capability_mask=unique_capability_mask,
                factorized_capability_vectors=unique_factorized_capability_vectors,
                factorized_capability_masks=unique_factorized_capability_masks,
                text_vectors=unique_text_vectors,
                text_mask=unique_text_mask,
                return_pooling_details=True,
                retrieval_direction=target_retrieval_direction,
            )
        elif needs_r2e_adapter_details:
            query_embeds = self.model.encode_queries(
                unique_query_vecs,
                retrieval_direction=query_retrieval_direction,
            )
            target_embeds, pooling_details = self.model.encode_targets(
                unique_residues,
                residue_padding_mask=unique_masks,
                score_residue_embeddings=unique_score_residues,
                score_residue_padding_mask=unique_score_masks,
                capability_vectors=unique_capability_vectors,
                capability_mask=unique_capability_mask,
                factorized_capability_vectors=unique_factorized_capability_vectors,
                factorized_capability_masks=unique_factorized_capability_masks,
                text_vectors=unique_text_vectors,
                text_mask=unique_text_mask,
                return_pooling_details=True,
                retrieval_direction=target_retrieval_direction,
            )
        else:
            query_embeds = self.model.encode_queries(
                unique_query_vecs,
                retrieval_direction=query_retrieval_direction,
            )
            target_embeds = self.model.encode_targets(
                unique_residues,
                residue_padding_mask=unique_masks,
                score_residue_embeddings=unique_score_residues,
                score_residue_padding_mask=unique_score_masks,
                capability_vectors=unique_capability_vectors,
                capability_mask=unique_capability_mask,
                factorized_capability_vectors=unique_factorized_capability_vectors,
                factorized_capability_masks=unique_factorized_capability_masks,
                text_vectors=unique_text_vectors,
                text_mask=unique_text_mask,
                retrieval_direction=target_retrieval_direction,
            )

        e2r_query_embeds = None
        e2r_target_embeds = None
        if self.training_stage == "bidirectional_adapters":
            if self.model.e2r_adapter is None or self.model.r2e_adapter is None:
                raise RuntimeError("Bidirectional adapter training requires both adapters")
            base_query_embeds = query_embeds
            base_target_embeds = target_embeds
            e2r_query_embeds, e2r_details = self.model.e2r_adapter(
                base_query_embeds,
                unique_query_vecs,
                return_details=True,
            )
            e2r_target_embeds = base_target_embeds
            target_embeds, r2e_details = self.model.r2e_adapter(
                base_target_embeds,
                return_details=True,
            )
            pooling_details = {} if pooling_details is None else dict(pooling_details)
            pooling_details.update(e2r_details)
            pooling_details.update(r2e_details)

        reaction_chemistry_consistency, chemistry_consistency_active = (
            self._reaction_chemistry_consistency_loss(
                query_embeds=query_embeds,
                query_inputs=unique_query_vecs,
                query_attention_details=query_attention_details,
                retrieval_direction=query_retrieval_direction,
            )
        )

        if self.detach_reaction_embeddings:
            query_embeds = query_embeds.detach()

        if self._distributed_enabled():
            loss = self._global_full_batch_loss(
                query_embeds=query_embeds,
                target_embeds=target_embeds,
                query_similarity_features=query_similarity_features,
                target_similarity_features=target_similarity_features,
                unique_query_ids=unique_query_ids,
                unique_target_ids=unique_target_ids,
                pair_query_ids=query_ids,
                pair_target_ids=target_ids,
                positive_pair_source=positive_pair_source,
                structure_terms_enabled=structure_terms_enabled,
                e2r_query_embeds=e2r_query_embeds,
                e2r_target_embeds=e2r_target_embeds,
            )
        else:
            query_idx, target_idx = self._positive_pair_indices(
                unique_query_ids,
                unique_target_ids,
                query_ids,
                target_ids,
                positive_pair_source=positive_pair_source,
            )
            reaction_ec_weights = None
            enzyme_ec_weights = None
            if structure_terms_enabled and isinstance(self.loss_fn, MultiAlignmentRetrievalLoss):
                datamodule = self.trainer.datamodule
                if self._needs_reaction_ec_weights():
                    reaction_ec_weights = self._ec_weight_matrix(
                        unique_query_ids,
                        getattr(datamodule, "_train_reaction_ec_sets", None),
                    )
                if self._needs_enzyme_ec_weights():
                    enzyme_ec_weights = self._ec_weight_matrix(
                        unique_target_ids,
                        getattr(datamodule, "_train_enzyme_ec_sets", None),
                    )
            reaction_degrees = None
            if isinstance(self.loss_fn, DegreeTemperedFullBatchMLNCELoss):
                reaction_degrees = self._full_train_query_degrees(
                    unique_query_ids,
                    query_embeds,
                )
            dists = self._compute_embedding_distances(query_embeds, target_embeds)
            e2r_dists = (
                self._compute_embedding_distances(e2r_query_embeds, e2r_target_embeds)
                if e2r_query_embeds is not None and e2r_target_embeds is not None
                else None
            )
            loss = self._loss_with_components(
                dists,
                query_idx,
                target_idx,
                query_embeds,
                target_embeds,
                reaction_similarity=pairwise_cosine_similarity(query_similarity_features),
                enzyme_similarity=pairwise_cosine_similarity(target_similarity_features),
                reaction_ec_weights=reaction_ec_weights,
                enzyme_ec_weights=enzyme_ec_weights,
                reaction_degrees=reaction_degrees,
                e2r_dists=e2r_dists,
                structure_terms_enabled=structure_terms_enabled,
            )

        loss_value, loss_components = loss
        loss_components = dict(loss_components)
        if self.reaction_chemistry_consistency_weight > 0:
            weighted_chemistry_consistency = (
                reaction_chemistry_consistency * self.reaction_chemistry_consistency_weight
            )
            loss_components["reaction_chemistry_consistency"] = reaction_chemistry_consistency
            loss_components["weighted_reaction_chemistry_consistency"] = (
                weighted_chemistry_consistency
            )
            loss_components["reaction_chemistry_consistency_active"] = chemistry_consistency_active
            loss_value = loss_value + weighted_chemistry_consistency
        if self.reaction_residual_identity_weight > 0:
            residual_identity = self._reaction_residual_identity_loss(
                query_embeds=query_embeds,
                query_attention_details=query_attention_details,
            )
            weighted_residual_identity = (
                residual_identity * self.reaction_residual_identity_weight
            )
            loss_components["reaction_residual_identity"] = residual_identity
            loss_components["weighted_reaction_residual_identity"] = (
                weighted_residual_identity
            )
            loss_value = loss_value + weighted_residual_identity
        if (
            self.training_stage in {"e2r_adapter", "bidirectional_adapters"}
            and self.e2r_identity_weight > 0
        ):
            if self.training_stage == "bidirectional_adapters":
                if e2r_query_embeds is None:
                    raise RuntimeError("Missing E2R embeddings for identity regularization")
                adapted_query_embeds = e2r_query_embeds
                base_query_embeds = query_embeds
            else:
                adapted_query_embeds = query_embeds
                with torch.no_grad():
                    base_query_embeds = self.model.encode_queries(
                        unique_query_vecs,
                        retrieval_direction="reaction_to_enzyme",
                    )
            identity_loss = (
                1.0
                - F.cosine_similarity(
                    adapted_query_embeds,
                    base_query_embeds,
                    dim=-1,
                )
            ).mean()
            weighted_identity = identity_loss * self.e2r_identity_weight
            loss_components["e2r_identity"] = identity_loss
            loss_components["weighted_e2r_identity"] = weighted_identity
            loss_value = loss_value + weighted_identity
        if (
            self.training_stage in {"r2e_adapter", "bidirectional_adapters"}
            and self.r2e_identity_weight > 0
        ):
            if pooling_details is None or "r2e_adapter_base_cosine" not in pooling_details:
                raise RuntimeError("R2E identity regularization requires adapter pooling details")
            identity_loss = (1.0 - pooling_details["r2e_adapter_base_cosine"]).mean()
            weighted_identity = identity_loss * self.r2e_identity_weight
            loss_components["r2e_identity"] = identity_loss
            loss_components["weighted_r2e_identity"] = weighted_identity
            loss_value = loss_value + weighted_identity
        if (
            "mlnce" not in loss_components
            and not isinstance(self.loss_fn, BidirectionalAnchorBalancedSupConLoss)
            and not isinstance(self.loss_fn, MultiAlignmentRetrievalLoss)
        ):
            loss_components["mlnce"] = loss_value
        residue_loss = self._residue_supervision_loss(
            pooling_details=pooling_details,
            residue_labels=unique_labels,
            residue_label_mask=unique_label_mask,
            residue_padding_mask=unique_masks,
        )
        if residue_loss is not None:
            loss_components["residue"] = residue_loss
            weighted_residue = residue_loss * self.lambda_residue
            loss_components["weighted_residue"] = weighted_residue
            loss_value = loss_value + weighted_residue
        anchor_loss = self._target_anchor_loss(
            target_embeds=target_embeds,
            residue_embeddings=unique_residues,
            residue_padding_mask=unique_masks,
            score_residue_embeddings=unique_score_residues,
            score_residue_padding_mask=unique_score_masks,
            capability_vectors=unique_capability_vectors,
            capability_mask=unique_capability_mask,
        )
        if anchor_loss is not None:
            loss_components["anchor"] = anchor_loss
            weighted_anchor = anchor_loss * self.anchor_weight
            loss_components["weighted_anchor"] = weighted_anchor
            loss_value = loss_value + weighted_anchor
        consistency_loss = self._capability_consistency_loss(
            target_embeds=target_embeds,
            capability_vectors=unique_capability_vectors,
            capability_mask=unique_capability_mask,
        )
        if consistency_loss is not None:
            loss_components["capability_consistency"] = consistency_loss
            weighted_consistency = consistency_loss * self.capability_consistency_weight
            loss_components["weighted_capability_consistency"] = weighted_consistency
            loss_value = loss_value + weighted_consistency
        biofp_loss, biofp_components = self._biofp_auxiliary_loss(
            pooling_details=pooling_details,
            biofp_targets=unique_biofp_targets,
        )
        if biofp_loss is not None:
            loss_components.update(biofp_components)
            loss_components["biofp"] = biofp_loss
            weighted_biofp = biofp_loss * self.biofp_aux_weight
            loss_components["weighted_biofp"] = weighted_biofp
            loss_value = loss_value + weighted_biofp
        if self.cross_tower_alignment_weight > 0:
            alignment_query_idx, alignment_target_idx = self._positive_pair_indices(
                unique_query_ids,
                unique_target_ids,
                query_ids,
                target_ids,
                positive_pair_source="observed_pairs",
            )
            alignment_loss, alignment_components = self._cross_tower_factor_alignment_loss(
                query_embeds=query_embeds,
                pooling_details=pooling_details,
                biofp_targets=unique_biofp_targets,
                query_indices=alignment_query_idx,
                target_indices=alignment_target_idx,
            )
            if alignment_loss is None:
                raise RuntimeError("Cross-tower factor alignment produced no loss")
            loss_components.update(alignment_components)
            loss_components["cross_tower_alignment"] = alignment_loss
            weighted_alignment = alignment_loss * self.cross_tower_alignment_weight
            loss_components["weighted_cross_tower_alignment"] = weighted_alignment
            loss_value = loss_value + weighted_alignment
        if self.reaction_attention_entropy_weight > 0:
            if not isinstance(query_attention_details, dict):
                raise RuntimeError(
                    "Reaction attention regularization requires modality attention details"
                )
            weights = query_attention_details.get("modality")
            modality_mask = query_attention_details.get("modality_mask")
            if weights is None or modality_mask is None:
                raise RuntimeError(
                    "Reaction attention regularization requires modality weights and mask"
                )
            valid_counts = modality_mask.sum(dim=1)
            valid_rows = valid_counts > 1
            if bool(valid_rows.any()):
                entropy = -(weights.clamp_min(1e-12).log() * weights).sum(dim=1)
                normalized_entropy = entropy[valid_rows] / torch.log(
                    valid_counts[valid_rows].to(dtype=weights.dtype)
                )
                entropy_penalty = (
                    F.relu(self.reaction_attention_min_normalized_entropy - normalized_entropy)
                    .pow(2)
                    .mean()
                )
                weighted_entropy = entropy_penalty * self.reaction_attention_entropy_weight
                loss_components["reaction_attention_entropy"] = entropy_penalty
                loss_components["weighted_reaction_attention_entropy"] = weighted_entropy
                loss_value = loss_value + weighted_entropy
        if self.enzyme_block_weight_kl_weight > 0:
            if pooling_details is None or "enzyme_block_weight_kl" not in pooling_details:
                raise RuntimeError("Enzyme block KL regularization requires learned block details")
            block_kl = pooling_details["enzyme_block_weight_kl"].mean()
            weighted_block_kl = block_kl * self.enzyme_block_weight_kl_weight
            loss_components["enzyme_block_weight_kl"] = block_kl
            loss_components["weighted_enzyme_block_weight_kl"] = weighted_block_kl
            loss_value = loss_value + weighted_block_kl
        return loss_value, len(query_ids), attention_stats, loss_components

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        should_log_attention = (
            self.log_attention_stats
            and self.attention_logging_interval > 0
            and batch_idx % self.attention_logging_interval == 0
        )
        loss, batch_size, attention_stats, loss_components = self._compute_full_batch_loss(
            batch,
            return_attention_stats=should_log_attention,
        )
        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        for name, value in attention_stats.items():
            self.log(
                f"train/{name}",
                value,
                on_epoch=False,
                on_step=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        self._log_loss_components(
            "train",
            loss_components,
            batch_size=batch_size,
            on_step=True,
            on_epoch=True,
        )
        if self.loss_fn.learn_beta:
            self.log(
                "train/beta",
                self.loss_fn.beta,
                on_epoch=True,
                on_step=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        return loss

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor | None:
        if dataloader_idx == 1 and self.validation_retrieval_metrics:
            self._validation_target_lookup_step(batch)
            return None
        if dataloader_idx == 2 and self.validation_retrieval_metrics:
            self._validation_query_lookup_step(batch)
            return None
        metric_loader_idx = 3
        if (
            "reaction_to_enzyme" in self.validation_retrieval_directions
            and dataloader_idx == metric_loader_idx
            and self.validation_retrieval_metrics
        ):
            self._validation_reaction_to_enzyme_step(batch)
            return None
        if "reaction_to_enzyme" in self.validation_retrieval_directions:
            metric_loader_idx += 1
        if (
            "enzyme_to_reaction" in self.validation_retrieval_directions
            and dataloader_idx == metric_loader_idx
            and self.validation_retrieval_metrics
        ):
            self._validation_enzyme_to_reaction_step(batch)
            return None
        if dataloader_idx != 0:
            return None

        loss, batch_size, _, loss_components = self._compute_full_batch_loss(
            batch,
            positive_pair_source="observed_pairs",
            structure_terms_enabled=self.apply_structure_terms_on_val,
        )
        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self._log_loss_components(
            "val",
            loss_components,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
