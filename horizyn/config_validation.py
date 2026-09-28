"""Ordered validation for core training and optional experimental configurations.

Every section is checked in the historical order, even when a variant is inactive.
Keep this order: callers rely on the first error and on legacy default mutations.
"""

import json
import math
from pathlib import Path
from typing import Any

from horizyn.config import DotDict, _has_global_or_split_values


def validate_config(config: DotDict) -> None:
    """Validate supported recipes without changing values or error precedence."""
    _validate_data(config)
    _validate_runtime_and_model_shape(config)
    _validate_adapters(config)
    _validate_prototypes_and_residuals(config)
    _validate_loss(config)
    _validate_enzyme_inputs(config)
    _validate_cross_tower_alignment(config)
    _validate_reaction_attention(config)
    _validate_sampling_and_cache(config)
    _validate_reaction_features(config)
    _validate_residue_pooling(config)
    _validate_hyperbolic_and_fingerprint(config)
    _validate_logging_and_precision(config)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_adapter_options(adapter, name, *, boolean_keys, dimension_keys):
    """Common scalar checks; direction-specific dependencies stay with each adapter."""
    prefix = f"model.{name}"
    if not isinstance(adapter, dict):
        raise ValueError(f"'{prefix}' must be a mapping")
    for key in boolean_keys:
        if key in adapter and not isinstance(adapter[key], bool):
            raise ValueError(f"'{prefix}.{key}' must be a boolean")
    for key in dimension_keys:
        if key in adapter:
            value = adapter[key]
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"'{prefix}.{key}' must be a positive integer")
    for key in ("dropout", "gate_init"):
        if key in adapter:
            value = adapter[key]
            if not _is_number(value) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"'{prefix}.{key}' must be in [0, 1]")
    if "gate_init" in adapter and not 0.0 < float(adapter.gate_init) < 1.0:
        raise ValueError(f"'{prefix}.gate_init' must be in (0, 1)")


def _validate_data(config: DotDict) -> None:
    required_sections = ["data", "model", "training"]
    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing required config section: '{section}'\n"
                f"Config must have sections: {required_sections}"
            )

    model_name = config.get("model", {}).get("name", "")

    # Validate data section
    required_data_keys = [
        "train_pairs_path",
        "train_reactions_path",
    ]
    if (
        model_name in {"ReactionConditionedDualModel", "ProteinPooledDualModel"}
        or "protein_residue_embeds_path" in config.data
    ):
        required_data_keys.append("protein_residue_embeds_path")
    else:
        required_data_keys.append("protein_embeds_path")

    for key in required_data_keys:
        if key not in config.data:
            raise ValueError(
                f"Missing required data config parameter: 'data.{key}'\n"
                f"Required data parameters: {required_data_keys}"
            )
    validation_enabled = config.training.get("validation_enabled", True)
    if type(validation_enabled) is not bool:
        raise ValueError("'training.validation_enabled' must be boolean")
    if validation_enabled:
        for test_key, validation_key in (
            ("test_pairs_path", "validation_pairs_path"),
            ("test_reactions_path", "validation_reactions_path"),
        ):
            if test_key not in config.data and validation_key not in config.data:
                raise ValueError(
                    "Missing required data config parameter: "
                    f"provide either 'data.{validation_key}' or 'data.{test_key}'"
                )
    else:
        if config.training.get("validation_retrieval_metrics", False):
            raise ValueError(
                "training.validation_retrieval_metrics must be false when validation is disabled"
            )
        if config.training.get("early_stopping", {}).get("enabled", False):
            raise ValueError("early stopping cannot be enabled when validation is disabled")
    if (
        "enzyme_ec_labels_path" in config.data
        and config.data.enzyme_ec_labels_path is not None
        and not isinstance(config.data.enzyme_ec_labels_path, str)
    ):
        raise ValueError("'data.enzyme_ec_labels_path' must be a string path")
    for key in (
        "protein_capability_vectors_path",
        "protein_capability_metadata_path",
        "protein_factorized_capability_vectors_path",
        "reaction_chemistry_vectors_path",
        "train_reaction_chemistry_vectors_path",
        "validation_reaction_chemistry_vectors_path",
        "reaction_directional_vectors_path",
        "train_reaction_directional_vectors_path",
        "validation_reaction_directional_vectors_path",
        "protein_text_vectors_path",
        "protein_text_metadata_path",
        "protein_biofp_targets_path",
        "protein_biofp_vocab_path",
        "typed_negative_pools_path",
        "indexed_pairs_dir",
        "validation_retrieval_query_ids_path",
    ):
        if (
            key in config.data
            and config.data[key] is not None
            and not isinstance(config.data[key], str)
        ):
            raise ValueError(f"'data.{key}' must be a string path or null")
    for key in (
        "capability_missing_policy",
        "factorized_capability_missing_policy",
        "text_vector_missing_policy",
        "biofp_missing_policy",
    ):
        if key in config.data and config.data[key] not in {"zero_with_mask"}:
            raise ValueError(f"'data.{key}' must be 'zero_with_mask'")
    for key in ("reaction_use_directional", "reaction_load_directional"):
        if key in config.data and not isinstance(config.data[key], bool):
            raise ValueError(f"'data.{key}' must be a boolean")
    if "hard_negative_direction" in config.data and config.data.hard_negative_direction not in {
        "reaction_to_enzyme",
        "enzyme_to_reaction",
    }:
        raise ValueError(
            "'data.hard_negative_direction' must be reaction_to_enzyme " "or enzyme_to_reaction"
        )
    typed_pool_path = config.data.get("typed_negative_pools_path", None)
    indexed_pairs_dir = config.data.get("indexed_pairs_dir", None)
    if indexed_pairs_dir is not None and typed_pool_path is not None:
        raise ValueError("indexed_pairs_dir and typed_negative_pools_path are mutually exclusive")
    configured_loss_name = str(
        config.training.get("loss", {}).get("name", "FullBatchMLNCELoss")
    ).lower()
    sampled_loss_names = {
        "sampledmultipositiveinfonceloss",
        "sampled_multi_positive_infonce",
        "bidirectionalsampledmultipositiveinfonceloss",
        "bidirectional_sampled_multi_positive_infonce",
    }
    uses_sampled_loss = configured_loss_name in sampled_loss_names
    if uses_sampled_loss and typed_pool_path is None and indexed_pairs_dir is None:
        raise ValueError(
            "SampledMultiPositiveInfoNCELoss requires data.typed_negative_pools_path or data.indexed_pairs_dir"
        )
    if typed_pool_path is not None or indexed_pairs_dir is not None:
        if not uses_sampled_loss:
            raise ValueError(
                "data.typed_negative_pools_path requires SampledMultiPositiveInfoNCELoss"
            )
        positive_pair_source = config.training.get("loss", {}).get(
            "positive_pair_source", "observed_pairs"
        )
        if positive_pair_source != "all_known_in_batch":
            raise ValueError(
                "typed negative sampling requires positive_pair_source=all_known_in_batch"
            )
        if config.data.get("hard_negative_pools_path", None) is not None:
            raise ValueError("typed and legacy hard-negative pools are mutually exclusive")
        if config.data.get("reaction_direction_mode", "bidirectional") != "forward_only":
            raise ValueError(
                "typed negative sampling requires reaction_direction_mode=forward_only"
            )
        expected_positive_fraction = 0.85 if indexed_pairs_dir is not None else 0.5
        positive_fraction = config.data.get(
            "typed_negative_positive_fraction", expected_positive_fraction
        )
        allowed_fractions = (0.85, 0.5) if indexed_pairs_dir is not None else (0.5,)
        if positive_fraction not in allowed_fractions:
            requirement = f"one of {allowed_fractions}" if indexed_pairs_dir is not None else "exactly 0.5"
            raise ValueError(
                f"'data.typed_negative_positive_fraction' must be {requirement}"
            )
        biological_fraction = config.data.get("typed_negative_biological_fraction", 0.5)
        if (
            not isinstance(biological_fraction, (int, float))
            or isinstance(biological_fraction, bool)
            or not 0.0 <= biological_fraction <= 1.0
        ):
            raise ValueError("'data.typed_negative_biological_fraction' must be in [0, 1]")
        train_batch_size = config.data.get("train_batch_size", 64)
        if (
            not isinstance(train_batch_size, int)
            or isinstance(train_batch_size, bool)
            or train_batch_size <= 0
        ):
            raise ValueError("'data.train_batch_size' must be a positive integer")
        if indexed_pairs_dir is not None and positive_fraction == 0.85 and train_batch_size % 20:
            raise ValueError(
                "85/15 indexed sampling requires data.train_batch_size divisible by 20"
            )
        if positive_fraction == 0.5 and train_batch_size % 2:
            raise ValueError("50/50 typed negative sampling requires an even data.train_batch_size")
        if indexed_pairs_dir is not None and positive_fraction == 0.5 and train_batch_size < 4:
            raise ValueError("50/50 indexed sampling requires data.train_batch_size >= 4")


def _validate_runtime_and_model_shape(config: DotDict) -> None:
    required_model_keys = ["query_encoder_dims", "target_encoder_dims", "embedding_dim"]
    for key in required_model_keys:
        if key not in config.model:
            raise ValueError(
                f"Missing required model config parameter: 'model.{key}'\n"
                f"Required model parameters: {required_model_keys}"
            )

    # Validate training section
    for section, keys in (
        ("data", ("persistent_workers",)),
        (
            "training",
            (
                "fused_adamw",
                "contrastive_fp32",
                "fp32_sensitive_modules",
                "ddp_gradient_as_bucket_view",
            ),
        ),
    ):
        for key in keys:
            if key in config[section] and not isinstance(config[section][key], bool):
                raise ValueError(f"'{section}.{key}' must be a boolean")
    for section, keys in (
        ("data", ("prefetch_factor", "worker_num_threads")),
        ("training", ("cpu_num_threads",)),
    ):
        for key in keys:
            value = config[section].get(key)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"'{section}.{key}' must be a positive integer or null")
    if "num_workers" in config.data:
        value = config.data.num_workers
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("'data.num_workers' must be a non-negative integer")
    if config.training.get("fp32_sensitive_modules", False) and config.training.get(
        "precision"
    ) in {"16-true", "bf16-true", "64-true"}:
        raise ValueError(
            "FP32 sensitive modules require mixed precision or FP32 parameters, not true reduced/double precision"
        )
    if "max_epochs" not in config.training:
        raise ValueError("Missing required training parameter: 'training.max_epochs'")

    # Type validation
    if not isinstance(config.training.max_epochs, int):
        raise ValueError(
            "'training.max_epochs' must be an integer, got "
            f"{type(config.training.max_epochs).__name__}"
        )
    if "training_stage" in config.training and config.training.training_stage not in {
        "joint",
        "enzyme_only_tuning",
        "joint_capability_retrieval",
        "e2r_adapter",
        "r2e_adapter",
        "bidirectional_adapters",
        "prototype_only",
        "biological_residual",
    }:
        raise ValueError(
            "'training.training_stage' must be one of: joint, "
            "enzyme_only_tuning, joint_capability_retrieval, e2r_adapter, "
            "r2e_adapter, bidirectional_adapters, prototype_only, "
            "biological_residual"
        )
    if "validation_interval_steps" in config.training:
        value = config.training.validation_interval_steps
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError("'training.validation_interval_steps' must be a positive integer")
    if "save_every_n_train_steps" in config.get("logging", {}):
        value = config.logging.save_every_n_train_steps
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError("'logging.save_every_n_train_steps' must be a positive integer")

    for key in ("query_encoder_dims", "target_encoder_dims"):
        if not isinstance(config.model[key], list):
            raise ValueError(
                f"'model.{key}' must be a list, got {type(config.model[key]).__name__}"
            )
    if not isinstance(config.model.embedding_dim, int):
        raise ValueError(
            "'model.embedding_dim' must be an integer, got "
            f"{type(config.model.embedding_dim).__name__}"
        )
    if (
        "query_encoder_checkpoint_path" in config.model
        and config.model.query_encoder_checkpoint_path is not None
        and not isinstance(config.model.query_encoder_checkpoint_path, str)
    ):
        raise ValueError("'model.query_encoder_checkpoint_path' must be a string path or null")


def _validate_adapters(config: DotDict) -> None:
    e2r_adapter = config.model.get("e2r_adapter", {})
    _validate_adapter_options(
        e2r_adapter,
        "e2r_adapter",
        boolean_keys=("enabled", "use_factorized_inputs", "use_directional_inputs"),
        dimension_keys=("hidden_dim", "directional_dim", "directional_hidden_dim"),
    )
    if e2r_adapter.get("enabled", False):
        if config.training.get("training_stage") == "e2r_adapter":
            if not config.training.get("init_from_checkpoint"):
                raise ValueError(
                    "training_stage=e2r_adapter requires " "'training.init_from_checkpoint'"
                )
        if e2r_adapter.get("use_factorized_inputs", False):
            if not config.data.get("reaction_use_chemistry", False):
                raise ValueError("factorized E2R inputs require data.reaction_use_chemistry=true")
        if e2r_adapter.get("use_directional_inputs", False):
            directional_dim = e2r_adapter.get(
                "directional_dim",
                config.data.get("reaction_directional_dim"),
            )
            if not isinstance(directional_dim, int) or directional_dim <= 0:
                raise ValueError("directional E2R inputs require a positive directional dimension")
            if not config.data.get("reaction_load_directional", False):
                raise ValueError(
                    "directional E2R inputs require " "data.reaction_load_directional=true"
                )

    r2e_adapter = config.model.get("r2e_adapter", {})
    _validate_adapter_options(
        r2e_adapter,
        "r2e_adapter",
        boolean_keys=("enabled", "use_factorized_inputs"),
        dimension_keys=("hidden_dim",),
    )
    if r2e_adapter.get("enabled", False):
        if config.training.get("training_stage") == "r2e_adapter":
            if not config.training.get("init_from_checkpoint"):
                raise ValueError(
                    "training_stage=r2e_adapter requires " "'training.init_from_checkpoint'"
                )
        if r2e_adapter.get("use_factorized_inputs", False):
            block_dims = r2e_adapter.get(
                "block_dims",
                config.model.get("enzyme_block_fusion", {}).get("dims"),
            )
            block_weights = r2e_adapter.get(
                "block_weights",
                config.model.get("enzyme_block_fusion", {}).get("weights"),
            )
            if not isinstance(block_dims, dict) or not isinstance(block_weights, dict):
                raise ValueError(
                    "factorized R2E inputs require enzyme block dimensions and weights"
                )
            if set(block_dims) != set(block_weights):
                raise ValueError(
                    "factorized R2E block dimensions and weights must have matching keys"
                )
            if sum(int(value) for value in block_dims.values()) != int(config.model.embedding_dim):
                raise ValueError("factorized R2E block dimensions must sum to model.embedding_dim")

    if config.training.get("training_stage") == "bidirectional_adapters":
        if not config.training.get("init_from_checkpoint"):
            raise ValueError(
                "training_stage=bidirectional_adapters requires " "'training.init_from_checkpoint'"
            )
        if not e2r_adapter.get("enabled", False) or not r2e_adapter.get("enabled", False):
            raise ValueError(
                "training_stage=bidirectional_adapters requires both model.e2r_adapter "
                "and model.r2e_adapter to be enabled"
            )


def _validate_prototypes_and_residuals(config: DotDict) -> None:
    prototype_config = config.model.get("enzyme_prototypes", {})
    if not isinstance(prototype_config, dict):
        raise ValueError("'model.enzyme_prototypes' must be a mapping")
    prototype_count = prototype_config.get("count", 1)
    bottleneck_dim = prototype_config.get("bottleneck_dim", 64)
    if not isinstance(prototype_count, int) or prototype_count <= 0:
        raise ValueError("'model.enzyme_prototypes.count' must be a positive integer")
    if not isinstance(bottleneck_dim, int) or bottleneck_dim <= 0:
        raise ValueError("'model.enzyme_prototypes.bottleneck_dim' must be a positive integer")
    for key in ("temperature",):
        value = prototype_config.get(key, 0.1)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0.0:
            raise ValueError(f"'model.enzyme_prototypes.{key}' must be positive")
    gate_init = prototype_config.get("residual_gate_init", 0.05)
    if (
        not isinstance(gate_init, (int, float))
        or isinstance(gate_init, bool)
        or not 0.0 < float(gate_init) < 1.0
    ):
        raise ValueError("'model.enzyme_prototypes.residual_gate_init' must be in (0, 1)")
    dropout = prototype_config.get("dropout", 0.0)
    if (
        not isinstance(dropout, (int, float))
        or isinstance(dropout, bool)
        or not 0.0 <= float(dropout) < 1.0
    ):
        raise ValueError("'model.enzyme_prototypes.dropout' must be in [0, 1)")
    for key in ("routing_entropy_weight", "load_balance_weight"):
        value = prototype_config.get(key, 0.0)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0.0:
            raise ValueError(f"'model.enzyme_prototypes.{key}' must be non-negative")
    if config.training.get("training_stage") == "prototype_only":
        if prototype_count <= 1:
            raise ValueError("training_stage=prototype_only requires enzyme_prototypes.count > 1")
        if not config.training.get("init_from_checkpoint"):
            raise ValueError(
                "training_stage=prototype_only requires 'training.init_from_checkpoint'"
            )
    biological_residual = config.model.get("biological_residual", {})
    if not isinstance(biological_residual, dict):
        raise ValueError("'model.biological_residual' must be a mapping")
    if config.training.get("training_stage") == "biological_residual":
        if not biological_residual.get("enabled", False):
            raise ValueError(
                "training_stage=biological_residual requires "
                "model.biological_residual.enabled=true"
            )
        if not config.training.get("init_from_checkpoint"):
            raise ValueError(
                "training_stage=biological_residual requires " "'training.init_from_checkpoint'"
            )
    if biological_residual:
        alpha = biological_residual.get("fusion_alpha", 0.0)
        max_alpha = biological_residual.get("max_fusion_alpha", 0.2)
        if (
            not isinstance(alpha, (int, float))
            or isinstance(alpha, bool)
            or not isinstance(max_alpha, (int, float))
            or isinstance(max_alpha, bool)
        ):
            raise ValueError("Biological residual fusion alphas must be numeric")
        if float(max_alpha) <= 0.0 or abs(float(alpha)) > float(max_alpha):
            raise ValueError(
                "model.biological_residual fusion_alpha must satisfy "
                "abs(fusion_alpha) <= max_fusion_alpha"
            )


def _validate_loss(config: DotDict) -> None:
    loss_config = config.training.get("loss", {})
    loss_name = loss_config.get("name", "FullBatchMLNCELoss")
    valid_loss_names = {
        "FullBatchMLNCELoss",
        "SampledMultiPositiveInfoNCELoss",
        "BidirectionalSampledMultiPositiveInfoNCELoss",
        "DegreeTemperedFullBatchMLNCELoss",
        "DecoupledAllPositiveInfoNCELoss",
        "HybridCardinalityRetrievalLoss",
        "BalancedSigmoidEBMLoss",
        "HorizynFGWLoss",
        "BidirectionalAnchorBalancedSupConLoss",
        "MultiAlignmentRetrievalLoss",
        "mlnce",
        "sampled_multi_positive_infonce",
        "bidirectional_sampled_multi_positive_infonce",
        "degree_tempered_mlnce",
        "decoupled_all_positive_infonce",
        "hybrid_cardinality_retrieval",
        "balanced_sigmoid_ebm",
        "fgw",
        "horizyn_fgw",
        "bidirectional_anchor_balanced_supcon",
        "anchor_balanced_supcon",
        "multi_alignment_retrieval",
        "multi_alignment",
    }
    if loss_name not in valid_loss_names:
        raise ValueError(
            "'training.loss.name' must be one of: " + ", ".join(sorted(valid_loss_names))
        )

    for key in (
        "lambda_r",
        "lambda_e",
        "lambda_g",
        "lambda_rr",
        "lambda_ee",
        "lambda_gw",
        "lambda_direction_gap",
        "lambda_r2e",
        "lambda_e2r",
        "lambda_r2e_hard_neg",
        "r2e_hard_neg_margin",
        "lambda_e2r_hard_neg",
        "e2r_hard_neg_margin",
        "e2r_identity_weight",
        "r2e_identity_weight",
        "anchor_weight",
        "capability_consistency_weight",
    ):
        if key in loss_config:
            value = loss_config[key]
            if not _is_number(value) or value < 0:
                raise ValueError(f"'training.loss.{key}' must be a non-negative number")
    if "r2e_hard_neg_top_k" in loss_config:
        value = loss_config.r2e_hard_neg_top_k
        if not isinstance(value, int) or value < 0:
            raise ValueError("'training.loss.r2e_hard_neg_top_k' must be a non-negative integer")
    if "e2r_hard_neg_top_k" in loss_config:
        value = loss_config.e2r_hard_neg_top_k
        if not isinstance(value, int) or value < 0:
            raise ValueError("'training.loss.e2r_hard_neg_top_k' must be a non-negative integer")
    if ("lambda_r2e" in loss_config or "lambda_e2r" in loss_config) and (
        float(loss_config.get("lambda_r2e", 0.5)) + float(loss_config.get("lambda_e2r", 0.5)) <= 0.0
    ):
        raise ValueError("'training.loss.lambda_r2e + lambda_e2r' must be positive")
    if "direction_balance_weight" in loss_config:
        value = loss_config.direction_balance_weight
        if not _is_number(value) or value < 0:
            raise ValueError("'training.loss.direction_balance_weight' must be non-negative")
    for key in ("tau_r", "tau_e", "tau_t", "tau_rr", "tau_ee", "tau_gw"):
        if key in loss_config:
            value = loss_config[key]
            if not _is_number(value) or value <= 0:
                raise ValueError(f"'training.loss.{key}' must be a positive number")
    for key in ("delta_r", "delta_e"):
        if key in loss_config and not _is_number(loss_config[key]):
            raise ValueError(f"'training.loss.{key}' must be a number")
    if "symmetric_gw" in loss_config and not isinstance(loss_config.symmetric_gw, bool):
        raise ValueError("'training.loss.symmetric_gw' must be a boolean")
    if "ec_positive_policy" in loss_config and loss_config.ec_positive_policy not in {
        "hierarchical_weighted",
    }:
        raise ValueError("'training.loss.ec_positive_policy' must be 'hierarchical_weighted'")
    if "ec_min_shared_depth" in loss_config and loss_config.ec_min_shared_depth not in {
        2,
        3,
        4,
    }:
        raise ValueError("'training.loss.ec_min_shared_depth' must be one of: 2, 3, 4")
    if "gw_max_anchors" in loss_config:
        value = loss_config.gw_max_anchors
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError("'training.loss.gw_max_anchors' must be a positive integer or null")
    if "apply_structure_terms_on_val" in loss_config and not isinstance(
        loss_config.apply_structure_terms_on_val,
        bool,
    ):
        raise ValueError("'training.loss.apply_structure_terms_on_val' must be a boolean")


def _enzyme_attention_regularization_weights(options, enzyme_input_mode: str) -> dict[str, float]:
    """Resolve regularizer defaults only for the multiview tower."""
    if options is not None and not isinstance(options, dict):
        raise ValueError("'training.enzyme_attention_regularization' must be a mapping or null")
    options = {} if options is None else options
    unknown = set(options) - {"entropy_weight", "diversity_weight"}
    if unknown:
        raise ValueError(f"Unknown enzyme_attention_regularization options: {sorted(unknown)}")
    active = enzyme_input_mode == "raw_mean_sleec_multiview"
    result = {}
    for key, default in (("entropy_weight", 0.01), ("diversity_weight", 0.001)):
        value = options.get(key, default if active else 0.0)
        if not _is_number(value) or not math.isfinite(value) or value < 0:
            raise ValueError(f"'training.enzyme_attention_regularization.{key}' must be finite and non-negative")
        result[key] = float(value)
    if not active and any(result.values()):
        raise ValueError("enzyme_attention_regularization requires raw_mean_sleec_multiview")
    return result


def validate_enzyme_multiview_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Shared, Torch-free architecture validation for YAML and model construction."""
    defaults = {
        "hidden_dim": 256, "num_slots": 4, "dropout": 0.1, "uniform_mix": 0.05,
        "min_effective_residues": 4.0, "diversity_margin": 0.9, "gate_floor": 0.05,
        "max_logit_scale": 10.0,
        "initial_residual_scale": 0.1,
    }
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("enzyme_multiview must be a mapping or None")
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown enzyme_multiview settings: {sorted(unknown)}")
    settings = {**defaults, **config}
    for name in ("hidden_dim", "num_slots"):
        value = settings[name]
        minimum = 0 if name == "num_slots" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            domain = "non-negative" if name == "num_slots" else "positive"
            raise ValueError(f"enzyme_multiview.{name} must be a {domain} integer")
    if settings["num_slots"] > settings["hidden_dim"]:
        raise ValueError("enzyme_multiview.num_slots cannot exceed hidden_dim")
    for name in set(defaults) - {"hidden_dim", "num_slots"}:
        value = settings[name]
        if not _is_number(value) or not math.isfinite(value):
            raise ValueError(f"enzyme_multiview.{name} must be a finite number")
        settings[name] = float(value)
    for name in ("dropout", "uniform_mix", "gate_floor"):
        if not 0 <= settings[name] < 1:
            raise ValueError(f"enzyme_multiview.{name} must be in [0, 1)")
    if not 0 <= settings["diversity_margin"] < 1:
        raise ValueError("enzyme_multiview.diversity_margin must be in [0, 1)")
    if settings["min_effective_residues"] < 1:
        raise ValueError("enzyme_multiview.min_effective_residues must be >= 1")
    if settings["max_logit_scale"] <= 0:
        raise ValueError("enzyme_multiview.max_logit_scale must be positive")
    if not 0 < settings["initial_residual_scale"] < 0.5:
        raise ValueError("enzyme_multiview.initial_residual_scale must be in (0, 0.5)")
    return settings


def _validate_enzyme_multiview_options(options, enzyme_input_mode: str) -> None:
    validate_enzyme_multiview_config(options)
    if options and enzyme_input_mode != "raw_mean_sleec_multiview":
        raise ValueError("enzyme_multiview options require raw_mean_sleec_multiview")


def _validate_enzyme_inputs(config: DotDict) -> None:
    enzyme_input_mode = config.model.get("enzyme_input_mode", "standard")
    _validate_enzyme_multiview_options(config.model.get("enzyme_multiview"), enzyme_input_mode)
    enzyme_regularization = _enzyme_attention_regularization_weights(
        config.training.get("enzyme_attention_regularization"), enzyme_input_mode)
    if any(enzyme_regularization.values()) and config.training.get("training_stage", "joint") not in {
        "joint", "enzyme_only_tuning", "joint_capability_retrieval",
    }:
        raise ValueError("Enzyme attention regularization requires a training stage that updates the enzyme tower")
    if enzyme_input_mode == "raw_mean_sleec_multiview":
        if config.model.get("pooling", "mean") != "sleec_guided_attention":
            raise ValueError("raw_mean_sleec_multiview requires pooling=sleec_guided_attention")
        if config.model.get("sleec_pooling", {}).get("freeze_scorer", False) is not True:
            raise ValueError("raw_mean_sleec_multiview requires a frozen SLEEC scorer")
        if not config.model.get("sleec_pooling", {}).get("checkpoint_path"):
            raise ValueError("raw_mean_sleec_multiview requires a SLEEC checkpoint_path for training")
        if config.model.get("hyperbolic_encoder", {}).get("checkpoint_path"):
            raise ValueError("raw_mean_sleec_multiview does not support an enzyme hyperbolic checkpoint")
        loss = config.training.get("loss", {})
        if loss.get("biofp_aux_weight", 0.0) != 0 and loss.get("biofp_aux_mode", "bce") != "positive_anchor":
            raise ValueError("Nonzero multiview biofp_aux_weight requires positive_anchor supervision")
        if config.model.get("enzyme_block_fusion", {}).get("weight_kl_weight", 0.0) != 0:
            raise ValueError("raw_mean_sleec_multiview does not support enzyme block KL regularization")
        for name in ("r2e_adapter", "biological_residual"):
            if config.model.get(name, {}).get("enabled", False):
                raise ValueError(f"raw_mean_sleec_multiview does not support {name}")
    loss = config.training.get("loss", {})
    aux_mode = loss.get("biofp_aux_mode", "bce")
    if aux_mode not in {"bce", "positive_anchor"}:
        raise ValueError("biofp_aux_mode must be bce or positive_anchor")
    labels = config.model.get("biofp", {}).get("positive_labels")
    if aux_mode == "positive_anchor":
        if enzyme_input_mode != "raw_mean_sleec_multiview" or not isinstance(labels, dict) or not labels:
            raise ValueError("positive_anchor requires multiview and positive_labels")
        if set(labels) - {"ec", "cofactor", "mechanism"}:
            raise ValueError("Unknown positive_anchor family")
        for family, names in labels.items():
            if not isinstance(names, list) or not names or any(not isinstance(x, str) or not x.strip() for x in names):
                raise ValueError(f"positive_labels.{family} must be a nonempty string list")
            if len(set(names)) != len(names):
                raise ValueError(f"Duplicate positive_labels.{family}")
        weights = loss.get("biofp_family_weights", {})
        if set(weights) != set(labels) or any(not _is_number(x) or not math.isfinite(x) or x <= 0 for x in weights.values()):
            raise ValueError("positive_anchor requires matching positive family weights")
        weight = loss.get("biofp_aux_weight", 0.0)
        if not _is_number(weight) or not math.isfinite(weight) or weight < 0:
            raise ValueError("biofp_aux_weight must be finite and non-negative")
        if loss.get("biofp_normalize_active_families") is not True:
            raise ValueError("positive_anchor requires biofp_normalize_active_families")
        for name in ("protein_biofp_targets_path", "protein_biofp_vocab_path"):
            if not config.data.get(name):
                raise ValueError(f"positive_anchor requires data.{name}")
        vocab_path = Path(config.data.protein_biofp_vocab_path)
        if vocab_path.exists():
            payload = json.loads(vocab_path.read_text())
            if payload.get("schema_version") != "positive_bio_v1" or payload.get("families") != labels:
                raise ValueError("Positive annotation vocabulary/schema mismatch")
    elif labels:
        raise ValueError("positive_labels requires biofp_aux_mode=positive_anchor")
    if "score_mode" in config.model and config.model.score_mode not in {
        "target_mlp",
        "projected_value",
    }:
        raise ValueError("'model.score_mode' must be one of: target_mlp, projected_value")
    if "pooling" in config.model and config.model.pooling not in {
        "mean",
        "avg",
        "attention",
        "sleec",
        "sleec_guided_attention",
    }:
        raise ValueError(
            "'model.pooling' must be one of: mean, avg, attention, sleec, " "sleec_guided_attention"
        )
    if "enzyme_input_mode" in config.model and config.model.enzyme_input_mode not in {
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
            "'model.enzyme_input_mode' must be one of: standard, "
            "raw_sleec_hyperbolic_concat, raw_mean_sleec_hyperbolic_gated, "
            "raw_mean_sleec_hyperbolic_capability_gated, "
            "raw_mean_sleec_hyperbolic_text_gated, raw_mean_sleec_blockwise, "
            "raw_mean_sleec_hyperbolic_blockwise, "
            "raw_mean_sleec_hyperbolic_capability_blockwise, "
            "raw_mean_sleec_hyperbolic_factorized_capability_blockwise, "
            "raw_mean_sleec_biofp_split, raw_mean_sleec_biological_factorized, raw_mean_sleec_multiview"
        )
    if config.model.get("enzyme_input_mode") in {
        "raw_mean_sleec_hyperbolic_capability_gated",
        "raw_mean_sleec_hyperbolic_capability_blockwise",
    }:
        if not config.data.get("protein_capability_vectors_path", None):
            raise ValueError(
                "data.protein_capability_vectors_path is required when "
                "model.enzyme_input_mode uses the capability branch"
            )
        capability_config = config.model.get("capability_vector", {})
        capability_dim = capability_config.get(
            "dim",
            config.model.get("capability_vector_dim", 256),
        )
        if not isinstance(capability_dim, int) or capability_dim <= 0:
            raise ValueError("'model.capability_vector.dim' must be a positive integer")
        capability_dropout = capability_config.get(
            "dropout",
            config.model.get("capability_dropout", 0.1),
        )
        if not _is_number(capability_dropout) or not (0.0 <= float(capability_dropout) <= 1.0):
            raise ValueError("'model.capability_vector.dropout' must be in [0, 1]")
        missing_policy = capability_config.get(
            "missing_policy",
            config.data.get("capability_missing_policy", "zero_with_mask"),
        )
        if missing_policy != "zero_with_mask":
            raise ValueError("'model.capability_vector.missing_policy' must be 'zero_with_mask'")
    if (
        config.model.get("enzyme_input_mode")
        == "raw_mean_sleec_hyperbolic_factorized_capability_blockwise"
    ):
        if not config.data.get("protein_factorized_capability_vectors_path", None):
            raise ValueError(
                "data.protein_factorized_capability_vectors_path is required when "
                "model.enzyme_input_mode uses factorized capability branches"
            )
        factorized_config = config.model.get("factorized_capability_vector", {})
        dims = factorized_config.get("dims", {})
        expected_families = {"cofactor", "center", "transition"}
        if set(dims) != expected_families:
            raise ValueError(
                "'model.factorized_capability_vector.dims' must contain exactly "
                f"{sorted(expected_families)}"
            )
        for family in expected_families:
            dim = dims[family]
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(
                    f"'model.factorized_capability_vector.dims.{family}' "
                    "must be a positive integer"
                )
        if "use_masks" in factorized_config and not isinstance(
            factorized_config.use_masks,
            bool,
        ):
            raise ValueError("'model.factorized_capability_vector.use_masks' must be boolean")
        missing_policy = factorized_config.get(
            "missing_policy",
            config.data.get("factorized_capability_missing_policy", "zero_with_mask"),
        )
        if missing_policy != "zero_with_mask":
            raise ValueError(
                "'model.factorized_capability_vector.missing_policy' must be " "'zero_with_mask'"
            )
    if config.model.get("enzyme_input_mode") in {
        "raw_mean_sleec_blockwise",
        "raw_mean_sleec_hyperbolic_blockwise",
        "raw_mean_sleec_hyperbolic_capability_blockwise",
        "raw_mean_sleec_hyperbolic_factorized_capability_blockwise",
        "raw_mean_sleec_biological_factorized",
    }:
        block_config = config.model.get("enzyme_block_fusion", {})
        dims = block_config.get("dims", {})
        weights = block_config.get("weights", {})
        if (
            config.model.get("enzyme_input_mode") == "raw_mean_sleec_biological_factorized"
            and not dims
            and not weights
        ):
            dims = {
                "core": 288,
                "site": 96,
                "mechanism": 64,
                "cofactor": 32,
                "ec": 32,
            }
            weights = {
                "core": 0.55,
                "site": 0.20,
                "mechanism": 0.12,
                "cofactor": 0.08,
                "ec": 0.05,
            }
        if not isinstance(dims, dict):
            raise ValueError("'model.enzyme_block_fusion.dims' must be a mapping")
        if not isinstance(weights, dict):
            raise ValueError("'model.enzyme_block_fusion.weights' must be a mapping")
        mode = config.model.get("enzyme_input_mode")
        expected_blocks = (
            set(dims) if mode == "raw_mean_sleec_biological_factorized" else {"core", "site"}
        )
        if mode == "raw_mean_sleec_biological_factorized" and not {
            "core",
            "site",
        }.issubset(expected_blocks):
            raise ValueError("Biological enzyme block layout must contain core and site")
        if mode in {
            "raw_mean_sleec_hyperbolic_blockwise",
            "raw_mean_sleec_hyperbolic_capability_blockwise",
            "raw_mean_sleec_hyperbolic_factorized_capability_blockwise",
        }:
            expected_blocks.add("ec")
        if mode == "raw_mean_sleec_hyperbolic_capability_blockwise":
            expected_blocks.add("capability")
        if mode == "raw_mean_sleec_hyperbolic_factorized_capability_blockwise":
            expected_blocks.update({"cofactor", "center", "transition"})
        if set(dims) != expected_blocks:
            raise ValueError(
                "'model.enzyme_block_fusion.dims' must contain exactly "
                f"{sorted(expected_blocks)} for {mode}"
            )
        if set(weights) != expected_blocks:
            raise ValueError(
                "'model.enzyme_block_fusion.weights' must contain exactly "
                f"{sorted(expected_blocks)} for {mode}"
            )
        embedding_dim = config.model.get("embedding_dim", config.model.target_encoder_dims[-1])
        dim_sum = 0
        for name in expected_blocks:
            dim = dims[name]
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"'model.enzyme_block_fusion.dims.{name}' must be positive")
            dim_sum += dim
            weight = weights[name]
            if not _is_number(weight) or float(weight) < 0.0:
                raise ValueError(f"'model.enzyme_block_fusion.weights.{name}' must be non-negative")
        if dim_sum != embedding_dim:
            raise ValueError("'model.enzyme_block_fusion.dims' must sum to model.embedding_dim")
        if sum(float(weights[name]) for name in expected_blocks) <= 0.0:
            raise ValueError("'model.enzyme_block_fusion.weights' must have positive sum")
        dropout = block_config.get("dropout", 0.0)
        if not _is_number(dropout) or not (0.0 <= float(dropout) <= 1.0):
            raise ValueError("'model.enzyme_block_fusion.dropout' must be in [0, 1]")
        if "learned_weights" in block_config and not isinstance(
            block_config.learned_weights,
            bool,
        ):
            raise ValueError("'model.enzyme_block_fusion.learned_weights' must be boolean")
        weight_kl = block_config.get("weight_kl_weight", 0.0)
        if not _is_number(weight_kl) or float(weight_kl) < 0.0:
            raise ValueError("'model.enzyme_block_fusion.weight_kl_weight' must be non-negative")
        if float(weight_kl) > 0.0 and not block_config.get("learned_weights", False):
            raise ValueError("enzyme block weight KL requires learned_weights=true")
    if config.model.get("enzyme_input_mode") == "raw_mean_sleec_hyperbolic_text_gated":
        if not config.data.get("protein_text_vectors_path", None):
            raise ValueError(
                "data.protein_text_vectors_path is required when "
                "model.enzyme_input_mode='raw_mean_sleec_hyperbolic_text_gated'"
            )
        text_config = config.model.get("text_vector", {})
        text_dim = text_config.get("dim", config.model.get("text_vector_dim", 768))
        if not isinstance(text_dim, int) or text_dim <= 0:
            raise ValueError("'model.text_vector.dim' must be a positive integer")
        fusion_dim = text_config.get("fusion_dim", config.model.get("text_fusion_dim", 512))
        if not isinstance(fusion_dim, int) or fusion_dim <= 0:
            raise ValueError("'model.text_vector.fusion_dim' must be a positive integer")
        num_heads = text_config.get("num_heads", config.model.get("text_num_heads", 8))
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError("'model.text_vector.num_heads' must be a positive integer")
        if fusion_dim % num_heads != 0:
            raise ValueError("'model.text_vector.fusion_dim' must be divisible by num_heads")
        text_dropout = text_config.get("dropout", config.model.get("text_dropout", 0.1))
        if not _is_number(text_dropout) or not (0.0 <= float(text_dropout) <= 1.0):
            raise ValueError("'model.text_vector.dropout' must be in [0, 1]")
        missing_policy = text_config.get(
            "missing_policy",
            config.data.get("text_vector_missing_policy", "zero_with_mask"),
        )
        if missing_policy != "zero_with_mask":
            raise ValueError("'model.text_vector.missing_policy' must be 'zero_with_mask'")
    if config.model.get("enzyme_input_mode") == "raw_mean_sleec_biofp_split":
        biofp_config = config.model.get("biofp", {})
        seq_dim = biofp_config.get("seq_dim", 384)
        bio_dim = biofp_config.get("dim", 128)
        hidden_dim = biofp_config.get("hidden_dim", 512)
        for name, value in {
            "seq_dim": seq_dim,
            "dim": bio_dim,
            "hidden_dim": hidden_dim,
        }.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"'model.biofp.{name}' must be a positive integer")
        if seq_dim + bio_dim != config.model.embedding_dim:
            raise ValueError("'model.biofp.seq_dim' + 'model.biofp.dim' must equal embedding_dim")
        for name in ("center_dim", "cofactor_dim", "transition_dim"):
            value = biofp_config.get(name, 0)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"'model.biofp.{name}' must be a non-negative integer")
        vocab_path = config.data.get("protein_biofp_vocab_path", None)
        if vocab_path is not None and Path(vocab_path).exists():
            with Path(vocab_path).open() as handle:
                vocab_payload = json.load(handle)
            families = vocab_payload.get("families", {})
            if isinstance(families, dict):
                expected_vocab_dims = {
                    "center": biofp_config.get("center_dim", 0),
                    "cofactor": biofp_config.get("cofactor_dim", 0),
                    "transition": biofp_config.get("transition_dim", 0),
                }
                for family, expected_dim in expected_vocab_dims.items():
                    labels = families.get(family)
                    if labels is None:
                        continue
                    if not isinstance(labels, list):
                        raise ValueError(
                            f"'data.protein_biofp_vocab_path' family '{family}' must be a list"
                        )
                    if len(labels) != expected_dim:
                        raise ValueError(
                            f"'model.biofp.{family}_dim' ({expected_dim}) must match "
                            f"{vocab_path} family '{family}' size ({len(labels)})"
                        )
        seq_weight = biofp_config.get("seq_weight", 0.75)
        if not _is_number(seq_weight) or not (0.0 <= float(seq_weight) <= 1.0):
            raise ValueError("'model.biofp.seq_weight' must be in [0, 1]")
        dropout = biofp_config.get("dropout", 0.1)
        if not _is_number(dropout) or not (0.0 <= float(dropout) <= 1.0):
            raise ValueError("'model.biofp.dropout' must be in [0, 1]")
        missing_policy = biofp_config.get(
            "missing_policy",
            config.data.get("biofp_missing_policy", "zero_with_mask"),
        )
        if missing_policy != "zero_with_mask":
            raise ValueError("'model.biofp.missing_policy' must be 'zero_with_mask'")
        loss_config = config.training.get("loss", {})
        biofp_aux_weight = loss_config.get("biofp_aux_weight", 0.0)
        warmup = loss_config.get("biofp_aux_warmup_epochs", 0)
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
            raise ValueError("'training.loss.biofp_aux_warmup_epochs' must be a non-negative integer")
        if not isinstance(loss_config.get("biofp_normalize_active_families", False), bool):
            raise ValueError("'training.loss.biofp_normalize_active_families' must be boolean")
        if not _is_number(biofp_aux_weight) or float(biofp_aux_weight) < 0:
            raise ValueError("'training.loss.biofp_aux_weight' must be non-negative")
        if float(biofp_aux_weight) > 0 and not config.data.get("protein_biofp_targets_path", None):
            raise ValueError(
                "data.protein_biofp_targets_path is required when "
                "training.loss.biofp_aux_weight > 0"
            )
        confidence_cap = loss_config.get("biofp_confidence_cap", 8.0)
        if not _is_number(confidence_cap) or float(confidence_cap) <= 0:
            raise ValueError("'training.loss.biofp_confidence_cap' must be positive")


def _validate_cross_tower_alignment(config: DotDict) -> None:
    loss_config = config.training.get("loss", {})
    alignment_weight = loss_config.get("cross_tower_alignment_weight", 0.0)
    if not _is_number(alignment_weight) or float(alignment_weight) < 0:
        raise ValueError("'training.loss.cross_tower_alignment_weight' must be non-negative")
    alignment_family_weights = loss_config.get("cross_tower_alignment_family_weights", {})
    if not isinstance(alignment_family_weights, dict):
        raise ValueError("'training.loss.cross_tower_alignment_family_weights' must be a mapping")
    for family, weight in alignment_family_weights.items():
        if not _is_number(weight) or float(weight) < 0:
            raise ValueError(
                "'training.loss.cross_tower_alignment_family_weights."
                f"{family}' must be non-negative"
            )
    if float(alignment_weight) > 0:
        if config.model.get("enzyme_input_mode") != "raw_mean_sleec_biological_factorized":
            raise ValueError(
                "training.loss.cross_tower_alignment_weight > 0 requires "
                "model.enzyme_input_mode='raw_mean_sleec_biological_factorized'"
            )
        if not config.data.get("protein_biofp_targets_path", None):
            raise ValueError(
                "data.protein_biofp_targets_path is required when "
                "training.loss.cross_tower_alignment_weight > 0"
            )
        block_dims = config.model.get("enzyme_block_fusion", {}).get("dims", {})
        biological_families = set(block_dims) - {"core", "site", "ec"}
        unknown_families = set(alignment_family_weights) - biological_families
        if unknown_families:
            raise ValueError(
                "cross-tower alignment families have no matching enzyme block: "
                f"{sorted(unknown_families)}"
            )
        family_dims = config.model.get("biofp", {}).get("family_dims", {})
        if not isinstance(family_dims, dict):
            raise ValueError("'model.biofp.family_dims' must be a mapping")
        supervised_weight = sum(
            float(weight)
            for family, weight in alignment_family_weights.items()
            if isinstance(family_dims.get(family), int) and family_dims[family] > 0
        )
        if supervised_weight <= 0:
            raise ValueError(
                "cross-tower factor alignment requires at least one positive-weight "
                "family with annotation targets"
            )


def _validate_reaction_attention(config: DotDict) -> None:
    if "reaction_pooling" in config.model and config.model.reaction_pooling not in {
        "attention",
        "mean",
        "interaction",
    }:
        raise ValueError("'model.reaction_pooling' must be one of: attention, mean, interaction")
    if (config.model.get("reaction_pooling") == "interaction" and
        config.model.get("query_encoder_type") != "multimodal_reaction_attention"):
        raise ValueError("interaction pooling requires query_encoder_type=multimodal_reaction_attention")
    reaction_attention_config = config.model.get("reaction_multimodal_attention", {})
    if reaction_attention_config:
        if "hidden_dim" in reaction_attention_config:
            value = reaction_attention_config.hidden_dim
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    "'model.reaction_multimodal_attention.hidden_dim' must be a positive integer"
                )
        if "modality_encoder_num_layers" in reaction_attention_config:
            value = reaction_attention_config.modality_encoder_num_layers
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    "'model.reaction_multimodal_attention.modality_encoder_num_layers' "
                    "must be a non-negative integer"
                )
        for key in (
            "dropout",
            "modality_dropout",
            "chemistry_dropout",
            "modality_encoder_dropout",
        ):
            if key in reaction_attention_config:
                value = reaction_attention_config[key]
                if not _is_number(value) or not (0.0 <= float(value) <= 1.0):
                    raise ValueError(
                        f"'model.reaction_multimodal_attention.{key}' must be in [0, 1]"
                    )
        for key in (
            "token_layer_norm",
            "token_normalization",
            "modality_encoder_use_layer_norm",
            "modality_encoder_layer_norm",
            "modality_encoder_normalise_output",
            "modality_l2_normalize",
        ):
            if key in reaction_attention_config and not isinstance(
                reaction_attention_config[key],
                bool,
            ):
                raise ValueError(f"'model.reaction_multimodal_attention.{key}' must be a boolean")
        if "modality_encoder_widths" in reaction_attention_config:
            widths = reaction_attention_config.modality_encoder_widths
            if isinstance(widths, int):
                if widths <= 0:
                    raise ValueError(
                        "'model.reaction_multimodal_attention.modality_encoder_widths' "
                        "must be positive"
                    )
            elif isinstance(widths, list):
                if not widths or any(not isinstance(width, int) or width <= 0 for width in widths):
                    raise ValueError(
                        "'model.reaction_multimodal_attention.modality_encoder_widths' "
                        "must be a non-empty list of positive integers"
                    )
            else:
                raise ValueError(
                    "'model.reaction_multimodal_attention.modality_encoder_widths' "
                    "must be an integer or list of integers"
                )
        if reaction_attention_config.get("side_composition", "directional_delta") not in {
            "directional_delta",
            "molecule_set",
            "signed_residual",
        }:
            raise ValueError(
                "'model.reaction_multimodal_attention.side_composition' must be "
                "directional_delta, molecule_set or signed_residual"
            )
        if reaction_attention_config.get("fusion", "attention") not in {
            "attention",
            "mean",
            "factorized_concat",
            "prior_bounded_attention",
            "concat",
            "gated_attention_concat",
            "feature_gate",
            "scalar_gate",
        }:
            raise ValueError(
                "'model.reaction_multimodal_attention.fusion' must be attention, mean, "
                "factorized_concat, prior_bounded_attention, concat, gated_attention_concat, "
                "feature_gate, or scalar_gate"
            )
        if reaction_attention_config.get("fusion") in {"concat", "gated_attention_concat"}:
            if reaction_attention_config.get("output_projection", "mlp") != "mlp":
                raise ValueError("concat fusions require an MLP output projection")
        if reaction_attention_config.get("fusion") == "gated_attention_concat":
            if config.model.query_encoder_dims[0] % 4:
                raise ValueError("gated_attention_concat requires token dimension divisible by four")
        if reaction_attention_config.get("output_projection", "mlp") not in {
            "mlp",
            "residual_mlp",
            "identity",
        }:
            raise ValueError(
                "'model.reaction_multimodal_attention.output_projection' must be "
                "mlp, residual_mlp, or identity"
            )
        residual_gate_init = reaction_attention_config.get("residual_gate_init", 0.1)
        if not _is_number(residual_gate_init) or not 0.0 < float(residual_gate_init) < 1.0:
            raise ValueError(
                "'model.reaction_multimodal_attention.residual_gate_init' must be in (0, 1)"
            )
        directional_gate_init = reaction_attention_config.get("directional_gate_init", 0.1)
        if not _is_number(directional_gate_init) or not 0.0 < float(directional_gate_init) < 1.0:
            raise ValueError(
                "'model.reaction_multimodal_attention.directional_gate_init' must be in (0, 1)"
            )
        symmetric_blocks = reaction_attention_config.get("symmetric_output_blocks", {})
        if not isinstance(symmetric_blocks, dict):
            raise ValueError(
                "'model.reaction_multimodal_attention.symmetric_output_blocks' " "must be a mapping"
            )
        if "enabled" in symmetric_blocks and not isinstance(symmetric_blocks.enabled, bool):
            raise ValueError(
                "'model.reaction_multimodal_attention.symmetric_output_blocks.enabled' "
                "must be boolean"
            )
        symmetric_block_dropout = symmetric_blocks.get("dropout", 0.0)
        if not _is_number(symmetric_block_dropout) or not (
            0.0 <= float(symmetric_block_dropout) < 1.0
        ):
            raise ValueError(
                "'model.reaction_multimodal_attention.symmetric_output_blocks.dropout' "
                "must be in [0, 1)"
            )
        if symmetric_blocks.get("enabled", False):
            if config.model.get("query_encoder_type") != "multimodal_reaction_attention":
                raise ValueError(
                    "symmetric reaction output blocks require "
                    "model.query_encoder_type=multimodal_reaction_attention"
                )
            if config.model.get("enzyme_input_mode") != "raw_mean_sleec_biological_factorized":
                raise ValueError(
                    "symmetric reaction output blocks require "
                    "model.enzyme_input_mode=raw_mean_sleec_biological_factorized"
                )
            enzyme_blocks = config.model.get("enzyme_block_fusion", {})
            if not isinstance(enzyme_blocks.get("dims"), dict) or not isinstance(
                enzyme_blocks.get("weights"),
                dict,
            ):
                raise ValueError(
                    "symmetric reaction output blocks require explicit "
                    "model.enzyme_block_fusion dimensions and weights"
                )
        reaction_fusion = reaction_attention_config.get("fusion", "attention")
        if reaction_fusion == "factorized_concat":
            dims = reaction_attention_config.get("factorized_dims")
            weights = reaction_attention_config.get("factorized_weights")
            if not isinstance(dims, dict) or not isinstance(weights, dict):
                raise ValueError(
                    f"{reaction_fusion} requires factorized_dims and factorized_weights mappings"
                )
            if set(dims) != set(weights):
                raise ValueError("reaction factorized dimension/weight names must match")
            if any(not isinstance(value, int) or value <= 0 for value in dims.values()):
                raise ValueError("reaction factorized dimensions must be positive integers")
            if sum(dims.values()) != int(config.model.embedding_dim):
                raise ValueError("reaction factorized dimensions must sum to model.embedding_dim")
            if any(not _is_number(value) or float(value) <= 0 for value in weights.values()):
                raise ValueError("reaction factorized weights must be positive numbers")
            if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-6:
                raise ValueError("reaction factorized weights must sum to 1")
        if reaction_fusion == "prior_bounded_attention":
            weights = reaction_attention_config.get("attention_prior_weights")
            if not isinstance(weights, dict) or not weights:
                raise ValueError("prior_bounded_attention requires attention_prior_weights mapping")
            if any(not _is_number(value) or float(value) <= 0 for value in weights.values()):
                raise ValueError("reaction attention prior weights must be positive numbers")
            if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-6:
                raise ValueError("reaction attention prior weights must sum to 1")
            adaptation_strength = reaction_attention_config.get(
                "attention_adaptation_strength",
                0.4,
            )
            if not _is_number(adaptation_strength) or not 0.0 <= float(adaptation_strength) < 1.0:
                raise ValueError(
                    "'model.reaction_multimodal_attention.attention_adaptation_strength' "
                    "must be in [0, 1)"
                )
    reaction_attention_regularization = config.training.get(
        "reaction_attention_regularization",
        {},
    )
    if reaction_attention_regularization:
        weight = reaction_attention_regularization.get("weight", 0.0)
        entropy_floor = reaction_attention_regularization.get(
            "min_normalized_entropy",
            0.75,
        )
        if not _is_number(weight) or float(weight) < 0.0:
            raise ValueError(
                "'training.reaction_attention_regularization.weight' must be non-negative"
            )
        if not _is_number(entropy_floor) or not 0.0 <= float(entropy_floor) <= 1.0:
            raise ValueError(
                "'training.reaction_attention_regularization.min_normalized_entropy' "
                "must be in [0, 1]"
            )
    reaction_chemistry_consistency = config.training.get(
        "reaction_chemistry_consistency",
        {},
    )
    if "reaction_chemistry_consistency" in config.training:
        if not isinstance(reaction_chemistry_consistency, dict):
            raise ValueError("'training.reaction_chemistry_consistency' must be a mapping")
        weight = reaction_chemistry_consistency.get("weight", 0.0)
        if not _is_number(weight) or float(weight) < 0.0:
            raise ValueError(
                "'training.reaction_chemistry_consistency.weight' must be non-negative"
            )
        if float(weight) > 0.0:
            if not config.data.get("reaction_use_chemistry", False):
                raise ValueError(
                    "reaction chemistry consistency requires " "data.reaction_use_chemistry=true"
                )
            if config.model.get("query_encoder_type") != "multimodal_reaction_attention":
                raise ValueError(
                    "reaction chemistry consistency requires "
                    "model.query_encoder_type=multimodal_reaction_attention"
                )
    reaction_geometry = config.training.get("reaction_geometry", {})
    biological_geometry = config.training.get('biological_geometry', {})
    if not isinstance(biological_geometry, dict):
        raise ValueError('training.biological_geometry must be a mapping')
    biology_weight = biological_geometry.get('weight', 0.)
    if not _is_number(biology_weight) or not math.isfinite(float(biology_weight)) or biology_weight < 0:
        raise ValueError('training.biological_geometry.weight must be finite and nonnegative')
    if biology_weight > 0 and not biological_geometry.get('path'):
        raise ValueError('Biological geometry requires a training-only annotation path')
    if biological_geometry.get('mode', 'attraction') not in ('attraction', 'relative'):
        raise ValueError('Biological geometry mode must be attraction or relative')
    biology_margin = biological_geometry.get('margin', .1)
    if not _is_number(biology_margin) or not math.isfinite(float(biology_margin)) or not 0 <= biology_margin <= 2:
        raise ValueError('Biological geometry margin must be between zero and two')
    protein_geometry = config.training.get("protein_geometry", {})
    if not isinstance(protein_geometry, dict):
        raise ValueError("'training.protein_geometry' must be a mapping")
    protein_geometry_weight = protein_geometry.get("weight", 0.0)
    if (not _is_number(protein_geometry_weight) or
            not math.isfinite(float(protein_geometry_weight)) or float(protein_geometry_weight) < 0):
        raise ValueError("'training.protein_geometry.weight' must be finite and non-negative")
    if not isinstance(reaction_geometry, dict):
        raise ValueError("'training.reaction_geometry' must be a mapping")
    geometry_weight = reaction_geometry.get("weight", 0.0)
    if not _is_number(geometry_weight) or not math.isfinite(float(geometry_weight)) or float(geometry_weight) < 0:
        raise ValueError("'training.reaction_geometry.weight' must be finite and non-negative")
    if float(geometry_weight) > 0:
        if not config.data.get("reaction_use_chemistry", False) or config.model.get("query_encoder_type") != "multimodal_reaction_attention":
            raise ValueError("Reaction geometry requires multimodal reaction chemistry inputs")
    reaction_residual_identity = config.training.get(
        "reaction_residual_identity",
        {},
    )
    if "reaction_residual_identity" in config.training:
        if not isinstance(reaction_residual_identity, dict):
            raise ValueError("'training.reaction_residual_identity' must be a mapping")
        weight = reaction_residual_identity.get("weight", 0.0)
        if not _is_number(weight) or float(weight) < 0.0:
            raise ValueError("'training.reaction_residual_identity.weight' must be non-negative")
        if float(weight) > 0.0:
            if config.model.get("query_encoder_type") != "multimodal_reaction_attention":
                raise ValueError(
                    "reaction residual identity requires "
                    "model.query_encoder_type=multimodal_reaction_attention"
                )
            if reaction_attention_config.get("output_projection") != "residual_mlp":
                raise ValueError(
                    "reaction residual identity requires "
                    "model.reaction_multimodal_attention.output_projection=residual_mlp"
                )


def _validate_sampling_and_cache(config: DotDict) -> None:
    if "reaction_representation" in config.data and config.data.reaction_representation not in {
        "fingerprint",
        "unimol2_attention",
        "hybrid_fingerprint_unimol2",
        "multimodal_reaction_attention",
    }:
        raise ValueError(
            "'data.reaction_representation' must be one of: fingerprint, "
            "unimol2_attention, hybrid_fingerprint_unimol2, "
            "multimodal_reaction_attention"
        )
    if "normalize_molecule_sets_as_self_reactions" in config.data and not isinstance(
        config.data.normalize_molecule_sets_as_self_reactions,
        bool,
    ):
        raise ValueError("'data.normalize_molecule_sets_as_self_reactions' must be a boolean")
    cached_enzyme_path = config.data.get("cached_enzyme_base_embeds_path", None)
    cached_train_reaction_path = config.data.get(
        "cached_train_reaction_base_embeds_path",
        None,
    )
    cached_validation_reaction_path = config.data.get(
        "cached_validation_reaction_base_embeds_path",
        None,
    )
    cached_paths = (
        cached_enzyme_path,
        cached_train_reaction_path,
        cached_validation_reaction_path,
    )
    if any(value is not None for value in cached_paths):
        if not isinstance(cached_enzyme_path, str) or not cached_enzyme_path.strip():
            raise ValueError(
                "'data.cached_enzyme_base_embeds_path' is required for cached training"
            )
        if (
            not isinstance(cached_train_reaction_path, str)
            or not cached_train_reaction_path.strip()
        ):
            raise ValueError(
                "'data.cached_train_reaction_base_embeds_path' is required for " "cached training"
            )
        if config.training.get("validation_enabled", True) and (
            not isinstance(cached_validation_reaction_path, str)
            or not cached_validation_reaction_path.strip()
        ):
            raise ValueError(
                "'data.cached_validation_reaction_base_embeds_path' is required "
                "when cached training has validation enabled"
            )
        if config.training.get("training_stage") not in {
            "prototype_only",
            "biological_residual",
        }:
            raise ValueError(
                "Cached base embeddings require training_stage=prototype_only or "
                "biological_residual"
            )
    train_sampler = config.data.get("train_sampler", {})
    if train_sampler:
        if not isinstance(train_sampler, dict):
            raise ValueError("'data.train_sampler' must be a mapping")
        if train_sampler.get("name", "shuffle") not in {
            "shuffle",
            "reaction_degree_balanced",
            "balanced_anchors",
            "enzyme_grouped",
            "hypergraph_grouped",
        }:
            raise ValueError(
                "'data.train_sampler.name' must be one of: shuffle, "
                "reaction_degree_balanced, balanced_anchors, enzyme_grouped, hypergraph_grouped"
            )
        exponent = train_sampler.get("degree_exponent", 0.5)
        if not _is_number(exponent) or float(exponent) < 0.0:
            raise ValueError("'data.train_sampler.degree_exponent' must be non-negative")
        sampler_seed = train_sampler.get("seed", config.get("seed", 42))
        if train_sampler.get("name") == "balanced_anchors":
            positives = train_sampler.get("positives_per_anchor", 4)
            if not isinstance(positives, int) or positives < 1:
                raise ValueError("Balanced positives_per_anchor must be a positive integer")
        if not isinstance(sampler_seed, int):
            raise ValueError("'data.train_sampler.seed' must be an integer")
        if train_sampler.get("name") in {"enzyme_grouped", "hypergraph_grouped"}:
            anchors = train_sampler.get("anchors_per_batch", 64)
            positives = train_sampler.get("positives_per_anchor", 4)
            if not isinstance(anchors, int) or anchors <= 0:
                raise ValueError(
                    "'data.train_sampler.anchors_per_batch' must be a positive integer"
                )
            if not isinstance(positives, int) or positives < 2:
                raise ValueError(
                    "'data.train_sampler.positives_per_anchor' must be an integer >= 2"
                )
            batch_size = config.data.get("train_batch_size", 64)
            if anchors * positives > batch_size:
                raise ValueError(
                    "grouped anchors_per_batch * positives_per_anchor must not "
                    "exceed data.train_batch_size"
                )
            if train_sampler.get("name") == "hypergraph_grouped":
                reaction_fraction = train_sampler.get("reaction_anchor_fraction", 0.5)
                if not _is_number(reaction_fraction) or not 0.0 <= float(reaction_fraction) <= 1.0:
                    raise ValueError(
                        "'data.train_sampler.reaction_anchor_fraction' must be in [0, 1]"
                    )
    source_replay = config.data.get("source_replay", {})
    if source_replay:
        if not isinstance(source_replay, dict):
            raise ValueError("'data.source_replay' must be a mapping")
        if source_replay.get("enabled", False):
            if not source_replay.get("config_path"):
                raise ValueError("Enabled source replay requires data.source_replay.config_path")
            replay_fraction = source_replay.get("fraction", 0.15)
            if not _is_number(replay_fraction) or not 0.0 < float(replay_fraction) < 1.0:
                raise ValueError("'data.source_replay.fraction' must lie in (0, 1)")


def _validate_reaction_features(config: DotDict) -> None:
    if config.data.get("reaction_representation", "fingerprint") == "unimol2_attention":
        if not _has_global_or_split_values(
            config.data,
            global_names=("reaction_embeds_path",),
            train_names=("train_reaction_embeds_path",),
            validation_names=("validation_reaction_embeds_path",),
        ):
            raise ValueError(
                "'data.reaction_embeds_path' or split-specific train/validation "
                "reaction embedding paths are required for unimol2_attention reactions"
            )
        if "reaction_unimol_dim" in config.data and not isinstance(
            config.data.reaction_unimol_dim,
            int,
        ):
            raise ValueError("'data.reaction_unimol_dim' must be an integer")
        if config.model.get("query_encoder_type", None) != "unimol2_reaction_attention":
            raise ValueError(
                "'model.query_encoder_type' must be 'unimol2_reaction_attention' "
                "when data.reaction_representation is 'unimol2_attention'"
            )
        expected_query_dim = 4 * config.data.get("reaction_unimol_dim", 768)
        if config.model.query_encoder_dims[0] != expected_query_dim:
            raise ValueError(
                f"'model.query_encoder_dims[0]' must be {expected_query_dim} "
                "for Uni-Mol2 reaction attention"
            )
    if config.data.get("reaction_representation", "fingerprint") == "hybrid_fingerprint_unimol2":
        if not _has_global_or_split_values(
            config.data,
            global_names=("reaction_embeds_path",),
            train_names=("train_reaction_embeds_path",),
            validation_names=("validation_reaction_embeds_path",),
        ):
            raise ValueError(
                "'data.reaction_embeds_path' or split-specific train/validation "
                "reaction embedding paths are required for hybrid_fingerprint_unimol2 reactions"
            )
        if "reaction_unimol_dim" in config.data and not isinstance(
            config.data.reaction_unimol_dim,
            int,
        ):
            raise ValueError("'data.reaction_unimol_dim' must be an integer")
        if config.model.get("query_encoder_type", None) != "hybrid_reaction":
            raise ValueError(
                "'model.query_encoder_type' must be 'hybrid_reaction' "
                "when data.reaction_representation is 'hybrid_fingerprint_unimol2'"
            )
        expected_query_dim = config.data.get("rdkit_fp_dim", 1024) + config.data.get(
            "drfp_dim", 1024
        )
        if config.model.query_encoder_dims[0] != expected_query_dim:
            raise ValueError(
                f"'model.query_encoder_dims[0]' must be {expected_query_dim} "
                "for hybrid fingerprint + Uni-Mol2 reactions"
            )
    if config.data.get("reaction_representation", "fingerprint") == "multimodal_reaction_attention":
        reaction_use_model = bool(config.data.get("reaction_use_model", True))
        if reaction_use_model and not _has_global_or_split_values(
            config.data,
            global_names=("reaction_t5v2_embeds_path", "reaction_model_embeds_path"),
            train_names=(
                "train_reaction_t5v2_embeds_path",
                "train_reaction_model_embeds_path",
            ),
            validation_names=(
                "validation_reaction_t5v2_embeds_path",
                "validation_reaction_model_embeds_path",
            ),
        ):
            raise ValueError(
                "'data.reaction_t5v2_embeds_path' or 'data.reaction_model_embeds_path' "
                "or split-specific train/validation reaction model embedding paths "
                "are required for multimodal_reaction_attention reactions"
            )
        if not _has_global_or_split_values(
            config.data,
            global_names=("reaction_unimol2_embeds_path", "reaction_embeds_path"),
            train_names=(
                "train_reaction_unimol2_embeds_path",
                "train_reaction_embeds_path",
            ),
            validation_names=(
                "validation_reaction_unimol2_embeds_path",
                "validation_reaction_embeds_path",
            ),
        ):
            raise ValueError(
                "'data.reaction_unimol2_embeds_path' or 'data.reaction_embeds_path' "
                "or split-specific train/validation UniMol2 embedding paths are "
                "required for multimodal_reaction_attention reactions"
            )
        reaction_use_chirality = config.data.get(
            "reaction_use_chiro",
            config.data.get(
                "reaction_use_chirality",
                config.data.get(
                    "reaction_use_chienn",
                    config.model.get(
                        "reaction_use_chiro",
                        config.model.get(
                            "reaction_use_chirality",
                            config.model.get("reaction_use_chienn", True),
                        ),
                    ),
                ),
            ),
        )
        if reaction_use_chirality and not _has_global_or_split_values(
            config.data,
            global_names=(
                "reaction_chiro_embeds_path",
                "reaction_chirality_embeds_path",
                "reaction_chienn_embeds_path",
            ),
            train_names=(
                "train_reaction_chiro_embeds_path",
                "train_reaction_chirality_embeds_path",
                "train_reaction_chienn_embeds_path",
            ),
            validation_names=(
                "validation_reaction_chiro_embeds_path",
                "validation_reaction_chirality_embeds_path",
                "validation_reaction_chienn_embeds_path",
            ),
        ):
            raise ValueError(
                "'data.reaction_chiro_embeds_path' or split-specific train/validation "
                "chirality embedding paths are required for multimodal_reaction_attention "
                "reactions when chirality is enabled"
            )
        dim_keys = ["reaction_unimol_dim"]
        if reaction_use_model:
            dim_keys.append("reaction_model_dim")
        if reaction_use_chirality:
            dim_keys.extend(["reaction_chiro_dim", "reaction_chirality_dim", "reaction_chienn_dim"])
        reaction_use_chemistry = bool(config.data.get("reaction_use_chemistry", False))
        if reaction_use_chemistry:
            dim_keys.append("reaction_chemistry_dim")
            if not _has_global_or_split_values(
                config.data,
                global_names=("reaction_chemistry_vectors_path",),
                train_names=("train_reaction_chemistry_vectors_path",),
                validation_names=("validation_reaction_chemistry_vectors_path",),
            ):
                raise ValueError(
                    "'data.reaction_chemistry_vectors_path' or split-specific "
                    "train/validation reaction chemistry vector paths are required "
                    "when data.reaction_use_chemistry=True"
                )
        reaction_use_directional = bool(config.data.get("reaction_use_directional", False))
        reaction_load_directional = bool(config.data.get("reaction_load_directional", False))
        if reaction_use_directional or reaction_load_directional:
            dim_keys.append("reaction_directional_dim")
            if not _has_global_or_split_values(
                config.data,
                global_names=("reaction_directional_vectors_path",),
                train_names=("train_reaction_directional_vectors_path",),
                validation_names=("validation_reaction_directional_vectors_path",),
            ):
                raise ValueError(
                    "global or split-specific reaction directional vector paths are required "
                    "when reaction directional vectors are enabled"
                )
            if config.data.get("reaction_directional_dim", 0) <= 0:
                raise ValueError("'data.reaction_directional_dim' must be a positive integer")
        for key in dim_keys:
            if key in config.data and not isinstance(config.data[key], int):
                raise ValueError(f"'data.{key}' must be an integer")
        if reaction_use_chemistry and config.data.get("reaction_chemistry_dim", 0) <= 0:
            raise ValueError("'data.reaction_chemistry_dim' must be a positive integer")
        if config.model.get("query_encoder_type", None) != "multimodal_reaction_attention":
            raise ValueError(
                "'model.query_encoder_type' must be 'multimodal_reaction_attention' "
                "when data.reaction_representation is 'multimodal_reaction_attention'"
            )
    if "query_encoder_type" in config.model and config.model.query_encoder_type not in {
        "mlp",
        "unimol2_reaction_attention",
        "hybrid_reaction",
        "multimodal_reaction_attention",
    }:
        raise ValueError(
            "'model.query_encoder_type' must be one of: mlp, "
            "unimol2_reaction_attention, hybrid_reaction, multimodal_reaction_attention"
        )


def _validate_residue_pooling(config: DotDict) -> None:
    if "protein_residue_embeds_path" in config.data:
        if "residue_dim" in config.data and not isinstance(config.data.residue_dim, int):
            raise ValueError("'data.residue_dim' must be an integer")
        if "protein_score_residue_embeds_path" in config.data:
            if not isinstance(config.data.protein_score_residue_embeds_path, str):
                raise ValueError("'data.protein_score_residue_embeds_path' must be a string")
            if "score_residue_dim" in config.data and not isinstance(
                config.data.score_residue_dim,
                int,
            ):
                raise ValueError("'data.score_residue_dim' must be an integer")
        if "max_protein_tokens" in config.data:
            max_tokens = config.data.max_protein_tokens
            if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
                raise ValueError("'data.max_protein_tokens' must be null or a positive integer")
        if "protein_truncation" in config.data and config.data.protein_truncation not in {
            "ends_center"
        }:
            raise ValueError("'data.protein_truncation' must be one of: ends_center")

    pooling_config = config.model.get("reaction_conditioned_pooling", {})
    if "attention_rank" in pooling_config:
        attention_rank = pooling_config.attention_rank
        if attention_rank is not None and (
            not isinstance(attention_rank, int) or attention_rank <= 0
        ):
            raise ValueError(
                "'model.reaction_conditioned_pooling.attention_rank' must be null or a "
                "positive integer"
            )
    if "value_projection_bias" in pooling_config and not isinstance(
        pooling_config.value_projection_bias, bool
    ):
        raise ValueError(
            "'model.reaction_conditioned_pooling.value_projection_bias' must be a boolean"
        )
    if "normalize_pooled_values" in pooling_config and not isinstance(
        pooling_config.normalize_pooled_values, bool
    ):
        raise ValueError(
            "'model.reaction_conditioned_pooling.normalize_pooled_values' must be a boolean"
        )

    protein_pooling_config = config.model.get("protein_attention_pooling", {})
    if "attention_bias" in protein_pooling_config and not isinstance(
        protein_pooling_config.attention_bias, bool
    ):
        raise ValueError("'model.protein_attention_pooling.attention_bias' must be a boolean")
    if "return_attention" in protein_pooling_config and not isinstance(
        protein_pooling_config.return_attention, bool
    ):
        raise ValueError("'model.protein_attention_pooling.return_attention' must be a boolean")

    sleec_pooling_config = config.model.get("sleec_pooling", {})
    if "mode" in sleec_pooling_config and sleec_pooling_config.mode not in {
        "topk",
        "soft",
        "threshold",
    }:
        raise ValueError("'model.sleec_pooling.mode' must be one of: topk, soft, threshold")
    if "topk_fraction" in sleec_pooling_config:
        value = sleec_pooling_config.topk_fraction
        if not _is_number(value) or not (0.0 < value <= 1.0):
            raise ValueError("'model.sleec_pooling.topk_fraction' must be in the range (0, 1]")
    if "threshold" in sleec_pooling_config and not _is_number(sleec_pooling_config.threshold):
        raise ValueError("'model.sleec_pooling.threshold' must be a number")
    if "scorer_hidden_dim" in sleec_pooling_config:
        value = sleec_pooling_config.scorer_hidden_dim
        if not isinstance(value, int) or value <= 0:
            raise ValueError("'model.sleec_pooling.scorer_hidden_dim' must be a positive integer")
    if "score_hidden_dim" in sleec_pooling_config:
        value = sleec_pooling_config.score_hidden_dim
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(
                "'model.sleec_pooling.score_hidden_dim' must be null or a positive integer"
            )
    if "checkpoint_path" in sleec_pooling_config and not isinstance(
        sleec_pooling_config.checkpoint_path,
        str,
    ):
        raise ValueError("'model.sleec_pooling.checkpoint_path' must be a string")
    if "freeze_scorer" in sleec_pooling_config and not isinstance(
        sleec_pooling_config.freeze_scorer,
        bool,
    ):
        raise ValueError("'model.sleec_pooling.freeze_scorer' must be a boolean")
    if "initial_bias_scale" in sleec_pooling_config:
        value = sleec_pooling_config.initial_bias_scale
        if not _is_number(value) or value <= 0:
            raise ValueError("'model.sleec_pooling.initial_bias_scale' must be positive")
    if "train_bias_scale" in sleec_pooling_config and not isinstance(
        sleec_pooling_config.train_bias_scale,
        bool,
    ):
        raise ValueError("'model.sleec_pooling.train_bias_scale' must be a boolean")
    if (
        "score_embedding_source" in sleec_pooling_config
        and sleec_pooling_config.score_embedding_source
        not in {
            "same",
            "external",
        }
    ):
        raise ValueError(
            "'model.sleec_pooling.score_embedding_source' must be one of: same, external"
        )
    if (
        config.model.get("pooling", None) in {"sleec", "sleec_guided_attention"}
        and sleec_pooling_config.get("score_embedding_source", "same") == "external"
    ):
        if "protein_score_residue_embeds_path" not in config.data:
            raise ValueError(
                "'data.protein_score_residue_embeds_path' is required when "
                "'model.sleec_pooling.score_embedding_source' is 'external'"
            )
        if "score_residue_dim" not in config.data:
            raise ValueError(
                "'data.score_residue_dim' is required when "
                "'model.sleec_pooling.score_embedding_source' is 'external'"
            )


def _validate_hyperbolic_and_fingerprint(config: DotDict) -> None:
    hyperbolic_config = config.model.get("hyperbolic_encoder", {})
    if "checkpoint_path" in hyperbolic_config and not isinstance(
        hyperbolic_config.checkpoint_path,
        str,
    ):
        raise ValueError("'model.hyperbolic_encoder.checkpoint_path' must be a string")
    for bool_key in (
        "freeze_projector",
        "use_tangent",
        "load_attention_pooler",
        "freeze_attention_pooler",
    ):
        if bool_key in hyperbolic_config and not isinstance(
            hyperbolic_config[bool_key],
            bool,
        ):
            raise ValueError(f"'model.hyperbolic_encoder.{bool_key}' must be a boolean")
    if "hyp_dim" in hyperbolic_config:
        value = hyperbolic_config.hyp_dim
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError("'model.hyperbolic_encoder.hyp_dim' must be a positive integer")
    enzyme_input_mode = config.model.get("enzyme_input_mode", "standard")
    if enzyme_input_mode in {
        "raw_sleec_hyperbolic_concat",
        "raw_mean_sleec_hyperbolic_gated",
        "raw_mean_sleec_hyperbolic_capability_gated",
        "raw_mean_sleec_hyperbolic_text_gated",
        "raw_mean_sleec_hyperbolic_blockwise",
        "raw_mean_sleec_hyperbolic_capability_blockwise",
    }:
        if not hyperbolic_config.get("checkpoint_path", None):
            raise ValueError(
                f"'model.enzyme_input_mode: {enzyme_input_mode}' requires "
                "'model.hyperbolic_encoder.checkpoint_path'"
            )
        if not hyperbolic_config.get("use_tangent", True):
            raise ValueError(
                f"'model.enzyme_input_mode: {enzyme_input_mode}' requires "
                "'model.hyperbolic_encoder.use_tangent: true'"
            )
        enzyme_fusion_config = config.model.get("enzyme_fusion", {})
        if "hidden_dim" in enzyme_fusion_config:
            hidden_dim = enzyme_fusion_config.hidden_dim
            if hidden_dim is not None and (not isinstance(hidden_dim, int) or hidden_dim <= 0):
                raise ValueError(
                    "'model.enzyme_fusion.hidden_dim' must be a positive integer or null"
                )
        if "dropout" in enzyme_fusion_config:
            dropout = enzyme_fusion_config.dropout
            if not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) <= 1.0:
                raise ValueError("'model.enzyme_fusion.dropout' must be a number in [0, 1]")
        hyp_dim = hyperbolic_config.get("hyp_dim", None)
        if enzyme_input_mode == "raw_sleec_hyperbolic_concat" and hyp_dim is not None:
            expected_target_dim = config.data.get("residue_dim", 1024) + hyp_dim
            if config.model.target_encoder_dims[0] != expected_target_dim:
                raise ValueError(
                    f"'model.target_encoder_dims[0]' must be {expected_target_dim} "
                    "for raw_sleec_hyperbolic_concat enzyme input"
                )

    reaction_hyperbolic_config = config.model.get("reaction_hyperbolic_encoder", {})
    if (
        "checkpoint_path" in reaction_hyperbolic_config
        and reaction_hyperbolic_config.checkpoint_path is not None
        and not isinstance(reaction_hyperbolic_config.checkpoint_path, str)
    ):
        raise ValueError(
            "'model.reaction_hyperbolic_encoder.checkpoint_path' must be a string or null"
        )
    for bool_key in ("freeze_encoder", "freeze_projector", "use_tangent"):
        if bool_key in reaction_hyperbolic_config and not isinstance(
            reaction_hyperbolic_config[bool_key],
            bool,
        ):
            raise ValueError(f"'model.reaction_hyperbolic_encoder.{bool_key}' must be a boolean")
    if "hyp_dim" in reaction_hyperbolic_config:
        value = reaction_hyperbolic_config.hyp_dim
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(
                "'model.reaction_hyperbolic_encoder.hyp_dim' must be a positive integer"
            )

    reaction_fingerprint_attention_config = config.model.get(
        "reaction_fingerprint_attention",
        {},
    )
    if "enabled" in reaction_fingerprint_attention_config and not isinstance(
        reaction_fingerprint_attention_config.enabled,
        bool,
    ):
        raise ValueError("'model.reaction_fingerprint_attention.enabled' must be a boolean")
    if bool(reaction_fingerprint_attention_config.get("enabled", False)):
        if reaction_hyperbolic_config.get("checkpoint_path", None):
            raise ValueError(
                "'model.reaction_fingerprint_attention.enabled' cannot be true when "
                "'model.reaction_hyperbolic_encoder.checkpoint_path' is set"
            )
        if config.data.get("reaction_representation", "fingerprint") != "fingerprint":
            raise ValueError(
                "'model.reaction_fingerprint_attention.enabled' requires "
                "'data.reaction_representation: fingerprint'"
            )
        expected_query_dim = reaction_fingerprint_attention_config.get("token_dim", 512)
        if config.model.query_encoder_dims[0] != expected_query_dim:
            raise ValueError(
                f"'model.query_encoder_dims[0]' must be {expected_query_dim} when "
                "'model.reaction_fingerprint_attention.enabled' is true"
            )
    for int_key in ("token_dim", "hidden_dim"):
        if int_key in reaction_fingerprint_attention_config:
            value = reaction_fingerprint_attention_config[int_key]
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"'model.reaction_fingerprint_attention.{int_key}' must be a positive integer"
                )
    if "dropout" in reaction_fingerprint_attention_config:
        value = reaction_fingerprint_attention_config.dropout
        if not _is_number(value) or not (0.0 <= value <= 1.0):
            raise ValueError(
                "'model.reaction_fingerprint_attention.dropout' must be in the range [0, 1]"
            )
    if "attention_bias" in reaction_fingerprint_attention_config and not isinstance(
        reaction_fingerprint_attention_config.attention_bias,
        bool,
    ):
        raise ValueError("'model.reaction_fingerprint_attention.attention_bias' must be a boolean")


def _validate_logging_and_precision(config: DotDict) -> None:
    if "lambda_residue" in config.training:
        value = config.training.lambda_residue
        if not _is_number(value) or value < 0:
            raise ValueError("'training.lambda_residue' must be a non-negative number")

    if "precision" in config.training and not isinstance(config.training.precision, (str, int)):
        raise ValueError("'training.precision' must be a string or integer")
    if (
        "float32_matmul_precision" in config.training
        and config.training.float32_matmul_precision
        not in {
            "highest",
            "high",
            "medium",
        }
    ):
        raise ValueError(
            "'training.float32_matmul_precision' must be one of: highest, high, medium"
        )

    if "logging" in config and "wandb" in config.logging:
        wandb_config = config.logging.wandb
        if "enabled" in wandb_config and not isinstance(wandb_config.enabled, bool):
            raise ValueError("'logging.wandb.enabled' must be a boolean")
        if "mode" in wandb_config and wandb_config.mode not in {"online", "offline", "disabled"}:
            raise ValueError("'logging.wandb.mode' must be one of: online, offline, disabled")
        if "tags" in wandb_config and not isinstance(wandb_config.tags, list):
            raise ValueError("'logging.wandb.tags' must be a list")
        if "log_model" in wandb_config and not isinstance(wandb_config.log_model, bool):
            raise ValueError("'logging.wandb.log_model' must be a boolean")
