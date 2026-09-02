"""
Configuration system for Horizyn.

This module provides a simple configuration system for loading and validating
YAML config files. It supports dot-notation access and command-line overrides.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class DotDict(dict):
    """A dictionary that supports dot notation access.

    Example:
        >>> config = DotDict({'model': {'layers': 3}})
        >>> config.model.layers  # Returns 3
    """

    def __init__(self, *args, **kwargs):
        """Initialize without mutating caller-owned mappings."""
        data: dict[str, Any] = {}
        for arg in args:
            if not isinstance(arg, dict):
                raise TypeError(f"DotDict expected a mapping, got {type(arg).__name__}")
            data.update(copy.deepcopy(arg))
        data.update(copy.deepcopy(kwargs))
        super().__init__()
        for key, value in data.items():
            super().__setitem__(key, self._convert(value))

    @classmethod
    def _convert(cls, value: Any) -> Any:
        if isinstance(value, DotDict):
            return DotDict(value)
        if isinstance(value, dict):
            return DotDict(value)
        if isinstance(value, list):
            return [cls._convert(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._convert(item) for item in value)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, self._convert(value))

    def update(self, *args, **kwargs) -> None:
        incoming = dict(*args, **kwargs)
        for key, value in incoming.items():
            self[key] = value

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]

    def __ior__(self, other):
        self.update(other)
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute using dot notation."""
        if isinstance(name, str):
            self[name] = value
        else:
            raise TypeError(f"attribute name must be string, not '{type(name).__name__}'")

    def __getattr__(self, name: str) -> Any:
        """Get attribute using dot notation."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from None

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default fallback."""
        try:
            return self[key]
        except KeyError:
            return default


def load_config(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None,
    validate: bool = True,
) -> DotDict:
    """
    Load a YAML configuration file and apply overrides.

    Args:
        config_path: Path to YAML config file.
        overrides: Dictionary of config overrides (supports dot notation keys).
        validate: Whether to validate the config structure.

    Returns:
        Loaded and validated configuration as a DotDict.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config validation fails.

    Example:
        >>> config = load_config('configs/sota.yaml')
        >>> config = load_config('configs/sota.yaml', {'training.max_epochs': 50})
    """
    # Load YAML file
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Make sure the path is correct relative to the working directory."
        )

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        raise ValueError(f"Config file is empty: {config_path}")

    # Convert to DotDict
    config = DotDict(config_dict)
    if "data" in config and "reaction_representation" not in config.data:
        config.data.reaction_representation = "fingerprint"

    # Apply overrides
    if overrides:
        config = apply_overrides(config, overrides)

    # Validate config structure
    if validate:
        validate_config(config)

    return config


def apply_overrides(config: DotDict, overrides: Dict[str, Any]) -> DotDict:
    """
    Apply command-line overrides to config.

    Supports dot notation for nested keys:
        {'training.max_epochs': 50} -> config.training.max_epochs = 50

    Args:
        config: Base configuration.
        overrides: Dictionary of overrides with dot notation keys.

    Returns:
        The same configuration object, updated in place.  This preserves the
        public API used by command-line and integration callers.
    """
    result = config
    for key, value in overrides.items():
        keys = key.split(".")
        current = result

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = DotDict()
            elif not isinstance(current[k], (dict, DotDict)):
                value_type = type(current[k]).__name__
                raise ValueError(f"Cannot override '{key}': '{k}' is not a dict (got {value_type})")
            current = current[k]

        # Set the final value
        current[keys[-1]] = value

    return result


def apply_overrides_copy(config: DotDict, overrides: Dict[str, Any]) -> DotDict:
    """Return an independently copied configuration with overrides applied."""

    return apply_overrides(DotDict(config), overrides)


def _has_any_config_value(config: DotDict, names: tuple[str, ...]) -> bool:
    return any(config.get(name, None) for name in names)


def _has_global_or_split_values(
    config: DotDict,
    *,
    global_names: tuple[str, ...],
    train_names: tuple[str, ...],
    validation_names: tuple[str, ...],
) -> bool:
    return _has_any_config_value(config, global_names) or (
        _has_any_config_value(config, train_names)
        and _has_any_config_value(config, validation_names)
    )


def validate_config(config: DotDict) -> None:
    """
    Validate the configuration structure.

    Ensures all required sections and parameters are present with correct types.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If validation fails with a helpful error message.
    """
    # Check required top-level sections
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
    for key in ("protein_capability_vectors_path", "protein_capability_metadata_path"):
        if (
            key in config.data
            and config.data[key] is not None
            and not isinstance(config.data[key], str)
        ):
            raise ValueError(f"'data.{key}' must be a string path or null")
    for key in (
        "protein_factorized_capability_vectors_path",
        "reaction_chemistry_vectors_path",
        "train_reaction_chemistry_vectors_path",
        "validation_reaction_chemistry_vectors_path",
        "reaction_directional_vectors_path",
        "train_reaction_directional_vectors_path",
        "validation_reaction_directional_vectors_path",
    ):
        if (
            key in config.data
            and config.data[key] is not None
            and not isinstance(config.data[key], str)
        ):
            raise ValueError(f"'data.{key}' must be a string path or null")
    for key in ("protein_text_vectors_path", "protein_text_metadata_path"):
        if (
            key in config.data
            and config.data[key] is not None
            and not isinstance(config.data[key], str)
        ):
            raise ValueError(f"'data.{key}' must be a string path or null")
    for key in ("protein_biofp_targets_path", "protein_biofp_vocab_path"):
        if (
            key in config.data
            and config.data[key] is not None
            and not isinstance(config.data[key], str)
        ):
            raise ValueError(f"'data.{key}' must be a string path or null")
    if (
        "capability_missing_policy" in config.data
        and config.data.capability_missing_policy not in {"zero_with_mask"}
    ):
        raise ValueError("'data.capability_missing_policy' must be 'zero_with_mask'")
    if (
        "factorized_capability_missing_policy" in config.data
        and config.data.factorized_capability_missing_policy not in {"zero_with_mask"}
    ):
        raise ValueError("'data.factorized_capability_missing_policy' must be 'zero_with_mask'")
    if (
        "text_vector_missing_policy" in config.data
        and config.data.text_vector_missing_policy not in {"zero_with_mask"}
    ):
        raise ValueError("'data.text_vector_missing_policy' must be 'zero_with_mask'")
    if "biofp_missing_policy" in config.data and config.data.biofp_missing_policy not in {
        "zero_with_mask"
    }:
        raise ValueError("'data.biofp_missing_policy' must be 'zero_with_mask'")
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

    # Validate model section
    required_model_keys = ["query_encoder_dims", "target_encoder_dims", "embedding_dim"]
    for key in required_model_keys:
        if key not in config.model:
            raise ValueError(
                f"Missing required model config parameter: 'model.{key}'\n"
                f"Required model parameters: {required_model_keys}"
            )

    # Validate training section
    if "max_epochs" not in config.training:
        raise ValueError("Missing required training parameter: 'training.max_epochs'")

    # Type validation
    if not isinstance(config.training.max_epochs, int):
        raise ValueError(
            f"'training.max_epochs' must be an integer, got {type(config.training.max_epochs).__name__}"
        )
    if "training_stage" in config.training and config.training.training_stage not in {
        "joint",
        "enzyme_only_tuning",
        "joint_capability_retrieval",
        "e2r_adapter",
        "r2e_adapter",
        "bidirectional_adapters",
    }:
        raise ValueError(
            "'training.training_stage' must be one of: joint, "
            "enzyme_only_tuning, joint_capability_retrieval, e2r_adapter, "
            "r2e_adapter, bidirectional_adapters"
        )
    if "validation_interval_steps" in config.training:
        value = config.training.validation_interval_steps
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError("'training.validation_interval_steps' must be a positive integer")
    if "save_every_n_train_steps" in config.get("logging", {}):
        value = config.logging.save_every_n_train_steps
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError("'logging.save_every_n_train_steps' must be a positive integer")

    if not isinstance(config.model.query_encoder_dims, list):
        raise ValueError(
            f"'model.query_encoder_dims' must be a list, got {type(config.model.query_encoder_dims).__name__}"
        )

    if not isinstance(config.model.target_encoder_dims, list):
        raise ValueError(
            f"'model.target_encoder_dims' must be a list, got {type(config.model.target_encoder_dims).__name__}"
        )

    if not isinstance(config.model.embedding_dim, int):
        raise ValueError(
            f"'model.embedding_dim' must be an integer, got {type(config.model.embedding_dim).__name__}"
        )
    if (
        "query_encoder_checkpoint_path" in config.model
        and config.model.query_encoder_checkpoint_path is not None
        and not isinstance(config.model.query_encoder_checkpoint_path, str)
    ):
        raise ValueError("'model.query_encoder_checkpoint_path' must be a string path or null")
    e2r_adapter = config.model.get("e2r_adapter", {})
    if not isinstance(e2r_adapter, dict):
        raise ValueError("'model.e2r_adapter' must be a mapping")
    if "enabled" in e2r_adapter and not isinstance(e2r_adapter.enabled, bool):
        raise ValueError("'model.e2r_adapter.enabled' must be a boolean")
    for key in ("use_factorized_inputs", "use_directional_inputs"):
        if key in e2r_adapter and not isinstance(e2r_adapter[key], bool):
            raise ValueError(f"'model.e2r_adapter.{key}' must be a boolean")
    for key in ("hidden_dim", "directional_dim", "directional_hidden_dim"):
        if key in e2r_adapter:
            value = e2r_adapter[key]
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"'model.e2r_adapter.{key}' must be a positive integer")
    for key in ("dropout", "gate_init"):
        if key in e2r_adapter:
            value = e2r_adapter[key]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"'model.e2r_adapter.{key}' must be in [0, 1]")
    if "gate_init" in e2r_adapter and not 0.0 < float(e2r_adapter.gate_init) < 1.0:
        raise ValueError("'model.e2r_adapter.gate_init' must be in (0, 1)")
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
    if not isinstance(r2e_adapter, dict):
        raise ValueError("'model.r2e_adapter' must be a mapping")
    if "enabled" in r2e_adapter and not isinstance(r2e_adapter.enabled, bool):
        raise ValueError("'model.r2e_adapter.enabled' must be a boolean")
    if "use_factorized_inputs" in r2e_adapter and not isinstance(
        r2e_adapter.use_factorized_inputs,
        bool,
    ):
        raise ValueError("'model.r2e_adapter.use_factorized_inputs' must be a boolean")
    for key in ("hidden_dim",):
        if key in r2e_adapter:
            value = r2e_adapter[key]
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"'model.r2e_adapter.{key}' must be a positive integer")
    for key in ("dropout", "gate_init"):
        if key in r2e_adapter:
            value = r2e_adapter[key]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"'model.r2e_adapter.{key}' must be in [0, 1]")
    if "gate_init" in r2e_adapter and not 0.0 < float(r2e_adapter.gate_init) < 1.0:
        raise ValueError("'model.r2e_adapter.gate_init' must be in (0, 1)")
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

    loss_config = config.training.get("loss", {})
    loss_name = loss_config.get("name", "FullBatchMLNCELoss")
    valid_loss_names = {
        "FullBatchMLNCELoss",
        "DegreeTemperedFullBatchMLNCELoss",
        "DecoupledAllPositiveInfoNCELoss",
        "HybridCardinalityRetrievalLoss",
        "BalancedSigmoidEBMLoss",
        "HorizynFGWLoss",
        "BidirectionalAnchorBalancedSupConLoss",
        "MultiAlignmentRetrievalLoss",
        "mlnce",
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

    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

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
    }:
        raise ValueError(
            "'model.enzyme_input_mode' must be one of: standard, "
            "raw_sleec_hyperbolic_concat, raw_mean_sleec_hyperbolic_gated, "
            "raw_mean_sleec_hyperbolic_capability_gated, "
            "raw_mean_sleec_hyperbolic_text_gated, raw_mean_sleec_blockwise, "
            "raw_mean_sleec_hyperbolic_blockwise, "
            "raw_mean_sleec_hyperbolic_capability_blockwise, "
            "raw_mean_sleec_hyperbolic_factorized_capability_blockwise, "
            "raw_mean_sleec_biofp_split, raw_mean_sleec_biological_factorized"
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
    if "reaction_pooling" in config.model and config.model.reaction_pooling not in {
        "attention",
        "mean",
    }:
        raise ValueError("'model.reaction_pooling' must be one of: attention, mean")
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
        }:
            raise ValueError(
                "'model.reaction_multimodal_attention.side_composition' must be "
                "directional_delta or molecule_set"
            )
        if reaction_attention_config.get("fusion", "attention") not in {
            "attention",
            "mean",
            "factorized_concat",
            "prior_bounded_attention",
        }:
            raise ValueError(
                "'model.reaction_multimodal_attention.fusion' must be attention, mean, "
                "factorized_concat, or prior_bounded_attention"
            )
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
    train_sampler = config.data.get("train_sampler", {})
    if train_sampler:
        if not isinstance(train_sampler, dict):
            raise ValueError("'data.train_sampler' must be a mapping")
        if train_sampler.get("name", "shuffle") not in {
            "shuffle",
            "reaction_degree_balanced",
        }:
            raise ValueError(
                "'data.train_sampler.name' must be one of: shuffle, " "reaction_degree_balanced"
            )
        exponent = train_sampler.get("degree_exponent", 0.5)
        if not _is_number(exponent) or float(exponent) < 0.0:
            raise ValueError("'data.train_sampler.degree_exponent' must be non-negative")
        sampler_seed = train_sampler.get("seed", config.get("seed", 42))
        if not isinstance(sampler_seed, int):
            raise ValueError("'data.train_sampler.seed' must be an integer")
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
                "'model.reaction_conditioned_pooling.attention_rank' must be null or a positive integer"
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


def parse_overrides(args: list[str]) -> Dict[str, Any]:
    """
    Parse command-line overrides in the format --key=value or --key value.

    Args:
        args: List of command-line arguments.

    Returns:
        Dictionary of parsed overrides.

    Example:
        >>> parse_overrides(['--training.max_epochs=50', '--training.learning_rate', '1e-3'])
        {'training.max_epochs': 50, 'training.learning_rate': 0.001}
    """
    overrides = {}
    i = 0

    while i < len(args):
        arg = args[i]

        if arg.startswith("--"):
            # Remove leading dashes
            arg = arg[2:]

            # Check for = format
            if "=" in arg:
                key, value = arg.split("=", 1)
                overrides[key] = _parse_value(value)
                i += 1
            else:
                # Check for space-separated format
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    key = arg
                    value = args[i + 1]
                    overrides[key] = _parse_value(value)
                    i += 2
                else:
                    # Boolean flag (no value provided)
                    overrides[arg] = True
                    i += 1
        else:
            i += 1

    return overrides


def _parse_value(value: str) -> Any:
    """
    Parse a string value to the appropriate Python type.

    Tries to parse as int, float, bool, or keeps as string.

    Args:
        value: String value to parse.

    Returns:
        Parsed value with appropriate type.
    """
    # Try boolean
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False

    # Try int
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Keep as string
    return value
