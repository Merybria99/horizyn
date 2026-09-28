"""Partial model warm starts, distinct from full trainer checkpoint resumes."""

from pathlib import Path

import torch

from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from horizyn.wandb_utils import rank_zero_print


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
    preserved_config: list[str] = []

    # These values define the new run rather than learned representation
    # parameters.  In particular, Stage A deliberately stores alpha=0 while
    # Stage B must use the alpha selected in its config.  Restoring this buffer
    # during a warm start would silently disable the fused Stage-B objective.
    preserve_from_current_config = {
        "model.biological_residual.fusion.alpha",
    }

    for key, value in source_state.items():
        if not key.startswith("model."):
            continue
        if key not in current_state:
            continue
        if key in preserve_from_current_config:
            preserved_config.append(key)
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
    if training_stage in {
        "e2r_adapter",
        "r2e_adapter",
        "bidirectional_adapters",
        "prototype_only",
        "biological_residual",
    }:
        if training_stage == "bidirectional_adapters":
            adapter_prefixes = ("model.e2r_adapter.", "model.r2e_adapter.")
        elif training_stage == "prototype_only":
            adapter_prefixes = ("model.enzyme_prototype_head.",)
        elif training_stage == "biological_residual":
            adapter_prefixes = ("model.biological_residual.",)
        else:
            adapter_prefixes = (f"model.{training_stage}.",)
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
        f"config_preserved={len(preserved_config)}, shape_skipped={len(skipped_shape)}, "
        f"missing={len(missing)}, "
        f"unexpected={len(unexpected)}"
    )
    if preserved_config:
        rank_zero_print("Config-controlled warm-start keys preserved:")
        for key in preserved_config:
            rank_zero_print(f"  {key}")
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
