"""Checkpoint loading helpers for optional pretrained encoders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _load_torch_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dict: {checkpoint_path}")
    return checkpoint


def _strip_known_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = (
        "module.model.query_encoder.",
        "model.query_encoder.",
        "module.query_encoder.",
        "query_encoder.",
        "module.",
    )
    output: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        stripped_key = key
        for prefix in prefixes:
            if stripped_key.startswith(prefix):
                stripped_key = stripped_key[len(prefix) :]
                break
        output[stripped_key] = value
    return output


def extract_query_encoder_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract a query/reaction encoder state dict from supported checkpoint shapes."""
    for key in ("query_encoder_state_dict", "reaction_encoder_state_dict"):
        state_dict = checkpoint.get(key)
        if isinstance(state_dict, dict):
            return _strip_known_prefixes(state_dict)

    lightning_state = checkpoint.get("state_dict")
    if isinstance(lightning_state, dict):
        query_state = {
            key: value
            for key, value in lightning_state.items()
            if key.startswith(("model.query_encoder.", "module.model.query_encoder."))
        }
        if query_state:
            return _strip_known_prefixes(query_state)

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return _strip_known_prefixes(checkpoint)

    raise KeyError(
        "Could not find query encoder weights. Expected one of "
        "'query_encoder_state_dict', 'reaction_encoder_state_dict', a Lightning "
        "'state_dict' with model.query_encoder.* keys, or a raw state dict."
    )


def load_query_encoder_checkpoint(
    query_encoder: torch.nn.Module,
    checkpoint_path: str | Path | None,
    *,
    strict: bool = True,
) -> None:
    """Load pretrained reaction/query encoder weights into an existing module."""
    if checkpoint_path is None or str(checkpoint_path) == "":
        return
    checkpoint = _load_torch_checkpoint(checkpoint_path)
    state_dict = extract_query_encoder_state_dict(checkpoint)
    query_encoder.load_state_dict(state_dict, strict=strict)
