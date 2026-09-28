"""Checkpoint loading shared by training evaluation and case studies."""

from pathlib import Path

import torch

from horizyn.config import load_config
from horizyn.generalization_residual import FrozenGeometryResidual


def model_from_checkpoint(config_path: Path, checkpoint: Path, device: str):
    # Construct on CPU: optimizer buffers must never occupy inference VRAM.
    from horizyn.benchmarks.retrieval import load_repo_checkpoint

    config = load_config(str(config_path))
    model, kind = load_repo_checkpoint(checkpoint, config, "cpu")
    if kind != "residue":
        raise ValueError("V4, F3, and CIRCEv2 require residue-level checkpoints")
    model.to(device).eval().requires_grad_(False)
    return model, config


def load_head(path, manifest, device):
    """Load residual weights only if their training-feature lineage matches."""
    state = torch.load(path, map_location=device, weights_only=False)
    registry = state["registry"]
    if registry["feature_manifest_sha256"] != manifest or registry["test_used"]:
        raise ValueError("Residual training lineage mismatch")
    model = FrozenGeometryResidual(**state["model_config"])
    model.load_state_dict(state["state_dict"], strict=True)
    model.training_registry = registry
    return model.to(device).eval().requires_grad_(False)
