#!/usr/bin/env python3
"""Freeze the EnzymeCAGE-data phase-2 extension before target P450 scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def identity(path: Path) -> dict:
    path = path.resolve()
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return {"path": str(path), "sha256": h.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--architecture-lock", type=Path, required=True)
    parser.add_argument("--phase2-recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    original = json.loads(args.phase2_recipe.read_text())
    expected_smooth = {"kernel": "exponential", "reaction_neighbors": None,
                       "reaction_temperature": 0.03, "protein_neighbors": 32,
                       "enzyme_temperature": 0.03}
    if original["composition"] != {"alpha": 0.25} or original["smooth"] != expected_smooth:
        raise ValueError("Historical phase-2 recipe differs from the declared target recipe")
    feature_manifest = identity(args.target / "features/manifest.json")
    dictionary_path = args.target / "smooth_dictionary/anchors.pt"
    dictionary = torch.load(dictionary_path, map_location="cpu", weights_only=False)
    if (dictionary["feature_manifest_sha256"] != feature_manifest["sha256"]
            or len(dictionary["train_reactions"]) != 9310):
        raise ValueError("Smooth dictionary and target training data disagree")
    residuals, density = {}, {}
    for seed in (42, 17, 73):
        residual_path = args.target / f"graph_uniform_s{seed}/selected.pt"
        residual = torch.load(residual_path, map_location="cpu", weights_only=False)
        if (residual["registry"]["feature_manifest_sha256"] != feature_manifest["sha256"]
                or residual["registry"]["test_used"] is not False):
            raise ValueError(f"Residual seed {seed} does not belong to target training data")
        residuals[str(seed)] = identity(residual_path)
        density_path = args.target / f"density_models/seed{seed}/density_bundle.json"
        density_spec = json.loads(density_path.read_text())
        if (density_spec["feature_manifest_sha256"] != feature_manifest["sha256"]
                or density_spec["seed"] != seed
                or density_spec["graph_checkpoint"]["sha256"] != residuals[str(seed)]["sha256"]
                or density_spec["test_used"] is not False):
            raise ValueError(f"Density seed {seed} provenance mismatch")
        density[str(seed)] = identity(density_path)
    freeze = {
        "schema": "enzymecage_phase2_target_freeze_v1",
        "scope": "positive-only F3 target retrain plus target-specific frozen-tower phase-2 extension",
        "architecture_lock": identity(args.architecture_lock),
        "historical_phase2_recipe": identity(args.phase2_recipe),
        "base_checkpoint": identity(args.base_checkpoint),
        "base_config": identity(args.base_config),
        "feature_manifest": feature_manifest,
        "smooth_dictionary": identity(dictionary_path),
        "smooth": expected_smooth,
        "alpha": 0.25,
        "residuals": residuals,
        "density_bundles": density,
        "selection": "Only EnzymeCAGE original validation positives; fixed phase-2 parameters from historical ReactZyme recipe",
        "negative_training_rows_used": False,
        "p450_scores_used_for_extension_selection": False,
        "p450_panel_previously_opened_for_f3_base": True,
        "comparison_is_training_controlled": False,
        "source": identity(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(freeze, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "feature_manifest": feature_manifest["sha256"]}))


if __name__ == "__main__":
    main()
