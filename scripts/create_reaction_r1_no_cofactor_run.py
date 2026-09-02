#!/usr/bin/env python3
"""Create R1: F3MC-EC with only reaction cofactor indicators removed."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from horizyn.capability.reaction_set_features import (
    COFACTOR_CAPACITY,
    REACTION_SET_BASE_DIM,
    REACTION_SET_FULL_DIM,
    materialize_reaction_set_features_without_cofactors,
)


SOURCE_RUN = ROOT / "runs/reactzyme_f3mc_ec_v1"
SOURCE_VARIANT = "F3MC_ec"
RUN_ROOT = ROOT / "runs/reactzyme_reaction_r1_v1"
VARIANT = "R1"
PROJECT = "horizyn-reactzyme-reaction-r1-v1"


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_source_config(config: dict[str, Any]) -> None:
    data = config["data"]
    model = config["model"]
    loss = config["training"]["loss"]
    expected_dims = {"core": 288, "site": 96, "mechanism": 64, "cofactor": 32, "ec": 32}
    expected_weights = {
        "core": 0.55,
        "site": 0.20,
        "mechanism": 0.12,
        "cofactor": 0.08,
        "ec": 0.05,
    }
    if int(data["reaction_chemistry_dim"]) != REACTION_SET_FULL_DIM:
        raise ValueError("R1 source must use the full 617D reaction-set vector")
    if model["enzyme_block_fusion"]["dims"] != expected_dims:
        raise ValueError("R1 source is not the expected F3 EC-block architecture")
    if model["enzyme_block_fusion"]["weights"] != expected_weights:
        raise ValueError("R1 source has unexpected F3 EC-block weights")
    if loss.get("biofp_aux_weight") != 1.0:
        raise ValueError("R1 source must use auxiliary weight 1.0")
    if loss.get("biofp_family_weights") != {"mechanism": 0.65, "cofactor": 0.35}:
        raise ValueError("R1 source must use the selected mechanism/cofactor mixture")
    if loss.get("cross_tower_alignment_weight") != 0.0:
        raise ValueError("R1 source must not use cross-tower alignment")


def feature_split_from_path(path: str) -> str:
    filename = Path(path).name
    for split in ("train", "validation", "test"):
        if filename.startswith(f"{split}_"):
            return split
    raise ValueError(f"Cannot identify reaction-set split from {path}")


def configure_r1(
    config: dict[str, Any],
    *,
    subset: str,
    feature_paths: dict[str, Path],
) -> dict[str, Any]:
    verify_source_config(config)
    data = config["data"]
    data["reaction_chemistry_dim"] = REACTION_SET_BASE_DIM
    for key in (
        "train_reaction_chemistry_vectors_path",
        "validation_reaction_chemistry_vectors_path",
        "reaction_chemistry_vectors_path",
    ):
        if key in data:
            split = feature_split_from_path(str(data[key]))
            data[key] = str(feature_paths[split])

    logging = config["logging"]
    logging["log_dir"] = str(RUN_ROOT / "logs" / VARIANT / subset)
    logging["checkpoint_dir"] = str(RUN_ROOT / "checkpoints" / VARIANT / "train")
    wandb = logging.setdefault("wandb", {})
    wandb.update(
        {
            "enabled": subset == "train",
            "project": PROJECT,
            "entity": "omnai",
            "run_name": f"{VARIANT}-reaction-smi-seed42",
            "mode": "online",
            "tags": [
                "reactzyme",
                "reaction-smi",
                "reaction-cluster-validation-0p85",
                "R1",
                "reaction-set-580d",
                "no-explicit-cofactor-indicators",
                "F3-MC-EC",
                "no-reaction-mechanism-loss",
                "no-cross-tower-alignment",
                "global-batch-2048",
                "bidirectional-selection",
                "seed-42",
            ],
            "log_model": False,
        }
    )
    config["ablation"] = {
        "campaign": "reactzyme_reaction_r1_v1",
        "run_id": "R1_reaction_smi_seed42",
        "variant": VARIANT,
        "label": "R1-no-reaction-cofactor-indicators",
        "split": "reaction_smi",
        "seed": 42,
        "evaluation_subset": subset,
        "description": (
            "F3MC-EC with the final 37 explicit reaction cofactor indicators removed; "
            "all remaining inputs, architecture, and losses are unchanged."
        ),
        "validation_protocol": "reaction_cluster_similarity_0p85",
        "test_protocol": "paper_test_candidates",
        "checkpoint_monitor": "val/mean_bidirectional_reactzyme_mrr",
        "reaction_chemistry_source_dim": REACTION_SET_FULL_DIM,
        "reaction_chemistry_dim": REACTION_SET_BASE_DIM,
        "removed_reaction_feature": "core_cofactor_indicators",
        "removed_reaction_feature_dim": COFACTOR_CAPACITY,
        "ec_block": True,
        "enzyme_biofp_aux_weight": 1.0,
        "enzyme_biofp_family_weights": {"mechanism": 0.65, "cofactor": 0.35},
        "reaction_auxiliary_supervision": False,
        "cross_tower_alignment_weight": 0.0,
        "global_batch_size": 2048,
        "devices": 2 if subset == "train" else 1,
    }
    return config


def main() -> None:
    source_configs = {
        subset: SOURCE_RUN / "configs" / SOURCE_VARIANT / f"{subset}.yaml"
        for subset in ("train", "validation", "test")
    }
    missing = [str(path) for path in source_configs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing R1 source configs: {missing}")

    source_train = load_yaml(source_configs["train"])
    verify_source_config(source_train)
    source_feature_paths = {
        "train": Path(source_train["data"]["train_reaction_chemistry_vectors_path"]),
        "validation": Path(source_train["data"]["validation_reaction_chemistry_vectors_path"]),
    }
    source_test = load_yaml(source_configs["test"])
    source_feature_paths["test"] = Path(source_test["data"]["reaction_chemistry_vectors_path"])
    missing = [str(path) for path in source_feature_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source reaction-set features: {missing}")

    feature_root = RUN_ROOT / "data/reaction_smi/reaction_set"
    feature_paths = {
        split: feature_root / f"{split}_reaction_set_features_580d.npz"
        for split in ("train", "validation", "test")
    }
    feature_reports = {
        split: materialize_reaction_set_features_without_cofactors(
            source_path=source_feature_paths[split],
            output_path=feature_paths[split],
        )
        for split in ("train", "validation", "test")
    }
    feature_schema = {
        "schema_version": "reactzyme_reaction_set_features_r1_no_cofactor_v1",
        "source_schema_version": "reactzyme_reaction_set_features_v1",
        "source_dimension": REACTION_SET_FULL_DIM,
        "output_dimension": REACTION_SET_BASE_DIM,
        "retained_blocks": [
            {"name": "morgan_mean", "start": 0, "end": 512, "dimension": 512},
            {"name": "descriptor_aggregates", "start": 512, "end": 576, "dimension": 64},
            {"name": "set_statistics", "start": 576, "end": 580, "dimension": 4},
        ],
        "removed_blocks": [
            {
                "name": "core_cofactor_indicators",
                "start": 580,
                "end": 617,
                "dimension": COFACTOR_CAPACITY,
            }
        ],
        "splits": feature_reports,
    }
    feature_root.mkdir(parents=True, exist_ok=True)
    (feature_root / "schema.json").write_text(
        json.dumps(feature_schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    config_root = RUN_ROOT / "configs" / VARIANT
    config_root.mkdir(parents=True, exist_ok=True)
    config_paths: dict[str, Path] = {}
    for subset, source in source_configs.items():
        output = config_root / f"{subset}.yaml"
        payload = configure_r1(
            copy.deepcopy(load_yaml(source)),
            subset=subset,
            feature_paths=feature_paths,
        )
        output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        config_paths[subset] = output

    manifest = {
        "schema_version": "reactzyme_reaction_r1_v1",
        "campaign": "reactzyme_reaction_r1_v1",
        "variant": VARIANT,
        "source_run": str(SOURCE_RUN),
        "source_variant": SOURCE_VARIANT,
        "source_configs": {
            subset: {"path": str(path), "sha256": sha256(path)}
            for subset, path in source_configs.items()
        },
        "configs": {subset: str(path) for subset, path in config_paths.items()},
        "reaction_feature_change": {
            "source_dimension": REACTION_SET_FULL_DIM,
            "output_dimension": REACTION_SET_BASE_DIM,
            "removed_block": "core_cofactor_indicators",
            "removed_dimension": COFACTOR_CAPACITY,
            "retained_dimension": REACTION_SET_BASE_DIM,
        },
        "feature_artifacts": {
            split: {
                **feature_reports[split],
                "sha256": sha256(feature_paths[split]),
            }
            for split in feature_paths
        },
        "invariants": {
            "enzyme_architecture": "F3-MC-EC",
            "enzyme_auxiliary_weight": 1.0,
            "enzyme_auxiliary_families": {"mechanism": 0.65, "cofactor": 0.35},
            "reaction_auxiliary_supervision": False,
            "cross_tower_alignment_weight": 0.0,
            "global_batch_size": 2048,
            "checkpoint_monitor": "val/mean_bidirectional_reactzyme_mrr",
        },
    }
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run_root": str(RUN_ROOT), "variant": VARIANT}, indent=2))


if __name__ == "__main__":
    main()
