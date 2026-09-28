#!/usr/bin/env python3
"""Create the strict-validation F3-MC run with the original F3 EC block."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "runs/reactzyme_f3_biological_no_ec_v1"
RUN_ROOT = ROOT / "runs/reactzyme_f3mc_ec_v1"
VARIANT = "F3MC_ec"
PROJECT = "horizyn-reactzyme-f3mc-ec-v1"

F3_DIMS = {"core": 288, "site": 96, "mechanism": 64, "cofactor": 32, "ec": 32}
F3_WEIGHTS = {
    "core": 0.55,
    "site": 0.20,
    "mechanism": 0.12,
    "cofactor": 0.08,
    "ec": 0.05,
}
HYPERBOLIC_ENCODER = {
    "hyp_dim": 128,
    "use_tangent": True,
    "freeze_projector": False,
    "load_attention_pooler": False,
    "freeze_attention_pooler": False,
}


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def restore_ec(config: dict[str, Any]) -> None:
    model = config["model"]
    fusion = model["enzyme_block_fusion"]
    expected_dims = {"core": 320, "site": 96, "mechanism": 64, "cofactor": 32}
    expected_weights = {"core": 0.60, "site": 0.20, "mechanism": 0.12, "cofactor": 0.08}
    if fusion["dims"] != expected_dims or fusion["weights"] != expected_weights:
        raise ValueError("Source no-EC layout changed; refusing an unaudited EC restoration")
    fusion["dims"] = copy.deepcopy(F3_DIMS)
    fusion["weights"] = copy.deepcopy(F3_WEIGHTS)
    model["hyperbolic_encoder"] = copy.deepcopy(HYPERBOLIC_ENCODER)


def configure(config: dict[str, Any], subset: str) -> dict[str, Any]:
    restore_ec(config)
    log = config["logging"]
    log["log_dir"] = str(RUN_ROOT / "logs" / VARIANT / subset)
    log["checkpoint_dir"] = str(RUN_ROOT / "checkpoints" / VARIANT / "train")
    wandb = log.setdefault("wandb", {})
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
                "F3",
                "ec-tangent-block",
                "mechanism-cofactor-supervision",
                "no-cross-tower-alignment",
                "two-gpu",
                "global-batch-2048",
                "bidirectional-selection",
                "seed-42",
            ],
            "log_model": False,
        }
    )
    config["ablation"] = {
        "campaign": "reactzyme_f3mc_ec_v1",
        "run_id": f"{VARIANT}_reaction_smi_seed42",
        "variant": VARIANT,
        "label": "F3-MC-EC",
        "split": "reaction_smi",
        "seed": 42,
        "evaluation_subset": subset,
        "description": (
            "Original F3 EC/tangent block with joint mechanism/cofactor supervision "
            "and no cross-tower alignment."
        ),
        "validation_protocol": "reaction_cluster_similarity_0p85",
        "test_protocol": "paper_test_candidates",
        "checkpoint_monitor": "val/mean_bidirectional_reactzyme_mrr",
        "global_batch_size": 2048,
        "devices": 2 if subset == "train" else 1,
        "ec_block": True,
        "ec_auxiliary_supervision": False,
        "biological_pretraining": False,
        "biofp_aux_weight": 1.0,
        "biofp_family_weights": {"mechanism": 0.65, "cofactor": 0.35},
        "cross_tower_alignment_weight": 0.0,
    }
    return config


def main() -> None:
    sources = {
        subset: SOURCE_ROOT / "configs" / "F3MC_no_ec" / f"{subset}.yaml"
        for subset in ("train", "validation", "test")
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source configs: {missing}")

    config_dir = RUN_ROOT / "configs" / VARIANT
    config_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for subset, source in sources.items():
        output = config_dir / f"{subset}.yaml"
        payload = configure(load_yaml(source), subset)
        output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        outputs[subset] = str(output)

    manifest = {
        "schema_version": "reactzyme_f3mc_ec_v1",
        "campaign": "reactzyme_f3mc_ec_v1",
        "variant": VARIANT,
        "source_variant": "F3MC_no_ec",
        "source_configs": {
            subset: {"path": str(path), "sha256": digest(path)}
            for subset, path in sources.items()
        },
        "configs": outputs,
        "architecture_change": {
            "restored_block": "ec",
            "ec_dim": 32,
            "ec_weight": 0.05,
            "core_dim": 288,
            "core_weight": 0.55,
            "hyperbolic_dim": 128,
            "use_tangent": True,
            "ec_auxiliary_supervision": False,
        },
        "supervision": {
            "biofp_aux_weight": 1.0,
            "families": {"mechanism": 0.65, "cofactor": 0.35},
            "cross_tower_alignment_weight": 0.0,
        },
        "devices": 2,
        "batch_size_per_gpu": 1024,
        "global_batch_size": 2048,
        "checkpoint_monitor": "val/mean_bidirectional_reactzyme_mrr",
    }
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run_root": str(RUN_ROOT), "variant": VARIANT}, indent=2))


if __name__ == "__main__":
    main()
