#!/usr/bin/env python3
"""Create the strict-split F3 run with exact reaction-modality L2 normalization."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/benchmarks/reactzyme_f3_cluster_proxy_v1.yaml"
TEST_SOURCE = ROOT / "runs/reactzyme_f3mc_ec_v1/configs/F3MC_ec/test.yaml"
RUN_ROOT = ROOT / "runs/reactzyme_f3_l2norm_strict_v1"
CONFIG = RUN_ROOT / "configs/F3_L2/train.yaml"
TEST_CONFIG = RUN_ROOT / "configs/F3_L2/test.yaml"
PROJECT = "horizyn-reactzyme-f3-l2norm-strict-v1"


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


def assert_f3_source(config: dict[str, Any]) -> None:
    attention = config["model"]["reaction_multimodal_attention"]
    loss = config["training"]["loss"]
    expected_panel = "reactzyme_reaction_cluster_validation_v1/similarity_0p85"
    checks = {
        "strict train panel": expected_panel in config["data"]["train_pairs_path"],
        "strict validation panel": expected_panel in config["data"]["validation_pairs_path"],
        "four-device source": int(config["training"]["devices"]) == 4,
        "512-row source batch": int(config["data"]["train_batch_size"]) == 512,
        "observed-pair retrieval": loss["positive_pair_source"] == "observed_pairs",
        "no biological auxiliary loss": float(loss["biofp_aux_weight"]) == 0.0,
        "attention fusion": attention["fusion"] == "attention",
        "token LayerNorm": bool(attention["token_layer_norm"]),
        "L2 disabled in source": not bool(attention.get("modality_l2_normalize", False)),
        "full checkpoint retention": int(config["logging"]["save_top_k"]) == -1,
        "no early stopping": not bool(config["training"]["early_stopping"]["enabled"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"F3 source audit failed: {failed}")


def configure(source: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(source)
    assert_f3_source(config)

    config["logging"].update(
        {
            "log_dir": str(RUN_ROOT / "logs/F3_L2/train"),
            "checkpoint_dir": str(RUN_ROOT / "checkpoints/F3_L2/train"),
            "checkpoint_monitor": "val/mean_bidirectional_reactzyme_mrr",
        }
    )
    config["logging"]["wandb"] = {
        "enabled": True,
        "project": PROJECT,
        "entity": "omnai",
        "run_name": "F3_L2-reaction-smi-strict-seed42",
        "mode": "online",
        "tags": [
            "reactzyme",
            "reaction-smi",
            "reaction-cluster-validation-0p85",
            "F3",
            "reaction-modality-l2-normalization",
            "global-batch-2048",
            "bidirectional-selection",
            "seed-42",
        ],
        "log_model": False,
    }

    config["model"]["reaction_multimodal_attention"]["modality_l2_normalize"] = True

    # Preserve F3's global batch (4 x 512) on the two currently free GPUs.
    config["data"]["train_batch_size"] = 1024
    config["training"]["devices"] = 2
    config["training"]["early_stopping"]["monitor"] = (
        "val/mean_bidirectional_reactzyme_mrr"
    )

    config["ablation"] = {
        "campaign": "reactzyme_f3_l2norm_strict_v1",
        "run_id": "F3_L2_reaction_smi_strict_seed42",
        "variant": "F3_L2",
        "label": "F3 + exact modality L2 normalization",
        "split": "reaction_smi",
        "seed": 42,
        "description": (
            "Strict-split F3 with scale-preserving L2 normalization of every "
            "reaction-modality token after LayerNorm and before attention/fusion."
        ),
        "validation_protocol": "reaction_cluster_similarity_0p85",
        "checkpoint_monitor": "val/mean_bidirectional_reactzyme_mrr",
        "source_devices": 4,
        "source_batch_size_per_gpu": 512,
        "devices": 2,
        "batch_size_per_gpu": 1024,
        "global_batch_size": 2048,
        "scientific_change": {"reaction_modality_l2_normalize": True},
    }
    return config


def configure_test(
    train_config: dict[str, Any], source_test: dict[str, Any]
) -> dict[str, Any]:
    """Build the paper-test config while preserving the trained F3-L2 model."""
    config = copy.deepcopy(source_test)
    config["model"] = copy.deepcopy(train_config["model"])
    config["training"] = copy.deepcopy(train_config["training"])

    config["data"]["protein_biofp_targets_path"] = None
    config["data"]["protein_biofp_vocab_path"] = None
    config["data"]["enzyme_ec_labels_path"] = None
    config["data"]["retrieval_batch_size"] = 512
    config["data"]["validation_retrieval_batch_size"] = 512

    test_candidates = config["data"]["validation_retrieval_candidate_ids_path"]
    config["training"].update(
        {
            "devices": 1,
            "accelerator": "gpu",
            "strategy": "auto",
            "validation_retrieval_candidate_set": "custom",
            "validation_retrieval_candidate_ids_path": test_candidates,
            "validation_retrieval_batch_size": 512,
            "validation_retrieval_directions": [
                "reaction_to_enzyme",
                "enzyme_to_reaction",
            ],
        }
    )

    config["logging"].update(
        {
            "log_dir": str(RUN_ROOT / "logs/F3_L2/test"),
            "checkpoint_dir": str(RUN_ROOT / "checkpoints/F3_L2/train"),
        }
    )
    config["logging"]["wandb"]["enabled"] = False
    config["ablation"] = copy.deepcopy(train_config["ablation"])
    config["ablation"].update(
        {
            "evaluation_subset": "test",
            "test_protocol": "paper_test_candidates",
            "selected_checkpoint_epoch": 4,
            "selection_rule": (
                "maximize validation E-to-R MRR subject to validation R-to-E "
                "MRR >= 0.04989478894; harmonic MRR is the tie-breaker"
            ),
        }
    )
    return config


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if not TEST_SOURCE.is_file():
        raise FileNotFoundError(TEST_SOURCE)
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    payload = configure(load_yaml(SOURCE))
    CONFIG.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    test_payload = configure_test(payload, load_yaml(TEST_SOURCE))
    TEST_CONFIG.write_text(
        yaml.safe_dump(test_payload, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "schema_version": "reactzyme_f3_l2norm_strict_v1",
        "campaign": "reactzyme_f3_l2norm_strict_v1",
        "source_config": {"path": str(SOURCE), "sha256": digest(SOURCE)},
        "train_config": str(CONFIG),
        "test_config": str(TEST_CONFIG),
        "scientific_diff": {
            "model.reaction_multimodal_attention.modality_l2_normalize": {
                "source": False,
                "variant": True,
            }
        },
        "operational_diff": {
            "training.devices": {"source": 4, "variant": 2},
            "data.train_batch_size": {"source": 512, "variant": 1024},
            "global_batch_size": {"source": 2048, "variant": 2048},
        },
        "selection": {
            "selected_epoch": 4,
            "primary_metric": "val/enzyme_to_reaction/reactzyme_mrr",
            "primary_mode": "max",
            "guard_metric": "val/reaction_to_enzyme/reactzyme_mrr",
            "guard_reference": 0.06489478894,
            "guard_tolerance": 0.015,
            "guard_floor": 0.04989478894,
            "tie_breaker": "harmonic_all_positive_mrr",
        },
        "max_epochs": 30,
        "early_stopping": False,
    }
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "train_config": str(CONFIG),
                "test_config": str(TEST_CONFIG),
                "run_root": str(RUN_ROOT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
