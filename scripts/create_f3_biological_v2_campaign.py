#!/usr/bin/env python3
"""Create the controlled F3-v2 mechanism/cofactor supervision pilot."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "runs/reactzyme_f3_reaction_smi_paper_v2_4gpu/configs"
RUN_ROOT = ROOT / "runs/reactzyme_f3_biological_v2"
TARGET_ROOT = RUN_ROOT / "data/reaction_smi/biofp"
TARGETS = TARGET_ROOT / "enzyme_biofp_soft_targets.npz"
VOCAB = TARGET_ROOT / "enzyme_biofp_vocab.json"
PROJECT = "horizyn-reactzyme-f3-biological-v2"

VARIANTS: dict[str, dict[str, Any]] = {
    "F3M_mechanism": {
        "label": "F3-M",
        "families": {"mechanism": 1.0},
        "description": "Exact audited F3-v2 with mechanism pretraining and joint supervision.",
    },
    "F3C_cofactor": {
        "label": "F3-C",
        "families": {"cofactor": 1.0},
        "description": "Exact audited F3-v2 with cofactor pretraining and joint supervision.",
    },
    "F3MC_mechanism_cofactor": {
        "label": "F3-MC",
        "families": {"mechanism": 0.65, "cofactor": 0.35},
        "description": "Exact audited F3-v2 with mechanism/cofactor pretraining and joint supervision.",
    },
}


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_id(variant: str) -> str:
    return f"{variant}_reaction_smi_seed42"


def configure_logging(
    config: dict[str, Any], variant: str, stage: str, *, enabled: bool
) -> None:
    metadata = VARIANTS[variant]
    identifier = run_id(variant)
    logging = config["logging"]
    logging.update(
        {
            "log_dir": str(RUN_ROOT / "logs" / variant / stage),
            "checkpoint_dir": str(RUN_ROOT / "checkpoints" / variant / stage),
            "checkpoint_on_validation_end": True,
        }
    )
    logging.setdefault("wandb", {}).update(
        {
            "enabled": enabled,
            "project": PROJECT,
            "entity": "omnai",
            "run_name": f"{identifier}-{stage}",
            "mode": "online",
            "tags": [
                "reactzyme",
                "reaction-smi",
                "paper-v2",
                "F3",
                metadata["label"],
                "biological-supervision",
                "four-gpu",
                "global-batch-2048",
                "seed-42",
            ],
            "log_model": False,
        }
    )


def configure_ablation(config: dict[str, Any], variant: str, stage: str) -> None:
    metadata = VARIANTS[variant]
    config["ablation"] = {
        "campaign": "reactzyme_f3_biological_v2",
        "run_id": run_id(variant),
        "variant": variant,
        "label": metadata["label"],
        "split": "reaction_smi",
        "seed": 42,
        "stage": stage,
        "base_architecture": "F3_reaction_smi_paper_v2_seed42_4gpu",
        "description": metadata["description"],
        "protocol_schema": "reactzyme_paper_protocol_v2",
        "validation_fidelity": "deterministic_positive_only_reactzyme_algorithm_analogue",
        "test_protocol": "paper_test_candidates",
        "audited_global_batch_size": 2048,
    }


def add_biological_targets(config: dict[str, Any]) -> None:
    config["data"].update(
        {
            "protein_biofp_targets_path": str(TARGETS),
            "protein_biofp_vocab_path": str(VOCAB),
            "biofp_missing_policy": "zero_with_mask",
        }
    )


def make_pretrain(base: dict[str, Any], variant: str) -> dict[str, Any]:
    config = copy.deepcopy(base)
    configure_logging(config, variant, "biological_pretrain", enabled=True)
    configure_ablation(config, variant, "biological_pretrain")
    add_biological_targets(config)
    source_data = config["data"]
    config["data"] = {
        "protein_residue_embeds_path": source_data["protein_residue_embeds_path"],
        "protein_biofp_targets_path": str(TARGETS),
        "protein_biofp_vocab_path": str(VOCAB),
        "biofp_missing_policy": "zero_with_mask",
        "residue_dim": source_data.get("residue_dim", 1024),
        "max_protein_tokens": source_data.get("max_protein_tokens", 1022),
        "protein_truncation": source_data.get("protein_truncation", "ends_center"),
    }
    config["training"] = {
        "max_epochs": 20,
        "batch_size": 128,
        "learning_rate": 1.0e-4,
        "weight_decay": 0.01,
        "validation_fraction": 0.10,
        "biofp_family_weights": VARIANTS[variant]["families"],
        "biofp_confidence_cap": 1.0,
        "checkpoint_monitor": "val/loss",
        "checkpoint_mode": "min",
        "early_stopping_patience": 4,
        "early_stopping_min_delta": 0.0001,
        "accelerator": "gpu",
        "devices": 4,
        "strategy": "ddp_find_unused_parameters_true",
        "precision": "32-true",
        "num_workers": 2,
        "pin_memory": False,
        "persistent_workers": True,
        "log_every_n_steps": 10,
        "enable_progress_bar": False,
    }
    config["wandb_args"] = {}
    return config


def make_retrieval(base: dict[str, Any], variant: str, stage: str) -> dict[str, Any]:
    config = copy.deepcopy(base)
    configure_logging(config, variant, stage, enabled=stage == "train")
    configure_ablation(config, variant, stage)
    add_biological_targets(config)
    loss = config["training"]["loss"]
    loss.update(
        {
            "biofp_aux_weight": 0.03,
            "biofp_family_weights": VARIANTS[variant]["families"],
            "biofp_confidence_cap": 1.0,
        }
    )
    if stage == "train":
        config["training"]["biofp_pretrain_checkpoint"] = str(
            RUN_ROOT / "checkpoints" / variant / "biological_pretrain/best.ckpt"
        )
    return config


def main() -> None:
    required = [SOURCE_ROOT / "train.yaml", SOURCE_ROOT / "validation.yaml", SOURCE_ROOT / "test.yaml", TARGETS, VOCAB]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing campaign inputs: {missing}")

    sources = {name: load_yaml(SOURCE_ROOT / f"{name}.yaml") for name in ("train", "validation", "test")}
    config_manifest: dict[str, dict[str, str]] = {}
    for variant in VARIANTS:
        config_dir = RUN_ROOT / "configs" / variant
        config_dir.mkdir(parents=True, exist_ok=True)
        payloads = {
            "biological_pretrain": make_pretrain(sources["train"], variant),
            "train": make_retrieval(sources["train"], variant, "train"),
            "validation": make_retrieval(sources["validation"], variant, "validation"),
            "test": make_retrieval(sources["test"], variant, "test"),
        }
        config_manifest[variant] = {}
        for stage, payload in payloads.items():
            path = config_dir / f"{stage}.yaml"
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            config_manifest[variant][stage] = str(path)

    manifest = {
        "schema_version": "reactzyme_f3_biological_v2_v1",
        "campaign": "reactzyme_f3_biological_v2",
        "scope": "reaction_smi_pilot",
        "control_run": str(ROOT / "runs/reactzyme_f3_reaction_smi_paper_v2_4gpu"),
        "source_configs": {
            name: {
                "path": str(SOURCE_ROOT / f"{name}.yaml"),
                "sha256": sha256(SOURCE_ROOT / f"{name}.yaml"),
            }
            for name in sources
        },
        "biological_targets": {
            "npz": str(TARGETS),
            "npz_sha256": sha256(TARGETS),
            "vocab": str(VOCAB),
            "vocab_sha256": sha256(VOCAB),
        },
        "variants": VARIANTS,
        "configs": config_manifest,
        "devices": [0, 1, 2, 3],
        "batch_size_per_gpu": 512,
        "global_batch_size": 2048,
        "auxiliary_weight": 0.03,
        "evaluation_metric": "reactzyme_all_positive_mrr",
    }
    (RUN_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run_root": str(RUN_ROOT), "variants": list(VARIANTS)}, indent=2))


if __name__ == "__main__":
    main()
