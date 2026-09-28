#!/usr/bin/env python3
"""Create full-budget F3MC-EC runs with reduced biological auxiliary weight."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/reactzyme_f3mc_ec_v1/configs/F3MC_ec/train.yaml"
RUN_ROOT = ROOT / "runs/reactzyme_f3mc_ec_aux_sweep_v1"
PROJECT = "horizyn-reactzyme-f3mc-ec-aux-sweep-v1"
WEIGHTS = (0.10, 0.30, 3.00)


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


def weight_tag(weight: float) -> str:
    return f"lambda_{weight:.2f}".replace(".", "p")


def configure(source: dict[str, Any], weight: float) -> dict[str, Any]:
    config = copy.deepcopy(source)
    tag = weight_tag(weight)
    variant = f"F3MC_ec_{tag}"

    config["training"]["loss"]["biofp_aux_weight"] = weight
    config["training"]["early_stopping"]["enabled"] = False
    config["logging"].update(
        {
            "log_dir": str(RUN_ROOT / "logs" / tag / "train"),
            "checkpoint_dir": str(RUN_ROOT / "checkpoints" / tag / "train"),
            "save_top_k": -1,
        }
    )
    config["logging"]["wandb"].update(
        {
            "project": PROJECT,
            "run_name": f"{variant}-reaction-smi-seed42",
            "tags": [
                "reactzyme",
                "reaction-smi",
                "reaction-cluster-validation-0p85",
                "F3MC-EC",
                "aux-weight-sweep",
                tag,
                "two-gpu",
                "global-batch-2048",
                "full-30-epochs",
                "seed-42",
            ],
        }
    )
    config["ablation"].update(
        {
            "campaign": "reactzyme_f3mc_ec_aux_sweep_v1",
            "run_id": f"{variant}_reaction_smi_seed42",
            "variant": variant,
            "label": f"F3-MC-EC, auxiliary weight {weight:.2f}",
            "description": (
                "Original unnormalized F3 architecture with joint mechanism "
                f"and cofactor supervision downweighted to {weight:.2f}."
            ),
            "biofp_aux_weight": weight,
            "full_training_budget": True,
        }
    )
    return config


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source = load_yaml(SOURCE)
    if source["training"]["loss"]["biofp_aux_weight"] != 1.0:
        raise ValueError("Expected the audited F3MC-EC source to use auxiliary weight 1.0")
    if source["model"]["reaction_multimodal_attention"].get(
        "modality_l2_normalize", False
    ):
        raise ValueError("The source unexpectedly enables reaction-modality L2")

    rows = []
    for weight in WEIGHTS:
        tag = weight_tag(weight)
        path = RUN_ROOT / "configs" / tag / "train.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(configure(source, weight), sort_keys=False),
            encoding="utf-8",
        )
        (RUN_ROOT / "logs" / tag / "train").mkdir(parents=True, exist_ok=True)
        (RUN_ROOT / "checkpoints" / tag / "train").mkdir(
            parents=True, exist_ok=True
        )
        rows.append(
            {
                "tag": tag,
                "weight": weight,
                "effective_mechanism_weight": 0.65 * weight,
                "effective_cofactor_weight": 0.35 * weight,
                "config": str(path),
            }
        )

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "reactzyme_f3mc_ec_aux_sweep_v1",
                "source_config": {"path": str(SOURCE), "sha256": sha256(SOURCE)},
                "fixed": {
                    "architecture": "original unnormalized F3 including EC block",
                    "family_weights": {"mechanism": 0.65, "cofactor": 0.35},
                    "max_epochs": 30,
                    "early_stopping": False,
                    "devices": 2,
                    "batch_size_per_gpu": 1024,
                    "global_batch_size": 2048,
                    "seed": 42,
                },
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"run_root": str(RUN_ROOT), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
