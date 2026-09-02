#!/usr/bin/env python3
"""Create the fixed-budget auxiliary-weight calibration configs."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

import create_f3_biological_no_ec_campaign as campaign


WEIGHTS = (0.03, 0.10, 0.30, 1.00)
EPOCHS = 3


def weight_tag(weight: float) -> str:
    return f"lambda_{weight:.2f}".replace(".", "p")


def main() -> None:
    base = campaign.load_yaml(campaign.BASE_TRAIN)
    rows = []
    for index, weight in enumerate(WEIGHTS):
        tag = weight_tag(weight)
        root = campaign.RUN_ROOT / "calibration" / tag
        config = campaign.make_train(
            base,
            "F3MC_no_ec",
            aux_weight=weight,
            max_epochs=EPOCHS,
        )
        config["logging"].update(
            {
                "log_dir": str(root / "logs"),
                "checkpoint_dir": str(root / "checkpoints"),
                "save_top_k": 1,
            }
        )
        config["logging"]["wandb"].update(
            {
                "run_name": f"F3MC-noEC-aux-calibration-{tag}-seed42",
                "tags": [
                    "reactzyme",
                    "reaction-smi",
                    "reaction-cluster-validation-0p85",
                    "F3-MC-noEC",
                    "aux-weight-calibration",
                    tag,
                    "two-gpu",
                    "global-batch-2048",
                    "seed-42",
                ],
            }
        )
        config["training"]["early_stopping"]["enabled"] = False
        config["ablation"].update(
            {
                "campaign": "reactzyme_f3_no_ec_aux_calibration_v1",
                "calibration": True,
                "calibration_weight": weight,
                "calibration_epochs": EPOCHS,
            }
        )
        path = root / "train.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        rows.append(
            {
                "index": index,
                "tag": tag,
                "weight": weight,
                "epochs": EPOCHS,
                "config": str(path),
                "checkpoint_dir": str(root / "checkpoints"),
                "metrics_glob": str(root / "logs/protein_pooling_training/*/metrics.csv"),
            }
        )
    manifest_path = campaign.RUN_ROOT / "calibration/manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "f3_no_ec_aux_calibration_v1",
                "selection_metric": campaign.MONITOR,
                "selection_rule": "maximum metric over the equal three-epoch budget",
                "variant": "F3MC_no_ec",
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
