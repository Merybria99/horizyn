#!/usr/bin/env python3
"""Extend the same EnzymeMap F3 fit before any held-out screening evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.base_config.read_text())
    training = config["training"]
    if (config["ablation"]["data_protocol"] != "official_EnzymeMap_train_positives_only"
            or training["max_epochs"] != 30
            or training["check_val_every_n_epoch"] != 1
            or training["early_stopping"]["patience"] != 5):
        raise ValueError("Unexpected original EnzymeMap F3 training config")
    training["max_epochs"] = 60
    training["check_val_every_n_epoch"] = 3
    training["early_stopping"]["patience"] = 5
    content = yaml.safe_dump(config, sort_keys=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_text() != content:
        raise ValueError("Extended config already exists with different contents")
    args.output.write_text(content)
    receipt = {"schema": "clipzyme_enzymemap_f3_extension_v1",
               "base_config_sha256": sha256(args.base_config),
               "extended_config_sha256": sha256(args.output),
               "architecture_changed": False,
               "training_data_changed": False,
               "validation_data_changed": False,
               "changes": {"max_epochs": [30, 60],
                           "check_val_every_n_epoch": [1, 3],
                           "early_stopping_patience_validation_checks": [5, 5]},
               "test_labels_or_scores_seen_before_extension": False,
               "source_sha256": sha256(Path(__file__))}
    (args.output.parent / "extended_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
