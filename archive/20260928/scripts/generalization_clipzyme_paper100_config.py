#!/usr/bin/env python3
"""Continue the fixed EnzymeMap F3 fit to FGW-CLIP's 100-epoch budget."""
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
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.base_config.read_text())
    training = config["training"]
    if (config["ablation"]["data_protocol"] != "official_EnzymeMap_train_positives_only"
            or training["max_epochs"] != 60
            or training["check_val_every_n_epoch"] != 3
            or not training["early_stopping"]["enabled"]):
        raise ValueError("Unexpected completed EnzymeMap F3 config")
    training["max_epochs"] = 100
    training["early_stopping"]["enabled"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    receipt = {"schema": "clipzyme_enzymemap_f3_paper100_continuation_v1",
               "base_config_sha256": sha256(args.base_config),
               "paper100_config_sha256": sha256(args.output),
               "architecture_changed": False,
               "training_data_changed": False,
               "validation_data_changed": False,
               "optimizer_or_learning_rate_changed": False,
               "changes": {"max_epochs": [60, 100],
                           "early_stopping_enabled": [True, False]},
               "checkpoint_selection": "last epoch (99), matching FGW-CLIP's stated EnzymeMap selection rule",
               "interim_test_scores_seen_before_extension": True,
               "interpretation": "Post-interim exploratory extension, not a blind confirmatory experiment.",
               "source_sha256": sha256(Path(__file__))}
    (args.output.parent / "paper100_continuation_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
