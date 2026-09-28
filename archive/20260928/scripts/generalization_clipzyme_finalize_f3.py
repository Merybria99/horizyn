#!/usr/bin/env python3
"""Freeze the validation-selected checkpoint of the matched EnzymeMap F3 fit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config_path = args.catalog / "configs/train_extended.yaml"
    config = yaml.safe_load(config_path.read_text())
    if (config["training"]["max_epochs"] != 60
            or config["training"]["check_val_every_n_epoch"] != 3
            or config["data"]["train_pairs_path"] != str((args.catalog / "train_pairs.csv").resolve())
            or config["data"]["validation_pairs_path"] != str((args.catalog / "validation_pairs.csv").resolve())):
        raise ValueError("EnzymeMap F3 training protocol changed")
    log_path = args.catalog / "training_extended.console.log"
    log = log_path.read_text()
    if "PROTEIN-POOLING TRAINING COMPLETE" not in log:
        raise ValueError("Training did not finish cleanly")
    last = args.catalog / "checkpoints/last.ckpt"
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    callbacks = [value for key, value in checkpoint["callbacks"].items()
                 if key.startswith("ModelCheckpoint")]
    if len(callbacks) != 1:
        raise ValueError("Expected one validation-selected checkpoint callback")
    callback = callbacks[0]
    best = Path(callback["best_model_path"])
    score = float(callback["best_model_score"])
    if not best.is_file() or best.parent.resolve() != (args.catalog / "checkpoints").resolve():
        raise ValueError("Selected checkpoint is outside the pinned run")
    metrics = []
    for version in (0, 1):
        path = args.catalog / f"logs/train/protein_pooling_training/version_{version}/metrics.csv"
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["val/mean_bidirectional_mrr"]:
                    metrics.append({"epoch": int(row["epoch"]),
                                    "step": int(row["step"]),
                                    "mrr": float(row["val/mean_bidirectional_mrr"]),
                                    "log_version": version})
    if not metrics or abs(max(row["mrr"] for row in metrics) - score) > 1e-6:
        raise ValueError("Callback best metric does not match the validation log")
    selected_epoch = int(best.stem.split("epoch=")[-1])
    selected_rows = [row for row in metrics if row["epoch"] == selected_epoch]
    if len(selected_rows) != 1 or abs(selected_rows[0]["mrr"] - score) > 1e-6:
        raise ValueError("Selected epoch does not match the validation log")
    receipt = {"schema": "clipzyme_enzymemap_f3_training_complete_v1",
               "selected_checkpoint": str(best.resolve()),
               "selected_checkpoint_sha256": sha256(best),
               "selected_epoch": selected_epoch,
               "selected_validation_bidirectional_mrr": score,
               "last_completed_epoch": int(checkpoint["epoch"]),
               "validation_observations": metrics,
               "training_input_rows": 34427,
               "validation_input_rows": 7287,
               "test_labels_read_for_training_or_selection": False,
               "base_config_sha256": sha256(args.catalog / "configs/train.yaml"),
               "extended_config_sha256": sha256(config_path),
               "train_pairs_sha256": sha256(args.catalog / "train_pairs.csv"),
               "validation_pairs_sha256": sha256(args.catalog / "validation_pairs.csv"),
               "training_log_sha256": sha256(log_path),
               "source_sha256": sha256(Path(__file__))}
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"epoch": selected_epoch, "validation_mrr": score,
                      "checkpoint": str(best)}), flush=True)


if __name__ == "__main__":
    main()
