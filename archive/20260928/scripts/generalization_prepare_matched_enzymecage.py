#!/usr/bin/env python3
"""Prepare a fresh EnzymeCAGE-data F3 run from the architecture-locked recipe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/enzymecage_f3_seed42/configs/train.yaml"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staging-receipt", type=Path,
                        help="Optional byte-verified local residue VDS; model inputs remain identical")
    args = parser.parse_args()
    lock = json.loads(args.architecture_lock.read_text())
    if lock["schema"] != "cross_paper_fixed_architecture_preparation_v3":
        raise ValueError("Unexpected architecture lock")
    if digest(SOURCE) != lock["source_configs"]["enzymecage"]["sha256"]:
        raise ValueError("EnzymeCAGE source config changed after lock")
    if not lock["same_model_and_base_training_in_local_enzymecage_config"]:
        raise ValueError("Architecture or training recipe differs")
    config = yaml.safe_load(SOURCE.read_text())
    staging = None
    if args.staging_receipt:
        staging = json.loads(args.staging_receipt.read_text())
        if (staging.get("schema") != "enzymecage_local_residue_staging_v1"
                or staging.get("model_input_values_changed") is not False
                or Path(staging["source_vds"]).resolve()
                   != Path(config["data"]["protein_residue_embeds_path"]).resolve()
                or digest(Path(staging["source_vds"])) != staging["source_vds_sha256"]
                or digest(Path(staging["staged_vds"])) != staging["staged_vds_sha256"]):
            raise ValueError("Invalid or changed local residue staging receipt")
        config["data"]["protein_residue_embeds_path"] = staging["staged_vds"]
    target = args.output.resolve()
    if target.exists():
        raise FileExistsError("Fresh matched run required; refusing to overwrite")
    (target / "configs").mkdir(parents=True)
    config["logging"]["log_dir"] = str(target / "logs")
    config["logging"]["checkpoint_dir"] = str(target / "checkpoints")
    config["logging"]["wandb"]["enabled"] = False
    config["ablation"]["run_id"] = "fixed_architecture_enzymecage_retrain_20260920"
    config["ablation"]["description"] = "Fresh F3 base retrain; fixed architecture and target train/validation only"
    prepared = target / "configs/train.yaml"
    prepared.write_text(yaml.safe_dump(config, sort_keys=False))
    report = {
        "schema": "matched_enzymecage_f3_preparation_v1",
        "architecture_lock": {"path": str(args.architecture_lock.resolve()), "sha256": digest(args.architecture_lock)},
        "source_config": {"path": str(SOURCE), "sha256": digest(SOURCE)},
        "prepared_config": {"path": str(prepared), "sha256": digest(prepared)},
        "model_matches_lock": hashlib.sha256(json.dumps(config["model"], sort_keys=True,
                                                    separators=(",", ":")).encode()).hexdigest()
                              == lock["f3_model_sha256"],
        "target_training_pairs": lock["enzymecage_local"]["train"]["pairs"],
        "external_test_labels_opened": False,
        "residue_staging": ({"receipt_path": str(args.staging_receipt.resolve()),
                              "receipt_sha256": digest(args.staging_receipt),
                              "identical_input_values": True} if staging else None),
        "status": "prepared; no training performed by this script",
    }
    if not report["model_matches_lock"]:
        raise ValueError("Prepared model mismatches architecture lock")
    (target / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"run": str(target), "config": str(prepared), "model_matches_lock": True}))


if __name__ == "__main__":
    main()
