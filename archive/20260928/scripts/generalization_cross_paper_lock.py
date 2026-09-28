#!/usr/bin/env python3
"""Audit the fixed F3 architecture and local assets for matched-paper retraining.

This is a preparation gate, not a benchmark evaluator. It never opens external
test labels or silently treats a ReactZyme-trained checkpoint as retrained.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REACTZYME = ROOT / "runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry/reaction_smi/train.yaml"
ENZYMECAGE = ROOT / "runs/enzymecage_f3_seed42/configs/train.yaml"
CLIPZYME = ROOT / "data/external/cyp_specificity_2026/clipzyme_official"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def pair_inventory(path: Path) -> dict:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"reaction_id", "protein_id"} <= set(rows[0]):
        raise ValueError(f"Invalid pair file: {path}")
    pairs = {(row["reaction_id"], row["protein_id"]) for row in rows}
    if len(pairs) != len(rows):
        raise ValueError(f"Duplicate pair rows: {path}")
    return {"source": record(path), "pairs": len(pairs),
            "reactions": len({r for r, _ in pairs}),
            "proteins": len({e for _, e in pairs}), "pair_set": pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Lock files are immutable; choose a fresh output path")
    react = yaml.safe_load(REACTZYME.read_text())
    cage = yaml.safe_load(ENZYMECAGE.read_text())
    if react["model"] != cage["model"]:
        raise ValueError("Local EnzymeCAGE and ReactZyme F3 model architectures differ")
    fixed_training_keys = ("max_epochs", "precision", "float32_matmul_precision",
                           "learning_rate", "weight_decay", "lambda_residue", "loss",
                           "devices", "strategy", "early_stopping")
    for key in fixed_training_keys:
        if react["training"].get(key) != cage["training"].get(key):
            raise ValueError(f"Local EnzymeCAGE and ReactZyme F3 training recipes differ at {key}")
    train_path = Path(cage["data"]["train_pairs_path"])
    valid_path = Path(cage["data"]["validation_pairs_path"])
    train = pair_inventory(train_path)
    valid = pair_inventory(valid_path)
    if train["pair_set"] & valid["pair_set"]:
        raise ValueError("Training and validation share an association")
    train.pop("pair_set")
    valid.pop("pair_set")
    clip_files = {name: CLIPZYME / "files" / name for name in (
        "enzymemap.json", "cached_enzymemap.p", "clipzyme_screening_set.p", "uniprot2sequence.p")}
    output = {
        "schema": "cross_paper_fixed_architecture_preparation_v3",
        "purpose": "Lock model/recipe identity and inventory inputs before matched-paper retraining; no test evaluation",
        "implementation": record(Path(__file__)),
        "f3_model_sha256": canonical_sha(react["model"]),
        "f3_model": react["model"],
        "base_training_sha256": canonical_sha({key: react["training"].get(key) for key in fixed_training_keys}),
        "base_training": {key: react["training"].get(key) for key in fixed_training_keys},
        "same_model_and_base_training_in_local_enzymecage_config": True,
        "source_configs": {"reactzyme": record(REACTZYME), "enzymecage": record(ENZYMECAGE)},
        "implementation_sources": {name: record(ROOT / name) for name in (
            "horizyn/model.py", "horizyn/protein_pooling_lightning_module.py",
            "horizyn/data_module.py", "horizyn/losses.py",
            "scripts/train_protein_pooling.py", "horizyn/generalization_residual.py",
            "horizyn/semantic_anchors.py", "horizyn/generalization_phase2.py",
            "horizyn/generalization_morgan.py", "scripts/generalization_full_graph.py")},
        "enzymecage_local": {"train": train, "validation": valid,
                               "positive_edges_only": True,
                               "existing_seed42_checkpoint_is_historical_not_new_retraining": True},
        "fgw_enzymemap_local": {"clipzyme_code": record(CLIPZYME / "README.md"),
                                "required_release_assets": {name: {"path": str(path), "present": path.is_file()}
                                                            for name, path in clip_files.items()},
                                "ready_for_matched_retraining": all(path.is_file() for path in clip_files.values())},
        "method_extension": {
            "independent_towers": True,
            "residual": {"hidden": 1024, "scale": 0.2, "steps": 1000, "validate_every": 25,
                         "temperature": 0.07, "identity_weight": 2.0, "optimizer": "AdamW",
                         "learning_rate": 0.0001, "weight_decay": 0.001,
                         "enzyme_weighting": "uniform", "max_directional_validation_drop": 0.005},
            "smooth_anchor": {"weight": 0.25, "reaction_kernel": "exponential",
                              "reaction_temperature": 0.03, "enzyme_temperature": 0.03,
                              "enzyme_training_neighbors": 32, "reaction_training_neighbors": "all"},
            "density_gate": {"rule": "f3_both_q25_cap1", "lower_training_quantile": 0.25,
                             "upper_training_quantile": 0.95,
                             "protein_calibration_size": 4096,
                             "protein_calibration_seed": 20260920},
            "morgan_reaction_anchor": {"radius": 3, "bits": 4096, "count_fingerprint": True,
                                       "chirality": True, "reaction_anchor_blend": 1.0},
            "all_dictionaries_fit_on_target_training_split_only": True,
        },
        "test_labels_read": False,
        "direct_cross_paper_claim_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".partial.json")
    temp.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temp.replace(args.output)
    print(json.dumps({"output": str(args.output), "f3_model_sha256": output["f3_model_sha256"],
                      "enzymecage_train_pairs": train["pairs"],
                      "fgw_assets_ready": output["fgw_enzymemap_local"]["ready_for_matched_retraining"]}))


if __name__ == "__main__":
    main()
