#!/usr/bin/env python3
"""Create the two-GPU F3 no-EC biological-supervision campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_TRAIN = ROOT / "configs/benchmarks/reactzyme_f3_cluster_proxy_v1.yaml"
BASE_EVAL = ROOT / "runs/reactzyme_f3_reaction_smi_paper_v2_4gpu/configs/test.yaml"
PROTOCOL_ROOT = (
    ROOT / "data/revised_protocols/reactzyme_reaction_cluster_validation_v1/similarity_0p85"
)
REACTION_FEATURE_ROOT = ROOT / "data/revised_protocols/reactzyme_official/features/reaction_smi"
CHEMISTRY_ROOT = PROTOCOL_ROOT / "features/reaction_set"
RUN_ROOT = ROOT / "runs/reactzyme_f3_biological_no_ec_v1"
TARGET_ROOT = RUN_ROOT / "data/reaction_smi/biofp"
TARGETS = TARGET_ROOT / "enzyme_biofp_soft_targets.npz"
VOCAB = TARGET_ROOT / "enzyme_biofp_vocab.json"
PROJECT = "horizyn-reactzyme-f3-biological-no-ec"
MONITOR = "val/mean_bidirectional_reactzyme_mrr"
MODALITY_FILES = {
    "reaction_t5v2": "reactiont5v2.h5",
    "reaction_unimol2": "unimol2.h5",
    "reaction_chiro": "chiro.h5",
}

VARIANTS: dict[str, dict[str, Any]] = {
    "F3_no_ec": {
        "label": "F3-noEC",
        "families": {},
        "description": "No-EC F3 control with retrieval supervision only.",
    },
    "F3M_no_ec": {
        "label": "F3-M-noEC",
        "families": {"mechanism": 1.0},
        "description": "No-EC F3 with joint mechanism supervision.",
    },
    "F3C_no_ec": {
        "label": "F3-C-noEC",
        "families": {"cofactor": 1.0},
        "description": "No-EC F3 with joint cofactor supervision.",
    },
    "F3MC_no_ec": {
        "label": "F3-MC-noEC",
        "families": {"mechanism": 0.65, "cofactor": 0.35},
        "description": "No-EC F3 with joint mechanism and cofactor supervision.",
    },
    "F3MC_align_no_ec": {
        "label": "F3-MC-XT-noEC",
        "families": {"mechanism": 0.65, "cofactor": 0.35},
        "alignment_families": {"mechanism": 0.65, "cofactor": 0.35},
        "description": (
            "No-EC F3 with mechanism/cofactor supervision and gated cross-tower "
            "factor alignment."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aux-weight", type=float, required=True)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=30)
    return parser.parse_args()


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


def remove_ec_block(config: dict[str, Any]) -> None:
    model = config["model"]
    fusion = model["enzyme_block_fusion"]
    expected_dims = {"core": 288, "site": 96, "mechanism": 64, "cofactor": 32, "ec": 32}
    expected_weights = {
        "core": 0.55,
        "site": 0.20,
        "mechanism": 0.12,
        "cofactor": 0.08,
        "ec": 0.05,
    }
    if fusion["dims"] != expected_dims or fusion["weights"] != expected_weights:
        raise ValueError("Source F3 enzyme layout changed; refusing an unaudited rewrite")
    fusion["dims"] = {"core": 320, "site": 96, "mechanism": 64, "cofactor": 32}
    fusion["weights"] = {"core": 0.60, "site": 0.20, "mechanism": 0.12, "cofactor": 0.08}
    model.pop("hyperbolic_encoder", None)


def configure_logging(config: dict[str, Any], variant: str, subset: str) -> None:
    metadata = VARIANTS[variant]
    logging = config["logging"]
    logging.update(
        {
            "log_dir": str(RUN_ROOT / "logs" / variant / subset),
            "checkpoint_dir": str(RUN_ROOT / "checkpoints" / variant / "train"),
            "checkpoint_monitor": MONITOR,
            "checkpoint_mode": "max",
            "checkpoint_on_validation_end": True,
            "save_top_k": 3,
        }
    )
    logging.setdefault("wandb", {}).update(
        {
            "enabled": subset == "train",
            "project": PROJECT,
            "entity": "omnai",
            "run_name": f"{variant}-reaction-smi-seed42",
            "mode": "online",
            "tags": [
                "reactzyme",
                "reaction-smi",
                "reaction-cluster-validation-0p85",
                "F3",
                "no-ec",
                metadata["label"],
                "two-gpu",
                "global-batch-2048",
                "bidirectional-selection",
                "seed-42",
            ],
            "log_model": False,
        }
    )


def configure_training_common(config: dict[str, Any], *, max_epochs: int) -> None:
    data = config["data"]
    training = config["training"]
    data["train_batch_size"] = 1024
    training.update(
        {
            "max_epochs": int(max_epochs),
            "devices": 2,
            "accelerator": "gpu",
            "strategy": "ddp_find_unused_parameters_true",
            "validation_retrieval_directions": ["reaction_to_enzyme", "enzyme_to_reaction"],
            "check_val_every_n_epoch": 1,
        }
    )
    training.setdefault("early_stopping", {}).update(
        {
            "enabled": True,
            "monitor": MONITOR,
            "mode": "max",
            "patience": 5,
            "min_delta": 0.0001,
        }
    )
    if int(data["train_batch_size"]) * int(training["devices"]) != 2048:
        raise ValueError("The audited global training batch must remain 2048")


def configure_train_data(config: dict[str, Any]) -> None:
    data = config["data"]
    data.update(
        {
            "train_pairs_path": str(PROTOCOL_ROOT / "train_pairs.csv"),
            "train_reactions_path": str(PROTOCOL_ROOT / "train_rxns.csv"),
            "validation_pairs_path": str(PROTOCOL_ROOT / "validation_pairs.csv"),
            "validation_reactions_path": str(PROTOCOL_ROOT / "validation_rxns.csv"),
            "validation_retrieval_candidate_ids_path": str(
                PROTOCOL_ROOT / "validation_candidate_ids.txt"
            ),
            "train_reaction_chemistry_vectors_path": str(
                CHEMISTRY_ROOT / "train_reaction_set_features.npz"
            ),
            "validation_reaction_chemistry_vectors_path": str(
                CHEMISTRY_ROOT / "validation_reaction_set_features.npz"
            ),
        }
    )
    for prefix, filename in MODALITY_FILES.items():
        data[f"train_{prefix}_embeds_path"] = str(REACTION_FEATURE_ROOT / "train" / filename)
        data[f"validation_{prefix}_embeds_path"] = str(REACTION_FEATURE_ROOT / "train" / filename)
    config["training"]["validation_retrieval_candidate_ids_path"] = str(
        PROTOCOL_ROOT / "validation_candidate_ids.txt"
    )


def configure_eval_data(config: dict[str, Any], subset: str) -> None:
    if subset not in {"validation", "test"}:
        raise ValueError(f"Unsupported evaluation subset: {subset}")
    feature_subset = "train" if subset == "validation" else "test"
    data = config["data"]
    data.update(
        {
            "train_pairs_path": str(PROTOCOL_ROOT / "train_pairs.csv"),
            "train_reactions_path": str(PROTOCOL_ROOT / "train_rxns.csv"),
            "test_pairs_path": str(PROTOCOL_ROOT / f"{subset}_pairs.csv"),
            "test_reactions_path": str(PROTOCOL_ROOT / f"{subset}_rxns.csv"),
            "validation_retrieval_candidate_ids_path": str(
                PROTOCOL_ROOT / f"{subset}_candidate_ids.txt"
            ),
            "train_reaction_chemistry_vectors_path": str(
                CHEMISTRY_ROOT / "train_reaction_set_features.npz"
            ),
            "validation_reaction_chemistry_vectors_path": str(
                CHEMISTRY_ROOT / f"{subset}_reaction_set_features.npz"
            ),
            "reaction_chemistry_vectors_path": str(
                CHEMISTRY_ROOT / f"{subset}_reaction_set_features.npz"
            ),
        }
    )
    for prefix, filename in MODALITY_FILES.items():
        data[f"train_{prefix}_embeds_path"] = str(REACTION_FEATURE_ROOT / "train" / filename)
        data[f"validation_{prefix}_embeds_path"] = str(
            REACTION_FEATURE_ROOT / feature_subset / filename
        )
        data[f"{prefix}_embeds_path"] = str(REACTION_FEATURE_ROOT / feature_subset / filename)
    config["training"].update(
        {
            "devices": 1,
            "strategy": "auto",
            "validation_retrieval_candidate_ids_path": str(
                PROTOCOL_ROOT / f"{subset}_candidate_ids.txt"
            ),
        }
    )


def configure_supervision(
    config: dict[str, Any],
    *,
    families: dict[str, float],
    aux_weight: float,
    alignment_families: dict[str, float],
    alignment_weight: float,
) -> None:
    config["training"]["loss"].update(
        {
            "biofp_aux_weight": float(aux_weight) if families else 0.0,
            "biofp_family_weights": families,
            "biofp_confidence_cap": 1.0,
            "cross_tower_alignment_weight": (
                float(alignment_weight) if alignment_families else 0.0
            ),
            "cross_tower_alignment_family_weights": dict(alignment_families),
        }
    )
    if families:
        config["data"].update(
            {
                "protein_biofp_targets_path": str(TARGETS),
                "protein_biofp_vocab_path": str(VOCAB),
                "biofp_missing_policy": "zero_with_mask",
            }
        )
    else:
        for key in (
            "protein_biofp_targets_path",
            "protein_biofp_vocab_path",
            "biofp_missing_policy",
        ):
            config["data"].pop(key, None)


def configure_ablation(
    config: dict[str, Any],
    variant: str,
    subset: str,
    aux_weight: float,
    alignment_weight: float,
) -> None:
    metadata = VARIANTS[variant]
    alignment_families = metadata.get("alignment_families", {})
    config["ablation"] = {
        "campaign": "reactzyme_f3_biological_no_ec_v1",
        "run_id": f"{variant}_reaction_smi_seed42",
        "variant": variant,
        "label": metadata["label"],
        "split": "reaction_smi",
        "seed": 42,
        "evaluation_subset": subset,
        "description": metadata["description"],
        "validation_protocol": "reaction_cluster_similarity_0p85",
        "test_protocol": "paper_test_candidates",
        "checkpoint_monitor": MONITOR,
        "global_batch_size": 2048,
        "devices": 2 if subset == "train" else 1,
        "ec_block": False,
        "biological_pretraining": False,
        "biofp_aux_weight": float(aux_weight) if metadata["families"] else 0.0,
        "cross_tower_alignment_weight": (float(alignment_weight) if alignment_families else 0.0),
        "cross_tower_alignment_family_weights": dict(alignment_families),
    }


def make_train(
    base: dict[str, Any],
    variant: str,
    *,
    aux_weight: float,
    alignment_weight: float,
    max_epochs: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    remove_ec_block(config)
    configure_logging(config, variant, "train")
    configure_train_data(config)
    configure_training_common(config, max_epochs=max_epochs)
    if VARIANTS[variant].get("alignment_families"):
        # Compare the new objective at a fixed training budget; auxiliary
        # losses can improve after retrieval MRR temporarily plateaus.
        config["training"]["early_stopping"]["enabled"] = False
    configure_supervision(
        config,
        families=VARIANTS[variant]["families"],
        aux_weight=aux_weight,
        alignment_families=VARIANTS[variant].get("alignment_families", {}),
        alignment_weight=alignment_weight,
    )
    configure_ablation(config, variant, "train", aux_weight, alignment_weight)
    return config


def make_eval(
    base: dict[str, Any],
    variant: str,
    subset: str,
    *,
    aux_weight: float,
    alignment_weight: float,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    remove_ec_block(config)
    configure_logging(config, variant, subset)
    configure_eval_data(config, subset)
    configure_supervision(
        config,
        families=VARIANTS[variant]["families"],
        aux_weight=aux_weight,
        alignment_families=VARIANTS[variant].get("alignment_families", {}),
        alignment_weight=alignment_weight,
    )
    configure_ablation(config, variant, subset, aux_weight, alignment_weight)
    return config


def main() -> None:
    args = parse_args()
    if args.aux_weight <= 0 or args.alignment_weight <= 0 or args.max_epochs <= 0:
        raise ValueError("--aux-weight, --alignment-weight, and --max-epochs must be positive")
    required = [BASE_TRAIN, BASE_EVAL, PROTOCOL_ROOT / "manifest.json", TARGETS, VOCAB]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing campaign inputs: {missing}")

    train_base = load_yaml(BASE_TRAIN)
    eval_base = load_yaml(BASE_EVAL)
    config_manifest: dict[str, dict[str, str]] = {}
    for variant in VARIANTS:
        config_dir = RUN_ROOT / "configs" / variant
        config_dir.mkdir(parents=True, exist_ok=True)
        payloads = {
            "train": make_train(
                train_base,
                variant,
                aux_weight=args.aux_weight,
                alignment_weight=args.alignment_weight,
                max_epochs=args.max_epochs,
            ),
            "validation": make_eval(
                eval_base,
                variant,
                "validation",
                aux_weight=args.aux_weight,
                alignment_weight=args.alignment_weight,
            ),
            "test": make_eval(
                eval_base,
                variant,
                "test",
                aux_weight=args.aux_weight,
                alignment_weight=args.alignment_weight,
            ),
        }
        config_manifest[variant] = {}
        for subset, payload in payloads.items():
            path = config_dir / f"{subset}.yaml"
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            config_manifest[variant][subset] = str(path)

    protocol_manifest = PROTOCOL_ROOT / "manifest.json"
    manifest = {
        "schema_version": "reactzyme_f3_biological_no_ec_v1",
        "campaign": "reactzyme_f3_biological_no_ec_v1",
        "scope": "reaction_smi_reaction_cluster_validation_0p85",
        "fresh_control_variant": "F3_no_ec",
        "architecture_change": {
            "removed_block": "ec",
            "dimension_reallocation": {"core": {"from": 288, "to": 320}},
            "weight_reallocation": {"core": {"from": 0.55, "to": 0.60}},
        },
        "base_train_config": {"path": str(BASE_TRAIN), "sha256": sha256(BASE_TRAIN)},
        "base_eval_config": {"path": str(BASE_EVAL), "sha256": sha256(BASE_EVAL)},
        "protocol_manifest": {"path": str(protocol_manifest), "sha256": sha256(protocol_manifest)},
        "biological_targets": {
            "npz": str(TARGETS),
            "npz_sha256": sha256(TARGETS),
            "vocab": str(VOCAB),
            "vocab_sha256": sha256(VOCAB),
        },
        "variants": VARIANTS,
        "configs": config_manifest,
        "devices_per_run": 2,
        "batch_size_per_gpu": 1024,
        "global_batch_size": 2048,
        "biological_pretraining": False,
        "auxiliary_weight": float(args.aux_weight),
        "cross_tower_alignment_weight": float(args.alignment_weight),
        "checkpoint_monitor": MONITOR,
        "evaluation_metric": "reactzyme_all_positive_mrr",
    }
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run_root": str(RUN_ROOT), "aux_weight": args.aux_weight}, indent=2))


if __name__ == "__main__":
    main()
