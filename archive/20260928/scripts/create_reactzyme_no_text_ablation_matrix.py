#!/usr/bin/env python3
"""Create the no-text ReactZyme ablation matrix and launch plans."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shlex
import stat
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "outputs/reactzyme_official_no_text_ablation_20260713"
DEFAULT_TRAIN_PYTHON = ROOT.parent / "env/bin/python"
DEFAULT_SETUP_PYTHON = ROOT.parent / ".capability-run-py/bin/python"
DEFAULT_WANDB_PROJECT = "horizyn-reactzyme-official-no-text-ablation-20260713"
DEFAULT_WANDB_ENTITY = "omnai"
DEFAULT_PROTOCOL_ROOT = ROOT / "data/revised_protocols/reactzyme_official"
DEFAULT_FEATURE_ROOT = DEFAULT_PROTOCOL_ROOT / "features"

SPLITS = {
    "time": {
        "template": "configs/reactzyme_clean_time_prott5_sleec_reactiont5v2_unimol2_chiro_observed_mlnce.yaml",
        "tag": "time-split",
    },
    "enzyme_smi": {
        "template": "configs/reactzyme_clean_enzyme_smi_prott5_sleec_reactiont5v2_unimol2_chiro_observed_mlnce.yaml",
        "tag": "enzyme-smi-split",
    },
    "reaction_smi": {
        "template": "configs/reactzyme_clean_reaction_smi_prott5_sleec_reactiont5v2_unimol2_chiro_observed_mlnce.yaml",
        "tag": "reaction-smi-split",
    },
}

VARIANTS: list[dict[str, Any]] = [
    {
        "id": "V0_observed_control",
        "label": "V0",
        "description": "Observed-pair clean control.",
        "loss_name": "FullBatchMLNCELoss",
        "positive_pair_source": "observed_pairs",
        "hard_negative": False,
        "biofp": False,
        "biofp_pretrain": False,
        "r2e_weighted": False,
        "structure": False,
    },
    {
        "id": "V1_allknown",
        "label": "V1",
        "description": "All-known multi-positive contrastive control.",
        "loss_name": "FullBatchMLNCELoss",
        "positive_pair_source": "all_known_in_batch",
        "hard_negative": False,
        "biofp": False,
        "biofp_pretrain": False,
        "r2e_weighted": False,
        "structure": False,
    },
    {
        "id": "V2_allknown_hardneg",
        "label": "V2",
        "description": "All-known positives plus mined R->E hard-negative batches.",
        "loss_name": "FullBatchMLNCELoss",
        "positive_pair_source": "all_known_in_batch",
        "hard_negative": True,
        "biofp": False,
        "biofp_pretrain": False,
        "r2e_weighted": False,
        "structure": False,
    },
    {
        "id": "V3_biofp_inline",
        "label": "V3",
        "description": "Hard-negative retrieval with weak inline enzyme BioFP branch.",
        "loss_name": "FullBatchMLNCELoss",
        "positive_pair_source": "all_known_in_batch",
        "hard_negative": True,
        "biofp": True,
        "biofp_pretrain": False,
        "r2e_weighted": False,
        "structure": False,
    },
    {
        "id": "V4_biofp_pretrained",
        "label": "V4",
        "description": "V3 initialized from same-split enzyme BioFP pretraining.",
        "loss_name": "FullBatchMLNCELoss",
        "positive_pair_source": "all_known_in_batch",
        "hard_negative": True,
        "biofp": True,
        "biofp_pretrain": True,
        "r2e_weighted": False,
        "structure": False,
    },
    {
        "id": "V5_r2e_weighted",
        "label": "V5",
        "description": "V4 with R->E-heavy multi-alignment loss and rank hard-negative term.",
        "loss_name": "MultiAlignmentRetrievalLoss",
        "positive_pair_source": "all_known_in_batch",
        "hard_negative": True,
        "biofp": True,
        "biofp_pretrain": True,
        "r2e_weighted": True,
        "structure": False,
    },
    {
        "id": "V6_structure_regularized",
        "label": "V6",
        "description": "V5 with light reaction/reaction, enzyme/enzyme, and GW structure terms.",
        "loss_name": "MultiAlignmentRetrievalLoss",
        "positive_pair_source": "all_known_in_batch",
        "hard_negative": True,
        "biofp": True,
        "biofp_pretrain": True,
        "r2e_weighted": True,
        "structure": True,
    },
]

CAPABILITY_DIR = ROOT / "data/processed/capability_features/train_exact_rhea_reconstructed"
CAPABILITY_REACTION_FEATURES = CAPABILITY_DIR / "reaction_features.parquet"
ENZYME_COFACTOR_LABELS = CAPABILITY_DIR / "enzyme_cofactor_labels_enhanced.csv"
COFACTOR_DICTIONARY = CAPABILITY_DIR / "cofactor_dictionary.csv"
EC_SOURCE = ROOT / "data/standardized/retrieval_training_source_collapse/hyperbolic_ec_labels/nr90_valid_prefix_ec_labels.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_TRAIN_PYTHON)
    parser.add_argument("--setup-python-bin", type=Path, default=DEFAULT_SETUP_PYTHON)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--retrieval-batch-size", type=int, default=128)
    parser.add_argument("--validation-retrieval-batch-size", type=int, default=512)
    parser.add_argument("--validation-interval-steps", type=int, default=5000)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hard-negative-max-per-query", type=int, default=256)
    parser.add_argument("--hard-negative-anchors-per-batch", type=int, default=48)
    parser.add_argument("--hard-negative-positives-per-query", type=int, default=2)
    parser.add_argument("--hard-negative-negatives-per-query", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return str(path)


PROTOCOL_ROOT = DEFAULT_PROTOCOL_ROOT
FEATURE_ROOT = DEFAULT_FEATURE_ROOT


def split_dir(split: str) -> Path:
    return PROTOCOL_ROOT / split


def feature_path(split: str, subset: str, modality: str) -> Path:
    return FEATURE_ROOT / split / subset / f"{modality}.h5"


def setup_paths(run_root: Path, split: str) -> dict[str, Path]:
    base = run_root / "data" / split
    reaction_feature_dir = base / "capability/reaction_features"
    return {
        "reaction_feature_dir": reaction_feature_dir,
        "reaction_features": reaction_feature_dir / "reaction_features.parquet",
        "biofp_npz": base / "biofp/enzyme_biofp_soft_targets.npz",
        "biofp_vocab": base / "biofp/enzyme_biofp_vocab.json",
        "hardneg_json": base / "hard_negatives/r2e_hard_negative_pools.json",
        "hardneg_parquet": base / "hard_negatives/r2e_hard_negative_pools.parquet",
        "hardneg_report": base / "hard_negatives/r2e_hard_negative_report.json",
        "enzyme_ec_labels": base / "ec/enzyme_ec_labels_train_only.csv",
        "ec_report": base / "ec/ec_label_report.json",
    }


def require_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def assert_no_text_fields(config: dict[str, Any], path: str = "") -> None:
    blocked_keys = {
        "protein_text_vectors_path",
        "protein_text_metadata_path",
        "text_vector_missing_policy",
        "text_vector",
        "text_fusion_dim",
        "text_vector_dim",
    }
    if isinstance(config, dict):
        for key, value in config.items():
            key_path = f"{path}.{key}" if path else str(key)
            if key in blocked_keys:
                raise ValueError(f"Generated no-text config still contains {key_path}")
            assert_no_text_fields(value, key_path)
    elif isinstance(config, list):
        for idx, value in enumerate(config):
            assert_no_text_fields(value, f"{path}[{idx}]")


def strip_text_fields(config: dict[str, Any]) -> None:
    data = config.get("data", {})
    model = config.get("model", {})
    for key in (
        "protein_text_vectors_path",
        "protein_text_metadata_path",
        "text_vector_missing_policy",
    ):
        data.pop(key, None)
    for key in ("text_vector", "text_vector_dim", "text_fusion_dim"):
        model.pop(key, None)


def unique_tags(*tag_groups: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for group in tag_groups:
        for tag in group:
            value = str(tag)
            if value not in seen:
                seen.add(value)
                result.append(value)
    return result


def apply_split_paths(config: dict[str, Any], split: str, subset: str) -> None:
    data = config["data"]
    base = split_dir(split)
    data["train_pairs_path"] = rel(base / "train_pairs.csv")
    data["train_reactions_path"] = rel(base / "train_rxns.csv")
    data["test_pairs_path"] = rel(base / f"{subset}_pairs.csv")
    data["test_reactions_path"] = rel(base / f"{subset}_rxns.csv")
    data["validation_reaction_t5v2_embeds_path"] = rel(feature_path(split, subset, "reactiont5v2"))
    data["validation_reaction_unimol2_embeds_path"] = rel(feature_path(split, subset, "unimol2"))
    data["validation_reaction_chiro_embeds_path"] = rel(feature_path(split, subset, "chiro"))
    data["validation_retrieval_candidate_set"] = "custom"
    data["validation_retrieval_candidate_ids_path"] = rel(base / "candidate_ids.txt")

    training = config["training"]
    training["validation_retrieval_candidate_set"] = "custom"
    training["validation_retrieval_candidate_ids_path"] = rel(base / "candidate_ids.txt")


def configure_common(
    config: dict[str, Any],
    *,
    run_root: Path,
    split: str,
    variant: dict[str, Any],
    args: argparse.Namespace,
    test: bool = False,
) -> None:
    split_tag = SPLITS[split]["tag"]
    variant_id = variant["id"]
    run_id = f"{split}_{variant_id}"
    run_name = f"reactzyme-no-text-{split}-{variant_id}"

    strip_text_fields(config)
    apply_split_paths(config, split, "test")

    logging = config["logging"]
    suffix = "test_configs" if test else "train"
    logging["log_dir"] = str(run_root / "logs" / split / variant_id / suffix)
    logging["checkpoint_dir"] = str(run_root / "checkpoints" / split / variant_id)
    logging["checkpoint_monitor"] = "val/reaction_to_enzyme/mrr"
    logging["checkpoint_mode"] = "max"
    logging["save_every_n_train_steps"] = 1000
    logging["log_every_n_steps"] = 10
    logging["wandb"] = {
        "enabled": (not test) and args.wandb_mode != "disabled",
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "run_name": run_name,
        "mode": args.wandb_mode,
        "tags": unique_tags(
            [
                "horizyn",
                "reactzyme-no-text-ablation",
                "official-reactzyme-protocol",
                split_tag,
                "prott5",
                "sleec-guided-attention",
                "reactiont5v2",
                "unimol2",
                "chiro",
                variant["label"],
                variant_id,
            ],
            ["hard-negative-sampler"] if variant["hard_negative"] else ["standard-sampler"],
            ["inline-biofp"] if variant["biofp"] else ["no-biofp"],
            ["biofp-pretrained"] if variant["biofp_pretrain"] else [],
            ["r2e-weighted-loss"] if variant["r2e_weighted"] else [],
            ["structure-regularized"] if variant["structure"] else [],
        ),
        "log_model": False,
    }

    data = config["data"]
    data["train_batch_size"] = args.train_batch_size
    data["retrieval_batch_size"] = args.retrieval_batch_size
    data["validation_retrieval_batch_size"] = args.validation_retrieval_batch_size
    data["num_workers"] = args.num_workers
    data["pin_memory"] = False
    data["reaction_use_chiro"] = True
    data["reaction_use_chirality"] = False
    data["reaction_use_chienn"] = False
    data["reaction_allow_missing_chiro"] = True
    data["reaction_allow_missing_chirality"] = False
    data["reaction_allow_missing_chienn"] = False

    model = config["model"]
    model["reaction_use_chiro"] = True
    model["reaction_use_chirality"] = False
    model["reaction_use_chienn"] = False
    model["reaction_chirality_name"] = "chiro"

    training = config["training"]
    training["max_epochs"] = args.max_epochs
    training["devices"] = 1
    training["accelerator"] = "gpu"
    training["strategy"] = "auto"
    training["validation_retrieval_metrics"] = True
    training["validation_retrieval_batch_size"] = args.validation_retrieval_batch_size
    training["validation_retrieval_directions"] = ["reaction_to_enzyme"]
    training["validation_interval_steps"] = args.validation_interval_steps
    training["num_sanity_val_steps"] = 0
    training["enable_progress_bar"] = False
    training["log_attention_stats"] = True
    training["attention_logging_interval"] = 10
    training["use_distributed_sampler"] = not bool(variant["hard_negative"])

    loss = training.setdefault("loss", {})
    loss.clear()
    loss.update(
        {
            "name": variant["loss_name"],
            "beta": 10.0,
            "learn_beta": False,
            "beta_min": 0.01,
            "beta_max": 100.0,
            "positive_pair_source": variant["positive_pair_source"],
        }
    )

    if variant["loss_name"] == "MultiAlignmentRetrievalLoss":
        loss.update(
            {
                "lambda_r2e": 0.85 if variant["r2e_weighted"] else 0.5,
                "lambda_e2r": 0.15 if variant["r2e_weighted"] else 0.5,
                "lambda_r2e_hard_neg": 0.25 if variant["r2e_weighted"] else 0.0,
                "r2e_hard_neg_top_k": 32 if variant["r2e_weighted"] else 0,
                "r2e_hard_neg_margin": 0.10 if variant["r2e_weighted"] else 0.0,
                "lambda_direction_gap": 0.05,
                "lambda_rr": 0.02 if variant["structure"] else 0.0,
                "lambda_ee": 0.02 if variant["structure"] else 0.0,
                "lambda_gw": 0.005 if variant["structure"] else 0.0,
                "tau_rr": 0.1,
                "tau_ee": 0.1,
                "tau_gw": 0.1,
                "ec_positive_policy": "hierarchical_weighted",
                "ec_min_shared_depth": 2,
                "gw_max_anchors": 512,
                "symmetric_gw": True,
                "apply_structure_terms_on_val": False,
            }
        )

    config["ablation"] = {
        "run_id": run_id,
        "split": split,
        "variant": variant_id,
        "description": variant["description"],
        "test_config": test,
        "no_enzyme_text": True,
    }
    assert_no_text_fields(config)


def configure_hard_negative(
    config: dict[str, Any],
    *,
    split: str,
    run_root: Path,
    args: argparse.Namespace,
) -> None:
    paths = setup_paths(run_root, split)
    data = config["data"]
    data["hard_negative_pools_path"] = str(paths["hardneg_json"])
    data["hard_negative_anchor_queries_per_batch"] = args.hard_negative_anchors_per_batch
    data["hard_negative_positives_per_query"] = args.hard_negative_positives_per_query
    data["hard_negative_negatives_per_query"] = args.hard_negative_negatives_per_query
    data["hard_negative_seed"] = args.seed


def configure_biofp(
    config: dict[str, Any],
    *,
    split: str,
    run_root: Path,
    pretrain: bool,
) -> None:
    paths = setup_paths(run_root, split)
    config["data"]["protein_biofp_targets_path"] = str(paths["biofp_npz"])
    config["data"]["protein_biofp_vocab_path"] = str(paths["biofp_vocab"])
    config["data"]["biofp_missing_policy"] = "zero_with_mask"
    config["model"]["enzyme_input_mode"] = "raw_mean_sleec_biofp_split"
    config["model"]["target_encoder_dims"] = [512, 512]
    config["model"]["biofp"] = {
        "center_dim": 8,
        "cofactor_dim": 16,
        "transition_dim": 16,
        "seq_dim": 384,
        "dim": 128,
        "hidden_dim": 512,
        "seq_weight": 0.90,
        "dropout": 0.10,
        "missing_policy": "zero_with_mask",
    }
    loss = config["training"]["loss"]
    loss["biofp_aux_weight"] = 0.03
    loss["biofp_center_weight"] = 0.45
    loss["biofp_transition_weight"] = 0.35
    loss["biofp_cofactor_weight"] = 0.20
    loss["biofp_confidence_cap"] = 8.0
    if pretrain:
        config["training"]["biofp_pretrain_checkpoint"] = str(
            run_root / "checkpoints" / split / "biofp_pretrain" / "last.ckpt"
        )


def configure_structure(config: dict[str, Any], *, split: str, run_root: Path) -> None:
    config["data"]["enzyme_ec_labels_path"] = str(setup_paths(run_root, split)["enzyme_ec_labels"])


def make_train_config(
    template: dict[str, Any],
    *,
    split: str,
    variant: dict[str, Any],
    run_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = copy.deepcopy(template)
    configure_common(cfg, run_root=run_root, split=split, variant=variant, args=args, test=False)
    if variant["hard_negative"]:
        configure_hard_negative(cfg, split=split, run_root=run_root, args=args)
    if variant["biofp"]:
        configure_biofp(cfg, split=split, run_root=run_root, pretrain=variant["biofp_pretrain"])
    if variant["structure"]:
        configure_structure(cfg, split=split, run_root=run_root)
    return cfg


def make_test_config(
    train_config: dict[str, Any],
    *,
    split: str,
    variant: dict[str, Any],
    run_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = copy.deepcopy(train_config)
    configure_common(cfg, run_root=run_root, split=split, variant=variant, args=args, test=True)
    if variant["hard_negative"]:
        configure_hard_negative(cfg, split=split, run_root=run_root, args=args)
    if variant["biofp"]:
        configure_biofp(cfg, split=split, run_root=run_root, pretrain=variant["biofp_pretrain"])
    if variant["structure"]:
        configure_structure(cfg, split=split, run_root=run_root)
    return cfg


def make_pretrain_config(
    template: dict[str, Any],
    *,
    split: str,
    run_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    paths = setup_paths(run_root, split)
    cfg = {
        "seed": args.seed,
        "logging": {
            "log_dir": str(run_root / "logs" / split / "biofp_pretrain"),
            "checkpoint_dir": str(run_root / "checkpoints" / split / "biofp_pretrain"),
            "wandb": {
                "enabled": args.wandb_mode != "disabled",
                "project": args.wandb_project,
                "entity": args.wandb_entity,
                "run_name": f"reactzyme-no-text-{split}-biofp-pretrain",
                "mode": args.wandb_mode,
                "tags": [
                    "horizyn",
                    "reactzyme-no-text-ablation",
                    SPLITS[split]["tag"],
                    "enzyme-biofp-pretrain",
                    "prott5",
                    "sleec-guided-attention",
                    "train-only-biofp",
                ],
                "log_model": False,
            },
        },
        "data": {
            "protein_residue_embeds_path": template["data"]["protein_residue_embeds_path"],
            "protein_biofp_targets_path": str(paths["biofp_npz"]),
            "biofp_missing_policy": "zero_with_mask",
            "residue_dim": template["data"].get("residue_dim", 1024),
            "max_protein_tokens": template["data"].get("max_protein_tokens", 1022),
            "protein_truncation": template["data"].get("protein_truncation", "ends_center"),
            "residue_in_memory": False,
        },
        "model": {
            "pooling": template["model"].get("pooling", "sleec_guided_attention"),
            "embedding_dim": 512,
            "biofp": {
                "center_dim": 8,
                "cofactor_dim": 16,
                "transition_dim": 16,
                "seq_dim": 384,
                "dim": 128,
                "hidden_dim": 512,
                "seq_weight": 0.90,
                "dropout": 0.10,
            },
            "sleec_pooling": copy.deepcopy(template["model"].get("sleec_pooling", {})),
        },
        "training": {
            "max_epochs": args.pretrain_epochs,
            "batch_size": 128,
            "learning_rate": 1.0e-4,
            "weight_decay": 0.01,
            "num_workers": args.num_workers,
            "pin_memory": False,
            "validation_fraction": 0.05,
            "checkpoint_monitor": "val/loss",
            "checkpoint_mode": "min",
            "precision": "32-true",
            "accelerator": "gpu",
            "devices": 1,
            "strategy": "auto",
            "gradient_clip_val": 1.0,
            "log_every_n_steps": 10,
            "enable_progress_bar": True,
            "biofp_center_weight": 0.45,
            "biofp_transition_weight": 0.35,
            "biofp_cofactor_weight": 0.20,
            "biofp_confidence_cap": 8.0,
        },
    }
    assert_no_text_fields(cfg)
    return cfg


def preflight(args: argparse.Namespace) -> None:
    paths: list[Path] = [
        args.python_bin,
        args.setup_python_bin,
        CAPABILITY_REACTION_FEATURES,
        ENZYME_COFACTOR_LABELS,
        COFACTOR_DICTIONARY,
        EC_SOURCE,
        ROOT / "scripts/train_protein_pooling.py",
        ROOT / "scripts/pretrain_enzyme_biofp_split.py",
        ROOT / "scripts/build_enzyme_biofp_soft_targets.py",
        ROOT / "scripts/mine_r2e_hard_negatives.py",
        ROOT / "scripts/evaluate_protein_pooling.py",
    ]
    for split, meta in SPLITS.items():
        base = split_dir(split)
        paths.extend(
            [
                ROOT / meta["template"],
                base / "train_pairs.csv",
                base / "test_pairs.csv",
                base / "train_rxns.csv",
                base / "test_rxns.csv",
                base / "candidate_ids.txt",
                feature_path(split, "train", "reactiont5v2"),
                feature_path(split, "train", "unimol2"),
                feature_path(split, "train", "chiro"),
                feature_path(split, "test", "reactiont5v2"),
                feature_path(split, "test", "unimol2"),
                feature_path(split, "test", "chiro"),
            ]
        )
    require_paths(paths)

    for split, meta in SPLITS.items():
        cfg = load_yaml(ROOT / meta["template"])
        sleec_checkpoint = cfg.get("model", {}).get("sleec_pooling", {}).get("checkpoint_path")
        residue_h5 = cfg.get("data", {}).get("protein_residue_embeds_path")
        if sleec_checkpoint:
            require_paths([Path(sleec_checkpoint)])
        if residue_h5:
            require_paths([ROOT / residue_h5])


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def shell_quote(value: Any) -> str:
    return shlex.quote(str(value))


def normalize_executable_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return (Path.cwd() / expanded).resolve()


def write_runtime_env(path: Path, args: argparse.Namespace, run_root: Path) -> None:
    gpus = " ".join(token.strip() for token in args.gpus.split(",") if token.strip())
    values = {
        "ROOT": ROOT,
        "RUN_ROOT": run_root,
        "PYTHON_BIN": args.python_bin,
        "SETUP_PYTHON_BIN": args.setup_python_bin,
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_MODE": args.wandb_mode,
        "MAX_PARALLEL": args.max_parallel,
        "GPUS": gpus,
        "CAPABILITY_REACTION_FEATURES": CAPABILITY_REACTION_FEATURES,
        "ENZYME_COFACTOR_LABELS": ENZYME_COFACTOR_LABELS,
        "COFACTOR_DICTIONARY": COFACTOR_DICTIONARY,
        "EC_SOURCE": EC_SOURCE,
        "HARD_NEGATIVE_MAX_PER_QUERY": args.hard_negative_max_per_query,
        "SEED": args.seed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{key}={shell_quote(value)}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.run_root = args.run_root.expanduser().resolve()
    args.protocol_root = args.protocol_root.expanduser().resolve()
    args.features_root = args.features_root.expanduser().resolve()
    args.python_bin = normalize_executable_path(args.python_bin)
    args.setup_python_bin = normalize_executable_path(args.setup_python_bin)
    global PROTOCOL_ROOT, FEATURE_ROOT
    PROTOCOL_ROOT = args.protocol_root
    FEATURE_ROOT = args.features_root
    if args.max_parallel <= 0:
        raise ValueError("--max-parallel must be positive")
    gpus = [token.strip() for token in args.gpus.split(",") if token.strip()]
    if not gpus:
        raise ValueError("--gpus must list at least one GPU")
    if args.max_parallel > len(gpus):
        raise ValueError("--max-parallel cannot exceed the number of listed GPUs")
    if not args.skip_preflight:
        preflight(args)

    run_root = args.run_root
    config_root = run_root / "configs"
    run_root.mkdir(parents=True, exist_ok=True)

    setup_rows: list[dict[str, Any]] = []
    pretrain_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    manifest_runs: list[dict[str, Any]] = []

    templates = {
        split: load_yaml(ROOT / meta["template"])
        for split, meta in SPLITS.items()
    }

    for split in SPLITS:
        paths = setup_paths(run_root, split)
        base = split_dir(split)
        setup_rows.append(
            {
                "split": split,
                "train_pairs": rel(base / "train_pairs.csv"),
                "train_reactions": rel(base / "train_rxns.csv"),
                "candidate_ids": rel(base / "candidate_ids.txt"),
                "reaction_features": paths["reaction_features"],
                "biofp_npz": paths["biofp_npz"],
                "biofp_vocab": paths["biofp_vocab"],
                "hardneg_json": paths["hardneg_json"],
                "hardneg_parquet": paths["hardneg_parquet"],
                "hardneg_report": paths["hardneg_report"],
                "enzyme_ec_labels": paths["enzyme_ec_labels"],
                "ec_report": paths["ec_report"],
            }
        )
        pretrain_cfg = make_pretrain_config(
            templates[split],
            split=split,
            run_root=run_root,
            args=args,
        )
        pretrain_config_path = config_root / split / "biofp_pretrain.yaml"
        write_yaml(pretrain_config_path, pretrain_cfg)
        pretrain_rows.append(
            {
                "split": split,
                "run_id": f"{split}_biofp_pretrain",
                "config": pretrain_config_path,
                "checkpoint_dir": run_root / "checkpoints" / split / "biofp_pretrain",
                "log_path": run_root / "logs" / split / "biofp_pretrain/stdout.log",
                "wandb_run_name": pretrain_cfg["logging"]["wandb"]["run_name"],
            }
        )

        for variant in VARIANTS:
            train_cfg = make_train_config(
                templates[split],
                split=split,
                variant=variant,
                run_root=run_root,
                args=args,
            )
            test_cfg = make_test_config(
                train_cfg,
                split=split,
                variant=variant,
                run_root=run_root,
                args=args,
            )
            variant_id = variant["id"]
            run_id = f"{split}_{variant_id}"
            train_config_path = config_root / split / f"{variant_id}.yaml"
            test_config_path = config_root / split / f"{variant_id}.test.yaml"
            write_yaml(train_config_path, train_cfg)
            write_yaml(test_config_path, test_cfg)

            checkpoint_dir = run_root / "checkpoints" / split / variant_id
            log_path = run_root / "logs" / split / variant_id / "stdout.log"
            eval_json = run_root / "eval" / split / variant_id / "test_r2e.json"
            row = {
                "run_id": run_id,
                "split": split,
                "variant": variant_id,
                "label": variant["label"],
                "config": train_config_path,
                "test_config": test_config_path,
                "checkpoint_dir": checkpoint_dir,
                "log_path": log_path,
                "eval_json": eval_json,
                "wandb_run_name": train_cfg["logging"]["wandb"]["run_name"],
                "requires_pretrain": str(bool(variant["biofp_pretrain"])).lower(),
            }
            train_rows.append(row)
            manifest_runs.append(
                {
                    **{key: str(value) for key, value in row.items()},
                    "description": variant["description"],
                    "hard_negative": variant["hard_negative"],
                    "biofp": variant["biofp"],
                    "biofp_pretrain": variant["biofp_pretrain"],
                    "r2e_weighted": variant["r2e_weighted"],
                    "structure": variant["structure"],
                }
            )

    write_tsv(
        run_root / "setup_plan.tsv",
        [
            "split",
            "train_pairs",
            "train_reactions",
            "candidate_ids",
            "reaction_features",
            "biofp_npz",
            "biofp_vocab",
            "hardneg_json",
            "hardneg_parquet",
            "hardneg_report",
            "enzyme_ec_labels",
            "ec_report",
        ],
        setup_rows,
    )
    write_tsv(
        run_root / "pretrain_plan.tsv",
        ["split", "run_id", "config", "checkpoint_dir", "log_path", "wandb_run_name"],
        pretrain_rows,
    )
    write_tsv(
        run_root / "train_plan.tsv",
        [
            "run_id",
            "split",
            "variant",
            "label",
            "config",
            "test_config",
            "checkpoint_dir",
            "log_path",
            "eval_json",
            "wandb_run_name",
            "requires_pretrain",
        ],
        train_rows,
    )
    write_runtime_env(run_root / "runtime.env", args, run_root)

    launcher = ROOT / "scripts/run_reactzyme_no_text_ablation_matrix.sh"
    manifest = {
        "run_root": str(run_root),
        "created_by": str(Path(__file__).resolve()),
        "launcher": str(launcher),
        "runtime_env": str(run_root / "runtime.env"),
        "setup_plan": str(run_root / "setup_plan.tsv"),
        "pretrain_plan": str(run_root / "pretrain_plan.tsv"),
        "train_plan": str(run_root / "train_plan.tsv"),
        "splits": list(SPLITS),
        "variants": VARIANTS,
        "wandb": {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "mode": args.wandb_mode,
        },
        "scheduler": {
            "gpus": gpus,
            "max_parallel": args.max_parallel,
            "single_gpu_jobs": True,
            "setup_python": str(args.setup_python_bin),
            "training_python": str(args.python_bin),
        },
        "inputs": {
            "reactzyme_protocol_root": str(args.protocol_root),
            "reactzyme_feature_root": str(args.features_root),
            "reactzyme_data_recipe": (
                "Official per-protocol ReactZyme train/test splits. No random validation "
                "split and no cross-protocol pooling."
            ),
            "capability_reaction_features": str(CAPABILITY_REACTION_FEATURES),
            "cofactor_dictionary": str(COFACTOR_DICTIONARY),
            "enzyme_cofactor_labels": str(ENZYME_COFACTOR_LABELS),
            "ec_source": str(EC_SOURCE),
        },
        "pretrains": [
            {key: str(value) for key, value in row.items()}
            for row in pretrain_rows
        ],
        "runs": manifest_runs,
    }
    write_json(run_root / "manifest.json", manifest)

    wrapper_path = run_root / "run_reactzyme_no_text_ablation_matrix.sh"
    wrapper_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shell_quote(ROOT)}\n"
        f"exec {shell_quote(launcher)} {shell_quote(run_root)} \"$@\"\n",
        encoding="utf-8",
    )
    wrapper_path.chmod(wrapper_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
