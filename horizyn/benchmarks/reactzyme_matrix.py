"""Generate ReactZyme paper-protocol ablation matrices."""

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

from horizyn.benchmarks.reactzyme_ablations import (
    MATRIX_NAMES,
    load_split_specs,
    load_variants,
)
from horizyn.benchmarks.reactzyme_protocol import collect_feature_coverage


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = ROOT / "outputs/reactzyme_latent_organization_paper_ablation_20260717"
DEFAULT_REPRESENTATION_RUN_ROOT = ROOT / "outputs/reactzyme_representation_paper_ablation_20260717"
DEFAULT_BIOLOGICAL_RUN_ROOT = ROOT / "runs/bio_aux_minimal_v1"
DEFAULT_REACTION_FEATURE_RUN_ROOT = ROOT / "runs/reactzyme_reaction_features_v1"
DEFAULT_TRAIN_PYTHON = ROOT.parent / "env/bin/python"
DEFAULT_SETUP_PYTHON = ROOT.parent / ".capability-run-py/bin/python"
DEFAULT_WANDB_PROJECT = "horizyn-reactzyme-paper-latent-organization-ablation"
DEFAULT_REPRESENTATION_WANDB_PROJECT = "horizyn-reactzyme-paper-representation-ablation"
DEFAULT_BIOLOGICAL_WANDB_PROJECT = "horizyn-reactzyme-biological-latent-v1"
DEFAULT_REACTION_FEATURE_WANDB_PROJECT = "horizyn-reactzyme-reaction-features-v1"
DEFAULT_WANDB_ENTITY = "omnai"
DEFAULT_PROTOCOL_ROOT = ROOT / "data/revised_protocols/reactzyme_paper"
DEFAULT_FEATURE_ROOT = ROOT / "data/revised_protocols/reactzyme_official/features"
EVALUATION_PROTOCOL = "paper_test_candidates"
SLEEC_CHECKPOINT = (
    ROOT / "checkpoints/SLEEC/sleec_stage1_prott5_uniref90_msa_4gpu_20260601_235157/best.ckpt"
)
HYPERBOLIC_CHECKPOINT = (
    ROOT
    / "checkpoints/hyperbolic_enzyme/hyperbolic-enzyme-prott5-lorentz-c0p25-b512-4gpu-20260601_150714/best.ckpt"
)
CAPABILITY_VECTORS = (
    ROOT
    / "outputs/enzyme_capability_pretrain/train_exact_rhea_reaction_demand_bio_composite_v3/reactzyme_official_hashed/enzyme_capability_vectors.npz"
)
CAPABILITY_METADATA = (
    ROOT
    / "outputs/enzyme_capability_pretrain/train_exact_rhea_reaction_demand_bio_composite_v3/reactzyme_official_hashed/enzyme_capability_metadata.json"
)
CAPABILITY_DIR = ROOT / "data/processed/capability_features/train_exact_rhea_reconstructed"
ENZYME_COFACTOR_LABELS = CAPABILITY_DIR / "enzyme_cofactor_labels_enhanced.csv"
COFACTOR_DICTIONARY = CAPABILITY_DIR / "cofactor_dictionary.csv"
EC_SOURCE = (
    ROOT
    / "data/standardized/retrieval_training_source_collapse/hyperbolic_ec_labels/nr90_valid_prefix_ec_labels.csv"
)
CLEANED_UNIPROT_RHEA = ROOT / "data/paper/reactzyme/raw/cleaned_uniprot_rhea.tsv"
RHEA_MOLECULES = ROOT / "data/paper/reactzyme/raw/rhea_molecules.tsv"
REACTION_T5_MODEL_PATH = (
    ROOT.parent
    / "hf_cache/hub/models--sagawa--ReactionT5v2-forward/snapshots/"
    "933114058cb2604dc1bf536dbebdfcefbe83d4fc"
)

REACTION_SET_FEATURE_DIM = 617
REACTION_DIRECTIONAL_F5_DIM = 2821
REACTION_DIRECTIONAL_F6_DIM = 3078

MATRIX_CONFIG_DIR = ROOT / "configs/benchmarks/reactzyme_paper"
SPLITS = load_split_specs(MATRIX_CONFIG_DIR / "splits.yaml")

PROTOCOL_ROOT = DEFAULT_PROTOCOL_ROOT
FEATURE_ROOT = DEFAULT_FEATURE_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        choices=MATRIX_NAMES,
        default="latent",
        help="Which ablation matrix to generate.",
    )
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_TRAIN_PYTHON)
    parser.add_argument("--setup-python-bin", type=Path, default=DEFAULT_SETUP_PYTHON)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--retrieval-batch-size", type=int, default=64)
    parser.add_argument("--validation-retrieval-batch-size", type=int, default=128)
    parser.add_argument("--accumulate-grad-batches", type=int, default=None)
    parser.add_argument(
        "--validation-interval-steps",
        type=int,
        default=None,
        help="Optional step-based validation override; paper runs validate every epoch.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--base-master-port", type=int, default=23900)
    parser.add_argument("--hard-negative-max-per-query", type=int, default=256)
    parser.add_argument("--hard-negative-anchors-per-batch", type=int, default=32)
    parser.add_argument("--hard-negative-positives-per-query", type=int, default=2)
    parser.add_argument("--hard-negative-negatives-per-query", type=int, default=6)
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


def rel(path: Path) -> str:
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return str(path)


def split_dir(split: str) -> Path:
    return PROTOCOL_ROOT / split


def feature_path(split: str, subset: str, modality: str) -> Path:
    return FEATURE_ROOT / split / subset / f"{modality}.h5"


def setup_paths(run_root: Path, split: str) -> dict[str, Path]:
    base = run_root / "data" / split
    reaction_feature_dir = base / "capability/reaction_features"
    validation_reaction_feature_dir = base / "capability/validation_reaction_features"
    test_reaction_feature_dir = base / "capability/test_reaction_features"
    reaction_chemistry_dir = base / "reaction_chemistry"
    reaction_set_dir = base / "reaction_set"
    reaction_directional_dir = base / "reaction_directional"
    return {
        "reaction_feature_dir": reaction_feature_dir,
        "reaction_features": reaction_feature_dir / "reaction_features.parquet",
        "validation_reaction_feature_dir": validation_reaction_feature_dir,
        "validation_reaction_features": (
            validation_reaction_feature_dir / "reaction_features.parquet"
        ),
        "test_reaction_feature_dir": test_reaction_feature_dir,
        "test_reaction_features": test_reaction_feature_dir / "reaction_features.parquet",
        "reaction_chemistry_train_npz": (
            reaction_chemistry_dir / "train_reaction_chemistry_vectors.npz"
        ),
        "reaction_chemistry_validation_npz": (
            reaction_chemistry_dir / "validation_reaction_chemistry_vectors.npz"
        ),
        "reaction_chemistry_test_npz": (
            reaction_chemistry_dir / "test_reaction_chemistry_vectors.npz"
        ),
        "reaction_chemistry_vocab": reaction_chemistry_dir / "reaction_chemistry_vocab.json",
        "reaction_set_train_npz": reaction_set_dir / "train_reaction_set_features.npz",
        "reaction_set_validation_npz": (
            reaction_set_dir / "validation_reaction_set_features.npz"
        ),
        "reaction_set_test_npz": reaction_set_dir / "test_reaction_set_features.npz",
        "reaction_set_schema": reaction_set_dir / "schema.json",
        "reaction_directional_f5_train_npz": (
            reaction_directional_dir / "train_reaction_directional_f5.npz"
        ),
        "reaction_directional_f5_validation_npz": (
            reaction_directional_dir / "validation_reaction_directional_f5.npz"
        ),
        "reaction_directional_f5_test_npz": (
            reaction_directional_dir / "test_reaction_directional_f5.npz"
        ),
        "reaction_directional_f6_train_npz": (
            reaction_directional_dir / "train_reaction_directional_f6.npz"
        ),
        "reaction_directional_f6_validation_npz": (
            reaction_directional_dir / "validation_reaction_directional_f6.npz"
        ),
        "reaction_directional_f6_test_npz": (
            reaction_directional_dir / "test_reaction_directional_f6.npz"
        ),
        "reaction_directional_schema": reaction_directional_dir / "schema.json",
        "factorized_capability_npz": (
            run_root / "data/factorized_capability/factorized_capability_vectors.npz"
        ),
        "factorized_capability_metadata": (
            run_root / "data/factorized_capability/factorized_capability_metadata.json"
        ),
        "biofp_npz": base / "biofp/enzyme_biofp_soft_targets.npz",
        "biofp_vocab": base / "biofp/enzyme_biofp_vocab.json",
        "rhea_match_dir": base / "rhea_matches",
        "rhea_matched_pairs": base / "rhea_matches/matched_pairs.csv",
        "rhea_matched_reactions": base / "rhea_matches/matched_reactions.csv",
        "rhea_matched_members": base / "rhea_matches/matched_members.csv",
        "rhea_match_report": base / "rhea_matches/report.json",
        "directional_feature_dir": base / "rhea_matches/directional_features",
        "directional_features": base / "rhea_matches/directional_features/reaction_features.parquet",
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


def assert_no_text_fields(config: Any, path: str = "") -> None:
    blocked = {
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
            if key in blocked:
                raise ValueError(f"Generated no-text config still contains {key_path}")
            assert_no_text_fields(value, key_path)
    elif isinstance(config, list):
        for idx, value in enumerate(config):
            assert_no_text_fields(value, f"{path}[{idx}]")


def unique_tags(*groups: list[str]) -> list[str]:
    seen = set()
    tags = []
    for group in groups:
        for tag in group:
            if tag not in seen:
                seen.add(tag)
                tags.append(tag)
    return tags


def apply_split_paths(config: dict[str, Any], split: str, subset: str) -> None:
    if subset not in {"validation", "test"}:
        raise ValueError(f"Unsupported ReactZyme paper subset: {subset}")
    data = config["data"]
    base = split_dir(split)
    feature_subset = "train" if subset == "validation" else "test"
    candidate_path = base / f"{subset}_candidate_ids.txt"
    data["train_pairs_path"] = rel(base / "train_pairs.csv")
    data["train_reactions_path"] = rel(base / "train_rxns.csv")
    if subset == "validation":
        data["validation_pairs_path"] = rel(base / "validation_pairs.csv")
        data["validation_reactions_path"] = rel(base / "validation_rxns.csv")
        data.pop("test_pairs_path", None)
        data.pop("test_reactions_path", None)
    else:
        data["test_pairs_path"] = rel(base / "test_pairs.csv")
        data["test_reactions_path"] = rel(base / "test_rxns.csv")
        data.pop("validation_pairs_path", None)
        data.pop("validation_reactions_path", None)
    data["train_reaction_t5v2_embeds_path"] = rel(feature_path(split, "train", "reactiont5v2"))
    data["validation_reaction_t5v2_embeds_path"] = rel(
        feature_path(split, feature_subset, "reactiont5v2")
    )
    data["train_reaction_unimol2_embeds_path"] = rel(feature_path(split, "train", "unimol2"))
    data["validation_reaction_unimol2_embeds_path"] = rel(
        feature_path(split, feature_subset, "unimol2")
    )
    data["train_reaction_chiro_embeds_path"] = rel(feature_path(split, "train", "chiro"))
    data["validation_reaction_chiro_embeds_path"] = rel(
        feature_path(split, feature_subset, "chiro")
    )
    data["reaction_direction_mode"] = "forward_only"
    data["validation_retrieval_candidate_set"] = "custom"
    data["validation_retrieval_candidate_ids_path"] = rel(candidate_path)
    training = config["training"]
    training["validation_retrieval_candidate_set"] = "custom"
    training["validation_retrieval_candidate_ids_path"] = rel(candidate_path)


def configure_loss(config: dict[str, Any], variant: dict[str, Any]) -> None:
    loss = config["training"].setdefault("loss", {})
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
                "lambda_r2e": 0.85,
                "lambda_e2r": 0.15,
                "lambda_direction_gap": 0.02,
                "lambda_r2e_hard_neg": variant.get("lambda_r2e_hard_neg", 0.25),
                "r2e_hard_neg_top_k": variant.get("r2e_hard_neg_top_k", 64),
                "r2e_hard_neg_margin": variant.get("r2e_hard_neg_margin", 0.05),
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
                "capability_consistency_weight": 0.02 if variant["capability"] else 0.0,
            }
        )
    if variant.get("biological", False):
        supervised = set(variant.get("supervised_families", []))
        family_weights: dict[str, float] = {}
        if supervised == {"mechanism", "cofactor"}:
            family_weights = {"mechanism": 0.65, "cofactor": 0.35}
        elif supervised:
            family_weights = {family: 1.0 for family in sorted(supervised)}
        loss["biofp_aux_weight"] = 0.03 if supervised else 0.0
        loss["biofp_family_weights"] = family_weights
        loss["biofp_confidence_cap"] = 1.0


def configure_model(
    config: dict[str, Any],
    variant: dict[str, Any],
    *,
    run_root: Path,
    split: str,
    evaluation_subset: str,
) -> None:
    model = config["model"]
    data = config["data"]
    paths = setup_paths(run_root, split)
    for key in (
        "protein_factorized_capability_vectors_path",
        "factorized_capability_missing_policy",
        "train_reaction_chemistry_vectors_path",
        "validation_reaction_chemistry_vectors_path",
        "reaction_chemistry_vectors_path",
        "reaction_chemistry_dim",
        "reaction_use_chemistry",
        "reaction_allow_missing_chemistry",
        "train_reaction_directional_vectors_path",
        "validation_reaction_directional_vectors_path",
        "reaction_directional_vectors_path",
        "reaction_directional_dim",
        "reaction_use_directional",
        "reaction_allow_missing_directional",
        "reaction_use_model",
    ):
        data.pop(key, None)
    model.pop("factorized_capability_vector", None)
    model["reaction_use_chiro"] = True
    model["reaction_use_chirality"] = False
    model["reaction_use_chienn"] = False
    model["reaction_chirality_name"] = "chiro"
    model.setdefault("reaction_multimodal_attention", {})
    model["reaction_multimodal_attention"].update(
        {
            "hidden_dim": 512,
            "dropout": 0.0,
            "modality_dropout": 0.10,
            "token_layer_norm": True,
            "modality_encoder_widths": [4096, 4096],
            "modality_encoder_use_layer_norm": False,
            "modality_encoder_dropout": 0.0,
            "modality_encoder_normalise_output": False,
        }
    )
    if variant.get("biological", False):
        model["reaction_multimodal_attention"]["modality_dropout"] = 0.0
        model["pooling"] = "sleec_guided_attention"
        model["enzyme_input_mode"] = "raw_mean_sleec_biological_factorized"
        model["target_encoder_dims"] = [512, 512]
        model["sleec_pooling"] = {
            "threshold": 0.34,
            "scorer_hidden_dim": 256,
            "checkpoint_path": str(SLEEC_CHECKPOINT),
            "freeze_scorer": True,
            "initial_bias_scale": 1.0,
            "train_bias_scale": True,
        }
        model["enzyme_block_fusion"] = {
            "dims": variant["block_dims"],
            "weights": variant["block_weights"],
            "dropout": 0.1,
        }
        model["biofp"] = {
            "family_dims": {"mechanism": 8, "cofactor": 10},
            "hidden_dim": 512,
            "dropout": 0.1,
        }
        hyperbolic: dict[str, Any] = {
            "hyp_dim": 128,
            "use_tangent": True,
            "freeze_projector": bool(variant.get("ec_pretrain", False)),
            "load_attention_pooler": False,
            "freeze_attention_pooler": False,
        }
        if variant.get("ec_pretrain", False):
            hyperbolic["checkpoint_path"] = str(
                run_root
                / "checkpoints"
                / split
                / variant["id"]
                / "ec_pretrain"
                / "last.ckpt"
            )
        model["hyperbolic_encoder"] = hyperbolic
        data["protein_biofp_targets_path"] = str(paths["biofp_npz"])
        data["protein_biofp_vocab_path"] = str(paths["biofp_vocab"])
        data["biofp_missing_policy"] = "zero_with_mask"
        for key in (
            "protein_capability_vectors_path",
            "protein_capability_metadata_path",
            "protein_factorized_capability_vectors_path",
        ):
            data.pop(key, None)
        model.pop("capability_vector", None)
        model.pop("factorized_capability_vector", None)
        if variant.get("reaction_feature_ablation", False):
            for key in (
                "protein_biofp_targets_path",
                "protein_biofp_vocab_path",
                "biofp_missing_policy",
            ):
                data.pop(key, None)
            attention = model["reaction_multimodal_attention"]
            attention.update(
                {
                    "side_composition": variant["reaction_side_composition"],
                    "fusion": variant["reaction_fusion"],
                    "output_projection": variant["reaction_output_projection"],
                    "modality_dropout": 0.0,
                }
            )
            if variant["reaction_fusion"] == "factorized_concat":
                attention["factorized_dims"] = variant["reaction_factorized_dims"]
                attention["factorized_weights"] = variant["reaction_factorized_weights"]
            else:
                attention.pop("factorized_dims", None)
                attention.pop("factorized_weights", None)
            attention["directional_gate_init"] = float(
                variant.get("reaction_directional_gate_init", 0.1)
            )
            data["reaction_use_model"] = bool(variant["reaction_use_model"])
            use_set_features = bool(variant.get("reaction_set_features", False))
            data["reaction_use_chemistry"] = use_set_features
            if use_set_features:
                data["train_reaction_chemistry_vectors_path"] = str(
                    paths["reaction_set_train_npz"]
                )
                data["validation_reaction_chemistry_vectors_path"] = str(
                    paths[
                        "reaction_set_test_npz"
                        if evaluation_subset == "test"
                        else "reaction_set_validation_npz"
                    ]
                )
                data["reaction_chemistry_dim"] = REACTION_SET_FEATURE_DIM
                data["reaction_allow_missing_chemistry"] = True
            directional_version = variant.get("reaction_directional")
            data["reaction_use_directional"] = directional_version in {"f5", "f6"}
            if directional_version in {"f5", "f6"}:
                prefix = f"reaction_directional_{directional_version}"
                data["train_reaction_directional_vectors_path"] = str(
                    paths[f"{prefix}_train_npz"]
                )
                data["validation_reaction_directional_vectors_path"] = str(
                    paths[
                        f"{prefix}_test_npz"
                        if evaluation_subset == "test"
                        else f"{prefix}_validation_npz"
                    ]
                )
                data["reaction_directional_dim"] = (
                    REACTION_DIRECTIONAL_F5_DIM
                    if directional_version == "f5"
                    else REACTION_DIRECTIONAL_F6_DIM
                )
                data["reaction_allow_missing_directional"] = True
        return
    if variant["mode"] == "standard":
        model["pooling"] = "mean"
        model["enzyme_input_mode"] = "standard"
        model["target_encoder_dims"] = [1024, 4096, 4096, 512]
        model.pop("enzyme_block_fusion", None)
        model.pop("hyperbolic_encoder", None)
        model.pop("capability_vector", None)
        model.pop("factorized_capability_vector", None)
        data.pop("protein_capability_vectors_path", None)
        data.pop("protein_capability_metadata_path", None)
        data.pop("protein_factorized_capability_vectors_path", None)
        return

    model["pooling"] = "sleec_guided_attention"
    model["enzyme_input_mode"] = variant["mode"]
    model["target_encoder_dims"] = [512, 512]
    model["sleec_pooling"] = {
        "threshold": 0.34,
        "scorer_hidden_dim": 256,
        "checkpoint_path": str(SLEEC_CHECKPOINT),
        "freeze_scorer": True,
        "initial_bias_scale": 1.0,
        "train_bias_scale": False,
    }
    model["enzyme_block_fusion"] = {
        "dims": variant["block_dims"],
        "weights": variant["block_weights"],
        "dropout": 0.0,
    }
    if "hyperbolic" in variant["mode"]:
        model["hyperbolic_encoder"] = {
            "checkpoint_path": str(HYPERBOLIC_CHECKPOINT),
            "hyp_dim": 512,
            "use_tangent": True,
            "freeze_projector": True,
            "load_attention_pooler": True,
            "freeze_attention_pooler": True,
        }
    else:
        model.pop("hyperbolic_encoder", None)
    if variant["capability"]:
        data["protein_capability_vectors_path"] = str(CAPABILITY_VECTORS)
        data["protein_capability_metadata_path"] = str(CAPABILITY_METADATA)
        data["capability_vector_in_memory"] = True
        data["capability_missing_policy"] = "zero_with_mask"
        model["capability_vector"] = {
            "dim": 256,
            "freeze": True,
            "adapter": False,
            "dropout": 0.1,
            "missing_policy": "zero_with_mask",
        }
    else:
        data.pop("protein_capability_vectors_path", None)
        data.pop("protein_capability_metadata_path", None)
        data.pop("capability_vector_in_memory", None)
        data.pop("capability_missing_policy", None)
        model.pop("capability_vector", None)
    if variant.get("factorized_capability", False):
        data["protein_factorized_capability_vectors_path"] = str(paths["factorized_capability_npz"])
        data["factorized_capability_missing_policy"] = "zero_with_mask"
        model["factorized_capability_vector"] = {
            "dims": {
                "cofactor": 128,
                "center": 128,
                "transition": 128,
            },
            "freeze": True,
            "use_masks": bool(variant.get("factorized_use_masks", True)),
            "missing_policy": "zero_with_mask",
        }
    else:
        data.pop("protein_factorized_capability_vectors_path", None)
        data.pop("factorized_capability_missing_policy", None)
        model.pop("factorized_capability_vector", None)
    if variant.get("reaction_chemistry", False):
        data["train_reaction_chemistry_vectors_path"] = str(paths["reaction_chemistry_train_npz"])
        data["validation_reaction_chemistry_vectors_path"] = str(
            paths[
                (
                    "reaction_chemistry_test_npz"
                    if evaluation_subset == "test"
                    else "reaction_chemistry_validation_npz"
                )
            ]
        )
        data["reaction_chemistry_dim"] = int(variant.get("reaction_chemistry_dim", 256))
        data["reaction_use_chemistry"] = True
        data["reaction_allow_missing_chemistry"] = True
        model["reaction_multimodal_attention"]["modality_dropout"] = 0.10


def configure_run(
    config: dict[str, Any],
    *,
    run_root: Path,
    split: str,
    variant: dict[str, Any],
    args: argparse.Namespace,
    test: bool = False,
) -> None:
    strip_text_fields(config)
    config["seed"] = int(args.seed)
    evaluation_subset = "test" if test else "validation"
    apply_split_paths(config, split, evaluation_subset)
    if test:
        data = config["data"]
        data["reaction_t5v2_embeds_path"] = data["validation_reaction_t5v2_embeds_path"]
        data["reaction_unimol2_embeds_path"] = data["validation_reaction_unimol2_embeds_path"]
        data["reaction_chiro_embeds_path"] = data["validation_reaction_chiro_embeds_path"]
        if "validation_reaction_chemistry_vectors_path" in data:
            data["reaction_chemistry_vectors_path"] = data[
                "validation_reaction_chemistry_vectors_path"
            ]
    configure_model(
        config,
        variant,
        run_root=run_root,
        split=split,
        evaluation_subset=evaluation_subset,
    )
    if test and variant.get("biological", False):
        # Biological labels shape the checkpoint during training; inference is
        # sequence-only and must not even load the target artifact.
        for key in (
            "protein_biofp_targets_path",
            "protein_biofp_vocab_path",
            "biofp_missing_policy",
        ):
            config["data"].pop(key, None)
    if test and "validation_reaction_chemistry_vectors_path" in data:
        data["reaction_chemistry_vectors_path"] = data["validation_reaction_chemistry_vectors_path"]
    if test and "validation_reaction_directional_vectors_path" in data:
        data["reaction_directional_vectors_path"] = data[
            "validation_reaction_directional_vectors_path"
        ]
    configure_loss(config, variant)
    split_tag = SPLITS[split]["tag"]
    variant_id = variant["id"]
    matrix_name = getattr(args, "matrix", "latent")
    run_name = f"reactzyme-{matrix_name}-{variant['label']}-{split}"
    if variant.get("biological", False):
        run_name += f"-seed{args.seed}"
    suffix = "test_configs" if test else "train"
    logging = config["logging"]
    logging["log_dir"] = str(run_root / "logs" / split / variant_id / suffix)
    logging["checkpoint_dir"] = str(run_root / "checkpoints" / split / variant_id)
    logging["checkpoint_monitor"] = (
        "val/mean_bidirectional_mrr"
        if variant.get("biological", False)
        else "val/loss"
    )
    logging["checkpoint_mode"] = "max" if variant.get("biological", False) else "min"
    logging["checkpoint_on_validation_end"] = True
    logging["save_every_n_train_steps"] = None
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
                f"reactzyme-{matrix_name}-ablation",
                "reactzyme-paper-protocol",
                "train-derived-validation",
                "released-test-final-only",
                "forward-only-reactions",
                split_tag,
                variant["label"],
                variant_id,
                "no-enzyme-text",
                "reactiont5v2",
                "unimol2",
                "chiro",
                "modality-mlp-attention",
            ],
            ["mean-baseline"] if variant["mode"] == "standard" else ["blockwise-enzyme"],
            ["sleec"] if variant["mode"] != "standard" else ["no-sleec"],
            (
                ["ec-hyperbolic"]
                if variant.get("ec_pretrain", False) or "hyperbolic" in variant["mode"]
                else ["no-ec-hyperbolic"]
            ),
            ["capability"] if variant["capability"] else ["no-capability"],
            (
                ["factorized-capability"]
                if variant.get("factorized_capability", False)
                else ["no-factorized-capability"]
            ),
            (
                ["reaction-chemistry-token"]
                if variant.get("reaction_chemistry", False)
                else ["no-reaction-chemistry-token"]
            ),
            ["hard-negative"] if variant["hard_negative"] else ["no-hard-negative"],
            ["structure-regularized"] if variant["structure"] else ["no-structure"],
            ["biofp-minimal-v1"] if variant.get("biological", False) else [],
            ["reaction-feature-ablation"] if variant.get("reaction_feature_ablation", False) else [],
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
    if variant["hard_negative"]:
        paths = setup_paths(run_root, split)
        data["hard_negative_pools_path"] = str(paths["hardneg_json"])
        data["hard_negative_anchor_queries_per_batch"] = args.hard_negative_anchors_per_batch
        if "hard_negative_anchors_per_batch" in variant:
            data["hard_negative_anchor_queries_per_batch"] = variant[
                "hard_negative_anchors_per_batch"
            ]
        data["hard_negative_positives_per_query"] = args.hard_negative_positives_per_query
        data["hard_negative_negatives_per_query"] = args.hard_negative_negatives_per_query
        if "hard_negative_negatives_per_query" in variant:
            data["hard_negative_negatives_per_query"] = variant["hard_negative_negatives_per_query"]
        data["hard_negative_seed"] = args.seed
    else:
        for key in (
            "hard_negative_pools_path",
            "hard_negative_anchor_queries_per_batch",
            "hard_negative_positives_per_query",
            "hard_negative_negatives_per_query",
            "hard_negative_seed",
        ):
            data.pop(key, None)
    if variant["structure"]:
        data["enzyme_ec_labels_path"] = str(setup_paths(run_root, split)["enzyme_ec_labels"])
    else:
        data["enzyme_ec_labels_path"] = None

    training = config["training"]
    training["max_epochs"] = args.max_epochs
    training["devices"] = 4
    training["accelerator"] = "gpu"
    training["strategy"] = "ddp"
    training["accumulate_grad_batches"] = args.accumulate_grad_batches
    training["validation_retrieval_metrics"] = True
    training["validation_retrieval_batch_size"] = args.validation_retrieval_batch_size
    training["validation_retrieval_directions"] = ["reaction_to_enzyme", "enzyme_to_reaction"]
    training["validation_interval_steps"] = args.validation_interval_steps
    training["check_val_every_n_epoch"] = 1
    training["num_sanity_val_steps"] = 0
    training["enable_progress_bar"] = False
    training["log_attention_stats"] = True
    training["attention_logging_interval"] = 10
    training["use_distributed_sampler"] = not bool(variant["hard_negative"])
    if variant.get("biological", False):
        if variant.get("pretrain", False) and not test:
            training["biofp_pretrain_checkpoint"] = str(
                run_root
                / "checkpoints"
                / split
                / variant_id
                / "biological_pretrain"
                / "best.ckpt"
            )
        else:
            training.pop("biofp_pretrain_checkpoint", None)
        training["early_stopping"] = {
            "enabled": True,
            "monitor": "val/mean_bidirectional_mrr",
            "mode": "max",
            "patience": 5,
            "min_delta": 0.0001,
        }
        # The architecture is fixed across ablations, so disabled auxiliary
        # heads are deliberately present but unused in some variants.
        training["strategy"] = "ddp_find_unused_parameters_true"
        training["accumulate_grad_batches"] = 1
    config["ablation"] = {
        "run_id": f"{split}_{variant_id}",
        "split": split,
        "variant": variant_id,
        "label": variant["label"],
        "description": variant["description"],
        "test_config": test,
        "execution_model": "independent-4gpu-ddp-concurrent-wave",
        "no_enzyme_text": True,
        "matrix": matrix_name,
        "data_protocol": "reactzyme-paper-90-10-validation",
        "evaluation_subset": evaluation_subset,
        "evaluation_protocol": EVALUATION_PROTOCOL,
    }
    assert_no_text_fields(config)


def make_config(
    templates: dict[str, dict[str, dict[str, Any]]],
    *,
    split: str,
    variant: dict[str, Any],
    run_root: Path,
    args: argparse.Namespace,
    test: bool = False,
) -> dict[str, Any]:
    template = templates[split][variant["template"]]
    config = copy.deepcopy(template)
    configure_run(config, run_root=run_root, split=split, variant=variant, args=args, test=test)
    return config


def make_biological_pretrain_config(
    train_config: dict[str, Any],
    *,
    run_root: Path,
    split: str,
    variant: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = copy.deepcopy(train_config)
    variant_id = variant["id"]
    stage_root = run_root / "checkpoints" / split / variant_id / "biological_pretrain"
    config["logging"]["checkpoint_dir"] = str(stage_root)
    config["logging"]["log_dir"] = str(
        run_root / "logs" / split / variant_id / "biological_pretrain"
    )
    config["logging"]["wandb"]["run_name"] = (
        f"reactzyme-biological-{variant['label']}-{split}-pretrain-seed{args.seed}"
    )
    config["logging"]["wandb"]["tags"] = unique_tags(
        config["logging"]["wandb"].get("tags", []),
        ["enzyme-only-pretraining", f"seed-{args.seed}"],
    )
    config["data"] = {
        "protein_residue_embeds_path": train_config["data"]["protein_residue_embeds_path"],
        "protein_biofp_targets_path": train_config["data"]["protein_biofp_targets_path"],
        "protein_biofp_vocab_path": train_config["data"]["protein_biofp_vocab_path"],
        "biofp_missing_policy": "zero_with_mask",
        "residue_dim": train_config["data"].get("residue_dim", 1024),
        "max_protein_tokens": train_config["data"].get("max_protein_tokens", 1022),
        "protein_truncation": train_config["data"].get("protein_truncation", "ends_center"),
    }
    families = set(variant.get("pretrain_families", []))
    weights = (
        {"mechanism": 0.65, "cofactor": 0.35}
        if families == {"mechanism", "cofactor"}
        else {family: 1.0 for family in sorted(families)}
    )
    config["training"] = {
        "max_epochs": 20,
        "batch_size": 128,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "validation_fraction": 0.10,
        "biofp_family_weights": weights,
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


def make_ec_pretrain_config(
    *,
    run_root: Path,
    split: str,
    variant: dict[str, Any],
    train_config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    stage_root = run_root / "checkpoints" / split / variant["id"] / "ec_pretrain"
    return {
        "residue_embeddings_path": train_config["data"]["protein_residue_embeds_path"],
        "ec_labels_path": str(setup_paths(run_root, split)["enzyme_ec_labels"]),
        "input_dim": int(train_config["data"].get("residue_dim", 1024)),
        "hyp_dim": 128,
        "curvature": 0.25,
        "batch_size": 128,
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "epochs": 20,
        "max_triplets_per_anchor": 32,
        "sleec_checkpoint_path": str(SLEEC_CHECKPOINT),
        "sleec_scorer_hidden_dim": 256,
        "freeze_sleec": True,
        "max_protein_tokens": int(train_config["data"].get("max_protein_tokens", 1022)),
        "protein_truncation": train_config["data"].get("protein_truncation", "ends_center"),
        "output_checkpoint": str(stage_root / "last.ckpt"),
        "seed": int(args.seed),
        "num_workers": 2,
        "device": "cuda",
        "wandb": args.wandb_mode != "disabled",
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_run_name": (
            f"reactzyme-biological-{variant['label']}-{split}-ec-pretrain-seed{args.seed}"
        ),
        "wandb_mode": args.wandb_mode,
        "wandb_tags": [
            "reactzyme-paper-protocol",
            "train-only-ec",
            "ec-hyperbolic-pretraining",
            split,
            variant["label"],
            f"seed-{args.seed}",
        ],
        "wandb_dir": str(run_root / "wandb"),
    }


def preflight(args: argparse.Namespace) -> None:
    paths = [
        args.python_bin,
        args.setup_python_bin,
        ROOT / "scripts/train_protein_pooling.py",
        ROOT / "scripts/evaluate_protein_pooling.py",
        SLEEC_CHECKPOINT,
        PROTOCOL_ROOT / "manifest.json",
    ]
    if args.matrix in {"biological", "reaction_features"}:
        paths.extend(
            [
                ROOT / "scripts/build_reaction_features.py",
                ENZYME_COFACTOR_LABELS,
                COFACTOR_DICTIONARY,
                RHEA_MOLECULES,
            ]
        )
        if args.matrix == "biological":
            paths.extend(
                [
                    ROOT / "scripts/build_reactzyme_train_rhea_matches.py",
                    ROOT / "scripts/build_enzyme_biological_targets.py",
                    ROOT / "scripts/pretrain_enzyme_biofp_split.py",
                    ROOT / "scripts/pretrain_hyperbolic_enzyme.py",
                    EC_SOURCE,
                    CLEANED_UNIPROT_RHEA,
                ]
            )
        else:
            paths.extend(
                [
                    ROOT / "scripts/build_reaction_set_features.py",
                    ROOT / "scripts/build_reaction_only_rhea_map.py",
                    ROOT / "scripts/build_reaction_directional_vectors.py",
                    ROOT / "scripts/extract_reaction_t5v2_embeddings.py",
                    ROOT / "scripts/extract_unimol2_reaction_embeddings.py",
                    ROOT / "scripts/extract_chiro_reaction_embeddings.py",
                ]
            )
    else:
        paths.extend(
            [
                ROOT / "scripts/build_reactzyme_paper_protocols.py",
                ROOT / "scripts/build_reaction_features.py",
                ROOT / "scripts/build_enzyme_biofp_soft_targets.py",
                ROOT / "scripts/mine_r2e_hard_negatives.py",
                ROOT / "scripts/build_factorized_capability_vectors.py",
                ROOT / "scripts/build_reaction_chemistry_vectors.py",
                HYPERBOLIC_CHECKPOINT,
                CAPABILITY_VECTORS,
                CAPABILITY_METADATA,
                ENZYME_COFACTOR_LABELS,
                COFACTOR_DICTIONARY,
                EC_SOURCE,
            ]
        )
    for split, meta in SPLITS.items():
        base = split_dir(split)
        paths.extend(
            [
                ROOT / meta["official_template"],
                ROOT / meta["sleec_template"],
                base / "train_pairs.csv",
                base / "validation_pairs.csv",
                base / "test_pairs.csv",
                base / "train_rxns.csv",
                base / "validation_rxns.csv",
                base / "test_rxns.csv",
                base / "train_candidate_ids.txt",
                base / "validation_candidate_ids.txt",
                base / "test_candidate_ids.txt",
                feature_path(split, "train", "reactiont5v2"),
                feature_path(split, "train", "unimol2"),
                feature_path(split, "train", "chiro"),
                feature_path(split, "test", "reactiont5v2"),
                feature_path(split, "test", "unimol2"),
                feature_path(split, "test", "chiro"),
            ]
        )
    require_paths(paths)


def write_runtime_env(path: Path, args: argparse.Namespace, run_root: Path) -> None:
    values = {
        "ROOT": ROOT,
        "RUN_ROOT": run_root,
        "MATRIX_NAME": args.matrix,
        "PROTOCOL_ROOT": PROTOCOL_ROOT,
        "PYTHON_BIN": args.python_bin,
        "SETUP_PYTHON_BIN": args.setup_python_bin,
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_MODE": args.wandb_mode,
        "GPUS": args.gpus,
        "BASE_MASTER_PORT": args.base_master_port,
        "COFACTOR_DICTIONARY": COFACTOR_DICTIONARY,
        "ENZYME_COFACTOR_LABELS": ENZYME_COFACTOR_LABELS,
        "EC_SOURCE": EC_SOURCE,
        "CLEANED_UNIPROT_RHEA": CLEANED_UNIPROT_RHEA,
        "RHEA_MOLECULES": RHEA_MOLECULES,
        "REACTION_T5_MODEL_PATH": REACTION_T5_MODEL_PATH,
        "CAPABILITY_VECTORS": CAPABILITY_VECTORS,
        "FACTORIZED_CAPABILITY_VECTORS": setup_paths(run_root, "time")["factorized_capability_npz"],
        "FACTORIZED_CAPABILITY_METADATA": setup_paths(run_root, "time")[
            "factorized_capability_metadata"
        ],
        "HARD_NEGATIVE_MAX_PER_QUERY": args.hard_negative_max_per_query,
        "SEED": args.seed,
        "EVALUATION_PROTOCOL": EVALUATION_PROTOCOL,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{key}={shell_quote(value)}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.max_epochs is None:
        args.max_epochs = 30 if args.matrix in {"biological", "reaction_features"} else 15
    if args.train_batch_size is None:
        args.train_batch_size = 512 if args.matrix in {"biological", "reaction_features"} else 128
    if args.accumulate_grad_batches is None:
        args.accumulate_grad_batches = 1 if args.matrix in {"biological", "reaction_features"} else 4
    if args.run_root is None:
        args.run_root = {
            "latent": DEFAULT_RUN_ROOT,
            "representation": DEFAULT_REPRESENTATION_RUN_ROOT,
            "biological": DEFAULT_BIOLOGICAL_RUN_ROOT,
            "reaction_features": DEFAULT_REACTION_FEATURE_RUN_ROOT,
        }[args.matrix]
    if args.wandb_project is None:
        args.wandb_project = {
            "latent": DEFAULT_WANDB_PROJECT,
            "representation": DEFAULT_REPRESENTATION_WANDB_PROJECT,
            "biological": DEFAULT_BIOLOGICAL_WANDB_PROJECT,
            "reaction_features": DEFAULT_REACTION_FEATURE_WANDB_PROJECT,
        }[args.matrix]
    args.run_root = args.run_root.expanduser().resolve()
    args.protocol_root = args.protocol_root.expanduser().resolve()
    args.features_root = args.features_root.expanduser().resolve()
    args.python_bin = normalize_executable_path(args.python_bin)
    args.setup_python_bin = normalize_executable_path(args.setup_python_bin)
    global PROTOCOL_ROOT, FEATURE_ROOT
    PROTOCOL_ROOT = args.protocol_root
    FEATURE_ROOT = args.features_root
    if args.accumulate_grad_batches <= 0:
        raise ValueError("--accumulate-grad-batches must be positive")
    if args.base_master_port <= 0:
        raise ValueError("--base-master-port must be positive")
    if not [token for token in args.gpus.split(",") if token.strip()]:
        raise ValueError("--gpus must list at least one GPU")
    if not args.skip_preflight:
        preflight(args)

    run_root = args.run_root
    config_root = run_root / "configs"
    run_root.mkdir(parents=True, exist_ok=True)
    templates = {
        split: {
            "official": load_yaml(ROOT / meta["official_template"]),
            "sleec": load_yaml(ROOT / meta["sleec_template"]),
        }
        for split, meta in SPLITS.items()
    }

    setup_rows: list[dict[str, Any]] = []
    pretrain_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    variants = load_variants(MATRIX_CONFIG_DIR / f"{args.matrix}.yaml")
    for split_index, split in enumerate(SPLITS):
        paths = setup_paths(run_root, split)
        base = split_dir(split)
        setup_rows.append(
            {
                "split": split,
                "train_pairs": rel(base / "train_pairs.csv"),
                "train_reactions": rel(base / "train_rxns.csv"),
                "candidate_ids": rel(base / "train_candidate_ids.txt"),
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
        for variant_index, variant in enumerate(variants):
            train_cfg = make_config(
                templates,
                split=split,
                variant=variant,
                run_root=run_root,
                args=args,
                test=False,
            )
            test_cfg = make_config(
                templates,
                split=split,
                variant=variant,
                run_root=run_root,
                args=args,
                test=True,
            )
            variant_id = variant["id"]
            train_config_path = config_root / variant_id / split / "train.yaml"
            test_config_path = config_root / variant_id / split / "test.yaml"
            write_yaml(train_config_path, train_cfg)
            write_yaml(test_config_path, test_cfg)
            if args.matrix == "biological" and variant.get("pretrain", False):
                biological_pretrain_config = make_biological_pretrain_config(
                    train_cfg,
                    run_root=run_root,
                    split=split,
                    variant=variant,
                    args=args,
                )
                biological_pretrain_path = (
                    config_root / variant_id / split / "biological_pretrain.yaml"
                )
                write_yaml(biological_pretrain_path, biological_pretrain_config)
            else:
                # Bash treats tab as IFS whitespace and collapses empty TSV fields.
                # Keep the plan rectangular so the two config columns cannot shift.
                biological_pretrain_path = "-"
            if args.matrix == "biological" and variant.get("ec_pretrain", False):
                ec_pretrain_config = make_ec_pretrain_config(
                    run_root=run_root,
                    split=split,
                    variant=variant,
                    train_config=train_cfg,
                    args=args,
                )
                ec_pretrain_path = config_root / variant_id / split / "ec_pretrain.yaml"
                write_yaml(ec_pretrain_path, ec_pretrain_config)
            else:
                ec_pretrain_path = "-"
            if args.matrix == "biological" and (
                variant.get("pretrain", False) or variant.get("ec_pretrain", False)
            ):
                pretrain_rows.append(
                    {
                        "wave": variant_index,
                        "variant": variant_id,
                        "label": variant["label"],
                        "split": split,
                        "run_id": f"{split}_{variant_id}",
                        "ec_config": ec_pretrain_path,
                        "biological_config": biological_pretrain_path,
                        "master_port": args.base_master_port + 1000 + variant_index * 10 + split_index,
                    }
                )
            train_rows.append(
                {
                    "wave": variant_index,
                    "variant": variant_id,
                    "label": variant["label"],
                    "split": split,
                    "run_id": f"{split}_{variant_id}",
                    "config": train_config_path,
                    "test_config": test_config_path,
                    "checkpoint_dir": run_root / "checkpoints" / split / variant_id,
                    "log_path": run_root / "logs" / split / variant_id / "stdout.log",
                    "eval_json": run_root / "eval" / split / variant_id / "test_both.json",
                    "wandb_run_name": train_cfg["logging"]["wandb"]["run_name"],
                    "master_port": args.base_master_port + variant_index * 10 + split_index,
                    "description": variant["description"],
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
        [
            "wave",
            "variant",
            "label",
            "split",
            "run_id",
            "ec_config",
            "biological_config",
            "master_port",
        ],
        pretrain_rows,
    )
    write_tsv(
        run_root / "train_plan.tsv",
        [
            "wave",
            "variant",
            "label",
            "split",
            "run_id",
            "config",
            "test_config",
            "checkpoint_dir",
            "log_path",
            "eval_json",
            "wandb_run_name",
            "master_port",
            "description",
        ],
        train_rows,
    )
    write_runtime_env(run_root / "runtime.env", args, run_root)
    launcher = ROOT / (
        "scripts/run_reactzyme_biological_ablation_matrix.sh"
        if args.matrix == "biological"
        else (
            "scripts/run_reactzyme_reaction_feature_chain.sh"
            if args.matrix == "reaction_features"
            else "scripts/run_reactzyme_paper_ablation_matrix.sh"
        )
    )
    manifest = {
        "run_root": str(run_root),
        "launcher": str(launcher),
        "runtime_env": str(run_root / "runtime.env"),
        "setup_plan": str(run_root / "setup_plan.tsv"),
        "pretrain_plan": str(run_root / "pretrain_plan.tsv"),
        "train_plan": str(run_root / "train_plan.tsv"),
        "splits": list(SPLITS),
        "variants": variants,
        "matrix": args.matrix,
        "data_protocol": {
            "name": "reactzyme-paper-90-10-validation",
            "protocol_root": str(PROTOCOL_ROOT),
            "feature_root": str(FEATURE_ROOT),
            "validation_source": "10% pair-level sample from released training split",
            "checkpoint_monitor": (
                "val/mean_bidirectional_mrr"
                if args.matrix in {"biological", "reaction_features"}
                else "val/loss"
            ),
            "reaction_direction_mode": "forward_only",
            "released_test_usage": "final evaluation only",
            "evaluation_protocol": EVALUATION_PROTOCOL,
        },
        "feature_coverage": collect_feature_coverage(
            protocol_root=PROTOCOL_ROOT,
            feature_root=FEATURE_ROOT,
            protocols=tuple(SPLITS),
        ),
        "execution_model": (
            "Independent split-specific 4-GPU DDP jobs. For each ablation wave, "
            "time/enzyme_smi/reaction_smi are launched concurrently on the same GPUs."
        ),
        "wandb": {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "mode": args.wandb_mode,
        },
        "scheduler": {
            "gpus": args.gpus,
            "devices_per_run": 4,
            "concurrent_runs_per_wave": 3,
            "base_master_port": args.base_master_port,
        },
        "runs": [{key: str(value) for key, value in row.items()} for row in train_rows],
    }
    write_json(run_root / "manifest.json", manifest)
    wrapper = run_root / (
        "run_reactzyme_biological_ablation_matrix.sh"
        if args.matrix == "biological"
        else (
            "run_reactzyme_reaction_feature_chain.sh"
            if args.matrix == "reaction_features"
            else "run_reactzyme_paper_ablation_matrix.sh"
        )
    )
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shell_quote(ROOT)}\n"
        f'exec {shell_quote(launcher)} {shell_quote(run_root)} "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
