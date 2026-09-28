"""Generate and promote the staged ReactZyme B0 E->R geometry ablation."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shlex
import stat
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import yaml

from horizyn.benchmarks import reactzyme_matrix
from horizyn.benchmarks.reactzyme_protocol import (
    REACTZYME_SPLITS,
    materialize_unseen_reaction_protocols,
)
from horizyn.benchmarks.reactzyme_runtime import (
    read_checkpoint_sidecar,
    write_checkpoint_sidecar,
)


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "configs/benchmarks/reactzyme_paper/geometry.yaml"
DEFAULT_RUN_ROOT = ROOT / "runs/reactzyme_b0_e2r_geometry_v1"
DEFAULT_PROTOCOL_ROOT = ROOT / "data/revised_protocols/reactzyme_unseen_reaction_v1"
DEFAULT_SOURCE_ROOT = ROOT / "data/revised_protocols/reactzyme_official"
DEFAULT_FEATURE_ROOT = ROOT / "data/revised_protocols/reactzyme_official/features"
DEFAULT_PYTHON = ROOT.parent / "env/bin/python"
DEFAULT_SETUP_PYTHON = ROOT.parent / ".capability-run-py/bin/python"
DEFAULT_WANDB_PROJECT = "reactzyme-b0-e2r-geometry-v1"
DEFAULT_WANDB_ENTITY = "omnai"
VALIDATION_PROTOCOL = "configured_forward_candidates"
TEST_PROTOCOL = "paper_test_candidates"
STAGES = ("recipe", "recipe_confirm", "reaction", "enzyme", "hard_negative")

PLAN_FIELDS = (
    "stage",
    "variant",
    "label",
    "seed",
    "split",
    "run_id",
    "config",
    "validation_config",
    "mining_config",
    "test_config",
    "checkpoint_dir",
    "checkpoint_sidecar",
    "validation_eval_json",
    "test_eval_json",
    "wandb_run_name",
    "master_port",
    "parent_checkpoint",
    "biological_pretrain_config",
    "ec_pretrain_config",
    "hard_negative_json",
    "description",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(os.path.abspath(expanded))


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    spec = _load_yaml(path)
    if spec.get("schema_version") != "reactzyme_e2r_geometry_v1":
        raise ValueError(f"Unsupported geometry matrix schema in {path}")
    stages = spec.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("geometry matrix must contain a stages mapping")
    expected = {"recipe", "reaction", "enzyme", "hard_negative"}
    if set(stages) != expected:
        raise ValueError(f"geometry stages must be exactly {sorted(expected)}")
    for stage_name, stage in stages.items():
        variants = stage.get("variants")
        if not isinstance(variants, list) or not variants:
            raise ValueError(f"{stage_name}.variants must be a non-empty list")
        ids = [str(variant.get("id", "")) for variant in variants]
        if any(not variant_id for variant_id in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{stage_name} has missing or duplicate variant IDs")
        if stage.get("baseline") not in ids:
            raise ValueError(f"{stage_name}.baseline is not a declared variant")
        for variant in variants:
            for field in ("id", "label", "description", "complexity", "overlay"):
                if field not in variant:
                    raise ValueError(f"{stage_name}.{variant.get('id')} missing {field}")
            if not isinstance(variant["overlay"], dict):
                raise ValueError(f"{stage_name}.{variant['id']}.overlay must be a mapping")
            enzyme = variant["overlay"].get("enzyme")
            if enzyme:
                dims = enzyme.get("dims", {})
                weights = enzyme.get("weights", {})
                if set(dims) != set(weights):
                    raise ValueError(f"{stage_name}.{variant['id']} enzyme blocks differ")
                if sum(int(value) for value in dims.values()) != 512:
                    raise ValueError(f"{stage_name}.{variant['id']} enzyme dims must sum to 512")
                if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-6:
                    raise ValueError(f"{stage_name}.{variant['id']} enzyme weights must sum to 1")
    return spec


def variants_for_stage(
    spec: dict[str, Any],
    stage: str,
    selected: set[str] | None = None,
) -> list[dict[str, Any]]:
    source_stage = "recipe" if stage == "recipe_confirm" else stage
    variants = copy.deepcopy(spec["stages"][source_stage]["variants"])
    if selected is not None:
        variants = [variant for variant in variants if variant["id"] in selected]
        missing = selected - {variant["id"] for variant in variants}
        if missing:
            raise ValueError(f"Unknown {source_stage} variants: {sorted(missing)}")
    return variants


def _base_variant() -> dict[str, Any]:
    variants = reactzyme_matrix.load_variants(
        ROOT / "configs/benchmarks/reactzyme_paper/biological.yaml"
    )
    return copy.deepcopy(next(variant for variant in variants if variant["label"] == "B0"))


def _generator_args(
    *,
    run_root: Path,
    protocol_root: Path,
    feature_root: Path,
    seed: int,
    python_bin: Path,
    setup_python_bin: Path,
    wandb_project: str,
    wandb_entity: str,
    wandb_mode: str,
    max_epochs: int,
    train_batch_size: int,
    retrieval_batch_size: int,
    validation_retrieval_batch_size: int,
    num_workers: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        matrix="biological",
        run_root=run_root,
        protocol_root=protocol_root,
        features_root=feature_root,
        python_bin=python_bin,
        setup_python_bin=setup_python_bin,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        max_epochs=max_epochs,
        train_batch_size=train_batch_size,
        retrieval_batch_size=retrieval_batch_size,
        validation_retrieval_batch_size=validation_retrieval_batch_size,
        accumulate_grad_batches=1,
        validation_interval_steps=None,
        num_workers=num_workers,
        gpus="0,1,2,3",
        base_master_port=24800,
        hard_negative_max_per_query=64,
        hard_negative_anchors_per_batch=32,
        hard_negative_positives_per_query=2,
        hard_negative_negatives_per_query=6,
        seed=seed,
        skip_preflight=True,
    )


def _load_templates() -> dict[str, dict[str, dict[str, Any]]]:
    return {
        split: {
            "official": _load_yaml(ROOT / meta["official_template"]),
            "sleec": _load_yaml(ROOT / meta["sleec_template"]),
        }
        for split, meta in reactzyme_matrix.SPLITS.items()
    }


def _base_configs(
    *,
    split: str,
    run_root: Path,
    protocol_root: Path,
    feature_root: Path,
    args: SimpleNamespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reactzyme_matrix.PROTOCOL_ROOT = protocol_root
    reactzyme_matrix.FEATURE_ROOT = feature_root
    templates = _load_templates()
    variant = _base_variant()
    train = reactzyme_matrix.make_config(
        templates,
        split=split,
        variant=variant,
        run_root=run_root,
        args=args,
        test=False,
    )
    test = reactzyme_matrix.make_config(
        templates,
        split=split,
        variant=variant,
        run_root=run_root,
        args=args,
        test=True,
    )
    _remove_training_labels(train)
    return train, test


def _set_loss(config: dict[str, Any], overlay: dict[str, Any]) -> None:
    if not overlay:
        return
    loss = config["training"].setdefault("loss", {})
    preserved = {
        "beta": loss.get("beta", 10.0),
        "learn_beta": False,
        "beta_min": loss.get("beta_min", 0.01),
        "beta_max": loss.get("beta_max", 100.0),
        "positive_pair_source": "observed_pairs",
        "biofp_aux_weight": loss.get("biofp_aux_weight", 0.0),
        "biofp_family_weights": loss.get("biofp_family_weights", {}),
        "biofp_confidence_cap": loss.get("biofp_confidence_cap", 1.0),
    }
    loss.clear()
    loss.update(preserved)
    loss.update(copy.deepcopy(overlay))


def _set_sampler(config: dict[str, Any], overlay: dict[str, Any], seed: int) -> None:
    name = overlay.get("name", "shuffle")
    if name == "shuffle":
        config["data"].pop("train_sampler", None)
        config["training"]["use_distributed_sampler"] = True
        return
    if name != "reaction_degree_balanced":
        raise ValueError(f"Unsupported sampler: {name}")
    config["data"]["train_sampler"] = {
        "name": name,
        "degree_exponent": float(overlay.get("degree_exponent", 0.5)),
        "seed": int(seed),
    }
    config["training"]["use_distributed_sampler"] = False


def _set_reaction(config: dict[str, Any], overlay: dict[str, Any]) -> None:
    if not overlay:
        return
    reaction = config["model"].setdefault("reaction_multimodal_attention", {})
    reaction.update(copy.deepcopy(overlay))


def _set_enzyme(
    config: dict[str, Any],
    overlay: dict[str, Any],
    *,
    run_root: Path,
    split: str,
    run_key: str,
) -> None:
    if not overlay:
        return
    fusion = config["model"].setdefault("enzyme_block_fusion", {})
    fusion["dims"] = copy.deepcopy(overlay["dims"])
    fusion["weights"] = copy.deepcopy(overlay["weights"])
    fusion["learned_weights"] = bool(overlay.get("learned_weights", False))
    fusion["weight_kl_weight"] = float(overlay.get("weight_kl_weight", 0.0))
    config["model"].setdefault("biofp", {})["family_dims"] = copy.deepcopy(
        overlay.get("family_dims", {})
    )
    if "ec" not in fusion["dims"]:
        config["model"].pop("hyperbolic_encoder", None)
    elif overlay.get("ec_pretrain", False):
        config["model"]["hyperbolic_encoder"] = {
            "checkpoint_path": str(
                run_root / "checkpoints" / split / run_key / "ec_pretrain" / "last.ckpt"
            ),
            "hyp_dim": int(fusion["dims"]["ec"]),
            "use_tangent": True,
            "freeze_projector": True,
            "load_attention_pooler": False,
            "freeze_attention_pooler": False,
        }


def _remove_training_labels(config: dict[str, Any]) -> None:
    for key in ("protein_biofp_targets_path", "protein_biofp_vocab_path", "biofp_missing_policy"):
        config["data"].pop(key, None)


def _apply_overlay(
    config: dict[str, Any],
    overlay: dict[str, Any],
    *,
    run_root: Path,
    split: str,
    run_key: str,
    seed: int,
) -> None:
    _set_loss(config, overlay.get("loss", {}))
    if "sampler" in overlay:
        _set_sampler(config, overlay["sampler"], seed)
    _set_reaction(config, overlay.get("reaction", {}))
    enzyme_overlay = copy.deepcopy(overlay.get("enzyme", {}))
    if overlay.get("ec_pretrain", False):
        enzyme_overlay["ec_pretrain"] = True
    _set_enzyme(
        config,
        enzyme_overlay,
        run_root=run_root,
        split=split,
        run_key=run_key,
    )
    attention_reg = overlay.get("reaction_attention_regularization")
    if attention_reg:
        config["training"]["reaction_attention_regularization"] = copy.deepcopy(attention_reg)
    elif "reaction" in overlay:
        config["training"].pop("reaction_attention_regularization", None)

    if "supervised_families" in overlay or "biofp_aux_weight" in overlay:
        supervised = list(overlay.get("supervised_families", []))
        aux_weight = float(overlay.get("biofp_aux_weight", 0.0))
        loss = config["training"]["loss"]
        loss["biofp_aux_weight"] = aux_weight
        loss["biofp_family_weights"] = {family: 1.0 for family in supervised}
        if aux_weight <= 0:
            _remove_training_labels(config)
    if overlay.get("model_hard_negatives", False):
        hard_negative_json = run_root / "hard_negatives" / split / run_key / "pools.json"
        config["data"].pop("train_sampler", None)
        config["data"].update(
            {
                "hard_negative_pools_path": str(hard_negative_json),
                "hard_negative_anchor_queries_per_batch": int(
                    overlay.get("hard_negative_anchors_per_batch", 32)
                ),
                "hard_negative_positives_per_query": int(
                    overlay.get("hard_negative_positives_per_query", 2)
                ),
                "hard_negative_negatives_per_query": int(
                    overlay.get("hard_negative_negatives_per_query", 6)
                ),
                "hard_negative_seed": int(seed),
            }
        )
        config["training"]["use_distributed_sampler"] = False
    else:
        for key in (
            "hard_negative_pools_path",
            "hard_negative_anchor_queries_per_batch",
            "hard_negative_positives_per_query",
            "hard_negative_negatives_per_query",
            "hard_negative_seed",
        ):
            config["data"].pop(key, None)


def _configure_identity(
    config: dict[str, Any],
    *,
    run_root: Path,
    stage: str,
    variant: dict[str, Any],
    split: str,
    seed: int,
    run_key: str,
    test: bool,
) -> None:
    run_id = f"{stage}_{variant['id']}_seed{seed}_{split}"
    config["seed"] = int(seed)
    logging = config["logging"]
    logging["log_dir"] = str(run_root / "logs" / stage / run_key / split)
    logging["checkpoint_dir"] = str(run_root / "checkpoints" / split / run_key)
    logging["checkpoint_monitor"] = "val/mean_bidirectional_mrr"
    logging["checkpoint_mode"] = "max"
    logging["wandb"]["enabled"] = not test
    logging["wandb"]["run_name"] = run_id
    logging["wandb"]["tags"] = [
        "reactzyme-b0-e2r-geometry-v1",
        stage,
        variant["id"],
        split,
        f"seed-{seed}",
        "no-enzyme-text",
        "forward-only-reactions",
    ]
    config["ablation"] = {
        "run_id": run_id,
        "stage": stage,
        "variant": variant["id"],
        "label": variant["label"],
        "split": split,
        "seed": int(seed),
        "description": variant["description"],
        "data_protocol": "reactzyme-unseen-reaction-validation-v1",
        "test_config": bool(test),
        "no_enzyme_text": True,
    }


def _make_validation_config(
    train: dict[str, Any],
    *,
    protocol_root: Path,
    split: str,
) -> dict[str, Any]:
    config = copy.deepcopy(train)
    data = config["data"]
    base = protocol_root / split
    data["test_pairs_path"] = os.path.relpath(base / "validation_pairs.csv", ROOT)
    data["test_reactions_path"] = os.path.relpath(base / "validation_rxns.csv", ROOT)
    data["reaction_t5v2_embeds_path"] = data["validation_reaction_t5v2_embeds_path"]
    data["reaction_unimol2_embeds_path"] = data["validation_reaction_unimol2_embeds_path"]
    data["reaction_chiro_embeds_path"] = data["validation_reaction_chiro_embeds_path"]
    data["validation_retrieval_candidate_ids_path"] = os.path.relpath(
        base / "validation_candidate_ids.txt", ROOT
    )
    _remove_training_labels(config)
    config["logging"]["wandb"]["enabled"] = False
    config["ablation"]["evaluation_subset"] = "validation"
    return config


def _make_mining_config(
    train: dict[str, Any],
    *,
    protocol_root: Path,
    split: str,
) -> dict[str, Any]:
    config = copy.deepcopy(train)
    data = config["data"]
    base = protocol_root / split
    data["test_pairs_path"] = os.path.relpath(base / "train_pairs.csv", ROOT)
    data["test_reactions_path"] = os.path.relpath(base / "train_rxns.csv", ROOT)
    data["reaction_t5v2_embeds_path"] = data["train_reaction_t5v2_embeds_path"]
    data["reaction_unimol2_embeds_path"] = data["train_reaction_unimol2_embeds_path"]
    data["reaction_chiro_embeds_path"] = data["train_reaction_chiro_embeds_path"]
    data["validation_retrieval_candidate_ids_path"] = os.path.relpath(
        base / "train_candidate_ids.txt", ROOT
    )
    _remove_training_labels(config)
    config["logging"]["wandb"]["enabled"] = False
    config["ablation"]["evaluation_subset"] = "train_hard_negative_mining"
    return config


def _parent_rows(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("entries", [])
    by_split = {str(row["split"]): {key: str(value) for key, value in row.items()} for row in rows}
    if set(by_split) != set(REACTZYME_SPLITS):
        raise ValueError(f"Promotion {path} does not contain all ReactZyme splits")
    return by_split


def _make_pretrain_configs(
    *,
    train: dict[str, Any],
    variant: dict[str, Any],
    run_root: Path,
    split: str,
    run_key: str,
    args: SimpleNamespace,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    overlay = variant["overlay"]
    biological = None
    ec = None
    families = list(overlay.get("biological_pretrain_families", []))
    if families:
        train_with_targets = copy.deepcopy(train)
        paths = reactzyme_matrix.setup_paths(run_root, split)
        train_with_targets["data"].update(
            {
                "protein_biofp_targets_path": str(paths["biofp_npz"]),
                "protein_biofp_vocab_path": str(paths["biofp_vocab"]),
                "biofp_missing_policy": "zero_with_mask",
            }
        )
        pretrain_variant = {
            **_base_variant(),
            "id": run_key,
            "label": variant["label"],
            "pretrain_families": families,
        }
        biological = reactzyme_matrix.make_biological_pretrain_config(
            train_with_targets,
            run_root=run_root,
            split=split,
            variant=pretrain_variant,
            args=args,
        )
        train["training"]["biofp_pretrain_checkpoint"] = str(
            run_root / "checkpoints" / split / run_key / "biological_pretrain" / "best.ckpt"
        )
        paths = reactzyme_matrix.setup_paths(run_root, split)
        train["data"].update(
            {
                "protein_biofp_targets_path": str(paths["biofp_npz"]),
                "protein_biofp_vocab_path": str(paths["biofp_vocab"]),
                "biofp_missing_policy": "zero_with_mask",
            }
        )
    if overlay.get("ec_pretrain", False):
        pretrain_variant = {
            **_base_variant(),
            "id": run_key,
            "label": variant["label"],
            "ec_pretrain": True,
        }
        ec = reactzyme_matrix.make_ec_pretrain_config(
            run_root=run_root,
            split=split,
            variant=pretrain_variant,
            train_config=train,
            args=args,
        )
        ec["hyp_dim"] = int(train["model"]["enzyme_block_fusion"]["dims"]["ec"])
    return biological, ec


def generate_stage(
    *,
    run_root: Path,
    protocol_root: Path,
    feature_root: Path,
    stage: str,
    seeds: list[int],
    selected: set[str] | None,
    parent_promotion: Path | None,
    max_epochs: int,
    train_batch_size: int,
    retrieval_batch_size: int,
    validation_retrieval_batch_size: int,
    num_workers: int,
    python_bin: Path,
    setup_python_bin: Path,
    wandb_project: str,
    wandb_entity: str,
    wandb_mode: str,
    base_master_port: int,
) -> Path:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    spec = load_spec()
    variants = variants_for_stage(spec, stage, selected)
    parents = _parent_rows(parent_promotion)
    if stage != "recipe" and not parents:
        raise ValueError(f"{stage} requires --parent-promotion")
    rows: list[dict[str, Any]] = []
    for variant_index, variant in enumerate(variants):
        for seed_index, seed in enumerate(seeds):
            args = _generator_args(
                run_root=run_root,
                protocol_root=protocol_root,
                feature_root=feature_root,
                seed=seed,
                python_bin=python_bin,
                setup_python_bin=setup_python_bin,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                wandb_mode=wandb_mode,
                max_epochs=max_epochs,
                train_batch_size=train_batch_size,
                retrieval_batch_size=retrieval_batch_size,
                validation_retrieval_batch_size=validation_retrieval_batch_size,
                num_workers=num_workers,
            )
            for split_index, split in enumerate(REACTZYME_SPLITS):
                if parents:
                    train = _load_yaml(Path(parents[split]["config"]))
                    test = _load_yaml(Path(parents[split]["test_config"]))
                else:
                    train, test = _base_configs(
                        split=split,
                        run_root=run_root,
                        protocol_root=protocol_root,
                        feature_root=feature_root,
                        args=args,
                    )
                run_key = f"{stage}_{variant['id']}_seed{seed}"
                _apply_overlay(
                    train,
                    variant["overlay"],
                    run_root=run_root,
                    split=split,
                    run_key=run_key,
                    seed=seed,
                )
                _apply_overlay(
                    test,
                    variant["overlay"],
                    run_root=run_root,
                    split=split,
                    run_key=run_key,
                    seed=seed,
                )
                if stage == "hard_negative":
                    continuation_epochs = int(variant["overlay"].get("continuation_epochs", 5))
                    parent_epochs = int(train["training"].get("max_epochs", max_epochs))
                    train["training"]["max_epochs"] = parent_epochs + continuation_epochs
                else:
                    train["training"]["max_epochs"] = int(max_epochs)
                _configure_identity(
                    train,
                    run_root=run_root,
                    stage=stage,
                    variant=variant,
                    split=split,
                    seed=seed,
                    run_key=run_key,
                    test=False,
                )
                _configure_identity(
                    test,
                    run_root=run_root,
                    stage=stage,
                    variant=variant,
                    split=split,
                    seed=seed,
                    run_key=run_key,
                    test=True,
                )
                biological, ec = _make_pretrain_configs(
                    train=train,
                    variant=variant,
                    run_root=run_root,
                    split=split,
                    run_key=run_key,
                    args=args,
                )
                validation = _make_validation_config(
                    train,
                    protocol_root=protocol_root,
                    split=split,
                )
                mining = _make_mining_config(
                    train,
                    protocol_root=protocol_root,
                    split=split,
                )
                config_dir = run_root / "configs" / stage / run_key / split
                train_path = config_dir / "train.yaml"
                validation_path = config_dir / "validation.yaml"
                test_path = config_dir / "test.yaml"
                mining_path = config_dir / "mining.yaml"
                _write_yaml(train_path, train)
                _write_yaml(validation_path, validation)
                _write_yaml(test_path, test)
                _write_yaml(mining_path, mining)
                biological_path = "-"
                ec_path = "-"
                if biological is not None:
                    biological_file = config_dir / "biological_pretrain.yaml"
                    _write_yaml(biological_file, biological)
                    biological_path = str(biological_file)
                if ec is not None:
                    ec_file = config_dir / "ec_pretrain.yaml"
                    _write_yaml(ec_file, ec)
                    ec_path = str(ec_file)
                checkpoint_dir = run_root / "checkpoints" / split / run_key
                checkpoint_sidecar = checkpoint_dir / "selected_checkpoint.json"
                hard_negative_json = run_root / "hard_negatives" / split / run_key / "pools.json"
                parent_checkpoint = parents.get(split, {}).get("checkpoint", "-")
                rows.append(
                    {
                        "stage": stage,
                        "variant": variant["id"],
                        "label": variant["label"],
                        "seed": seed,
                        "split": split,
                        "run_id": f"{stage}_{variant['id']}_seed{seed}_{split}",
                        "config": train_path,
                        "validation_config": validation_path,
                        "mining_config": mining_path,
                        "test_config": test_path,
                        "checkpoint_dir": checkpoint_dir,
                        "checkpoint_sidecar": checkpoint_sidecar,
                        "validation_eval_json": run_root
                        / "validation"
                        / stage
                        / run_key
                        / f"{split}.json",
                        "test_eval_json": run_root / "test" / stage / run_key / f"{split}.json",
                        "wandb_run_name": train["logging"]["wandb"]["run_name"],
                        "master_port": base_master_port
                        + variant_index * 100
                        + seed_index * 10
                        + split_index,
                        "parent_checkpoint": parent_checkpoint if stage == "hard_negative" else "-",
                        "biological_pretrain_config": biological_path,
                        "ec_pretrain_config": ec_path,
                        "hard_negative_json": (
                            hard_negative_json
                            if variant["overlay"].get("model_hard_negatives", False)
                            else "-"
                        ),
                        "description": variant["description"],
                    }
                )
    plan_path = run_root / "plans" / f"{stage}.tsv"
    _write_tsv(plan_path, rows, PLAN_FIELDS)
    _write_json(
        run_root / "plans" / f"{stage}.json",
        {
            "schema_version": "reactzyme_geometry_stage_plan_v1",
            "stage": stage,
            "seeds": seeds,
            "variants": variants,
            "parent_promotion": None if parent_promotion is None else str(parent_promotion),
            "plan": str(plan_path),
            "runs": len(rows),
        },
    )
    return plan_path


def _metric(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"Missing validation metric {key}")
    return float(value)


def _variant_statistics(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    stats: dict[str, dict[str, Any]] = {}
    for variant, items in grouped.items():
        by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in items:
            eval_path = Path(row["validation_eval_json"])
            if not eval_path.is_file():
                raise FileNotFoundError(f"Missing validation evaluation: {eval_path}")
            by_split[row["split"]].append(json.loads(eval_path.read_text(encoding="utf-8")))
        if set(by_split) != set(REACTZYME_SPLITS):
            raise ValueError(f"{variant} does not have all three split evaluations")
        split_stats = {}
        for split, payloads in by_split.items():
            split_stats[split] = {
                "balanced_mrr": sum(_metric(item, "balanced_mrr") for item in payloads)
                / len(payloads),
                "r2e_mrr": sum(
                    _metric(item, "reaction_to_enzyme/mrr") for item in payloads
                )
                / len(payloads),
                "e2r_mrr": sum(
                    _metric(item, "enzyme_to_reaction/mrr") for item in payloads
                )
                / len(payloads),
                "seeds": len(payloads),
            }
        stats[variant] = {
            "splits": split_stats,
            "macro_balanced_mrr": sum(
                item["balanced_mrr"] for item in split_stats.values()
            )
            / len(split_stats),
            "reaction_smi_e2r_mrr": split_stats["reaction_smi"]["e2r_mrr"],
        }
    return stats


def promote(
    *,
    plan_path: Path,
    output_path: Path,
    top_k: int,
) -> dict[str, Any]:
    rows = _read_tsv(plan_path)
    if not rows:
        raise ValueError(f"Empty stage plan: {plan_path}")
    stage = rows[0]["stage"]
    source_stage = "recipe" if stage == "recipe_confirm" else stage
    spec = load_spec()
    stage_spec = spec["stages"][source_stage]
    baseline = str(stage_spec["baseline"])
    stats = _variant_statistics(rows)
    if baseline not in stats:
        raise ValueError(f"Baseline {baseline} is absent from {plan_path}")
    base = stats[baseline]
    gate = spec["promotion"]
    complexity = {
        variant["id"]: int(variant["complexity"])
        for variant in stage_spec["variants"]
    }
    eligible = []
    rejected: dict[str, list[str]] = {}
    for variant, item in stats.items():
        reasons = []
        if item["macro_balanced_mrr"] < (
            base["macro_balanced_mrr"] - float(gate["macro_balanced_drop"])
        ):
            reasons.append("macro balanced MRR gate")
        for split in REACTZYME_SPLITS:
            if item["splits"][split]["balanced_mrr"] < (
                base["splits"][split]["balanced_mrr"]
                - float(gate["per_split_balanced_drop"])
            ):
                reasons.append(f"{split} balanced MRR gate")
        if reasons:
            rejected[variant] = reasons
        else:
            eligible.append(variant)
    if not eligible:
        eligible = [baseline]

    tie = float(gate["reaction_smi_e2r_tie"])
    max_e2r = max(stats[variant]["reaction_smi_e2r_mrr"] for variant in eligible)
    leading = [
        variant
        for variant in eligible
        if max_e2r - stats[variant]["reaction_smi_e2r_mrr"] <= tie
    ]
    leading.sort(
        key=lambda variant: (
            -stats[variant]["macro_balanced_mrr"],
            complexity[variant],
            variant,
        )
    )
    remaining = sorted(
        (variant for variant in eligible if variant not in leading),
        key=lambda variant: (
            -stats[variant]["reaction_smi_e2r_mrr"],
            -stats[variant]["macro_balanced_mrr"],
            complexity[variant],
            variant,
        ),
    )
    selected = (leading + remaining)[:top_k]
    winner = selected[0]
    winner_rows = [row for row in rows if row["variant"] == winner]
    entries = []
    for split in REACTZYME_SPLITS:
        split_rows = [row for row in winner_rows if row["split"] == split]
        parent_row = next((row for row in split_rows if int(row["seed"]) == 42), split_rows[0])
        checkpoint = read_checkpoint_sidecar(Path(parent_row["checkpoint_sidecar"]))
        if checkpoint is None:
            raise FileNotFoundError(
                f"Missing selected checkpoint for {parent_row['run_id']}: "
                f"{parent_row['checkpoint_sidecar']}"
            )
        entries.append(
            {
                **parent_row,
                "checkpoint": checkpoint,
                "checkpoint_sha256": _sha256(Path(checkpoint)),
            }
        )
    payload = {
        "schema_version": "reactzyme_geometry_promotion_v1",
        "stage": stage,
        "source_plan": str(plan_path),
        "baseline": baseline,
        "winner": winner,
        "selected_variants": selected,
        "eligible_variants": eligible,
        "rejected": rejected,
        "statistics": stats,
        "gate": gate,
        "entries": entries,
    }
    _write_json(output_path, payload)
    return payload


def merge_plans(run_root: Path, output: Path) -> Path:
    rows = []
    for stage in STAGES:
        path = run_root / "plans" / f"{stage}.tsv"
        if path.is_file():
            rows.extend(_read_tsv(path))
    if not rows:
        raise ValueError(f"No stage plans found under {run_root}")
    _write_tsv(output, rows, PLAN_FIELDS)
    return output


def freeze_all(run_root: Path, output: Path) -> dict[str, Any]:
    plan = merge_plans(run_root, run_root / "plans" / "all_runs.tsv")
    entries = []
    for row in _read_tsv(plan):
        checkpoint = read_checkpoint_sidecar(Path(row["checkpoint_sidecar"]))
        if checkpoint is None:
            raise FileNotFoundError(f"Incomplete run: {row['run_id']}")
        entries.append(
            {
                **row,
                "checkpoint": checkpoint,
                "checkpoint_sha256": _sha256(Path(checkpoint)),
                "eval_json": row["test_eval_json"],
            }
        )
    payload = {
        "schema_version": "reactzyme_geometry_frozen_selection_v1",
        "source_plan": str(plan),
        "source_plan_sha256": _sha256(plan),
        "entries": entries,
    }
    _write_json(output, payload)
    return payload


def frozen_rows(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "reactzyme_geometry_frozen_selection_v1":
        raise ValueError(f"Unsupported frozen selection: {path}")
    rows = []
    for entry in payload.get("entries", []):
        checkpoint = Path(entry["checkpoint"])
        if not checkpoint.is_file() or _sha256(checkpoint) != entry["checkpoint_sha256"]:
            raise ValueError(f"Frozen checkpoint changed: {checkpoint}")
        rows.append({key: str(value) for key, value in entry.items()})
    return rows


def write_report(run_root: Path, output_csv: Path, output_markdown: Path) -> None:
    plan = merge_plans(run_root, run_root / "plans" / "all_runs.tsv")
    rows = []
    for row in _read_tsv(plan):
        eval_path = Path(row["test_eval_json"])
        payload = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.is_file() else {}
        r2e_first = payload.get(
            "reaction_to_enzyme/first_positive_mrr",
            payload.get("reaction_to_enzyme/mrr"),
        )
        e2r_first = payload.get(
            "enzyme_to_reaction/first_positive_mrr",
            payload.get("enzyme_to_reaction/mrr"),
        )
        r2e_paper = payload.get("reaction_to_enzyme/reactzyme_mrr")
        e2r_paper = payload.get("enzyme_to_reaction/reactzyme_mrr")
        rows.append(
            {
                "stage": row["stage"],
                "variant": row["variant"],
                "seed": row["seed"],
                "split": row["split"],
                "r2e_reactzyme_mrr": r2e_paper,
                "r2e_first_positive_mrr": r2e_first,
                "r2e_top1": payload.get("reaction_to_enzyme/top_1"),
                "r2e_top10": payload.get("reaction_to_enzyme/top_10"),
                "e2r_reactzyme_mrr": e2r_paper,
                "e2r_first_positive_mrr": e2r_first,
                "e2r_top1": payload.get("enzyme_to_reaction/top_1"),
                "e2r_top10": payload.get("enzyme_to_reaction/top_10"),
                "balanced_reactzyme_mrr": payload.get("balanced_reactzyme_mrr"),
                "balanced_first_positive_mrr": payload.get(
                    "balanced_first_positive_mrr",
                    payload.get("balanced_mrr"),
                ),
                "eval_json": str(eval_path),
            }
        )
    fields = list(rows[0])
    _write_tsv(output_csv, rows, fields)

    def fmt(value: Any) -> str:
        return "" if value is None or value == "" else f"{float(value):.4f}"

    lines = [
        "# ReactZyme B0 E->R Geometry Ablation",
        "",
        "Official released tests were evaluated only after every checkpoint identity was frozen.",
        "MRR uses ReactZyme's all-positive definition; first-positive MRR is shown only as a diagnostic.",
        "",
    ]
    for split in REACTZYME_SPLITS:
        lines.extend(
            [
                f"## {split}",
                "",
                "| Stage | Variant | Seed | R->E MRR | R->E First+ MRR | R->E Top1 | R->E Top10 | E->R MRR | E->R First+ MRR | E->R Top1 | E->R Top10 | Balanced MRR |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in (item for item in rows if item["split"] == split):
            lines.append(
                f"| {row['stage']} | {row['variant']} | {row['seed']} | "
                f"{fmt(row['r2e_reactzyme_mrr'])} | {fmt(row['r2e_first_positive_mrr'])} | "
                f"{fmt(row['r2e_top1'])} | {fmt(row['r2e_top10'])} | "
                f"{fmt(row['e2r_reactzyme_mrr'])} | {fmt(row['e2r_first_positive_mrr'])} | "
                f"{fmt(row['e2r_top1'])} | {fmt(row['e2r_top10'])} | "
                f"{fmt(row['balanced_reactzyme_mrr'])} |"
            )
        lines.append("")
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


def init_run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    protocol_root = args.protocol_root.resolve()
    source_root = args.source_root.resolve()
    if not (protocol_root / "manifest.json").is_file() or args.overwrite_protocol:
        materialize_unseen_reaction_protocols(
            source_root=source_root,
            out_root=protocol_root,
            validation_fraction=0.10,
            seed=42,
            overwrite=protocol_root.exists(),
        )
    run_root.mkdir(parents=True, exist_ok=True)
    setup_rows = []
    for split in REACTZYME_SPLITS:
        base = protocol_root / split
        paths = reactzyme_matrix.setup_paths(run_root, split)
        setup_rows.append(
            {
                "split": split,
                "train_pairs": os.path.relpath(base / "train_pairs.csv", ROOT),
                "train_reactions": os.path.relpath(base / "train_rxns.csv", ROOT),
                "candidate_ids": os.path.relpath(base / "train_candidate_ids.txt", ROOT),
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
    _write_tsv(
        run_root / "setup_plan.tsv",
        setup_rows,
        (
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
        ),
    )
    runtime = {
        "ROOT": ROOT,
        "RUN_ROOT": run_root,
        "PROTOCOL_ROOT": protocol_root,
        "FEATURE_ROOT": args.feature_root.resolve(),
        "PYTHON_BIN": _absolute_without_resolving_symlinks(args.python_bin),
        "SETUP_PYTHON_BIN": _absolute_without_resolving_symlinks(args.setup_python_bin),
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_MODE": args.wandb_mode,
        "GPUS": args.gpus,
        "BASE_MASTER_PORT": 24800,
        "EVALUATION_PROTOCOL": TEST_PROTOCOL,
        "COFACTOR_DICTIONARY": reactzyme_matrix.COFACTOR_DICTIONARY,
        "ENZYME_COFACTOR_LABELS": reactzyme_matrix.ENZYME_COFACTOR_LABELS,
        "EC_SOURCE": reactzyme_matrix.EC_SOURCE,
        "CLEANED_UNIPROT_RHEA": reactzyme_matrix.CLEANED_UNIPROT_RHEA,
        "RHEA_MOLECULES": reactzyme_matrix.RHEA_MOLECULES,
    }
    (run_root / "runtime.env").write_text(
        "\n".join(f"{key}={shlex.quote(str(value))}" for key, value in runtime.items()) + "\n",
        encoding="utf-8",
    )
    launcher = ROOT / "scripts/run_reactzyme_geometry_ablation_matrix.sh"
    wrapper = run_root / "run.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(ROOT))}\n"
        f"exec {shlex.quote(str(launcher))} {shlex.quote(str(run_root))} \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    manifest = {
        "schema_version": "reactzyme_b0_e2r_geometry_run_v1",
        "run_root": str(run_root),
        "protocol_root": str(protocol_root),
        "source_root": str(source_root),
        "feature_root": str(args.feature_root.resolve()),
        "matrix_spec": str(SPEC_PATH),
        "matrix_spec_sha256": _sha256(SPEC_PATH),
        "wandb": {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "mode": args.wandb_mode,
        },
        "execution": (
            "Three independent split-specific four-GPU DDP jobs run concurrently per "
            "variant wave on the same GPU set. Variant waves are sequential."
        ),
        "validation_protocol": VALIDATION_PROTOCOL,
        "test_protocol": TEST_PROTOCOL,
    }
    _write_json(run_root / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    init.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    init.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    init.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    init.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    init.add_argument("--setup-python-bin", type=Path, default=DEFAULT_SETUP_PYTHON)
    init.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    init.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY)
    init.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    init.add_argument("--gpus", default="0,1,2,3")
    init.add_argument("--overwrite-protocol", action="store_true")

    generate = subparsers.add_parser("generate-stage")
    generate.add_argument("--run-root", type=Path, required=True)
    generate.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    generate.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    generate.add_argument("--stage", choices=STAGES, required=True)
    generate.add_argument("--seed", type=int, action="append", dest="seeds")
    generate.add_argument("--variant", action="append", dest="variants")
    generate.add_argument("--parent-promotion", type=Path, default=None)
    generate.add_argument("--max-epochs", type=int, default=30)
    generate.add_argument("--train-batch-size", type=int, default=512)
    generate.add_argument("--retrieval-batch-size", type=int, default=64)
    generate.add_argument("--validation-retrieval-batch-size", type=int, default=128)
    generate.add_argument("--num-workers", type=int, default=2)
    generate.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    generate.add_argument("--setup-python-bin", type=Path, default=DEFAULT_SETUP_PYTHON)
    generate.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    generate.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY)
    generate.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    generate.add_argument("--base-master-port", type=int, default=24800)

    promotion = subparsers.add_parser("promote")
    promotion.add_argument("--plan", type=Path, required=True)
    promotion.add_argument("--output", type=Path, required=True)
    promotion.add_argument("--top-k", type=int, default=1)

    merge = subparsers.add_parser("merge-plans")
    merge.add_argument("--run-root", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)

    freeze = subparsers.add_parser("freeze-all")
    freeze.add_argument("--run-root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    selection = subparsers.add_parser("selection-rows")
    selection.add_argument("selection", type=Path)

    report = subparsers.add_parser("write-report")
    report.add_argument("--run-root", type=Path, required=True)
    report.add_argument("--output-csv", type=Path, required=True)
    report.add_argument("--output-markdown", type=Path, required=True)

    checkpoint = subparsers.add_parser("write-checkpoint")
    checkpoint.add_argument("sidecar", type=Path)
    checkpoint.add_argument("run_id")
    checkpoint.add_argument("checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init":
        print(json.dumps(init_run(args), indent=2, sort_keys=True))
    elif args.command == "generate-stage":
        plan = generate_stage(
            run_root=args.run_root.resolve(),
            protocol_root=args.protocol_root.resolve(),
            feature_root=args.feature_root.resolve(),
            stage=args.stage,
            seeds=args.seeds or [42],
            selected=None if args.variants is None else set(args.variants),
            parent_promotion=args.parent_promotion,
            max_epochs=args.max_epochs,
            train_batch_size=args.train_batch_size,
            retrieval_batch_size=args.retrieval_batch_size,
            validation_retrieval_batch_size=args.validation_retrieval_batch_size,
            num_workers=args.num_workers,
            python_bin=_absolute_without_resolving_symlinks(args.python_bin),
            setup_python_bin=_absolute_without_resolving_symlinks(args.setup_python_bin),
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_mode=args.wandb_mode,
            base_master_port=args.base_master_port,
        )
        print(plan)
    elif args.command == "promote":
        print(
            json.dumps(
                promote(plan_path=args.plan, output_path=args.output, top_k=args.top_k),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "merge-plans":
        print(merge_plans(args.run_root, args.output))
    elif args.command == "freeze-all":
        print(json.dumps(freeze_all(args.run_root, args.output), indent=2, sort_keys=True))
    elif args.command == "selection-rows":
        fields = (
            "stage",
            "variant",
            "seed",
            "split",
            "run_id",
            "checkpoint",
            "checkpoint_sha256",
            "test_config",
            "test_eval_json",
        )
        writer = csv.DictWriter(
            os.sys.stdout,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in frozen_rows(args.selection):
            writer.writerow({field: row[field] for field in fields})
    elif args.command == "write-report":
        write_report(args.run_root, args.output_csv, args.output_markdown)
        print(args.output_markdown)
    elif args.command == "write-checkpoint":
        write_checkpoint_sidecar(args.sidecar, run_id=args.run_id, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
