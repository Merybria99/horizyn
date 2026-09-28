"""Generate the directional E2R Pareto campaign from the exact F3 recipe."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("time", "enzyme_smi", "reaction_smi")
VARIANTS: dict[str, dict[str, Any]] = {
    "Q0_f3": {
        "label": "Q0",
        "description": "Fresh exact F3 control.",
    },
    "Q1_e2r_adapter": {
        "label": "Q1",
        "description": "Frozen F3 with an E2R-only residual reaction adapter.",
    },
    "Q2_e2r_hardneg": {
        "label": "Q2",
        "description": "Q1 plus E2R mined hard-negative batches and loss.",
        "hard_negative": True,
    },
    "Q3_dense_transform": {
        "label": "Q3",
        "description": "Q1 plus train-fitted dense reaction transformations.",
        "dense": True,
    },
    "Q4_factorized_modalities": {
        "label": "Q4",
        "description": "Q1 plus F4-style factorized raw modality inputs.",
        "factorized": True,
    },
    "Q5_combined": {
        "label": "Q5",
        "description": "Combined E2R adapter, hard negatives, dense chemistry, and factorization.",
        "hard_negative": True,
        "dense": True,
        "factorized": True,
    },
}
DENSE_REACTION_DIM = 1114


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _selected_checkpoint(
    run_root: Path,
    seed: int,
    split: str,
    variant: str,
) -> Path:
    return run_root / "checkpoints" / f"seed{seed}" / split / variant / "selected.ckpt"


def _dense_paths(run_root: Path, split: str) -> dict[str, Path]:
    root = run_root / "data" / split / "dense_reaction"
    return {
        subset: root / f"{subset}_dense_reaction_features.npz"
        for subset in ("train", "validation", "test")
    }


def _hard_negative_path(run_root: Path, split: str) -> Path:
    return (
        run_root
        / "hard_negatives"
        / "seed42"
        / split
        / "Q1_e2r_adapter.json"
    )


def _configure_logging(
    config: dict[str, Any],
    *,
    run_root: Path,
    seed: int,
    split: str,
    variant: str,
    project: str,
    entity: str,
    mode: str,
    enabled: bool,
) -> None:
    variant_spec = VARIANTS[variant]
    logging = config.setdefault("logging", {})
    logging["log_dir"] = str(
        run_root / "logs" / f"seed{seed}" / split / variant / "train"
    )
    logging["checkpoint_dir"] = str(
        run_root / "checkpoints" / f"seed{seed}" / split / variant
    )
    if variant != "Q0_f3":
        logging["checkpoint_monitor"] = "val/enzyme_to_reaction/reactzyme_mrr"
        logging["checkpoint_mode"] = "max"
    wandb = logging.setdefault("wandb", {})
    wandb.update(
        {
            "enabled": enabled,
            "project": project,
            "entity": entity,
            "run_name": (
                f"reactzyme-e2r-{variant_spec['label']}-{split}-seed{seed}"
            ),
            "mode": mode,
            "log_model": False,
        }
    )
    wandb["tags"] = [
        "reactzyme",
        "paper-protocol",
        "e2r-pareto",
        "f3-parent",
        "no-enzyme-text",
        split,
        variant_spec["label"],
        f"seed-{seed}",
    ]


def _configure_variant(
    config: dict[str, Any],
    *,
    run_root: Path,
    seed: int,
    split: str,
    variant: str,
) -> None:
    spec = VARIANTS[variant]
    config["seed"] = seed
    model = config.setdefault("model", {})
    data = config.setdefault("data", {})
    training = config.setdefault("training", {})
    loss = training.setdefault("loss", {})

    model.pop("e2r_adapter", None)
    data.pop("reaction_load_directional", None)
    for key in (
        "train_reaction_directional_vectors_path",
        "validation_reaction_directional_vectors_path",
        "reaction_directional_vectors_path",
        "hard_negative_pools_path",
        "hard_negative_direction",
        "hard_negative_anchor_queries_per_batch",
        "hard_negative_positives_per_query",
        "hard_negative_negatives_per_query",
        "hard_negative_seed",
    ):
        data.pop(key, None)

    if variant == "Q0_f3":
        training.pop("training_stage", None)
        training.pop("init_from_checkpoint", None)
        return

    dense = bool(spec.get("dense", False))
    factorized = bool(spec.get("factorized", False))
    model["e2r_adapter"] = {
        "enabled": True,
        "hidden_dim": 512,
        "dropout": 0.1,
        "gate_init": 0.1,
        "use_factorized_inputs": factorized,
        "use_directional_inputs": dense,
        "directional_dim": DENSE_REACTION_DIM if dense else None,
        "directional_hidden_dim": 128,
    }
    if not dense:
        model["e2r_adapter"].pop("directional_dim")

    training.update(
        {
            "training_stage": "e2r_adapter",
            "init_from_checkpoint": str(
                _selected_checkpoint(run_root, seed, split, "Q0_f3")
            ),
            "max_epochs": 15,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
        }
    )
    if spec.get("hard_negative"):
        # DirectionalHardNegativeBatchSampler shards batches by DDP rank.
        # Lightning must not replace its sampler with another distributed one.
        training["use_distributed_sampler"] = False
    training["early_stopping"] = {
        "enabled": True,
        "monitor": "val/enzyme_to_reaction/reactzyme_mrr",
        "mode": "max",
        "patience": 4,
        "min_delta": 1e-4,
    }
    training["loss"] = {
        "name": "MultiAlignmentRetrievalLoss",
        "beta": 10.0,
        "learn_beta": False,
        "beta_min": 0.01,
        "beta_max": 100.0,
        "positive_pair_source": "observed_pairs",
        "lambda_r2e": 0.0,
        "lambda_e2r": 1.0,
        "lambda_direction_gap": 0.0,
        "lambda_rr": 0.0,
        "lambda_ee": 0.0,
        "lambda_gw": 0.0,
        "lambda_r2e_hard_neg": 0.0,
        "r2e_hard_neg_top_k": 0,
        "r2e_hard_neg_margin": 0.0,
        "lambda_e2r_hard_neg": 0.25 if spec.get("hard_negative") else 0.0,
        "e2r_hard_neg_top_k": 64 if spec.get("hard_negative") else 0,
        "e2r_hard_neg_margin": 0.05 if spec.get("hard_negative") else 0.0,
        "e2r_identity_weight": 0.05,
        "biofp_aux_weight": 0.0,
        "biofp_family_weights": {},
        "biofp_confidence_cap": 1.0,
    }

    if dense:
        paths = _dense_paths(run_root, split)
        data.update(
            {
                "reaction_load_directional": True,
                "reaction_use_directional": False,
                "reaction_directional_dim": DENSE_REACTION_DIM,
                "reaction_allow_missing_directional": False,
                "train_reaction_directional_vectors_path": str(paths["train"]),
                "validation_reaction_directional_vectors_path": str(
                    paths["validation"]
                ),
            }
        )
    if spec.get("hard_negative"):
        data.update(
            {
                "hard_negative_pools_path": str(
                    _hard_negative_path(run_root, split)
                ),
                "hard_negative_direction": "enzyme_to_reaction",
                "hard_negative_anchor_queries_per_batch": 48,
                "hard_negative_positives_per_query": 2,
                "hard_negative_negatives_per_query": 8,
                "hard_negative_seed": seed,
            }
        )


def _configure_eval_subset(
    config: dict[str, Any],
    train_source: dict[str, Any],
    test_source: dict[str, Any],
    *,
    subset: str,
    run_root: Path,
    split: str,
    variant: str,
) -> None:
    data = config["data"]
    if subset == "test":
        source_data = test_source["data"]
    elif subset in {"validation", "train"}:
        source_data = train_source["data"]
        data["test_pairs_path"] = source_data[
            "validation_pairs_path" if subset == "validation" else "train_pairs_path"
        ]
        data["test_reactions_path"] = source_data[
            "validation_reactions_path"
            if subset == "validation"
            else "train_reactions_path"
        ]
        modality_subset = "validation" if subset == "validation" else "train"
        for short, split_key in (
            ("reaction_t5v2_embeds_path", f"{modality_subset}_reaction_t5v2_embeds_path"),
            ("reaction_unimol2_embeds_path", f"{modality_subset}_reaction_unimol2_embeds_path"),
            ("reaction_chiro_embeds_path", f"{modality_subset}_reaction_chiro_embeds_path"),
        ):
            data[short] = source_data[split_key]
        chemistry_key = f"{modality_subset}_reaction_chemistry_vectors_path"
        data["reaction_chemistry_vectors_path"] = source_data[chemistry_key]
        if subset == "validation":
            data["validation_retrieval_candidate_ids_path"] = source_data[
                "validation_retrieval_candidate_ids_path"
            ]
    else:
        raise ValueError(f"Unsupported evaluation subset: {subset}")

    if VARIANTS[variant].get("dense"):
        dense_subset = "train" if subset == "train" else subset
        data["reaction_directional_vectors_path"] = str(
            _dense_paths(run_root, split)[dense_subset]
        )


def generate_campaign(
    *,
    run_root: Path,
    source_run_root: Path,
    project: str,
    entity: str,
    mode: str,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    run_root = run_root.resolve()
    source_run_root = source_run_root.resolve()
    manifest: dict[str, Any] = {
        "schema_version": "reactzyme_e2r_pareto_v1",
        "run_root": str(run_root),
        "source_run_root": str(source_run_root),
        "wandb_project": project,
        "wandb_entity": entity,
        "splits": list(SPLITS),
        "seeds": list(seeds),
        "variants": VARIANTS,
        "selection": {
            "primary": "validation reaction_smi enzyme_to_reaction/reactzyme_mrr",
            "r2e_absolute_tolerance": 1e-6,
            "non_primary_e2r_drop_tolerance": 0.01,
            "promotion_seeds": [7, 23],
            "promotion_count": 2,
        },
        "runs": [],
    }

    for seed in seeds:
        for split in SPLITS:
            source_train_path = (
                source_run_root
                / "configs"
                / "F3_set_chemistry"
                / split
                / "train.yaml"
            )
            source_test_path = source_train_path.with_name("test.yaml")
            source_train = load_yaml(source_train_path)
            source_test = load_yaml(source_test_path)
            for variant, spec in VARIANTS.items():
                config_dir = run_root / "configs" / f"seed{seed}" / variant / split
                train = copy.deepcopy(source_train)
                _configure_variant(
                    train,
                    run_root=run_root,
                    seed=seed,
                    split=split,
                    variant=variant,
                )
                _configure_logging(
                    train,
                    run_root=run_root,
                    seed=seed,
                    split=split,
                    variant=variant,
                    project=project,
                    entity=entity,
                    mode=mode,
                    enabled=True,
                )
                train["ablation"] = {
                    "run_id": f"seed{seed}_{split}_{variant}",
                    "split": split,
                    "variant": variant,
                    "label": spec["label"],
                    "description": spec["description"],
                    "matrix": "e2r_pareto",
                    "parent": "Q0_f3" if variant != "Q0_f3" else None,
                    "selection_subset": "validation",
                    "final_evaluation_subset": "test",
                    "evaluation_protocol": "paper_test_candidates",
                    "no_enzyme_text": True,
                }
                train_path = config_dir / "train.yaml"
                write_yaml(train_path, train)

                eval_paths = {}
                for subset in ("validation", "test"):
                    evaluation = copy.deepcopy(source_test)
                    _configure_variant(
                        evaluation,
                        run_root=run_root,
                        seed=seed,
                        split=split,
                        variant=variant,
                    )
                    _configure_logging(
                        evaluation,
                        run_root=run_root,
                        seed=seed,
                        split=split,
                        variant=variant,
                        project=project,
                        entity=entity,
                        mode=mode,
                        enabled=False,
                    )
                    _configure_eval_subset(
                        evaluation,
                        source_train,
                        source_test,
                        subset=subset,
                        run_root=run_root,
                        split=split,
                        variant=variant,
                    )
                    evaluation["ablation"] = copy.deepcopy(train["ablation"])
                    evaluation["ablation"]["evaluation_subset"] = subset
                    eval_path = config_dir / f"{subset}.yaml"
                    write_yaml(eval_path, evaluation)
                    eval_paths[subset] = str(eval_path)

                mine = copy.deepcopy(source_test)
                _configure_variant(
                    mine,
                    run_root=run_root,
                    seed=seed,
                    split=split,
                    variant=variant,
                )
                _configure_logging(
                    mine,
                    run_root=run_root,
                    seed=seed,
                    split=split,
                    variant=variant,
                    project=project,
                    entity=entity,
                    mode=mode,
                    enabled=False,
                )
                _configure_eval_subset(
                    mine,
                    source_train,
                    source_test,
                    subset="train",
                    run_root=run_root,
                    split=split,
                    variant=variant,
                )
                mine_path = config_dir / "mine_train.yaml"
                write_yaml(mine_path, mine)

                manifest["runs"].append(
                    {
                        "seed": seed,
                        "split": split,
                        "variant": variant,
                        "train_config": str(train_path),
                        "validation_config": eval_paths["validation"],
                        "test_config": eval_paths["test"],
                        "mine_config": str(mine_path),
                        "checkpoint": str(
                            _selected_checkpoint(run_root, seed, split, variant)
                        ),
                    }
                )

    runtime = {
        "ROOT": ROOT,
        "RUN_ROOT": run_root,
        "SOURCE_RUN_ROOT": source_run_root,
        "PYTHON_BIN": ROOT.parent / "env/bin/python",
        "SETUP_PYTHON_BIN": ROOT.parent / ".capability-run-py/bin/python",
        "WANDB_PROJECT": project,
        "WANDB_ENTITY": entity,
        "WANDB_MODE": mode,
        "GPUS": "0,1,2,3",
        "BASE_MASTER_PORT": "28600",
        "EVALUATION_PROTOCOL": "paper_test_candidates",
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "runtime.env").write_text(
        "\n".join(f"{key}={value}" for key, value in runtime.items()) + "\n",
        encoding="utf-8",
    )
    write_json(run_root / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "runs/reactzyme_e2r_pareto_v1",
    )
    parser.add_argument(
        "--source-run-root",
        type=Path,
        default=ROOT / "runs/reactzyme_reaction_features_v1",
    )
    parser.add_argument("--wandb-project", default="reactzyme-e2r-pareto-v1")
    parser.add_argument("--wandb-entity", default="omnai")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--seeds", default="42,7,23")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    manifest = generate_campaign(
        run_root=args.run_root,
        source_run_root=args.source_run_root,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        seeds=seeds,
    )
    print(
        f"Generated {len(manifest['runs'])} run definitions under "
        f"{manifest['run_root']}"
    )


if __name__ == "__main__":
    main()
