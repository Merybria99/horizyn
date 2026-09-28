"""Reproduce Q5 from Q0/F3 while jointly training E2R and R2E adapters."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("time", "enzyme_smi", "reaction_smi")
VARIANT = "Q6_bidirectional_adapters"
LABEL = "Q6"
ENZYME_BLOCK_DIMS = {
    "core": 288,
    "site": 96,
    "mechanism": 64,
    "cofactor": 32,
    "ec": 32,
}
ENZYME_BLOCK_WEIGHTS = {
    "core": 0.55,
    "site": 0.20,
    "mechanism": 0.12,
    "cofactor": 0.08,
    "ec": 0.05,
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _parent_checkpoint(source_run_root: Path, split: str) -> Path:
    return source_run_root / "checkpoints" / "seed42" / split / "Q0_f3" / "selected.ckpt"


def _selected_checkpoint(run_root: Path, split: str) -> Path:
    return run_root / "checkpoints" / "seed42" / split / VARIANT / "selected.ckpt"


def _configure_q6(
    config: dict[str, Any],
    *,
    run_root: Path,
    source_run_root: Path,
    split: str,
    project: str,
    entity: str,
    mode: str,
    wandb_enabled: bool,
) -> None:
    parent_checkpoint = _parent_checkpoint(source_run_root, split)
    if not parent_checkpoint.is_file():
        raise FileNotFoundError(f"Missing selected Q0/F3 parent checkpoint: {parent_checkpoint}")

    config["seed"] = 42
    model = config.setdefault("model", {})
    model["r2e_adapter"] = {
        "enabled": True,
        "hidden_dim": 512,
        "dropout": 0.1,
        "gate_init": 0.1,
        "use_factorized_inputs": True,
        "block_dims": copy.deepcopy(ENZYME_BLOCK_DIMS),
        "block_weights": copy.deepcopy(ENZYME_BLOCK_WEIGHTS),
    }

    training = config.setdefault("training", {})
    training.update(
        {
            "training_stage": "bidirectional_adapters",
            "init_from_checkpoint": str(parent_checkpoint.resolve()),
            "max_epochs": 15,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            # Preserve Q5's E2R hard-negative batch sampler under DDP.
            "use_distributed_sampler": False,
            "validation_retrieval_directions": [
                "reaction_to_enzyme",
                "enzyme_to_reaction",
            ],
        }
    )
    training["early_stopping"] = {
        "enabled": True,
        "monitor": "val/balanced_mrr",
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
        "lambda_r2e": 1.0,
        "lambda_e2r": 1.0,
        "lambda_direction_gap": 0.0,
        "lambda_rr": 0.0,
        "lambda_ee": 0.0,
        "lambda_gw": 0.0,
        "lambda_r2e_hard_neg": 0.0,
        "r2e_hard_neg_top_k": 0,
        "r2e_hard_neg_margin": 0.0,
        "lambda_e2r_hard_neg": 0.25,
        "e2r_hard_neg_top_k": 64,
        "e2r_hard_neg_margin": 0.05,
        "e2r_identity_weight": 0.05,
        "r2e_identity_weight": 0.05,
        "biofp_aux_weight": 0.0,
        "biofp_family_weights": {},
        "biofp_confidence_cap": 1.0,
    }

    logging = config.setdefault("logging", {})
    logging.update(
        {
            "log_dir": str(run_root / "logs" / "seed42" / split / VARIANT / "train"),
            "checkpoint_dir": str(run_root / "checkpoints" / "seed42" / split / VARIANT),
            "checkpoint_monitor": "val/balanced_mrr",
            "checkpoint_mode": "max",
        }
    )
    logging["wandb"] = {
        "enabled": wandb_enabled,
        "project": project,
        "entity": entity,
        "run_name": f"reactzyme-bidirectional-Q6-{split}-seed42",
        "mode": mode,
        "tags": [
            "reactzyme",
            "paper-protocol",
            "bidirectional-adapters",
            "q5-reproduction",
            "q0-f3-parent",
            "e2r-adapter",
            "r2e-adapter",
            split,
            "Q6",
            "seed-42",
        ],
        "log_model": False,
    }
    config["ablation"] = {
        "run_id": f"seed42_{split}_{VARIANT}",
        "split": split,
        "variant": VARIANT,
        "label": LABEL,
        "description": (
            "Frozen Q0/F3 parent with Q5 dense chemistry, factorized modalities, "
            "E2R hard negatives, and jointly trained E2R and R2E residual adapters."
        ),
        "matrix": "bidirectional_adapter",
        "parent": "Q0_f3",
        "selection_subset": "validation",
        "final_evaluation_subset": "test",
        "evaluation_protocol": "paper_test_candidates",
        "no_enzyme_text": True,
    }


def generate_campaign(
    *,
    run_root: Path,
    source_run_root: Path,
    project: str,
    entity: str,
    mode: str,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    source_run_root = source_run_root.resolve()
    manifest: dict[str, Any] = {
        "schema_version": "reactzyme_bidirectional_adapter_v1",
        "run_root": str(run_root),
        "source_run_root": str(source_run_root),
        "wandb_project": project,
        "wandb_entity": entity,
        "variant": VARIANT,
        "splits": list(SPLITS),
        "selection": "split-specific validation harmonic R2E/E2R MRR",
        "runs": [],
    }

    for split in SPLITS:
        source_dir = source_run_root / "configs" / "seed42" / "Q5_combined" / split
        config_dir = run_root / "configs" / "seed42" / VARIANT / split
        paths: dict[str, str] = {}
        for subset in ("train", "validation", "test"):
            source_path = source_dir / f"{subset}.yaml"
            config = load_yaml(source_path)
            _configure_q6(
                config,
                run_root=run_root,
                source_run_root=source_run_root,
                split=split,
                project=project,
                entity=entity,
                mode=mode,
                wandb_enabled=subset == "train",
            )
            if subset != "train":
                config["ablation"]["evaluation_subset"] = subset
            output_path = config_dir / f"{subset}.yaml"
            write_yaml(output_path, config)
            paths[subset] = str(output_path)
        manifest["runs"].append(
            {
                "seed": 42,
                "split": split,
                "variant": VARIANT,
                "parent_checkpoint": str(_parent_checkpoint(source_run_root, split)),
                "checkpoint": str(_selected_checkpoint(run_root, split)),
                "train_config": paths["train"],
                "validation_config": paths["validation"],
                "test_config": paths["test"],
            }
        )

    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_root / "runtime.env").write_text(
        "\n".join(
            (
                f"ROOT={ROOT}",
                f"RUN_ROOT={run_root}",
                f"SOURCE_RUN_ROOT={source_run_root}",
                f"PYTHON_BIN={ROOT.parent / 'env/bin/python'}",
                f"WANDB_PROJECT={project}",
                f"WANDB_ENTITY={entity}",
                f"WANDB_MODE={mode}",
                "SHORT_TMPDIR=/datastor2/deep-proteins/EnzymeDiscovery/.q6tmp",
                "GPUS=0,1,2,3",
                "BASE_MASTER_PORT=28900",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "runs/reactzyme_bidirectional_adapter_v1",
    )
    parser.add_argument(
        "--source-run-root",
        type=Path,
        default=ROOT / "runs/reactzyme_e2r_pareto_v1",
    )
    parser.add_argument(
        "--wandb-project",
        default="reactzyme-bidirectional-adapter-v1",
    )
    parser.add_argument("--wandb-entity", default="omnai")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = generate_campaign(
        run_root=args.run_root,
        source_run_root=args.source_run_root,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
    )
    print(f"Generated {len(manifest['runs'])} Q6 run definitions in {manifest['run_root']}")


if __name__ == "__main__":
    main()
