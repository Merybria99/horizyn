"""Generate and select the post-ablation F3 geometry campaign."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from horizyn.benchmarks.reactzyme_f3_shortcut_campaign import (
    DEFAULT_BASE_ROOT,
    DEFAULT_FEATURE_ROOT,
    DEFAULT_PROTOCOL_ROOT,
    MODALITIES,
    _fixed_feature_path,
    _load_yaml,
    _rewrite_data,
    _select_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = ROOT / "runs/reactzyme_f3_geometry_v1"
SPLITS = ("time", "enzyme_smi", "reaction_smi")
SCREEN_SEED = 42
REPLICATE_SEEDS = (17, 73)
ALL_SEEDS = (SCREEN_SEED, *REPLICATE_SEEDS)
WANDB_PROJECT = "horizyn-reactzyme-f3-geometry-v1"
REACTION_R2E_GUARD = 0.01
FACTOR_DIMS = {
    "reaction_model": 128,
    "unimol2": 192,
    "chiro": 96,
    "reaction_chemistry": 96,
}
FACTOR_WEIGHTS = {
    "reaction_model": 0.25,
    "unimol2": 0.375,
    "chiro": 0.1875,
    "reaction_chemistry": 0.1875,
}
VARIANTS: dict[str, dict[str, Any]] = {
    "M0": {
        "label": "audited F3 control",
        "description": "Exact F3/S0 architecture and observed-pair FullBatchMLNCE loss.",
    },
    "M1": {
        "label": "normalized F3",
        "description": "M0 with scale-preserving modality L2 normalization.",
        "normalize": True,
    },
    "M2": {
        "label": "factorized gated residual",
        "description": (
            "F4 normalized modality blocks followed by a gate-0.10 residual MLP and "
            "cosine identity regularization at weight 0.02."
        ),
        "factorized_residual": True,
        "identity_weight": 0.02,
    },
    "M3": {
        "label": "factorized gated residual + consistency",
        "description": (
            "M2 with detached chemistry-free consistency regularization at weight 0.03."
        ),
        "factorized_residual": True,
        "identity_weight": 0.02,
        "consistency_weight": 0.03,
    },
}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _apply_variant(config: dict[str, Any], variant: str) -> None:
    spec = VARIANTS[variant]
    attention = config["model"]["reaction_multimodal_attention"]
    for key in (
        "modality_l2_normalize",
        "factorized_dims",
        "factorized_weights",
        "residual_gate_init",
        "chemistry_dropout",
        "attention_prior_weights",
        "attention_adaptation_strength",
    ):
        attention.pop(key, None)
    config["training"].pop("reaction_residual_identity", None)
    config["training"].pop("reaction_chemistry_consistency", None)
    attention["fusion"] = "attention"
    attention["output_projection"] = "mlp"

    if spec.get("normalize"):
        attention["modality_l2_normalize"] = True
    if spec.get("factorized_residual"):
        attention.update(
            {
                "fusion": "factorized_concat",
                "output_projection": "residual_mlp",
                "factorized_dims": copy.deepcopy(FACTOR_DIMS),
                "factorized_weights": copy.deepcopy(FACTOR_WEIGHTS),
                "residual_gate_init": 0.10,
            }
        )
    if "identity_weight" in spec:
        config["training"]["reaction_residual_identity"] = {
            "weight": float(spec["identity_weight"]),
        }
    if "consistency_weight" in spec:
        config["training"]["reaction_chemistry_consistency"] = {
            "weight": float(spec["consistency_weight"]),
        }


def _configure_execution(config: dict[str, Any]) -> None:
    """Preserve the audited global batch while assigning one H200 per run."""
    old_devices = int(config["training"].get("devices", 1))
    old_local_batch = int(config["data"]["train_batch_size"])
    config["data"]["train_batch_size"] = old_devices * old_local_batch
    config["training"]["devices"] = 1
    config["training"]["strategy"] = "auto"


def configure_run(
    base: dict[str, Any],
    *,
    run_root: Path,
    protocol_root: Path,
    feature_root: Path,
    variant: str,
    split: str,
    seed: int,
    evaluation_subset: str,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown geometry variant: {variant}")
    config = copy.deepcopy(base)
    config["seed"] = int(seed)
    _apply_variant(config, variant)
    _configure_execution(config)
    _rewrite_data(
        config,
        run_root=run_root,
        protocol_root=protocol_root,
        feature_root=feature_root,
        split=split,
        evaluation_subset=evaluation_subset,
    )

    run_id = f"{variant}_{split}_seed{seed}"
    config["logging"]["log_dir"] = str(run_root / "logs" / run_id / evaluation_subset)
    config["logging"]["checkpoint_dir"] = str(run_root / "checkpoints" / run_id)
    monitor = (
        "val/enzyme_to_reaction/reactzyme_mrr"
        if split == "reaction_smi"
        else "val/mean_bidirectional_reactzyme_mrr"
    )
    config["logging"].update(
        {
            "checkpoint_monitor": monitor,
            "checkpoint_mode": "max",
            "checkpoint_on_validation_end": True,
            "save_top_k": 3,
        }
    )
    config["logging"].setdefault("wandb", {}).update(
        {
            "enabled": evaluation_subset == "validation",
            "project": WANDB_PROJECT,
            "run_name": f"reactzyme-f3-geometry-{run_id}",
            "tags": [
                "reactzyme-f3-geometry",
                "one-stage",
                "single-head",
                variant,
                split,
                f"seed-{seed}",
            ],
        }
    )
    config["training"].setdefault("early_stopping", {}).update(
        {"enabled": True, "monitor": monitor, "mode": "max"}
    )
    config["ablation"] = {
        "campaign": "reactzyme_f3_geometry_v1",
        "run_id": run_id,
        "variant": variant,
        "split": split,
        "seed": int(seed),
        "evaluation_subset": evaluation_subset,
        "description": VARIANTS[variant]["description"],
        "audited_global_batch_size": int(config["data"]["train_batch_size"]),
        "released_test_final_only": True,
        "cluster_validation_used": False,
    }
    return config


def generate_campaign(
    run_root: Path = DEFAULT_RUN_ROOT,
    *,
    base_root: Path = DEFAULT_BASE_ROOT,
    protocol_root: Path = DEFAULT_PROTOCOL_ROOT,
    feature_root: Path = DEFAULT_FEATURE_ROOT,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    protocol_root = protocol_root.resolve()
    feature_root = feature_root.resolve()
    if not (protocol_root / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing audited protocol: {protocol_root}")

    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for split in SPLITS:
            train_base = _load_yaml(base_root / split / "train.yaml")
            test_base = _load_yaml(base_root / split / "test.yaml")
            for seed in ALL_SEEDS:
                train = configure_run(
                    train_base,
                    run_root=run_root,
                    protocol_root=protocol_root,
                    feature_root=feature_root,
                    variant=variant,
                    split=split,
                    seed=seed,
                    evaluation_subset="validation",
                )
                validation = configure_run(
                    test_base,
                    run_root=run_root,
                    protocol_root=protocol_root,
                    feature_root=feature_root,
                    variant=variant,
                    split=split,
                    seed=seed,
                    evaluation_subset="validation",
                )
                validation["data"]["test_pairs_path"] = str(
                    protocol_root / split / "validation_pairs.csv"
                )
                validation["data"]["test_reactions_path"] = str(
                    protocol_root / split / "validation_rxns.csv"
                )
                for config_name, artifact_name in MODALITIES:
                    validation["data"][f"reaction_{config_name}_embeds_path"] = str(
                        _fixed_feature_path(feature_root, split, "validation", artifact_name)
                    )
                validation["data"]["reaction_chemistry_vectors_path"] = str(
                    run_root
                    / "data"
                    / split
                    / "reaction_set"
                    / "validation_reaction_set_features.npz"
                )
                test = configure_run(
                    test_base,
                    run_root=run_root,
                    protocol_root=protocol_root,
                    feature_root=feature_root,
                    variant=variant,
                    split=split,
                    seed=seed,
                    evaluation_subset="test",
                )
                if variant == "M0":
                    if train["model"] != train_base["model"]:
                        raise AssertionError("M0 changed the F3 model architecture")
                    if train["training"]["loss"] != train_base["training"]["loss"]:
                        raise AssertionError("M0 changed the F3 retrieval loss")

                config_dir = run_root / "configs" / variant / split / f"seed{seed}"
                paths = {
                    "train": config_dir / "train.yaml",
                    "validation": config_dir / "validation.yaml",
                    "test": config_dir / "test.yaml",
                }
                _write_yaml(paths["train"], train)
                _write_yaml(paths["validation"], validation)
                _write_yaml(paths["test"], test)
                rows.append(
                    {
                        "variant": variant,
                        "split": split,
                        "seed": seed,
                        "train_config": str(paths["train"]),
                        "validation_config": str(paths["validation"]),
                        "test_config": str(paths["test"]),
                        "checkpoint_dir": train["logging"]["checkpoint_dir"],
                    }
                )

    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / "campaign.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": "reactzyme_f3_geometry_v1",
        "run_root": str(run_root),
        "base_root": str(base_root.resolve()),
        "protocol_root": str(protocol_root),
        "feature_root": str(feature_root),
        "wandb_project": WANDB_PROJECT,
        "variants": VARIANTS,
        "screen_seed": SCREEN_SEED,
        "replicate_seeds": list(REPLICATE_SEEDS),
        "reaction_r2e_guard": REACTION_R2E_GUARD,
        "selection": (
            "Use only standard validation. Reaction-Sim selects E->R with an M0-relative "
            "R->E guard of 0.01; other splits select mean bidirectional ReactZyme MRR. "
            "Promote at most two seed-42 variants, replicate seeds 17 and 73, then freeze "
            "one winner before official test access."
        ),
        "execution": {
            "gpus_per_run": 1,
            "train_batch_size_per_gpu": 2048,
            "effective_global_batch_size": 2048,
        },
    }
    _write_json(run_root / "campaign.json", manifest)
    runtime = run_root / "runtime.env"
    if not runtime.exists():
        runtime.write_text(
            "\n".join(
                (
                    f"PYTHON_BIN={ROOT.parent / 'env/bin/python'}",
                    f"SETUP_PYTHON_BIN={ROOT.parent / '.capability-run-py/bin/python'}",
                    f"WANDB_PROJECT={WANDB_PROJECT}",
                    "WANDB_ENTITY=omnai",
                    "WANDB_MODE=online",
                    "GPU_LIST=0,1,2,3",
                    "",
                )
            ),
            encoding="utf-8",
        )
    return manifest


def _validation_results(
    run_root: Path,
    variant: str,
    split: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    result_dir = run_root / "results" / "validation" / variant / split / f"seed{seed}"
    for path in sorted(result_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("checkpoint")
            and payload.get("enzyme_to_reaction/reactzyme_mrr") is not None
            and payload.get("reaction_to_enzyme/reactzyme_mrr") is not None
        ):
            rows.append(
                {
                    "path": str(path),
                    "checkpoint": payload["checkpoint"],
                    "e2r": payload["enzyme_to_reaction/reactzyme_mrr"],
                    "r2e": payload["reaction_to_enzyme/reactzyme_mrr"],
                }
            )
    return rows


def _seed_selection(
    run_root: Path,
    variants: list[str],
    seed: int,
    *,
    baseline_r2e: float,
) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for variant in variants:
        split_rows = {}
        for split in SPLITS:
            row = _select_checkpoint(
                _validation_results(run_root, variant, split, seed),
                split=split,
                reaction_r2e_floor=(
                    baseline_r2e - REACTION_R2E_GUARD
                    if split == "reaction_smi" and variant != "M0"
                    else None
                ),
            )
            if row is not None:
                split_rows[split] = row
        cells = [float(row[key]) for row in split_rows.values() for key in ("e2r", "r2e")]
        selected[variant] = {
            "splits": split_rows,
            "complete": len(split_rows) == len(SPLITS),
            "six_cell_macro_mrr": sum(cells) / len(cells) if cells else None,
        }
    return selected


def select_screen(run_root: Path, count: int = 2) -> dict[str, Any]:
    control = _select_checkpoint(
        _validation_results(run_root, "M0", "reaction_smi", SCREEN_SEED),
        split="reaction_smi",
        reaction_r2e_floor=None,
    )
    if control is None:
        raise RuntimeError("M0 Reaction-Sim validation is incomplete")
    baseline_r2e = float(control["r2e"])
    variants = _seed_selection(
        run_root,
        list(VARIANTS),
        SCREEN_SEED,
        baseline_r2e=baseline_r2e,
    )
    ranked = sorted(
        (
            (name, payload["six_cell_macro_mrr"])
            for name, payload in variants.items()
            if payload["complete"] and payload["six_cell_macro_mrr"] is not None
        ),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    if len(ranked) < count:
        raise RuntimeError(f"Need {count} complete variants, found {len(ranked)}")
    report = {
        "stage": "screen",
        "seed": SCREEN_SEED,
        "reaction_m0_r2e": baseline_r2e,
        "selected_variants": [name for name, _ in ranked[:count]],
        "variants": variants,
    }
    _write_json(run_root / "reports" / "screen_selection.json", report)
    (run_root / "reports" / "selected_variants.txt").write_text(
        "\n".join(report["selected_variants"]) + "\n",
        encoding="utf-8",
    )
    return report


def select_final(run_root: Path) -> dict[str, Any]:
    screen = json.loads((run_root / "reports/screen_selection.json").read_text())
    selected_variants = list(screen["selected_variants"])
    baseline_r2e = float(screen["reaction_m0_r2e"])
    per_seed = {
        str(seed): _seed_selection(
            run_root,
            selected_variants,
            seed,
            baseline_r2e=baseline_r2e,
        )
        for seed in ALL_SEEDS
    }
    summaries = {}
    for variant in selected_variants:
        payloads = [per_seed[str(seed)][variant] for seed in ALL_SEEDS]
        values = [payload["six_cell_macro_mrr"] for payload in payloads]
        complete = all(payload["complete"] for payload in payloads)
        summaries[variant] = {
            "complete": complete,
            "mean_six_cell_macro_mrr": (
                sum(float(value) for value in values) / len(values) if complete else None
            ),
        }
    eligible = [
        (name, payload["mean_six_cell_macro_mrr"])
        for name, payload in summaries.items()
        if payload["complete"] and payload["mean_six_cell_macro_mrr"] is not None
    ]
    if not eligible:
        raise RuntimeError("No replicated variant is complete")
    winner = max(eligible, key=lambda item: float(item[1]))[0]
    report = {"stage": "final", "winner": winner, "summaries": summaries, "seeds": per_seed}
    _write_json(run_root / "reports/final_selection.json", report)
    return report


def freeze_winner(run_root: Path) -> dict[str, Any]:
    selection_path = run_root / "reports/final_selection.json"
    if not selection_path.is_file():
        select_final(run_root)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    winner = selection["winner"]
    records = []
    for seed in ALL_SEEDS:
        for split in SPLITS:
            row = selection["seeds"][str(seed)][winner]["splits"][split]
            source = Path(row["checkpoint"])
            destination = run_root / "frozen" / winner / f"seed{seed}" / split / "best.ckpt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            records.append(
                {
                    "variant": winner,
                    "seed": seed,
                    "split": split,
                    "source": str(source),
                    "frozen": str(destination),
                    "sha256": digest,
                }
            )
    manifest = {
        "schema_version": "reactzyme_f3_geometry_frozen_v1",
        "winner": winner,
        "validation_selection": str(selection_path),
        "checkpoints": records,
    }
    _write_json(run_root / "frozen/manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "select-screen", "select-final", "freeze"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
        if command == "generate":
            subparser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
            subparser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
            subparser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_root = args.run_root.resolve()
    if args.command == "generate":
        result = generate_campaign(
            run_root,
            base_root=args.base_root,
            protocol_root=args.protocol_root,
            feature_root=args.feature_root,
        )
    elif args.command == "select-screen":
        result = select_screen(run_root)
    elif args.command == "select-final":
        result = select_final(run_root)
    else:
        result = freeze_winner(run_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
