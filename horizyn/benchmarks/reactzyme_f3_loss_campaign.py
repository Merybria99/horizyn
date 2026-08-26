"""Generate and summarize the canonical-isomeric F3 loss campaign."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = ROOT / "runs/reactzyme_f3_loss_ablation_v1"
DEFAULT_BASE_ROOT = ROOT / "runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry"
PROTOCOL_ROOT = ROOT / "data/revised_protocols/reactzyme_paper"
SPLITS = ("time", "enzyme_smi", "reaction_smi")
SUBSETS = ("train", "validation", "test")
SCREEN_SEED = 42
REPLICATE_SEEDS = (17, 73)
WANDB_PROJECT = "horizyn-reactzyme-f3-loss-ablation-v1"
VARIANT_ORDER = ("L0", "L1", "L3", "L4", "L2", "L5", "L6", "L7")


LOSS_VARIANTS: dict[str, dict[str, Any]] = {
    "L0": {"name": "FullBatchMLNCELoss"},
    "L1": {"name": "DegreeTemperedFullBatchMLNCELoss", "degree_alpha": 0.5},
    "L2": {"name": "BidirectionalAnchorBalancedSupConLoss"},
    "L3": {"name": "DecoupledAllPositiveInfoNCELoss"},
    "L4": {
        "name": "HybridCardinalityRetrievalLoss",
        "cardinality_weight": 0.3,
        "cardinality_warmup_epochs": 5,
    },
    "L5": {
        "name": "HybridCardinalityRetrievalLoss",
        "cardinality_weight": 0.3,
        "cardinality_warmup_epochs": 5,
        "separate_direction_temperatures": True,
        "beta_r2e": 10.0,
        "beta_e2r": 10.0,
        "beta_min": 3.0,
        "beta_max": 30.0,
        "temperature_regularization_weight": 1e-3,
    },
    "L6": {
        "name": "HybridCardinalityRetrievalLoss",
        "cardinality_weight": 0.3,
        "cardinality_warmup_epochs": 5,
        "soft_rank_weight": 0.05,
        "soft_rank_tau": 0.1,
        "soft_rank_top_k": 128,
    },
    "L7": {
        "name": "BalancedSigmoidEBMLoss",
        "sigmoid_bias_init": 0.0,
        "sigmoid_learn_bias": True,
        "sigmoid_negative_weight": 1.0,
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def normalized_reaction_path(run_root: Path, split: str, subset: str) -> Path:
    return run_root / "data/isomeric" / split / f"{subset}_rxns.csv"


def feature_path(run_root: Path, split: str, subset: str, modality: str) -> Path:
    return run_root / "data/features" / split / subset / f"{modality}.h5"


def set_feature_path(run_root: Path, split: str, subset: str) -> Path:
    return run_root / "data/features" / split / "reaction_set" / f"{subset}_reaction_set_features.npz"


def loss_config(variant: str, split: str) -> dict[str, Any]:
    if variant not in LOSS_VARIANTS:
        raise ValueError(f"Unknown loss variant: {variant}")
    directional = (
        {"lambda_r2e": 0.45, "lambda_e2r": 0.55}
        if split == "reaction_smi" and variant in {"L3", "L4", "L5", "L6"}
        else {"lambda_r2e": 0.5, "lambda_e2r": 0.5}
    )
    payload: dict[str, Any] = {
        "name": LOSS_VARIANTS[variant]["name"],
        "beta": 10.0,
        "learn_beta": False,
        "beta_min": 0.01,
        "beta_max": 100.0,
        "positive_pair_source": "all_known_in_batch",
        "biofp_aux_weight": 0.0,
        "biofp_family_weights": {},
        "biofp_confidence_cap": 1.0,
    }
    payload.update(directional)
    payload.update({key: value for key, value in LOSS_VARIANTS[variant].items() if key != "name"})
    return payload


def _rewrite_data_paths(
    config: dict[str, Any],
    *,
    run_root: Path,
    split: str,
    test_config: bool,
) -> None:
    data = config["data"]
    data["train_reactions_path"] = str(normalized_reaction_path(run_root, split, "train"))
    data["train_reaction_t5v2_embeds_path"] = str(
        feature_path(run_root, split, "train", "reactiont5v2")
    )
    data["train_reaction_unimol2_embeds_path"] = str(
        feature_path(run_root, split, "train", "unimol2")
    )
    data["train_reaction_chiro_embeds_path"] = str(
        feature_path(run_root, split, "train", "chiro")
    )
    evaluation_subset = "test" if test_config else "validation"
    data[f"{evaluation_subset}_reactions_path"] = str(
        normalized_reaction_path(run_root, split, evaluation_subset)
    )
    for modality in ("reactiont5v2", "unimol2", "chiro"):
        data[f"validation_reaction_{modality}_embeds_path"] = str(
            feature_path(run_root, split, evaluation_subset, modality)
        )
    data["train_reaction_chemistry_vectors_path"] = str(
        set_feature_path(run_root, split, "train")
    )
    data["validation_reaction_chemistry_vectors_path"] = str(
        set_feature_path(run_root, split, evaluation_subset)
    )
    if test_config:
        for modality in ("reactiont5v2", "unimol2", "chiro"):
            data[f"reaction_{modality}_embeds_path"] = str(
                feature_path(run_root, split, "test", modality)
            )
        data["reaction_chemistry_vectors_path"] = str(
            set_feature_path(run_root, split, "test")
        )


def configure_f3_run(
    base: dict[str, Any],
    *,
    run_root: Path,
    variant: str,
    split: str,
    seed: int,
    test_config: bool = False,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["seed"] = int(seed)
    _rewrite_data_paths(config, run_root=run_root, split=split, test_config=test_config)
    run_id = f"{variant}_{split}_seed{seed}"
    stage = "test_config" if test_config else "train"
    config["logging"]["log_dir"] = str(run_root / "logs" / run_id / stage)
    config["logging"]["checkpoint_dir"] = str(run_root / "checkpoints" / run_id)
    config["logging"]["checkpoint_monitor"] = "val/mean_bidirectional_reactzyme_mrr"
    config["logging"]["checkpoint_mode"] = "max"
    wandb = config["logging"].setdefault("wandb", {})
    wandb.update(
        {
            "enabled": not test_config,
            "project": WANDB_PROJECT,
            "run_name": f"reactzyme-f3-loss-{run_id}",
            "tags": [
                "reactzyme-paper-protocol",
                "f3-fixed-architecture",
                "canonical-isomeric-smiles",
                "all-known-positives",
                variant,
                split,
                f"seed-{seed}",
            ],
        }
    )
    config["training"]["loss"] = loss_config(variant, split)
    early_stopping = config["training"].setdefault("early_stopping", {})
    early_stopping.update(
        {
            "enabled": True,
            "monitor": "val/mean_bidirectional_reactzyme_mrr",
            "mode": "max",
        }
    )
    config["training"]["devices"] = 4
    config["ablation"] = {
        "run_id": run_id,
        "split": split,
        "variant": variant,
        "seed": int(seed),
        "base_architecture": "F3_set_chemistry",
        "smiles_mode": "canonical_isomeric",
        "test_config": bool(test_config),
    }
    return config


def assert_f3_architecture_unchanged(base: dict[str, Any], generated: dict[str, Any]) -> None:
    if generated.get("model") != base.get("model"):
        raise AssertionError("Loss campaign changed the F3 model architecture")
    immutable_training = ("learning_rate", "weight_decay", "max_epochs", "precision")
    for key in immutable_training:
        if generated["training"].get(key) != base["training"].get(key):
            raise AssertionError(f"Loss campaign changed training.{key}")


def generate_campaign(
    run_root: Path = DEFAULT_RUN_ROOT,
    *,
    base_root: Path = DEFAULT_BASE_ROOT,
    variants: tuple[str, ...] = tuple(LOSS_VARIANTS),
    seeds: tuple[int, ...] = (SCREEN_SEED, *REPLICATE_SEEDS),
) -> dict[str, Any]:
    run_root = run_root.resolve()
    rows: list[dict[str, Any]] = []
    for variant in variants:
        if variant not in LOSS_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        for split in SPLITS:
            train_base = _load_yaml(base_root / split / "train.yaml")
            test_base = _load_yaml(base_root / split / "test.yaml")
            for seed in seeds:
                train = configure_f3_run(
                    train_base,
                    run_root=run_root,
                    variant=variant,
                    split=split,
                    seed=seed,
                )
                test = configure_f3_run(
                    test_base,
                    run_root=run_root,
                    variant=variant,
                    split=split,
                    seed=seed,
                    test_config=True,
                )
                assert_f3_architecture_unchanged(train_base, train)
                assert_f3_architecture_unchanged(test_base, test)
                config_dir = run_root / "configs" / variant / split / f"seed{seed}"
                train_path = config_dir / "train.yaml"
                test_path = config_dir / "test.yaml"
                _write_yaml(train_path, train)
                _write_yaml(test_path, test)
                rows.append(
                    {
                        "variant": variant,
                        "split": split,
                        "seed": seed,
                        "train_config": str(train_path),
                        "test_config": str(test_path),
                        "checkpoint_dir": train["logging"]["checkpoint_dir"],
                        "train_log": str(run_root / "logs" / f"{variant}_{split}_seed{seed}" / "train.log"),
                        "test_json": str(run_root / "results" / variant / f"seed{seed}" / f"{split}.json"),
                    }
                )

    run_root.mkdir(parents=True, exist_ok=True)
    plan_path = run_root / "campaign.tsv"
    with plan_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "campaign": "reactzyme_f3_loss_ablation_v1",
        "wandb_project": WANDB_PROJECT,
        "base_config_root": str(base_root.resolve()),
        "run_root": str(run_root),
        "smiles_mode": "canonical_isomeric",
        "normalizer_required": True,
        "splits": list(SPLITS),
        "subsets": list(SUBSETS),
        "variant_order": list(VARIANT_ORDER),
        "screen_seed": SCREEN_SEED,
        "replicate_seeds": list(REPLICATE_SEEDS),
        "num_generated_runs": len(rows),
        "loss_variants": LOSS_VARIANTS,
        "checkpoint_monitor": "val/mean_bidirectional_reactzyme_mrr",
    }
    (run_root / "campaign.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    runtime = run_root / "runtime.env"
    if not runtime.exists():
        runtime.write_text(
            "\n".join(
                (
                    f"PYTHON_BIN={ROOT.parent / 'env/bin/python'}",
                    f"SETUP_PYTHON_BIN={ROOT.parent / '.capability-run-py/bin/python'}",
                    f"REACTION_T5_MODEL_PATH={ROOT.parent / 'hf_cache/hub/models--sagawa--ReactionT5v2-forward/snapshots/933114058cb2604dc1bf536dbebdfcefbe83d4fc'}",
                    f"WANDB_PROJECT={WANDB_PROJECT}",
                    "WANDB_ENTITY=omnai",
                    "WANDB_MODE=online",
                    "GPUS=0,1,2,3",
                    "BASE_MASTER_PORT=26700",
                    "",
                )
            ),
            encoding="utf-8",
        )
    return manifest


def _best_validation_row(metrics_path: Path) -> dict[str, float] | None:
    frame = pd.read_csv(metrics_path)
    primary = "val/mean_bidirectional_reactzyme_mrr"
    if primary not in frame:
        return None
    valid = frame[frame[primary].notna()]
    if valid.empty:
        return None
    row = valid.loc[valid[primary].astype(float).idxmax()]
    keys = (
        primary,
        "val/reaction_to_enzyme/reactzyme_mrr",
        "val/enzyme_to_reaction/reactzyme_mrr",
    )
    return {key: float(row[key]) for key in keys if key in row and not pd.isna(row[key])}


def collect_validation_results(run_root: Path, seed: int = SCREEN_SEED) -> dict[str, Any]:
    results: dict[str, Any] = {"seed": seed, "variants": {}}
    for variant in LOSS_VARIANTS:
        split_rows: dict[str, Any] = {}
        for split in SPLITS:
            log_root = run_root / "logs" / f"{variant}_{split}_seed{seed}" / "train"
            metrics_paths = sorted(log_root.glob("protein_pooling_training/version_*/metrics.csv"))
            candidates = [row for path in metrics_paths if (row := _best_validation_row(path))]
            if candidates:
                split_rows[split] = max(
                    candidates,
                    key=lambda row: row["val/mean_bidirectional_reactzyme_mrr"],
                )
        macro_values = [
            row["val/mean_bidirectional_reactzyme_mrr"] for row in split_rows.values()
        ]
        results["variants"][variant] = {
            "splits": split_rows,
            "complete": len(split_rows) == len(SPLITS),
            "macro_mrr": sum(macro_values) / len(macro_values) if macro_values else None,
        }
    return results


def select_top_variants(run_root: Path, count: int = 2) -> list[str]:
    report = collect_validation_results(run_root)
    control = report["variants"].get("L0", {})
    control_splits = control.get("splits", {})
    for variant, payload in report["variants"].items():
        guardrail_failures: list[str] = []
        if payload["complete"] and control.get("complete"):
            for split in SPLITS:
                for metric in (
                    "val/reaction_to_enzyme/reactzyme_mrr",
                    "val/enzyme_to_reaction/reactzyme_mrr",
                ):
                    value = payload["splits"][split].get(metric)
                    baseline = control_splits[split].get(metric)
                    if value is not None and baseline is not None and value < baseline - 0.01:
                        guardrail_failures.append(f"{split}:{metric}")
        payload["guardrail_failures_vs_L0"] = guardrail_failures
        payload["guardrail_pass"] = payload["complete"] and not guardrail_failures
    complete = [
        (variant, payload["macro_mrr"])
        for variant, payload in report["variants"].items()
        if payload["complete"] and payload["macro_mrr"] is not None
    ]
    if len(complete) < count:
        raise RuntimeError(f"Need {count} complete screen variants, found {len(complete)}")
    eligible = [
        item for item in complete if report["variants"][item[0]]["guardrail_pass"]
    ]
    selection_pool = eligible if len(eligible) >= count else complete
    selected = [
        variant
        for variant, _ in sorted(selection_pool, key=lambda item: item[1], reverse=True)[:count]
    ]
    report["selected_variants"] = selected
    reports_dir = run_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "screen_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports_dir / "selected_variants.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    return selected


def write_test_report(run_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_root / "results").glob("L*/seed*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        variant = path.parents[1].name
        seed = int(path.parent.name.removeprefix("seed"))
        split = path.stem
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "split": split,
                "E_to_R_reactzyme_mrr": payload.get(
                    "enzyme_to_reaction/reactzyme_mrr"
                ),
                "R_to_E_reactzyme_mrr": payload.get(
                    "reaction_to_enzyme/reactzyme_mrr"
                ),
                "balanced_reactzyme_mrr": payload.get("balanced_reactzyme_mrr"),
                "checkpoint": payload.get("checkpoint"),
            }
        )
    report_dir = run_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(report_dir / "test_results.tsv", sep="\t", index=False)
    summary_rows: list[dict[str, Any]] = []
    if not frame.empty:
        for variant, variant_frame in frame.groupby("variant"):
            row: dict[str, Any] = {"variant": variant}
            for split in SPLITS:
                split_frame = variant_frame[variant_frame["split"] == split]
                for direction in ("E_to_R", "R_to_E"):
                    values = split_frame[f"{direction}_reactzyme_mrr"].dropna().astype(float)
                    row[f"{split}_{direction}_mean"] = float(values.mean()) if len(values) else None
                    row[f"{split}_{direction}_std"] = (
                        float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else None
                    )
            six_cells = [
                value
                for key, value in row.items()
                if key.endswith("_mean") and value is not None
            ]
            row["six_cell_macro_mrr"] = sum(six_cells) / len(six_cells) if six_cells else None
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(report_dir / "test_summary.tsv", sep="\t", index=False)
    markdown = ["# F3 Loss Ablation Test Results", "", "Canonical isomeric SMILES; paper test candidates; corrected ReactZyme MRR.", ""]
    if summary.empty:
        markdown.append("No completed test results.")
    else:
        display_columns = ["variant", "six_cell_macro_mrr"] + [
            f"{split}_{direction}_mean"
            for split in SPLITS
            for direction in ("E_to_R", "R_to_E")
        ]
        markdown.append("| " + " | ".join(display_columns) + " |")
        markdown.append("| " + " | ".join("---" for _ in display_columns) + " |")
        for row in summary[display_columns].to_dict("records"):
            values = []
            for column in display_columns:
                value = row[column]
                values.append(
                    ""
                    if value is None or (isinstance(value, float) and math.isnan(value))
                    else f"{value:.4f}"
                    if isinstance(value, float)
                    else str(value)
                )
            markdown.append("| " + " | ".join(values) + " |")
    (report_dir / "README_test_results.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return {"num_test_results": len(rows), "num_variants": len(summary_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    generate.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    select = subparsers.add_parser("select")
    select.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    select.add_argument("--count", type=int, default=2)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    collect.add_argument("--seed", type=int, default=SCREEN_SEED)
    report = subparsers.add_parser("report")
    report.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    if args.command == "generate":
        payload = generate_campaign(args.run_root, base_root=args.base_root)
    elif args.command == "select":
        payload = {"selected_variants": select_top_variants(args.run_root, args.count)}
    elif args.command == "collect":
        payload = collect_validation_results(args.run_root, args.seed)
    else:
        payload = write_test_report(args.run_root)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
