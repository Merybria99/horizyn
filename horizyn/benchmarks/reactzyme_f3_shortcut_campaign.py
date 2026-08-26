"""Generate, select, freeze, and report the single-head F3 shortcut campaign."""

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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = ROOT / "runs/reactzyme_f3_shortcut_ablation_v1"
DEFAULT_BASE_ROOT = ROOT / "runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry"
DEFAULT_PROTOCOL_ROOT = ROOT / "data/revised_protocols/reactzyme_unseen_reaction_v1"
DEFAULT_FEATURE_ROOT = ROOT / "data/revised_protocols/reactzyme_official/features"
SPLITS = ("time", "enzyme_smi", "reaction_smi")
SCREEN_SEED = 42
REPLICATE_SEEDS = (17, 73)
ALL_SEEDS = (SCREEN_SEED, *REPLICATE_SEEDS)
WANDB_PROJECT = "horizyn-reactzyme-f3-shortcut-ablation-v1"
REACTION_R2E_GUARD = 0.01
PRIOR_WEIGHTS = {
    "reaction_model": 0.25,
    "unimol2": 0.375,
    "chiro": 0.1875,
    "reaction_chemistry": 0.1875,
}
ACCEPTANCE_THRESHOLDS = {
    "time": {"enzyme_to_reaction": 0.7913, "reaction_to_enzyme": 0.5544},
    "enzyme_smi": {"enzyme_to_reaction": 0.9550, "reaction_to_enzyme": 0.6700},
    "reaction_smi": {"enzyme_to_reaction": 0.5023, "reaction_to_enzyme": 0.3909},
}
MODALITIES = (
    ("t5v2", "reactiont5v2"),
    ("unimol2", "unimol2"),
    ("chiro", "chiro"),
)

VARIANTS: dict[str, dict[str, Any]] = {
    "S0": {
        "label": "F3 control",
        "description": "Exact F3 single-head attention architecture on the audited protocol.",
    },
    "S1": {
        "label": "normalized F3",
        "description": "S0 with scale-preserving modality L2 normalization.",
        "normalize": True,
    },
    "S2": {
        "label": "bounded F3",
        "description": "S1 with prior-bounded adaptive single-head attention.",
        "normalize": True,
        "bounded": True,
    },
    "S3": {
        "label": "bounded + chemistry dropout",
        "description": "S2 with chemistry-only dropout at probability 0.25.",
        "normalize": True,
        "bounded": True,
        "chemistry_dropout": 0.25,
    },
    "S4": {
        "label": "bounded + consistency",
        "description": "S2 with detached chemistry-free consistency at weight 0.03.",
        "normalize": True,
        "bounded": True,
        "consistency_weight": 0.03,
    },
    "S5": {
        "label": "bounded + both",
        "description": "S2 with chemistry dropout and detached consistency.",
        "normalize": True,
        "bounded": True,
        "chemistry_dropout": 0.25,
        "consistency_weight": 0.03,
    },
}


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


def chemistry_path(run_root: Path, split: str, subset: str) -> Path:
    return (
        run_root
        / "data"
        / split
        / "reaction_set"
        / f"{subset}_reaction_set_features.npz"
    )


def _fixed_feature_path(feature_root: Path, split: str, subset: str, modality: str) -> Path:
    feature_subset = "test" if subset == "test" else "train"
    return feature_root / split / feature_subset / f"{modality}.h5"


def _apply_variant(config: dict[str, Any], variant: str) -> None:
    spec = VARIANTS[variant]
    attention = config["model"]["reaction_multimodal_attention"]
    for key in (
        "modality_l2_normalize",
        "chemistry_dropout",
        "attention_prior_weights",
        "attention_adaptation_strength",
    ):
        attention.pop(key, None)
    attention["fusion"] = "attention"
    config["training"].pop("reaction_chemistry_consistency", None)

    if spec.get("normalize"):
        attention["modality_l2_normalize"] = True
    if spec.get("bounded"):
        attention["fusion"] = "prior_bounded_attention"
        attention["attention_prior_weights"] = copy.deepcopy(PRIOR_WEIGHTS)
        attention["attention_adaptation_strength"] = 0.4
    if "chemistry_dropout" in spec:
        attention["chemistry_dropout"] = float(spec["chemistry_dropout"])
    if "consistency_weight" in spec:
        config["training"]["reaction_chemistry_consistency"] = {
            "weight": float(spec["consistency_weight"]),
        }


def _rewrite_data(
    config: dict[str, Any],
    *,
    run_root: Path,
    protocol_root: Path,
    feature_root: Path,
    split: str,
    evaluation_subset: str,
) -> None:
    data = config["data"]
    split_root = protocol_root / split
    data["train_pairs_path"] = str(split_root / "train_pairs.csv")
    data["train_reactions_path"] = str(split_root / "train_rxns.csv")
    for config_name, artifact_name in MODALITIES:
        data[f"train_reaction_{config_name}_embeds_path"] = str(
            _fixed_feature_path(feature_root, split, "train", artifact_name)
        )
        data[f"validation_reaction_{config_name}_embeds_path"] = str(
            _fixed_feature_path(feature_root, split, evaluation_subset, artifact_name)
        )
    data["train_reaction_chemistry_vectors_path"] = str(
        chemistry_path(run_root, split, "train")
    )
    data["validation_reaction_chemistry_vectors_path"] = str(
        chemistry_path(run_root, split, evaluation_subset)
    )

    if evaluation_subset == "validation":
        data["validation_pairs_path"] = str(split_root / "validation_pairs.csv")
        data["validation_reactions_path"] = str(split_root / "validation_rxns.csv")
        data.pop("test_pairs_path", None)
        data.pop("test_reactions_path", None)
    else:
        data["test_pairs_path"] = str(split_root / "test_pairs.csv")
        data["test_reactions_path"] = str(split_root / "test_rxns.csv")

    candidate_path = split_root / f"{evaluation_subset}_candidate_ids.txt"
    data["validation_retrieval_candidate_set"] = "custom"
    data["validation_retrieval_candidate_ids_path"] = str(candidate_path)
    config["training"]["validation_retrieval_candidate_set"] = "custom"
    config["training"]["validation_retrieval_candidate_ids_path"] = str(candidate_path)

    if evaluation_subset in {"validation", "test"} and "test_pairs_path" in data:
        for config_name, artifact_name in MODALITIES:
            data[f"reaction_{config_name}_embeds_path"] = str(
                _fixed_feature_path(feature_root, split, evaluation_subset, artifact_name)
            )
        data["reaction_chemistry_vectors_path"] = str(
            chemistry_path(run_root, split, evaluation_subset)
        )


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
        raise ValueError(f"Unknown shortcut variant: {variant}")
    config = copy.deepcopy(base)
    config["seed"] = int(seed)
    _apply_variant(config, variant)
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
        }
    )
    wandb = config["logging"].setdefault("wandb", {})
    wandb.update(
        {
            "enabled": evaluation_subset == "validation",
            "project": WANDB_PROJECT,
            "run_name": f"reactzyme-f3-shortcut-{run_id}",
            "tags": [
                "reactzyme-f3-shortcut",
                "single-head",
                "reaction-smiles-disjoint-validation"
                if split == "reaction_smi"
                else "pair-random-validation",
                variant,
                split,
                f"seed-{seed}",
            ],
        }
    )
    early_stopping = config["training"].setdefault("early_stopping", {})
    early_stopping.update({"enabled": True, "monitor": monitor, "mode": "max"})
    config["ablation"] = {
        "campaign": "reactzyme_f3_shortcut_ablation_v1",
        "run_id": run_id,
        "variant": variant,
        "split": split,
        "seed": int(seed),
        "evaluation_subset": evaluation_subset,
        "description": VARIANTS[variant]["description"],
        "released_test_final_only": True,
    }
    return config


def _assert_control_architecture(base: dict[str, Any], control: dict[str, Any]) -> None:
    if control["model"] != base["model"]:
        raise AssertionError("S0 changed the F3 model architecture")
    if control["training"]["loss"] != base["training"]["loss"]:
        raise AssertionError("S0 changed the F3 retrieval loss")


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
        raise FileNotFoundError(
            f"Missing audited protocol at {protocol_root}; run "
            "scripts/build_reaction_disjoint_validation.py first"
        )

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
                        _fixed_feature_path(
                            feature_root,
                            split,
                            "validation",
                            artifact_name,
                        )
                    )
                validation["data"]["reaction_chemistry_vectors_path"] = str(
                    chemistry_path(run_root, split, "validation")
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
                if variant == "S0":
                    _assert_control_architecture(train_base, train)

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
        "schema_version": "reactzyme_f3_shortcut_ablation_v1",
        "run_root": str(run_root),
        "base_root": str(base_root.resolve()),
        "protocol_root": str(protocol_root),
        "feature_root": str(feature_root),
        "wandb_project": WANDB_PROJECT,
        "variants": VARIANTS,
        "screen_seed": SCREEN_SEED,
        "replicate_seeds": list(REPLICATE_SEEDS),
        "reaction_r2e_guard": REACTION_R2E_GUARD,
        "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
        "selection": (
            "Reaction-Sim checkpoints maximize validation E->R among the saved top three "
            "while retaining R->E within 0.01 of S0. Other splits maximize the mean of "
            "validation E->R and R->E. The top two seed-42 variants are repeated at "
            "seeds 17 and 73; only the frozen final winner reaches released test data."
        ),
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
                    "GPUS=0,1,2,3",
                    "BASE_MASTER_PORT=27600",
                    "",
                )
            ),
            encoding="utf-8",
        )
    return manifest


def _validation_results(run_root: Path, variant: str, split: str, seed: int) -> list[dict[str, Any]]:
    rows = []
    result_dir = run_root / "results" / "validation" / variant / split / f"seed{seed}"
    for path in sorted(result_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "path": str(path),
                "checkpoint": payload.get("checkpoint"),
                "e2r": payload.get("enzyme_to_reaction/reactzyme_mrr"),
                "r2e": payload.get("reaction_to_enzyme/reactzyme_mrr"),
            }
        )
    return [row for row in rows if row["checkpoint"] and row["e2r"] is not None and row["r2e"] is not None]


def _select_checkpoint(
    rows: list[dict[str, Any]],
    *,
    split: str,
    reaction_r2e_floor: float | None,
) -> dict[str, Any] | None:
    if split == "reaction_smi":
        eligible = rows
        if reaction_r2e_floor is not None:
            eligible = [row for row in rows if float(row["r2e"]) >= reaction_r2e_floor]
        if not eligible:
            return None
        return max(eligible, key=lambda row: float(row["e2r"]))
    if not rows:
        return None
    return max(rows, key=lambda row: (float(row["e2r"]) + float(row["r2e"])) / 2.0)


def _seed_selection(
    run_root: Path,
    variants: list[str],
    seed: int,
    *,
    baseline_r2e: float,
) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for variant in variants:
        split_rows: dict[str, Any] = {}
        for split in SPLITS:
            candidate = _select_checkpoint(
                _validation_results(run_root, variant, split, seed),
                split=split,
                reaction_r2e_floor=(
                    baseline_r2e - REACTION_R2E_GUARD
                    if split == "reaction_smi" and variant != "S0"
                    else None
                ),
            )
            if candidate is not None:
                split_rows[split] = candidate
        cells = [float(row[key]) for row in split_rows.values() for key in ("e2r", "r2e")]
        selected[variant] = {
            "splits": split_rows,
            "complete": len(split_rows) == len(SPLITS),
            "six_cell_macro_mrr": sum(cells) / len(cells) if cells else None,
        }
    return selected


def select_screen(run_root: Path, count: int = 2) -> dict[str, Any]:
    control_rows = _validation_results(run_root, "S0", "reaction_smi", SCREEN_SEED)
    control = _select_checkpoint(control_rows, split="reaction_smi", reaction_r2e_floor=None)
    if control is None:
        raise RuntimeError("S0 Reaction-Sim validation candidates are incomplete")
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
        raise RuntimeError(f"Need {count} complete screen variants, found {len(ranked)}")
    selected = [name for name, _ in ranked[:count]]
    report = {
        "stage": "screen",
        "seed": SCREEN_SEED,
        "reaction_s0_r2e": baseline_r2e,
        "selected_variants": selected,
        "variants": variants,
    }
    _write_json(run_root / "reports/screen_selection.json", report)
    (run_root / "reports/selected_variants.txt").write_text(
        "\n".join(selected) + "\n",
        encoding="utf-8",
    )
    return report


def select_final(run_root: Path) -> dict[str, Any]:
    screen = json.loads((run_root / "reports/screen_selection.json").read_text(encoding="utf-8"))
    selected_variants = list(screen["selected_variants"])
    baseline_r2e = float(screen["reaction_s0_r2e"])
    per_seed = {
        str(seed): _seed_selection(
            run_root,
            selected_variants,
            seed,
            baseline_r2e=baseline_r2e,
        )
        for seed in ALL_SEEDS
    }
    summaries: dict[str, Any] = {}
    for variant in selected_variants:
        payloads = [per_seed[str(seed)][variant] for seed in ALL_SEEDS]
        values = [payload["six_cell_macro_mrr"] for payload in payloads]
        complete = all(payload["complete"] for payload in payloads)
        summaries[variant] = {
            "complete": complete,
            "mean_six_cell_macro_mrr": (
                sum(float(value) for value in values) / len(values)
                if complete and all(value is not None for value in values)
                else None
            ),
        }
    ranked = sorted(
        (
            (name, payload["mean_six_cell_macro_mrr"])
            for name, payload in summaries.items()
            if payload["complete"] and payload["mean_six_cell_macro_mrr"] is not None
        ),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    if not ranked:
        raise RuntimeError("No replicated variant has complete validation results")
    winner = ranked[0][0]
    report = {
        "stage": "final",
        "winner": winner,
        "selected_variants": selected_variants,
        "summaries": summaries,
        "per_seed": per_seed,
    }
    _write_json(run_root / "reports/final_selection.json", report)
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_winner(run_root: Path) -> dict[str, Any]:
    selection_path = run_root / "reports/final_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("Final validation selection must be completed before freezing")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    winner = selection["winner"]
    rows = []
    for seed in ALL_SEEDS:
        for split in SPLITS:
            source = Path(selection["per_seed"][str(seed)][winner]["splits"][split]["checkpoint"])
            if not source.is_file():
                raise FileNotFoundError(f"Selected checkpoint is missing: {source}")
            destination = run_root / "frozen" / winner / f"seed{seed}" / split / "best.ckpt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            rows.append(
                {
                    "variant": winner,
                    "seed": seed,
                    "split": split,
                    "source": str(source),
                    "frozen": str(destination),
                    "sha256": _sha256(destination),
                }
            )
    manifest = {
        "schema_version": "reactzyme_f3_shortcut_frozen_winner_v1",
        "winner": winner,
        "validation_selection": str(selection_path),
        "checkpoints": rows,
    }
    _write_json(run_root / "frozen/manifest.json", manifest)
    return manifest


def write_test_report(run_root: Path) -> dict[str, Any]:
    frozen_path = run_root / "frozen/manifest.json"
    if not frozen_path.is_file():
        raise FileNotFoundError("A frozen winner is required before test reporting")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    winner = frozen["winner"]
    rows = []
    for path in sorted((run_root / "results/test" / winner).glob("seed*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "variant": winner,
                "seed": int(path.parent.name.removeprefix("seed")),
                "split": path.stem,
                "e2r": payload.get("enzyme_to_reaction/reactzyme_mrr"),
                "r2e": payload.get("reaction_to_enzyme/reactzyme_mrr"),
            }
        )
    summary: dict[str, Any] = {"winner": winner, "rows": rows, "splits": {}}
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        split_summary = {}
        for key, direction in (("e2r", "enzyme_to_reaction"), ("r2e", "reaction_to_enzyme")):
            values = [float(row[key]) for row in split_rows if row[key] is not None]
            mean = sum(values) / len(values) if values else None
            split_summary[key] = mean
            split_summary[f"{key}_threshold"] = ACCEPTANCE_THRESHOLDS[split][direction]
            split_summary[f"{key}_pass"] = (
                mean is not None and mean >= ACCEPTANCE_THRESHOLDS[split][direction]
            )
        summary["splits"][split] = split_summary
    summary["all_acceptance_cells_pass"] = all(
        split_summary[f"{key}_pass"]
        for split_summary in summary["splits"].values()
        for key in ("e2r", "r2e")
    )
    _write_json(run_root / "reports/test_results.json", summary)

    lines = [
        "# F3 Shortcut Ablation Test Results",
        "",
        f"Frozen validation winner: **{winner}**",
        "",
        "| Split | E->R mean | Gate | Pass | R->E mean | Gate | Pass |",
        "| --- | ---: | ---: | :---: | ---: | ---: | :---: |",
    ]
    for split in SPLITS:
        item = summary["splits"][split]
        fmt = lambda value: "" if value is None else f"{float(value):.4f}"
        lines.append(
            f"| {split} | {fmt(item['e2r'])} | {item['e2r_threshold']:.4f} | "
            f"{'yes' if item['e2r_pass'] else 'no'} | {fmt(item['r2e'])} | "
            f"{item['r2e_threshold']:.4f} | {'yes' if item['r2e_pass'] else 'no'} |"
        )
    (run_root / "reports/README_test_results.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    generate.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    generate.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    generate.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    screen = subparsers.add_parser("select-screen")
    screen.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    final = subparsers.add_parser("select-final")
    final.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    report = subparsers.add_parser("report")
    report.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    if args.command == "generate":
        payload = generate_campaign(
            args.run_root,
            base_root=args.base_root,
            protocol_root=args.protocol_root,
            feature_root=args.feature_root,
        )
    elif args.command == "select-screen":
        payload = select_screen(args.run_root)
    elif args.command == "select-final":
        payload = select_final(args.run_root)
    elif args.command == "freeze":
        payload = freeze_winner(args.run_root)
    else:
        payload = write_test_report(args.run_root)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
