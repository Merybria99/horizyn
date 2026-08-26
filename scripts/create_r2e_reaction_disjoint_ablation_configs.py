#!/usr/bin/env python3
"""Create reaction-disjoint R->E ablation configs and a sequential launcher."""

from __future__ import annotations

import argparse
import copy
import json
import os
import stat
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHAIN_ROOT = (
    ROOT
    / "outputs/biofp_from_scratch_chains/biofp-fresh-chain-nohome-20260709_172734"
)
DEFAULT_ABLATION_ROOT = (
    DEFAULT_CHAIN_ROOT / "r2e_reaction_disjoint_ablation_20260710_1800"
)
DEFAULT_BASE_CONFIG = DEFAULT_CHAIN_ROOT / "configs/joint_biofp_retrieval.yaml"
DEFAULT_PYTHON = Path("/datastor2/deep-proteins/EnzymeDiscovery/.capability-run-py/bin/python")
DEFAULT_SHORT_TMP_DIR = Path("/tmp/horizyn_r2e_disjoint_ablation_20260710_1800")


RUNS = [
    {
        "id": "B_reaction_disjoint_current_biofp",
        "label": "B",
        "seq_weight": 0.75,
        "hard_negative": False,
        "description": "Reaction-disjoint current BioFP baseline.",
        "master_port": 29710,
    },
    {
        "id": "C_reaction_disjoint_hardneg_seq075",
        "label": "C",
        "seq_weight": 0.75,
        "hard_negative": True,
        "description": "Hard-negative training with current BioFP split weight.",
        "master_port": 29711,
    },
    {
        "id": "D_reaction_disjoint_hardneg_seq090",
        "label": "D",
        "seq_weight": 0.90,
        "hard_negative": True,
        "description": "Hard-negative training with weaker BioFP subspace.",
        "master_port": 29712,
    },
    {
        "id": "E_reaction_disjoint_hardneg_seq095",
        "label": "E",
        "seq_weight": 0.95,
        "hard_negative": True,
        "description": "Hard-negative training with BioFP almost only auxiliary.",
        "master_port": 29713,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION_ROOT)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--short-tmp-dir",
        type=Path,
        default=DEFAULT_SHORT_TMP_DIR,
        help=(
            "Short temp directory used for multiprocessing sockets. "
            "Keep this path well below the AF_UNIX path length limit."
        ),
    )
    parser.add_argument(
        "--wandb-project",
        default="horizyn-r2e-reaction-disjoint-ablation-20260710",
    )
    parser.add_argument("--wandb-entity", default="omnai")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--validation-interval-steps", type=int, default=1000)
    parser.add_argument("--validation-retrieval-batch-size", type=int, default=128)
    parser.add_argument("--hard-negative-anchor-queries-per-batch", type=int, default=48)
    parser.add_argument("--hard-negative-positives-per-query", type=int, default=2)
    parser.add_argument("--hard-negative-negatives-per-query", type=int, default=8)
    parser.add_argument("--hard-negative-seed", type=int, default=42)
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


def split_paths(ablation_root: Path) -> dict[str, Path]:
    split_root = ablation_root / "data/reaction_disjoint_split"
    biofp_root = split_root / "biofp_train_only"
    hardneg_root = ablation_root / "data/hard_negatives"
    return {
        "train_pairs": split_root / "train_pairs_reaction_disjoint.csv",
        "train_reactions": split_root / "train_rxns_reaction_disjoint.csv",
        "val_pairs": split_root / "val_pairs_reaction_disjoint.csv",
        "val_reactions": split_root / "val_rxns_reaction_disjoint.csv",
        "val_candidates": split_root / "val_candidate_ids_reaction_disjoint.txt",
        "biofp_targets": biofp_root / "enzyme_biofp_soft_targets.npz",
        "biofp_vocab": biofp_root / "enzyme_biofp_vocab.json",
        "hardneg_json": hardneg_root / "r2e_hard_negative_pools.json",
    }


def require_paths(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required ablation inputs:\n" + "\n".join(missing))


def unique_tags(existing: list[Any], extra: list[str]) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for value in [*existing, *extra]:
        tag = str(value)
        if tag not in seen:
            tags.append(tag)
            seen.add(tag)
    return tags


def make_config(
    base_config: dict[str, Any],
    run: dict[str, Any],
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    run_id = str(run["id"])
    run_name = f"r2e-disjoint-{run['label']}-{run_id}"

    cfg["logging"]["log_dir"] = str(args.ablation_root / "logs" / run_id)
    cfg["logging"]["checkpoint_dir"] = str(args.ablation_root / "checkpoints" / run_id)
    cfg["logging"]["checkpoint_monitor"] = "val/reaction_to_enzyme/mrr"
    cfg["logging"]["checkpoint_mode"] = "max"
    cfg["logging"]["save_every_n_train_steps"] = 1000
    cfg["logging"]["log_every_n_steps"] = 1
    wandb_cfg = cfg["logging"].setdefault("wandb", {})
    wandb_cfg["enabled"] = args.wandb_mode != "disabled"
    wandb_cfg["project"] = args.wandb_project
    wandb_cfg["entity"] = args.wandb_entity
    wandb_cfg["run_name"] = run_name
    wandb_cfg["mode"] = args.wandb_mode
    wandb_cfg["log_model"] = False
    wandb_cfg["tags"] = unique_tags(
        list(wandb_cfg.get("tags", [])),
        [
            "r2e-rank-geometry",
            "reaction-disjoint-validation",
            "train-only-biofp-targets",
            run["label"],
            run_id,
            "hard-negative-sampler" if run["hard_negative"] else "baseline-sampler",
            f"biofp-seq-weight-{run['seq_weight']:.2f}",
        ],
    )

    data_cfg = cfg["data"]
    data_cfg["train_pairs_path"] = str(paths["train_pairs"])
    data_cfg["train_reactions_path"] = str(paths["train_reactions"])
    data_cfg["test_pairs_path"] = str(paths["val_pairs"])
    data_cfg["test_reactions_path"] = str(paths["val_reactions"])
    data_cfg["validation_retrieval_candidate_ids_path"] = str(paths["val_candidates"])
    data_cfg["protein_biofp_targets_path"] = str(paths["biofp_targets"])
    data_cfg["protein_biofp_vocab_path"] = str(paths["biofp_vocab"])
    data_cfg["train_batch_size"] = 512
    data_cfg["retrieval_batch_size"] = args.validation_retrieval_batch_size

    for key in (
        "hard_negative_pools_path",
        "hard_negative_anchor_queries_per_batch",
        "hard_negative_positives_per_query",
        "hard_negative_negatives_per_query",
        "hard_negative_seed",
    ):
        data_cfg.pop(key, None)
    if run["hard_negative"]:
        data_cfg["hard_negative_pools_path"] = str(paths["hardneg_json"])
        data_cfg["hard_negative_anchor_queries_per_batch"] = (
            args.hard_negative_anchor_queries_per_batch
        )
        data_cfg["hard_negative_positives_per_query"] = args.hard_negative_positives_per_query
        data_cfg["hard_negative_negatives_per_query"] = args.hard_negative_negatives_per_query
        data_cfg["hard_negative_seed"] = args.hard_negative_seed

    model_cfg = cfg["model"]
    model_cfg.setdefault("biofp", {})["seq_weight"] = float(run["seq_weight"])

    training_cfg = cfg["training"]
    training_cfg["max_epochs"] = args.max_epochs
    training_cfg["validation_retrieval_metrics"] = True
    training_cfg["validation_retrieval_candidate_set"] = "custom"
    training_cfg["validation_retrieval_candidate_ids_path"] = str(paths["val_candidates"])
    training_cfg["validation_retrieval_batch_size"] = args.validation_retrieval_batch_size
    training_cfg["validation_interval_steps"] = args.validation_interval_steps
    training_cfg["devices"] = 4
    training_cfg["accelerator"] = "gpu"
    training_cfg["strategy"] = "ddp_find_unused_parameters_true"
    training_cfg["use_distributed_sampler"] = not bool(run["hard_negative"])
    loss_cfg = training_cfg.setdefault("loss", {})
    loss_cfg["name"] = "FullBatchMLNCELoss"
    loss_cfg["positive_pair_source"] = "all_known_in_batch"
    loss_cfg["biofp_aux_weight"] = 0.03

    cfg["ablation"] = {
        "id": run_id,
        "label": run["label"],
        "description": run["description"],
        "reaction_disjoint_split": str(paths["val_reactions"]),
        "validation_candidate_ids": str(paths["val_candidates"]),
        "hard_negative_pools": str(paths["hardneg_json"]) if run["hard_negative"] else None,
        "biofp_seq_weight": run["seq_weight"],
    }
    return cfg


def launcher_text(
    args: argparse.Namespace,
    run_configs: list[dict[str, str]],
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'ROOT="{ROOT}"',
        f'AB_ROOT="{args.ablation_root}"',
        f'PYTHON_BIN="{args.python_bin}"',
        f'SHORT_TMP_DIR="{args.short_tmp_dir}"',
        f'WANDB_PROJECT="{args.wandb_project}"',
        f'WANDB_ENTITY="{args.wandb_entity}"',
        f'WANDB_MODE="${{WANDB_MODE:-{args.wandb_mode}}}"',
        f'export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-{args.devices}}}"',
        'export PYTHONUNBUFFERED="1"',
        'export PYTHONPATH="$ROOT"',
        'export HOME="$AB_ROOT/nohome"',
        'export TMPDIR="$SHORT_TMP_DIR"',
        'export TEMP="$TMPDIR"',
        'export TMP="$TMPDIR"',
        'export HF_HOME="$AB_ROOT/cache/huggingface"',
        'export HF_DATASETS_CACHE="$HF_HOME/datasets"',
        'export TRANSFORMERS_CACHE="$HF_HOME/transformers"',
        'export TORCH_HOME="$AB_ROOT/cache/torch"',
        'export XDG_CACHE_HOME="$AB_ROOT/xdg_cache"',
        'export XDG_CONFIG_HOME="$AB_ROOT/xdg_config"',
        'export XDG_STATE_HOME="$AB_ROOT/xdg_state"',
        'export MPLCONFIGDIR="$AB_ROOT/matplotlib"',
        'export WANDB_ROOT="$AB_ROOT/wandb"',
        'export WANDB_DIR="$AB_ROOT/wandb"',
        'export WANDB_DATA_DIR="$AB_ROOT/wandb/data"',
        'export WANDB_CACHE_DIR="$AB_ROOT/wandb/cache"',
        'export WANDB_CONFIG_DIR="$AB_ROOT/wandb/config"',
        'export TOKENIZERS_PARALLELISM="false"',
        'mkdir -p "$HOME" "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" '
        '"$XDG_CONFIG_HOME" "$XDG_STATE_HOME" "$MPLCONFIGDIR" "$WANDB_DIR" '
        '"$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$AB_ROOT/logs/chain"',
        'STATUS_LOG="$AB_ROOT/logs/chain_status.jsonl"',
        'status() {',
        '  local run_id="$1"',
        '  local event="$2"',
        '  local rc="${3:-}"',
        '  printf \'{"time":"%s","run_id":"%s","event":"%s","returncode":"%s"}\\n\' '
        '"$(date -Iseconds)" "$run_id" "$event" "$rc" >> "$STATUS_LOG"',
        '}',
        'run_one() {',
        '  local run_id="$1"',
        '  local cfg="$2"',
        '  local run_name="$3"',
        '  local master_port="$4"',
        '  local log_path="$AB_ROOT/logs/chain/${run_id}.stdout.log"',
        '  export MASTER_PORT="$master_port"',
        '  status "$run_id" start ""',
        '  echo "[$(date -Iseconds)] START $run_id cfg=$cfg port=$MASTER_PORT"',
        '  set +e',
        '  "$PYTHON_BIN" "$ROOT/scripts/train_protein_pooling.py" \\',
        '    --config "$cfg" \\',
        '    --wandb \\',
        '    --wandb-project "$WANDB_PROJECT" \\',
        '    --wandb-entity "$WANDB_ENTITY" \\',
        '    --wandb-run-name "$run_name" \\',
        '    --wandb-mode "$WANDB_MODE" \\',
        '    --wandb-tags r2e-rank-geometry reaction-disjoint-validation "$run_id" \\',
        '    > "$log_path" 2>&1',
        '  local rc=$?',
        '  set -e',
        '  status "$run_id" end "$rc"',
        '  echo "[$(date -Iseconds)] END $run_id rc=$rc log=$log_path"',
        '  if [ "$rc" -ne 0 ]; then',
        '    exit "$rc"',
        '  fi',
        '}',
    ]
    for item in run_configs:
        lines.append(
            "run_one "
            f'"{item["id"]}" '
            f'"{item["config"]}" '
            f'"{item["run_name"]}" '
            f'"{item["master_port"]}"'
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.ablation_root = args.ablation_root.resolve()
    args.base_config = args.base_config.resolve()
    args.short_tmp_dir = args.short_tmp_dir.expanduser()
    if not args.short_tmp_dir.is_absolute():
        args.short_tmp_dir = (Path.cwd() / args.short_tmp_dir).resolve()
    args.python_bin = args.python_bin.expanduser()
    if not args.python_bin.is_absolute():
        args.python_bin = (Path.cwd() / args.python_bin).resolve()

    paths = split_paths(args.ablation_root)
    require_paths(paths)
    if not args.base_config.exists():
        raise FileNotFoundError(args.base_config)
    if not args.python_bin.exists():
        raise FileNotFoundError(args.python_bin)

    base_config = load_yaml(args.base_config)
    config_dir = args.ablation_root / "configs"
    run_configs: list[dict[str, str]] = []
    for run in RUNS:
        cfg = make_config(base_config, run, args, paths)
        config_path = config_dir / f"{run['id']}.yaml"
        write_yaml(config_path, cfg)
        run_configs.append(
            {
                "id": run["id"],
                "label": run["label"],
                "run_name": cfg["logging"]["wandb"]["run_name"],
                "config": str(config_path),
                "master_port": str(run["master_port"]),
                "hard_negative": str(run["hard_negative"]),
                "seq_weight": str(run["seq_weight"]),
            }
        )

    launcher_path = args.ablation_root / "run_b_c_d_e_chain.sh"
    launcher_path.write_text(launcher_text(args, run_configs), encoding="utf-8")
    current_mode = launcher_path.stat().st_mode
    launcher_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP)

    manifest = {
        "ablation_root": str(args.ablation_root),
        "base_config": str(args.base_config),
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_mode": args.wandb_mode,
        "short_tmp_dir": str(args.short_tmp_dir),
        "max_epochs": args.max_epochs,
        "validation_interval_steps": args.validation_interval_steps,
        "inputs": {key: str(path) for key, path in paths.items()},
        "runs": run_configs,
        "launcher": str(launcher_path),
    }
    manifest_path = args.ablation_root / "ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
