#!/usr/bin/env python3
"""Materialize split-specific prototype train and test configs.

The reaction-SMILES configs are the canonical recipe.  This script changes
only split-dependent data, artifact, and parent-checkpoint paths so the
enzyme-SMILES and time experiments remain directly comparable while using
their own leakage-safe frozen parents.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_TEMPLATE = ROOT / "configs/source_collapse_promiscuity_k2_reaction_smi_pretrain.yaml"
FINETUNE_TEMPLATE = ROOT / "configs/reactzyme_reaction_smi_promiscuity_k2.yaml"
TEST_TEMPLATE = ROOT / "configs/reactzyme_reaction_smi_promiscuity_k2_test.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("reaction_smi", "enzyme_smi", "time"),
        required=True,
    )
    parser.add_argument("--prototype-count", type=int, default=2)
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def as_project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def materialize_source(
    template: dict[str, Any],
    *,
    split: str,
    parent: str,
    run_root: str,
    prototype_count: int,
) -> dict[str, Any]:
    config = copy.deepcopy(template)
    prototype_tag = f"k{prototype_count}"
    config["model"]["enzyme_prototypes"]["count"] = prototype_count
    config["logging"]["log_dir"] = f"{run_root}/logs/source_pretrain"
    config["logging"]["checkpoint_dir"] = f"{run_root}/checkpoints/source_pretrain"
    wandb = config["logging"]["wandb"]
    wandb["run_name"] = (
        f"source-collapse-promiscuity-{prototype_tag}-{split}-safe-seed42"
    )
    wandb["tags"] = [
        "source-collapse",
        "promiscuity",
        "enzyme-prototypes",
        prototype_tag,
        f"{split}-safe",
    ]

    data = config["data"]
    data["train_pairs_path"] = f"{run_root}/data/source_pretrain/train_pairs.csv"
    data["train_reactions_path"] = f"{run_root}/data/source_pretrain/train_rxns.csv"
    data["cached_enzyme_base_embeds_path"] = f"{run_root}/cache/source/enzyme_base.h5"
    data["cached_train_reaction_base_embeds_path"] = (
        f"{run_root}/cache/source/train_reaction_base.h5"
    )

    config["training"]["init_from_checkpoint"] = parent
    ablation = config["ablation"]
    ablation["run_id"] = (
        f"source_collapse_promiscuity_{prototype_tag}_{split}_safe_seed42"
    )
    ablation["split"] = f"source_collapse_{split}_safe"
    ablation["description"] = (
        f"Frozen split-specific {split} parent with {prototype_count} residual enzyme "
        "prototypes and grouped multi-reaction batches."
    )
    ablation["variant"] = f"enzyme_prototypes_{prototype_tag}"
    ablation["label"] = (
        f"Source-collapse K={prototype_count} prototype pretraining"
    )
    return config


def materialize_finetune(
    template: dict[str, Any],
    *,
    split: str,
    run_root: str,
    prototype_count: int,
) -> dict[str, Any]:
    config = copy.deepcopy(template)
    prototype_tag = f"k{prototype_count}"
    config["model"]["enzyme_prototypes"]["count"] = prototype_count
    config["logging"]["log_dir"] = f"{run_root}/logs/reactzyme_finetune"
    config["logging"]["checkpoint_dir"] = f"{run_root}/checkpoints/reactzyme_finetune"
    wandb = config["logging"]["wandb"]
    wandb["run_name"] = (
        f"reactzyme-{split}-promiscuity-{prototype_tag}-seed42"
    )
    wandb["tags"] = [
        "reactzyme",
        split,
        "promiscuity",
        "enzyme-prototypes",
        prototype_tag,
    ]

    paper_root = f"data/revised_protocols/reactzyme_paper/{split}"
    feature_root = f"data/revised_protocols/reactzyme_official/features/{split}/train"
    chemistry_root = f"runs/reactzyme_reaction_features_v1/data/{split}/reaction_set"
    data = config["data"]
    data.update(
        {
            "train_pairs_path": f"{paper_root}/train_pairs.csv",
            "train_reactions_path": f"{paper_root}/train_rxns.csv",
            "validation_pairs_path": f"{paper_root}/validation_pairs.csv",
            "validation_reactions_path": f"{paper_root}/validation_rxns.csv",
            "cached_enzyme_base_embeds_path": f"{run_root}/cache/reactzyme/enzyme_base.h5",
            "cached_train_reaction_base_embeds_path": f"{run_root}/cache/reactzyme/train_reaction_base.h5",
            "cached_validation_reaction_base_embeds_path": f"{run_root}/cache/reactzyme/validation_reaction_base.h5",
            "train_reaction_t5v2_embeds_path": f"{feature_root}/reactiont5v2.h5",
            "validation_reaction_t5v2_embeds_path": f"{feature_root}/reactiont5v2.h5",
            "train_reaction_unimol2_embeds_path": f"{feature_root}/unimol2.h5",
            "validation_reaction_unimol2_embeds_path": f"{feature_root}/unimol2.h5",
            "train_reaction_chiro_embeds_path": f"{feature_root}/chiro.h5",
            "validation_reaction_chiro_embeds_path": f"{feature_root}/chiro.h5",
            "train_reaction_chemistry_vectors_path": f"{chemistry_root}/train_reaction_set_features.npz",
            "validation_reaction_chemistry_vectors_path": f"{chemistry_root}/validation_reaction_set_features.npz",
            "typed_negative_pools_path": f"{run_root}/data/reactzyme_negative_pools.json",
            "validation_retrieval_candidate_ids_path": f"{paper_root}/validation_candidate_ids.txt",
        }
    )

    training = config["training"]
    training["init_from_checkpoint"] = f"{run_root}/checkpoints/source_pretrain/last.ckpt"
    training["validation_retrieval_candidate_ids_path"] = (
        f"{paper_root}/validation_candidate_ids.txt"
    )
    ablation = config["ablation"]
    ablation["run_id"] = (
        f"reactzyme_{split}_promiscuity_{prototype_tag}_seed42"
    )
    ablation["split"] = split
    ablation["description"] = (
        f"{prototype_count} residual enzyme prototypes pretrained on a split-specific, "
        "leakage-filtered source-collapse graph and adapted on ReactZyme."
    )
    ablation["variant"] = (
        f"source_pretrained_enzyme_prototypes_{prototype_tag}"
    )
    ablation["label"] = (
        f"Source-pretrained K={prototype_count} enzyme prototypes"
    )
    return config


def materialize_test(
    template: dict[str, Any],
    *,
    split: str,
    run_root: str,
    prototype_count: int,
) -> dict[str, Any]:
    config = copy.deepcopy(template)
    config["model"]["enzyme_prototypes"]["count"] = prototype_count
    paper_root = f"data/revised_protocols/reactzyme_paper/{split}"
    feature_root = f"data/revised_protocols/reactzyme_official/features/{split}/test"
    chemistry_root = f"runs/reactzyme_reaction_features_v1/data/{split}/reaction_set"
    data = config["data"]
    data.update(
        {
            "train_pairs_path": f"{paper_root}/train_pairs.csv",
            "train_reactions_path": f"{paper_root}/train_rxns.csv",
            "test_pairs_path": f"{paper_root}/test_pairs.csv",
            "test_reactions_path": f"{paper_root}/test_rxns.csv",
            "reaction_t5v2_embeds_path": f"{feature_root}/reactiont5v2.h5",
            "reaction_unimol2_embeds_path": f"{feature_root}/unimol2.h5",
            "reaction_chiro_embeds_path": f"{feature_root}/chiro.h5",
            "reaction_chemistry_vectors_path": f"{chemistry_root}/test_reaction_set_features.npz",
            "validation_retrieval_candidate_ids_path": f"{paper_root}/test_candidate_ids.txt",
        }
    )
    config["training"]["init_from_checkpoint"] = (
        f"{run_root}/checkpoints/reactzyme_finetune/last.ckpt"
    )
    return config


def main() -> None:
    args = parse_args()
    if args.prototype_count < 1:
        raise ValueError("--prototype-count must be positive")
    parent_path = args.parent_checkpoint.resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(f"Split-specific parent checkpoint not found: {parent_path}")

    run_root = as_project_path(args.run_root)
    parent = as_project_path(parent_path)
    output_dir = args.run_root.resolve() / "configs"
    outputs = {
        "source": output_dir / "source_pretrain.yaml",
        "finetune": output_dir / "finetune.yaml",
        "test": output_dir / "test.yaml",
    }
    write_yaml(
        outputs["source"],
        materialize_source(
            read_yaml(SOURCE_TEMPLATE),
            split=args.split,
            parent=parent,
            run_root=run_root,
            prototype_count=args.prototype_count,
        ),
    )
    write_yaml(
        outputs["finetune"],
        materialize_finetune(
            read_yaml(FINETUNE_TEMPLATE),
            split=args.split,
            run_root=run_root,
            prototype_count=args.prototype_count,
        ),
    )
    write_yaml(
        outputs["test"],
        materialize_test(
            read_yaml(TEST_TEMPLATE),
            split=args.split,
            run_root=run_root,
            prototype_count=args.prototype_count,
        ),
    )

    sys.path.insert(0, str(ROOT))
    from horizyn.config import load_config

    for output in outputs.values():
        load_config(output)
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
