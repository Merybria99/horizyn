#!/usr/bin/env python3
"""Materialize split-specific biological-residual configs from validated templates."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SPLITS = ("enzyme_smi", "time")
SOURCE_TEMPLATE = ROOT / "configs/source_collapse_biological_residual_reaction_smi_pretrain.yaml"
FINETUNE_TEMPLATE = ROOT / "configs/reactzyme_reaction_smi_biological_residual.yaml"
SHARED_SOURCE_TOKENS = Path(
    "runs/biological_residual_shared/cache/source_full/protein_functional_tokens.h5"
)
SHARED_REACTZYME_TOKENS = Path(
    "runs/biological_residual_reaction_smi/cache/reactzyme/protein_functional_tokens.h5"
)


SPLIT_DATA_KEYS = (
    "train_pairs_path",
    "train_reactions_path",
    "validation_pairs_path",
    "validation_reactions_path",
    "protein_residue_embeds_path",
    "reaction_representation",
    "train_reaction_embeds_path",
    "validation_reaction_embeds_path",
    "train_reaction_t5v2_embeds_path",
    "validation_reaction_t5v2_embeds_path",
    "train_reaction_model_embeds_path",
    "validation_reaction_model_embeds_path",
    "train_reaction_unimol2_embeds_path",
    "validation_reaction_unimol2_embeds_path",
    "train_reaction_chiro_embeds_path",
    "validation_reaction_chiro_embeds_path",
    "train_reaction_chirality_embeds_path",
    "validation_reaction_chirality_embeds_path",
    "train_reaction_chienn_embeds_path",
    "validation_reaction_chienn_embeds_path",
    "train_reaction_chemistry_vectors_path",
    "validation_reaction_chemistry_vectors_path",
    "train_reaction_directional_vectors_path",
    "validation_reaction_directional_vectors_path",
    "reaction_model_dim",
    "reaction_unimol_dim",
    "reaction_chiro_dim",
    "reaction_chirality_dim",
    "reaction_chienn_dim",
    "reaction_chemistry_dim",
    "reaction_directional_dim",
    "reaction_use_model",
    "reaction_use_chiro",
    "reaction_use_chirality",
    "reaction_use_chienn",
    "reaction_use_chemistry",
    "reaction_use_directional",
    "reaction_load_directional",
    "reaction_allow_missing_unimol2",
    "reaction_allow_missing_chiro",
    "reaction_allow_missing_chirality",
    "reaction_allow_missing_chienn",
    "reaction_allow_missing_chemistry",
    "reaction_allow_missing_directional",
    "normalize_molecule_sets_as_self_reactions",
    "reaction_direction_mode",
    "enzyme_ec_labels_path",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return payload


def _write_yaml_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, width=100),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def materialize_split_configs(
    split: str,
    *,
    output_dir: Path | None = None,
    source_template_path: Path = SOURCE_TEMPLATE,
    finetune_template_path: Path = FINETUNE_TEMPLATE,
    shared_source_tokens: Path = SHARED_SOURCE_TOKENS,
    shared_reactzyme_tokens: Path = SHARED_REACTZYME_TOKENS,
) -> dict[str, Path]:
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"split must be one of {SUPPORTED_SPLITS}, got {split!r}")

    run_root = Path(f"runs/biological_residual_{split}")
    output_dir = run_root / "configs" if output_dir is None else Path(output_dir)
    source_config_path = output_dir / "source_pretrain.yaml"
    finetune_config_path = output_dir / "reactzyme_finetune.yaml"
    f3_config_path = Path(
        f"runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry/{split}/train.yaml"
    )
    f3_checkpoint_path = Path(
        f"runs/reactzyme_reaction_features_v1/checkpoints/{split}/"
        "F3_set_chemistry/protein-pooling-epoch=29.ckpt"
    )

    source = copy.deepcopy(_read_yaml(source_template_path))
    finetune = copy.deepcopy(_read_yaml(finetune_template_path))
    f3 = _read_yaml(ROOT / f3_config_path)
    f3_data = f3.get("data", {})
    if not isinstance(f3_data, dict):
        raise ValueError(f"F3 config has no data mapping: {f3_config_path}")

    source["logging"]["log_dir"] = str(run_root / "logs/source_pretrain")
    source["logging"]["checkpoint_dir"] = str(run_root / "checkpoints/source_pretrain")
    source["logging"]["wandb"]["run_name"] = (
        f"biological-residual-source-pretrain-{split}-seed42"
    )
    source["logging"]["wandb"]["tags"] = [
        "source-collapse",
        "biological-residual",
        "hypergraph",
        f"{split}-firewall",
    ]
    source["data"]["train_pairs_path"] = str(run_root / "data/source_pretrain/train_pairs.csv")
    source["data"]["train_reactions_path"] = str(
        run_root / "data/source_pretrain/train_rxns.csv"
    )
    source["data"]["protein_functional_tokens_path"] = str(shared_source_tokens)
    source["data"]["cached_enzyme_base_embeds_path"] = str(
        run_root / "cache/source/enzyme_base.h5"
    )
    source["data"]["cached_train_reaction_base_embeds_path"] = str(
        run_root / "cache/source/train_reaction_base.h5"
    )
    source["training"]["init_from_checkpoint"] = str(f3_checkpoint_path)
    source["ablation"]["run_id"] = f"source_collapse_biological_residual_{split}_safe_seed42"
    source["ablation"]["split"] = f"source_collapse_{split}_strict"

    for key in SPLIT_DATA_KEYS:
        if key in f3_data:
            finetune["data"][key] = f3_data[key]
        else:
            finetune["data"].pop(key, None)
    # Keep raw HDF5 access lazy: frozen CIRCE embeddings and functional tokens
    # are already cached, while molecule-set features are small per reaction.
    finetune["data"]["reaction_embedding_in_memory"] = False
    finetune["data"]["protein_functional_tokens_path"] = str(shared_reactzyme_tokens)
    finetune["data"]["cached_enzyme_base_embeds_path"] = str(
        run_root / "cache/reactzyme/enzyme_base.h5"
    )
    finetune["data"]["cached_train_reaction_base_embeds_path"] = str(
        run_root / "cache/reactzyme/train_reaction_base.h5"
    )
    finetune["data"]["cached_validation_reaction_base_embeds_path"] = str(
        run_root / "cache/reactzyme/validation_reaction_base.h5"
    )
    finetune["data"]["source_replay"]["config_path"] = str(source_config_path)
    finetune["logging"]["log_dir"] = str(run_root / "logs/reactzyme_finetune")
    finetune["logging"]["checkpoint_dir"] = str(
        run_root / "checkpoints/reactzyme_finetune"
    )
    finetune["logging"]["wandb"]["run_name"] = (
        f"biological-residual-reactzyme-{split}-seed42"
    )
    finetune["logging"]["wandb"]["tags"] = [
        "reactzyme",
        split,
        "circe-f3",
        "biological-residual",
        "hypergraph",
    ]
    finetune["training"]["init_from_checkpoint"] = str(
        run_root / "checkpoints/source_pretrain/last.ckpt"
    )
    finetune["ablation"]["run_id"] = f"reactzyme_{split}_biological_residual_seed42"
    finetune["ablation"]["split"] = split

    _write_yaml_atomic(source_config_path, source)
    _write_yaml_atomic(finetune_config_path, finetune)
    return {
        "source_config": source_config_path,
        "finetune_config": finetune_config_path,
        "f3_config": f3_config_path,
        "f3_checkpoint": f3_checkpoint_path,
        "run_root": run_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=SUPPORTED_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--shared-source-tokens", type=Path, default=SHARED_SOURCE_TOKENS)
    parser.add_argument(
        "--shared-reactzyme-tokens",
        type=Path,
        default=SHARED_REACTZYME_TOKENS,
    )
    args = parser.parse_args()
    paths = materialize_split_configs(
        args.split,
        output_dir=args.output_dir,
        shared_source_tokens=args.shared_source_tokens,
        shared_reactzyme_tokens=args.shared_reactzyme_tokens,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
