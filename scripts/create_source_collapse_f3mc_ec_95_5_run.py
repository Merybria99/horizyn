#!/usr/bin/env python3
"""Create a leakage-safe 95/5 F3-MC-EC run on the collapsed source union.

The split is reaction-disjoint and targets exactly five percent of association
rows in the held-out partition.  F3's original 8-way mechanism and 10-way
cofactor targets are rebuilt from training associations only.  Curated
cofactor supervision is restricted to UniProt-derived columns so that
reaction-derived annotations from held-out associations cannot leak into the
training targets.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))
SNAPSHOT = (
    ROOT
    / "outputs/biofp_from_scratch_chains/"
    "biofp-fresh-chain-nohome-20260709_172734/data"
)
SOURCE_PAIRS = (
    SNAPSHOT
    / "standardized/retrieval_training_source_collapse/train_exact/"
    "train_pairs_valid_rxn_rhea_reconstructed.csv"
)
SOURCE_REACTIONS = (
    SNAPSHOT
    / "standardized/retrieval_training_source_collapse/train_exact/"
    "train_rxns_valid_rxn_rhea_reconstructed.csv"
)
SOURCE_PROTEIN_H5 = (
    SNAPSHOT
    / "standardized/retrieval_training_source_collapse/train_exact/"
    "fit_proteins_with_clipzyme_eval_prott5_residue.h5"
)
SOURCE_REACTION_T5_H5 = (
    SNAPSHOT
    / "standardized/retrieval_training_source_collapse/train_exact/"
    "rxns_reactiont5v2_forward_768_exact_rhea_plus_clipzyme_eval.h5"
)
SOURCE_UNIMOL2_H5 = (
    SNAPSHOT
    / "standardized/retrieval_training_source_collapse/train_exact/"
    "rxns_unimol2_84m_exact_rhea_plus_clipzyme_eval.h5"
)
SOURCE_CHIRO_H5 = (
    SNAPSHOT
    / "standardized/retrieval_training_source_collapse/train_exact/"
    "rxns_chiro_256_exact_rhea_plus_clipzyme_eval.h5"
)
CAPABILITY_ROOT = (
    SNAPSHOT
    / "processed/capability_features/train_exact_rhea_reconstructed"
)
SOURCE_REACTION_FEATURES = CAPABILITY_ROOT / "reaction_features.parquet"
SOURCE_ENZYME_COFACTORS = CAPABILITY_ROOT / "enzyme_cofactor_labels_enhanced.csv"
COFACTOR_DICTIONARY = CAPABILITY_ROOT / "cofactor_dictionary.csv"
SOURCE_CONFIG = (
    ROOT / "runs/reactzyme_f3mc_ec_aux_sweep_v1/configs/lambda_3p00/train.yaml"
)
RUN_ROOT = ROOT / "runs/source_collapse_f3mc_ec_lambda3_95_5_v1"
PROJECT = "horizyn-source-collapse-f3mc-ec-lambda3-95-5-v1"
SEED = 42
TEST_FRACTION = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
    return count


def choose_test_reactions(
    pair_rows: list[dict[str, str]],
    *,
    seed: int,
    test_fraction: float,
) -> tuple[set[str], int]:
    """Choose whole reaction groups while hitting the requested row count."""

    degree = Counter(row["reaction_id"] for row in pair_rows)
    target_rows = round(len(pair_rows) * test_fraction)
    reaction_ids = sorted(degree)
    random.Random(seed).shuffle(reaction_ids)

    chosen: set[str] = set()
    remaining = target_rows
    for reaction_id in reaction_ids:
        size = degree[reaction_id]
        if size <= remaining:
            chosen.add(reaction_id)
            remaining -= size
        if remaining == 0:
            break
    if remaining != 0:
        raise RuntimeError(
            f"Could not construct an exact {target_rows}-row reaction-disjoint holdout; "
            f"remaining={remaining}"
        )
    return chosen, target_rows


def split_rows(
    pair_rows: list[dict[str, str]], test_reactions: set[str]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    for row in pair_rows:
        destination = test_rows if row["reaction_id"] in test_reactions else train_rows
        new_row = dict(row)
        new_row["pr_id"] = str(len(destination))
        destination.append(new_row)
    return train_rows, test_rows


def write_uniprot_only_cofactors(
    output_path: Path,
    train_protein_ids: set[str],
) -> int:
    fields = [
        "enzyme_id",
        "uniprot_core_cofactor_labels_train",
        "enzyme_uniprotkb_cofactor_labels_train",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with SOURCE_ENZYME_COFACTORS.open(newline="", encoding="utf-8") as source, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source)
        missing = set(fields) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{SOURCE_ENZYME_COFACTORS} is missing columns {sorted(missing)}"
            )
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            if row["enzyme_id"] not in train_protein_ids:
                continue
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
    return count


def create_config(paths: dict[str, Path], counts: dict[str, int]) -> dict[str, Any]:
    source = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError(f"Expected YAML mapping in {SOURCE_CONFIG}")
    if float(source["training"]["loss"]["biofp_aux_weight"]) != 3.0:
        raise ValueError("Source F3-MC-EC config must use biological auxiliary weight 3")

    config = copy.deepcopy(source)
    config["logging"].update(
        {
            "log_dir": str(RUN_ROOT / "logs/train"),
            "checkpoint_dir": str(RUN_ROOT / "checkpoints/train"),
            "save_top_k": 0,
            "checkpoint_monitor": "train/loss_epoch",
            "checkpoint_mode": "min",
        }
    )
    config["logging"]["wandb"].update(
        {
            "enabled": True,
            "project": PROJECT,
            "entity": "omnai",
            "run_name": "F3MC-EC-lambda3-source-collapse-rxn-disjoint-95-5-seed42",
            "mode": "online",
            "tags": [
                "source-collapse",
                "exact-sequence-collapse",
                "rhea-reconstructed",
                "reaction-disjoint",
                "train-95-test-5",
                "F3MC-EC",
                "lambda-3",
                "four-gpu",
                "global-batch-4096",
                "fixed-30-epochs",
                "test-untouched-during-training",
                "seed-42",
            ],
        }
    )

    data = config["data"]
    data.update(
        {
            "train_pairs_path": str(paths["train_pairs"]),
            "train_reactions_path": str(paths["train_reactions"]),
            "validation_pairs_path": str(paths["test_pairs"]),
            "validation_reactions_path": str(paths["test_reactions"]),
            "test_pairs_path": str(paths["test_pairs"]),
            "test_reactions_path": str(paths["test_reactions"]),
            "protein_residue_embeds_path": str(SOURCE_PROTEIN_H5),
            "train_reaction_t5v2_embeds_path": str(SOURCE_REACTION_T5_H5),
            "validation_reaction_t5v2_embeds_path": str(SOURCE_REACTION_T5_H5),
            "train_reaction_unimol2_embeds_path": str(SOURCE_UNIMOL2_H5),
            "validation_reaction_unimol2_embeds_path": str(SOURCE_UNIMOL2_H5),
            "train_reaction_chiro_embeds_path": str(SOURCE_CHIRO_H5),
            "validation_reaction_chiro_embeds_path": str(SOURCE_CHIRO_H5),
            "train_reaction_chemistry_vectors_path": str(paths["train_chemistry"]),
            "validation_reaction_chemistry_vectors_path": str(paths["test_chemistry"]),
            "protein_biofp_targets_path": str(paths["biofp_targets"]),
            "protein_biofp_vocab_path": str(paths["biofp_vocab"]),
            "validation_retrieval_candidate_set": "custom",
            "validation_retrieval_candidate_ids_path": str(paths["candidate_ids"]),
            "train_batch_size": 1024,
            "retrieval_batch_size": 512,
            "validation_retrieval_batch_size": 512,
            "num_workers": 1,
            "pin_memory": False,
            "reaction_direction_mode": "forward_only",
            "normalize_molecule_sets_as_self_reactions": False,
        }
    )

    training = config["training"]
    training.update(
        {
            "max_epochs": 30,
            "devices": 4,
            "accelerator": "gpu",
            "strategy": "ddp_find_unused_parameters_true",
            "validation_enabled": False,
            "validation_retrieval_metrics": False,
            "check_val_every_n_epoch": 1,
            "num_sanity_val_steps": 0,
            "use_distributed_sampler": True,
        }
    )
    training["early_stopping"]["enabled"] = False
    training["loss"]["biofp_aux_weight"] = 3.0
    training["loss"]["biofp_family_weights"] = {
        "mechanism": 0.65,
        "cofactor": 0.35,
    }

    config["ablation"] = {
        "campaign": "source_collapse_f3mc_ec_lambda3_95_5_v1",
        "run_id": "F3MC_ec_lambda3_source_collapse_reaction_disjoint_seed42",
        "variant": "F3MC_ec_lambda_3p00",
        "label": "F3-MC-EC lambda=3 on source-collapse aggregate",
        "description": (
            "Original F3-MC-EC architecture trained on the exact-sequence-collapsed, "
            "Rhea-reconstructed aggregate with a reaction-disjoint 95/5 split."
        ),
        "seed": SEED,
        "split": "reaction_disjoint_95_5",
        "train_pairs": counts["train_pairs"],
        "test_pairs": counts["test_pairs"],
        "train_reactions": counts["train_reactions"],
        "test_reactions": counts["test_reactions"],
        "evaluation_subset": "heldout_test",
        "validation_protocol": "disabled_fixed_epoch_refit",
        "test_protocol": "reaction_disjoint_full_candidate_pool",
        "global_batch_size": 4096,
        "devices": 4,
        "ec_block": True,
        "ec_auxiliary_supervision": False,
        "biofp_aux_weight": 3.0,
        "biofp_family_weights": {"mechanism": 0.65, "cofactor": 0.35},
    }
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [
        SOURCE_PAIRS,
        SOURCE_REACTIONS,
        SOURCE_PROTEIN_H5,
        SOURCE_REACTION_T5_H5,
        SOURCE_UNIMOL2_H5,
        SOURCE_CHIRO_H5,
        SOURCE_REACTION_FEATURES,
        SOURCE_ENZYME_COFACTORS,
        COFACTOR_DICTIONARY,
        SOURCE_CONFIG,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))

    data_dir = RUN_ROOT / "data"
    paths = {
        "train_pairs": data_dir / "split/train_pairs.csv",
        "test_pairs": data_dir / "split/test_pairs.csv",
        "train_reactions": data_dir / "split/train_rxns.csv",
        "test_reactions": data_dir / "split/test_rxns.csv",
        "candidate_ids": data_dir / "split/candidate_ids.txt",
        "direct_members": data_dir / "biology/train_direct_members.csv",
        "safe_cofactors": data_dir / "biology/train_uniprot_cofactor_labels.csv",
        "biofp_targets": data_dir / "biology/enzyme_biofp_soft_targets.npz",
        "biofp_vocab": data_dir / "biology/enzyme_biofp_vocab.json",
        "train_chemistry": data_dir / "reaction_set/train_reaction_set_features.npz",
        "test_chemistry": data_dir / "reaction_set/test_reaction_set_features.npz",
        "config": RUN_ROOT / "configs/train.yaml",
    }
    if paths["config"].exists() and not args.force:
        raise FileExistsError(
            f"Run already prepared: {paths['config']}. Pass --force to rebuild."
        )

    pair_fields, pair_rows = read_csv(SOURCE_PAIRS)
    reaction_fields, reaction_rows = read_csv(SOURCE_REACTIONS)
    reaction_by_id = {row["reaction_id"]: row for row in reaction_rows}
    if len(reaction_by_id) != len(reaction_rows):
        raise ValueError(f"Duplicate reaction IDs in {SOURCE_REACTIONS}")

    test_reaction_ids, target_test_rows = choose_test_reactions(
        pair_rows,
        seed=SEED,
        test_fraction=TEST_FRACTION,
    )
    train_rows, test_rows = split_rows(pair_rows, test_reaction_ids)
    train_reaction_ids = {row["reaction_id"] for row in train_rows}
    if train_reaction_ids & test_reaction_ids:
        raise RuntimeError("Reaction-disjoint split invariant failed")
    if len(test_rows) != target_test_rows:
        raise RuntimeError(
            f"Holdout row count mismatch: expected={target_test_rows}, got={len(test_rows)}"
        )

    train_reactions = [reaction_by_id[rid] for rid in reaction_by_id if rid in train_reaction_ids]
    test_reactions = [reaction_by_id[rid] for rid in reaction_by_id if rid in test_reaction_ids]
    write_csv(paths["train_pairs"], pair_fields, train_rows)
    write_csv(paths["test_pairs"], pair_fields, test_rows)
    write_csv(paths["train_reactions"], reaction_fields, train_reactions)
    write_csv(paths["test_reactions"], reaction_fields, test_reactions)

    all_proteins = sorted({row["protein_id"] for row in pair_rows})
    paths["candidate_ids"].parent.mkdir(parents=True, exist_ok=True)
    paths["candidate_ids"].write_text("\n".join(all_proteins) + "\n", encoding="utf-8")

    direct_member_fields = [
        "source_protein_id",
        "source_reaction_id",
        "reaction_id",
        "match_confidence",
    ]
    write_csv(
        paths["direct_members"],
        direct_member_fields,
        (
            {
                "source_protein_id": row["protein_id"],
                "source_reaction_id": row["reaction_id"],
                "reaction_id": row["reaction_id"],
                "match_confidence": 1.0,
            }
            for row in train_rows
        ),
    )
    train_proteins = {row["protein_id"] for row in train_rows}
    safe_cofactor_rows = write_uniprot_only_cofactors(
        paths["safe_cofactors"], train_proteins
    )

    from horizyn.capability.biological_targets import (
        build_minimal_targets,
        write_minimal_targets,
    )
    from horizyn.capability.reaction_set_features import (
        build_reaction_set_feature_splits,
    )

    ids, arrays, biofp_metadata = build_minimal_targets(
        train_pairs_path=paths["train_pairs"],
        matched_members_path=paths["direct_members"],
        directional_features_path=SOURCE_REACTION_FEATURES,
        enzyme_cofactor_labels_path=paths["safe_cofactors"],
    )
    write_minimal_targets(
        paths["biofp_targets"], paths["biofp_vocab"], ids, arrays, biofp_metadata
    )

    reaction_set_report = build_reaction_set_feature_splits(
        train_reactions_path=paths["train_reactions"],
        validation_reactions_path=paths["test_reactions"],
        test_reactions_path=paths["test_reactions"],
        cofactor_dictionary_path=COFACTOR_DICTIONARY,
        out_dir=paths["train_chemistry"].parent,
        morgan_bits=512,
    )

    counts = {
        "source_pairs": len(pair_rows),
        "train_pairs": len(train_rows),
        "test_pairs": len(test_rows),
        "source_reactions": len(reaction_rows),
        "train_reactions": len(train_reactions),
        "test_reactions": len(test_reactions),
        "all_proteins": len(all_proteins),
        "train_proteins": len(train_proteins),
        "test_positive_proteins": len({row["protein_id"] for row in test_rows}),
        "safe_uniprot_cofactor_rows": safe_cofactor_rows,
    }
    config = create_config(paths, counts)
    paths["config"].parent.mkdir(parents=True, exist_ok=True)
    paths["config"].write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    source_combinations = Counter(row["source_datasets"] for row in pair_rows)
    train_source_combinations = Counter(row["source_datasets"] for row in train_rows)
    test_source_combinations = Counter(row["source_datasets"] for row in test_rows)
    manifest = {
        "schema_version": "source_collapse_f3mc_ec_95_5_v1",
        "seed": SEED,
        "split_policy": "reaction_disjoint_exact_pair_fraction",
        "requested_test_fraction": TEST_FRACTION,
        "actual_test_fraction": len(test_rows) / len(pair_rows),
        "test_annotations_used_during_training": False,
        "checkpoint_selection": "fixed_epoch_30_last_checkpoint",
        "source": {
            "pairs": str(SOURCE_PAIRS),
            "pairs_sha256": sha256(SOURCE_PAIRS),
            "reactions": str(SOURCE_REACTIONS),
            "reactions_sha256": sha256(SOURCE_REACTIONS),
            "base_config": str(SOURCE_CONFIG),
            "base_config_sha256": sha256(SOURCE_CONFIG),
        },
        "counts": counts,
        "source_dataset_combinations": dict(source_combinations),
        "train_dataset_combinations": dict(train_source_combinations),
        "test_dataset_combinations": dict(test_source_combinations),
        "biofp": biofp_metadata,
        "reaction_set": reaction_set_report,
        "paths": {name: str(path) for name, path in paths.items()},
    }
    (RUN_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run_root": str(RUN_ROOT), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
