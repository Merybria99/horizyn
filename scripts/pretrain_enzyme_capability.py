#!/usr/bin/env python3
"""Pretrain an enzyme-only capability encoder."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import shutil
import sys
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Sampler, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from horizyn.capability.enzyme_capability_dataset import (
    EnzymeCapabilityPretrainDataset,
    enzyme_capability_collate,
)
from horizyn.capability.enzyme_capability_model import EnzymeCapabilityLitModule


BIO_STAGE1_LABEL_COLUMNS = {
    "cofactor": (
        "cofactor_architecture_bins_train",
        "cofactor_chemistry_bins_train",
        "combined_core_cofactor_labels_train",
    ),
    "transition": ("substrate_product_transition_labels_train",),
    "substrate": ("substrate_class_labels_train",),
    "product": ("product_class_labels_train",),
    "reaction_center": ("reaction_center_labels_train",),
}


class StratifiedHardNegativeBatchSampler(Sampler[list[int]]):
    """Build batches with a related-pair stratum plus random negatives."""

    def __init__(
        self,
        pairs,
        batch_size: int,
        hard_negative_count: int,
        random_negative_count: int,
        seed: int,
    ) -> None:
        self.pairs = pairs.reset_index(drop=True)
        self.batch_size = max(1, int(batch_size))
        self.hard_negative_count = max(1, int(hard_negative_count))
        self.random_negative_count = max(1, int(random_negative_count))
        self.seed = int(seed)
        self.indices = np.arange(len(self.pairs))
        self.groups: dict[tuple, np.ndarray] = {}
        group_columns = [
            column
            for column in (
                "ec_overlap_level",
                "cofactor_overlap",
                "reaction_type_overlap",
                "substrate_class_overlap",
                "center_label_overlap",
            )
            if column in self.pairs.columns
        ]
        if not group_columns:
            self.groups[("all",)] = self.indices
        else:
            for key, group in self.pairs.groupby(group_columns, dropna=False):
                if not isinstance(key, tuple):
                    key = (key,)
                values = group.index.to_numpy(dtype=np.int64)
                if len(values) > 1:
                    self.groups[key] = values
            if not self.groups:
                self.groups[("all",)] = self.indices
        self.group_keys = list(self.groups)

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(len(self)):
            key = self.group_keys[int(rng.integers(0, len(self.group_keys)))]
            group_indices = self.groups[key]
            hard_count = min(
                len(group_indices),
                max(self.hard_negative_count + 1, self.batch_size // 2),
            )
            hard = rng.choice(group_indices, size=hard_count, replace=len(group_indices) < hard_count)
            remaining = max(0, self.batch_size - len(hard))
            random_part = rng.choice(
                self.indices,
                size=remaining,
                replace=len(self.indices) < max(1, remaining),
            )
            batch = np.concatenate([hard, random_part])[: self.batch_size]
            rng.shuffle(batch)
            yield batch.astype(int).tolist()


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _feature_paths(paths: dict, model_config: dict) -> dict[str, str | None]:
    features: dict[str, str | None] = {}
    if bool(model_config.get("use_prot5_mean", True)):
        features["prot5_mean"] = paths.get("prot5_mean")
    if bool(model_config.get("use_sleec_pool", True)):
        features["prot5_sleec"] = paths.get("prot5_sleec")
    if bool(model_config.get("use_lorentz", True)):
        features["lorentz_tangent"] = paths.get("lorentz_tangent")
    return features


def _residue_feature_path(paths: dict, model_config: dict) -> str | None:
    if not bool(model_config.get("use_prot5_residue_multiquery_pooling", False)):
        return None
    return paths.get("prot5_residue") or paths.get("protein_residue_embeds_path")


def _loss_value(loss_config: dict, new_key: str, old_key: str, default: float) -> float:
    return float(loss_config.get(new_key, loss_config.get(old_key, default)))


def _temperature(loss_config: dict) -> float:
    if "capability_temperature" in loss_config:
        return float(loss_config["capability_temperature"])
    if "capability_beta" in loss_config:
        beta = float(loss_config["capability_beta"])
        if beta <= 0:
            raise ValueError("loss.capability_beta must be positive")
        return 1.0 / beta
    return float(loss_config.get("temperature", 0.1))


def _as_label_list(value) -> list[str]:
    if value is None:
        return []
    try:
        import pandas as pd

        if bool(pd.isna(value)):
            return []
    except Exception:
        pass
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [str(item) for item in value if str(item)]
    return []


def _first_nonempty_labels(row: dict, columns: tuple[str, ...]) -> list[str]:
    for column in columns:
        labels = _as_label_list(row.get(column))
        if labels:
            return labels
    return []


def _composite_family_weights(loss_config: dict) -> dict[str, float] | None:
    explicit = loss_config.get("composite_family_weights")
    if isinstance(explicit, dict):
        return {str(name): float(value) for name, value in explicit.items()}
    weights = {}
    key_map = {
        "cofactor": "composite_cofactor_weight",
        "transition": "composite_transition_weight",
        "reaction_center": "composite_reaction_center_weight",
        "substrate": "composite_substrate_weight",
        "product": "composite_product_weight",
    }
    for family_name, key in key_map.items():
        if key in loss_config:
            weights[family_name] = float(loss_config[key])
    return weights or None


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    return float(overlap) / float(len(left | right))


def _composite_positive_count_stats(
    *,
    family_label_sets: dict[str, list[set[str]]],
    selected_families: tuple[str, ...],
    required_positive_families: tuple[str, ...],
    loss_config: dict,
    label_quality_config: dict,
    seed: int,
) -> dict:
    if not family_label_sets:
        return {
            "usable_anchor_count": 0,
            "usable_anchor_fraction": 0.0,
            "positive_count_mode": "none",
        }
    total = len(next(iter(family_label_sets.values())))
    family_weights = _composite_family_weights(loss_config) or {
        "cofactor": 0.35,
        "reaction_center": 0.25,
        "substrate": 0.20,
        "product": 0.20,
    }
    positive_family_names = tuple(
        str(value)
        for value in loss_config.get("composite_positive_family_names", selected_families)
    )
    known_family_names = tuple(
        str(value)
        for value in loss_config.get("composite_known_family_names", selected_families)
    )
    positive_threshold = float(loss_config.get("composite_positive_threshold", 0.20))
    min_positive_families = int(loss_config.get("composite_min_positive_families", 2))
    min_known_families = int(loss_config.get("composite_min_known_families", 2))
    max_candidate_checks = int(label_quality_config.get("max_candidate_checks_per_anchor", 4096))
    max_anchor_checks = int(label_quality_config.get("max_anchor_checks", 5000))
    if max_anchor_checks <= 0:
        return {
            "sampled_anchor_count": 0,
            "estimated_usable_anchor_count": None,
            "usable_anchor_count": None,
            "usable_anchor_fraction": None,
            "mean_positives_per_usable_anchor": None,
            "median_positives_per_usable_anchor": None,
            "p95_positives_per_usable_anchor": None,
            "capped_anchor_count": 0,
            "max_anchor_checks": int(max_anchor_checks),
            "max_candidate_checks_per_anchor": int(max_candidate_checks),
            "positive_count_mode": "skipped_by_config",
            "positive_probe_skipped": True,
        }

    label_index: dict[str, dict[str, set[int]]] = {
        family_name: {}
        for family_name in set(selected_families) | set(required_positive_families)
    }
    for family_name, values in family_label_sets.items():
        if family_name not in label_index:
            continue
        for row_idx, labels in enumerate(values):
            for label in labels:
                label_index[family_name].setdefault(label, set()).add(row_idx)

    rng = np.random.default_rng(seed)
    if max_anchor_checks > 0 and total > max_anchor_checks:
        anchor_indices = rng.choice(
            np.arange(total, dtype=np.int64),
            size=max_anchor_checks,
            replace=False,
        )
        positive_count_mode_prefix = "sampled"
    else:
        anchor_indices = np.arange(total, dtype=np.int64)
        positive_count_mode_prefix = "full"
    positive_counts = np.zeros(len(anchor_indices), dtype=np.int64)
    capped_anchor_count = 0
    required_or_positive = (
        required_positive_families
        if required_positive_families
        else positive_family_names
    )
    for out_idx, row_idx in enumerate(anchor_indices.tolist()):
        candidate_pool: set[int] = set()
        for family_name in required_or_positive:
            for label in family_label_sets.get(family_name, [set()] * total)[row_idx]:
                candidate_pool.update(label_index.get(family_name, {}).get(label, set()))
        candidate_pool.discard(row_idx)
        if not candidate_pool:
            continue
        candidates = np.fromiter(candidate_pool, dtype=np.int64)
        if max_candidate_checks > 0 and len(candidates) > max_candidate_checks:
            capped_anchor_count += 1
            candidates = rng.choice(
                candidates,
                size=max_candidate_checks,
                replace=False,
            )
        count = 0
        for candidate_idx in candidates.tolist():
            known_count = 0
            positive_count = 0
            required_any = False
            weighted_overlap = 0.0
            available_weight = 0.0
            for family_name, weight in family_weights.items():
                left = family_label_sets.get(family_name, [set()] * total)[row_idx]
                right = family_label_sets.get(family_name, [set()] * total)[candidate_idx]
                both_known = bool(left) and bool(right)
                if family_name in known_family_names and both_known:
                    known_count += 1
                jaccard = _jaccard(left, right)
                if family_name in positive_family_names and jaccard > 0:
                    positive_count += 1
                if family_name in required_positive_families and jaccard > 0:
                    required_any = True
                if float(weight) > 0 and both_known:
                    weighted_overlap += float(weight) * jaccard
                    available_weight += float(weight)
            if known_count < min_known_families:
                continue
            if positive_count < min_positive_families:
                continue
            if required_positive_families and not required_any:
                continue
            composite = weighted_overlap / available_weight if available_weight > 0 else 0.0
            if composite >= positive_threshold:
                count += 1
        positive_counts[out_idx] = count

    usable = positive_counts > 0
    usable_counts = positive_counts[usable]
    usable_fraction = float(usable.mean()) if len(anchor_indices) else 0.0
    return {
        "sampled_anchor_count": int(len(anchor_indices)),
        "estimated_usable_anchor_count": int(round(usable_fraction * total)),
        "usable_anchor_count": int(usable.sum()),
        "usable_anchor_fraction": usable_fraction,
        "mean_positives_per_usable_anchor": float(usable_counts.mean()) if len(usable_counts) else 0.0,
        "median_positives_per_usable_anchor": float(np.median(usable_counts)) if len(usable_counts) else 0.0,
        "p95_positives_per_usable_anchor": float(np.percentile(usable_counts, 95)) if len(usable_counts) else 0.0,
        "capped_anchor_count": int(capped_anchor_count),
        "max_anchor_checks": int(max_anchor_checks),
        "max_candidate_checks_per_anchor": int(max_candidate_checks),
        "positive_count_mode": (
            f"{positive_count_mode_prefix}_exact"
            if max_candidate_checks <= 0 or capped_anchor_count == 0
            else f"{positive_count_mode_prefix}_capped_candidate_scan"
        ),
    }


def _write_label_quality_report(
    *,
    labels_path: str | Path,
    output_dir: Path,
    selected_families: tuple[str, ...],
    required_positive_families: tuple[str, ...],
    config: dict,
    loss_config: dict,
    seed: int,
) -> dict:
    import pandas as pd

    label_quality_config = config.get("label_quality", {})
    labels = pd.read_parquet(labels_path)
    total = int(len(labels))
    records = labels.to_dict("records")
    family_reports: dict[str, dict] = {}
    family_label_sets: dict[str, list[set[str]]] = {}
    for family_name in selected_families + tuple(
        name for name in required_positive_families if name not in selected_families
    ):
        columns = BIO_STAGE1_LABEL_COLUMNS.get(family_name)
        if columns is None:
            continue
        values = [_first_nonempty_labels(row, columns) for row in records]
        family_label_sets[family_name] = [set(labels_for_row) for labels_for_row in values]
        counts = Counter(label for labels_for_row in values for label in labels_for_row)
        per_row_counts = np.asarray([len(labels_for_row) for labels_for_row in values], dtype=np.int64)
        nonempty = per_row_counts > 0
        family_reports[family_name] = {
            "columns": list(columns),
            "num_labels": int(len(counts)),
            "nonempty_count": int(nonempty.sum()),
            "coverage": float(nonempty.mean()) if total else 0.0,
            "mean_label_count": float(per_row_counts.mean()) if total else 0.0,
            "median_label_count": float(np.median(per_row_counts)) if total else 0.0,
            "p95_label_count": float(np.percentile(per_row_counts, 95)) if total else 0.0,
            "top_labels": dict(counts.most_common(50)),
        }

    min_known_families = int(label_quality_config.get("min_known_label_families", 2))
    positive_stats = _composite_positive_count_stats(
        family_label_sets=family_label_sets,
        selected_families=selected_families,
        required_positive_families=required_positive_families,
        loss_config=loss_config,
        label_quality_config=label_quality_config,
        seed=seed,
    )

    thresholds = {
        "cofactor": float(label_quality_config.get("min_core_cofactor_coverage", 0.50)),
        "transition": float(label_quality_config.get("min_transition_coverage", 0.75)),
        "substrate": float(label_quality_config.get("min_substrate_coverage", 0.80)),
        "product": float(label_quality_config.get("min_product_coverage", 0.80)),
    }
    failed_families = []
    for family_name, threshold in thresholds.items():
        if family_name not in family_reports:
            continue
        if family_reports[family_name]["coverage"] < threshold:
            failed_families.append(family_name)
    usable_threshold = float(label_quality_config.get("min_usable_anchor_fraction", 0.70))
    usable_fraction = positive_stats.get("usable_anchor_fraction")
    if (
        not bool(positive_stats.get("positive_probe_skipped", False))
        and float(usable_fraction or 0.0) < usable_threshold
    ):
        failed_families.append("usable_anchor_fraction")

    report = {
        "labels_path": str(labels_path),
        "num_enzymes": total,
        "selected_families": list(selected_families),
        "required_positive_families": list(required_positive_families),
        "min_known_label_families": min_known_families,
        "family_reports": family_reports,
        **positive_stats,
        "failed_gates": failed_families,
        "quality_gate_passed": not failed_families,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "label_quality_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if bool(label_quality_config.get("enabled", False)) and failed_families:
        policy = str(label_quality_config.get("failed_family_policy", "abort"))
        if policy == "abort":
            raise RuntimeError(
                "Label quality gates failed: "
                f"{failed_families}. See {output_dir / 'label_quality_report.json'}"
            )
    return report


def _build_logger(config: dict, config_path: Path, output_dir: Path):
    wandb_config = config.get("logging", {}).get("wandb", {})
    if not bool(wandb_config.get("enabled", False)):
        return None
    try:
        logger = pl.loggers.WandbLogger(
            project=wandb_config.get("project"),
            entity=wandb_config.get("entity"),
            name=wandb_config.get("run_name"),
            save_dir=str(output_dir),
            tags=wandb_config.get("tags", None),
            mode=wandb_config.get("mode", "online"),
            log_model=bool(wandb_config.get("log_model", False)),
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize W&B logger: {exc}") from exc
    logger.log_hyperparams(
        {
            "config_path": str(config_path),
            **config,
        }
    )
    return logger


def split_dataset_by_enzyme_id(
    dataset: EnzymeCapabilityPretrainDataset,
    validation_fraction: float,
    seed: int,
) -> tuple[Subset, Subset | None, dict]:
    enzyme_ids = dataset.pairs["enzyme_id"].astype(str).to_numpy()
    unique_enzymes = np.asarray(sorted(set(enzyme_ids)))
    if len(unique_enzymes) <= 1 or validation_fraction <= 0:
        stats = {
            "num_train_pairs": int(len(dataset)),
            "num_val_pairs": 0,
            "num_train_enzymes": int(len(unique_enzymes)),
            "num_val_enzymes": 0,
            "train_val_enzyme_overlap": 0,
        }
        return Subset(dataset, list(range(len(dataset)))), None, stats
    rng = np.random.default_rng(seed)
    shuffled = unique_enzymes.copy()
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * float(validation_fraction))))
    val_count = min(val_count, len(shuffled) - 1)
    val_enzyme_ids = set(str(value) for value in shuffled[:val_count])
    train_indices = [
        idx
        for idx, enzyme_id in enumerate(enzyme_ids)
        if str(enzyme_id) not in val_enzyme_ids
    ]
    val_indices = [
        idx
        for idx, enzyme_id in enumerate(enzyme_ids)
        if str(enzyme_id) in val_enzyme_ids
    ]
    train_enzyme_ids = set(str(enzyme_ids[idx]) for idx in train_indices)
    val_enzyme_ids_observed = set(str(enzyme_ids[idx]) for idx in val_indices)
    overlap = train_enzyme_ids & val_enzyme_ids_observed
    if overlap:
        raise AssertionError(
            f"Train/validation enzyme split leaked {len(overlap)} enzymes"
        )
    stats = {
        "num_train_pairs": int(len(train_indices)),
        "num_val_pairs": int(len(val_indices)),
        "num_train_enzymes": int(len(train_enzyme_ids)),
        "num_val_enzymes": int(len(val_enzyme_ids_observed)),
        "train_val_enzyme_overlap": 0,
    }
    return Subset(dataset, train_indices), Subset(dataset, val_indices), stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = _load_yaml(config_path)
    paths = config["paths"]
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    loss_config = config.get("loss", {})
    output_dir = Path(paths.get("output_dir", "outputs/enzyme_capability_pretrain"))
    output_dir.mkdir(parents=True, exist_ok=True)

    vocabs_path = Path(paths.get("enzyme_label_vocabs", Path(paths["enzyme_capability_labels"]).with_name("enzyme_label_vocabs.json")))
    vocabs = _load_json(vocabs_path)
    selected_biological_families = tuple(
        str(value)
        for value in loss_config.get(
            "selected_biological_families",
            ("cofactor", "reaction_center", "substrate", "product"),
        )
    )
    composite_required_positive_families = tuple(
        str(value)
        for value in loss_config.get("composite_required_positive_families", ())
    )
    print("Running enzyme Stage 1 label-quality preflight...", flush=True)
    _write_label_quality_report(
        labels_path=paths["enzyme_capability_labels"],
        output_dir=output_dir,
        selected_families=selected_biological_families,
        required_positive_families=composite_required_positive_families,
        config=config,
        loss_config=loss_config,
        seed=int(config.get("training", {}).get("seed", 13)),
    )
    print("Label-quality preflight complete.", flush=True)
    print("Loading enzyme capability pretraining dataset...", flush=True)
    dataset = EnzymeCapabilityPretrainDataset(
        pair_capability_training_path=paths["pair_capability_training"],
        reaction_demand_vectors_path=paths["reaction_demand_vectors"],
        enzyme_capability_labels_path=paths["enzyme_capability_labels"],
        enzyme_feature_paths=_feature_paths(paths, model_config),
        enzyme_residue_path=_residue_feature_path(paths, model_config),
        enzyme_label_vocabs=vocabs,
        reaction_features_path=paths.get("reaction_features"),
        reaction_demand_metadata_path=paths.get("reaction_demand_metadata"),
        require_directional_reaction=bool(
            config.get("data", {}).get("require_directional_reaction", False)
        ),
        residue_max_tokens=model_config.get("residue_pooling", {}).get(
            "max_tokens",
            config.get("data", {}).get("residue_max_tokens", None),
        ),
        residue_truncation=str(
            model_config.get("residue_pooling", {}).get(
                "truncation",
                config.get("data", {}).get("residue_truncation", "ends_center"),
            )
        ),
        residue_in_memory=bool(
            model_config.get("residue_pooling", {}).get(
                "in_memory",
                config.get("data", {}).get("residue_in_memory", False),
            )
        ),
    )
    print(f"Loaded {len(dataset)} enzyme capability training rows.", flush=True)
    seed = int(config.get("training", {}).get("seed", 13))
    val_fraction = float(config.get("training", {}).get("validation_fraction", 0.05))
    train_dataset, val_dataset, split_report = split_dataset_by_enzyme_id(
        dataset,
        validation_fraction=val_fraction,
        seed=seed,
    )
    with (output_dir / "enzyme_validation_split_report.json").open("w", encoding="utf-8") as handle:
        json.dump(split_report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Enzyme-level split: {split_report}", flush=True)

    sample = dataset[0]
    enzyme_input_dims = {
        name: int(sample[name].numel())
        for name in ("prot5_mean", "prot5_sleec", "lorentz_tangent")
        if name in sample
    }
    residue_input_dim = (
        int(sample["prot5_residue_embeddings"].shape[-1])
        if "prot5_residue_embeddings" in sample
        else None
    )
    residue_pooling_config = model_config.get("residue_pooling", {})
    label_output_dims = {
        "cofactor_targets": len(vocabs.get("cofactor_labels", [])),
        "core_cofactor_targets": len(vocabs.get("core_cofactor_labels", [])),
        "enzyme_derived_cofactor_targets": len(vocabs.get("enzyme_derived_cofactor_labels", [])),
        "enzyme_derived_core_cofactor_targets": len(
            vocabs.get("enzyme_derived_core_cofactor_labels", [])
        ),
        "combined_cofactor_targets": len(vocabs.get("combined_cofactor_labels", [])),
        "combined_core_cofactor_targets": len(vocabs.get("combined_core_cofactor_labels", [])),
        "cofactor_architecture_targets": len(vocabs.get("cofactor_architecture_bins", [])),
        "cofactor_chemistry_targets": len(vocabs.get("cofactor_chemistry_bins", [])),
        "metal_ion_targets": len(vocabs.get("metal_ion_labels", [])),
        "auxiliary_participant_targets": len(vocabs.get("auxiliary_participant_labels", [])),
        "reaction_center_targets": len(vocabs.get("reaction_center_labels", [])),
        "substrate_targets": len(vocabs.get("substrate_class_labels", [])),
        "product_targets": len(vocabs.get("product_class_labels", [])),
        "substrate_product_transition_targets": len(
            vocabs.get("substrate_product_transition_labels", [])
        ),
        "reaction_type_targets": len(vocabs.get("reaction_type_labels", [])),
        "ec_targets": len(vocabs.get("ec_labels", [])),
    }
    model = EnzymeCapabilityLitModule(
        enzyme_input_dims=enzyme_input_dims,
        reaction_demand_dim=int(sample["reaction_demand_vec"].numel()),
        label_output_dims=label_output_dims,
        capability_dim=int(model_config.get("capability_dim", 256)),
        hidden_dim=int(model_config.get("hidden_dim", 512)),
        dropout=float(model_config.get("dropout", 0.1)),
        lr=float(training_config.get("lr", 3e-4)),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
        temperature=_temperature(loss_config),
        capability_mode=str(model_config.get("capability_mode", "single")),
        family_dim=int(model_config.get("family_dim", 64)),
        residue_input_dim=residue_input_dim,
        use_residue_multiquery_pooling=bool(
            model_config.get("use_prot5_residue_multiquery_pooling", False)
        ),
        residue_pool_dropout=float(residue_pooling_config.get("dropout", 0.0)),
        residue_pool_scale=float(residue_pooling_config.get("initial_scale", 0.1)),
        biological_family_names=tuple(
            str(value)
            for value in model_config.get("biological_family_names", ())
        )
        or None,
        pretraining_objective=str(
            loss_config.get(
                "objective",
                config.get("pretraining_objective", "reaction_demand"),
            )
        ),
        r2e_capability_weight=_loss_value(
            loss_config,
            "cap_r2e_weight",
            "r2e_capability_weight",
            1.0,
        ),
        e2r_capability_weight=_loss_value(
            loss_config,
            "cap_e2r_weight",
            "e2r_capability_weight",
            0.5,
        ),
        cofactor_subspace_weight=float(loss_config.get("cofactor_subspace_weight", 0.0)),
        reaction_center_subspace_weight=float(
            loss_config.get("reaction_center_subspace_weight", 0.0)
        ),
        substrate_subspace_weight=float(loss_config.get("substrate_subspace_weight", 0.0)),
        product_subspace_weight=float(loss_config.get("product_subspace_weight", 0.0)),
        family_e2r_weight=float(loss_config.get("family_e2r_weight", 0.4)),
        cofactor_bce_weight=float(loss_config.get("cofactor_bce_weight", 0.0)),
        core_cofactor_bce_weight=float(loss_config.get("core_cofactor_bce_weight", 0.0)),
        enzyme_derived_cofactor_bce_weight=float(
            loss_config.get("enzyme_derived_cofactor_bce_weight", 0.0)
        ),
        enzyme_derived_core_cofactor_bce_weight=float(
            loss_config.get("enzyme_derived_core_cofactor_bce_weight", 0.0)
        ),
        combined_cofactor_bce_weight=float(loss_config.get("combined_cofactor_bce_weight", 0.0)),
        combined_core_cofactor_bce_weight=float(
            loss_config.get("combined_core_cofactor_bce_weight", 0.0)
        ),
        cofactor_architecture_bce_weight=float(
            loss_config.get(
                "cofactor_architecture_bce_weight",
                loss_config.get("attribute_weight", 0.10),
            )
        ),
        cofactor_chemistry_bce_weight=float(
            loss_config.get(
                "cofactor_chemistry_bce_weight",
                loss_config.get("attribute_weight", 0.10),
            )
        ),
        metal_ion_bce_weight=float(
            loss_config.get("metal_ion_bce_weight", loss_config.get("attribute_weight", 0.03))
        ),
        auxiliary_participant_bce_weight=float(
            loss_config.get(
                "auxiliary_participant_bce_weight",
                loss_config.get("attribute_weight", 0.02),
            )
        ),
        reaction_center_bce_weight=float(
            loss_config.get("reaction_center_bce_weight", loss_config.get("attribute_weight", 0.10))
        ),
        substrate_product_bce_weight=float(
            loss_config.get("substrate_product_bce_weight", loss_config.get("attribute_weight", 0.10))
        ),
        substrate_product_transition_bce_weight=float(
            loss_config.get(
                "substrate_product_transition_bce_weight",
                loss_config.get("substrate_product_bce_weight", loss_config.get("attribute_weight", 0.10)),
            )
        ),
        reaction_type_bce_weight=float(
            loss_config.get("reaction_type_bce_weight", loss_config.get("attribute_weight", 0.10))
        ),
        ec_aux_weight=float(loss_config.get("ec_weight", loss_config.get("ec_aux_weight", 0.03))),
        bio_cofactor_weight=float(loss_config.get("cofactor_weight", 1.0)),
        bio_transition_weight=float(loss_config.get("transition_weight", 1.0)),
        bio_reaction_center_weight=float(loss_config.get("reaction_center_weight", 1.0)),
        bio_substrate_weight=float(loss_config.get("substrate_weight", 0.7)),
        bio_product_weight=float(loss_config.get("product_weight", 0.7)),
        bio_global_weight=float(loss_config.get("global_weight", 0.5)),
        bio_attribute_weight=float(loss_config.get("attribute_bce_weight", 0.10)),
        bio_composite_weight=float(loss_config.get("composite_weight", 1.0)),
        composite_family_weights=_composite_family_weights(loss_config),
        composite_positive_family_names=loss_config.get("composite_positive_family_names"),
        composite_known_family_names=loss_config.get("composite_known_family_names"),
        composite_required_positive_families=composite_required_positive_families,
        bio_selected_families=selected_biological_families,
        same_center_hard_negative_weight=float(
            loss_config.get("same_center_hard_negative_weight", 0.3)
        ),
        composite_positive_threshold=float(
            loss_config.get("composite_positive_threshold", 0.20)
        ),
        composite_min_positive_families=int(
            loss_config.get("composite_min_positive_families", 2)
        ),
        composite_min_known_families=int(
            loss_config.get("composite_min_known_families", 2)
        ),
        same_center_hard_negative_margin=float(
            loss_config.get("same_center_hard_negative_margin", 0.10)
        ),
        hard_negative_temperature=float(
            loss_config.get(
                "hard_negative_temperature",
                loss_config.get("capability_temperature", _temperature(loss_config)),
            )
        ),
        factor_diversity_weight=float(loss_config.get("factor_diversity_weight", 0.02)),
        reaction_to_enzymes=dataset.reaction_to_enzymes,
        enzyme_to_reactions=dataset.enzyme_to_reactions,
        reaction_family_masks=dataset.reaction_family_masks,
        enzyme_family_masks=dataset.enzyme_family_masks,
        freeze_reaction_encoder_after_epochs=config.get("reaction_demand_encoder", {}).get(
            "freeze_after_epochs",
            None,
        ),
        deduplicate_entities_before_loss=bool(
            loss_config.get("deduplicate_entities_before_loss", True)
        ),
    )
    batch_size = int(training_config.get("batch_size", 512))
    num_workers = int(training_config.get("num_workers", 8))
    sampling_config = config.get("sampling", {})
    if sampling_config.get("use_stratified_hard_negatives", False):
        if hasattr(train_dataset, "dataset"):
            base_dataset = train_dataset.dataset
            train_indices = np.asarray(train_dataset.indices, dtype=np.int64)
            train_pairs = base_dataset.pairs.iloc[train_indices].reset_index(drop=True)
        else:
            train_pairs = train_dataset.pairs.reset_index(drop=True)
        batch_sampler = StratifiedHardNegativeBatchSampler(
            train_pairs,
            batch_size=batch_size,
            hard_negative_count=int(sampling_config.get("hard_negative_count", 8)),
            random_negative_count=int(sampling_config.get("random_negative_count", 16)),
            seed=seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=bool(training_config.get("pin_memory", True)),
            collate_fn=enzyme_capability_collate,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=bool(training_config.get("pin_memory", True)),
            collate_fn=enzyme_capability_collate,
        )
    val_loader = (
        None
        if val_dataset is None
        else DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(training_config.get("pin_memory", True)),
            collate_fn=enzyme_capability_collate,
        )
    )
    default_monitor = "capability_space/r2e_mrr" if val_loader is not None else "train/loss"
    checkpoint_monitor = training_config.get("checkpoint_monitor", default_monitor)
    checkpoint_mode = training_config.get(
        "checkpoint_mode",
        "max" if any(token in str(checkpoint_monitor) for token in ("mrr", "recall")) else "min",
    )
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename="best_enzyme_capability",
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
        save_top_k=1,
        save_last=True,
    )
    logger = _build_logger(config, config_path, output_dir)
    trainer = pl.Trainer(
        max_epochs=int(training_config.get("epochs", 30)),
        accelerator=training_config.get("accelerator", "auto"),
        devices=training_config.get("devices", 1),
        strategy=training_config.get("strategy", "auto"),
        precision=training_config.get("precision", "32-true"),
        callbacks=[checkpoint],
        logger=logger,
        gradient_clip_val=float(training_config.get("gradient_clip_val", 1.0)),
        log_every_n_steps=int(training_config.get("log_every_n_steps", 10)),
        deterministic=True,
    )
    resume_checkpoint = training_config.get("resume_from_checkpoint", None)
    if resume_checkpoint is not None and not Path(resume_checkpoint).exists():
        print(f"Resume checkpoint not found, starting fresh: {resume_checkpoint}")
        resume_checkpoint = None
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_checkpoint)
    if checkpoint.best_model_path:
        target = output_dir / "best_enzyme_capability.ckpt"
        source = Path(checkpoint.best_model_path)
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
    with (output_dir / "enzyme_capability_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


if __name__ == "__main__":
    main()
