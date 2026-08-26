"""Unified enzyme-reaction retrieval benchmark utilities."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from horizyn.config import load_config
from horizyn.datasets.base import BaseDataset
from horizyn.datasets.csv import CSVDataset
from horizyn.datasets.hdf5 import EmbedDataset
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from horizyn.reaction_features import build_reaction_feature_dataset
from horizyn.utils import residue_collate_fn, unimol2_reaction_collate_fn
from horizyn.capability.enzyme_capability_dataset import CapabilityVectorDataset, TextVectorDataset


DEFAULT_REACTZYME_TOP_K = [1, 2, 3, 4, 5, 10, 20, 50]
DEFAULT_SOTA_TOP_K = [1, 10, 100, 1000]
DEFAULT_BEDROC_ALPHAS = [85.0, 20.0]
DEFAULT_EF_FRACTIONS = [0.05, 0.10]
COSINE_EPS = 1e-12
TARGET_CACHE_SCHEMA_VERSION = 1
TARGET_CACHE_LOCK_POLL_SECONDS = 10.0
TARGET_CACHE_LOCK_STALE_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class BenchmarkTask:
    """Resolved benchmark task definition."""

    name: str
    task_type: str
    dataset: str
    task_label: str
    split: str
    pairs: Path
    reactions: Path
    train_pairs: Path | None = None
    candidate_ids: Path | None = None
    candidate_embedding_h5: Path | None = None
    candidate_residue_h5: dict[str, Path] = field(default_factory=dict)
    candidate_score_residue_h5: dict[str, Path] = field(default_factory=dict)
    reaction_embeds_h5: Path | None = None
    reaction_model_embeds_h5: Path | None = None
    reaction_unimol2_embeds_h5: Path | None = None
    reaction_chiro_embeds_h5: Path | None = None
    reaction_chirality_embeds_h5: Path | None = None
    reaction_chienn_embeds_h5: Path | None = None
    reaction_id_col: str = "reaction_id"
    reaction_smiles_col: str = "reaction_smiles"
    protein_id_col: str = "protein_id"
    directions: tuple[str, ...] = ("reaction_to_enzyme", "enzyme_to_reaction")
    top_k: tuple[int, ...] = tuple(DEFAULT_REACTZYME_TOP_K)
    bedroc_alphas: tuple[float, ...] = tuple(DEFAULT_BEDROC_ALPHAS)
    ef_fractions: tuple[float, ...] = tuple(DEFAULT_EF_FRACTIONS)

    @property
    def is_screening(self) -> bool:
        return self.task_type == "screening"

    @property
    def is_retrieval(self) -> bool:
        return self.task_type == "retrieval"


def _resolve_path(value: str | None, project_root: Path) -> Path | None:
    if value in {None, ""}:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def _resolve_path_map(value: Any, project_root: Path) -> dict[str, Path]:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        return {"default": _resolve_path(value, project_root)}
    if not isinstance(value, dict):
        raise TypeError("path map values must be a string or mapping")
    return {
        str(key): resolved
        for key, item in value.items()
        if (resolved := _resolve_path(str(item), project_root)) is not None
    }


def load_benchmark_suite(
    path: str | Path, project_root: str | Path | None = None
) -> list[BenchmarkTask]:
    """Load and resolve a benchmark suite YAML file."""

    suite_path = Path(path)
    if project_root is None:
        project_root_path = Path.cwd()
    else:
        project_root_path = Path(project_root)
    with suite_path.open("r", encoding="utf-8") as handle:
        raw_suite = yaml.safe_load(handle) or {}
    tasks = raw_suite.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"Benchmark suite has no tasks: {suite_path}")

    resolved_tasks: list[BenchmarkTask] = []
    for raw_task in tasks:
        if not isinstance(raw_task, dict):
            raise TypeError("Each benchmark task must be a mapping")
        name = str(raw_task["name"])
        task_type = str(raw_task.get("type", "retrieval"))
        if task_type not in {"retrieval", "screening"}:
            raise ValueError(f"Unsupported task type for {name}: {task_type}")
        pairs = _resolve_path(raw_task.get("pairs"), project_root_path)
        reactions = _resolve_path(raw_task.get("reactions"), project_root_path)
        if pairs is None or reactions is None:
            raise ValueError(f"Task {name} requires pairs and reactions paths")

        default_top_k = DEFAULT_SOTA_TOP_K if name == "horizyn_sota" else DEFAULT_REACTZYME_TOP_K
        top_k = tuple(int(value) for value in raw_task.get("top_k", default_top_k))
        bedroc_alphas = tuple(
            float(value) for value in raw_task.get("bedroc_alphas", DEFAULT_BEDROC_ALPHAS)
        )
        ef_fractions = tuple(
            float(value) for value in raw_task.get("ef_fractions", DEFAULT_EF_FRACTIONS)
        )
        directions = tuple(str(value) for value in raw_task.get("directions", ["both"]))
        if "both" in directions:
            directions = ("reaction_to_enzyme", "enzyme_to_reaction")

        resolved_tasks.append(
            BenchmarkTask(
                name=name,
                task_type=task_type,
                dataset=str(raw_task.get("dataset", name.split("_")[0])),
                task_label=str(raw_task.get("task_label", name)),
                split=str(raw_task.get("split", name)),
                pairs=pairs,
                reactions=reactions,
                train_pairs=_resolve_path(raw_task.get("train_pairs"), project_root_path),
                candidate_ids=_resolve_path(raw_task.get("candidate_ids"), project_root_path),
                candidate_embedding_h5=_resolve_path(
                    raw_task.get("candidate_embedding_h5"), project_root_path
                ),
                candidate_residue_h5=_resolve_path_map(
                    raw_task.get("candidate_residue_h5"), project_root_path
                ),
                candidate_score_residue_h5=_resolve_path_map(
                    raw_task.get("candidate_score_residue_h5"), project_root_path
                ),
                reaction_embeds_h5=_resolve_path(
                    raw_task.get("reaction_embeds_h5"), project_root_path
                ),
                reaction_model_embeds_h5=_resolve_path(
                    raw_task.get("reaction_model_embeds_h5"), project_root_path
                ),
                reaction_unimol2_embeds_h5=_resolve_path(
                    raw_task.get("reaction_unimol2_embeds_h5"), project_root_path
                ),
                reaction_chiro_embeds_h5=_resolve_path(
                    raw_task.get("reaction_chiro_embeds_h5"), project_root_path
                ),
                reaction_chirality_embeds_h5=_resolve_path(
                    raw_task.get("reaction_chirality_embeds_h5"), project_root_path
                ),
                reaction_chienn_embeds_h5=_resolve_path(
                    raw_task.get("reaction_chienn_embeds_h5"), project_root_path
                ),
                reaction_id_col=str(raw_task.get("reaction_id_col", "reaction_id")),
                reaction_smiles_col=str(raw_task.get("reaction_smiles_col", "reaction_smiles")),
                protein_id_col=str(raw_task.get("protein_id_col", "protein_id")),
                directions=directions,
                top_k=top_k,
                bedroc_alphas=bedroc_alphas,
                ef_fractions=ef_fractions,
            )
        )
    return resolved_tasks


def task_to_dict(task: BenchmarkTask) -> dict[str, Any]:
    """Return a JSON-serializable task definition."""

    return {
        "name": task.name,
        "type": task.task_type,
        "dataset": task.dataset,
        "task_label": task.task_label,
        "split": task.split,
        "pairs": str(task.pairs),
        "reactions": str(task.reactions),
        "train_pairs": None if task.train_pairs is None else str(task.train_pairs),
        "candidate_ids": None if task.candidate_ids is None else str(task.candidate_ids),
        "candidate_embedding_h5": (
            None if task.candidate_embedding_h5 is None else str(task.candidate_embedding_h5)
        ),
        "candidate_residue_h5": {
            key: str(value) for key, value in task.candidate_residue_h5.items()
        },
        "candidate_score_residue_h5": {
            key: str(value) for key, value in task.candidate_score_residue_h5.items()
        },
        "reaction_embeds_h5": (
            None if task.reaction_embeds_h5 is None else str(task.reaction_embeds_h5)
        ),
        "reaction_model_embeds_h5": (
            None if task.reaction_model_embeds_h5 is None else str(task.reaction_model_embeds_h5)
        ),
        "reaction_unimol2_embeds_h5": (
            None
            if task.reaction_unimol2_embeds_h5 is None
            else str(task.reaction_unimol2_embeds_h5)
        ),
        "reaction_chiro_embeds_h5": (
            None if task.reaction_chiro_embeds_h5 is None else str(task.reaction_chiro_embeds_h5)
        ),
        "reaction_chirality_embeds_h5": (
            None
            if task.reaction_chirality_embeds_h5 is None
            else str(task.reaction_chirality_embeds_h5)
        ),
        "reaction_chienn_embeds_h5": (
            None if task.reaction_chienn_embeds_h5 is None else str(task.reaction_chienn_embeds_h5)
        ),
        "directions": list(task.directions),
        "top_k": list(task.top_k),
        "bedroc_alphas": list(task.bedroc_alphas),
        "ef_fractions": list(task.ef_fractions),
    }


def read_id_list(path: str | Path | None, column: str | None = None) -> list[str] | None:
    if path is None:
        return None
    id_path = Path(path)
    if not id_path.exists():
        raise FileNotFoundError(f"ID file not found: {id_path}")
    if column is not None:
        with id_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{column}' not found in {id_path}; columns={reader.fieldnames}"
                )
            return [row[column] for row in reader if row.get(column)]
    ids = []
    with id_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line.split(",")[0].split()[0])
    return ids


def read_pairs(
    path: str | Path, reaction_id_col: str, protein_id_col: str
) -> list[tuple[str, str]]:
    pairs_path = Path(path)
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pair file not found: {pairs_path}")
    pairs: list[tuple[str, str]] = []
    with pairs_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Pair file has no header: {pairs_path}")
        missing = [col for col in (reaction_id_col, protein_id_col) if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns in {pairs_path}: {missing}")
        for row in reader:
            reaction_id = row.get(reaction_id_col)
            protein_id = row.get(protein_id_col)
            if reaction_id and protein_id:
                pairs.append((reaction_id, protein_id))
    return pairs


def group_pairs(
    pairs: Iterable[tuple[str, str]],
    allowed_reactions: set[str] | None = None,
    allowed_proteins: set[str] | None = None,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    reaction_to_proteins: dict[str, set[str]] = defaultdict(set)
    protein_to_reactions: dict[str, set[str]] = defaultdict(set)
    for reaction_id, protein_id in pairs:
        if allowed_reactions is not None and reaction_id not in allowed_reactions:
            continue
        if allowed_proteins is not None and protein_id not in allowed_proteins:
            continue
        reaction_to_proteins[reaction_id].add(protein_id)
        protein_to_reactions[protein_id].add(reaction_id)
    return (
        {key: sorted(values) for key, values in reaction_to_proteins.items()},
        {key: sorted(values) for key, values in protein_to_reactions.items()},
    )


def mean_dict(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = sorted({key for row in metric_rows for key in row})
    return {key: sum(row.get(key, 0.0) for row in metric_rows) / len(metric_rows) for key in keys}


def rank_metrics_for_query(
    scores: torch.Tensor,
    positive_indices: list[int],
    top_k_values: list[int] | tuple[int, ...],
) -> dict[str, float]:
    """Compute retrieval metrics for one query."""

    if not positive_indices:
        return {}
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(scores.numel(), device=scores.device, dtype=order.dtype)
    positive_tensor = torch.as_tensor(positive_indices, device=scores.device, dtype=torch.long)
    positive_ranks = torch.sort(ranks[positive_tensor].to(torch.float32) + 1.0).values
    if positive_ranks.numel() == 0:
        return {}

    metrics: dict[str, float] = {
        "mean_rank": float(positive_ranks.mean().item()),
        "mrr": float((1.0 / positive_ranks[0]).item()),
    }
    relevant_count = int(positive_ranks.numel())
    r_cutoff = min(relevant_count, int(scores.numel()))
    metrics["r_precision"] = float((positive_ranks <= r_cutoff).sum().item() / relevant_count)
    precisions = (
        torch.arange(
            1,
            relevant_count + 1,
            device=scores.device,
            dtype=torch.float32,
        )
        / positive_ranks
    )
    metrics["avg_precision"] = float(precisions.mean().item())
    for k in top_k_values:
        hits = int((positive_ranks <= k).sum().item())
        metrics[f"top_{k}"] = float(hits > 0)
        metrics[f"top_{k}_n"] = float(hits / k)
    return metrics


def _metric_key(value: float) -> str:
    return f"{value:g}".replace(".", "_")


def screening_metrics_for_query(
    scores: torch.Tensor,
    positive_indices: list[int],
    bedroc_alphas: list[float] | tuple[float, ...],
    ef_fractions: list[float] | tuple[float, ...],
) -> dict[str, float]:
    """Compute BEDROC and enrichment metrics for one screening query."""

    metrics: dict[str, float] = {}
    num_mol = int(scores.numel())
    num_actives = len(positive_indices)
    if num_mol == 0:
        raise ValueError("score list is empty")
    if num_actives == 0:
        for alpha in bedroc_alphas:
            metrics[f"bedroc_{_metric_key(alpha)}"] = 0.0
        for fraction in ef_fractions:
            metrics[f"ef_{_metric_key(fraction)}"] = 0.0
        return metrics

    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(num_mol, device=scores.device, dtype=order.dtype)
    positive_tensor = torch.as_tensor(positive_indices, device=scores.device, dtype=torch.long)
    positive_ranks = ranks[positive_tensor]

    for alpha in bedroc_alphas:
        if alpha <= 0:
            raise ValueError("BEDROC alpha must be greater than zero")
        alpha_key = _metric_key(alpha)
        ratio = float(num_actives) / float(num_mol)
        denom = (1.0 / num_mol) * ((-math.expm1(-alpha)) / math.expm1(alpha / num_mol))
        sum_exp = torch.exp(-alpha * (positive_ranks.to(torch.float64) + 1.0) / num_mol).sum()
        rie = float(sum_exp.item()) / (num_actives * denom)
        rie_max = (-math.expm1(-alpha * ratio)) / (ratio * (-math.expm1(-alpha)))
        rie_min = math.expm1(alpha * ratio) / (ratio * math.expm1(alpha))
        metrics[f"bedroc_{alpha_key}"] = (
            float((rie - rie_min) / (rie_max - rie_min)) if rie_max != rie_min else 1.0
        )

    for fraction in ef_fractions:
        if fraction < 0 or fraction > 1:
            raise ValueError("enrichment fractions must be in [0, 1]")
        fraction_key = _metric_key(fraction)
        cutoff = math.ceil(num_mol * fraction)
        if cutoff <= 0:
            metrics[f"ef_{fraction_key}"] = 0.0
            continue
        hits = int((positive_ranks < cutoff).sum().item())
        metrics[f"ef_{fraction_key}"] = float(hits * num_mol / (cutoff * num_actives))
    return metrics


def normalize_reaction_smiles(_key: str, sample: dict) -> dict:
    """Turn molecule-set strings into pseudo reactions for fingerprint encoders."""

    smiles = sample.get("reaction_smiles", "")
    if isinstance(smiles, str) and smiles.count(">") < 2:
        sample = dict(sample)
        sample["reaction_smiles"] = f"{smiles}>>{smiles}"
    return sample


def reaction_input_mode(config: Any) -> str:
    representation = config.data.get("reaction_representation", "fingerprint")
    query_encoder_type = config.model.get("query_encoder_type", "mlp")
    if representation == "hybrid_fingerprint_unimol2" or query_encoder_type == "hybrid_reaction":
        return "hybrid_fingerprint_unimol2"
    if (
        representation == "multimodal_reaction_attention"
        or query_encoder_type == "multimodal_reaction_attention"
    ):
        return "multimodal_reaction_attention"
    if representation == "unimol2_attention" or query_encoder_type == "unimol2_reaction_attention":
        return "unimol2_attention"
    return "fingerprint"


def build_reaction_inputs(task: BenchmarkTask, config: Any) -> BaseDataset:
    config_for_reactions = config
    mode = reaction_input_mode(config)
    if mode in {"hybrid_fingerprint_unimol2", "unimol2_attention"}:
        if task.reaction_embeds_h5 is None:
            raise FileNotFoundError(
                f"Task {task.name} needs UniMol2 reaction embeddings for config mode {mode}"
            )
        if not task.reaction_embeds_h5.exists():
            raise FileNotFoundError(f"Reaction embedding HDF5 not found: {task.reaction_embeds_h5}")
        config_for_reactions = copy.deepcopy(config)
        config_for_reactions.data.reaction_embeds_path = str(task.reaction_embeds_h5)
    elif mode == "multimodal_reaction_attention":
        reaction_model_h5 = task.reaction_model_embeds_h5
        if reaction_model_h5 is None:
            configured = config.data.get("reaction_t5v2_embeds_path", None) or config.data.get(
                "reaction_model_embeds_path", None
            )
            reaction_model_h5 = None if configured is None else Path(configured)
        reaction_unimol2_h5 = task.reaction_unimol2_embeds_h5 or task.reaction_embeds_h5
        if reaction_unimol2_h5 is None:
            configured = config.data.get("reaction_unimol2_embeds_path", None) or config.data.get(
                "reaction_embeds_path", None
            )
            reaction_unimol2_h5 = None if configured is None else Path(configured)
        reaction_use_chirality = bool(
            config.data.get(
                "reaction_use_chiro",
                config.data.get(
                    "reaction_use_chirality",
                    config.data.get(
                        "reaction_use_chienn",
                        config.model.get(
                            "reaction_use_chiro",
                            config.model.get(
                                "reaction_use_chirality",
                                config.model.get("reaction_use_chienn", True),
                            ),
                        ),
                    ),
                ),
            )
        )
        reaction_chirality_h5 = (
            task.reaction_chiro_embeds_h5
            or task.reaction_chirality_embeds_h5
            or task.reaction_chienn_embeds_h5
        )
        if reaction_use_chirality and reaction_chirality_h5 is None:
            configured = (
                config.data.get("reaction_chiro_embeds_path", None)
                or config.data.get("reaction_chirality_embeds_path", None)
                or config.data.get("reaction_chienn_embeds_path", None)
            )
            reaction_chirality_h5 = None if configured is None else Path(configured)

        required_paths = {
            "reaction_model_embeds_h5": reaction_model_h5,
            "reaction_unimol2_embeds_h5": reaction_unimol2_h5,
        }
        if reaction_use_chirality:
            required_paths["reaction_chiro_embeds_h5"] = reaction_chirality_h5
        for label, path in required_paths.items():
            if path is None:
                raise FileNotFoundError(f"Task {task.name} needs {label} for config mode {mode}")
            if not path.exists():
                raise FileNotFoundError(f"Reaction embedding HDF5 not found: {path}")

        config_for_reactions = copy.deepcopy(config)
        config_for_reactions.data.reaction_t5v2_embeds_path = str(reaction_model_h5)
        config_for_reactions.data.reaction_unimol2_embeds_path = str(reaction_unimol2_h5)
        if reaction_use_chirality:
            config_for_reactions.data.reaction_chiro_embeds_path = str(reaction_chirality_h5)
            config_for_reactions.data.reaction_chirality_embeds_path = str(reaction_chirality_h5)
            config_for_reactions.data.reaction_chienn_embeds_path = str(reaction_chirality_h5)

    return build_reaction_feature_dataset(
        reactions_path=task.reactions,
        config=config_for_reactions,
        key_column=task.reaction_id_col,
        smiles_column=task.reaction_smiles_col,
        bidirectional=False,
        transforms=normalize_reaction_smiles,
    )


def model_kind_from_config(config: Any) -> str:
    model_name = str(config.model.get("name", ""))
    if model_name == "DualContrastiveModel" or "protein_embeds_path" in config.data:
        return "pooled"
    if model_name == "ProteinPooledDualModel" or "protein_residue_embeds_path" in config.data:
        return "residue"
    raise ValueError(
        "Unsupported model config for unified benchmark. Expected DualContrastiveModel "
        "or ProteinPooledDualModel."
    )


def load_repo_checkpoint(checkpoint: str | Path, config: Any, device: str):
    kind = model_kind_from_config(config)
    if kind == "pooled":
        from horizyn.lightning_module import HorizynLitModule

        module = HorizynLitModule.load_from_checkpoint(str(checkpoint), map_location=device)
    else:
        from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule

        try:
            module = ProteinPooledLitModule.load_from_checkpoint(
                str(checkpoint), map_location=device
            )
        except RuntimeError as error:
            message = str(error)
            legacy_factorized_buffer = "model.query_encoder.factorized_weight_values"
            missing_line = next(
                (
                    line.strip()
                    for line in message.splitlines()
                    if line.strip().startswith("Missing key(s) in state_dict:")
                ),
                "",
            )
            expected_missing_line = (
                "Missing key(s) in state_dict: " f'"{legacy_factorized_buffer}".'
            )
            if (
                missing_line != expected_missing_line
                or "Unexpected key(s) in state_dict" in message
            ):
                raise
            # Older attention-fusion checkpoints predate this persistent buffer.
            # It is unused unless modality_fusion == "factorized_concat", and its
            # deterministic value is reconstructed by the current constructor.
            module = ProteinPooledLitModule.load_from_checkpoint(
                str(checkpoint), map_location=device, strict=False
            )
    module.eval()
    module.to(device)
    return module, kind


def select_residue_h5(task: BenchmarkTask, protein_embedding: str) -> Path:
    if protein_embedding in task.candidate_residue_h5:
        return task.candidate_residue_h5[protein_embedding]
    if "default" in task.candidate_residue_h5:
        return task.candidate_residue_h5["default"]
    available = sorted(task.candidate_residue_h5)
    raise ValueError(
        f"Task {task.name} has no residue HDF5 for '{protein_embedding}'. Available={available}"
    )


def needs_score_residue_embeddings(config: Any) -> bool:
    """Return whether target encoding needs a separate residue store for SLEEC scores."""

    sleec_pooling = config.model.get("sleec_pooling", {})
    if hasattr(sleec_pooling, "get") and sleec_pooling.get("score_embedding_source") == "external":
        return True
    return bool(config.data.get("protein_score_residue_embeds_path"))


def needs_capability_vectors(config: Any) -> bool:
    return str(config.model.get("enzyme_input_mode", "standard")) in {
        "raw_mean_sleec_hyperbolic_capability_gated",
        "raw_mean_sleec_hyperbolic_capability_blockwise",
    }


def needs_text_vectors(config: Any) -> bool:
    return (
        str(config.model.get("enzyme_input_mode", "standard"))
        == "raw_mean_sleec_hyperbolic_text_gated"
    )


def load_capability_vectors_from_config(config: Any) -> CapabilityVectorDataset | None:
    if not needs_capability_vectors(config):
        return None
    path = config.data.get("protein_capability_vectors_path", None)
    if path in {None, ""}:
        raise ValueError(
            "data.protein_capability_vectors_path is required for "
            "capability-branch benchmark evaluation"
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Protein capability vector file not found: {path}")
    return CapabilityVectorDataset(path)


def load_text_vectors_from_config(config: Any) -> TextVectorDataset | None:
    if not needs_text_vectors(config):
        return None
    path = config.data.get("protein_text_vectors_path", None)
    if path in {None, ""}:
        raise ValueError(
            "data.protein_text_vectors_path is required for "
            "raw_mean_sleec_hyperbolic_text_gated benchmark evaluation"
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Protein text vector file not found: {path}")
    return TextVectorDataset(path)


def select_score_residue_h5(task: BenchmarkTask, score_protein_embedding: str) -> Path:
    """Select the task-local residue HDF5 used for external SLEEC score embeddings."""

    for key in (score_protein_embedding, "default"):
        if key in task.candidate_score_residue_h5:
            return task.candidate_score_residue_h5[key]

    # Backward-compatible fallback for manifests that list all residue families
    # under candidate_residue_h5 only.
    for key in (score_protein_embedding, "default"):
        if key in task.candidate_residue_h5:
            return task.candidate_residue_h5[key]

    available = sorted(set(task.candidate_score_residue_h5) | set(task.candidate_residue_h5))
    raise ValueError(
        f"Task {task.name} has no score residue HDF5 for "
        f"'{score_protein_embedding}'. Available={available}"
    )


def load_candidate_keys_from_embedding(
    dataset: EmbedDataset,
    candidate_ids_path: Path | None,
) -> tuple[list[str], dict[str, int]]:
    requested_ids = read_id_list(candidate_ids_path)
    key_set = set(dataset.keys)
    if requested_ids is None:
        return list(dataset.keys), {
            "requested_candidate_count": len(dataset.keys),
            "candidate_count": len(dataset.keys),
            "missing_candidate_id_count": 0,
            "zero_length_candidate_count": 0,
        }
    candidate_keys = [protein_id for protein_id in requested_ids if protein_id in key_set]
    return candidate_keys, {
        "requested_candidate_count": len(requested_ids),
        "candidate_count": len(candidate_keys),
        "missing_candidate_id_count": len(requested_ids) - len(candidate_keys),
        "zero_length_candidate_count": 0,
    }


def load_candidate_keys_from_residue(
    dataset: ResidueEmbedDataset,
    candidate_ids_path: Path | None,
) -> tuple[list[str], dict[str, int]]:
    residue_lengths = dataset.offsets[1:] - dataset.offsets[:-1]
    length_by_key = {
        protein_id: int(length.item()) for protein_id, length in zip(dataset.keys, residue_lengths)
    }
    requested_ids = read_id_list(candidate_ids_path)
    key_set = set(dataset.keys)
    source_ids = list(dataset.keys) if requested_ids is None else requested_ids
    candidate_keys = [
        protein_id
        for protein_id in source_ids
        if protein_id in key_set and length_by_key.get(protein_id, 0) > 0
    ]
    return candidate_keys, {
        "requested_candidate_count": len(source_ids),
        "candidate_count": len(candidate_keys),
        "missing_candidate_id_count": sum(protein_id not in key_set for protein_id in source_ids),
        "zero_length_candidate_count": sum(
            protein_id in key_set and length_by_key.get(protein_id, 0) <= 0
            for protein_id in source_ids
        ),
    }


def filter_candidate_keys_by_score_residue(
    candidate_keys: list[str],
    score_dataset: ResidueEmbedDataset,
) -> tuple[list[str], dict[str, int]]:
    score_lengths = score_dataset.offsets[1:] - score_dataset.offsets[:-1]
    score_length_by_key = {
        protein_id: int(length.item())
        for protein_id, length in zip(score_dataset.keys, score_lengths)
    }
    score_key_set = set(score_dataset.keys)
    filtered_keys = [
        protein_id
        for protein_id in candidate_keys
        if protein_id in score_key_set and score_length_by_key.get(protein_id, 0) > 0
    ]
    return filtered_keys, {
        "score_candidate_store_count": len(score_dataset.keys),
        "score_candidate_store_overlap": len(set(candidate_keys) & score_key_set),
        "score_missing_candidate_id_count": sum(
            protein_id not in score_key_set for protein_id in candidate_keys
        ),
        "score_zero_length_candidate_count": sum(
            protein_id in score_key_set and score_length_by_key.get(protein_id, 0) <= 0
            for protein_id in candidate_keys
        ),
    }


def validate_task_inputs(
    task: BenchmarkTask,
    reaction_inputs: BaseDataset,
    target_keys: list[str],
    target_store_keys: list[str],
    eval_pairs: list[tuple[str, str]],
    candidate_stats: dict[str, int],
) -> dict[str, int | str]:
    target_key_set = set(target_keys)
    store_key_set = set(target_store_keys)
    pair_reactions = {reaction_id for reaction_id, _protein_id in eval_pairs}
    pair_proteins = {protein_id for _reaction_id, protein_id in eval_pairs}
    reaction_key_set = set(reaction_inputs.keys)
    stats: dict[str, int | str] = {
        "pair_count": len(eval_pairs),
        "unique_pair_reactions": len(pair_reactions),
        "unique_pair_proteins": len(pair_proteins),
        "candidate_count": len(target_keys),
        "candidate_store_count": len(target_store_keys),
        "candidate_store_overlap": len(target_key_set & store_key_set),
        "missing_reaction_count": len(pair_reactions - reaction_key_set),
        "missing_candidate_positive_count": len(pair_proteins - target_key_set),
        "missing_store_positive_count": len(pair_proteins - store_key_set),
        "reaction_id_col": task.reaction_id_col,
        "protein_id_col": task.protein_id_col,
        **candidate_stats,
    }
    if not target_keys:
        raise ValueError(f"Task {task.name} has no evaluable candidates")
    if stats["candidate_store_overlap"] <= 0:
        raise ValueError(f"Task {task.name} candidate IDs do not overlap target store")
    return stats


def build_query_inputs(
    reaction_inputs: BaseDataset,
    reaction_ids: list[str],
    device: str,
) -> torch.Tensor | dict[str, torch.Tensor]:
    samples = [reaction_inputs[reaction_id] for reaction_id in reaction_ids]
    if isinstance(samples[0], dict):
        batch = unimol2_reaction_collate_fn(samples)
        return {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if torch.is_tensor(value)
        }
    return torch.stack(samples).to(device, non_blocking=True)


def encode_model_queries(
    model: torch.nn.Module,
    query_inputs: dict | torch.Tensor,
    retrieval_direction: str = "reaction_to_enzyme",
) -> torch.Tensor:
    if hasattr(model, "encode_queries"):
        try:
            return model.encode_queries(
                query_inputs,
                retrieval_direction=retrieval_direction,
            )
        except TypeError as exc:
            if "retrieval_direction" not in str(exc):
                raise
            return model.encode_queries(query_inputs)
    if isinstance(query_inputs, dict):
        return model.query_encoder(**query_inputs)
    return model.query_encoder(query_inputs)


def encode_reactions(
    module: torch.nn.Module,
    reaction_inputs: BaseDataset,
    reaction_ids: list[str],
    device: str,
    batch_size: int,
    retrieval_direction: str = "reaction_to_enzyme",
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch_start in range(0, len(reaction_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(reaction_ids))
            query_vecs = build_query_inputs(
                reaction_inputs,
                reaction_ids[batch_start:batch_end],
                device,
            )
            encoded = encode_model_queries(
                module.model,
                query_vecs,
                retrieval_direction=retrieval_direction,
            )
            output.append(encoded.detach())
    if not output:
        return torch.empty(0, 0, dtype=torch.float32, device=device)
    return torch.cat(output, dim=0)


def encode_pooled_targets(
    module: torch.nn.Module,
    dataset: EmbedDataset,
    target_keys: list[str],
    device: str,
    batch_size: int,
    store_on_device: bool,
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    storage_device = device if store_on_device else "cpu"
    with torch.inference_mode():
        for batch_start in range(0, len(target_keys), batch_size):
            batch_end = min(batch_start + batch_size, len(target_keys))
            vectors = torch.stack(
                [dataset[target_id] for target_id in target_keys[batch_start:batch_end]]
            )
            encoded = module.model.target_encoder(vectors.to(device, non_blocking=True))
            output.append(encoded.detach().to(storage_device))
    if not output:
        return torch.empty(0, 0, dtype=torch.float32, device=storage_device)
    return torch.cat(output, dim=0)


def encode_residue_targets(
    module: torch.nn.Module,
    dataset: ResidueEmbedDataset,
    target_keys: list[str],
    device: str,
    batch_size: int,
    store_on_device: bool,
    score_dataset: ResidueEmbedDataset | None = None,
    capability_dataset: CapabilityVectorDataset | None = None,
    text_dataset: TextVectorDataset | None = None,
    progress_every_batches: int = 0,
    retrieval_direction: str = "reaction_to_enzyme",
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    storage_device = device if store_on_device else "cpu"
    model_dtype = next(
        (
            parameter.dtype
            for parameter in module.model.parameters()
            if parameter.is_floating_point()
        ),
        torch.float32,
    )
    capability_key_set = set(capability_dataset.keys) if capability_dataset is not None else set()
    text_key_set = set(text_dataset.keys) if text_dataset is not None else set()
    capability_zero = (
        torch.zeros(capability_dataset.vec_dim, dtype=torch.float32)
        if capability_dataset is not None
        else None
    )
    text_zero = (
        torch.zeros(text_dataset.vec_dim, dtype=torch.float32) if text_dataset is not None else None
    )
    total_batches = math.ceil(len(target_keys) / batch_size)
    started_at = time.monotonic()
    with torch.inference_mode():
        for batch_index, batch_start in enumerate(
            range(0, len(target_keys), batch_size),
            start=1,
        ):
            batch_end = min(batch_start + batch_size, len(target_keys))
            batch_target_keys = target_keys[batch_start:batch_end]
            samples = []
            for target_id in batch_target_keys:
                sample = dict(dataset[target_id])
                if score_dataset is not None:
                    score_sample = score_dataset[target_id]
                    sample["score_residue_embeddings"] = score_sample["residue_embeddings"]
                sample["target_id"] = target_id
                samples.append(sample)
            batch = residue_collate_fn(samples)
            residues = batch["residue_embeddings"].to(
                device=device,
                dtype=model_dtype,
                non_blocking=True,
            )
            mask = batch["residue_padding_mask"].to(device, non_blocking=True)
            score_residues = batch.get("score_residue_embeddings")
            score_mask = batch.get("score_residue_padding_mask")
            if score_residues is not None:
                score_residues = score_residues.to(
                    device=device,
                    dtype=model_dtype,
                    non_blocking=True,
                )
                score_mask = score_mask.to(device, non_blocking=True)
            capability_vectors = None
            capability_mask = None
            text_vectors = None
            text_mask = None
            if capability_dataset is not None:
                capability_rows = []
                capability_mask_values = []
                for target_id in batch_target_keys:
                    if target_id in capability_key_set:
                        capability_rows.append(capability_dataset[target_id].float())
                        capability_mask_values.append(True)
                    else:
                        if capability_zero is None:
                            raise RuntimeError("capability_zero was not initialized")
                        capability_rows.append(capability_zero.clone())
                        capability_mask_values.append(False)
                capability_vectors = torch.stack(capability_rows).to(device, non_blocking=True)
                capability_mask = torch.as_tensor(
                    capability_mask_values,
                    dtype=torch.bool,
                    device=device,
                )
            if text_dataset is not None:
                text_rows = []
                text_mask_values = []
                for target_id in batch_target_keys:
                    if target_id in text_key_set:
                        text_rows.append(text_dataset[target_id].float())
                        text_mask_values.append(True)
                    else:
                        if text_zero is None:
                            raise RuntimeError("text_zero was not initialized")
                        text_rows.append(text_zero.clone())
                        text_mask_values.append(False)
                text_vectors = torch.stack(text_rows).to(device, non_blocking=True)
                text_mask = torch.as_tensor(
                    text_mask_values,
                    dtype=torch.bool,
                    device=device,
                )
            encoded = module.model.encode_targets(
                residues,
                residue_padding_mask=mask,
                score_residue_embeddings=score_residues,
                score_residue_padding_mask=score_mask,
                capability_vectors=capability_vectors,
                capability_mask=capability_mask,
                text_vectors=text_vectors,
                text_mask=text_mask,
                retrieval_direction=retrieval_direction,
            )
            output.append(encoded.detach().to(storage_device))
            if progress_every_batches > 0 and (
                batch_index % progress_every_batches == 0 or batch_index == total_batches
            ):
                encoded_count = min(batch_start + batch_size, len(target_keys))
                elapsed = max(time.monotonic() - started_at, 1e-9)
                print(
                    "Encoded candidate enzymes: "
                    f"{encoded_count:,}/{len(target_keys):,} "
                    f"({encoded_count / max(len(target_keys), 1):.1%}); "
                    f"{encoded_count / elapsed:,.1f} proteins/s",
                    flush=True,
                )
    if not output:
        return torch.empty(0, 0, dtype=torch.float32, device=storage_device)
    return torch.cat(output, dim=0)


def _stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _path_fingerprint(path: str | Path | None) -> dict[str, Any] | None:
    if path in {None, ""}:
        return None
    resolved = Path(path).expanduser().resolve()
    stat_result = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
    }


def _target_cache_base_metadata(
    *,
    kind: str,
    checkpoint: str | Path,
    config_path: str | Path,
    protein_embedding: str,
    score_protein_embedding: str,
    residue_h5: str | Path | None = None,
    score_residue_h5: str | Path | None = None,
    candidate_embedding_h5: str | Path | None = None,
    capability_vectors_path: str | Path | None = None,
    capability_missing_policy: str | None = None,
    text_vectors_path: str | Path | None = None,
    text_vector_missing_policy: str | None = None,
    max_tokens: int | None = None,
    truncation: str | None = None,
    retrieval_direction: str = "reaction_to_enzyme",
) -> dict[str, Any]:
    return {
        "schema_version": TARGET_CACHE_SCHEMA_VERSION,
        "kind": kind,
        "checkpoint": _path_fingerprint(checkpoint),
        "config": _path_fingerprint(config_path),
        "protein_embedding": protein_embedding,
        "score_protein_embedding": score_protein_embedding,
        "residue_h5": _path_fingerprint(residue_h5),
        "score_residue_h5": _path_fingerprint(score_residue_h5),
        "candidate_embedding_h5": _path_fingerprint(candidate_embedding_h5),
        "capability_vectors": _path_fingerprint(capability_vectors_path),
        "capability_missing_policy": capability_missing_policy,
        "text_vectors": _path_fingerprint(text_vectors_path),
        "text_vector_missing_policy": text_vector_missing_policy,
        "max_tokens": None if max_tokens is None else int(max_tokens),
        "truncation": truncation,
        "retrieval_direction": retrieval_direction,
        "embedding_dtype": "float32",
    }


def _target_cache_paths(
    target_cache_dir: str | Path,
    base_metadata: dict[str, Any],
    target_keys: list[str],
) -> tuple[str, Path, Path, Path]:
    target_keys_sha256 = _sha256_strings(target_keys)
    cache_key = hashlib.sha256(
        _stable_json_dumps(
            {
                "base": base_metadata,
                "target_keys_sha256": target_keys_sha256,
                "num_targets": len(target_keys),
            }
        ).encode("utf-8")
    ).hexdigest()
    cache_dir = Path(target_cache_dir)
    return (
        cache_key,
        cache_dir / f"{cache_key}.pt",
        cache_dir / f"{cache_key}.json",
        cache_dir / f"{cache_key}.lock",
    )


def _cache_sidecar_matches(
    sidecar: dict[str, Any],
    base_metadata: dict[str, Any],
) -> bool:
    return (
        sidecar.get("schema_version") == TARGET_CACHE_SCHEMA_VERSION
        and sidecar.get("base_metadata") == base_metadata
    )


def _load_target_cache_payload(
    tensor_path: Path,
    *,
    expected_num_targets: int,
) -> torch.Tensor:
    payload = torch.load(tensor_path, map_location="cpu")
    if not isinstance(payload, dict) or "target_embeds" not in payload:
        raise ValueError(f"Target cache payload is malformed: {tensor_path}")
    target_embeds = payload["target_embeds"]
    if not isinstance(target_embeds, torch.Tensor):
        raise ValueError(f"Target cache embeddings are not a tensor: {tensor_path}")
    if target_embeds.shape[0] != expected_num_targets:
        raise ValueError(
            f"Target cache row count mismatch for {tensor_path}: "
            f"expected {expected_num_targets}, got {target_embeds.shape[0]}"
        )
    return target_embeds.float().cpu()


def _move_target_cache_tensor(
    target_embeds: torch.Tensor,
    *,
    device: str,
    store_on_device: bool,
) -> torch.Tensor:
    return target_embeds.to(device) if store_on_device else target_embeds.cpu()


def _read_target_cache_sidecar(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _try_load_target_cache(
    target_cache_dir: str | Path,
    base_metadata: dict[str, Any],
    target_keys: list[str],
    *,
    device: str,
    store_on_device: bool,
) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    cache_dir = Path(target_cache_dir)
    cache_key, tensor_path, sidecar_path, _ = _target_cache_paths(
        cache_dir,
        base_metadata,
        target_keys,
    )
    target_keys_sha256 = _sha256_strings(target_keys)

    sidecar = _read_target_cache_sidecar(sidecar_path)
    if (
        sidecar is not None
        and _cache_sidecar_matches(sidecar, base_metadata)
        and sidecar.get("target_keys_sha256") == target_keys_sha256
        and sidecar.get("target_keys") == target_keys
        and tensor_path.exists()
    ):
        target_embeds = _load_target_cache_payload(
            tensor_path,
            expected_num_targets=len(target_keys),
        )
        return (
            _move_target_cache_tensor(
                target_embeds,
                device=device,
                store_on_device=store_on_device,
            ),
            {
                "status": "hit_exact",
                "cache_key": cache_key,
                "cache_path": str(tensor_path),
                "num_targets": len(target_keys),
            },
        )

    target_key_set = set(target_keys)
    for candidate_sidecar_path in sorted(cache_dir.glob("*.json")):
        if candidate_sidecar_path == sidecar_path:
            continue
        candidate_sidecar = _read_target_cache_sidecar(candidate_sidecar_path)
        if candidate_sidecar is None or not _cache_sidecar_matches(
            candidate_sidecar,
            base_metadata,
        ):
            continue
        cached_keys = candidate_sidecar.get("target_keys")
        if not isinstance(cached_keys, list) or len(cached_keys) < len(target_keys):
            continue
        cached_key_set = set(cached_keys)
        if not target_key_set.issubset(cached_key_set):
            continue
        source_tensor_path = candidate_sidecar_path.with_suffix(".pt")
        if not source_tensor_path.exists():
            continue
        source_embeds = _load_target_cache_payload(
            source_tensor_path,
            expected_num_targets=len(cached_keys),
        )
        cached_key_to_idx = {key: idx for idx, key in enumerate(cached_keys)}
        rows = torch.as_tensor(
            [cached_key_to_idx[key] for key in target_keys],
            dtype=torch.long,
        )
        target_embeds = source_embeds.index_select(0, rows)
        return (
            _move_target_cache_tensor(
                target_embeds,
                device=device,
                store_on_device=store_on_device,
            ),
            {
                "status": "hit_superset",
                "cache_key": cache_key,
                "source_cache_path": str(source_tensor_path),
                "source_num_targets": len(cached_keys),
                "num_targets": len(target_keys),
            },
        )
    return None, None


def _try_acquire_target_cache_lock(lock_path: Path) -> bool:
    now = time.time()
    try:
        stat_result = lock_path.stat()
    except FileNotFoundError:
        stat_result = None
    if stat_result is not None and now - stat_result.st_mtime > TARGET_CACHE_LOCK_STALE_SECONDS:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "created_at": now}))
    return True


def prepare_target_embedding_cache(
    target_cache_dir: str | Path | None,
    base_metadata: dict[str, Any],
    target_keys: list[str],
    *,
    device: str,
    store_on_device: bool,
) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    if target_cache_dir in {None, ""}:
        return None, {"status": "disabled"}

    cache_dir = Path(target_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key, tensor_path, sidecar_path, lock_path = _target_cache_paths(
        cache_dir,
        base_metadata,
        target_keys,
    )

    while True:
        cached_embeds, cache_info = _try_load_target_cache(
            cache_dir,
            base_metadata,
            target_keys,
            device=device,
            store_on_device=store_on_device,
        )
        if cached_embeds is not None:
            print(
                f"Target embedding cache {cache_info['status']}: "
                f"{cache_info.get('cache_path', cache_info.get('source_cache_path'))}",
                flush=True,
            )
            return cached_embeds, cache_info

        if _try_acquire_target_cache_lock(lock_path):
            return None, {
                "status": "miss_encode",
                "cache_key": cache_key,
                "cache_path": str(tensor_path),
                "sidecar_path": str(sidecar_path),
                "lock_path": str(lock_path),
                "num_targets": len(target_keys),
            }

        print(f"Waiting for target embedding cache lock: {lock_path}", flush=True)
        time.sleep(TARGET_CACHE_LOCK_POLL_SECONDS)


def write_target_embedding_cache(
    cache_info: dict[str, Any] | None,
    base_metadata: dict[str, Any],
    target_keys: list[str],
    target_embeds: torch.Tensor,
) -> None:
    if cache_info is None or cache_info.get("status") != "miss_encode":
        return

    tensor_path = Path(cache_info["cache_path"])
    sidecar_path = Path(cache_info["sidecar_path"])
    lock_path = Path(cache_info["lock_path"])
    cache_dir = tensor_path.parent
    target_embeds_cpu = target_embeds.detach().float().cpu()
    sidecar = {
        "schema_version": TARGET_CACHE_SCHEMA_VERSION,
        "base_metadata": base_metadata,
        "target_keys_sha256": _sha256_strings(target_keys),
        "target_keys": target_keys,
        "num_targets": len(target_keys),
        "embedding_shape": list(target_embeds_cpu.shape),
        "tensor_file": tensor_path.name,
        "created_at": time.time(),
    }
    payload = {
        "schema_version": TARGET_CACHE_SCHEMA_VERSION,
        "base_metadata": base_metadata,
        "target_keys": target_keys,
        "target_embeds": target_embeds_cpu,
    }

    temp_tensor_path = cache_dir / f".{tensor_path.name}.{os.getpid()}.tmp"
    temp_sidecar_path = cache_dir / f".{sidecar_path.name}.{os.getpid()}.tmp"
    try:
        torch.save(payload, temp_tensor_path)
        with temp_sidecar_path.open("w", encoding="utf-8") as handle:
            json.dump(sidecar, handle, indent=2)
        os.replace(temp_tensor_path, tensor_path)
        os.replace(temp_sidecar_path, sidecar_path)
        print(f"Saved target embedding cache: {tensor_path}", flush=True)
    finally:
        for temp_path in (temp_tensor_path, temp_sidecar_path):
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def release_target_embedding_cache_lock(cache_info: dict[str, Any] | None) -> None:
    if cache_info is None or cache_info.get("status") != "miss_encode":
        return
    lock_path = Path(cache_info["lock_path"])
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def evaluate_retrieval_direction(
    query_ids: list[str],
    candidate_ids: list[str],
    score_matrix: torch.Tensor,
    query_to_candidates: dict[str, list[str]],
    top_k_values: list[int] | tuple[int, ...],
) -> dict[str, float | int]:
    candidate_to_idx = {candidate_id: idx for idx, candidate_id in enumerate(candidate_ids)}
    metric_rows: list[dict[str, float]] = []
    for row_idx, query_id in enumerate(query_ids):
        positives = [
            candidate_to_idx[candidate_id]
            for candidate_id in query_to_candidates.get(query_id, [])
            if candidate_id in candidate_to_idx
        ]
        if not positives:
            continue
        metric_rows.append(
            rank_metrics_for_query(score_matrix[row_idx], positives, top_k_values=top_k_values)
        )
    results: dict[str, float | int] = mean_dict(metric_rows)
    results["num_queries"] = len(metric_rows)
    results["num_targets"] = len(candidate_ids)
    return results


def l2_normalize_embeddings(embeds: torch.Tensor) -> torch.Tensor:
    """Return embeddings normalized for cosine retrieval scoring."""
    return F.normalize(embeds.float(), p=2, dim=-1, eps=COSINE_EPS)


def cosine_scores(query_embeds: torch.Tensor, target_embeds: torch.Tensor) -> torch.Tensor:
    """Compute cosine-similarity scores after defensive L2 normalization."""
    query_embeds = l2_normalize_embeddings(query_embeds)
    target_embeds = l2_normalize_embeddings(target_embeds)
    return torch.matmul(query_embeds, target_embeds.T)


def dot_scores(query_embeds: torch.Tensor, target_embeds: torch.Tensor) -> torch.Tensor:
    """Compute raw dot-product scores without norm correction."""
    return torch.matmul(query_embeds.float(), target_embeds.float().T)


def parse_scoring_mode(value: str | None, *, allow_both: bool, default: str) -> str:
    mode = (value or default).strip().lower()
    allowed = {"dot", "cosine"}
    if allow_both:
        allowed.add("both")
    if mode not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"Invalid scoring mode {mode!r}; expected one of: {allowed_text}")
    return mode


def score_embeddings(
    query_embeds: torch.Tensor,
    target_embeds: torch.Tensor,
    scoring_mode: str,
) -> torch.Tensor:
    if scoring_mode == "cosine":
        return cosine_scores(query_embeds, target_embeds)
    if scoring_mode == "dot":
        return dot_scores(query_embeds, target_embeds)
    raise ValueError(f"score_embeddings expects 'dot' or 'cosine', got {scoring_mode!r}")


def scoring_metadata(scoring_mode: str) -> dict[str, str]:
    if scoring_mode == "cosine":
        return {
            "score_type": "cosine_similarity",
            "embedding_normalization": "l2",
        }
    if scoring_mode == "dot":
        return {
            "score_type": "dot_product",
            "embedding_normalization": "none",
        }
    raise ValueError(f"Unknown scoring mode: {scoring_mode!r}")


def evaluate_retrieval(
    module: torch.nn.Module,
    reaction_inputs: BaseDataset,
    target_embeds: torch.Tensor,
    candidate_keys: list[str],
    reaction_to_proteins: dict[str, list[str]],
    protein_to_reactions: dict[str, list[str]],
    directions: tuple[str, ...],
    top_k_values: tuple[int, ...],
    device: str,
    batch_size: int,
    score_dump_task_name: str | None = None,
    e2r_target_embeds: torch.Tensor | None = None,
) -> dict[str, Any]:
    reaction_key_set = set(reaction_inputs.keys)
    reaction_ids = sorted({rid for rid in reaction_to_proteins if rid in reaction_key_set})
    reaction_embeds = encode_reactions(
        module,
        reaction_inputs,
        reaction_ids,
        device,
        batch_size,
        retrieval_direction="reaction_to_enzyme",
    )
    protein_embeds = (
        target_embeds.to(device) if target_embeds.device.type == "cpu" else target_embeds
    )
    e2r_protein_embeds = (
        protein_embeds
        if e2r_target_embeds is None
        else (
            e2r_target_embeds.to(device)
            if e2r_target_embeds.device.type == "cpu"
            else e2r_target_embeds
        )
    )
    e2r_reaction_embeds = (
        encode_reactions(
            module,
            reaction_inputs,
            reaction_ids,
            device,
            batch_size,
            retrieval_direction="enzyme_to_reaction",
        )
        if "enzyme_to_reaction" in directions
        else reaction_embeds
    )

    requested_scoring_mode = parse_scoring_mode(
        os.environ.get("HORIZYN_RETRIEVAL_SCORING_MODE"),
        allow_both=True,
        default="cosine",
    )
    scoring_modes = (
        ("dot", "cosine") if requested_scoring_mode == "both" else (requested_scoring_mode,)
    )
    primary_scoring_mode = "cosine" if requested_scoring_mode == "both" else requested_scoring_mode
    primary_meta = scoring_metadata(primary_scoring_mode)
    results: dict[str, Any] = {
        "setting": "retrieval",
        "requested_scoring_mode": requested_scoring_mode,
        "primary_scoring_mode": primary_scoring_mode,
        **primary_meta,
    }
    with torch.inference_mode():
        dump_root = os.environ.get("HORIZYN_RETRIEVAL_SCORE_DUMP_DIR")
        if dump_root and score_dump_task_name:
            dump_dir = Path(dump_root) / score_dump_task_name
            dump_dir.mkdir(parents=True, exist_ok=True)
            for scoring_mode in scoring_modes:
                full_scores = score_embeddings(
                    e2r_protein_embeds,
                    e2r_reaction_embeds,
                    scoring_mode,
                )
                score_filenames = (
                    [f"scores_enzyme_by_reaction_{scoring_mode}.npz"]
                    if requested_scoring_mode == "both"
                    else ["scores_enzyme_by_reaction.npz"]
                )
                if requested_scoring_mode == "both" and scoring_mode == primary_scoring_mode:
                    score_filenames.append("scores_enzyme_by_reaction.npz")
                for score_filename in score_filenames:
                    np.savez(
                        dump_dir / score_filename,
                        scores=full_scores.detach().float().cpu().numpy(),
                        enzyme_ids=np.asarray(candidate_keys, dtype=str),
                        reaction_ids=np.asarray(reaction_ids, dtype=str),
                    )
                    metadata_filename = (
                        "score_metadata.json"
                        if score_filename == "scores_enzyme_by_reaction.npz"
                        else f"score_metadata_{scoring_mode}.json"
                    )
                    with (dump_dir / metadata_filename).open("w", encoding="utf-8") as handle:
                        json.dump(
                            {
                                "task": score_dump_task_name,
                                "format": "npz_dense_enzyme_by_reaction",
                                "scores": score_filename,
                                **scoring_metadata(scoring_mode),
                                "shape": [len(candidate_keys), len(reaction_ids)],
                                "rows": "enzyme_ids",
                                "columns": "reaction_ids",
                            },
                            handle,
                            indent=2,
                        )

        reaction_id_set = set(reaction_ids)
        protein_query_ids = [
            protein_id
            for protein_id in candidate_keys
            if any(
                reaction_id in reaction_id_set
                for reaction_id in protein_to_reactions.get(protein_id, [])
            )
        ]
        protein_to_idx = {protein_id: idx for idx, protein_id in enumerate(candidate_keys)}
        protein_rows = [protein_to_idx[protein_id] for protein_id in protein_query_ids]

        for scoring_mode in scoring_modes:
            suffix = "" if requested_scoring_mode != "both" else f"_{scoring_mode}"
            if "reaction_to_enzyme" in directions:
                scores = score_embeddings(reaction_embeds, protein_embeds, scoring_mode)
                metrics = evaluate_retrieval_direction(
                    query_ids=reaction_ids,
                    candidate_ids=candidate_keys,
                    score_matrix=scores,
                    query_to_candidates=reaction_to_proteins,
                    top_k_values=top_k_values,
                )
                results[f"reaction_to_enzyme{suffix}"] = metrics
                if scoring_mode == primary_scoring_mode:
                    results["reaction_to_enzyme"] = metrics

            if "enzyme_to_reaction" in directions:
                scores = score_embeddings(
                    e2r_protein_embeds[protein_rows],
                    e2r_reaction_embeds,
                    scoring_mode,
                )
                metrics = evaluate_retrieval_direction(
                    query_ids=protein_query_ids,
                    candidate_ids=reaction_ids,
                    score_matrix=scores,
                    query_to_candidates=protein_to_reactions,
                    top_k_values=top_k_values,
                )
                results[f"enzyme_to_reaction{suffix}"] = metrics
                if scoring_mode == primary_scoring_mode:
                    results["enzyme_to_reaction"] = metrics
    return results


def evaluate_screening(
    module: torch.nn.Module,
    reaction_inputs: BaseDataset,
    target_embeds: torch.Tensor,
    candidate_keys: list[str],
    reaction_to_proteins: dict[str, list[str]],
    device: str,
    batch_size: int,
    bedroc_alphas: tuple[float, ...],
    ef_fractions: tuple[float, ...],
) -> dict[str, Any]:
    candidate_to_idx = {protein_id: idx for idx, protein_id in enumerate(candidate_keys)}
    reaction_key_set = set(reaction_inputs.keys)
    query_ids = [
        reaction_id
        for reaction_id in sorted(reaction_to_proteins)
        if reaction_id in reaction_key_set
        and any(protein_id in candidate_to_idx for protein_id in reaction_to_proteins[reaction_id])
    ]
    if not query_ids:
        raise ValueError("No evaluable screening queries after candidate/positive filtering")
    target_embeds_for_scoring = (
        target_embeds.to(device) if target_embeds.device.type == "cpu" else target_embeds
    )
    screening_scoring_mode = parse_scoring_mode(
        os.environ.get("HORIZYN_SCREENING_SCORING_MODE"),
        allow_both=False,
        default="dot",
    )
    metric_rows: list[dict[str, float]] = []
    with torch.inference_mode():
        for query_start in range(0, len(query_ids), batch_size):
            query_end = min(query_start + batch_size, len(query_ids))
            batch_ids = query_ids[query_start:query_end]
            query_vecs = build_query_inputs(reaction_inputs, batch_ids, device)
            query_embeds = encode_model_queries(module.model, query_vecs)
            score_batch = score_embeddings(
                query_embeds,
                target_embeds_for_scoring,
                screening_scoring_mode,
            )
            for row_idx, reaction_id in enumerate(batch_ids):
                positives = [
                    candidate_to_idx[protein_id]
                    for protein_id in reaction_to_proteins[reaction_id]
                    if protein_id in candidate_to_idx
                ]
                metric_rows.append(
                    screening_metrics_for_query(
                        score_batch[row_idx],
                        positives,
                        bedroc_alphas=bedroc_alphas,
                        ef_fractions=ef_fractions,
                    )
                )
    results: dict[str, Any] = mean_dict(metric_rows)
    results.update(
        {
            "setting": "screening",
            "scoring_mode": screening_scoring_mode,
            **scoring_metadata(screening_scoring_mode),
            "num_queries": len(metric_rows),
            "num_targets": len(candidate_keys),
        }
    )
    return results


def run_benchmark_task(
    task: BenchmarkTask,
    checkpoint: str | Path,
    config_path: str | Path,
    *,
    protein_embedding: str = "prott5",
    device: str = "cuda",
    query_batch_size: int = 128,
    target_batch_size: int = 512,
    store_targets_on_cpu: bool = False,
    validate_only: bool = False,
    score_protein_embedding: str = "esm2",
    target_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run or validate one benchmark task for an existing repo checkpoint."""

    config = load_config(str(config_path))
    kind = model_kind_from_config(config)
    target_retrieval_direction = (
        "reaction_to_enzyme"
        if task.is_screening or "reaction_to_enzyme" in task.directions
        else "enzyme_to_reaction"
    )
    module = None
    capability_dataset = None
    capability_vectors_path = None
    capability_missing_policy = None
    text_dataset = None
    text_vectors_path = None
    text_vector_missing_policy = None

    reaction_inputs = build_reaction_inputs(task, config)
    eval_pairs = read_pairs(task.pairs, task.reaction_id_col, task.protein_id_col)

    if kind == "pooled":
        if task.candidate_embedding_h5 is None:
            raise ValueError(f"Task {task.name} does not define candidate_embedding_h5")
        target_dataset = EmbedDataset(str(task.candidate_embedding_h5), in_memory=False)
        candidate_keys, candidate_stats = load_candidate_keys_from_embedding(
            target_dataset,
            task.candidate_ids,
        )
        target_store_keys = list(target_dataset.keys)
        target_cache_base_metadata = _target_cache_base_metadata(
            kind=kind,
            checkpoint=checkpoint,
            config_path=config_path,
            protein_embedding=protein_embedding,
            score_protein_embedding=score_protein_embedding,
            candidate_embedding_h5=task.candidate_embedding_h5,
            retrieval_direction=target_retrieval_direction,
        )
    else:
        residue_h5 = select_residue_h5(task, protein_embedding)
        max_tokens = config.data.get("max_protein_tokens", 1024)
        truncation = config.data.get("protein_truncation", "ends_center")
        target_dataset = ResidueEmbedDataset(
            str(residue_h5),
            in_memory=False,
            max_tokens=max_tokens,
            truncation=truncation,
            retrieval_direction=target_retrieval_direction,
        )
        candidate_keys, candidate_stats = load_candidate_keys_from_residue(
            target_dataset,
            task.candidate_ids,
        )
        score_target_dataset = None
        if needs_score_residue_embeddings(config):
            score_residue_h5 = select_score_residue_h5(task, score_protein_embedding)
            score_target_dataset = ResidueEmbedDataset(
                str(score_residue_h5),
                in_memory=False,
                max_tokens=max_tokens,
                truncation=truncation,
            )
            candidate_keys, score_candidate_stats = filter_candidate_keys_by_score_residue(
                candidate_keys,
                score_target_dataset,
            )
            candidate_stats.update(score_candidate_stats)
        capability_dataset = load_capability_vectors_from_config(config)
        if capability_dataset is not None:
            capability_vectors_path = config.data.get("protein_capability_vectors_path", None)
            capability_missing_policy = config.data.get(
                "capability_missing_policy",
                "zero_with_mask",
            )
            capability_key_set = set(capability_dataset.keys)
            missing_capability = len(set(candidate_keys) - capability_key_set)
            candidate_stats["capability_vector_missing_count"] = missing_capability
            candidate_stats["capability_vector_dim"] = capability_dataset.vec_dim
        text_dataset = load_text_vectors_from_config(config)
        if text_dataset is not None:
            text_vectors_path = config.data.get("protein_text_vectors_path", None)
            text_vector_missing_policy = config.data.get(
                "text_vector_missing_policy",
                "zero_with_mask",
            )
            text_key_set = set(text_dataset.keys)
            missing_text = len(set(candidate_keys) - text_key_set)
            candidate_stats["text_vector_missing_count"] = missing_text
            candidate_stats["text_vector_dim"] = text_dataset.vec_dim
        target_store_keys = list(target_dataset.keys)
        target_cache_base_metadata = _target_cache_base_metadata(
            kind=kind,
            checkpoint=checkpoint,
            config_path=config_path,
            protein_embedding=protein_embedding,
            score_protein_embedding=score_protein_embedding,
            residue_h5=residue_h5,
            score_residue_h5=score_residue_h5 if score_target_dataset is not None else None,
            capability_vectors_path=capability_vectors_path,
            capability_missing_policy=capability_missing_policy,
            text_vectors_path=text_vectors_path,
            text_vector_missing_policy=text_vector_missing_policy,
            max_tokens=max_tokens,
            truncation=truncation,
        )

    validation_stats = validate_task_inputs(
        task,
        reaction_inputs,
        candidate_keys,
        target_store_keys,
        eval_pairs,
        candidate_stats,
    )
    reaction_to_proteins, protein_to_reactions = group_pairs(
        eval_pairs,
        allowed_reactions=set(reaction_inputs.keys),
        allowed_proteins=set(candidate_keys),
    )

    result: dict[str, Any] = {
        "task": task.name,
        "dataset": task.dataset,
        "task_label": task.task_label,
        "setting": task.task_type,
        "split": task.split,
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "protein_embedding": protein_embedding,
        "score_protein_embedding": score_protein_embedding,
        "candidate_pool_policy": "published",
        "pairs": str(task.pairs),
        "reactions": str(task.reactions),
        "candidate_ids": None if task.candidate_ids is None else str(task.candidate_ids),
        "candidate_pool_size": len(candidate_keys),
        "reaction_input_mode": reaction_input_mode(config),
        "validation": validation_stats,
        "validate_only": validate_only,
    }
    if kind == "pooled":
        result["candidate_embedding_h5"] = str(task.candidate_embedding_h5)
    else:
        result["candidate_residue_h5"] = str(select_residue_h5(task, protein_embedding))
        if needs_score_residue_embeddings(config):
            result["candidate_score_residue_h5"] = str(
                select_score_residue_h5(task, score_protein_embedding)
            )
        if capability_vectors_path is not None:
            result["candidate_capability_vectors"] = str(capability_vectors_path)
        if text_vectors_path is not None:
            result["candidate_text_vectors"] = str(text_vectors_path)
    if task.reaction_embeds_h5 is not None:
        result["reaction_embeds_h5"] = str(task.reaction_embeds_h5)

    if validate_only:
        return result

    with torch.inference_mode():
        target_embeds, target_cache_info = prepare_target_embedding_cache(
            target_cache_dir,
            target_cache_base_metadata,
            candidate_keys,
            device=device,
            store_on_device=not store_targets_on_cpu,
        )
        if kind == "pooled":
            if target_embeds is None:
                module, loaded_kind = load_repo_checkpoint(checkpoint, config, device)
                if loaded_kind != kind:
                    raise RuntimeError(f"Loaded model kind changed from {kind} to {loaded_kind}")
                try:
                    target_embeds = encode_pooled_targets(
                        module,
                        target_dataset,
                        candidate_keys,
                        device,
                        target_batch_size,
                        store_on_device=not store_targets_on_cpu,
                    )
                except Exception:
                    release_target_embedding_cache_lock(target_cache_info)
                    raise
                write_target_embedding_cache(
                    target_cache_info,
                    target_cache_base_metadata,
                    candidate_keys,
                    target_embeds,
                )
        else:
            if target_embeds is None:
                module, loaded_kind = load_repo_checkpoint(checkpoint, config, device)
                if loaded_kind != kind:
                    raise RuntimeError(f"Loaded model kind changed from {kind} to {loaded_kind}")
                try:
                    target_embeds = encode_residue_targets(
                        module,
                        target_dataset,
                        candidate_keys,
                        device,
                        target_batch_size,
                        store_on_device=not store_targets_on_cpu,
                        score_dataset=score_target_dataset,
                        capability_dataset=capability_dataset,
                        text_dataset=text_dataset,
                        retrieval_direction=target_retrieval_direction,
                    )
                except Exception:
                    release_target_embedding_cache_lock(target_cache_info)
                    raise
                write_target_embedding_cache(
                    target_cache_info,
                    target_cache_base_metadata,
                    candidate_keys,
                    target_embeds,
                )

        result["target_embedding_cache"] = target_cache_info
        if module is None:
            module, loaded_kind = load_repo_checkpoint(checkpoint, config, device)
            if loaded_kind != kind:
                raise RuntimeError(f"Loaded model kind changed from {kind} to {loaded_kind}")

        e2r_target_embeds = None
        if (
            not task.is_screening
            and "enzyme_to_reaction" in task.directions
            and target_retrieval_direction != "enzyme_to_reaction"
            and getattr(module.model, "r2e_adapter", None) is not None
        ):
            if kind != "residue":
                raise ValueError("R2E adapters require residue-level enzyme inputs")
            e2r_target_embeds = encode_residue_targets(
                module,
                target_dataset,
                candidate_keys,
                device,
                target_batch_size,
                store_on_device=not store_targets_on_cpu,
                score_dataset=score_target_dataset,
                capability_dataset=capability_dataset,
                text_dataset=text_dataset,
                retrieval_direction="enzyme_to_reaction",
            )

        if task.is_screening:
            metrics = evaluate_screening(
                module,
                reaction_inputs,
                target_embeds,
                candidate_keys,
                reaction_to_proteins,
                device,
                query_batch_size,
                task.bedroc_alphas,
                task.ef_fractions,
            )
        else:
            metrics = evaluate_retrieval(
                module,
                reaction_inputs,
                target_embeds,
                candidate_keys,
                reaction_to_proteins,
                protein_to_reactions,
                task.directions,
                task.top_k,
                device,
                query_batch_size,
                score_dump_task_name=task.name,
                e2r_target_embeds=e2r_target_embeds,
            )
    result.update(metrics)
    return result


def flatten_result_metrics(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten task result metrics into summary rows."""

    rows: list[dict[str, Any]] = []
    base = {
        "dataset": result.get("dataset", ""),
        "task": result.get("task", ""),
        "task_label": result.get("task_label", result.get("task", "")),
        "setting": result.get("setting", ""),
        "split": result.get("split", ""),
        "candidate_pool_size": result.get("candidate_pool_size", ""),
    }
    for direction in ("reaction_to_enzyme", "enzyme_to_reaction"):
        metrics = result.get(direction)
        if not isinstance(metrics, dict):
            continue
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                rows.append({**base, "direction": direction, "metric": metric, "value": value})
    screening_metric_names = [
        key
        for key, value in result.items()
        if key.startswith(("bedroc_", "ef_")) and isinstance(value, float)
    ]
    for metric in sorted(screening_metric_names):
        rows.append(
            {**base, "direction": "reaction_to_enzyme", "metric": metric, "value": result[metric]}
        )
    for metric in ("num_queries", "num_targets"):
        if isinstance(result.get(metric), (int, float)):
            rows.append(
                {
                    **base,
                    "direction": "reaction_to_enzyme",
                    "metric": metric,
                    "value": result[metric],
                }
            )
    return rows


def write_summary_csv(results: list[dict[str, Any]], path: str | Path) -> None:
    rows = [row for result in results for row in flatten_result_metrics(result)]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "task",
        "task_label",
        "setting",
        "split",
        "direction",
        "metric",
        "value",
        "candidate_pool_size",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary_markdown(results: list[dict[str, Any]], path: str | Path) -> None:
    rows = [row for result in results for row in flatten_result_metrics(result)]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Unified Retrieval Benchmark Summary",
        "",
        "| Dataset | Task | Direction | Metric | Value |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        value = row.get("value", "")
        if isinstance(value, float):
            value_text = f"{value:.6g}"
        else:
            value_text = str(value)
        lines.append(
            f"| {row.get('dataset', '')} | {row.get('task', '')} | "
            f"{row.get('direction', '')} | "
            f"{row.get('metric', '')} | {value_text} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


WIDE_METRIC_ORDER = [
    "top_1",
    "top_2",
    "top_3",
    "top_4",
    "top_5",
    "top_10",
    "top_20",
    "top_50",
    "top_100",
    "top_1000",
    "mrr",
    "avg_precision",
    "r_precision",
    "mean_rank",
    "num_queries",
    "num_targets",
]


def _safe_table_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return safe.strip("_") or "unknown"


def _wide_metric_columns(rows: list[dict[str, Any]]) -> list[str]:
    metrics = sorted({str(row.get("metric", "")) for row in rows if row.get("metric")})
    ordered = [metric for metric in WIDE_METRIC_ORDER if metric in metrics]
    ordered.extend(metric for metric in metrics if metric not in ordered)
    return ordered


def pivot_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pivot long benchmark rows into one row per dataset/task/direction."""

    grouped: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("task", "")),
            str(row.get("task_label", "")),
            str(row.get("setting", "")),
            str(row.get("split", "")),
            str(row.get("direction", "")),
        )
        item = grouped.setdefault(
            key,
            {
                "dataset": key[0],
                "task": key[1],
                "task_label": key[2],
                "setting": key[3],
                "split": key[4],
                "direction": key[5],
                "candidate_pool_size": row.get("candidate_pool_size", ""),
            },
        )
        metric = row.get("metric")
        if metric:
            item[str(metric)] = row.get("value", "")
    return [grouped[key] for key in sorted(grouped)]


def write_wide_metric_table(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write a wide CSV table for a subset of flattened benchmark rows."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_columns = _wide_metric_columns(rows)
    fieldnames = [
        "dataset",
        "task",
        "task_label",
        "setting",
        "split",
        "direction",
        "candidate_pool_size",
        *metric_columns,
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in pivot_metric_rows(rows):
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_wide_metric_markdown(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write a compact Markdown table for a subset of flattened benchmark rows."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_columns = _wide_metric_columns(rows)
    columns = [
        "dataset",
        "task",
        "direction",
        "candidate_pool_size",
        *metric_columns,
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in pivot_metric_rows(rows):
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_grouped_summary_tables(results: list[dict[str, Any]], output_dir: str | Path) -> None:
    """Write per-dataset and per-task benchmark performance tables."""

    rows = [row for result in results for row in flatten_result_metrics(result)]
    root = Path(output_dir)
    write_wide_metric_table(rows, root / "summary_wide.csv")
    write_wide_metric_markdown(rows, root / "summary_wide.md")

    by_dataset = root / "tables" / "by_dataset"
    for dataset in sorted({str(row.get("dataset", "")) for row in rows}):
        subset = [row for row in rows if str(row.get("dataset", "")) == dataset]
        name = _safe_table_name(dataset)
        write_wide_metric_table(subset, by_dataset / f"{name}.csv")
        write_wide_metric_markdown(subset, by_dataset / f"{name}.md")

    by_task = root / "tables" / "by_task"
    for task in sorted({str(row.get("task", "")) for row in rows}):
        subset = [row for row in rows if str(row.get("task", "")) == task]
        name = _safe_table_name(task)
        write_wide_metric_table(subset, by_task / f"{name}.csv")
        write_wide_metric_markdown(subset, by_task / f"{name}.md")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
