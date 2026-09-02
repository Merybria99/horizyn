"""Unified enzyme-reaction retrieval benchmark utilities."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import socket
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from horizyn.config import load_config
from horizyn.artifacts import (
    ArtifactManifestV2,
    SplitRoleManifestV2,
    current_git_revision,
    fingerprint_file,
    sha256_file,
    sha256_strings,
)
from horizyn.datasets.base import BaseDataset
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
TARGET_CACHE_SCHEMA_VERSION = 2
METRIC_SCHEMA_VERSION = 3
TARGET_CACHE_LOCK_POLL_SECONDS = 10.0
TARGET_CACHE_LOCK_STALE_SECONDS = 24 * 60 * 60
TARGET_CACHE_LOCK_TIMEOUT_SECONDS = 5 * 60.0
DEFAULT_CANDIDATE_SCORE_BATCH_SIZE = 8192
VALID_RETRIEVAL_DIRECTIONS = {"reaction_to_enzyme", "enzyme_to_reaction"}
VALID_METRIC_PROTOCOLS = {"canonical", "horizyn", "reactzyme", "screening"}

# These identities describe the released Level-1 benchmark data after the
# repository's lossless CSV conversion.  The digests are semantic (sorted
# ``reaction_id<TAB>protein_id`` records), so newline style and row order do
# not affect compliance.  A task must opt in explicitly; metric names alone
# are not enough to claim that a result follows a published protocol.
LEVEL1_BENCHMARK_PROTOCOLS: dict[str, dict[str, Any]] = {
    "horizyn_release_v1": {
        "upstream_revision": "6944198303f2f0946d448a259ab788589cfd27b3",
        "metric_protocol": "horizyn",
        "candidates_from_test_positives": False,
        "bidirectional_reactions": True,
        "directions": ("reaction_to_enzyme",),
        "top_k": (1, 10, 100, 1000),
        "test_pair_count": 33_996,
        "test_pair_digest": "f829c74c592c68ac194aa5717ef0d14708ec2a214459a3b4e513fcd59ebbf770",
        "train_pair_count": 257_733,
        "train_pair_digest": "11c5f2397ff4d52b6fe8e0f9204be0e4dd418bb110bfa3661fee19146dabe2dc",
        "reaction_row_count": 1_012,
        "reaction_digest": "88353fafcf09e4b93996699bb0873e78013e7affb021da0a3b9228d414040df6",
        "raw_query_count": 1_012,
        "evaluation_query_count": 2_024,
        "positive_protein_count": 32_100,
        "candidate_count": 216_132,
        "candidate_digest": "44b2d8fd728b90ddad39d468f2cf92a123a918f49bf2021d9e87989834ad12f6",
    },
    "reactzyme_time_release_v2": {
        "upstream_revision": "c4d1554640a01e5b8ae8e2de716b452ce93b1bc6",
        "metric_protocol": "reactzyme",
        "candidates_from_test_positives": True,
        "bidirectional_reactions": False,
        "directions": ("reaction_to_enzyme", "enzyme_to_reaction"),
        "top_k": (1, 2, 3, 4, 5, 10, 20, 50),
        "test_pair_count": 12_287,
        "test_pair_digest": "3b4b0cd64ef5d49fc93ca7c4795e80c8204e01357fafc6ddb1f70be50e4a1e69",
        "train_pair_count": 166_172,
        "train_pair_digest": "0a6852e6ae0271400a4b9095ed3a8efa636e6078813809834ee4d52c23160d7a",
        "reaction_row_count": 7_726,
        "reaction_digest": "513f0a42ba76fed7a4c3cd6ebcfb2783ec47ec80f40957ece4fe7551cbca1995",
        "raw_query_count": 2_634,
        "evaluation_query_count": 2_634,
        "positive_protein_count": 12_277,
        "candidate_count": 12_277,
        "candidate_digest": "f0099878ad4f96af9c6d6740863f53c9cc899386a367384037180d1950d63c38",
    },
    "reactzyme_enzyme_smi_release_v2": {
        "upstream_revision": "c4d1554640a01e5b8ae8e2de716b452ce93b1bc6",
        "metric_protocol": "reactzyme",
        "candidates_from_test_positives": True,
        "bidirectional_reactions": False,
        "directions": ("reaction_to_enzyme", "enzyme_to_reaction"),
        "top_k": (1, 2, 3, 4, 5, 10, 20, 50),
        "test_pair_count": 8_739,
        "test_pair_digest": "5a0b5d5f5d6b10718902b67b2e475de273b67437e7ffca57fe4815922da1b388",
        "train_pair_count": 169_720,
        "train_pair_digest": "d83fbac087810a223cdbcee8f7e94a54cad13431fb4b712d25f0f2c401b7ffd3",
        "reaction_row_count": 7_726,
        "reaction_digest": "513f0a42ba76fed7a4c3cd6ebcfb2783ec47ec80f40957ece4fe7551cbca1995",
        "raw_query_count": 1_573,
        "evaluation_query_count": 1_573,
        "positive_protein_count": 8_734,
        "candidate_count": 8_734,
        "candidate_digest": "a8f4ce9bcc4174401b9f9c8975a30161ac15de2850f0af0c11a666dde1433893",
    },
    "reactzyme_reaction_smi_release_v2": {
        "upstream_revision": "c4d1554640a01e5b8ae8e2de716b452ce93b1bc6",
        "metric_protocol": "reactzyme",
        "candidates_from_test_positives": True,
        "bidirectional_reactions": False,
        "directions": ("reaction_to_enzyme", "enzyme_to_reaction"),
        "top_k": (1, 2, 3, 4, 5, 10, 20, 50),
        "test_pair_count": 14_689,
        "test_pair_digest": "3df12f1ea5a13160b7e0250b64ef5c635a9447d9582e3af5388db8a9b4bbb3fc",
        "train_pair_count": 163_770,
        "train_pair_digest": "25db2904f00109afae4c3490b8363c367bc7bbd08ac6ff688ea68c60a17d8c0a",
        "reaction_row_count": 7_726,
        "reaction_digest": "513f0a42ba76fed7a4c3cd6ebcfb2783ec47ec80f40957ece4fe7551cbca1995",
        "raw_query_count": 386,
        "evaluation_query_count": 386,
        "positive_protein_count": 14_688,
        "candidate_count": 14_688,
        "candidate_digest": "40f24848473f4b8291bf0d6ca9e44f98f5ef4f5ab290a16edec88fae1156f64e",
    },
}


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
    candidates_from_test_positives: bool = False
    bidirectional_reactions: bool = False
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
    metric_protocol: str = "canonical"
    benchmark_protocol: str = "custom"
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
    if not isinstance(raw_suite, dict):
        raise TypeError(f"Benchmark suite root must be a mapping: {suite_path}")
    tasks = raw_suite.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"Benchmark suite has no tasks: {suite_path}")

    resolved_tasks: list[BenchmarkTask] = []
    for raw_task in tasks:
        if not isinstance(raw_task, dict):
            raise TypeError("Each benchmark task must be a mapping")
        raw_name = raw_task.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("Each benchmark task requires a non-empty string name")
        name = raw_name.strip()
        raw_task_type = raw_task.get("type", "retrieval")
        if not isinstance(raw_task_type, str):
            raise ValueError(f"Task {name} type must be a string")
        task_type = raw_task_type.strip()
        if task_type not in {"retrieval", "screening"}:
            raise ValueError(f"Unsupported task type for {name}: {task_type}")
        pairs = _resolve_path(raw_task.get("pairs"), project_root_path)
        reactions = _resolve_path(raw_task.get("reactions"), project_root_path)
        if pairs is None or reactions is None:
            raise ValueError(f"Task {name} requires pairs and reactions paths")

        dataset = str(raw_task.get("dataset", name.split("_")[0]))
        default_top_k = DEFAULT_SOTA_TOP_K if name == "horizyn_sota" else DEFAULT_REACTZYME_TOP_K
        raw_top_k = raw_task.get("top_k", default_top_k)
        if not isinstance(raw_top_k, (list, tuple)) or not raw_top_k:
            raise ValueError(f"Task {name} top_k must be a non-empty list")
        if any(type(value) is not int or value <= 0 for value in raw_top_k):
            raise ValueError(f"Task {name} top_k must contain positive integers")
        if len(set(raw_top_k)) != len(raw_top_k):
            raise ValueError(f"Task {name} top_k must not contain duplicates")
        top_k = tuple(raw_top_k)

        def finite_float_tuple(field_name: str, default: list[float]) -> tuple[float, ...]:
            raw_values = raw_task.get(field_name, default)
            if not isinstance(raw_values, (list, tuple)) or not raw_values:
                raise ValueError(f"Task {name} {field_name} must be a non-empty list")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in raw_values
            ):
                raise ValueError(f"Task {name} {field_name} must contain finite numbers")
            values = tuple(float(value) for value in raw_values)
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"Task {name} {field_name} must contain finite numbers")
            if len(set(values)) != len(values):
                raise ValueError(f"Task {name} {field_name} must not contain duplicates")
            return values

        bedroc_alphas = finite_float_tuple("bedroc_alphas", DEFAULT_BEDROC_ALPHAS)
        if any(value <= 0 for value in bedroc_alphas):
            raise ValueError(f"Task {name} bedroc_alphas must be greater than zero")
        ef_fractions = finite_float_tuple("ef_fractions", DEFAULT_EF_FRACTIONS)
        if any(value < 0 or value > 1 for value in ef_fractions):
            raise ValueError(f"Task {name} ef_fractions must be in [0, 1]")

        default_directions = ["reaction_to_enzyme"] if task_type == "screening" else ["both"]
        raw_directions = raw_task.get("directions", default_directions)
        if not isinstance(raw_directions, (list, tuple)) or not raw_directions:
            raise ValueError(f"Task {name} directions must be a non-empty list")
        if any(not isinstance(value, str) for value in raw_directions):
            raise ValueError(f"Task {name} directions must contain strings")
        directions = tuple(raw_directions)
        allowed_directions = VALID_RETRIEVAL_DIRECTIONS | {"both"}
        unknown_directions = sorted(set(directions) - allowed_directions)
        if unknown_directions:
            raise ValueError(f"Task {name} has unsupported directions: {unknown_directions}")
        if "both" in directions and len(directions) != 1:
            raise ValueError(f"Task {name} must use 'both' alone")
        if "both" in directions:
            directions = ("reaction_to_enzyme", "enzyme_to_reaction")
        if len(set(directions)) != len(directions):
            raise ValueError(f"Task {name} directions must not contain duplicates")
        if task_type == "screening" and directions != ("reaction_to_enzyme",):
            raise ValueError(f"Screening task {name} supports reaction_to_enzyme only")

        metric_protocol = str(raw_task.get("metric_protocol", "auto")).strip().lower()
        if metric_protocol == "auto":
            dataset_key = dataset.strip().lower()
            if task_type == "screening":
                metric_protocol = "screening"
            elif dataset_key == "reactzyme" or name.lower().startswith("reactzyme_"):
                metric_protocol = "reactzyme"
            elif dataset_key == "horizyn" or name.lower().startswith("horizyn_"):
                metric_protocol = "horizyn"
            else:
                metric_protocol = "canonical"
        if metric_protocol not in VALID_METRIC_PROTOCOLS:
            raise ValueError(
                f"Task {name} metric_protocol must be one of {sorted(VALID_METRIC_PROTOCOLS)}"
            )
        if task_type == "screening" and metric_protocol != "screening":
            raise ValueError(f"Screening task {name} must use metric_protocol=screening")
        candidates_from_test_positives = raw_task.get("candidates_from_test_positives", False)
        if type(candidates_from_test_positives) is not bool:
            raise ValueError(f"Task {name} candidates_from_test_positives must be boolean")
        if candidates_from_test_positives and raw_task.get("candidate_ids") not in {None, ""}:
            raise ValueError(
                f"Task {name} cannot combine candidate_ids with " "candidates_from_test_positives"
            )
        bidirectional_reactions = raw_task.get(
            "bidirectional_reactions", metric_protocol == "horizyn"
        )
        if type(bidirectional_reactions) is not bool:
            raise ValueError(f"Task {name} bidirectional_reactions must be boolean")
        benchmark_protocol = str(raw_task.get("benchmark_protocol", "custom")).strip().lower()
        if benchmark_protocol != "custom":
            if benchmark_protocol not in LEVEL1_BENCHMARK_PROTOCOLS:
                raise ValueError(
                    f"Task {name} benchmark_protocol must be 'custom' or one of "
                    f"{sorted(LEVEL1_BENCHMARK_PROTOCOLS)}"
                )
            expected = LEVEL1_BENCHMARK_PROTOCOLS[benchmark_protocol]
            structural_values = {
                "metric_protocol": metric_protocol,
                "candidates_from_test_positives": candidates_from_test_positives,
                "bidirectional_reactions": bidirectional_reactions,
                "directions": directions,
                "top_k": top_k,
            }
            mismatches = {
                key: {"expected": expected[key], "actual": value}
                for key, value in structural_values.items()
                if value != expected[key]
            }
            if mismatches:
                raise ValueError(
                    f"Task {name} does not implement {benchmark_protocol}: {mismatches}"
                )

        resolved_tasks.append(
            BenchmarkTask(
                name=name,
                task_type=task_type,
                dataset=dataset,
                task_label=str(raw_task.get("task_label", name)),
                split=str(raw_task.get("split", name)),
                pairs=pairs,
                reactions=reactions,
                train_pairs=_resolve_path(raw_task.get("train_pairs"), project_root_path),
                candidate_ids=_resolve_path(raw_task.get("candidate_ids"), project_root_path),
                candidates_from_test_positives=candidates_from_test_positives,
                bidirectional_reactions=bidirectional_reactions,
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
                metric_protocol=metric_protocol,
                benchmark_protocol=benchmark_protocol,
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
        "candidates_from_test_positives": task.candidates_from_test_positives,
        "bidirectional_reactions": task.bidirectional_reactions,
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
        "metric_protocol": task.metric_protocol,
        "benchmark_protocol": task.benchmark_protocol,
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
    if id_path.suffix.lower() in {".fa", ".faa", ".fasta", ".fas"}:
        if column is not None:
            raise ValueError("A FASTA candidate manifest does not accept a CSV column")
        ids = []
        with id_path.open("r", encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                if not line.startswith(">"):
                    continue
                header = line[1:].strip()
                if not header:
                    raise ValueError(
                        f"FASTA candidate manifest {id_path} has an empty header "
                        f"at row {row_number}"
                    )
                ids.append(header.split()[0])
        if not ids:
            raise ValueError(f"FASTA candidate manifest contains no records: {id_path}")
        return _require_unique_ids(ids, f"candidate FASTA {id_path}")
    if column is not None:
        with id_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{column}' not found in {id_path}; columns={reader.fieldnames}"
                )
            ids = []
            for row_number, row in enumerate(reader, start=2):
                raw_id = row.get(column)
                if raw_id is None:
                    raise ValueError(
                        f"Candidate ID file {id_path} has a missing cell at row {row_number}"
                    )
                identifier = raw_id.strip()
                if identifier:
                    ids.append(identifier)
            _require_unique_ids(ids, f"candidate ID file {id_path}")
            return ids
    ids = []
    with id_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line.split(",")[0].split()[0])
    _require_unique_ids(ids, f"candidate ID file {id_path}")
    return ids


def _require_unique_ids(values: Iterable[str], label: str) -> list[str]:
    ordered = [str(value) for value in values]
    if any(not value.strip() for value in ordered):
        raise ValueError(f"{label} contains an empty ID")
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in ordered:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        preview = duplicates[:10]
        raise ValueError(f"{label} contains duplicate IDs: {preview}")
    return ordered


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
        seen_pairs: set[tuple[str, str]] = set()
        for row_number, row in enumerate(reader, start=2):
            raw_reaction_id = row.get(reaction_id_col)
            raw_protein_id = row.get(protein_id_col)
            reaction_id = "" if raw_reaction_id is None else raw_reaction_id.strip()
            protein_id = "" if raw_protein_id is None else raw_protein_id.strip()
            if not reaction_id or not protein_id:
                raise ValueError(f"Pair file {pairs_path} has an empty ID at row {row_number}")
            pair = (reaction_id, protein_id)
            if pair in seen_pairs:
                raise ValueError(f"Pair file {pairs_path} contains duplicate pair {pair}")
            seen_pairs.add(pair)
            pairs.append(pair)
    if not pairs:
        raise ValueError(f"Pair file contains no positive pairs: {pairs_path}")
    return pairs


def _semantic_pair_digest(pairs: Iterable[tuple[str, str]]) -> str:
    return sha256_strings(
        sorted(f"{reaction_id}\t{protein_id}" for reaction_id, protein_id in pairs)
    )


def _read_reaction_records(task: BenchmarkTask) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    with task.reactions.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Reaction file has no header: {task.reactions}")
        missing = [
            column
            for column in (task.reaction_id_col, task.reaction_smiles_col)
            if column not in reader.fieldnames
        ]
        if missing:
            raise ValueError(f"Missing columns in {task.reactions}: {missing}")
        seen_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            raw_id = row.get(task.reaction_id_col)
            raw_smiles = row.get(task.reaction_smiles_col)
            reaction_id = "" if raw_id is None else raw_id.strip()
            reaction_smiles = "" if raw_smiles is None else raw_smiles.strip()
            if not reaction_id or not reaction_smiles:
                raise ValueError(
                    f"Reaction file {task.reactions} has an empty ID or SMILES at row "
                    f"{row_number}"
                )
            if reaction_id in seen_ids:
                raise ValueError(
                    f"Reaction file {task.reactions} contains duplicate ID {reaction_id!r}"
                )
            seen_ids.add(reaction_id)
            records.append((reaction_id, reaction_smiles))
    if not records:
        raise ValueError(f"Reaction file contains no records: {task.reactions}")
    return records


def _require_protocol_value(
    task: BenchmarkTask,
    field: str,
    actual: Any,
    expected: Any,
) -> None:
    if actual != expected:
        raise ValueError(
            f"Task {task.name} is not compliant with {task.benchmark_protocol}: "
            f"{field} expected {expected!r}, got {actual!r}"
        )


def validate_level1_benchmark_sources(
    task: BenchmarkTask,
    raw_pairs: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Validate released Level-1 split identity independently of model inputs."""

    if task.benchmark_protocol == "custom":
        return {"benchmark_protocol": "custom", "release_compliance": "not_claimed"}
    spec = LEVEL1_BENCHMARK_PROTOCOLS[task.benchmark_protocol]
    if raw_pairs is None:
        raw_pairs = read_pairs(task.pairs, task.reaction_id_col, task.protein_id_col)
    pair_digest = _semantic_pair_digest(raw_pairs)
    _require_protocol_value(task, "test_pair_count", len(raw_pairs), spec["test_pair_count"])
    _require_protocol_value(task, "test_pair_digest", pair_digest, spec["test_pair_digest"])

    raw_reactions = {reaction_id for reaction_id, _protein_id in raw_pairs}
    positive_proteins = {protein_id for _reaction_id, protein_id in raw_pairs}
    _require_protocol_value(task, "raw_query_count", len(raw_reactions), spec["raw_query_count"])
    _require_protocol_value(
        task,
        "positive_protein_count",
        len(positive_proteins),
        spec["positive_protein_count"],
    )

    if task.train_pairs is None:
        raise ValueError(
            f"Task {task.name} must declare train_pairs to claim {task.benchmark_protocol}"
        )
    train_pairs = read_pairs(task.train_pairs, task.reaction_id_col, task.protein_id_col)
    train_digest = _semantic_pair_digest(train_pairs)
    _require_protocol_value(task, "train_pair_count", len(train_pairs), spec["train_pair_count"])
    _require_protocol_value(task, "train_pair_digest", train_digest, spec["train_pair_digest"])

    reaction_records = _read_reaction_records(task)
    reaction_digest = sha256_strings(
        sorted(f"{reaction_id}\t{smiles}" for reaction_id, smiles in reaction_records)
    )
    _require_protocol_value(
        task, "reaction_row_count", len(reaction_records), spec["reaction_row_count"]
    )
    _require_protocol_value(task, "reaction_digest", reaction_digest, spec["reaction_digest"])

    declared_candidate_ids: list[str] | None
    if task.candidates_from_test_positives:
        declared_candidate_ids = sorted(positive_proteins)
    else:
        declared_candidate_ids = read_id_list(task.candidate_ids)
    candidate_source = "embedding_store"
    if declared_candidate_ids is not None:
        candidate_source = (
            "test_positive_entities"
            if task.candidates_from_test_positives
            else "candidate_manifest"
        )
        _require_protocol_value(
            task,
            "candidate_count",
            len(declared_candidate_ids),
            spec["candidate_count"],
        )
        _require_protocol_value(
            task,
            "candidate_digest",
            sha256_strings(sorted(declared_candidate_ids)),
            spec["candidate_digest"],
        )

    return {
        "benchmark_protocol": task.benchmark_protocol,
        "release_compliance": "source_files_verified",
        "upstream_revision": spec["upstream_revision"],
        "test_pair_count": len(raw_pairs),
        "test_pair_digest": pair_digest,
        "train_pair_count": len(train_pairs),
        "train_pair_digest": train_digest,
        "reaction_row_count": len(reaction_records),
        "reaction_digest": reaction_digest,
        "raw_query_count": len(raw_reactions),
        "positive_protein_count": len(positive_proteins),
        "declared_candidate_source": candidate_source,
        "declared_candidate_count": (
            None if declared_candidate_ids is None else len(declared_candidate_ids)
        ),
    }


def validate_level1_candidate_pool(
    task: BenchmarkTask,
    candidate_keys: list[str],
) -> dict[str, Any]:
    """Verify the resolved candidate universe before a published-protocol run."""

    if task.benchmark_protocol == "custom":
        return {"release_candidate_pool_compliance": "not_claimed"}
    spec = LEVEL1_BENCHMARK_PROTOCOLS[task.benchmark_protocol]
    candidate_digest = sha256_strings(sorted(candidate_keys))
    _require_protocol_value(task, "candidate_count", len(candidate_keys), spec["candidate_count"])
    _require_protocol_value(task, "candidate_digest", candidate_digest, spec["candidate_digest"])
    return {
        "release_candidate_pool_compliance": "verified",
        "candidate_count": len(candidate_keys),
        "candidate_digest": candidate_digest,
    }


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


def candidate_pool_policy(
    task: BenchmarkTask,
    candidate_keys: list[str],
    eval_pairs: list[tuple[str, str]],
) -> str:
    """Classify the candidate universe without overstating protocol fidelity."""

    positive_proteins = {protein_id for _, protein_id in eval_pairs}
    if task.candidates_from_test_positives:
        if set(candidate_keys) != positive_proteins:
            raise ValueError(
                f"Task {task.name} declares test-positive candidates but its candidate "
                "universe does not match the test positives"
            )
        return "published_test_positive_entities"
    if task.benchmark_protocol == "horizyn_release_v1":
        return "published_full_embedding_store"
    if task.candidate_ids is None:
        return "full_embedding_store"
    if task.metric_protocol == "reactzyme" and set(candidate_keys) == positive_proteins:
        return "published_test_positive_entities"
    return "explicit_candidate_manifest"


def restrict_candidates_to_test_positives(
    task: BenchmarkTask,
    eval_pairs: list[tuple[str, str]],
    available_keys: list[str],
    candidate_stats: dict[str, int],
) -> tuple[list[str], dict[str, int]]:
    """Derive ReactZyme's published candidate universe from test positives."""

    if not task.candidates_from_test_positives:
        return available_keys, candidate_stats
    requested = {protein_id for _, protein_id in eval_pairs}
    available_set = set(available_keys)
    missing = sorted(requested - available_set)
    if missing:
        raise ValueError(
            f"Task {task.name} is missing {len(missing)} test-positive candidates from "
            f"the embedding store: {missing[:10]}"
        )
    keys = [key for key in available_keys if key in requested]
    stats = dict(candidate_stats)
    stats["candidate_source_store_count"] = len(available_keys)
    stats["requested_candidate_count"] = len(requested)
    stats["candidate_count"] = len(keys)
    stats["candidate_manifest_count"] = len(keys)
    stats["candidate_source_test_positives"] = 1
    return keys, stats


def expand_bidirectional_pairs(
    task: BenchmarkTask,
    pairs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Mirror Horizyn's published forward/reverse reaction-query expansion."""

    if not task.bidirectional_reactions:
        return pairs
    expanded = []
    for reaction_id, protein_id in pairs:
        expanded.append((f"{reaction_id}_f", protein_id))
        expanded.append((f"{reaction_id}_r", protein_id))
    return expanded


def mean_dict(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = sorted({key for row in metric_rows for key in row})
    return {key: sum(row.get(key, 0.0) for row in metric_rows) / len(metric_rows) for key in keys}


def rank_metrics_for_query(
    scores: torch.Tensor,
    positive_indices: list[int],
    top_k_values: list[int] | tuple[int, ...],
    *,
    metric_protocol: str = "canonical",
) -> dict[str, float]:
    """Compute retrieval metrics for one query."""

    if scores.dim() != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty 1D tensor")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must all be finite")
    if not positive_indices:
        return {}
    if any(type(index) is not int for index in positive_indices):
        raise ValueError("positive_indices must contain integers")
    unique_positive_indices = sorted(set(positive_indices))
    if unique_positive_indices[0] < 0 or unique_positive_indices[-1] >= scores.numel():
        raise ValueError("positive_indices contains an out-of-range candidate index")
    if metric_protocol not in {"canonical", "horizyn", "reactzyme"}:
        raise ValueError(f"Unsupported retrieval metric protocol: {metric_protocol!r}")
    if not top_k_values or any(type(k) is not int or k <= 0 for k in top_k_values):
        raise ValueError("top_k_values must contain positive integers")
    if len(set(top_k_values)) != len(top_k_values):
        raise ValueError("top_k_values must not contain duplicates")
    order = torch.argsort(scores, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(scores.numel(), device=scores.device, dtype=order.dtype)
    positive_tensor = torch.as_tensor(
        unique_positive_indices, device=scores.device, dtype=torch.long
    )
    positive_ranks = torch.sort(ranks[positive_tensor].to(torch.float32) + 1.0).values
    if positive_ranks.numel() == 0:
        return {}

    first_positive_mrr = float(torch.reciprocal(positive_ranks[0]).item())
    reactzyme_mrr = float(torch.reciprocal(positive_ranks).mean().item())
    metrics: dict[str, float] = {
        "mean_rank": float(positive_ranks.mean().item()),
        "first_positive_mrr": first_positive_mrr,
        "reactzyme_mrr": reactzyme_mrr,
        "mrr": reactzyme_mrr if metric_protocol == "reactzyme" else first_positive_mrr,
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
        cutoff = min(int(k), int(scores.numel()))
        hits = int((positive_ranks <= cutoff).sum().item())
        metrics[f"top_{k}"] = float(hits > 0)
        # ReactZyme calls this Acc-N and divides by the requested k even when
        # the candidate pool contains fewer than k entries.
        metrics[f"top_{k}_n"] = float(hits / int(k))
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
    if scores.dim() != 1 or not torch.isfinite(scores).all():
        raise ValueError("screening scores must be a finite 1D tensor")
    if any(type(index) is not int for index in positive_indices):
        raise ValueError("positive_indices must contain integers")
    positive_indices = sorted(set(positive_indices))
    if positive_indices and (positive_indices[0] < 0 or positive_indices[-1] >= num_mol):
        raise ValueError("positive_indices contains an out-of-range candidate index")
    num_actives = len(positive_indices)
    if num_mol == 0:
        raise ValueError("score list is empty")
    if (
        not bedroc_alphas
        or any(not math.isfinite(alpha) or alpha <= 0 for alpha in bedroc_alphas)
        or len(set(bedroc_alphas)) != len(bedroc_alphas)
    ):
        raise ValueError("BEDROC alphas must be unique, finite, and greater than zero")
    if (
        not ef_fractions
        or any(
            not math.isfinite(fraction) or fraction < 0 or fraction > 1 for fraction in ef_fractions
        )
        or len(set(ef_fractions)) != len(ef_fractions)
    ):
        raise ValueError("enrichment fractions must be unique, finite, and in [0, 1]")
    if num_actives == 0:
        for alpha in bedroc_alphas:
            metrics[f"bedroc_{_metric_key(alpha)}"] = 0.0
        for fraction in ef_fractions:
            metrics[f"ef_{_metric_key(fraction)}"] = 0.0
        return metrics

    order = torch.argsort(scores, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(num_mol, device=scores.device, dtype=order.dtype)
    positive_tensor = torch.as_tensor(positive_indices, device=scores.device, dtype=torch.long)
    positive_ranks = ranks[positive_tensor]

    for alpha in bedroc_alphas:
        alpha_key = _metric_key(alpha)
        ratio = float(num_actives) / float(num_mol)

        def log_expm1(value: float) -> float:
            if value > 50.0:
                return value + math.log1p(-math.exp(-value))
            return math.log(math.expm1(value))

        # Cancel the common exp(-alpha / N) factor analytically.  The
        # textbook expm1(alpha / N) form overflows for large alpha even though
        # the final BEDROC value remains bounded.
        scaled_ranks = -(positive_ranks.to(torch.float64) / num_mol) * alpha
        shifted_sum_exp = float(torch.exp(scaled_ranks).sum().item())
        rie = (
            shifted_sum_exp
            * num_mol
            * (-math.expm1(-alpha / num_mol))
            / (num_actives * (-math.expm1(-alpha)))
        )
        rie_max = (-math.expm1(-alpha * ratio)) / (ratio * (-math.expm1(-alpha)))
        log_rie_min = log_expm1(alpha * ratio) - math.log(ratio) - log_expm1(alpha)
        rie_min = math.exp(log_rie_min)
        bedroc = float((rie - rie_min) / (rie_max - rie_min)) if rie_max != rie_min else 1.0
        metrics[f"bedroc_{alpha_key}"] = min(1.0, max(0.0, bedroc))

    for fraction in ef_fractions:
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
        bidirectional=task.bidirectional_reactions,
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
        candidate_keys = _require_unique_ids(dataset.keys, "embedding candidate store")
        return candidate_keys, {
            "requested_candidate_count": len(dataset.keys),
            "candidate_count": len(dataset.keys),
            "missing_candidate_id_count": 0,
            "zero_length_candidate_count": 0,
        }
    candidate_keys = [protein_id for protein_id in requested_ids if protein_id in key_set]
    missing_count = len(requested_ids) - len(candidate_keys)
    if missing_count:
        missing = [protein_id for protein_id in requested_ids if protein_id not in key_set]
        raise ValueError(
            f"Candidate manifest references {missing_count} IDs absent from the embedding store: "
            f"{missing[:10]}"
        )
    return _require_unique_ids(candidate_keys, "resolved embedding candidates"), {
        "requested_candidate_count": len(requested_ids),
        "candidate_count": len(candidate_keys),
        "missing_candidate_id_count": missing_count,
        "zero_length_candidate_count": 0,
    }


def load_candidate_keys_from_residue(
    dataset: ResidueEmbedDataset,
    candidate_ids_path: Path | None,
) -> tuple[list[str], dict[str, int]]:
    length_by_key = dataset.length_by_key
    requested_ids = read_id_list(candidate_ids_path)
    key_set = set(dataset.keys)
    source_ids = list(dataset.keys) if requested_ids is None else requested_ids
    candidate_keys = [
        protein_id
        for protein_id in source_ids
        if protein_id in key_set and length_by_key.get(protein_id, 0) > 0
    ]
    missing_ids = [protein_id for protein_id in source_ids if protein_id not in length_by_key]
    zero_length_ids = [
        protein_id
        for protein_id in source_ids
        if protein_id in length_by_key and length_by_key[protein_id] <= 0
    ]
    if requested_ids is not None and (missing_ids or zero_length_ids):
        raise ValueError(
            "Candidate manifest cannot be represented exactly by residue embeddings: "
            f"missing={missing_ids[:10]}, zero_length={zero_length_ids[:10]}"
        )
    candidate_keys = _require_unique_ids(candidate_keys, "resolved residue candidates")
    return candidate_keys, {
        "requested_candidate_count": len(source_ids),
        "candidate_count": len(candidate_keys),
        "missing_candidate_id_count": len(missing_ids),
        "zero_length_candidate_count": len(zero_length_ids),
    }


def filter_candidate_keys_by_score_residue(
    candidate_keys: list[str],
    score_dataset: ResidueEmbedDataset,
) -> tuple[list[str], dict[str, int]]:
    score_length_by_key = score_dataset.length_by_key
    score_key_set = set(score_dataset.keys)
    filtered_keys = [
        protein_id
        for protein_id in candidate_keys
        if protein_id in score_key_set and score_length_by_key.get(protein_id, 0) > 0
    ]
    missing_ids = [
        protein_id for protein_id in candidate_keys if protein_id not in score_length_by_key
    ]
    zero_length_ids = [
        protein_id
        for protein_id in candidate_keys
        if protein_id in score_length_by_key and score_length_by_key[protein_id] <= 0
    ]
    if missing_ids or zero_length_ids:
        raise ValueError(
            "Score-residue store cannot represent the candidate manifest exactly: "
            f"missing={missing_ids[:10]}, zero_length={zero_length_ids[:10]}"
        )
    filtered_keys = _require_unique_ids(filtered_keys, "score-residue candidates")
    return filtered_keys, {
        "score_candidate_store_count": len(score_dataset.keys),
        "score_candidate_store_overlap": len(set(candidate_keys) & score_key_set),
        "score_missing_candidate_id_count": len(missing_ids),
        "score_zero_length_candidate_count": len(zero_length_ids),
    }


def validate_task_inputs(
    task: BenchmarkTask,
    reaction_inputs: BaseDataset,
    target_keys: list[str],
    target_store_keys: list[str],
    eval_pairs: list[tuple[str, str]],
    candidate_stats: dict[str, int],
) -> dict[str, Any]:
    _require_unique_ids(target_keys, f"task {task.name} candidates")
    _require_unique_ids(target_store_keys, f"task {task.name} target store")
    _require_unique_ids(reaction_inputs.keys, f"task {task.name} reaction inputs")
    target_key_set = set(target_keys)
    store_key_set = set(target_store_keys)
    pair_reactions = {reaction_id for reaction_id, _protein_id in eval_pairs}
    pair_proteins = {protein_id for _reaction_id, protein_id in eval_pairs}
    reaction_key_set = set(reaction_inputs.keys)
    stats: dict[str, Any] = {
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
    if task.benchmark_protocol != "custom":
        spec = LEVEL1_BENCHMARK_PROTOCOLS[task.benchmark_protocol]
        expected_pair_count = int(spec["test_pair_count"]) * (
            2 if task.bidirectional_reactions else 1
        )
        _require_protocol_value(task, "evaluation_pair_count", len(eval_pairs), expected_pair_count)
        _require_protocol_value(
            task,
            "evaluation_query_count",
            len(pair_reactions),
            spec["evaluation_query_count"],
        )
    if not target_keys:
        raise ValueError(f"Task {task.name} has no evaluable candidates")
    if stats["candidate_store_overlap"] != len(target_keys):
        raise ValueError(f"Task {task.name} candidate IDs are not fully present in target store")
    if stats["missing_reaction_count"]:
        raise ValueError(
            f"Task {task.name} has {stats['missing_reaction_count']} positive reactions "
            "without query representations"
        )
    if stats["missing_candidate_positive_count"]:
        raise ValueError(
            f"Task {task.name} has {stats['missing_candidate_positive_count']} positive proteins "
            "outside the candidate manifest"
        )
    if task.train_pairs is not None:
        train_pairs = set(
            expand_bidirectional_pairs(
                task,
                read_pairs(task.train_pairs, task.reaction_id_col, task.protein_id_col),
            )
        )
        overlap = train_pairs & set(eval_pairs)
        stats["train_eval_pair_overlap_count"] = len(overlap)
        if overlap:
            raise ValueError(
                f"Task {task.name} leaks {len(overlap)} exact positive pairs from training"
            )
    pair_tokens = [f"{reaction_id}\t{protein_id}" for reaction_id, protein_id in eval_pairs]
    split_manifest = SplitRoleManifestV2(
        split=task.split,
        pairs_sha256=sha256_strings(sorted(pair_tokens)),
        candidate_ids_sha256=sha256_strings(target_keys),
        positive_pair_count=len(eval_pairs),
        query_count=len(pair_reactions),
        candidate_count=len(target_keys),
    )
    stats["split_manifest"] = split_manifest.to_dict()
    return stats


def benchmark_artifact_inputs(
    task: BenchmarkTask,
    target_cache_base_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build strict source identities shared by benchmark results and score dumps."""

    return {
        "task": task_to_dict(task),
        "pairs": fingerprint_file(task.pairs),
        "reactions": fingerprint_file(task.reactions),
        "train_pairs": _path_fingerprint(task.train_pairs),
        "candidate_ids": _path_fingerprint(task.candidate_ids),
        "target_encoding": target_cache_base_metadata,
    }


def attach_benchmark_artifact_manifest(
    result: dict[str, Any],
    *,
    task: BenchmarkTask,
    candidate_keys: list[str],
    artifact_inputs: dict[str, Any],
    config_sha256: str,
    validate_only: bool,
) -> None:
    """Attach the v2 provenance contract to a benchmark result in-place."""

    if task.is_screening:
        direction = "reaction_to_enzyme"
    elif set(task.directions) == {"reaction_to_enzyme", "enzyme_to_reaction"}:
        direction = "both"
    elif len(task.directions) == 1:
        direction = task.directions[0]
    else:
        raise ValueError(f"Task {task.name} has unsupported directions: {task.directions}")
    result["schema_version"] = 2
    result["metric_schema_version"] = METRIC_SCHEMA_VERSION
    result["score_semantics"] = "higher_is_better_except_mean_rank"
    result["artifact_manifest"] = ArtifactManifestV2(
        artifact_type="benchmark_validation" if validate_only else "benchmark_result",
        role="validation" if validate_only else "evaluation",
        direction=direction,
        inputs=artifact_inputs,
        code_revision=current_git_revision(),
        config_sha256=config_sha256,
        candidate_ids_sha256=sha256_strings(candidate_keys),
        row_count=int(result["validation"]["pair_count"]),
    ).to_dict()


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
    return sha256_strings(values)


def _path_fingerprint(path: str | Path | None) -> dict[str, Any] | None:
    if path in {None, ""}:
        return None
    return fingerprint_file(path)


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
        "code_revision": current_git_revision(),
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
    if (
        sidecar.get("schema_version") != TARGET_CACHE_SCHEMA_VERSION
        or sidecar.get("base_metadata") != base_metadata
    ):
        return False
    try:
        manifest = ArtifactManifestV2.from_dict(sidecar.get("artifact_manifest", {}))
    except (TypeError, ValueError):
        return False
    config_fingerprint = base_metadata.get("config")
    return (
        manifest.artifact_type == "target_embedding_cache"
        and manifest.role == "candidate_store"
        and manifest.direction == base_metadata.get("retrieval_direction")
        and dict(manifest.inputs) == base_metadata
        and manifest.code_revision == base_metadata.get("code_revision")
        and isinstance(config_fingerprint, dict)
        and manifest.config_sha256 == config_fingerprint.get("sha256")
    )


def _load_target_cache_payload(
    tensor_path: Path,
    *,
    expected_num_targets: int,
    expected_base_metadata: dict[str, Any],
    expected_target_keys: list[str],
    expected_artifact_manifest: dict[str, Any],
) -> torch.Tensor:
    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "target_embeds" not in payload:
        raise ValueError(f"Target cache payload is malformed: {tensor_path}")
    if payload.get("schema_version") != TARGET_CACHE_SCHEMA_VERSION:
        raise ValueError(f"Target cache schema mismatch: {tensor_path}")
    if payload.get("base_metadata") != expected_base_metadata:
        raise ValueError(f"Target cache provenance mismatch: {tensor_path}")
    if payload.get("target_keys") != expected_target_keys:
        raise ValueError(f"Target cache candidate order mismatch: {tensor_path}")
    if payload.get("artifact_manifest") != expected_artifact_manifest:
        raise ValueError(f"Target cache manifest mismatch: {tensor_path}")
    manifest = ArtifactManifestV2.from_dict(expected_artifact_manifest)
    target_embeds = payload["target_embeds"]
    if not isinstance(target_embeds, torch.Tensor):
        raise ValueError(f"Target cache embeddings are not a tensor: {tensor_path}")
    if target_embeds.dim() != 2:
        raise ValueError(f"Target cache embeddings must be rank-2: {tensor_path}")
    if target_embeds.shape[0] != expected_num_targets:
        raise ValueError(
            f"Target cache row count mismatch for {tensor_path}: "
            f"expected {expected_num_targets}, got {target_embeds.shape[0]}"
        )
    if not torch.isfinite(target_embeds).all():
        raise ValueError(f"Target cache contains non-finite embeddings: {tensor_path}")
    expected_digest = sha256_strings(expected_target_keys)
    expected_dtype = str(target_embeds.dtype).removeprefix("torch.")
    if (
        manifest.candidate_ids_sha256 != expected_digest
        or manifest.row_count != expected_num_targets
        or manifest.shape != tuple(int(value) for value in target_embeds.shape)
        or manifest.dtype != expected_dtype
    ):
        raise ValueError(f"Target cache manifest does not describe its payload: {tensor_path}")
    return target_embeds.float().cpu()


def _cache_tensor_matches(sidecar: dict[str, Any], tensor_path: Path) -> bool:
    expected_digest = sidecar.get("tensor_sha256")
    return (
        isinstance(expected_digest, str)
        and len(expected_digest) == 64
        and tensor_path.is_file()
        and sha256_file(tensor_path) == expected_digest
    )


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
        and _cache_tensor_matches(sidecar, tensor_path)
    ):
        try:
            target_embeds = _load_target_cache_payload(
                tensor_path,
                expected_num_targets=len(target_keys),
                expected_base_metadata=base_metadata,
                expected_target_keys=target_keys,
                expected_artifact_manifest=sidecar["artifact_manifest"],
            )
        except (KeyError, TypeError, ValueError, RuntimeError):
            target_embeds = None
    else:
        target_embeds = None
    if target_embeds is not None:
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
        try:
            _require_unique_ids(cached_keys, "target cache candidate IDs")
        except ValueError:
            continue
        cached_key_set = set(cached_keys)
        if not target_key_set.issubset(cached_key_set):
            continue
        source_tensor_path = candidate_sidecar_path.with_suffix(".pt")
        if not _cache_tensor_matches(candidate_sidecar, source_tensor_path):
            continue
        try:
            source_embeds = _load_target_cache_payload(
                source_tensor_path,
                expected_num_targets=len(cached_keys),
                expected_base_metadata=base_metadata,
                expected_target_keys=cached_keys,
                expected_artifact_manifest=candidate_sidecar["artifact_manifest"],
            )
        except (KeyError, TypeError, ValueError, RuntimeError):
            continue
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


def _try_acquire_target_cache_lock(lock_path: Path) -> str | None:
    now = time.time()
    try:
        stat_result = lock_path.stat()
    except FileNotFoundError:
        stat_result = None
    owner_dead = False
    owner_live = False
    if stat_result is not None:
        try:
            owner = json.loads(lock_path.read_text(encoding="utf-8"))
            owner_pid = owner.get("pid")
            if (
                owner.get("hostname") == socket.gethostname()
                and type(owner_pid) is int
                and owner_pid > 0
            ):
                try:
                    os.kill(owner_pid, 0)
                except ProcessLookupError:
                    owner_dead = True
                except PermissionError:
                    owner_live = True
                else:
                    owner_live = True
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    if stat_result is not None and (
        owner_dead
        or (not owner_live and now - stat_result.st_mtime > TARGET_CACHE_LOCK_STALE_SECONDS)
    ):
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    lock_token = uuid.uuid4().hex
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "created_at": now,
                    "token": lock_token,
                }
            )
        )
    return lock_token


def _release_target_cache_lock(lock_path: Path, expected_token: str | None) -> None:
    """Release only the lock generation acquired by this process."""

    try:
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return
    if expected_token is None or owner.get("token") != expected_token:
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def prepare_target_embedding_cache(
    target_cache_dir: str | Path | None,
    base_metadata: dict[str, Any],
    target_keys: list[str],
    *,
    device: str,
    store_on_device: bool,
    lock_timeout_seconds: float | None = None,
) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    if target_cache_dir in {None, ""}:
        return None, {"status": "disabled"}

    _require_unique_ids(target_keys, "target cache candidate IDs")

    cache_dir = Path(target_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key, tensor_path, sidecar_path, lock_path = _target_cache_paths(
        cache_dir,
        base_metadata,
        target_keys,
    )
    if lock_timeout_seconds is None:
        lock_timeout_seconds = float(
            os.environ.get(
                "HORIZYN_TARGET_CACHE_LOCK_TIMEOUT_SECONDS",
                TARGET_CACHE_LOCK_TIMEOUT_SECONDS,
            )
        )
    if not math.isfinite(lock_timeout_seconds) or lock_timeout_seconds <= 0:
        raise ValueError("target cache lock timeout must be finite and positive")
    wait_started = time.monotonic()

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

        lock_token = _try_acquire_target_cache_lock(lock_path)
        if lock_token is not None:
            return None, {
                "status": "miss_encode",
                "cache_key": cache_key,
                "cache_path": str(tensor_path),
                "sidecar_path": str(sidecar_path),
                "lock_path": str(lock_path),
                "lock_token": lock_token,
                "num_targets": len(target_keys),
            }

        if time.monotonic() - wait_started >= lock_timeout_seconds:
            owner_text = "unreadable"
            try:
                owner_text = lock_path.read_text(encoding="utf-8")
            except OSError:
                pass
            raise TimeoutError(
                f"Timed out after {lock_timeout_seconds:g}s waiting for target cache lock "
                f"{lock_path}; owner={owner_text}"
            )

        print(f"Waiting for target embedding cache lock: {lock_path}", flush=True)
        remaining = lock_timeout_seconds - (time.monotonic() - wait_started)
        time.sleep(min(TARGET_CACHE_LOCK_POLL_SECONDS, max(0.0, remaining)))


def _write_target_embedding_cache(
    cache_info: dict[str, Any] | None,
    base_metadata: dict[str, Any],
    target_keys: list[str],
    target_embeds: torch.Tensor,
) -> None:
    if cache_info is None or cache_info.get("status") != "miss_encode":
        return

    tensor_path = Path(cache_info["cache_path"])
    sidecar_path = Path(cache_info["sidecar_path"])
    cache_dir = tensor_path.parent
    target_embeds_cpu = target_embeds.detach().float().cpu()
    if target_embeds_cpu.dim() != 2 or target_embeds_cpu.shape[0] != len(target_keys):
        raise ValueError("Target embeddings must be rank-2 and aligned one-to-one with target_keys")
    if not torch.isfinite(target_embeds_cpu).all():
        raise ValueError("Refusing to cache non-finite target embeddings")
    candidate_digest = sha256_strings(target_keys)
    config_fingerprint = base_metadata.get("config")
    if not isinstance(config_fingerprint, dict) or "sha256" not in config_fingerprint:
        raise ValueError("Target cache metadata requires a content-hashed config")
    artifact_manifest = ArtifactManifestV2(
        artifact_type="target_embedding_cache",
        role="candidate_store",
        direction=str(base_metadata.get("retrieval_direction")),
        inputs=base_metadata,
        code_revision=str(base_metadata.get("code_revision", "unavailable")),
        config_sha256=str(config_fingerprint["sha256"]),
        candidate_ids_sha256=candidate_digest,
        row_count=len(target_keys),
        shape=tuple(int(value) for value in target_embeds_cpu.shape),
        dtype=str(target_embeds_cpu.dtype).removeprefix("torch."),
    ).to_dict()
    sidecar = {
        "schema_version": TARGET_CACHE_SCHEMA_VERSION,
        "base_metadata": base_metadata,
        "target_keys_sha256": candidate_digest,
        "target_keys": target_keys,
        "num_targets": len(target_keys),
        "embedding_shape": list(target_embeds_cpu.shape),
        "tensor_file": tensor_path.name,
        "created_at": time.time(),
        "artifact_manifest": artifact_manifest,
    }
    payload = {
        "schema_version": TARGET_CACHE_SCHEMA_VERSION,
        "base_metadata": base_metadata,
        "target_keys": target_keys,
        "target_embeds": target_embeds_cpu,
        "artifact_manifest": artifact_manifest,
    }

    temp_tensor_path = cache_dir / f".{tensor_path.name}.{os.getpid()}.tmp"
    temp_sidecar_path = cache_dir / f".{sidecar_path.name}.{os.getpid()}.tmp"
    try:
        torch.save(payload, temp_tensor_path)
        sidecar["tensor_sha256"] = sha256_file(temp_tensor_path)
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


def write_target_embedding_cache(
    cache_info: dict[str, Any] | None,
    base_metadata: dict[str, Any],
    target_keys: list[str],
    target_embeds: torch.Tensor,
) -> None:
    """Atomically write a target cache and always release the owned lock."""

    try:
        _write_target_embedding_cache(cache_info, base_metadata, target_keys, target_embeds)
    finally:
        if cache_info is not None and cache_info.get("status") == "miss_encode":
            _release_target_cache_lock(Path(cache_info["lock_path"]), cache_info.get("lock_token"))


def release_target_embedding_cache_lock(cache_info: dict[str, Any] | None) -> None:
    if cache_info is None or cache_info.get("status") != "miss_encode":
        return
    lock_path = Path(cache_info["lock_path"])
    _release_target_cache_lock(lock_path, cache_info.get("lock_token"))


def evaluate_retrieval_direction(
    query_ids: list[str],
    candidate_ids: list[str],
    score_matrix: torch.Tensor,
    query_to_candidates: dict[str, list[str]],
    top_k_values: list[int] | tuple[int, ...],
    *,
    metric_protocol: str = "canonical",
) -> dict[str, float | int]:
    _require_unique_ids(query_ids, label="retrieval query IDs")
    _require_unique_ids(candidate_ids, label="retrieval candidate IDs")
    if not query_ids or not candidate_ids:
        raise ValueError("retrieval query and candidate IDs must be non-empty")
    if score_matrix.dim() != 2:
        raise ValueError(f"score_matrix must be rank-2, got shape={tuple(score_matrix.shape)}")
    expected_shape = (len(query_ids), len(candidate_ids))
    if tuple(score_matrix.shape) != expected_shape:
        raise ValueError(
            f"score_matrix shape mismatch: expected {expected_shape}, "
            f"got {tuple(score_matrix.shape)}"
        )
    if not torch.isfinite(score_matrix).all():
        raise ValueError("score_matrix contains non-finite values")
    candidate_to_idx = {candidate_id: idx for idx, candidate_id in enumerate(candidate_ids)}
    metric_rows: list[dict[str, float]] = []
    for row_idx, query_id in enumerate(query_ids):
        declared_positives = query_to_candidates.get(query_id, [])
        missing_positives = [
            candidate_id
            for candidate_id in declared_positives
            if candidate_id not in candidate_to_idx
        ]
        if missing_positives:
            raise ValueError(
                f"Query {query_id!r} has positives outside the candidate IDs: "
                f"{missing_positives[:10]}"
            )
        positives = [candidate_to_idx[candidate_id] for candidate_id in declared_positives]
        if not positives:
            raise ValueError(f"Query {query_id!r} has no declared positive candidates")
        metric_rows.append(
            rank_metrics_for_query(
                score_matrix[row_idx],
                positives,
                top_k_values=top_k_values,
                metric_protocol=metric_protocol,
            )
        )
    results: dict[str, float | int] = mean_dict(metric_rows)
    results["num_queries"] = len(metric_rows)
    results["num_targets"] = len(candidate_ids)
    return results


def evaluate_embedding_retrieval_direction(
    query_ids: list[str],
    candidate_ids: list[str],
    query_embeds: torch.Tensor,
    candidate_embeds: torch.Tensor,
    query_to_candidates: dict[str, list[str]],
    top_k_values: list[int] | tuple[int, ...],
    *,
    scoring_mode: str,
    query_batch_size: int,
    candidate_batch_size: int | None = None,
    metric_protocol: str = "canonical",
) -> dict[str, float | int]:
    """Evaluate retrieval without materializing the full query-by-candidate matrix."""

    _require_unique_ids(query_ids, label="retrieval query IDs")
    _require_unique_ids(candidate_ids, label="retrieval candidate IDs")
    if not query_ids or not candidate_ids:
        raise ValueError("retrieval query and candidate IDs must be non-empty")
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    if query_embeds.dim() != 2 or candidate_embeds.dim() != 2:
        raise ValueError("query_embeds and candidate_embeds must both be rank-2")
    if query_embeds.shape[0] != len(query_ids):
        raise ValueError("query_embeds rows must align one-to-one with query_ids")
    if candidate_embeds.shape[0] != len(candidate_ids):
        raise ValueError("candidate_embeds rows must align one-to-one with candidate_ids")
    if query_embeds.shape[1] != candidate_embeds.shape[1]:
        raise ValueError(
            "query and candidate embedding dimensions differ: "
            f"{query_embeds.shape[1]} vs {candidate_embeds.shape[1]}"
        )
    if not torch.isfinite(query_embeds).all() or not torch.isfinite(candidate_embeds).all():
        raise ValueError("retrieval embeddings contain non-finite values")

    if candidate_batch_size is None:
        candidate_batch_size = int(
            os.environ.get(
                "HORIZYN_RETRIEVAL_CANDIDATE_BATCH_SIZE",
                DEFAULT_CANDIDATE_SCORE_BATCH_SIZE,
            )
        )
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    if scoring_mode == "cosine":
        prepared_queries = l2_normalize_embeddings(query_embeds)
        prepared_candidates = l2_normalize_embeddings(candidate_embeds)
    elif scoring_mode == "dot":
        prepared_queries = query_embeds.float()
        prepared_candidates = candidate_embeds.float()
    else:
        raise ValueError(f"Unsupported scoring mode: {scoring_mode!r}")

    candidate_to_idx = {candidate_id: idx for idx, candidate_id in enumerate(candidate_ids)}
    metric_rows: list[dict[str, float]] = []
    for query_start in range(0, len(query_ids), query_batch_size):
        query_end = min(query_start + query_batch_size, len(query_ids))
        score_batch = score_prepared_embeddings_chunked(
            prepared_queries[query_start:query_end],
            prepared_candidates,
            candidate_batch_size=candidate_batch_size,
        )
        for row_idx, query_id in enumerate(query_ids[query_start:query_end]):
            declared_positives = query_to_candidates.get(query_id, [])
            missing_positives = [
                candidate_id
                for candidate_id in declared_positives
                if candidate_id not in candidate_to_idx
            ]
            if missing_positives:
                raise ValueError(
                    f"Query {query_id!r} has positives outside the candidate IDs: "
                    f"{missing_positives[:10]}"
                )
            positives = [candidate_to_idx[candidate_id] for candidate_id in declared_positives]
            if not positives:
                raise ValueError(f"Query {query_id!r} has no declared positive candidates")
            metric_rows.append(
                rank_metrics_for_query(
                    score_batch[row_idx],
                    positives,
                    top_k_values=top_k_values,
                    metric_protocol=metric_protocol,
                )
            )

    results: dict[str, float | int] = mean_dict(metric_rows)
    results["num_queries"] = len(metric_rows)
    results["num_targets"] = len(candidate_ids)
    return results


def l2_normalize_embeddings(embeds: torch.Tensor) -> torch.Tensor:
    """Return embeddings normalized for cosine retrieval scoring."""
    if embeds.dim() != 2 or embeds.shape[0] == 0 or embeds.shape[1] == 0:
        raise ValueError(f"embeddings must be a non-empty rank-2 tensor, got {embeds.shape}")
    embeds_float = embeds.float()
    if not torch.isfinite(embeds_float).all():
        raise ValueError("embeddings contain non-finite values")
    norms = torch.linalg.vector_norm(embeds_float, ord=2, dim=-1)
    if bool((norms <= COSINE_EPS).any()):
        raise ValueError("cosine scoring is undefined for zero-norm embeddings")
    return F.normalize(embeds_float, p=2, dim=-1, eps=COSINE_EPS)


def cosine_scores(query_embeds: torch.Tensor, target_embeds: torch.Tensor) -> torch.Tensor:
    """Compute cosine-similarity scores after defensive L2 normalization."""
    query_embeds = l2_normalize_embeddings(query_embeds)
    target_embeds = l2_normalize_embeddings(target_embeds)
    return torch.matmul(query_embeds, target_embeds.T)


def dot_scores(query_embeds: torch.Tensor, target_embeds: torch.Tensor) -> torch.Tensor:
    """Compute raw dot-product scores without norm correction."""
    if query_embeds.dim() != 2 or target_embeds.dim() != 2:
        raise ValueError("dot scoring expects rank-2 query and target embeddings")
    if query_embeds.shape[1] != target_embeds.shape[1]:
        raise ValueError("query and target embedding dimensions must match")
    query_float = query_embeds.float()
    target_float = target_embeds.float()
    if not torch.isfinite(query_float).all() or not torch.isfinite(target_float).all():
        raise ValueError("embeddings contain non-finite values")
    return torch.matmul(query_float, target_float.T)


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


def score_prepared_embeddings_chunked(
    query_embeds: torch.Tensor,
    target_embeds: torch.Tensor,
    *,
    candidate_batch_size: int,
) -> torch.Tensor:
    """Score prepared embeddings without moving a full CPU target store to an accelerator."""

    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    if query_embeds.dim() != 2 or target_embeds.dim() != 2:
        raise ValueError("prepared embeddings must be rank-2")
    if query_embeds.shape[0] == 0 or target_embeds.shape[0] == 0:
        raise ValueError("prepared embeddings must contain at least one row")
    if query_embeds.shape[1] != target_embeds.shape[1]:
        raise ValueError("prepared embedding dimensions must match")
    if query_embeds.device == target_embeds.device:
        return torch.matmul(query_embeds, target_embeds.T)
    score_chunks = []
    for start in range(0, target_embeds.shape[0], candidate_batch_size):
        end = min(start + candidate_batch_size, target_embeds.shape[0])
        target_chunk = target_embeds[start:end].to(query_embeds.device)
        score_chunks.append(torch.matmul(query_embeds, target_chunk.T).cpu())
    return torch.cat(score_chunks, dim=1)


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
    artifact_inputs: dict[str, Any] | None = None,
    config_sha256: str | None = None,
    metric_protocol: str = "canonical",
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
    protein_embeds = target_embeds
    e2r_protein_embeds = protein_embeds if e2r_target_embeds is None else e2r_target_embeds
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
        "metric_protocol": metric_protocol,
        "mrr_definition": (
            "mean_reciprocal_rank_over_all_positives"
            if metric_protocol == "reactzyme"
            else "reciprocal_rank_of_first_positive"
        ),
    }
    with torch.inference_mode():
        dump_root = os.environ.get("HORIZYN_RETRIEVAL_SCORE_DUMP_DIR")
        if dump_root and score_dump_task_name:
            dump_dir = Path(dump_root) / score_dump_task_name
            dump_dir.mkdir(parents=True, exist_ok=True)
            for scoring_mode in scoring_modes:
                if scoring_mode == "cosine":
                    dump_queries = l2_normalize_embeddings(e2r_protein_embeds)
                    dump_candidates = l2_normalize_embeddings(e2r_reaction_embeds)
                else:
                    dump_queries = e2r_protein_embeds.float()
                    dump_candidates = e2r_reaction_embeds.float()
                full_scores = score_prepared_embeddings_chunked(
                    dump_queries,
                    dump_candidates,
                    candidate_batch_size=DEFAULT_CANDIDATE_SCORE_BATCH_SIZE,
                )
                score_filenames = (
                    [f"scores_enzyme_by_reaction_{scoring_mode}.npz"]
                    if requested_scoring_mode == "both"
                    else ["scores_enzyme_by_reaction.npz"]
                )
                if requested_scoring_mode == "both" and scoring_mode == primary_scoring_mode:
                    score_filenames.append("scores_enzyme_by_reaction.npz")
                for score_filename in score_filenames:
                    score_path = dump_dir / score_filename
                    temporary_score_path = dump_dir / f".{score_filename}.{os.getpid()}.tmp"
                    with temporary_score_path.open("wb") as handle:
                        np.savez(
                            handle,
                            scores=full_scores.detach().float().cpu().numpy(),
                            enzyme_ids=np.asarray(candidate_keys, dtype=str),
                            reaction_ids=np.asarray(reaction_ids, dtype=str),
                        )
                    os.replace(temporary_score_path, score_path)
                    metadata_filename = (
                        "score_metadata.json"
                        if score_filename == "scores_enzyme_by_reaction.npz"
                        else f"score_metadata_{scoring_mode}.json"
                    )
                    score_meta = scoring_metadata(scoring_mode)
                    metadata: dict[str, Any] = {
                        "schema_version": 2,
                        "task": score_dump_task_name,
                        "format": "npz_dense_enzyme_by_reaction",
                        "scores": score_filename,
                        "score_sha256": sha256_file(score_path),
                        **score_meta,
                        "higher_is_better": True,
                        "shape": [len(candidate_keys), len(reaction_ids)],
                        "rows": "enzyme_ids",
                        "columns": "reaction_ids",
                    }
                    if artifact_inputs is not None and config_sha256 is not None:
                        metadata["artifact_manifest"] = ArtifactManifestV2(
                            artifact_type="retrieval_score_matrix",
                            role="benchmark_score_dump",
                            direction="enzyme_to_reaction",
                            inputs=artifact_inputs,
                            code_revision=current_git_revision(),
                            config_sha256=config_sha256,
                            candidate_ids_sha256=sha256_strings(candidate_keys),
                            score_type=score_meta["score_type"],
                            higher_is_better=True,
                            row_count=len(candidate_keys),
                            shape=(len(candidate_keys), len(reaction_ids)),
                            dtype="float32",
                        ).to_dict()
                    write_json(dump_dir / metadata_filename, metadata)

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
                metrics = evaluate_embedding_retrieval_direction(
                    query_ids=reaction_ids,
                    candidate_ids=candidate_keys,
                    query_embeds=reaction_embeds,
                    candidate_embeds=protein_embeds,
                    query_to_candidates=reaction_to_proteins,
                    top_k_values=top_k_values,
                    scoring_mode=scoring_mode,
                    query_batch_size=batch_size,
                    metric_protocol=metric_protocol,
                )
                results[f"reaction_to_enzyme{suffix}"] = metrics
                if scoring_mode == primary_scoring_mode:
                    results["reaction_to_enzyme"] = metrics

            if "enzyme_to_reaction" in directions:
                metrics = evaluate_embedding_retrieval_direction(
                    query_ids=protein_query_ids,
                    candidate_ids=reaction_ids,
                    query_embeds=e2r_protein_embeds[protein_rows],
                    candidate_embeds=e2r_reaction_embeds,
                    query_to_candidates=protein_to_reactions,
                    top_k_values=top_k_values,
                    scoring_mode=scoring_mode,
                    query_batch_size=batch_size,
                    metric_protocol=metric_protocol,
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
    screening_scoring_mode = parse_scoring_mode(
        os.environ.get("HORIZYN_SCREENING_SCORING_MODE"),
        allow_both=False,
        default="dot",
    )
    if screening_scoring_mode == "cosine":
        prepared_targets = l2_normalize_embeddings(target_embeds)
    else:
        prepared_targets = target_embeds.float()
        if not torch.isfinite(prepared_targets).all():
            raise ValueError("screening target embeddings contain non-finite values")
    candidate_batch_size = int(
        os.environ.get(
            "HORIZYN_RETRIEVAL_CANDIDATE_BATCH_SIZE",
            DEFAULT_CANDIDATE_SCORE_BATCH_SIZE,
        )
    )
    if candidate_batch_size <= 0:
        raise ValueError("HORIZYN_RETRIEVAL_CANDIDATE_BATCH_SIZE must be positive")
    metric_rows: list[dict[str, float]] = []
    with torch.inference_mode():
        for query_start in range(0, len(query_ids), batch_size):
            query_end = min(query_start + batch_size, len(query_ids))
            batch_ids = query_ids[query_start:query_end]
            query_vecs = build_query_inputs(reaction_inputs, batch_ids, device)
            query_embeds = encode_model_queries(module.model, query_vecs)
            prepared_queries = (
                l2_normalize_embeddings(query_embeds)
                if screening_scoring_mode == "cosine"
                else query_embeds.float()
            )
            score_batch = score_prepared_embeddings_chunked(
                prepared_queries,
                prepared_targets,
                candidate_batch_size=candidate_batch_size,
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

    raw_eval_pairs = read_pairs(task.pairs, task.reaction_id_col, task.protein_id_col)
    level1_source_validation = validate_level1_benchmark_sources(task, raw_eval_pairs)
    reaction_inputs = build_reaction_inputs(task, config)
    eval_pairs = expand_bidirectional_pairs(task, raw_eval_pairs)

    if kind == "pooled":
        if task.candidate_embedding_h5 is None:
            raise ValueError(f"Task {task.name} does not define candidate_embedding_h5")
        target_dataset = EmbedDataset(str(task.candidate_embedding_h5), in_memory=False)
        candidate_keys, candidate_stats = load_candidate_keys_from_embedding(
            target_dataset,
            task.candidate_ids,
        )
        candidate_keys, candidate_stats = restrict_candidates_to_test_positives(
            task, eval_pairs, candidate_keys, candidate_stats
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
        candidate_keys, candidate_stats = restrict_candidates_to_test_positives(
            task, eval_pairs, candidate_keys, candidate_stats
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
            retrieval_direction=target_retrieval_direction,
        )

    level1_candidate_validation = validate_level1_candidate_pool(task, candidate_keys)
    validation_stats = validate_task_inputs(
        task,
        reaction_inputs,
        candidate_keys,
        target_store_keys,
        eval_pairs,
        candidate_stats,
    )
    validation_stats["level1_protocol"] = {
        **level1_source_validation,
        **level1_candidate_validation,
    }
    reaction_to_proteins, protein_to_reactions = group_pairs(
        eval_pairs,
        allowed_reactions=set(reaction_inputs.keys),
        allowed_proteins=set(candidate_keys),
    )

    pool_policy = candidate_pool_policy(task, candidate_keys, eval_pairs)

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
        "candidate_pool_policy": pool_policy,
        "metric_protocol": task.metric_protocol,
        "benchmark_protocol": task.benchmark_protocol,
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

    artifact_inputs = benchmark_artifact_inputs(task, target_cache_base_metadata)
    config_fingerprint = target_cache_base_metadata.get("config")
    if not isinstance(config_fingerprint, dict) or "sha256" not in config_fingerprint:
        raise ValueError("Benchmark provenance requires a content-hashed config")
    config_sha256 = str(config_fingerprint["sha256"])

    if validate_only:
        attach_benchmark_artifact_manifest(
            result,
            task=task,
            candidate_keys=candidate_keys,
            artifact_inputs=artifact_inputs,
            config_sha256=config_sha256,
            validate_only=True,
        )
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
                artifact_inputs=artifact_inputs,
                config_sha256=config_sha256,
                metric_protocol=task.metric_protocol,
            )
    result.update(metrics)
    attach_benchmark_artifact_manifest(
        result,
        task=task,
        candidate_keys=candidate_keys,
        artifact_inputs=artifact_inputs,
        config_sha256=config_sha256,
        validate_only=False,
    )
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
    "reactzyme_mrr",
    "first_positive_mrr",
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
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, output_path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
