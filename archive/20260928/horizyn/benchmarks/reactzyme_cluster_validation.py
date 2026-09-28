"""Reaction-cluster-held-out validation panels for ReactZyme Reaction-Sim."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np

from horizyn.benchmarks.reactzyme_protocol import (
    PAIR_REQUIRED_FIELDS,
    read_csv_rows,
    reaction_rows_for_pairs,
    sha256,
    unique_protein_ids,
    write_csv_rows,
    write_ids,
)


SCHEMA_VERSION = "reactzyme_reaction_cluster_validation_v1"
DEFAULT_THRESHOLDS = (0.80, 0.85, 0.90)
DEFAULT_PRIMARY_THRESHOLD = 0.85
DEFAULT_MORGAN_WEIGHT = 0.70
DEFAULT_REACTIONT5_WEIGHT = 0.30


def threshold_tag(threshold: float) -> str:
    """Return a path-safe, stable threshold label."""

    return f"similarity_{threshold:.2f}".replace(".", "p")


def _decode_id(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _l2_normalize(vectors: np.ndarray, *, name: str) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    bad = np.flatnonzero(norms[:, 0] <= 0)
    if len(bad):
        raise ValueError(f"{name} contains {len(bad)} zero-norm rows")
    return vectors / norms


def load_npz_vector_union(
    paths: Sequence[Path],
    reaction_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load checkpoint-compatible chemistry vectors from one or more NPZ files."""

    vectors_by_id: dict[str, np.ndarray] = {}
    masks_by_id: dict[str, bool] = {}
    vector_dim: int | None = None
    for path in paths:
        with np.load(path, allow_pickle=True) as payload:
            if "ids" not in payload or "vectors" not in payload:
                raise KeyError(f"{path} must contain ids and vectors")
            ids = [_decode_id(value) for value in payload["ids"]]
            vectors = np.asarray(payload["vectors"], dtype=np.float32)
            masks = (
                np.asarray(payload["mask"], dtype=bool)
                if "mask" in payload
                else np.ones(len(ids), dtype=bool)
            )
        if vectors.ndim != 2 or vectors.shape[0] != len(ids):
            raise ValueError(f"Malformed vector matrix in {path}: {vectors.shape}")
        if masks.shape != (len(ids),):
            raise ValueError(f"Malformed mask in {path}: {masks.shape}")
        if vector_dim is None:
            vector_dim = int(vectors.shape[1])
        elif vectors.shape[1] != vector_dim:
            raise ValueError(f"Vector dimension mismatch in {path}")
        for reaction_id, vector, mask in zip(ids, vectors, masks, strict=True):
            if reaction_id in vectors_by_id and not np.array_equal(
                vectors_by_id[reaction_id], vector
            ):
                raise ValueError(f"Conflicting vectors for {reaction_id}")
            vectors_by_id[reaction_id] = vector
            masks_by_id[reaction_id] = bool(mask)

    missing = [reaction_id for reaction_id in reaction_ids if reaction_id not in vectors_by_id]
    if missing:
        raise ValueError(
            f"Chemistry artifacts miss {len(missing)} reactions; examples: {missing[:5]}"
        )
    invalid = [reaction_id for reaction_id in reaction_ids if not masks_by_id[reaction_id]]
    if invalid:
        raise ValueError(
            f"Chemistry artifacts mark {len(invalid)} reactions invalid; examples: {invalid[:5]}"
        )
    ordered = np.stack([vectors_by_id[reaction_id] for reaction_id in reaction_ids])
    ordered_mask = np.asarray([masks_by_id[reaction_id] for reaction_id in reaction_ids])
    return ordered, ordered_mask


def load_forward_h5_vectors(path: Path, reaction_ids: Sequence[str]) -> np.ndarray:
    """Load frozen forward ReactionT5 vectors in the requested reaction order."""

    with h5py.File(path, "r") as handle:
        ids = [_decode_id(value) for value in handle["ids"][:]]
        vectors = np.asarray(handle["vectors"][:], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(ids):
        raise ValueError(f"Malformed HDF5 vector matrix in {path}: {vectors.shape}")
    index = {reaction_id: row for row, reaction_id in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError(f"Duplicate feature IDs in {path}")
    requested = [f"{reaction_id}_f" for reaction_id in reaction_ids]
    missing = [reaction_id for reaction_id in requested if reaction_id not in index]
    if missing:
        raise ValueError(
            f"ReactionT5 artifact misses {len(missing)} forward IDs; examples: {missing[:5]}"
        )
    return vectors[[index[reaction_id] for reaction_id in requested]]


def build_hybrid_features(
    chemistry_vectors: np.ndarray,
    reactiont5_vectors: np.ndarray,
    *,
    morgan_dims: int,
    morgan_weight: float,
    reactiont5_weight: float,
) -> np.ndarray:
    """Combine molecule-set Morgan and frozen ReactionT5 cosine features."""

    if chemistry_vectors.shape[0] != reactiont5_vectors.shape[0]:
        raise ValueError("Chemistry and ReactionT5 row counts differ")
    if not 0 < morgan_dims <= chemistry_vectors.shape[1]:
        raise ValueError("morgan_dims must fit inside the chemistry vector width")
    if morgan_weight <= 0 or reactiont5_weight <= 0:
        raise ValueError("Hybrid feature weights must be positive")
    total_weight = morgan_weight + reactiont5_weight
    morgan_weight /= total_weight
    reactiont5_weight /= total_weight
    morgan = _l2_normalize(
        chemistry_vectors[:, :morgan_dims], name="Morgan molecule-set features"
    )
    reactiont5 = _l2_normalize(reactiont5_vectors, name="ReactionT5 features")
    hybrid = np.concatenate(
        (
            math.sqrt(morgan_weight) * morgan,
            math.sqrt(reactiont5_weight) * reactiont5,
        ),
        axis=1,
    )
    return _l2_normalize(hybrid, name="hybrid reaction features")


def threshold_graph_components(
    vectors: np.ndarray,
    *,
    similarity_threshold: float,
    block_size: int = 256,
) -> np.ndarray:
    """Find components joined by any cosine edge at or above the threshold."""

    if not 0.0 < similarity_threshold < 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")
    if vectors.ndim != 2 or vectors.shape[0] < 2:
        raise ValueError("At least two rank-2 reaction vectors are required")
    num_reactions = vectors.shape[0]
    parent = np.arange(num_reactions, dtype=np.int64)
    sizes = np.ones(num_reactions, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        sizes[left_root] += sizes[right_root]

    for start in range(0, num_reactions, block_size):
        stop = min(start + block_size, num_reactions)
        similarities = vectors[start:stop] @ vectors.T
        for local_index, reaction_index in enumerate(range(start, stop)):
            neighbors = np.flatnonzero(
                similarities[local_index, reaction_index + 1 :] >= similarity_threshold
            )
            neighbors += reaction_index + 1
            for neighbor in neighbors:
                union(reaction_index, int(neighbor))
    return np.asarray([find(index) for index in range(num_reactions)], dtype=np.int64)


def stable_cluster_assignments(
    reaction_ids: Sequence[str], labels: np.ndarray
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Convert implementation-specific integer labels to content-addressed IDs."""

    if labels.shape != (len(reaction_ids),):
        raise ValueError("Cluster label count does not match reaction IDs")
    members_by_label: dict[int, list[str]] = defaultdict(list)
    for reaction_id, label in zip(reaction_ids, labels, strict=True):
        members_by_label[int(label)].append(reaction_id)
    members_by_cluster: dict[str, list[str]] = {}
    assignment: dict[str, str] = {}
    for members in members_by_label.values():
        members.sort()
        digest = hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest()[:16]
        cluster_id = f"cluster_{digest}"
        members_by_cluster[cluster_id] = members
        assignment.update({reaction_id: cluster_id for reaction_id in members})
    return assignment, members_by_cluster


def select_validation_clusters(
    members_by_cluster: dict[str, list[str]],
    pair_count_by_reaction: dict[str, int],
    *,
    validation_fraction: float,
    seed: int,
    threshold: float,
) -> set[str]:
    """Select complete clusters near a target fraction of pair rows."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(members_by_cluster) < 2:
        raise ValueError("A held-out panel requires at least two reaction clusters")
    cluster_weights = {
        cluster_id: sum(pair_count_by_reaction[reaction_id] for reaction_id in members)
        for cluster_id, members in members_by_cluster.items()
    }
    total = sum(cluster_weights.values())
    target = validation_fraction * total
    ordered = sorted(
        cluster_weights,
        key=lambda cluster_id: hashlib.sha256(
            f"{seed}:{threshold:.8f}:{cluster_id}".encode("utf-8")
        ).hexdigest(),
    )
    selected: set[str] = set()
    selected_count = 0
    for cluster_id in ordered:
        candidate_count = selected_count + cluster_weights[cluster_id]
        if abs(target - candidate_count) < abs(target - selected_count):
            selected.add(cluster_id)
            selected_count = candidate_count
    if not selected:
        selected.add(min(ordered, key=lambda cluster_id: cluster_weights[cluster_id]))
    if len(selected) == len(members_by_cluster):
        selected.remove(
            min(selected, key=lambda cluster_id: cluster_weights[cluster_id])
        )
    return selected


def cross_split_similarity_audit(
    vectors: np.ndarray,
    validation_indices: Sequence[int],
    train_indices: Sequence[int],
    *,
    threshold: float,
    block_size: int = 512,
) -> dict[str, float]:
    """Summarize each held-out reaction's nearest training reaction."""

    validation = vectors[np.asarray(validation_indices)]
    train = vectors[np.asarray(train_indices)]
    nearest = np.empty(len(validation), dtype=np.float32)
    for start in range(0, len(validation), block_size):
        stop = min(start + block_size, len(validation))
        nearest[start:stop] = np.max(validation[start:stop] @ train.T, axis=1)
    return {
        "minimum": float(np.min(nearest)),
        "median": float(np.median(nearest)),
        "p90": float(np.quantile(nearest, 0.90)),
        "p95": float(np.quantile(nearest, 0.95)),
        "maximum": float(np.max(nearest)),
        "fraction_at_or_above_cut_similarity": float(np.mean(nearest >= threshold)),
    }


def _pair_key(row: dict[str, str]) -> tuple[str, str]:
    return row["reaction_id"], row["protein_id"]


def _write_cluster_tables(
    panel_dir: Path,
    *,
    reaction_ids: Sequence[str],
    assignment: dict[str, str],
    held_clusters: set[str],
    members_by_cluster: dict[str, list[str]],
    pair_count_by_reaction: dict[str, int],
) -> None:
    reaction_rows = [
        {
            "reaction_id": reaction_id,
            "cluster_id": assignment[reaction_id],
            "subset": "validation" if assignment[reaction_id] in held_clusters else "train",
            "pair_count": str(pair_count_by_reaction[reaction_id]),
        }
        for reaction_id in reaction_ids
    ]
    write_csv_rows(
        panel_dir / "reaction_clusters.csv",
        ("reaction_id", "cluster_id", "subset", "pair_count"),
        reaction_rows,
    )
    summary_rows = []
    for cluster_id, members in sorted(members_by_cluster.items()):
        summary_rows.append(
            {
                "cluster_id": cluster_id,
                "subset": "validation" if cluster_id in held_clusters else "train",
                "reaction_count": str(len(members)),
                "pair_count": str(
                    sum(pair_count_by_reaction[reaction_id] for reaction_id in members)
                ),
            }
        )
    write_csv_rows(
        panel_dir / "cluster_summary.csv",
        ("cluster_id", "subset", "reaction_count", "pair_count"),
        summary_rows,
    )


def _materialize_panel(
    *,
    source_dir: Path,
    out_root: Path,
    reaction_ids: Sequence[str],
    source_pair_fields: Sequence[str],
    source_pairs: Sequence[dict[str, str]],
    reaction_fields: Sequence[str],
    source_reactions: Sequence[dict[str, str]],
    test_pairs: Sequence[dict[str, str]],
    vectors: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    validation_fraction: float,
    seed: int,
    primary_threshold: float,
) -> dict[str, Any]:
    assignment, members_by_cluster = stable_cluster_assignments(reaction_ids, labels)
    pair_count_by_reaction = Counter(row["reaction_id"] for row in source_pairs)
    held_clusters = select_validation_clusters(
        members_by_cluster,
        pair_count_by_reaction,
        validation_fraction=validation_fraction,
        seed=seed,
        threshold=threshold,
    )
    validation_ids = {
        reaction_id
        for reaction_id, cluster_id in assignment.items()
        if cluster_id in held_clusters
    }
    train_pairs = [row for row in source_pairs if row["reaction_id"] not in validation_ids]
    validation_pairs = [row for row in source_pairs if row["reaction_id"] in validation_ids]
    train_reactions = reaction_rows_for_pairs(source_reactions, train_pairs)
    validation_reactions = reaction_rows_for_pairs(source_reactions, validation_pairs)

    panel_dir = out_root / threshold_tag(threshold)
    panel_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(panel_dir / "train_pairs.csv", source_pair_fields, train_pairs)
    write_csv_rows(panel_dir / "validation_pairs.csv", source_pair_fields, validation_pairs)
    write_csv_rows(panel_dir / "train_rxns.csv", reaction_fields, train_reactions)
    write_csv_rows(panel_dir / "validation_rxns.csv", reaction_fields, validation_reactions)
    for name in ("test_pairs.csv", "test_rxns.csv"):
        shutil.copy2(source_dir / name, panel_dir / name)
    for subset, rows in (
        ("train", train_pairs),
        ("validation", validation_pairs),
        ("test", test_pairs),
    ):
        write_ids(panel_dir / f"{subset}_candidate_ids.txt", unique_protein_ids(rows))
    _write_cluster_tables(
        panel_dir,
        reaction_ids=reaction_ids,
        assignment=assignment,
        held_clusters=held_clusters,
        members_by_cluster=members_by_cluster,
        pair_count_by_reaction=pair_count_by_reaction,
    )

    train_ids = {row["reaction_id"] for row in train_pairs}
    train_clusters = {assignment[reaction_id] for reaction_id in train_ids}
    validation_clusters = {assignment[reaction_id] for reaction_id in validation_ids}
    train_smiles = {row["reaction_smiles"] for row in train_pairs}
    validation_smiles = {row["reaction_smiles"] for row in validation_pairs}
    train_keys = {_pair_key(row) for row in train_pairs}
    validation_keys = {_pair_key(row) for row in validation_pairs}
    source_keys = {_pair_key(row) for row in source_pairs}
    if train_ids & validation_ids:
        raise ValueError("Reaction ID leakage across cluster split")
    if train_clusters & validation_clusters:
        raise ValueError("Reaction cluster leakage across cluster split")
    if train_smiles & validation_smiles:
        raise ValueError("Exact reaction SMILES leakage across cluster split")
    if train_keys & validation_keys:
        raise ValueError("Exact pair leakage across cluster split")
    if train_keys | validation_keys != source_keys:
        raise ValueError("Cluster split does not reconstruct source pair keys")
    if len(train_pairs) + len(validation_pairs) != len(source_pairs):
        raise ValueError("Cluster split does not reconstruct source pair rows")
    for name in ("test_pairs.csv", "test_rxns.csv"):
        if sha256(source_dir / name) != sha256(panel_dir / name):
            raise ValueError(f"Released test file changed while copying: {name}")

    index_by_id = {reaction_id: index for index, reaction_id in enumerate(reaction_ids)}
    train_indices = [index_by_id[reaction_id] for reaction_id in train_ids]
    validation_indices = [index_by_id[reaction_id] for reaction_id in validation_ids]
    cluster_sizes = [len(members) for members in members_by_cluster.values()]
    similarity_audit = cross_split_similarity_audit(
        vectors,
        validation_indices,
        train_indices,
        threshold=threshold,
    )
    if similarity_audit["fraction_at_or_above_cut_similarity"] != 0.0:
        raise ValueError(
            "Threshold-graph split leaked a validation-to-train similarity edge"
        )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "panel": threshold_tag(threshold),
        "primary": math.isclose(threshold, primary_threshold),
        "similarity_threshold": threshold,
        "graph_edge_cut": f"cosine_similarity >= {threshold}",
        "clustering": "connected_components",
        "metric": "cosine",
        "validation_fraction_target": validation_fraction,
        "seed": seed,
        "candidate_pool_protocol": (
            "validation reactions and all unique proteins occurring in validation pairs"
        ),
        "counts": {
            "source_train_pairs": len(source_pairs),
            "train_pairs": len(train_pairs),
            "validation_pairs": len(validation_pairs),
            "validation_pair_fraction": len(validation_pairs) / len(source_pairs),
            "train_reactions": len(train_ids),
            "validation_reactions": len(validation_ids),
            "clusters": len(members_by_cluster),
            "train_clusters": len(train_clusters),
            "validation_clusters": len(validation_clusters),
            "singleton_clusters": sum(size == 1 for size in cluster_sizes),
            "largest_cluster_reactions": max(cluster_sizes),
            "train_candidates": len(unique_protein_ids(train_pairs)),
            "validation_candidates": len(unique_protein_ids(validation_pairs)),
            "test_candidates": len(unique_protein_ids(test_pairs)),
        },
        "audits": {
            "train_validation_reaction_id_overlap": 0,
            "train_validation_cluster_overlap": 0,
            "train_validation_reaction_smiles_overlap": 0,
            "train_validation_exact_pair_overlap": 0,
            "source_pair_keys_reconstructed": train_keys | validation_keys == source_keys,
            "source_pair_rows_reconstructed": (
                len(train_pairs) + len(validation_pairs) == len(source_pairs)
            ),
            "test_files_byte_identical_to_release": True,
            "nearest_train_hybrid_cosine": similarity_audit,
        },
    }
    (panel_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def materialize_reaction_cluster_validation(
    *,
    source_dir: Path,
    out_root: Path,
    chemistry_npz_paths: Sequence[Path],
    reactiont5_h5_path: Path,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    primary_threshold: float = DEFAULT_PRIMARY_THRESHOLD,
    validation_fraction: float = 0.10,
    seed: int = 42,
    morgan_dims: int = 512,
    morgan_weight: float = DEFAULT_MORGAN_WEIGHT,
    reactiont5_weight: float = DEFAULT_REACTIONT5_WEIGHT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build all Reaction-Sim cluster-held-out validation panels."""

    thresholds = tuple(sorted(set(float(value) for value in thresholds)))
    if not thresholds:
        raise ValueError("At least one similarity threshold is required")
    if primary_threshold not in thresholds:
        raise ValueError("primary_threshold must be included in thresholds")
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    pair_fields, source_pairs = read_csv_rows(source_dir / "train_pairs.csv")
    reaction_fields, source_reactions = read_csv_rows(source_dir / "train_rxns.csv")
    _, test_pairs = read_csv_rows(source_dir / "test_pairs.csv")
    missing_fields = PAIR_REQUIRED_FIELDS - set(pair_fields)
    if missing_fields:
        raise ValueError(f"Source pairs miss fields: {sorted(missing_fields)}")
    reaction_ids = [row["reaction_id"] for row in source_reactions]
    if len(reaction_ids) != len(set(reaction_ids)):
        raise ValueError("Source train_rxns.csv contains duplicate reaction IDs")
    pair_reaction_ids = {row["reaction_id"] for row in source_pairs}
    if pair_reaction_ids != set(reaction_ids):
        raise ValueError("Source pair and reaction tables have different reaction IDs")

    chemistry_vectors, chemistry_mask = load_npz_vector_union(
        chemistry_npz_paths, reaction_ids
    )
    reactiont5_vectors = load_forward_h5_vectors(reactiont5_h5_path, reaction_ids)
    hybrid = build_hybrid_features(
        chemistry_vectors,
        reactiont5_vectors,
        morgan_dims=morgan_dims,
        morgan_weight=morgan_weight,
        reactiont5_weight=reactiont5_weight,
    )
    labels_by_threshold = {
        threshold: threshold_graph_components(
            hybrid,
            similarity_threshold=threshold,
        )
        for threshold in thresholds
    }

    artifact_dir = out_root / "clustering"
    artifact_dir.mkdir()
    np.savez_compressed(
        artifact_dir / "hybrid_features.npz",
        ids=np.asarray(reaction_ids, dtype=str),
        vectors=hybrid,
    )
    np.savez_compressed(
        artifact_dir / "threshold_graph_components.npz",
        **{
            f"labels_{threshold_tag(threshold)}": labels
            for threshold, labels in labels_by_threshold.items()
        },
    )
    panels = [
        _materialize_panel(
            source_dir=source_dir,
            out_root=out_root,
            reaction_ids=reaction_ids,
            source_pair_fields=pair_fields,
            source_pairs=source_pairs,
            reaction_fields=reaction_fields,
            source_reactions=source_reactions,
            test_pairs=test_pairs,
            vectors=hybrid,
            labels=labels_by_threshold[threshold],
            threshold=threshold,
            validation_fraction=validation_fraction,
            seed=seed,
            primary_threshold=primary_threshold,
        )
        for threshold in thresholds
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_dir": str(source_dir.resolve()),
        "source_train_only": True,
        "thresholds": list(thresholds),
        "primary_threshold": primary_threshold,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "representation": {
            "description": (
                "L2-normalized train-fitted Morgan molecule-set block concatenated "
                "with frozen forward ReactionT5v2; block weights use square-root scaling"
            ),
            "morgan_dims": morgan_dims,
            "morgan_weight": morgan_weight,
            "reactiont5_weight": reactiont5_weight,
            "hybrid_dim": int(hybrid.shape[1]),
            "chemistry_npz_paths": [str(path.resolve()) for path in chemistry_npz_paths],
            "reactiont5_h5_path": str(reactiont5_h5_path.resolve()),
            "all_chemistry_masks_valid": bool(chemistry_mask.all()),
        },
        "clustering": {
            "algorithm": "threshold_graph_connected_components",
            "metric": "cosine",
            "edge_rule": "connect reactions when cosine similarity >= threshold",
            "fit_scope": "released Reaction-Sim training reactions only",
        },
        "source_sha256": {
            name: sha256(source_dir / name)
            for name in ("train_pairs.csv", "train_rxns.csv", "test_pairs.csv", "test_rxns.csv")
        },
        "feature_sha256": {
            "chemistry_npz": {str(path.resolve()): sha256(path) for path in chemistry_npz_paths},
            "reactiont5_h5": sha256(reactiont5_h5_path),
        },
        "panels": panels,
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_root / "README.md").write_text(
        """# ReactZyme Reaction-Cluster Validation v1

This protocol derives validation data only from the released Reaction-Sim training
split. It clusters complete reactions as connected components of a cosine-threshold
graph over a hybrid of the 512-bit F3 Morgan molecule-set block and frozen forward
ReactionT5v2 features.
No released test row or test feature participates in fitting or cluster selection.

The ReactZyme paper describes Needleman-Wunsch reaction-SMILES similarity, but the
local release does not contain its train similarity matrix or cluster assignments.
The hybrid representation is therefore the documented stable fallback, not a claim
to reproduce unavailable benchmark clustering metadata.

`similarity_0p85` is the primary checkpoint-selection panel. The 0.80 and 0.90
panels are sensitivity checks. Every pair belonging to a selected reaction cluster
is held out, and each panel uses its validation reactions and unique validation
proteins as the two candidate pools. The released test CSV files are copied
byte-for-byte for provenance only and remain excluded from validation.

Connected components enforce that no validation-to-training reaction edge reaches
the panel's similarity threshold. The 0.80 panel is intentionally smaller than 10%
because its giant component leaves too few leakage-free pair rows; its manifest
records the attainable fraction. Each panel also reports nearest-train similarities.
""",
        encoding="utf-8",
    )
    return manifest
