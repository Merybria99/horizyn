import csv
from pathlib import Path

import h5py
import numpy as np

from horizyn.benchmarks.reactzyme_cluster_validation import (
    materialize_reaction_cluster_validation,
    select_validation_clusters,
    stable_cluster_assignments,
    threshold_graph_components,
    threshold_tag,
)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_threshold_graph_and_content_addressed_clusters_are_stable():
    reaction_ids = ["r0", "r1", "r2", "r3"]
    vectors = np.asarray(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]],
        dtype=np.float32,
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    labels = threshold_graph_components(
        vectors,
        similarity_threshold=0.80,
    )
    assignment, members = stable_cluster_assignments(reaction_ids, labels)

    assert assignment["r0"] == assignment["r1"]
    assert assignment["r2"] == assignment["r3"]
    assert assignment["r0"] != assignment["r2"]
    assert sorted(sorted(cluster) for cluster in members.values()) == [
        ["r0", "r1"],
        ["r2", "r3"],
    ]


def test_cluster_selection_is_deterministic_and_uses_pair_rows():
    clusters = {
        "large": ["r0"],
        "small_a": ["r1"],
        "small_b": ["r2"],
        "small_c": ["r3"],
    }
    pair_counts = {"r0": 90, "r1": 4, "r2": 3, "r3": 3}

    first = select_validation_clusters(
        clusters,
        pair_counts,
        validation_fraction=0.10,
        seed=42,
        threshold=0.8,
    )
    second = select_validation_clusters(
        clusters,
        pair_counts,
        validation_fraction=0.10,
        seed=42,
        threshold=0.8,
    )

    assert first == second
    assert "large" not in first
    assert sum(pair_counts[clusters[name][0]] for name in first) == 10


def test_materialized_panel_holds_complete_clusters_and_preserves_test(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    reaction_ids = [f"r{index}" for index in range(6)]
    reaction_rows = [
        {"reaction_id": reaction_id, "reaction_smiles": f"SMILES_{reaction_id}"}
        for reaction_id in reaction_ids
    ]
    pair_rows = []
    pair_id = 0
    for reaction_id, pair_count in zip(reaction_ids, [4, 3, 3, 2, 2, 2], strict=True):
        for index in range(pair_count):
            pair_rows.append(
                {
                    "pr_id": str(pair_id),
                    "reaction_id": reaction_id,
                    "protein_id": f"p_{reaction_id}_{index}",
                    "reaction_smiles": f"SMILES_{reaction_id}",
                    "protein_sequence": "AAAA",
                }
            )
            pair_id += 1
    test_pairs = [
        {
            "pr_id": "test_0",
            "reaction_id": "test_r0",
            "protein_id": "test_p0",
            "reaction_smiles": "TEST",
            "protein_sequence": "AAAA",
        }
    ]
    test_reactions = [{"reaction_id": "test_r0", "reaction_smiles": "TEST"}]
    pair_fields = list(pair_rows[0])
    reaction_fields = list(reaction_rows[0])
    _write_csv(source / "train_pairs.csv", pair_fields, pair_rows)
    _write_csv(source / "train_rxns.csv", reaction_fields, reaction_rows)
    _write_csv(source / "test_pairs.csv", pair_fields, test_pairs)
    _write_csv(source / "test_rxns.csv", reaction_fields, test_reactions)
    original_test_bytes = (source / "test_pairs.csv").read_bytes()

    chemistry = np.asarray(
        [
            [1.0, 0.00],
            [0.99, 0.01],
            [0.0, 1.00],
            [0.01, 0.99],
            [-1.0, 0.00],
            [-0.99, 0.01],
        ],
        dtype=np.float32,
    )
    chemistry_path = tmp_path / "chemistry.npz"
    np.savez(
        chemistry_path,
        ids=np.asarray(reaction_ids, dtype=str),
        vectors=chemistry,
        mask=np.ones(len(reaction_ids), dtype=bool),
    )
    reactiont5_path = tmp_path / "reactiont5.h5"
    with h5py.File(reactiont5_path, "w") as handle:
        handle.create_dataset(
            "ids",
            data=np.asarray([f"{reaction_id}_f" for reaction_id in reaction_ids], dtype="S8"),
        )
        handle.create_dataset("vectors", data=chemistry)

    output = tmp_path / "output"
    manifest = materialize_reaction_cluster_validation(
        source_dir=source,
        out_root=output,
        chemistry_npz_paths=[chemistry_path],
        reactiont5_h5_path=reactiont5_path,
        thresholds=[0.8],
        primary_threshold=0.8,
        validation_fraction=0.30,
        morgan_dims=2,
    )

    panel = output / threshold_tag(0.8)
    train_pairs = _read_csv(panel / "train_pairs.csv")
    validation_pairs = _read_csv(panel / "validation_pairs.csv")
    cluster_rows = _read_csv(panel / "reaction_clusters.csv")
    subset_by_cluster: dict[str, set[str]] = {}
    for row in cluster_rows:
        subset_by_cluster.setdefault(row["cluster_id"], set()).add(row["subset"])

    assert all(len(subsets) == 1 for subsets in subset_by_cluster.values())
    assert {row["reaction_id"] for row in train_pairs}.isdisjoint(
        {row["reaction_id"] for row in validation_pairs}
    )
    assert len(train_pairs) + len(validation_pairs) == len(pair_rows)
    assert (panel / "test_pairs.csv").read_bytes() == original_test_bytes
    assert manifest["panels"][0]["audits"]["train_validation_cluster_overlap"] == 0
    assert manifest["panels"][0]["primary"] is True
