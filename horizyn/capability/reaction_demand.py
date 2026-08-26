"""Reaction demand vector construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Iterable

import numpy as np
import pandas as pd

from horizyn.capability.io import ensure_dir, write_json


def _vocab(values: list[list[str]]) -> list[str]:
    return sorted({item for row in values for item in _as_label_list(row)})


def _as_label_list(labels: Any) -> list[str]:
    if labels is None or isinstance(labels, str):
        return []
    if isinstance(labels, Iterable):
        return [str(label) for label in labels if str(label)]
    return []


def _multihot(labels: list[str], vocab: list[str]) -> np.ndarray:
    index = {label: idx for idx, label in enumerate(vocab)}
    vec = np.zeros(len(vocab), dtype=np.float32)
    for label in _as_label_list(labels):
        idx = index.get(label)
        if idx is not None:
            vec[idx] = 1.0
    return vec


def _column_lists(features: pd.DataFrame, column: str) -> list[Any]:
    if column not in features.columns:
        return [[] for _ in range(len(features))]
    return features[column].tolist()


def build_reaction_demand_vectors(
    *,
    reaction_features_path: str | Path,
    reaction_drfp_path: str | Path,
    out_dir: str | Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    out = ensure_dir(out_dir)
    features = pd.read_parquet(reaction_features_path)
    drfp_npz = np.load(reaction_drfp_path, allow_pickle=True)
    if "vectors" not in drfp_npz:
        raise KeyError(f"{reaction_drfp_path} must contain a 'vectors' array")
    drfp = np.asarray(drfp_npz["vectors"], dtype=np.float32)
    if drfp.shape[0] != len(features):
        raise ValueError(
            f"DRFP row count {drfp.shape[0]} does not match reaction features {len(features)}"
        )

    vocab = {
        "drfp_num_bits": int(drfp.shape[1]),
        "reaction_center_labels": _vocab(_column_lists(features, "reaction_center_coarse_labels")),
        "cofactor_labels": _vocab(_column_lists(features, "cofactor_labels")),
        "core_cofactor_labels": _vocab(_column_lists(features, "core_cofactor_labels")),
        "metal_ion_labels": _vocab(_column_lists(features, "metal_ion_labels")),
        "auxiliary_participant_labels": _vocab(
            _column_lists(features, "auxiliary_participant_labels")
        ),
        "substrate_class_labels": _vocab(_column_lists(features, "substrate_class_labels")),
        "product_class_labels": _vocab(_column_lists(features, "product_class_labels")),
        "substrate_product_transition_labels": _vocab(
            _column_lists(features, "substrate_product_transition_labels")
        ),
        "reaction_type_labels": _vocab(_column_lists(features, "reaction_type_labels")),
        "ec_labels": _vocab(_column_lists(features, "ec_numbers")),
    }
    components: list[np.ndarray] = []
    slices: dict[str, list[int]] = {}
    offset = 0
    drfp_component = drfp.astype(np.float32)
    components.append(drfp_component)
    slices["drfp"] = [offset, offset + drfp_component.shape[1]]
    offset += drfp_component.shape[1]

    for column, vocab_key in (
        ("reaction_center_coarse_labels", "reaction_center_labels"),
        ("cofactor_labels", "cofactor_labels"),
        ("core_cofactor_labels", "core_cofactor_labels"),
        ("metal_ion_labels", "metal_ion_labels"),
        ("auxiliary_participant_labels", "auxiliary_participant_labels"),
        ("substrate_class_labels", "substrate_class_labels"),
        ("product_class_labels", "product_class_labels"),
        ("substrate_product_transition_labels", "substrate_product_transition_labels"),
        ("reaction_type_labels", "reaction_type_labels"),
    ):
        mat = np.stack(
            [_multihot(labels, vocab[vocab_key]) for labels in _column_lists(features, column)]
        )
        components.append(mat)
        slices[vocab_key] = [offset, offset + mat.shape[1]]
        offset += mat.shape[1]

    vectors = np.concatenate(components, axis=1).astype(np.float32)
    reaction_ids = features["reaction_id"].astype(str).to_numpy()
    np.savez_compressed(
        out / "reaction_demand_vectors.npz",
        ids=reaction_ids,
        vectors=vectors,
        component_slices=np.array([slices], dtype=object),
    )
    metadata_columns = [
        column
        for column in [
            "reaction_id",
            "ec_numbers",
            "cofactor_labels",
            "core_cofactor_labels",
            "metal_ion_labels",
            "auxiliary_participant_labels",
            "reaction_center_coarse_labels",
            "substrate_class_labels",
            "product_class_labels",
            "substrate_product_transition_labels",
            "reaction_type_labels",
            "quality_flags",
        ]
        if column in features.columns
    ]
    metadata = features[metadata_columns].copy()
    metadata["vector_index"] = np.arange(len(metadata), dtype=np.int64)
    metadata.to_parquet(out / "reaction_demand_metadata.parquet", index=False)
    payload: dict[str, Any] = dict(vocab)
    payload["component_slices"] = slices
    write_json(out / "reaction_demand_vocab.json", payload)
    return vectors, metadata
