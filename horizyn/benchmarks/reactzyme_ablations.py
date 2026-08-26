"""Load and validate declarative ReactZyme ablation matrices."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from horizyn.benchmarks.reactzyme_protocol import REACTZYME_SPLITS


MATRIX_NAMES = ("latent", "representation", "biological", "reaction_features")
REQUIRED_VARIANT_FIELDS = frozenset(
    {
        "id",
        "label",
        "description",
        "template",
        "mode",
        "loss_name",
        "positive_pair_source",
        "hard_negative",
        "structure",
        "capability",
        "block_dims",
        "block_weights",
    }
)


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _validate_block_layout(variant: dict[str, Any], path: Path) -> None:
    dims = variant["block_dims"]
    weights = variant["block_weights"]
    if dims is None or weights is None:
        if dims is not None or weights is not None:
            raise ValueError(f"{path}: {variant['id']} must define both block maps or neither")
        return
    if not isinstance(dims, dict) or not isinstance(weights, dict):
        raise ValueError(f"{path}: {variant['id']} block layouts must be mappings")
    if set(dims) != set(weights):
        raise ValueError(f"{path}: {variant['id']} block dimension/weight names differ")
    if sum(int(value) for value in dims.values()) != 512:
        raise ValueError(f"{path}: {variant['id']} block dimensions must sum to 512")
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-6:
        raise ValueError(f"{path}: {variant['id']} block weights must sum to 1")


def load_split_specs(path: Path) -> dict[str, dict[str, str]]:
    """Load the three independent ReactZyme split template definitions."""

    payload = _load_mapping(path)
    if tuple(payload) != REACTZYME_SPLITS:
        raise ValueError(f"{path}: split order must be {REACTZYME_SPLITS}, got {tuple(payload)}")
    for split, spec in payload.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{path}: {split} specification must be a mapping")
        missing = {"official_template", "sleec_template", "tag"} - set(spec)
        if missing:
            raise ValueError(f"{path}: {split} missing fields: {sorted(missing)}")
    return copy.deepcopy(payload)


def load_variants(path: Path) -> list[dict[str, Any]]:
    """Load one matrix and reject incomplete or internally inconsistent variants."""

    payload = _load_mapping(path)
    variants = payload.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError(f"{path}: variants must be a non-empty list")

    seen_ids: set[str] = set()
    seen_labels: set[str] = set()
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"{path}: variant {index} must be a mapping")
        missing = REQUIRED_VARIANT_FIELDS - set(variant)
        if missing:
            raise ValueError(f"{path}: variant {index} missing fields: {sorted(missing)}")
        variant_id = str(variant["id"])
        label = str(variant["label"])
        if variant_id in seen_ids or label in seen_labels:
            raise ValueError(f"{path}: duplicate variant id or label: {variant_id}/{label}")
        if variant["template"] not in {"official", "sleec"}:
            raise ValueError(f"{path}: {variant_id} has unsupported template")
        for field in ("hard_negative", "structure", "capability"):
            if not isinstance(variant[field], bool):
                raise ValueError(f"{path}: {variant_id}.{field} must be boolean")
        _validate_block_layout(variant, path)
        seen_ids.add(variant_id)
        seen_labels.add(label)
    return copy.deepcopy(variants)


def load_matrix_specs(
    config_dir: Path,
    matrix: str,
) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    if matrix not in MATRIX_NAMES:
        raise ValueError(f"Unknown ReactZyme matrix: {matrix}")
    splits = load_split_specs(config_dir / "splits.yaml")
    variants = load_variants(config_dir / f"{matrix}.yaml")
    return splits, variants


__all__ = [
    "MATRIX_NAMES",
    "load_matrix_specs",
    "load_split_specs",
    "load_variants",
]
