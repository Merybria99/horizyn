"""Datasets for enzyme capability pretraining and retrieval integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Iterable

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from horizyn.capability.io import load_vector_file
from horizyn.datasets.base import BaseDataset
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset

try:
    import pandas as pd
except Exception:  # pragma: no cover - depends on runtime environment
    pd = None


class CapabilityVectorDataset(BaseDataset[str]):
    """Key-addressable static capability vectors loaded from npz/h5/pt/npy."""

    def __init__(self, file_path: str | Path, dtype: torch.dtype = torch.float32):
        ids, vectors = load_vector_file(file_path)
        self.data = vectors.to(dtype=dtype)
        self.vec_dim = int(self.data.shape[1])
        super().__init__(keys=ids, use_key_to_idx=True)

    def __getitem__(self, key: str | int) -> torch.Tensor:
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            idx = key
            actual_key = self.keys[key]
        else:
            actual_key = key
            idx = self._get_idx(actual_key)
        return self._apply_transforms(actual_key, self.data[idx])


class FactorizedCapabilityVectorDataset(BaseDataset[str]):
    """Key-addressable cofactor/center/transition capability vectors."""

    FAMILIES = ("cofactor", "center", "transition")

    def __init__(self, file_path: str | Path, dtype: torch.dtype = torch.float32):
        payload = np.load(file_path, allow_pickle=True)
        if "ids" not in payload:
            raise KeyError(f"{file_path} must contain an 'ids' array")
        ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in payload["ids"]
        ]
        self.vectors: dict[str, torch.Tensor] = {}
        self.masks: dict[str, torch.Tensor] = {}
        self.vec_dims: dict[str, int] = {}
        for family in self.FAMILIES:
            vector_key = f"{family}_vectors"
            if vector_key not in payload:
                raise KeyError(f"{file_path} must contain '{vector_key}'")
            vectors = torch.as_tensor(payload[vector_key], dtype=dtype)
            if vectors.ndim != 2:
                raise ValueError(
                    f"{vector_key} must be rank-2, got shape={tuple(vectors.shape)}"
                )
            if vectors.shape[0] != len(ids):
                raise ValueError(
                    f"{vector_key} rows ({vectors.shape[0]}) do not match ids ({len(ids)})"
                )
            mask_key = f"{family}_mask"
            mask = (
                torch.as_tensor(payload[mask_key], dtype=torch.bool)
                if mask_key in payload
                else vectors.abs().sum(dim=1).gt(0)
            )
            if mask.shape != (len(ids),):
                raise ValueError(
                    f"{mask_key} must have shape ({len(ids)},), got {tuple(mask.shape)}"
                )
            self.vectors[family] = vectors
            self.masks[family] = mask
            self.vec_dims[family] = int(vectors.shape[1])
        super().__init__(keys=ids, use_key_to_idx=True)

    def _row(self, idx: int) -> dict[str, torch.Tensor]:
        sample: dict[str, torch.Tensor] = {}
        for family in self.FAMILIES:
            sample[f"capability_{family}_vec"] = self.vectors[family][idx]
            sample[f"capability_{family}_mask"] = self.masks[family][idx]
        return sample

    def __getitem__(self, key: str | int) -> dict[str, torch.Tensor]:
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            idx = key
            actual_key = self.keys[key]
        else:
            actual_key = key
            idx = self._get_idx(actual_key)
        return self._apply_transforms(actual_key, self._row(idx))


class TextVectorDataset(BaseDataset[str]):
    """Key-addressable static text vectors loaded from npz/h5/pt/npy."""

    def __init__(self, file_path: str | Path, dtype: torch.dtype = torch.float32):
        ids, vectors = load_vector_file(file_path)
        self.data = vectors.to(dtype=dtype)
        self.vec_dim = int(self.data.shape[1])
        super().__init__(keys=ids, use_key_to_idx=True)

    def __getitem__(self, key: str | int) -> torch.Tensor:
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            idx = key
            actual_key = self.keys[key]
        else:
            actual_key = key
            idx = self._get_idx(actual_key)
        return self._apply_transforms(actual_key, self.data[idx])


class BioFPTargetDataset(BaseDataset[str]):
    """Key-addressable soft BioFP targets loaded from an NPZ artifact."""

    LEGACY_FAMILIES = ("center", "cofactor", "transition")

    def __init__(self, file_path: str | Path, dtype: torch.dtype = torch.float32):
        payload = np.load(file_path, allow_pickle=True)
        if "ids" not in payload:
            raise KeyError(f"{file_path} must contain an 'ids' array")
        ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in payload["ids"]
        ]
        self.targets: dict[str, torch.Tensor] = {}
        self.masks: dict[str, torch.Tensor] = {}
        self.denominators: dict[str, torch.Tensor] = {}
        self.confidences: dict[str, torch.Tensor] = {}
        self.target_dims: dict[str, int] = {}
        discovered = {
            key[: -len("_targets")]
            for key in payload.files
            if key.endswith("_targets") and key != "targets"
        }
        if not discovered:
            raise KeyError(f"{file_path} must contain at least one '<family>_targets' array")
        ordered = [family for family in self.LEGACY_FAMILIES if family in discovered]
        ordered.extend(sorted(discovered - set(ordered)))
        self.FAMILIES = tuple(ordered)
        for family in self.FAMILIES:
            target_key = f"{family}_targets"
            if target_key not in payload:
                raise KeyError(f"{file_path} must contain '{target_key}'")
            targets = torch.as_tensor(payload[target_key], dtype=dtype)
            if targets.ndim != 2:
                raise ValueError(
                    f"{target_key} must be rank-2, got shape={tuple(targets.shape)}"
                )
            if targets.shape[0] != len(ids):
                raise ValueError(
                    f"{target_key} rows ({targets.shape[0]}) do not match ids ({len(ids)})"
                )
            mask_key = f"{family}_mask"
            denom_key = f"{family}_denominator"
            mask = (
                torch.as_tensor(payload[mask_key], dtype=torch.bool)
                if mask_key in payload
                else targets.sum(dim=1).gt(0)
            )
            denominator = (
                torch.as_tensor(payload[denom_key], dtype=dtype)
                if denom_key in payload
                else mask.to(dtype=dtype)
            )
            confidence_key = f"{family}_confidence"
            confidence = (
                torch.as_tensor(payload[confidence_key], dtype=dtype)
                if confidence_key in payload
                else denominator.clone()
            )
            valid_shapes = {(len(ids),), tuple(targets.shape)}
            if tuple(mask.shape) not in valid_shapes:
                raise ValueError(
                    f"{mask_key} must have shape ({len(ids)},) or {tuple(targets.shape)}, "
                    f"got {tuple(mask.shape)}"
                )
            if tuple(denominator.shape) not in valid_shapes:
                raise ValueError(
                    f"{denom_key} must have shape ({len(ids)},) or {tuple(targets.shape)}, got "
                    f"{tuple(denominator.shape)}"
                )
            if tuple(confidence.shape) not in valid_shapes:
                raise ValueError(
                    f"{confidence_key} must have shape ({len(ids)},) or "
                    f"{tuple(targets.shape)}, got {tuple(confidence.shape)}"
                )
            self.targets[family] = targets
            self.masks[family] = mask
            self.denominators[family] = denominator
            self.confidences[family] = confidence
            self.target_dims[family] = int(targets.shape[1])
        super().__init__(keys=ids, use_key_to_idx=True)

    def _row(self, idx: int) -> dict[str, torch.Tensor]:
        sample: dict[str, torch.Tensor] = {}
        for family in self.FAMILIES:
            sample[f"biofp_{family}_targets"] = self.targets[family][idx]
            sample[f"biofp_{family}_mask"] = self.masks[family][idx]
            sample[f"biofp_{family}_denominator"] = self.denominators[family][idx]
            sample[f"biofp_{family}_confidence"] = self.confidences[family][idx]
        return sample

    def __getitem__(self, key: str | int) -> dict[str, torch.Tensor]:
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            idx = key
            actual_key = self.keys[key]
        else:
            actual_key = key
            idx = self._get_idx(actual_key)
        return self._apply_transforms(actual_key, self._row(idx))


class TargetWithCapabilityDataset(BaseDataset[str]):
    """Attach static capability vectors to a target dataset without dropping IDs."""

    def __init__(
        self,
        target_dataset: BaseDataset[str],
        capability_dataset: CapabilityVectorDataset,
        capability_key: str = "capability_vec",
        mask_key: str = "capability_mask",
        missing_policy: str = "zero_with_mask",
    ):
        if missing_policy != "zero_with_mask":
            raise ValueError("Only missing_policy='zero_with_mask' is supported")
        self.target_dataset = target_dataset
        self.capability_dataset = capability_dataset
        self.capability_key = capability_key
        self.mask_key = mask_key
        self.missing_policy = missing_policy
        self._capability_key_set = set(capability_dataset.keys)
        self._zero_vector = torch.zeros(capability_dataset.vec_dim, dtype=capability_dataset.data.dtype)
        super().__init__(keys=list(target_dataset.keys), use_key_to_idx=True)

    @property
    def missing_count(self) -> int:
        return len(set(self.target_dataset.keys) - self._capability_key_set)

    def __getitem__(self, key: str | int) -> dict[str, Any]:
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            actual_key = self.keys[key]
        else:
            actual_key = key
            self._get_idx(actual_key)
        target = self.target_dataset[actual_key]
        sample = dict(target) if isinstance(target, dict) else {"target_vec": target}
        if actual_key in self._capability_key_set:
            sample[self.capability_key] = self.capability_dataset[actual_key]
            sample[self.mask_key] = torch.tensor(True, dtype=torch.bool)
        else:
            sample[self.capability_key] = self._zero_vector.clone()
            sample[self.mask_key] = torch.tensor(False, dtype=torch.bool)
        return sample


class TargetWithFactorizedCapabilityDataset(BaseDataset[str]):
    """Attach factorized static capability vectors to a target dataset."""

    def __init__(
        self,
        target_dataset: BaseDataset[str],
        capability_dataset: FactorizedCapabilityVectorDataset,
        missing_policy: str = "zero_with_mask",
    ):
        if missing_policy != "zero_with_mask":
            raise ValueError("Only missing_policy='zero_with_mask' is supported")
        self.target_dataset = target_dataset
        self.capability_dataset = capability_dataset
        self.missing_policy = missing_policy
        self._capability_key_set = set(capability_dataset.keys)
        self._zero_sample = {}
        for family in capability_dataset.FAMILIES:
            self._zero_sample[f"capability_{family}_vec"] = torch.zeros(
                capability_dataset.vec_dims[family],
                dtype=capability_dataset.vectors[family].dtype,
            )
            self._zero_sample[f"capability_{family}_mask"] = torch.tensor(
                False,
                dtype=torch.bool,
            )
        super().__init__(keys=list(target_dataset.keys), use_key_to_idx=True)

    @property
    def missing_count(self) -> int:
        return len(set(self.target_dataset.keys) - self._capability_key_set)

    def __getitem__(self, key: str | int) -> dict[str, Any]:
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            actual_key = self.keys[key]
        else:
            actual_key = key
            self._get_idx(actual_key)
        target = self.target_dataset[actual_key]
        sample = dict(target) if isinstance(target, dict) else {"target_vec": target}
        if actual_key in self._capability_key_set:
            sample.update(self.capability_dataset[actual_key])
        else:
            sample.update({name: value.clone() for name, value in self._zero_sample.items()})
        return sample


class TargetWithBioFPTargetDataset(BaseDataset[str]):
    """Attach train-only soft BioFP supervision to target samples."""

    def __init__(
        self,
        target_dataset: BaseDataset[str],
        biofp_dataset: BioFPTargetDataset,
        missing_policy: str = "zero_with_mask",
    ):
        if missing_policy != "zero_with_mask":
            raise ValueError("Only missing_policy='zero_with_mask' is supported")
        self.target_dataset = target_dataset
        self.biofp_dataset = biofp_dataset
        self.missing_policy = missing_policy
        self._biofp_key_set = set(biofp_dataset.keys)
        self._zero_sample = {
            **{
                f"biofp_{family}_targets": torch.zeros(dim, dtype=biofp_dataset.targets[family].dtype)
                for family, dim in biofp_dataset.target_dims.items()
            },
            **{
                f"biofp_{family}_mask": torch.zeros(
                    dim,
                    dtype=torch.bool,
                )
                if biofp_dataset.masks[family].ndim == 2
                else torch.tensor(False, dtype=torch.bool)
                for family, dim in biofp_dataset.target_dims.items()
            },
            **{
                f"biofp_{family}_confidence": torch.zeros(
                    dim,
                    dtype=biofp_dataset.confidences[family].dtype,
                )
                if biofp_dataset.confidences[family].ndim == 2
                else torch.tensor(
                    0.0,
                    dtype=biofp_dataset.confidences[family].dtype,
                )
                for family, dim in biofp_dataset.target_dims.items()
            },
            **{
                f"biofp_{family}_denominator": torch.zeros(
                    dim,
                    dtype=biofp_dataset.denominators[family].dtype,
                )
                if biofp_dataset.denominators[family].ndim == 2
                else torch.tensor(
                    0.0,
                    dtype=biofp_dataset.denominators[family].dtype,
                )
                for family, dim in biofp_dataset.target_dims.items()
            },
        }
        super().__init__(keys=list(target_dataset.keys), use_key_to_idx=True)

    @property
    def missing_count(self) -> int:
        return len(set(self.target_dataset.keys) - self._biofp_key_set)

    def __getitem__(self, key: str | int) -> dict[str, Any]:
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            actual_key = self.keys[key]
        else:
            actual_key = key
            self._get_idx(actual_key)
        target = self.target_dataset[actual_key]
        sample = dict(target) if isinstance(target, dict) else {"target_vec": target}
        if actual_key in self._biofp_key_set:
            sample.update(self.biofp_dataset[actual_key])
        else:
            sample.update({name: value.clone() for name, value in self._zero_sample.items()})
        return sample


class TargetWithTextDataset(BaseDataset[str]):
    """Attach static protein text vectors to a target dataset without dropping IDs."""

    def __init__(
        self,
        target_dataset: BaseDataset[str],
        text_dataset: TextVectorDataset,
        text_key: str = "text_vec",
        mask_key: str = "text_mask",
        missing_policy: str = "zero_with_mask",
    ):
        if missing_policy != "zero_with_mask":
            raise ValueError("Only missing_policy='zero_with_mask' is supported")
        self.target_dataset = target_dataset
        self.text_dataset = text_dataset
        self.text_key = text_key
        self.mask_key = mask_key
        self.missing_policy = missing_policy
        self._text_key_set = set(text_dataset.keys)
        self._zero_vector = torch.zeros(text_dataset.vec_dim, dtype=text_dataset.data.dtype)
        super().__init__(keys=list(target_dataset.keys), use_key_to_idx=True)

    @property
    def missing_count(self) -> int:
        return len(set(self.target_dataset.keys) - self._text_key_set)

    def __getitem__(self, key: str | int) -> dict[str, Any]:
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            actual_key = self.keys[key]
        else:
            actual_key = key
            self._get_idx(actual_key)
        target = self.target_dataset[actual_key]
        sample = dict(target) if isinstance(target, dict) else {"target_vec": target}
        if actual_key in self._text_key_set:
            sample[self.text_key] = self.text_dataset[actual_key]
            sample[self.mask_key] = torch.tensor(True, dtype=torch.bool)
        else:
            sample[self.text_key] = self._zero_vector.clone()
            sample[self.mask_key] = torch.tensor(False, dtype=torch.bool)
        return sample


def _load_demand_npz(path: str | Path) -> tuple[list[str], torch.Tensor]:
    payload = np.load(path, allow_pickle=True)
    if "ids" not in payload or "vectors" not in payload:
        raise KeyError(f"{path} must contain 'ids' and 'vectors'")
    ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in payload["ids"]]
    vectors = torch.as_tensor(payload["vectors"], dtype=torch.float32)
    return ids, vectors


def _as_label_list(labels: Any) -> list[str]:
    if labels is None or isinstance(labels, str):
        return []
    if pd is not None:
        try:
            if bool(pd.isna(labels)):
                return []
        except (TypeError, ValueError):
            pass
    if isinstance(labels, Iterable):
        return [str(label) for label in labels if str(label)]
    return []


def _multihot(labels: list[str], vocab: list[str]) -> torch.Tensor:
    index = {label: idx for idx, label in enumerate(vocab)}
    out = torch.zeros(len(vocab), dtype=torch.float32)
    for label in _as_label_list(labels):
        idx = index.get(label)
        if idx is not None:
            out[idx] = 1.0
    return out


def _has_any_label(row: dict[str, Any], columns: tuple[str, ...]) -> bool:
    return any(_as_label_list(row.get(column, [])) for column in columns)


class EnzymeCapabilityPretrainDataset(Dataset):
    """Positive train-pair dataset for enzyme capability pretraining."""

    def __init__(
        self,
        pair_capability_training_path: str | Path,
        reaction_demand_vectors_path: str | Path,
        enzyme_capability_labels_path: str | Path,
        enzyme_feature_paths: dict[str, str | Path | None],
        enzyme_residue_path: str | Path | None = None,
        enzyme_label_vocabs: dict[str, list[str]] | None = None,
        reaction_features_path: str | Path | None = None,
        reaction_demand_metadata_path: str | Path | None = None,
        require_directional_reaction: bool = False,
        exclude_reaction_ids: set[str] | None = None,
        residue_max_tokens: int | None = None,
        residue_truncation: str = "ends_center",
        residue_in_memory: bool = False,
    ) -> None:
        if pd is None:
            raise ImportError("pandas is required for EnzymeCapabilityPretrainDataset")
        self.pairs = pd.read_parquet(pair_capability_training_path)
        if "source_split" in self.pairs.columns and set(self.pairs["source_split"].unique()) != {"train"}:
            raise AssertionError("pair_capability_training must contain only source_split='train'")
        if "is_positive" in self.pairs.columns:
            self.pairs = self.pairs[self.pairs["is_positive"].astype(int) == 1].copy()
        self.pairs["enzyme_id"] = self.pairs["enzyme_id"].astype(str)
        self.pairs["reaction_id"] = self.pairs["reaction_id"].astype(str)
        reaction_ids, demand_vectors = _load_demand_npz(reaction_demand_vectors_path)
        self.reaction_demand = {rid: demand_vectors[idx] for idx, rid in enumerate(reaction_ids)}
        self.valid_directional_reaction_ids = self._load_valid_reaction_ids(
            reaction_features_path=reaction_features_path,
            reaction_demand_metadata_path=reaction_demand_metadata_path,
        )
        if exclude_reaction_ids:
            excluded = set(str(value) for value in exclude_reaction_ids)
            self.pairs = self.pairs[~self.pairs["reaction_id"].isin(excluded)].copy()
        if require_directional_reaction:
            if not self.valid_directional_reaction_ids:
                raise ValueError(
                    "require_directional_reaction=True requires reaction_features_path "
                    "or reaction_demand_metadata_path with valid directional reaction IDs"
                )
            self.pairs = self.pairs[
                self.pairs["reaction_id"].isin(self.valid_directional_reaction_ids)
            ].copy()

        self.features: dict[str, tuple[dict[str, int], torch.Tensor]] = {}
        for name, path in enzyme_feature_paths.items():
            if path is None:
                continue
            ids, vectors = load_vector_file(path)
            self.features[name] = ({enzyme_id: idx for idx, enzyme_id in enumerate(ids)}, vectors)
        if not self.features:
            raise ValueError("At least one enzyme feature file is required")
        self.residue_dataset: ResidueEmbedDataset | None = None
        self.residue_key_set: set[str] = set()
        if enzyme_residue_path is not None:
            self.residue_dataset = ResidueEmbedDataset(
                str(enzyme_residue_path),
                in_memory=bool(residue_in_memory),
                max_tokens=residue_max_tokens,
                truncation=residue_truncation,
            )
            self.residue_key_set = set(self.residue_dataset.keys)

        labels = pd.read_parquet(enzyme_capability_labels_path)
        self.label_rows = {str(row["enzyme_id"]): row for row in labels.to_dict("records")}
        self.label_vocabs = enzyme_label_vocabs or {}
        self.reaction_family_masks = self._load_reaction_family_masks(reaction_features_path)
        self.enzyme_family_masks = self._build_enzyme_family_masks(labels)

        valid_indices = []
        for idx, row in self.pairs.iterrows():
            enzyme_id = str(row["enzyme_id"])
            reaction_id = str(row["reaction_id"])
            if reaction_id not in self.reaction_demand:
                continue
            if all(enzyme_id in mapping for mapping, _ in self.features.values()):
                if self.residue_dataset is not None and enzyme_id not in self.residue_key_set:
                    continue
                valid_indices.append(int(idx))
        if not valid_indices:
            raise ValueError("No pretraining pairs have all required reaction and enzyme features")
        self.pairs = self.pairs.loc[valid_indices].reset_index(drop=True)
        if "source_split" in self.pairs.columns:
            assert set(self.pairs["source_split"].unique()) == {"train"}
        self.reaction_to_enzymes, self.enzyme_to_reactions = self._build_positive_maps()
        self.num_masked_nondirectional_reactions = int(
            0
            if not self.valid_directional_reaction_ids
            else sum(
                reaction_id not in self.valid_directional_reaction_ids
                for reaction_id in self.pairs["reaction_id"].astype(str)
            )
        )

    @staticmethod
    def _load_reaction_family_masks(
        reaction_features_path: str | Path | None,
    ) -> dict[str, dict[str, bool]]:
        if reaction_features_path is None or not Path(reaction_features_path).exists():
            return {}
        features = pd.read_parquet(reaction_features_path)
        if "reaction_id" not in features.columns:
            return {}
        family_columns = {
            "cofactor": ("core_cofactor_labels", "cofactor_labels", "metal_ion_labels"),
            "reaction_center": (
                "reaction_center_coarse_labels",
                "reaction_center_raw_labels",
            ),
            "transition": ("substrate_product_transition_labels",),
            "substrate": ("substrate_class_labels",),
            "product": ("product_class_labels",),
        }
        out: dict[str, dict[str, bool]] = {}
        for row in features.to_dict("records"):
            reaction_id = str(row["reaction_id"])
            out[reaction_id] = {
                "global": True,
                **{
                    family: _has_any_label(row, columns)
                    for family, columns in family_columns.items()
                },
            }
        return out

    @staticmethod
    def _build_enzyme_family_masks(labels) -> dict[str, dict[str, bool]]:
        family_columns = {
            "cofactor": (
                "cofactor_architecture_bins_train",
                "cofactor_chemistry_bins_train",
                "combined_core_cofactor_labels_train",
                "core_cofactor_labels_train",
                "cofactor_labels_train",
                "metal_ion_labels_train",
            ),
            "reaction_center": ("reaction_center_labels_train",),
            "transition": ("substrate_product_transition_labels_train",),
            "substrate": ("substrate_class_labels_train",),
            "product": ("product_class_labels_train",),
        }
        out: dict[str, dict[str, bool]] = {}
        for row in labels.to_dict("records"):
            enzyme_id = str(row["enzyme_id"])
            out[enzyme_id] = {
                "global": True,
                **{
                    family: _has_any_label(row, columns)
                    for family, columns in family_columns.items()
                },
            }
        return out

    @staticmethod
    def _load_valid_reaction_ids(
        *,
        reaction_features_path: str | Path | None,
        reaction_demand_metadata_path: str | Path | None,
    ) -> set[str]:
        valid: set[str] = set()
        if reaction_features_path is not None and Path(reaction_features_path).exists():
            features = pd.read_parquet(reaction_features_path)
            if "reaction_id" not in features.columns:
                raise ValueError("reaction_features_path must contain reaction_id")
            if "canonical_reaction_smiles" in features.columns:
                smiles = features["canonical_reaction_smiles"].fillna("").astype(str)
                valid.update(features.loc[smiles.str.contains(">>", regex=False), "reaction_id"].astype(str))
        if reaction_demand_metadata_path is not None and Path(reaction_demand_metadata_path).exists():
            metadata = pd.read_parquet(reaction_demand_metadata_path)
            reaction_col = "reaction_id" if "reaction_id" in metadata.columns else None
            if reaction_col is not None:
                if "is_directional" in metadata.columns:
                    valid.update(metadata.loc[metadata["is_directional"].astype(bool), reaction_col].astype(str))
                elif "canonical_reaction_smiles" in metadata.columns:
                    smiles = metadata["canonical_reaction_smiles"].fillna("").astype(str)
                    valid.update(metadata.loc[smiles.str.contains(">>", regex=False), reaction_col].astype(str))
        return valid

    def _build_positive_maps(self) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        reaction_to_enzymes: dict[str, set[str]] = {}
        enzyme_to_reactions: dict[str, set[str]] = {}
        for row in self.pairs[["reaction_id", "enzyme_id"]].to_dict("records"):
            reaction_id = str(row["reaction_id"])
            enzyme_id = str(row["enzyme_id"])
            reaction_to_enzymes.setdefault(reaction_id, set()).add(enzyme_id)
            enzyme_to_reactions.setdefault(enzyme_id, set()).add(reaction_id)
        return reaction_to_enzymes, enzyme_to_reactions

    def __len__(self) -> int:
        return len(self.pairs)

    def _feature(self, name: str, enzyme_id: str) -> torch.Tensor:
        mapping, vectors = self.features[name]
        return vectors[mapping[enzyme_id]]

    @staticmethod
    def _target_mask(label_row: dict[str, Any], row_key: str) -> torch.Tensor:
        labels = _as_label_list(label_row.get(row_key, []))
        return torch.tensor(bool(labels), dtype=torch.bool)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.pairs.iloc[index]
        enzyme_id = str(row["enzyme_id"])
        reaction_id = str(row["reaction_id"])
        label_row = self.label_rows.get(enzyme_id, {})
        sample: dict[str, Any] = {
            "enzyme_id": enzyme_id,
            "reaction_id": reaction_id,
            "reaction_demand_vec": self.reaction_demand[reaction_id],
            "directional_reaction_mask": torch.tensor(
                not self.valid_directional_reaction_ids
                or reaction_id in self.valid_directional_reaction_ids,
                dtype=torch.bool,
            ),
        }
        reaction_family = self.reaction_family_masks.get(reaction_id, {})
        enzyme_family = self.enzyme_family_masks.get(enzyme_id, {})
        for family in ("cofactor", "reaction_center", "transition", "substrate", "product"):
            sample[f"reaction_{family}_mask"] = torch.tensor(
                bool(reaction_family.get(family, False)),
                dtype=torch.bool,
            )
            sample[f"enzyme_{family}_mask"] = torch.tensor(
                bool(enzyme_family.get(family, False)),
                dtype=torch.bool,
            )
        for name in self.features:
            sample[name] = self._feature(name, enzyme_id)
        if self.residue_dataset is not None:
            sample["prot5_residue_embeddings"] = self.residue_dataset[enzyme_id][
                "residue_embeddings"
            ]
        label_specs = {
            "cofactor_targets": ("cofactor_labels_train", "cofactor_labels"),
            "core_cofactor_targets": ("core_cofactor_labels_train", "core_cofactor_labels"),
            "enzyme_derived_cofactor_targets": (
                "enzyme_derived_cofactor_labels_train",
                "enzyme_derived_cofactor_labels",
            ),
            "enzyme_derived_core_cofactor_targets": (
                "enzyme_derived_core_cofactor_labels_train",
                "enzyme_derived_core_cofactor_labels",
            ),
            "combined_cofactor_targets": (
                "combined_cofactor_labels_train",
                "combined_cofactor_labels",
            ),
            "combined_core_cofactor_targets": (
                "combined_core_cofactor_labels_train",
                "combined_core_cofactor_labels",
            ),
            "cofactor_architecture_targets": (
                "cofactor_architecture_bins_train",
                "cofactor_architecture_bins",
            ),
            "cofactor_chemistry_targets": (
                "cofactor_chemistry_bins_train",
                "cofactor_chemistry_bins",
            ),
            "metal_ion_targets": ("metal_ion_labels_train", "metal_ion_labels"),
            "auxiliary_participant_targets": (
                "auxiliary_participant_labels_train",
                "auxiliary_participant_labels",
            ),
            "reaction_center_targets": (
                "reaction_center_labels_train",
                "reaction_center_labels",
            ),
            "substrate_targets": ("substrate_class_labels_train", "substrate_class_labels"),
            "product_targets": ("product_class_labels_train", "product_class_labels"),
            "substrate_product_transition_targets": (
                "substrate_product_transition_labels_train",
                "substrate_product_transition_labels",
            ),
            "reaction_type_targets": ("reaction_type_labels_train", "reaction_type_labels"),
            "ec_targets": ("ec_numbers_train", "ec_labels"),
        }
        for target_key, (row_key, vocab_key) in label_specs.items():
            vocab = self.label_vocabs.get(vocab_key, [])
            sample[target_key] = _multihot(label_row.get(row_key, []), vocab)
            sample[f"{target_key}_mask"] = self._target_mask(label_row, row_key)
        return sample


def enzyme_capability_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate capability samples, padding optional residue-level ProT5 tensors."""

    if not batch:
        return {}
    out: dict[str, Any] = {}
    residue_key = "prot5_residue_embeddings"
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        first = values[0]
        if key == residue_key:
            lengths = torch.tensor([int(value.shape[0]) for value in values], dtype=torch.long)
            out[key] = pad_sequence(values, batch_first=True)
            max_len = int(out[key].shape[1])
            out["prot5_residue_mask"] = (
                torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
            )
        elif torch.is_tensor(first):
            out[key] = torch.stack(values)
        else:
            out[key] = values
    return out
