"""
Shared reaction feature dataset builders.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from horizyn.datasets.base import BaseDataset
from horizyn.datasets.collection import MergeDataset
from horizyn.datasets.csv import CSVDataset
from horizyn.datasets.fingerprints import DRFPFingerprintDataset, RDKitPlusFingerprintDataset
from horizyn.datasets.hdf5 import EmbedDataset
from horizyn.datasets.reaction_hdf5 import UniMol2ReactionEmbedDataset
from horizyn.datasets.transform import ConcatTensorTransform


def build_bidirectional_reactions(reactions: BaseDataset) -> BaseDataset:
    augmented_keys = []
    augmented_data = []
    for rxn_id in reactions.keys:
        rxn_data = reactions[rxn_id]
        smiles = rxn_data["reaction_smiles"]
        augmented_keys.append(f"{rxn_id}_f")
        augmented_data.append({"reaction_smiles": smiles})
        if ">>" in smiles:
            parts = smiles.split(">>")
            if len(parts) == 2:
                augmented_keys.append(f"{rxn_id}_r")
                augmented_data.append({"reaction_smiles": f"{parts[1]}>>{parts[0]}"})
        else:
            augmented_keys.append(f"{rxn_id}_r")
            augmented_data.append({"reaction_smiles": smiles})
    return BaseDataset(keys=augmented_keys, array_data=augmented_data)


def build_reaction_fingerprint_dataset(
    reactions_path: str | Path,
    config,
    key_column: str = "reaction_id",
    smiles_column: str = "reaction_smiles",
    bidirectional: bool = True,
    transforms: Callable | None = None,
) -> BaseDataset:
    reactions = CSVDataset(
        file_path=str(reactions_path),
        key_column=key_column,
        columns=[smiles_column],
        rename_map={smiles_column: "reaction_smiles"},
        transforms=transforms,
    )
    if bidirectional:
        reactions = build_bidirectional_reactions(reactions)

    rdkit_fp = RDKitPlusFingerprintDataset(
        reaction_dataset=reactions,
        vec_dim=config.data.get("rdkit_fp_dim", 1024),
        mol_fp_type="morgan",
        rxn_fp_type="struct",
        use_chirality=True,
        standardize=config.data.get("standardize_reactions", True),
        standardize_hypervalent=config.data.get("standardize_hypervalent", True),
        standardize_remove_hs=config.data.get("standardize_remove_hs", True),
        standardize_kekulize=config.data.get("standardize_kekulize", False),
        standardize_uncharge=config.data.get("standardize_uncharge", True),
        standardize_metals=config.data.get("standardize_metals", True),
    )
    drfp_fp = DRFPFingerprintDataset(
        reaction_dataset=reactions,
        vec_dim=config.data.get("drfp_dim", 1024),
        radius=3,
        rings=True,
        standardize=config.data.get("standardize_reactions", True),
        standardize_hypervalent=config.data.get("standardize_hypervalent", True),
        standardize_remove_hs=config.data.get("standardize_remove_hs", True),
        standardize_kekulize=config.data.get("standardize_kekulize", False),
        standardize_uncharge=config.data.get("standardize_uncharge", True),
        standardize_metals=config.data.get("standardize_metals", True),
    )
    merged_fp = MergeDataset(
        datasets={"rdkit": rdkit_fp, "drfp": drfp_fp},
        add_prefix=False,
    )
    merged_fp.append_transforms(ConcatTensorTransform(labels=["rdkit", "drfp"], dim=0))
    return merged_fp


class HybridReactionFeatureDataset(BaseDataset[str]):
    """
    Reaction feature dataset with complete fingerprints and optional Uni-Mol2 sets.

    The dataset keeps the fingerprint key space as authoritative so reactions
    missing from the Uni-Mol2 HDF5 remain trainable/evaluable. Missing Uni-Mol2
    entries receive one zero reactant and product vector plus ``has_unimol2=False``.
    """

    def __init__(
        self,
        fingerprint_dataset: BaseDataset,
        unimol2_dataset: UniMol2ReactionEmbedDataset,
        unimol_dim: int,
        transforms: Callable | None = None,
    ):
        self.fingerprint_dataset = fingerprint_dataset
        self.unimol2_dataset = unimol2_dataset
        self.unimol_dim = unimol_dim
        self.unimol2_key_set = set(unimol2_dataset.keys)
        super().__init__(
            keys=list(fingerprint_dataset.keys),
            use_key_to_idx=True,
            transforms=transforms,
        )

    def __getitem__(self, key: str | int) -> dict[str, torch.Tensor]:
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            actual_key = self.keys[key]
        else:
            actual_key = key
            self._get_idx(actual_key)

        fingerprint = self.fingerprint_dataset[actual_key]
        sample = {
            "fingerprint": fingerprint,
            "has_unimol2": torch.tensor(actual_key in self.unimol2_key_set, dtype=torch.bool),
        }
        if actual_key in self.unimol2_key_set:
            sample.update(self.unimol2_dataset[actual_key])
        else:
            dtype = fingerprint.dtype if torch.is_tensor(fingerprint) else torch.float32
            sample.update(
                {
                    "reactant_embeddings": torch.zeros(1, self.unimol_dim, dtype=dtype),
                    "product_embeddings": torch.zeros(1, self.unimol_dim, dtype=dtype),
                }
            )
        return self._apply_transforms(actual_key, sample)


class MultimodalReactionFeatureDataset(BaseDataset[str]):
    """
    Reaction dataset for learned reaction, Uni-Mol2, and optional chirality modalities.

    The key space is the ordered intersection of the reaction CSV keys and the
    configured HDF5 stores. The chirality branch is optional so the same encoder
    can ablate it without changing the learned reaction and Uni-Mol2 inputs.
    Legacy configs may still call this branch ChiENN, while new configs should
    point it at ChIRo reaction embeddings.
    """

    def __init__(
        self,
        reaction_keys: list[str],
        reaction_embedding_dataset: EmbedDataset | None,
        unimol2_dataset: UniMol2ReactionEmbedDataset,
        chirality_dataset: UniMol2ReactionEmbedDataset | None = None,
        chemistry_dataset: "ReactionChemistryVectorDataset | None" = None,
        directional_dataset: "ReactionDirectionalVectorDataset | None" = None,
        allow_missing_unimol2: bool = False,
        allow_missing_chirality: bool = False,
        allow_missing_chemistry: bool = True,
        allow_missing_directional: bool = True,
        transforms: Callable | None = None,
    ):
        self.reaction_embedding_dataset = reaction_embedding_dataset
        self.unimol2_dataset = unimol2_dataset
        self.chirality_dataset = chirality_dataset
        self.chemistry_dataset = chemistry_dataset
        self.directional_dataset = directional_dataset
        self.allow_missing_unimol2 = allow_missing_unimol2
        self.allow_missing_chirality = allow_missing_chirality
        self.allow_missing_chemistry = allow_missing_chemistry
        self.allow_missing_directional = allow_missing_directional
        reaction_embedding_keys = (
            None if reaction_embedding_dataset is None else set(reaction_embedding_dataset.keys)
        )
        unimol2_keys = set(unimol2_dataset.keys)
        chirality_keys = set(chirality_dataset.keys) if chirality_dataset is not None else None
        keys = [
            key
            for key in reaction_keys
            if (reaction_embedding_keys is None or key in reaction_embedding_keys)
            and (allow_missing_unimol2 or key in unimol2_keys)
            and (chirality_keys is None or allow_missing_chirality or key in chirality_keys)
        ]
        if not keys:
            chirality_count = "disabled" if chirality_keys is None else len(chirality_keys)
            raise ValueError(
                "No reactions have all multimodal representations. "
                f"reaction_csv={len(reaction_keys)}, "
                "reaction_embedding="
                f"{'disabled' if reaction_embedding_keys is None else len(reaction_embedding_keys)}, "
                f"unimol2={len(unimol2_keys)}, chirality={chirality_count}"
            )
        self.num_requested_reactions = len(reaction_keys)
        self.num_filtered_reactions = len(reaction_keys) - len(keys)
        self.reaction_embedding_dim = (
            None if reaction_embedding_dataset is None else reaction_embedding_dataset.vec_dim
        )
        self.unimol_dim = unimol2_dataset.embedding_dim
        self.chirality_dim = None if chirality_dataset is None else chirality_dataset.embedding_dim
        self.chienn_dataset = chirality_dataset
        self.chienn_dim = self.chirality_dim
        self.chemistry_dim = None if chemistry_dataset is None else chemistry_dataset.vec_dim
        self.unimol2_key_set = unimol2_keys
        self.chirality_key_set = chirality_keys
        self.chemistry_key_set = (
            None if chemistry_dataset is None else set(chemistry_dataset.keys)
        )
        self.directional_dim = (
            None if directional_dataset is None else directional_dataset.vec_dim
        )
        self.directional_key_set = (
            None if directional_dataset is None else set(directional_dataset.keys)
        )
        super().__init__(
            keys=keys,
            use_key_to_idx=True,
            transforms=transforms,
        )

    def __getitem__(self, key: str | int) -> dict[str, torch.Tensor]:
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            actual_key = self.keys[key]
        else:
            actual_key = key
            self._get_idx(actual_key)

        reaction_embedding = (
            None
            if self.reaction_embedding_dataset is None
            else self.reaction_embedding_dataset[actual_key]
        )
        if actual_key in self.unimol2_key_set:
            unimol2_sample = self.unimol2_dataset[actual_key]
            has_unimol2 = True
        elif self.allow_missing_unimol2:
            dtype = reaction_embedding.dtype if torch.is_tensor(reaction_embedding) else torch.float32
            unimol2_sample = {
                "reactant_embeddings": torch.zeros(1, self.unimol_dim, dtype=dtype),
                "product_embeddings": torch.zeros(1, self.unimol_dim, dtype=dtype),
            }
            has_unimol2 = False
        else:
            raise KeyError(f"Missing Uni-Mol2 reaction embedding for {actual_key}")
        sample = {
            "reactant_embeddings": unimol2_sample["reactant_embeddings"],
            "product_embeddings": unimol2_sample["product_embeddings"],
            "has_unimol2": torch.tensor(has_unimol2, dtype=torch.bool),
        }
        if reaction_embedding is not None:
            sample["reaction_embedding"] = reaction_embedding
        if self.chirality_dataset is not None:
            if self.chirality_key_set is not None and actual_key in self.chirality_key_set:
                chirality_sample = self.chirality_dataset[actual_key]
                has_chirality = True
            elif self.allow_missing_chirality:
                dtype = reaction_embedding.dtype if torch.is_tensor(reaction_embedding) else torch.float32
                chirality_sample = {
                    "reactant_embeddings": torch.zeros(1, self.chirality_dim, dtype=dtype),
                    "product_embeddings": torch.zeros(1, self.chirality_dim, dtype=dtype),
                }
                has_chirality = False
            else:
                raise KeyError(f"Missing ChIRo/chirality reaction embedding for {actual_key}")
            sample.update(
                {
                    "reactant_chirality_embeddings": chirality_sample["reactant_embeddings"],
                    "product_chirality_embeddings": chirality_sample["product_embeddings"],
                    "has_chirality": torch.tensor(has_chirality, dtype=torch.bool),
                    "has_chiro": torch.tensor(has_chirality, dtype=torch.bool),
                    "has_chienn": torch.tensor(has_chirality, dtype=torch.bool),
                }
            )
        if self.chemistry_dataset is not None:
            chemistry_key = actual_key
            if chemistry_key not in self.chemistry_key_set and actual_key.endswith(("_f", "_r")):
                chemistry_key = actual_key[:-2]
            if chemistry_key in self.chemistry_key_set:
                chemistry_sample = self.chemistry_dataset[chemistry_key]
                has_chemistry = bool(chemistry_sample["reaction_chemistry_mask"])
                chemistry_vector = chemistry_sample["reaction_chemistry_vector"]
            elif self.allow_missing_chemistry:
                chemistry_vector = torch.zeros(
                    int(self.chemistry_dim),
                    dtype=reaction_embedding.dtype
                    if torch.is_tensor(reaction_embedding)
                    else torch.float32,
                )
                has_chemistry = False
            else:
                raise KeyError(f"Missing reaction chemistry vector for {actual_key}")
            sample.update(
                {
                    "reaction_chemistry_vector": chemistry_vector,
                    "has_reaction_chemistry": torch.tensor(
                        has_chemistry,
                        dtype=torch.bool,
                    ),
                }
            )
        if self.directional_dataset is not None:
            directional_key = actual_key
            if (
                directional_key not in self.directional_key_set
                and actual_key.endswith(("_f", "_r"))
            ):
                directional_key = actual_key[:-2]
            if directional_key in self.directional_key_set:
                directional_sample = self.directional_dataset[directional_key]
                directional_vector = directional_sample["reaction_directional_vector"]
                has_directional = bool(
                    directional_sample["reaction_directional_mask"]
                )
            elif self.allow_missing_directional:
                directional_vector = torch.zeros(
                    int(self.directional_dim),
                    dtype=(
                        reaction_embedding.dtype
                        if torch.is_tensor(reaction_embedding)
                        else torch.float32
                    ),
                )
                has_directional = False
            else:
                raise KeyError(f"Missing reaction directional vector for {actual_key}")
            sample.update(
                {
                    "reaction_directional_vector": directional_vector,
                    "has_reaction_directional": torch.tensor(
                        has_directional,
                        dtype=torch.bool,
                    ),
                }
            )
        return self._apply_transforms(actual_key, sample)


class ReactionChemistryVectorDataset(BaseDataset[str]):
    """Key-addressable reaction chemistry multihot vectors loaded from NPZ."""

    def __init__(self, file_path: str | Path, dtype: torch.dtype = torch.float32):
        payload = np.load(file_path, allow_pickle=True)
        if "ids" not in payload or "vectors" not in payload:
            raise KeyError(f"{file_path} must contain 'ids' and 'vectors'")
        ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in payload["ids"]
        ]
        vectors = torch.as_tensor(payload["vectors"], dtype=dtype)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be rank-2, got shape={tuple(vectors.shape)}")
        if vectors.shape[0] != len(ids):
            raise ValueError(
                f"vectors rows ({vectors.shape[0]}) do not match ids ({len(ids)})"
            )
        mask = (
            torch.as_tensor(payload["mask"], dtype=torch.bool)
            if "mask" in payload
            else vectors.sum(dim=1).gt(0)
        )
        if mask.shape != (len(ids),):
            raise ValueError(f"mask must have shape ({len(ids)},), got {tuple(mask.shape)}")
        self.data = vectors
        self.mask = mask
        self.vec_dim = int(vectors.shape[1])
        super().__init__(keys=ids, use_key_to_idx=True)

    def __getitem__(self, key: str | int) -> dict[str, torch.Tensor]:
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            idx = key
            actual_key = self.keys[key]
        else:
            actual_key = key
            idx = self._get_idx(actual_key)
        return self._apply_transforms(
            actual_key,
            {
                "reaction_chemistry_vector": self.data[idx],
                "reaction_chemistry_mask": self.mask[idx],
            },
        )


class ReactionDirectionalVectorDataset(BaseDataset[str]):
    """Key-addressable optional directional vectors loaded from an NPZ bundle."""

    def __init__(self, file_path: str | Path, dtype: torch.dtype = torch.float32):
        payload = np.load(file_path, allow_pickle=True)
        if "ids" not in payload or "vectors" not in payload:
            raise KeyError(f"{file_path} must contain 'ids' and 'vectors'")
        ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in payload["ids"]
        ]
        vectors = torch.as_tensor(payload["vectors"], dtype=dtype)
        if vectors.ndim != 2 or vectors.shape[0] != len(ids):
            raise ValueError(
                "Directional vectors must have shape [num_ids, dim], got "
                f"{tuple(vectors.shape)} for {len(ids)} ids"
            )
        mask = (
            torch.as_tensor(payload["mask"], dtype=torch.bool)
            if "mask" in payload
            else vectors.abs().sum(dim=1).gt(0)
        )
        if mask.shape != (len(ids),):
            raise ValueError(f"mask must have shape ({len(ids)},), got {tuple(mask.shape)}")
        self.data = vectors
        self.mask = mask
        self.vec_dim = int(vectors.shape[1])
        super().__init__(keys=ids, use_key_to_idx=True)

    def __getitem__(self, key: str | int) -> dict[str, torch.Tensor]:
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            idx = key
            actual_key = self.keys[key]
        else:
            actual_key = key
            idx = self._get_idx(actual_key)
        return self._apply_transforms(
            actual_key,
            {
                "reaction_directional_vector": self.data[idx],
                "reaction_directional_mask": self.mask[idx],
            },
        )


def _split_prefix(split_name: str | None) -> str | None:
    if split_name in {"train", "training"}:
        return "train"
    if split_name in {"validation", "val", "valid", "eval", "test"}:
        return "validation"
    return None


def _required_path(config, *names: str, split_name: str | None = None) -> str:
    prefix = _split_prefix(split_name)
    if prefix is not None:
        for name in names:
            value = config.data.get(f"{prefix}_{name}", None)
            if value:
                return value
    for name in names:
        value = config.data.get(name, None)
        if value:
            return value
    raise ValueError(
        "Missing required data config path. Provide one of: "
        + ", ".join(f"data.{name}" for name in names)
    )


def _config_get_first(config, *names: str, default=None):
    for name in names:
        value = config.data.get(name, None)
        if value is not None:
            return value
    return default


def _optional_path(config, *names: str, split_name: str | None = None) -> str | None:
    prefix = _split_prefix(split_name)
    if prefix is not None:
        for name in names:
            value = config.data.get(f"{prefix}_{name}", None)
            if value:
                return value
    for name in names:
        value = config.data.get(name, None)
        if value:
            return value
    return None


def build_multimodal_reaction_dataset(
    reactions_path: str | Path,
    config,
    key_column: str = "reaction_id",
    smiles_column: str = "reaction_smiles",
    bidirectional: bool = True,
    transforms: Callable | None = None,
    split_name: str | None = None,
) -> MultimodalReactionFeatureDataset:
    reactions = CSVDataset(
        file_path=str(reactions_path),
        key_column=key_column,
        columns=[smiles_column],
        rename_map={smiles_column: "reaction_smiles"},
        transforms=transforms,
    )
    if bidirectional:
        reactions = build_bidirectional_reactions(reactions)

    use_reaction_model = bool(config.data.get("reaction_use_model", True))
    reaction_embedding_path = (
        _required_path(
            config,
            "reaction_t5v2_embeds_path",
            "reaction_model_embeds_path",
            split_name=split_name,
        )
        if use_reaction_model
        else None
    )
    unimol2_path = _required_path(
        config,
        "reaction_unimol2_embeds_path",
        "reaction_embeds_path",
        split_name=split_name,
    )
    use_chirality = bool(
        _config_get_first(
            config,
            "reaction_use_chiro",
            "reaction_use_chirality",
            "reaction_use_chienn",
            default=True,
        )
    )
    chirality_path = (
        _required_path(
            config,
            "reaction_chiro_embeds_path",
            "reaction_chirality_embeds_path",
            "reaction_chienn_embeds_path",
            split_name=split_name,
        )
        if use_chirality
        else None
    )

    reaction_embedding_dataset = (
        EmbedDataset(
            file_path=reaction_embedding_path,
            in_memory=config.data.get("reaction_embedding_in_memory", True),
        )
        if reaction_embedding_path is not None
        else None
    )
    expected_reaction_dim = config.data.get("reaction_model_dim", None)
    if (
        reaction_embedding_dataset is not None
        and expected_reaction_dim is not None
        and reaction_embedding_dataset.vec_dim != int(expected_reaction_dim)
    ):
        raise ValueError(
            "Reaction model embedding dim mismatch: "
            f"expected {expected_reaction_dim}, got {reaction_embedding_dataset.vec_dim}"
        )

    unimol2_dataset = UniMol2ReactionEmbedDataset(
        file_path=unimol2_path,
        expected_dim=config.data.get("reaction_unimol_dim", 768),
    )
    chirality_dim = _config_get_first(
        config,
        "reaction_chiro_dim",
        "reaction_chirality_dim",
        "reaction_chienn_dim",
        default=None,
    )
    chirality_dataset = (
        UniMol2ReactionEmbedDataset(
            file_path=chirality_path,
            expected_dim=chirality_dim,
        )
        if use_chirality
        else None
    )
    chemistry_path = _optional_path(
        config,
        "reaction_chemistry_vectors_path",
        split_name=split_name,
    )
    chemistry_dataset = (
        ReactionChemistryVectorDataset(chemistry_path)
        if chemistry_path is not None
        else None
    )
    directional_path = _optional_path(
        config,
        "reaction_directional_vectors_path",
        split_name=split_name,
    )
    directional_dataset = (
        ReactionDirectionalVectorDataset(directional_path)
        if directional_path is not None
        else None
    )
    expected_chemistry_dim = config.data.get("reaction_chemistry_dim", None)
    if (
        chemistry_dataset is not None
        and expected_chemistry_dim is not None
        and chemistry_dataset.vec_dim != int(expected_chemistry_dim)
    ):
        raise ValueError(
            "Reaction chemistry vector dim mismatch: "
            f"expected {expected_chemistry_dim}, got {chemistry_dataset.vec_dim}"
        )
    expected_directional_dim = config.data.get("reaction_directional_dim", None)
    if (
        directional_dataset is not None
        and expected_directional_dim is not None
        and directional_dataset.vec_dim != int(expected_directional_dim)
    ):
        raise ValueError(
            "Reaction directional vector dim mismatch: "
            f"expected {expected_directional_dim}, got {directional_dataset.vec_dim}"
        )
    return MultimodalReactionFeatureDataset(
        reaction_keys=list(reactions.keys),
        reaction_embedding_dataset=reaction_embedding_dataset,
        unimol2_dataset=unimol2_dataset,
        chirality_dataset=chirality_dataset,
        chemistry_dataset=chemistry_dataset,
        directional_dataset=directional_dataset,
        allow_missing_unimol2=config.data.get("reaction_allow_missing_unimol2", False),
        allow_missing_chirality=_config_get_first(
            config,
            "reaction_allow_missing_chiro",
            "reaction_allow_missing_chirality",
            "reaction_allow_missing_chienn",
            default=False,
        ),
        allow_missing_chemistry=config.data.get("reaction_allow_missing_chemistry", True),
        allow_missing_directional=config.data.get(
            "reaction_allow_missing_directional",
            True,
        ),
        transforms=transforms,
    )


def build_reaction_feature_dataset(
    reactions_path: str | Path,
    config,
    key_column: str = "reaction_id",
    smiles_column: str = "reaction_smiles",
    bidirectional: bool = True,
    transforms: Callable | None = None,
    split_name: str | None = None,
) -> BaseDataset:
    representation = config.data.get("reaction_representation", "fingerprint")
    if representation == "fingerprint":
        return build_reaction_fingerprint_dataset(
            reactions_path=reactions_path,
            config=config,
            key_column=key_column,
            smiles_column=smiles_column,
            bidirectional=bidirectional,
            transforms=transforms,
        )
    if representation == "unimol2_attention":
        return UniMol2ReactionEmbedDataset(
            file_path=_required_path(config, "reaction_embeds_path", split_name=split_name),
            expected_dim=config.data.get("reaction_unimol_dim", 768),
        )
    if representation == "hybrid_fingerprint_unimol2":
        fingerprint_dataset = build_reaction_fingerprint_dataset(
            reactions_path=reactions_path,
            config=config,
            key_column=key_column,
            smiles_column=smiles_column,
            bidirectional=bidirectional,
        )
        unimol_dim = config.data.get("reaction_unimol_dim", 768)
        unimol2_dataset = UniMol2ReactionEmbedDataset(
            file_path=_required_path(config, "reaction_embeds_path", split_name=split_name),
            expected_dim=unimol_dim,
        )
        return HybridReactionFeatureDataset(
            fingerprint_dataset=fingerprint_dataset,
            unimol2_dataset=unimol2_dataset,
            unimol_dim=unimol_dim,
            transforms=transforms,
        )
    if representation == "multimodal_reaction_attention":
        return build_multimodal_reaction_dataset(
            reactions_path=reactions_path,
            config=config,
            key_column=key_column,
            smiles_column=smiles_column,
            bidirectional=bidirectional,
            transforms=transforms,
            split_name=split_name,
        )
    raise ValueError(f"Unsupported reaction_representation: {representation}")


__all__ = [
    "build_bidirectional_reactions",
    "HybridReactionFeatureDataset",
    "MultimodalReactionFeatureDataset",
    "ReactionChemistryVectorDataset",
    "build_reaction_feature_dataset",
    "build_reaction_fingerprint_dataset",
    "build_multimodal_reaction_dataset",
]
