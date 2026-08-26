"""
Lightning DataModule for Horizyn contrastive learning.

This module loads pre-split datasets and creates dataloaders for training
and validation. All data is loaded into memory at setup time.
"""

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset

from horizyn.config import DotDict
from horizyn.capability.enzyme_capability_dataset import (
    BioFPTargetDataset,
    CapabilityVectorDataset,
    FactorizedCapabilityVectorDataset,
    TargetWithBioFPTargetDataset,
    TargetWithCapabilityDataset,
    TargetWithFactorizedCapabilityDataset,
    TargetWithTextDataset,
    TextVectorDataset,
)
from horizyn.datasets.base import BaseDataset
from horizyn.datasets.collection import MergeDataset, TupleDataset
from horizyn.datasets.csv import CSVDataset
from horizyn.datasets.fingerprints import (
    DRFPFingerprintDataset,
    RDKitPlusFingerprintDataset,
)
from horizyn.datasets.hdf5 import EmbedDataset
from horizyn.datasets.reaction_hdf5 import UniMol2ReactionEmbedDataset
from horizyn.datasets.transform import ConcatTensorTransform
from horizyn.reaction_features import build_reaction_feature_dataset
from horizyn.utils import default_collate, dict_collate_fn


_EC_SPLIT_PATTERN = re.compile(r"[;,|]\s*")


def normalize_ec_number(ec_number: str) -> str | None:
    """Normalize one EC string and drop invalid empty labels."""

    ec_number = str(ec_number).strip()
    if not ec_number or ec_number.lower() in {"nan", "none", "null"}:
        return None
    parts = [part.strip() for part in ec_number.split(".")]
    if not parts:
        return None
    normalized_parts = []
    for part in parts[:4]:
        if not part or part == "-":
            break
        normalized_parts.append(part)
    if not normalized_parts:
        return None
    return ".".join(normalized_parts)


def split_ec_numbers(ec_numbers: str) -> list[str]:
    """Split a potentially multi-EC field into normalized EC labels."""

    values = []
    for raw_ec in _EC_SPLIT_PATTERN.split(str(ec_numbers)):
        normalized = normalize_ec_number(raw_ec)
        if normalized is not None:
            values.append(normalized)
    return values


def ec_shared_depth(first: str, second: str) -> int:
    """Return the deepest shared EC prefix depth for two normalized EC labels."""

    first_parts = first.split(".")
    second_parts = second.split(".")
    depth = 0
    for first_part, second_part in zip(first_parts, second_parts):
        if first_part != second_part:
            break
        depth += 1
    return depth


def hierarchical_ec_shared_weight(
    first_ecs: tuple[str, ...] | list[str],
    second_ecs: tuple[str, ...] | list[str],
    min_shared_depth: int = 2,
) -> float:
    """Compute the maximum hierarchical EC positive weight for two EC sets."""

    best_depth = 0
    for first_ec in first_ecs:
        for second_ec in second_ecs:
            best_depth = max(best_depth, ec_shared_depth(first_ec, second_ec))
    if best_depth < min_shared_depth:
        return 0.0
    if best_depth >= 4:
        return 1.0
    if best_depth == 3:
        return 0.5
    if best_depth == 2:
        return 0.25
    return 0.0


class IndexedDataset(Dataset):
    """Wrap a dataset and include each sample's original row index."""

    def __init__(self, dataset: Dataset, value_key: str, index_key: str):
        self.dataset = dataset
        self.value_key = value_key
        self.index_key = index_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        if isinstance(sample, dict):
            output = dict(sample)
        else:
            output = {self.value_key: sample}
        output[self.index_key] = index
        return output


class HorizynDataModule(pl.LightningDataModule):
    """
    DataModule for Horizyn SOTA contrastive learning.

    Loads pre-split training and validation data, generates fingerprints for
    reactions, and creates dataloaders. All data is loaded into memory at
    setup time for fast training.

    SOTA Configuration:
        - Reactions: RDKit+ (1024-dim) + DRFP (1024-dim) concatenated
        - Proteins: T5 embeddings (1024-dim) pre-computed
        - Training batch size: 16384
        - Retrieval batch size: 128

    Args:
        train_pairs_path: Path to training pairs CSV file.
        test_pairs_path: Path to test pairs CSV file.
        train_reactions_path: Path to training reactions CSV file.
        test_reactions_path: Path to test reactions CSV file.
        protein_embeds_path: Path to protein embeddings HDF5 file.
        train_batch_size: Batch size for training. Defaults to 16384.
        retrieval_batch_size: Batch size for retrieval metrics. Defaults to 128.
        num_workers: Number of dataloader workers. Defaults to 4.
        pin_memory: Whether to pin memory. Defaults to False.
        rdkit_fp_dim: Dimension of RDKit+ fingerprints. Defaults to 1024.
        drfp_dim: Dimension of DRFP fingerprints. Defaults to 1024.
        standardize_reactions: Whether to standardize reactions. Defaults to True.
        standardize_hypervalent: Whether to standardize hypervalent atoms. Defaults to True.
        standardize_remove_hs: Whether to remove explicit hydrogen atoms. Defaults to True.
        standardize_kekulize: Whether to kekulize aromatic compounds. Defaults to False.
        standardize_uncharge: Whether to uncharge molecules. Defaults to True.
        standardize_metals: Whether to standardize metals. Defaults to True.

    Example:
        >>> dm = HorizynDataModule(
        ...     train_pairs_path="data/sota/train_pairs.csv",
        ...     test_pairs_path="data/sota/test_pairs.csv",
        ...     train_reactions_path="data/sota/train_rxns.csv",
        ...     test_reactions_path="data/sota/test_rxns.csv",
        ...     protein_embeds_path="data/sota/prots_t5.h5",
        ... )
        >>> dm.setup("fit")
        >>> train_loader = dm.train_dataloader()
        >>> val_loaders = dm.val_dataloader()
    """

    def __init__(
        self,
        train_pairs_path: str,
        test_pairs_path: str,
        train_reactions_path: str,
        test_reactions_path: str,
        protein_embeds_path: str,
        train_batch_size: int = 16384,
        retrieval_batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = False,
        rdkit_fp_dim: int = 1024,
        drfp_dim: int = 1024,
        reaction_representation: str = "fingerprint",
        reaction_embeds_path: str | None = None,
        reaction_t5v2_embeds_path: str | None = None,
        reaction_model_embeds_path: str | None = None,
        reaction_unimol2_embeds_path: str | None = None,
        reaction_chiro_embeds_path: str | None = None,
        reaction_chirality_embeds_path: str | None = None,
        reaction_chienn_embeds_path: str | None = None,
        reaction_chemistry_vectors_path: str | None = None,
        reaction_directional_vectors_path: str | None = None,
        train_reaction_embeds_path: str | None = None,
        validation_reaction_embeds_path: str | None = None,
        train_reaction_t5v2_embeds_path: str | None = None,
        validation_reaction_t5v2_embeds_path: str | None = None,
        train_reaction_model_embeds_path: str | None = None,
        validation_reaction_model_embeds_path: str | None = None,
        train_reaction_unimol2_embeds_path: str | None = None,
        validation_reaction_unimol2_embeds_path: str | None = None,
        train_reaction_chiro_embeds_path: str | None = None,
        validation_reaction_chiro_embeds_path: str | None = None,
        train_reaction_chirality_embeds_path: str | None = None,
        validation_reaction_chirality_embeds_path: str | None = None,
        train_reaction_chienn_embeds_path: str | None = None,
        validation_reaction_chienn_embeds_path: str | None = None,
        train_reaction_chemistry_vectors_path: str | None = None,
        validation_reaction_chemistry_vectors_path: str | None = None,
        train_reaction_directional_vectors_path: str | None = None,
        validation_reaction_directional_vectors_path: str | None = None,
        reaction_model_dim: int | None = None,
        reaction_unimol_dim: int = 768,
        reaction_chiro_dim: int | None = None,
        reaction_chirality_dim: int | None = None,
        reaction_chienn_dim: int | None = None,
        reaction_chemistry_dim: int | None = None,
        reaction_directional_dim: int | None = None,
        reaction_use_model: bool = True,
        reaction_use_chiro: bool | None = None,
        reaction_use_chirality: bool | None = None,
        reaction_use_chienn: bool = True,
        reaction_use_chemistry: bool = False,
        reaction_use_directional: bool = False,
        reaction_load_directional: bool = False,
        reaction_allow_missing_unimol2: bool = False,
        reaction_allow_missing_chiro: bool | None = None,
        reaction_allow_missing_chirality: bool | None = None,
        reaction_allow_missing_chienn: bool = False,
        reaction_allow_missing_chemistry: bool = True,
        reaction_allow_missing_directional: bool = True,
        reaction_embedding_in_memory: bool = True,
        standardize_reactions: bool = True,
        standardize_hypervalent: bool = True,
        standardize_remove_hs: bool = True,
        standardize_kekulize: bool = False,
        standardize_uncharge: bool = True,
        standardize_metals: bool = True,
        normalize_molecule_sets_as_self_reactions: bool = False,
        enzyme_ec_labels_path: str | None = None,
        protein_capability_vectors_path: str | None = None,
        protein_capability_metadata_path: str | None = None,
        capability_vector_in_memory: bool = True,
        capability_missing_policy: str = "zero_with_mask",
        protein_factorized_capability_vectors_path: str | None = None,
        factorized_capability_missing_policy: str = "zero_with_mask",
        protein_text_vectors_path: str | None = None,
        protein_text_metadata_path: str | None = None,
        text_vector_missing_policy: str = "zero_with_mask",
        protein_biofp_targets_path: str | None = None,
        protein_biofp_vocab_path: str | None = None,
        biofp_missing_policy: str = "zero_with_mask",
        reaction_direction_mode: str = "bidirectional",
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store paths
        self.train_pairs_path = Path(train_pairs_path)
        self.test_pairs_path = Path(test_pairs_path)
        self.train_reactions_path = Path(train_reactions_path)
        self.test_reactions_path = Path(test_reactions_path)
        self.protein_embeds_path = Path(protein_embeds_path)

        # Batch sizes
        self.train_batch_size = train_batch_size
        self.retrieval_batch_size = retrieval_batch_size

        # DataLoader settings
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Fingerprint settings
        self.rdkit_fp_dim = rdkit_fp_dim
        self.drfp_dim = drfp_dim
        self.reaction_representation = reaction_representation
        self.reaction_embeds_path = (
            None if reaction_embeds_path is None else Path(reaction_embeds_path)
        )
        self.reaction_t5v2_embeds_path = (
            None if reaction_t5v2_embeds_path is None else Path(reaction_t5v2_embeds_path)
        )
        self.reaction_model_embeds_path = (
            None if reaction_model_embeds_path is None else Path(reaction_model_embeds_path)
        )
        self.reaction_unimol2_embeds_path = (
            None if reaction_unimol2_embeds_path is None else Path(reaction_unimol2_embeds_path)
        )
        chirality_embeds_path = (
            reaction_chiro_embeds_path
            or reaction_chirality_embeds_path
            or reaction_chienn_embeds_path
        )
        self.reaction_chiro_embeds_path = (
            None if reaction_chiro_embeds_path is None else Path(reaction_chiro_embeds_path)
        )
        self.reaction_chirality_embeds_path = (
            None if reaction_chirality_embeds_path is None else Path(reaction_chirality_embeds_path)
        )
        self.reaction_chienn_embeds_path = (
            None if chirality_embeds_path is None else Path(chirality_embeds_path)
        )
        self.reaction_chemistry_vectors_path = (
            None
            if reaction_chemistry_vectors_path is None
            else Path(reaction_chemistry_vectors_path)
        )
        self.reaction_directional_vectors_path = (
            None
            if reaction_directional_vectors_path is None
            else Path(reaction_directional_vectors_path)
        )
        self.train_reaction_embeds_path = (
            None if train_reaction_embeds_path is None else Path(train_reaction_embeds_path)
        )
        self.validation_reaction_embeds_path = (
            None
            if validation_reaction_embeds_path is None
            else Path(validation_reaction_embeds_path)
        )
        self.train_reaction_t5v2_embeds_path = (
            None
            if train_reaction_t5v2_embeds_path is None
            else Path(train_reaction_t5v2_embeds_path)
        )
        self.validation_reaction_t5v2_embeds_path = (
            None
            if validation_reaction_t5v2_embeds_path is None
            else Path(validation_reaction_t5v2_embeds_path)
        )
        self.train_reaction_model_embeds_path = (
            None
            if train_reaction_model_embeds_path is None
            else Path(train_reaction_model_embeds_path)
        )
        self.validation_reaction_model_embeds_path = (
            None
            if validation_reaction_model_embeds_path is None
            else Path(validation_reaction_model_embeds_path)
        )
        self.train_reaction_unimol2_embeds_path = (
            None
            if train_reaction_unimol2_embeds_path is None
            else Path(train_reaction_unimol2_embeds_path)
        )
        self.validation_reaction_unimol2_embeds_path = (
            None
            if validation_reaction_unimol2_embeds_path is None
            else Path(validation_reaction_unimol2_embeds_path)
        )
        train_chirality_path = (
            train_reaction_chiro_embeds_path
            or train_reaction_chirality_embeds_path
            or train_reaction_chienn_embeds_path
        )
        validation_chirality_path = (
            validation_reaction_chiro_embeds_path
            or validation_reaction_chirality_embeds_path
            or validation_reaction_chienn_embeds_path
        )
        self.train_reaction_chiro_embeds_path = (
            None if train_reaction_chiro_embeds_path is None else Path(train_reaction_chiro_embeds_path)
        )
        self.validation_reaction_chiro_embeds_path = (
            None
            if validation_reaction_chiro_embeds_path is None
            else Path(validation_reaction_chiro_embeds_path)
        )
        self.train_reaction_chirality_embeds_path = (
            None
            if train_reaction_chirality_embeds_path is None
            else Path(train_reaction_chirality_embeds_path)
        )
        self.validation_reaction_chirality_embeds_path = (
            None
            if validation_reaction_chirality_embeds_path is None
            else Path(validation_reaction_chirality_embeds_path)
        )
        self.train_reaction_chienn_embeds_path = (
            None if train_chirality_path is None else Path(train_chirality_path)
        )
        self.validation_reaction_chienn_embeds_path = (
            None if validation_chirality_path is None else Path(validation_chirality_path)
        )
        self.train_reaction_chemistry_vectors_path = (
            None
            if train_reaction_chemistry_vectors_path is None
            else Path(train_reaction_chemistry_vectors_path)
        )
        self.validation_reaction_chemistry_vectors_path = (
            None
            if validation_reaction_chemistry_vectors_path is None
            else Path(validation_reaction_chemistry_vectors_path)
        )
        self.train_reaction_directional_vectors_path = (
            None
            if train_reaction_directional_vectors_path is None
            else Path(train_reaction_directional_vectors_path)
        )
        self.validation_reaction_directional_vectors_path = (
            None
            if validation_reaction_directional_vectors_path is None
            else Path(validation_reaction_directional_vectors_path)
        )
        self.reaction_model_dim = reaction_model_dim
        self.reaction_unimol_dim = reaction_unimol_dim
        self.reaction_chiro_dim = reaction_chiro_dim
        self.reaction_chirality_dim = reaction_chirality_dim
        self.reaction_chienn_dim = (
            reaction_chiro_dim or reaction_chirality_dim or reaction_chienn_dim
        )
        self.reaction_chemistry_dim = reaction_chemistry_dim
        self.reaction_directional_dim = reaction_directional_dim
        self.reaction_use_model = bool(reaction_use_model)
        self.reaction_use_chienn = (
            reaction_use_chiro
            if reaction_use_chiro is not None
            else (
                reaction_use_chirality
                if reaction_use_chirality is not None
                else reaction_use_chienn
            )
        )
        self.reaction_use_chiro = self.reaction_use_chienn
        self.reaction_use_chirality = self.reaction_use_chienn
        self.reaction_use_chemistry = bool(reaction_use_chemistry)
        self.reaction_use_directional = bool(reaction_use_directional)
        self.reaction_load_directional = bool(reaction_load_directional)
        self.reaction_allow_missing_unimol2 = reaction_allow_missing_unimol2
        self.reaction_allow_missing_chienn = (
            reaction_allow_missing_chiro
            if reaction_allow_missing_chiro is not None
            else (
                reaction_allow_missing_chirality
                if reaction_allow_missing_chirality is not None
                else reaction_allow_missing_chienn
            )
        )
        self.reaction_allow_missing_chiro = self.reaction_allow_missing_chienn
        self.reaction_allow_missing_chirality = self.reaction_allow_missing_chienn
        self.reaction_allow_missing_chemistry = bool(reaction_allow_missing_chemistry)
        self.reaction_allow_missing_directional = bool(
            reaction_allow_missing_directional
        )
        self.reaction_embedding_in_memory = reaction_embedding_in_memory
        self.standardize_reactions = standardize_reactions
        self.standardize_hypervalent = standardize_hypervalent
        self.standardize_remove_hs = standardize_remove_hs
        self.standardize_kekulize = standardize_kekulize
        self.standardize_uncharge = standardize_uncharge
        self.standardize_metals = standardize_metals
        self.normalize_molecule_sets_as_self_reactions = normalize_molecule_sets_as_self_reactions
        self.enzyme_ec_labels_path = (
            None if enzyme_ec_labels_path is None else Path(enzyme_ec_labels_path)
        )
        self.protein_capability_vectors_path = (
            None
            if protein_capability_vectors_path is None
            else Path(protein_capability_vectors_path)
        )
        self.protein_capability_metadata_path = (
            None
            if protein_capability_metadata_path is None
            else Path(protein_capability_metadata_path)
        )
        self.capability_vector_in_memory = bool(capability_vector_in_memory)
        self.capability_missing_policy = capability_missing_policy
        self._capability_vector_data = None
        self.protein_factorized_capability_vectors_path = (
            None
            if protein_factorized_capability_vectors_path is None
            else Path(protein_factorized_capability_vectors_path)
        )
        self.factorized_capability_missing_policy = factorized_capability_missing_policy
        self._factorized_capability_vector_data = None
        self.protein_text_vectors_path = (
            None if protein_text_vectors_path is None else Path(protein_text_vectors_path)
        )
        self.protein_text_metadata_path = (
            None if protein_text_metadata_path is None else Path(protein_text_metadata_path)
        )
        self.text_vector_missing_policy = text_vector_missing_policy
        self._text_vector_data = None
        self.protein_biofp_targets_path = (
            None
            if protein_biofp_targets_path is None
            else Path(protein_biofp_targets_path)
        )
        self.protein_biofp_vocab_path = (
            None if protein_biofp_vocab_path is None else Path(protein_biofp_vocab_path)
        )
        self.biofp_missing_policy = biofp_missing_policy
        self._biofp_target_data = None
        if reaction_direction_mode not in {"bidirectional", "forward_only"}:
            raise ValueError(
                "reaction_direction_mode must be one of: bidirectional, forward_only"
            )
        self.reaction_direction_mode = reaction_direction_mode

        # Dataset containers (populated in setup)
        self._train_data = None
        self._val_data = None
        self._val_query_data = None
        self._train_query_data = None  # Training query dataset (train reactions)
        self._val_query_data_raw = None  # Validation query dataset (val reactions)
        self._target_data = None  # Protein embeddings
        self._screening_target_data = None  # Full screening set
        self._train_query_to_targets = None
        self._train_target_to_queries = None
        self._train_enzyme_ec_sets = {}
        self._train_reaction_ec_sets = {}

    def _create_capability_dataset(self) -> CapabilityVectorDataset | None:
        if self.protein_capability_vectors_path is None:
            return None
        if not self.protein_capability_vectors_path.exists():
            raise FileNotFoundError(
                f"Protein capability vector file not found: {self.protein_capability_vectors_path}"
            )
        return CapabilityVectorDataset(self.protein_capability_vectors_path)

    def _maybe_attach_capability_vectors(self, target_dataset: BaseDataset) -> BaseDataset:
        if self.protein_capability_vectors_path is None:
            return target_dataset
        if self._capability_vector_data is None:
            self._capability_vector_data = self._create_capability_dataset()
        if self._capability_vector_data is None:
            return target_dataset
        merged = TargetWithCapabilityDataset(
            target_dataset=target_dataset,
            capability_dataset=self._capability_vector_data,
            capability_key="capability_vec",
            mask_key="capability_mask",
            missing_policy=self.capability_missing_policy,
        )
        missing = merged.missing_count
        if missing > 0:
            print(
                "  Capability vectors missing for "
                f"{missing}/{len(target_dataset.keys)} target proteins; "
                "using zero vectors with capability_mask=False"
            )
        return merged

    def _create_factorized_capability_dataset(
        self,
    ) -> FactorizedCapabilityVectorDataset | None:
        if self.protein_factorized_capability_vectors_path is None:
            return None
        if not self.protein_factorized_capability_vectors_path.exists():
            raise FileNotFoundError(
                "Protein factorized capability vector file not found: "
                f"{self.protein_factorized_capability_vectors_path}"
            )
        return FactorizedCapabilityVectorDataset(self.protein_factorized_capability_vectors_path)

    def _maybe_attach_factorized_capability_vectors(
        self,
        target_dataset: BaseDataset,
    ) -> BaseDataset:
        if self.protein_factorized_capability_vectors_path is None:
            return target_dataset
        if self._factorized_capability_vector_data is None:
            self._factorized_capability_vector_data = (
                self._create_factorized_capability_dataset()
            )
        if self._factorized_capability_vector_data is None:
            return target_dataset
        merged = TargetWithFactorizedCapabilityDataset(
            target_dataset=target_dataset,
            capability_dataset=self._factorized_capability_vector_data,
            missing_policy=self.factorized_capability_missing_policy,
        )
        missing = merged.missing_count
        if missing > 0:
            print(
                "  Factorized capability vectors missing for "
                f"{missing}/{len(target_dataset.keys)} target proteins; "
                "using zero vectors with family masks=False"
            )
        return merged

    def _create_text_dataset(self) -> TextVectorDataset | None:
        if self.protein_text_vectors_path is None:
            return None
        if not self.protein_text_vectors_path.exists():
            raise FileNotFoundError(
                f"Protein text vector file not found: {self.protein_text_vectors_path}"
            )
        return TextVectorDataset(self.protein_text_vectors_path)

    def _maybe_attach_text_vectors(self, target_dataset: BaseDataset) -> BaseDataset:
        if self.protein_text_vectors_path is None:
            return target_dataset
        if self._text_vector_data is None:
            self._text_vector_data = self._create_text_dataset()
        if self._text_vector_data is None:
            return target_dataset
        merged = TargetWithTextDataset(
            target_dataset=target_dataset,
            text_dataset=self._text_vector_data,
            text_key="text_vec",
            mask_key="text_mask",
            missing_policy=self.text_vector_missing_policy,
        )
        missing = merged.missing_count
        if missing > 0:
            print(
                "  Text vectors missing for "
                f"{missing}/{len(target_dataset.keys)} target proteins; "
                "using zero vectors with text_mask=False"
            )
        return merged

    def _create_biofp_dataset(self) -> BioFPTargetDataset | None:
        if self.protein_biofp_targets_path is None:
            return None
        if not self.protein_biofp_targets_path.exists():
            raise FileNotFoundError(
                f"Protein BioFP target file not found: {self.protein_biofp_targets_path}"
            )
        return BioFPTargetDataset(self.protein_biofp_targets_path)

    def _maybe_attach_biofp_targets(self, target_dataset: BaseDataset) -> BaseDataset:
        if self.protein_biofp_targets_path is None:
            return target_dataset
        if self._biofp_target_data is None:
            self._biofp_target_data = self._create_biofp_dataset()
        if self._biofp_target_data is None:
            return target_dataset
        merged = TargetWithBioFPTargetDataset(
            target_dataset=target_dataset,
            biofp_dataset=self._biofp_target_data,
            missing_policy=self.biofp_missing_policy,
        )
        missing = merged.missing_count
        if missing > 0:
            print(
                "  BioFP targets missing for "
                f"{missing}/{len(target_dataset.keys)} target proteins; "
                "using zero targets with BioFP masks=False"
            )
        return merged

    def _maybe_attach_target_side_vectors(self, target_dataset: BaseDataset) -> BaseDataset:
        target_dataset = self._maybe_attach_capability_vectors(target_dataset)
        target_dataset = self._maybe_attach_factorized_capability_vectors(target_dataset)
        target_dataset = self._maybe_attach_text_vectors(target_dataset)
        target_dataset = self._maybe_attach_biofp_targets(target_dataset)
        return target_dataset

    def setup(self, stage: Optional[str] = None):
        """
        Load all datasets into memory.

        This loads pairs, reactions, and proteins, generates fingerprints,
        and caches everything in memory for fast training.

        Args:
            stage: Stage ('fit', 'validate', 'test', or 'predict').
        """
        if stage in (None, "fit", "validate"):
            self._setup_training_data()
            self._setup_validation_data()

    def _augment_pairs_bidirectional(self, pairs: BaseDataset) -> BaseDataset:
        """
        Augment pairs with bidirectional reaction variants.

        For each pair (reaction_id, protein_id):
        - Forward: (reaction_id_f, protein_id)
        - Backward: (reaction_id_r, protein_id)

        Args:
            pairs: Dataset with query_id (reaction_id) and target_id (protein_id).

        Returns:
            Augmented dataset with both forward and backward pairs.
        """
        augmented_keys = []
        augmented_data = []

        for pair_key in pairs.keys:
            pair_data = pairs[pair_key]
            query_id = pair_data["query_id"]
            target_id = pair_data["target_id"]

            # Forward direction
            augmented_keys.append(f"{pair_key}_f")
            augmented_data.append(
                {
                    "query_id": f"{query_id}_f",
                    "target_id": target_id,
                }
            )

            # Backward direction
            augmented_keys.append(f"{pair_key}_r")
            augmented_data.append(
                {
                    "query_id": f"{query_id}_r",
                    "target_id": target_id,
                }
            )

        return BaseDataset(keys=augmented_keys, array_data=augmented_data)

    @staticmethod
    def _augment_pairs_forward_only(pairs: BaseDataset) -> BaseDataset:
        """Map released reaction rows to their canonical ``_f`` feature IDs."""

        augmented_keys = []
        augmented_data = []
        for pair_key in pairs.keys:
            pair_data = pairs[pair_key]
            augmented_keys.append(f"{pair_key}_f")
            augmented_data.append(
                {
                    "query_id": f"{pair_data['query_id']}_f",
                    "target_id": pair_data["target_id"],
                }
            )
        return BaseDataset(keys=augmented_keys, array_data=augmented_data)

    def _augment_pairs_for_direction_mode(self, pairs: BaseDataset) -> BaseDataset:
        if self.reaction_direction_mode == "forward_only":
            return self._augment_pairs_forward_only(pairs)
        return self._augment_pairs_bidirectional(pairs)

    def _filter_pairs_to_available_queries(
        self,
        pairs: BaseDataset,
        query_data: BaseDataset,
        split_name: str,
    ) -> BaseDataset:
        """Drop pairs whose reaction representation is unavailable."""

        query_key_set = set(query_data.keys)
        kept_keys = []
        kept_data = []
        for pair_key in pairs.keys:
            pair_data = pairs[pair_key]
            if pair_data["query_id"] not in query_key_set:
                continue
            kept_keys.append(pair_key)
            kept_data.append(pair_data)
        removed = len(pairs) - len(kept_keys)
        if removed > 0:
            print(
                f"  Filtered {removed} {split_name} pairs with missing reaction "
                "representations"
            )
        if not kept_keys:
            raise ValueError(
                f"No {split_name} pairs remain after filtering to available reaction "
                "representations"
            )
        return BaseDataset(keys=kept_keys, array_data=kept_data)

    @staticmethod
    def _build_pair_lookup_maps(
        pairs: BaseDataset,
        pair_keys: list[str] | None = None,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Build deterministic positive lookup maps from final pair rows."""

        query_to_targets = defaultdict(set)
        target_to_queries = defaultdict(set)
        keys = pairs.keys if pair_keys is None else pair_keys
        for pair_key in keys:
            pair = pairs[pair_key]
            query_id = pair["query_id"]
            target_id = pair["target_id"]
            query_to_targets[query_id].add(target_id)
            target_to_queries[target_id].add(query_id)
        return (
            {query_id: sorted(target_ids) for query_id, target_ids in query_to_targets.items()},
            {target_id: sorted(query_ids) for target_id, query_ids in target_to_queries.items()},
        )

    def _load_enzyme_ec_sets(self) -> dict[str, tuple[str, ...]]:
        """Load enzyme EC labels keyed by protein ID."""

        if self.enzyme_ec_labels_path is None:
            return {}
        if not self.enzyme_ec_labels_path.exists():
            raise FileNotFoundError(
                f"Enzyme EC labels file not found: {self.enzyme_ec_labels_path}"
            )

        protein_to_ecs: dict[str, set[str]] = defaultdict(set)
        with self.enzyme_ec_labels_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required_columns = {"protein_id", "ec_number"}
            missing = required_columns - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Enzyme EC labels file is missing columns: {sorted(missing)}"
                )
            for row in reader:
                protein_id = str(row.get("protein_id", "")).strip()
                if not protein_id:
                    continue
                for ec_number in split_ec_numbers(row.get("ec_number", "")):
                    protein_to_ecs[protein_id].add(ec_number)

        return {
            protein_id: tuple(sorted(ec_numbers))
            for protein_id, ec_numbers in protein_to_ecs.items()
            if ec_numbers
        }

    @staticmethod
    def _derive_reaction_ec_sets(
        pairs: BaseDataset,
        enzyme_ec_sets: dict[str, tuple[str, ...]],
        pair_keys: list[str] | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Derive reaction EC labels from EC labels of known positive enzymes."""

        reaction_to_ecs: dict[str, set[str]] = defaultdict(set)
        keys = pairs.keys if pair_keys is None else pair_keys
        for pair_key in keys:
            pair = pairs[pair_key]
            query_id = pair["query_id"]
            target_id = pair["target_id"]
            target_ecs = enzyme_ec_sets.get(target_id)
            if target_ecs is None:
                continue
            reaction_to_ecs[query_id].update(target_ecs)

        return {
            query_id: tuple(sorted(ec_numbers))
            for query_id, ec_numbers in reaction_to_ecs.items()
            if ec_numbers
        }

    def _setup_training_data(self):
        """Setup training dataset with fingerprints and embeddings."""
        print("Setting up training data...")

        # Load training pairs
        train_pairs = CSVDataset(
            file_path=str(self.train_pairs_path),
            key_column="pr_id",
            columns=["reaction_id", "protein_id"],
            rename_map={"reaction_id": "query_id", "protein_id": "target_id"},
        )
        original_pair_count = len(train_pairs)
        print(f"  Loaded {original_pair_count} training pairs")

        train_pairs = self._augment_pairs_for_direction_mode(train_pairs)
        print(
            f"  Prepared {len(train_pairs)} pairs with "
            f"reaction_direction_mode={self.reaction_direction_mode}"
        )

        # Load query data (train reactions with fingerprints)
        self._train_query_data = self._create_query_dataset(
            self.train_reactions_path,
            split_name="train",
        )
        print(f"  Loaded {len(self._train_query_data)} train reactions")
        train_pairs = self._filter_pairs_to_available_queries(
            train_pairs,
            self._train_query_data,
            split_name="training",
        )

        # Load target data (protein embeddings)
        self._target_data = self._maybe_attach_target_side_vectors(self._create_target_dataset())
        print(f"  Loaded {len(self._target_data)} proteins")

        # Merge pairs with query and target data
        self._train_data = TupleDataset(
            tuple_dataset=train_pairs,
            key_name_to_dataset={
                "query_id": self._train_query_data,
                "target_id": self._target_data,
            },
            rename_map={
                "query_id": "query_vec",
                "target_id": "target_vec",
            },
        )
        self._train_query_to_targets, self._train_target_to_queries = (
            self._build_pair_lookup_maps(train_pairs, list(self._train_data.keys))
        )
        all_enzyme_ec_sets = self._load_enzyme_ec_sets()
        self._train_enzyme_ec_sets = {
            target_id: all_enzyme_ec_sets[target_id]
            for target_id in self._train_target_to_queries
            if target_id in all_enzyme_ec_sets
        }
        self._train_reaction_ec_sets = self._derive_reaction_ec_sets(
            train_pairs,
            self._train_enzyme_ec_sets,
            pair_keys=list(self._train_data.keys),
        )
        print(
            "  Train positive map: "
            f"{len(self._train_query_to_targets)} queries, "
            f"{len(self._train_target_to_queries)} targets"
        )
        if self.enzyme_ec_labels_path is not None:
            print(
                "  Train EC maps: "
                f"{len(self._train_reaction_ec_sets)} reactions, "
                f"{len(self._train_enzyme_ec_sets)} enzymes"
            )

        print(f"Training dataset ready: {len(self._train_data)} samples")

    def _setup_validation_data(self):
        """Setup validation dataset for retrieval metrics."""
        print("Setting up validation data...")

        # Load validation pairs
        val_pairs = CSVDataset(
            file_path=str(self.test_pairs_path),
            key_column="pr_id",
            columns=["reaction_id", "protein_id"],
            rename_map={"reaction_id": "query_id", "protein_id": "target_id"},
        )
        original_val_pair_count = len(val_pairs)
        print(f"  Loaded {original_val_pair_count} validation pairs")

        val_pairs = self._augment_pairs_for_direction_mode(val_pairs)
        print(
            f"  Prepared {len(val_pairs)} pairs with "
            f"reaction_direction_mode={self.reaction_direction_mode}"
        )

        # Load validation query data (test reactions with fingerprints)
        self._val_query_data_raw = self._create_query_dataset(
            self.test_reactions_path,
            split_name="validation",
        )
        print(f"  Loaded {len(self._val_query_data_raw)} test reactions")
        val_pairs = self._filter_pairs_to_available_queries(
            val_pairs,
            self._val_query_data_raw,
            split_name="validation",
        )

        # Ensure target data is loaded
        if self._target_data is None:
            raise RuntimeError(
                "Training data must be setup before validation data. "
                "Target dataset must be loaded first."
            )

        # Load full protein embeddings for screening
        full_protein_dataset = EmbedDataset(
            file_path=str(self.protein_embeds_path),
            in_memory=True,
        )
        self._screening_target_data = self._maybe_attach_target_side_vectors(full_protein_dataset)

        # Collect protein stats
        train_pairs_for_stats = CSVDataset(
            file_path=str(self.train_pairs_path),
            key_column="pr_id",
            columns=["reaction_id", "protein_id"],
            rename_map={"reaction_id": "query_id", "protein_id": "target_id"},
        )
        train_protein_ids = set(
            train_pairs_for_stats[k]["target_id"] for k in train_pairs_for_stats.keys
        )
        val_protein_ids = set(val_pairs[k]["target_id"] for k in val_pairs.keys)
        screening_protein_ids = set(self._screening_target_data.keys)

        print(f"  Training proteins: {len(train_protein_ids)}")
        print(f"  Validation proteins: {len(val_protein_ids)}")
        print(f"  Screening set (full): {len(screening_protein_ids)} proteins")
        print(f"  Overlap (train ∩ val): {len(train_protein_ids & val_protein_ids)}")
        print(f"  Val-only proteins: {len(val_protein_ids - train_protein_ids)}")

        # Validation data for loss computation
        self._val_data = TupleDataset(
            tuple_dataset=val_pairs,
            key_name_to_dataset={
                "query_id": self._val_query_data_raw,
                "target_id": self._target_data,
            },
            rename_map={
                "query_id": "query_vec",
                "target_id": "target_vec",
            },
        )

        # Group pairs by query_id for multi-label retrieval metrics
        query_to_targets = defaultdict(list)
        for pair_key in val_pairs.keys:
            pair = val_pairs[pair_key]
            query_id = pair["query_id"]
            target_id = pair["target_id"]
            query_to_targets[query_id].append(target_id)

        # Get unique query IDs (sorted for determinism)
        unique_query_ids = sorted(query_to_targets.keys())

        # Create retrieval dataset: maps query_id -> list of target_ids
        retrieval_array_data = [query_to_targets[qid] for qid in unique_query_ids]

        # Store query-to-targets mapping
        self._query_to_targets = query_to_targets

        # Create dataset for retrieval queries (unique queries only)
        self._val_query_data = TupleDataset(
            tuple_dataset=BaseDataset(
                keys=unique_query_ids,
                array_data=[{"query_id": qid} for qid in unique_query_ids],
            ),
            key_name_to_dataset={
                "query_id": self._val_query_data_raw,
            },
            rename_map={
                "query_id": "query_vec",
            },
        )

        # Store target list dataset for metrics computation
        self._val_retrieval_targets = BaseDataset(
            keys=unique_query_ids,
            array_data=retrieval_array_data,
        )

        print(f"Validation dataset ready: {len(self._val_data)} samples")
        print(f"  Unique validation queries: {len(unique_query_ids)}")
        print(f"  Avg targets per query: {len(val_pairs) / len(unique_query_ids):.2f}")

    def _augment_reactions_bidirectional(self, reactions: BaseDataset) -> BaseDataset:
        """
        Augment reactions with bidirectional variants (forward and reverse).

        For each reaction with key 'rxn_id' and SMILES 'A>>B':
        - Forward: key='rxn_id_f', smiles='A>>B'
        - Backward: key='rxn_id_r', smiles='B>>A'

        Args:
            reactions: Dataset with reaction_smiles.

        Returns:
            Augmented dataset with both forward and backward reactions.
        """
        augmented_keys = []
        augmented_data = []

        for rxn_id in reactions.keys:
            rxn_data = reactions[rxn_id]
            smiles = rxn_data["reaction_smiles"]

            # Forward direction
            augmented_keys.append(f"{rxn_id}_f")
            augmented_data.append({"reaction_smiles": smiles})

            # Backward direction (reverse reactants and products)
            if ">>" in smiles:
                parts = smiles.split(">>")
                if len(parts) == 2:
                    reversed_smiles = f"{parts[1]}>>{parts[0]}"
                    augmented_keys.append(f"{rxn_id}_r")
                    augmented_data.append({"reaction_smiles": reversed_smiles})
                else:
                    # Malformed SMILES - skip backward
                    print(f"Warning: Malformed reaction SMILES for {rxn_id}: {smiles}")
            else:
                # No '>>' separator - skip backward
                print(f"Warning: No '>>' separator in reaction SMILES for {rxn_id}: {smiles}")

        return BaseDataset(keys=augmented_keys, array_data=augmented_data)

    def _normalize_reaction_smiles(self, _key: str, sample: dict) -> dict:
        """Optionally convert molecule-set strings into pseudo self-reactions."""

        if not self.normalize_molecule_sets_as_self_reactions:
            return sample
        smiles = sample.get("reaction_smiles", "")
        if isinstance(smiles, str) and smiles.count(">") < 2:
            sample = dict(sample)
            sample["reaction_smiles"] = f"{smiles}>>{smiles}"
        return sample

    @staticmethod
    def _is_train_split(split_name: str | None) -> bool:
        return split_name in {"train", "training"}

    @staticmethod
    def _is_validation_split(split_name: str | None) -> bool:
        return split_name in {"validation", "val", "valid", "eval", "test"}

    def _select_split_path(
        self,
        split_name: str | None,
        *,
        train_attr: str,
        validation_attr: str,
        fallback_attr: str,
    ) -> Path | None:
        if self._is_train_split(split_name):
            split_path = getattr(self, train_attr)
            if split_path is not None:
                return split_path
        if self._is_validation_split(split_name):
            split_path = getattr(self, validation_attr)
            if split_path is not None:
                return split_path
        return getattr(self, fallback_attr)

    def _select_reaction_embeds_path(self, split_name: str | None) -> Path | None:
        return self._select_split_path(
            split_name,
            train_attr="train_reaction_embeds_path",
            validation_attr="validation_reaction_embeds_path",
            fallback_attr="reaction_embeds_path",
        )

    def _select_reaction_model_path(self, split_name: str | None) -> Path | None:
        return self._select_split_path(
            split_name,
            train_attr="train_reaction_t5v2_embeds_path",
            validation_attr="validation_reaction_t5v2_embeds_path",
            fallback_attr="reaction_t5v2_embeds_path",
        ) or self._select_split_path(
            split_name,
            train_attr="train_reaction_model_embeds_path",
            validation_attr="validation_reaction_model_embeds_path",
            fallback_attr="reaction_model_embeds_path",
        )

    def _select_reaction_unimol2_path(self, split_name: str | None) -> Path | None:
        return self._select_split_path(
            split_name,
            train_attr="train_reaction_unimol2_embeds_path",
            validation_attr="validation_reaction_unimol2_embeds_path",
            fallback_attr="reaction_unimol2_embeds_path",
        ) or self._select_reaction_embeds_path(split_name)

    def _select_reaction_chirality_path(self, split_name: str | None) -> Path | None:
        return self._select_split_path(
            split_name,
            train_attr="train_reaction_chiro_embeds_path",
            validation_attr="validation_reaction_chiro_embeds_path",
            fallback_attr="reaction_chiro_embeds_path",
        ) or self._select_split_path(
            split_name,
            train_attr="train_reaction_chirality_embeds_path",
            validation_attr="validation_reaction_chirality_embeds_path",
            fallback_attr="reaction_chirality_embeds_path",
        ) or self._select_split_path(
            split_name,
            train_attr="train_reaction_chienn_embeds_path",
            validation_attr="validation_reaction_chienn_embeds_path",
            fallback_attr="reaction_chienn_embeds_path",
        )

    def _select_reaction_chemistry_path(self, split_name: str | None) -> Path | None:
        return self._select_split_path(
            split_name,
            train_attr="train_reaction_chemistry_vectors_path",
            validation_attr="validation_reaction_chemistry_vectors_path",
            fallback_attr="reaction_chemistry_vectors_path",
        )

    def _select_reaction_directional_path(self, split_name: str | None) -> Path | None:
        return self._select_split_path(
            split_name,
            train_attr="train_reaction_directional_vectors_path",
            validation_attr="validation_reaction_directional_vectors_path",
            fallback_attr="reaction_directional_vectors_path",
        )

    def _create_query_dataset(self, reactions_path: Path, split_name: str | None = None):
        """
        Create query dataset with RDKit+ and DRFP fingerprints.

        Reactions are augmented bidirectionally (forward + backward) to double
        the training data and ensure reversible reactions are learned properly.

        Args:
            reactions_path: Path to reactions CSV file.

        Returns:
            Dataset that returns concatenated 2048-dim fingerprints.
        """
        if self.reaction_representation == "unimol2_attention":
            reaction_embeds_path = self._select_reaction_embeds_path(split_name)
            if reaction_embeds_path is None:
                raise ValueError(
                    "data.reaction_embeds_path is required when "
                    "reaction_representation='unimol2_attention'"
                )
            return UniMol2ReactionEmbedDataset(
                file_path=str(reaction_embeds_path),
                expected_dim=self.reaction_unimol_dim,
            )
        if self.reaction_representation == "hybrid_fingerprint_unimol2":
            reaction_embeds_path = self._select_reaction_embeds_path(split_name)
            if reaction_embeds_path is None:
                raise ValueError(
                    "data.reaction_embeds_path is required when "
                    "reaction_representation='hybrid_fingerprint_unimol2'"
                )
            return build_reaction_feature_dataset(
                reactions_path=reactions_path,
                config=DotDict(
                    {
                        "data": {
                            "rdkit_fp_dim": self.rdkit_fp_dim,
                            "drfp_dim": self.drfp_dim,
                            "reaction_representation": self.reaction_representation,
                            "reaction_embeds_path": str(reaction_embeds_path),
                            "reaction_unimol_dim": self.reaction_unimol_dim,
                            "standardize_reactions": self.standardize_reactions,
                            "standardize_hypervalent": self.standardize_hypervalent,
                            "standardize_remove_hs": self.standardize_remove_hs,
                            "standardize_kekulize": self.standardize_kekulize,
                            "standardize_uncharge": self.standardize_uncharge,
                            "standardize_metals": self.standardize_metals,
                        }
                    }
                ),
                bidirectional=True,
                transforms=self._normalize_reaction_smiles,
                split_name=split_name,
            )
        if self.reaction_representation == "multimodal_reaction_attention":
            reaction_model_path = self._select_reaction_model_path(split_name)
            unimol2_path = self._select_reaction_unimol2_path(split_name)
            if self.reaction_use_model and reaction_model_path is None:
                raise ValueError(
                    "data.reaction_t5v2_embeds_path or data.reaction_model_embeds_path "
                    "is required when reaction_representation='multimodal_reaction_attention'"
                )
            if unimol2_path is None:
                raise ValueError(
                    "data.reaction_unimol2_embeds_path or data.reaction_embeds_path "
                    "is required when reaction_representation='multimodal_reaction_attention'"
                )
            reaction_chirality_path = self._select_reaction_chirality_path(split_name)
            if self.reaction_use_chienn and reaction_chirality_path is None:
                raise ValueError(
                    "data.reaction_chiro_embeds_path is required when "
                    "reaction_representation='multimodal_reaction_attention' and "
                    "chirality is enabled"
                )
            reaction_chirality_path = (
                None if reaction_chirality_path is None else str(reaction_chirality_path)
            )
            reaction_chemistry_path = (
                self._select_reaction_chemistry_path(split_name)
                if self.reaction_use_chemistry
                else None
            )
            reaction_directional_path = (
                self._select_reaction_directional_path(split_name)
                if self.reaction_use_directional or self.reaction_load_directional
                else None
            )
            if (
                self.reaction_use_directional or self.reaction_load_directional
            ) and reaction_directional_path is None:
                raise ValueError(
                    "data.reaction_directional_vectors_path is required when "
                    "reaction directional vectors are loaded"
                )
            return build_reaction_feature_dataset(
                reactions_path=reactions_path,
                config=DotDict(
                    {
                        "data": {
                            "reaction_representation": self.reaction_representation,
                            "reaction_t5v2_embeds_path": (
                                None
                                if reaction_model_path is None
                                else str(reaction_model_path)
                            ),
                            "reaction_unimol2_embeds_path": str(unimol2_path),
                            "reaction_chiro_embeds_path": reaction_chirality_path,
                            "reaction_chirality_embeds_path": reaction_chirality_path,
                            "reaction_chienn_embeds_path": reaction_chirality_path,
                            "reaction_chemistry_vectors_path": (
                                None
                                if reaction_chemistry_path is None
                                else str(reaction_chemistry_path)
                            ),
                            "reaction_directional_vectors_path": (
                                None
                                if reaction_directional_path is None
                                else str(reaction_directional_path)
                            ),
                            "reaction_model_dim": self.reaction_model_dim,
                            "reaction_unimol_dim": self.reaction_unimol_dim,
                            "reaction_chiro_dim": self.reaction_chienn_dim,
                            "reaction_chirality_dim": self.reaction_chienn_dim,
                            "reaction_chienn_dim": self.reaction_chienn_dim,
                            "reaction_chemistry_dim": self.reaction_chemistry_dim,
                            "reaction_directional_dim": self.reaction_directional_dim,
                            "reaction_use_model": self.reaction_use_model,
                            "reaction_use_chiro": self.reaction_use_chienn,
                            "reaction_use_chirality": self.reaction_use_chienn,
                            "reaction_use_chienn": self.reaction_use_chienn,
                            "reaction_use_chemistry": self.reaction_use_chemistry,
                            "reaction_use_directional": self.reaction_use_directional,
                            "reaction_allow_missing_unimol2": (
                                self.reaction_allow_missing_unimol2
                            ),
                            "reaction_allow_missing_chiro": self.reaction_allow_missing_chienn,
                            "reaction_allow_missing_chirality": self.reaction_allow_missing_chienn,
                            "reaction_allow_missing_chienn": self.reaction_allow_missing_chienn,
                            "reaction_allow_missing_chemistry": (
                                self.reaction_allow_missing_chemistry
                            ),
                            "reaction_allow_missing_directional": (
                                self.reaction_allow_missing_directional
                            ),
                            "reaction_embedding_in_memory": self.reaction_embedding_in_memory,
                        }
                    }
                ),
                bidirectional=True,
                transforms=self._normalize_reaction_smiles,
                split_name=split_name,
            )
        if self.reaction_representation != "fingerprint":
            raise ValueError(
                "reaction_representation must be one of: fingerprint, "
                "unimol2_attention, hybrid_fingerprint_unimol2, "
                "multimodal_reaction_attention"
            )

        # Load reaction SMILES
        reactions = CSVDataset(
            file_path=str(reactions_path),
            key_column="reaction_id",
            columns=["reaction_smiles"],
            transforms=self._normalize_reaction_smiles,
        )

        # Augment with bidirectional reactions (forward + backward)
        reactions = self._augment_reactions_bidirectional(reactions)
        print(f"  Augmented to {len(reactions)} bidirectional reactions")

        # Generate RDKit+ fingerprints (1024-dim)
        rdkit_fp = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions,
            vec_dim=self.rdkit_fp_dim,
            mol_fp_type="morgan",
            rxn_fp_type="struct",
            use_chirality=True,
            standardize=self.standardize_reactions,
            standardize_hypervalent=self.standardize_hypervalent,
            standardize_remove_hs=self.standardize_remove_hs,
            standardize_kekulize=self.standardize_kekulize,
            standardize_uncharge=self.standardize_uncharge,
            standardize_metals=self.standardize_metals,
        )

        # Generate DRFP fingerprints (1024-dim)
        drfp_fp = DRFPFingerprintDataset(
            reaction_dataset=reactions,
            vec_dim=self.drfp_dim,
            radius=3,
            rings=True,
            standardize=self.standardize_reactions,
            standardize_hypervalent=self.standardize_hypervalent,
            standardize_remove_hs=self.standardize_remove_hs,
            standardize_kekulize=self.standardize_kekulize,
            standardize_uncharge=self.standardize_uncharge,
            standardize_metals=self.standardize_metals,
        )

        # Merge fingerprints
        merged = MergeDataset(
            datasets={"rdkit": rdkit_fp, "drfp": drfp_fp},
            add_prefix=False,
        )

        # Concatenate into single vector (2048-dim)
        merged.append_transforms(ConcatTensorTransform(labels=["rdkit", "drfp"], dim=0))

        return merged

    def _create_target_dataset(self):
        """
        Create target dataset with pre-computed T5 embeddings.

        Returns:
            Dataset that returns 1024-dim T5 embeddings.
        """
        return EmbedDataset(
            file_path=str(self.protein_embeds_path),
            in_memory=True,
        )

    @property
    def train_data(self):
        """Access to training dataset (read-only)."""
        return self._train_data

    @property
    def val_data(self):
        """Access to validation dataset (read-only)."""
        return self._val_data

    @property
    def val_query_data(self):
        """Access to validation query dataset (read-only)."""
        return self._val_query_data

    @property
    def val_retrieval_pairs(self):
        """Access to validation query dataset (alias for compatibility)."""
        return self._val_query_data

    def train_dataloader(self) -> DataLoader:
        """
        Create training dataloader.

        Returns:
            DataLoader with training data.
        """
        if self._train_data is None:
            raise RuntimeError("Training data not setup. Call setup() first.")

        return DataLoader(
            self._train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=dict_collate_fn,
        )

    def val_dataloader(self) -> List[DataLoader]:
        """
        Create validation dataloaders.

        Returns list of dataloaders:
            1. Validation loss dataloader
            2. Target lookup table dataloader (FULL screening set)
            3. Query retrieval metrics dataloader

        Returns:
            List of DataLoaders for validation.
        """
        if self._val_data is None or self._screening_target_data is None:
            raise RuntimeError("Validation data not setup. Call setup() first.")

        dataloaders = []

        # 1. Validation loss dataloader
        dataloaders.append(
            DataLoader(
                self._val_data,
                batch_size=self.train_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=dict_collate_fn,
            )
        )

        # 2. Target lookup table (FULL screening set for retrieval metrics)
        dataloaders.append(
            DataLoader(
                IndexedDataset(
                    self._screening_target_data,
                    value_key="target_vec",
                    index_key="target_lookup_row_idx",
                ),
                batch_size=self.train_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=dict_collate_fn,
            )
        )

        # 3. Query retrieval metrics
        dataloaders.append(
            DataLoader(
                self._val_query_data,
                batch_size=self.retrieval_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=default_collate,
            )
        )

        return dataloaders
