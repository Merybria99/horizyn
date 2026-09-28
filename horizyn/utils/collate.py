"""
Batch collation functions for PyTorch DataLoader.
"""

from typing import Any, Dict, List

import torch
from torch.utils.data._utils.collate import default_collate


REACTION_SET_KEYS = {
    "reactant_embeddings",
    "product_embeddings",
}

CHIRALITY_SET_KEYS = {
    "reactant_chirality_embeddings",
    "product_chirality_embeddings",
}

HYBRID_REACTION_KEYS = {
    "fingerprint",
    "has_unimol2",
    "has_chirality",
    "has_chiro",
    "has_chienn",
    "reaction_embedding",
    "reaction_chemistry_vector",
    "has_reaction_chemistry",
    "reaction_directional_vector",
    "has_reaction_directional",
}


def _common_mapping_keys(batch: List[Dict[str, Any]]) -> set[str]:
    """Return keys present in every sample of a non-empty mapping batch."""
    common_keys = set(batch[0].keys())
    for sample in batch[1:]:
        common_keys.intersection_update(sample.keys())
    return common_keys


def _validate_partial_reaction_keys_are_metadata(
    batch: List[Dict[str, Any]],
    common_keys: set[str],
) -> None:
    """Reject partially present model inputs while allowing optional metadata.

    Primary and replay datasets can expose different string annotations (for
    example ``reaction_smiles``). Those annotations are not model inputs and
    are omitted from a heterogeneous batch. A tensor or nested mapping that is
    present in only part of a batch is instead almost certainly a malformed
    feature batch and must fail loudly rather than being silently discarded.
    """
    all_keys = set().union(*(sample.keys() for sample in batch))
    for key in sorted(all_keys - common_keys):
        present_values = [sample[key] for sample in batch if key in sample]
        if any(isinstance(value, (torch.Tensor, dict)) for value in present_values):
            raise KeyError(
                f"Reaction feature {key!r} must be present in every sample; "
                "only non-tensor metadata may differ between primary and replay rows"
            )


def _is_reaction_set_sample(value: Any) -> bool:
    return isinstance(value, dict) and (
        REACTION_SET_KEYS.issubset(value.keys()) or CHIRALITY_SET_KEYS.issubset(value.keys())
    )


def _pad_reaction_side_pair(
    batch: List[Dict[str, torch.Tensor]],
    reactant_key: str,
    product_key: str,
    reactant_mask_key: str,
    product_mask_key: str,
    label: str,
) -> Dict[str, torch.Tensor]:
    reactant_lengths = [int(sample[reactant_key].shape[0]) for sample in batch]
    product_lengths = [int(sample[product_key].shape[0]) for sample in batch]
    if any(length <= 0 for length in reactant_lengths):
        raise ValueError(f"Reaction samples must contain at least one {label} reactant embedding")
    if any(length <= 0 for length in product_lengths):
        raise ValueError(f"Reaction samples must contain at least one {label} product embedding")

    embedding_dim = int(batch[0][reactant_key].shape[1])
    dtype = batch[0][reactant_key].dtype
    max_reactants = max(reactant_lengths)
    max_products = max(product_lengths)

    reactant_embeddings = torch.zeros(len(batch), max_reactants, embedding_dim, dtype=dtype)
    product_embeddings = torch.zeros(len(batch), max_products, embedding_dim, dtype=dtype)
    reactant_padding_mask = torch.ones(len(batch), max_reactants, dtype=torch.bool)
    product_padding_mask = torch.ones(len(batch), max_products, dtype=torch.bool)

    for idx, sample in enumerate(batch):
        reactants = sample[reactant_key]
        products = sample[product_key]
        if reactants.ndim != 2 or products.ndim != 2:
            raise ValueError(f"{label} reactant/product embeddings must be rank-2 tensors")
        if reactants.shape[1] != embedding_dim or products.shape[1] != embedding_dim:
            raise ValueError(
                f"All {label} molecule embeddings must have dim {embedding_dim}; "
                f"got reactants={reactants.shape[1]}, products={products.shape[1]}"
            )

        r_len = reactant_lengths[idx]
        p_len = product_lengths[idx]
        reactant_embeddings[idx, :r_len] = reactants
        product_embeddings[idx, :p_len] = products
        reactant_padding_mask[idx, :r_len] = False
        product_padding_mask[idx, :p_len] = False

    return {
        reactant_key: reactant_embeddings,
        reactant_mask_key: reactant_padding_mask,
        product_key: product_embeddings,
        product_mask_key: product_padding_mask,
    }


def unimol2_reaction_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    extra_keys: set[str] | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Pad ragged reaction-side molecule embedding sets.

    Returned padding masks use ``True`` for padded positions.
    """
    if not batch:
        return {}
    if any(not _is_reaction_set_sample(sample) for sample in batch):
        raise KeyError(
            "All reaction samples must contain a supported reactant/product "
            "embedding pair"
        )

    common_keys = _common_mapping_keys(batch)
    collated: Dict[str, Any] = {}
    padded_source_keys = set()
    has_unimol2 = [REACTION_SET_KEYS.issubset(sample.keys()) for sample in batch]
    if any(has_unimol2) and not all(has_unimol2):
        raise KeyError(
            "Uni-Mol2 reactant/product embeddings must be present in every "
            "sample when any sample provides them"
        )
    if all(has_unimol2):
        collated.update(
            _pad_reaction_side_pair(
                batch,
                reactant_key="reactant_embeddings",
                product_key="product_embeddings",
                reactant_mask_key="reactant_padding_mask",
                product_mask_key="product_padding_mask",
                label="Uni-Mol2",
            )
        )
        padded_source_keys.update(REACTION_SET_KEYS)

    has_chirality = [CHIRALITY_SET_KEYS.issubset(sample.keys()) for sample in batch]
    if any(has_chirality) and not all(has_chirality):
        raise KeyError(
            "Chirality reactant/product embeddings must be present in every "
            "sample when any sample provides them"
        )
    if all(has_chirality):
        collated.update(
            _pad_reaction_side_pair(
                batch,
                reactant_key="reactant_chirality_embeddings",
                product_key="product_chirality_embeddings",
                reactant_mask_key="reactant_chirality_padding_mask",
                product_mask_key="product_chirality_padding_mask",
                label="chirality",
            )
        )
        padded_source_keys.update(CHIRALITY_SET_KEYS)

    if extra_keys is None:
        extra_keys = common_keys - padded_source_keys
    else:
        extra_keys = set(extra_keys) & common_keys

    for key in batch[0].keys():
        if key in padded_source_keys or key not in extra_keys:
            continue
        values = [sample[key] for sample in batch]
        if isinstance(values[0], str):
            collated[key] = values
        else:
            collated[key] = default_collate(values)

    return collated


def dict_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate a batch of dictionaries into a single dictionary of batched tensors.

    This function takes a list of dictionaries (one per sample) and combines them
    into a single dictionary where each value is a batched tensor. It uses PyTorch's
    default_collate for the actual batching.

    This is the standard collation function for the SOTA model, which uses
    dictionaries containing query and target vectors.

    Args:
        batch: List of dictionaries, one per sample. Each dict should have the
            same keys and compatible tensor values.

    Returns:
        Dictionary with same keys as input dicts, where each value is a batched
        tensor of shape [batch_size, ...].

    Example:
        >>> batch = [
        ...     {"query_vec": torch.tensor([1, 2, 3]), "target_vec": torch.tensor([4, 5, 6])},
        ...     {"query_vec": torch.tensor([7, 8, 9]), "target_vec": torch.tensor([10, 11, 12])}
        ... ]
        >>> result = dict_collate_fn(batch)
        >>> result["query_vec"].shape
        torch.Size([2, 3])
    """
    if not batch:
        return {}

    top_level_reaction = any(_is_reaction_set_sample(sample) for sample in batch)
    nested_reaction = any(
        "query_vec" in sample and _is_reaction_set_sample(sample["query_vec"])
        for sample in batch
    )
    if top_level_reaction:
        common_keys = _common_mapping_keys(batch)
        _validate_partial_reaction_keys_are_metadata(batch, common_keys)
        collated: Dict[str, Any] = {}
        skip_keys = set()
        query_extra_keys = HYBRID_REACTION_KEYS & common_keys
        collated.update(unimol2_reaction_collate_fn(batch, extra_keys=query_extra_keys))
        skip_keys.update(REACTION_SET_KEYS | CHIRALITY_SET_KEYS | query_extra_keys)

        for key in batch[0].keys():
            if key in skip_keys or key not in common_keys:
                continue
            values = [sample[key] for sample in batch]
            if key == "query_vec" and _is_reaction_set_sample(values[0]):
                collated[key] = unimol2_reaction_collate_fn(values)
            elif isinstance(values[0], str):
                collated[key] = values
            else:
                collated[key] = default_collate(values)
        return collated

    if nested_reaction:
        common_keys = _common_mapping_keys(batch)
        _validate_partial_reaction_keys_are_metadata(batch, common_keys)
        collated = {}
        for key in batch[0].keys():
            if key not in common_keys:
                continue
            values = [sample[key] for sample in batch]
            if key == "query_vec":
                collated[key] = unimol2_reaction_collate_fn(values)
            elif isinstance(values[0], str):
                collated[key] = values
            else:
                collated[key] = default_collate(values)
        return collated

    return default_collate(batch)


def residue_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate reaction-conditioned residue batches with sequence padding.

    Input samples must contain ``residue_embeddings`` with shape ``[L, D]``.
    The returned ``residue_padding_mask`` has shape ``[B, Lmax]`` and uses
    ``True`` for padded positions.
    """
    if not batch:
        return {}

    if any("residue_embeddings" not in sample for sample in batch):
        raise KeyError("All samples must contain 'residue_embeddings'")

    lengths = [int(sample["residue_embeddings"].shape[0]) for sample in batch]
    if any(length <= 0 for length in lengths):
        raise ValueError("Residue embedding sequences must be non-empty")

    max_length = max(lengths)
    residue_dim = int(batch[0]["residue_embeddings"].shape[1])
    dtype = batch[0]["residue_embeddings"].dtype

    residue_embeddings = torch.zeros(len(batch), max_length, residue_dim, dtype=dtype)
    residue_padding_mask = torch.ones(len(batch), max_length, dtype=torch.bool)
    has_score_residues = any("score_residue_embeddings" in sample for sample in batch)
    if has_score_residues and not all("score_residue_embeddings" in sample for sample in batch):
        raise KeyError("All samples must contain 'score_residue_embeddings' when any sample does")
    score_residue_embeddings = None
    score_residue_padding_mask = None
    if has_score_residues:
        score_dim = int(batch[0]["score_residue_embeddings"].shape[1])
        score_dtype = batch[0]["score_residue_embeddings"].dtype
        score_residue_embeddings = torch.zeros(
            len(batch),
            max_length,
            score_dim,
            dtype=score_dtype,
        )
        score_residue_padding_mask = torch.ones(len(batch), max_length, dtype=torch.bool)
    has_residue_labels = any("residue_labels" in sample for sample in batch)
    has_residue_label_mask = any("residue_label_mask" in sample for sample in batch)
    if has_residue_labels and not all("residue_labels" in sample for sample in batch):
        raise KeyError("All samples must contain 'residue_labels' when any sample does")
    if has_residue_label_mask and not all("residue_label_mask" in sample for sample in batch):
        raise KeyError("All samples must contain 'residue_label_mask' when any sample does")
    residue_labels = (
        torch.zeros(len(batch), max_length, dtype=torch.float32) if has_residue_labels else None
    )
    residue_label_mask = (
        torch.zeros(len(batch), max_length, dtype=torch.bool) if has_residue_label_mask else None
    )

    for idx, sample in enumerate(batch):
        residues = sample["residue_embeddings"]
        if residues.ndim != 2:
            raise ValueError(
                f"'residue_embeddings' must be rank-2, got shape={tuple(residues.shape)}"
            )
        if residues.shape[1] != residue_dim:
            raise ValueError(
                f"All residue embeddings must have dim {residue_dim}, got {residues.shape[1]}"
            )
        length = lengths[idx]
        residue_embeddings[idx, :length] = residues
        residue_padding_mask[idx, :length] = False
        if score_residue_embeddings is not None:
            score_residues = sample["score_residue_embeddings"]
            if score_residues.ndim != 2:
                raise ValueError(
                    "'score_residue_embeddings' must be rank-2, "
                    f"got shape={tuple(score_residues.shape)}"
                )
            if score_residues.shape[0] != length:
                raise ValueError(
                    "'score_residue_embeddings' must match residue length, "
                    f"got length={score_residues.shape[0]} for value length={length}"
                )
            if score_residues.shape[1] != score_residue_embeddings.shape[2]:
                raise ValueError(
                    "All score residue embeddings must have dim "
                    f"{score_residue_embeddings.shape[2]}, got {score_residues.shape[1]}"
                )
            score_residue_embeddings[idx, :length] = score_residues
            score_residue_padding_mask[idx, :length] = False
        if residue_labels is not None:
            labels = torch.as_tensor(sample["residue_labels"])
            if labels.ndim != 1 or labels.shape[0] != length:
                raise ValueError(
                    "'residue_labels' must be rank-1 and match residue length, "
                    f"got shape={tuple(labels.shape)} for length={length}"
                )
            residue_labels[idx, :length] = labels.to(dtype=torch.float32)
        if residue_label_mask is not None:
            label_mask = torch.as_tensor(sample["residue_label_mask"])
            if label_mask.ndim != 1 or label_mask.shape[0] != length:
                raise ValueError(
                    "'residue_label_mask' must be rank-1 and match residue length, "
                    f"got shape={tuple(label_mask.shape)} for length={length}"
                )
            residue_label_mask[idx, :length] = label_mask.to(dtype=torch.bool)

    collated: Dict[str, Any] = {
        "residue_embeddings": residue_embeddings,
        "residue_padding_mask": residue_padding_mask,
    }
    if score_residue_embeddings is not None:
        collated["score_residue_embeddings"] = score_residue_embeddings
        collated["score_residue_padding_mask"] = score_residue_padding_mask
    if residue_labels is not None:
        collated["residue_labels"] = residue_labels
    if residue_label_mask is not None:
        collated["residue_label_mask"] = residue_label_mask

    skip_keys = {
        "residue_embeddings",
        "score_residue_embeddings",
        "score_residue_padding_mask",
        "residue_labels",
        "residue_label_mask",
    }
    if _is_reaction_set_sample(batch[0]):
        query_extra_keys = HYBRID_REACTION_KEYS & set(batch[0].keys())
        if "query_vec" not in batch[0]:
            collated["query_vec"] = unimol2_reaction_collate_fn(
                batch,
                extra_keys=query_extra_keys,
            )
        skip_keys.update(REACTION_SET_KEYS | CHIRALITY_SET_KEYS | query_extra_keys)

    remaining_keys = [key for key in batch[0].keys() if key not in skip_keys]
    for key in remaining_keys:
        values = [sample[key] for sample in batch]
        if key == "query_vec" and _is_reaction_set_sample(values[0]):
            collated[key] = unimol2_reaction_collate_fn(values)
        elif isinstance(values[0], str):
            collated[key] = values
        else:
            collated[key] = default_collate(values)

    return collated


__all__ = [
    "default_collate",
    "dict_collate_fn",
    "residue_collate_fn",
    "unimol2_reaction_collate_fn",
]
