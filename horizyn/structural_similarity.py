"""
Utilities for structure-aware loss similarities.

These helpers build detached in-batch external similarity matrices from the raw
features already present in Horizyn batches. They intentionally do not add new
dataset dependencies: fingerprints drive reaction similarity when available,
and mean-pooled protein features drive enzyme similarity.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def flatten_feature_matrix(features: torch.Tensor) -> torch.Tensor:
    """Flatten all non-batch dimensions into a feature matrix."""
    if features.ndim < 2:
        raise ValueError(f"features must have a batch dimension and feature dim, got {features.shape}")
    return features.reshape(features.shape[0], -1).detach()


def masked_mean_features(
    embeddings: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean-pool sequence or molecule embeddings while ignoring padded rows."""
    if embeddings.ndim != 3:
        return flatten_feature_matrix(embeddings)

    if padding_mask is None:
        return embeddings.mean(dim=1).detach()

    valid = ~padding_mask.to(device=embeddings.device, dtype=torch.bool)
    weights = valid.unsqueeze(-1).to(dtype=embeddings.dtype)
    lengths = weights.sum(dim=1).clamp_min(1.0)
    return (embeddings * weights).sum(dim=1).div(lengths).detach()


def reaction_feature_matrix(
    query_vecs: torch.Tensor | dict[str, torch.Tensor],
) -> Optional[torch.Tensor]:
    """
    Build detached reaction features for external RR similarity.

    Hybrid reaction batches contain an RDKit+DRFP fingerprint branch; this is the
    preferred external chemical signal. Uni-Mol2-only batches fall back to the
    same reactant/product mean composition used by the mean-pooled reaction branch.
    """
    if isinstance(query_vecs, torch.Tensor):
        return flatten_feature_matrix(query_vecs)

    fingerprint = query_vecs.get("fingerprint")
    if torch.is_tensor(fingerprint):
        return flatten_feature_matrix(fingerprint)

    multimodal_features: list[torch.Tensor] = []
    reaction_embedding = query_vecs.get("reaction_embedding")
    if torch.is_tensor(reaction_embedding):
        multimodal_features.append(flatten_feature_matrix(reaction_embedding))

    reactants = query_vecs.get("reactant_embeddings")
    products = query_vecs.get("product_embeddings")
    if torch.is_tensor(reactants) and torch.is_tensor(products):
        reactant_mean = masked_mean_features(
            reactants,
            query_vecs.get("reactant_padding_mask"),
        )
        product_mean = masked_mean_features(
            products,
            query_vecs.get("product_padding_mask"),
        )
        multimodal_features.append(
            torch.cat(
                [
                    reactant_mean,
                    product_mean,
                    product_mean - reactant_mean,
                    (product_mean - reactant_mean).abs(),
                ],
                dim=-1,
            )
        )

    chirality_reactants = query_vecs.get("reactant_chirality_embeddings")
    chirality_products = query_vecs.get("product_chirality_embeddings")
    if torch.is_tensor(chirality_reactants) and torch.is_tensor(chirality_products):
        reactant_chirality_mean = masked_mean_features(
            chirality_reactants,
            query_vecs.get("reactant_chirality_padding_mask"),
        )
        product_chirality_mean = masked_mean_features(
            chirality_products,
            query_vecs.get("product_chirality_padding_mask"),
        )
        multimodal_features.append(
            torch.cat(
                [
                    reactant_chirality_mean,
                    product_chirality_mean,
                    product_chirality_mean - reactant_chirality_mean,
                    (product_chirality_mean - reactant_chirality_mean).abs(),
                ],
                dim=-1,
            )
        )

    if not multimodal_features:
        return None
    if len(multimodal_features) == 1:
        return multimodal_features[0].detach()
    return torch.cat(multimodal_features, dim=-1).detach()


def pairwise_cosine_similarity(features: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Compute a detached cosine similarity matrix for external structure labels."""
    if features is None:
        return None
    if features.ndim != 2:
        features = flatten_feature_matrix(features)
    features = F.normalize(features.detach().float(), p=2, dim=-1, eps=1e-12)
    return torch.matmul(features, features.t()).detach()
