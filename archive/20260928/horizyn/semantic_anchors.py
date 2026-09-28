"""Independent enzyme/reaction encoders anchored in training associations.

Both sides produce a vector indexed by the same training reaction catalog.
There is no dependence on the other side's current query/candidate batch.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def row_unit(raw):
    """Normalize with FP64 reductions, then store in the original dtype.

    FP32 CUDA norm kernels may choose different reduction layouts for a full
    catalog and a small inference batch. Rounding after a FP64 reduction pins
    the feature map more tightly, especially for near-identical homologs.
    """
    return F.normalize(raw.double(), dim=-1).to(raw.dtype)


def stable_topk(scores, count):
    """Top-k membership with the lower anchor index winning cutoff ties."""
    count = min(count, scores.shape[1])
    initial, _ = torch.topk(scores, count, dim=1)
    cutoff = initial[:, -1:]
    greater = scores > cutoff
    ties = scores == cutoff
    needed = count - greater.sum(1, keepdim=True)
    chosen = greater | (ties & (ties.cumsum(1, dtype=torch.int32) <= needed))
    return torch.topk(scores.masked_fill(~chosen, float("-inf")), count, dim=1, sorted=True)


def fit_center(raw, indices, mask=None):
    unit = row_unit(raw)
    chosen = unit[indices]
    if mask is not None:
        chosen = chosen[mask[indices]]
    if not len(chosen):
        raise ValueError("Cannot fit a feature center without training observations")
    return chosen.double().mean(0).to(raw.dtype)


def centered_unit(raw, center, mask=None):
    result = row_unit(row_unit(raw) - center)
    if mask is not None:
        result = result * mask[:, None]
    return result


def reaction_features(raw_blocks, centers, masks, modalities):
    values = [centered_unit(raw_blocks[key], centers[key], masks[key]) for key in modalities]
    return row_unit(torch.cat(values, dim=1))


def make_protein_reaction_map(reaction_index, enzyme_index, n_enzymes):
    """Padded training-only adjacency; -1 marks unused entries."""
    edges = torch.unique(torch.stack((enzyme_index, reaction_index), dim=1), dim=0)
    order = torch.argsort(edges[:, 0], stable=True)
    edges = edges[order]
    counts = torch.bincount(edges[:, 0], minlength=n_enzymes)
    if bool((counts == 0).any()):
        raise ValueError("Every anchor protein must have an observed association")
    starts = counts.cumsum(0) - counts
    slots = torch.arange(len(edges), device=edges.device) - starts[edges[:, 0]]
    result = torch.full((n_enzymes, int(counts.max())), -1, dtype=torch.long, device=edges.device)
    result[edges[:, 0], slots] = edges[:, 1]
    return result


@torch.inference_mode()
def nearest_training_proteins(encoded, anchors, top_k=32, batch_size=2048):
    if top_k < 1 or batch_size < 1:
        raise ValueError("Positive top_k and batch_size required")
    values, indices = [], []
    precise_anchors = anchors.double()
    for start in range(0, len(encoded), batch_size):
        score = (encoded[start:start + batch_size].double() @ precise_anchors.T).to(encoded.dtype)
        val, idx = stable_topk(score, top_k)
        values.append(val)
        indices.append(idx)
    return torch.cat(values), torch.cat(indices)


def enzyme_anchor_features(neighbor_values, neighbor_indices, adjacency, n_reactions, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    labels = adjacency[neighbor_indices]
    # Rowwise rescaling avoids underflow and vanishes after L2 normalization.
    weights = torch.exp((neighbor_values - neighbor_values[:, :1]) / temperature)
    weights = weights[:, :, None].expand_as(labels).reshape(len(labels), -1)
    labels = labels.reshape(len(labels), -1)
    valid = labels >= 0
    result = torch.zeros((len(labels), n_reactions), device=weights.device, dtype=weights.dtype)
    result.scatter_reduce_(1, labels.clamp_min(0), weights * valid, reduce="amax")
    return row_unit(result)


def reaction_anchor_features(encoded, anchors, temperature, top_k=16):
    if temperature <= 0 or top_k < 1:
        raise ValueError("Temperature and top_k must be positive")
    similarity = (encoded.double() @ anchors.double().T).to(encoded.dtype)
    values, indices = stable_topk(similarity, top_k)
    weights = torch.exp((values - values[:, :1]) / temperature)
    result = torch.zeros_like(similarity)
    result.scatter_(1, indices, weights)
    return row_unit(result)
