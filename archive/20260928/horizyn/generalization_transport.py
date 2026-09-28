"""Training-neighborhood refinement of independently encoded endpoints.

Reference keys and values come only from training endpoints. Query and
candidate catalogs never become reference banks or fit normalization statistics.
The underlying SLEEC encoder and the phase-2 semantic coordinates are retained.
"""
from __future__ import annotations

import torch

from .semantic_anchors import row_unit, stable_topk
from .semantic_smooth import stable_neighbor_order


@torch.inference_mode()
def transport_embeddings(query_keys, training_keys, training_values, *, neighbors=32,
                         temperature=0.03, batch_size=128):
    if neighbors < 1 or temperature <= 0 or batch_size < 1:
        raise ValueError('Positive neighbors, temperature, and batch size required')
    if (query_keys.ndim != 2 or training_keys.ndim != 2 or training_values.ndim != 2
            or query_keys.shape[1] != training_keys.shape[1]
            or len(training_keys) != len(training_values) or not len(training_keys)):
        raise ValueError('Aligned nonempty training keys and values required')
    if any(not bool(torch.isfinite(x).all()) for x in (query_keys, training_keys, training_values)):
        raise ValueError('Transport inputs must be finite')
    if not len(query_keys):
        return training_values.new_empty((0, training_values.shape[1]))
    precise_keys = training_keys.double()
    results = []
    for query in query_keys.split(batch_size):
        similarity = (query.double() @ precise_keys.T).to(query.dtype)
        values, indices = stable_topk(similarity, min(neighbors, len(training_keys)))
        values, indices = stable_neighbor_order(values, indices)
        weights = torch.softmax(values.double() / temperature, dim=1)
        interpolated = (weights[:, :, None] * training_values[indices].double()).sum(1)
        results.append(row_unit(interpolated).to(training_values.dtype))
    return torch.cat(results)


def refine_embeddings(base, transported, strength):
    """A single endpoint vector; strength zero is a bitwise identity."""
    if not 0 <= strength <= 1 or base.shape != transported.shape:
        raise ValueError('Aligned endpoints and strength in [0, 1] required')
    if strength == 0:
        return base
    return row_unit((1 - strength) * base.double() + strength * transported.double()).to(base.dtype)


def refine_phase2(embedding, transported, strength, *, alpha, dense_dimension=512):
    """Refine only the learned SLEEC branch, keeping semantic features fixed."""
    if not 0 <= alpha < 1 or embedding.shape[1] < dense_dimension:
        raise ValueError('A nonzero learned phase-2 branch is required')
    if strength == 0:
        return embedding
    scale = (1 - alpha) ** .5
    refined = refine_embeddings(embedding[:, :dense_dimension] / scale, transported, strength)
    return torch.cat((refined * scale, embedding[:, dense_dimension:]), dim=1)
