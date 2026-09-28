"""Train-only, anchor-balanced linear residual alignment of two endpoint spaces."""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


@torch.inference_mode()
def graph_statistics(enzymes, reactions, edges):
    """Return sufficient statistics for equally weighted observed endpoint anchors.

    Each anchor predicts the mean of its unique positive partners. The resulting
    squared-error objective equals the mean positive-pair squared error up to a
    parameter-independent constant. No unobserved pair becomes a target.
    """
    if enzymes.ndim != 2 or reactions.ndim != 2 or enzymes.shape[1] != reactions.shape[1]:
        raise ValueError("Aligned two-dimensional endpoint spaces are required")
    if not torch.isfinite(enzymes).all() or not torch.isfinite(reactions).all():
        raise ValueError("Nonfinite training endpoints")
    edges = torch.as_tensor(edges, dtype=torch.long, device=enzymes.device)
    if edges.ndim != 2 or edges.shape[1] != 2 or not len(edges):
        raise ValueError("Nonempty reaction/enzyme training edge pairs are required")
    if (edges < 0).any() or (edges[:, 0] >= len(reactions)).any() or (edges[:, 1] >= len(enzymes)).any():
        raise ValueError("Training edge outside endpoint catalog")
    edges = torch.unique(edges, dim=0)
    e, r = enzymes.double(), reactions.double()
    result = {}
    for name, values, partners, own, other in (
        ("enzyme", e, r, edges[:, 1], edges[:, 0]),
        ("reaction", r, e, edges[:, 0], edges[:, 1]),
    ):
        counts = torch.bincount(own, minlength=len(values))
        targets = torch.zeros_like(values).index_add_(0, own, partners[other])
        present = counts > 0
        x = values[present]
        y = targets[present] / counts[present, None]
        result[name] = dict(gram=x.T @ x, cross=x.T @ (y-x), anchors=len(x), dimension=x.shape[1])
    return result


@torch.inference_mode()
def fit_alignment(statistics, relative_ridge):
    if not math.isfinite(relative_ridge) or relative_ridge <= 0:
        raise ValueError("A positive finite ridge penalty is required")
    result = {}
    for name, row in statistics.items():
        gram = row["gram"]
        penalty = relative_ridge * row["anchors"] / row["dimension"]
        result[name] = torch.linalg.solve(
            gram + penalty * torch.eye(len(gram), device=gram.device, dtype=gram.dtype), row["cross"])
    return result


@torch.inference_mode()
def align_composed(composed, delta, strength, alpha, dimension=512):
    """Change only the learned coordinates; preserve semantic coordinates exactly."""
    if not 0 <= strength <= 1 or not 0 <= alpha < 1:
        raise ValueError("Invalid residual strength or semantic composition weight")
    if strength == 0:
        return composed
    dense = composed[:, :dimension].double() / math.sqrt(1-alpha)
    learned = F.normalize(dense + strength * (dense @ delta.double()), dim=-1).to(composed.dtype)
    return torch.cat((math.sqrt(1-alpha)*learned, composed[:, dimension:]), dim=1)
