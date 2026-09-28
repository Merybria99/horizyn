"""Small independent residual encoders for frozen retrieval representations.

The full training association graph is used by the experimental trainer. This
module does not change the F3/CIRCE base models or their checkpoint schemas.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FrozenGeometryResidual(nn.Module):
    """Two independently encodable residuals with exact identity initialization."""

    def __init__(self, dimension=512, hidden=1024, scale=0.2):
        super().__init__()
        self.dimension, self.hidden, self.scale = dimension, hidden, float(scale)
        self.enzyme = self._tower(dimension, hidden)
        self.reaction = self._tower(dimension, hidden)

    @staticmethod
    def _tower(dimension, hidden):
        tower = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, hidden),
                              nn.GELU(), nn.Linear(hidden, dimension))
        nn.init.zeros_(tower[-1].weight)
        nn.init.zeros_(tower[-1].bias)
        return tower

    def encode_enzymes(self, base):
        return F.normalize(base + self.scale * self.enzyme(base), dim=-1)

    def encode_reactions(self, base):
        return F.normalize(base + self.scale * self.reaction(base), dim=-1)


def full_graph_contrastive_loss(logits, reaction_index, enzyme_index,
                                enzyme_weighting="uniform"):
    """Bidirectional cross entropy against uniform *known-positive* targets.

    Every unique train reaction and enzyme is present in the denominator.
    Edges must be unique. A protein may have several positive reactions. The
    optional reaction-balanced enzyme weighting gives equal total E->R weight
    to each reaction, divided among its observed edges; it does not sample or
    access held-out labels. All reactions receive equal R->E weight.
    """
    nr, ne = logits.shape
    r, e = reaction_index, enzyme_index
    r_count = torch.bincount(r, minlength=nr).to(logits)
    e_count = torch.bincount(e, minlength=ne).to(logits)
    if torch.any(r_count == 0) or torch.any(e_count == 0):
        raise ValueError("Every training node must have at least one positive")
    positive = logits[r, e]
    r_positive = torch.zeros(nr, device=logits.device, dtype=logits.dtype)
    e_positive = torch.zeros(ne, device=logits.device, dtype=logits.dtype)
    r_positive.scatter_add_(0, r, positive)
    e_positive.scatter_add_(0, e, positive)
    r_loss = (torch.logsumexp(logits, dim=1) - r_positive / r_count).mean()
    e_losses = torch.logsumexp(logits, dim=0) - e_positive / e_count
    if enzyme_weighting == "uniform":
        e_loss = e_losses.mean()
    elif enzyme_weighting == "reaction_balanced":
        weights = torch.zeros(ne, device=logits.device, dtype=logits.dtype)
        weights.scatter_add_(0, e, r_count[r].reciprocal())
        e_loss = (e_losses * weights).sum() / weights.sum()
    else:
        raise ValueError(f"Unknown enzyme weighting: {enzyme_weighting}")
    return (r_loss + e_loss) / 2, r_loss.detach(), e_loss.detach()
