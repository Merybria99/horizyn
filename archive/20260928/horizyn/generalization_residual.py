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


def sampled_smooth_ap_loss(scores, positive_mask, anchor_indices, temperature=0.05):
    """Uniform-anchor Smooth-AP, comparing every positive to the whole bank.

    Anchor subsampling bounds memory; candidates and positives are not sampled.
    Unannotated associations remain unknowns used as contrastive negatives.
    """
    if temperature <= 0 or anchor_indices.numel() == 0:
        raise ValueError("Positive temperature and nonempty anchors required")
    selected = scores[anchor_indices]
    mask = positive_mask[anchor_indices]
    row, col = mask.nonzero(as_tuple=True)
    counts = mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("Every sampled anchor must have a known positive")
    comparisons = torch.sigmoid((selected[row] - selected[row, col, None]) / temperature)
    # Sigmoid(self-self)=1/2; +1/2 makes the self rank exactly one.
    all_rank = 0.5 + comparisons.sum(dim=1)
    positive_rank = 0.5 + (comparisons * mask[row]).sum(dim=1)
    per_positive_ap = positive_rank / all_rank
    anchor_ap = torch.zeros(len(anchor_indices), device=scores.device, dtype=scores.dtype)
    anchor_ap.scatter_add_(0, row, per_positive_ap)
    return 1 - (anchor_ap / counts).mean()


def full_graph_decoupled_loss(logits, positive_mask, reaction_index, enzyme_index):
    """Anchor-balanced decoupled InfoNCE with all train positives excluded from negatives."""
    r, e = reaction_index, enzyme_index
    negatives = logits.masked_fill(positive_mask, torch.finfo(logits.dtype).min)
    positive = logits[r, e]

    def reduce(log_mass, indices, n, bank_size):
        counts = torch.bincount(indices, minlength=n)
        sums = logits.new_zeros(n)
        sums.scatter_add_(0, indices, F.softplus(log_mass[indices] - positive))
        valid = (counts > 0) & (counts < bank_size)
        return (sums / counts.clamp_min(1)).masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1)

    r_loss = reduce(torch.logsumexp(negatives, dim=1), r, logits.shape[0], logits.shape[1])
    e_loss = reduce(torch.logsumexp(negatives, dim=0), e, logits.shape[1], logits.shape[0])
    return (r_loss + e_loss) / 2, r_loss.detach(), e_loss.detach()
