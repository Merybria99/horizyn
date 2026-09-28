"""Small, independently encoded biological blocks on frozen CIRCE features.

Missing labels are neutral. Category anchors express annotation identity, not
measured chemical distances. The normalized supervised blocks are scored
directly, with no unconstrained projection between supervision and retrieval.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from horizyn.positive_bio import BIOLOGICAL_FAMILIES, build_positive_anchors


class ScoredBiology(nn.Module):
    def __init__(self, enzyme_dim, reaction_dim, labels, block_dim=64, hidden_dim=256):
        super().__init__()
        self.block_dim = block_dim
        self.labels = labels
        def branch(dim):
            return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim),
                                 nn.GELU(), nn.Linear(hidden_dim, 3 * block_dim))
        self.enzyme = branch(enzyme_dim)
        self.reaction = branch(reaction_dim)
        for family in BIOLOGICAL_FAMILIES:
            self.register_buffer(f"anchors_{family}",
                build_positive_anchors(family, labels[family], block_dim))

    def encode(self, features, side):
        if side not in ("enzyme", "reaction"):
            raise ValueError(side)
        return F.normalize(getattr(self, side)(features).reshape(-1, 3, self.block_dim), dim=-1)

    def supervision(self, blocks, targets):
        losses = {}
        active = []
        for index, family in enumerate(BIOLOGICAL_FAMILIES):
            ids, confidence = targets[family]
            valid = (ids >= 0) & (confidence > 0)
            anchors = getattr(self, f"anchors_{family}")[ids.clamp_min(0)]
            distance = 1 - (blocks[:, index, None] * anchors).sum(-1).clamp(-1, 1)
            # Divide by observed count, NOT confidence sum: 0.4 remains weaker.
            per_row = (distance * confidence * valid).sum(-1) / valid.sum(-1).clamp_min(1)
            observed = valid.any(-1)
            loss = per_row[observed].mean() if observed.any() else blocks.sum() * 0
            losses[family] = loss
            if observed.any():
                active.append(loss)
        total = torch.stack(active).mean() if active else blocks.sum() * 0
        return total, losses


def mixed_scores(base_queries, base_targets, query_blocks, target_blocks, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    base = base_queries @ base_targets.T
    if alpha == 0:
        return base
    bio = query_blocks.flatten(1) @ target_blocks.flatten(1).T / 3
    return (1 - alpha) * base + alpha * bio


def known_positive_mask(query_ids, protein_ids, sorted_edges, num_proteins):
    """All retained training edges, not just the sampled diagonal."""
    keys = query_ids[:, None] * num_proteins + protein_ids[None, :]
    positions = torch.searchsorted(sorted_edges, keys)
    return (positions < len(sorted_edges)) & (sorted_edges[positions.clamp_max(len(sorted_edges)-1)] == keys)


@torch.no_grad()
def retrieval_metrics(base_q, base_p, bio_q, bio_p, positives, alpha, chunk_size=128):
    """Query-macro all-positive MRR, first-positive MRR and recall@10.

Candidate order must be fixed by the caller. Ties use stable candidate order
identically for the baseline and all variants. No sampled candidate pools.
"""
    totals = torch.zeros(4, device=base_q.device)
    count = 0
    for start in range(0, len(base_q), chunk_size):
        stop = min(start + chunk_size, len(base_q))
        scores = mixed_scores(base_q[start:stop], base_p, bio_q[start:stop], bio_p, alpha)
        order = scores.argsort(dim=1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(1, len(base_p)+1, device=order.device).expand_as(order))
        for local, indices in enumerate(positives[start:stop]):
            if not indices:
                continue
            positive_ranks = ranks[local, indices].float()
            totals += torch.stack((positive_ranks.reciprocal().mean(), positive_ranks.min().reciprocal(),
                                   (positive_ranks <= 10).float().mean(), (positive_ranks.min() <= 10).float()))
            count += 1
    if not count:
        raise ValueError("No evaluable positive queries")
    values = (totals / count).cpu().tolist()
    return dict(zip(("mrr", "first_mrr", "recall10", "hit10"), values), queries=count, candidates=len(base_p))
