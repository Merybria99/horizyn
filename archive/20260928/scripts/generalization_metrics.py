"""Stable, all-positive retrieval metrics for the 2026-09-19 validation screens.

The score matrix is reaction by enzyme. Each stratum retains the full candidate
universe and restricts positive edges; queries without a positive in the stratum
are omitted. In particular, an enzyme may occur in both reaction-seen strata.
"""
from __future__ import annotations

import numpy as np
import torch


def _summarize(ranks, anchors, mask, n_anchors):
    selected_ranks = ranks[mask].float()
    selected_anchors = anchors[mask]
    counts = torch.zeros(n_anchors, device=ranks.device)
    reciprocal_sum = torch.zeros_like(counts)
    first = torch.full_like(counts, float("inf"))
    counts.scatter_add_(0, selected_anchors, torch.ones_like(selected_ranks))
    reciprocal_sum.scatter_add_(0, selected_anchors, selected_ranks.reciprocal())
    first.scatter_reduce_(0, selected_anchors, selected_ranks, reduce="amin")
    valid = counts > 0
    first = first[valid]
    per_query = {
        "query_index": valid.nonzero().flatten().cpu().numpy(),
        "first_rank": first.cpu().numpy(),
        "reactzyme_mrr": (reciprocal_sum[valid] / counts[valid]).cpu().numpy(),
        "positive_count": counts[valid].cpu().numpy(),
    }
    metrics = {"num_queries": int(valid.sum().item()), "num_positive_edges": int(mask.sum().item())}
    metrics["first_positive_mrr"] = float(first.reciprocal().mean()) if first.numel() else None
    metrics["reactzyme_mrr"] = float(np.mean(per_query["reactzyme_mrr"])) if first.numel() else None
    for k in (1, 5, 10):
        hits = (first <= k).float()
        per_query[f"top_{k}"] = hits.cpu().numpy()
        metrics[f"top_{k}"] = float(hits.mean()) if first.numel() else None
    return metrics, per_query


@torch.inference_mode()
def evaluate_scores(scores, truth, batch_size=512, directions=None):
    """Return summaries and rank arrays using stable candidate-index tie breaks.

    ``truth`` requires aligned ``reaction_index`` and ``enzyme_index`` arrays.
    Optional boolean ``reaction_seen`` and ``enzyme_seen`` arrays indicate
    membership in the training graph. Empty strata report null metrics.
    """
    if scores.ndim != 2 or batch_size <= 0 or not bool(torch.isfinite(scores).all()):
        raise ValueError("Scores must be a finite matrix; batch_size must be positive")
    n_reactions, n_enzymes = scores.shape
    reaction = torch.as_tensor(truth["reaction_index"], device=scores.device, dtype=torch.long)
    enzyme = torch.as_tensor(truth["enzyme_index"], device=scores.device, dtype=torch.long)
    if reaction.ndim != 1 or enzyme.shape != reaction.shape:
        raise ValueError("Positive edge indices must be aligned one-dimensional arrays")
    if bool(((reaction < 0) | (reaction >= n_reactions) | (enzyme < 0) | (enzyme >= n_enzymes)).any()):
        raise ValueError("Positive edge index outside score matrix")
    edges = torch.unique(reaction * n_enzymes + enzyme, sorted=True)
    reaction, enzyme = edges // n_enzymes, edges % n_enzymes
    strata = {"all": torch.ones_like(reaction, dtype=torch.bool)}
    for label, index, count in (("reaction", reaction, n_reactions), ("enzyme", enzyme, n_enzymes)):
        key = f"{label}_seen"
        if key not in truth:
            continue
        seen = torch.as_tensor(truth[key], device=scores.device, dtype=torch.bool)
        if seen.shape != (count,):
            raise ValueError(f"{key} has incorrect shape")
        strata[f"seen_{label}"] = seen[index]
        strata[f"unseen_{label}"] = ~seen[index]
    output = {"summary": {}, "per_query": {}, "per_positive": {}}
    for direction, matrix, anchor, candidate in (
        ("reaction_to_enzyme", scores, reaction, enzyme),
        ("enzyme_to_reaction", scores.T, enzyme, reaction),
    ):
        if directions is not None and direction not in directions:
            continue
        ranks = torch.empty_like(anchor)
        for start in range(0, len(matrix), batch_size):
            stop = min(start + batch_size, len(matrix))
            mask = (anchor >= start) & (anchor < stop)
            if not bool(mask.any()):
                continue
            order = torch.argsort(matrix[start:stop], dim=1, descending=True, stable=True)
            inverse = torch.empty_like(order)
            inverse.scatter_(1, order, torch.arange(1, matrix.shape[1] + 1, device=scores.device).expand_as(order))
            ranks[mask] = inverse[anchor[mask] - start, candidate[mask]]
        output["summary"][direction] = {}
        output["per_query"][direction] = {}
        for label, mask in strata.items():
            summary, per_query = _summarize(ranks, anchor, mask, len(matrix))
            output["summary"][direction][label] = summary
            output["per_query"][direction][label] = per_query
        output["per_positive"][direction] = {
            "reaction_index": reaction.cpu().numpy(),
            "enzyme_index": enzyme.cpu().numpy(),
            "rank": ranks.cpu().numpy(),
        }
    return output
