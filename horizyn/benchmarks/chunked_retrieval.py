"""Exact positive ranks without allocating a query-by-catalog score matrix.

Ties are resolved by ascending candidate row, making results reproducible even
for identical embeddings. Candidate tensors may live on CPU (including mmap).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


@torch.inference_mode()
def chunked_positive_ranks(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    positives: Sequence[Sequence[int]],
    *,
    chunk_size: int = 32768,
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    if chunk_size <= 0 or queries.ndim != 2 or candidates.ndim != 2:
        raise ValueError("Positive chunk_size and two-dimensional embeddings required")
    if queries.shape[1] != candidates.shape[1] or len(positives) != len(queries):
        raise ValueError("Embedding dimensions or positive-row count mismatch")
    score_fn = score_fn or (lambda q, c: q.float() @ c.float().T)
    ids = [torch.tensor(sorted(set(p)), device=queries.device, dtype=torch.long) for p in positives]
    for row in ids:
        if row.numel() and (row.min() < 0 or row.max() >= len(candidates)):
            raise ValueError("Positive index outside candidate catalog")
    positive_scores = [queries.new_empty(len(row), dtype=torch.float32) for row in ids]
    # Read positive scores through the same block scorer used for ranking. A
    # separate elementwise dot product can differ by an ulp and mis-rank itself.
    for start in range(0, len(candidates), chunk_size):
        end = min(start + chunk_size, len(candidates))
        masks = [(row >= start) & (row < end) for row in ids]
        if not any(bool(mask.any()) for mask in masks):
            continue
        scores = score_fn(queries, candidates[start:end].to(queries.device)).float()
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite retrieval scores")
        for i, mask in enumerate(masks):
            positive_scores[i][mask] = scores[i, ids[i][mask] - start]
    ranks = [torch.ones_like(row) for row in ids]
    for start in range(0, len(candidates), chunk_size):
        end = min(start + chunk_size, len(candidates))
        scores = score_fn(queries, candidates[start:end].to(queries.device)).float()
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite retrieval scores")
        values, order = scores.sort(dim=1, stable=True)
        inverse = torch.empty_like(order)
        inverse.scatter_(1, order, torch.arange(end - start, device=order.device).expand_as(order))
        for i, row in enumerate(ids):
            if not row.numel():
                continue
            low = torch.searchsorted(values[i].contiguous(), positive_scores[i], right=False)
            high = torch.searchsorted(values[i].contiguous(), positive_scores[i], right=True)
            tied_before = torch.where(row >= end, high - low, torch.zeros_like(low))
            within = (row >= start) & (row < end)
            tied_before[within] = inverse[i, row[within] - start] - low[within]
            ranks[i] += end - start - high + tied_before
    return ranks


def rank_metrics(ranks: Sequence[torch.Tensor], top_k: Sequence[int]) -> dict[str, torch.Tensor]:
    """Macro per-query metrics; rows without an in-catalog positive are omitted."""
    valid = [r.sort().values.float() for r in ranks if r.numel()]
    if not valid:
        return {}
    output = {
        "mrr": torch.stack([r[0].reciprocal() for r in valid]),
        "mean_rank": torch.stack([r[0] for r in valid]),
        "reactzyme_mrr": torch.stack([r.reciprocal().mean() for r in valid]),
        "r_precision": torch.stack([(r <= len(r)).float().mean() for r in valid]),
        "avg_precision": torch.stack([
            (torch.arange(1, len(r) + 1, device=r.device) / r).mean() for r in valid
        ]),
    }
    for k in top_k:
        if k <= 0:
            raise ValueError("top_k must contain positive integers")
        output[f"top_{k}"] = torch.stack([(r[0] <= k).float() for r in valid])
        output[f"recall_{k}"] = torch.stack([(r <= k).float().mean() for r in valid])
    return output
