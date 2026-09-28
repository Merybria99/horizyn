"""Evaluate aligned endpoint vectors without re-encoding or changing the pool."""

from collections import defaultdict

import numpy as np
import torch

from horizyn.benchmarks.retrieval import mean_dict, rank_metrics_for_query, screening_metrics_for_query
from .scoring import canonical_dot


def evaluate_embeddings(reactions, enzymes, edges, *, protocol="reactzyme", batch_size=128):
    """Return official per-query metrics with stable candidate-index tie breaks.

    `edges` indexes the supplied, explicitly ordered reaction/enzyme banks.
    ReactZyme uses both retrieval directions. Screening uses R→E with
    BEDROC85, BEDROC20, EF5 and EF10; MRR is not its selection metric.
    """
    if protocol not in {"reactzyme", "screening"} or batch_size < 1:
        raise ValueError("Choose reactzyme or screening and a positive batch size")
    if reactions.ndim != 2 or enzymes.ndim != 2 or reactions.shape[1] != enzymes.shape[1]:
        raise ValueError("Reaction/enzyme banks must have matching embedding dimensions")
    if not torch.isfinite(reactions).all() or not torch.isfinite(enzymes).all():
        raise ValueError("Embedding banks must be finite")
    if len(reactions) == 0 or len(enzymes) == 0:
        raise ValueError("Candidate banks must not be empty")
    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("Positive edges must be integer (reaction, enzyme) pairs")
    if not len(edges) or np.any(edges < 0) or np.any(edges[:, 0] >= len(reactions)) or np.any(edges[:, 1] >= len(enzymes)):
        raise ValueError("Positive edges must be nonempty and index the supplied banks")
    directions = [("reaction_to_enzyme", reactions, enzymes, edges)]
    if protocol == "reactzyme":
        directions.append(("enzyme_to_reaction", enzymes, reactions, edges[:, ::-1]))
    results = {}
    for name, queries, candidates, directed_edges in directions:
        positives = defaultdict(set)
        for q, p in directed_edges:
            positives[int(q)].add(int(p))
        query_ids = sorted(positives)
        metrics = []
        for start in range(0, len(query_ids), batch_size):
            selected = query_ids[start:start + batch_size]
            scores = canonical_dot(queries[selected], candidates)
            for index, query in enumerate(selected):
                indices = sorted(positives[query])
                if protocol == "screening":
                    row = screening_metrics_for_query(scores[index], indices, (85.0, 20.0), (0.05, 0.10))
                else:
                    row = rank_metrics_for_query(scores[index], indices, (1, 2, 3, 4, 5, 10, 20, 50),
                                                 metric_protocol="reactzyme")
                metrics.append(row)
        results[name] = {"num_queries": len(query_ids), "candidate_count": len(candidates),
                         "metrics": mean_dict(metrics)}
    return results
