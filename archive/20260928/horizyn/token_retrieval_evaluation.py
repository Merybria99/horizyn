"""Adapt B1/B2 scores to CIRCE-V2's existing validation and test evaluators.

The encoders and pair scores belong to B1/B2. Candidate selection, forward
query IDs, positive lookups, and ranking metrics come from CIRCE's code. Its
validation and standalone test routines retain their distinct candidate and
positive-set protocols; both break score ties in original candidate order.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import inspect
from types import SimpleNamespace

import numpy as np
import torch

from horizyn.data_module import HorizynDataModule
from horizyn.datasets.csv import CSVDataset
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from horizyn.reaction_conditioned_data_module import _load_id_list
from scripts.evaluate_protein_pooling import (
    CONFIGURED_FORWARD_CANDIDATES, HIT_RATE_CUTOFFS, PAPER_TEST_CANDIDATES,
    append_retrieval_metrics, build_forward_pairs, mean_metric_results,
    select_candidate_keys,
)

EVALUATOR_ID = "circe_v2_shared_v2_stable_ties"
VALIDATION_TOP_K = (1, 10, 100, 1000)
VALIDATION_BATCH_SIZE = 512


def evaluator_provenance():
    functions = [HorizynDataModule._augment_pairs_forward_only,
                 HorizynDataModule._build_pair_lookup_maps,
                 ProteinPooledLitModule._batched_retrieval_metric_values,
                 append_retrieval_metrics, build_forward_pairs, select_candidate_keys,
                 mean_metric_results]
    return {"implementation": EVALUATOR_ID, "functions": {
        f"{fn.__module__}.{fn.__qualname__}": hashlib.sha256(
            inspect.getsource(fn).encode()).hexdigest() for fn in functions}}


@dataclass
class CirceEvaluationPlan:
    split: str
    query_ids: list[str]
    candidate_ids: list[str]
    raw_query_ids: dict[str, str]
    query_to_targets: dict[str, list[str]]
    target_to_queries: dict[str, list[str]]
    candidate_stats: dict[str, int]

    @classmethod
    def from_protocol(cls, protocol, split):
        if split not in {"validation", "test"}:
            raise ValueError(f"Unsupported evaluation split: {split}")
        raw = CSVDataset(
            file_path=str(protocol.split_dir / f"{split}_pairs.csv"), key_column="pr_id",
            columns=["reaction_id", "protein_id"],
            rename_map={"reaction_id": "query_id", "protein_id": "target_id"})
        raw_ids = {f"{raw[key]['query_id']}_f": raw[key]["query_id"] for key in raw.keys}
        target_ids = [raw[key]["target_id"] for key in raw.keys]
        configured = _load_id_list(protocol.split_dir / f"{split}_candidate_ids.txt")
        candidates, stats = select_candidate_keys(
            evaluation_protocol=PAPER_TEST_CANDIDATES if split == "test" else CONFIGURED_FORWARD_CANDIDATES,
            available_keys=list(protocol.sequences), configured_candidate_ids=configured,
            test_target_ids=target_ids)
        if stats["missing_count"] or set(target_ids) - set(candidates):
            raise ValueError("B1/B2 must cover CIRCE's complete candidate/positive set")
        if set(raw_ids.values()) - set(protocol.smiles):
            raise ValueError("B1/B2 must cover every CIRCE reaction query")
        if split == "validation":
            pairs = HorizynDataModule._augment_pairs_forward_only(raw)
            q2e, e2q = HorizynDataModule._build_pair_lookup_maps(pairs)
        else:
            pairs = build_forward_pairs(raw)
            # Preserve the standalone evaluator's list semantics, including
            # any duplicate positive rows (validation uses sets instead).
            q2e, e2q = defaultdict(list), defaultdict(list)
            for key in pairs.keys:
                pair = pairs[key]
                q2e[pair["query_id"]].append(pair["target_id"])
            for q, targets in q2e.items():
                for target in targets:
                    e2q[target].append(q)
        return cls(split, sorted(q2e), candidates, raw_ids, dict(q2e), dict(e2q), stats)


def circe_directional_metrics(scores, anchor_ids, candidate_ids, positives, split):
    """Call the production CIRCE routines, with no alternate ranking formula."""
    if scores.shape != (len(anchor_ids), len(candidate_ids)):
        raise ValueError("Score matrix is not aligned with the evaluation IDs")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("Non-finite retrieval scores")
    candidate_lookup = {identifier: i for i, identifier in enumerate(candidate_ids)}
    valid_ids = [q for q in anchor_ids if any(e in candidate_lookup for e in positives.get(q, []))]
    if valid_ids != anchor_ids:
        raise ValueError("Evaluation anchors must have an available ground-truth positive")
    per_query = defaultdict(list)
    if split == "validation":
        # These are the only model attributes read by CIRCE's metric method;
        # the actual CIRCE backbone is not instantiated or substituted for B1/B2.
        context = SimpleNamespace(retrieval_metric_top_k=VALIDATION_TOP_K,
                                  metric_functionals={"r_precision", "avg_precision"})
        for start in range(0, len(anchor_ids), VALIDATION_BATCH_SIZE):
            ids = anchor_ids[start:start + VALIDATION_BATCH_SIZE]
            count, values = ProteinPooledLitModule._batched_retrieval_metric_values(
                context, scores[start:start + len(ids)], ids, positives, candidate_lookup)
            if count != len(ids):
                raise ValueError("CIRCE validation unexpectedly excluded an anchor")
            for key, value in values.items():
                per_query[key].extend(value.cpu().tolist())
        per_query["first_positive_mrr"] = list(per_query["mrr"])
    elif split == "test":
        for row, identifier in enumerate(anchor_ids):
            indices = torch.tensor([candidate_lookup[e] for e in positives[identifier]
                                    if e in candidate_lookup], device=scores.device, dtype=torch.long)
            append_retrieval_metrics(per_query, scores[row], indices)
    else:
        raise ValueError(split)
    metrics = mean_metric_results(per_query)
    metrics["num_queries"] = len(anchor_ids)
    return metrics, {key: np.asarray(values) for key, values in per_query.items()}


def evaluate_circe_score_matrix(scores, plan, output_prefix=None):
    """Evaluate a full, ordered B1/B2 score grid with CIRCE's two directions."""
    enzyme_queries = sorted(plan.target_to_queries)
    eindex = {e: i for i, e in enumerate(plan.candidate_ids)}
    enzyme_rows = torch.tensor([eindex[e] for e in enzyme_queries], device=scores.device)
    result = {}
    for direction, matrix, anchors, candidates, positives in [
        ("reaction_to_enzyme", scores, plan.query_ids, plan.candidate_ids, plan.query_to_targets),
        ("enzyme_to_reaction", scores.T.index_select(0, enzyme_rows), enzyme_queries,
         plan.query_ids, plan.target_to_queries),
    ]:
        metrics, per_query = circe_directional_metrics(matrix, anchors, candidates, positives, plan.split)
        result.update({f"{direction}/{key}": value for key, value in metrics.items()})
        if output_prefix is not None:
            np.savez_compressed(f"{output_prefix}_{direction}_per_query.npz",
                                query_ids=np.asarray(anchors), candidate_ids=np.asarray(candidates), **per_query)
    for metric in ["mrr", "reactzyme_mrr", "top_1"]:
        first, second = [result[f"{d}/{metric}"] for d in ["reaction_to_enzyme", "enzyme_to_reaction"]]
        arithmetic = (first + second) / 2
        result[f"mean_bidirectional_{metric}"] = arithmetic
        result[f"balanced_{metric}"] = (
            ProteinPooledLitModule._harmonic_mean(torch.tensor(first), torch.tensor(second)).item()
            if plan.split == "validation" else arithmetic)
    # Explicit arithmetic aliases retained for the campaign report and test JSON.
    result["balanced_first_positive_mrr"] = result["mean_bidirectional_mrr"]
    result["arithmetic_reactzyme_mrr"] = result["mean_bidirectional_reactzyme_mrr"]
    result.update(evaluation_implementation=EVALUATOR_ID,
                  evaluator_provenance=evaluator_provenance(),
                  evaluation_protocol=PAPER_TEST_CANDIDATES if plan.split == "test" else "custom_validation_candidates",
                  ranking="stable_descending_candidate_order",
                  ground_truth_pairs=f"{plan.split}_pairs_only", reaction_query_expansion="canonical_forward_only",
                  num_enzyme_candidates=len(plan.candidate_ids), num_reaction_candidates=len(plan.query_ids),
                  num_targets=len(plan.candidate_ids), direction="both",
                  hit_rate_cutoffs=list(HIT_RATE_CUTOFFS if plan.split == "test" else VALIDATION_TOP_K),
                  configured_candidate_count=plan.candidate_stats["configured_count"],
                  test_candidate_count=plan.candidate_stats["test_count"], full_candidate_coverage=True)
    return result
