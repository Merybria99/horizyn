#!/usr/bin/env python3
"""Score F3 screening ranks with the released CLIPZyme notebook definitions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def notebook_metrics(sorted_labels: np.ndarray) -> dict[str, float]:
    """Equivalent to bedroc_score/enrichment_score in Results.ipynb cells 8-9."""
    big_n = len(sorted_labels)
    n = int(sorted_labels.sum())
    if n == 0 or n == big_n:
        raise ValueError("BEDROC needs both positive and negative candidates")
    ranks = np.flatnonzero(sorted_labels)
    r_a = n / big_n
    metrics = {}
    for alpha in (85.0, 20.0):
        s = np.exp(-alpha * ranks / big_n).sum()
        rand_sum = r_a * (1 - np.exp(-alpha)) / np.expm1(alpha / big_n)
        fac = r_a * np.sinh(alpha / 2) / (
            np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * r_a))
        cte = 1 / (1 - np.exp(alpha * (1 - r_a)))
        metrics[f"bedroc{int(alpha)}"] = float(s * fac / rand_sum + cte)
    for chi in (0.05, 0.1):
        k = math.floor(chi * big_n)
        metrics[f"ef{chi}"] = float(sorted_labels[:k].sum() / (chi * n))
    return metrics


def evaluate_query(arguments):
    """Exact notebook ranking per query; safe for ordered thread-pool mapping."""
    index, key, values, positive_indices, kept = arguments
    if not len(positive_indices):
        return None
    if not np.isfinite(values).all():
        raise ValueError(f'Nonfinite validation scores for query {key}')
    labels = np.zeros(len(values), dtype=bool)
    labels[positive_indices] = True
    table1 = notebook_metrics(labels[np.argsort(-values)])
    reduced = labels[kept]
    table2 = (notebook_metrics(reduced[np.argsort(-values[kept])])
              if reduced.any() else None)
    return dict(query_index=index, reaction_id=key,
                positives_table1=int(labels.sum()), positives_table2=int(reduced.sum()),
                table1=table1, table2=table2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True,
                        help="float32 .npy matrix, query x ordered candidate")
    parser.add_argument("--query-ids", type=Path, required=True)
    parser.add_argument("--candidate-ids", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric-workers", type=int, default=8)
    args = parser.parse_args()
    if args.metric_workers < 1:
        parser.error("metric-workers must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads((args.protocol / "receipt.json").read_text())
    if (protocol["schema"] != "clipzyme_screening_notebook_protocol_v1"
            or sha256(args.protocol / "queries.csv") != protocol["queries_sha256"]
            or sha256(args.protocol / "query_inputs.csv") != protocol["query_inputs_sha256"]
            or sha256(args.protocol / "train_uniprot_ids.txt") != protocol["train_uniprot_ids_sha256"]):
        raise ValueError("Official screening query protocol changed")
    with (args.protocol / "queries.csv").open(newline="") as handle:
        queries = list(csv.DictReader(handle))
    query_ids = args.query_ids.read_text().splitlines()
    candidate_ids = args.candidate_ids.read_text().splitlines()
    if (query_ids != [row["reaction_id"] for row in queries]
            or len(candidate_ids) != 261907 or len(set(candidate_ids)) != len(candidate_ids)):
        raise ValueError("Score matrix axes do not match official screening protocol")
    scores = np.load(args.scores, mmap_mode="r")
    if scores.shape != (1521, 261907) or scores.dtype != np.float32:
        raise ValueError("Expected full float32 score matrix on fixed screening axes")
    train_ids = set((args.protocol / "train_uniprot_ids.txt").read_text().splitlines())
    candidate_index = {key: i for i, key in enumerate(candidate_ids)}
    kept = np.array([i for i, key in enumerate(candidate_ids) if key not in train_ids], dtype=np.int32)
    if len(kept) != protocol["table2_candidate_ids"]:
        raise ValueError("Training-enzyme exclusion pool changed")
    args.output.mkdir(parents=True)
    arguments = []
    for i, row in enumerate(queries):
        positives = json.loads(row["positive_uniprot_ids_json"])
        if not positives:
            raise ValueError("Test query has no positive candidates")
        arguments.append((i, row["reaction_id"], np.asarray(scores[i]),
                          np.asarray([candidate_index[key] for key in positives], dtype=np.int64), kept))
    result_rows = []
    with ThreadPoolExecutor(max_workers=args.metric_workers) as pool:
        for i, row in enumerate(pool.map(evaluate_query, arguments)):
            result_rows.append(row)
            if (i + 1) % 100 == 0:
                print(json.dumps({"evaluated_queries": i + 1}), flush=True)
    if (len(result_rows) != protocol["table1_queries"]
            or sum(row["positives_table1"] for row in result_rows)
               != protocol["table1_candidate_positive_labels"]
            or sum(row["table2"] is not None for row in result_rows)
               != protocol["table2_queries_with_positive"]):
        raise ValueError("Screening positives or query denominator changed")
    summary = {}
    for table in ("table1", "table2"):
        rows = [row[table] for row in result_rows if row[table] is not None]
        summary[table] = {"queries": len(rows), "candidate_ids":
                          len(candidate_ids) if table == "table1" else len(kept),
                          **{name: float(np.mean([row[name] for row in rows]))
                             for name in ("bedroc85", "bedroc20", "ef0.05", "ef0.1")}}
    per_query = args.output / "per_query.jsonl"
    per_query.write_text("".join(json.dumps(row) + "\n" for row in result_rows))
    receipt = {"schema": "clipzyme_f3_official_screening_evaluation_v1",
               "protocol_receipt_sha256": sha256(args.protocol / "receipt.json"),
               "scores_sha256": sha256(args.scores),
               "query_ids_sha256": sha256(args.query_ids),
               "candidate_ids_sha256": sha256(args.candidate_ids),
               "per_query_sha256": sha256(per_query),
               "notebook_sha256": protocol["source_notebook_sha256"],
               "selection_used_test_scores": False,
               "retrospective_test_metadata_previously_opened": True,
               "summary": summary,
               "metric_workers": args.metric_workers,
               "source_sha256": sha256(Path(__file__))}
    (args.output / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
