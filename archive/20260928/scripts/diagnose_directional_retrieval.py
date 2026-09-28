#!/usr/bin/env python3
"""Directional retrieval diagnostics for bimodal enzyme-reaction models.

This script evaluates enzyme -> reaction and reaction -> enzyme retrieval on
triplet data with multiple positives per query. It is intentionally standalone:
it only requires pandas, numpy, and matplotlib for plots. Optional analyses are
skipped when their inputs are not provided.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import pickle
import re
import sys
import textwrap
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting should be optional
    plt = None


KS = (1, 5, 10)
TRIPLET_COLUMNS = ("enzyme_id", "reaction_id", "ec")
LONG_SCORE_COLUMNS = ("enzyme_id", "reaction_id", "score")
UNKNOWN_EC_VALUES = {"", "-", "?", "nan", "none", "null", "na", "n/a"}


# ---------------------------------------------------------------------------
# Generic IO helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_table_auto(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return normalize_columns(pd.read_parquet(path))
    if suffix in {".pkl", ".pickle"}:
        return normalize_columns(pd.read_pickle(path))
    if suffix in {".json", ".jsonl"}:
        try:
            return normalize_columns(pd.read_json(path, lines=suffix == ".jsonl"))
        except ValueError:
            return normalize_columns(pd.read_json(path))
    return normalize_columns(pd.read_csv(path, sep=None, engine="python"))


def split_ec_values(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = re.split(r"[;,|]", text)
    return [part.strip() for part in parts if part.strip()]


def read_ec_source(path: Optional[str | Path]) -> Dict[str, Set[str]]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    df = read_table_auto(source)
    seq_col = next((c for c in ("Sequence", "sequence", "protein_sequence") if c in df.columns), None)
    ec_col = next((c for c in ("EC number", "ec_number", "ec", "EC") if c in df.columns), None)
    if seq_col is None or ec_col is None:
        return {}
    mapping: Dict[str, Set[str]] = defaultdict(set)
    for row in df[[seq_col, ec_col]].itertuples(index=False):
        seq = "" if pd.isna(row[0]) else str(row[0]).strip()
        if not seq:
            continue
        for ec in split_ec_values(row[1]):
            mapping[seq].add(ec)
    return dict(mapping)


def read_triplets(path: str | Path, name: str, ec_by_sequence: Optional[Mapping[str, Set[str]]] = None) -> pd.DataFrame:
    df = read_table_auto(path)
    enzyme_col = next((c for c in ("enzyme_id", "protein_id", "uniprot_id", "Entry") if c in df.columns), None)
    reaction_col = next((c for c in ("reaction_id", "rxn_id", "Rhea ID", "rhea_id") if c in df.columns), None)
    ec_col = next((c for c in ("ec", "ec_number", "EC number", "EC") if c in df.columns), None)
    if enzyme_col is None or reaction_col is None:
        raise ValueError(
            f"{name} triplets must contain enzyme/protein and reaction columns; "
            f"available columns={list(df.columns)}"
        )
    out = pd.DataFrame(
        {
            "enzyme_id": df[enzyme_col],
            "reaction_id": df[reaction_col],
        }
    )
    if ec_col is not None:
        out["ec"] = df[ec_col]
    elif ec_by_sequence and "protein_sequence" in df.columns:
        out["ec"] = df["protein_sequence"].map(
            lambda seq: ";".join(sorted(ec_by_sequence.get(str(seq).strip(), set())))
            if not pd.isna(seq)
            else ""
        )
    else:
        out["ec"] = ""
    df = out
    df["enzyme_id"] = df["enzyme_id"].map(normalize_id)
    df["reaction_id"] = df["reaction_id"].map(normalize_id)
    df["ec"] = df["ec"].map(lambda x: "" if pd.isna(x) else str(x).strip())
    df = df[(df["enzyme_id"] != "") & (df["reaction_id"] != "")].copy()
    exploded_rows = []
    for row in df.itertuples(index=False):
        ecs = split_ec_values(row.ec)
        if not ecs:
            exploded_rows.append(
                {"enzyme_id": row.enzyme_id, "reaction_id": row.reaction_id, "ec": ""}
            )
        else:
            for ec in ecs:
                exploded_rows.append(
                    {"enzyme_id": row.enzyme_id, "reaction_id": row.reaction_id, "ec": ec}
                )
    return pd.DataFrame(exploded_rows).drop_duplicates().reset_index(drop=True)


def read_scores(path: str | Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return scores as enzyme rows x reaction columns."""
    meta: Dict[str, Any] = {"source": str(path)}
    path = Path(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        required = {"scores", "enzyme_ids", "reaction_ids"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Score npz is missing keys: {sorted(missing)}")
        scores = np.asarray(data["scores"], dtype=np.float32)
        enzyme_ids = [normalize_id(x) for x in data["enzyme_ids"].tolist()]
        reaction_ids = [normalize_id(x) for x in data["reaction_ids"].tolist()]
        if scores.shape != (len(enzyme_ids), len(reaction_ids)):
            raise ValueError(
                f"Score npz shape {scores.shape} does not match "
                f"{len(enzyme_ids)} enzyme ids x {len(reaction_ids)} reaction ids"
            )
        meta.update({"format": "npz_dense", "rows": "enzyme_id", "columns": "reaction_id"})
        return pd.DataFrame(scores, index=enzyme_ids, columns=reaction_ids).sort_index().sort_index(axis=1), meta

    df = read_table_auto(path)
    cols = set(df.columns)

    if all(c in cols for c in LONG_SCORE_COLUMNS):
        long_df = df.loc[:, list(LONG_SCORE_COLUMNS)].copy()
        long_df["enzyme_id"] = long_df["enzyme_id"].map(normalize_id)
        long_df["reaction_id"] = long_df["reaction_id"].map(normalize_id)
        long_df["score"] = pd.to_numeric(long_df["score"], errors="coerce")
        long_df = long_df.dropna(subset=["score"])
        long_df = long_df[(long_df["enzyme_id"] != "") & (long_df["reaction_id"] != "")]
        score_matrix = long_df.pivot_table(
            index="enzyme_id",
            columns="reaction_id",
            values="score",
            aggfunc="max",
        )
        score_matrix.index = score_matrix.index.map(str)
        score_matrix.columns = score_matrix.columns.map(str)
        meta.update(
            {
                "format": "long",
                "n_score_rows": int(len(long_df)),
                "duplicate_pair_policy": "max score",
            }
        )
        return score_matrix.sort_index().sort_index(axis=1), meta

    if len(df.columns) < 2:
        raise ValueError(
            "Dense score matrix must have one enzyme-id column plus one or more reaction score columns."
        )

    id_col = None
    for candidate in ("enzyme_id", "enzyme", "protein_id", "id", "Unnamed: 0"):
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        id_col = df.columns[0]

    dense = df.copy()
    enzyme_ids = dense[id_col].map(normalize_id)
    dense = dense.drop(columns=[id_col])
    dense.index = enzyme_ids
    dense.columns = [normalize_id(c) for c in dense.columns]
    dense = dense.apply(pd.to_numeric, errors="coerce")
    dense = dense.loc[dense.index != ""]
    dense = dense.loc[:, [c for c in dense.columns if c != ""]]
    meta.update({"format": "dense", "id_column": str(id_col)})
    return dense.sort_index().sort_index(axis=1), meta


def parse_embedding_value(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return np.asarray([], dtype=float)
        try:
            parsed = json.loads(stripped)
        except Exception:
            parsed = ast.literal_eval(stripped)
        return np.asarray(parsed, dtype=float)
    return np.asarray([], dtype=float)


def read_embeddings(path: str | Path, id_candidates: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    meta: Dict[str, Any] = {"source": str(path)}

    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        keys = set(data.files)
        id_key = next((k for k in ("ids", "enzyme_ids", "reaction_ids", "id") if k in keys), None)
        emb_key = next((k for k in ("embeddings", "embedding", "vectors", "arr_0") if k in keys), None)
        if id_key is None or emb_key is None:
            raise ValueError(f"Could not infer id/embedding keys from npz file {path}; keys={sorted(keys)}")
        ids = [normalize_id(x) for x in data[id_key].tolist()]
        arr = np.asarray(data[emb_key], dtype=float)
        meta.update({"format": "npz", "id_key": id_key, "embedding_key": emb_key})
        return pd.DataFrame(arr, index=ids).sort_index(), meta

    if suffix == ".npy":
        raise ValueError("Raw .npy embeddings do not contain IDs. Use csv/parquet/npz/pickle with IDs.")

    if suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as handle:
            obj = pickle.load(handle)
        if isinstance(obj, dict):
            ids = [normalize_id(k) for k in obj.keys()]
            arr = np.vstack([parse_embedding_value(v) for v in obj.values()])
            meta.update({"format": "pickle_dict"})
            return pd.DataFrame(arr, index=ids).sort_index(), meta
        if isinstance(obj, pd.DataFrame):
            df = normalize_columns(obj)
        else:
            raise ValueError(f"Unsupported pickle embedding object type: {type(obj)}")
    else:
        df = read_table_auto(path)

    id_col = next((c for c in id_candidates if c in df.columns), None)
    if id_col is None:
        id_col = next((c for c in ("id", "protein_id", "enzyme", "reaction") if c in df.columns), None)
    if id_col is None:
        id_col = df.columns[0]

    ids = df[id_col].map(normalize_id)
    if "embedding" in df.columns:
        arr = np.vstack(df["embedding"].map(parse_embedding_value).to_numpy())
    elif "embeddings" in df.columns:
        arr = np.vstack(df["embeddings"].map(parse_embedding_value).to_numpy())
    else:
        numeric = df.drop(columns=[id_col]).apply(pd.to_numeric, errors="coerce")
        numeric = numeric.loc[:, numeric.notna().any(axis=0)]
        arr = numeric.to_numpy(dtype=float)
    meta.update({"format": suffix.lstrip(".") or "table", "id_column": str(id_col)})
    return pd.DataFrame(arr, index=ids).sort_index(), meta


def write_table(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def df_to_markdown(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_No rows._"
    shown = df.head(max_rows).copy()

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            if np.isfinite(value):
                return f"{float(value):.6g}"
        text = str(value)
        text = text.replace("\n", "<br>")
        text = text.replace("|", "\\|")
        return text

    note = ""
    if len(df) > len(shown):
        note = f"\n\n_Showing first {len(shown)} of {len(df)} rows. Full CSV tables are in `tables/`._"

    if shown.shape[1] <= 8:
        headers = [str(col).replace("|", "\\|") for col in shown.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in shown.iterrows():
            lines.append("| " + " | ".join(fmt(row[col]) for col in shown.columns) + " |")
        return "\n".join(lines) + note

    identity_cols = [
        "split",
        "direction",
        "positive_bin",
        "k",
        "rank",
        "ranked_factor",
        "classification",
        "metric",
    ]
    blocks: List[str] = []
    for row_number, (_, row) in enumerate(shown.iterrows(), start=1):
        title_parts = []
        for col in identity_cols:
            if col in shown.columns and not pd.isna(row[col]):
                title_parts.append(f"{col}={fmt(row[col])}")
        title = ", ".join(title_parts) if title_parts else f"row={row_number}"
        lines = [
            f"#### {title}",
            "",
            "| Field | Value |",
            "| --- | --- |",
        ]
        for col in shown.columns:
            field = str(col).replace("|", "\\|")
            lines.append(f"| {field} | {fmt(row[col])} |")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + note


def save_json(data: Mapping[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# EC parsing and matching
# ---------------------------------------------------------------------------


def normalize_ec(ec: Any) -> str:
    if pd.isna(ec):
        return ""
    text = str(ec).strip()
    text = re.sub(r"^EC[:\s]*", "", text, flags=re.IGNORECASE)
    text = text.replace("_", ".")
    text = re.sub(r"\s+", "", text)
    return text


def ec_parts(ec: Any) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    text = normalize_ec(ec)
    if text.lower() in UNKNOWN_EC_VALUES:
        return (None, None, None, None)
    pieces = text.split(".")
    pieces = (pieces + [""] * 4)[:4]
    out: List[Optional[str]] = []
    for part in pieces:
        part = part.strip()
        if part.lower() in UNKNOWN_EC_VALUES:
            out.append(None)
        else:
            out.append(part)
    return tuple(out)  # type: ignore[return-value]


def ec_known_depth(ec: Any) -> int:
    parts = ec_parts(ec)
    depth = 0
    for part in parts:
        if part is None:
            break
        depth += 1
    return depth


def ec_prefix(ec: Any, level: int) -> Optional[str]:
    parts = ec_parts(ec)
    if level < 1 or level > 4:
        return None
    selected = parts[:level]
    if any(part is None for part in selected):
        return None
    return ".".join(str(part) for part in selected)


def exact_ec_match(ec_a: Any, ec_b: Any) -> bool:
    a = normalize_ec(ec_a)
    b = normalize_ec(ec_b)
    if not a or not b:
        return False
    return a == b


def prefix_ec_match(ec_a: Any, ec_b: Any, level: int) -> bool:
    pa = ec_prefix(ec_a, level)
    pb = ec_prefix(ec_b, level)
    return pa is not None and pa == pb


def any_ec_match(true_ecs: Iterable[str], candidate_ecs: Iterable[str], mode: str) -> bool:
    true_list = [e for e in true_ecs if normalize_ec(e)]
    cand_list = [e for e in candidate_ecs if normalize_ec(e)]
    if not true_list or not cand_list:
        return False
    if mode == "exact":
        return any(exact_ec_match(a, b) for a in true_list for b in cand_list)
    if mode.startswith("prefix"):
        level = int(mode.replace("prefix", ""))
        return any(prefix_ec_match(a, b, level) for a in true_list for b in cand_list)
    raise ValueError(f"Unknown EC match mode: {mode}")


# ---------------------------------------------------------------------------
# Metrics and ranking helpers
# ---------------------------------------------------------------------------


def build_positive_sets(df: pd.DataFrame) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    e_to_r = (
        df.groupby("enzyme_id")["reaction_id"]
        .apply(lambda s: set(map(str, s.dropna().unique())))
        .to_dict()
    )
    r_to_e = (
        df.groupby("reaction_id")["enzyme_id"]
        .apply(lambda s: set(map(str, s.dropna().unique())))
        .to_dict()
    )
    return e_to_r, r_to_e


def sorted_one_positive(possets: Mapping[str, Set[str]]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for query, positives in possets.items():
        if positives:
            out[query] = {sorted(positives)[0]}
    return out


def rank_indices_desc(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) == 0:
        return np.asarray([], dtype=int)
    valid_scores = scores[valid_idx]
    order = np.argsort(-valid_scores, kind="mergesort")
    return valid_idx[order]


def top_items_from_indices(candidates: Sequence[str], indices: Sequence[int]) -> List[str]:
    return [str(candidates[int(i)]) for i in indices]


def evaluate_direction(
    score_matrix: pd.DataFrame,
    possets: Mapping[str, Set[str]],
    direction: str,
    ks: Sequence[int] = KS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one retrieval direction.

    direction must be "enzyme_to_reaction" or "reaction_to_enzyme".
    score_matrix is always enzyme rows x reaction columns.
    """
    if direction not in {"enzyme_to_reaction", "reaction_to_enzyme"}:
        raise ValueError(direction)

    max_k = max(ks)
    rows: List[Dict[str, Any]] = []

    if direction == "enzyme_to_reaction":
        query_axis = score_matrix.index.map(str).tolist()
        candidate_axis = score_matrix.columns.map(str).tolist()
        query_to_pos = {str(k): set(map(str, v)) for k, v in possets.items()}
        query_available = set(query_axis)
        candidate_to_idx = {c: i for i, c in enumerate(candidate_axis)}

        def get_scores(query: str) -> np.ndarray:
            return score_matrix.loc[query].to_numpy(dtype=float)

    else:
        query_axis = score_matrix.columns.map(str).tolist()
        candidate_axis = score_matrix.index.map(str).tolist()
        query_to_pos = {str(k): set(map(str, v)) for k, v in possets.items()}
        query_available = set(query_axis)
        candidate_to_idx = {c: i for i, c in enumerate(candidate_axis)}

        def get_scores(query: str) -> np.ndarray:
            return score_matrix.loc[:, query].to_numpy(dtype=float)

    for query in sorted(query_to_pos):
        positives_all = set(query_to_pos[query])
        row: Dict[str, Any] = {
            "direction": direction,
            "query_id": query,
            "n_positives": len(positives_all),
            "positives": ";".join(sorted(positives_all)),
            "candidate_count": len(candidate_axis),
            "query_has_scores": query in query_available,
        }
        for k in ks:
            row[f"hit@{k}"] = np.nan
            row[f"recall@{k}"] = np.nan
            row[f"top{k}"] = ""
            row[f"top{k}_scores"] = ""
        row.update(
            {
                "first_positive_rank": np.nan,
                "mrr": np.nan,
                "scored_candidate_count": 0,
                "n_positive_candidates": 0,
                "n_positive_scored": 0,
                "evaluated": False,
                "skip_reason": "",
            }
        )

        if query not in query_available:
            row["skip_reason"] = "query_not_in_score_matrix"
            rows.append(row)
            continue

        scores = get_scores(query)
        valid = np.isfinite(scores)
        positive_idx = [candidate_to_idx[p] for p in positives_all if p in candidate_to_idx]
        scored_positive_idx = [idx for idx in positive_idx if valid[idx]]
        row["scored_candidate_count"] = int(valid.sum())
        row["n_positive_candidates"] = int(len(positive_idx))
        row["n_positive_scored"] = int(len(scored_positive_idx))

        if len(scored_positive_idx) == 0:
            row["skip_reason"] = "no_scored_positive_for_query"
            rows.append(row)
            continue

        ranked_idx = rank_indices_desc(scores)
        if len(ranked_idx) == 0:
            row["skip_reason"] = "no_finite_scores_for_query"
            rows.append(row)
            continue

        rank_lookup = {int(idx): rank + 1 for rank, idx in enumerate(ranked_idx)}
        positive_ranks = [rank_lookup[idx] for idx in scored_positive_idx if idx in rank_lookup]
        first_rank = min(positive_ranks) if positive_ranks else np.nan
        row["first_positive_rank"] = float(first_rank)
        row["mrr"] = 1.0 / float(first_rank) if np.isfinite(first_rank) and first_rank > 0 else np.nan
        row["evaluated"] = True

        for k in ks:
            top_idx = ranked_idx[:k]
            top_set = set(map(int, top_idx))
            hits = len(top_set.intersection(scored_positive_idx))
            row[f"hit@{k}"] = float(hits > 0)
            row[f"recall@{k}"] = float(hits / len(scored_positive_idx))
            row[f"top{k}"] = ";".join(top_items_from_indices(candidate_axis, top_idx))
            row[f"top{k}_scores"] = ";".join(f"{scores[int(i)]:.8g}" for i in top_idx)

        rows.append(row)

    details = pd.DataFrame(rows)
    evaluated = details[details["evaluated"] == True].copy()  # noqa: E712
    metric_row: Dict[str, Any] = {
        "direction": direction,
        "n_queries_total": int(len(details)),
        "n_queries_evaluated": int(len(evaluated)),
        "n_queries_skipped": int(len(details) - len(evaluated)),
        "mean_rank_first_positive": np.nan,
        "median_rank_first_positive": np.nan,
        "mrr": np.nan,
    }
    for k in ks:
        metric_row[f"hit@{k}"] = np.nan
        metric_row[f"recall@{k}"] = np.nan
    if not evaluated.empty:
        metric_row["mean_rank_first_positive"] = float(evaluated["first_positive_rank"].mean())
        metric_row["median_rank_first_positive"] = float(evaluated["first_positive_rank"].median())
        metric_row["mrr"] = float(evaluated["mrr"].mean())
        for k in ks:
            metric_row[f"hit@{k}"] = float(evaluated[f"hit@{k}"].mean())
            metric_row[f"recall@{k}"] = float(evaluated[f"recall@{k}"].mean())
    return pd.DataFrame([metric_row]), details


def make_quantile_summary(values: pd.Series, prefix: str) -> Dict[str, Any]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {f"{prefix}_{name}": np.nan for name in ("mean", "median", "max", "q25", "q75", "q90", "q95", "q99")}
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(values.median()),
        f"{prefix}_max": int(values.max()),
        f"{prefix}_q25": float(values.quantile(0.25)),
        f"{prefix}_q75": float(values.quantile(0.75)),
        f"{prefix}_q90": float(values.quantile(0.90)),
        f"{prefix}_q95": float(values.quantile(0.95)),
        f"{prefix}_q99": float(values.quantile(0.99)),
    }


def dataset_cardinality(train: pd.DataFrame, test: pd.DataFrame, val: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for name, df in [("train", train), ("val", val), ("test", test)]:
        if df is None:
            continue
        unique_pairs = df[["enzyme_id", "reaction_id"]].drop_duplicates()
        per_enzyme = unique_pairs.groupby("enzyme_id")["reaction_id"].nunique()
        per_reaction = unique_pairs.groupby("reaction_id")["enzyme_id"].nunique()
        row: Dict[str, Any] = {
            "split": name,
            "unique_enzymes": int(df["enzyme_id"].nunique()),
            "unique_reactions": int(df["reaction_id"].nunique()),
            "positive_triplets": int(len(df.drop_duplicates())),
            "unique_enzyme_reaction_pairs": int(len(unique_pairs)),
            "avg_positives_per_enzyme": float(per_enzyme.mean()) if not per_enzyme.empty else np.nan,
            "avg_positives_per_reaction": float(per_reaction.mean()) if not per_reaction.empty else np.nan,
        }
        row.update(make_quantile_summary(per_enzyme, "positives_per_enzyme"))
        row.update(make_quantile_summary(per_reaction, "positives_per_reaction"))
        rows.append(row)
    cardinality = pd.DataFrame(rows)
    n_test_e = int(test["enzyme_id"].nunique())
    n_test_r = int(test["reaction_id"].nunique())
    random_baselines = pd.DataFrame(
        [
            {
                "direction": "enzyme_to_reaction",
                "candidate_count": n_test_r,
                "random_top1": 1.0 / n_test_r if n_test_r else np.nan,
            },
            {
                "direction": "reaction_to_enzyme",
                "candidate_count": n_test_e,
                "random_top1": 1.0 / n_test_e if n_test_e else np.nan,
            },
        ]
    )
    return cardinality, random_baselines


def stratified_metrics(details: pd.DataFrame, direction: str) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame()

    def bin_label(n: int) -> str:
        if n == 1:
            return "1"
        if 2 <= n <= 5:
            return "2-5"
        if 6 <= n <= 20:
            return "6-20"
        return ">20"

    evaluated = details[details["evaluated"] == True].copy()  # noqa: E712
    if evaluated.empty:
        return pd.DataFrame()
    evaluated["positive_bin"] = evaluated["n_positives"].map(lambda x: bin_label(int(x)))
    rows = []
    for label in ["1", "2-5", "6-20", ">20"]:
        sub = evaluated[evaluated["positive_bin"] == label]
        row: Dict[str, Any] = {"direction": direction, "positive_bin": label, "n_queries": int(len(sub))}
        for k in KS:
            row[f"hit@{k}"] = float(sub[f"hit@{k}"].mean()) if not sub.empty else np.nan
            row[f"recall@{k}"] = float(sub[f"recall@{k}"].mean()) if not sub.empty else np.nan
        row["mrr"] = float(sub["mrr"].mean()) if not sub.empty else np.nan
        row["mean_rank_first_positive"] = float(sub["first_positive_rank"].mean()) if not sub.empty else np.nan
        row["median_rank_first_positive"] = float(sub["first_positive_rank"].median()) if not sub.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EC-level and false-positive analyses
# ---------------------------------------------------------------------------


def build_ec_maps(*dfs: pd.DataFrame) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[Tuple[str, str], Set[str]]]:
    combined = pd.concat([df for df in dfs if df is not None], ignore_index=True)
    combined = combined.drop_duplicates()
    enzyme_ecs = (
        combined.groupby("enzyme_id")["ec"]
        .apply(lambda s: set(normalize_ec(x) for x in s.dropna().astype(str) if normalize_ec(x)))
        .to_dict()
    )
    reaction_ecs = (
        combined.groupby("reaction_id")["ec"]
        .apply(lambda s: set(normalize_ec(x) for x in s.dropna().astype(str) if normalize_ec(x)))
        .to_dict()
    )
    pair_ecs: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for row in combined.itertuples(index=False):
        ec = normalize_ec(row.ec)
        if ec:
            pair_ecs[(str(row.enzyme_id), str(row.reaction_id))].add(ec)
    return enzyme_ecs, reaction_ecs, pair_ecs


def top_items(row: pd.Series, k: int) -> List[str]:
    text = row.get(f"top{k}", "")
    if not isinstance(text, str) or text == "":
        return []
    return [x for x in text.split(";") if x]


def ec_level_reaction_to_enzyme(
    r2e_details: pd.DataFrame,
    test: pd.DataFrame,
    enzyme_ecs: Mapping[str, Set[str]],
) -> pd.DataFrame:
    true_reaction_ecs = (
        test.groupby("reaction_id")["ec"]
        .apply(lambda s: set(normalize_ec(x) for x in s.dropna().astype(str) if normalize_ec(x)))
        .to_dict()
    )
    rows = []
    evaluated = r2e_details[r2e_details["evaluated"] == True].copy()  # noqa: E712
    for k in KS:
        totals = {
            "exact_enzyme_hit": [],
            "exact_ec_hit": [],
            "ec_prefix3_hit": [],
            "ec_prefix2_hit": [],
            "ec_prefix1_hit": [],
        }
        for _, row in evaluated.iterrows():
            query = str(row["query_id"])
            true_enzymes = set(str(x) for x in str(row["positives"]).split(";") if x)
            true_ecs = true_reaction_ecs.get(query, set())
            retrieved = top_items(row, k)
            totals["exact_enzyme_hit"].append(float(bool(set(retrieved).intersection(true_enzymes))))
            retrieved_ecs: Set[str] = set()
            for enzyme_id in retrieved:
                retrieved_ecs.update(enzyme_ecs.get(enzyme_id, set()))
            totals["exact_ec_hit"].append(float(any_ec_match(true_ecs, retrieved_ecs, "exact")))
            totals["ec_prefix3_hit"].append(float(any_ec_match(true_ecs, retrieved_ecs, "prefix3")))
            totals["ec_prefix2_hit"].append(float(any_ec_match(true_ecs, retrieved_ecs, "prefix2")))
            totals["ec_prefix1_hit"].append(float(any_ec_match(true_ecs, retrieved_ecs, "prefix1")))
        out: Dict[str, Any] = {"direction": "reaction_to_enzyme", "k": k, "n_queries": int(len(evaluated))}
        for metric, values in totals.items():
            out[metric] = float(np.mean(values)) if values else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def read_cluster_file(path: Optional[str | Path]) -> Dict[str, str]:
    if not path:
        return {}
    df = read_table_auto(path)
    id_col = next((c for c in ("enzyme_id", "protein_id", "id") if c in df.columns), df.columns[0])
    cluster_col = next((c for c in ("cluster_id", "cluster", "sequence_cluster") if c in df.columns), None)
    if cluster_col is None:
        candidates = [c for c in df.columns if c != id_col]
        if not candidates:
            return {}
        cluster_col = candidates[0]
    return dict(zip(df[id_col].map(normalize_id), df[cluster_col].map(normalize_id)))


def false_positive_plausibility(
    r2e_details: pd.DataFrame,
    test: pd.DataFrame,
    combined_annotations: pd.DataFrame,
    enzyme_ecs: Mapping[str, Set[str]],
    cluster_map: Optional[Mapping[str, str]] = None,
    k: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    true_reaction_ecs = (
        test.groupby("reaction_id")["ec"]
        .apply(lambda s: set(normalize_ec(x) for x in s.dropna().astype(str) if normalize_ec(x)))
        .to_dict()
    )
    reaction_to_any_enzyme = (
        combined_annotations.groupby("reaction_id")["enzyme_id"]
        .apply(lambda s: set(map(str, s.dropna().unique())))
        .to_dict()
    )
    cluster_map = cluster_map or {}
    rows = []
    evaluated = r2e_details[r2e_details["evaluated"] == True].copy()  # noqa: E712
    for _, row in evaluated.iterrows():
        reaction_id = str(row["query_id"])
        true_enzymes = set(str(x) for x in str(row["positives"]).split(";") if x)
        true_ecs = true_reaction_ecs.get(reaction_id, set())
        true_clusters = {cluster_map[e] for e in true_enzymes if e in cluster_map}
        for rank, enzyme_id in enumerate(top_items(row, k), start=1):
            if enzyme_id in true_enzymes:
                continue
            candidate_ecs = enzyme_ecs.get(enzyme_id, set())
            candidate_cluster = cluster_map.get(enzyme_id)
            rows.append(
                {
                    "reaction_id": reaction_id,
                    "rank": rank,
                    "retrieved_enzyme_id": enzyme_id,
                    "exact_ec_match": any_ec_match(true_ecs, candidate_ecs, "exact"),
                    "ec_prefix3_match": any_ec_match(true_ecs, candidate_ecs, "prefix3"),
                    "ec_prefix2_match": any_ec_match(true_ecs, candidate_ecs, "prefix2"),
                    "ec_prefix1_match": any_ec_match(true_ecs, candidate_ecs, "prefix1"),
                    "shares_sequence_cluster": bool(candidate_cluster and candidate_cluster in true_clusters),
                    "appears_with_same_reaction_any_split": enzyme_id in reaction_to_any_enzyme.get(reaction_id, set()),
                    "true_ecs": ";".join(sorted(true_ecs)),
                    "retrieved_ecs": ";".join(sorted(candidate_ecs)),
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(), detail
    summary = pd.DataFrame(
        [
            {
                "direction": "reaction_to_enzyme",
                "k": k,
                "n_false_positive_retrievals": int(len(detail)),
                "pct_exact_ec_match": float(detail["exact_ec_match"].mean()),
                "pct_ec_prefix3_match": float(detail["ec_prefix3_match"].mean()),
                "pct_ec_prefix2_match": float(detail["ec_prefix2_match"].mean()),
                "pct_ec_prefix1_match": float(detail["ec_prefix1_match"].mean()),
                "pct_shares_sequence_cluster": float(detail["shares_sequence_cluster"].mean()) if cluster_map else np.nan,
                "pct_appears_with_same_reaction_any_split": float(detail["appears_with_same_reaction_any_split"].mean()),
            }
        ]
    )
    return summary, detail


# ---------------------------------------------------------------------------
# Hubness and score calibration
# ---------------------------------------------------------------------------


def gini(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cumulative = np.cumsum(arr)
    return float((n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n)


def entropy_from_counts(values: Sequence[int]) -> Tuple[float, float]:
    counts = np.asarray(values, dtype=float)
    total = counts.sum()
    if total <= 0:
        return np.nan, np.nan
    p = counts[counts > 0] / total
    entropy = -float(np.sum(p * np.log(p)))
    norm = entropy / math.log(len(counts)) if len(counts) > 1 else np.nan
    return entropy, norm


def hubness(details: pd.DataFrame, direction: str, candidate_count: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Counter]]:
    evaluated = details[details["evaluated"] == True].copy()  # noqa: E712
    top_rows = []
    summary_rows = []
    counters: Dict[int, Counter] = {}
    for k in KS:
        counter: Counter = Counter()
        for _, row in evaluated.iterrows():
            counter.update(top_items(row, k))
        counters[k] = counter
        total_slots = int(sum(counter.values()))
        counts = list(counter.values())
        entropy, norm_entropy = entropy_from_counts(counts)
        sorted_counts = sorted(counts, reverse=True)
        top1_frac = sum(sorted_counts[:1]) / total_slots if total_slots else np.nan
        top5_frac = sum(sorted_counts[:5]) / total_slots if total_slots else np.nan
        top10_frac = sum(sorted_counts[:10]) / total_slots if total_slots else np.nan
        summary_rows.append(
            {
                "direction": direction,
                "k": k,
                "n_queries": int(len(evaluated)),
                "candidate_count": int(candidate_count),
                "total_retrieved_slots": total_slots,
                "unique_retrieved_entities": int(len(counter)),
                "top1_entity_fraction": float(top1_frac) if np.isfinite(top1_frac) else np.nan,
                "top5_entities_fraction": float(top5_frac) if np.isfinite(top5_frac) else np.nan,
                "top10_entities_fraction": float(top10_frac) if np.isfinite(top10_frac) else np.nan,
                "gini_retrieval_counts": gini(counts),
                "entropy_retrieval_counts": entropy,
                "normalized_entropy_retrieval_counts": norm_entropy,
            }
        )
        for rank, (entity, count) in enumerate(counter.most_common(20), start=1):
            top_rows.append(
                {
                    "direction": direction,
                    "k": k,
                    "rank": rank,
                    "entity_id": entity,
                    "retrieval_count": int(count),
                    "fraction_of_retrieved_slots": float(count / total_slots) if total_slots else np.nan,
                    "fraction_of_queries": float(count / len(evaluated)) if len(evaluated) else np.nan,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(top_rows), counters


def stable_softmax_entropy(scores: np.ndarray) -> Tuple[float, float, float]:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return np.nan, np.nan, np.nan
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    total = exp.sum()
    if total <= 0:
        return np.nan, np.nan, np.nan
    p = exp / total
    entropy = -float(np.sum(p * np.log(p + 1e-300)))
    norm_entropy = entropy / math.log(len(scores)) if len(scores) > 1 else np.nan
    top = np.sort(scores)[::-1]
    margin = float(top[0] - top[1]) if len(top) > 1 else np.nan
    return entropy, norm_entropy, margin


def score_distribution_diagnostics(
    score_matrix: pd.DataFrame,
    e_to_r: Mapping[str, Set[str]],
    r_to_e: Mapping[str, Set[str]],
    max_unlabeled_samples: int = 100000,
    seed: int = 13,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    distributions = []

    for direction, possets in [("enzyme_to_reaction", e_to_r), ("reaction_to_enzyme", r_to_e)]:
        entropy_values = []
        norm_entropy_values = []
        max_scores = []
        margins = []
        positive_scores = []
        unlabeled_scores = []

        if direction == "enzyme_to_reaction":
            candidates = score_matrix.columns.map(str).tolist()
            candidate_to_idx = {c: i for i, c in enumerate(candidates)}
            queries = [q for q in sorted(possets) if q in score_matrix.index]

            def get_scores(q: str) -> np.ndarray:
                return score_matrix.loc[q].to_numpy(dtype=float)

        else:
            candidates = score_matrix.index.map(str).tolist()
            candidate_to_idx = {c: i for i, c in enumerate(candidates)}
            queries = [q for q in sorted(possets) if q in score_matrix.columns]

            def get_scores(q: str) -> np.ndarray:
                return score_matrix.loc[:, q].to_numpy(dtype=float)

        for q in queries:
            scores = get_scores(q)
            valid = np.isfinite(scores)
            if not valid.any():
                continue
            entropy, norm_entropy, margin = stable_softmax_entropy(scores)
            entropy_values.append(entropy)
            norm_entropy_values.append(norm_entropy)
            max_scores.append(float(np.nanmax(scores)))
            margins.append(margin)
            positive_idx = [candidate_to_idx[p] for p in possets[q] if p in candidate_to_idx and valid[candidate_to_idx[p]]]
            positive_scores.extend([float(scores[i]) for i in positive_idx])
            unlabeled_idx = np.flatnonzero(valid)
            if positive_idx:
                positive_idx_set = set(positive_idx)
                unlabeled_idx = np.asarray([i for i in unlabeled_idx if i not in positive_idx_set], dtype=int)
            if len(unlabeled_idx) > 0:
                per_query_cap = max(1, max_unlabeled_samples // max(1, len(queries)))
                take = min(len(unlabeled_idx), per_query_cap)
                sampled = rng.choice(unlabeled_idx, size=take, replace=False)
                unlabeled_scores.extend([float(scores[i]) for i in sampled])

        def stat(values: Sequence[float], fn: str) -> float:
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return np.nan
            if fn == "mean":
                return float(arr.mean())
            if fn == "median":
                return float(np.median(arr))
            if fn == "std":
                return float(arr.std())
            if fn == "q05":
                return float(np.quantile(arr, 0.05))
            if fn == "q95":
                return float(np.quantile(arr, 0.95))
            raise ValueError(fn)

        rows.append(
            {
                "direction": direction,
                "n_queries_scored": int(len(entropy_values)),
                "mean_entropy": stat(entropy_values, "mean"),
                "median_entropy": stat(entropy_values, "median"),
                "mean_normalized_entropy": stat(norm_entropy_values, "mean"),
                "median_normalized_entropy": stat(norm_entropy_values, "median"),
                "mean_max_score": stat(max_scores, "mean"),
                "median_max_score": stat(max_scores, "median"),
                "mean_top1_top2_margin": stat(margins, "mean"),
                "median_top1_top2_margin": stat(margins, "median"),
                "mean_positive_score": stat(positive_scores, "mean"),
                "median_positive_score": stat(positive_scores, "median"),
                "mean_unlabeled_score": stat(unlabeled_scores, "mean"),
                "median_unlabeled_score": stat(unlabeled_scores, "median"),
                "positive_minus_unlabeled_mean": stat(positive_scores, "mean") - stat(unlabeled_scores, "mean"),
                "n_positive_scores": int(np.isfinite(np.asarray(positive_scores, dtype=float)).sum()),
                "n_unlabeled_scores_sampled": int(np.isfinite(np.asarray(unlabeled_scores, dtype=float)).sum()),
            }
        )
        distributions.extend(
            {"direction": direction, "distribution": "entropy", "value": v} for v in entropy_values if np.isfinite(v)
        )
        distributions.extend(
            {"direction": direction, "distribution": "margin", "value": v} for v in margins if np.isfinite(v)
        )
        distributions.extend(
            {"direction": direction, "distribution": "positive_score", "value": v}
            for v in positive_scores
            if np.isfinite(v)
        )
        distributions.extend(
            {"direction": direction, "distribution": "sampled_unlabeled_score", "value": v}
            for v in unlabeled_scores
            if np.isfinite(v)
        )

    return pd.DataFrame(rows), pd.DataFrame(distributions)


# ---------------------------------------------------------------------------
# Embedding diagnostics
# ---------------------------------------------------------------------------


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def norm_summary(emb: pd.DataFrame, entity_type: str) -> pd.DataFrame:
    norms = np.linalg.norm(emb.to_numpy(dtype=float), axis=1)
    return pd.DataFrame(
        [
            {
                "entity_type": entity_type,
                "n_entities": int(len(norms)),
                "mean_l2_norm": float(np.mean(norms)) if len(norms) else np.nan,
                "median_l2_norm": float(np.median(norms)) if len(norms) else np.nan,
                "std_l2_norm": float(np.std(norms)) if len(norms) else np.nan,
                "min_l2_norm": float(np.min(norms)) if len(norms) else np.nan,
                "max_l2_norm": float(np.max(norms)) if len(norms) else np.nan,
                "q95_l2_norm": float(np.quantile(norms, 0.95)) if len(norms) else np.nan,
                "q99_l2_norm": float(np.quantile(norms, 0.99)) if len(norms) else np.nan,
            }
        ]
    )


def embedding_diagnostics(
    enzyme_emb: Optional[pd.DataFrame],
    reaction_emb: Optional[pd.DataFrame],
    test: pd.DataFrame,
    score_matrix: pd.DataFrame,
    e_to_r: Mapping[str, Set[str]],
    r_to_e: Mapping[str, Set[str]],
    max_pairs: int = 10000000,
    seed: int = 13,
) -> Dict[str, pd.DataFrame]:
    outputs: Dict[str, pd.DataFrame] = {}
    norm_rows = []
    if enzyme_emb is not None:
        norm_rows.append(norm_summary(enzyme_emb, "enzyme"))
    if reaction_emb is not None:
        norm_rows.append(norm_summary(reaction_emb, "reaction"))
    if norm_rows:
        outputs["embedding_norms"] = pd.concat(norm_rows, ignore_index=True)

    if enzyme_emb is None or reaction_emb is None:
        return outputs

    common_enzymes = sorted(set(enzyme_emb.index.map(str)).intersection(score_matrix.index.map(str)))
    common_reactions = sorted(set(reaction_emb.index.map(str)).intersection(score_matrix.columns.map(str)))
    enzyme_emb = enzyme_emb.loc[common_enzymes]
    reaction_emb = reaction_emb.loc[common_reactions]
    if enzyme_emb.empty or reaction_emb.empty:
        return outputs

    rng = np.random.default_rng(seed)
    e_norm = l2_normalize(enzyme_emb.to_numpy(dtype=float))
    r_norm = l2_normalize(reaction_emb.to_numpy(dtype=float))
    e_idx = {e: i for i, e in enumerate(enzyme_emb.index.map(str))}
    r_idx = {r: i for i, r in enumerate(reaction_emb.index.map(str))}

    positive_pairs = [
        (str(row.enzyme_id), str(row.reaction_id))
        for row in test.itertuples(index=False)
        if str(row.enzyme_id) in e_idx and str(row.reaction_id) in r_idx
    ]
    positive_pair_set = set(positive_pairs)
    positive_cos = [float(np.dot(e_norm[e_idx[e]], r_norm[r_idx[r]])) for e, r in positive_pairs]

    n_samples = min(100000, max(1000, len(positive_pairs) * 5)) if positive_pairs else 10000
    all_e = list(e_idx.keys())
    all_r = list(r_idx.keys())
    unlabeled_cos = []
    tries = 0
    while len(unlabeled_cos) < n_samples and tries < n_samples * 20 and all_e and all_r:
        tries += 1
        e = all_e[int(rng.integers(0, len(all_e)))]
        r = all_r[int(rng.integers(0, len(all_r)))]
        if (e, r) in positive_pair_set:
            continue
        unlabeled_cos.append(float(np.dot(e_norm[e_idx[e]], r_norm[r_idx[r]])))

    outputs["embedding_similarity_metrics"] = pd.DataFrame(
        [
            {
                "n_positive_pairs": int(len(positive_cos)),
                "n_unlabeled_pairs_sampled": int(len(unlabeled_cos)),
                "mean_positive_cosine": float(np.mean(positive_cos)) if positive_cos else np.nan,
                "median_positive_cosine": float(np.median(positive_cos)) if positive_cos else np.nan,
                "mean_sampled_unlabeled_cosine": float(np.mean(unlabeled_cos)) if unlabeled_cos else np.nan,
                "median_sampled_unlabeled_cosine": float(np.median(unlabeled_cos)) if unlabeled_cos else np.nan,
                "positive_minus_unlabeled_cosine_mean": (
                    float(np.mean(positive_cos) - np.mean(unlabeled_cos)) if positive_cos and unlabeled_cos else np.nan
                ),
            }
        ]
    )

    score_overlap = score_matrix.loc[common_enzymes, common_reactions]
    enzyme_norms = np.linalg.norm(enzyme_emb.to_numpy(dtype=float), axis=1)
    avg_scores = score_overlap.mean(axis=1, skipna=True).to_numpy(dtype=float)
    valid = np.isfinite(enzyme_norms) & np.isfinite(avg_scores)
    if valid.sum() >= 3:
        pearson = float(np.corrcoef(enzyme_norms[valid], avg_scores[valid])[0, 1])
        spearman = float(pd.Series(enzyme_norms[valid]).rank().corr(pd.Series(avg_scores[valid]).rank()))
    else:
        pearson = np.nan
        spearman = np.nan
    outputs["embedding_norm_score_correlation"] = pd.DataFrame(
        [
            {
                "n_enzymes_overlap": int(valid.sum()),
                "pearson_enzyme_norm_vs_average_score": pearson,
                "spearman_enzyme_norm_vs_average_score": spearman,
            }
        ]
    )

    pair_count = len(common_enzymes) * len(common_reactions)
    if pair_count <= max_pairs:
        cosine_scores = pd.DataFrame(
            e_norm @ r_norm.T,
            index=common_enzymes,
            columns=common_reactions,
        )
        e2r_cos_metrics, _ = evaluate_direction(cosine_scores, e_to_r, "enzyme_to_reaction")
        r2e_cos_metrics, _ = evaluate_direction(cosine_scores, r_to_e, "reaction_to_enzyme")
        cosine_metrics = pd.concat([e2r_cos_metrics, r2e_cos_metrics], ignore_index=True)
        cosine_metrics.insert(0, "score_type", "cosine_from_embeddings")
        outputs["cosine_retrieval_metrics"] = cosine_metrics
    else:
        outputs["cosine_retrieval_metrics"] = pd.DataFrame(
            [
                {
                    "score_type": "cosine_from_embeddings",
                    "status": "skipped",
                    "reason": f"{pair_count} enzyme-reaction pairs exceeds --max-cosine-pairs={max_pairs}",
                }
            ]
        )

    avg_cos_to_reactions = e_norm @ r_norm.T
    avg_cos = avg_cos_to_reactions.mean(axis=1)
    top_avg = np.argsort(-avg_cos)[:20]
    outputs["enzyme_average_cosine_hubs"] = pd.DataFrame(
        [
            {
                "rank": rank,
                "enzyme_id": common_enzymes[int(i)],
                "average_cosine_to_reactions": float(avg_cos[int(i)]),
                "l2_norm": float(enzyme_norms[int(i)]),
            }
            for rank, i in enumerate(top_avg, start=1)
        ]
    )
    return outputs


# ---------------------------------------------------------------------------
# Leakage, loss inspection, examples, and cause ranking
# ---------------------------------------------------------------------------


def leakage_summary(train: pd.DataFrame, test: pd.DataFrame, val: Optional[pd.DataFrame], reaction_smiles: Optional[str]) -> pd.DataFrame:
    train_pairs = set(map(tuple, train[["enzyme_id", "reaction_id"]].drop_duplicates().to_numpy()))
    test_pairs = set(map(tuple, test[["enzyme_id", "reaction_id"]].drop_duplicates().to_numpy()))
    train_reactions = set(train["reaction_id"].unique())
    test_reactions = set(test["reaction_id"].unique())
    train_enzymes = set(train["enzyme_id"].unique())
    test_enzymes = set(test["enzyme_id"].unique())

    train_reaction_ecs = (
        train.groupby("reaction_id")["ec"]
        .apply(lambda s: set(normalize_ec(x) for x in s.dropna().astype(str) if normalize_ec(x)))
        .to_dict()
    )
    test_reaction_ecs = (
        test.groupby("reaction_id")["ec"]
        .apply(lambda s: set(normalize_ec(x) for x in s.dropna().astype(str) if normalize_ec(x)))
        .to_dict()
    )
    same_reaction_train_ec_share = []
    for reaction_id in sorted(test_reactions):
        if reaction_id not in train_reactions:
            continue
        same_reaction_train_ec_share.append(
            bool(test_reaction_ecs.get(reaction_id, set()).intersection(train_reaction_ecs.get(reaction_id, set())))
        )

    row: Dict[str, Any] = {
        "test_reactions": int(len(test_reactions)),
        "test_enzymes": int(len(test_enzymes)),
        "test_pairs": int(len(test_pairs)),
        "fraction_test_reactions_seen_in_train": len(test_reactions.intersection(train_reactions)) / len(test_reactions)
        if test_reactions
        else np.nan,
        "fraction_test_enzymes_seen_in_train": len(test_enzymes.intersection(train_enzymes)) / len(test_enzymes)
        if test_enzymes
        else np.nan,
        "fraction_test_pairs_seen_in_train": len(test_pairs.intersection(train_pairs)) / len(test_pairs) if test_pairs else np.nan,
        "fraction_seen_test_reactions_with_train_ec_overlap": float(np.mean(same_reaction_train_ec_share))
        if same_reaction_train_ec_share
        else np.nan,
        "canonical_reaction_duplicate_check": "not_provided",
        "fraction_test_reactions_with_same_smiles_in_train": np.nan,
    }

    if reaction_smiles:
        smiles_df = read_table_auto(reaction_smiles)
        rid_col = next((c for c in ("reaction_id", "id", "rxn_id") if c in smiles_df.columns), smiles_df.columns[0])
        smiles_col = next((c for c in ("canonical_smiles", "smiles", "reaction_smiles", "rxn_smiles") if c in smiles_df.columns), None)
        if smiles_col is not None:
            smiles_map = dict(zip(smiles_df[rid_col].map(normalize_id), smiles_df[smiles_col].astype(str).str.strip()))
            train_smiles = {smiles_map[r] for r in train_reactions if r in smiles_map and smiles_map[r]}
            test_with_smiles = [r for r in test_reactions if r in smiles_map and smiles_map[r]]
            if test_with_smiles:
                dup_count = sum(1 for r in test_with_smiles if smiles_map[r] in train_smiles)
                row["canonical_reaction_duplicate_check"] = "computed"
                row["fraction_test_reactions_with_same_smiles_in_train"] = dup_count / len(test_with_smiles)
            else:
                row["canonical_reaction_duplicate_check"] = "no_test_smiles_available"
        else:
            row["canonical_reaction_duplicate_check"] = "smiles_column_not_found"

    return pd.DataFrame([row])


def inspect_loss(config_path: Optional[str | Path]) -> Tuple[pd.DataFrame, str]:
    texts: List[Tuple[str, str]] = []
    if config_path:
        path = Path(config_path)
        if path.exists():
            texts.append((str(path), path.read_text(errors="ignore")))
    else:
        for candidate in (
            Path("horizyn/horizyn/losses.py"),
            Path("horizyn/horizyn/protein_pooling_lightning_module.py"),
            Path("horizyn/configs"),
        ):
            if candidate.is_file():
                texts.append((str(candidate), candidate.read_text(errors="ignore")))
            elif candidate.is_dir():
                for path in list(candidate.glob("*.yaml"))[:50]:
                    texts.append((str(path), path.read_text(errors="ignore")))

    joined = "\n".join(text for _, text in texts)
    lower = joined.lower()
    has_mlnce = "mlnce" in lower or "multi" in lower and "positive" in lower
    has_bidirectional = (
        "reaction_to_enzyme" in lower and "enzyme_to_reaction" in lower
    ) or "bidirectional" in lower
    has_e_anchor = "enzyme_to_reaction" in lower or "protein_to_reaction" in lower or "e_to_r" in lower
    has_r_anchor = "reaction_to_enzyme" in lower or "reaction_to_protein" in lower or "r_to_e" in lower

    if has_bidirectional and has_mlnce:
        classification = "likely_multi_positive_bidirectional"
    elif has_bidirectional:
        classification = "likely_bidirectional_unknown_positive_handling"
    elif has_e_anchor and not has_r_anchor:
        classification = "likely_enzyme_anchored_only"
    elif has_r_anchor and not has_e_anchor:
        classification = "likely_reaction_anchored_only"
    elif has_mlnce:
        classification = "likely_multi_positive_but_direction_unclear"
    else:
        classification = "unknown"

    table = pd.DataFrame(
        [
            {
                "classification": classification,
                "has_mlnce_or_multi_positive_keywords": bool(has_mlnce),
                "has_bidirectional_keywords": bool(has_bidirectional),
                "has_enzyme_anchor_keywords": bool(has_e_anchor),
                "has_reaction_anchor_keywords": bool(has_r_anchor),
                "sources_scanned": ";".join(source for source, _ in texts),
            }
        ]
    )
    md = textwrap.dedent(
        f"""
        # Loss inspection

        Classification: `{classification}`

        Sources scanned: `{'; '.join(source for source, _ in texts) if texts else 'none'}`

        Heuristic flags:

        - MLNCE or multi-positive keywords: `{has_mlnce}`
        - Bidirectional keywords: `{has_bidirectional}`
        - Enzyme-anchored keywords: `{has_e_anchor}`
        - Reaction-anchored keywords: `{has_r_anchor}`

        Recommended objective if reaction-to-enzyme is weak:

        $$
        L_{{E \\rightarrow R}}
        =
        -\\log
        \\frac{{
        \\sum_{{R^+ \\in P(E)}} \\exp(s(E,R^+)/\\tau)
        }}{{
        \\sum_{{R'}} \\exp(s(E,R')/\\tau)
        }}
        $$

        $$
        L_{{R \\rightarrow E}}
        =
        -\\log
        \\frac{{
        \\sum_{{E^+ \\in P(R)}} \\exp(s(E^+,R)/\\tau)
        }}{{
        \\sum_{{E'}} \\exp(s(E',R)/\\tau)
        }}
        $$

        $$
        L = L_{{E \\rightarrow R}} + \\lambda L_{{R \\rightarrow E}}
        $$

        If reaction-to-enzyme is the weak direction, test `lambda > 1`.
        """
    ).strip()
    return table, md


def error_examples(
    e2r_details: pd.DataFrame,
    r2e_details: pd.DataFrame,
    test: pd.DataFrame,
    enzyme_ecs: Mapping[str, Set[str]],
    reaction_ecs: Mapping[str, Set[str]],
    n: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    evaluated_r2e = r2e_details[r2e_details["evaluated"] == True].copy()  # noqa: E712
    failed = evaluated_r2e[evaluated_r2e["hit@10"] == 0].copy()
    failed = failed.sort_values(["first_positive_rank", "mrr"], ascending=[False, True]).head(n)
    rows = []
    for _, row in failed.iterrows():
        reaction_id = str(row["query_id"])
        true_enzymes = [x for x in str(row["positives"]).split(";") if x]
        retrieved = top_items(row, 10)
        true_ecs = reaction_ecs.get(reaction_id, set())
        retrieved_ecs = {e: enzyme_ecs.get(e, set()) for e in retrieved}
        rows.append(
            {
                "reaction_id": reaction_id,
                "true_enzyme_ids": ";".join(true_enzymes),
                "true_ec_labels": ";".join(sorted(true_ecs)),
                "top10_retrieved_enzyme_ids": ";".join(retrieved),
                "top10_retrieved_enzyme_ecs": json.dumps({e: sorted(v) for e, v in retrieved_ecs.items()}),
                "rank_of_first_true_enzyme": row["first_positive_rank"],
                "top10_has_exact_ec": any_ec_match(true_ecs, set().union(*retrieved_ecs.values()) if retrieved_ecs else set(), "exact"),
                "top10_has_ec_prefix3": any_ec_match(true_ecs, set().union(*retrieved_ecs.values()) if retrieved_ecs else set(), "prefix3"),
                "top10_has_ec_prefix2": any_ec_match(true_ecs, set().union(*retrieved_ecs.values()) if retrieved_ecs else set(), "prefix2"),
                "top10_has_ec_prefix1": any_ec_match(true_ecs, set().union(*retrieved_ecs.values()) if retrieved_ecs else set(), "prefix1"),
            }
        )
    bad_r2e = pd.DataFrame(rows)

    e2r_lookup = e2r_details.set_index("query_id") if not e2r_details.empty else pd.DataFrame()
    r2e_lookup = r2e_details.set_index("query_id") if not r2e_details.empty else pd.DataFrame()
    asym_rows = []
    for row in test.drop_duplicates(["enzyme_id", "reaction_id"]).itertuples(index=False):
        enzyme_id = str(row.enzyme_id)
        reaction_id = str(row.reaction_id)
        if enzyme_id not in e2r_lookup.index or reaction_id not in r2e_lookup.index:
            continue
        e2r_row = e2r_lookup.loc[enzyme_id]
        r2e_row = r2e_lookup.loc[reaction_id]
        if isinstance(e2r_row, pd.DataFrame):
            e2r_row = e2r_row.iloc[0]
        if isinstance(r2e_row, pd.DataFrame):
            r2e_row = r2e_row.iloc[0]
        if not bool(e2r_row.get("evaluated", False)) or not bool(r2e_row.get("evaluated", False)):
            continue
        e2r_top10 = set(top_items(e2r_row, 10))
        r2e_top10 = set(top_items(r2e_row, 10))
        if reaction_id in e2r_top10 and enzyme_id not in r2e_top10:
            top_enzymes = top_items(r2e_row, 10)
            true_ecs = reaction_ecs.get(reaction_id, set())
            retrieved_ecs = set()
            for e in top_enzymes:
                retrieved_ecs.update(enzyme_ecs.get(e, set()))
            asym_rows.append(
                {
                    "enzyme_id": enzyme_id,
                    "reaction_id": reaction_id,
                    "ec": normalize_ec(row.ec),
                    "enzyme_to_reaction_rank_first_positive": e2r_row.get("first_positive_rank"),
                    "reaction_to_enzyme_rank_first_positive": r2e_row.get("first_positive_rank"),
                    "reaction_to_enzyme_top10": ";".join(top_enzymes),
                    "reaction_to_enzyme_top10_has_exact_ec": any_ec_match(true_ecs, retrieved_ecs, "exact"),
                    "reaction_to_enzyme_top10_has_ec_prefix3": any_ec_match(true_ecs, retrieved_ecs, "prefix3"),
                    "reaction_to_enzyme_top10_has_ec_prefix2": any_ec_match(true_ecs, retrieved_ecs, "prefix2"),
                    "reaction_to_enzyme_top10_has_ec_prefix1": any_ec_match(true_ecs, retrieved_ecs, "prefix1"),
                }
            )
        if len(asym_rows) >= n:
            break
    return bad_r2e, pd.DataFrame(asym_rows)


def rank_likely_causes(
    cardinality: pd.DataFrame,
    multi_metrics: pd.DataFrame,
    delta_metrics: pd.DataFrame,
    ec_metrics: pd.DataFrame,
    fp_summary: pd.DataFrame,
    hub_summary: pd.DataFrame,
    score_summary: pd.DataFrame,
    leakage: pd.DataFrame,
    loss_table: pd.DataFrame,
    embedding_outputs: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    def get_metric(direction: str, column: str) -> float:
        sub = multi_metrics[multi_metrics["direction"] == direction]
        if sub.empty or column not in sub.columns:
            return np.nan
        return float(sub.iloc[0][column])

    test_row = cardinality[cardinality["split"] == "test"].iloc[0]
    n_e = float(test_row["unique_enzymes"])
    n_r = float(test_row["unique_reactions"])
    r2e_hit1 = get_metric("reaction_to_enzyme", "hit@1")
    e2r_hit1 = get_metric("enzyme_to_reaction", "hit@1")

    if n_e > n_r * 2:
        score = min(1.0, n_e / max(n_r, 1.0) / 10.0)
        rows.append(
            {
                "ranked_factor": "larger enzyme candidate pool",
                "evidence_score": score,
                "evidence": f"test has {int(n_e)} enzymes vs {int(n_r)} reactions; random R->E is harder",
            }
        )

    if not delta_metrics.empty:
        sub = delta_metrics[delta_metrics["direction"] == "reaction_to_enzyme"]
        if not sub.empty:
            improvement = float(sub.iloc[0].get("hit@10_delta_multi_minus_single", np.nan))
            if np.isfinite(improvement) and improvement > 0.02:
                rows.append(
                    {
                        "ranked_factor": "multi-positive ambiguity",
                        "evidence_score": min(1.0, improvement * 5),
                        "evidence": f"R->E Hit@10 improves by {improvement:.4f} with multi-positive evaluation",
                    }
                )

    if not fp_summary.empty:
        p2 = float(fp_summary.iloc[0].get("pct_ec_prefix2_match", np.nan))
        p3 = float(fp_summary.iloc[0].get("pct_ec_prefix3_match", np.nan))
        if np.isfinite(p2) and p2 > 0.2:
            rows.append(
                {
                    "ranked_factor": "incomplete annotations / false negatives",
                    "evidence_score": min(1.0, p2),
                    "evidence": f"{p2:.1%} of top false positives share EC prefix-2; prefix-3 share is {p3:.1%}" if np.isfinite(p3) else f"{p2:.1%} share EC prefix-2",
                }
            )

    if not ec_metrics.empty:
        k10 = ec_metrics[ec_metrics["k"] == 10]
        if not k10.empty:
            exact = float(k10.iloc[0].get("exact_enzyme_hit", np.nan))
            prefix3 = float(k10.iloc[0].get("ec_prefix3_hit", np.nan))
            if np.isfinite(exact) and np.isfinite(prefix3) and prefix3 - exact > 0.1:
                rows.append(
                    {
                        "ranked_factor": "reaction under-specification",
                        "evidence_score": min(1.0, prefix3 - exact),
                        "evidence": f"R->E exact enzyme Hit@10={exact:.4f}, but EC prefix-3 Hit@10={prefix3:.4f}",
                    }
                )

    if not hub_summary.empty:
        r2e = hub_summary[(hub_summary["direction"] == "reaction_to_enzyme") & (hub_summary["k"] == 1)]
        e2r = hub_summary[(hub_summary["direction"] == "enzyme_to_reaction") & (hub_summary["k"] == 1)]
        if not r2e.empty:
            r2e_gini = float(r2e.iloc[0].get("gini_retrieval_counts", np.nan))
            e2r_gini = float(e2r.iloc[0].get("gini_retrieval_counts", 0.0)) if not e2r.empty else 0.0
            top5 = float(r2e.iloc[0].get("top5_entities_fraction", np.nan))
            if (np.isfinite(r2e_gini) and r2e_gini > e2r_gini + 0.15) or (np.isfinite(top5) and top5 > 0.25):
                rows.append(
                    {
                        "ranked_factor": "hubness",
                        "evidence_score": min(1.0, max(r2e_gini - e2r_gini, top5 if np.isfinite(top5) else 0.0)),
                        "evidence": f"R->E top-1 retrieval Gini={r2e_gini:.4f}, top-5 hubs occupy {top5:.1%} of top-1 predictions",
                    }
                )

    if not score_summary.empty:
        r2e = score_summary[score_summary["direction"] == "reaction_to_enzyme"]
        e2r = score_summary[score_summary["direction"] == "enzyme_to_reaction"]
        if not r2e.empty and not e2r.empty:
            r_margin = float(r2e.iloc[0].get("median_top1_top2_margin", np.nan))
            e_margin = float(e2r.iloc[0].get("median_top1_top2_margin", np.nan))
            r_entropy = float(r2e.iloc[0].get("median_normalized_entropy", np.nan))
            if np.isfinite(r_margin) and np.isfinite(e_margin) and r_margin < e_margin * 0.5:
                rows.append(
                    {
                        "ranked_factor": "embedding normalization/calibration issue",
                        "evidence_score": min(1.0, 1.0 - r_margin / max(e_margin, 1e-12)),
                        "evidence": f"R->E median margin={r_margin:.4g} vs E->R={e_margin:.4g}; normalized entropy={r_entropy:.4f}",
                    }
                )

    if not loss_table.empty:
        classification = str(loss_table.iloc[0].get("classification", "unknown"))
        if classification not in {"likely_multi_positive_bidirectional"}:
            rows.append(
                {
                    "ranked_factor": "loss-direction mismatch",
                    "evidence_score": 0.6 if classification != "unknown" else 0.3,
                    "evidence": f"loss inspection classified training as {classification}",
                }
            )

    if not leakage.empty:
        frac_rxn = float(leakage.iloc[0].get("fraction_test_reactions_seen_in_train", np.nan))
        frac_pair = float(leakage.iloc[0].get("fraction_test_pairs_seen_in_train", np.nan))
        if (np.isfinite(frac_rxn) and frac_rxn > 0.2) or (np.isfinite(frac_pair) and frac_pair > 0.0):
            rows.append(
                {
                    "ranked_factor": "train/test leakage or split artifact",
                    "evidence_score": min(1.0, max(frac_rxn if np.isfinite(frac_rxn) else 0.0, frac_pair * 5 if np.isfinite(frac_pair) else 0.0)),
                    "evidence": f"{frac_rxn:.1%} test reactions seen in train; {frac_pair:.1%} exact test pairs seen in train",
                }
            )

    if not rows:
        rows.append(
            {
                "ranked_factor": "no dominant cause identified by heuristics",
                "evidence_score": 0.0,
                "evidence": "Inspect generated tables and error examples manually.",
            }
        )

    ranked = pd.DataFrame(rows).sort_values("evidence_score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def save_plot(path: Path) -> None:
    if plt is None:
        return
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_positive_histograms(test: pd.DataFrame, outdir: Path) -> None:
    if plt is None:
        return
    pairs = test[["enzyme_id", "reaction_id"]].drop_duplicates()
    per_e = pairs.groupby("enzyme_id")["reaction_id"].nunique()
    per_r = pairs.groupby("reaction_id")["enzyme_id"].nunique()
    plt.figure(figsize=(8, 4))
    plt.hist(per_e, bins=50, alpha=0.7, label="positives per enzyme")
    plt.hist(per_r, bins=50, alpha=0.7, label="positives per reaction")
    plt.yscale("log")
    plt.xlabel("Known positives per query")
    plt.ylabel("Query count, log scale")
    plt.legend()
    plt.title("Test positives per query")
    save_plot(outdir / "positives_per_query_hist.png")


def plot_single_multi(single: pd.DataFrame, multi: pd.DataFrame, outdir: Path) -> None:
    if plt is None or single.empty or multi.empty:
        return
    rows = []
    for label, df in [("single", single), ("multi", multi)]:
        for _, row in df.iterrows():
            for metric in ["hit@1", "hit@5", "hit@10", "mrr"]:
                rows.append({"evaluation": label, "direction": row["direction"], "metric": metric, "value": row.get(metric)})
    data = pd.DataFrame(rows)
    if data.empty:
        return
    for direction in data["direction"].unique():
        sub = data[data["direction"] == direction]
        pivot = sub.pivot(index="metric", columns="evaluation", values="value").reindex(["hit@1", "hit@5", "hit@10", "mrr"])
        pivot.plot(kind="bar", figsize=(7, 4))
        plt.ylabel("Metric")
        plt.ylim(0, 1)
        plt.title(f"Single-positive vs multi-positive: {direction}")
        save_plot(outdir / f"single_vs_multi_{direction}.png")


def plot_stratified(stratified: pd.DataFrame, outdir: Path) -> None:
    if plt is None or stratified.empty:
        return
    for direction in stratified["direction"].unique():
        sub = stratified[stratified["direction"] == direction].set_index("positive_bin").reindex(["1", "2-5", "6-20", ">20"])
        cols = ["hit@1", "hit@5", "hit@10", "mrr"]
        sub[cols].plot(kind="bar", figsize=(8, 4))
        plt.ylabel("Metric")
        plt.ylim(0, 1)
        plt.title(f"Retrieval by positives-per-query bin: {direction}")
        save_plot(outdir / f"stratified_metrics_{direction}.png")


def plot_ec_ladder(ec_metrics: pd.DataFrame, outdir: Path) -> None:
    if plt is None or ec_metrics.empty:
        return
    metrics = ["exact_enzyme_hit", "exact_ec_hit", "ec_prefix3_hit", "ec_prefix2_hit", "ec_prefix1_hit"]
    for k in sorted(ec_metrics["k"].unique()):
        row = ec_metrics[ec_metrics["k"] == k].iloc[0]
        values = [row.get(m, np.nan) for m in metrics]
        plt.figure(figsize=(8, 4))
        plt.bar(metrics, values)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Hit rate")
        plt.ylim(0, 1)
        plt.title(f"Reaction-to-enzyme EC hierarchy Hit@{k}")
        save_plot(outdir / f"ec_hierarchy_hit_at_{k}.png")


def plot_false_positive(fp_summary: pd.DataFrame, outdir: Path) -> None:
    if plt is None or fp_summary.empty:
        return
    row = fp_summary.iloc[0]
    metrics = ["pct_exact_ec_match", "pct_ec_prefix3_match", "pct_ec_prefix2_match", "pct_ec_prefix1_match"]
    plt.figure(figsize=(7, 4))
    plt.bar(metrics, [row.get(m, np.nan) for m in metrics])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Fraction of top false positives")
    plt.ylim(0, 1)
    plt.title("EC plausibility among reaction-to-enzyme false positives")
    save_plot(outdir / "false_positive_ec_plausibility.png")


def plot_hubness(counters_by_direction: Mapping[str, Mapping[int, Counter]], outdir: Path) -> None:
    if plt is None:
        return
    for direction, counters in counters_by_direction.items():
        for k, counter in counters.items():
            counts = np.asarray(sorted(counter.values(), reverse=True), dtype=float)
            if counts.size == 0:
                continue
            plt.figure(figsize=(7, 4))
            plt.plot(np.arange(1, len(counts) + 1), counts)
            plt.yscale("log")
            plt.xlabel("Entity rank by retrieval frequency")
            plt.ylabel("Retrieval count, log scale")
            plt.title(f"Hubness counts: {direction} top-{k}")
            save_plot(outdir / f"hubness_counts_{direction}_top{k}.png")

            sorted_asc = np.sort(counts)
            cum = np.cumsum(sorted_asc) / sorted_asc.sum()
            x = np.arange(1, len(cum) + 1) / len(cum)
            plt.figure(figsize=(5, 5))
            plt.plot(x, cum, label="retrieval distribution")
            plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="uniform")
            plt.xlabel("Cumulative fraction of entities")
            plt.ylabel("Cumulative fraction of retrieved slots")
            plt.title(f"Hubness Lorenz curve: {direction} top-{k}")
            plt.legend()
            save_plot(outdir / f"hubness_lorenz_{direction}_top{k}.png")


def plot_score_distributions(score_distributions: pd.DataFrame, outdir: Path) -> None:
    if plt is None or score_distributions.empty:
        return
    for dist in score_distributions["distribution"].unique():
        plt.figure(figsize=(8, 4))
        for direction in score_distributions["direction"].unique():
            values = score_distributions[
                (score_distributions["distribution"] == dist) & (score_distributions["direction"] == direction)
            ]["value"].to_numpy(dtype=float)
            if len(values):
                plt.hist(values, bins=60, alpha=0.55, density=True, label=direction)
        plt.xlabel(dist)
        plt.ylabel("Density")
        plt.legend()
        plt.title(f"Score diagnostic distribution: {dist}")
        save_plot(outdir / f"score_distribution_{dist}.png")


def plot_embedding_outputs(outputs: Mapping[str, pd.DataFrame], outdir: Path) -> None:
    if plt is None:
        return
    norms = outputs.get("embedding_norms")
    if norms is not None and not norms.empty:
        plt.figure(figsize=(6, 4))
        x = np.arange(len(norms))
        plt.bar(x, norms["median_l2_norm"], yerr=norms["std_l2_norm"].fillna(0))
        plt.xticks(x, norms["entity_type"])
        plt.ylabel("L2 norm")
        plt.title("Embedding norm summary")
        save_plot(outdir / "embedding_norm_summary.png")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def make_delta(single: pd.DataFrame, multi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for direction in sorted(set(single["direction"]).intersection(multi["direction"])):
        s = single[single["direction"] == direction].iloc[0]
        m = multi[multi["direction"] == direction].iloc[0]
        row = {"direction": direction}
        for metric in ["hit@1", "hit@5", "hit@10", "recall@1", "recall@5", "recall@10", "mrr"]:
            row[f"{metric}_single"] = s.get(metric, np.nan)
            row[f"{metric}_multi"] = m.get(metric, np.nan)
            row[f"{metric}_delta_multi_minus_single"] = m.get(metric, np.nan) - s.get(metric, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def write_readme(
    outdir: Path,
    args: argparse.Namespace,
    score_meta: Mapping[str, Any],
    cardinality: pd.DataFrame,
    random_baselines: pd.DataFrame,
    multi_metrics: pd.DataFrame,
    single_metrics: pd.DataFrame,
    delta_metrics: pd.DataFrame,
    stratified: pd.DataFrame,
    ec_metrics: pd.DataFrame,
    fp_summary: pd.DataFrame,
    hub_summary: pd.DataFrame,
    score_summary: pd.DataFrame,
    leakage: pd.DataFrame,
    loss_table: pd.DataFrame,
    ranked_causes: pd.DataFrame,
    embedding_outputs: Mapping[str, pd.DataFrame],
) -> None:
    test_row = cardinality[cardinality["split"] == "test"].iloc[0]
    candidate_sentence = ""
    if test_row["unique_enzymes"] > test_row["unique_reactions"] * 2:
        candidate_sentence = (
            f"Reaction-to-enzyme has a larger test candidate space: "
            f"{int(test_row['unique_enzymes'])} enzymes vs {int(test_row['unique_reactions'])} reactions."
        )
    elif test_row["unique_reactions"] > test_row["unique_enzymes"] * 2:
        candidate_sentence = (
            f"Enzyme-to-reaction has a larger test candidate space: "
            f"{int(test_row['unique_reactions'])} reactions vs {int(test_row['unique_enzymes'])} enzymes."
        )
    else:
        candidate_sentence = (
            f"The two test candidate spaces are similar in size: "
            f"{int(test_row['unique_enzymes'])} enzymes and {int(test_row['unique_reactions'])} reactions."
        )

    embedding_section = ""
    if embedding_outputs:
        for name, df in embedding_outputs.items():
            embedding_section += f"\n### {name}\n\n{df_to_markdown(df)}\n"
    else:
        embedding_section = "\n_No embedding diagnostics were run because embeddings were not provided._\n"

    content = f"""# ReactZyme Directional Retrieval Diagnostics

Generated: `{datetime.now().isoformat(timespec='seconds')}`

## Inputs

- Train triplets: `{args.train_triplets}`
- Test triplets: `{args.test_triplets}`
- Validation triplets: `{args.val_triplets or 'None'}`
- Scores: `{args.scores}`
- Score format: `{score_meta.get('format')}`
- Enzyme embeddings: `{args.enzyme_embeddings or 'None'}`
- Reaction embeddings: `{args.reaction_embeddings or 'None'}`

## A. Main quantitative findings

{candidate_sentence}

### Dataset cardinality

{df_to_markdown(cardinality)}

### Random top-1 baselines

{df_to_markdown(random_baselines)}

### Multi-positive retrieval metrics

{df_to_markdown(multi_metrics)}

### Single-positive retrieval metrics

{df_to_markdown(single_metrics)}

### Multi-positive improvement

{df_to_markdown(delta_metrics)}

### Positives-per-query stratified metrics

{df_to_markdown(stratified, max_rows=50)}

### Reaction-to-enzyme EC hierarchy metrics

{df_to_markdown(ec_metrics)}

### False-positive EC plausibility

{df_to_markdown(fp_summary)}

### Hubness summary

{df_to_markdown(hub_summary, max_rows=50)}

### Score entropy and calibration

{df_to_markdown(score_summary)}

### Leakage summary

{df_to_markdown(leakage)}

### Training objective inspection

{df_to_markdown(loss_table)}

## B. Ranked likely reasons reaction-to-enzyme is weak

{df_to_markdown(ranked_causes, max_rows=20)}

## C. Concrete next steps

1. Use multi-positive evaluation as the main exact-enzyme metric.
2. Train with multi-positive bidirectional contrastive loss if not already used.
3. Increase the reaction-to-enzyme loss weight, for example `lambda > 1`.
4. Normalize embeddings and evaluate cosine retrieval if dot-product scores show norm or hubness issues.
5. Add EC-prefix-aware soft positives so near-family retrieval is not treated as fully wrong.
6. Add two-stage retrieval: reaction -> EC or EC prefix -> candidate enzymes -> rerank.
7. Report exact enzyme retrieval together with EC-prefix retrieval.
8. If EC-prefix retrieval is high but exact enzyme retrieval is low, improve reaction specificity with reaction-center, chirality, cofactors, and condition features.

## Optional embedding diagnostics

{embedding_section}

## Notes

- Missing enzyme-reaction pairs are treated as unlabeled, not guaranteed negatives.
- Positive-score vs unlabeled-score diagnostics are descriptive only.
- Partial EC labels are used only at known prefix levels.
- Single-positive evaluation uses the lexicographically first known positive per query for determinism.
"""
    (outdir / "README.md").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run complete diagnostics for enzyme->reaction and reaction->enzyme retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-triplets", required=True)
    parser.add_argument("--test-triplets", required=True)
    parser.add_argument("--val-triplets", default=None)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--enzyme-embeddings", default=None)
    parser.add_argument("--reaction-embeddings", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--enzyme-clusters", default=None)
    parser.add_argument("--reaction-smiles", default=None)
    parser.add_argument(
        "--ec-source",
        default="data/paper/reactzyme/raw/cleaned_uniprot_rhea.tsv",
        help="Optional table with protein sequences and EC labels used when triplets lack an EC column.",
    )
    parser.add_argument("--max-unlabeled-score-samples", type=int, default=100000)
    parser.add_argument("--max-cosine-pairs", type=int, default=10000000)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.output_dir)
    tables_dir = outdir / "tables"
    plots_dir = outdir / "plots"
    examples_dir = outdir / "examples"
    ensure_dir(tables_dir)
    ensure_dir(plots_dir)
    ensure_dir(examples_dir)

    ec_by_sequence = read_ec_source(args.ec_source)
    train = read_triplets(args.train_triplets, "train", ec_by_sequence=ec_by_sequence)
    test = read_triplets(args.test_triplets, "test", ec_by_sequence=ec_by_sequence)
    val = read_triplets(args.val_triplets, "val", ec_by_sequence=ec_by_sequence) if args.val_triplets else None
    score_matrix, score_meta = read_scores(args.scores)
    save_json(score_meta, outdir / "score_metadata.json")

    combined_annotations = pd.concat([df for df in (train, val, test) if df is not None], ignore_index=True).drop_duplicates()
    e_to_r, r_to_e = build_positive_sets(test)

    cardinality, random_baselines = dataset_cardinality(train, test, val)
    write_table(cardinality, tables_dir / "dataset_cardinality.csv")
    write_table(random_baselines, tables_dir / "random_baselines.csv")

    e2r_metrics, e2r_details = evaluate_direction(score_matrix, e_to_r, "enzyme_to_reaction")
    r2e_metrics, r2e_details = evaluate_direction(score_matrix, r_to_e, "reaction_to_enzyme")
    multi_metrics = pd.concat([e2r_metrics, r2e_metrics], ignore_index=True)
    write_table(multi_metrics, tables_dir / "retrieval_metrics_multi_positive.csv")
    write_table(e2r_details, tables_dir / "query_details_enzyme_to_reaction_multi_positive.csv")
    write_table(r2e_details, tables_dir / "query_details_reaction_to_enzyme_multi_positive.csv")

    e2r_single_metrics, e2r_single_details = evaluate_direction(score_matrix, sorted_one_positive(e_to_r), "enzyme_to_reaction")
    r2e_single_metrics, r2e_single_details = evaluate_direction(score_matrix, sorted_one_positive(r_to_e), "reaction_to_enzyme")
    single_metrics = pd.concat([e2r_single_metrics, r2e_single_metrics], ignore_index=True)
    write_table(single_metrics, tables_dir / "retrieval_metrics_single_positive.csv")
    write_table(e2r_single_details, tables_dir / "query_details_enzyme_to_reaction_single_positive.csv")
    write_table(r2e_single_details, tables_dir / "query_details_reaction_to_enzyme_single_positive.csv")

    delta_metrics = make_delta(single_metrics, multi_metrics)
    write_table(delta_metrics, tables_dir / "single_vs_multi_delta.csv")

    stratified = pd.concat(
        [
            stratified_metrics(e2r_details, "enzyme_to_reaction"),
            stratified_metrics(r2e_details, "reaction_to_enzyme"),
        ],
        ignore_index=True,
    )
    write_table(stratified, tables_dir / "stratified_metrics.csv")

    enzyme_ecs, reaction_ecs, _ = build_ec_maps(train, test, val if val is not None else pd.DataFrame(columns=TRIPLET_COLUMNS))
    ec_metrics = ec_level_reaction_to_enzyme(r2e_details, test, enzyme_ecs)
    write_table(ec_metrics, tables_dir / "ec_level_metrics_reaction_to_enzyme.csv")

    cluster_map = read_cluster_file(args.enzyme_clusters) if args.enzyme_clusters else {}
    fp_summary, fp_detail = false_positive_plausibility(
        r2e_details,
        test,
        combined_annotations,
        enzyme_ecs,
        cluster_map=cluster_map,
        k=10,
    )
    write_table(fp_summary, tables_dir / "false_positive_ec_plausibility.csv")
    write_table(fp_detail, tables_dir / "false_positive_ec_plausibility_detail.csv")

    e2r_hub_summary, e2r_hub_top, e2r_counters = hubness(e2r_details, "enzyme_to_reaction", score_matrix.shape[1])
    r2e_hub_summary, r2e_hub_top, r2e_counters = hubness(r2e_details, "reaction_to_enzyme", score_matrix.shape[0])
    hub_summary = pd.concat([e2r_hub_summary, r2e_hub_summary], ignore_index=True)
    hub_top = pd.concat([e2r_hub_top, r2e_hub_top], ignore_index=True)
    write_table(hub_summary, tables_dir / "hubness_summary.csv")
    write_table(hub_top, tables_dir / "hubness_top_entities.csv")

    score_summary, score_distributions = score_distribution_diagnostics(
        score_matrix,
        e_to_r,
        r_to_e,
        max_unlabeled_samples=args.max_unlabeled_score_samples,
        seed=args.seed,
    )
    write_table(score_summary, tables_dir / "score_entropy_calibration.csv")
    write_table(score_distributions, tables_dir / "score_distributions_long.csv")

    enzyme_emb = None
    reaction_emb = None
    embedding_outputs: Dict[str, pd.DataFrame] = {}
    if args.enzyme_embeddings:
        enzyme_emb, enzyme_emb_meta = read_embeddings(args.enzyme_embeddings, ["enzyme_id", "protein_id", "id"])
        save_json(enzyme_emb_meta, outdir / "enzyme_embedding_metadata.json")
    if args.reaction_embeddings:
        reaction_emb, reaction_emb_meta = read_embeddings(args.reaction_embeddings, ["reaction_id", "rxn_id", "id"])
        save_json(reaction_emb_meta, outdir / "reaction_embedding_metadata.json")
    if enzyme_emb is not None or reaction_emb is not None:
        embedding_outputs = embedding_diagnostics(
            enzyme_emb,
            reaction_emb,
            test,
            score_matrix,
            e_to_r,
            r_to_e,
            max_pairs=args.max_cosine_pairs,
            seed=args.seed,
        )
        for name, df in embedding_outputs.items():
            write_table(df, tables_dir / f"{name}.csv")

    leakage = leakage_summary(train, test, val, args.reaction_smiles)
    write_table(leakage, tables_dir / "leakage_summary.csv")

    loss_table, loss_md = inspect_loss(args.config_path)
    write_table(loss_table, tables_dir / "loss_inspection.csv")
    (outdir / "loss_inspection.md").write_text(loss_md, encoding="utf-8")

    bad_r2e, asym = error_examples(e2r_details, r2e_details, test, enzyme_ecs, reaction_ecs, n=20)
    write_table(bad_r2e, examples_dir / "bad_reaction_to_enzyme_examples.csv")
    write_table(asym, examples_dir / "asymmetric_success_failure_examples.csv")

    ranked_causes = rank_likely_causes(
        cardinality,
        multi_metrics,
        delta_metrics,
        ec_metrics,
        fp_summary,
        hub_summary,
        score_summary,
        leakage,
        loss_table,
        embedding_outputs,
    )
    write_table(ranked_causes, tables_dir / "main_findings_ranked.csv")

    plot_positive_histograms(test, plots_dir)
    plot_single_multi(single_metrics, multi_metrics, plots_dir)
    plot_stratified(stratified, plots_dir)
    plot_ec_ladder(ec_metrics, plots_dir)
    plot_false_positive(fp_summary, plots_dir)
    plot_hubness(
        {"enzyme_to_reaction": e2r_counters, "reaction_to_enzyme": r2e_counters},
        plots_dir,
    )
    plot_score_distributions(score_distributions, plots_dir)
    plot_embedding_outputs(embedding_outputs, plots_dir)

    write_readme(
        outdir,
        args,
        score_meta,
        cardinality,
        random_baselines,
        multi_metrics,
        single_metrics,
        delta_metrics,
        stratified,
        ec_metrics,
        fp_summary,
        hub_summary,
        score_summary,
        leakage,
        loss_table,
        ranked_causes,
        embedding_outputs,
    )

    print(f"Wrote directional retrieval diagnostics to {outdir}")
    print(f"Main report: {outdir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
