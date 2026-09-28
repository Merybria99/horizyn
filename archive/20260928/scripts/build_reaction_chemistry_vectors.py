#!/usr/bin/env python3
"""Build reaction chemistry multihot vectors from reaction-feature parquet rows."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_COLUMNS = (
    "core_cofactor_labels",
    "reaction_center_coarse_labels",
    "substrate_product_transition_labels",
    "reaction_type_labels",
)


def _as_labels(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        if bool(pd.isna(value)):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist() if str(item)]
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return [str(item) for item in parsed if str(item)]
    return [text]


def _build_vocab(rows, columns: tuple[str, ...], min_count: int) -> list[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for column in columns:
            family = column.replace("_labels", "").replace("_coarse", "")
            for label in _as_labels(row.get(column)):
                counts[f"{family}:{label}"] += 1
    return sorted(label for label, count in counts.items() if count >= min_count)


def _load_vocab(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        vocab = payload.get("vocab", payload.get("labels", []))
    else:
        vocab = payload
    if not isinstance(vocab, list):
        raise ValueError(f"Invalid vocab JSON: {path}")
    return [str(item) for item in vocab]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaction-features", required=True, type=Path)
    parser.add_argument("--out-npz", required=True, type=Path)
    parser.add_argument("--out-vocab", type=Path, default=None)
    parser.add_argument("--vocab-json", type=Path, default=None)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--output-dim", type=int, default=None)
    parser.add_argument("--columns", nargs="*", default=list(DEFAULT_COLUMNS))
    args = parser.parse_args()

    if args.min_count <= 0:
        raise ValueError("--min-count must be positive")
    frame = pd.read_parquet(args.reaction_features)
    if "reaction_id" not in frame.columns:
        raise ValueError("--reaction-features must contain reaction_id")
    columns = tuple(column for column in args.columns if column in frame.columns)
    if not columns:
        raise ValueError("None of the requested label columns are present")
    rows = frame.to_dict("records")
    vocab = _load_vocab(args.vocab_json) if args.vocab_json else _build_vocab(rows, columns, args.min_count)
    if args.output_dim is not None:
        if args.output_dim <= 0:
            raise ValueError("--output-dim must be positive")
        if len(vocab) > args.output_dim:
            vocab = vocab[: args.output_dim]
        elif len(vocab) < args.output_dim:
            vocab = vocab + [f"__pad_{idx}" for idx in range(args.output_dim - len(vocab))]
    index = {label: idx for idx, label in enumerate(vocab)}
    ids = []
    vectors = np.zeros((len(rows), len(vocab)), dtype=np.float32)
    mask = np.zeros(len(rows), dtype=bool)
    for row_idx, row in enumerate(rows):
        ids.append(str(row["reaction_id"]))
        for column in columns:
            family = column.replace("_labels", "").replace("_coarse", "")
            for label in _as_labels(row.get(column)):
                idx = index.get(f"{family}:{label}")
                if idx is not None:
                    vectors[row_idx, idx] = 1.0
        mask[row_idx] = bool(vectors[row_idx].sum() > 0)

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        ids=np.asarray(ids, dtype=object),
        vectors=vectors,
        mask=mask,
        labels=np.asarray(vocab, dtype=object),
    )
    if args.out_vocab is not None:
        args.out_vocab.parent.mkdir(parents=True, exist_ok=True)
        args.out_vocab.write_text(
            json.dumps(
                {
                    "vocab": vocab,
                    "columns": list(columns),
                    "num_reactions": int(len(ids)),
                    "num_labeled": int(mask.sum()),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
