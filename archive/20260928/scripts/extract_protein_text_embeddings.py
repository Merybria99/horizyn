#!/usr/bin/env python3
"""Extract per-protein text embeddings for TIGER-style enzyme fusion."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


DEFAULT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


def read_text_rows(path: Path, id_column: str, text_column: str) -> list[tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd

        table = pd.read_parquet(path)
        for column in (id_column, text_column):
            if column not in table.columns:
                raise ValueError(f"Missing required column {column!r} in {path}")
        raw_rows = (
            (str(row[id_column]).strip(), "" if row[text_column] is None else str(row[text_column]))
            for _, row in table[[id_column, text_column]].iterrows()
        )
    elif suffix in {".tsv", ".tab", ".csv"}:
        delimiter = "\t" if suffix in {".tsv", ".tab"} else ","
        handle = path.open(newline="", encoding="utf-8")
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            handle.close()
            raise ValueError(f"Input table has no header: {path}")
        for column in (id_column, text_column):
            if column not in reader.fieldnames:
                handle.close()
                raise ValueError(f"Missing required column {column!r} in {path}")
        raw_rows = (
            (str(row.get(id_column, "")).strip(), str(row.get(text_column, "") or ""))
            for row in reader
        )
    else:
        raise ValueError(f"Unsupported input extension: {path.suffix}")

    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    try:
        for protein_id, text in raw_rows:
            if not protein_id or protein_id in seen:
                continue
            seen.add(protein_id)
            rows.append((protein_id, text))
    finally:
        if "handle" in locals():
            handle.close()
    return rows


def read_table(path: Path):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input extension: {path.suffix}")


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(dtype=last_hidden.dtype).unsqueeze(-1)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="CSV/TSV/parquet with protein text")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--id-column", default="protein_id", help="Protein ID column")
    parser.add_argument("--text-column", default="text", help="Text description column")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Hugging Face model name/path")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--metadata-output", default=None, help="Optional metadata JSON path")
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=100,
        help="Print extraction progress every N batches; set <=0 to disable.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    if Path(args.output).suffix.lower() != ".npz":
        raise ValueError("--output must end with .npz")


def main() -> None:
    args = parse_args()
    validate_args(args)

    from transformers import AutoModel, AutoTokenizer

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_text_rows(input_path, args.id_column, args.text_column)
    if not rows:
        raise ValueError("No valid protein text rows found")

    ids = [protein_id for protein_id, _text in rows]
    texts = [text for _protein_id, text in rows]
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.eval().to(device)

    vectors: list[np.ndarray] = []
    total_batches = (len(texts) + args.batch_size - 1) // args.batch_size
    with torch.inference_mode():
        for batch_idx, start in enumerate(range(0, len(texts), args.batch_size), start=1):
            batch_texts = texts[start : start + args.batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            if args.pooling == "cls":
                pooled = output.last_hidden_state[:, 0]
            else:
                pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            vectors.append(pooled.detach().cpu().float().numpy())
            if args.progress_every_batches > 0 and (
                batch_idx == 1
                or batch_idx == total_batches
                or batch_idx % args.progress_every_batches == 0
            ):
                end = min(start + args.batch_size, len(texts))
                print(
                    f"Encoded text batch {batch_idx}/{total_batches} "
                    f"({end}/{len(texts)} proteins)",
                    flush=True,
                )

    matrix = np.concatenate(vectors, axis=0).astype(np.float32, copy=False)
    np.savez_compressed(
        output_path,
        ids=np.asarray(ids, dtype=object),
        vectors=matrix,
    )

    metadata: dict[str, Any] = {
        "input": str(input_path),
        "output": str(output_path),
        "num_proteins": int(len(ids)),
        "embedding_dim": int(matrix.shape[1]),
        "model_name": args.model_name,
        "pooling": args.pooling,
        "max_length": int(args.max_length),
    }
    metadata_path = (
        Path(args.metadata_output)
        if args.metadata_output
        else output_path.with_suffix(".metadata.json")
    )
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {len(ids)} text embeddings to {output_path}")


if __name__ == "__main__":
    main()
