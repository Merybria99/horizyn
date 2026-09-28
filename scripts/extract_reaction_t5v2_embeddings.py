#!/usr/bin/env python3
"""
Extract frozen transformer reaction embeddings from reaction SMILES.

The output HDF5 schema is:
    /ids      string reaction IDs
    /vectors  fixed reaction embeddings [N, D]

This script is intentionally model-name driven. For ReactionT5v2, pass the
appropriate Hugging Face model identifier or local checkpoint with --model-name.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from horizyn.chemistry.standardizer import Standardizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reactions", nargs="+", required=True, help="Reaction CSV files")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument("--model-name", required=True, help="HF/local ReactionT5v2 model name")
    parser.add_argument("--key-column", default="reaction_id")
    parser.add_argument("--smiles-column", default="reaction_smiles")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pooling", choices=("mean", "cls"), default="mean")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--limit-reactions",
        type=int,
        default=None,
        help="Optional maximum number of base reactions to read for smoke tests.",
    )
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--standardize-hypervalent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--standardize-remove-hs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--standardize-kekulize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--standardize-uncharge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--standardize-metals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-pseudo-reactions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Convert molecule-set strings with no reaction arrow into self-reactions.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output if it exists")
    return parser.parse_args()


def normalize_smiles(
    smiles: str,
    standardizer: Standardizer | None,
    allow_pseudo_reactions: bool,
) -> str:
    if smiles.count(">") < 2:
        if not allow_pseudo_reactions:
            return smiles
        smiles = f"{smiles}>>{smiles}"
    return standardizer.standardize_reaction(smiles) if standardizer is not None else smiles


def read_reactions(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    standardizer = (
        Standardizer(
            standardize_hypervalent=args.standardize_hypervalent,
            standardize_remove_hs=args.standardize_remove_hs,
            standardize_kekulize=args.standardize_kekulize,
            standardize_uncharge=args.standardize_uncharge,
            standardize_metals=args.standardize_metals,
        )
        if args.standardize
        else None
    )
    ids: list[str] = []
    smiles_values: list[str] = []
    seen: set[str] = set()
    for path_str in args.reactions:
        with Path(path_str).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if args.key_column not in reader.fieldnames or args.smiles_column not in reader.fieldnames:
                raise KeyError(
                    f"{path_str} must contain {args.key_column!r} and {args.smiles_column!r}"
                )
            for row in reader:
                if args.limit_reactions is not None and len(seen) >= args.limit_reactions:
                    break
                base_id = str(row[args.key_column])
                smiles = normalize_smiles(
                    str(row[args.smiles_column]),
                    standardizer,
                    allow_pseudo_reactions=args.allow_pseudo_reactions,
                )
                directions = (
                    ((base_id, smiles),)
                    if not args.bidirectional or ">>" not in smiles
                    else (
                        (f"{base_id}_f", smiles),
                        (f"{base_id}_r", ">>".join(reversed(smiles.split(">>", maxsplit=1)))),
                    )
                )
                for reaction_id, reaction_smiles in directions:
                    if reaction_id in seen:
                        continue
                    seen.add(reaction_id)
                    ids.append(reaction_id)
                    smiles_values.append(reaction_smiles)
    return ids, smiles_values


def load_transformer(model_name: str, device: str):
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for ReactionT5v2 extraction. "
            "Install it in the active environment or precompute embeddings externally."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval().to(device)
    return tokenizer, model


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    if pooling == "cls":
        return hidden_states[:, 0]
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def encode_smiles(args: argparse.Namespace, smiles_values: list[str]) -> np.ndarray:
    tokenizer, model = load_transformer(args.model_name, args.device)
    vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(smiles_values), args.batch_size):
            batch_smiles = smiles_values[start : start + args.batch_size]
            tokens = tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(args.device) for key, value in tokens.items()}
            if getattr(model.config, "is_encoder_decoder", False):
                encoder = model.get_encoder()
                output = encoder(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                )
            else:
                output = model(**tokens)
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                raise ValueError("Model output does not contain last_hidden_state")
            pooled = pool_hidden_states(hidden, tokens["attention_mask"], args.pooling)
            vectors.append(pooled.detach().cpu().float().numpy())
    return np.concatenate(vectors, axis=0)


def write_hdf5(
    path: Path,
    ids: list[str],
    vectors: np.ndarray,
    dtype: str,
    model_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text_dtype = h5py.string_dtype("utf-8")
    stored_vectors = vectors.astype(np.float16 if dtype == "float16" else np.float32)
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("ids", data=np.asarray(ids, dtype=object), dtype=text_dtype)
        h5_file.create_dataset("vectors", data=stored_vectors)
        h5_file.attrs["embedding_dim"] = int(stored_vectors.shape[1])
        h5_file.attrs["model_name"] = model_name
        h5_file.attrs["representation"] = "reaction_transformer"


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output}. Use --force to overwrite.")
    ids, smiles_values = read_reactions(args)
    if not ids:
        raise ValueError("No reactions found")
    vectors = encode_smiles(args, smiles_values)
    write_hdf5(output, ids, vectors, args.dtype, args.model_name)
    print(f"Wrote {len(ids)} reaction embeddings to {output} with dim {vectors.shape[1]}")


if __name__ == "__main__":
    main()
