#!/usr/bin/env python3
"""Build the one-time compact SLEEC/ProtT5 functional-residue cache.

The script supports ``torchrun``.  Each rank reads a disjoint set of proteins,
writes a temporary shard, and rank zero merges the shards into one HDF5 file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from horizyn.biological_residual import (  # noqa: E402
    FUNCTIONAL_TOKEN_SCHEMA_VERSION,
    select_functional_token_indices,
)
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset  # noqa: E402
from horizyn.model import FunctionalResidueScorer  # noqa: E402
from horizyn.utils import residue_collate_fn  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residue-h5", required=True, type=Path)
    parser.add_argument("--sleec-checkpoint", type=Path)
    parser.add_argument("--selection", choices=("sleec", "random"), default="sleec")
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument(
        "--pairs",
        required=True,
        action="append",
        type=Path,
        help="Pair CSV used to select proteins; may be supplied more than once",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--context-k", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1022)
    parser.add_argument("--scorer-hidden-dim", type=int, default=256)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def canonical_protein_id(value: str) -> str:
    text = str(value).strip()
    for prefix in ("prot_", "uprot_", "nr90_"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def requested_proteins(paths: list[Path]) -> set[str]:
    values: set[str] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "protein_id" not in (reader.fieldnames or []):
                raise ValueError(f"Pair CSV lacks protein_id: {path}")
            values.update(
                str(row["protein_id"]).strip()
                for row in reader
                if str(row.get("protein_id", "")).strip()
            )
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_signature(path: Path, *, hash_content: bool) -> dict[str, object]:
    resolved = path.resolve()
    stat = resolved.stat()
    signature: dict[str, object] = {
        "path": str(resolved),
        "size": int(stat.st_size),
    }
    if hash_content:
        signature["sha256"] = _sha256(resolved)
    else:
        signature["mtime_ns"] = int(stat.st_mtime_ns)
    return signature


def _cache_metadata(
    args: argparse.Namespace,
    *,
    requested_count: int,
    resolved_count: int,
    missing_count: int,
) -> dict[str, object]:
    metadata = {
        "residue_h5": _file_signature(args.residue_h5, hash_content=False),
        "sleec_checkpoint": (
            _file_signature(args.sleec_checkpoint, hash_content=True)
            if args.selection == "sleec"
            else None
        ),
        "pair_files": [_file_signature(path, hash_content=True) for path in args.pairs],
        "requested_proteins": requested_count,
        "resolved_proteins": resolved_count,
        "missing_proteins": missing_count,
        "top_k": args.top_k,
        "context_k": args.context_k,
        "max_tokens": args.max_tokens,
        "scorer_hidden_dim": args.scorer_hidden_dim,
        "bf16": bool(args.bf16),
    }
    # Preserve compatibility with existing guided caches; random caches must
    # never be mistaken for them or reused with a different selection seed.
    if args.selection != "sleec":
        metadata.update(selection=args.selection, selection_seed=args.selection_seed)
    return metadata


def random_selection_scores(ids: list[str], valid: torch.Tensor, seed: int) -> torch.Tensor:
    """Per-protein random priorities, invariant to rank, padding and batch order."""
    scores = torch.zeros(valid.shape, dtype=torch.float32)
    for row, protein_id in enumerate(ids):
        digest = hashlib.sha256(f"{seed}:{canonical_protein_id(protein_id)}".encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
        positions = valid[row].nonzero().flatten().cpu()
        # A permutation avoids ties and gives uniform sampling without replacement.
        scores[row, positions] = torch.randperm(len(positions), generator=generator).float()
    return scores.to(valid.device)


def _cache_is_compatible(
    output: Path,
    expected_ids: list[str],
    expected_metadata: dict[str, object],
) -> bool:
    try:
        with h5py.File(output, "r") as handle:
            schema = handle.attrs.get("schema_version", "")
            if isinstance(schema, bytes):
                schema = schema.decode("utf-8")
            metadata_json = handle.attrs.get("metadata_json", "{}")
            if isinstance(metadata_json, bytes):
                metadata_json = metadata_json.decode("utf-8")
            metadata = json.loads(str(metadata_json))
            stored_ids = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in handle["ids"][:]
            ]
            return (
                schema == FUNCTIONAL_TOKEN_SCHEMA_VERSION
                and metadata == expected_metadata
                and stored_ids == expected_ids
                and handle["token_vectors"].shape[:2]
                == (len(expected_ids), _token_width(expected_metadata))
            )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _token_width(metadata: dict[str, object]) -> int:
    return int(metadata["top_k"]) + int(metadata["context_k"])


def resolve_ids(requested: set[str], available: list[str]) -> tuple[list[str], list[str]]:
    available_set = set(available)
    by_canonical: dict[str, list[str]] = {}
    for value in available:
        by_canonical.setdefault(canonical_protein_id(value), []).append(value)
    resolved: set[str] = set()
    missing: list[str] = []
    for value in requested:
        if value in available_set:
            resolved.add(value)
            continue
        candidates = by_canonical.get(canonical_protein_id(value), [])
        if len(candidates) == 1:
            resolved.add(candidates[0])
        else:
            missing.append(value)
    return sorted(resolved), sorted(missing)


def distributed_context() -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-process cache extraction requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return rank, world_size, torch.device("cuda", local_rank)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def write_shard(
    path: Path,
    ids: list[str],
    dataset: ResidueEmbedDataset,
    scorer: FunctionalResidueScorer | None,
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
) -> None:
    width = args.top_k + args.context_k
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = FUNCTIONAL_TOKEN_SCHEMA_VERSION
        handle.create_dataset("ids", data=np.asarray(ids, dtype=object), dtype=string_dtype)
        vector_options = (
            {"chunks": (1, width, dataset.vec_dim), "compression": "lzf"} if ids else {}
        )
        vectors_out = handle.create_dataset(
            "token_vectors",
            shape=(len(ids), width, dataset.vec_dim),
            dtype="float16",
            **vector_options,
        )
        mask_out = handle.create_dataset("token_mask", shape=(len(ids), width), dtype="bool")
        scores_out = handle.create_dataset("sleec_scores", shape=(len(ids), width), dtype="float16")
        positions_out = handle.create_dataset(
            "token_positions", shape=(len(ids), width), dtype="int32"
        )
        starts = range(0, len(ids), args.batch_size)
        progress = tqdm(
            starts,
            total=(len(ids) + args.batch_size - 1) // args.batch_size,
            desc=f"rank {rank} {args.selection} tokens",
            disable=rank != 0,
        )
        with torch.inference_mode():
            for start in progress:
                batch_ids = ids[start : start + args.batch_size]
                samples = [dataset[value] for value in batch_ids]
                batch = residue_collate_fn(samples)
                residues = batch["residue_embeddings"].to(device, non_blocking=True)
                valid = ~batch["residue_padding_mask"].to(device, non_blocking=True)
                autocast = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" and args.bf16
                    else nullcontext()
                )
                if args.selection == "random":
                    scores = random_selection_scores(batch_ids, valid, args.selection_seed)
                else:
                    with autocast:
                        _logits, scores = scorer(residues, attention_mask=valid)
                indices, selected_mask = select_functional_token_indices(
                    scores.float(), valid, top_k=args.top_k, context_k=args.context_k
                )
                gather_idx = indices.unsqueeze(-1).expand(-1, -1, residues.shape[-1])
                selected = residues.gather(1, gather_idx).float()
                selected_scores = scores.float().gather(1, indices)
                if args.selection == "random":
                    selected_scores.zero_()  # No biological or random-priority scoring bias.
                selected = selected.masked_fill(~selected_mask.unsqueeze(-1), 0.0)
                selected_scores = selected_scores.masked_fill(~selected_mask, 0.0)
                if not bool(torch.isfinite(selected).all()):
                    raise ValueError("Non-finite selected residue token")
                end = start + len(batch_ids)
                vectors_out[start:end] = selected.cpu().half().numpy()
                mask_out[start:end] = selected_mask.cpu().numpy()
                scores_out[start:end] = selected_scores.cpu().half().numpy()
                positions_out[start:end] = indices.cpu().int().numpy()


def merge_shards(output: Path, shards: list[Path], metadata: dict[str, object]) -> None:
    counts: list[int] = []
    width = 0
    dim = 0
    for shard in shards:
        with h5py.File(shard, "r") as handle:
            counts.append(len(handle["ids"]))
            width = int(handle["token_vectors"].shape[1])
            dim = int(handle["token_vectors"].shape[2])
    total = sum(counts)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".merge.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(temporary, "w") as target:
        target.attrs["schema_version"] = FUNCTIONAL_TOKEN_SCHEMA_VERSION
        target.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        ids_out = target.create_dataset("ids", shape=(total,), dtype=string_dtype)
        vectors_out = target.create_dataset(
            "token_vectors",
            shape=(total, width, dim),
            dtype="float16",
            chunks=(1, width, dim),
            compression="lzf",
        )
        mask_out = target.create_dataset("token_mask", shape=(total, width), dtype="bool")
        scores_out = target.create_dataset("sleec_scores", shape=(total, width), dtype="float16")
        positions_out = target.create_dataset(
            "token_positions", shape=(total, width), dtype="int32"
        )
        cursor = 0
        for shard, count in zip(shards, counts):
            with h5py.File(shard, "r") as source:
                for local_start in range(0, count, 256):
                    local_end = min(local_start + 256, count)
                    target_start = cursor + local_start
                    target_end = cursor + local_end
                    ids_out[target_start:target_end] = source["ids"][local_start:local_end]
                    vectors_out[target_start:target_end] = source["token_vectors"][
                        local_start:local_end
                    ]
                    mask_out[target_start:target_end] = source["token_mask"][local_start:local_end]
                    scores_out[target_start:target_end] = source["sleec_scores"][
                        local_start:local_end
                    ]
                    positions_out[target_start:target_end] = source["token_positions"][
                        local_start:local_end
                    ]
                cursor += count
        target.flush()
    os.replace(temporary, output)


def main() -> None:
    args = parse_args()
    if args.selection == "sleec" and args.sleec_checkpoint is None:
        raise ValueError("SLEEC selection requires --sleec-checkpoint")
    if args.batch_size <= 0 or args.top_k < 0 or args.context_k < 0:
        raise ValueError("Invalid batch/token counts")
    dataset = ResidueEmbedDataset(
        str(args.residue_h5),
        in_memory=False,
        max_tokens=args.max_tokens,
        truncation="ends_center",
    )
    requested = requested_proteins(args.pairs)
    ids, missing = resolve_ids(requested, list(dataset.keys))
    if not ids:
        raise ValueError("None of the requested proteins exist in the residue HDF5")
    metadata = _cache_metadata(
        args,
        requested_count=len(requested),
        resolved_count=len(ids),
        missing_count=len(missing),
    )
    if args.output.exists() and not args.force:
        if _cache_is_compatible(args.output, ids, metadata):
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"Reusing compatible functional-token cache: {args.output}")
            return
        raise FileExistsError(
            f"Functional-token cache is incompatible or incomplete: {args.output}; "
            "rerun with --force"
        )
    rank, world_size, device = distributed_context()
    shard_ids = ids[len(ids) * rank // world_size : len(ids) * (rank + 1) // world_size]
    scorer = None
    if args.selection == "sleec":
        scorer = FunctionalResidueScorer(
            hidden_dim=dataset.vec_dim,
            scorer_hidden_dim=args.scorer_hidden_dim,
        )
        scorer.load_stage1_checkpoint(str(args.sleec_checkpoint))
        scorer.eval().to(device)
    shard_paths = [
        args.output.with_name(f".{args.output.name}.rank{value}.tmp.h5")
        for value in range(world_size)
    ]
    write_shard(shard_paths[rank], shard_ids, dataset, scorer, args, device, rank)
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        merge_shards(args.output, shard_paths, metadata)
        for shard in shard_paths:
            shard.unlink(missing_ok=True)
        print(json.dumps(metadata, indent=2))
        if missing:
            print(f"Warning: {len(missing)} requested proteins were not resolved")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
