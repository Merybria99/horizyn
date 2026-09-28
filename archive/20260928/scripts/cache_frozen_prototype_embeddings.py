#!/usr/bin/env python3
"""Cache frozen CIRCE tower outputs for fast prototype-only training.

Run directly for one GPU or through ``torch.distributed.run`` for contiguous
multi-GPU sharding.  Each rank encodes a disjoint portion of the requested
reaction/enzyme IDs; rank zero merges compact float16 HDF5 caches.
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
from typing import Any, Iterable

import h5py
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from horizyn.config import load_config
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from horizyn.reaction_conditioned_data_module import DualResidueEmbedDataset
from horizyn.reaction_features import build_reaction_feature_dataset
from horizyn.utils import residue_collate_fn, unimol2_reaction_collate_fn


SCHEMA_VERSION = "frozen_prototype_embedding_cache_v1"


class KeySubsetDataset:
    def __init__(self, dataset: Any, keys: Iterable[str]) -> None:
        self.dataset = dataset
        self.keys = list(keys)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, key: str | int):
        actual_key = self.keys[key] if isinstance(key, int) else key
        return self.dataset[actual_key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--enzyme-batch-size", type=int, default=96)
    parser.add_argument("--reaction-batch-size", type=int, default=512)
    parser.add_argument(
        "--precision",
        choices=("32", "bf16"),
        default="32",
        help="Inference precision used while building the cache",
    )
    parser.add_argument(
        "--output-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Storage dtype; use float32 for exact CIRCE validation/test baselines",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Cache only the configured validation/test reactions and candidates",
    )
    return parser.parse_args()


def file_signature(
    path: str | Path | None,
    *,
    content_hash: bool = False,
) -> dict[str, Any] | None:
    if path is None or str(path) == "":
        return None
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Cache input does not exist: {resolved}")
    stat = resolved.stat()
    signature = {
        "path": str(resolved),
        "size": int(stat.st_size),
    }
    if content_hash:
        signature["sha256"] = sha256(resolved)
    else:
        signature["mtime_ns"] = int(stat.st_mtime_ns)
    return signature


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pair_ids(path: str | Path, direction_mode: str) -> tuple[set[str], set[str]]:
    reaction_ids: set[str] = set()
    enzyme_ids: set[str] = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"reaction_id", "protein_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Pair CSV {path} is missing columns {sorted(missing)}")
        for row in reader:
            reaction_id = str(row["reaction_id"]).strip()
            enzyme_id = str(row["protein_id"]).strip()
            if not reaction_id or not enzyme_id:
                continue
            reaction_ids.add(f"{reaction_id}_f")
            if direction_mode == "bidirectional":
                reaction_ids.add(f"{reaction_id}_r")
            enzyme_ids.add(enzyme_id)
    return reaction_ids, enzyme_ids


def read_id_list(path: str | Path | None) -> set[str]:
    if path is None or str(path) == "":
        return set()
    return {
        line.strip().split(",")[0].split()[0]
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def read_typed_negative_ids(path: str | Path | None) -> set[str]:
    if path is None or str(path) == "":
        return set()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    pools = payload.get("reaction_to_negatives", payload)
    if not isinstance(pools, dict):
        raise ValueError(f"Typed-negative pool has invalid schema: {path}")
    target_ids: set[str] = set()
    for typed_pool in pools.values():
        if not isinstance(typed_pool, dict):
            continue
        for pool_name in ("biological", "random"):
            values = typed_pool.get(pool_name, [])
            if isinstance(values, list):
                target_ids.update(str(value) for value in values)
    return target_ids


def canonical_protein_id(protein_id: str) -> str:
    value = str(protein_id).strip()
    for prefix in ("prot_", "uprot_", "nr90_"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def resolve_enzyme_ids(
    requested_ids: set[str],
    available_ids: set[str],
) -> tuple[set[str], list[str]]:
    """Resolve pool IDs to the unique prefix variant present in the HDF5."""

    available_by_canonical: dict[str, list[str]] = {}
    for available_id in available_ids:
        available_by_canonical.setdefault(
            canonical_protein_id(available_id),
            [],
        ).append(available_id)
    resolved: set[str] = set()
    missing: list[str] = []
    for requested_id in requested_ids:
        if requested_id in available_ids:
            resolved.add(requested_id)
            continue
        matches = available_by_canonical.get(canonical_protein_id(requested_id), [])
        if len(matches) == 1:
            resolved.add(matches[0])
        else:
            missing.append(requested_id)
    return resolved, sorted(missing)


def contiguous_shard(values: list[str], rank: int, world_size: int) -> list[str]:
    start = len(values) * rank // world_size
    end = len(values) * (rank + 1) // world_size
    return values[start:end]


def distributed_context() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed cache extraction requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, torch.device("cuda", local_rank)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, world_size, device


def move_query_inputs(samples: list[Any], device: torch.device) -> Any:
    if not samples:
        raise ValueError("Cannot collate an empty reaction batch")
    if isinstance(samples[0], dict):
        batch = unimol2_reaction_collate_fn(samples)
        return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    return torch.stack(samples).to(device, non_blocking=True)


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def encode_reactions(
    module: ProteinPooledLitModule,
    dataset: Any,
    reaction_ids: list[str],
    *,
    device: torch.device,
    batch_size: int,
    precision: str,
    output_dtype: torch.dtype = torch.float16,
    rank: int,
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    starts = range(0, len(reaction_ids), batch_size)
    iterator = tqdm(
        starts,
        total=(len(reaction_ids) + batch_size - 1) // batch_size,
        desc=f"rank {rank} reactions",
        disable=rank != 0,
    )
    with torch.inference_mode():
        for start in iterator:
            batch_ids = reaction_ids[start : start + batch_size]
            query_inputs = move_query_inputs(
                [dataset[reaction_id] for reaction_id in batch_ids],
                device,
            )
            with autocast_context(device, precision):
                encoded = module.model.encode_queries(
                    query_inputs,
                    retrieval_direction="reaction_to_enzyme",
                )
            encoded = encoded.detach().float().cpu()
            if not bool(torch.isfinite(encoded).all()):
                raise ValueError("Non-finite cached reaction embeddings")
            output.append(encoded.to(dtype=output_dtype))
    embedding_dim = module.model.enzyme_prototype_head.embedding_dim
    return (
        torch.cat(output, dim=0)
        if output
        else torch.empty((0, embedding_dim), dtype=output_dtype)
    )


def encode_enzymes(
    module: ProteinPooledLitModule,
    dataset: Any,
    enzyme_ids: list[str],
    *,
    device: torch.device,
    batch_size: int,
    precision: str,
    output_dtype: torch.dtype = torch.float16,
    rank: int,
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    starts = range(0, len(enzyme_ids), batch_size)
    iterator = tqdm(
        starts,
        total=(len(enzyme_ids) + batch_size - 1) // batch_size,
        desc=f"rank {rank} enzymes",
        disable=rank != 0,
    )
    with torch.inference_mode():
        for start in iterator:
            batch_ids = enzyme_ids[start : start + batch_size]
            samples = []
            for enzyme_id in batch_ids:
                sample = dict(dataset[enzyme_id])
                sample["target_id"] = enzyme_id
                samples.append(sample)
            batch = residue_collate_fn(samples)
            residues = batch["residue_embeddings"].to(device, non_blocking=True)
            residue_mask = batch["residue_padding_mask"].to(device, non_blocking=True)
            score_residues = batch.get("score_residue_embeddings")
            score_mask = batch.get("score_residue_padding_mask")
            if score_residues is not None:
                score_residues = score_residues.to(device, non_blocking=True)
            if score_mask is not None:
                score_mask = score_mask.to(device, non_blocking=True)
            with autocast_context(device, precision):
                encoded = module.model.encode_targets(
                    residues,
                    residue_padding_mask=residue_mask,
                    score_residue_embeddings=score_residues,
                    score_residue_padding_mask=score_mask,
                    retrieval_direction="reaction_to_enzyme",
                )
            encoded = encoded.detach().float().cpu()
            if not bool(torch.isfinite(encoded).all()):
                raise ValueError("Non-finite cached enzyme embeddings")
            output.append(encoded.to(dtype=output_dtype))
    embedding_dim = module.model.enzyme_prototype_head.embedding_dim
    return (
        torch.cat(output, dim=0)
        if output
        else torch.empty((0, embedding_dim), dtype=output_dtype)
    )


def write_embedding_hdf5(
    path: Path,
    ids: list[str],
    vectors: torch.Tensor,
    *,
    checkpoint: Path,
    config: Path,
    output_dtype: str = "float16",
) -> None:
    if vectors.ndim != 2 or vectors.shape[0] != len(ids):
        raise ValueError(f"Invalid cache shape for {path}: {tuple(vectors.shape)}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Cache IDs are not unique for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    if output_dtype not in {"float16", "float32"}:
        raise ValueError("output_dtype must be float16 or float32")
    numpy_dtype = np.float16 if output_dtype == "float16" else np.float32
    array = vectors.detach().cpu().numpy().astype(numpy_dtype, copy=False)
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("ids", data=np.asarray(ids, dtype=object), dtype=string_dtype)
        handle.create_dataset(
            "vectors",
            data=array,
            chunks=(min(max(len(ids), 1), 4096), vectors.shape[1]),
        )
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["checkpoint"] = str(checkpoint.resolve())
        handle.attrs["config"] = str(config.resolve())
        handle.attrs["dtype"] = output_dtype
    os.replace(temporary, path)


def build_signature(args: argparse.Namespace, config: Any) -> dict[str, Any]:
    validation_enabled = bool(config.training.get("validation_enabled", True))
    candidate_path = config.training.get(
        "validation_retrieval_candidate_ids_path",
        config.data.get("validation_retrieval_candidate_ids_path", None),
    )
    reaction_feature_path_keys = (
        "reaction_embeds_path",
        "reaction_t5v2_embeds_path",
        "reaction_model_embeds_path",
        "reaction_unimol2_embeds_path",
        "reaction_chiro_embeds_path",
        "reaction_chirality_embeds_path",
        "reaction_chienn_embeds_path",
        "reaction_chemistry_vectors_path",
        "reaction_directional_vectors_path",
        "train_reaction_embeds_path",
        "validation_reaction_embeds_path",
        "train_reaction_t5v2_embeds_path",
        "validation_reaction_t5v2_embeds_path",
        "train_reaction_model_embeds_path",
        "validation_reaction_model_embeds_path",
        "train_reaction_unimol2_embeds_path",
        "validation_reaction_unimol2_embeds_path",
        "train_reaction_chiro_embeds_path",
        "validation_reaction_chiro_embeds_path",
        "train_reaction_chirality_embeds_path",
        "validation_reaction_chirality_embeds_path",
        "train_reaction_chienn_embeds_path",
        "validation_reaction_chienn_embeds_path",
        "train_reaction_chemistry_vectors_path",
        "validation_reaction_chemistry_vectors_path",
        "train_reaction_directional_vectors_path",
        "validation_reaction_directional_vectors_path",
    )
    reaction_feature_inputs = {
        key: file_signature(config.data.get(key, None))
        for key in reaction_feature_path_keys
        if config.data.get(key, None)
    }
    signatures = {
        "config_sha256": sha256(args.config.resolve()),
        "checkpoint": file_signature(args.checkpoint),
        "train_pairs": file_signature(config.data.train_pairs_path, content_hash=True),
        "train_reactions": file_signature(
            config.data.train_reactions_path,
            content_hash=True,
        ),
        "protein_residues": file_signature(config.data.protein_residue_embeds_path),
        "protein_score_residues": file_signature(
            config.data.get("protein_score_residue_embeds_path", None)
        ),
        "reaction_feature_inputs": reaction_feature_inputs,
        "typed_negative_pools": file_signature(
            config.data.get("typed_negative_pools_path", None),
            content_hash=True,
        ),
        "validation_pairs": None,
        "validation_reactions": None,
        "validation_candidates": file_signature(candidate_path, content_hash=True),
    }
    if validation_enabled:
        signatures["validation_pairs"] = file_signature(
            config.data.validation_pairs_path,
            content_hash=True,
        )
        signatures["validation_reactions"] = file_signature(
            config.data.validation_reactions_path,
            content_hash=True,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "precision": args.precision,
        "output_dtype": getattr(args, "output_dtype", "float16"),
        "validation_only": bool(getattr(args, "validation_only", False)),
        "inputs": signatures,
    }


def expected_outputs(
    output_dir: Path,
    validation_enabled: bool,
    validation_only: bool = False,
) -> list[Path]:
    outputs = [output_dir / "enzyme_base.h5"]
    if not validation_only:
        outputs.append(output_dir / "train_reaction_base.h5")
    if validation_enabled:
        outputs.append(output_dir / "validation_reaction_base.h5")
    return outputs


def cache_is_reusable(
    output_dir: Path,
    signature: dict[str, Any],
    validation_enabled: bool,
    validation_only: bool = False,
) -> bool:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    if any(
        not path.is_file() or path.stat().st_size == 0
        for path in expected_outputs(
            output_dir,
            validation_enabled,
            validation_only,
        )
    ):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("signature") == signature


def main() -> None:
    args = parse_args()
    if args.enzyme_batch_size <= 0 or args.reaction_batch_size <= 0:
        raise ValueError("Cache batch sizes must be positive")
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    config = load_config(args.config)
    # Evaluation configs conventionally expose the released split as ``test_*``
    # while this cache writer calls its second output ``validation``.  Accept
    # either spelling so the exact test protocol can be cached without copying
    # a large model config solely to rename two paths.
    if config.data.get("validation_pairs_path", None) is None:
        test_pairs_path = config.data.get("test_pairs_path", None)
        test_reactions_path = config.data.get("test_reactions_path", None)
        if test_pairs_path is not None and test_reactions_path is not None:
            config.data.validation_pairs_path = test_pairs_path
            config.data.validation_reactions_path = test_reactions_path
    if config.data.get("reaction_direction_mode", "bidirectional") not in {
        "forward_only",
        "bidirectional",
    }:
        raise ValueError("Unsupported reaction_direction_mode for cache extraction")
    for unsupported_key in (
        "protein_capability_vectors_path",
        "protein_factorized_capability_vectors_path",
        "protein_text_vectors_path",
    ):
        if config.data.get(unsupported_key, None):
            raise ValueError(
                f"Cache extraction does not yet support target-side input {unsupported_key}"
            )

    rank, world_size, device = distributed_context()
    validation_enabled = bool(config.training.get("validation_enabled", True))
    signature = build_signature(args, config)
    if args.validation_only and not validation_enabled:
        raise ValueError("--validation-only requires a configured validation/test split")
    reusable = not args.force and cache_is_reusable(
        args.output_dir,
        signature,
        validation_enabled,
        validation_only=args.validation_only,
    )
    if world_size > 1:
        flag = torch.tensor([int(reusable)], dtype=torch.int64, device=device)
        dist.broadcast(flag, src=0)
        reusable = bool(flag.item())
    if reusable:
        if rank == 0:
            print(f"Reusing compatible frozen embedding cache: {args.output_dir}")
        if world_size > 1:
            dist.destroy_process_group()
        return

    direction_mode = config.data.get("reaction_direction_mode", "bidirectional")
    if args.validation_only:
        train_reaction_ids: set[str] = set()
        enzyme_ids: set[str] = set()
    else:
        train_reaction_ids, enzyme_ids = read_pair_ids(
            config.data.train_pairs_path,
            direction_mode,
        )
    validation_reaction_ids: set[str] = set()
    if validation_enabled:
        validation_reaction_ids, validation_enzyme_ids = read_pair_ids(
            config.data.validation_pairs_path,
            direction_mode,
        )
        enzyme_ids.update(validation_enzyme_ids)
        candidate_path = config.training.get(
            "validation_retrieval_candidate_ids_path",
            config.data.get("validation_retrieval_candidate_ids_path", None),
        )
        enzyme_ids.update(read_id_list(candidate_path))
    enzyme_ids.update(read_typed_negative_ids(config.data.get("typed_negative_pools_path", None)))

    residue_dataset = ResidueEmbedDataset(
        file_path=config.data.protein_residue_embeds_path,
        in_memory=False,
        max_tokens=config.data.get("max_protein_tokens", 1024),
        truncation=config.data.get("protein_truncation", "ends_center"),
        validate_finite_on_access=False,
        allow_uncertified_finite_skip=True,
    )
    score_residue_path = config.data.get("protein_score_residue_embeds_path", None)
    if score_residue_path:
        score_dataset = ResidueEmbedDataset(
            file_path=score_residue_path,
            in_memory=False,
            max_tokens=config.data.get("max_protein_tokens", 1024),
            truncation=config.data.get("protein_truncation", "ends_center"),
            validate_finite_on_access=False,
            allow_uncertified_finite_skip=True,
        )
        target_dataset: Any = DualResidueEmbedDataset(
            value_dataset=residue_dataset,
            score_dataset=score_dataset,
        )
    else:
        target_dataset = residue_dataset

    available_target_ids = set(target_dataset.keys)
    enzyme_ids, missing_targets = resolve_enzyme_ids(
        enzyme_ids,
        available_target_ids,
    )
    if missing_targets:
        preview = ", ".join(missing_targets[:5])
        raise ValueError(
            f"{len(missing_targets)} requested enzymes are absent from residue HDF5; "
            f"examples: {preview}"
        )
    physical_order = {key: idx for idx, key in enumerate(residue_dataset.source_keys)}
    ordered_enzyme_ids = sorted(enzyme_ids, key=physical_order.__getitem__)

    train_reaction_dataset = None
    ordered_train_reaction_ids: list[str] = []
    if not args.validation_only:
        train_reaction_dataset = build_reaction_feature_dataset(
            reactions_path=config.data.train_reactions_path,
            config=config,
            bidirectional=True,
            split_name="train",
        )
        missing_train_reactions = sorted(
            train_reaction_ids - set(train_reaction_dataset.keys)
        )
        if missing_train_reactions:
            raise ValueError(
                f"{len(missing_train_reactions)} train reactions lack raw features; "
                f"examples: {', '.join(missing_train_reactions[:5])}"
            )
        ordered_train_reaction_ids = sorted(train_reaction_ids)

    validation_reaction_dataset = None
    ordered_validation_reaction_ids: list[str] = []
    if validation_enabled:
        validation_reaction_dataset = build_reaction_feature_dataset(
            reactions_path=config.data.validation_reactions_path,
            config=config,
            bidirectional=True,
            split_name="validation",
        )
        missing_validation_reactions = sorted(
            validation_reaction_ids - set(validation_reaction_dataset.keys)
        )
        if missing_validation_reactions:
            raise ValueError(
                f"{len(missing_validation_reactions)} validation reactions lack raw "
                f"features; examples: {', '.join(missing_validation_reactions[:5])}"
            )
        ordered_validation_reaction_ids = sorted(validation_reaction_ids)

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            "Frozen cache request: "
            f"enzymes={len(ordered_enzyme_ids)}, "
            f"train_reactions={len(ordered_train_reaction_ids)}, "
            f"validation_reactions={len(ordered_validation_reaction_ids)}, "
            f"world_size={world_size}, precision={args.precision}, "
            f"output_dtype={args.output_dtype}"
        )
    if world_size > 1:
        dist.barrier()

    module = ProteinPooledLitModule.load_from_checkpoint(
        args.checkpoint,
        map_location="cpu",
    )
    if module.model.e2r_adapter is not None or module.model.r2e_adapter is not None:
        raise ValueError(
            "A single frozen cache is invalid for a checkpoint with directional " "E2R/R2E adapters"
    )
    module.eval()
    module.to(device)
    torch.set_float32_matmul_precision(
        "highest"
        if args.precision == "32" and args.output_dtype == "float32"
        else "high"
    )
    output_dtype = torch.float16 if args.output_dtype == "float16" else torch.float32

    local_enzyme_ids = contiguous_shard(ordered_enzyme_ids, rank, world_size)
    local_train_reaction_ids = contiguous_shard(
        ordered_train_reaction_ids,
        rank,
        world_size,
    )
    local_validation_reaction_ids = contiguous_shard(
        ordered_validation_reaction_ids,
        rank,
        world_size,
    )
    local_payload = {
        "enzyme_ids": local_enzyme_ids,
        "enzyme_vectors": encode_enzymes(
            module,
            KeySubsetDataset(target_dataset, local_enzyme_ids),
            local_enzyme_ids,
            device=device,
            batch_size=args.enzyme_batch_size,
            precision=args.precision,
            output_dtype=output_dtype,
            rank=rank,
        ),
        "train_reaction_ids": local_train_reaction_ids,
        "train_reaction_vectors": (
            encode_reactions(
                module,
                train_reaction_dataset,
                local_train_reaction_ids,
                device=device,
                batch_size=args.reaction_batch_size,
                precision=args.precision,
                output_dtype=output_dtype,
                rank=rank,
            )
            if train_reaction_dataset is not None
            else torch.empty(
                (0, module.model.enzyme_prototype_head.embedding_dim),
                dtype=output_dtype,
            )
        ),
        "validation_reaction_ids": local_validation_reaction_ids,
        "validation_reaction_vectors": (
            encode_reactions(
                module,
                validation_reaction_dataset,
                local_validation_reaction_ids,
                device=device,
                batch_size=args.reaction_batch_size,
                precision=args.precision,
                output_dtype=output_dtype,
                rank=rank,
            )
            if validation_reaction_dataset is not None
            else torch.empty(
                (0, module.model.enzyme_prototype_head.embedding_dim),
                dtype=output_dtype,
            )
        ),
    }
    shard_path = args.output_dir / f".rank-{rank:04d}.pt"
    torch.save(local_payload, shard_path)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        shards = [
            torch.load(
                args.output_dir / f".rank-{shard_rank:04d}.pt",
                map_location="cpu",
                weights_only=False,
            )
            for shard_rank in range(world_size)
        ]

        def merge(prefix: str) -> tuple[list[str], torch.Tensor]:
            ids = [value for shard in shards for value in shard[f"{prefix}_ids"]]
            vectors = torch.cat(
                [shard[f"{prefix}_vectors"] for shard in shards],
                dim=0,
            )
            return ids, vectors

        merged_enzyme_ids, merged_enzyme_vectors = merge("enzyme")
        merged_train_ids, merged_train_vectors = merge("train_reaction")
        write_embedding_hdf5(
            args.output_dir / "enzyme_base.h5",
            merged_enzyme_ids,
            merged_enzyme_vectors,
            checkpoint=args.checkpoint,
            config=args.config,
            output_dtype=args.output_dtype,
        )
        if not args.validation_only:
            write_embedding_hdf5(
                args.output_dir / "train_reaction_base.h5",
                merged_train_ids,
                merged_train_vectors,
                checkpoint=args.checkpoint,
                config=args.config,
                output_dtype=args.output_dtype,
            )
        counts = {
            "enzymes": len(merged_enzyme_ids),
            "train_reactions": len(merged_train_ids),
            "validation_reactions": 0,
        }
        if validation_enabled:
            merged_validation_ids, merged_validation_vectors = merge("validation_reaction")
            write_embedding_hdf5(
                args.output_dir / "validation_reaction_base.h5",
                merged_validation_ids,
                merged_validation_vectors,
                checkpoint=args.checkpoint,
                config=args.config,
                output_dtype=args.output_dtype,
            )
            counts["validation_reactions"] = len(merged_validation_ids)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "signature": signature,
            "counts": counts,
            "embedding_dim": int(merged_enzyme_vectors.shape[1]),
            "dtype": args.output_dtype,
            "world_size": world_size,
            "enzyme_batch_size_per_rank": args.enzyme_batch_size,
            "reaction_batch_size_per_rank": args.reaction_batch_size,
        }
        manifest_path = args.output_dir / "manifest.json"
        temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + f".tmp.{os.getpid()}")
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
        for shard_rank in range(world_size):
            (args.output_dir / f".rank-{shard_rank:04d}.pt").unlink(missing_ok=True)
        print(json.dumps(manifest, indent=2, sort_keys=True))

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
