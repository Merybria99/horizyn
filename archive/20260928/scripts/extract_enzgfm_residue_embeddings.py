#!/usr/bin/env python
"""Extract unpooled EnzGFM residue embeddings into Horizyn ragged HDF5.

This adapter uses the authors' ``EnzGFM_Model`` implementation and tokenizer,
which are distributed separately from the pretrained weights. The output has
the same ID/vector/offset schema as the existing ProtT5 and ESM2 extractors so
it can be used as ``data.protein_residue_embeds_path`` while ProtT5 embeddings
remain available through ``data.protein_score_residue_embeds_path``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import h5py
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_esm2_residue_embeddings import (  # noqa: E402
    ProteinRecord,
    extract_residue_tokens,
    get_shard_path,
    hdf5_compression,
    iter_batches,
    merge_shards,
    numpy_dtype,
    parse_fasta,
    residue_embedding_dim,
    truncate_sequence,
    write_embedding_attrs,
)


DEFAULT_MODEL_LOCATION = "checkpoints/EnzGFM-650M"
DEFAULT_ENZGFM_REPO = "third_party/EnzGFM"
DEFAULT_MAX_SEQUENCE_LENGTH = 1022


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, help="Input protein FASTA")
    parser.add_argument("--output", required=True, help="Final merged ragged HDF5 output")
    parser.add_argument(
        "--model-location",
        default=DEFAULT_MODEL_LOCATION,
        help="Local EnzGFM pretrained model directory (for example EnzGFM-650M)",
    )
    parser.add_argument(
        "--enzgfm-repo",
        default=DEFAULT_ENZGFM_REPO,
        help="Checkout of the official DeepBxM/EnzGFM repository",
    )
    parser.add_argument(
        "--tokenizer-location",
        default=None,
        help="Tokenizer directory; defaults to <enzgfm-repo>/EsmTokenizer",
    )
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="Directory for rank shard files; defaults beside the output",
    )
    parser.add_argument("--rank", type=int, default=0, help="Shard rank for this worker")
    parser.add_argument("--world-size", type=int, default=1, help="Number of shard workers")
    parser.add_argument("--device", default="cuda", help="Torch device for extraction")
    parser.add_argument("--batch-size", type=int, default=4, help="Maximum proteins per batch")
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=2048,
        help="Soft residue-token budget per batch",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
        help="Residues retained before adding the two ESM special tokens",
    )
    parser.add_argument(
        "--sequence-truncation",
        choices=("ends_center",),
        default="ends_center",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Model/storage dtype (CPU extraction always loads the model in float32)",
    )
    parser.add_argument(
        "--compression",
        choices=("none", "gzip", "lzf"),
        default="none",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--disable-fast-kernels",
        action="store_true",
        help="Use the reference Mamba path, primarily for CPU/debug extraction",
    )
    parser.add_argument("--load-only", action="store_true", help="Load model/tokenizer and exit")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cleanup-shards", action="store_true")
    return parser.parse_args()


def tokenizer_location(args: argparse.Namespace) -> Path:
    if args.tokenizer_location is not None:
        return Path(args.tokenizer_location)
    return Path(args.enzgfm_repo) / "EsmTokenizer"


def load_official_models_package(repo_path: str | Path) -> ModuleType:
    """Load only the authors' EnzGFM modules, avoiding unrelated package imports."""
    models_dir = Path(repo_path).resolve() / "models"
    config_path = models_dir / "configuration_EnzGFM.py"
    model_path = models_dir / "modeling_EnzGFM.py"
    if not config_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(
            "EnzGFM model sources were not found. Clone "
            "https://github.com/DeepBxM/EnzGFM and pass --enzgfm-repo; "
            f"expected {config_path} and {model_path}"
        )

    module_name = "_horizyn_official_enzgfm_models"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    # Executing the official models/__init__.py also imports pretraining helpers
    # (and pandas), none of which are required for feature extraction. Register a
    # lightweight package so the two required files can still use relative imports.
    module = ModuleType(module_name)
    module.__package__ = module_name
    module.__path__ = [str(models_dir)]
    sys.modules[module_name] = module

    for source_name, source_path in (
        ("configuration_EnzGFM", config_path),
        ("modeling_EnzGFM", model_path),
    ):
        qualified_name = f"{module_name}.{source_name}"
        spec = importlib.util.spec_from_file_location(qualified_name, source_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create an import specification for {source_path}")
        source_module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = source_module
        spec.loader.exec_module(source_module)

    module.EnzGFMConfig = sys.modules[f"{module_name}.configuration_EnzGFM"].EnzGFMConfig
    modeling_module = sys.modules[f"{module_name}.modeling_EnzGFM"]
    module.EnzGFM_Model = modeling_module.EnzGFM_Model
    return module


def load_model_and_tokenizer(
    model_location: str,
    enzgfm_repo: str,
    tokenizer_path: str | Path,
    device: str,
    dtype_name: str,
    *,
    use_fast_kernels: bool,
) -> tuple[torch.nn.Module, Any]:
    try:
        from transformers import EsmTokenizer
    except ImportError as exc:
        raise ImportError(
            "EnzGFM extraction requires transformers plus the dependencies in the "
            "official EnzGFM environment."
        ) from exc

    package = load_official_models_package(enzgfm_repo)
    config = package.EnzGFMConfig.from_pretrained(model_location)
    config.use_cache = False
    config.use_mamba_kernels = bool(use_fast_kernels and str(device).startswith("cuda"))

    torch_dtype = torch.float16 if dtype_name == "float16" else torch.float32
    if not str(device).startswith("cuda"):
        torch_dtype = torch.float32
    model = package.EnzGFM_Model.from_pretrained(
        model_location,
        config=config,
        torch_dtype=torch_dtype,
    )
    model.eval()
    model.to(device)
    installed_guards = install_mamba_projection_dtype_guards(model)
    if torch_dtype != torch.float32 and installed_guards == 0:
        raise RuntimeError(
            "No EnzGFM Mamba output projections were found for the mixed-precision "
            "dtype compatibility guard"
        )
    tokenizer = EsmTokenizer.from_pretrained(str(tokenizer_path))
    return model, tokenizer


def repository_revision(repo_path: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(repo_path).resolve()), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _match_linear_input_dtype(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...] | None:
    """Match an upstream FP32 Mamba scan output to its projection weights."""
    if not inputs or not isinstance(inputs[0], torch.Tensor):
        return None
    weight = getattr(module, "weight", None)
    if weight is None or inputs[0].dtype == weight.dtype:
        return None
    return (inputs[0].to(dtype=weight.dtype), *inputs[1:])


def install_mamba_projection_dtype_guards(model: torch.nn.Module) -> int:
    """Work around the released EnzGFM FP32-scan/FP16-projection mismatch."""
    installed = 0
    for module in model.modules():
        if module.__class__.__name__ != "EnzGFMMambaMixer":
            continue
        output_projection = getattr(module, "out_proj", None)
        if output_projection is None:
            continue
        output_projection.register_forward_pre_hook(_match_linear_input_dtype)
        installed += 1
    return installed


def set_metadata_args(args: argparse.Namespace) -> None:
    args.embedding_model_type = "enzgfm"
    args.embedding_backend = "official_deepbxm"


def write_enzgfm_attrs(h5_file: h5py.File, args: argparse.Namespace) -> None:
    h5_file.attrs["tokenizer_location"] = str(tokenizer_location(args).resolve())
    h5_file.attrs["enzgfm_source_repo"] = str(Path(args.enzgfm_repo).resolve())
    h5_file.attrs["enzgfm_source_revision"] = repository_revision(args.enzgfm_repo)
    h5_file.attrs["use_fast_kernels"] = not bool(args.disable_fast_kernels)
    h5_file.attrs["mamba_projection_dtype_guard"] = args.dtype != "float32"


def write_shard(args: argparse.Namespace) -> Path:
    set_metadata_args(args)
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError(f"rank must be in [0, {args.world_size}), got {args.rank}")

    shard_path = get_shard_path(args.output, args.tmp_dir, args.rank, args.world_size)
    partial_path = shard_path.with_suffix(f"{shard_path.suffix}.partial")
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    if shard_path.exists():
        if args.force:
            shard_path.unlink()
        elif args.resume:
            print(f"Rank {args.rank}: completed shard exists, skipping: {shard_path}", flush=True)
            return shard_path
        else:
            raise FileExistsError(f"Shard already exists: {shard_path} (use --force)")
    if partial_path.exists():
        partial_path.unlink()

    all_records = parse_fasta(args.fasta)
    records = [
        ProteinRecord(
            index=record.index,
            protein_id=record.protein_id,
            sequence=truncate_sequence(
                record.sequence,
                max_length=args.max_sequence_length,
                strategy=args.sequence_truncation,
            ),
        )
        for record in all_records
        if record.index % args.world_size == args.rank
    ]
    total_residues = sum(len(record.sequence) for record in records)
    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    if records:
        offsets[1:] = np.cumsum([len(record.sequence) for record in records], dtype=np.int64)

    model, tokenizer = load_model_and_tokenizer(
        args.model_location,
        args.enzgfm_repo,
        tokenizer_location(args),
        args.device,
        args.dtype,
        use_fast_kernels=not args.disable_fast_kernels,
    )
    residue_dim = residue_embedding_dim(model)
    out_dtype = numpy_dtype(args.dtype)
    text_dtype = h5py.string_dtype("utf-8")
    vector_chunks = (min(max(total_residues, 1), 256), residue_dim)

    with h5py.File(partial_path, "w") as h5_file:
        write_embedding_attrs(h5_file, args, residue_dim)
        write_enzgfm_attrs(h5_file, args)
        h5_file.attrs["rank"] = args.rank
        h5_file.attrs["world_size"] = args.world_size
        h5_file.create_dataset(
            "ids",
            data=np.array([record.protein_id for record in records], dtype=object),
            dtype=text_dtype,
        )
        h5_file.create_dataset(
            "indices",
            data=np.array([record.index for record in records], dtype=np.int64),
        )
        h5_file.create_dataset("offsets", data=offsets, dtype=np.int64)
        vectors = h5_file.create_dataset(
            "vectors",
            shape=(total_residues, residue_dim),
            dtype=out_dtype,
            chunks=vector_chunks,
            compression=hdf5_compression(args.compression),
        )

        processed = 0
        for batch in iter_batches(records, args.batch_size, args.max_tokens_per_batch):
            tokenized = tokenizer(
                [record.sequence for record in batch],
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].to(args.device)
            attention_mask = tokenized["attention_mask"].to(args.device)
            with torch.inference_mode():
                hidden = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state

            for row_idx, record in enumerate(batch):
                if not record.sequence:
                    continue
                local_idx = (record.index - args.rank) // args.world_size
                start, end = int(offsets[local_idx]), int(offsets[local_idx + 1])
                residue_tensor = extract_residue_tokens(
                    hidden[row_idx],
                    input_ids[row_idx],
                    attention_mask[row_idx],
                    tokenizer,
                    len(record.sequence),
                ).detach()
                if residue_tensor.shape[0] != end - start:
                    raise ValueError(
                        f"Residue token count mismatch for {record.protein_id}: "
                        f"got {residue_tensor.shape[0]}, expected {end - start}"
                    )
                storage_dtype = torch.float16 if args.dtype == "float16" else torch.float32
                vectors[start:end] = residue_tensor.cpu().to(storage_dtype).numpy()

            processed += len(batch)
            if processed == len(records) or processed % args.progress_every == 0:
                print(
                    f"Rank {args.rank}: processed {processed}/{len(records)} proteins",
                    flush=True,
                )
        h5_file.flush()

    os.replace(partial_path, shard_path)
    print(f"Rank {args.rank}: wrote shard {shard_path}", flush=True)
    return shard_path


def merge_enzgfm_shards(args: argparse.Namespace) -> Path:
    set_metadata_args(args)
    output_path = merge_shards(args)
    with h5py.File(output_path, "r+") as h5_file:
        write_enzgfm_attrs(h5_file, args)
    return output_path


def main() -> None:
    args = parse_args()
    set_metadata_args(args)
    if args.load_only:
        model, _tokenizer = load_model_and_tokenizer(
            args.model_location,
            args.enzgfm_repo,
            tokenizer_location(args),
            args.device,
            args.dtype,
            use_fast_kernels=not args.disable_fast_kernels,
        )
        print(
            f"Loaded {args.model_location} (residue_dim={residue_embedding_dim(model)})",
            flush=True,
        )
        return
    if args.merge_only:
        merge_enzgfm_shards(args)
        return
    write_shard(args)


if __name__ == "__main__":
    main()
