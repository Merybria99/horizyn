#!/usr/bin/env python
"""
Extract unpooled Biohub ESMC residue embeddings into a ragged HDF5 file.

The final HDF5 schema is:
    /ids      string protein identifiers, shape [N]
    /vectors  flattened residue embeddings, shape [sum_L, residue_dim]
    /offsets  integer offsets into vectors, shape [N + 1]

For multi-GPU extraction, launch one process per rank. Each rank writes a shard
with the same ragged schema plus /indices. A final merge pass restores FASTA
order and writes the output file expected by ResidueEmbedDataset.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

DEFAULT_MODEL_NAME = "Biohub/ESMC-6B"
DEFAULT_HF_CACHE_DIR = "/datastor2/deep-proteins/EnzymeDiscovery/hf_cache"
DEFAULT_RESIDUE_DIM = 2560
DEFAULT_MAX_SEQUENCE_LENGTH = 1022
DEFAULT_HIDDEN_LAYER = -1
AMINO_ACID_PATTERN = re.compile(r"[^ACDEFGHIKLMNPQRSTVWYX]")


@dataclass(frozen=True)
class ProteinRecord:
    index: int
    protein_id: str
    sequence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", default="data/sota/prots.fasta", help="Input protein FASTA")
    parser.add_argument(
        "--output",
        default="data/sota/prots_esmc_6B_residue.h5",
        help="Final merged ragged HDF5 output",
    )
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="Directory for rank shard HDF5 files. Defaults to <output_stem>_shards",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Biohub/Hugging Face ESMC model name",
    )
    parser.add_argument(
        "--backend",
        choices=("biohub", "transformers"),
        default="biohub",
        help=(
            "ESMC loading backend. 'biohub' uses the Biohub/esm local ESMC "
            "implementation; 'transformers' uses Hugging Face AutoModel."
        ),
    )
    parser.add_argument(
        "--hidden-layer",
        type=int,
        default=DEFAULT_HIDDEN_LAYER,
        help=(
            "Hidden-state index to store. ESMC-6B has layer 60 as its final layer; "
            "-1 stores the final hidden state and works across ESMC variants."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face model/tokenizer loading.",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_HF_CACHE_DIR,
        help="Hugging Face cache root for tokenizer/model weights",
    )
    parser.add_argument("--rank", type=int, default=0, help="Shard rank for this worker")
    parser.add_argument("--world-size", type=int, default=1, help="Number of shard workers")
    parser.add_argument("--device", default="cuda", help="Torch device for extraction")
    parser.add_argument("--batch-size", type=int, default=2, help="Maximum proteins per batch")
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=4096,
        help="Soft residue-token budget per batch; very long proteins run as singleton batches",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
        help="Deterministic per-protein sequence truncation before ESMC",
    )
    parser.add_argument(
        "--sequence-truncation",
        choices=("ends_center",),
        default="ends_center",
        help="Sequence truncation strategy used with --max-sequence-length",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Stored embedding dtype",
    )
    parser.add_argument(
        "--compression",
        choices=("none", "gzip", "lzf"),
        default="none",
        help="HDF5 compression for vectors",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N proteins per shard",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download tokenizer/model into cache and exit",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge existing shards into the final output and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing shard/final output",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip extraction if this rank's completed shard already exists",
    )
    parser.add_argument(
        "--cleanup-shards",
        action="store_true",
        help="Delete shard files after a successful merge",
    )
    return parser.parse_args()


def normalize_sequence(sequence: str) -> str:
    sequence = re.sub(r"\s+", "", sequence.upper())
    sequence = re.sub(r"[UZOB]", "X", sequence)
    return AMINO_ACID_PATTERN.sub("X", sequence)


def truncate_sequence(sequence: str, max_length: int | None, strategy: str = "ends_center") -> str:
    if max_length is None or len(sequence) <= max_length:
        return sequence
    if max_length <= 0:
        raise ValueError("max_sequence_length must be positive when provided")
    if strategy != "ends_center":
        raise ValueError(f"Unsupported sequence truncation strategy: {strategy}")

    first_count = max_length // 4
    last_count = max_length // 4
    middle_count = max_length - first_count - last_count
    middle_start = max((len(sequence) - middle_count) // 2, first_count)
    middle_end = min(middle_start + middle_count, len(sequence) - last_count)
    middle_start = max(middle_end - middle_count, first_count)

    first = sequence[:first_count]
    middle = sequence[middle_start:middle_end]
    last = sequence[len(sequence) - last_count :] if last_count else ""
    return first + middle + last


def parse_fasta(path: str | Path) -> list[ProteinRecord]:
    path = Path(path)
    records: list[ProteinRecord] = []
    protein_id: str | None = None
    chunks: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if protein_id is not None:
                    records.append(
                        ProteinRecord(
                            index=len(records),
                            protein_id=protein_id,
                            sequence=normalize_sequence("".join(chunks)),
                        )
                    )
                protein_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)

    if protein_id is not None:
        records.append(
            ProteinRecord(
                index=len(records),
                protein_id=protein_id,
                sequence=normalize_sequence("".join(chunks)),
            )
        )

    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def get_tmp_dir(output: str | Path, tmp_dir: str | None) -> Path:
    if tmp_dir is not None:
        return Path(tmp_dir)
    output_path = Path(output)
    return output_path.with_name(f"{output_path.stem}_shards")


def get_shard_path(output: str | Path, tmp_dir: str | None, rank: int, world_size: int) -> Path:
    output_path = Path(output)
    shard_dir = get_tmp_dir(output_path, tmp_dir)
    return shard_dir / f"{output_path.stem}.shard{rank:02d}-of-{world_size:02d}.h5"


def numpy_dtype(dtype_name: str) -> np.dtype:
    if dtype_name == "float16":
        return np.dtype("float16")
    if dtype_name == "float32":
        return np.dtype("float32")
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def hdf5_compression(compression: str) -> str | None:
    return None if compression == "none" else compression


def iter_batches(
    records: list[ProteinRecord],
    batch_size: int,
    max_tokens_per_batch: int,
) -> list[list[ProteinRecord]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_tokens_per_batch <= 0:
        raise ValueError("max_tokens_per_batch must be positive")

    batches: list[list[ProteinRecord]] = []
    current: list[ProteinRecord] = []
    current_tokens = 0

    for record in records:
        token_count = max(len(record.sequence), 1)
        would_exceed_batch = len(current) >= batch_size
        would_exceed_tokens = current and current_tokens + token_count > max_tokens_per_batch
        if would_exceed_batch or would_exceed_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(record)
        current_tokens += token_count

    if current:
        batches.append(current)
    return batches


def configure_hf_cache(cache_dir: str) -> tuple[Path, Path]:
    cache_path = Path(cache_dir).expanduser()
    hub_cache_path = cache_path / "hub"
    hub_cache_path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_path)
    os.environ["HF_HUB_CACHE"] = str(hub_cache_path)
    return cache_path, hub_cache_path


def normalize_biohub_model_name(model_name: str) -> str:
    normalized = model_name.strip()
    lower = normalized.lower()
    if lower in {"biohub/esmc-6b", "biohub/esmc-6b-2024-12", "biohub/esmc_6b"}:
        return "esmc_6b"
    if lower in {"biohub/esmc-600m", "biohub/esmc-600m-2024-12", "biohub/esmc_600m"}:
        return "esmc_600m"
    if lower in {"biohub/esmc-300m", "biohub/esmc-300m-2024-12", "biohub/esmc_300m"}:
        return "esmc_300m"
    if normalized in {"esmc_6b", "esmc_600m", "esmc_300m"}:
        return normalized
    return normalized


def biohub_snapshot_repo(model_name: str) -> str:
    normalized = normalize_biohub_model_name(model_name)
    if normalized == "esmc_6b":
        return "biohub/esmc-6b-2024-12"
    if normalized == "esmc_600m":
        return "biohub/esmc-600m-2024-12"
    if normalized == "esmc_300m":
        return "biohub/esmc-300m-2024-12"
    raise ValueError(f"Unsupported Biohub ESMC model name: {model_name}")


def download_model_artifacts(model_name: str, cache_dir: str, *, backend: str = "biohub") -> Path:
    _cache_path, hub_cache_path = configure_hf_cache(cache_dir)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "ESMC download requires huggingface_hub, which is installed with transformers. "
            "Install it in the project env with: "
            "uv pip install --python /datastor2/deep-proteins/EnzymeDiscovery/env/bin/python "
            "transformers accelerate"
        ) from exc

    repo_id = biohub_snapshot_repo(model_name) if backend == "biohub" else model_name
    return Path(snapshot_download(repo_id=repo_id, cache_dir=str(hub_cache_path)))


def load_safetensors_assign(
    module: torch.nn.Module,
    snapshot_path: str | Path,
    device: str | torch.device,
) -> tuple[list[str], list[str]]:
    """Load sharded safetensors into a meta-initialized module with assign=True."""
    try:
        from safetensors.torch import load_file as load_safetensors_file
    except ImportError as exc:
        raise ImportError("Biohub ESMC safetensors loading requires safetensors") from exc

    snapshot_path = Path(snapshot_path)
    index_path = snapshot_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = sorted(path.name for path in snapshot_path.glob("*.safetensors"))
    if not shard_names:
        raise FileNotFoundError(f"No safetensors shards found in {snapshot_path}")

    expected_keys = set(module.state_dict().keys())
    loaded_keys: set[str] = set()
    unexpected_keys: set[str] = set()
    for shard_name in shard_names:
        shard_path = snapshot_path / shard_name
        raw_state_dict = load_safetensors_file(str(shard_path), device=str(device))
        state_dict = {}
        for key, tensor in raw_state_dict.items():
            mapped_key = key
            if key.startswith("lm_head."):
                candidate_key = f"esmc.sequence_head.{key[len('lm_head.'):]}"
                if candidate_key in expected_keys:
                    mapped_key = candidate_key
            state_dict[mapped_key] = tensor
        loaded_keys.update(state_dict.keys())
        unexpected_keys.update(key for key in state_dict if key not in expected_keys)
        module.load_state_dict(state_dict, strict=False, assign=True)
        del raw_state_dict
        del state_dict
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    missing_keys = sorted(expected_keys - loaded_keys)
    return missing_keys, sorted(unexpected_keys)


def load_biohub_esmc_model(model_name: str, device: str, cache_dir: str):
    _cache_path, hub_cache_path = configure_hf_cache(cache_dir)
    try:
        from accelerate import init_empty_weights
        from esm.models.esmc import ESMC
        from esm.tokenization import get_esmc_model_tokenizers
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Biohub ESMC extraction requires the Biohub/esm package on PYTHONPATH. "
            "Clone https://github.com/Biohub/esm and prepend it to PYTHONPATH."
        ) from exc

    normalized_name = normalize_biohub_model_name(model_name)
    model_specs = {
        "esmc_300m": {"d_model": 960, "n_heads": 15, "n_layers": 30},
        "esmc_600m": {"d_model": 1152, "n_heads": 18, "n_layers": 36},
        "esmc_6b": {"d_model": 2560, "n_heads": 40, "n_layers": 80},
    }
    if normalized_name not in model_specs:
        raise ValueError(f"Unsupported Biohub ESMC model name: {model_name}")
    spec = model_specs[normalized_name]

    # Biohub's HF checkpoints store weights under an "esmc." prefix. Build the
    # native local model on meta tensors, wrap it under that prefix, then stream
    # safetensors directly onto the requested device.
    with init_empty_weights():
        model = ESMC(
            d_model=spec["d_model"],
            n_heads=spec["n_heads"],
            n_layers=spec["n_layers"],
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=True,
        ).eval()
    wrapper = torch.nn.Module()
    wrapper.esmc = model
    snapshot_path = snapshot_download(
        repo_id=biohub_snapshot_repo(normalized_name),
        cache_dir=str(hub_cache_path),
    )
    missing_keys, unexpected_keys = load_safetensors_assign(
        wrapper,
        snapshot_path,
        device=torch.device(device),
    )
    if missing_keys or unexpected_keys:
        print(
            "Biohub ESMC load warnings: "
            f"missing_keys={len(missing_keys)} unexpected_keys={len(unexpected_keys)}",
            flush=True,
        )
    model = wrapper.esmc
    meta_parameters = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if meta_parameters:
        preview = ", ".join(meta_parameters[:5])
        raise RuntimeError(
            "Biohub ESMC checkpoint load left parameters on meta tensors: "
            f"{preview}{'...' if len(meta_parameters) > 5 else ''}"
        )
    model = model.to(torch.device(device))
    if str(device).startswith("cuda"):
        model = model.to(torch.bfloat16)
    model.eval()
    model._horizyn_esmc_backend = "biohub"  # type: ignore[attr-defined]
    return model, model.tokenizer


def load_transformers_model_and_tokenizer(
    model_name: str,
    device: str,
    dtype_name: str,
    cache_dir: str,
    *,
    trust_remote_code: bool = False,
):
    _cache_path, hub_cache_path = configure_hf_cache(cache_dir)

    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "ESMC extraction requires transformers. "
            "Install it in the project env with: "
            "uv pip install --python /datastor2/deep-proteins/EnzymeDiscovery/env/bin/python "
            "transformers accelerate"
        ) from exc

    torch_dtype = torch.float16 if dtype_name == "float16" else torch.float32
    if not str(device).startswith("cuda"):
        torch_dtype = torch.float32

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(hub_cache_path),
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            cache_dir=str(hub_cache_path),
            trust_remote_code=trust_remote_code,
        )
    except ValueError as exc:
        if "model type `esmc`" in str(exc) or "model_type `esmc`" in str(exc):
            raise ImportError(
                "Biohub ESMC local extraction requires transformers>=4.57.6. "
                "Refresh the project environment, for example: "
                "uv pip install --python /datastor2/deep-proteins/EnzymeDiscovery/env/bin/python "
                "'transformers>=4.57.6' accelerate"
            ) from exc
        raise
    model.eval()
    model.to(device)
    model._horizyn_esmc_backend = "transformers"  # type: ignore[attr-defined]
    return model, tokenizer


def load_model_and_tokenizer(
    model_name: str,
    device: str,
    dtype_name: str,
    cache_dir: str,
    *,
    backend: str = "biohub",
    trust_remote_code: bool = False,
):
    if backend == "biohub":
        return load_biohub_esmc_model(model_name, device, cache_dir)
    if backend == "transformers":
        return load_transformers_model_and_tokenizer(
            model_name,
            device,
            dtype_name,
            cache_dir,
            trust_remote_code=trust_remote_code,
        )
    raise ValueError(f"Unsupported ESMC backend: {backend}")


def residue_embedding_dim(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    for attr in ("hidden_size", "d_model", "embed_dim"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    embed = getattr(model, "embed", None)
    embedding_dim = getattr(embed, "embedding_dim", None)
    if embedding_dim is not None:
        return int(embedding_dim)
    raise ValueError("Could not infer ESMC residue embedding dimension from model config")


def select_hidden_tensor(model_output: Any, hidden_layer: int) -> torch.Tensor:
    """Select the residue embedding tensor from an ESMC model output."""
    embeddings = getattr(model_output, "embeddings", None)
    if embeddings is not None and hidden_layer == -1:
        return embeddings

    hidden_states = getattr(model_output, "hidden_states", None)
    if hidden_states is None and isinstance(model_output, dict):
        hidden_states = model_output.get("hidden_states")
    if hidden_states is not None:
        if isinstance(hidden_states, torch.Tensor):
            if hidden_states.ndim >= 4:
                return hidden_states[hidden_layer]
            if hidden_layer in {-1, 0}:
                return hidden_states
            raise ValueError(
                f"Model returned a hidden_states tensor with shape {tuple(hidden_states.shape)}; "
                f"cannot select hidden_layer={hidden_layer}"
            )
        if not hidden_states:
            raise ValueError("Model returned an empty hidden_states sequence")
        try:
            return hidden_states[hidden_layer]
        except IndexError as exc:
            raise IndexError(
                f"Requested hidden_layer={hidden_layer}, but model returned "
                f"{len(hidden_states)} hidden-state tensors"
            ) from exc

    last_hidden = getattr(model_output, "last_hidden_state", None)
    if last_hidden is None and isinstance(model_output, dict):
        last_hidden = model_output.get("last_hidden_state")
    if last_hidden is not None:
        if hidden_layer not in {-1, 0}:
            raise ValueError(
                "Model output did not include hidden_states; only final-layer embeddings "
                f"are available, but hidden_layer={hidden_layer} was requested"
            )
        return last_hidden

    raise ValueError(
        "Could not find hidden_states or last_hidden_state in ESMC output. "
        "Try loading with a model/tokenizer version that exposes transformer hidden states."
    )


def extract_residue_tokens(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
    sequence_length: int,
) -> torch.Tensor:
    """Return only residue-token hidden states, excluding ESM special/pad tokens."""
    if sequence_length == 0:
        return hidden.new_empty((0, hidden.shape[-1]))

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    valid_positions = attention_mask.to(torch.bool).nonzero(as_tuple=False).flatten()
    residue_positions = [
        int(pos.item())
        for pos in valid_positions
        if int(input_ids[int(pos.item())].item()) not in special_ids
    ]
    if len(residue_positions) < sequence_length:
        raise ValueError(
            f"Tokenizer produced {len(residue_positions)} residue tokens for a sequence "
            f"of length {sequence_length}"
        )
    if len(residue_positions) > sequence_length:
        residue_positions = residue_positions[:sequence_length]
    positions = torch.as_tensor(residue_positions, dtype=torch.long, device=hidden.device)
    return hidden.index_select(0, positions)


def write_embedding_attrs(h5_file: h5py.File, args: argparse.Namespace, residue_dim: int) -> None:
    h5_file.attrs["model_name"] = args.model_name
    h5_file.attrs["model_cache_dir"] = str(Path(args.cache_dir).expanduser())
    h5_file.attrs["embedding_model_type"] = "esmc"
    h5_file.attrs["embedding_backend"] = args.backend
    h5_file.attrs["hidden_layer"] = int(args.hidden_layer)
    h5_file.attrs["residue_dim"] = residue_dim
    h5_file.attrs["source_fasta"] = str(Path(args.fasta).resolve())
    h5_file.attrs["max_sequence_length"] = (
        -1 if args.max_sequence_length is None else args.max_sequence_length
    )
    h5_file.attrs["sequence_truncation"] = args.sequence_truncation
    h5_file.attrs["created_unix_time"] = time.time()


def write_shard(args: argparse.Namespace) -> Path:
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
            raise FileExistsError(f"Shard already exists: {shard_path} (use --force to overwrite)")
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

    print(
        f"Rank {args.rank}/{args.world_size}: {len(records)} proteins, "
        f"{total_residues} residues",
        flush=True,
    )

    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        args.device,
        args.dtype,
        args.cache_dir,
        backend=args.backend,
        trust_remote_code=bool(args.trust_remote_code),
    )
    residue_dim = residue_embedding_dim(model)
    out_dtype = numpy_dtype(args.dtype)
    text_dtype = h5py.string_dtype("utf-8")
    compression = hdf5_compression(args.compression)
    vector_chunks = (min(max(total_residues, 1), 256), residue_dim)

    with h5py.File(partial_path, "w") as h5_file:
        write_embedding_attrs(h5_file, args, residue_dim)
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
            dtype=np.int64,
        )
        h5_file.create_dataset("offsets", data=offsets, dtype=np.int64)
        vectors = h5_file.create_dataset(
            "vectors",
            shape=(total_residues, residue_dim),
            dtype=out_dtype,
            chunks=vector_chunks,
            compression=compression,
        )

        batches = iter_batches(records, args.batch_size, args.max_tokens_per_batch)
        processed = 0
        for batch in batches:
            batch_sequences = [record.sequence for record in batch]
            backend = getattr(model, "_horizyn_esmc_backend", args.backend)
            if backend == "biohub":
                input_ids = model._tokenize(batch_sequences)  # type: ignore[attr-defined]
                pad_id = tokenizer.pad_token_id
                attention_mask = (input_ids != pad_id).to(torch.long)
            else:
                tokenized = tokenizer(
                    batch_sequences,
                    add_special_tokens=True,
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = tokenized["input_ids"].to(args.device)
                attention_mask = tokenized["attention_mask"].to(args.device)

            with torch.inference_mode():
                if backend == "biohub":
                    output = model(sequence_tokens=input_ids)
                else:
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                hidden = select_hidden_tensor(output, int(args.hidden_layer))

            for row_idx, record in enumerate(batch):
                if len(record.sequence) == 0:
                    continue
                shard_idx = (record.index - args.rank) // args.world_size
                start = int(offsets[shard_idx])
                end = int(offsets[shard_idx + 1])
                residue_tensor = extract_residue_tokens(
                    hidden=hidden[row_idx],
                    input_ids=input_ids[row_idx],
                    attention_mask=attention_mask[row_idx],
                    tokenizer=tokenizer,
                    sequence_length=len(record.sequence),
                ).detach()
                if residue_tensor.shape[0] != end - start:
                    raise ValueError(
                        f"Residue token count mismatch for {record.protein_id}: "
                        f"got {residue_tensor.shape[0]}, expected {end - start}"
                    )
                if args.dtype == "float16":
                    residue_array = residue_tensor.cpu().to(torch.float16).numpy()
                else:
                    residue_array = residue_tensor.cpu().to(torch.float32).numpy()
                vectors[start:end] = residue_array

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


def read_ids(dataset) -> list[str]:
    values = dataset[:]
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def merge_shards(args: argparse.Namespace) -> Path:
    output_path = Path(args.output)
    partial_path = output_path.with_suffix(f"{output_path.suffix}.partial")
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path} (use --force to overwrite)")
    if partial_path.exists():
        partial_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shard_paths = [
        get_shard_path(args.output, args.tmp_dir, rank, args.world_size)
        for rank in range(args.world_size)
    ]
    missing = [str(path) for path in shard_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shard files: {missing}")

    shard_files = [h5py.File(path, "r") for path in shard_paths]
    try:
        entries: list[tuple[int, int, int, str, int]] = []
        residue_dim: int | None = None
        out_dtype = None
        for rank, h5_file in enumerate(shard_files):
            for dataset_name in ("ids", "indices", "vectors", "offsets"):
                if dataset_name not in h5_file:
                    raise KeyError(f"{shard_paths[rank]} missing dataset {dataset_name}")
            if residue_dim is None:
                residue_dim = int(h5_file["vectors"].shape[1])
                out_dtype = h5_file["vectors"].dtype
            elif int(h5_file["vectors"].shape[1]) != residue_dim:
                raise ValueError("Shard residue dimensions do not match")

            ids = read_ids(h5_file["ids"])
            indices = h5_file["indices"][:]
            offsets = h5_file["offsets"][:]
            for local_idx, protein_id in enumerate(ids):
                length = int(offsets[local_idx + 1] - offsets[local_idx])
                entries.append((int(indices[local_idx]), rank, local_idx, protein_id, length))

        if residue_dim is None or out_dtype is None:
            raise ValueError("No shard entries found")

        entries.sort(key=lambda item: item[0])
        expected_indices = list(range(len(entries)))
        actual_indices = [entry[0] for entry in entries]
        if actual_indices != expected_indices:
            raise ValueError("Shard indices do not form a contiguous FASTA-order range")

        final_offsets = np.zeros(len(entries) + 1, dtype=np.int64)
        if entries:
            final_offsets[1:] = np.cumsum([entry[4] for entry in entries], dtype=np.int64)
        total_residues = int(final_offsets[-1])
        compression = hdf5_compression(args.compression)
        vector_chunks = (min(max(total_residues, 1), 256), residue_dim)
        text_dtype = h5py.string_dtype("utf-8")

        with h5py.File(partial_path, "w") as output_file:
            write_embedding_attrs(output_file, args, residue_dim)
            output_file.attrs["world_size"] = args.world_size
            output_file.create_dataset(
                "ids",
                data=np.array([entry[3] for entry in entries], dtype=object),
                dtype=text_dtype,
            )
            output_file.create_dataset("offsets", data=final_offsets, dtype=np.int64)
            output_vectors = output_file.create_dataset(
                "vectors",
                shape=(total_residues, residue_dim),
                dtype=out_dtype,
                chunks=vector_chunks,
                compression=compression,
            )

            write_start = 0
            for output_idx, (_global_idx, rank, local_idx, _protein_id, length) in enumerate(
                entries
            ):
                if length:
                    input_offsets = shard_files[rank]["offsets"]
                    input_start = int(input_offsets[local_idx])
                    input_end = int(input_offsets[local_idx + 1])
                    output_vectors[write_start : write_start + length] = shard_files[rank][
                        "vectors"
                    ][input_start:input_end]
                write_start += length
                if (output_idx + 1) % args.progress_every == 0 or output_idx + 1 == len(entries):
                    print(
                        f"Merged {output_idx + 1}/{len(entries)} proteins into {output_path}",
                        flush=True,
                    )

            output_file.flush()

        os.replace(partial_path, output_path)
    finally:
        for h5_file in shard_files:
            h5_file.close()

    if args.cleanup_shards:
        for path in shard_paths:
            path.unlink(missing_ok=True)

    print(f"Wrote merged ESMC residue HDF5: {output_path}", flush=True)
    return output_path


def main() -> None:
    args = parse_args()

    if args.download_only:
        snapshot_path = download_model_artifacts(
            args.model_name,
            args.cache_dir,
            backend=args.backend,
        )
        print(
            f"Downloaded {args.model_name} to {snapshot_path} "
            f"(HF_HOME={Path(args.cache_dir).expanduser()})",
            flush=True,
        )
        return

    if args.merge_only:
        merge_shards(args)
        return

    write_shard(args)


if __name__ == "__main__":
    main()
