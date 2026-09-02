#!/usr/bin/env python
"""
Extract unpooled ESM2 residue embeddings into a ragged HDF5 file.

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
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


DEFAULT_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DEFAULT_RESIDUE_DIM = 1280
DEFAULT_MAX_SEQUENCE_LENGTH = 1022
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
        default="data/sota/prots_esm2_650m_residue.h5",
        help="Final merged ragged HDF5 output",
    )
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="Directory for rank shard HDF5 files. Defaults to <output_stem>_shards",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Hugging Face model name")
    parser.add_argument("--rank", type=int, default=0, help="Shard rank for this worker")
    parser.add_argument("--world-size", type=int, default=1, help="Number of shard workers")
    parser.add_argument("--device", default="cuda", help="Torch device for extraction")
    parser.add_argument("--batch-size", type=int, default=8, help="Maximum proteins per batch")
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
        help="Deterministic per-protein sequence truncation before ESM2",
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


def load_model_and_tokenizer(model_name: str, device: str, dtype_name: str):
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "ESM2 extraction requires transformers. "
            "Install it in the project env with: "
            "uv pip install --python /datastor2/deep-proteins/EnzymeDiscovery/env/bin/python "
            "transformers accelerate"
        ) from exc

    torch_dtype = torch.float16 if dtype_name == "float16" else torch.float32
    if not str(device).startswith("cuda"):
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.eval()
    model.to(device)
    return model, tokenizer


def residue_embedding_dim(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    for attr in ("hidden_size", "d_model", "embed_dim"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer ESM2 residue embedding dimension from model config")


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
    model_name = getattr(args, "model_name", getattr(args, "model_location", ""))
    h5_file.attrs["model_name"] = model_name
    h5_file.attrs["embedding_model_type"] = getattr(args, "embedding_model_type", "esm2")
    h5_file.attrs["embedding_backend"] = getattr(
        args,
        "embedding_backend",
        "huggingface_auto_model",
    )
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

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device, args.dtype)
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
            tokenized = tokenizer(
                batch_sequences,
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].to(args.device)
            attention_mask = tokenized["attention_mask"].to(args.device)

            with torch.inference_mode():
                hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

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

    embedding_type = getattr(args, "embedding_model_type", "esm2")
    print(f"Wrote merged {embedding_type} residue HDF5: {output_path}", flush=True)
    return output_path


def main() -> None:
    args = parse_args()

    if args.download_only:
        model, _tokenizer = load_model_and_tokenizer(args.model_name, args.device, args.dtype)
        print(
            f"Downloaded and loaded {args.model_name} "
            f"(residue_dim={residue_embedding_dim(model)})",
            flush=True,
        )
        return

    if args.merge_only:
        merge_shards(args)
        return

    write_shard(args)


if __name__ == "__main__":
    main()
