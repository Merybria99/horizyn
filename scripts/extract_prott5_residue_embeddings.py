#!/usr/bin/env python
"""
Extract unpooled ProtT5 residue embeddings into a ragged HDF5 file.

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
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

DEFAULT_MODEL_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"
AMINO_ACID_PATTERN = re.compile(r"[^ACDEFGHIKLMNPQRSTVWYX]")
PARTIAL_FORMAT_VERSION = 1
PROCESSED_COUNT_ATTR = "processed_count"


class TerminationRequested(BaseException):
    """Raised after a termination signal so HDF5 context managers can close."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


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
        default="data/sota/prots_t5_residue.h5",
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
        help="Soft token budget per batch; very long proteins run as singleton batches",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=None,
        help="Optional deterministic per-protein sequence truncation before ProtT5",
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
        "--checkpoint-every",
        type=int,
        default=100,
        help="Flush data and commit resumable progress every N proteins per shard",
    )
    parser.add_argument(
        "--merge-proteins-per-chunk",
        type=int,
        default=256,
        help="Proteins assembled per blockwise HDF5 merge write",
    )
    parser.add_argument(
        "--merge-order",
        choices=("fasta", "shard"),
        default="fasta",
        help="Final protein storage order",
    )
    parser.add_argument(
        "--merge-storage",
        choices=("copy", "virtual"),
        default="copy",
        help="Copy vectors or create an HDF5 virtual dataset over immutable shards",
    )
    parser.add_argument(
        "--merged-ids-output",
        default=None,
        help="Optional newline-delimited IDs in final HDF5 storage order",
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
        help="Resume a checkpointed partial shard or skip a completed shard",
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


def format_for_prott5(sequence: str) -> str:
    return " ".join(sequence)


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
        from transformers import T5EncoderModel, T5Tokenizer
    except ImportError as exc:
        raise ImportError(
            "ProtT5 extraction requires transformers and sentencepiece. "
            "Install them in the project env with: "
            "uv pip install --python /datastor2/deep-proteins/EnzymeDiscovery/env/bin/python "
            "transformers sentencepiece accelerate"
        ) from exc

    torch_dtype = torch.float16 if dtype_name == "float16" else torch.float32
    if not str(device).startswith("cuda"):
        torch_dtype = torch.float32

    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.eval()
    model.to(device)
    return model, tokenizer


def _decode_text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _validate_numeric_dataset(
    dataset: h5py.Dataset,
    expected: np.ndarray,
    *,
    name: str,
    chunk_size: int = 100_000,
) -> None:
    if dataset.shape != expected.shape:
        raise ValueError(
            f"Partial shard /{name} shape mismatch: {dataset.shape} != {expected.shape}"
        )
    for start in range(0, len(expected), chunk_size):
        end = min(start + chunk_size, len(expected))
        if not np.array_equal(dataset[start:end], expected[start:end]):
            raise ValueError(f"Partial shard /{name} differs at rows {start}:{end}")


def _validate_id_dataset(
    dataset: h5py.Dataset,
    records: list[ProteinRecord],
    *,
    chunk_size: int = 100_000,
) -> None:
    if dataset.shape != (len(records),):
        raise ValueError(f"Partial shard /ids shape mismatch: {dataset.shape} != {(len(records),)}")
    for start in range(0, len(records), chunk_size):
        end = min(start + chunk_size, len(records))
        actual = [_decode_text(value) for value in dataset[start:end]]
        expected = [record.protein_id for record in records[start:end]]
        if actual != expected:
            raise ValueError(f"Partial shard /ids differs at rows {start}:{end}")


def _checkpoint_is_batch_boundary(
    batches: list[list[ProteinRecord]],
    processed_count: int,
) -> bool:
    if processed_count == 0:
        return True
    processed = 0
    for batch in batches:
        processed += len(batch)
        if processed == processed_count:
            return True
        if processed > processed_count:
            return False
    return False


def validate_partial_shard(
    h5_file: h5py.File,
    args: argparse.Namespace,
    records: list[ProteinRecord],
    offsets: np.ndarray,
    batches: list[list[ProteinRecord]],
    *,
    expected_residue_dim: int | None = None,
    checkpoint_override: int | None = None,
) -> int:
    """Validate provenance and return the last committed protein count."""

    required_datasets = {"ids", "indices", "offsets", "vectors"}
    missing = sorted(required_datasets.difference(h5_file.keys()))
    if missing:
        raise ValueError(f"Partial shard is missing datasets: {', '.join(missing)}")

    expected_attrs: dict[str, object] = {
        "model_name": args.model_name,
        "source_fasta": str(Path(args.fasta).resolve()),
        "max_sequence_length": (
            -1 if args.max_sequence_length is None else args.max_sequence_length
        ),
        "sequence_truncation": args.sequence_truncation,
        "rank": args.rank,
        "world_size": args.world_size,
    }
    for name, expected in expected_attrs.items():
        if name not in h5_file.attrs:
            raise ValueError(f"Partial shard is missing attribute {name!r}")
        actual = h5_file.attrs[name]
        if isinstance(expected, str):
            actual = _decode_text(actual)
        else:
            actual = int(actual)
        if actual != expected:
            raise ValueError(
                f"Partial shard attribute {name!r} mismatch: {actual!r} != {expected!r}"
            )

    vectors = h5_file["vectors"]
    residue_dim = int(h5_file.attrs.get("residue_dim", -1))
    if expected_residue_dim is not None and residue_dim != expected_residue_dim:
        raise ValueError(
            f"Partial shard residue dimension mismatch: {residue_dim} != {expected_residue_dim}"
        )
    expected_vector_shape = (int(offsets[-1]), residue_dim)
    if vectors.shape != expected_vector_shape:
        raise ValueError(
            f"Partial shard /vectors shape mismatch: {vectors.shape} != {expected_vector_shape}"
        )
    if vectors.dtype != numpy_dtype(args.dtype):
        raise ValueError(
            f"Partial shard /vectors dtype mismatch: {vectors.dtype} != {numpy_dtype(args.dtype)}"
        )
    if vectors.compression != hdf5_compression(args.compression):
        raise ValueError(
            "Partial shard /vectors compression mismatch: "
            f"{vectors.compression!r} != {hdf5_compression(args.compression)!r}"
        )

    expected_indices = np.asarray([record.index for record in records], dtype=np.int64)
    _validate_numeric_dataset(h5_file["indices"], expected_indices, name="indices")
    _validate_numeric_dataset(h5_file["offsets"], offsets, name="offsets")
    _validate_id_dataset(h5_file["ids"], records)

    if PROCESSED_COUNT_ATTR in h5_file.attrs:
        processed_count = int(h5_file.attrs[PROCESSED_COUNT_ATTR])
        if checkpoint_override is not None and processed_count != checkpoint_override:
            raise ValueError(
                "Partial-shard checkpoint differs from the supplied recovery checkpoint: "
                f"{processed_count} != {checkpoint_override}"
            )
    elif checkpoint_override is not None:
        processed_count = checkpoint_override
    else:
        raise ValueError(
            "Partial shard has no committed progress checkpoint. Recover and validate the "
            f"legacy file before setting {PROCESSED_COUNT_ATTR!r}."
        )
    if not 0 <= processed_count <= len(records):
        raise ValueError(
            f"Invalid partial-shard checkpoint {processed_count}; expected 0..{len(records)}"
        )
    if not _checkpoint_is_batch_boundary(batches, processed_count):
        raise ValueError(
            f"Partial-shard checkpoint {processed_count} is not a deterministic batch boundary"
        )
    return processed_count


def commit_checkpoint(h5_file: h5py.File, processed_count: int) -> None:
    """Commit vectors first, then the count that makes those vectors resumable."""

    h5_file.flush()
    h5_file.attrs[PROCESSED_COUNT_ATTR] = processed_count
    h5_file.attrs["checkpoint_unix_time"] = time.time()
    h5_file.flush()


def _write_pending_batches(
    h5_file: h5py.File,
    args: argparse.Namespace,
    records: list[ProteinRecord],
    offsets: np.ndarray,
    batches: list[list[ProteinRecord]],
    model: object,
    tokenizer: object,
    *,
    processed_count: int,
) -> None:
    vectors = h5_file["vectors"]
    processed = 0
    next_progress = ((processed_count // args.progress_every) + 1) * args.progress_every
    next_checkpoint = ((processed_count // args.checkpoint_every) + 1) * args.checkpoint_every

    for batch in batches:
        batch_end = processed + len(batch)
        if batch_end <= processed_count:
            processed = batch_end
            continue
        if processed < processed_count:
            raise ValueError(
                f"Checkpoint {processed_count} falls inside batch {processed}:{batch_end}"
            )

        batch_text = [format_for_prott5(record.sequence) for record in batch]
        tokenized = tokenizer(
            batch_text,
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
            residue_tensor = hidden[row_idx, : len(record.sequence), :].detach().cpu()
            if args.dtype == "float16":
                residue_array = residue_tensor.to(torch.float16).numpy()
            else:
                residue_array = residue_tensor.to(torch.float32).numpy()
            vectors[start:end] = residue_array

        processed = batch_end
        checkpoint_due = processed == len(records) or processed >= next_checkpoint
        if checkpoint_due:
            commit_checkpoint(h5_file, processed)
            while next_checkpoint <= processed:
                next_checkpoint += args.checkpoint_every

        if processed == len(records) or processed >= next_progress:
            if not checkpoint_due:
                commit_checkpoint(h5_file, processed)
                while next_checkpoint <= processed:
                    next_checkpoint += args.checkpoint_every
            print(
                f"Rank {args.rank}: processed {processed}/{len(records)} proteins",
                flush=True,
            )
            while next_progress <= processed:
                next_progress += args.progress_every

    if int(h5_file.attrs[PROCESSED_COUNT_ATTR]) != len(records):
        commit_checkpoint(h5_file, len(records))


def write_shard(args: argparse.Namespace) -> Path:
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError(f"rank must be in [0, {args.world_size}), got {args.rank}")

    shard_path = get_shard_path(args.output, args.tmp_dir, args.rank, args.world_size)
    partial_path = shard_path.with_suffix(f"{shard_path.suffix}.partial")
    shard_path.parent.mkdir(parents=True, exist_ok=True)

    if shard_path.exists():
        if args.resume:
            print(f"Rank {args.rank}: completed shard exists, skipping: {shard_path}", flush=True)
            return shard_path
        if not args.force:
            raise FileExistsError(f"Shard already exists: {shard_path} (use --force to overwrite)")
        shard_path.unlink()
    if partial_path.exists():
        if args.force:
            partial_path.unlink()
        elif not args.resume:
            raise FileExistsError(
                f"Partial shard already exists: {partial_path} (use --resume or --force)"
            )

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
    batches = iter_batches(records, args.batch_size, args.max_tokens_per_batch)

    print(
        f"Rank {args.rank}/{args.world_size}: {len(records)} proteins, "
        f"{total_residues} residues",
        flush=True,
    )

    resume_count = 0
    if partial_path.exists():
        try:
            with h5py.File(partial_path, "r") as h5_file:
                resume_count = validate_partial_shard(
                    h5_file,
                    args,
                    records,
                    offsets,
                    batches,
                )
        except OSError as exc:
            raise RuntimeError(
                f"Cannot open partial shard {partial_path}; repair the interrupted HDF5 "
                "container before resuming"
            ) from exc
        print(
            f"Rank {args.rank}: resuming partial shard at "
            f"{resume_count}/{len(records)} proteins",
            flush=True,
        )

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device, args.dtype)
    residue_dim = int(model.config.d_model)
    out_dtype = numpy_dtype(args.dtype)
    text_dtype = h5py.string_dtype("utf-8")
    compression = hdf5_compression(args.compression)
    vector_chunks = (min(max(total_residues, 1), 256), residue_dim)

    if partial_path.exists():
        with h5py.File(partial_path, "r+") as h5_file:
            validate_partial_shard(
                h5_file,
                args,
                records,
                offsets,
                batches,
                expected_residue_dim=residue_dim,
            )
            _write_pending_batches(
                h5_file,
                args,
                records,
                offsets,
                batches,
                model,
                tokenizer,
                processed_count=resume_count,
            )
    else:
        with h5py.File(partial_path, "w") as h5_file:
            h5_file.attrs["partial_format_version"] = PARTIAL_FORMAT_VERSION
            h5_file.attrs["model_name"] = args.model_name
            h5_file.attrs["embedding_model_type"] = "prott5"
            h5_file.attrs["embedding_backend"] = "huggingface_t5_encoder"
            h5_file.attrs["residue_dim"] = residue_dim
            h5_file.attrs["source_fasta"] = str(Path(args.fasta).resolve())
            h5_file.attrs["max_sequence_length"] = (
                -1 if args.max_sequence_length is None else args.max_sequence_length
            )
            h5_file.attrs["sequence_truncation"] = args.sequence_truncation
            h5_file.attrs["rank"] = args.rank
            h5_file.attrs["world_size"] = args.world_size
            h5_file.attrs["created_unix_time"] = time.time()
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
            h5_file.create_dataset(
                "vectors",
                shape=(total_residues, residue_dim),
                dtype=out_dtype,
                chunks=vector_chunks,
                compression=compression,
            )
            h5_file.attrs[PROCESSED_COUNT_ATTR] = 0
            h5_file.flush()
            _write_pending_batches(
                h5_file,
                args,
                records,
                offsets,
                batches,
                model,
                tokenizer,
                processed_count=0,
            )

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
    if args.merge_proteins_per_chunk <= 0:
        raise ValueError("merge_proteins_per_chunk must be positive")
    if args.merge_storage == "virtual" and args.merge_order != "shard":
        raise ValueError("Virtual merge storage requires --merge-order shard")
    if args.merge_storage == "virtual" and args.cleanup_shards:
        raise ValueError("Virtual merge storage requires retaining its source shards")
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

        entries_by_fasta = sorted(entries, key=lambda item: item[0])
        expected_indices = list(range(len(entries)))
        actual_indices = [entry[0] for entry in entries_by_fasta]
        if actual_indices != expected_indices:
            raise ValueError("Shard indices do not form a contiguous FASTA-order range")
        entries = (
            entries_by_fasta
            if args.merge_order == "fasta"
            else sorted(entries, key=lambda item: (item[1], item[2]))
        )

        final_offsets = np.zeros(len(entries) + 1, dtype=np.int64)
        if entries:
            final_offsets[1:] = np.cumsum([entry[4] for entry in entries], dtype=np.int64)
        total_residues = int(final_offsets[-1])
        compression = hdf5_compression(args.compression)
        vector_chunks = (min(max(total_residues, 1), 256), residue_dim)
        text_dtype = h5py.string_dtype("utf-8")

        with h5py.File(partial_path, "w") as output_file:
            output_file.attrs["model_name"] = shard_files[0].attrs.get("model_name", "")
            output_file.attrs["embedding_model_type"] = shard_files[0].attrs.get(
                "embedding_model_type",
                "prott5",
            )
            output_file.attrs["embedding_backend"] = shard_files[0].attrs.get(
                "embedding_backend",
                "huggingface_t5_encoder",
            )
            output_file.attrs["residue_dim"] = residue_dim
            output_file.attrs["source_fasta"] = str(Path(args.fasta).resolve())
            output_file.attrs["max_sequence_length"] = shard_files[0].attrs.get(
                "max_sequence_length",
                -1,
            )
            output_file.attrs["sequence_truncation"] = shard_files[0].attrs.get(
                "sequence_truncation",
                "",
            )
            output_file.attrs["world_size"] = args.world_size
            output_file.attrs["merge_order"] = args.merge_order
            output_file.attrs["merge_storage"] = args.merge_storage
            output_file.attrs["created_unix_time"] = time.time()
            output_file.create_dataset(
                "ids",
                data=np.array([entry[3] for entry in entries], dtype=object),
                dtype=text_dtype,
            )
            output_file.create_dataset("offsets", data=final_offsets, dtype=np.int64)
            if args.merge_storage == "virtual":
                layout = h5py.VirtualLayout(
                    shape=(total_residues, residue_dim),
                    dtype=out_dtype,
                )
                for rank in range(args.world_size):
                    source_shape = shard_files[rank]["vectors"].shape
                    if source_shape[0] == 0:
                        continue
                    rank_start = sum(
                        int(shard_files[previous]["vectors"].shape[0]) for previous in range(rank)
                    )
                    source = h5py.VirtualSource(
                        str(shard_paths[rank].resolve()),
                        "vectors",
                        shape=source_shape,
                    )
                    layout[rank_start : rank_start + source_shape[0]] = source
                output_file.create_virtual_dataset("vectors", layout)
                print(
                    f"Mapped {len(entries)} proteins from {args.world_size} immutable shards",
                    flush=True,
                )
            else:
                output_vectors = output_file.create_dataset(
                    "vectors",
                    shape=(total_residues, residue_dim),
                    dtype=out_dtype,
                    chunks=vector_chunks,
                    compression=compression,
                )
                shard_offsets = [h5_file["offsets"][:] for h5_file in shard_files]
                next_progress = args.progress_every
                for chunk_start in range(0, len(entries), args.merge_proteins_per_chunk):
                    chunk_end = min(chunk_start + args.merge_proteins_per_chunk, len(entries))
                    chunk_entries = entries[chunk_start:chunk_end]
                    output_start = int(final_offsets[chunk_start])
                    output_end = int(final_offsets[chunk_end])
                    output_chunk = np.empty(
                        (output_end - output_start, residue_dim),
                        dtype=out_dtype,
                    )

                    rank_blocks: dict[int, tuple[int, np.ndarray]] = {}
                    for rank in range(args.world_size):
                        local_indices = [
                            local_idx
                            for (
                                _global_idx,
                                entry_rank,
                                local_idx,
                                _protein_id,
                                _length,
                            ) in chunk_entries
                            if entry_rank == rank
                        ]
                        if not local_indices:
                            continue
                        first_local = local_indices[0]
                        last_local = local_indices[-1]
                        if local_indices != list(range(first_local, last_local + 1)):
                            raise ValueError("Per-rank merge indices are not contiguous")
                        input_start = int(shard_offsets[rank][first_local])
                        input_end = int(shard_offsets[rank][last_local + 1])
                        rank_blocks[rank] = (
                            input_start,
                            shard_files[rank]["vectors"][input_start:input_end],
                        )

                    chunk_write_start = 0
                    for _global_idx, rank, local_idx, _protein_id, length in chunk_entries:
                        if length:
                            input_start = int(shard_offsets[rank][local_idx])
                            block_start, block = rank_blocks[rank]
                            relative_start = input_start - block_start
                            output_chunk[chunk_write_start : chunk_write_start + length] = block[
                                relative_start : relative_start + length
                            ]
                        chunk_write_start += length
                    output_vectors[output_start:output_end] = output_chunk

                    if chunk_end == len(entries) or chunk_end >= next_progress:
                        print(
                            f"Merged {chunk_end}/{len(entries)} proteins into {output_path}",
                            flush=True,
                        )
                        while next_progress <= chunk_end:
                            next_progress += args.progress_every

            output_file.flush()

        os.replace(partial_path, output_path)
        if args.merged_ids_output:
            merged_ids_path = Path(args.merged_ids_output)
            merged_ids_path.parent.mkdir(parents=True, exist_ok=True)
            merged_ids_partial = merged_ids_path.with_suffix(merged_ids_path.suffix + ".partial")
            with merged_ids_partial.open("w", encoding="utf-8") as handle:
                for entry in entries:
                    handle.write(entry[3] + "\n")
            os.replace(merged_ids_partial, merged_ids_path)
    finally:
        for h5_file in shard_files:
            h5_file.close()

    if args.cleanup_shards:
        for path in shard_paths:
            path.unlink(missing_ok=True)

    print(f"Wrote merged residue HDF5: {output_path}", flush=True)
    return output_path


def main() -> None:
    args = parse_args()
    if args.progress_every <= 0:
        raise ValueError("progress_every must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")

    def request_termination(signum: int, _frame: object) -> None:
        raise TerminationRequested(signum)

    handled_signals = (signal.SIGHUP, signal.SIGTERM)
    previous_handlers = {signum: signal.getsignal(signum) for signum in handled_signals}
    for signum in handled_signals:
        signal.signal(signum, request_termination)
    try:
        if args.download_only:
            load_model_and_tokenizer(args.model_name, args.device, args.dtype)
            print(f"Downloaded and loaded {args.model_name}", flush=True)
            return

        if args.merge_only:
            merge_shards(args)
            return

        write_shard(args)
    except TerminationRequested as exc:
        print(
            f"Termination signal {exc.signum} received; closed at the last committed checkpoint",
            flush=True,
        )
        raise SystemExit(128 + exc.signum) from None
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
