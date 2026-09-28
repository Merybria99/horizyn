"""
Dataset for loading ragged residue-level protein embeddings from HDF5 files.
"""

import json
import os
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import h5py
import numpy as np
import torch

from horizyn.datasets.base import BaseDataset

FINITE_CERTIFICATE_SCHEMA = "horizyn_residue_hdf5_finite_validation_v1"


def residue_finite_certificate_path(file_path: str | Path) -> Path:
    """Return the default sidecar path for a residue HDF5 store."""

    resolved = Path(file_path).expanduser().resolve(strict=True)
    return resolved.with_name(resolved.name + ".finite.json")


def _file_identity(path: Path) -> dict[str, int | str]:
    resolved = path.expanduser().resolve(strict=True)
    stat_result = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
    }


def residue_hdf5_store_identity(file_path: str | Path) -> dict[str, Any]:
    """Describe the HDF5 container and all files backing its vector dataset."""

    resolved = Path(file_path).expanduser().resolve(strict=True)
    with h5py.File(resolved, "r") as h5_file:
        missing = [name for name in ("ids", "vectors", "offsets") if name not in h5_file]
        if missing:
            raise ValueError(f"Residue HDF5 is missing datasets: {', '.join(missing)}")
        vectors = h5_file["vectors"]
        if vectors.ndim != 2:
            raise ValueError(f"Residue HDF5 vectors must be rank-2, got {vectors.shape}")

        source_paths = {resolved}
        if vectors.is_virtual:
            for source in vectors.virtual_sources():
                source_name = os.fsdecode(source.file_name)
                if source_name == ".":
                    source_path = resolved
                else:
                    source_path = Path(source_name).expanduser()
                    if not source_path.is_absolute():
                        source_path = resolved.parent / source_path
                source_paths.add(source_path.resolve(strict=True))

        return {
            "container": _file_identity(resolved),
            "backing_files": [
                _file_identity(path) for path in sorted(source_paths, key=lambda item: str(item))
            ],
            "ids_shape": list(h5_file["ids"].shape),
            "ids_dtype": str(h5_file["ids"].dtype),
            "offsets_shape": list(h5_file["offsets"].shape),
            "offsets_dtype": str(h5_file["offsets"].dtype),
            "vectors_shape": list(vectors.shape),
            "vectors_dtype": str(vectors.dtype),
            "vectors_is_virtual": bool(vectors.is_virtual),
        }


def _scan_residue_vector_range(
    file_path: str,
    start: int,
    stop: int,
    rows_per_chunk: int,
    progress_every_chunks: int,
    worker_index: int,
) -> tuple[int, int]:
    rows_scanned = 0
    chunks_scanned = 0
    with h5py.File(file_path, "r") as h5_file:
        vectors = h5_file["vectors"]
        vector_dim = int(vectors.shape[1])
        for chunk_start in range(start, stop, rows_per_chunk):
            chunk_end = min(chunk_start + rows_per_chunk, stop)
            block = np.asarray(vectors[chunk_start:chunk_end])
            finite = np.isfinite(block)
            if not bool(finite.all()):
                invalid = np.argwhere(~finite)[0]
                raise ValueError(
                    "Residue HDF5 contains a non-finite value at "
                    f"vectors[{chunk_start + int(invalid[0])}, {int(invalid[1])}]"
                )
            rows_scanned += chunk_end - chunk_start
            chunks_scanned += 1
            if progress_every_chunks > 0 and (
                chunks_scanned % progress_every_chunks == 0 or chunk_end == stop
            ):
                print(
                    f"Finite validation worker {worker_index}: "
                    f"{rows_scanned:,}/{stop - start:,} rows "
                    f"({rows_scanned / max(stop - start, 1):.1%})",
                    flush=True,
                )
    return rows_scanned, rows_scanned * vector_dim


def validate_residue_hdf5_finite(
    file_path: str | Path,
    *,
    certificate_path: str | Path | None = None,
    rows_per_chunk: int = 8192,
    progress_every_chunks: int = 100,
    workers: int = 1,
) -> Path:
    """Scan every residue vector and atomically certify that all values are finite."""

    if type(rows_per_chunk) is not int or rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be a positive integer")
    if type(progress_every_chunks) is not int or progress_every_chunks < 0:
        raise ValueError("progress_every_chunks must be a non-negative integer")
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive integer")

    resolved = Path(file_path).expanduser().resolve(strict=True)
    output_path = (
        residue_finite_certificate_path(resolved)
        if certificate_path is None
        else Path(certificate_path).expanduser().resolve()
    )
    store_identity = residue_hdf5_store_identity(resolved)
    vectors_shape = store_identity["vectors_shape"]
    total_rows = int(vectors_shape[0])
    effective_workers = min(workers, max(total_rows, 1))
    ranges = [
        (
            total_rows * worker_index // effective_workers,
            total_rows * (worker_index + 1) // effective_workers,
        )
        for worker_index in range(effective_workers)
    ]
    arguments = [
        (
            str(resolved),
            start,
            stop,
            rows_per_chunk,
            progress_every_chunks,
            worker_index,
        )
        for worker_index, (start, stop) in enumerate(ranges)
    ]
    if effective_workers == 1:
        scan_results = [_scan_residue_vector_range(*arguments[0])]
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            scan_results = list(executor.map(_scan_residue_vector_range_from_tuple, arguments))

    rows_scanned = sum(result[0] for result in scan_results)
    values_scanned = sum(result[1] for result in scan_results)
    if rows_scanned != total_rows:
        raise RuntimeError(f"Finite validation scanned {rows_scanned} rows, expected {total_rows}")

    # Reject a certificate if the container or a virtual source changed mid-scan.
    final_identity = residue_hdf5_store_identity(resolved)
    if final_identity != store_identity:
        raise RuntimeError("Residue HDF5 store changed while it was being validated")

    certificate = {
        "schema": FINITE_CERTIFICATE_SCHEMA,
        "finite": True,
        "store": store_identity,
        "rows_scanned": rows_scanned,
        "values_scanned": values_scanned,
        "rows_per_chunk": rows_per_chunk,
        "workers": effective_workers,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def _scan_residue_vector_range_from_tuple(
    arguments: tuple[str, int, int, int, int, int],
) -> tuple[int, int]:
    return _scan_residue_vector_range(*arguments)


def verify_residue_hdf5_finite_certificate(
    file_path: str | Path,
    certificate_path: str | Path | None = None,
) -> Path:
    """Verify that a finite-value certificate describes the current HDF5 store."""

    resolved = Path(file_path).expanduser().resolve(strict=True)
    sidecar = (
        residue_finite_certificate_path(resolved)
        if certificate_path is None
        else Path(certificate_path).expanduser().resolve(strict=True)
    )
    try:
        certificate = json.loads(sidecar.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "Per-access finite checks may be disabled only for a certified residue store. "
            f"Missing certificate: {sidecar}"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid residue finite-value certificate: {sidecar}") from error

    current_identity = residue_hdf5_store_identity(resolved)
    expected_rows = int(current_identity["vectors_shape"][0])
    expected_values = expected_rows * int(current_identity["vectors_shape"][1])
    if (
        not isinstance(certificate, dict)
        or certificate.get("schema") != FINITE_CERTIFICATE_SCHEMA
        or certificate.get("finite") is not True
        or certificate.get("store") != current_identity
        or certificate.get("rows_scanned") != expected_rows
        or certificate.get("values_scanned") != expected_values
    ):
        raise ValueError(
            f"Residue finite-value certificate does not match the current store: {sidecar}"
        )
    return sidecar


def truncate_residue_embeddings(
    embeddings: torch.Tensor,
    max_tokens: int | None = None,
    strategy: str = "ends_center",
) -> torch.Tensor:
    """
    Deterministically truncate residue embeddings.

    The default strategy keeps the first 25%, centered 50%, and last 25% of the
    requested token budget, preserving original residue order.
    """
    if max_tokens is None or embeddings.shape[0] <= max_tokens:
        return embeddings
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive when provided")
    if strategy != "ends_center":
        raise ValueError(f"Unsupported protein truncation strategy: {strategy}")

    length = embeddings.shape[0]
    first_count = max_tokens // 4
    last_count = max_tokens // 4
    middle_count = max_tokens - first_count - last_count

    first_idx = torch.arange(first_count, dtype=torch.long)
    middle_start = max((length - middle_count) // 2, first_count)
    middle_end = min(middle_start + middle_count, length - last_count)
    middle_start = max(middle_end - middle_count, first_count)
    middle_idx = torch.arange(middle_start, middle_end, dtype=torch.long)
    last_idx = torch.arange(length - last_count, length, dtype=torch.long)

    indices = torch.cat([first_idx, middle_idx, last_idx])
    return embeddings[indices]


class ResidueEmbedDataset(BaseDataset[str]):
    """
    Dataset for unpooled ProtT5 residue embeddings stored in ragged HDF5 format.

    Expected HDF5 schema:
        - ``ids``: protein identifiers, shape ``[N]``
        - ``vectors``: flattened residue embeddings, shape ``[sum_L, D]``
        - ``offsets``: integer offsets into ``vectors``, shape ``[N + 1]``
    """

    def __init__(
        self,
        file_path: str,
        in_memory: bool = False,
        dtype: torch.dtype = torch.float32,
        max_tokens: int | None = None,
        truncation: str = "ends_center",
        drop_empty: bool = True,
        validate_finite_on_access: bool = True,
        finite_validation_sidecar: str | None = None,
        allow_uncertified_finite_skip: bool = False,
        transforms: Optional[Callable[[str, Any], Any]] = None,
        **kwargs,
    ):
        file_path_obj = Path(file_path)
        if not file_path_obj.is_file():
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")

        self.file_path = str(file_path_obj)
        self.in_memory = in_memory
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.truncation = truncation
        self.drop_empty = drop_empty
        if type(validate_finite_on_access) is not bool:
            raise TypeError("validate_finite_on_access must be boolean")
        if type(allow_uncertified_finite_skip) is not bool:
            raise TypeError("allow_uncertified_finite_skip must be boolean")
        self.validate_finite_on_access = validate_finite_on_access
        self.finite_validation_sidecar = None
        if not validate_finite_on_access:
            if allow_uncertified_finite_skip:
                warnings.warn(
                    "Skipping residue finite checks without a store certificate; the caller "
                    "must validate every derived output batch before use",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                self.finite_validation_sidecar = str(
                    verify_residue_hdf5_finite_certificate(
                        file_path_obj,
                        finite_validation_sidecar,
                    )
                )
        self.file: h5py.File | None = None
        self._file_pid: int | None = None
        self.data: torch.Tensor | None = None
        self.h5_indices: torch.Tensor | None = None

        with h5py.File(self.file_path, "r") as h5_file:
            for dataset_name in ("ids", "vectors", "offsets"):
                if dataset_name not in h5_file:
                    available = list(h5_file.keys())
                    raise KeyError(
                        f"Required dataset '{dataset_name}' not found in HDF5 file. "
                        f"Available datasets: {available}"
                    )

            vectors_shape = h5_file["vectors"].shape
            if len(vectors_shape) != 2:
                raise ValueError(f"'vectors' must be rank-2, got shape={vectors_shape}")
            if vectors_shape[1] == 0:
                raise ValueError(f"'vectors' embedding dimension must be positive: {vectors_shape}")
            if h5_file["vectors"].dtype.kind not in {"f", "i", "u"}:
                raise ValueError("'vectors' must have a numeric dtype")
            self.num_residues, self.vec_dim = vectors_shape

            if h5_file["ids"].ndim != 1:
                raise ValueError(f"'ids' must be rank-1, got shape={h5_file['ids'].shape}")
            ids_data = h5_file["ids"][:]
            if ids_data.dtype.kind in {"S", "O"}:
                keys = [
                    id_val.decode("utf-8") if isinstance(id_val, bytes) else str(id_val)
                    for id_val in ids_data
                ]
            else:
                keys = [str(id_val) for id_val in ids_data]
            if any(not key.strip() for key in keys):
                raise ValueError("Residue HDF5 'ids' must not contain empty strings")
            if len(keys) != len(set(keys)):
                raise ValueError("Residue HDF5 'ids' must be unique")

            if h5_file["offsets"].dtype.kind not in {"i", "u"}:
                raise ValueError("'offsets' must have an integer dtype")
            offsets_tensor = torch.as_tensor(h5_file["offsets"][:], dtype=torch.long)
            if offsets_tensor.ndim != 1:
                raise ValueError("'offsets' must be rank-1")
            if len(offsets_tensor) != len(keys) + 1:
                raise ValueError(
                    f"'offsets' length must equal len(ids)+1, got {len(offsets_tensor)} "
                    f"for {len(keys)} ids"
                )
            if int(offsets_tensor[0].item()) != 0:
                raise ValueError("'offsets' must start at 0")
            if int(offsets_tensor[-1].item()) != self.num_residues:
                raise ValueError(
                    f"'offsets' final value ({int(offsets_tensor[-1].item())}) must equal "
                    f"number of residue vectors ({self.num_residues})"
                )
            if not torch.all(offsets_tensor[1:] >= offsets_tensor[:-1]):
                raise ValueError("'offsets' must be monotonically non-decreasing")

            self.offsets = offsets_tensor
            lengths = offsets_tensor[1:] - offsets_tensor[:-1]
            self.source_keys = tuple(keys)
            self.source_lengths = lengths.clone()
            self.length_by_key = {
                protein_id: int(length.item())
                for protein_id, length in zip(self.source_keys, self.source_lengths)
            }
            if drop_empty:
                h5_indices = torch.nonzero(lengths > 0, as_tuple=False).flatten()
                keys = [keys[int(idx.item())] for idx in h5_indices]
            else:
                h5_indices = torch.arange(len(keys), dtype=torch.long)
            self.h5_indices = h5_indices
            self.lengths = lengths.index_select(0, h5_indices).clone()

            if self.in_memory:
                self.data = torch.from_numpy(h5_file["vectors"][:]).to(dtype=self.dtype)

        super().__init__(keys=keys, use_key_to_idx=True, transforms=transforms, **kwargs)

    def _ensure_file(self) -> h5py.File:
        current_pid = os.getpid()
        if self.file is not None and self._file_pid != current_pid:
            self.close()
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")
            self._file_pid = current_pid
        return self.file

    def __getitem__(self, key: str | int) -> dict[str, torch.Tensor]:
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            actual_key = self.keys[key]
            idx = key
        else:
            actual_key = key
            idx = self._get_idx(actual_key)

        if self.h5_indices is None:
            raise RuntimeError("Residue HDF5 index mapping is not initialized")
        h5_idx = int(self.h5_indices[idx].item())
        start = int(self.offsets[h5_idx].item())
        end = int(self.offsets[h5_idx + 1].item())

        if self.in_memory:
            if self.data is None:
                raise RuntimeError("In-memory residue data is not initialized")
            residue_embeddings = self.data[start:end]
        else:
            h5_file = self._ensure_file()
            residue_embeddings = torch.from_numpy(h5_file["vectors"][start:end]).to(
                dtype=self.dtype
            )

        residue_embeddings = truncate_residue_embeddings(
            residue_embeddings,
            max_tokens=self.max_tokens,
            strategy=self.truncation,
        )
        if self.validate_finite_on_access and not torch.isfinite(residue_embeddings).all():
            raise ValueError(f"Residue embeddings for {actual_key!r} contain non-finite values")
        sample = {"residue_embeddings": residue_embeddings}
        return self._apply_transforms(actual_key, sample)

    def close(self) -> None:
        file_handle = getattr(self, "file", None)
        if file_handle is not None:
            try:
                file_handle.close()
            except Exception:
                pass
        self.file = None
        self._file_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["file"] = None
        state["_file_pid"] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
