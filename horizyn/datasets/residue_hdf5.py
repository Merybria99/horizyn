"""
Dataset for loading ragged residue-level protein embeddings from HDF5 files.
"""

from pathlib import Path
import os
from typing import Any, Callable, Optional

import h5py
import torch

from horizyn.datasets.base import BaseDataset


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
        self.validate_finite_on_access = validate_finite_on_access
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
