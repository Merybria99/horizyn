"""
Datasets for ragged reaction-level molecule embeddings.
"""

from pathlib import Path
import os
from typing import Any, Callable, Optional

import h5py
import torch

from horizyn.datasets.base import BaseDataset


class UniMol2ReactionEmbedDataset(BaseDataset[str]):
    """
    Load frozen Uni-Mol2 molecule embeddings grouped by reaction side.

    Expected HDF5 schema:
        - ``ids``: reaction IDs, shape ``[N]``
        - ``reactant_vectors``: flattened molecule embeddings, shape ``[sum_R, D]``
        - ``reactant_offsets``: offsets into reactant_vectors, shape ``[N + 1]``
        - ``product_vectors``: flattened molecule embeddings, shape ``[sum_P, D]``
        - ``product_offsets``: offsets into product_vectors, shape ``[N + 1]``
    """

    def __init__(
        self,
        file_path: str,
        dtype: torch.dtype = torch.float32,
        expected_dim: int | None = None,
        transforms: Optional[Callable[[str, Any], Any]] = None,
        **kwargs,
    ):
        file_path_obj = Path(file_path)
        if not file_path_obj.is_file():
            raise FileNotFoundError(f"Reaction HDF5 file not found: {file_path}")

        self.file_path = str(file_path_obj)
        self.dtype = dtype
        self.file: h5py.File | None = None
        self._file_pid: int | None = None

        with h5py.File(self.file_path, "r") as h5_file:
            required = [
                "ids",
                "reactant_vectors",
                "reactant_offsets",
                "product_vectors",
                "product_offsets",
            ]
            missing = [name for name in required if name not in h5_file]
            if missing:
                raise KeyError(
                    f"Missing required reaction HDF5 dataset(s): {missing}. "
                    f"Available datasets: {list(h5_file.keys())}"
                )

            ids_data = h5_file["ids"][:]
            if ids_data.dtype.kind in {"S", "O"}:
                keys = [
                    value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    for value in ids_data
                ]
            else:
                keys = [str(value) for value in ids_data]
            if any(not key.strip() for key in keys):
                raise ValueError("Reaction HDF5 'ids' must not contain empty strings")
            if len(keys) != len(set(keys)):
                raise ValueError("Reaction HDF5 'ids' must be unique")

            num_ids = len(keys)
            if len(h5_file["reactant_offsets"]) != num_ids + 1:
                raise ValueError("reactant_offsets length must equal len(ids) + 1")
            if len(h5_file["product_offsets"]) != num_ids + 1:
                raise ValueError("product_offsets length must equal len(ids) + 1")

            self.reactant_offsets = h5_file["reactant_offsets"][:].astype("int64")
            self.product_offsets = h5_file["product_offsets"][:].astype("int64")
            reactant_shape = h5_file["reactant_vectors"].shape
            product_shape = h5_file["product_vectors"].shape
            if len(reactant_shape) != 2 or len(product_shape) != 2:
                raise ValueError("Reaction vector datasets must be rank-2")
            if h5_file["reactant_vectors"].dtype.kind not in {"f", "i", "u"} or h5_file[
                "product_vectors"
            ].dtype.kind not in {"f", "i", "u"}:
                raise ValueError("Reaction vector datasets must have numeric dtypes")
            if h5_file["reactant_offsets"].dtype.kind not in {"i", "u"} or h5_file[
                "product_offsets"
            ].dtype.kind not in {"i", "u"}:
                raise ValueError("Reaction offset datasets must have integer dtypes")
            self.embedding_dim = int(reactant_shape[1])
            product_dim = int(product_shape[1])
            if product_dim != self.embedding_dim:
                raise ValueError(
                    f"Reactant/product embedding dims differ: {self.embedding_dim} vs {product_dim}"
                )
            if expected_dim is not None and self.embedding_dim != expected_dim:
                raise ValueError(
                    f"Reaction embedding dim mismatch: expected {expected_dim}, "
                    f"got {self.embedding_dim}"
                )
            for side, offsets, vector_count in (
                ("reactant", self.reactant_offsets, reactant_shape[0]),
                ("product", self.product_offsets, product_shape[0]),
            ):
                if offsets[0] != 0 or offsets[-1] != vector_count:
                    raise ValueError(f"{side}_offsets must span exactly [0, {vector_count}]")
                if (offsets[1:] - offsets[:-1] <= 0).any():
                    raise ValueError(f"Every reaction must contain at least one {side} embedding")

        super().__init__(keys=keys, use_key_to_idx=True, transforms=transforms, **kwargs)

    def _slice_side(self, vector_key: str, offsets, idx: int) -> torch.Tensor:
        start = int(offsets[idx])
        end = int(offsets[idx + 1])
        vectors = self._ensure_file()[vector_key][start:end]
        return torch.from_numpy(vectors).to(dtype=self.dtype)

    def __getitem__(self, key: str | int) -> dict[str, torch.Tensor]:
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            actual_key = self.keys[key]
            idx = key
        else:
            actual_key = key
            idx = self._get_idx(actual_key)

        sample = {
            "reactant_embeddings": self._slice_side(
                "reactant_vectors",
                self.reactant_offsets,
                idx,
            ),
            "product_embeddings": self._slice_side(
                "product_vectors",
                self.product_offsets,
                idx,
            ),
        }
        return self._apply_transforms(actual_key, sample)

    def _ensure_file(self) -> h5py.File:
        current_pid = os.getpid()
        if self.file is not None and self._file_pid != current_pid:
            self.close()
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")
            self._file_pid = current_pid
        return self.file

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


__all__ = ["UniMol2ReactionEmbedDataset"]
