"""
Datasets for ragged reaction-level molecule embeddings.
"""

from pathlib import Path
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
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Reaction HDF5 file not found: {file_path}")

        self.file_path = str(file_path_obj)
        self.dtype = dtype
        self.file = h5py.File(self.file_path, "r")

        required = [
            "ids",
            "reactant_vectors",
            "reactant_offsets",
            "product_vectors",
            "product_offsets",
        ]
        missing = [name for name in required if name not in self.file]
        if missing:
            available = list(self.file.keys())
            self.file.close()
            raise KeyError(
                f"Missing required reaction HDF5 dataset(s): {missing}. "
                f"Available datasets: {available}"
            )

        ids_data = self.file["ids"][:]
        if ids_data.dtype.kind in {"S", "O"}:
            keys = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in ids_data
            ]
        else:
            keys = [str(value) for value in ids_data]

        num_ids = len(keys)
        if len(self.file["reactant_offsets"]) != num_ids + 1:
            self.file.close()
            raise ValueError("reactant_offsets length must equal len(ids) + 1")
        if len(self.file["product_offsets"]) != num_ids + 1:
            self.file.close()
            raise ValueError("product_offsets length must equal len(ids) + 1")

        self.reactant_offsets = self.file["reactant_offsets"][:].astype("int64")
        self.product_offsets = self.file["product_offsets"][:].astype("int64")
        self.embedding_dim = int(self.file["reactant_vectors"].shape[1])
        product_dim = int(self.file["product_vectors"].shape[1])
        if product_dim != self.embedding_dim:
            self.file.close()
            raise ValueError(
                f"Reactant/product embedding dims differ: {self.embedding_dim} vs {product_dim}"
            )
        if expected_dim is not None and self.embedding_dim != expected_dim:
            self.file.close()
            raise ValueError(
                f"Reaction embedding dim mismatch: expected {expected_dim}, "
                f"got {self.embedding_dim}"
            )
        if (self.reactant_offsets[1:] - self.reactant_offsets[:-1] <= 0).any():
            self.file.close()
            raise ValueError("Every reaction must contain at least one reactant embedding")
        if (self.product_offsets[1:] - self.product_offsets[:-1] <= 0).any():
            self.file.close()
            raise ValueError("Every reaction must contain at least one product embedding")

        super().__init__(keys=keys, use_key_to_idx=True, transforms=transforms, **kwargs)

    def _slice_side(self, vector_key: str, offsets, idx: int) -> torch.Tensor:
        start = int(offsets[idx])
        end = int(offsets[idx + 1])
        vectors = self.file[vector_key][start:end]
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

    def __del__(self):
        if hasattr(self, "file") and self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass


__all__ = ["UniMol2ReactionEmbedDataset"]
