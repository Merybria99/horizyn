"""
Dataset for loading embeddings from HDF5 files.
"""

from pathlib import Path
import os
from typing import Any, Callable, Optional

import h5py
import torch

from horizyn.datasets.base import BaseDataset


class EmbedDataset(BaseDataset[str]):
    """
    Dataset for loading pre-computed embeddings from HDF5 files.

    This dataset loads vector embeddings (e.g., protein T5 embeddings) from HDF5
    files with support for in-memory caching. HDF5 files are expected to have:
        - 'ids': dataset of string identifiers (shape: [N])
        - 'vectors': dataset of embedding vectors (shape: [N, D])

    Where N is the number of embeddings and D is the embedding dimension.

    For training efficiency, the entire embedding matrix can be loaded into
    memory at initialization. This is recommended for datasets that fit in RAM
    (typically <16GB for embedding matrices).

    Attributes:
        file_path (str): Path to the HDF5 file.
        in_memory (bool): Whether embeddings are loaded into memory.
        dtype (torch.dtype): Data type for returned tensors.
        num_vecs (int): Number of vectors in the dataset.
        vec_dim (int): Dimension of each vector.
        file (h5py.File): Open HDF5 file handle (if not in_memory).
        data (torch.Tensor): In-memory tensor of all embeddings (if in_memory).

    Note:
        All keys are converted to strings, even if the 'ids' dataset contains integers.
        This eliminates ambiguity between integer indices (for DataLoader) and integer keys.
        - Integer access: `dataset[0]` → array index (first item)
        - String access: `dataset["P12345"]` → protein ID

    Example:
        >>> # Load protein T5 embeddings
        >>> protein_embeds = EmbedDataset(
        ...     file_path="data/sota/prots_t5.h5",
        ...     in_memory=True,
        ...     dtype=torch.float32
        ... )
        >>> embedding = protein_embeds["P12345"]  # Returns tensor of shape [1024]
        >>> print(f"Dataset has {len(protein_embeds)} proteins")
        >>> print(f"Embedding dim: {protein_embeds.vec_dim}")
    """

    def __init__(
        self,
        file_path: str,
        in_memory: bool = True,
        dtype: torch.dtype = torch.float32,
        transforms: Optional[Callable[[str, Any], Any]] = None,
        **kwargs,
    ):
        """
        Initialize the EmbedDataset.

        Args:
            file_path: Path to the HDF5 file. Must exist and contain 'ids' and
                'vectors' datasets.
            in_memory: Whether to load all embeddings into memory at initialization.
                Recommended for training (faster access). Defaults to True.
            dtype: PyTorch data type for returned tensors. Defaults to torch.float32.
            transforms: Optional transform function. Defaults to None.
            **kwargs: Additional keyword arguments.

        Raises:
            FileNotFoundError: If the HDF5 file doesn't exist.
            KeyError: If 'ids' or 'vectors' datasets are missing from the file.
            ValueError: If 'ids' and 'vectors' have mismatched lengths.
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.is_file():
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")

        self.file_path = str(file_path_obj)
        self.in_memory = in_memory
        self.dtype = dtype

        self.file: h5py.File | None = None
        self._file_pid: int | None = None
        with h5py.File(self.file_path, "r") as h5_file:
            for dataset_name in ("ids", "vectors"):
                if dataset_name not in h5_file:
                    raise KeyError(
                        f"Required dataset '{dataset_name}' not found in HDF5 file. "
                        f"Available datasets: {list(h5_file.keys())}"
                    )

            vector_shape = h5_file["vectors"].shape
            if len(vector_shape) != 2:
                raise ValueError(f"'vectors' must be rank-2, got shape={vector_shape}")
            if h5_file["vectors"].dtype.kind not in {"f", "i", "u"}:
                raise ValueError("HDF5 'vectors' must have a numeric dtype")
            self.num_vecs, self.vec_dim = vector_shape
            num_ids = len(h5_file["ids"])
            if self.num_vecs != num_ids:
                raise ValueError(
                    f"Mismatch between number of ids ({num_ids}) and vectors ({self.num_vecs})"
                )

            ids_data = h5_file["ids"][:]
            if ids_data.dtype.kind in {"S", "O"}:
                keys = [
                    id_val.decode("utf-8") if isinstance(id_val, bytes) else str(id_val)
                    for id_val in ids_data
                ]
            else:
                keys = [str(id_val) for id_val in ids_data]
            if any(not key.strip() for key in keys):
                raise ValueError("HDF5 'ids' must not contain empty strings")
            if len(keys) != len(set(keys)):
                raise ValueError("HDF5 'ids' must be unique")

            self.data = (
                torch.from_numpy(h5_file["vectors"][:]).to(dtype=self.dtype)
                if self.in_memory
                else None
            )

        # Initialize base dataset with key_to_idx mapping enabled
        super().__init__(keys=keys, use_key_to_idx=True, transforms=transforms, **kwargs)

    def __getitem__(self, key: str | int) -> torch.Tensor:
        """
        Get an embedding vector by key or integer index.

        Args:
            key: String identifier from the 'ids' dataset, or integer index (0 to len-1).
                 All HDF5 keys are strings (even if originally integers).

        Returns:
            Tensor of shape [vec_dim] containing the embedding.

        Raises:
            KeyError: If the key is not found in the dataset.
            IndexError: If integer index is out of bounds.
        """
        # Handle integer indexing (for DataLoader)
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            actual_key = self.keys[key]
            idx = key
        else:
            # String key lookup
            actual_key = key
            # Get the index for this key
            idx = self._get_idx(actual_key)

        if self.in_memory:
            if self.data is None:
                raise RuntimeError("Data not initialized")
            vector = self.data[idx]
        else:
            # Load from disk on-the-fly
            # torch.from_numpy avoids copying and shares memory with numpy array
            vector = torch.from_numpy(self._ensure_file()["vectors"][idx]).to(dtype=self.dtype)

        return self._apply_transforms(actual_key, vector)

    def _ensure_file(self) -> h5py.File:
        current_pid = os.getpid()
        if self.file is not None and self._file_pid != current_pid:
            self.close()
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")
            self._file_pid = current_pid
        return self.file

    def close(self) -> None:
        """Close this process's lazy HDF5 handle."""
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
