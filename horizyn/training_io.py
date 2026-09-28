"""Opt-in training I/O, isolated from the fingerprinted extraction pipeline.

These adapters do not alter sampling, annotations, the model, or its loss.
Importing this module does not activate them; the dedicated launcher does.
"""
from contextlib import contextmanager
import time

import h5py
import numpy as np
import torch

from horizyn.datasets.indexed_pairs import ARRAY_NAMES, IndexedPairs, IndexedPositiveMap
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset, truncate_residue_embeddings
from horizyn.reaction_conditioned_data_module import ReactionConditionedDataModule
from horizyn.validation_runtime import FastValidationLitModule, ShardedValidationDataMixin


class ResidentIndexedPairs(IndexedPairs):
    """One immutable RAM copy per DDP rank; fork workers share read-only pages.

Spawned workers only fetch rows, so reopen the original read-only memmaps
instead of serializing/copying another full resident index into each worker.
"""

    def __init__(self, directory, max_bytes=2 * 1024**3):
        start = time.perf_counter()
        super().__init__(directory)  # Preserve every original schema check.
        total = sum(getattr(self, name).nbytes for name in ARRAY_NAMES)
        if total > max_bytes:
            raise ValueError(f"Resident index needs {total} bytes; limit is {max_bytes}")
        for name in ARRAY_NAMES:
            value = np.array(getattr(self, name), copy=True, order="C")
            value.flags.writeable = False
            setattr(self, name, value)
        # The maps hold ID-array references, not just a reference to the index.
        self.query_to_targets = IndexedPositiveMap(self)
        self.target_to_queries = IndexedPositiveMap(self, reverse=True)
        self.resident_bytes = total
        print(f"Resident training index: {total / 1024**2:.1f} MiB, "
              f"{time.perf_counter() - start:.1f}s; original sampling policy", flush=True)

    def __reduce__(self):
        return IndexedPairs, (str(self.directory),)


class StoragePrecisionResidues(ResidueEmbedDataset):
    """Keep FP16 small and retain the VDS handle (and its source-file caches)."""

    def __init__(self, file_path, **kwargs):
        self._vectors = None
        with h5py.File(file_path, "r") as handle:
            source_dtype = handle["vectors"].dtype
        kwargs["dtype"] = torch.float16 if source_dtype == np.dtype("float16") else torch.float32
        super().__init__(file_path, **kwargs)

    def __getitem__(self, key):
        if self.in_memory:
            return super().__getitem__(key)
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            actual_key, index = self.keys[key], key
        else:
            actual_key, index = key, self._get_idx(key)
        if self.h5_indices is None:
            raise RuntimeError("Residue HDF5 index mapping is not initialized")
        row = int(self.h5_indices[index].item())
        start, end = int(self.offsets[row].item()), int(self.offsets[row + 1].item())
        handle = self._ensure_file()  # Also closes inherited handles after fork.
        if self._vectors is None:
            self._vectors = handle["vectors"]
        # Reopening the VDS dataset for each protein discards its source-file
        # handles/chunk metadata caches. Keep it alive for this worker instead.
        residues = torch.from_numpy(self._vectors[start:end]).to(dtype=self.dtype)
        residues = truncate_residue_embeddings(residues, self.max_tokens, self.truncation)
        if self.validate_finite_on_access and not torch.isfinite(residues).all():
            raise ValueError(f"Residue embeddings for {actual_key!r} contain non-finite values")
        return self._apply_transforms(actual_key, {"residue_embeddings": residues})

    def close(self):
        self._vectors = None
        super().close()

    def __getstate__(self):
        state = super().__getstate__()
        state["_vectors"] = None
        return state


_RESIDUE_FIELDS = frozenset((
    "residue_embeddings", "protein_residue_embeddings",
    "score_residue_embeddings", "protein_score_residue_embeddings",
))


def restore_residue_precision(batch):
    """Restore the original FP32 model input *after* the device transfer.

The cast is exact for stored FP16 values. Other tensors (labels, masks,
reaction features) and non-floating metadata are left alone.
"""
    if isinstance(batch, dict):
        return {
            key: value.float() if key in _RESIDUE_FIELDS and torch.is_tensor(value)
            else restore_residue_precision(value)
            for key, value in batch.items()
        }
    if isinstance(batch, list):
        return [restore_residue_precision(value) for value in batch]
    if isinstance(batch, tuple):
        return tuple(restore_residue_precision(value) for value in batch)
    return batch


class ResidueTransportDataModule(ReactionConditionedDataModule):
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return restore_residue_precision(batch)


class FastIODataModule(ShardedValidationDataMixin, ResidueTransportDataModule):
    pass


@contextmanager
def training_io_adapters(entrypoint, *, residue_only=False):
    """Install process-local factories, restoring them even on failure.

Keeping the existing controller/data files unchanged protects extraction
receipts and other running interpreters. No patch is written to those files.
"""
    import horizyn.reaction_conditioned_data_module as data_module

    original = (data_module.IndexedPairs, data_module.ResidueEmbedDataset,
                entrypoint.ReactionConditionedDataModule)
    original_module = getattr(entrypoint, "ProteinPooledLitModule", None)
    if not residue_only:
        data_module.IndexedPairs = ResidentIndexedPairs
    data_module.ResidueEmbedDataset = StoragePrecisionResidues
    entrypoint.ReactionConditionedDataModule = ResidueTransportDataModule if residue_only else FastIODataModule
    if original_module is not None and not residue_only:
        entrypoint.ProteinPooledLitModule = FastValidationLitModule
    try:
        yield
    finally:
        (data_module.IndexedPairs, data_module.ResidueEmbedDataset,
         entrypoint.ReactionConditionedDataModule) = original
        if original_module is not None:
            entrypoint.ProteinPooledLitModule = original_module
