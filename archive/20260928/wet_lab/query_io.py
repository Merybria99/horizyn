"""Process-local RefSeq I/O acceleration; scoring and cache identities stay unchanged."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch

from horizyn.artifacts import sha256_file
from horizyn.benchmarks.retrieval import (
    _prepare_target_embedding_chunk_store,
    _target_embedding_chunk_path,
    encode_residue_targets,
)
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from horizyn.training_io import StoragePrecisionResidues


class PrefetchedRows:
    """Ordered CPU rows: the current batch plus at most one look-ahead batch.

    Only the reader thread accesses the dataset. The original encoder still
    validates every resumed chunk and performs collation and all CUDA work.
    """

    def __init__(self, dataset, keys, batch_size, chunk_dir=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batches = self._pending_batches(keys, batch_size, chunk_dir)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="refseq-reader")
        self.future = None
        self.rows = deque()

    @staticmethod
    def _pending_batches(keys, batch_size, chunk_dir):
        for start in range(0, len(keys), batch_size):
            end = min(start + batch_size, len(keys))
            # The encoder, not this existence check, decides cache validity.
            if (
                chunk_dir is None
                or not _target_embedding_chunk_path(chunk_dir, start, end).exists()
            ):
                yield keys[start:end]

    def _read(self, keys):
        with torch.inference_mode():
            return deque((key, self.dataset[key]) for key in keys)

    def _submit(self):
        keys = next(self.batches, None)
        self.future = None if keys is None else self.executor.submit(self._read, keys)

    def __getitem__(self, key):
        if not self.rows:
            if self.future is None:
                self._submit()
            if self.future is None:
                raise KeyError(f"No pending residue row for {key!r}")
            self.rows = self.future.result()
            self._submit()
        expected, sample = self.rows.popleft()
        if key != expected:
            raise ValueError(f"Prefetch order mismatch: requested {key!r}, expected {expected!r}")
        return sample

    def close(self):
        # Never close an HDF5 handle while the reader is using it.
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.future = None
        self.rows.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def encode_prefetched_targets(
    module,
    dataset,
    target_keys,
    device,
    batch_size,
    store_on_device,
    *,
    score_dataset=None,
    **kwargs,
):
    chunk_dir = _prepare_target_embedding_chunk_store(
        kwargs.get("cache_info"),
        target_keys,
        batch_size,
    )
    with ExitStack() as stack:
        dataset = stack.enter_context(PrefetchedRows(dataset, target_keys, batch_size, chunk_dir))
        if score_dataset is not None:
            score_dataset = stack.enter_context(
                PrefetchedRows(score_dataset, target_keys, batch_size, chunk_dir)
            )
        return encode_residue_targets(
            module,
            dataset,
            target_keys,
            device,
            batch_size,
            store_on_device,
            score_dataset=score_dataset,
            **kwargs,
        )


@contextmanager
def screening_io_adapters(query):
    """Reuse the tested persistent reader without editing shared training code.

    Keep the exact requested dtype, truncation, finite checks, batch boundaries
    and original encoder. An independent receipt records this I/O-only adapter;
    existing checkpoint-specific cache provenance is never overridden.
    """
    original = query.ResidueEmbedDataset, query.encode_residue_targets
    readers = []

    def reader(*args, **kwargs):
        if kwargs.get("in_memory", False):
            result = ResidueEmbedDataset(*args, **kwargs)
        else:
            result = StoragePrecisionResidues(*args, **kwargs)
            result.dtype = kwargs.get("dtype", torch.float32)
        readers.append(result)
        return result

    query.ResidueEmbedDataset = reader
    query.encode_residue_targets = encode_prefetched_targets
    try:
        print(
            "RefSeq fast I/O: persistent HDF5 handles; one CPU batch prefetched; unchanged scoring",
            flush=True,
        )
        yield {
            "mode": "persistent_hdf5_prefetch_v1",
            "prefetch_batches": 1,
            "adapter_sha256": sha256_file(__file__),
            "reader_sha256": sha256_file(
                Path(__file__).resolve().parents[1] / "horizyn/training_io.py"
            ),
        }
    finally:
        query.ResidueEmbedDataset, query.encode_residue_targets = original
        for dataset in readers:
            dataset.close()
