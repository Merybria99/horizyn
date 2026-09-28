import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from horizyn.benchmarks.retrieval import encode_residue_targets
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from wet_lab.query_io import PrefetchedRows, encode_prefetched_targets, screening_io_adapters


@pytest.fixture
def residues(tmp_path):
    lengths = [2, 8, 4, 6, 1, 10, 3]
    offsets = np.cumsum([0, *lengths])
    values = np.random.default_rng(7).normal(size=(sum(lengths), 8)).astype("float16")
    layout = h5py.VirtualLayout(shape=values.shape, dtype="float16")
    for i, (start, end) in enumerate(((0, 17), (17, len(values)))):
        source = tmp_path / f"source{i}.h5"
        with h5py.File(source, "w") as handle:
            handle.create_dataset("vectors", data=values[start:end], chunks=(2, 8))
        layout[start:end] = h5py.VirtualSource(str(source), "vectors", shape=(end - start, 8))
    path = tmp_path / "residues.h5"
    with h5py.File(path, "w", libver="latest") as handle:
        handle.create_virtual_dataset("vectors", layout)
        handle["offsets"] = offsets
        handle["ids"] = np.asarray([f"p{i}" for i in range(len(lengths))], dtype="S")
    return path


def query_namespace():
    return SimpleNamespace(
        ResidueEmbedDataset=ResidueEmbedDataset,
        encode_residue_targets=encode_residue_targets,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("in_memory", [False, True])
def test_adapter_keeps_exact_dtype_values_truncation_and_closes(residues, dtype, in_memory):
    query = query_namespace()
    baseline = ResidueEmbedDataset(str(residues), dtype=dtype, max_tokens=5, in_memory=in_memory)
    with pytest.raises(RuntimeError, match="sentinel"):
        with screening_io_adapters(query) as receipt:
            assert receipt["prefetch_batches"] == 1
            assert len(receipt["adapter_sha256"]) == 64
            fast = query.ResidueEmbedDataset(
                str(residues),
                dtype=dtype,
                max_tokens=5,
                in_memory=in_memory,
            )
            for key in reversed(baseline.keys):
                expected, actual = baseline[key], fast[key]
                torch.testing.assert_close(
                    actual["residue_embeddings"], expected["residue_embeddings"], rtol=0, atol=0
                )
            if not in_memory:
                retained = fast._vectors
                fast[0]
                assert fast._vectors is retained
            raise RuntimeError("sentinel")
    assert fast.file is None
    assert query.ResidueEmbedDataset is ResidueEmbedDataset
    assert query.encode_residue_targets is encode_residue_targets
    baseline.close()


def test_prefetch_is_bounded_ordered_and_off_main_thread():
    calls = []
    ready = threading.Event()

    class Reader:
        def __getitem__(self, key):
            calls.append((key, threading.get_ident()))
            if len(calls) == 4:
                ready.set()
            return key

    keys = [f"p{i}" for i in range(8)]
    with PrefetchedRows(Reader(), keys, 2) as rows:
        assert rows["p0"] == "p0"
        assert ready.wait(5)
        assert [key for key, _ in calls] == keys[:4]  # Current batch + ONE next batch.
        assert all(tid != threading.get_ident() for _, tid in calls)
        assert [rows[key] for key in keys[1:]] == keys[1:]
    assert rows.future is None and not rows.rows
    assert not any(t.is_alive() for t in rows.executor._threads)


def test_prefetch_propagates_reader_errors_and_rejects_wrong_order():
    class Reader:
        def __getitem__(self, key):
            raise OSError("NFS read failed")

    with PrefetchedRows(Reader(), ["p0"], 1) as rows:
        with pytest.raises(OSError, match="NFS read failed"):
            rows["p0"]
    with PrefetchedRows({"p0": 0, "p1": 1}, ["p0", "p1"], 1) as rows:
        with pytest.raises(ValueError, match="order mismatch"):
            rows["p1"]


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def encode_targets(
        self, residues, *, residue_padding_mask, score_residue_embeddings=None, **kwargs
    ):
        valid = (~residue_padding_mask).unsqueeze(-1)
        result = (residues * valid).sum(1) / valid.sum(1)
        if score_residue_embeddings is not None:
            result = result + (score_residue_embeddings * valid).sum(1) / valid.sum(1)
        return result


@pytest.mark.parametrize("with_scores", [False, True])
def test_exact_encoding_and_noncontiguous_chunk_resume(residues, tmp_path, with_scores):
    query = query_namespace()
    dataset = ResidueEmbedDataset(str(residues), dtype=torch.float16, max_tokens=5)
    keys = list(reversed(dataset.keys))  # Preserve supplied order, not file order.
    module = SimpleNamespace(model=Encoder())
    cache = {"status": "miss_encode", "cache_key": "test", "cache_path": str(tmp_path / "cache.pt")}
    expected = encode_residue_targets(
        module,
        dataset,
        keys,
        "cpu",
        2,
        False,
        cache_info=cache,
        score_dataset=dataset if with_scores else None,
    )
    chunks = sorted(Path(cache["chunk_dir"]).glob("chunk_*.pt"))
    before = {p: p.read_bytes() for p in chunks}
    chunks[1].unlink()
    chunks[-1].unlink()
    with screening_io_adapters(query):
        fast = query.ResidueEmbedDataset(str(residues), dtype=torch.float16, max_tokens=5)
        calls = []

        class TrackedReader:
            def __getitem__(self, key):
                calls.append(key)
                return fast[key]

        actual = query.encode_residue_targets(
            module,
            TrackedReader(),
            keys,
            "cpu",
            2,
            False,
            cache_info=cache,
            score_dataset=fast if with_scores else None,
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == keys[2:4] + keys[6:]
    for path in (chunks[0], chunks[2]):
        assert path.read_bytes() == before[path]
    # An entirely cached pass must not touch the residue store at all.
    actual = encode_prefetched_targets(module, {}, keys, "cpu", 2, False, cache_info=cache)
    assert torch.equal(actual, expected)
    dataset.close()


def test_corrupt_cached_chunk_is_not_silently_skipped(residues, tmp_path):
    dataset = ResidueEmbedDataset(str(residues))
    module = SimpleNamespace(model=Encoder())
    cache = {"status": "miss_encode", "cache_key": "test", "cache_path": str(tmp_path / "cache.pt")}
    encode_residue_targets(module, dataset, dataset.keys, "cpu", 2, False, cache_info=cache)
    path = next(Path(cache["chunk_dir"]).glob("chunk_*.pt"))
    payload = torch.load(path, weights_only=True)
    payload["target_keys_sha256"] = "wrong"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="provenance mismatch"):
        encode_prefetched_targets(module, dataset, dataset.keys, "cpu", 2, False, cache_info=cache)
    dataset.close()


def test_fast_reader_keeps_nonfinite_guard(residues):
    with h5py.File(residues.parent / "source0.h5", "r+") as handle:
        handle["vectors"][0, 0] = np.nan
    query = query_namespace()
    with screening_io_adapters(query):
        reader = query.ResidueEmbedDataset(str(residues), dtype=torch.float16)
        with pytest.raises(ValueError, match="non-finite"):
            query.encode_residue_targets(
                SimpleNamespace(model=Encoder()), reader, reader.keys, "cpu", 2, False
            )


def test_real_factorized_enzyme_tower_matches_exactly(tmp_path):
    source = Path(__file__).with_name("test_horizyn1_multimodal_training_smoke.py")
    spec = importlib.util.spec_from_file_location("refseq_io_fixture", source)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    _, module = fixture.make_multimodal_fixture(tmp_path, True)
    module.eval()
    path = str(tmp_path / "proteins.h5")
    baseline = ResidueEmbedDataset(path, dtype=torch.float16)
    keys = baseline.keys
    expected = encode_residue_targets(module, baseline, keys, "cpu", 2, False)
    query = query_namespace()
    with screening_io_adapters(query):
        reader = query.ResidueEmbedDataset(path, dtype=torch.float16)
        actual = query.encode_residue_targets(module, reader, keys, "cpu", 2, False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    baseline.close()
