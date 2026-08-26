from argparse import Namespace
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from scripts import extract_prott5_residue_embeddings as extraction
from scripts.extract_prott5_residue_embeddings import get_shard_path, merge_shards


def _write_shard(path, rank, indices, ids, vectors):
    offsets = np.zeros(len(vectors) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(value) for value in vectors])
    concatenated = np.concatenate(vectors, axis=0)
    with h5py.File(path, "w") as handle:
        handle.attrs["model_name"] = "test"
        handle.attrs["residue_dim"] = 2
        handle.create_dataset("ids", data=np.asarray(ids, dtype=h5py.string_dtype("utf-8")))
        handle.create_dataset("indices", data=np.asarray(indices, dtype=np.int64))
        handle.create_dataset("offsets", data=offsets)
        handle.create_dataset("vectors", data=concatenated)


def test_blockwise_merge_restores_fasta_order(tmp_path):
    output = tmp_path / "merged.h5"
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    args = Namespace(
        output=str(output),
        tmp_dir=str(shard_dir),
        world_size=2,
        force=False,
        fasta=str(tmp_path / "proteins.fasta"),
        compression="none",
        progress_every=2,
        merge_proteins_per_chunk=3,
        merge_order="fasta",
        merge_storage="copy",
        merged_ids_output=None,
        cleanup_shards=False,
    )
    shard0 = get_shard_path(output, shard_dir, 0, 2)
    shard1 = get_shard_path(output, shard_dir, 1, 2)
    _write_shard(
        shard0,
        0,
        [0, 2],
        ["P0", "P2"],
        [
            np.asarray([[0, 0]], dtype=np.float16),
            np.asarray([[2, 0], [2, 1]], dtype=np.float16),
        ],
    )
    _write_shard(
        shard1,
        1,
        [1, 3],
        ["P1", "P3"],
        [
            np.asarray([[1, 0], [1, 1], [1, 2]], dtype=np.float16),
            np.asarray([[3, 0]], dtype=np.float16),
        ],
    )

    merge_shards(args)

    with h5py.File(output, "r") as handle:
        assert [value.decode() for value in handle["ids"][:]] == ["P0", "P1", "P2", "P3"]
        assert handle["offsets"][:].tolist() == [0, 1, 4, 6, 7]
        assert handle["vectors"][:].tolist() == [
            [0, 0],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 0],
            [2, 1],
            [3, 0],
        ]


def test_virtual_shard_order_merge_maps_without_copying_vectors(tmp_path):
    output = tmp_path / "virtual.h5"
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    merged_ids = tmp_path / "ids.txt"
    args = Namespace(
        output=str(output),
        tmp_dir=str(shard_dir),
        world_size=2,
        force=False,
        fasta=str(tmp_path / "proteins.fasta"),
        compression="none",
        progress_every=2,
        merge_proteins_per_chunk=3,
        merge_order="shard",
        merge_storage="virtual",
        merged_ids_output=str(merged_ids),
        cleanup_shards=False,
    )
    shard0 = get_shard_path(output, shard_dir, 0, 2)
    shard1 = get_shard_path(output, shard_dir, 1, 2)
    _write_shard(
        shard0,
        0,
        [0, 2],
        ["P0", "P2"],
        [
            np.asarray([[0, 0]], dtype=np.float16),
            np.asarray([[2, 0], [2, 1]], dtype=np.float16),
        ],
    )
    _write_shard(
        shard1,
        1,
        [1, 3],
        ["P1", "P3"],
        [
            np.asarray([[1, 0], [1, 1], [1, 2]], dtype=np.float16),
            np.asarray([[3, 0]], dtype=np.float16),
        ],
    )

    merge_shards(args)

    with h5py.File(output, "r") as handle:
        assert handle["vectors"].is_virtual
        assert [value.decode() for value in handle["ids"][:]] == ["P0", "P2", "P1", "P3"]
        assert handle["offsets"][:].tolist() == [0, 1, 3, 6, 7]
        assert handle["vectors"][:].tolist() == [
            [0, 0],
            [2, 0],
            [2, 1],
            [1, 0],
            [1, 1],
            [1, 2],
            [3, 0],
        ]
    assert merged_ids.read_text(encoding="utf-8") == "P0\nP2\nP1\nP3\n"


class _DummyTokenizer:
    def __call__(self, batch_text, **_kwargs):
        lengths = [max(text.count(" ") + 1, 1) for text in batch_text]
        width = max(lengths) + 1
        return {
            "input_ids": torch.ones((len(batch_text), width), dtype=torch.int64),
            "attention_mask": torch.ones((len(batch_text), width), dtype=torch.int64),
        }


class _DummyModel:
    def __init__(self, *, fail_on_call=None):
        self.config = SimpleNamespace(d_model=2)
        self.calls = 0
        self.fail_on_call = fail_on_call

    def __call__(self, *, input_ids, attention_mask):
        del attention_mask
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated interruption")
        hidden = torch.arange(
            input_ids.shape[0] * input_ids.shape[1] * 2,
            dtype=torch.float32,
        ).reshape(input_ids.shape[0], input_ids.shape[1], 2)
        return SimpleNamespace(last_hidden_state=hidden)


def _extraction_args(tmp_path):
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">P0\nAA\n>P1\nCC\n>P2\nDD\n>P3\nEE\n", encoding="utf-8")
    return Namespace(
        output=str(tmp_path / "merged.h5"),
        tmp_dir=str(tmp_path / "shards"),
        rank=0,
        world_size=1,
        resume=True,
        force=False,
        fasta=str(fasta),
        model_name="dummy-model",
        device="cpu",
        dtype="float16",
        compression="none",
        max_sequence_length=1024,
        sequence_truncation="ends_center",
        batch_size=2,
        max_tokens_per_batch=4,
        progress_every=2,
        checkpoint_every=2,
    )


def test_interrupted_partial_shard_resumes_from_committed_batch(tmp_path, monkeypatch):
    args = _extraction_args(tmp_path)
    interrupted_model = _DummyModel(fail_on_call=2)
    monkeypatch.setattr(
        extraction,
        "load_model_and_tokenizer",
        lambda *_args: (interrupted_model, _DummyTokenizer()),
    )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        extraction.write_shard(args)

    shard = get_shard_path(args.output, args.tmp_dir, 0, 1)
    partial = shard.with_suffix(".h5.partial")
    assert partial.is_file()
    with h5py.File(partial, "r") as handle:
        assert int(handle.attrs[extraction.PROCESSED_COUNT_ATTR]) == 2

    resumed_model = _DummyModel()
    monkeypatch.setattr(
        extraction,
        "load_model_and_tokenizer",
        lambda *_args: (resumed_model, _DummyTokenizer()),
    )
    extraction.write_shard(args)

    assert resumed_model.calls == 1
    assert shard.is_file()
    assert not partial.exists()
    with h5py.File(shard, "r") as handle:
        assert int(handle.attrs[extraction.PROCESSED_COUNT_ATTR]) == 4
        assert [value.decode() for value in handle["ids"][:]] == ["P0", "P1", "P2", "P3"]
