from argparse import Namespace
from pathlib import Path
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
    def __init__(self, *, fail_on_call=None, non_finite=False):
        self.config = SimpleNamespace(d_model=2)
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.non_finite = non_finite

    def __call__(self, *, input_ids, attention_mask):
        del attention_mask
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated interruption")
        hidden = torch.arange(
            input_ids.shape[0] * input_ids.shape[1] * 2,
            dtype=torch.float32,
        ).reshape(input_ids.shape[0], input_ids.shape[1], 2)
        if self.non_finite:
            hidden[0, 0, 0] = float("nan")
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


def test_extraction_rejects_non_finite_model_output_before_checkpoint(tmp_path, monkeypatch):
    args = _extraction_args(tmp_path)
    monkeypatch.setattr(
        extraction,
        "load_model_and_tokenizer",
        lambda *_args: (_DummyModel(non_finite=True), _DummyTokenizer()),
    )

    with pytest.raises(ValueError, match="P0"):
        extraction.write_shard(args)

    shard = get_shard_path(args.output, args.tmp_dir, 0, 1)
    partial = shard.with_suffix(".h5.partial")
    with h5py.File(partial, "r") as handle:
        assert int(handle.attrs[extraction.PROCESSED_COUNT_ATTR]) == 0


def test_padded_token_budget_includes_padding_and_eos():
    records = [extraction.ProteinRecord(i, str(i), "A" * length)
               for i, length in enumerate([1, 9, 2, 20])]
    batches = extraction.iter_batches(records, 8, 12, padded_token_budget=True)
    assert [[len(r.sequence) for r in batch] for batch in batches] == [[1], [9], [2], [20]]
    assert [r.index for batch in batches for r in batch] == list(range(4))
    assert all(len(batch) == 1 or len(batch) * (max(len(r.sequence) for r in batch) + 1) <= 12
               for batch in batches)


def test_fasta_iterator_preserves_ids_normalization_and_indices(tmp_path):
    fasta = tmp_path / "test.fasta"
    fasta.write_text(">p0 description\nAcUZ\n>P1\nDD\n")
    records = list(extraction.iter_fasta_records(fasta))
    assert records == extraction.parse_fasta(fasta)
    assert [(r.index, r.protein_id, r.sequence) for r in records] == [(0, "p0", "ACXX"), (1, "P1", "DD")]


@pytest.mark.parametrize("rank,world_size", [(0, 1), (0, 2), (1, 2)])
def test_length_sorted_shard_writes_embeddings_to_correct_protein(tmp_path, monkeypatch, rank, world_size):
    args = _extraction_args(tmp_path)
    args.rank, args.world_size = rank, world_size
    args.length_sort = args.padded_token_budget = True
    args.max_tokens_per_batch = 100
    Path(args.fasta).write_text(">P0\nAAAA\n>P1\nC\n>P2\nDDD\n>P3\nEE\n")

    class IdentityTokenizer:
        def __call__(self, texts, **kwargs):
            width = max(len(text.split()) for text in texts) + 1
            ids = torch.zeros(len(texts), width, dtype=torch.long)
            for i, text in enumerate(texts):
                ids[i, :len(text.split())] = torch.tensor([ord(x) for x in text.split()])
            return {"input_ids": ids, "attention_mask": (ids != 0).long()}

    class IdentityModel:
        config = SimpleNamespace(d_model=2)

        def __call__(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1).expand(-1, -1, 2))

    monkeypatch.setattr(extraction, "load_model_and_tokenizer", lambda *args: (IdentityModel(), IdentityTokenizer()))
    monkeypatch.setattr(extraction, "parse_fasta", lambda *args: pytest.fail("Worker must stream input, not materialize all ranks"))
    shard = extraction.write_shard(args)
    sequences = {"P0": "AAAA", "P1": "C", "P2": "DDD", "P3": "EE"}
    with h5py.File(shard, "r") as handle:
        ids = [x.decode() for x in handle["ids"][:]]
        assert ids == sorted([f"P{i}" for i in range(4) if i % world_size == rank], key=lambda p: len(sequences[p]))
        assert handle["indices"][:].tolist() == [int(p[1:]) for p in ids]
        offsets = handle["offsets"][:]
        for i, pid in enumerate(ids):
            values = handle["vectors"][offsets[i]:offsets[i + 1]]
            assert values.shape == (len(sequences[pid]), 2)
            assert np.all(values == ord(sequences[pid][0]))
