import argparse
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from horizyn.config import DotDict, validate_config


def load_esm2_script():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "extract_esm2_residue_embeddings.py"
    spec = importlib.util.spec_from_file_location("extract_esm2_residue_embeddings", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_esm2_sequence_normalization_and_truncation():
    esm2 = load_esm2_script()

    normalized = esm2.normalize_sequence("acduzob-*")
    truncated = esm2.truncate_sequence("A" * 2000, max_length=1022)

    assert normalized == "ACDXXXXXX"
    assert len(truncated) == 1022


def test_esm2_residue_token_slice_excludes_special_tokens():
    esm2 = load_esm2_script()

    class Tokenizer:
        all_special_ids = [0, 1, 2]

    hidden = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    input_ids = torch.tensor([0, 5, 6, 2, 1, 1])
    attention_mask = torch.tensor([1, 1, 1, 1, 0, 0])

    residues = esm2.extract_residue_tokens(
        hidden=hidden,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=Tokenizer(),
        sequence_length=2,
    )

    assert torch.equal(residues, hidden[[1, 2]])


def test_esm2_shard_merge_preserves_fasta_order_and_offsets(tmp_path):
    esm2 = load_esm2_script()
    output = tmp_path / "esm2_residues.h5"
    tmp_dir = tmp_path / "shards"
    tmp_dir.mkdir()
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">p0\nAA\n>p1\nCC\n>p2\nD\n")

    args = argparse.Namespace(
        output=str(output),
        tmp_dir=str(tmp_dir),
        world_size=2,
        force=True,
        cleanup_shards=False,
        compression="none",
        progress_every=1,
        fasta=str(fasta),
        model_name=esm2.DEFAULT_MODEL_NAME,
        max_sequence_length=1022,
        sequence_truncation="ends_center",
    )

    text_dtype = h5py.string_dtype("utf-8")
    shard0 = esm2.get_shard_path(args.output, args.tmp_dir, rank=0, world_size=2)
    with h5py.File(shard0, "w") as h5_file:
        h5_file.create_dataset("ids", data=np.array(["p0", "p2"], dtype=object), dtype=text_dtype)
        h5_file.create_dataset("indices", data=np.array([0, 2], dtype=np.int64))
        h5_file.create_dataset("offsets", data=np.array([0, 2, 3], dtype=np.int64))
        h5_file.create_dataset("vectors", data=np.array([[0.0], [1.0], [2.0]], dtype=np.float16))

    shard1 = esm2.get_shard_path(args.output, args.tmp_dir, rank=1, world_size=2)
    with h5py.File(shard1, "w") as h5_file:
        h5_file.create_dataset("ids", data=np.array(["p1"], dtype=object), dtype=text_dtype)
        h5_file.create_dataset("indices", data=np.array([1], dtype=np.int64))
        h5_file.create_dataset("offsets", data=np.array([0, 2], dtype=np.int64))
        h5_file.create_dataset("vectors", data=np.array([[10.0], [11.0]], dtype=np.float16))

    esm2.merge_shards(args)

    with h5py.File(output, "r") as h5_file:
        ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in h5_file["ids"][:]
        ]
        assert ids == ["p0", "p1", "p2"]
        assert h5_file["offsets"][:].tolist() == [0, 2, 4, 5]
        assert h5_file["vectors"][:].astype(np.float32).squeeze(1).tolist() == [
            0.0,
            1.0,
            10.0,
            11.0,
            2.0,
        ]
        assert h5_file.attrs["embedding_model_type"] == "esm2"
        assert int(h5_file.attrs["residue_dim"]) == 1
        assert int(h5_file.attrs["max_sequence_length"]) == 1022


def test_esm2_write_shard_with_fake_model(tmp_path):
    esm2 = load_esm2_script()
    fasta = tmp_path / "proteins.fasta"
    output = tmp_path / "esm2_fake.h5"
    fasta.write_text(">p0\nACD\n>p1\nEF\n")

    class FakeTokenizer:
        all_special_ids = [0, 1, 2]

        def __call__(self, sequences, add_special_tokens=True, padding=True, return_tensors="pt"):
            rows = []
            max_len = max(len(sequence) + 2 for sequence in sequences)
            for sequence in sequences:
                residue_ids = list(range(10, 10 + len(sequence)))
                ids = [0, *residue_ids, 2]
                ids = ids + [1] * (max_len - len(ids))
                rows.append(ids)
            input_ids = torch.tensor(rows, dtype=torch.long)
            attention_mask = (input_ids != 1).to(torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class FakeModel:
        class Config:
            hidden_size = 4

        config = Config()

        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, input_ids, attention_mask):
            batch, seq_len = input_ids.shape
            hidden = torch.arange(batch * seq_len * 4, dtype=torch.float32).reshape(
                batch,
                seq_len,
                4,
            )
            return type("Output", (), {"last_hidden_state": hidden})

    esm2.load_model_and_tokenizer = lambda model_name, device, dtype_name: (
        FakeModel(),
        FakeTokenizer(),
    )
    args = argparse.Namespace(
        fasta=str(fasta),
        output=str(output),
        tmp_dir=None,
        model_name=esm2.DEFAULT_MODEL_NAME,
        rank=0,
        world_size=1,
        device="cpu",
        batch_size=2,
        max_tokens_per_batch=16,
        max_sequence_length=1022,
        sequence_truncation="ends_center",
        dtype="float32",
        compression="none",
        progress_every=1,
        resume=False,
        force=True,
    )

    shard_path = esm2.write_shard(args)

    with h5py.File(shard_path, "r") as h5_file:
        ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in h5_file["ids"][:]
        ]
        assert ids == ["p0", "p1"]
        assert h5_file["offsets"][:].tolist() == [0, 3, 5]
        assert h5_file["vectors"].shape == (5, 4)
        assert h5_file.attrs["embedding_model_type"] == "esm2"
        assert int(h5_file.attrs["residue_dim"]) == 4


def test_esm2_attention_pooling_config_validation_accepts_1280_dim():
    config = DotDict(
        {
            "data": {
                "train_pairs_path": "data/train_pairs.csv",
                "test_pairs_path": "data/test_pairs.csv",
                "train_reactions_path": "data/train_rxns.csv",
                "test_reactions_path": "data/test_rxns.csv",
                "protein_residue_embeds_path": "data/prots_esm2_650m_residue.h5",
                "residue_dim": 1280,
                "max_protein_tokens": 1022,
                "protein_truncation": "ends_center",
            },
            "model": {
                "name": "ProteinPooledDualModel",
                "pooling": "attention",
                "query_encoder_dims": [2048, 512],
                "target_encoder_dims": [1280, 512],
                "embedding_dim": 512,
            },
            "training": {"max_epochs": 1},
        }
    )

    validate_config(config)
