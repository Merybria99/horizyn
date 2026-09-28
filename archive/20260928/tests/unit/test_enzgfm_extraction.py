import argparse
import importlib.util
import sys
from pathlib import Path

import h5py
import torch


def load_enzgfm_script():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "extract_enzgfm_residue_embeddings.py"
    )
    spec = importlib.util.spec_from_file_location(
        "extract_enzgfm_residue_embeddings",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_args(tmp_path, fasta, output):
    return argparse.Namespace(
        fasta=str(fasta),
        output=str(output),
        tmp_dir=None,
        model_location="EnzGFM-650M",
        enzgfm_repo=str(tmp_path / "EnzGFM"),
        tokenizer_location=str(tmp_path / "EsmTokenizer"),
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
        disable_fast_kernels=True,
        resume=False,
        force=True,
        cleanup_shards=False,
    )


def test_enzgfm_write_shard_preserves_one_vector_per_residue(tmp_path):
    enzgfm = load_enzgfm_script()
    fasta = tmp_path / "proteins.fasta"
    output = tmp_path / "enzgfm_residues.h5"
    fasta.write_text(">p0\nACD\n>p1\nEF\n")

    class FakeTokenizer:
        all_special_ids = [0, 1, 2]

        def __call__(self, sequences, add_special_tokens=True, padding=True, return_tensors="pt"):
            rows = []
            max_len = max(len(sequence) + 2 for sequence in sequences)
            for sequence in sequences:
                residue_ids = list(range(10, 10 + len(sequence)))
                ids = [0, *residue_ids, 2]
                rows.append(ids + [1] * (max_len - len(ids)))
            input_ids = torch.tensor(rows, dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": (input_ids != 1).to(torch.long),
            }

    class FakeModel:
        class Config:
            hidden_size = 6

        config = Config()

        def __call__(self, input_ids, attention_mask, use_cache, return_dict):
            batch, length = input_ids.shape
            hidden = torch.arange(batch * length * 6, dtype=torch.float32).reshape(batch, length, 6)
            return type("Output", (), {"last_hidden_state": hidden})

    enzgfm.load_model_and_tokenizer = lambda *args, **kwargs: (
        FakeModel(),
        FakeTokenizer(),
    )
    enzgfm.repository_revision = lambda repo_path: "test-revision"
    args = _fake_args(tmp_path, fasta, output)

    shard_path = enzgfm.write_shard(args)

    with h5py.File(shard_path, "r") as h5_file:
        assert h5_file["offsets"][:].tolist() == [0, 3, 5]
        assert h5_file["vectors"].shape == (5, 6)
        assert h5_file.attrs["embedding_model_type"] == "enzgfm"
        assert h5_file.attrs["embedding_backend"] == "official_deepbxm"
        assert h5_file.attrs["enzgfm_source_revision"] == "test-revision"


def test_enzgfm_merge_preserves_metadata(tmp_path):
    enzgfm = load_enzgfm_script()
    fasta = tmp_path / "proteins.fasta"
    output = tmp_path / "enzgfm_residues.h5"
    fasta.write_text(">p0\nACD\n")

    class FakeTokenizer:
        all_special_ids = [0, 1, 2]

        def __call__(self, sequences, add_special_tokens=True, padding=True, return_tensors="pt"):
            input_ids = torch.tensor([[0, 10, 11, 12, 2]], dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    class FakeModel:
        class Config:
            hidden_size = 2

        config = Config()

        def __call__(self, input_ids, attention_mask, use_cache, return_dict):
            return type(
                "Output",
                (),
                {"last_hidden_state": torch.ones((1, 5, 2), dtype=torch.float32)},
            )

    enzgfm.load_model_and_tokenizer = lambda *args, **kwargs: (
        FakeModel(),
        FakeTokenizer(),
    )
    enzgfm.repository_revision = lambda repo_path: "test-revision"
    args = _fake_args(tmp_path, fasta, output)
    enzgfm.write_shard(args)
    enzgfm.merge_enzgfm_shards(args)

    with h5py.File(output, "r") as h5_file:
        assert h5_file.attrs["embedding_model_type"] == "enzgfm"
        assert h5_file.attrs["model_name"] == "EnzGFM-650M"
        assert h5_file.attrs["enzgfm_source_revision"] == "test-revision"
        assert int(h5_file.attrs["residue_dim"]) == 2


def test_mamba_projection_guard_matches_mixed_precision_input_dtype():
    enzgfm = load_enzgfm_script()

    class EnzGFMMambaMixer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.out_proj = torch.nn.Linear(4, 3, bias=False).half()

        def forward(self, value):
            return self.out_proj(value.float())

    model = torch.nn.Sequential(EnzGFMMambaMixer())
    assert enzgfm.install_mamba_projection_dtype_guards(model) == 1

    result = model(torch.randn(2, 4, dtype=torch.float16))

    assert result.dtype == torch.float16
    assert result.shape == (2, 3)
