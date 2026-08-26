import argparse
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from horizyn.config import DotDict


def load_script(module_name: str, script_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stage1_training_infers_input_dim_from_residue_hdf5(tmp_path):
    train_sleec = load_script("train_sleec_stage1_for_test", "train_sleec_stage1.py")
    h5_path = tmp_path / "residues.h5"
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("vectors", data=np.zeros((5, 7), dtype=np.float32))
        h5_file.attrs["embedding_model_type"] = "toy"
        h5_file.attrs["model_name"] = "toy-model"

    config = DotDict({"model": {"input_dim": "auto"}})

    input_dim = train_sleec.resolve_model_input_dim(config, h5_path)
    metadata = train_sleec.read_residue_embedding_metadata(h5_path)

    assert input_dim == 7
    assert config.model.input_dim == 7
    assert metadata["residue_embedding_dim"] == 7
    assert metadata["embedding_model_type"] == "toy"
    assert metadata["model_name"] == "toy-model"


def test_esmc_hidden_state_selection_supports_final_and_indexed_layers():
    esmc = load_script("extract_esmc_residue_embeddings_for_test", "extract_esmc_residue_embeddings.py")
    hidden0 = torch.zeros(2, 4, 3)
    hidden1 = torch.ones(2, 4, 3)
    output = argparse.Namespace(hidden_states=(hidden0, hidden1))

    assert torch.equal(esmc.select_hidden_tensor(output, 0), hidden0)
    assert torch.equal(esmc.select_hidden_tensor(output, -1), hidden1)

    final_only = argparse.Namespace(last_hidden_state=hidden1)
    assert torch.equal(esmc.select_hidden_tensor(final_only, -1), hidden1)
    with pytest.raises(ValueError, match="hidden_states"):
        esmc.select_hidden_tensor(final_only, 1)


def test_esmc_embedding_attrs_mark_esmc_backend(tmp_path):
    esmc = load_script("extract_esmc_residue_embeddings_attrs_for_test", "extract_esmc_residue_embeddings.py")
    h5_path = tmp_path / "attrs.h5"
    args = argparse.Namespace(
        model_name="Biohub/ESMC-6B",
        cache_dir="/tmp/hf",
        fasta="proteins.fasta",
        max_sequence_length=1022,
        sequence_truncation="ends_center",
        hidden_layer=-1,
    )

    with h5py.File(h5_path, "w") as h5_file:
        esmc.write_embedding_attrs(h5_file, args, residue_dim=2560)

    with h5py.File(h5_path, "r") as h5_file:
        assert h5_file.attrs["embedding_model_type"] == "esmc"
        assert h5_file.attrs["embedding_backend"] == "biohub_huggingface_transformers"
        assert h5_file.attrs["model_name"] == "Biohub/ESMC-6B"
        assert int(h5_file.attrs["residue_dim"]) == 2560
