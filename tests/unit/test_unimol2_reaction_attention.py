import importlib.util
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from horizyn.config import DotDict, validate_config
from horizyn.datasets.reaction_hdf5 import UniMol2ReactionEmbedDataset
from horizyn.model import (
    HybridReactionEncoder,
    MLP,
    MoleculeSetAttentionPooling,
    MoleculeSetMeanPooling,
    MultimodalReactionAttentionEncoder,
    UniMol2ReactionAttentionEncoder,
)
from horizyn.datasets.base import BaseDataset
from horizyn.utils import dict_collate_fn, residue_collate_fn, unimol2_reaction_collate_fn
from horizyn.reaction_features import HybridReactionFeatureDataset, build_reaction_feature_dataset


def write_reaction_hdf5(path: Path):
    text_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset(
            "ids",
            data=np.array(["r1_f", "r1_r"], dtype=object),
            dtype=text_dtype,
        )
        h5_file.create_dataset(
            "reactant_vectors",
            data=np.array([[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32),
        )
        h5_file.create_dataset("reactant_offsets", data=np.array([0, 2, 3], dtype=np.int64))
        h5_file.create_dataset(
            "product_vectors",
            data=np.array([[4.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        )
        h5_file.create_dataset("product_offsets", data=np.array([0, 1, 3], dtype=np.int64))
        h5_file.attrs["reaction_representation"] = "unimol2_attention"
        h5_file.attrs["embedding_dim"] = 2


def write_vector_hdf5(path: Path, ids: list[str], vectors: np.ndarray):
    text_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset(
            "ids",
            data=np.array(ids, dtype=object),
            dtype=text_dtype,
        )
        h5_file.create_dataset("vectors", data=vectors.astype(np.float32))


def test_unimol2_reaction_hdf5_loader_and_reverse_swap(tmp_path):
    h5_path = tmp_path / "reactions.h5"
    write_reaction_hdf5(h5_path)

    dataset = UniMol2ReactionEmbedDataset(str(h5_path), expected_dim=2)

    assert len(dataset) == 2
    assert dataset.embedding_dim == 2
    assert torch.equal(dataset["r1_f"]["reactant_embeddings"], torch.tensor([[1.0, 0.0], [2.0, 0.0]]))
    assert torch.equal(dataset["r1_f"]["product_embeddings"], torch.tensor([[4.0, 0.0]]))
    assert torch.equal(dataset["r1_r"]["reactant_embeddings"], dataset["r1_f"]["product_embeddings"])
    assert torch.equal(dataset["r1_r"]["product_embeddings"], dataset["r1_f"]["reactant_embeddings"])


def test_unimol2_reaction_collate_padding_masks(tmp_path):
    h5_path = tmp_path / "reactions.h5"
    write_reaction_hdf5(h5_path)
    dataset = UniMol2ReactionEmbedDataset(str(h5_path), expected_dim=2)

    batch = unimol2_reaction_collate_fn([dataset["r1_f"], dataset["r1_r"]])

    assert batch["reactant_embeddings"].shape == (2, 2, 2)
    assert batch["product_embeddings"].shape == (2, 2, 2)
    assert batch["reactant_padding_mask"].tolist() == [[False, False], [False, True]]
    assert batch["product_padding_mask"].tolist() == [[False, True], [False, False]]


def test_hybrid_reaction_dataset_and_collate_keep_missing_unimol2(tmp_path):
    h5_path = tmp_path / "reactions.h5"
    write_reaction_hdf5(h5_path)
    unimol_dataset = UniMol2ReactionEmbedDataset(str(h5_path), expected_dim=2)
    fingerprint_dataset = BaseDataset(
        keys=["r1_f", "missing_f"],
        array_data=torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        use_key_to_idx=True,
    )
    hybrid = HybridReactionFeatureDataset(
        fingerprint_dataset=fingerprint_dataset,
        unimol2_dataset=unimol_dataset,
        unimol_dim=2,
    )

    present = hybrid["r1_f"]
    missing = hybrid["missing_f"]
    collated = unimol2_reaction_collate_fn([present, missing])

    assert present["has_unimol2"].item() is True
    assert missing["has_unimol2"].item() is False
    assert torch.equal(missing["reactant_embeddings"], torch.zeros(1, 2))
    assert collated["fingerprint"].shape == (2, 4)
    assert collated["has_unimol2"].tolist() == [True, False]
    assert collated["reactant_embeddings"].shape == (2, 2, 2)


def test_multimodal_reaction_dataset_and_collate(tmp_path):
    reaction_model_h5 = tmp_path / "reaction_model.h5"
    unimol_h5 = tmp_path / "unimol.h5"
    chiro_h5 = tmp_path / "chiro.h5"
    reactions_csv = tmp_path / "reactions.csv"

    write_vector_hdf5(
        reaction_model_h5,
        ["r1_f", "r1_r", "extra_f"],
        np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float32,
        ),
    )
    write_reaction_hdf5(unimol_h5)
    write_reaction_hdf5(chiro_h5)
    reactions_csv.write_text("reaction_id,reaction_smiles\nr1,CCO>>CC=O\n", encoding="utf-8")

    config = DotDict(
        {
            "data": {
                "reaction_representation": "multimodal_reaction_attention",
                "reaction_t5v2_embeds_path": str(reaction_model_h5),
                "reaction_unimol2_embeds_path": str(unimol_h5),
                "reaction_chiro_embeds_path": str(chiro_h5),
                "reaction_model_dim": 3,
                "reaction_unimol_dim": 2,
                "reaction_chiro_dim": 2,
            }
        }
    )
    dataset = build_reaction_feature_dataset(
        reactions_csv,
        config=config,
        bidirectional=True,
    )

    assert len(dataset) == 2
    sample = dataset["r1_f"]
    assert sample["reaction_embedding"].shape == (3,)
    assert sample["reactant_embeddings"].shape == (2, 2)
    assert sample["product_embeddings"].shape == (1, 2)
    assert sample["reactant_chirality_embeddings"].shape == (2, 2)

    batch = unimol2_reaction_collate_fn([dataset["r1_f"], dataset["r1_r"]])
    assert batch["reaction_embedding"].shape == (2, 3)
    assert batch["reactant_embeddings"].shape == (2, 2, 2)
    assert batch["reactant_chirality_embeddings"].shape == (2, 2, 2)
    assert batch["reactant_chirality_padding_mask"].shape == (2, 2)


def test_multimodal_reaction_dataset_allows_missing_chiro(tmp_path):
    reaction_model_h5 = tmp_path / "reaction_model.h5"
    unimol_h5 = tmp_path / "unimol.h5"
    chiro_h5 = tmp_path / "chiro.h5"
    reactions_csv = tmp_path / "reactions.csv"

    write_vector_hdf5(
        reaction_model_h5,
        ["r1_f", "r1_r", "r2_f", "r2_r"],
        np.arange(12, dtype=np.float32).reshape(4, 3),
    )
    write_reaction_hdf5(unimol_h5)
    write_reaction_hdf5(chiro_h5)
    reactions_csv.write_text(
        "reaction_id,reaction_smiles\nr1,CCO>>CC=O\nr2,CCN>>CC=N\n",
        encoding="utf-8",
    )

    config = DotDict(
        {
            "data": {
                "reaction_representation": "multimodal_reaction_attention",
                "reaction_t5v2_embeds_path": str(reaction_model_h5),
                "reaction_unimol2_embeds_path": str(unimol_h5),
                "reaction_chiro_embeds_path": str(chiro_h5),
                "reaction_model_dim": 3,
                "reaction_unimol_dim": 2,
                "reaction_chiro_dim": 2,
                "reaction_allow_missing_unimol2": True,
                "reaction_allow_missing_chiro": True,
            }
        }
    )
    dataset = build_reaction_feature_dataset(
        reactions_csv,
        config=config,
        bidirectional=True,
    )

    assert len(dataset) == 4
    present = dataset["r1_f"]
    missing = dataset["r2_f"]
    assert present["has_chirality"].item() is True
    assert missing["has_chirality"].item() is False
    assert torch.equal(missing["reactant_chirality_embeddings"], torch.zeros(1, 2))

    batch = unimol2_reaction_collate_fn([present, missing])
    assert batch["has_chirality"].tolist() == [True, False]
    assert batch["has_chiro"].tolist() == [True, False]
    assert batch["reactant_chirality_embeddings"].shape == (2, 2, 2)


def test_dict_collate_pads_top_level_unimol2_reaction_sets():
    batch = [
        {
            "query_id": "r1",
            "reactant_embeddings": torch.tensor([[1.0, 0.0]]),
            "product_embeddings": torch.tensor([[2.0, 0.0], [3.0, 0.0]]),
        },
        {
            "query_id": "r2",
            "reactant_embeddings": torch.tensor([[4.0, 0.0], [5.0, 0.0]]),
            "product_embeddings": torch.tensor([[6.0, 0.0]]),
        },
    ]

    collated = dict_collate_fn(batch)

    assert collated["query_id"] == ["r1", "r2"]
    assert collated["reactant_embeddings"].shape == (2, 2, 2)
    assert collated["product_embeddings"].shape == (2, 2, 2)
    assert collated["reactant_padding_mask"].tolist() == [[False, True], [False, False]]
    assert collated["product_padding_mask"].tolist() == [[False, False], [False, True]]


def test_residue_collate_wraps_top_level_unimol2_reactions_as_query_vec():
    batch = [
        {
            "query_id": "r1",
            "target_id": "p1",
            "reactant_embeddings": torch.tensor([[1.0, 0.0]]),
            "product_embeddings": torch.tensor([[2.0, 0.0], [3.0, 0.0]]),
            "residue_embeddings": torch.ones(2, 3),
        },
        {
            "query_id": "r2",
            "target_id": "p2",
            "reactant_embeddings": torch.tensor([[4.0, 0.0], [5.0, 0.0]]),
            "product_embeddings": torch.tensor([[6.0, 0.0]]),
            "residue_embeddings": torch.ones(1, 3) * 2,
        },
    ]

    collated = residue_collate_fn(batch)
    query_vec = collated["query_vec"]

    assert collated["query_id"] == ["r1", "r2"]
    assert collated["target_id"] == ["p1", "p2"]
    assert "reactant_embeddings" not in collated
    assert "product_embeddings" not in collated
    assert collated["residue_embeddings"].shape == (2, 2, 3)
    assert collated["residue_padding_mask"].tolist() == [[False, False], [False, True]]
    assert query_vec["reactant_embeddings"].shape == (2, 2, 2)
    assert query_vec["product_embeddings"].shape == (2, 2, 2)
    assert query_vec["reactant_padding_mask"].tolist() == [[False, True], [False, False]]
    assert query_vec["product_padding_mask"].tolist() == [[False, False], [False, True]]


def test_residue_collate_wraps_top_level_hybrid_reactions_as_query_vec():
    batch = [
        {
            "query_id": "r1",
            "target_id": "p1",
            "fingerprint": torch.ones(4),
            "has_unimol2": torch.tensor(True),
            "reaction_chemistry_vector": torch.ones(7),
            "has_reaction_chemistry": torch.tensor(True),
            "reaction_directional_vector": torch.ones(9),
            "has_reaction_directional": torch.tensor(True),
            "reactant_embeddings": torch.tensor([[1.0, 0.0]]),
            "product_embeddings": torch.tensor([[2.0, 0.0]]),
            "residue_embeddings": torch.ones(2, 3),
        },
        {
            "query_id": "r2",
            "target_id": "p2",
            "fingerprint": torch.ones(4) * 2,
            "has_unimol2": torch.tensor(False),
            "reaction_chemistry_vector": torch.ones(7) * 2,
            "has_reaction_chemistry": torch.tensor(False),
            "reaction_directional_vector": torch.ones(9) * 2,
            "has_reaction_directional": torch.tensor(False),
            "reactant_embeddings": torch.tensor([[0.0, 0.0]]),
            "product_embeddings": torch.tensor([[0.0, 0.0]]),
            "residue_embeddings": torch.ones(1, 3),
        },
    ]

    collated = residue_collate_fn(batch)
    query_vec = collated["query_vec"]

    assert "fingerprint" not in collated
    assert "has_unimol2" not in collated
    assert "reaction_directional_vector" not in collated
    assert "has_reaction_directional" not in collated
    assert query_vec["fingerprint"].shape == (2, 4)
    assert query_vec["has_unimol2"].tolist() == [True, False]
    assert query_vec["reaction_chemistry_vector"].shape == (2, 7)
    assert query_vec["has_reaction_chemistry"].tolist() == [True, False]
    assert query_vec["reaction_directional_vector"].shape == (2, 9)
    assert query_vec["has_reaction_directional"].tolist() == [True, False]
    assert query_vec["reactant_embeddings"].shape == (2, 1, 2)


def test_molecule_set_attention_pooling_matches_manual_softmax():
    pooling = MoleculeSetAttentionPooling(hidden_dim=2, attention_bias=False)
    with torch.no_grad():
        pooling.attention.weight.copy_(torch.tensor([[1.0, 0.0]]))

    molecules = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [5.0, 5.0]]])
    valid_mask = torch.tensor([[True, True, False]])

    pooled, weights = pooling(molecules, valid_mask, return_attention=True)

    manual_weights = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)
    manual_pooled = manual_weights[0] * molecules[0, 0] + manual_weights[1] * molecules[0, 1]
    assert torch.allclose(weights[0, :2], manual_weights)
    assert weights[0, 2].item() == 0.0
    assert torch.allclose(pooled[0], manual_pooled)


def test_molecule_set_mean_pooling_ignores_padding():
    pooling = MoleculeSetMeanPooling()
    molecules = torch.tensor([[[1.0, 0.0], [9.0, 9.0], [3.0, 4.0]]])
    valid_mask = torch.tensor([[True, False, True]])

    pooled, weights = pooling(molecules, valid_mask, return_attention=True)

    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))
    assert torch.allclose(weights, torch.tensor([[0.5, 0.0, 0.5]]))


def test_unimol2_reaction_attention_encoder_shape_and_normalization():
    encoder = UniMol2ReactionAttentionEncoder(
        input_dim=8,
        output_dim=3,
        num_layers=0,
        widths=[],
        unimol_dim=2,
    )
    reactants = torch.randn(4, 2, 2)
    products = torch.randn(4, 3, 2)
    reactant_mask = torch.tensor(
        [[False, False], [False, True], [False, False], [False, True]]
    )
    product_mask = torch.tensor(
        [[False, False, True], [False, False, False], [False, True, True], [False, False, True]]
    )

    encoded, attention = encoder(
        reactants,
        reactant_padding_mask=reactant_mask,
        product_embeddings=products,
        product_padding_mask=product_mask,
        return_attention=True,
    )

    assert encoded.shape == (4, 3)
    assert torch.allclose(encoded.norm(dim=1), torch.ones(4), atol=1e-5)
    assert attention["reactant"].shape == (4, 2)
    assert attention["product"].shape == (4, 3)
    assert torch.equal(attention["reactant"].masked_select(reactant_mask), torch.zeros(2))
    assert torch.equal(attention["product"].masked_select(product_mask), torch.zeros(4))


def test_hybrid_reaction_encoder_shape_and_normalization():
    encoder = HybridReactionEncoder(
        input_dim=4,
        output_dim=3,
        num_layers=0,
        widths=[],
        unimol_dim=2,
    )

    encoded, attention = encoder(
        fingerprint=torch.randn(2, 4),
        reactant_embeddings=torch.randn(2, 1, 2),
        product_embeddings=torch.randn(2, 2, 2),
        reactant_padding_mask=torch.zeros(2, 1, dtype=torch.bool),
        product_padding_mask=torch.tensor([[False, True], [False, False]]),
        has_unimol2=torch.tensor([True, False]),
        return_attention=True,
    )

    assert encoded.shape == (2, 3)
    assert torch.allclose(encoded.norm(dim=1), torch.ones(2), atol=1e-5)
    assert attention["reactant"].shape == (2, 1)
    assert attention["product"].shape == (2, 2)


def test_hybrid_reaction_encoder_mean_pooling_handles_missing_unimol2():
    encoder = HybridReactionEncoder(
        input_dim=4,
        output_dim=512,
        num_layers=0,
        widths=[],
        unimol_dim=2,
        reaction_pooling="mean",
    )

    encoded, weights = encoder(
        fingerprint=torch.randn(2, 4),
        reactant_embeddings=torch.tensor(
            [
                [[1.0, 0.0], [3.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
        product_embeddings=torch.tensor(
            [
                [[0.0, 2.0], [0.0, 4.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
        reactant_padding_mask=torch.tensor([[False, False], [False, True]]),
        product_padding_mask=torch.tensor([[False, False], [False, True]]),
        has_unimol2=torch.tensor([True, False]),
        return_attention=True,
    )

    assert encoded.shape == (2, 512)
    assert torch.allclose(encoded.norm(dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(weights["reactant"][0], torch.tensor([0.5, 0.5]))
    assert torch.allclose(weights["product"][0], torch.tensor([0.5, 0.5]))
    assert torch.allclose(weights["reactant"][1], torch.tensor([1.0, 0.0]))
    assert torch.allclose(weights["product"][1], torch.tensor([1.0, 0.0]))


def test_multimodal_reaction_attention_encoder_shape_and_attention():
    encoder = MultimodalReactionAttentionEncoder(
        input_dim=5,
        output_dim=3,
        num_layers=0,
        widths=[],
        reaction_model_dim=3,
        unimol_dim=2,
        chienn_dim=4,
        modality_dropout=0.25,
        modality_token_layer_norm=True,
    )
    encoder.eval()

    encoded, attention = encoder(
        reaction_embedding=torch.randn(4, 3),
        reactant_embeddings=torch.randn(4, 2, 2),
        product_embeddings=torch.randn(4, 3, 2),
        reactant_padding_mask=torch.tensor(
            [[False, False], [False, True], [False, False], [False, True]]
        ),
        product_padding_mask=torch.tensor(
            [[False, False, True], [False, False, False], [False, True, True], [False, False, True]]
        ),
        reactant_chirality_embeddings=torch.randn(4, 2, 4),
        product_chirality_embeddings=torch.randn(4, 3, 4),
        reactant_chirality_padding_mask=torch.tensor(
            [[False, False], [False, True], [False, False], [False, True]]
        ),
        product_chirality_padding_mask=torch.tensor(
            [[False, False, True], [False, False, False], [False, True, True], [False, False, True]]
        ),
        has_chirality=torch.tensor([True, False, True, False]),
        return_attention=True,
    )

    assert encoded.shape == (4, 3)
    assert torch.allclose(encoded.norm(dim=1), torch.ones(4), atol=1e-5)
    assert attention["modality"].shape == (4, 3)
    assert torch.allclose(attention["modality"].sum(dim=1), torch.ones(4), atol=1e-6)
    assert torch.equal(attention["modality_mask"][:, 2], torch.tensor([True, False, True, False]))
    assert torch.equal(attention["modality"][~attention["modality_mask"]], torch.zeros(2))
    assert torch.isfinite(attention["modality_entropy"]).all()
    assert attention["unimol2_reactant"].shape == (4, 2)
    assert attention["chiro_product"].shape == (4, 3)
    assert attention["chirality_product"].shape == (4, 3)


def test_multimodal_reaction_attention_encoder_uses_modality_mlps():
    encoder = MultimodalReactionAttentionEncoder(
        input_dim=5,
        output_dim=3,
        num_layers=0,
        widths=[],
        reaction_model_dim=3,
        unimol_dim=2,
        chienn_dim=4,
        modality_encoder_widths=[7, 6],
        modality_encoder_use_layer_norm=True,
        modality_encoder_dropout=0.1,
        modality_token_layer_norm=True,
    )
    encoder.eval()

    assert isinstance(encoder.reaction_projection, MLP)
    assert isinstance(encoder.unimol_projection, MLP)
    assert isinstance(encoder.chienn_projection, MLP)

    encoded, attention = encoder(
        reaction_embedding=torch.randn(2, 3),
        reactant_embeddings=torch.randn(2, 2, 2),
        product_embeddings=torch.randn(2, 2, 2),
        reactant_chirality_embeddings=torch.randn(2, 2, 4),
        product_chirality_embeddings=torch.randn(2, 2, 4),
        return_attention=True,
    )

    assert encoded.shape == (2, 3)
    assert torch.allclose(encoded.norm(dim=1), torch.ones(2), atol=1e-5)
    assert attention["modality"].shape == (2, 3)


def test_multimodal_molecule_set_mode_ignores_artificial_product_copy():
    encoder = MultimodalReactionAttentionEncoder(
        input_dim=4,
        output_dim=4,
        num_layers=1,
        widths=[8],
        reaction_model_dim=3,
        unimol_dim=2,
        chienn_dim=2,
        side_composition="molecule_set",
        modality_fusion="mean",
        output_projection="residual_mlp",
        residual_gate_init=0.1,
    )
    encoder.eval()
    inputs = {
        "reaction_embedding": torch.randn(2, 3),
        "reactant_embeddings": torch.randn(2, 2, 2),
        "product_embeddings": torch.randn(2, 3, 2),
        "reactant_chirality_embeddings": torch.randn(2, 2, 2),
        "product_chirality_embeddings": torch.randn(2, 3, 2),
        "return_attention": True,
    }

    first, attention = encoder(**inputs)
    inputs["product_embeddings"] = torch.randn(2, 5, 2) * 100.0
    inputs["product_chirality_embeddings"] = torch.randn(2, 5, 2) * 100.0
    second, _ = encoder(**inputs)

    assert torch.allclose(first, second, atol=1e-6)
    assert torch.allclose(attention["modality"], torch.full((2, 3), 1.0 / 3.0))
    assert attention["unimol2_product"] is None
    assert encoder.projection_encoder.gate.item() == pytest.approx(0.1)
    assert torch.allclose(first.norm(dim=1), torch.ones(2), atol=1e-5)


def load_unimol2_script():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "extract_unimol2_reaction_embeddings.py"
    )
    spec = importlib.util.spec_from_file_location("extract_unimol2_reaction_embeddings", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_unimol2_extractor_caches_molecules_with_mocked_unimol(monkeypatch):
    script = load_unimol2_script()
    calls = []

    class FakeUniMolRepr:
        def __init__(self, model_name, model_size, batch_size):
            self.model_name = model_name
            self.model_size = model_size
            self.batch_size = batch_size

        def get_repr(self, smiles, return_atomic_reprs=False):
            calls.extend(smiles)
            return {
                "cls_repr": np.asarray(
                    [[float(len(value)), float(index)] for index, value in enumerate(smiles)],
                    dtype=np.float32,
                )
            }

    fake_module = types.ModuleType("unimol_tools")
    fake_module.UniMolRepr = FakeUniMolRepr
    monkeypatch.setitem(sys.modules, "unimol_tools", fake_module)

    embeddings = script.encode_molecules(
        molecules=["CCO", "N", "CCO"],
        model_name="unimolv2",
        model_size="84m",
        batch_size=2,
        skip_invalid_molecules=False,
    )

    assert calls == ["CCO", "N"]
    assert set(embeddings) == {"CCO", "N"}
    assert embeddings["N"].shape == (2,)


def test_unimol2_config_validation_accepts_attention_query():
    config = DotDict(
        {
            "data": {
                "train_pairs_path": "train_pairs.csv",
                "test_pairs_path": "test_pairs.csv",
                "train_reactions_path": "train_rxns.csv",
                "test_reactions_path": "test_rxns.csv",
                "protein_residue_embeds_path": "prots.h5",
                "reaction_representation": "unimol2_attention",
                "reaction_embeds_path": "rxns.h5",
                "reaction_unimol_dim": 2,
                "residue_dim": 4,
            },
            "model": {
                "query_encoder_type": "unimol2_reaction_attention",
                "query_encoder_dims": [8, 3],
                "target_encoder_dims": [4, 3],
                "embedding_dim": 3,
            },
            "training": {"max_epochs": 1},
        }
    )

    validate_config(config)


def test_hybrid_config_validation_accepts_hybrid_query():
    config = DotDict(
        {
            "data": {
                "train_pairs_path": "train_pairs.csv",
                "test_pairs_path": "test_pairs.csv",
                "train_reactions_path": "train_rxns.csv",
                "test_reactions_path": "test_rxns.csv",
                "protein_residue_embeds_path": "prots.h5",
                "reaction_representation": "hybrid_fingerprint_unimol2",
                "reaction_embeds_path": "rxns.h5",
                "reaction_unimol_dim": 2,
                "rdkit_fp_dim": 2,
                "drfp_dim": 2,
                "residue_dim": 4,
            },
            "model": {
                "query_encoder_type": "hybrid_reaction",
                "reaction_pooling": "mean",
                "query_encoder_dims": [4, 3],
                "target_encoder_dims": [4, 3],
                "embedding_dim": 3,
            },
            "training": {"max_epochs": 1},
        }
    )

    validate_config(config)


def test_multimodal_config_validation_accepts_multimodal_query():
    config = DotDict(
        {
            "data": {
                "train_pairs_path": "train_pairs.csv",
                "test_pairs_path": "test_pairs.csv",
                "train_reactions_path": "train_rxns.csv",
                "test_reactions_path": "test_rxns.csv",
                "protein_residue_embeds_path": "prots.h5",
                "reaction_representation": "multimodal_reaction_attention",
                "reaction_t5v2_embeds_path": "reaction_model.h5",
                "reaction_unimol2_embeds_path": "unimol.h5",
                "reaction_chiro_embeds_path": "chiro.h5",
                "reaction_model_dim": 3,
                "reaction_unimol_dim": 2,
                "reaction_chiro_dim": 4,
                "residue_dim": 4,
            },
            "model": {
                "query_encoder_type": "multimodal_reaction_attention",
                "reaction_pooling": "attention",
                "query_encoder_dims": [5, 3],
                "target_encoder_dims": [4, 3],
                "embedding_dim": 3,
            },
            "training": {"max_epochs": 1},
        }
    )

    validate_config(config)


def test_hybrid_config_validation_rejects_invalid_reaction_pooling():
    config = DotDict(
        {
            "data": {
                "train_pairs_path": "train_pairs.csv",
                "test_pairs_path": "test_pairs.csv",
                "train_reactions_path": "train_rxns.csv",
                "test_reactions_path": "test_rxns.csv",
                "protein_residue_embeds_path": "prots.h5",
                "reaction_representation": "hybrid_fingerprint_unimol2",
                "reaction_embeds_path": "rxns.h5",
                "reaction_unimol_dim": 2,
                "rdkit_fp_dim": 2,
                "drfp_dim": 2,
                "residue_dim": 4,
            },
            "model": {
                "query_encoder_type": "hybrid_reaction",
                "reaction_pooling": "max",
                "query_encoder_dims": [4, 3],
                "target_encoder_dims": [4, 3],
                "embedding_dim": 3,
            },
            "training": {"max_epochs": 1},
        }
    )

    with pytest.raises(ValueError, match="model.reaction_pooling"):
        validate_config(config)


def test_multimodal_encoder_runs_without_reaction_model_token():
    encoder = MultimodalReactionAttentionEncoder(
        input_dim=4,
        output_dim=4,
        num_layers=0,
        widths=[],
        reaction_model_dim=3,
        unimol_dim=2,
        chienn_dim=2,
        use_reaction_model=False,
        use_chienn=True,
        side_composition="molecule_set",
    )
    encoded, attention = encoder(
        reactant_embeddings=torch.randn(2, 3, 2),
        product_embeddings=torch.randn(2, 3, 2),
        reactant_chirality_embeddings=torch.randn(2, 3, 2),
        product_chirality_embeddings=torch.randn(2, 3, 2),
        return_attention=True,
    )

    assert encoded.shape == (2, 4)
    assert attention["modality_names"] == ("unimol2", "chiro")
    assert attention["modality"].shape == (2, 2)


def test_multimodal_dataset_does_not_require_reaction_model_hdf5(tmp_path):
    unimol_h5 = tmp_path / "unimol.h5"
    chiro_h5 = tmp_path / "chiro.h5"
    reactions_csv = tmp_path / "reactions.csv"
    write_reaction_hdf5(unimol_h5)
    write_reaction_hdf5(chiro_h5)
    reactions_csv.write_text("reaction_id,reaction_smiles\nr1,CCO>>CC=O\n", encoding="utf-8")
    config = DotDict(
        {
            "data": {
                "reaction_representation": "multimodal_reaction_attention",
                "reaction_use_model": False,
                "reaction_unimol2_embeds_path": str(unimol_h5),
                "reaction_chiro_embeds_path": str(chiro_h5),
                "reaction_unimol_dim": 2,
                "reaction_chiro_dim": 2,
            }
        }
    )

    dataset = build_reaction_feature_dataset(reactions_csv, config, bidirectional=True)

    assert len(dataset) == 2
    assert "reaction_embedding" not in dataset["r1_f"]


def test_factorized_reaction_fusion_renormalizes_missing_blocks():
    encoder = MultimodalReactionAttentionEncoder(
        input_dim=4,
        output_dim=4,
        num_layers=0,
        widths=[],
        reaction_model_dim=3,
        unimol_dim=2,
        chienn_dim=2,
        side_composition="molecule_set",
        modality_fusion="factorized_concat",
        factorized_dims={"reaction_model": 1, "unimol2": 2, "chiro": 1},
        factorized_weights={"reaction_model": 0.25, "unimol2": 0.5, "chiro": 0.25},
        output_projection="identity",
    )
    encoder.eval()
    encoded, attention = encoder(
        reaction_embedding=torch.randn(2, 3),
        reactant_embeddings=torch.randn(2, 2, 2),
        product_embeddings=torch.randn(2, 2, 2),
        reactant_chirality_embeddings=torch.randn(2, 2, 2),
        product_chirality_embeddings=torch.randn(2, 2, 2),
        has_chiro=torch.tensor([False, True]),
        return_attention=True,
    )

    assert torch.allclose(
        attention["modality"][0],
        torch.tensor([1.0 / 3.0, 2.0 / 3.0, 0.0]),
        atol=1e-6,
    )
    assert torch.allclose(attention["modality"][1], torch.tensor([0.25, 0.5, 0.25]))
    assert torch.allclose(encoded.norm(dim=1), torch.ones(2), atol=1e-6)


def test_factorized_residual_projection_reports_geometry_diagnostics():
    encoder = MultimodalReactionAttentionEncoder(
        input_dim=4,
        output_dim=4,
        num_layers=1,
        widths=[8],
        reaction_model_dim=3,
        unimol_dim=2,
        chienn_dim=2,
        side_composition="molecule_set",
        modality_fusion="factorized_concat",
        factorized_dims={"reaction_model": 1, "unimol2": 2, "chiro": 1},
        factorized_weights={"reaction_model": 0.25, "unimol2": 0.5, "chiro": 0.25},
        output_projection="residual_mlp",
        residual_gate_init=0.1,
    )
    encoded, details = encoder(
        reaction_embedding=torch.randn(2, 3),
        reactant_embeddings=torch.randn(2, 2, 2),
        product_embeddings=torch.randn(2, 2, 2),
        reactant_chirality_embeddings=torch.randn(2, 2, 2),
        product_chirality_embeddings=torch.randn(2, 2, 2),
        return_attention=True,
    )

    assert encoded.shape == (2, 4)
    assert torch.allclose(details["residual_gate"], torch.full((2,), 0.1))
    assert details["residual_base_cosine"].shape == (2,)
    assert details["residual_base_norm_ratio"].shape == (2,)
    assert torch.all(details["residual_base_norm_ratio"] >= 0)

    (1.0 - details["residual_base_cosine"]).mean().backward()
    assert encoder.projection_encoder.raw_gate.grad is not None


def test_prior_bounded_attention_starts_at_prior_and_renormalizes_missing_modalities():
    encoder = MultimodalReactionAttentionEncoder(
        input_dim=4,
        output_dim=4,
        num_layers=0,
        widths=[],
        reaction_model_dim=3,
        unimol_dim=2,
        chienn_dim=2,
        side_composition="molecule_set",
        modality_fusion="prior_bounded_attention",
        attention_prior_weights={"reaction_model": 0.25, "unimol2": 0.5, "chiro": 0.25},
        attention_adaptation_strength=0.4,
        output_projection="identity",
    )
    encoder.eval()
    encoded, attention = encoder(
        reaction_embedding=torch.randn(2, 3),
        reactant_embeddings=torch.randn(2, 2, 2),
        product_embeddings=torch.randn(2, 2, 2),
        reactant_chirality_embeddings=torch.randn(2, 2, 2),
        product_chirality_embeddings=torch.randn(2, 2, 2),
        has_chiro=torch.tensor([False, True]),
        return_attention=True,
    )

    expected = torch.tensor([[1.0 / 3.0, 2.0 / 3.0, 0.0], [0.25, 0.5, 0.25]])
    assert torch.allclose(attention["modality_prior"], expected, atol=1e-6)
    assert torch.allclose(attention["modality_adaptive"], expected, atol=1e-6)
    assert torch.allclose(attention["modality"], expected, atol=1e-6)
    assert torch.allclose(encoded.norm(dim=1), torch.ones(2), atol=1e-6)


def test_attention_prior_is_checkpoint_compatible_derived_state():
    kwargs = {
        "input_dim": 4,
        "output_dim": 4,
        "num_layers": 0,
        "widths": [],
        "reaction_model_dim": 3,
        "unimol_dim": 2,
        "chienn_dim": 2,
        "side_composition": "molecule_set",
        "modality_fusion": "prior_bounded_attention",
        "attention_prior_weights": {
            "reaction_model": 0.25,
            "unimol2": 0.5,
            "chiro": 0.25,
        },
        "output_projection": "identity",
    }
    source = MultimodalReactionAttentionEncoder(**kwargs)
    state_dict = source.state_dict()
    assert "attention_prior_values" not in state_dict

    # Checkpoints created before the buffer became non-persistent may contain it.
    state_dict["attention_prior_values"] = torch.tensor([0.1, 0.1, 0.8])
    target = MultimodalReactionAttentionEncoder(**kwargs)
    target.load_state_dict(state_dict, strict=True)
    assert torch.allclose(
        target.attention_prior_values,
        torch.tensor([0.25, 0.5, 0.25]),
    )


def test_prior_bounded_attention_is_trainable_and_respects_bounds():
    encoder = MultimodalReactionAttentionEncoder(
        input_dim=4,
        output_dim=4,
        num_layers=0,
        widths=[],
        reaction_model_dim=3,
        unimol_dim=2,
        chienn_dim=2,
        side_composition="molecule_set",
        modality_fusion="prior_bounded_attention",
        attention_prior_weights={"reaction_model": 0.25, "unimol2": 0.5, "chiro": 0.25},
        attention_adaptation_strength=0.4,
        output_projection="identity",
    )
    inputs = {
        "reaction_embedding": torch.randn(2, 3),
        "reactant_embeddings": torch.randn(2, 2, 2),
        "product_embeddings": torch.randn(2, 2, 2),
        "reactant_chirality_embeddings": torch.randn(2, 2, 2),
        "product_chirality_embeddings": torch.randn(2, 2, 2),
    }
    _, attention = encoder(return_attention=True, **inputs)
    attention["modality"][:, 0].sum().backward()
    final_gate = encoder.modality_attention[-1]
    assert final_gate.weight.grad is not None
    assert final_gate.weight.grad.abs().sum() > 0

    with torch.no_grad():
        final_gate.weight.normal_(mean=0.0, std=100.0)
    _, attention = encoder(return_attention=True, **inputs)

    lower = 0.6 * attention["modality_prior"]
    upper = lower + 0.4
    assert torch.all(attention["modality"] >= lower - 1e-6)
    assert torch.all(attention["modality"] <= upper + 1e-6)


def test_modality_l2_normalization_preserves_f3_token_scale():
    encoder = MultimodalReactionAttentionEncoder(
        input_dim=4,
        output_dim=4,
        num_layers=0,
        widths=[],
        reaction_model_dim=3,
        unimol_dim=2,
        chienn_dim=2,
        side_composition="molecule_set",
        modality_l2_normalize=True,
        output_projection="identity",
    )
    _, attention = encoder(
        reaction_embedding=torch.randn(2, 3),
        reactant_embeddings=torch.randn(2, 2, 2),
        product_embeddings=torch.randn(2, 2, 2),
        reactant_chirality_embeddings=torch.randn(2, 2, 2),
        product_chirality_embeddings=torch.randn(2, 2, 2),
        return_attention=True,
    )

    assert torch.allclose(
        attention["modality_token_norms"],
        torch.full((2, 3), 2.0),
        atol=1e-5,
    )


def test_chemistry_dropout_only_masks_chemistry_during_training():
    encoder = MultimodalReactionAttentionEncoder(
        input_dim=4,
        output_dim=4,
        num_layers=0,
        widths=[],
        reaction_model_dim=3,
        unimol_dim=2,
        chienn_dim=2,
        reaction_chemistry_dim=5,
        use_reaction_chemistry=True,
        side_composition="molecule_set",
        chemistry_dropout=1.0,
        output_projection="identity",
    )
    inputs = {
        "reaction_embedding": torch.randn(2, 3),
        "reactant_embeddings": torch.randn(2, 2, 2),
        "product_embeddings": torch.randn(2, 2, 2),
        "reactant_chirality_embeddings": torch.randn(2, 2, 2),
        "product_chirality_embeddings": torch.randn(2, 2, 2),
        "reaction_chemistry_vector": torch.randn(2, 5),
    }
    encoder.train()
    _, train_attention = encoder(return_attention=True, **inputs)
    encoder.eval()
    _, eval_attention = encoder(return_attention=True, **inputs)

    chemistry_index = train_attention["modality_names"].index("reaction_chemistry")
    assert not train_attention["modality_mask"][:, chemistry_index].any()
    assert train_attention["modality_mask"][:, :chemistry_index].all()
    assert eval_attention["modality_mask"][:, chemistry_index].all()


def test_masked_directional_residual_is_exact_noop():
    encoder = MultimodalReactionAttentionEncoder(
        input_dim=4,
        output_dim=4,
        num_layers=0,
        widths=[],
        reaction_model_dim=3,
        unimol_dim=2,
        chienn_dim=2,
        reaction_directional_dim=5,
        use_reaction_directional=True,
        side_composition="molecule_set",
    )
    encoder.eval()
    common = {
        "reaction_embedding": torch.randn(2, 3),
        "reactant_embeddings": torch.randn(2, 2, 2),
        "product_embeddings": torch.randn(2, 2, 2),
        "reactant_chirality_embeddings": torch.randn(2, 2, 2),
        "product_chirality_embeddings": torch.randn(2, 2, 2),
        "has_reaction_directional": torch.zeros(2, dtype=torch.bool),
    }
    first = encoder(reaction_directional_vector=torch.randn(2, 5), **common)
    second = encoder(reaction_directional_vector=torch.randn(2, 5) * 100.0, **common)

    assert torch.equal(first, second)
