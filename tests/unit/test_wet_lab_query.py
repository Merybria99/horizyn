from types import SimpleNamespace

import torch

from wet_lab.query import (
    _load_fasta_sequences,
    bucket_candidate_keys_by_residue_length,
    candidate_pool_with_root_override,
    streaming_topk,
    validate_reaction_smiles,
)


def test_streaming_topk_matches_dense_cosine():
    query = torch.tensor([1.0, 1.0])
    targets = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
        ]
    )

    scores, indices = streaming_topk(
        query,
        targets,
        top_k=3,
        device="cpu",
        batch_size=2,
    )

    expected = torch.mv(
        torch.nn.functional.normalize(targets, dim=1),
        torch.nn.functional.normalize(query, dim=0),
    )
    expected_scores, expected_indices = torch.topk(expected, 3)
    assert torch.equal(indices, expected_indices)
    assert torch.allclose(scores, expected_scores)


def test_validate_reaction_smiles_reports_unassigned_stereochemistry():
    warnings = validate_reaction_smiles(
        "OCC(=O)[C@@H](O)[C@@H](O)C(O)CO>>OCC(=O)[C@@H](O)[C@H](O)C(O)CO"
    )

    assert any("unassigned stereocenters" in warning for warning in warnings)


def test_validate_reaction_smiles_rejects_non_reaction():
    try:
        validate_reaction_smiles("CCO")
    except ValueError as error:
        assert "reaction SMILES" in str(error)
    else:
        raise AssertionError("Expected non-reaction SMILES to be rejected")


def test_load_fasta_sequences_selects_requested_ids(tmp_path):
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(
        ">P1 first\nAAAA\n>unused\nCCCC\n>P2\nGG\nTT\n",
        encoding="utf-8",
    )

    assert _load_fasta_sequences(fasta, {"P1", "P2"}) == {
        "P1": "AAAA",
        "P2": "GGTT",
    }


def test_bucket_candidate_keys_by_residue_length_reduces_padding():
    dataset = SimpleNamespace(
        keys=["P1", "P2", "P3", "P4", "P5", "P6"],
        offsets=torch.tensor([0, 10, 110, 130, 210, 240, 310]),
        h5_indices=torch.arange(6),
        max_tokens=50,
    )

    ordered, stats = bucket_candidate_keys_by_residue_length(
        dataset,
        list(dataset.keys),
        batch_size=2,
        window_size=4,
    )

    assert ordered == ["P1", "P3", "P2", "P4", "P5", "P6"]
    assert stats["encoding_actual_token_count"] == 210
    assert stats["encoding_padded_token_count_before"] == 300
    assert stats["encoding_padded_token_count"] == 240


def test_candidate_pool_root_override_repoints_standard_artifacts(tmp_path):
    root = tmp_path / "refseq"
    root.mkdir()
    settings = {
        "ids": "old/ids.txt",
        "residue_embeddings": "old/residues.h5",
        "fasta": "old/proteins.fasta",
        "metadata_csv": "old/proteins.csv",
        "target_cache_dir": "cache/refseq",
    }

    updated = candidate_pool_with_root_override(settings, root)

    assert updated == {
        "ids": str(root / "candidate_ids_prott5_order.txt"),
        "residue_embeddings": str(root / "proteins_prott5_residue.h5"),
        "fasta": str(root / "proteins.fasta"),
        "metadata_csv": str(root / "proteins.csv"),
        "target_cache_dir": "cache/refseq",
    }
    assert settings["ids"] == "old/ids.txt"
