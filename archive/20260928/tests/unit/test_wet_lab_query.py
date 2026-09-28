import os
from types import SimpleNamespace

import pytest
import torch

from wet_lab.query import (
    _configure_cpu_threads,
    _configure_cuda_memory,
    _load_fasta_sequences,
    bucket_candidate_keys_by_residue_length,
    candidate_pool_with_root_override,
    streaming_topk,
    validate_reaction_smiles,
    reaction_smiles_for_model,
    resolve_reaction_input_policy,
)


def test_wrap_flag_does_not_turn_directional_training_into_participant_model(tmp_path):
    path = tmp_path / "training.csv"
    path.write_text("reaction_id,reaction_smiles\nq1,CCO>>CC=O\nq2,C>>C\n")
    model = SimpleNamespace(data={"normalize_molecule_sets_as_self_reactions": True,
                                  "train_reactions_path": str(path)})
    config = {"feature_generation": {}}
    policy = resolve_reaction_input_policy(config, model)
    assert policy["policy"] == "physical_reaction"
    assert policy["counts"]["directional_reactions"] == 1
    path.write_text("reaction_id,reaction_smiles\nq1,CCO.CC=O\nq2,C.O>>O.C\n")
    assert resolve_reaction_input_policy(config, model)["policy"] == "participant_self_reaction"
    path.write_text("reaction_id,reaction_smiles\nq1,CCO.CC=O\nq2,C>>O\n")
    with pytest.raises(ValueError, match="Ambiguous mixed"):
        resolve_reaction_input_policy(config, model)
    config["feature_generation"]["reaction_input_policy"] = "physical_reaction"
    assert resolve_reaction_input_policy(config, model)["source"] == "explicit_query_configuration"


def test_participant_representation_includes_products_and_is_idempotent():
    # Product identity and stereo survive the conversion; an existing S>>S
    # must not double its participants on a subsequent query/resume.
    physical = "[CH3:1][C@@H:2](O)CO>>[CH3:1][C@H:2](O)CO"
    encoded = reaction_smiles_for_model(physical, normalize_molecule_sets_as_self_reactions=True)
    left, right = encoded.split(">>")
    assert left == right
    assert len(left.split(".")) == 2
    assert "@" in left and ":1" not in left
    assert reaction_smiles_for_model(encoded, normalize_molecule_sets_as_self_reactions=True) == encoded
    assert reaction_smiles_for_model(physical, normalize_molecule_sets_as_self_reactions=False) == physical
    repeated = reaction_smiles_for_model("C.C>>O", normalize_molecule_sets_as_self_reactions=True)
    assert repeated == "C.C.O>>C.C.O"


def test_every_extractor_receives_training_representation_and_old_cache_is_preserved(tmp_path, monkeypatch):
    import csv
    import json
    from wet_lab import query

    calls = []
    monkeypatch.setattr(query, "_resolve_path", lambda value, **kwargs: tmp_path / str(value))
    monkeypatch.setattr(query, "_run_logged", lambda **kwargs: calls.append(kwargs))
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    (feature_dir / "unimol2.h5").write_bytes(b"old directed cache")
    (tmp_path / "feature_manifest.json").write_text(json.dumps({
        "reaction_id": "query", "reaction_smiles": "CCO>>CC=O"
    }))
    reaction_csv, _, manifest = query.prepare_reaction_features({
        "reaction": {"id": "query", "smiles": "CCO>>CC=O"},
        "feature_generation": {"reaction_t5_model": "model", "reaction_chemistry_schema": "schema", "cofactor_dictionary": "cofactors"},
    }, tmp_path, force=False, normalize_molecule_sets_as_self_reactions=True)
    assert len(calls) == 4
    for call in calls:
        command = call["command"]
        assert command[command.index("--reactions") + 1] == str(reaction_csv)
    with reaction_csv.open() as handle:
        assert next(csv.DictReader(handle))["reaction_smiles"] == "CC=O.CCO>>CC=O.CCO"
    with (tmp_path / "reaction.csv").open() as handle:
        assert next(csv.DictReader(handle))["reaction_smiles"] == "CCO>>CC=O"
    assert manifest["reaction_smiles"] == "CCO>>CC=O"
    assert manifest["reaction_feature_policy"] == "participant_self_reaction_v1"
    backups = list((tmp_path / "feature_history").glob("*/unimol2.h5"))
    assert len(backups) == 1 and backups[0].read_bytes() == b"old directed cache"


def test_cuda_memory_limit_applies_to_visible_device(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=80 * 1024**3),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: calls.append((fraction, str(device))),
    )
    _configure_cuda_memory({"device": "cuda:0", "cuda_memory_limit_gib": 16})
    assert calls == [(0.2, "cuda:0")]
    _configure_cuda_memory({"device": "cpu"})
    assert len(calls) == 1
    for invalid in (True, -1, 0, float("nan"), float("inf"), 81):
        with pytest.raises(ValueError):
            _configure_cuda_memory({"device": "cuda", "cuda_memory_limit_gib": invalid})
    assert len(calls) == 1


@pytest.mark.parametrize(
    "configured,current,expected",
    [("cuda", 0, 0), ("cuda", 2, 2), ("cuda:1", 0, 1), (None, 0, 0)],
)
def test_cuda_memory_limit_uses_real_torch_device_validation(
    monkeypatch, configured, current, expected
):
    # Keep the public PyTorch setter and its index validation real; only fake
    # driver/backend access so this regression can run without a GPU.
    calls = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=80 * 1024**3)
    )
    monkeypatch.setattr(torch.cuda.memory, "_lazy_init", lambda: None)
    monkeypatch.setattr(
        torch._C,
        "_cuda_setMemoryFraction",
        lambda fraction, device: calls.append((fraction, device)),
    )
    inference = {"cuda_memory_limit_gib": 16}
    if configured is not None:
        inference["device"] = configured
    _configure_cuda_memory(inference)
    assert calls == [(0.2, expected)]


def test_cpu_reaction_features_hide_gpu_from_all_extractors(tmp_path, monkeypatch):
    from wet_lab import query

    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(query, "_resolve_path", lambda value, **kwargs: tmp_path / str(value))
    monkeypatch.setattr(query, "_run_logged", lambda **kwargs: calls.append(kwargs))
    query.prepare_reaction_features(
        {
            "reaction": {"id": "test", "smiles": "C>>C"},
            "inference": {"device": "cuda"},
            "feature_generation": {
                "device": "cpu",
                "reaction_t5_model": "model",
                "reaction_chemistry_schema": "schema",
                "cofactor_dictionary": "cofactors",
            },
        },
        tmp_path,
        force=False,
    )
    assert len(calls) == 4
    for call in calls:
        assert call["environment"]["CUDA_VISIBLE_DEVICES"] == ""
        command = call["command"]
        if "--device" in command:
            assert command[command.index("--device") + 1] == "cpu"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_configure_cpu_threads_caps_torch_and_native_pools(monkeypatch):
    calls = {}
    monkeypatch.delenv("HORIZYN_CPU_THREADS", raising=False)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(torch, "set_num_threads", lambda value: calls.setdefault("intra", value))
    monkeypatch.setattr(
        torch,
        "set_num_interop_threads",
        lambda value: calls.setdefault("interop", value),
    )
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 96)
    monkeypatch.setattr(
        "wet_lab.query.threadpool_limits",
        lambda *, limits: calls.setdefault("native", limits),
    )

    _configure_cpu_threads({"cpu_threads": 16})

    assert calls == {"native": 16, "intra": 16, "interop": 4}
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert os.environ[variable] == "16"


def test_configure_cpu_threads_environment_overrides_config(monkeypatch):
    calls = {}
    monkeypatch.setenv("HORIZYN_CPU_THREADS", "3")
    monkeypatch.setattr(torch, "set_num_threads", lambda value: calls.setdefault("intra", value))
    monkeypatch.setattr(
        torch,
        "set_num_interop_threads",
        lambda value: calls.setdefault("interop", value),
    )
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 96)
    monkeypatch.setattr(
        "wet_lab.query.threadpool_limits",
        lambda *, limits: calls.setdefault("native", limits),
    )

    _configure_cpu_threads({"cpu_threads": 16})

    assert calls == {"native": 3, "intra": 3, "interop": 3}


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
