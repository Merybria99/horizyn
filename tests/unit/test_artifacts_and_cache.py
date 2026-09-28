import json
import os
import socket
import subprocess
from pathlib import Path

import pytest
import torch

from horizyn.artifacts import (
    ArtifactManifestV2,
    current_git_revision,
    sha256_file,
    sha256_strings,
    write_manifest,
)
from horizyn.benchmarks.retrieval import (
    _target_cache_base_metadata,
    _target_cache_paths,
    encode_residue_targets,
    prepare_target_embedding_cache,
    release_target_embedding_cache_lock,
    write_target_embedding_cache,
)


class _ResidueMapping:
    def __init__(self):
        self.rows = {
            "p1": torch.tensor([[1.0, 0.0]]),
            "p2": torch.tensor([[2.0, 0.0], [2.0, 0.0]]),
            "p3": torch.tensor([[3.0, 0.0]]),
            "p4": torch.tensor([[4.0, 0.0], [4.0, 0.0]]),
            "p5": torch.tensor([[5.0, 0.0]]),
        }

    def __getitem__(self, key):
        return {"residue_embeddings": self.rows[key]}


class _CountingResidueEncoder(torch.nn.Module):
    def __init__(self, *, non_finite=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0
        self.non_finite = non_finite

    def encode_targets(self, residues, *, residue_padding_mask, **_kwargs):
        self.calls += 1
        valid = (~residue_padding_mask).unsqueeze(-1)
        encoded = (residues * valid).sum(dim=1) / valid.sum(dim=1)
        if self.non_finite:
            encoded[0, 0] = float("nan")
        return encoded


class _ResidueModule:
    def __init__(self, model):
        self.model = model


def _write_source_files(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "model.ckpt"
    config = tmp_path / "config.yaml"
    embeddings = tmp_path / "embeddings.h5"
    checkpoint.write_bytes(b"checkpoint")
    config.write_text("model: {}\n", encoding="utf-8")
    embeddings.write_bytes(b"embedding-store")
    return checkpoint, config, embeddings


def _cache_metadata(tmp_path, *, direction):
    checkpoint, config, embeddings = _write_source_files(tmp_path)
    return _target_cache_base_metadata(
        kind="pooled",
        checkpoint=checkpoint,
        config_path=config,
        protein_embedding="prott5",
        score_protein_embedding="esm2",
        candidate_embedding_h5=embeddings,
        retrieval_direction=direction,
    )


def test_artifact_manifest_round_trip_and_atomic_write(tmp_path):
    digest = sha256_strings(["config"])
    manifest = ArtifactManifestV2(
        artifact_type="test",
        role="unit_test",
        direction="not_applicable",
        inputs={"source": "synthetic"},
        code_revision="abc123",
        config_sha256=digest,
    )
    path = tmp_path / "manifest.json"

    write_manifest(path, manifest)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert ArtifactManifestV2.from_dict(payload) == manifest
    assert not list(tmp_path.glob(".*.tmp"))


def test_artifact_manifest_rejects_non_hex_digest():
    manifest = ArtifactManifestV2(
        artifact_type="test",
        role="unit_test",
        direction="not_applicable",
        inputs={},
        code_revision="abc123",
        config_sha256="z" * 64,
    )

    with pytest.raises(ValueError, match="SHA-256"):
        manifest.validate()


def test_target_cache_identity_includes_retrieval_direction(tmp_path):
    r2e = _cache_metadata(tmp_path / "r2e", direction="reaction_to_enzyme")
    e2r = _cache_metadata(tmp_path / "e2r", direction="enzyme_to_reaction")
    keys = ["p1", "p2"]

    r2e_key = _target_cache_paths(tmp_path, r2e, keys)[0]
    e2r_key = _target_cache_paths(tmp_path, e2r, keys)[0]

    assert r2e_key != e2r_key


def test_target_cache_rejects_tampered_tensor(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    base_metadata = _cache_metadata(sources, direction="reaction_to_enzyme")
    cache_dir = tmp_path / "cache"
    keys = ["p1", "p2"]
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    cached, cache_info = prepare_target_embedding_cache(
        cache_dir,
        base_metadata,
        keys,
        device="cpu",
        store_on_device=False,
    )
    assert cached is None and cache_info["status"] == "miss_encode"
    write_target_embedding_cache(cache_info, base_metadata, keys, embeddings)

    cached, hit_info = prepare_target_embedding_cache(
        cache_dir,
        base_metadata,
        keys,
        device="cpu",
        store_on_device=False,
    )
    assert hit_info["status"] == "hit_exact"
    assert torch.equal(cached, embeddings)

    tensor_path = next(cache_dir.glob("*.pt"))
    with tensor_path.open("ab") as handle:
        handle.write(b"tampered")

    cached, miss_info = prepare_target_embedding_cache(
        cache_dir,
        base_metadata,
        keys,
        device="cpu",
        store_on_device=False,
    )
    try:
        assert cached is None
        assert miss_info["status"] == "miss_encode"
    finally:
        release_target_embedding_cache_lock(miss_info)


def test_target_cache_rejects_payload_sidecar_manifest_disagreement(tmp_path):
    base_metadata = _cache_metadata(tmp_path / "sources", direction="reaction_to_enzyme")
    cache_dir = tmp_path / "cache"
    keys = ["p1", "p2"]
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    _, cache_info = prepare_target_embedding_cache(
        cache_dir, base_metadata, keys, device="cpu", store_on_device=False
    )
    write_target_embedding_cache(cache_info, base_metadata, keys, embeddings)
    tensor_path = next(cache_dir.glob("*.pt"))
    sidecar_path = next(cache_dir.glob("*.json"))
    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    payload["artifact_manifest"] = dict(payload["artifact_manifest"])
    payload["artifact_manifest"]["role"] = "untrusted_role"
    torch.save(payload, tensor_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["tensor_sha256"] = sha256_file(tensor_path)
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    cached, miss_info = prepare_target_embedding_cache(
        cache_dir, base_metadata, keys, device="cpu", store_on_device=False
    )
    try:
        assert cached is None
        assert miss_info["status"] == "miss_encode"
    finally:
        release_target_embedding_cache_lock(miss_info)


def test_target_cache_lock_wait_has_a_timeout(tmp_path):
    base_metadata = _cache_metadata(tmp_path / "sources", direction="reaction_to_enzyme")
    keys = ["p1"]
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    lock_path = _target_cache_paths(cache_dir, base_metadata, keys)[3]
    lock_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "created_at": 0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TimeoutError, match="Timed out"):
        prepare_target_embedding_cache(
            cache_dir,
            base_metadata,
            keys,
            device="cpu",
            store_on_device=False,
            lock_timeout_seconds=0.01,
        )


def test_target_cache_write_failure_releases_owned_lock(tmp_path):
    base_metadata = _cache_metadata(tmp_path / "sources", direction="reaction_to_enzyme")
    cache_dir = tmp_path / "cache"
    keys = ["p1"]
    _, cache_info = prepare_target_embedding_cache(
        cache_dir, base_metadata, keys, device="cpu", store_on_device=False
    )

    with pytest.raises(ValueError, match="non-finite"):
        write_target_embedding_cache(
            cache_info,
            base_metadata,
            keys,
            torch.tensor([[float("nan")]]),
        )

    assert not Path(cache_info["lock_path"]).exists()


def test_residue_target_encoding_resumes_completed_chunks(tmp_path):
    base_metadata = _cache_metadata(tmp_path / "sources", direction="reaction_to_enzyme")
    keys = ["p1", "p2", "p3", "p4", "p5"]
    _, cache_info = prepare_target_embedding_cache(
        tmp_path / "cache",
        base_metadata,
        keys,
        device="cpu",
        store_on_device=False,
    )
    dataset = _ResidueMapping()
    first_model = _CountingResidueEncoder()

    first = encode_residue_targets(
        _ResidueModule(first_model),
        dataset,
        keys,
        "cpu",
        batch_size=2,
        store_on_device=False,
        cache_info=cache_info,
    )
    chunk_dir = Path(cache_info["chunk_dir"])
    chunks = sorted(chunk_dir.glob("chunk_*.pt"))
    assert first_model.calls == 3
    assert len(chunks) == 3

    chunks[-1].unlink()
    resumed_model = _CountingResidueEncoder()
    resumed = encode_residue_targets(
        _ResidueModule(resumed_model),
        dataset,
        keys,
        "cpu",
        batch_size=2,
        store_on_device=False,
        cache_info=cache_info,
    )

    assert resumed_model.calls == 1
    assert torch.equal(resumed, first)
    write_target_embedding_cache(cache_info, base_metadata, keys, resumed)
    assert not chunk_dir.exists()
    assert not Path(cache_info["lock_path"]).exists()


def test_residue_target_encoding_rejects_non_finite_batch_before_chunk_write(tmp_path):
    base_metadata = _cache_metadata(tmp_path / "sources", direction="reaction_to_enzyme")
    keys = ["p1", "p2"]
    _, cache_info = prepare_target_embedding_cache(
        tmp_path / "cache",
        base_metadata,
        keys,
        device="cpu",
        store_on_device=False,
    )

    try:
        with pytest.raises(ValueError, match="p1"):
            encode_residue_targets(
                _ResidueModule(_CountingResidueEncoder(non_finite=True)),
                _ResidueMapping(),
                keys,
                "cpu",
                batch_size=2,
                store_on_device=False,
                cache_info=cache_info,
            )
        assert not list(Path(cache_info["chunk_dir"]).glob("chunk_*.pt"))
    finally:
        release_target_embedding_cache_lock(cache_info)


def test_residue_target_encoding_without_cache_preserves_model_dtype():
    model = _CountingResidueEncoder().half()

    encoded = encode_residue_targets(
        _ResidueModule(model),
        _ResidueMapping(),
        ["p1", "p2"],
        "cpu",
        batch_size=2,
        store_on_device=False,
    )

    assert encoded.dtype == torch.float16


def test_current_git_revision_distinguishes_tracked_dirty_state(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "baseline"], check=True)

    clean_revision = current_git_revision(tmp_path)
    source.write_text("value = 2\n", encoding="utf-8")
    dirty_revision = current_git_revision(tmp_path)

    assert "+dirty." not in clean_revision
    assert dirty_revision.startswith(clean_revision + "+dirty.")
