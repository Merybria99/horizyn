import fcntl
import json
from pathlib import Path

import pytest
import yaml

from scripts import repair_horizyn1_pilot_config as repair
from scripts.horizyn1_circe_v2_reaction_holdout import Pipeline, parse_args


@pytest.fixture
def failed_run(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("GPU_COUNT", "4")
    monkeypatch.delenv("CIRCE_PROFILE", raising=False)
    run = tmp_path / "run"
    pipeline = Pipeline(parse_args(["--profile", "base", "--run-root", str(run)]))
    (run / "pipeline.lock").touch()
    config = run / "pilot/train.yaml"
    pipeline.config(config, pilot=run / "pilot")
    old = yaml.safe_load(config.read_text())
    for section in ("data", "training"):
        old[section]["validation_retrieval_candidate_set"] = "all"
    config.write_text(yaml.safe_dump(old, sort_keys=False))
    artifact = run / "cached-feature.bin"
    artifact.write_bytes(b"immutable feature data")
    record = repair.signature(artifact)
    for name in repair.COMPLETE:
        value = {"status": "complete", "fingerprint": {
            "controller_sha256": repair.OLD_CONTROLLER_SHA, "inputs": [record], "parameters": {}},
            "outputs": [record], "inventory": [record]}
        if name == "preflight":
            value = {"status": "complete", "profile": "base", "required_inputs": [record]}
        (run / f"state/{name}.json").write_bytes(repair.payload(value))
    binding = {"dataset_protocol": "horizyn80_clustered_reaction_holdout_v1", "seed": 42,
               "validation_fraction": .05, "test_fraction": .05}
    (run / "state/dataset_protocol.json").write_bytes(repair.payload(binding))
    failed = {"status": "failed", "error": "pilot_training failed with exit code 1", "fingerprint": {
        "controller_sha256": repair.OLD_CONTROLLER_SHA,
        "inputs": [repair.signature(config), record], "parameters": {}}}
    (run / "state/pilot_training.json").write_bytes(repair.payload(failed))
    (run / "logs/pilot_training.log").write_text(repair.ERROR)
    return run


def snapshot(run):
    return {p.relative_to(run): (p.read_bytes(), repair.signature(p))
            for p in run.rglob("*") if p.is_file()}


def test_dry_run_checks_without_changing_files(failed_run):
    before = snapshot(failed_run)
    assert repair.repair(failed_run)["status"] == "verified"
    assert snapshot(failed_run) == before


def test_apply_preserves_artifacts_backs_up_and_keeps_failed_stage_failed(failed_run):
    before = snapshot(failed_run)
    result = repair.repair(failed_run, apply=True)
    assert result["status"] == "applied"
    bundle = Path(result["backup"])
    config = yaml.safe_load((failed_run / "pilot/train.yaml").read_text())
    for section in ("data", "training"):
        assert config[section]["validation_retrieval_candidate_set"] == "validation"
    assert config["training"]["validation_enabled"] is False
    for name in repair.COMPLETE | {"pilot_training"}:
        doc = json.loads((failed_run / f"state/{name}.json").read_text())
        assert doc["status"] == ("failed" if name == "pilot_training" else "complete")
        if "fingerprint" in doc:
            assert doc["fingerprint"]["controller_sha256"] == repair.sha(repair.CONTROLLER.read_bytes())
    for name, (content, sig) in before.items():
        if str(name).startswith("state/") or str(name) == "pilot/train.yaml":
            assert (bundle / "original" / name).read_bytes() == content
        else:
            assert (failed_run / name).read_bytes() == content
            assert repair.signature(failed_run / name) == sig
    documents = {p.name: json.loads(p.read_text()) for p in (failed_run / "state").glob("*.json")}
    repair.validate_artifacts(documents)
    after = snapshot(failed_run)
    assert repair.repair(failed_run, apply=True)["status"] == "already_applied"
    assert snapshot(failed_run) == after


@pytest.mark.parametrize("mutation,error", [
    ("artifact", "Saved artifact changed"),
    ("stage", "Expected completed"),
    ("checkpoint", "before any pilot checkpoint"),
    ("unrelated_failure", "before any pilot checkpoint"),
    ("config", "regenerated launch configuration"),
])
def test_repair_rejects_unrelated_changes(failed_run, mutation, error):
    if mutation == "artifact":
        (failed_run / "cached-feature.bin").write_bytes(b"other data")
    elif mutation == "stage":
        path = failed_run / "state/index.json"
        doc = json.loads(path.read_text())
        doc["status"] = "running"
        path.write_bytes(repair.payload(doc))
    elif mutation == "checkpoint":
        path = failed_run / "pilot/checkpoints/last.ckpt"
        path.parent.mkdir()
        path.write_bytes(b"trained weights")
    elif mutation == "unrelated_failure":
        (failed_run / "logs/pilot_training.log").write_text("CUDA out of memory")
    else:
        path = failed_run / "pilot/train.yaml"
        # Same byte length/inode so only regenerated-content validation catches it.
        path.write_text(path.read_text().replace("seed: 42", "seed: 43"))
    before = snapshot(failed_run)
    with pytest.raises(ValueError, match=error):
        repair.repair(failed_run, apply=True)
    assert snapshot(failed_run) == before


def test_config_timestamp_drift_requires_exact_regenerated_bytes(failed_run):
    import os
    config = failed_run / "pilot/train.yaml"
    old = config.stat()
    os.utime(config, ns=(old.st_atime_ns, old.st_mtime_ns + 1000000))
    assert repair.repair(failed_run, apply=True)["status"] == "applied"
    plan = json.loads((failed_run / "metadata_repairs/pilot_candidate_set_v1/plan.json").read_text())
    assert "timestamp drift" in plan["config_verification"]


def test_repair_refuses_live_controller_lock(failed_run):
    with (failed_run / "pipeline.lock").open("r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Pipeline is running"):
            repair.repair(failed_run, apply=True)


def test_repair_resumes_its_own_interrupted_publication(failed_run, monkeypatch):
    original_publish = repair.publish
    def interrupt(path, data):
        if Path(path) == failed_run / "state/pilot_training.json":
            raise OSError("simulated interruption")
        original_publish(path, data)
    monkeypatch.setattr(repair, "publish", interrupt)
    with pytest.raises(OSError, match="simulated"):
        repair.repair(failed_run, apply=True)
    monkeypatch.setattr(repair, "publish", original_publish)
    assert repair.repair(failed_run, apply=True)["status"] == "applied"
    docs = {p.name: json.loads(p.read_text()) for p in (failed_run / "state").glob("*.json")}
    repair.validate_artifacts(docs)


def test_repair_rejects_other_controller_edits(failed_run, tmp_path, monkeypatch):
    modified = tmp_path / "changed_controller.py"
    modified.write_bytes(repair.CONTROLLER.read_bytes() + b"\n# unrelated edit\n")
    monkeypatch.setattr(repair, "CONTROLLER", modified)
    with pytest.raises(ValueError, match="beyond the audited pilot fix"):
        repair.repair(failed_run, apply=True)


@pytest.mark.parametrize("profile", ["base", "h200"])
def test_generated_holdout_pilot_config_uses_supported_candidate_mode(tmp_path, monkeypatch, profile):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("GPU_COUNT", "4")
    pipeline = Pipeline(parse_args(["--profile", profile, "--run-root", str(tmp_path / "run")]))
    path = tmp_path / "pilot.yaml"
    pipeline.config(path, pilot=pipeline.run / "pilot")
    config = yaml.safe_load(path.read_text())
    assert config["training"]["validation_enabled"] is False
    assert config["training"]["validation_retrieval_metrics"] is False
    for section in ("data", "training"):
        assert config[section]["validation_retrieval_candidate_set"] == "validation"
