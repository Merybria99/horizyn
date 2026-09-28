from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import horizyn1_parallel_reactions as parallel


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2,3")
    monkeypatch.setenv("GPU_COUNT", "3")
    main = tmp_path / "main"
    (main / "state").mkdir(parents=True)
    (main / "data").mkdir()
    (main / "state/dataset_protocol.json").write_text(json.dumps({
        "dataset_protocol": parallel.holdout.PROTOCOL, "seed": 42,
        "validation_fraction": .05, "test_fraction": .05}))
    for name in ("reactions.csv", "train_rxns.csv"):
        (main / "data" / name).write_text("reaction_id,reaction_smiles\nr1,CC>>C=C\n")
    args, _ = parallel.parse_args(["extract", "--cache-dir", str(tmp_path / "cache"),
                                   "--run-root", str(main), "--profile", "h200"])
    parallel.bind_cache(args, create=True)
    return args


def test_cli_keeps_original_h200_resume_parameters(setup):
    args, rest = parallel.parse_args(["resume", "--run-root", setup.run_root,
        "--cache-dir", str(setup.cache_dir), "--profile", "h200",
        "--extraction-batch-size", "256", "--extraction-max-tokens", "65536",
        "--extraction-length-sort", "--extraction-padded-token-budget"])
    assert args.extraction_batch_size == 256
    assert args.extraction_max_tokens == 65536
    assert args.extraction_padded_token_budget
    assert args.stage == "all"
    assert "--cache-dir" not in rest


def test_binding_does_not_touch_source_and_rejects_changes(setup):
    source = Path(setup.run_root) / "state/dataset_protocol.json"
    original = parallel.base.signature(source)
    parallel.bind_cache(setup)
    assert parallel.base.signature(source) == original
    setup.seed = 99
    with pytest.raises(ValueError, match="protocol"):
        parallel.bind_cache(setup)


def test_binding_rejects_nested_or_unowned_caches(setup):
    setup.cache_dir = Path(setup.run_root) / "nested"
    with pytest.raises(ValueError, match="separate"):
        parallel.bind_cache(setup, create=True)
    setup.cache_dir = Path(setup.run_root).parent / "unowned"
    setup.cache_dir.mkdir()
    (setup.cache_dir / "foreign").write_text("keep")
    with pytest.raises(RuntimeError, match="unowned"):
        parallel.bind_cache(setup, create=True)
    assert (setup.cache_dir / "foreign").read_text() == "keep"


def test_plans_reuse_original_commands_and_training_scope(setup):
    values = parallel.plans(setup)
    assert set(values) == parallel.STAGES
    for name, value in values.items():
        assert value["fingerprint"]["controller_sha256"] == parallel.digest(Path(parallel.base.__file__))
        assert all(str(setup.cache_dir) in path for path in value["outputs"])
        assert value["fingerprint"]["parameters"]["dataset_protocol"] == parallel.holdout.PROTOCOL
        assert str(Path(setup.run_root) / "data/reactions.csv") in [
            item["path"] for item in value["fingerprint"]["inputs"]]
    chemistry_inputs = values["full_chemistry"]["fingerprint"]["inputs"]
    assert str(Path(setup.run_root) / "data/train_rxns.csv") in [p["path"] for p in chemistry_inputs]
    command = values["full_chiro"]["fingerprint"]["parameters"]["command"]
    assert command[command.index("--molecule-cache") + 1] == str(setup.cache_dir / "features/chiro_molecules.sqlite")


def test_worker_only_runs_selected_modality(setup, monkeypatch):
    setup.modality = "unimol2"
    calls = []
    monkeypatch.setattr(parallel.holdout.Pipeline, "step", lambda self, name, *a, **k: calls.append(name))
    runner = parallel.cache_runner(setup, parallel.WorkerPipeline)
    parallel.invoke_features(runner, Path(setup.run_root))
    assert calls == ["full_unimol2"]


def test_plan_fingerprint_matches_actual_existing_step(setup):
    expected = parallel.plans(setup)["full_reactiont5v2"]
    setup.modality = "reactiont5v2"
    runner = parallel.cache_runner(setup, parallel.WorkerPipeline)
    # Exercise the original step/action/rename path without allocating a GPU.
    def fake_command(command, name, **kwargs):
        target = Path(str(command[command.index("--output") + 1]))
        target.write_bytes(b"fake output")
    runner.command = fake_command
    parallel.invoke_features(runner, Path(setup.run_root))
    actual = json.loads((runner.state / "full_reactiont5v2.json").read_text())
    assert actual["fingerprint"] == expected["fingerprint"]
    assert [item["path"] for item in actual["outputs"]] == expected["outputs"]


def build_receipt(setup):
    name = "full_reactiont5v2"
    plan = parallel.plans(setup)[name]
    path = Path(plan["outputs"][0])
    path.write_bytes(b"feature contents")
    state = setup.cache_dir / "state"
    parallel.base.atomic_json(state / f"{name}.json", {
        "status": "complete", "fingerprint": plan["fingerprint"],
        "outputs": [parallel.base.signature(path)],
    })
    audit = {"status": "complete", "stage": parallel.base.signature(state / f"{name}.json"),
             "outputs": [{"signature": parallel.base.signature(path), "sha256": parallel.digest(path)}]}
    parallel.base.atomic_json(state / f"{name}.validated.json", audit)
    return name, plan, audit


def test_verification_rejects_mutated_outputs(setup):
    name, plan, audit = build_receipt(setup)
    assert parallel.verified_cache(setup, name, plan) == audit
    Path(plan["outputs"][0]).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="changed"):
        parallel.verified_cache(setup, name, plan)


def test_verification_rejects_wrong_settings(setup):
    name, plan, _ = build_receipt(setup)
    plan["fingerprint"]["parameters"]["direction"] = "wrong"
    with pytest.raises(RuntimeError, match="fingerprint"):
        parallel.verified_cache(setup, name, plan)


def test_atomic_import_idempotent_and_preserves_source(setup):
    _, plan, audit = build_receipt(setup)
    dest = Path(setup.run_root) / "features/reactiont5v2.h5"
    journal = Path(setup.run_root) / "state/external_imports/full_reactiont5v2.json"
    before = parallel.base.signature(Path(plan["outputs"][0]))
    parallel.publish_files([dest], audit, journal)
    signature = parallel.base.signature(dest)
    parallel.publish_files([dest], audit, journal)
    assert parallel.base.signature(dest) == signature
    assert parallel.base.signature(Path(plan["outputs"][0])) == before
    assert dest.read_bytes() == b"feature contents"


def test_import_refuses_foreign_or_changed_destination(setup):
    _, _, audit = build_receipt(setup)
    dest = Path(setup.run_root) / "foreign.h5"
    journal = Path(setup.run_root) / "journal.json"
    dest.write_bytes(b"keep")
    with pytest.raises(RuntimeError, match="unowned"):
        parallel.publish_files([dest], audit, journal)
    assert dest.read_bytes() == b"keep"


def test_import_retry_after_partial_publication(setup, monkeypatch):
    _, _, audit = build_receipt(setup)
    audit["outputs"].append(audit["outputs"][0])
    outputs = [Path(setup.run_root) / "a.h5", Path(setup.run_root) / "b.h5"]
    journal = Path(setup.run_root) / "journal.json"
    real_link = parallel.os.link
    def interrupted(source, target):
        if target == outputs[1]:
            raise OSError("simulated interruption")
        return real_link(source, target)
    monkeypatch.setattr(parallel.os, "link", interrupted)
    with pytest.raises(OSError):
        parallel.publish_files(outputs, audit, journal)
    assert outputs[0].exists() and not outputs[1].exists()
    first = parallel.base.signature(outputs[0])
    monkeypatch.setattr(parallel.os, "link", real_link)
    parallel.publish_files(outputs, audit, journal)
    assert parallel.base.signature(outputs[0]) == first
    assert outputs[1].read_bytes() == b"feature contents"


def test_lock_refuses_concurrent_owner(tmp_path):
    path = tmp_path / "lock"
    with parallel.exclusive(path):
        with pytest.raises(RuntimeError, match="Another process"):
            with parallel.exclusive(path):
                pass


def test_validation_checks_real_hdf5_ids_finiteness(setup):
    h5py = pytest.importorskip("h5py")
    path = setup.cache_dir / "test.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ids", data=[b"r1"])
        handle.create_dataset("vectors", data=np.zeros((1, 768), dtype=np.float16))
    assert parallel.validate_output("full_reactiont5v2", [path], Path(setup.run_root))["coverage"] == 1
    with h5py.File(path, "r+") as handle:
        handle["vectors"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        parallel.validate_output("full_reactiont5v2", [path], Path(setup.run_root))


def test_resume_import_keeps_original_stage_fingerprint(setup, monkeypatch):
    cache_plan = parallel.plans(setup)["full_reactiont5v2"]
    imports = []
    runner = parallel.ImportPipeline(setup)
    def import_fake(name, outputs):
        imports.append(name)
        for path in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"imported")
    monkeypatch.setattr(runner, "import_feature", import_fake)
    runner.reaction_features(Path(setup.run_root) / "data/reactions.csv", runner.features, "full")
    actual = json.loads((runner.state / "full_reactiont5v2.json").read_text())
    assert actual["fingerprint"] == cache_plan["fingerprint"]
    assert imports == ["full_reactiont5v2", "full_unimol2", "full_chiro"]
    runner.reaction_features(Path(setup.run_root) / "data/reactions.csv", runner.features, "full")
    assert len(imports) == 3


def test_real_validated_cache_handoff_all_modalities(setup, monkeypatch):
    h5py = pytest.importorskip("h5py")
    # Run original extraction actions with tiny stand-in encoder outputs, then
    # real finite/ID checks, SHA-256 checks, publication, and normal reuse gates.
    monkeypatch.setattr(parallel.holdout.Pipeline, "require_free_gpus", lambda self: None)
    def tiny_encoder(self, command, name, **kwargs):
        if name == "full_chemistry":
            target = Path(command[command.index("--out-dir") + 1])
            target.mkdir(parents=True, exist_ok=True)
            (target / "schema.json").write_text('{"train_fitted": true}')
            np.savez(target / "validation_reaction_set_features.npz",
                     ids=np.array(["r1"]), vectors=np.zeros((1, 617)), mask=np.ones(1))
            return
        target = Path(command[command.index("--output") + 1])
        with h5py.File(target, "w") as handle:
            handle.create_dataset("ids", data=[b"r1"])
            dim = 256 if name == "full_chiro" else 768
            if name == "full_reactiont5v2":
                handle.create_dataset("vectors", data=np.zeros((1, dim)))
            else:
                for side in ("reactant", "product"):
                    handle.create_dataset(f"{side}_vectors", data=np.zeros((1, dim)))
                    handle.create_dataset(f"{side}_offsets", data=[0, 1])
    monkeypatch.setattr(parallel.WorkerPipeline, "command", tiny_encoder)
    monkeypatch.setattr(parallel.signal, "signal", lambda *args: None)
    for modality in parallel.MODALITIES:
        setup.modality = modality
        parallel.worker(setup)
    runner = parallel.ImportPipeline(setup)
    source = Path(setup.run_root)
    parallel.invoke_features(runner, source)
    for name in parallel.STAGES:
        runner.require_stage(name)
        record = json.loads((runner.state / f"{name}.json").read_text())
        assert record["status"] == "complete"
        assert record["inventory"]
    # A subsequent original controller accepts the receipts and skips compute.
    old = parallel.holdout.Pipeline(setup)
    old.command = lambda *a, **k: pytest.fail("Imported features must not be recomputed")
    parallel.invoke_features(old, source)
    for name in parallel.STAGES:
        old.require_stage(name)


def test_matching_pilot_report_is_not_repaired(setup):
    runner = parallel.ImportPipeline(setup)
    report = runner.run / "pilot/report.json"
    report.parent.mkdir()
    report.write_text('{"status": "passed"}')
    marker = runner.state / "pilot.json"
    parallel.base.atomic_json(marker, {"status": "complete", "report": parallel.base.signature(report)})
    before = parallel.base.signature(marker)
    parallel.refresh_pilot_report(runner)
    assert parallel.base.signature(marker) == before
    assert not (runner.run / "metadata_repairs").exists()


def test_pilot_report_identity_change_is_never_repaired(setup):
    runner = parallel.ImportPipeline(setup)
    report = runner.run / "pilot/report.json"
    report.parent.mkdir()
    report.write_text('{"status": "passed"}')
    marker = runner.state / "pilot.json"
    parallel.base.atomic_json(marker, {"status": "complete", "report": parallel.base.signature(report)})
    report.write_text('{"status": "failed", "changed": true}')
    before = parallel.base.signature(marker)
    with pytest.raises(RuntimeError, match="not limited to timestamps"):
        parallel.refresh_pilot_report(runner)
    assert parallel.base.signature(marker) == before


def test_timestamp_only_pilot_report_requires_dependency_revalidation(setup):
    runner = parallel.ImportPipeline(setup)
    report = runner.run / "pilot/report.json"
    report.parent.mkdir()
    report.write_text('{"status": "passed"}')
    previous = parallel.base.signature(report)
    previous["mtime_ns"] -= 1
    marker = runner.state / "pilot.json"
    parallel.base.atomic_json(marker, {"status": "complete", "report": previous})
    before = parallel.base.signature(marker)
    with pytest.raises(RuntimeError, match="Run stage"):
        parallel.refresh_pilot_report(runner)
    assert parallel.base.signature(marker) == before
