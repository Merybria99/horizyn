from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/horizyn1_circe_v2_pipeline.py"
spec = importlib.util.spec_from_file_location("horizyn1_pipeline", SCRIPT)
pipeline = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pipeline)


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("GPU_COUNT", "4")
    args = pipeline.parse_args(["preflight", "--run-root", str(tmp_path / "run"), "--data-root", str(tmp_path / "source")])
    return pipeline.Pipeline(args)


def test_default_training_agreement(runner):
    yaml = pytest.importorskip("yaml")
    output = runner.run / "configs/train.yaml"
    runner.config(output)
    value = yaml.safe_load(output.read_text())
    assert value["data"]["typed_negative_positive_fraction"] == .85
    assert value["data"]["typed_negative_pools_path"] is None
    assert value["training"]["loss"]["biofp_aux_weight"] == 0
    assert value["training"]["loss"]["sampled_require_both_directions"]
    assert value["training"]["loss"]["sampled_share_indexed_negatives"]
    assert value["training"]["loss"]["name"] == "BidirectionalSampledMultiPositiveInfoNCELoss"
    assert value["model"]["reaction_multimodal_attention"]["side_composition"] == "directional_delta"
    assert value["model"]["sleec_pooling"]["freeze_scorer"]
    assert value["training"]["devices"] == 4
    assert value["training"]["precision"] == "32-true"
    assert value["data"]["protein_residue_embeds_path"].startswith(str(runner.run))
    before = output.stat().st_mtime_ns
    runner.config(output)
    assert output.stat().st_mtime_ns == before


def test_pilot_is_real_training_with_separate_outputs(runner):
    yaml = pytest.importorskip("yaml")
    pilot = runner.run / "pilot"
    runner.config(pilot / "train.yaml", pilot=pilot)
    value = yaml.safe_load((pilot / "train.yaml").read_text())
    assert value["training"]["max_steps"] == 20
    assert value["training"]["validation_enabled"] is False
    assert value["data"]["validation_retrieval_candidate_set"] == "validation"
    assert value["training"]["validation_retrieval_candidate_set"] == "validation"
    assert value["training"]["early_stopping"]["enabled"] is False
    assert value["data"]["indexed_pairs_dir"] == str(pilot / "index")
    assert value["logging"]["checkpoint_dir"] == str(pilot / "checkpoints")
    assert value["data"]["typed_negative_pools_path"] is None


def test_completed_step_is_reused_only_when_inputs_outputs_match(runner, tmp_path):
    source, output = tmp_path / "source.txt", tmp_path / "output.txt"
    source.write_text("input")
    calls = []
    def action():
        calls.append(1)
        output.write_text("output")
    runner.step("toy", [source], [output], action)
    runner.step("toy", [source], [output], action)
    assert calls == [1]
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="Inputs/options changed"):
        runner.step("toy", [source], [output], action)


def test_unowned_output_is_never_overwritten(runner, tmp_path):
    output = tmp_path / "preexisting.txt"
    output.write_text("user data")
    with pytest.raises(RuntimeError, match="Unowned outputs"):
        runner.step("foreign", [], [output], lambda: output.write_text("bad"))
    assert output.read_text() == "user data"


def test_failure_records_failed_stage_and_propagates(runner, tmp_path):
    def fail():
        raise ValueError("deliberate failure")
    with pytest.raises(ValueError, match="deliberate"):
        runner.step("failure", [], [tmp_path / "never"], fail)
    assert json.loads((runner.state / "failure.json").read_text())["status"] == "failed"
    with pytest.raises(RuntimeError, match="Run stage"):
        runner.require_stage("failure")


def test_command_exit_code_is_not_hidden_by_logging(runner):
    with pytest.raises(RuntimeError, match="exit code 7"):
        runner.command([sys.executable, "-c", "raise SystemExit(7)"], "exit7")


def test_stale_stage_output_rejected(runner, tmp_path):
    output = tmp_path / "artifact.txt"
    runner.step("artifact", [], [output], lambda: output.write_text("first"))
    output.write_text("changed")
    with pytest.raises(RuntimeError, match="Stale artifact"):
        runner.require_stage("artifact")


def test_compact_index_inventory_is_checked_on_resume(runner):
    directory = runner.data / "index"
    def build():
        directory.mkdir()
        (directory / "manifest.json").write_text("{}")
        (directory / "pairs.npy").write_bytes(b"original array")
    runner.step("index", [], [directory / "manifest.json"], build)
    (directory / "pairs.npy").write_bytes(b"changed array")
    with pytest.raises(RuntimeError, match="Stale index"):
        runner.step("index", [], [directory / "manifest.json"], build)


def test_failed_owned_index_is_preserved_and_restartable(runner):
    directory = runner.data / "index"
    def partial_build():
        directory.mkdir()
        (directory / "partial.npy").write_bytes(b"partial data")
        raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.step("index", [], [directory / "manifest.json"], partial_build)
    def finish():
        directory.mkdir()
        (directory / "manifest.json").write_text("{}")
    runner.step("index", [], [directory / "manifest.json"],
                lambda: runner.rebuild_directory(directory, finish))
    archives = list(runner.data.glob(".index.interrupted.*"))
    assert len(archives) == 1
    assert (archives[0] / "partial.npy").read_bytes() == b"partial data"
    assert (directory / "manifest.json").is_file()


def test_foreign_partial_index_is_not_archived_or_overwritten(runner):
    directory = runner.data / "index"
    directory.mkdir()
    (directory / "user.npy").write_bytes(b"user data")
    with pytest.raises(RuntimeError, match="Unowned nonempty"):
        runner.step("index", [], [directory / "manifest.json"],
                    lambda: runner.rebuild_directory(directory, lambda: None))
    assert (directory / "user.npy").read_bytes() == b"user data"
    assert not list(runner.data.glob(".index.interrupted.*"))


def test_report_signature_is_checked_before_feature_gate(runner, tmp_path):
    report = tmp_path / "report.json"
    report.write_text("original")
    pipeline.atomic_json(runner.state / "pilot.json", {"status": "complete", "report": pipeline.signature(report)})
    report.write_text("tampered")
    with pytest.raises(RuntimeError, match="Stale pilot"):
        runner.require_stage("pilot")


@pytest.mark.parametrize("batch", [0, 19, 25, 99, 101])
def test_pair_ratio_requires_exact_batch_quota(batch):
    with pytest.raises(SystemExit):
        pipeline.parse_args(["--train-batch-size", str(batch)])


@pytest.mark.parametrize("pairs,batch,world,expected", [(1, 100, 4, 1), (220, 100, 4, 1),
    (221, 100, 4, 2), (1000, 100, 4, 5), (44, 20, 4, 1), (45, 20, 4, 2)])
def test_epoch_projection_matches_55_percent_base_sweep(pairs, batch, world, expected):
    assert pipeline.indexed_epoch_steps(pairs, batch, world) == expected


def test_resume_rejects_changed_controller_implementation(runner, tmp_path):
    output = tmp_path / "output"
    runner.step("code_bound", [], [output], lambda: output.write_text("done"))
    marker = runner.state / "code_bound.json"
    saved = json.loads(marker.read_text())
    saved["fingerprint"]["controller_sha256"] = "old implementation"
    pipeline.atomic_json(marker, saved)
    with pytest.raises(RuntimeError, match="controller implementation"):
        runner.require_stage("code_bound")


def test_hdf5_validation_fails_nonfinite_wrong_ids_and_bad_offsets(tmp_path):
    h5py, np = pytest.importorskip("h5py"), pytest.importorskip("numpy")
    path = tmp_path / "features.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ids", data=[b"a", b"b"])
        handle.create_dataset("vectors", data=np.ones((4, 3)))
        handle.create_dataset("offsets", data=[0, 2, 4])
    assert pipeline.validate_h5(path, {"a", "b"}, 3)["coverage"] == 1
    with pytest.raises(ValueError, match="Feature ID mismatch"):
        pipeline.validate_h5(path, {"a", "b", "c"}, 3)
    with h5py.File(path, "r+") as handle:
        handle["vectors"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        pipeline.validate_h5(path, {"a", "b"}, 3)
    with h5py.File(path, "r+") as handle:
        handle["vectors"][0, 0] = 1
        handle["offsets"][1] = 0
    with pytest.raises(ValueError, match="Invalid ragged offsets"):
        pipeline.validate_h5(path, {"a", "b"}, 3)


def test_missing_virtual_shard_fails_instead_of_silent_zero_fill(tmp_path):
    h5py, np = pytest.importorskip("h5py"), pytest.importorskip("numpy")
    path = tmp_path / "virtual.h5"
    layout = h5py.VirtualLayout(shape=(2, 3), dtype="float16")
    layout[:] = h5py.VirtualSource(str(tmp_path / "missing.h5"), "vectors", shape=(2, 3))
    with h5py.File(path, "w", libver="latest") as handle:
        handle.create_dataset("ids", data=[b"a"])
        handle.create_dataset("offsets", data=[0, 2])
        handle.create_virtual_dataset("vectors", layout)
    with pytest.raises(ValueError, match="Missing immutable VDS shard"):
        pipeline.validate_h5(path, {"a"}, 3)


def test_chemistry_features_require_correct_dimensions_and_ids(tmp_path):
    np = pytest.importorskip("numpy")
    path = tmp_path / "chemistry.npz"
    np.savez(path, ids=np.array(["a"], dtype=object), vectors=np.ones((1, 617)), mask=np.array([True]))
    assert pipeline.validate_chemistry(path, {"a"})["coverage"] == 1
    with pytest.raises(ValueError, match="ID mismatch"):
        pipeline.validate_chemistry(path, {"b"})


def test_signature_binds_real_path_and_change(tmp_path):
    path = tmp_path / "file"
    path.write_text("data")
    signature = pipeline.signature(path)
    assert signature["path"] == str(path.resolve())
    path.write_text("new data")
    assert pipeline.signature(path) != signature


def test_pilot_pairs_keep_required_unique_pr_id(runner):
    fasta = runner.source / "clustered/proteins.fasta"
    fasta.parent.mkdir(parents=True)
    fasta.write_text("".join(f">p{i}\n{'A' * (100 + i % 5)}\n" for i in range(250)))
    pipeline.write_csv(runner.data / "train_pairs.csv", ["pr_id", "reaction_id", "protein_id"],
                       ({"pr_id": str(i + 1000), "reaction_id": f"r{i % 10}", "protein_id": f"p{i}"} for i in range(250)))
    pipeline.write_csv(runner.data / "reactions.csv", ["reaction_id", "reaction_smiles"],
                       ({"reaction_id": f"r{i}", "reaction_smiles": "CCO>>CC=O"} for i in range(10)))
    pilot = runner.run / "pilot"
    runner.select_pilot(pilot)
    rows = list(pipeline.csv_rows(pilot / "train_pairs.csv"))
    assert len(rows) == 250
    assert len({row["pr_id"] for row in rows}) == 250
    assert set(rows[0]) == {"pr_id", "reaction_id", "protein_id"}


@pytest.fixture
def reconstruction_audit(tmp_path):
    path = tmp_path / "clustered_manifest.json"
    path.write_text('{"clustered_proteins": 3}\n')
    stat = path.stat()
    fields = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    return path, {
        "status": "passed",
        "errors": [],
        "artifact_signatures": {str(path.resolve()): fields},
    }


def test_reconstruction_audit_accepts_matching_real_file(reconstruction_audit):
    path, audit = reconstruction_audit
    assert pipeline.validate_reconstruction_audit(audit) == []
    assert pipeline.validate_reconstruction_audit(audit, expected_paths=[path]) == []


def test_reconstruction_audit_accepts_only_device_difference(reconstruction_audit):
    path, audit = reconstruction_audit
    stored = audit["artifact_signatures"][str(path.resolve())]
    stored[0] += 999
    before = list(stored)
    assert pipeline.validate_reconstruction_audit(audit, expected_paths=[path]) == [str(path.resolve())]
    assert stored == before  # Validation must not rewrite the historical audit.


@pytest.mark.parametrize("index,field", [(1, "inode"), (2, "size"), (3, "mtime"), (4, "ctime")])
def test_reconstruction_audit_rejects_each_nondevice_change(reconstruction_audit, index, field):
    path, audit = reconstruction_audit
    stored = audit["artifact_signatures"][str(path.resolve())]
    actual_value = stored[index]
    stored[index] += 1
    with pytest.raises(RuntimeError) as error:
        pipeline.validate_reconstruction_audit(audit)
    message = str(error.value)
    assert field in message
    assert str(path.resolve()) in message
    assert str(stored[index]) in message
    assert str(actual_value) in message
    assert "expected" in message.lower()
    assert "actual" in message.lower()


def test_reconstruction_audit_does_not_hide_changed_mtime_behind_device_mismatch(reconstruction_audit):
    path, audit = reconstruction_audit
    stored = audit["artifact_signatures"][str(path.resolve())]
    stored[0] += 999
    stored[3] += 1
    with pytest.raises(RuntimeError, match="mtime"):
        pipeline.validate_reconstruction_audit(audit)


def test_reconstruction_audit_reports_multiple_changed_nondevice_fields(reconstruction_audit):
    path, audit = reconstruction_audit
    stored = audit["artifact_signatures"][str(path.resolve())]
    stored[1] += 1
    stored[2] += 1
    stored[4] += 1
    with pytest.raises(RuntimeError) as error:
        pipeline.validate_reconstruction_audit(audit)
    assert all(field in str(error.value) for field in ("inode", "size", "ctime"))


@pytest.mark.parametrize("signatures", [None, {}, [], "not a signature map"])
def test_reconstruction_audit_rejects_empty_or_malformed_signature_map(reconstruction_audit, signatures):
    _, audit = reconstruction_audit
    audit["artifact_signatures"] = signatures
    with pytest.raises(RuntimeError):
        pipeline.validate_reconstruction_audit(audit)


def test_reconstruction_audit_rejects_missing_signature_map(reconstruction_audit):
    _, audit = reconstruction_audit
    del audit["artifact_signatures"]
    with pytest.raises(RuntimeError):
        pipeline.validate_reconstruction_audit(audit)


@pytest.mark.parametrize("stored", [None, [], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6],
                                     [1, 2, "3", 4, 5], [1, 2, 3.0, 4, 5],
                                     [True, 2, 3, 4, 5], {"device": 1}, "1,2,3,4,5"])
def test_reconstruction_audit_rejects_malformed_five_integer_signature(reconstruction_audit, stored):
    path, audit = reconstruction_audit
    audit["artifact_signatures"][str(path.resolve())] = stored
    with pytest.raises(RuntimeError):
        pipeline.validate_reconstruction_audit(audit)


def test_reconstruction_audit_rejects_absent_required_artifact(reconstruction_audit, tmp_path):
    path, audit = reconstruction_audit
    another_required_file = tmp_path / "proteins.fasta"
    another_required_file.write_text(">p1\nAAAA\n")
    with pytest.raises(RuntimeError) as error:
        pipeline.validate_reconstruction_audit(audit, expected_paths=[path, another_required_file])
    assert str(another_required_file.resolve()) in str(error.value)


@pytest.mark.parametrize("status,errors", [("failed", []), ("running", []), (None, []),
                                         ("passed", ["bad sequence"]), ("passed", {"bad": "edge"})])
def test_reconstruction_audit_rejects_nonpassed_or_erroneous_audit(reconstruction_audit, status, errors):
    _, audit = reconstruction_audit
    audit.update(status=status, errors=errors)
    with pytest.raises(RuntimeError):
        pipeline.validate_reconstruction_audit(audit)


def test_h200_cli_defaults_preserve_global_rows_and_use_new_run(monkeypatch):
    monkeypatch.delenv("RUN_ROOT", raising=False)
    monkeypatch.delenv("SCRATCH_ROOT", raising=False)
    args = pipeline.parse_args(["all", "--profile", "h200"])
    assert Path(args.run_root).name == "horizyn1_circe_v2_h200"
    assert args.train_batch_size == 100
    assert args.extraction_batch_size == 32
    assert args.extraction_max_tokens == 32768
    assert args.extraction_length_sort is True
    assert args.extraction_padded_token_budget is True
    assert args.extraction_checkpoint_every == args.extraction_progress_every == 1000
    assert args.cpu_threads == 4
    assert args.scratch_root is None  # No local SSD requirement.


def test_base_extraction_defaults_remain_unchanged():
    args = pipeline.parse_args(["all", "--profile", "base"])
    assert args.extraction_batch_size == 8
    assert args.extraction_max_tokens == 4096
    assert not args.extraction_length_sort and not args.extraction_padded_token_budget
    assert args.extraction_checkpoint_every == args.extraction_progress_every == 100


def test_h200_configuration_and_pilot_keep_real_loader_profile(tmp_path, monkeypatch):
    yaml = pytest.importorskip("yaml")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("GPU_COUNT", "4")
    args = pipeline.parse_args(["all", "--profile", "h200", "--run-root", str(tmp_path / "h200")])
    run = pipeline.Pipeline(args)
    run.config(run.run / "configs/train.yaml")
    run.config(run.run / "pilot/train.yaml", pilot=run.run / "pilot")
    for path in (run.run / "configs/train.yaml", run.run / "pilot/train.yaml"):
        value = yaml.safe_load(path.read_text())
        assert value["training"]["precision"] == "bf16-mixed"
        assert value["training"]["fused_adamw"] is True
        assert value["training"]["contrastive_fp32"] is True
        assert value["training"]["fp32_sensitive_modules"] is True
        assert value["training"]["cpu_num_threads"] == 4
        assert value["data"]["num_workers"] == 4
        assert value["data"]["persistent_workers"] is True
        assert value["data"]["prefetch_factor"] == 2
        assert value["data"]["train_batch_size"] * value["training"]["devices"] == 400
        assert str(tmp_path / "h200") in value["data"]["protein_residue_embeds_path"]
    assert run.runtime_env["OMP_NUM_THREADS"] == "4"


def test_shared_work_is_default_without_local_probe(runner, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Default shared mode must not probe or require local scratch")
    monkeypatch.setattr(pipeline, "inspect_scratch_root", forbidden)
    assert runner.work_directory() == runner.run / "work/preparation"
    binding = json.loads((runner.state / "work_directory.json").read_text())
    assert binding["storage"] == "shared"
    assert runner.work_directory(check_space=False) == runner.run / "work/preparation"


def test_work_binding_change_is_not_silently_migrated(runner):
    runner.work_directory()
    runner.args.profile = "h200"
    with pytest.raises(RuntimeError, match="binding changed"):
        runner.work_directory()


def test_explicit_network_scratch_is_rejected_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "scratch_filesystem", lambda path: "nfs4")
    with pytest.raises(RuntimeError, match="local disk filesystem"):
        pipeline.inspect_scratch_root(tmp_path, 0)


def test_optional_local_scratch_is_unique_owned_and_lost_state_fails(runner, tmp_path, monkeypatch):
    runner.args.scratch_root = tmp_path
    monkeypatch.setattr(pipeline, "inspect_scratch_root", lambda path, minimum: {
        "scratch_root": str(tmp_path), "filesystem": "ext4", "free_bytes": 10**15})
    work = runner.work_directory()
    assert work.parent.name.startswith("circe-")
    assert work.parent.parent == tmp_path
    owner = work.parent / ".circe_owner.json"
    assert owner.is_file()
    assert runner.work_directory() == work
    owner.unlink()
    with pytest.raises(RuntimeError, match="scratch is missing"):
        runner.work_directory()


def test_prepare_passes_shared_work_log_and_cache_controls(runner, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "require_stage", lambda name: None)
    monkeypatch.setattr(runner, "command", lambda argv, name: calls.append(list(map(str, argv))))
    def step(name, inputs, outputs, action, parameters=None):
        assert parameters["work_dir"] == str(runner.run / "work/preparation")
        assert parameters["sqlite_cache_mib"] == 512
        action()
    monkeypatch.setattr(runner, "step", step)
    runner.prepare()
    command = calls[0]
    assert command[command.index("--work-dir") + 1] == str(runner.run / "work/preparation")
    assert command[command.index("--log-dir") + 1] == str(runner.logs / "preparation")
    assert command[command.index("--sqlite-cache-mib") + 1] == "512"


def test_extraction_performance_options_are_fingerprinted(runner, tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"weights")
    runner.prott5 = str(model)
    runner.args.extraction_max_tokens = 32768
    runner.args.extraction_batch_size = 32
    runner.args.extraction_length_sort = True
    runner.args.extraction_padded_token_budget = True
    captured = []
    monkeypatch.setattr(runner, "step", lambda name, inputs, outputs, action, parameters: captured.append(parameters))
    runner.protein_features(tmp_path / "unused.fasta", tmp_path / "unused.h5", "pilot")
    assert captured[0]["token_budget"] == 32768
    assert captured[0]["batch_size"] == 32
    assert captured[0]["length_sort"] is True
    assert captured[0]["padded_token_budget"] is True


def test_termination_signals_verified_descendants_but_never_self(runner, monkeypatch):
    import os
    import signal
    from scripts import stop_horizyn1_circe_v2 as stop
    current = stop.Process(os.getpid(), 1, 123, "R")
    child = stop.Process(100001, current.pid, 124, "S")
    grandchild = stop.Process(100002, child.pid, 125, "S")
    sent = []
    monkeypatch.setattr(stop, "read_process", lambda pid: current)
    monkeypatch.setattr(stop, "process_tree", lambda root: {p.pid: p for p in (current, child, grandchild)})
    monkeypatch.setattr(stop, "send_signal", lambda process, sig: sent.append((process.pid, sig)))
    monkeypatch.setattr(stop, "alive", lambda process: False)
    monkeypatch.setattr(pipeline.signal, "signal", lambda *args: None)
    with pytest.raises(SystemExit) as error:
        runner.terminate(signal.SIGTERM)
    assert error.value.code == 128 + signal.SIGTERM
    assert sent == [(child.pid, signal.SIGTERM), (grandchild.pid, signal.SIGTERM)]
