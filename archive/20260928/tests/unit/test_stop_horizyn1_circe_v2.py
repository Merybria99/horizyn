from pathlib import Path
from types import SimpleNamespace
import os
import signal

import pytest

from scripts import stop_horizyn1_circe_v2 as stop


def test_option_values_accept_separate_and_equals_forms():
    assert stop.option_value(["all", "--run-root", "/a"], "--run-root") == "/a"
    assert stop.option_value(["--run-root=/b"], "--run-root") == "/b"
    assert stop.option_value([], "--run-root") is None
    with pytest.raises(RuntimeError, match="Malformed"):
        stop.option_value(["--run-root"], "--run-root")


def test_real_unrelated_process_is_not_accepted_as_pipeline(tmp_path):
    with pytest.raises(RuntimeError, match="does not identify"):
        stop.verify_controller(os.getpid(), tmp_path)


def test_pid_reuse_and_zombies_are_not_signalled(monkeypatch):
    observed = stop.Process(12345, 1, 10, "S")
    sent = []
    monkeypatch.setattr(stop.os, "kill", lambda *args: sent.append(args))
    for current in (None, stop.Process(12345, 1, 11, "S"), stop.Process(12345, 1, 10, "Z")):
        monkeypatch.setattr(stop, "read_process", lambda pid, current=current: current)
        stop.send_signal(observed, signal.SIGTERM)
    assert not sent
    monkeypatch.setattr(stop, "read_process", lambda pid: observed)
    stop.send_signal(observed, signal.SIGTERM)
    assert sent == [(12345, signal.SIGTERM)]


@pytest.mark.parametrize("profile", [None, "h200"])
@pytest.mark.parametrize("profile_in_environment", [False, True])
@pytest.mark.parametrize("reaction_holdout", [False, True])
def test_controller_run_root_and_start_identity_are_verified(tmp_path, monkeypatch, profile, profile_in_environment, reaction_holdout):
    pid = os.getpid()
    initial = stop.Process(pid, 1, 55, "S")
    monkeypatch.setattr(stop, "read_process", lambda pid: initial)
    original_read = Path.read_bytes
    controller = "horizyn1_circe_v2_reaction_holdout.py" if reaction_holdout else "horizyn1_circe_v2_pipeline.py"
    arguments = ["python", str(stop.ROOT / "scripts" / controller), "all"]
    if profile and not profile_in_environment:
        arguments += ["--profile", profile]

    def fake_read(path):
        if str(path) == f"/proc/{pid}/cmdline":
            return b"\0".join(os.fsencode(x) for x in arguments) + b"\0"
        if str(path) == f"/proc/{pid}/environ":
            return (b"PYTHONUNBUFFERED=1\0" +
                    (os.fsencode("CIRCE_PROFILE=" + profile) + b"\0" if profile and profile_in_environment else b""))
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", fake_read)
    default = stop.ROOT / "runs" / ("horizyn1_circe_v2_h200" if profile else "horizyn1_circe_v2")
    if reaction_holdout:
        default = stop.ROOT / "runs" / f"horizyn1_circe_v2_reaction_holdout_{profile or 'base'}"
    assert stop.verify_controller(pid, default) == initial
    with pytest.raises(RuntimeError, match="different RUN_ROOT"):
        stop.verify_controller(pid, tmp_path)
    arguments += ["--run-root", str(tmp_path)]
    assert stop.verify_controller(pid, tmp_path) == initial
    sequence = iter([initial, stop.Process(pid, 1, 56, "S")])
    monkeypatch.setattr(stop, "read_process", lambda pid: next(sequence))
    with pytest.raises(RuntimeError, match="PID changed"):
        stop.verify_controller(pid, tmp_path)


def test_dry_run_never_sends_signals(tmp_path, monkeypatch):
    (tmp_path / "pipeline.pid").write_text("54321\n")
    controller = stop.Process(54321, 1, 100, "S")
    child = stop.Process(54322, 54321, 110, "S")
    monkeypatch.setattr(stop, "verify_controller", lambda *args: controller)
    monkeypatch.setattr(stop, "process_tree", lambda *args: {controller.pid: controller, child.pid: child})
    monkeypatch.setattr(stop, "send_signal", lambda *args: pytest.fail("dry run must not send signals"))
    stop.main(["--run-root", str(tmp_path), "--dry-run"])


def test_stop_signals_only_observed_controller_and_descendants(tmp_path, monkeypatch):
    (tmp_path / "pipeline.pid").write_text("54321\n")
    controller = stop.Process(54321, 1, 100, "S")
    child = stop.Process(54322, 54321, 110, "S")
    monkeypatch.setattr(stop, "verify_controller", lambda *args: controller)
    monkeypatch.setattr(stop, "process_tree", lambda *args: {controller.pid: controller, child.pid: child})
    signals = []
    monkeypatch.setattr(stop, "send_signal", lambda process, signum: signals.append((process.pid, signum)))
    monkeypatch.setattr(stop, "alive", lambda *args: False)
    stop.main(["--run-root", str(tmp_path)])
    assert signals == [(54321, signal.SIGTERM), (54322, signal.SIGTERM)]
    assert (tmp_path / "pipeline.pid").is_file()


def test_missing_pid_file_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(stop, "send_signal", lambda *args: pytest.fail("no PID means no signal"))
    stop.main(["--run-root", str(tmp_path)])


def test_explicit_pid_is_verified_without_shared_pid_file(tmp_path, monkeypatch):
    controller = stop.Process(54321, 1, 100, "S")
    verified = []
    def verify(pid, root):
        verified.append((pid, root))
        return controller
    monkeypatch.setattr(stop, "verify_controller", verify)
    monkeypatch.setattr(stop, "process_tree", lambda *args: {controller.pid: controller})
    monkeypatch.setattr(stop, "send_signal", lambda *args: pytest.fail("dry run must not send signals"))
    stop.main(["--run-root", str(tmp_path), "--pid", "54321", "--dry-run"])
    assert verified == [(54321, tmp_path.resolve())]


def test_fast_io_pid_requires_exact_output_path_and_stable_identity(tmp_path, monkeypatch):
    pid = os.getpid()
    initial = stop.Process(pid, 1, 55, "S")
    monkeypatch.setattr(stop, "read_process", lambda pid: initial)
    original = Path.read_bytes
    argv = ["python", str(stop.ROOT / "scripts/train_protein_pooling_fast_io.py"),
            "--config", "/unrelated/config.yaml", "--io-output-dir", str(tmp_path)]
    def fake_read(path):
        if str(path) == f"/proc/{pid}/cmdline":
            return b"\0".join(os.fsencode(x) for x in argv) + b"\0"
        if str(path) == f"/proc/{pid}/environ":
            pytest.fail("Fast-I/O verification does not need the environment")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", fake_read)
    assert stop.verify_controller(pid, tmp_path) == initial
    with pytest.raises(RuntimeError, match="different RUN_ROOT"):
        stop.verify_controller(pid, tmp_path / "other")
    sequence = iter([initial, stop.Process(pid, 1, 56, "S")])
    monkeypatch.setattr(stop, "read_process", lambda pid: next(sequence))
    with pytest.raises(RuntimeError, match="PID changed"):
        stop.verify_controller(pid, tmp_path)
    monkeypatch.setattr(stop, "read_process", lambda pid: initial)
    del argv[-2:]
    with pytest.raises(RuntimeError, match="different RUN_ROOT"):
        stop.verify_controller(pid, tmp_path)


@pytest.mark.parametrize("mode", ["resume", "extract", "worker"])
def test_parallel_controller_only_allows_main_resume(tmp_path, monkeypatch, mode):
    pid = os.getpid()
    initial = stop.Process(pid, 1, 55, "S")
    monkeypatch.setattr(stop, "read_process", lambda pid: initial)
    original = Path.read_bytes
    argv = ["python", str(stop.ROOT / "scripts/horizyn1_parallel_reactions.py"),
            mode, "--run-root", str(tmp_path), "--cache-dir", str(tmp_path.parent / "cache")]
    def fake_read(path):
        if str(path) == f"/proc/{pid}/cmdline":
            return b"\0".join(os.fsencode(x) for x in argv) + b"\0"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", fake_read)
    if mode == "resume":
        assert stop.verify_controller(pid, tmp_path) == initial
    else:
        with pytest.raises(RuntimeError, match="not a main-run"):
            stop.verify_controller(pid, tmp_path)
