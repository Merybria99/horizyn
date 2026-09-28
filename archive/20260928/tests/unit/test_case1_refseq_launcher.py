import os
import shutil
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import run_case1_refseq_best16k_parallel as parallel

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "occupied_uuid,free_memory,existing_session,expected_exit",
    [
        ("GPU-selected", "140000", "0", 1),
        ("GPU-other", "140000", "0", 0),
        ("GPU-other", "20000", "0", 1),
        ("GPU-other", "140000", "1", 1),
    ],
)
def test_launcher_guards_selected_gpu_and_session(
    tmp_path, occupied_uuid, free_memory, existing_session, expected_exit
):
    project = tmp_path / "project with spaces"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    launcher = scripts / "launch_case1_refseq_best16k.sh"
    shutil.copyfile(ROOT / "scripts" / launcher.name, launcher)
    python = tmp_path / ".capability-run-py/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to("/bin/true")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    commands = {
        "nvidia-smi": """#!/bin/sh
case "$*" in
  *--query-gpu=uuid*) echo GPU-selected ;;
  *--query-gpu=memory.free*) echo "$TEST_FREE" ;;
  *--query-compute-apps=*) echo "$TEST_OCCUPIED, 123" ;;
  *) exit 1 ;;
esac
""",
        "tmux": """#!/bin/sh
if [ "$1" = has-session ]; then [ "$TEST_SESSION" = 1 ]; exit $?; fi
if [ "$1" = new-session ]; then touch "$TEST_LAUNCHED"; exit 0; fi
exit 1
""",
    }
    for name, source in commands.items():
        path = bin_dir / name
        path.write_text(source)
        path.chmod(0o755)
    launched = tmp_path / "launched"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TEST_OCCUPIED": occupied_uuid,
        "TEST_FREE": free_memory,
        "TEST_SESSION": existing_session,
        "TEST_LAUNCHED": str(launched),
    }
    result = subprocess.run(["bash", str(launcher), "2"], env=env, capture_output=True, text=True)
    assert result.returncode == expected_exit, result.stdout + result.stderr
    assert launched.exists() == (expected_exit == 0)


def test_refseq_uses_same_model_reaction_and_chemistry_as_workbook():
    small = yaml.safe_load(
        (ROOT / "wet_lab/Case1/restricted_setting/query_horizyn1_best16k.yaml").read_text()
    )
    refseq = yaml.safe_load((ROOT / "wet_lab/configs/refseq/horizyn1_best16k.yaml").read_text())
    for key in ("model", "reaction", "feature_generation"):
        assert refseq[key] == small[key]
    assert (
        refseq["candidate_pool"]["target_cache_dir"] != small["candidate_pool"]["target_cache_dir"]
    )
    assert "five_shards" in refseq["candidate_pool"]["ids"]
    assert refseq["candidate_pool"]["validate_finite_on_access"] is True


def test_parallel_allows_authorized_sharing_but_requires_headroom():
    devices = "0,GPU-0,20000\n1,GPU-1,81000\n2,GPU-2,78000\n3,GPU-3,81000"
    busy = "GPU-0,100\nGPU-1,101\nGPU-2,200\nGPU-3,301"
    parallel.check_gpus(devices, busy)
    for gpu in (1, 2, 3):
        for free_mib in ("48000", "nan"):
            low_memory = "\n".join(
                f"{i},GPU-{i},{free_mib if i == gpu else '81000'}" for i in (1, 2, 3)
            )
            with pytest.raises(RuntimeError, match=f"GPU {gpu} needs 48 GiB"):
                parallel.check_gpus(low_memory, busy)
    parallel.check_gpus("1,GPU-1,49152\n2,GPU-2,49152\n3,GPU-3,49152", busy)
    with pytest.raises(RuntimeError, match="GPU 3 is unavailable"):
        parallel.check_gpus("1,GPU-1,81000\n2,GPU-2,78000", "")


def test_parallel_configs_preserve_scoring_and_limit_shared_gpu():
    base = yaml.safe_load(parallel.BASE.read_text())
    configs = [parallel.shard_config(base, i, b, m) for i, (_, b, m, _) in enumerate(parallel.PLAN)]
    assert [gpu for gpu, *_ in parallel.PLAN] == [1, 2, 3]
    assert [c["candidate_pool"]["encoding_batch_size"] for c in configs] == [64, 64, 64]
    assert [c["inference"]["cuda_memory_limit_gib"] for c in configs] == [16, 16, 16]
    for i, c in enumerate(configs):
        assert c["model"] == base["model"]
        assert c["reaction"] == base["reaction"]
        assert c["inference"]["top_k"] == base["inference"]["top_k"]
        assert c["feature_generation"]["device"] == "cpu"
        assert f"shard_{i}" in c["candidate_pool"]["ids"]
        assert f"shard_{i}" in c["output"]["directory"]
    assert "device" not in base["feature_generation"]


def test_parallel_partition_rejects_missing_and_duplicate_ids(tmp_path):
    full = tmp_path / "full.txt"
    full.write_text("A\nB\nC\nD\n")
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    first.write_text("A\nB\n")
    second.write_text("C\nD\n")
    base = {"candidate_pool": {"ids": str(full)}}
    configs = [{"candidate_pool": {"ids": str(p)}} for p in (first, second)]
    assert parallel.validate_partition(base, configs) == 4
    for bad in ("B\nD\n", "C\n", "C\nD\nE\n"):
        second.write_text(bad)
        with pytest.raises(ValueError, match="partition|cover"):
            parallel.validate_partition(base, configs)


def test_parallel_failure_stops_only_its_own_live_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel, "OUTPUT", tmp_path / "run")
    monkeypatch.setattr(parallel, "LOCK", tmp_path / "launch.lock")
    monkeypatch.setattr(parallel, "preflight", lambda: None)
    monkeypatch.setattr(parallel, "validate_partition", lambda *_: 6)
    monkeypatch.setattr(parallel.signal, "signal", lambda *_: None)
    spawned, stopped = [], []

    def spawn(command, **kwargs):
        gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
        assert kwargs["start_new_session"] is True
        assert kwargs["pass_fds"]
        assert kwargs["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "backend:native"
        process = SimpleNamespace(pid=9000 + len(spawned), returncode=1 if gpu == "1" else None)
        process.poll = lambda: process.returncode
        process.wait = lambda **_: setattr(process, "returncode", -15)
        spawned.append((gpu, process))
        return process

    monkeypatch.setattr(parallel.subprocess, "Popen", spawn)
    monkeypatch.setattr(parallel.os, "killpg", lambda pid, sig: stopped.append((pid, sig)))
    with pytest.raises(RuntimeError, match="worker failed"):
        parallel.run()
    assert [gpu for gpu, _ in spawned] == ["1", "2", "3"]
    assert stopped == [(9001, signal.SIGTERM), (9002, signal.SIGTERM)]
    assert not (tmp_path / "run/merged").exists()


def test_parallel_uses_exclusive_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel, "LOCK", tmp_path / "launch.lock")
    with parallel.run_lock():
        with pytest.raises(RuntimeError, match="RefSeq best16k run is active"):
            with parallel.run_lock():
                pytest.fail("Acquired an already-held run lock")


def test_restart_refuses_wrong_host(monkeypatch):
    monkeypatch.setattr(parallel.socket, "gethostname", lambda: "tyrosine")
    monkeypatch.setattr(parallel, "launch", lambda: pytest.fail("Unexpected launch"))
    with pytest.raises(RuntimeError, match="on glutamine"):
        parallel.restart()


@pytest.mark.parametrize("survives", [False, True])
def test_restart_signals_only_verified_controller_and_refuses_survivors(monkeypatch, survives):
    from scripts import stop_horizyn1_circe_v2 as stop

    controller = stop.Process(123, 1, 10, "S")
    child = stop.Process(124, 123, 11, "S")
    monkeypatch.setattr(parallel.socket, "gethostname", lambda: "glutamine")
    monkeypatch.setattr(parallel.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(parallel.subprocess, "check_output", lambda *a, **k: "123\n")
    monkeypatch.setattr(parallel, "verify_restart_controller", lambda pid: controller)
    monkeypatch.setattr(stop, "process_tree", lambda p: {123: controller, 124: child})
    monkeypatch.setattr(stop, "alive", lambda p: survives and p.pid == 124)
    ticks = iter([0, 31])
    monkeypatch.setattr(parallel.time, "monotonic", lambda: next(ticks))
    calls = []
    monkeypatch.setattr(stop, "send_signal", lambda p, s: calls.append((p.pid, s)))
    monkeypatch.setattr(parallel, "launch", lambda: calls.append("launch"))
    if survives:
        with pytest.raises(RuntimeError, match="replacement NOT launched"):
            parallel.restart()
    else:
        parallel.restart()
    assert calls == [(123, signal.SIGTERM)] + ([] if survives else ["launch"])


def test_restart_rejects_unrelated_local_process():
    # The pytest process belongs to us, but is not the screen controller.
    with pytest.raises(RuntimeError, match="not this RefSeq controller"):
        parallel.verify_restart_controller(os.getpid())
