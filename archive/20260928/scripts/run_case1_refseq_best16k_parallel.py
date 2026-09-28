#!/usr/bin/env python3
"""Case 1 RefSeq shards on shared GPUs 1/2/3, with a small per-worker budget."""

from __future__ import annotations

import argparse
import copy
import csv
from contextlib import ExitStack, contextmanager
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "wet_lab/configs/refseq/horizyn1_best16k.yaml"
OUTPUT = ROOT / "wet_lab/runs/refseq/prokaryotes/horizyn1_best16k_parallel"
LOCK = OUTPUT.parent / "horizyn1_best16k/launch.lock"
SESSION = "case1_refseq_best16k_parallel"
# Physical GPU, batch size, allocator limit (GiB), required spare memory (GiB).
PLAN = ((1, 64, 16, 32), (2, 64, 16, 32), (3, 64, 16, 32))


def check_gpus(gpu_csv: str, process_csv: str) -> None:
    devices = {int(r[0]): r for r in csv.reader(gpu_csv.splitlines()) if r}
    busy = {r[0].strip() for r in csv.reader(process_csv.splitlines()) if r}
    for gpu, _, limit, reserve in PLAN:
        if gpu not in devices:
            raise RuntimeError(f"GPU {gpu} is unavailable")
        _, uuid, free_mib = devices[gpu]
        if not float(free_mib) >= (limit + reserve) * 1024:
            raise RuntimeError(f"GPU {gpu} needs {(limit + reserve)} GiB free before launch")
        if uuid.strip() in busy:
            print(
                f"Sharing GPU {gpu}: our allocator budget is {limit} GiB; existing jobs are left running",
                flush=True,
            )


def preflight() -> None:
    def query(fields, kind):
        return subprocess.check_output(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            text=True,
            timeout=15,
        )

    check_gpus(query("index,uuid,memory.free", "gpu"), query("gpu_uuid,pid", "compute-apps"))


@contextmanager
def run_lock():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Another single- or multi-GPU RefSeq best16k run is active"
            ) from None
        yield handle


def shard_config(base, shard, batch_size, limit):
    config = copy.deepcopy(base)
    pool = ROOT / f"wet_lab/databases/refseq/prokaryotes/query_shards_3/shard_{shard}"
    for key, name in {
        "ids": "candidate_ids_prott5_order.txt",
        "residue_embeddings": "proteins_prott5_residue.h5",
        "fasta": "proteins.fasta",
        "metadata_csv": "proteins.csv",
    }.items():
        config["candidate_pool"][key] = str(pool / name)
    config["candidate_pool"].update(
        encoding_batch_size=batch_size,
        encoding_progress_every_batches=25,
        target_cache_dir=str(
            ROOT / f"wet_lab/cache/refseq/prokaryotes/horizyn1_best16k/shard_{shard}"
        ),
    )
    config["inference"].update(device="cuda", cpu_threads=4, cuda_memory_limit_gib=limit)
    config["feature_generation"]["device"] = "cpu"
    config["output"]["directory"] = str(OUTPUT / f"shard_{shard}")
    return config


def validate_partition(base, configs):
    """Verify full coverage and disjointness before relying on a top-k merge."""
    full_path = ROOT / base["candidate_pool"]["ids"]
    with full_path.open() as full:
        count = 0
        seen = set()
        for config in configs:
            with Path(config["candidate_pool"]["ids"]).open() as shard:
                for line in shard:
                    key = line.strip()
                    if not key or key in seen or key != full.readline().strip():
                        raise ValueError("RefSeq shard IDs are not a disjoint ordered partition")
                    seen.add(key)
                    count += 1
        if full.readline() or not count:
            raise ValueError("RefSeq shards do not cover the full pool")
    return count


def worker(config_path):
    from wet_lab import query
    from wet_lab.query_io import screening_io_adapters

    with screening_io_adapters(query) as receipt:
        config_path.with_suffix(".io.json").write_text(json.dumps(receipt, indent=2) + "\n")
        result = query.run_query(query.load_query_config(config_path))
    config_path.with_suffix(".result.json").write_text(json.dumps({"result": str(result)}))


def interrupt(signum, _frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def run():
    with run_lock() as lock, ExitStack() as logs:
        preflight()
        base = yaml.safe_load(BASE.read_text())
        configs = [
            shard_config(base, i, batch, limit) for i, (_, batch, limit, _) in enumerate(PLAN)
        ]
        print(
            f"Verified {validate_partition(base, configs):,} disjoint RefSeq candidates", flush=True
        )
        OUTPUT.mkdir(parents=True, exist_ok=True)
        processes, paths = [], []
        for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(signum, interrupt)
        try:
            for shard, ((gpu, batch, limit, _), config) in enumerate(zip(PLAN, configs)):
                path = OUTPUT / f"shard_{shard}.yaml"
                if path.exists() and yaml.safe_load(path.read_text()) != config:
                    raise ValueError(f"Existing config changed; refusing to overwrite {path}")
                if not path.exists():
                    path.write_text(yaml.safe_dump(config, sort_keys=False))
                paths.append(path)
                env = {
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONUNBUFFERED": "1",
                    "HORIZYN_CPU_THREADS": "4",
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "4",
                    "PYTORCH_CUDA_ALLOC_CONF": "backend:native",
                }
                for name in (
                    "WET_LAB_CANDIDATE_POOL_ROOT_OVERRIDE",
                    "WET_LAB_RESIDUE_EMBEDDINGS_OVERRIDE",
                ):
                    env.pop(name, None)
                log = logs.enter_context((OUTPUT / f"gpu{gpu}.log").open("a"))
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", str(path)],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(lock.fileno(),),
                )
                processes.append(process)
                print(
                    f"GPU {gpu}: shard {shard}, batch {batch}, allocator cap {limit} GiB, PID {process.pid}",
                    flush=True,
                )
            while any(p.poll() is None for p in processes):
                if any(p.poll() not in (None, 0) for p in processes):
                    raise RuntimeError(
                        "A RefSeq worker failed; inspect gpu*.log. Completed chunks remain reusable."
                    )
                time.sleep(2)
            if any(p.returncode != 0 for p in processes):
                raise RuntimeError("A RefSeq worker failed; merge was not run")
        finally:
            # Only process groups created above; never discover or stop foreign jobs.
            live = [p for p in processes if p.poll() is None]
            for p in live:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + 10
            for p in live:
                try:
                    p.wait(timeout=max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait(timeout=10)
        from wet_lab.merge_query_shards import merge_sharded_results

        results = [
            Path(json.loads(p.with_suffix(".result.json").read_text())["result"]) for p in paths
        ]
        result = merge_sharded_results(results, OUTPUT / "merged")
        print(f"COMPLETE: {result}", flush=True)


def launch():
    preflight()
    with run_lock():
        if (
            subprocess.run(
                ["tmux", "has-session", "-t", f"={SESSION}"], capture_output=True
            ).returncode
            == 0
        ):
            raise RuntimeError(f"Session {SESSION} already exists")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    command = shlex.join(
        [
            "/usr/bin/env",
            "-u",
            "BASH_ENV",
            "-u",
            "ENV",
            "PYTHONUNBUFFERED=1",
            sys.executable,
            str(Path(__file__).resolve()),
            "--run",
        ]
    )
    log_path = OUTPUT / "pipeline.log"
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            SESSION,
            "-c",
            str(ROOT),
            f"exec {command} >> {shlex.quote(str(log_path))} 2>&1",
        ],
        check=True,
    )
    print(f"Detached launch requested: {SESSION}\nWatch: tail -f {log_path}")


def verify_restart_controller(pid):
    from scripts.stop_horizyn1_circe_v2 import read_process

    before = read_process(pid)
    if before is None or before.state == "Z":
        raise RuntimeError("The tmux controller has exited; retry after the session closes")
    proc = Path(f"/proc/{pid}")
    if proc.stat().st_uid != os.getuid():
        raise RuntimeError("Controller belongs to another user; no signals sent")
    argv = [os.fsdecode(v) for v in (proc / "cmdline").read_bytes().split(b"\0") if v]
    cwd = (proc / "cwd").resolve(strict=True)
    expected = Path(__file__).resolve()
    if len(argv) != 3 or (cwd / argv[1]).resolve() != expected or argv[2] != "--run":
        raise RuntimeError("tmux pane is not this RefSeq controller; no signals sent")
    after = read_process(pid)
    if after is None or after.started != before.started:
        raise RuntimeError("Controller PID changed during verification; no signals sent")
    return after


def restart():
    """Stop only the verified local screen; never launch over surviving workers."""
    from scripts.stop_horizyn1_circe_v2 import alive, process_tree, send_signal

    if socket.gethostname().split(".")[0] != "glutamine":
        raise RuntimeError("Run --restart on glutamine; no processes stopped or launched")
    session = subprocess.run(
        ["tmux", "has-session", "-t", f"={SESSION}"],
        capture_output=True,
        timeout=5,
    )
    if session.returncode == 0:
        panes = subprocess.check_output(
            ["tmux", "list-panes", "-s", "-t", f"={SESSION}", "-F", "#{pane_pid}"],
            text=True,
            timeout=5,
        ).splitlines()
        if len(panes) != 1 or not panes[0].isdigit():
            raise RuntimeError("Expected exactly one RefSeq controller pane; no signals sent")
        controller = verify_restart_controller(int(panes[0]))
        observed = process_tree(controller)
        print(
            f"Stopping verified RefSeq controller {controller.pid} and its workers only...",
            flush=True,
        )
        send_signal(controller, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while any(alive(p) for p in observed.values()) and time.monotonic() < deadline:
            time.sleep(0.2)
        remaining = [p.pid for p in observed.values() if alive(p)]
        if remaining:
            raise RuntimeError(
                f"Old RefSeq processes remain: {remaining}; replacement NOT launched"
            )
        print("Old screen stopped. Reaction features and completed chunks preserved.", flush=True)
    # launch() rechecks the shared lock, GPU headroom and duplicate session.
    launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", type=Path)
    mode.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker:
        worker(args.worker)
    elif args.launch:
        launch()
    elif args.restart:
        restart()
    else:
        run()
