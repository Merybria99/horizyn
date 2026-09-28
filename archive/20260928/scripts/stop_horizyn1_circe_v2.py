#!/usr/bin/env python3
"""Stop only a verified CIRCE-v2 controller and its observed descendants.

Run this on the machine running the job, never by applying a shared PID file
blindly on a different node. No run artifacts or tmux sessions are removed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    started: int
    state: str


def read_process(pid: int) -> Process | None:
    try:
        record = Path(f"/proc/{pid}/stat").read_text()
        fields = record[record.rfind(")") + 2:].split()
        return Process(pid, int(fields[1]), int(fields[19]), fields[0])
    except (FileNotFoundError, ProcessLookupError):
        return None


def option_value(argv: list[str], name: str) -> str | None:
    result = None
    for i, arg in enumerate(argv):
        if arg == name:
            if i + 1 == len(argv):
                raise RuntimeError(f"Malformed controller argument: {name}")
            result = argv[i + 1]
        elif arg.startswith(name + "="):
            result = arg.split("=", 1)[1]
    return result


def resolve_process_path(value: str, cwd: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else cwd / path).resolve()


def verify_controller(pid: int, run_root: Path) -> Process | None:
    before = read_process(pid)
    if before is None or before.state == "Z":
        return None
    proc = Path(f"/proc/{pid}")
    if proc.stat().st_uid != os.getuid():
        raise RuntimeError("PID belongs to another user; refusing to signal it")
    argv = [value.decode() for value in (proc / "cmdline").read_bytes().split(b"\0") if value]
    cwd = (proc / "cwd").resolve(strict=True)
    controller_names = ("horizyn1_circe_v2_pipeline.py", "horizyn1_circe_v2_reaction_holdout.py",
                        "horizyn1_parallel_reactions.py", "train_protein_pooling_fast_io.py")
    controllers = {(ROOT / "scripts" / name).resolve() for name in controller_names}
    matches = [i for i, arg in enumerate(argv)
               if arg.endswith(controller_names)
               and resolve_process_path(arg, cwd) in controllers]
    if len(matches) != 1:
        raise RuntimeError("Shared PID does not identify this pipeline on this machine; no signals sent")
    arguments = argv[matches[0] + 1:]
    if Path(argv[matches[0]]).name == "train_protein_pooling_fast_io.py":
        configured = option_value(arguments, "--io-output-dir")
        if configured is None or resolve_process_path(configured, cwd) != run_root.resolve():
            raise RuntimeError("Training PID belongs to a different RUN_ROOT; no signals sent")
        after = read_process(pid)
        if after is None or after.started != before.started:
            raise RuntimeError("Controller PID changed during verification; retry safely")
        return after
    if Path(argv[matches[0]]).name == "horizyn1_parallel_reactions.py" and (
            not arguments or arguments[0] != "resume"):
        raise RuntimeError("Parallel extraction is not a main-run resume controller; no signals sent")
    # Read only the relevant environment field; never print the process environment.
    run_env = profile_env = None
    for item in (proc / "environ").read_bytes().split(b"\0"):
        if item.startswith(b"RUN_ROOT="):
            run_env = os.fsdecode(item[len(b"RUN_ROOT="):])
        elif item.startswith(b"CIRCE_PROFILE="):
            profile_env = os.fsdecode(item[len(b"CIRCE_PROFILE="):])
    profile = option_value(arguments, "--profile") or profile_env or "base"
    default = ROOT / "runs" / ("horizyn1_circe_v2_h200" if profile == "h200" else "horizyn1_circe_v2")
    if Path(argv[matches[0]]).name in ("horizyn1_circe_v2_reaction_holdout.py",
                                      "horizyn1_parallel_reactions.py"):
        default = ROOT / "runs" / f"horizyn1_circe_v2_reaction_holdout_{profile}"
    configured = option_value(arguments, "--run-root") or run_env or str(default)
    if resolve_process_path(configured, cwd) != run_root.resolve():
        raise RuntimeError("PID belongs to a different RUN_ROOT; no signals sent")
    after = read_process(pid)
    if after is None or after.started != before.started:
        raise RuntimeError("Controller PID changed during verification; retry safely")
    return after


def process_tree(root: Process) -> dict[int, Process]:
    processes = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            process = read_process(int(entry.name))
        except (FileNotFoundError, PermissionError):
            continue
        if process is not None:
            processes[process.pid] = process
    result = {root.pid: root}
    changed = True
    while changed:
        changed = False
        for pid, process in processes.items():
            if pid not in result and process.ppid in result:
                result[pid] = process
                changed = True
    return result


def alive(process: Process) -> bool:
    current = read_process(process.pid)
    return current is not None and current.started == process.started and current.state != "Z"


def send_signal(process: Process, signum: int) -> None:
    if alive(process):
        try:
            os.kill(process.pid, signum)
        except ProcessLookupError:
            pass


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--pid", type=int, default=None,
                        help="Verify this local PID against RUN_ROOT instead of reading pipeline.pid")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--force-after-timeout", action="store_true",
                        help="SIGKILL only verified surviving job processes after TERM times out")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 60:
        parser.error("--timeout must be between 0 and 60 seconds")
    run_root = args.run_root.resolve()
    pid_file = run_root / "pipeline.pid"
    if args.pid is None and not pid_file.is_file():
        print(f"No controller PID file: {pid_file}; no processes signalled", flush=True)
        return
    pid = args.pid if args.pid is not None else int(pid_file.read_text().strip())
    if pid <= 1 or pid == os.getpid():
        raise RuntimeError("Invalid controller PID; refusing to signal it")
    controller = verify_controller(pid, run_root)
    if controller is None:
        print("Recorded controller is not running on this machine; no processes signalled", flush=True)
        return
    observed = process_tree(controller)
    print(f"Verified controller {pid}; {len(observed) - 1} descendant processes; RUN_ROOT={run_root}", flush=True)
    if args.dry_run:
        print("Dry run: no signals sent", flush=True)
        return
    # Let the controller forward TERM to its own subprocess groups first.
    send_signal(controller, signal.SIGTERM)
    for process in observed.values():
        if process.pid != pid:
            send_signal(process, signal.SIGTERM)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        if not any(alive(process) for process in observed.values()):
            print("Pipeline stopped. All run files were preserved.", flush=True)
            return
        time.sleep(.2)
    remaining = [process for process in observed.values() if alive(process)]
    if args.force_after_timeout:
        print(f"TERM timed out; sending KILL to {len(remaining)} verified surviving job processes", flush=True)
        for process in remaining:
            send_signal(process, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(alive(process) for process in remaining):
            time.sleep(.2)
        remaining = [process for process in remaining if alive(process)]
    if remaining:
        raise SystemExit(f"Processes still present: {[p.pid for p in remaining]}; do not launch a replacement yet")
    print("Pipeline stopped. All run files were preserved.", flush=True)


if __name__ == "__main__":
    main()
