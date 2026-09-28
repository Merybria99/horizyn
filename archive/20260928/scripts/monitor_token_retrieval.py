#!/usr/bin/env python3
"""Read-only monitoring of a launched B1/B2 campaign until both jobs finish."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone


def atomic_write(path, text):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def process_running(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] not in {"Z", "X"}
    except FileNotFoundError:
        return False


def snapshot_run(launch):
    directory = Path(launch["run_dir"])
    status = read_json(directory / "status.json", {})
    manifest = read_json(directory / "run_manifest.json", {})
    records = []
    try:
        for line in (directory / "steps.jsonl").read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A writer may be appending its last line.
    except FileNotFoundError:
        pass
    last = records[-1] if records else {}
    recent = records[-20:]
    epochs = []
    for path in (directory / "evaluation").glob("validation_epoch*.json"):
        item = read_json(path)
        if item is not None and item.get("evaluation_implementation") == "circe_v2_shared_v1":
            epochs.append(item)
    epochs.sort(key=lambda item: item["epoch"])
    best = max(epochs, key=lambda item: item["balanced_first_positive_mrr"], default=None)
    latest = epochs[-1] if epochs else None
    alive = process_running(launch["launcher_pid"])
    phase = status.get("state", "starting")
    steps_per_epoch = manifest.get("steps_per_epoch")
    if (phase in {"training", "preparing_batch"} and last
            and last["batch"] + 1 == steps_per_epoch
            and (latest is None or latest["epoch"] < last["epoch"])):
        phase = "validating"
    observed_paths = [directory / "status.json", directory / "steps.jsonl", directory / "train.log"]
    ages = [time.time() - path.stat().st_mtime for path in observed_paths if path.exists()]
    silence = min(ages, default=0)
    alerts = []
    if phase == "failed" or list(directory.glob("failure_rank*.json")):
        alerts.append("Run reported a failure; inspect train.log and failure_rank*.json.")
    if not alive and phase != "complete":
        alerts.append("Supervisor exited before completion.")
    if alive and silence > 600:
        alerts.append(f"No log or status updates for {silence:.0f} seconds; inspect for a stall.")
    if any(not math.isfinite(row["loss"]) for row in records):
        alerts.append("A non-finite loss was recorded.")
    return {
        "variant": launch["variant"], "gpus": launch["gpus"], "pid": launch["launcher_pid"],
        "alive": alive, "phase": phase, "epoch": status.get("epoch", last.get("epoch")),
        "completed_steps": len(records), "steps_per_epoch": steps_per_epoch,
        "last_step": last, "activity_age_seconds": silence,
        "recent_mean_loss": statistics.mean(row["loss"] for row in recent) if recent else None,
        "recent_mean_step_seconds": statistics.mean(row["step_seconds"] for row in recent) if recent else None,
        "latest_validation": latest, "best_validation": best,
        "test": read_json(directory / "evaluation/test_both.json"),
        "checkpoints": [path.name for path in (directory / "checkpoints").glob("*.pt")],
        "alerts": alerts, "run_dir": str(directory),
    }


def markdown(report):
    lines = ["# B1/B2 live monitoring", "", f"Updated: {report['utc']} (UTC).",
             "", "Checks run every 30 seconds. This monitor observes jobs and does not restart or modify them.",
             "", "| Run | GPUs | Phase | Epoch (1-based) | Completed steps | Recent loss | Seconds/step |",
             "|---|---|---|---:|---:|---:|---:|"]
    for run in report["runs"]:
        epoch = run["epoch"] + 1 if run["epoch"] is not None else "—"
        loss = f"{run['recent_mean_loss']:.4f}" if run["recent_mean_loss"] is not None else "—"
        duration = f"{run['recent_mean_step_seconds']:.2f}" if run["recent_mean_step_seconds"] is not None else "—"
        lines.append(f"| {run['variant']} | {run['gpus']} | {run['phase']} | {epoch} | {run['completed_steps']} | {loss} | {duration} |")
    lines += ["", "Loss and timing averages use the most recent 20 steps. Losses alone do not establish retrieval improvement.",
              "", "| Run | Latest validation epoch (1-based) | R→E first-positive MRR | E→R first-positive MRR | Mean first-positive MRR | Mean all-positive MRR |",
              "|---|---:|---:|---:|---:|---:|"]
    for run in report["runs"]:
        val = run["latest_validation"]
        if val:
            lines.append(f"| {run['variant']} | {val['epoch']+1} | {val['reaction_to_enzyme/first_positive_mrr']:.4f} | "
                         f"{val['enzyme_to_reaction/first_positive_mrr']:.4f} | {val['balanced_first_positive_mrr']:.4f} | "
                         f"{val['mean_bidirectional_reactzyme_mrr']:.4f} |")
        else:
            lines.append(f"| {run['variant']} | Pending | — | — | — | — |")
    lines += ["", "First-positive MRR selects checkpoints; all-positive MRR is a separate reported metric.", ""]
    alerts = [f"{run['variant']}: {alert}" for run in report["runs"] for alert in run["alerts"]]
    if report["disk_free_gib"] < 100:
        alerts.append("Less than 100 GiB free on the local cache filesystem.")
    lines += ["## Alerts", "", *([f"- {item}" for item in alerts] or ["No alerts."]),
              "", f"Local cache filesystem free: {report['disk_free_gib']:.1f} GiB.", "",
              "```text", report["gpu_status"].strip(), "```", ""]
    if report["all_terminal"]:
        lines += ["Both jobs have finished or exited; monitoring has stopped.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    campaign = Path(args.campaign_dir).resolve()
    launch = read_json(campaign / "launches.json")
    if not launch:
        raise ValueError("Campaign launches.json is missing or invalid")
    output = campaign / "monitor"
    output.mkdir(exist_ok=True)
    lock = (output / "monitor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_write(output / "monitor.pid", f"{os.getpid()}\n")
    while True:
        launch = read_json(campaign / "launches.json", launch)
        runs = [snapshot_run(item) for item in launch["launches"]]
        try:
            gpus = subprocess.check_output([
                "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu", "--format=csv"
            ], text=True, timeout=15)
        except (subprocess.SubprocessError, OSError) as error:
            gpus = f"GPU query failed: {error}"
        report = {"utc": datetime.now(timezone.utc).isoformat(), "runs": runs,
                  "disk_free_gib": shutil.disk_usage("/tmp").free / 2**30, "gpu_status": gpus,
                  "all_terminal": all(run["phase"] in {"complete", "failed"} or not run["alive"] for run in runs)}
        atomic_write(output / "latest.json", json.dumps(report, indent=2) + "\n")
        atomic_write(output / "status.md", markdown(report))
        with (output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(report) + "\n")
        print(json.dumps({"utc": report["utc"], "runs": [
            {key: run[key] for key in ["variant", "phase", "completed_steps", "alerts"]} for run in runs]}), flush=True)
        if report["all_terminal"] or args.once:
            break
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    main()
