#!/usr/bin/env python3
"""Run explicitly requested intermediate tests once immutable inputs are ready."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_job(job, protocol):
    output = Path(job["output"])
    embeddings = output / "test_embeddings"
    embeddings.mkdir(parents=True, exist_ok=True)
    write_json(output / "test_request.json", dict(job=job, requested_utc=protocol["created_utc"],
        intermediate=True, test_requested_by_user=True, selection_used_test_scores=False,
        policy=job["policy"], repeated_test_evaluation=True))
    for source in Path(job["protein_source"]).glob("protein_*"):
        target = embeddings / source.name
        if not target.exists():
            target.symlink_to(source.resolve())
    if job.get("reaction_source"):
        for name in ("reaction_embeddings.npy", "query_ids.txt", "reaction_receipt.json", "screen_head.pt"):
            source = Path(job["reaction_source"]) / name
            if source.exists() and not (embeddings / name).exists():
                (embeddings / name).symlink_to(source.resolve())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job.get("gpu", 3)),
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")

    def invoke(script, arguments, name):
        with (output / f"{name}.log").open("a") as log:
            subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / script), *map(str, arguments)],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    shared = ["--catalog", protocol["catalog"], "--protocol", protocol["screening_protocol"],
              "--config", job["config"], "--checkpoint", job["checkpoint"], "--output", embeddings]
    if not (embeddings / "reaction_receipt.json").exists():
        invoke("generalization_clipzyme_f3_screen.py", [*shared, "--phase", "reactions", "--scope", "test",
               "--batch-size", 256, *(["--mask-chemistry"] if job.get("mask_chemistry") else [])], "reactions")
    if not (embeddings / "score_receipt.json").exists():
        invoke("generalization_clipzyme_f3_screen.py", [*shared, "--phase", "scores", "--batch-size", 64,
               *(["--refiner", job["refiner"]] if job.get("refiner") else []),
               *(["--fusion-calibration", job["fusion_calibration"]] if job.get("fusion_calibration") else [])], "scores")
    if not (output / "test_evaluation/summary.json").exists():
        invoke("generalization_clipzyme_screening_evaluate.py", ["--scores", embeddings / "scores.npy",
               "--query-ids", embeddings / "query_ids.txt", "--candidate-ids", embeddings / "candidate_ids.txt",
               "--protocol", protocol["screening_protocol"], "--output", output / "test_evaluation"], "evaluate")
    write_json(output / "complete.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                test_requested_by_user=True, intermediate=True, selection_used_test_scores=False))


def ready(job):
    source = Path(job["protein_source"])
    return (Path(job["checkpoint"]).exists() and all(
        (source / f"protein_rank{rank:02d}_of_04.receipt.json").exists() for rank in range(4)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--hours", type=float, default=8)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    started = time.monotonic()
    active, failures = {}, {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        while time.monotonic() - started < 3600 * a.hours:
            for label, future in list(active.items()):
                if future.done():
                    try:
                        future.result()
                        print(json.dumps(dict(completed=label)), flush=True)
                    except Exception as exc:
                        failures[label] = repr(exc)
                        print(json.dumps(dict(failed=label, error=repr(exc))), flush=True)
                    del active[label]
            completed = []
            for job in plan["jobs"]:
                label = job["label"]
                if (Path(job["output"]) / "complete.json").exists():
                    completed.append(label)
                elif label not in failures and label not in active and len(active) < 2 and ready(job):
                    print(json.dumps(dict(starting=label)), flush=True)
                    active[label] = pool.submit(run_job, job, plan)
            write_json(a.plan.parent / "status.json", dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                completed=completed, active=list(active), failures=failures,
                waiting=[j["label"] for j in plan["jobs"] if j["label"] not in completed + list(active) + list(failures)]))
            if len(completed) + len(failures) == len(plan["jobs"]):
                return
            time.sleep(15)


if __name__ == "__main__":
    main()
