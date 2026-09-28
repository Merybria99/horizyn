#!/usr/bin/env python3
"""Validate predeclared F3 snapshots as they arrive; never open test labels."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def invoke(script, arguments, log, gpu):
    command = [sys.executable, "-u", str(ROOT / "scripts" / script), *map(str, arguments)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    with log.open("a") as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def evaluate(arm, epoch, args):
    checkpoint = arm / "checkpoints/screen_selection" / f"screen-epoch={epoch:02d}.ckpt"
    config = arm / "configs/train.yaml"
    output = arm / f"screen_epoch{epoch}"
    proteins, embeddings = output / "proteins", output / "validation_embeddings"
    for path in (output, proteins, embeddings):
        path.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "watch_request.json", dict(
        created_utc=datetime.now(timezone.utc).isoformat(), checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint), config_sha256=sha(config),
        epoch_zero_based=epoch, scopes=["validation"], test_labels_read=False,
        selection_metric="validation.table1.bedroc85", checkpoint_grid=args.epochs,
        gpu_protein_export=args.gpus, protein_batch_size=512, source_sha256=sha(Path(__file__))))
    shared = ["--catalog", args.catalog, "--protocol", args.protocol,
              "--config", config, "--checkpoint", checkpoint]

    def protein_rank(rank):
        stem = f"protein_rank{rank:02d}_of_04"
        if not (proteins / f"{stem}.receipt.json").exists():
            invoke("generalization_clipzyme_f3_screen.py", [*shared, "--phase", "proteins",
                   "--output", proteins, "--rank", rank, "--world-size", 4,
                   "--shard-block-size", 4096, "--batch-size", 512],
                   output / f"protein_rank{rank}.log", args.gpus[rank])

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(protein_rank, range(4)))
    for path in proteins.glob("protein_*"):
        target = embeddings / path.name
        if not target.exists():
            target.symlink_to(path.resolve())
    if not (embeddings / "reaction_receipt.json").exists():
        invoke("generalization_clipzyme_f3_screen.py", [*shared, "--phase", "reactions",
               "--scope", "validation", "--output", embeddings, "--batch-size", 256],
               output / "validation_reactions.log", args.gpus[0])
    evaluation = output / "validation_evaluation"
    if not (evaluation / "summary.json").exists():
        invoke("generalization_clipzyme_f3_validation.py", ["--manifest", args.manifest,
               "--catalog", args.catalog, "--embeddings", embeddings, "--output", evaluation,
               "--batch-size", 64], output / "validation.log", args.gpus[0])
    atomic_json(output / "watch_complete.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                validation_summary=str(evaluation / "summary.json"), test_labels_read=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", type=Path, action="append", required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, nargs="+", default=[19, 39, 59, 79, 99])
    p.add_argument("--gpus", type=int, nargs=4, default=[0, 1, 2, 3])
    p.add_argument("--max-hours", type=float, default=8)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    failed = {}
    # Earlier fixed epochs get first attention across all arms.
    while time.monotonic() - started < a.max_hours * 3600:
        pending, completed = [], []
        for epoch in a.epochs:
            for arm in a.arm:
                key = f"{arm.name}/epoch{epoch}"
                result = arm / f"screen_epoch{epoch}/validation_evaluation/summary.json"
                checkpoint = arm / "checkpoints/screen_selection" / f"screen-epoch={epoch:02d}.ckpt"
                if result.exists():
                    validation = json.loads(result.read_text())
                    completed.append(dict(arm=str(arm), epoch=epoch,
                        bedroc85=validation["selection_value"],
                        summary={table: {metric: validation["summary"][table][metric]
                                         for metric in ("bedroc85", "bedroc20", "ef0.05", "ef0.1")}
                                 for table in ("table1", "table2")}))
                elif key not in failed and checkpoint.exists() and time.time() - checkpoint.stat().st_mtime > 30:
                    pending.append((arm, epoch))
        state = dict(updated_utc=datetime.now(timezone.utc).isoformat(), completed=completed,
                     failed=failed, ready=[dict(arm=str(p), epoch=e) for p, e in pending],
                     test_labels_read=False)
        atomic_json(a.output / "status.json", state)
        for arm in a.arm:
            evaluations = [arm / f"screen_epoch{epoch}/validation_evaluation/summary.json" for epoch in a.epochs]
            selection = arm / "validation_selected.json"
            if all(path.exists() for path in evaluations) and not selection.exists():
                invoke("generalization_clipzyme_checkpoint_select.py",
                       [item for path in evaluations for item in ("--evaluation", path)] + ["--output", selection],
                       a.output / f"selection_{arm.name}.log", a.gpus[0])
        if len(completed) + len(failed) == len(a.arm) * len(a.epochs):
            atomic_json(a.output / "complete.json", state)
            return
        if pending:
            arm, epoch = pending[0]
            key = f"{arm.name}/epoch{epoch}"
            print(json.dumps(dict(starting=key)), flush=True)
            try:
                evaluate(arm, epoch, a)
            except Exception as exc:
                failed[key] = repr(exc)
                print(json.dumps(dict(failed=key, error=repr(exc))), flush=True)
            continue
        time.sleep(30)
    atomic_json(a.output / "deadline.json", dict(completed=False, reason="watch time limit"))


if __name__ == "__main__":
    main()
