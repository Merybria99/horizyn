#!/usr/bin/env python3
"""Run the preregistered frozen variants across benchmark/external panels.

One worker per GPU; jobs are labels-free feature inference. Completed outputs
are reused only when their bundle, catalog and score hashes match. Evaluation
of the produced immutable score files is deliberately a separate operation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--variants", nargs="+", default=["seed42", "seed17", "seed73", "ensemble", "graph_only", "anchors_blend", "anchors_only"])
    args = p.parse_args()
    campaign = args.campaign.resolve()
    frozen = campaign / "frozen_recipe.json"
    if not json.loads(frozen.read_text()).get("frozen_before_held_out_evaluation"):
        raise ValueError("Recipe must be frozen before prediction")
    panels = {}
    for split in ("reaction_smi", "time", "enzyme_smi"):
        source = campaign / f"features_test_{split}"
        panels[split] = dict(split=split, base=source / "f3_features.npz",
            protein_means=source / "protein_mean.h5", reaction_features=source / "reaction_features.npz",
            catalog=source / "catalog.json", reaction_key="reactions")
    for name, source in (("case1", campaign / "case1_audit/features"),
                         ("p450", campaign / "p450_audit/features")):
        panels[name] = dict(split="reaction_smi", base=source / "f3_epoch29/features.npz",
            protein_means=source / "raw/protein_mean.h5", reaction_features=source / "raw/reaction_features.npz",
            catalog=source / "catalog.json", reaction_key="query_matched" if name == "case1" else "reactions")
    pending = queue.Queue()
    all_jobs = []
    for name, panel in panels.items():
        for variant in args.variants:
            bundle = campaign / "models" / panel["split"] / variant / "bundle.json"
            output = campaign / "predictions" / name / variant
            job = dict(panel=name, variant=variant, bundle=str(bundle), output=str(output),
                       **{key: str(value) for key, value in panel.items()})
            all_jobs.append(job)
            pending.put(job)
    results = []
    started = time.monotonic()
    def worker(gpu):
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            output = Path(job["output"])
            output.mkdir(parents=True, exist_ok=True)
            receipt = output / "complete.json"
            try:
                if receipt.exists():
                    saved = json.loads(receipt.read_text())
                    if saved["bundle"]["sha256"] != sha(job["bundle"]) or saved["inputs"]["catalog"]["sha256"] != sha(job["catalog"]) or saved["output_sha256"] != sha(output / "scores.npz"):
                        raise ValueError("Completed prediction provenance mismatch")
                    result = dict(**job, status="reused", gpu=gpu)
                else:
                    command = [args.python, str(ROOT / "scripts/generalization_predict.py"),
                               "--bundle", job["bundle"], "--output", str(output),
                               "--device", f"cuda:{gpu}", "--reaction-key", job["reaction_key"]]
                    for key in ("base", "catalog", "protein_means", "reaction_features"):
                        command.extend(["--" + key.replace("_", "-"), job[key]])
                    if job["variant"] == "seed42":
                        command.append("--save-embeddings")
                    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
                    with (output / "run.log").open("w") as log:
                        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                    result = dict(**job, status="complete", gpu=gpu)
            except Exception as error:
                result = dict(**job, status="failed", gpu=gpu, error=str(error))
            results.append(result)
            print(json.dumps({k: result[k] for k in ("panel", "variant", "status", "gpu")}), flush=True)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        list(pool.map(worker, args.gpus))
    output = dict(schema="frozen_prediction_index_v1", freeze=dict(path=str(frozen), sha256=sha(frozen)),
                  jobs=results, elapsed_seconds=time.monotonic() - started,
                  source_sha256=sha(__file__))
    (campaign / "prediction_index.json").write_text(json.dumps(output, indent=2) + "\n")
    if any(row["status"] == "failed" for row in results):
        raise SystemExit("Some predictions failed; inspect per-output logs and prediction_index.json")


if __name__ == "__main__":
    main()
