#!/usr/bin/env python3
"""Short phase-2 training, selected only by full-library screening validation."""
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

from generalization_clipzyme_test_queue import run_job, write_json

ROOT = Path(__file__).resolve().parents[1]
RECIPES = [
    dict(name="positive_ce", objective="positive_ce", temperature=.1, ranking=0.),
    dict(name="decoupled", objective="decoupled", temperature=.1, ranking=0.),
    dict(name="ranking", objective="positive_ce", temperature=.1, ranking=1.),
    dict(name="soft_ce", objective="positive_ce", temperature=.2, ranking=0.),
]
STEPS = [5, 20, 50, 100]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def invoke(script, arguments, gpu, log):
    with log.open("a") as stream:
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / script),
                        *map(str, arguments)], cwd=ROOT,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                                OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4"),
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if (out / "training_protocol.json").exists():
        raise FileExistsError("Use a fresh campaign")
    base = json.loads((out / "protocol.json").read_text())
    features = out / "features"
    if not (features / "complete.json").exists():
        raise ValueError("Frozen base feature export incomplete")
    cross = out.parent
    base_run = Path(base["base_config"]).parents[1]
    screen = Path(base["validation_summary"]).parent.parent
    if screen.parent != base_run:
        raise ValueError("Validation summary does not belong to the selected base run")
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        recipes=RECIPES, steps=STEPS, identity_weight=10., base=base,
        selection_metric="full_library_validation.table1.bedroc85",
        reported_metrics=["bedroc85", "bedroc20", "ef0.05", "ef0.1"],
        selection_includes_unmodified_base=True, test_used_for_selection=False,
        catalog=str(cross / "clipzyme_f3_catalog_v1"),
        screening_protocol=str(cross / "clipzyme_screening_evaluation_protocol_v2"),
        features_manifest_sha256=sha(features / "manifest.json"),
        source_sha256=sha(__file__), trainer_sha256=sha(ROOT / "scripts/generalization_full_graph.py"))
    write_json(out / "training_protocol.json", protocol)

    def arm(item):
        gpu, recipe = item
        run = out / recipe["name"]
        run.mkdir()
        # Each graph fits comfortably alongside a base trainer. Check the
        # available device memory before importing/loading the training job.
        while True:
            free = subprocess.check_output(["nvidia-smi", "--id", str(gpu),
                "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
            if int(free.strip()) >= 24000:
                break
            time.sleep(5)
        invoke("generalization_full_graph.py", ["--features", features,
            "--output", run / "training", "--steps", 100, "--snapshot-every", 5,
            "--selection-method", "external_screening", "--identity-weight", 10,
            "--temperature", recipe["temperature"], "--contrastive-objective", recipe["objective"],
            "--ranking-weight", recipe["ranking"], "--ranking-anchors", 64,
            "--cpu-threads", 4], gpu, run / "training.log")
        evaluations = []
        for step in STEPS:
            evaluation = run / f"validation_step{step}"
            invoke("generalization_clipzyme_f3_validation.py", [
                "--manifest", cross / "clipzyme_manifests_v2/manifest.json",
                "--catalog", cross / "clipzyme_f3_catalog_v1",
                "--embeddings", screen / "validation_embeddings", "--output", evaluation,
                "--refiner", run / f"training/step{step:04d}.pt", "--batch-size", 64,
                "--metric-workers", 8], gpu, run / f"validation_step{step}.log")
            evaluations.append(evaluation / "summary.json")
        return evaluations

    with ThreadPoolExecutor(max_workers=4) as pool:
        blocks = list(pool.map(arm, enumerate(RECIPES)))
    evaluations = [screen / "validation_evaluation/summary.json", *[p for block in blocks for p in block]]
    selection = out / "validation_selected.json"
    invoke("generalization_clipzyme_checkpoint_select.py",
           [v for path in evaluations for v in ("--evaluation", path)] + ["--output", selection],
           1, out / "selection.log")
    selected = json.loads(selection.read_text())
    write_json(out / "validation_comparison.json", dict(
        rows=[dict(path=str(path), **json.loads(path.read_text())) for path in evaluations],
        selection=selected))
    if selected["selected_index"] == 0:
        write_json(out / "complete.json", dict(base_retained=True,
            reason="No phase-2 snapshot improved full-library validation BEDROC85", test_run=False))
        return
    result = json.loads(Path(selected["selected"]["evaluation"]).read_text())
    refiner = result["refiner"]
    if sha(refiner["path"]) != refiner["sha256"]:
        raise ValueError("Selected phase-2 weights changed")
    job = dict(label="validation_selected_phase2", config=base["base_config"],
        checkpoint=base["base_checkpoint"], refiner=refiner["path"],
        output=str(out / "selected_test"), protein_source=str(screen / "proteins"),
        reaction_source=str(screen / "requested_test/test_embeddings"), gpu=1,
        policy="One checkpoint selected from the fixed phase-2 grid by full-library validation BEDROC85")
    write_json(out / "selected_test_plan.json", dict(job=job, selection_sha256=sha(selection)))
    run_job(job, protocol)
    write_json(out / "complete.json", dict(base_retained=False, test_run=True,
        selected_test=str(out / "selected_test/test_evaluation/summary.json")))


if __name__ == "__main__":
    main()
