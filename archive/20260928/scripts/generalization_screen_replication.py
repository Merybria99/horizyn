#!/usr/bin/env python3
"""Reproduce the frozen SLEEC screening recipe with two new training seeds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from generalization_clipzyme_ablation_watch import evaluate
from generalization_clipzyme_test_queue import run_job, write_json
from generalization_gpu_budget import free_memory_mib

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def invoke(script, args, gpu, log):
    with log.open("a") as stream:
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / script), *map(str,args)],
            cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4"),
            stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign",type=Path,required=True)
    p.add_argument("--resume-seed",type=int,help="Resume evaluation of a seed with a verified completed training checkpoint")
    a=p.parse_args()
    out=a.campaign.resolve();cross=out.parent
    plan=json.loads((out/"protocol.json").read_text())
    prior=json.loads((Path(plan["prerequisite"])/"protocol.json").read_text())
    while not all((Path(t["run_root"])/"train_complete.json").exists() for t in prior["tasks"]):
        time.sleep(10)
    def replicate(task):
        run=Path(task["run_root"]);gpu=task["gpu"]
        epoch=task.get("epoch",20)
        try:
            if sha(task["config"])!=task["config_sha256"]:
                raise ValueError("Replication config changed")
            checkpoint=run/f"checkpoints/screen_selection/screen-epoch={epoch-1:02d}.ckpt"
            if a.resume_seed is not None:
                receipt=json.loads((run/"train_complete.json").read_text())
                if receipt['checkpoint_sha256']!=sha(checkpoint):
                    raise ValueError('Completed training checkpoint changed before evaluation recovery')
                if (run/"phase2").exists():
                    raise ValueError('Evaluation recovery expects a failure before phase2; preserve partial phase2 separately')
                write_json(run/"evaluation_recovery.json",dict(started_utc=datetime.now(timezone.utc).isoformat(),
                    checkpoint_sha256=sha(checkpoint),training_repeated=False,original_failure=json.loads((run/"failure.json").read_text())))
            else:
                if task.get('prerequisite_receipt'):
                    while not Path(task['prerequisite_receipt']).exists():
                        time.sleep(10)
                while True:
                    if free_memory_mib(gpu)>=95000:break
                    time.sleep(10)
                write_json(run/"state.json",dict(stage="fresh_f3_training",seed=task["seed"],
                           started_utc=datetime.now(timezone.utc).isoformat()))
                invoke("train_protein_pooling_fast_io.py",["--config",task["config"],
                    "--io-mode","residue","--io-output-dir",run/"training","--io-prefetch",1],
                    gpu,run/"training.log")
                write_json(run/"train_complete.json",dict(checkpoint_sha256=sha(checkpoint),seed=task["seed"]))
            write_json(run/"state.json",dict(stage="full_library_base_validation",seed=task["seed"]))
            evaluate(run,epoch-1,SimpleNamespace(catalog=cross/"clipzyme_f3_catalog_v1",
                protocol=cross/"clipzyme_screening_evaluation_protocol_v2",
                manifest=cross/"clipzyme_manifests_v2/manifest.json",epochs=[epoch-1],gpus=[0,1,2,3]))
            phase2=run/"phase2";features=phase2/"features";features.mkdir(parents=True)
            source=cross/"sleec_multiview_phase2_v1/features"
            for name in ("catalog.json","pairs.npz","reaction_features.npz"):
                (features/name).symlink_to((source/name).resolve())
            shutil.copyfile(source/"protein_mean.h5",features/"protein_mean.h5")
            manifest=json.loads((source/"manifest.json").read_text())
            for key in ("checkpoint","f3_export_config","f3_residue_cache"):
                manifest.pop(key,None)
            manifest["raw_feature_reuse"]=dict(source_manifest=str(source/"manifest.json"),
                sha256=sha(source/"manifest.json"),purpose="Same raw train/dev inputs; learned vectors regenerated for this seed")
            manifest["sources"]["config"]=dict(path=task["config"],sha256=task["config_sha256"])
            write_json(features/"manifest.json",manifest)
            write_json(run/"state.json",dict(stage="phase2_feature_export",seed=task["seed"]))
            invoke("generalization_clipzyme_phase2_export.py",["--stage","f3",
                "--catalog",cross/"clipzyme_f3_catalog_v1","--config",task["config"],
                "--checkpoint",checkpoint,"--output",features,"--batch-size",128,
                "--residue-cache","/tmp/enzymediscovery_f3_20260920/train_validation_prott5.h5"],
                gpu,run/"phase2_export.log")
            write_json(run/"state.json",dict(stage="fixed_phase2_training",seed=task["seed"]))
            invoke("generalization_full_graph.py",["--features",features,"--output",phase2/"training",
                "--steps",100,"--snapshot-every",100,"--selection-method","external_screening",
                "--identity-weight",10,"--temperature",.2,"--contrastive-objective","positive_ce",
                "--seed",task["seed"],"--cpu-threads",4],gpu,run/"phase2_training.log")
            refiner=phase2/"training/step0100.pt"
            invoke("generalization_clipzyme_f3_validation.py",["--manifest",cross/"clipzyme_manifests_v2/manifest.json",
                "--catalog",cross/"clipzyme_f3_catalog_v1","--embeddings",run/f"screen_epoch{epoch-1}/validation_embeddings",
                "--output",phase2/"validation","--refiner",refiner,"--batch-size",64],
                gpu,run/"phase2_validation.log")
            # This is a fixed-recipe replication. Its test is reported even if
            # its validation score is below seed 42; there is no seed selection.
            write_json(run/"state.json",dict(stage="fixed_recipe_test",seed=task["seed"]))
            job=dict(label=f"replication_seed{task['seed']}",config=task["config"],
                checkpoint=str(checkpoint),refiner=str(refiner),output=str(phase2/"test"),
                protein_source=str(run/f"screen_epoch{epoch-1}/proteins"),gpu=gpu,
                policy=f"Fixed predeclared architecture, epoch{epoch} and phase2 step100; no seed selection")
            run_job(job,dict(created_utc=plan["created_utc"],catalog=str(cross/"clipzyme_f3_catalog_v1"),
                screening_protocol=str(cross/"clipzyme_screening_evaluation_protocol_v2")))
            if plan.get("semantic_alpha") is not None:
                write_json(run/"state.json",dict(stage="fixed_semantic_composition",seed=task["seed"]))
                write_json(phase2/"protocol.json",dict(base_config=task["config"],base_checkpoint=str(checkpoint),
                    base_checkpoint_sha256=sha(checkpoint),validation_summary=str(run/f"screen_epoch{epoch-1}/validation_evaluation/summary.json"),
                    seed=task["seed"],fixed_recipe=True,test_used_for_selection=False))
                write_json(phase2/"validation_selected.json",dict(selected=dict(evaluation=str(phase2/"validation/summary.json")),
                    selection="Fixed phase2 step100 recipe; no checkpoint or seed selection"))
                (phase2/"selected_test").symlink_to(phase2/"test",target_is_directory=True)
                invoke("generalization_clipzyme_phase2_dictionary.py",["--features",features,"--output",phase2/"anchors.pt"],
                    gpu,run/"dictionary.log")
                invoke("generalization_screen_phase2_compose.py",["--campaign",phase2,"--cross-root",cross,
                    "--fixed-alpha",plan["semantic_alpha"],"--fixed-cap",1],gpu,run/"composition.log")
            write_json(run/"complete.json",dict(seed=task["seed"],test=str(phase2/"test/test_evaluation/summary.json")))
            write_json(run/"state.json",dict(stage="complete",seed=task["seed"]))
        except Exception as exc:
            write_json(run/"failure.json",dict(error=repr(exc),seed=task["seed"]))
            raise
    tasks=[t for t in plan['tasks'] if a.resume_seed is None or t['seed']==a.resume_seed]
    if not tasks:raise ValueError('Requested recovery seed is not in the protocol')
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(replicate,tasks))
    if a.resume_seed is not None:
        write_json(out/f'recovery_seed{a.resume_seed}_complete.json',dict(seed=a.resume_seed,training_repeated=False))
    if all((Path(t['run_root'])/'complete.json').exists() for t in plan['tasks']):
        write_json(out/"complete.json",dict(seeds=[t["seed"] for t in plan["tasks"]],ensembles=False))


if __name__=="__main__":main()
