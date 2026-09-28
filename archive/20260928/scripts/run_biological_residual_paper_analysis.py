#!/usr/bin/env python3
"""Evaluate existing residual checkpoints consistently; never train or extract backbones."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.biological_residual_paper_analysis import SPLITS, digest, pair_inventory

DEFAULT_RUN = ROOT / "runs/biological_residual_paper_analysis_v1"
ANALYSIS_PYTHON = ROOT.parent / ".capability-run-py/bin/python"
EVALUATOR = ROOT / "scripts/evaluate_biological_residual.py"
ANALYSIS = ROOT / "scripts/biological_residual_paper_analysis.py"


def file_record(path, content=False):
    path = Path(path).resolve()
    stat = path.stat()
    result = dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if content:
        result["sha256"] = digest(path)
    return result


def plans(project=ROOT):
    result = {}
    for split in SPLITS:
        run = project / "runs" / f"biological_residual_{split}"
        protocol = project / "data/revised_protocols/reactzyme_paper" / split
        config = (
            project / "configs/reactzyme_reaction_smi_biological_residual.yaml"
            if split == "reaction_smi"
            else run / "configs/reactzyme_finetune.yaml"
        )
        result[split] = dict(
            checkpoint=run / "checkpoints/reactzyme_finetune/last.ckpt",
            parent=project
            / "runs/reactzyme_reaction_features_v1/checkpoints"
            / split
            / "F3_set_chemistry/protein-pooling-epoch=29.ckpt",
            config=config,
            test_config=project
            / "runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry"
            / split
            / "test.yaml",
            protocol=protocol,
            run=run,
            tokens=project
            / yaml.safe_load(config.read_text())["data"]["protein_functional_tokens_path"],
        )
    return result


def validate_parent_weights(parent, residual):
    import torch

    base = torch.load(parent, map_location="cpu", weights_only=False, mmap=True)["state_dict"]
    full = torch.load(residual, map_location="cpu", weights_only=False, mmap=True)["state_dict"]
    names = [
        k for k in base if k.startswith("model.") and not k.startswith("model.biological_residual.")
    ]
    if not names:
        raise ValueError("Parent has no model weights")
    for key in names:
        if key not in full or not torch.equal(base[key], full[key]):
            raise ValueError(f"Residual differs from frozen F3 parent: {key}")
    return len(names)


def validate_record(record):
    current = file_record(record["path"], content="sha256" in record)
    for key in ("size", "sha256" if "sha256" in record else "mtime_ns"):
        if key in record and current[key] != record[key]:
            raise ValueError(f"Recorded input changed: {record['path']} ({key})")


def preflight(jobs):
    import h5py
    import numpy
    import torch
    from scripts.evaluate_biological_residual import validate_exact_base_cache

    snapshot = {
        "schema": "biological_residual_paper_analysis_v1",
        "policy": {
            "selection": "balanced_reactzyme_mrr",
            "alphas": [0, 0.025, 0.05, 0.075, 0.1],
            "max_directional_drop": 0.005,
            "checkpoint": "fixed existing final residual / F3 epoch 29",
            "test_previously_inspected": True,
        },
        "splits": {},
    }
    for split, job in jobs.items():
        print(
            f"Preflight: {split}, frozen F3 identity, pair files and cache provenance", flush=True
        )
        count = validate_parent_weights(job["parent"], job["checkpoint"])
        inputs = {
            job["checkpoint"],
            job["parent"],
            job["config"],
            job["test_config"],
            job["tokens"],
        }
        train_pairs = job["protocol"] / "train_pairs.csv"
        train_edges = pair_inventory(train_pairs)[0]
        for subset, config in (("validation", job["config"]), ("test", job["test_config"])):
            cache = (
                job["run"]
                / "cache"
                / ("reactzyme_validation_eval" if subset == "validation" else "reactzyme_test")
            )
            manifest_path = cache / "manifest.json"
            signature = json.loads(manifest_path.read_text())["signature"]
            recorded = signature["inputs"]
            if Path(recorded["checkpoint"]["path"]).resolve() != job["parent"].resolve():
                raise ValueError("Cache belongs to a different F3 checkpoint")
            validate_record(recorded["checkpoint"])
            if recorded["config_sha256"] != digest(config):
                raise ValueError(f"Cache configuration changed: {config}")
            for field, filename in (
                ("train_pairs", "train_pairs.csv"),
                ("train_reactions", "train_rxns.csv"),
                ("validation_pairs", f"{subset}_pairs.csv"),
                ("validation_reactions", f"{subset}_rxns.csv"),
                ("validation_candidates", f"{subset}_candidate_ids.txt"),
            ):
                actual = job["protocol"] / filename
                # Validation caches may derive candidates from positive pairs
                # rather than recording an explicit candidate-ID file.
                if recorded[field] is None and field == "validation_candidates":
                    inputs.add(actual)
                    continue
                if recorded[field] is None or recorded[field]["sha256"] != digest(actual):
                    raise ValueError(f"Cache/protocol contents disagree: {actual}")
                inputs.add(actual)
            for record in recorded["reaction_feature_inputs"].values():
                if record is not None:
                    validate_record(record)
                    inputs.add(Path(record["path"]))
            edges, proteins, reactions = pair_inventory(job["protocol"] / f"{subset}_pairs.csv")
            if edges & train_edges:
                raise ValueError("Evaluation pairs overlap optimization pairs")
            candidates = (job["protocol"] / f"{subset}_candidate_ids.txt").read_text().splitlines()
            if len(candidates) != len(set(candidates)) or set(candidates) != set(proteins):
                raise ValueError(
                    "Candidate universe must equal unique evaluation-positive proteins"
                )
            inputs.add(manifest_path)
            for filename, expected_ids in (
                ("enzyme_base.h5", set(proteins)),
                ("validation_reaction_base.h5", {r + "_f" for r in reactions}),
            ):
                path = cache / filename
                validate_exact_base_cache(path)
                with h5py.File(path, "r") as handle:
                    keys = {
                        v.decode() if isinstance(v, bytes) else str(v) for v in handle["ids"][:]
                    }
                    if not expected_ids <= keys:
                        raise ValueError(f"Missing cached entities: {path}")
                    handle["vectors"][0]
                inputs.add(path)
        source = job["run"] / "data/source_pretrain"
        source_manifest = json.loads((source / "manifest.json").read_text())
        for filename, field in (
            ("train_pairs.csv", "output_pairs_sha256"),
            ("train_rxns.csv", "output_reactions_sha256"),
        ):
            if digest(source / filename) != source_manifest[field]:
                raise ValueError("Source training graph changed")
            inputs.add(source / filename)
        inputs.add(source / "manifest.json")
        with h5py.File(job["tokens"], "r") as handle:
            handle["token_vectors"][0]
        snapshot["splits"][split] = dict(
            frozen_tensors_verified=count,
            inputs=[file_record(path, content=path.suffix != ".h5") for path in sorted(inputs)],
        )
    code = [
        EVALUATOR,
        ANALYSIS,
        Path(__file__),
        ROOT / "scripts/report_biological_residual_controls.py",
    ]
    code.extend(sorted((ROOT / "horizyn").rglob("*.py")))
    snapshot["code_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in code}
    snapshot["metadata_python"] = str(ANALYSIS_PYTHON)
    snapshot["runtime"] = dict(
        python=sys.version.split()[0],
        torch=torch.__version__,
        numpy=numpy.__version__,
        h5py=h5py.__version__,
    )
    return snapshot


def command_for(job, subset, directory):
    cache = (
        job["run"]
        / "cache"
        / ("reactzyme_validation_eval" if subset == "validation" else "reactzyme_test")
    )
    command = [
        sys.executable,
        str(EVALUATOR),
        "--checkpoint",
        str(job["checkpoint"]),
        "--config",
        str(job["config"] if subset == "validation" else job["test_config"]),
        "--pairs",
        str(job["protocol"] / f"{subset}_pairs.csv"),
        "--reactions",
        str(job["protocol"] / f"{subset}_rxns.csv"),
        "--candidate-ids",
        str(job["protocol"] / f"{subset}_candidate_ids.txt"),
        "--reaction-base-cache",
        str(cache / "validation_reaction_base.h5"),
        "--enzyme-base-cache",
        str(cache / "enzyme_base.h5"),
        "--functional-token-cache",
        str(job["tokens"]),
        "--train-pairs",
        str(job["protocol"] / "train_pairs.csv"),
        "--evaluation-split",
        subset,
        "--selection-metric",
        "reactzyme_mrr",
        "--max-directional-mrr-drop",
        "0.005",
        "--device",
        "cuda:0",
        "--output",
        str(directory / f"{subset}.json"),
        "--per-query-output",
        str(directory / f"{subset}.per_query.csv"),
    ]
    if subset == "test":
        command += ["--selection-from", str(directory / "validation.json")]
    else:
        command += ["--alphas", "0", "0.025", "0.05", "0.075", "0.1"]
    return command


def completed_outputs(directory, label):
    marker = directory / f"{label}.complete.json"
    previous = json.loads(marker.read_text())
    if previous["outputs"] != {p: digest(p) for p in previous["outputs"]}:
        raise ValueError(f"Completed artifact changed: {marker}")
    if label == "test" and previous["request"]["selection_sha256"] != digest(
        directory / "validation.json"
    ):
        raise ValueError(f"Validation selection changed: {marker}")
    return previous


def checked_job(command, directory, label, env=None):
    """Only a successful process plus matching output hashes permits reuse."""
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / f"{label}.complete.json"
    outputs = (
        [directory / f"{label}.json", directory / f"{label}.per_query.csv"]
        if label in ("validation", "test")
        else [
            directory / "metadata_provenance.json",
            *[directory / s / "query_metadata.csv" for s in SPLITS],
        ]
    )
    selection = digest(directory / "validation.json") if label == "test" else None
    request = dict(command=command, selection_sha256=selection)
    if marker.exists():
        previous = completed_outputs(directory, label)
        if previous["request"] != request or previous["outputs"] != {
            str(p): digest(p) for p in outputs
        }:
            raise ValueError(f"Completed artifact changed: {marker}")
        print(f"Reusing {directory.name}/{label}", flush=True)
        return
    print(f"Starting {directory.name}/{label}; log: {directory / (label + '.log')}", flush=True)
    with (directory / f"{label}.log").open("a") as stream:
        subprocess.run(
            command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True
        )
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(dict(request=request, outputs={str(p): digest(p) for p in outputs}), indent=2)
        + "\n"
    )
    os.replace(temporary, marker)
    print(f"Finished {directory.name}/{label}", flush=True)


def check_gpus(gpus):
    if not gpus or len(gpus) != len(set(gpus)) or not all(g.isdigit() for g in gpus):
        raise ValueError("Provide distinct numeric GPU indices")
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
        text=True,
        timeout=15,
    )
    selected = {}
    for line in inventory.splitlines():
        index, uuid, free = map(str.strip, line.split(","))
        if index in gpus:
            if int(free) < 16000:
                raise ValueError(f"GPU {index} has less than 16 GB free")
            selected[uuid] = index
    if len(selected) != len(gpus):
        raise ValueError("A selected GPU is not visible")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
        timeout=15,
    )
    if any(line.split(",")[0].strip() in selected for line in apps.splitlines()):
        raise RuntimeError("A selected GPU is occupied; no process was stopped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "report"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--gpus", default="0,1,2", help="One evaluation per GPU; other GPUs untouched"
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    output = args.run_root.resolve()
    if output.parent != ROOT / "runs" or not output.name.startswith(
        "biological_residual_paper_analysis_"
    ):
        raise ValueError("Use a new runs/biological_residual_paper_analysis_* directory")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = plans()
        snapshot = preflight(jobs)
        audit = ROOT / "results/reactzyme_split_biology"
        snapshot["audit_sha256"] = {
            str(p.relative_to(audit)): digest(p)
            for p in sorted(audit.rglob("*"))
            if p.is_file()
            and (p.suffix in (".parquet", ".csv", ".fasta") or p.name == "test_to_train.tsv")
        }
        manifest = output / "manifest.json"
        encoded = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
        if manifest.exists() and manifest.read_text() != encoded:
            raise ValueError("Inputs/code differ from the saved analysis; use a new run root")
        if not manifest.exists():
            manifest.write_text(encoded)
        if args.action == "preflight":
            print("Preflight passed. No GPU inference or training launched.", flush=True)
            return
        if args.action == "run":
            gpus = args.gpus.split(",")
            check_gpus(gpus)
            checked_job(
                [str(ANALYSIS_PYTHON), str(ANALYSIS), "metadata", "--output", str(output)],
                output,
                "metadata",
            )
            check_gpus(gpus)
            # Complete every validation before starting any test. Alpha never sees test scores.
            for subset in ("validation", "test"):

                def worker(slot):
                    for index, (split, job) in enumerate(jobs.items()):
                        if index % len(gpus) == slot:
                            env = dict(
                                os.environ,
                                CUDA_VISIBLE_DEVICES=gpus[slot],
                                CUDA_DEVICE_ORDER="PCI_BUS_ID",
                                OMP_NUM_THREADS="4",
                                OPENBLAS_NUM_THREADS="4",
                                MKL_NUM_THREADS="4",
                                PYTHONUNBUFFERED="1",
                                HF_HUB_OFFLINE="1",
                                TRANSFORMERS_OFFLINE="1",
                            )
                            checked_job(
                                command_for(job, subset, output / split),
                                output / split,
                                subset,
                                env,
                            )

                with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                    list(pool.map(worker, range(len(gpus))))
        completed_outputs(output, "metadata")
        for split in SPLITS:
            for subset in ("validation", "test"):
                completed_outputs(output / split, subset)
        subprocess.run(
            [str(ANALYSIS_PYTHON), str(ANALYSIS), "report", "--output", str(output)], check=True
        )
        print(
            "Completed: harmonized tables, validation alpha curves, biological strata and coverage.",
            flush=True,
        )


if __name__ == "__main__":
    main()
