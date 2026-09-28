#!/usr/bin/env python3
"""Matched reaction-smi controls: reuse R1; train R2 and R3; select on validation."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "runs/biological_residual_reaction_smi"
PROTOCOL = ROOT / "data/revised_protocols/reactzyme_paper/reaction_smi"
DEFAULT_RUN = ROOT / "runs/biological_residual_controls_reaction_smi_v1"
VARIANTS = ("R1_full", "R2_no_source_pretrain", "R3_unguided")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, text: str) -> None:
    """Never silently change an experiment or a partial run's configuration."""
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"Existing experiment differs: {path}; use a new run root")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(text)


def build_configs(run_root: Path) -> dict[str, dict]:
    source = yaml.safe_load(
        (
            ROOT / "configs/source_collapse_biological_residual_reaction_smi_pretrain.yaml"
        ).read_text()
    )
    finetune = yaml.safe_load(
        (ROOT / "configs/reactzyme_reaction_smi_biological_residual.yaml").read_text()
    )
    configs = {}
    for variant in VARIANTS[1:]:
        for stage, template in (("source", source), ("finetune", finetune)):
            config = copy.deepcopy(template)
            config["logging"]["log_dir"] = str(run_root / variant / "logs" / stage)
            config["logging"]["checkpoint_dir"] = str(run_root / variant / "checkpoints" / stage)
            config["logging"]["wandb"]["enabled"] = False
            config["ablation"] = {
                "variant": variant,
                "stage": stage,
                "split": "reaction_smi",
                "seed": 42,
            }
            if variant == "R3_unguided":
                config["model"]["biological_residual"].update(sleec_bias=0.0, sleec_pool_scale=0.0)
                subset = "source" if stage == "source" else "reactzyme"
                config["data"]["protein_functional_tokens_path"] = str(
                    run_root / variant / "cache" / f"{subset}_tokens.h5"
                )
            if stage == "finetune":
                config["data"]["source_replay"]["config_path"] = str(
                    run_root / "configs" / f"{variant}_source.yaml"
                )
                config["training"]["init_from_checkpoint"] = (
                    source["training"]["init_from_checkpoint"]
                    if variant == "R2_no_source_pretrain"
                    else str(run_root / variant / "checkpoints/source/last.ckpt")
                )
            configs[f"{variant}_{stage}"] = config
    return configs


def prepare(run_root: Path) -> dict[str, dict]:
    if run_root == REFERENCE or run_root in REFERENCE.parents or REFERENCE in run_root.parents:
        raise ValueError("Use a new experiment directory outside the original residual run")
    configs = build_configs(run_root)
    for name, config in configs.items():
        write_once(run_root / "configs" / f"{name}.yaml", yaml.safe_dump(config, sort_keys=False))
    manifest = {
        "schema": "biological_residual_controls_v1",
        "seed": 42,
        "devices": 4,
        "reference": str(REFERENCE),
        "variants": list(VARIANTS),
        "selection_metric": "balanced_reactzyme_mrr",
        "alpha_grid": [0, 0.025, 0.05, 0.075, 0.1],
        "directional_drop_guard": 0.005,
        "checkpoint_policy": "last epoch of each fixed-duration stage",
        "configs": {name: sha256(run_root / "configs" / f"{name}.yaml") for name in configs},
    }
    write_once(run_root / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return configs


def check_gpus(gpus: str) -> None:
    selected = gpus.split(",")
    if len(selected) != 4 or len(set(selected)) != 4 or not all(x.isdigit() for x in selected):
        raise ValueError("This matched recipe requires four distinct GPU indices")
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
        text=True,
        timeout=15,
    )
    uuids = set()
    for line in inventory.splitlines():
        index, uuid, free = (part.strip() for part in line.split(","))
        if index in selected:
            if int(free) < 16000:
                raise RuntimeError(f"GPU {index} has less than 16 GB free")
            uuids.add(uuid)
    if len(uuids) != 4:
        raise RuntimeError("Not all selected GPUs are visible to nvidia-smi")
    # Some drivers ignore -i for compute-app queries; explicitly filter UUIDs.
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
        timeout=15,
    )
    for line in apps.splitlines():
        if line.split(",")[0].strip() in uuids:
            raise RuntimeError(f"A selected GPU is occupied: {line}. No processes stopped.")


def preflight(configs: dict[str, dict]) -> None:
    """Check only inputs actually used by this cached recipe; never re-extract."""
    import h5py

    print(
        "Checking existing feature stores, F3 identity and audited source-pair hashes...",
        flush=True,
    )
    sys.path.insert(0, str(ROOT))
    from scripts.cache_sleec_functional_tokens import _file_signature

    source = configs["R2_no_source_pretrain_source"]
    f3 = ROOT / source["training"]["init_from_checkpoint"]
    for subset in ("source", "reactzyme", "reactzyme_validation_eval", "reactzyme_test"):
        cache = REFERENCE / "cache" / subset
        metadata = json.loads((cache / "manifest.json").read_text())
        recorded = metadata["signature"]["inputs"]["checkpoint"]
        if recorded != _file_signature(f3, hash_content=False):
            raise ValueError(f"Frozen F3 cache provenance mismatch: {cache}")
        if "eval" in subset or subset == "reactzyme_test":
            if (
                metadata["signature"]["precision"] != "32"
                or metadata["signature"]["output_dtype"] != "float32"
            ):
                raise ValueError(f"Evaluation cache is not exact FP32: {cache}")
    source_manifest = json.loads((REFERENCE / "data/source_pretrain/manifest.json").read_text())
    for kind in ("pairs", "reactions"):
        path = ROOT / source["data"][f"train_{kind}_path"]
        if sha256(path) != source_manifest[f"output_{kind}_sha256"]:
            raise ValueError(f"Previously audited source graph changed: {path}")
    inputs = {REFERENCE / "checkpoints/reactzyme_finetune/last.ckpt", f3}
    for config in configs.values():
        for key, value in config["data"].items():
            if isinstance(value, str) and key.endswith("_path"):
                path = ROOT / value
                if "R3_unguided/cache" not in str(path):
                    inputs.add(path)
    for subset in ("validation", "test"):
        inputs.update(
            PROTOCOL / f"{subset}_{suffix}"
            for suffix in ("pairs.csv", "rxns.csv", "candidate_ids.txt")
        )
        cache_subset = "reactzyme_validation_eval" if subset == "validation" else "reactzyme_test"
        inputs.update(
            REFERENCE / "cache" / cache_subset / filename
            for filename in ("enzyme_base.h5", "validation_reaction_base.h5")
        )
    for path in sorted(inputs):
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing input: {path}")
        if path.suffix == ".h5":
            with h5py.File(path, "r") as handle:
                # Read a small real dataset value, not just the directory entry.
                key = next(
                    (k for k in ("vectors", "token_vectors") if k in handle), next(iter(handle))
                )
                obj = handle[key]
                if isinstance(obj, h5py.Dataset) and obj.size:
                    obj[0]
        else:
            with path.open("rb") as handle:
                handle.read(4096)
    print(
        f"Preflight passed: {len(inputs)} inputs readable; frozen F3/source graph unchanged.",
        flush=True,
    )


def child(command: list[str], env: dict[str, str] | None = None) -> None:
    print("RUN " + shlex.join(command), flush=True)
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    try:
        code = process.wait()
    except BaseException:
        # Signal only the subprocess group this controller just created.
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            pass
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def completed_training_epochs(saved: dict) -> int:
    """Count finished epochs in both validation-end and post-fit checkpoints."""
    try:
        loop = saved["loops"]["fit_loop"]
        epochs = loop["epoch_progress"]["current"]
        batches = loop["epoch_loop.batch_progress"]
        completed, started = epochs["completed"], epochs["started"]
        ready, done = batches["current"]["ready"], batches["current"]["completed"]
        last_batch = batches["is_last_batch"]
    except (KeyError, TypeError) as error:
        raise ValueError("Checkpoint lacks training-progress counters") from error
    if (
        any(type(value) is not int or value < 0 for value in (completed, started, ready, done))
        or started not in (completed, completed + 1)
        or done > ready
        or type(last_batch) is not bool
    ):
        raise ValueError("Checkpoint has inconsistent training-progress counters")
    # Validation-end saves precede the epoch counter increment. A post-fit save
    # has already incremented it; a partial final epoch must still be resumed.
    if started == completed + 1 and last_batch and ready > 0 and done == ready:
        return started
    return completed


def train(config_path: Path) -> None:
    import torch

    config = yaml.safe_load(config_path.read_text())
    last = Path(config["logging"]["checkpoint_dir"]) / "last.ckpt"
    command = [
        sys.executable,
        "scripts/train_protein_pooling.py",
        "--config",
        str(config_path),
        "--wandb-mode",
        "disabled",
    ]
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False, mmap=True)
        epochs = completed_training_epochs(saved)
        del saved
        if epochs > config["training"]["max_epochs"]:
            raise ValueError(f"Checkpoint exceeds this experiment's epoch budget: {last}")
        if epochs == config["training"]["max_epochs"]:
            print(f"Reusing completed stage: {last}", flush=True)
            return
        command.extend(["--resume", str(last)])
    child(command)
    if not last.is_file():
        raise RuntimeError(f"Training returned without its last checkpoint: {last}")
    saved = torch.load(last, map_location="cpu", weights_only=False, mmap=True)
    if completed_training_epochs(saved) != config["training"]["max_epochs"]:
        raise RuntimeError(f"Stage did not finish its prescribed epoch budget: {last}")


def cache_random_tokens(configs: dict[str, dict]) -> None:
    for stage in ("source", "finetune"):
        data = configs[f"R3_unguided_{stage}"]["data"]
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            "scripts/cache_sleec_functional_tokens.py",
            "--selection",
            "random",
            "--selection-seed",
            "42",
            "--residue-h5",
            data["protein_residue_embeds_path"],
            "--output",
            data["protein_functional_tokens_path"],
            "--top-k",
            "48",
            "--context-k",
            "16",
            "--bf16",
        ]
        pairs = (
            [ROOT / data["train_pairs_path"]]
            if stage == "source"
            else [PROTOCOL / f"{part}_pairs.csv" for part in ("train", "validation", "test")]
        )
        for path in pairs:
            command.extend(["--pairs", str(path)])
        child(command)


def evaluate(run_root: Path, variant: str, subset: str) -> None:
    output = run_root / variant / "results" / f"{subset}.json"
    checkpoint = (
        REFERENCE / "checkpoints/reactzyme_finetune/last.ckpt"
        if variant == "R1_full"
        else run_root / variant / "checkpoints/finetune/last.ckpt"
    )
    if output.exists():
        if not output.with_suffix(".per_query.csv").is_file():
            raise RuntimeError(f"Incomplete evaluation artifacts: {output}")
        previous = json.loads(output.read_text())
        signature = {"size": checkpoint.stat().st_size, "mtime_ns": checkpoint.stat().st_mtime_ns}
        if previous.get("checkpoint_signature") != signature or previous.get("checkpoint") != str(
            checkpoint.resolve()
        ):
            raise ValueError(f"Checkpoint changed since evaluation: {output}")
        print(f"Reusing evaluation: {output}", flush=True)
        return
    config = (
        ROOT / "configs/reactzyme_reaction_smi_biological_residual.yaml"
        if subset == "validation"
        else ROOT
        / "runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry/reaction_smi/test.yaml"
    )
    cache = (
        REFERENCE
        / "cache"
        / ("reactzyme_validation_eval" if subset == "validation" else "reactzyme_test")
    )
    tokens = (
        run_root / variant / "cache/reactzyme_tokens.h5"
        if variant == "R3_unguided"
        else REFERENCE / "cache/reactzyme/protein_functional_tokens.h5"
    )
    command = [
        sys.executable,
        "scripts/evaluate_biological_residual.py",
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(config),
        "--pairs",
        str(PROTOCOL / f"{subset}_pairs.csv"),
        "--reactions",
        str(PROTOCOL / f"{subset}_rxns.csv"),
        "--reaction-base-cache",
        str(cache / "validation_reaction_base.h5"),
        "--enzyme-base-cache",
        str(cache / "enzyme_base.h5"),
        "--functional-token-cache",
        str(tokens),
        "--candidate-ids",
        str(PROTOCOL / f"{subset}_candidate_ids.txt"),
        "--selection-metric",
        "reactzyme_mrr",
        "--evaluation-split",
        subset,
        "--train-pairs",
        str(PROTOCOL / "train_pairs.csv"),
        "--output",
        str(output),
        "--per-query-output",
        str(output.with_suffix(".per_query.csv")),
    ]
    if subset == "test":
        command.extend(["--selection-from", str(output.with_name("validation.json"))])
    child(command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "run", "report"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument(
        "--query-metadata", type=Path, help="Optional CSV: direction,query_id and stratum_* columns"
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    run_root = args.run_root.resolve()
    if run_root == REFERENCE or run_root in REFERENCE.parents or REFERENCE in run_root.parents:
        raise ValueError("Use a separate control-run directory")
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / "controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "This experiment already has a controller; no duplicate launched"
            ) from None
        configs = prepare(run_root)
        if args.action == "prepare":
            print(f"Prepared matched configurations: {run_root}")
            return
        if args.action in ("run", "preflight"):
            if args.action == "run":
                check_gpus(args.gpus)
            preflight(configs)
        if args.action == "preflight":
            return
        if args.action == "run":
            check_gpus(args.gpus)
            code = {
                str(path.relative_to(ROOT)): sha256(path)
                for path in (
                    Path(__file__),
                    ROOT / "scripts/cache_sleec_functional_tokens.py",
                    ROOT / "scripts/evaluate_biological_residual.py",
                    ROOT / "horizyn/biological_residual.py",
                    ROOT / "horizyn/protein_pooling_lightning_module.py",
                    ROOT / "horizyn/training_options.py",
                    ROOT / "horizyn/training_warm_start.py",
                    ROOT / "scripts/train_protein_pooling.py",
                )
            }
            write_once(
                run_root / "execution_signature.json",
                json.dumps(code, indent=2, sort_keys=True) + "\n",
            )
            os.environ.update(
                CUDA_VISIBLE_DEVICES=args.gpus,
                CUDA_DEVICE_ORDER="PCI_BUS_ID",
                OMP_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="4",
                MKL_NUM_THREADS="4",
                PYTHONUNBUFFERED="1",
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
            )

            def stop(_signum, _frame):
                raise KeyboardInterrupt

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGHUP, stop)
            (run_root / "controller.json").write_text(
                json.dumps({"pid": os.getpid(), "host": socket.gethostname()})
            )
            evaluate(run_root, "R1_full", "validation")
            train(run_root / "configs/R2_no_source_pretrain_finetune.yaml")
            evaluate(run_root, "R2_no_source_pretrain", "validation")
            cache_random_tokens(configs)
            train(run_root / "configs/R3_unguided_source.yaml")
            train(run_root / "configs/R3_unguided_finetune.yaml")
            evaluate(run_root, "R3_unguided", "validation")
            for variant in VARIANTS:
                evaluate(run_root, variant, "test")
        sys.path.insert(0, str(ROOT))
        from scripts.report_biological_residual_controls import report

        report(run_root, args.query_metadata)


if __name__ == "__main__":
    main()
