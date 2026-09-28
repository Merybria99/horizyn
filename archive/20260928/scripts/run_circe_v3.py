#!/usr/bin/env python3
"""Prepare, check, train or evaluate isolated CIRCE-v3 cached-feature runs."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horizyn.circe_v3 import F3_ROOT, SPLITS, prepare_run, rooted, verify_run


def preflight(run):
    import h5py
    import numpy as np
    from horizyn.config import load_config
    from horizyn.circe_v3 import read_pairs
    from horizyn.datasets.residue_hdf5 import residue_hdf5_store_identity

    manifest = verify_run(run)
    config = load_config(manifest["configs"]["train"])
    # Read cache ID indices, not the enormous residue arrays.
    data = config.data
    train = read_pairs(data.train_pairs_path)
    valid = read_pairs(data.validation_pairs_path)
    for subset, pairs in (("train", train), ("validation", valid)):
        queries = {q for q, _ in pairs}
        for modality in ("t5v2", "unimol2", "chiro"):
            path = rooted(data[f"{subset}_reaction_{modality}_embeds_path"])
            with h5py.File(path, "r") as store:
                ids = {x.decode() if isinstance(x, bytes) else str(x) for x in store["ids"][:]}
            # The F3 training loader requests _f IDs, even in forward-only mode.
            missing = {q + "_f" for q in queries} - ids
            if missing:
                if modality == "t5v2" or not data.get(f"reaction_allow_missing_{modality}", False) or len(missing) == len(queries):
                    raise ValueError(f"Missing forward {modality} cache IDs: {len(missing)} in {path}")
                print(f"{subset}: {len(missing)} missing {modality} modalities use the existing F3 missing-modality mask", flush=True)
        path = rooted(data[f"{subset}_reaction_chemistry_vectors_path"])
        with np.load(path, allow_pickle=True) as store:
            ids = set(map(str, store["ids"]))
            if queries - ids:
                raise ValueError(f"Missing chemistry IDs: {path}")
    protein_path = rooted(data.protein_residue_embeds_path)
    residue_hdf5_store_identity(protein_path)  # Also rejects missing VDS backing shards.
    with h5py.File(protein_path, "r") as store:
        ids = ({x.decode() if isinstance(x, bytes) else str(x) for x in store["ids"][:]}
               if "ids" in store else set(store.keys()))
        missing = {p for _, p in train + valid} - ids
        if missing:
            raise ValueError(f"Missing protein cache IDs: {len(missing)}")
        candidate_path = data.get("validation_retrieval_candidate_ids_path")
        if candidate_path:
            candidates = set(rooted(candidate_path).read_text().splitlines()) - {""}
            if candidates - ids or {p for _, p in valid} - candidates:
                raise ValueError("Validation candidate pool lacks cached proteins or evaluation positives")
        vectors = store["vectors"]
        for index in {0, len(vectors) // 2, len(vectors) - 1}:
            if not np.isfinite(vectors[index]).all():
                raise ValueError(f"Nonfinite protein feature probe at row {index}")
    scorer = config.model.get("sleec_pooling", {}).get("checkpoint_path")
    if not scorer or not rooted(scorer).is_file():
        raise ValueError("Missing pretrained SLEEC scorer")
    print(json.dumps({"status": "preflight passed", "train_pairs": len(train),
                      "auxiliary": manifest["auxiliary"]}, indent=2), flush=True)
    return config


def check_gpus(devices):
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(selected) != devices or any(not x.isdigit() for x in selected) or len(set(selected)) != devices:
        raise ValueError(f"Set CUDA_VISIBLE_DEVICES to {devices} distinct physical GPU indices")
    result = subprocess.run(["nvidia-smi", "-i", ",".join(selected),
                             "--query-gpu=uuid", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=15)
    selected_uuids = set(result.stdout.split())
    if len(selected_uuids) != devices:
        raise RuntimeError("Could not verify selected GPU identities")
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                             "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=15)
    if any(line.split(",")[0].strip() in selected_uuids for line in result.stdout.splitlines()):
        raise RuntimeError("Selected GPUs have compute jobs; no training launched and no jobs stopped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--dataset", choices=["reactzyme", "enzymecage"], required=True)
    prep.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    prep.add_argument("--source-config-root", type=Path, default=F3_ROOT)
    prep.add_argument("--source-config", type=Path,
                      default=ROOT / "runs/enzymecage_f3_seed42/configs/train.yaml")
    prep.add_argument("--run-root", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--devices", type=int, default=4)
    prep.add_argument("--hidden-fraction", type=float, default=0.05)
    prep.add_argument("--aux-evidence", type=Path)
    for action in ("check", "train", "evaluate"):
        cmd = sub.add_parser(action)
        cmd.add_argument("--run-dir", type=Path, required=True)
        if action == "train":
            cmd.add_argument("--resume", type=Path)
            cmd.add_argument("--detach", metavar="TMUX_SESSION")
        if action == "evaluate":
            cmd.add_argument("--subset", choices=["validation", "hidden", "test"], required=True)
            cmd.add_argument("--checkpoint", type=Path, required=True)
            cmd.add_argument("--device", default="cuda")
            cmd.add_argument("--family-map", type=Path,
                             help="Development-only within-family R→E evaluation; CSV protein_id,family_id")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == "prepare":
        if args.devices < 1:
            parser.error("devices must be positive")
        specifications = ([(split, rooted(args.source_config_root) / split / "train.yaml",
                             rooted(args.source_config_root) / split / "test.yaml") for split in args.splits]
                          if args.dataset == "reactzyme" else [("original", args.source_config, None)])
        for split, source, test in specifications:
            output = rooted(args.run_root) / args.dataset / split / f"seed{args.seed}"
            report = prepare_run(source, output, test_template=test, seed=args.seed,
                                 devices=args.devices, holdout_fraction=args.hidden_fraction,
                                 evidence=rooted(args.aux_evidence) if args.aux_evidence else None)
            print(f"Prepared {output}: {report['train_pairs']} train / {report['hidden_pairs']} hidden", flush=True)
            print(json.dumps(report["auxiliary"]["families"], indent=2), flush=True)
        return
    run = rooted(args.run_dir)
    if args.action == "check":
        preflight(run)
        return
    if args.action == "train" and args.detach:
        if subprocess.run(["tmux", "has-session", "-t", f"={args.detach}"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            raise RuntimeError("Detached session already exists; not replacing it")
        command = [sys.executable, str(Path(__file__).resolve()), "train", "--run-dir", str(run)]
        if args.resume:
            command += ["--resume", str(rooted(args.resume))]
        import shlex
        # tmux's server may retain a different CUDA environment from an older run.
        command = ["env", f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
                   "PYTHONUNBUFFERED=1", *command]
        subprocess.run(["tmux", "new-session", "-d", "-s", args.detach, "-c", str(ROOT),
                        "exec " + shlex.join(command) + " >> " + shlex.quote(str(run / "pipeline.log")) + " 2>&1"], check=True)
        print(f"Detached session: {args.detach}\nLog: {run / 'pipeline.log'}")
        return
    with (run / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = verify_run(run)
        if args.action == "train":
            # Refuse accidental retraining over existing logs/checkpoints.
            if not args.resume and (run / "training.started.json").exists():
                raise RuntimeError("Run has already started; supply an explicit --resume checkpoint")
            with (run / "pipeline.log").open("a", buffering=1) as log:
                import contextlib
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    config = preflight(run)
                    check_gpus(int(config.training.devices))
                    command = [sys.executable, "scripts/train_protein_pooling.py", "--config", manifest["configs"]["train"]]
                    if args.resume:
                        resume = rooted(args.resume)
                        if not resume.is_relative_to(run / "checkpoints") or not resume.is_file():
                            raise ValueError("Resume must use a checkpoint from this run")
                        command += ["--resume", str(resume)]
                    (run / "training.started.json").write_text(json.dumps({"pid": os.getpid(), "command": command}) + "\n")
                    subprocess.run(command, check=True, stdout=log, stderr=log,
                                   env={**os.environ, "PYTHONUNBUFFERED": "1"})
        else:
            if args.subset not in manifest["configs"]:
                raise ValueError(f"No {args.subset} subset in this preparation")
            checkpoint = rooted(args.checkpoint)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if args.family_map and args.subset == "test":
                raise ValueError("Within-family development evaluation cannot modify the released test protocol")
            output = run / "evaluation" / (args.subset + ("_within_family" if args.family_map else "")) / checkpoint.stem
            output.mkdir(parents=True, exist_ok=False)
            config_path = manifest["configs"][args.subset]
            if args.family_map:
                import yaml
                from horizyn.circe_v3 import sha256
                family_path = rooted(args.family_map)
                config = yaml.safe_load(Path(config_path).read_text())
                config["data"]["evaluation_protein_families_path"] = str(family_path)
                config_path = str(output / "config.yaml")
                Path(config_path).write_text(yaml.safe_dump(config, sort_keys=False))
                (output / "family_map_provenance.json").write_text(json.dumps({"path": str(family_path), "sha256": sha256(family_path)}) + "\n")
            command = [sys.executable, "scripts/evaluate_protein_pooling.py", "--config", config_path,
                       "--checkpoint", str(checkpoint), "--direction", "reaction_to_enzyme" if args.family_map else "both", "--device", args.device,
                       "--evaluation-protocol", "paper_test_candidates" if args.subset == "test" else "configured_forward_candidates",
                       "--store-targets-on-cpu", "--output", str(output / "metrics.json")]
            with (output / "evaluation.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=log)


if __name__ == "__main__":
    main()
