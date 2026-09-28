#!/usr/bin/env python3
"""Detached, restartable RefSeq Case 1 screen using the completed workbook model."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horizyn.generalization_diagnostics import atomic_json
from scripts.run_circe_generalization_diagnostics import check_gpus, sha

WORKBOOK = ROOT / "wet_lab/Case1/restricted_setting/runs/feature_gate_case1.0EGTB4/result"
POOL = ROOT / "wet_lab/databases/refseq/prokaryotes/five_shards"
DEFAULT_OUTPUT = ROOT / "wet_lab/runs/refseq/prokaryotes/feature_gate_epoch30"


def partition_ranges(count, workers):
    if workers < 1 or count < workers:
        raise ValueError("Need at least one candidate per worker")
    return [(count * i // workers, count * (i + 1) // workers) for i in range(workers)]


def prepare(output, gpus, batch_size):
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    workbook = json.loads((WORKBOOK / "results.json").read_text())
    if not (WORKBOOK / "complete.json").is_file():
        raise ValueError("Workbook run is incomplete")
    inputs = [WORKBOOK / name for name in
              ("results.json", "feature_manifest.json", "participants.csv", "reactiont5.h5", "unimol2.h5", "chiro.h5")]
    inputs += [Path(workbook["model"][name]) for name in ("config", "checkpoint")]
    inputs += [POOL / "candidate_ids_prott5_order.txt"]
    code = [Path(__file__), ROOT / "wet_lab/query_io.py", ROOT / "horizyn/model.py",
            ROOT / "horizyn/enzyme_multiview.py", ROOT / "horizyn/gated_reaction_fusion.py",
            ROOT / "horizyn/benchmarks/retrieval.py", ROOT / "wet_lab/query.py"]
    ids = (POOL / "candidate_ids_prott5_order.txt").read_text().splitlines()
    if len(ids) != 3944613 or len(set(ids)) != len(ids) or any(not p for p in ids):
        raise ValueError("Unexpected RefSeq candidate count, blanks or duplicate IDs")
    ranges = partition_ranges(len(ids), len(gpus))
    manifest = dict(model=workbook["model"], reaction=workbook["reaction"],
        workbook=str(WORKBOOK), pool=str(POOL), candidate_count=len(ids),
        workers=len(gpus), batch_size=batch_size, top_k=[1, 5, 10, 25, 50, 100, 1000],
        inputs={str(p): sha(p) for p in inputs}, code={str(p): sha(p) for p in code},
        partitions=ranges, residue_dtype="float32", device="cuda:0")
    receipt = output / "manifest.json"
    if receipt.exists():
        if json.loads(receipt.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError("Inputs/configuration changed; use a new --output directory")
    else:
        atomic_json(receipt, manifest)
    for i, (start, end) in enumerate(ranges):
        path = output / f"shard_{i}.ids"
        text = "\n".join(ids[start:end]) + "\n"
        if path.exists():
            if path.read_text() != text:
                raise ValueError(f"Changed candidate partition: {path}")
        else:
            with path.open("x") as handle:
                handle.write(text)
    print(f"Verified {len(ids):,} unique candidates across {len(gpus)} disjoint partitions.", flush=True)
    return manifest


def worker(output, shard):
    import h5py
    import numpy as np
    import torch
    from horizyn.config import load_config
    from horizyn.training_io import StoragePrecisionResidues
    from horizyn.benchmarks.retrieval import (
        BenchmarkTask, _target_cache_base_metadata, build_reaction_inputs, encode_reactions,
        load_candidate_keys_from_residue, load_repo_checkpoint, prepare_target_embedding_cache,
        release_target_embedding_cache_lock, write_target_embedding_cache,
    )
    from wet_lab.query import bucket_candidate_keys_by_residue_length, streaming_topk, write_rankings
    from wet_lab.query_io import encode_prefetched_targets

    torch.set_num_threads(4)
    torch.manual_seed(42)
    manifest = json.loads((output / "manifest.json").read_text())
    device = manifest["device"]
    work = output / f"shard_{shard}"
    work.mkdir(exist_ok=True)
    model = manifest["model"]
    config = load_config(model["config"])
    if (config.model.enzyme_input_mode != "raw_mean_sleec_multiview"
            or config.model.reaction_multimodal_attention.side_composition != "molecule_set"
            or config.model.reaction_use_chemistry or config.model.reaction_use_directional):
        raise ValueError("Unexpected model/participant-set recipe")
    feature_manifest = json.loads((WORKBOOK / "feature_manifest.json").read_text())
    query_id = manifest["reaction"]["id"]
    for name, dim in (("reactiont5", 768), ("unimol2", 768), ("chiro", 256)):
        with h5py.File(WORKBOOK / f"{name}.h5") as handle:
            if list(handle["ids"].asstr()[:]) != [query_id]:
                raise ValueError("Cached reaction IDs do not match")
            matrix = handle["vectors" if name == "reactiont5" else "reactant_vectors"][:]
            if matrix.shape != (1 if name == "reactiont5" else 2, dim) or not np.isfinite(matrix).all():
                raise ValueError("Invalid participant features")
    task = BenchmarkTask(name="case1_refseq", task_type="screening", dataset="refseq",
        task_label=query_id, split="prepared_pool", pairs=WORKBOOK / "participants.csv",
        reactions=WORKBOOK / "participants.csv", reaction_model_embeds_h5=WORKBOOK / "reactiont5.h5",
        reaction_unimol2_embeds_h5=WORKBOOK / "unimol2.h5", reaction_chiro_embeds_h5=WORKBOOK / "chiro.h5")
    reaction_inputs = build_reaction_inputs(task, config)
    residue_path = POOL / "proteins_prott5_residue.h5"
    dataset = StoragePrecisionResidues(str(residue_path), in_memory=False,
        max_tokens=config.data.max_protein_tokens, truncation=config.data.protein_truncation)
    dataset.dtype = torch.float32
    cache_info = None
    try:
        keys, stats = load_candidate_keys_from_residue(dataset, output / f"shard_{shard}.ids")
        keys, padding_stats = bucket_candidate_keys_by_residue_length(dataset, keys,
            batch_size=manifest["batch_size"], window_size=8192)
        stats.update(padding_stats)
        print(f"Shard {shard}: {len(keys):,} proteins; batch {manifest['batch_size']}; padding {padding_stats}", flush=True)
        metadata = _target_cache_base_metadata(kind="residue", checkpoint=model["checkpoint"],
            config_path=model["config"], protein_embedding="prott5", score_protein_embedding="prott5",
            residue_h5=residue_path, max_tokens=config.data.max_protein_tokens,
            truncation=config.data.protein_truncation)
        metadata["residue_load_dtype"] = "float32"
        targets, cache_info = prepare_target_embedding_cache(work / "target_cache", metadata, keys,
            device=device, store_on_device=False)
        module, _ = load_repo_checkpoint(model["checkpoint"], config, device)
        if targets is None:
            targets = encode_prefetched_targets(module, dataset, keys, device,
                manifest["batch_size"], False, progress_every_batches=25, cache_info=cache_info)
            write_target_embedding_cache(cache_info, metadata, keys, targets)
        query = encode_reactions(module, reaction_inputs, [query_id], device, 1)
        scores, indices = streaming_topk(query, targets, top_k=max(manifest["top_k"]),
            device=device, batch_size=32768)
        (work / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2))
        result = write_rankings(workspace=work, reaction=manifest["reaction"], model=model,
            candidate_stats=stats, cache_info=cache_info, top_k_values=manifest["top_k"],
            candidate_keys=keys, scores=scores, indices=indices,
            candidate_config=dict(fasta=str(POOL / "proteins.fasta"), metadata_csv=str(POOL / "proteins.csv"),
                                  metadata_id_column=manifest.get("metadata_id_column", "protein_id")),
            feature_manifest=feature_manifest)
        atomic_json(work / "complete.json", dict(result=str(result)))
        print(f"Shard complete: {result}", flush=True)
    finally:
        release_target_embedding_cache_lock(cache_info)
        dataset.close()


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"Interrupted by signal {signum}")


def run(output, gpus, batch_size):
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((output / ".controller.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_gpus(gpus)
        manifest = prepare(output, gpus, batch_size)
        if (output / "complete.json").exists():
            print(f"Already complete: {output / 'complete.json'}", flush=True)
            return
        processes = []
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, interrupted)
        try:
            for shard, gpu in enumerate(gpus):
                if (output / f"shard_{shard}/complete.json").exists():
                    print(f"Reusing completed shard {shard}", flush=True)
                    continue
                log = stack.enter_context((output / f"gpu{gpu}.log").open("a"))
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                    PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
                for key in ("BASH_ENV", "ENV", "WET_LAB_CANDIDATE_POOL_ROOT_OVERRIDE", "WET_LAB_RESIDUE_EMBEDDINGS_OVERRIDE"):
                    env.pop(key, None)
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker",
                    "--output", str(output), "--shard", str(shard)], cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
                processes.append(process)
                print(f"GPU {gpu}: shard {shard}, PID {process.pid}", flush=True)
            while any(p.poll() is None for p in processes):
                if any(p.poll() not in (None, 0) for p in processes):
                    raise RuntimeError("A worker failed; inspect gpu*.log. Completed batches are preserved.")
                time.sleep(2)
            if any(p.returncode != 0 for p in processes):
                raise RuntimeError("A worker failed; refusing an incomplete merge")
            from wet_lab.merge_query_shards import merge_sharded_results
            result = merge_sharded_results([output / f"shard_{i}/results.json" for i in range(len(gpus))], output / "merged")
            merged = json.loads(result.read_text())
            if merged["candidate_pool"]["candidate_count"] != manifest["candidate_count"]:
                raise ValueError("Merged coverage is incomplete")
            atomic_json(output / "complete.json", dict(result=str(result), candidates=manifest["candidate_count"]))
            print(f"COMPLETE: {result}", flush=True)
        finally:
            for process in processes:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print(f"Worker {process.pid} is still stopping; do not restart yet.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("launch", "prepare", "run", "worker"))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    gpus = args.gpus.split(",")
    if len(gpus) < 2 or len(gpus) > 4 or len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
        parser.error("Use 2-4 distinct physical GPU IDs")
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    if args.stage == "worker":
        if args.shard is None:
            parser.error("--shard required for worker")
        worker(output, args.shard)
        return
    if args.stage == "prepare":
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            prepare(output, gpus, args.batch_size)
        return
    if args.stage == "run":
        run(output, gpus, args.batch_size)
        return
    check_gpus(gpus)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    session = "case1_feature_refseq_" + hashlib.sha256(str(output).encode()).hexdigest()[:10]
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), "run", "--output", str(output),
                         "--gpus", args.gpus, "--batch-size", str(args.batch_size)])
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(ROOT),
        "/usr/bin/env", "-u", "BASH_ENV", "-u", "ENV", "/bin/bash", "--noprofile", "--norc", "-c",
        f"exec {command} >> {shlex.quote(str(output / 'pipeline.log'))} 2>&1"], check=True)
    print(f"Detached session: {session}\nLog: {output / 'pipeline.log'}")


if __name__ == "__main__":
    main()
