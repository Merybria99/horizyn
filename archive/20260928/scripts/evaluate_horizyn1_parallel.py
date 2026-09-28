#!/usr/bin/env python3
"""Resumable multi-GPU full-catalog evaluation; no DDP/NCCL or training changes.

Workers preserve the serial evaluator's FP32 encoding batches, candidate order,
positive gold and exact ranker. Only disjoint work is assigned to different GPUs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_horizyn1_circe_v2 as serial
from scripts.stop_horizyn1_circe_v2 import (
    alive, option_value, read_process, resolve_process_path, send_signal,
)

PATH_FLAGS = ("checkpoint", "config", "pairs", "reactions", "protein_candidates",
              "reaction_candidates", "enzyme_query_ids", "reaction_query_ids", "output")
METRICS = {"mrr", "mean_rank", "reactzyme_mrr", "r_precision", "avg_precision",
           *[f"{name}_{k}" for name in ("top", "recall") for k in (1, 10, 100, 1000)]}
DIRECTIONS = ("reaction_to_enzyme", "enzyme_to_reaction")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def blocks(count, size):
    return [(start, min(start + size, count)) for start in range(0, count, size)]


def owned_blocks(count, size, rank, workers):
    return [(i, start, stop) for i, (start, stop) in enumerate(blocks(count, size))
            if i % workers == rank]


def tensor_path(cache, start, stop):
    return cache / f"proteins-{start:09d}-{stop:09d}.pt"


def read_tensor(path, signature, start, stop, dim=512):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    value = payload["embeddings"]
    if (payload.get("signature") != signature or payload.get("start") != start
            or payload.get("stop") != stop or value.shape != (stop - start, dim)
            or value.dtype != torch.float32 or not torch.isfinite(value).all()):
        raise ValueError(f"Invalid or incompatible encoded chunk: {path}")
    return value


def save_tensor(path, signature, start, stop, value):
    value = value.detach().cpu().contiguous()
    if value.shape != (stop - start, 512) or value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise ValueError("Invalid encoder output; no chunk committed")
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    torch.save(dict(signature=signature, start=start, stop=stop, embeddings=value), temporary)
    os.replace(temporary, path)


def make_signature(args):
    print("Checking checkpoint, input catalogs and feature provenance...", flush=True)
    sources = {name: {**serial.file_signature(getattr(args, name)),
                      "sha256": serial.digest(getattr(args, name))}
               for name in PATH_FLAGS if name != "output"}
    config = serial.load_config(args.config)
    for name, value in config.data.items():
        if not isinstance(value, str) or not ("embed" in name or "vectors_path" in name):
            continue
        path = Path(value)
        if not path.is_file():
            continue
        sources[name] = serial.file_signature(path)
        if path.suffix in {".h5", ".hdf5"}:
            import h5py
            with h5py.File(path, "r") as handle:
                if "vectors" in handle and handle["vectors"].is_virtual:
                    sources[name]["backing"] = [serial.file_signature(
                        Path(os.fsdecode(s.file_name)) if Path(os.fsdecode(s.file_name)).is_absolute()
                        else path.parent / os.fsdecode(s.file_name))
                        for s in handle["vectors"].virtual_sources()]
    return dict(schema="horizyn_parallel_test_v1", sources=sources,
                implementation={**serial.implementation_signatures(),
                                "parallel": serial.digest(__file__)},
                protocol=args.protocol, target_batch_size=args.target_batch_size,
                batch_size=args.batch_size, candidate_chunk_size=args.candidate_chunk_size,
                encoding_chunk_size=args.encoding_chunk_size, score_block_size=args.score_block_size,
                score_dtype="float32", ties="score_desc_candidate_row_asc",
                torch_version=str(torch.__version__), numpy_version=np.__version__)


def catalog_ids(args):
    return serial.read_ids(args.protein_candidates), serial.read_ids(args.reaction_candidates)


def encode_worker(args, cache, signature, rank, workers, device):
    proteins, reactions = catalog_ids(args)
    tasks = owned_blocks(len(proteins), args.encoding_chunk_size, rank, workers)
    pending = []
    for _, start, stop in tasks:
        path = tensor_path(cache, start, stop)
        if path.exists():
            read_tensor(path, signature, start, stop)
        else:
            pending.append((start, stop))
    reaction_path = cache / "reactions.pt"
    need_reactions = rank == 0 and not reaction_path.exists()
    if rank == 0 and reaction_path.exists():
        read_tensor(reaction_path, signature, 0, len(reactions))
    if not pending and not need_reactions:
        print(f"Worker {rank}: all encoding chunks already complete", flush=True)
        return
    config = serial.load_config(args.config)
    module = serial.ProteinPooledLitModule.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    if (module.model.enzyme_prototype_count != 1 or module.embedding_similarity != "cosine"
            or getattr(module.model, "e2r_adapter", None) is not None
            or getattr(module.model, "r2e_adapter", None) is not None):
        raise ValueError("Requires the same shared-vector cosine architecture as serial evaluation")
    module.eval().to(device)
    with torch.inference_mode():
        if need_reactions:
            features = serial.build_reaction_feature_dataset(
                args.reactions, config, bidirectional=not bool(config.data.get("indexed_pairs_dir")),
                split_name="train")
            available = set(features.keys)
            keys = [r if r in available else f"{r}_f" for r in reactions]
            if set(keys) - available:
                raise ValueError("Missing mandatory reaction features")
            value = serial.encode_reactions(module, features, keys, device, args.batch_size).cpu()
            save_tensor(reaction_path, signature, 0, len(reactions), value)
            del features, value
        if pending:
            residues = serial.ResidueEmbedDataset(
                config.data.protein_residue_embeds_path, in_memory=False,
                max_tokens=config.data.get("max_protein_tokens", 1022),
                truncation=config.data.get("protein_truncation", "ends_center"))
            available = set(residues.keys)
            if any(p not in available for start, stop in pending for p in proteins[start:stop]):
                raise ValueError("Missing protein features")
            del available
            for start, stop in pending:
                value = serial.encode_targets(module, serial.KeySubsetDataset(residues, proteins[start:stop]),
                                              device, args.target_batch_size, store_on_device=False)
                save_tensor(tensor_path(cache, start, stop), signature, start, stop, value)
                print(f"Committed proteins [{start:,}, {stop:,}); {stop - start:,} rows", flush=True)


def merge_catalog(cache, signature, count, size):
    """CPU-only, streamed merge, after encoding workers have exited."""
    output, marker = cache / "proteins.npy", cache / "proteins.complete.json"
    if marker.exists():
        info = json.loads(marker.read_text())
        if info.get("signature") != signature or info.get("shape") != [count, 512] or info.get("sha256") != serial.digest(output):
            raise ValueError("Merged catalog has changed")
        return output
    temporary = cache / f"proteins.partial.{os.getpid()}.npy"
    merged = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(count, 512))
    for start, stop in blocks(count, size):
        value = read_tensor(tensor_path(cache, start, stop), signature, start, stop)
        merged[start:stop] = value.numpy()
    merged.flush()
    del merged
    os.replace(temporary, output)
    atomic_json(marker, dict(signature=signature, shape=[count, 512], sha256=serial.digest(output)))
    return output


def read_score(path, signature, direction, start, stop, candidates):
    value = json.loads(path.read_text())
    result = value["metrics"]
    if (value.get("signature") != signature or value.get("direction") != direction
            or value.get("start") != start or value.get("stop") != stop
            or result.get("queries") != stop - start or result.get("candidates") != candidates
            or set(result) != METRICS | {"queries", "candidates"}
            or any(not math.isfinite(float(result[k])) for k in METRICS)):
        raise ValueError(f"Invalid score chunk: {path}")
    return result


def score_path(cache, direction, start, stop):
    return cache / f"{direction}-{start:09d}-{stop:09d}.json"


def score_worker(args, cache, signature, rank, workers, device):
    proteins, reactions = catalog_ids(args)
    plookup, rlookup = {p: i for i, p in enumerate(proteins)}, {r: i for i, r in enumerate(reactions)}
    rqueries, equeries = serial.read_ids(args.reaction_query_ids), serial.read_ids(args.enzyme_query_ids)
    rtruth, etruth = serial.query_truth(args.pairs, set(rqueries), set(equeries))
    if set(rqueries) - rlookup.keys() or set(equeries) - plookup.keys():
        raise ValueError("Query entities missing from catalog")
    if args.protocol == "reaction_holdout_clustered" and set(reactions) != set(rqueries):
        raise ValueError("Reaction-held-out E2R must use exactly the held-out reaction catalog")
    # Copy-on-write mmap avoids loading another 12+ GB host copy per worker.
    pvalues = torch.from_numpy(np.load(cache / "proteins.npy", mmap_mode="c"))
    rvalues = read_tensor(cache / "reactions.pt", signature, 0, len(reactions))
    if args.catalog_on_gpu:
        pvalues = pvalues.to(device)
    rvalues = rvalues.to(device)
    for direction, queries, anchors, lookup, candidates, truth, clookup in (
        (DIRECTIONS[0], rqueries, rvalues, rlookup, pvalues, rtruth, plookup),
        (DIRECTIONS[1], equeries, pvalues, plookup, rvalues, etruth, rlookup),
    ):
        for _, start, stop in owned_blocks(len(queries), args.score_block_size, rank, workers):
            path = score_path(cache, direction, start, stop)
            if path.exists():
                read_score(path, signature, direction, start, stop, len(candidates))
                continue
            ids = queries[start:stop]
            result = serial.evaluate_direction(
                anchors[[lookup[q] for q in ids]], candidates, ids, truth, clookup,
                batch_size=args.batch_size, chunk_size=args.candidate_chunk_size,
                device=device, label=f"{direction} [{start}:{stop}]")
            atomic_json(path, dict(signature=signature, direction=direction,
                                   start=start, stop=stop, metrics=result))
            print(f"Committed {direction} queries [{start:,}, {stop:,})", flush=True)


def merge_scores(cache, signature, direction, queries, candidates, size):
    sums = defaultdict(float)
    for start, stop in blocks(queries, size):
        values = read_score(score_path(cache, direction, start, stop), signature,
                            direction, start, stop, candidates)
        for key in METRICS:
            sums[key] += values[key] * (stop - start)
    return dict(queries=queries, candidates=candidates, **{k: v / queries for k, v in sums.items()})


def script_arguments(argv, executable, cwd, script):
    """Match Python's actual script operand, never strings inside launcher arguments.

Fail closed for -c/-m, wrappers, unknown interpreter flags and other binaries.
The legacy command uses ``python -u SCRIPT ...``; no wrapper needs signalling.
"""
    if Path(executable).resolve() != Path(sys.executable).resolve() or len(argv) < 2:
        return None
    index = 1
    while index < len(argv) and argv[index] in {"-u", "-B", "-I", "-E", "-s", "-S", "-O", "-OO"}:
        index += 1
    if index < len(argv) and argv[index] == "--":
        index += 1
    if index == len(argv) or argv[index].startswith("-"):
        return None
    if resolve_process_path(argv[index], cwd) != Path(script).resolve():
        return None
    return argv[index + 1:]


def protected_processes():
    """The controller and its ancestors (including tmux) are never stop targets."""
    protected, pid = set(), os.getpid()
    while pid > 0 and pid not in protected:
        protected.add(pid)
        process = read_process(pid)
        if process is None:
            break
        pid = process.ppid
    return protected


def process_arguments(pid, script):
    path = Path("/proc") / str(pid)
    if path.stat().st_uid != os.getuid():
        return None, None
    argv = [os.fsdecode(x) for x in (path / "cmdline").read_bytes().split(b"\0") if x]
    cwd = (path / "cwd").resolve(strict=True)
    executable = (path / "exe").resolve(strict=True)
    return script_arguments(argv, executable, cwd, script), cwd


def verify_legacy_process(pid, output, checkpoint):
    if pid in protected_processes():
        return None
    try:
        before = read_process(pid)
        if before is None or before.state in {"Z", "X"}:
            return None
        arguments, cwd = process_arguments(pid, serial.__file__)
        if arguments is None:
            return None
        out, ckpt = option_value(arguments, "--output"), option_value(arguments, "--checkpoint")
        if out is None or resolve_process_path(out, cwd) != output.resolve():
            return None
        if ckpt is None or resolve_process_path(ckpt, cwd) != checkpoint.resolve():
            raise RuntimeError("Previous output belongs to a different checkpoint; no signals sent")
        after = read_process(pid)
        return after if after is not None and after.started == before.started else None
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def legacy_processes(output, checkpoint):
    result = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            process = verify_legacy_process(int(path.name), output, checkpoint)
            if process is not None:
                result[process.pid] = process  # Do not expand to a launcher's process tree.
    return result


def refuse_other_controllers():
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            arguments, _ = process_arguments(int(path.name), __file__)
            if arguments and arguments[0] == "run":
                raise RuntimeError(f"Another parallel test controller is present (PID {path.name}); not launching")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue


def check_gpus(gpus, allowed_pids=()):
    if not gpus or len(gpus) != len(set(gpus)) or not all(g.isdigit() for g in gpus):
        raise ValueError("GPU indices must be distinct integers")
    catalog = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
                                       "--format=csv,noheader,nounits"], text=True, timeout=15)
    selected = {}
    for row in catalog.splitlines():
        index, uuid, free = [x.strip() for x in row.split(",")]
        if index in gpus:
            selected[uuid] = (index, int(free))
    if len(selected) != len(gpus):
        raise ValueError("A selected GPU does not exist")
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                         "--format=csv,noheader,nounits"], text=True, timeout=15)
    for row in processes.splitlines():
        uuid, pid = [x.strip() for x in row.split(",")]
        if uuid in selected and int(pid) not in allowed_pids:
            raise RuntimeError(f"GPU {selected[uuid][0]} is occupied by PID {pid}; not launching")
    if not allowed_pids and any(free < 20000 for _, free in selected.values()):
        raise RuntimeError("Each selected GPU needs at least 20,000 MiB free")


def stop_verified(processes, output, checkpoint):
    if not processes:
        return
    print(f"Stopping only verified previous test processes: {sorted(processes)}", flush=True)
    for sig, timeout in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
        for p in processes.values():
            if not alive(p):
                continue
            verified = verify_legacy_process(p.pid, output, checkpoint)
            if verified is None or verified.started != p.started:
                raise RuntimeError("Evaluator identity changed before signalling; replacement not launched")
            print(f"Sending {sig.name} to verified evaluator PID {p.pid}", flush=True)
            send_signal(p, sig)
        deadline = time.monotonic() + timeout
        while any(alive(p) for p in processes.values()) and time.monotonic() < deadline:
            time.sleep(.2)
        if not any(alive(p) for p in processes.values()):
            print("Previous evaluator exited; launchers and saved files were left untouched.", flush=True)
            return
    raise RuntimeError("Previous test processes remain; replacement was not launched")


def run_workers(job, phase, gpus):
    processes, handles = [], []
    cache = job.parent
    try:
        for rank, gpu in enumerate(gpus):
            log = cache / f"{phase}.gpu{gpu}.log"
            handle = log.open("a")
            handles.append(handle)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            p = subprocess.Popen([sys.executable, __file__, "worker", "--job", str(job),
                                  "--phase", phase, "--rank", str(rank), "--workers", str(len(gpus))],
                                 stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            processes.append(p)
            print(f"Started {phase} GPU {gpu}, PID {p.pid}; log: {log}", flush=True)
        last_report = 0
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(f"A {phase} worker failed; see worker logs. Completed chunks preserved.")
            if time.monotonic() - last_report >= 30:
                pattern = "proteins-*.pt" if phase == "encode" else "*_to_*-*.json"
                print(f"{phase}: {sum(1 for _ in cache.glob(pattern))} chunks committed", flush=True)
                last_report = time.monotonic()
            time.sleep(1)
        if any(p.returncode for p in processes):
            raise RuntimeError(f"A {phase} worker failed; see worker logs")
    finally:
        for p in processes:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 10
        while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
            time.sleep(.2)
        for p in processes:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for p in processes:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass  # Blocked I/O may delay exit; the next GPU preflight rejects survivors.
        for handle in handles:
            handle.close()


def controller(args):
    cache = args.output.parent / "parallel_cache"
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refuse_other_controllers()
        signature = make_signature(args)
        marker = cache / "manifest.json"
        if marker.exists() and json.loads(marker.read_text()) != signature:
            raise ValueError("Inputs/implementation changed; use a new output directory")
        if not marker.exists():
            atomic_json(marker, signature)
        signature_id = serial.digest(marker)
        if args.output.exists():
            if json.loads(args.output.read_text()).get("signature") != signature:
                raise ValueError("Completed output has a different signature")
            print(f"Already complete: {args.output}", flush=True)
            return
        proteins, reactions = catalog_ids(args)
        print(f"Full catalog: {len(proteins):,} proteins, {len(reactions):,} reactions", flush=True)
        old = legacy_processes(args.stop_previous_output, args.checkpoint) if args.stop_previous_output else {}
        gpus = args.gpus.split(",")
        check_gpus(gpus, old)
        stop_verified(old, args.stop_previous_output, args.checkpoint)
        # Driver context cleanup may briefly lag process exit.
        for attempt in range(10):
            try:
                check_gpus(gpus)
                break
            except RuntimeError:
                if attempt == 9:
                    raise
                time.sleep(1)
        job = cache / "job.json"
        atomic_json(job, dict(args=vars(args), signature=signature_id))
        print("Protocol and FP32 batch sizes unchanged; independent resumable workers.", flush=True)
        run_workers(job, "encode", gpus)
        print("All encoders exited. Merging completed chunks on CPU...", flush=True)
        merge_catalog(cache, signature_id, len(proteins), args.encoding_chunk_size)
        check_gpus(gpus)
        run_workers(job, "score", gpus)
        results = dict(signature=signature,
                       gold_semantics="source-collapsed reaction-enzyme associations; not all experimentally verified",
                       split_semantics="reaction-held-out; protein overlap is allowed, not an enzyme-cold test",
                       enzyme_to_reaction_protocol="enzymes with test associations ranked against held-out reactions only",
                       reaction_direction_augmentation="forward and reverse chemical directions")
        for direction, queries, candidates in (
            (DIRECTIONS[0], serial.read_ids(args.reaction_query_ids), proteins),
            (DIRECTIONS[1], serial.read_ids(args.enzyme_query_ids), reactions),
        ):
            results[direction] = merge_scores(cache, signature_id, direction, len(queries),
                                               len(candidates), args.score_block_size)
        atomic_json(args.output, results)
        print(json.dumps({k: v for k, v in results.items() if k != "signature"}, indent=2), flush=True)
        print(f"TEST COMPLETE: {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    run = modes.add_parser("run")
    for name in PATH_FLAGS:
        run.add_argument("--" + name.replace("_", "-"), type=str, required=True)
    run.add_argument("--gpus", default="0,1,2,3")
    run.add_argument("--protocol", choices=("reaction_holdout_clustered",), default="reaction_holdout_clustered")
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--target-batch-size", type=int, default=64)
    run.add_argument("--candidate-chunk-size", type=int, default=32768)
    run.add_argument("--encoding-chunk-size", type=int, default=8192)
    run.add_argument("--score-block-size", type=int, default=1024)
    run.add_argument("--catalog-on-gpu", action="store_true")
    run.add_argument("--stop-previous-output", type=str)
    worker = modes.add_parser("worker")
    worker.add_argument("--job", type=Path, required=True)
    worker.add_argument("--phase", choices=("encode", "score"), required=True)
    worker.add_argument("--rank", type=int, required=True)
    worker.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == "worker":
        job = json.loads(args.job.read_text())
        values = argparse.Namespace(**job["args"])
        if not 0 <= args.rank < args.workers:
            raise ValueError("Invalid worker rank")
        function = encode_worker if args.phase == "encode" else score_worker
        function(values, args.job.parent, job["signature"], args.rank, args.workers, "cuda:0")
        return
    for name in (*PATH_FLAGS, "stop_previous_output"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, str(Path(value).resolve()))
    if (min(args.batch_size, args.target_batch_size, args.candidate_chunk_size,
            args.encoding_chunk_size, args.score_block_size) <= 0
            or args.encoding_chunk_size % args.target_batch_size
            or args.score_block_size % args.batch_size):
        raise ValueError("Positive chunk sizes must preserve complete serial encoding/scoring batches")
    args.output = Path(args.output)
    args.checkpoint = Path(args.checkpoint)
    args.stop_previous_output = Path(args.stop_previous_output) if args.stop_previous_output else None
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("Controller terminated; committed chunks preserved"))
    controller(args)


if __name__ == "__main__":
    main()
