#!/usr/bin/env python3
"""Rebuild ONLY ReactZyme's frozen residue cache, without touching old caches.

The controller and preparation use the standard library. GPU extraction and
HDF5 validation run in isolated, timed children using the existing ML environment.
This script never deletes an old cache, edits training configs, or starts training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = ROOT / "scripts/extract_prott5_residue_embeddings.py"
DEFAULT_MODEL = ROOT.parent / "hf_cache/hub/models--Rostlab--prot_t5_xl_half_uniref50-enc/snapshots/94a6abc029ae13029317b140b7424e012bf8dfbf"
PROTOCOLS = ("reaction_smi", "enzyme_smi", "time")
PARTS = ("train", "validation", "test")
GIB = 1024 ** 3


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 ** 2), b""):
            result.update(block)
    return result.hexdigest()


def signature(path):
    st = path.stat()
    return dict(path=str(path.resolve()), size=st.st_size, mtime_ns=st.st_mtime_ns)


def fasta_records(path):
    protein_id, pieces = None, []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if protein_id is not None:
                    if not pieces:
                        raise ValueError(f"Empty sequence: {protein_id}")
                    yield protein_id, "".join(pieces).upper()
                protein_id, pieces = line[1:].split()[0], []
            else:
                if protein_id is None or any(c.isspace() for c in line):
                    raise ValueError(f"Malformed FASTA: {path}")
                pieces.append(line)
        if protein_id is not None:
            if not pieces:
                raise ValueError(f"Empty sequence: {protein_id}")
            yield protein_id, "".join(pieces).upper()


def collect_sequences(project_root):
    """Check all original IDs and conflicting aliases BEFORE ProtT5 normalization."""
    groups, inputs = {}, []
    for split in PROTOCOLS:
        for part in PARTS:
            path = project_root / "data/revised_protocols/reactzyme_paper" / split / f"{part}_candidate_ids.txt"
            inputs.append(signature(path))
            with path.open() as handle:
                rows = [line.strip() for line in handle if line.strip()]
            if not rows or len(rows) != len(set(rows)):
                raise ValueError(f"Empty or duplicate candidate IDs: {path}")
            groups[f"{split}/{part}"] = set(rows)
    required = set().union(*groups.values())
    records = {}
    for split in PROTOCOLS:
        path = project_root / "data/revised_protocols/reactzyme_official" / split / "proteins.fasta"
        inputs.append(signature(path))
        print(f"Reading original {split} sequences: {path}", flush=True)
        for protein_id, sequence in fasta_records(path):
            if protein_id not in required:
                continue
            if protein_id in records and records[protein_id] != sequence:
                raise ValueError(f"Conflicting original sequence: {protein_id}")
            records[protein_id] = sequence
    missing = required - records.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} original sequences; examples: {sorted(missing)[:5]}")
    for item in inputs:
        if signature(Path(item["path"])) != item:
            raise ValueError("Source changed during preparation: " + item["path"])
    return records, groups, inputs


def write_fasta(path, records):
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w") as handle:
        for protein_id, sequence in records:
            handle.write(f">{protein_id}\n{sequence}\n")
    partial.replace(path)


def pilot_ids(records, world_size):
    # Two maximum-length batches per GPU plus a spread of shorter examples.
    ordered = sorted(records, key=lambda key: (-min(len(records[key]), 1022), key))
    selected = ordered[:128 * world_size]
    selected_set = set(selected)
    if len(ordered) > len(selected):
        for i in range(64):
            key = ordered[i * (len(ordered) - 1) // 63]
            if key not in selected_set:
                selected.append(key)
                selected_set.add(key)
    return selected


def prepare(args):
    manifest_path = args.run_root / "preparation.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["source_inputs"]:
            if signature(Path(item["path"])) != item:
                raise ValueError("Preparation source changed; use a new run root")
        for name, expected in manifest["prepared_sha256"].items():
            if digest(args.run_root / name) != expected:
                raise ValueError("Prepared file changed: " + name)
        print(f"Reusing verified preparation: {manifest['proteins']:,} proteins", flush=True)
        return manifest
    records, groups, inputs = collect_sequences(ROOT)
    if len(records) != args.expected_proteins:
        raise ValueError(f"Expected {args.expected_proteins:,} proteins, got {len(records):,}")
    write_fasta(args.run_root / "proteins.fasta", sorted(records.items()))
    selected = pilot_ids(records, len(args.gpus))
    write_fasta(args.run_root / "pilot.fasta", [(key, records[key]) for key in selected])
    residues = sum(min(len(sequence), 1022) for sequence in records.values())
    manifest = dict(
        proteins=len(records), truncated_residues=residues,
        fp16_payload_bytes=residues * 1024 * 2,
        pilot_proteins=len(selected),
        split_candidates={key: len(ids) for key, ids in groups.items()},
        source_inputs=inputs,
        prepared_sha256={name: digest(args.run_root / name) for name in ("proteins.fasta", "pilot.fasta")},
        old_cache_touched=False,
    )
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def storage_probe(args):
    # Only our new, uniquely-created probe is ever removed.
    fd, filename = tempfile.mkstemp(prefix="storage_probe_", dir=args.run_root)
    path = Path(filename)
    payload = os.urandom(8 * 1024 ** 2)
    started = time.monotonic()
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        with path.open("rb") as handle:
            if handle.read() != payload:
                raise ValueError("Storage read-back mismatch")
        report = dict(bytes=len(payload), seconds=time.monotonic() - started,
                      available_bytes=shutil.disk_usage(args.run_root).free,
                      note="Small fsync/read-back probe, not a sustained NFS throughput guarantee")
        preparation = args.run_root / "preparation.json"
        if preparation.exists():
            planned = json.loads(preparation.read_text())["fp16_payload_bytes"]
            # A VDS merge does not duplicate the full residue payload. This is
            # deliberately conservative on resume: no credit for partial files.
            report["required_free_bytes"] = int(planned * 1.10) + 15 * GIB
            if report["available_bytes"] < report["required_free_bytes"]:
                raise ValueError(f"Insufficient filesystem space: {report}")
        atomic_json(args.run_root / "storage_probe.json", report)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        path.unlink(missing_ok=True)


def paths(args, tag):
    base = args.run_root / ("pilot_features" if tag == "pilot" else "features")
    return base / "proteins_prott5_residue.h5", base / "shards"


def extraction_command(args, tag, rank=None):
    output, shards = paths(args, tag)
    command = [str(args.python), str(EXTRACTOR), "--fasta",
               str(args.run_root / ("pilot.fasta" if tag == "pilot" else "proteins.fasta")),
               "--output", str(output), "--tmp-dir", str(shards),
               "--model-name", str(args.model), "--world-size", str(len(args.gpus)),
               "--max-sequence-length", "1022", "--sequence-truncation", "ends_center",
               "--dtype", "float16", "--compression", "none",
               "--batch-size", str(args.batch_size), "--max-tokens-per-batch", str(args.max_tokens),
               "--length-sort", "--padded-token-budget", "--progress-every", "1000",
               "--checkpoint-every", "1000", "--merge-order", "shard", "--merge-storage", "virtual"]
    if rank is None:
        command.append("--merge-only")
    else:
        command += ["--rank", str(rank), "--device", "cuda", "--resume"]
    return command


def verify(args, tag):
    # Imports intentionally stay out of the CPU/controller startup path.
    import h5py
    import numpy as np
    records = list(fasta_records(args.run_root / ("pilot.fasta" if tag == "pilot" else "proteins.fasta")))
    lengths = {key: min(len(sequence), 1022) for key, sequence in records}
    output, shards = paths(args, tag)
    total_residues = sum(lengths.values())
    all_shard_ids, shard_files = [], []
    for rank in range(len(args.gpus)):
        path = shards / f"proteins_prott5_residue.shard{rank:02d}-of-{len(args.gpus):02d}.h5"
        expected = sorted([(i, key, lengths[key]) for i, (key, _) in enumerate(records)
                           if i % len(args.gpus) == rank], key=lambda row: (row[2], row[0]))
        with h5py.File(path, "r") as handle:
            ids = list(handle["ids"].asstr()[:])
            if ids != [row[1] for row in expected] or list(handle["indices"][:]) != [row[0] for row in expected]:
                raise ValueError(f"Wrong shard IDs/order: {path}")
            offsets = handle["offsets"][:]
            if len(offsets) != len(ids) + 1 or offsets[0] != 0 or not np.array_equal(np.diff(offsets), [row[2] for row in expected]):
                raise ValueError(f"Wrong shard offsets: {path}")
            if handle["vectors"].shape != (int(offsets[-1]), 1024) or handle["vectors"].dtype != np.dtype("float16"):
                raise ValueError(f"Wrong shard shape/dtype: {path}")
            expected_attrs = dict(model_name=str(args.model), max_sequence_length=1022,
                                  sequence_truncation="ends_center", rank=rank,
                                  world_size=len(args.gpus), processed_count=len(ids),
                                  source_fasta=str((args.run_root / ("pilot.fasta" if tag == "pilot" else "proteins.fasta")).resolve()),
                                  length_sort=True, padded_token_budget=True)
            for key, value in expected_attrs.items():
                if handle.attrs.get(key) != value:
                    raise ValueError(f"Wrong shard attribute {key}: {path}")
        all_shard_ids.extend(ids)
        shard_files.append(path)
    with h5py.File(output, "r") as handle:
        if list(handle["ids"].asstr()[:]) != all_shard_ids or len(set(all_shard_ids)) != len(records):
            raise ValueError("Merged protein IDs do not exactly cover the prepared catalog")
        offsets = handle["offsets"][:]
        if len(offsets) != len(records) + 1 or offsets[0] != 0 or not np.array_equal(np.diff(offsets), [lengths[key] for key in all_shard_ids]):
            raise ValueError("Merged lengths/offsets are inconsistent")
        vectors = handle["vectors"]
        if vectors.shape != (total_residues, 1024) or vectors.dtype != np.dtype("float16") or not vectors.is_virtual:
            raise ValueError("Wrong merged shape/dtype/storage")
        actual_sources = {str(Path(item.file_name).resolve()) for item in vectors.virtual_sources()}
        if actual_sources != {str(path.resolve()) for path in shard_files}:
            raise ValueError("Virtual dataset does not reference exactly this run's shards")
        # HDF5 may silently fill a missing VDS source with zeroes. Opening every
        # shard above is essential; additionally test real source-to-VDS mapping.
        base = 0
        for path in shard_files:
            with h5py.File(path, "r") as source:
                size = source["vectors"].shape[0]
                for position in (0, max(0, size - 8)):
                    count = min(8, size - position)
                    if not np.array_equal(vectors[base + position:base + position + count],
                                          source["vectors"][position:position + count]):
                        raise ValueError("Virtual dataset read-back differs from its source")
                base += size
        if tag == "pilot" or args.full_scan:
            for start in range(0, total_residues, 32768):
                if not np.isfinite(vectors[start:start + 32768]).all():
                    raise ValueError("Nonfinite stored residue vectors")
    # Exercise the actual CIRCE-v2 data reader, not only direct HDF5 slices.
    sys.path.insert(0, str(ROOT))
    import torch
    from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
    torch.set_num_threads(4)
    dataset = ResidueEmbedDataset(str(output), max_tokens=1022)
    rng = random.Random(42)
    sampled = set(rng.sample(all_shard_ids, min(128, len(all_shard_ids))))
    # Include every shard's first and last entries, not just random rows.
    for rank in range(len(args.gpus)):
        rank_ids = [key for i, (key, _) in enumerate(records) if i % len(args.gpus) == rank]
        sampled.update((min(rank_ids, key=lambda key: lengths[key]), max(rank_ids, key=lambda key: lengths[key])))
    try:
        for key in sampled:
            tensor = dataset[key]["residue_embeddings"]
            if tensor.shape != (lengths[key], 1024) or not torch.isfinite(tensor).all() or not bool(tensor.any()):
                raise ValueError(f"Training-reader validation failed: {key}")
    finally:
        dataset.close()
    report = dict(proteins=len(records), residues=total_residues,
                  sampled_training_reads=len(sampled), full_stored_finite_scan=tag == "pilot" or args.full_scan,
                  all_extraction_outputs_finite_checked=True, output=str(output),
                  output_signature=signature(output), shard_signatures=[signature(path) for path in shard_files])
    atomic_json(args.run_root / f"{tag}_validation.json", report)
    print(json.dumps(report, indent=2), flush=True)


class Runner:
    def __init__(self, args):
        self.args, self.children = args, []
        self.env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}

    def stage(self, name, commands, timeout=None):
        atomic_json(self.args.run_root / "status.json", dict(stage=name, host=socket.gethostname(), pid=os.getpid(), time=time.time()))
        print(f"Stage: {name}; logs: {self.args.run_root / 'logs'}", flush=True)
        started = time.monotonic()
        processes, handles = [], []
        try:
            for suffix, command, extra_env in commands:
                handle = (self.args.run_root / "logs" / f"{name}{suffix}.log").open("a")
                handles.append(handle)
                child = subprocess.Popen(command, cwd=ROOT, env={**self.env, **extra_env},
                                         stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                self.children.append(child)
                processes.append(child)
            while True:
                codes = [child.poll() for child in processes]
                if any(code not in (None, 0) for code in codes):
                    raise RuntimeError(f"{name} failed; inspect its log(s)")
                if all(code == 0 for code in codes):
                    return
                if timeout is not None and time.monotonic() - started > timeout:
                    raise TimeoutError(f"{name} timed out; inspect storage/model access before retrying")
                time.sleep(1)
        except BaseException:
            self.stop()
            raise
        finally:
            for handle in handles:
                handle.close()

    def stop(self):
        live = [child for child in self.children if child.poll() is None]
        for child in live:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 10
        while any(child.poll() is None for child in live) and time.monotonic() < deadline:
            time.sleep(0.2)
        for child in live:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 2
        while any(child.poll() is None for child in live) and time.monotonic() < deadline:
            time.sleep(0.1)
        survivors = [child.pid for child in live if child.poll() is None]
        if survivors:
            print(f"BLOCKED processes remain: {survivors}. Do not relaunch. Controller lock retained.", flush=True)
        return survivors


def gpu_check(args):
    selected = ",".join(args.gpus)
    common = ["nvidia-smi", "-i", selected]
    result = subprocess.run([*common, "--query-compute-apps=pid", "--format=csv,noheader"],
                            check=True, text=True, capture_output=True, timeout=10)
    if result.stdout.strip():
        raise RuntimeError("Selected GPUs have compute processes; none were stopped")
    result = subprocess.run([*common, "--query-gpu=index,name,memory.free", "--format=csv,noheader,nounits"],
                            check=True, text=True, capture_output=True, timeout=10)
    rows = [row.split(",") for row in result.stdout.strip().splitlines()]
    if len(rows) != len(args.gpus):
        raise RuntimeError("GPU query returned the wrong number of devices")
    for row in rows:
        if "H200" not in row[1] or int(row[2].strip()) < 125 * 1024:
            raise RuntimeError("This profile requires idle H200s with at least 125 GiB free per GPU")
    print(result.stdout, flush=True)


def helper_command(args, action, ml=False):
    command = [str(args.python) if ml else "/usr/bin/python3", str(Path(__file__).resolve()), action,
               "--run-root", str(args.run_root), "--python", str(args.python), "--model", str(args.model),
               "--gpus", ",".join(args.gpus), "--batch-size", str(args.batch_size),
               "--max-tokens", str(args.max_tokens), "--expected-proteins", str(args.expected_proteins)]
    if args.full_scan:
        command.append("--full-scan")
    return command


def run(args):
    if socket.gethostname().split(".")[0] != args.expected_host:
        raise RuntimeError(f"Run the GPU stages on {args.expected_host}; no GPU processes started")
    lock = args.run_root / ".controller.lock"
    lock.mkdir()  # Never take over an existing/stale lock automatically.
    atomic_json(lock / "owner.json", dict(host=socket.gethostname(), pid=os.getpid(), time=time.time()))
    runner = Runner(args)
    try:
        def request_stop(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, request_stop)
        ready = args.run_root / "READY.json"
        if ready.exists():
            ready.rename(args.run_root / f"READY.previous.{time.time_ns()}.json")
        runner.stage("storage_probe", [("", helper_command(args, "probe"), {})], 30)
        runner.stage("prepare", [("", helper_command(args, "prepare"), {})], 600)
        runner.stage("space_check", [("", helper_command(args, "probe"), {})], 30)
        runner.stage("runtime_probe", [("", [str(args.python), "-u", "-c",
                     "print('Checking ML runtime imports', flush=True); "
                     "import torch, h5py, numpy, transformers; "
                     "print('ML runtime imports OK', flush=True)"], {"CUDA_VISIBLE_DEVICES": ""})], 180)
        for tag in ("pilot", "full"):
            output, _ = paths(args, tag)
            if not output.exists():
                gpu_check(args)
                runner.stage(f"{tag}_extract", [(f"_rank{rank}", extraction_command(args, tag, rank),
                                                 {"CUDA_VISIBLE_DEVICES": gpu}) for rank, gpu in enumerate(args.gpus)],
                             900 if tag == "pilot" else None)
                runner.stage(f"{tag}_merge", [("", extraction_command(args, tag), {"CUDA_VISIBLE_DEVICES": ""})], 600)
            runner.stage(f"{tag}_verify", [("", helper_command(args, f"verify-{tag}", ml=True),
                                            {"CUDA_VISIBLE_DEVICES": ""})], 1800 if args.full_scan else 600)
        atomic_json(args.run_root / "READY.json", dict(
            protein_residue_embeds_path=str(paths(args, "full")[0]),
            preparation_sha256=digest(args.run_root / "preparation.json"),
            validation_sha256=digest(args.run_root / "full_validation.json"),
            old_cache_deleted=False, training_started=False,
            note="ReactZyme-only replacement. Keep its VDS source shards. No existing training config was edited."))
        atomic_json(args.run_root / "status.json", dict(stage="complete", time=time.time()))
        print("READY: fresh ReactZyme cache verified. Old cache and all training runs were left untouched.", flush=True)
    except BaseException as exc:
        atomic_json(args.run_root / "status.json", dict(stage="failed", error=str(exc), time=time.time()))
        raise
    finally:
        survivors = runner.stop()
        if not survivors:
            (lock / "owner.json").unlink()
            lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "probe", "run", "verify-pilot", "verify-full"))
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/reactzyme_prott5_reextract_20260910")
    parser.add_argument("--python", type=Path, default=ROOT.parent / "env/bin/python")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--expected-proteins", type=int, default=178327)
    parser.add_argument("--expected-host", default="slurm-node-014")
    parser.add_argument("--full-scan", action="store_true", help="Additionally scan every stored full-cache vector for finiteness")
    args = parser.parse_args()
    print(f"Cache action: {args.action}; checking paths and pinned extraction recipe", flush=True)
    args.run_root, args.model, args.python = args.run_root.resolve(), args.model.resolve(), args.python.absolute()
    args.gpus = args.gpus.split(",")
    if len(set(args.gpus)) != len(args.gpus) or not all(gpu.isdigit() for gpu in args.gpus):
        parser.error("GPU indices must be unique integers")
    if not args.run_root.is_relative_to(ROOT / "runs") or args.run_root == ROOT / "runs":
        parser.error("Use a dedicated directory beneath this repository's runs directory")
    if args.batch_size <= 0 or args.max_tokens < 1023 or args.expected_proteins < len(args.gpus):
        parser.error("Invalid extraction size or token budget")
    if args.action == "run" and socket.gethostname().split(".")[0] != args.expected_host:
        parser.error(f"GPU launch must be on {args.expected_host}; current host is {socket.gethostname()}")
    recipe = dict(schema=1, world_size=len(args.gpus), model=str(args.model),
                  batch_size=args.batch_size, max_tokens=args.max_tokens,
                  max_sequence_length=1022, sequence_truncation="ends_center", dtype="float16",
                  expected_proteins=args.expected_proteins, extractor_sha256=digest(EXTRACTOR),
                  model_files=[signature(args.model / name) for name in
                               ("config.json", "pytorch_model.bin", "spiece.model", "tokenizer_config.json", "special_tokens_map.json")])
    if args.run_root.exists() and not (args.run_root / "recipe.json").exists():
        parser.error("Existing directory is not owned by this cache pipeline; choose a fresh run root")
    args.run_root.mkdir(parents=True, exist_ok=True)
    recipe_path = args.run_root / "recipe.json"
    if recipe_path.exists():
        if json.loads(recipe_path.read_text()) != recipe:
            parser.error("Extraction recipe changed; use a new run root (do not overwrite old shards)")
    else:
        atomic_json(recipe_path, recipe)
    (args.run_root / "logs").mkdir(exist_ok=True)
    if args.action == "prepare":
        prepare(args)
    elif args.action == "probe":
        storage_probe(args)
    elif args.action.startswith("verify-"):
        verify(args, args.action.removeprefix("verify-"))
    else:
        run(args)


if __name__ == "__main__":
    main()
