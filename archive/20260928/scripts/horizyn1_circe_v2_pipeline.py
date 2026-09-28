#!/usr/bin/env python3
"""Locked, restartable CIRCE-v2 preparation, feature pilot, training and test runner.

No stage is launched on import. This controller uses the existing extraction
programs; all pretrained encoders and the SLEEC scorer remain frozen.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
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
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("preflight", "prepare", "labels", "index", "pilot", "features", "train", "test")
LOCAL_SCRATCH_FILESYSTEMS = frozenset({"ext2", "ext3", "ext4", "xfs", "btrfs", "zfs", "overlay"})


def scratch_filesystem(path: Path) -> str:
    """Fail closed for unknown/network/memory-backed scratch filesystems."""
    try:
        result = subprocess.check_output(
            ["findmnt", "--noheadings", "--output", "FSTYPE", "--target", str(path)],
            text=True, stderr=subprocess.PIPE,
        ).strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot determine scratch filesystem for {path}; findmnt is required") from exc
    if len(result) != 1:
        raise RuntimeError(f"Ambiguous scratch filesystem for {path}: {result}")
    return result[0].strip()


def inspect_scratch_root(path: Path, minimum_bytes: int) -> dict:
    if not path.is_absolute():
        raise RuntimeError(f"Scratch root must be an absolute path: {path}")
    path = path.resolve()
    if not path.is_dir() or not os.access(path, os.W_OK | os.X_OK):
        raise RuntimeError(f"Scratch root is not an existing writable directory: {path}")
    filesystem = scratch_filesystem(path)
    if filesystem not in LOCAL_SCRATCH_FILESYSTEMS:
        raise RuntimeError(f"Scratch root {path} uses {filesystem}; a local disk filesystem is required, not network/tmpfs storage")
    free = shutil.disk_usage(path).free
    if free < minimum_bytes:
        raise RuntimeError(f"Scratch root {path} has {free / 1e9:.1f} GB free; need at least {minimum_bytes / 1e9:.1f} GB")
    return {"scratch_root": str(path), "filesystem": filesystem, "free_bytes": free}


def signature(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns, "inode": stat.st_ino}


def validate_reconstruction_audit(
    audit: dict, expected_paths: Iterable[Path] | None = None,
) -> list[str]:
    """Validate an existing audit of shared files across cluster nodes.

    The legacy audit stores [device, inode, size, mtime_ns, ctime_ns]. A
    filesystem's device number is local to its mount on a node, so it is not
    portable. All other fields must still match; this does not accept copied,
    replaced, or modified data or rewrite the audit's recorded signatures.
    """
    if not isinstance(audit, dict) or audit.get("status") != "passed" or audit.get("errors"):
        raise RuntimeError("The clustered reconstruction audit must pass first")
    recorded = audit.get("artifact_signatures")
    if not isinstance(recorded, dict) or not recorded:
        raise RuntimeError("Invalid reconstruction audit: missing artifact_signatures")
    for path, expected in recorded.items():
        if (not isinstance(path, str) or not path
                or not isinstance(expected, list) or len(expected) != 5
                or any(type(value) is not int for value in expected)):
            raise RuntimeError(f"Invalid reconstruction audit signature: {path!r}")
    if expected_paths is not None:
        covered = {Path(path).resolve() for path in recorded}
        missing = sorted(str(Path(path).resolve()) for path in expected_paths
                         if Path(path).resolve() not in covered)
        if missing:
            raise RuntimeError("Reconstruction audit missing required artifacts: " + ", ".join(missing))
    device_changes = []
    fields = ("device", "inode", "size", "mtime_ns", "ctime_ns")
    for path, expected in recorded.items():
        info = Path(path).stat()
        actual = [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
        changes = [f"{field}: expected {before}, actual {after}"
                   for field, before, after in zip(fields[1:], expected[1:], actual[1:])
                   if before != after]
        if changes:
            raise RuntimeError(f"Stale reconstruction audit: {path} ({'; '.join(changes)}). "
                               "Only node-local device differences are allowed; re-audit changed data.")
        if actual[0] != expected[0]:
            device_changes.append(path)
    return device_changes


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def fasta_records(path: Path) -> Iterable[tuple[str, str]]:
    protein_id, sequence = None, []
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                if protein_id is not None:
                    yield protein_id, "".join(sequence)
                protein_id, sequence = line[1:].split()[0], []
            else:
                sequence.append(line.strip())
    if protein_id is not None:
        yield protein_id, "".join(sequence)


def csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ",")


def write_csv(path: Path, columns: list[str], rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def indexed_epoch_steps(pair_count: int, batch_size: int, world_size: int) -> int:
    """Match IndexedTypedNegativeBatchSampler's base-edge sweep exactly."""
    if pair_count <= 0 or batch_size <= 0 or batch_size % 20 or world_size <= 0:
        raise ValueError("Positive pair/world counts and a batch size divisible by 20 are required")
    negatives = batch_size * 3 // 20
    base_rows = batch_size - 3 * negatives  # 55% base + 30% endpoint support + 15% negatives.
    global_base_rows = base_rows * world_size
    return (pair_count + global_base_rows - 1) // global_base_rows


def validate_h5(path: Path, expected: set[str], dim: int, *, ragged: bool = False,
                optional: bool = False, sample_only: bool = False) -> dict:
    import h5py
    import numpy as np
    with h5py.File(path, "r") as handle:
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in handle["ids"][:]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate feature IDs: {path}")
        unexpected = set(ids) - expected
        missing = expected - set(ids)
        if unexpected or (missing and not optional) or not ids:
            raise ValueError(f"Feature ID mismatch in {path}: missing={len(missing)}, unexpected={len(unexpected)}")
        vector_names = ["reactant_vectors", "product_vectors"] if ragged else ["vectors"]
        for name in vector_names:
            vectors = handle[name]
            if vectors.ndim != 2 or vectors.shape[1] != dim:
                raise ValueError(f"Unexpected {name} shape {vectors.shape}: {path}")
            if vectors.is_virtual:
                for source in vectors.virtual_sources():
                    source_path = Path(os.fsdecode(source.file_name))
                    if not source_path.is_absolute():
                        source_path = path.parent / source_path
                    if not source_path.is_file():
                        raise ValueError(f"Missing immutable VDS shard: {source_path}")
            offset_name = name.replace("vectors", "offsets")
            if offset_name in handle:
                offsets = handle[offset_name][:]
                if (len(offsets) != len(ids) + 1 or offsets[0] != 0
                        or offsets[-1] != len(vectors) or np.any(np.diff(offsets) <= 0)):
                    raise ValueError(f"Invalid ragged offsets: {path}/{offset_name}")
            elif len(vectors) != len(ids):
                raise ValueError(f"Feature rows do not match IDs: {path}")
            starts = range(0, len(vectors), 4096)
            if sample_only and len(vectors) > 4096 * 64:
                starts = np.linspace(0, max(0, len(vectors) - 4096), 64, dtype=int)
            for start in starts:
                if not np.isfinite(vectors[start:start + 4096]).all():
                    raise ValueError(f"Non-finite features in {path}/{name} near row {start}")
        return {"rows": len(ids), "missing": len(missing), "coverage": len(ids) / len(expected),
                "finiteness": "sampled_64_blocks" if sample_only else "all_vectors"}


def validate_chemistry(path: Path, expected: set[str]) -> dict:
    import numpy as np
    with np.load(path, allow_pickle=True) as payload:
        ids = list(map(str, payload["ids"]))
        vectors, mask = payload["vectors"], payload["mask"]
        if set(ids) != expected or len(ids) != len(expected):
            raise ValueError(f"Chemistry feature ID mismatch: {path}")
        if vectors.shape != (len(ids), 617) or mask.shape != (len(ids),):
            raise ValueError(f"Unexpected chemistry feature dimensions: {path}")
        if not np.isfinite(vectors).all():
            raise ValueError(f"Non-finite chemistry features: {path}")
        return {"rows": len(ids), "valid_rows": int(mask.sum()), "coverage": float(mask.mean())}


class Pipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run = Path(args.run_root).resolve()
        self.template = ROOT / "configs" / ("horizyn1_circe_v2_h200.yaml" if args.profile == "h200" else "horizyn1_circe_v2.yaml")
        self.source = Path(args.data_root).resolve()
        self.data = self.run / "data"
        self.features = self.run / "features"
        self.logs = self.run / "logs"
        self.state = self.run / "state"
        self.python = os.environ.get("PYTHON_BIN", str(ROOT.parent / "env/bin/python"))
        self.setup_python = os.environ.get("SETUP_PYTHON_BIN", str(ROOT.parent / ".capability-run-py/bin/python"))
        self.annotation_python = os.environ.get("ANNOTATION_PYTHON", str(ROOT / ".venv-horizyn1-annotations/bin/python"))
        self.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
        if len(self.gpus) != int(os.environ.get("GPU_COUNT", "4")) or any(not gpu for gpu in self.gpus):
            raise ValueError("GPU_COUNT must equal the number of CUDA_VISIBLE_DEVICES entries")
        self.sleec = Path(os.environ.get("SLEEC_CHECKPOINT", str(ROOT / "checkpoints/SLEEC/sleec_stage1_prott5_uniref90_msa_4gpu_20260601_235157/best.ckpt")))
        self.prott5 = os.environ.get("PROTT5_MODEL", str(ROOT.parent / "hf_cache/hub/models--Rostlab--prot_t5_xl_half_uniref50-enc/snapshots/94a6abc029ae13029317b140b7424e012bf8dfbf"))
        self.reaction_t5 = Path(os.environ.get("REACTION_T5_MODEL_PATH", str(ROOT.parent / "hf_cache/hub/models--sagawa--ReactionT5v2-forward/snapshots/933114058cb2604dc1bf536dbebdfcefbe83d4fc")))
        self.cofactor_dictionary = Path(os.environ.get("COFACTOR_DICTIONARY", str(ROOT / "data/processed/capability_features/train_exact_rhea_reconstructed/cofactor_dictionary.csv")))
        self.mmseqs = os.environ.get("MMSEQS_BIN", shutil.which("mmseqs") or str(ROOT / ".deps/mmseqs/bin/mmseqs"))
        self.children: list[subprocess.Popen] = []
        self.runtime_env = ({name: str(args.cpu_threads) for name in
                            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
                            if args.cpu_threads is not None else {})
        for directory in (self.logs, self.state, self.data, self.features):
            directory.mkdir(parents=True, exist_ok=True)

    def work_directory(self, *, check_space: bool = True) -> Path:
        """Bind new run work without moving any old/live preparation database."""
        marker = self.state / "work_directory.json"
        previous = json.loads(marker.read_text()) if marker.exists() else None
        if self.args.scratch_root is None:
            binding = {"schema": 1, "run_root": str(self.run), "profile": self.args.profile,
                       "storage": "shared", "work_dir": str(self.run / "work/preparation")}
            if previous is not None and previous != binding:
                raise RuntimeError("Work-directory/profile binding changed; use a new RUN_ROOT, never migrate a live database")
        else:
            info = inspect_scratch_root(self.args.scratch_root,
                                        int(self.args.min_scratch_gb * 1e9) if check_space else 0)
            identity = {"schema": 1, "run_root": str(self.run), "profile": self.args.profile,
                        "storage": "local_opt_in", "scratch_root": info["scratch_root"],
                        "hostname": socket.gethostname(), "uid": os.getuid()}
            if previous is not None:
                if any(previous.get(key) != value for key, value in identity.items()):
                    raise RuntimeError("Local scratch path/node/profile changed; resume on the original node or use a new RUN_ROOT")
                binding = previous
                owner = Path(binding["owned_root"]) / ".circe_owner.json"
                if not owner.is_file() or json.loads(owner.read_text()) != binding:
                    raise RuntimeError("Owned local scratch is missing or changed; use an explicit new-run/rebuild, not silent cache reuse")
            else:
                run_hash = hashlib.sha256(str(self.run).encode()).hexdigest()[:12]
                owned = Path(tempfile.mkdtemp(prefix=f"circe-{os.getuid()}-{run_hash}-", dir=info["scratch_root"]))
                binding = {**identity, "owned_root": str(owned), "work_dir": str(owned / "preparation")}
                atomic_json(owned / ".circe_owner.json", binding)
        if previous is None:
            atomic_json(marker, binding)
            self.log(f"Preparation work directory: {binding['work_dir']} ({binding['storage']}); existing run data is not moved")
        return Path(binding["work_dir"])

    def log(self, message: str) -> None:
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}", flush=True)

    def terminate(self, signum: int, frame=None) -> None:
        # Child extractors/trainers use separate sessions. Snapshot descendants,
        # not just the controller's process group, and verify PID start identity.
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from scripts.stop_horizyn1_circe_v2 import alive, process_tree, read_process, send_signal
        for watched in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            signal.signal(watched, signal.SIG_IGN)
        self.log(f"Received signal {signum}; forwarding TERM to verified job descendants across sessions")
        root_process = read_process(os.getpid())
        observed = process_tree(root_process) if root_process is not None else {}
        observed.pop(os.getpid(), None)
        for process in observed.values():
            send_signal(process, signal.SIGTERM)
        next_report = time.monotonic() + 30
        while any(alive(process) for process in observed.values()):
            if root_process is not None:
                for pid, process in process_tree(root_process).items():
                    if pid != os.getpid() and pid not in observed:
                        observed[pid] = process
                        send_signal(process, signal.SIGTERM)
            for child in self.children:
                child.poll()  # Reap direct children without waiting for unrelated PIDs.
            if time.monotonic() >= next_report:
                self.log("Still waiting for owned job processes to stop; verified stop helper supports --force-after-timeout if necessary")
                next_report = time.monotonic() + 30
            time.sleep(.2)
        raise SystemExit(128 + signum)

    def command(self, command: list, name: str, *, env: dict | None = None) -> None:
        command = list(map(str, command))
        self.log(f"{name}: {' '.join(command)}")
        with (self.logs / f"{name}.log").open("a") as output:
            child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **self.runtime_env, **(env or {})},
                                     stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            self.children.append(child)
            (self.logs / f"{name}.pid").write_text(str(child.pid) + "\n")
            code = child.wait()
        if code:
            raise RuntimeError(f"{name} failed with exit code {code}; see {self.logs / (name + '.log')}")

    def step(self, name: str, inputs: list[Path], outputs: list[Path], action: Callable[[], None],
             parameters: dict | None = None) -> None:
        marker = self.state / f"{name}.json"
        fingerprint = {"inputs": [signature(p) for p in inputs], "parameters": parameters or {},
                       "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        old = json.loads(marker.read_text()) if marker.exists() else None
        if old and old["fingerprint"] != fingerprint:
            raise RuntimeError(f"Inputs/options changed for {name}. Use a new RUN_ROOT; refusing stale cache reuse.")
        if old and old.get("status") == "complete":
            if old["outputs"] != [signature(path) for path in outputs]:
                raise RuntimeError(f"Completed {name} outputs changed; investigate or use a new RUN_ROOT")
            self.validate_records(old.get("inventory", []), name)
            self.log(f"{name}: verified complete, skipping")
            return
        if not old and any(path.exists() for path in outputs):
            raise RuntimeError(f"Unowned outputs already exist for {name}; use a new RUN_ROOT")
        if not old and name in ("labels", "index", "pilot_index"):
            directory = self.run / "pilot/index" if name == "pilot_index" else self.data / name
            if directory.exists() and any(directory.iterdir()):
                raise RuntimeError(f"Unowned nonempty build directory for {name}: {directory}")
        start = time.time()
        atomic_json(marker, {"status": "running", "fingerprint": fingerprint, "started": start})
        self.log(f"{name}: starting")
        try:
            action()
            records = [signature(path) for path in outputs]
        except BaseException as exc:
            atomic_json(marker, {"status": "failed", "fingerprint": fingerprint,
                                 "started": start, "error": str(exc)})
            raise
        atomic_json(marker, {"status": "complete", "fingerprint": fingerprint, "outputs": records,
                             "inventory": self.inventory(name), "started": start, "seconds": time.time() - start})
        self.log(f"{name}: complete in {(time.time() - start) / 60:.1f} minutes")

    def require_stage(self, stage: str) -> None:
        marker = self.state / f"{stage}.json"
        if not marker.exists() or json.loads(marker.read_text()).get("status") != "complete":
            raise RuntimeError(f"Run stage '{stage}' first (or use 'all')")
        saved = json.loads(marker.read_text())
        if stage == "prepare" and (self.state / "work_directory.json").exists():
            work = self.work_directory(check_space=False)
            if not (work / "preparation_owner.json").is_file() or not (work / "preparation.sqlite").is_file():
                raise RuntimeError("Bound preparation work is missing; restore a consistent stopped backup or use a new RUN_ROOT")
        controller = saved.get("fingerprint", {}).get("controller_sha256")
        if controller and controller != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise RuntimeError(f"Stale {stage} stage controller implementation; use a new RUN_ROOT")
        records = [*saved.get("outputs", []), *saved.get("fingerprint", {}).get("inputs", []),
                   *saved.get("virtual_shards", []), *saved.get("inventory", []), *saved.get("artifacts", []),
                   *saved.get("required_inputs", [])]
        for key in ("config", "report", "last", "best_checkpoint"):
            if saved.get(key):
                records.append(saved[key])
        self.validate_records(records, stage)
        for dependency in {"labels": ["prepare"], "index": ["labels"],
                           "pilot": ["index", "pilot_selection", "pilot_prott5", "pilot_reactiont5v2", "pilot_unimol2", "pilot_chiro", "pilot_chemistry", "pilot_index", "pilot_training"],
                           "features": ["pilot", "full_prott5", "full_reactiont5v2", "full_unimol2", "full_chiro", "full_chemistry"],
                           "train": ["features"]}.get(stage, []):
            self.require_stage(dependency)

    def validate_records(self, records: list[dict], stage: str) -> None:
        for expected in records:
            if signature(Path(expected["path"])) != expected:
                raise RuntimeError(f"Stale {stage} stage artifact: {expected['path']}")

    def inventory(self, name: str) -> list[dict]:
        paths: list[Path] = []
        if name == "prepare":
            for suffix in ("*.csv", "*.txt", "*.json"):
                paths.extend(self.data.glob(suffix))
                paths.extend((self.data / "panels").glob(suffix))
        elif name in ("labels", "index", "pilot_index"):
            directory = self.run / "pilot/index" if name == "pilot_index" else self.data / name
            paths.extend(path for path in directory.iterdir() if path.is_file())
        elif name.endswith("_prott5"):
            directory = self.run / "pilot/features" if name.startswith("pilot_") else self.features
            paths.extend(directory.glob("proteins_prott5_residue_shards/*.h5"))
        return [signature(path) for path in sorted(set(paths))]

    def rebuild_directory(self, directory: Path, action: Callable[[], None]) -> None:
        """Retry only a directory whose ownership was checked by step()."""
        if directory.exists() and any(directory.iterdir()):
            archived = directory.with_name(f".{directory.name}.interrupted.{time.time_ns()}")
            directory.rename(archived)
            self.log(f"Preserved incomplete pipeline-owned output at {archived}; rebuilding {directory}")
        action()

    def require_free_gpus(self) -> None:
        if self.args.allow_busy_gpus:
            self.log("WARNING: --allow-busy-gpus was explicitly selected")
            return
        command = ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
        devices = subprocess.check_output(command, text=True).splitlines()
        uuids = {line.split(",")[1].strip() for line in devices
                 if line.split(",")[0].strip() in self.gpus or line.split(",")[1].strip() in self.gpus}
        if len(uuids) != len(self.gpus):
            raise RuntimeError("Selected CUDA devices could not be resolved with nvidia-smi")
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                             "--format=csv,noheader,nounits"], text=True).splitlines()
        busy = [line for line in processes if line.split(",")[0].strip() in uuids]
        if busy:
            raise RuntimeError("Selected GPUs have compute jobs. Wait or select free GPUs; no jobs were stopped: " + "; ".join(busy))

    def preflight(self) -> None:
        required = [self.template, self.source / "clustered/proteins.fasta", self.source / "clustered/clusters.tsv",
                    self.source / "raw/raw_pairs.tsv", self.source / "raw/raw_reactions.tsv",
                    self.source / "annotations/native_uniprot.tsv.gz",
                    self.source / "annotations/cofactor_v2/reaction_annotations.json",
                    self.source / "logs/clustered_integrity_audit.json", self.sleec,
                    Path(self.prott5) / "config.json", Path(self.prott5) / "pytorch_model.bin",
                    self.reaction_t5 / "config.json", self.reaction_t5 / "model.safetensors", self.cofactor_dictionary,
                    Path(os.environ.get("UNIMOL_WEIGHT_DIR", str(ROOT.parent / "unimol_weights"))) / "modelzoo/84M/checkpoint.pt",
                    ROOT / ".deps/ChIRo/paper_results/RS_experiment/ChIRo/results_RS_ChIRo_seed1/best_model.pt",
                    ROOT / ".deps/ChIRo/paper_results/RS_experiment/ChIRo/params_RS_ChIRo.json"]
        for path in required:
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError(f"Missing required input: {path}")
        audit = json.loads((self.source / "logs/clustered_integrity_audit.json").read_text())
        device_changes = validate_reconstruction_audit(audit, [
            self.source / relative for relative in (
                "clustered/clustered_manifest.json", "clustered/clusters.tsv",
                "clustered/pairs.tsv", "clustered/proteins.fasta",
                "raw/raw_manifest.json", "raw/raw_pairs.tsv",
                "raw/raw_proteins.fasta", "raw/raw_reactions.tsv",
            )
        ])
        if device_changes:
            self.log(f"Reconstruction audit accepted {len(device_changes)} node-local device-ID "
                     "differences; inode, size, mtime and ctime still match for every audited file.")
        if not Path(self.mmseqs).is_file() or not os.access(self.mmseqs, os.X_OK):
            raise RuntimeError("mmseqs is required for sequence-family grouping; add its binary directory to PATH")
        self.command([self.python, "-c", "import torch, lightning, transformers, sentencepiece, h5py, yaml; print(torch.__version__)"], "preflight_training_env")
        self.command([self.annotation_python, "-c", "import numpy, pandas, pyarrow, yaml"], "preflight_annotation_env")
        self.command([self.setup_python, "-c", "import numpy, rdkit, pandas, h5py"], "preflight_chemistry_env")
        # Local-only loading checks tokenizer/model configuration, without allocating model/GPU memory.
        self.command([self.python, "-c", "from transformers import AutoConfig, AutoTokenizer, T5Tokenizer; import sys; [AutoConfig.from_pretrained(p, local_files_only=True) for p in sys.argv[1:]]; T5Tokenizer.from_pretrained(sys.argv[1], do_lower_case=False, local_files_only=True); AutoTokenizer.from_pretrained(sys.argv[2], trust_remote_code=True, local_files_only=True)", self.prott5, self.reaction_t5], "preflight_models")
        unimol_path = f"{ROOT}/.deps/unimol_tools:{ROOT.parent}/env/unimol2_site:{ROOT}"
        chiro_path = f"{ROOT}/.deps/python:{ROOT}/.deps/ChIRo:{ROOT}"
        self.command([self.python, "-c", "from unimol_tools import UniMolRepr"], "preflight_unimol", env={"PYTHONPATH": unimol_path})
        self.command([self.python, "-c", "from scripts.extract_chiro_reaction_embeddings import add_runtime_paths, install_chiro_compatibility; add_runtime_paths('.deps/ChIRo','.deps/python'); install_chiro_compatibility('cpu'); from model.alpha_encoder import Encoder; print(Encoder.__name__)"], "preflight_chiro", env={"PYTHONPATH": chiro_path})
        free = shutil.disk_usage(self.run).free
        if free < self.args.min_free_tb * 1e12:
            raise RuntimeError(f"Only {free / 1e12:.2f} TB free; require {self.args.min_free_tb:.2f} TB")
        work = self.work_directory()
        self.log(f"Profile={self.args.profile}; extraction batch={self.args.extraction_batch_size}, "
                 f"token budget={self.args.extraction_max_tokens}; training global pair rows={self.args.train_batch_size * len(self.gpus)}")
        atomic_json(self.state / "preflight.json", {"status": "complete", "free_tb": free / 1e12,
                    "profile": self.args.profile, "work_dir": str(work),
                    "required_inputs": [signature(path) for path in required], "time": time.time()})
        self.log(f"Preflight passed; {free / 1e12:.2f} TB free (shared, not reserved). GPUs checked at GPU-stage entry.")

    # Split and compact-index interfaces are intentionally isolated here.
    def prepare(self) -> None:
        self.require_stage("preflight")
        work = self.work_directory()
        self.step("prepare", [ROOT / "scripts/prepare_horizyn1_training.py",
                              ROOT / "horizyn/datasets/horizyn1_training.py",
                              self.state / "work_directory.json",
                              Path(self.mmseqs),
                              self.source / "clustered/proteins.fasta", self.source / "raw/raw_pairs.tsv",
                              self.source / "clustered/pairs.tsv",
                              self.source / "raw/raw_reactions.tsv"],
                  [self.data / "preparation_manifest.json"],
                  lambda: self.command([self.setup_python, ROOT / "scripts/prepare_horizyn1_training.py",
                      "--representative-fasta", self.source / "clustered/proteins.fasta",
                      "--raw-pairs", self.source / "raw/raw_pairs.tsv", "--reactions", self.source / "raw/raw_reactions.tsv",
                      "--clustered-pairs", self.source / "clustered/pairs.tsv",
                      "--output-dir", self.data, "--mmseqs", self.mmseqs,
                      "--work-dir", work, "--log-dir", self.logs / "preparation",
                      "--threads", self.args.preparation_threads,
                      "--sqlite-cache-mib", self.args.sqlite_cache_mib,
                      "--validation-fraction", ".05", "--test-fraction", ".05", "--min-seq-id", ".5",
                      "--coverage", ".8", "--reaction-similarity-threshold", ".8", "--resume",
                      "--seed", self.args.seed], "prepare"),
                  {"seed": self.args.seed, "validation_fraction": .05, "test_fraction": .05,
                   "min_seq_id": .5, "coverage": .8, "reaction_similarity_threshold": .8,
                   "work_dir": str(work), "profile": self.args.profile,
                   "threads": self.args.preparation_threads, "sqlite_cache_mib": self.args.sqlite_cache_mib})

    def labels(self) -> None:
        self.require_stage("prepare")
        self.step("labels", [self.data / "preparation_manifest.json", self.data / "train_own_raw_associations.csv",
                             self.source / "annotations/native_uniprot.tsv.gz",
                             self.source / "annotations/native_uniprot_manifest.json",
                             self.source / "annotations/cofactor_v2/reaction_annotations.json",
                             self.source / "annotations/cofactor_v2/reaction_annotations_manifest.json",
                             ROOT / "scripts/build_horizyn1_circe_v2_labels_v2.py"],
                  [self.data / "labels/label_manifest.json"],
                  lambda: self.rebuild_directory(self.data / "labels", lambda: self.command([self.annotation_python, ROOT / "scripts/build_horizyn1_circe_v2_labels_v2.py",
                      "--representative-fasta", self.source / "clustered/proteins.fasta",
                      "--cluster-map", self.source / "clustered/clusters.tsv",
                      "--native-annotations", self.source / "annotations/native_uniprot.tsv.gz",
                      "--native-manifest", self.source / "annotations/native_uniprot_manifest.json",
                      "--reaction-annotations", self.source / "annotations/cofactor_v2/reaction_annotations.json",
                      "--reaction-manifest", self.source / "annotations/cofactor_v2/reaction_annotations_manifest.json",
                      "--association-pairs", self.data / "train_own_raw_associations.csv", "--pair-scope", "train",
                      "--output-dir", self.data / "labels"], "labels")))

    def build_index(self, pairs: Path, output: Path, name: str) -> None:
        self.rebuild_directory(output, lambda: self.command([self.python, ROOT / "scripts/build_indexed_training_pairs.py",
                      "--train-pairs", pairs,
                      "--train-reactions", pairs.parent / ("reactions.csv" if pairs.parent.name == "pilot" else "train_rxns.csv"),
                      "--ec-labels", self.data / "labels/enzyme_ec_labels.csv",
                      "--biofp-targets", self.data / "labels/enzyme_biofp_targets.npz",
                      "--candidate-eligibility", self.data / "labels/candidate_eligibility.csv",
                      "--exclude-pairs", self.data / "train_transferred_uncertain_pairs.csv",
                      "--output-dir", output], name))

    def index(self) -> None:
        self.require_stage("labels")
        self.step("index", [self.data / "train_pairs.csv", self.data / "labels/label_manifest.json",
                            ROOT / "scripts/build_indexed_training_pairs.py", ROOT / "horizyn/datasets/indexed_pairs.py"],
                  [self.data / "index/manifest.json"],
                  lambda: self.build_index(self.data / "train_pairs.csv", self.data / "index", "index"))

    def select_pilot(self, pilot: Path) -> None:
        # Length-stratified reservoirs require one streaming FASTA pass, not a RAM copy.
        rng = random.Random(self.args.seed)
        bins: list[list[tuple[str, str]]] = [[] for _ in range(5)]
        seen = [0] * 5
        quota = max(1, self.args.pilot_proteins // 5)
        total, residues = 0, 0
        for protein_id, sequence in fasta_records(self.source / "clustered/proteins.fasta"):
            length = len(sequence)
            bucket = sum(length > bound for bound in (128, 256, 512, 1022))
            total += 1
            residues += min(length, 1022)
            seen[bucket] += 1
            record = protein_id, sequence
            if len(bins[bucket]) < quota:
                bins[bucket].append(record)
            else:
                position = rng.randrange(seen[bucket])
                if position < quota:
                    bins[bucket][position] = record
        selected = {protein_id: sequence for bucket in bins for protein_id, sequence in bucket}
        rows = [row for row in csv_rows(self.data / "train_pairs.csv") if row["protein_id"] in selected]
        if len(rows) < 200:
            raise ValueError("Pilot contains fewer than 200 training pairs; increase --pilot-proteins")
        used = {row["protein_id"] for row in rows}
        reaction_ids = {row["reaction_id"] for row in rows}
        pilot.mkdir(parents=True, exist_ok=True)
        with (pilot / "proteins.fasta").open("w") as handle:
            for protein_id, sequence in selected.items():
                if protein_id in used:
                    handle.write(f">{protein_id}\n{sequence}\n")
        write_csv(pilot / "train_pairs.csv", ["pr_id", "reaction_id", "protein_id"],
                  ({**row, "pr_id": str(index)} for index, row in enumerate(rows)))
        reaction_rows = [row for row in csv_rows(self.data / "reactions.csv") if row["reaction_id"] in reaction_ids]
        if len(reaction_rows) != len(reaction_ids):
            raise ValueError("Pilot training pairs have missing reaction definitions")
        write_csv(pilot / "reactions.csv", ["reaction_id", "reaction_smiles"], reaction_rows)
        atomic_json(pilot / "selection.json", {"proteins": len(used), "pairs": len(rows),
                    "reactions": len(reaction_ids), "length_bin_population": seen,
                    "pilot_truncated_residues": sum(min(len(selected[pid]), 1022) for pid in used),
                    "full_proteins": total, "full_truncated_residues": residues,
                    "fp16_residue_payload_bytes": residues * 1024 * 2})

    def protein_features(self, fasta: Path, output: Path, tag: str) -> None:
        common = [self.python, ROOT / "scripts/extract_prott5_residue_embeddings.py", "--fasta", fasta,
                  "--output", output, "--model-name", self.prott5, "--world-size", len(self.gpus),
                  "--max-sequence-length", "1022", "--sequence-truncation", "ends_center", "--dtype", "float16",
                  "--compression", "none", "--batch-size", self.args.extraction_batch_size,
                  "--max-tokens-per-batch", self.args.extraction_max_tokens,
                  "--progress-every", self.args.extraction_progress_every,
                  "--checkpoint-every", self.args.extraction_checkpoint_every,
                  "--merge-order", "shard", "--merge-storage", "virtual", "--resume"]
        if self.args.extraction_length_sort:
            common.append("--length-sort")
        if self.args.extraction_padded_token_budget:
            common.append("--padded-token-budget")
        def extract():
            if not output.exists():
                processes = []
                try:
                    for rank, gpu in enumerate(self.gpus):
                        log = (self.logs / f"{tag}_prott5_rank{rank}.log").open("a")
                        child = subprocess.Popen(list(map(str, [*common, "--rank", rank, "--device", "cuda"])),
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, **self.runtime_env, "CUDA_VISIBLE_DEVICES": gpu}, start_new_session=True)
                        self.children.append(child)
                        processes.append((child, log))
                    while any(child.poll() is None for child, _ in processes):
                        failed = [child for child, _ in processes if child.poll() not in (None, 0)]
                        if failed:
                            for child, _ in processes:
                                if child.poll() is None:
                                    os.killpg(child.pid, signal.SIGTERM)
                            raise RuntimeError(f"{tag} ProtT5 worker failed; remaining workers stopped")
                        time.sleep(1)
                    if any(child.returncode != 0 for child, _ in processes):
                        raise RuntimeError(f"{tag} ProtT5 worker failed")
                finally:
                    for child, log in processes:
                        if child.poll() is None:
                            os.killpg(child.pid, signal.SIGTERM)
                        child.wait()
                        log.close()
                self.command([*common, "--merge-only"], f"{tag}_prott5_merge")
        self.step(f"{tag}_prott5", [fasta, ROOT / "scripts/extract_prott5_residue_embeddings.py",
                                   *sorted(path for path in Path(self.prott5).iterdir() if path.is_file())], [output], extract,
                  {"model": self.prott5, "gpus": self.gpus, "max_length": 1022,
                   "batch_size": self.args.extraction_batch_size, "token_budget": self.args.extraction_max_tokens,
                   "length_sort": self.args.extraction_length_sort,
                   "padded_token_budget": self.args.extraction_padded_token_budget,
                   "checkpoint_every": self.args.extraction_checkpoint_every,
                   "progress_every": self.args.extraction_progress_every,
                   "cpu_threads": self.args.cpu_threads})

    def reaction_features(self, reactions: Path, output: Path, tag: str) -> None:
        output.mkdir(parents=True, exist_ok=True)
        commands = {
            "reactiont5v2": ([self.python, ROOT / "scripts/extract_reaction_t5v2_embeddings.py",
                "--model-name", self.reaction_t5, "--batch-size", "64", "--max-length", "512", "--pooling", "mean",
                "--dtype", "float16", "--device", "cuda"], {}),
            "unimol2": ([self.python, ROOT / "scripts/extract_unimol2_reaction_embeddings.py",
                "--batch-size", "64", "--dtype", "float16", "--compression", "none",
                "--skip-invalid-molecules", "--skip-invalid-reactions"],
                {"PYTHONPATH": f"{ROOT}/.deps/unimol_tools:{ROOT.parent}/env/unimol2_site:{ROOT}"}),
            "chiro": ([self.python, ROOT / "scripts/extract_chiro_reaction_embeddings.py", "--device", "cuda",
                "--batch-size", "64", "--num-workers", self.args.reaction_workers, "--dtype", "float16",
                "--molecule-cache", self.features / "chiro_molecules.sqlite"],
                {"PYTHONPATH": f"{ROOT}/.deps/python:{ROOT}/.deps/ChIRo:{ROOT}"}),
        }
        for modality, (command, env) in commands.items():
            target = output / f"{modality}.h5"
            temporary = target.with_suffix(".h5.partial")
            def extract(command=command, env=env, target=target, temporary=temporary, modality=modality):
                # Preserve interrupted files; never force-overwrite a foreign cache.
                if temporary.exists():
                    temporary.rename(temporary.with_name(temporary.name + f".interrupted.{time.time_ns()}"))
                self.command([*command, "--reactions", reactions, "--output", temporary,
                              "--no-bidirectional", "--no-allow-pseudo-reactions"], f"{tag}_{modality}",
                             env={**env, "CUDA_VISIBLE_DEVICES": self.gpus[0]})
                temporary.replace(target)
            weights = {"reactiont5v2": [path for path in self.reaction_t5.iterdir() if path.is_file()],
                       "unimol2": [Path(os.environ.get("UNIMOL_WEIGHT_DIR", str(ROOT.parent / "unimol_weights"))) / "modelzoo/84M/checkpoint.pt"],
                       "chiro": [ROOT / ".deps/ChIRo/paper_results/RS_experiment/ChIRo/results_RS_ChIRo_seed1/best_model.pt",
                                 ROOT / ".deps/ChIRo/paper_results/RS_experiment/ChIRo/params_RS_ChIRo.json"]}[modality]
            self.step(f"{tag}_{modality}", [reactions, Path(command[1]), *sorted(weights)], [target], extract,
                      {"command": list(map(str, command)), "direction": "forward_unsuffixed"})

    def chemistry(self, train: Path, reactions: Path, output: Path, tag: str) -> None:
        def build():
            self.command([self.setup_python, ROOT / "scripts/build_reaction_set_features.py",
                          "--train-reactions", train, "--validation-reactions", reactions,
                          "--test-reactions", reactions, "--cofactor-dictionary", self.cofactor_dictionary,
                          "--out-dir", output], f"{tag}_chemistry")
            # Validation file contains all requested records transformed by the TRAIN-fitted schema.
            shutil.copyfile(output / "validation_reaction_set_features.npz", output / "all_reaction_set_features.npz")
        self.step(f"{tag}_chemistry", [train, reactions, self.cofactor_dictionary],
                  [output / "schema.json", output / "all_reaction_set_features.npz"], build)

    def config(self, output: Path, *, pilot: Path | None = None) -> None:
        import yaml
        value = yaml.safe_load(self.template.read_text())
        def resolve(item):
            if isinstance(item, dict):
                return {key: resolve(val) for key, val in item.items()}
            if isinstance(item, list):
                return [resolve(val) for val in item]
            if isinstance(item, str):
                for prefix in ("runs/horizyn1_circe_v2_h200/", "runs/horizyn1_circe_v2/"):
                    if item.startswith(prefix):
                        return str(self.run / item.removeprefix(prefix))
            return item
        value = resolve(value)
        value["model"]["sleec_pooling"]["checkpoint_path"] = str(self.sleec.resolve())
        value["training"]["devices"] = len(self.gpus)
        value["data"]["train_batch_size"] = self.args.train_batch_size
        value["training"]["max_steps"] = self.args.max_steps
        value["training"]["max_epochs"] = self.args.max_epochs
        if self.args.cpu_threads is not None:
            value["training"]["cpu_num_threads"] = self.args.cpu_threads
        if self.args.loader_workers is not None:
            value["data"]["num_workers"] = self.args.loader_workers
        if value["data"]["num_workers"] == 0:
            value["data"]["persistent_workers"] = False
        value["seed"] = self.args.seed
        if pilot:
            data = value["data"]
            data["train_pairs_path"] = str(pilot / "train_pairs.csv")
            data["train_reactions_path"] = str(pilot / "reactions.csv")
            data["validation_pairs_path"] = str(pilot / "train_pairs.csv")
            data["validation_reactions_path"] = str(pilot / "reactions.csv")
            data["indexed_pairs_dir"] = str(pilot / "index")
            data["typed_negative_pools_path"] = None
            data["protein_residue_embeds_path"] = str(pilot / "features/proteins_prott5_residue.h5")
            data["validation_retrieval_candidate_ids_path"] = None
            data["validation_retrieval_query_ids_path"] = None
            data["validation_retrieval_candidate_set"] = "validation"
            if self.args.profile == "base":
                data["num_workers"] = 0
                data["persistent_workers"] = False
            for split in ("train", "validation"):
                for key, filename in (("t5v2_embeds", "reactiont5v2.h5"), ("unimol2_embeds", "unimol2.h5"),
                                      ("chiro_embeds", "chiro.h5"), ("chemistry_vectors", "chemistry/all_reaction_set_features.npz")):
                    data[f"{split}_reaction_{key}_path"] = str(pilot / "features" / filename)
            training = value["training"]
            training.update(max_steps=self.args.pilot_steps, max_epochs=100, validation_enabled=False, validation_retrieval_metrics=False,
                            validation_retrieval_candidate_set="validation", validation_retrieval_candidate_ids_path=None,
                            validation_interval_steps=None, limit_val_batches=0)
            training["early_stopping"]["enabled"] = False
            value["logging"].update(log_dir=str(pilot / "logs/train"), checkpoint_dir=str(pilot / "checkpoints"),
                checkpoint_monitor="train/loss", checkpoint_mode="min", checkpoint_on_validation_end=False,
                save_every_n_train_steps=self.args.pilot_steps, log_every_n_steps=1)
        output.parent.mkdir(parents=True, exist_ok=True)
        serialized = yaml.safe_dump(value, sort_keys=False)
        if output.exists() and output.read_text() != serialized:
            raise RuntimeError(f"Training configuration changed at {output}; use a new RUN_ROOT")
        if not output.exists():
            output.write_text(serialized)

    def pilot(self) -> None:
        self.require_stage("index")
        pilot = self.run / "pilot"
        self.step("pilot_selection", [self.source / "clustered/proteins.fasta", self.data / "train_pairs.csv",
                                      self.data / "reactions.csv"],
                  [pilot / "selection.json", pilot / "proteins.fasta", pilot / "train_pairs.csv", pilot / "reactions.csv"],
                  lambda: self.select_pilot(pilot), {"proteins": self.args.pilot_proteins, "seed": self.args.seed})
        self.require_free_gpus()
        start = time.time()
        self.protein_features(pilot / "proteins.fasta", pilot / "features/proteins_prott5_residue.h5", "pilot")
        self.reaction_features(pilot / "reactions.csv", pilot / "features", "pilot")
        self.chemistry(pilot / "reactions.csv", pilot / "reactions.csv", pilot / "features/chemistry", "pilot")
        expected_proteins = {protein_id for protein_id, _ in fasta_records(pilot / "proteins.fasta")}
        expected_reactions = {row["reaction_id"] for row in csv_rows(pilot / "reactions.csv")}
        report = {"selection": json.loads((pilot / "selection.json").read_text()), "features": {},
                  "profile": self.args.profile,
                  "extraction_settings": {"batch_size": self.args.extraction_batch_size,
                      "max_tokens_per_batch": self.args.extraction_max_tokens,
                      "length_sort": self.args.extraction_length_sort,
                      "padded_token_budget": self.args.extraction_padded_token_budget,
                      "checkpoint_every": self.args.extraction_checkpoint_every,
                      "cpu_threads": self.args.cpu_threads},
                  "training_global_pair_rows": self.args.train_batch_size * len(self.gpus)}
        for name, ids, dim, ragged, optional in (("proteins_prott5_residue", expected_proteins, 1024, False, False),
                ("reactiont5v2", expected_reactions, 768, False, False), ("unimol2", expected_reactions, 768, True, True),
                ("chiro", expected_reactions, 256, True, True)):
            report["features"][name] = validate_h5(pilot / "features" / f"{name}.h5", ids, dim,
                                                 ragged=ragged, optional=optional)
        import h5py
        import numpy as np
        with h5py.File(pilot / "features/proteins_prott5_residue.h5", "r") as handle:
            offsets = handle["offsets"][:]
            order = np.random.default_rng(self.args.seed).permutation(len(offsets) - 1)
            read_start = time.time()
            read_bytes = 0
            for index in order:
                values = handle["vectors"][offsets[index]:offsets[index + 1]]
                read_bytes += values.nbytes
            report["shuffled_read_MB_per_second"] = read_bytes / max(1e-6, time.time() - read_start) / 1e6
        extraction_state = json.loads((self.state / "pilot_prott5.json").read_text())
        report["projected_prott5_seconds"] = extraction_state["seconds"] * report["selection"]["full_truncated_residues"] / report["selection"]["pilot_truncated_residues"]
        report["projection_note"] = "Length-stratified pilot extrapolation, not a runtime guarantee; excludes reaction extraction and training."
        self.step("pilot_index", [pilot / "train_pairs.csv", self.data / "labels/label_manifest.json",
                                  ROOT / "scripts/build_indexed_training_pairs.py", ROOT / "horizyn/datasets/indexed_pairs.py"],
                  [pilot / "index/manifest.json"], lambda: self.build_index(pilot / "train_pairs.csv", pilot / "index", "pilot_index"))
        self.config(pilot / "train.yaml", pilot=pilot)
        self.step("pilot_training", [pilot / "train.yaml", pilot / "index/manifest.json", self.sleec,
                                     ROOT / "horizyn/model.py", ROOT / "horizyn/protein_pooling_lightning_module.py",
                                     ROOT / "horizyn/reaction_conditioned_data_module.py", ROOT / "horizyn/losses.py",
                                     ROOT / "horizyn/config.py", ROOT / "horizyn/reaction_features.py",
                                     ROOT / "horizyn/utils/collate.py", ROOT / "horizyn/datasets/residue_hdf5.py",
                                     ROOT / "scripts/train_protein_pooling.py"],
                  [pilot / "training_passed.json"], lambda: self._pilot_train(pilot))
        training_seconds = json.loads((self.state / "pilot_training.json").read_text())["seconds"]
        train_pairs = json.loads((self.data / "index/manifest.json").read_text())["num_pairs"]
        projected_epoch_steps = indexed_epoch_steps(train_pairs, self.args.train_batch_size, len(self.gpus))
        report["training_steps_per_epoch"] = projected_epoch_steps
        report["sampler_rows_per_rank_batch"] = {
            "base_positive_sweep": self.args.train_batch_size * 11 // 20,
            "endpoint_support_positive": self.args.train_batch_size * 6 // 20,
            "explicit_negative": self.args.train_batch_size * 3 // 20,
        }
        report["pilot_training_steps_per_second_including_startup"] = self.args.pilot_steps / training_seconds
        report["projected_training_seconds_per_epoch_excluding_validation"] = projected_epoch_steps * training_seconds / self.args.pilot_steps
        report["training_projection_note"] = "Short pilot includes startup/checkpoint checks and warm-cache I/O; full shuffled shared-storage I/O and periodic validation can differ substantially."
        report["status"] = "passed"
        report["features"]["chemistry"] = validate_chemistry(pilot / "features/chemistry/all_reaction_set_features.npz", expected_reactions)
        report["elapsed_this_invocation_seconds"] = time.time() - start
        atomic_json(pilot / "report.json", report)
        atomic_json(self.state / "pilot.json", {"status": "complete", "report": signature(pilot / "report.json")})
        self.log(f"Pilot passed; projected protein payload {report['selection']['fp16_residue_payload_bytes'] / 1e12:.2f} TB")

    def _pilot_train(self, pilot: Path) -> None:
        self.command([self.python, ROOT / "scripts/train_protein_pooling.py", "--config", pilot / "train.yaml"], "pilot_training")
        checkpoint = pilot / "checkpoints/last.ckpt"
        code = "import sys,torch; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); assert c.get('global_step',0)>=int(sys.argv[2]), 'Pilot stopped before requested steps'; bad=[k for k,v in c['state_dict'].items() if torch.is_floating_point(v) and not torch.isfinite(v).all()]; assert not bad, 'Nonfinite pilot parameters: '+str(bad); print('Pilot checkpoint finite at global_step',c['global_step'])"
        self.command([self.python, "-c", code, checkpoint, self.args.pilot_steps], "pilot_checkpoint_audit")
        atomic_json(pilot / "training_passed.json", {"status": "complete", "max_steps": self.args.pilot_steps,
                    "devices": len(self.gpus), "bidirectional_active_anchor_guard": True,
                    "checkpoint": signature(checkpoint)})

    def features_stage(self) -> None:
        self.require_stage("pilot")
        pilot = self.run / "pilot"
        report = json.loads((pilot / "report.json").read_text())
        required = report["selection"]["fp16_residue_payload_bytes"] * 1.25
        existing_shards = sum(path.stat().st_size for path in self.features.glob("proteins_prott5_residue_shards/*.h5*"))
        if shutil.disk_usage(self.run).free + existing_shards < required:
            raise RuntimeError(f"Insufficient space for feature payload plus 25% headroom: {required / 1e12:.2f} TB")
        self.require_free_gpus()
        self.protein_features(self.source / "clustered/proteins.fasta", self.features / "proteins_prott5_residue.h5", "full")
        self.reaction_features(self.data / "reactions.csv", self.features, "full")
        self.chemistry(self.data / "train_rxns.csv", self.data / "reactions.csv", self.features / "chemistry", "full")
        ids = {row["reaction_id"] for row in csv_rows(self.data / "reactions.csv")}
        checks = {}
        for name, dim, ragged, optional in (("reactiont5v2", 768, False, False), ("unimol2", 768, True, True), ("chiro", 256, True, True)):
            checks[name] = validate_h5(self.features / f"{name}.h5", ids, dim, ragged=ragged, optional=optional)
        protein_ids = {protein_id for protein_id, _ in fasta_records(self.source / "clustered/proteins.fasta")}
        checks["proteins"] = validate_h5(self.features / "proteins_prott5_residue.h5", protein_ids, 1024, sample_only=True)
        checks["chemistry"] = validate_chemistry(self.features / "chemistry/all_reaction_set_features.npz", ids)
        self.config(self.run / "configs/train.yaml")
        atomic_json(self.state / "features.json", {"status": "complete", "checks": checks,
                    "config": signature(self.run / "configs/train.yaml"),
                    "artifacts": [signature(self.sleec), signature(self.template)],
                    "virtual_shards": [signature(path) for path in sorted(self.features.glob("proteins_prott5_residue_shards/*.h5"))]})

    def train(self) -> None:
        self.require_stage("features")
        self.config(self.run / "configs/train.yaml")
        marker = self.state / "train.json"
        if marker.exists() and json.loads(marker.read_text()).get("status") == "complete":
            self.log("Training already completed; skipping")
            return
        self.require_free_gpus()
        command = [self.python, ROOT / "scripts/train_protein_pooling.py", "--config", self.run / "configs/train.yaml"]
        last = self.run / "checkpoints/last.ckpt"
        if self.args.resume:
            command.extend(["--resume", self.args.resume])
        elif last.exists():
            command.extend(["--resume", last])
        self.command(command, "train")
        # Lightning's last checkpoint records the validation-selected checkpoint.
        code = "import json,sys,torch; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); print(json.dumps([v.get('best_model_path') for v in c.get('callbacks',{}).values() if isinstance(v,dict) and v.get('best_model_path')]))"
        choices = json.loads(subprocess.check_output([self.python, "-c", code, str(last)], cwd=ROOT, text=True))
        choices = [Path(choice) if Path(choice).is_absolute() else ROOT / choice for choice in choices]
        choices = [choice.resolve() for choice in choices if choice.is_file() and choice.resolve().is_relative_to(self.run)]
        if len(set(choices)) != 1:
            raise RuntimeError("Training exited but no unique validation-selected checkpoint was found in last.ckpt")
        atomic_json(marker, {"status": "complete", "time": time.time(), "last": signature(last),
                             "best_checkpoint": signature(choices[0])})

    def test(self) -> None:
        self.require_stage("train")
        self.require_free_gpus()
        selected = self.args.test_checkpoint or json.loads((self.state / "train.json").read_text())["best_checkpoint"]["path"]
        self.command([self.python, ROOT / "scripts/evaluate_horizyn1_circe_v2.py",
            "--checkpoint", selected,
            "--config", self.run / "configs/train.yaml", "--pairs", self.data / "test_query_gold.csv",
            "--reactions", self.data / "reactions.csv", "--protein-candidates", self.data / "all_candidate_ids.txt",
            "--reaction-candidates", self.data / "all_reaction_ids.txt", "--output", self.run / "results/test.json",
            "--enzyme-query-ids", self.data / "test_enzyme_query_ids.txt",
            "--reaction-query-ids", self.data / "test_reaction_query_ids.txt",
            "--embedding-cache", self.run / "results/embedding_cache.pt",
            "--batch-size", "128", "--candidate-chunk-size", "32768", "--device", "cuda"], "test",
            env={"CUDA_VISIBLE_DEVICES": self.gpus[0]})
        atomic_json(self.state / "test.json", {"status": "complete", "time": time.time()})


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--profile", choices=("base", "h200"), default=os.environ.get("CIRCE_PROFILE", "base"))
    parser.add_argument("--run-root", default=os.environ.get("RUN_ROOT"))
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", str(ROOT / "data/reconstructed/horizyn1_2023_05")))
    parser.add_argument("--scratch-root", type=Path, default=os.environ.get("SCRATCH_ROOT"),
                        help="Optional allocated LOCAL disk root; default work stays on shared storage. No automatic SSD requirement.")
    parser.add_argument("--min-scratch-gb", type=float, default=100.0,
                        help="Free-space gate only when --scratch-root is explicitly selected")
    parser.add_argument("--preparation-threads", type=int, default=64)
    parser.add_argument("--sqlite-cache-mib", type=int, default=512)
    parser.add_argument("--cpu-threads", type=int, default=None, help="Per-process Torch/BLAS threads; H200 default4")
    parser.add_argument("--loader-workers", type=int, default=None, help="Override training DataLoader workers per rank")
    parser.add_argument("--reaction-workers", type=int, default=None, help="ChIRo molecule preprocessing workers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot-proteins", type=int, default=5000)
    parser.add_argument("--pilot-steps", type=int, default=20)
    parser.add_argument("--extraction-batch-size", type=int, default=None)
    parser.add_argument("--extraction-max-tokens", "--extraction-token-budget", type=int, default=None)
    parser.add_argument("--extraction-progress-every", type=int, default=None)
    parser.add_argument("--extraction-checkpoint-every", type=int, default=None)
    parser.add_argument("--extraction-length-sort", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--extraction-padded-token-budget", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--train-batch-size", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--min-free-tb", type=float, default=6.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--test-checkpoint", type=Path)
    parser.add_argument("--allow-busy-gpus", action="store_true", help="Explicitly permit sharing selected GPUs")
    args = parser.parse_args(argv)
    if args.run_root is None:
        args.run_root = str(ROOT / "runs" / ("horizyn1_circe_v2_h200" if args.profile == "h200" else "horizyn1_circe_v2"))
    h200 = args.profile == "h200"
    defaults = {"extraction_batch_size": 32 if h200 else 8, "extraction_max_tokens": 32768 if h200 else 4096,
                "extraction_progress_every": 1000 if h200 else 100, "extraction_checkpoint_every": 1000 if h200 else 100,
                "extraction_length_sort": h200, "extraction_padded_token_budget": h200,
                "reaction_workers": 8 if h200 else 4}
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    if args.cpu_threads is None and h200:
        args.cpu_threads = 4
    if args.train_batch_size < 20 or args.train_batch_size % 20:
        parser.error("--train-batch-size must be a positive multiple of 20 for an exact 85/15 ratio")
    if min(args.pilot_proteins, args.pilot_steps, args.extraction_batch_size, args.max_steps, args.max_epochs,
           args.extraction_max_tokens, args.extraction_progress_every, args.extraction_checkpoint_every,
           args.preparation_threads, args.sqlite_cache_mib) <= 0:
        parser.error("Counts and step budgets must be positive")
    if ((args.cpu_threads is not None and args.cpu_threads <= 0)
            or (args.loader_workers is not None and args.loader_workers < 0) or args.reaction_workers < 0):
        parser.error("CPU threads must be positive; loader/reaction workers must be nonnegative")
    if args.scratch_root is not None and (not args.scratch_root.is_absolute() or args.min_scratch_gb <= 0):
        parser.error("Explicit scratch root must be absolute and its free-space gate positive")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    pipeline = Pipeline(args)
    with (pipeline.run / "pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another pipeline holds the RUN_ROOT lock; no duplicate run was launched")
        (pipeline.run / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, pipeline.terminate)
        stages = STAGES if args.stage == "all" else [args.stage]
        for stage in stages:
            (pipeline.run / "current_stage.txt").write_text(stage + "\n")
            getattr(pipeline, "features_stage" if stage == "features" else stage)()
        pipeline.log(f"Requested stage '{args.stage}' completed")


if __name__ == "__main__":
    main()
