#!/usr/bin/env python3
"""Extract reaction features on a second node, then safely resume the main run.

Existing controllers/extractors are reused unchanged. External workers never
write into the source run. The resume controller takes the source pipeline lock,
resumes protein extraction, and imports validated reaction caches instead of
recomputing them. All normal stage fingerprints and training gates still apply.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import horizyn1_circe_v2_pipeline as base
from scripts import horizyn1_circe_v2_reaction_holdout as holdout

MODALITIES = ("reactiont5v2", "unimol2", "chiro", "chemistry")
STAGES = {f"full_{name}" for name in MODALITIES}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def sync_file(path):
    # Finish NFS writeback before recording cross-node ownership signatures.
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def durable_json(path, value):
    base.atomic_json(path, value)
    sync_file(path)


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process holds {path}") from None
        yield


def expected_binding(args):
    source = Path(args.run_root).resolve()
    cache = args.cache_dir.resolve()
    if cache == source or source in cache.parents or cache in source.parents:
        raise ValueError("Cache and main run must be separate, non-nested directories")
    protocol = source / "state/dataset_protocol.json"
    value = json.loads(protocol.read_text())
    expected = {"dataset_protocol": holdout.PROTOCOL, "seed": args.seed,
                "validation_fraction": args.validation_fraction,
                "test_fraction": args.test_fraction}
    if value != expected:
        raise ValueError("Source reaction-holdout protocol/options do not match")
    return {"format": 1, "source_run": str(source), "cache_dir": str(cache),
            "protocol": base.signature(protocol),
            "implementations": {str(path): digest(path) for path in
                                [Path(__file__).resolve(), Path(base.__file__).resolve(),
                                 *holdout.IMPLEMENTATIONS,
                                 ROOT / "scripts/build_reaction_set_features.py",
                                 ROOT / "horizyn/capability/reaction_set_features.py"]}}


def bind_cache(args, *, create=False):
    expected = expected_binding(args)
    marker = args.cache_dir / "binding.json"
    if not marker.exists():
        if not create:
            raise RuntimeError(f"External cache has not been initialized: {marker}")
        if args.cache_dir.exists() and any(args.cache_dir.iterdir()):
            raise RuntimeError("Refusing to claim a nonempty unowned cache directory")
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        durable_json(marker, expected)
    if json.loads(marker.read_text()) != expected:
        raise RuntimeError("External cache binding/code/source changed; use a new cache directory")


def cache_runner(args, cls=holdout.Pipeline):
    cache_args = copy.copy(args)
    cache_args.run_root = str(args.cache_dir)
    return cls(cache_args)


def invoke_features(runner, source):
    runner.reaction_features(source / "data/reactions.csv", runner.features, "full")
    runner.chemistry(source / "data/train_rxns.csv", source / "data/reactions.csv",
                     runner.features / "chemistry", "full")


class PlanPipeline(holdout.Pipeline):
    def step(self, name, inputs, outputs, action, parameters=None):
        self.plans[name] = {
            "fingerprint": {
                "inputs": [base.signature(path) for path in [*inputs, *holdout.IMPLEMENTATIONS]],
                "parameters": {**(parameters or {}), "dataset_protocol": holdout.PROTOCOL,
                               "validation_fraction": self.args.validation_fraction,
                               "test_fraction": self.args.test_fraction},
                "controller_sha256": digest(Path(base.__file__)),
            },
            "outputs": [str(path) for path in outputs],
        }


def plans(args):
    runner = cache_runner(args, PlanPipeline)
    runner.plans = {}
    invoke_features(runner, Path(args.run_root).resolve())
    return runner.plans


def validate_output(name, paths, source):
    ids = {row["reaction_id"] for row in base.csv_rows(source / "data/reactions.csv")}
    if name == "full_chemistry":
        # Both the train-fitted schema and all-catalog values must be present.
        json.loads(paths[0].read_text())
        return base.validate_chemistry(paths[1], ids)
    modality = name.removeprefix("full_")
    return base.validate_h5(paths[0], ids, 256 if modality == "chiro" else 768,
                            ragged=modality != "reactiont5v2",
                            optional=modality != "reactiont5v2")


class WorkerPipeline(holdout.Pipeline):
    def step(self, name, inputs, outputs, action, parameters=None):
        if name != f"full_{self.args.modality}":
            return
        super().step(name, inputs, outputs, action, parameters)


def worker(args):
    bind_cache(args)
    source = Path(args.run_root).resolve()
    runner = cache_runner(args, WorkerPipeline)
    name = f"full_{args.modality}"
    with exclusive(args.cache_dir / f"{name}.lock"):
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, runner.terminate)
        try:
            if args.modality != "chemistry":
                runner.require_free_gpus()
            invoke_features(runner, source)
            record = json.loads((runner.state / f"{name}.json").read_text())
            runner.validate_records(record["fingerprint"]["inputs"], name)
            outputs = [Path(item["path"]) for item in record["outputs"]]
            checks = validate_output(name, outputs, source)
            bind_cache(args)
            for path in outputs:
                sync_file(path)
            record["outputs"] = [base.signature(path) for path in outputs]
            durable_json(runner.state / f"{name}.json", record)
            durable_json(runner.state / f"{name}.validated.json", {
                "status": "complete", "stage": base.signature(runner.state / f"{name}.json"),
                "outputs": [{"signature": base.signature(path), "sha256": digest(path)}
                            for path in outputs], "checks": checks,
                "host": socket.gethostname(), "finished": time.time(),
            })
            runner.log(f"{name}: cache validated and ready for the main run")
        except BaseException as exc:
            durable_json(runner.state / f"{name}.worker_error.json",
                             {"error": str(exc), "time": time.time()})
            raise


def preflight(args):
    """Validate prepared data/index and extractor imports before dispatch."""
    source = holdout.Pipeline(args)
    # Frozen reaction extraction does not depend on a pilot timing report.
    source.require_stage("index")
    source.require_free_gpus()
    probes = [
        (source.python, "import torch,transformers,sentencepiece,h5py; print(torch.__version__)", str(ROOT)),
        (source.python, "from unimol_tools import UniMolRepr; print('Uni-Mol2 import OK')",
         f"{ROOT}/.deps/unimol_tools:{ROOT.parent}/env/unimol2_site:{ROOT}"),
        (source.python, "from scripts.extract_chiro_reaction_embeddings import add_runtime_paths, install_chiro_compatibility; add_runtime_paths('.deps/ChIRo','.deps/python'); install_chiro_compatibility('cpu'); from model.alpha_encoder import Encoder; print('ChIRo import OK')",
         f"{ROOT}/.deps/python:{ROOT}/.deps/ChIRo:{ROOT}"),
        (source.setup_python, "import rdkit,pandas,numpy; print('Chemistry import OK')", str(ROOT)),
    ]
    for python, code, path in probes:
        subprocess.run([python, "-c", code], cwd=ROOT, check=True,
                       env={**os.environ, **source.runtime_env, "PYTHONPATH": path})
    if shutil.disk_usage(args.cache_dir).free < 20 * 10**9:
        raise RuntimeError("External reaction cache needs at least 20 GB free")


def extract(args, forwarded):
    bind_cache(args, create=True)
    with exclusive(args.cache_dir / "extract.lock"):
        runner = cache_runner(args)
        if len(runner.gpus) != 3:
            raise ValueError("Select exactly three free GPUs for external extraction")
        preflight(args)
        expected_plans = plans(args)
        durable_json(args.cache_dir / "launch.json", {
            "host": socket.gethostname(), "pid": os.getpid(), "started": time.time(),
            "gpu_assignment": dict(zip(MODALITIES[:3], runner.gpus)),
            "source_run": str(Path(args.run_root).resolve()), "plans": expected_plans,
        })
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, runner.terminate)
        children = []
        try:
            for index, modality in enumerate(MODALITIES):
                log = (runner.logs / f"{modality}_worker.log").open("a")
                gpu = runner.gpus[index] if index < 3 else runner.gpus[0]
                env = {**os.environ, **runner.runtime_env,
                       "CUDA_VISIBLE_DEVICES": gpu, "GPU_COUNT": "1"}
                command = [sys.executable, str(Path(__file__).resolve()), "worker",
                           "--cache-dir", str(args.cache_dir), "--modality", modality, *forwarded]
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                runner.children.append(child)
                children.append((modality, child, log))
                runner.log(f"Launched {modality} PID={child.pid} on "
                           f"{'GPU ' + gpu if index < 3 else 'CPU'}")
            # Independent modalities continue even if one fails; completed caches
            # remain usable. Only explicitly owned descendants are terminated.
            results = {name: child.wait() for name, child, _ in children}
            durable_json(args.cache_dir / "result.json", {
                "status": "complete" if all(code == 0 for code in results.values()) else "failed",
                "exit_codes": results, "finished": time.time(),
            })
            if any(results.values()):
                raise RuntimeError(f"External extraction failed: {results}")
            runner.log("All four external reaction/chemistry caches are ready")
        finally:
            for _, child, log in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                child.wait()
                log.close()


def verified_cache(args, name, plan):
    bind_cache(args)
    state = args.cache_dir / "state"
    record = json.loads((state / f"{name}.json").read_text())
    audit = json.loads((state / f"{name}.validated.json").read_text())
    if record.get("status") != "complete" or record["fingerprint"] != plan["fingerprint"]:
        raise RuntimeError(f"External {name} configuration or input fingerprint mismatch")
    if audit.get("status") != "complete" or audit["stage"] != base.signature(state / f"{name}.json"):
        raise RuntimeError(f"External {name} validation is stale")
    if [item["path"] for item in record["outputs"]] != plan["outputs"]:
        raise RuntimeError(f"External {name} output paths mismatch")
    if [item["signature"] for item in audit["outputs"]] != record["outputs"]:
        raise RuntimeError(f"External {name} validation outputs mismatch")
    for item in [*record["fingerprint"]["inputs"], *record["outputs"]]:
        if base.signature(Path(item["path"])) != item:
            raise RuntimeError(f"External {name} artifact changed: {item['path']}")
    for item in audit["outputs"]:
        if digest(Path(item["signature"]["path"])) != item["sha256"]:
            raise RuntimeError(f"External {name} content hash mismatch")
    return audit


def publish_files(outputs, audit, journal):
    """Copy atomically without overwriting; a journal makes retries recoverable."""
    if len(outputs) != len(audit["outputs"]):
        raise RuntimeError("Import output count mismatch")
    planned = {"importer_sha256": digest(Path(__file__)),
               "files": [{"source": entry, "destination": str(destination)}
                         for destination, entry in zip(outputs, audit["outputs"])]}
    if journal.exists():
        if json.loads(journal.read_text()) != planned:
            raise RuntimeError("Import journal no longer matches the validated cache")
    else:
        if any(path.exists() for path in outputs):
            raise RuntimeError("Refusing to overwrite unowned main-run feature outputs")
        durable_json(journal, planned)
    for destination, entry in zip(outputs, audit["outputs"]):
        if destination.exists():
            if digest(destination) != entry["sha256"]:
                raise RuntimeError(f"Interrupted import output changed: {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".reaction-import-", dir=destination.parent)
        os.close(fd)
        temporary = Path(temporary)
        try:
            shutil.copyfile(entry["signature"]["path"], temporary)
            sync_file(temporary)
            if digest(temporary) != entry["sha256"]:
                raise RuntimeError("Imported feature content hash mismatch")
            os.link(temporary, destination)  # Atomic no-clobber publication on shared storage.
            sync_file(destination)
        finally:
            temporary.unlink(missing_ok=True)


class ImportPipeline(holdout.Pipeline):
    def inventory(self, name):
        records = super().inventory(name)
        journal = self.state / "external_imports" / f"{name}.json"
        if name in STAGES and journal.exists():
            records.append(base.signature(journal))
        return records

    def step(self, name, inputs, outputs, action, parameters=None):
        if name in STAGES:
            action = lambda: self.import_feature(name, outputs)
        # Same step/input/options signatures as the original sequential run.
        return super().step(name, inputs, outputs, action, parameters)

    def import_feature(self, name, outputs):
        plan = plans(self.args)[name]
        deadline = time.monotonic() + self.args.wait_hours * 3600
        state = self.args.cache_dir / "state"
        next_report = 0
        while True:
            ready = state / f"{name}.validated.json"
            if ready.exists():
                try:
                    with exclusive(self.args.cache_dir / f"{name}.lock"):
                        audit = verified_cache(self.args, name, plan)
                        validate_output(name, [Path(p) for p in plan["outputs"]], self.run)
                        publish_files(outputs, audit, self.state / "external_imports" / f"{name}.json")
                        self.log(f"{name}: imported validated external cache; GPU extraction skipped")
                        return
                except RuntimeError as exc:
                    if not str(exc).startswith("Another process holds "):
                        raise
            # A completed validated artifact takes precedence over an old error
            # from a failed attempt; a live retry is allowed to finish.
            error = state / f"{name}.worker_error.json"
            if error.exists() and not ready.exists():
                try:
                    with exclusive(self.args.cache_dir / f"{name}.lock"):
                        raise ValueError(f"External {name} failed; inspect {error}")
                except RuntimeError as exc:
                    if not str(exc).startswith("Another process holds "):
                        raise
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for external {name}; no fallback/recomputation launched")
            if time.monotonic() >= next_report:
                self.log(f"Waiting for validated external {name} in {self.args.cache_dir}")
                next_report = time.monotonic() + 60
            time.sleep(10)


def audit_pilot_report(pipeline):
    """Revalidate a reporting-only file after timestamp drift, never model data.

There was no historic content hash in the original receipt. We do not claim
historical byte equality: all meaningful fields are recomputed from independently
validated stage artifacts, while timing-only diagnostics must be finite/positive.
"""
    marker = pipeline.state / "pilot.json"
    saved = json.loads(marker.read_text())
    path = pipeline.run / "pilot/report.json"
    actual = base.signature(path)
    expected = saved.get("report", {})
    if saved.get("status") != "complete":
        raise RuntimeError("Cannot repair an incomplete pilot")
    if actual == expected:
        return None
    changed = {key for key in actual.keys() | expected.keys() if actual.get(key) != expected.get(key)}
    if not changed <= {"mtime_ns", "ctime_ns"}:
        raise RuntimeError("Pilot report change is not limited to timestamps; refusing repair")
    before_hash = digest(path)
    report = json.loads(path.read_text())
    for stage in ("index", "pilot_selection", "pilot_prott5", "pilot_reactiont5v2", "pilot_unimol2",
                  "pilot_chiro", "pilot_chemistry", "pilot_index", "pilot_training"):
        pipeline.require_stage(stage)
    pilot = pipeline.run / "pilot"
    selection = json.loads((pilot / "selection.json").read_text())
    args = pipeline.args
    settings = {"batch_size": args.extraction_batch_size,
                "max_tokens_per_batch": args.extraction_max_tokens,
                "length_sort": args.extraction_length_sort,
                "padded_token_budget": args.extraction_padded_token_budget,
                "checkpoint_every": args.extraction_checkpoint_every,
                "cpu_threads": args.cpu_threads}
    pair_count = json.loads((pipeline.data / "index/manifest.json").read_text())["num_pairs"]
    steps = base.indexed_epoch_steps(pair_count, args.train_batch_size, len(pipeline.gpus))
    fixed = {"status": "passed", "profile": args.profile, "selection": selection,
             "extraction_settings": settings,
             "training_global_pair_rows": args.train_batch_size * len(pipeline.gpus),
             "training_steps_per_epoch": steps,
             "sampler_rows_per_rank_batch": {"base_positive_sweep": args.train_batch_size * 11 // 20,
                 "endpoint_support_positive": args.train_batch_size * 6 // 20,
                 "explicit_negative": args.train_batch_size * 3 // 20}}
    for key, value in fixed.items():
        if report.get(key) != value:
            raise RuntimeError(f"Pilot report semantic mismatch: {key}")
    extract_seconds = json.loads((pipeline.state / "pilot_prott5.json").read_text())["seconds"]
    train_seconds = json.loads((pipeline.state / "pilot_training.json").read_text())["seconds"]
    projections = {"projected_prott5_seconds": extract_seconds * selection["full_truncated_residues"] / selection["pilot_truncated_residues"],
                   "pilot_training_steps_per_second_including_startup": args.pilot_steps / train_seconds,
                   "projected_training_seconds_per_epoch_excluding_validation": steps * train_seconds / args.pilot_steps}
    for key, value in projections.items():
        if not math.isclose(report.get(key, float("nan")), value, rel_tol=1e-12):
            raise RuntimeError(f"Pilot report projection mismatch: {key}")
    for key in ("elapsed_this_invocation_seconds", "shuffled_read_MB_per_second"):
        value = report.get(key, float("nan"))
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"Invalid pilot timing diagnostic: {key}")
    proteins = {pid for pid, _ in base.fasta_records(pilot / "proteins.fasta")}
    reactions = {row["reaction_id"] for row in base.csv_rows(pilot / "reactions.csv")}
    checks = {}
    for name, dim, ragged, optional in (("proteins_prott5_residue", 1024, False, False),
            ("reactiont5v2", 768, False, False), ("unimol2", 768, True, True), ("chiro", 256, True, True)):
        checks[name] = base.validate_h5(pilot / "features" / f"{name}.h5",
                                      proteins if name.startswith("proteins_") else reactions,
                                      dim, ragged=ragged, optional=optional)
    checks["chemistry"] = base.validate_chemistry(pilot / "features/chemistry/all_reaction_set_features.npz", reactions)
    if report.get("features") != checks:
        raise RuntimeError("Pilot report feature audit disagrees with actual cached features")
    if base.signature(path) != actual or digest(path) != before_hash:
        raise RuntimeError("Pilot report changed during audit")
    return {"old_marker": saved, "new_signature": actual, "report_sha256": before_hash,
            "checks": checks, "note": "Timestamp-only drift; current semantic contents independently revalidated. Historic byte equality is not asserted."}


def refresh_pilot_report(pipeline):
    audit = audit_pilot_report(pipeline)
    if audit is None:
        return
    directory = pipeline.run / "metadata_repairs"
    directory.mkdir(exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix="pilot_report_timestamps_", dir=directory))
    marker = pipeline.state / "pilot.json"
    shutil.copyfile(marker, backup / "pilot.json.before")
    shutil.copyfile(pipeline.run / "pilot/report.json", backup / "report.json")
    sync_file(backup / "pilot.json.before")
    sync_file(backup / "report.json")
    durable_json(backup / "audit.json", audit)
    value = dict(audit["old_marker"])
    value["report"] = audit["new_signature"]
    durable_json(marker, value)
    pipeline.log(f"Revalidated pilot reporting-only timestamp drift; original receipt preserved in {backup}")


def resume(args):
    bind_cache(args)
    pipeline = ImportPipeline(args)
    with exclusive(pipeline.run / "pipeline.lock"):
        # No protocol rewrites and no changes to the old controller implementation.
        expected_binding(args)
        refresh_pilot_report(pipeline)
        (pipeline.run / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, pipeline.terminate)
        for stage in base.STAGES:
            (pipeline.run / "current_stage.txt").write_text(stage + "\n")
            getattr(pipeline, "features_stage" if stage == "features" else stage)()
        pipeline.log("Parallel-cache pipeline completed training and test")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("command", choices=("extract", "worker", "resume"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--modality", choices=MODALITIES)
    parser.add_argument("--wait-hours", type=float, default=24)
    specific, forwarded = parser.parse_known_args(argv)
    args = holdout.parse_args(["all", *forwarded])
    for name, value in vars(specific).items():
        setattr(args, name, value)
    args.cache_dir = args.cache_dir.resolve()
    if args.wait_hours <= 0 or (args.command == "worker" and args.modality is None):
        parser.error("Positive --wait-hours and a --modality for worker are required")
    return args, forwarded


def main(argv=None):
    args, forwarded = parse_args(argv)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("UNIMOL_WEIGHT_DIR", str(ROOT.parent / "unimol_weights"))
    if args.command == "extract":
        extract(args, forwarded)
    elif args.command == "worker":
        worker(args)
    else:
        resume(args)


if __name__ == "__main__":
    main()
