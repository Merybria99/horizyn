#!/usr/bin/env python3
"""Verify a permission-only CIRCE-v2 metadata repair; apply only to stopped work.

Never changes the pipeline implementation, data, SQLite files, or file modes.
Only the reconstruction audit and preparation controller receipt are published.
Normal preflight must run again after apply; un-hashed model provenance is not
silently rewritten. A fresh graph audit establishes new evidence where the old
audit had no byte hashes; it is not a claim of historical byte identity there.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "horizyn1_metadata_repair_v1"
INPUT_NAMES = {"builder", "implementation", "mmseqs", "representative_fasta",
               "raw_pairs", "reactions", "clustered_pairs"}
ARTIFACTS = ("raw/raw_manifest.json", "raw/raw_proteins.fasta", "raw/raw_reactions.tsv",
             "raw/raw_pairs.tsv", "clustered/clustered_manifest.json", "clustered/proteins.fasta",
             "clustered/clusters.tsv", "clustered/pairs.tsv")
AUDIT_CHECKS = {"cluster_membership", "clustered_manifest", "clustered_pairs", "clustered_proteins",
                "collapsed_edge_union", "collapsed_source_union", "completed_inputs", "input_stability",
                "raw_manifest", "raw_pairs", "raw_proteins", "raw_reactions", "representative_sequences"}


def progress(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def signature(path):
    path = Path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Not a regular file: {path}")
    return {"path": str(path.resolve()), "inode": info.st_ino, "size": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}


def digest(path):
    before = signature(path)
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    if signature(path) != before:
        raise ValueError(f"File changed while hashing: {path}")
    return {**before, "sha256": value.hexdigest()}


def read_json(path):
    return json.loads(Path(path).read_text())


def payload(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_bytes(path, contents):
    """Same-directory atomic replacement, preserving existing permissions."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Refusing to replace a symlink: {path}")
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(contents)
        os.fchmod(handle.fileno(), mode)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def only_ctime_changed(before, after):
    if set(before) != set(after) or any(before[k] != after[k] for k in before if k != "ctime_ns"):
        raise ValueError(f"Change is not ctime-only: {before.get('path')}")


def audit_signatures_current(report):
    for path, expected in report["artifact_signatures"].items():
        info = Path(path).stat()
        # Device IDs differ between NFS clients, but inode/size/times must match.
        if list(expected[1:]) != [info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]:
            raise ValueError(f"Fresh audit is stale: {path}")
        if Path(path + ".partial").exists():
            raise ValueError(f"Partial reconstruction artifact exists: {path}")


def validate_audit(report, source):
    expected = {str((source / name).resolve()) for name in ARTIFACTS}
    if (report.get("status") != "passed" or report.get("stage") != "clustered"
            or report.get("errors") != [] or set(report.get("checks", {})) != AUDIT_CHECKS
            or set(report.get("artifact_signatures", {})) != expected
            or set(report.get("paths", {}).values()) != expected):
        raise ValueError("A complete, passed, correctly bound clustered audit is required")
    if any(v.get("status") != "passed" or v.get("error_count") != 0 for v in report["checks"].values()):
        raise ValueError("Reconstruction audit contains failed checks")


def fresh_audit(source):
    spec = importlib.util.spec_from_file_location("metadata_repair_graph_audit", ROOT / "scripts/audit_horizyn1_reconstruction.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Audit(source, "clustered").run()


def load_context(run):
    run = Path(run).resolve()
    state = run / "state"
    if any(p.name not in {"preflight.json", "prepare.json", "work_directory.json"}
           for p in state.glob("*.json")):
        raise ValueError("Downstream stage records exist; this repair is preparation-only")
    prepare = read_json(state / "prepare.json")
    binding = read_json(run / "data/preparation_state.json")
    work_binding = read_json(state / "work_directory.json")
    if (binding.get("schema_version") != "horizyn1_circe_v2_training_preparation_v1"
            or set(binding.get("inputs", {})) != INPUT_NAMES):
        raise ValueError("Unsupported preparation input binding")
    for name, record in binding["inputs"].items():
        if not {"path", "size", "mtime_ns", "sha256"}.issubset(record):
            raise ValueError(f"Incomplete historical content binding: {name}")
    fp = prepare["fingerprint"]
    if digest(ROOT / "scripts/horizyn1_circe_v2_pipeline.py")["sha256"] != fp["controller_sha256"]:
        raise ValueError("Controller code changed; metadata repair cannot authorize code migration")
    work = Path(binding["work_dir"])
    expected = {"schema": 1, "profile": fp["parameters"]["profile"], "run_root": str(run),
                "storage": "shared", "work_dir": str(work)}
    if work_binding != expected or work != run / "work/preparation":
        raise ValueError("Work-directory binding changed or unsupported scratch layout")
    owner = {"schema_version": binding["schema_version"], "output_dir": str(run / "data"), "work_dir": str(work)}
    if read_json(work / "preparation_owner.json") != owner:
        raise ValueError("Scratch ownership changed")
    if not (work / "preparation.sqlite").is_file() or not (work / "preparation.sqlite").stat().st_size:
        raise ValueError("Bound preparation database is missing; never recreate it during metadata repair")
    if fp["parameters"]["work_dir"] != str(work):
        raise ValueError("Preparation work parameter changed")
    for name, value in fp["parameters"].items():
        if name not in {"profile", "work_dir"} and binding["parameters"].get(name) != value:
            raise ValueError(f"Preparation parameter changed: {name}")
    source = Path(binding["inputs"]["representative_fasta"]["path"]).parent.parent
    paths = {name: Path(record["path"]) for name, record in binding["inputs"].items()}
    expected_paths = {"builder": ROOT / "scripts/prepare_horizyn1_training.py",
                      "implementation": ROOT / "horizyn/datasets/horizyn1_training.py",
                      "representative_fasta": source / "clustered/proteins.fasta",
                      "clustered_pairs": source / "clustered/pairs.tsv",
                      "raw_pairs": source / "raw/raw_pairs.tsv", "reactions": source / "raw/raw_reactions.tsv"}
    if any(paths[name] != path.resolve() for name, path in expected_paths.items()):
        raise ValueError("Preparation source paths changed")
    wanted = {str(p.resolve()) for p in paths.values()} | {str(state / "work_directory.json")}
    records = fp["inputs"]
    if len(records) != len(wanted) or {r["path"] for r in records} != wanted:
        raise ValueError("Preparation fingerprint input coverage changed")
    for record in records:
        only_ctime_changed(record, signature(record["path"]))
    return {"run": run, "source": source, "work": work, "prepare": prepare, "binding": binding,
            "audit_path": source / "logs/clustered_integrity_audit.json"}


def verify_historical_inputs(context):
    result = {}
    for name, expected in context["binding"]["inputs"].items():
        progress(f"Verifying historical SHA-256: {name}")
        actual = digest(expected["path"])
        if any(actual[key] != expected[key] for key in ("path", "size", "mtime_ns", "sha256")):
            raise ValueError(f"Historical content binding changed: {name}")
        result[actual["path"]] = actual
    return result


def verify(run_root: Path, bundle_dir: Path) -> dict:
    context = load_context(run_root)
    run, source = context["run"], context["source"]
    bundle = Path(bundle_dir).resolve()
    if run not in bundle.parents or any(base == bundle or base in bundle.parents
                                       for base in (run / "state", run / "data", run / "work")):
        raise ValueError("Repair bundle must be a new directory beneath RUN_ROOT, outside state/data/work")
    if bundle.exists() and any(bundle.iterdir()):
        raise ValueError("Repair bundle is not empty; use a new directory")
    bundle.mkdir(parents=True, exist_ok=True)
    old_audit_bytes = context["audit_path"].read_bytes()
    old_audit = json.loads(old_audit_bytes)
    validate_audit(old_audit, source)
    # Do not reinterpret an inode/content-stat change as a permission-only event.
    for path, old in old_audit["artifact_signatures"].items():
        current = signature(path)
        if list(old[1:4]) != [current["inode"], current["size"], current["mtime_ns"]]:
            raise ValueError(f"Reconstruction artifact changed beyond ctime: {path}")
    hashes = verify_historical_inputs(context)
    auditor = digest(ROOT / "scripts/audit_horizyn1_reconstruction.py")
    progress("Running a fresh READ-ONLY reconstruction integrity audit")
    audit = fresh_audit(source)
    validate_audit(audit, source)
    for key in ("checks", "observed_counts", "paths", "provenance_sources", "paper_targets", "paper_deltas", "stage"):
        if audit.get(key) != old_audit.get(key):
            raise ValueError(f"Fresh reconstruction audit disagrees with previous evidence: {key}")
    audit_signatures_current(audit)
    for path in audit["artifact_signatures"]:
        if path not in hashes:
            progress(f"Recording newly revalidated content SHA-256: {Path(path).name}")
            hashes[path] = digest(path)
    for path, proof in hashes.items():
        if signature(path) != {k: v for k, v in proof.items() if k != "sha256"}:
            raise ValueError(f"Input changed during verification: {path}")
    final = load_context(run)
    if final["prepare"]["fingerprint"] != context["prepare"]["fingerprint"] or final["binding"] != context["binding"]:
        raise ValueError("Preparation binding changed during verification")
    if context["audit_path"].read_bytes() != old_audit_bytes:
        raise ValueError("Original reconstruction audit changed during verification")
    if digest(ROOT / "scripts/audit_horizyn1_reconstruction.py") != auditor:
        raise ValueError("Auditor implementation changed during verification")
    plan = {"schema": SCHEMA, "run_root": str(run), "source_root": str(source),
            "prepare_fingerprint": context["prepare"]["fingerprint"], "preparation_binding": context["binding"],
            "audit_original_sha256": sha_bytes(old_audit_bytes), "audit_candidate_sha256": sha_bytes(payload(audit)),
            "input_proofs": hashes, "auditor_sha256": auditor["sha256"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "verification_note": "Historical SHA-256 checked where available; other graph artifacts freshly re-audited and newly hashed. No historical byte-identity claim for those artifacts."}
    publish_bytes(bundle / "reconstruction_audit.json", payload(audit))
    publish_bytes(bundle / "original_reconstruction_audit.json", old_audit_bytes)
    publish_bytes(bundle / "bundle.json", payload(plan))
    progress(f"Verification complete. No live receipts changed. Apply after stopping the remote job: {bundle}")
    return plan


@contextmanager
def held_lock(path, *, create=False):
    path = Path(path)
    if path.is_symlink() or (not create and not path.is_file()):
        raise ValueError(f"Missing or unsafe lock: {path}")
    with path.open("a+" if create else "r+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Run is still active or another repair holds {path}; stop on the execution node first") from exc
        yield


def fingerprint_without_ctime(fp):
    value = json.loads(json.dumps(fp))
    for record in value["inputs"]:
        record.pop("ctime_ns", None)
    return value


def apply(bundle_dir: Path) -> dict:
    bundle = Path(bundle_dir).resolve()
    plan = read_json(bundle / "bundle.json")
    if plan.get("schema") != SCHEMA:
        raise ValueError("Unknown repair bundle schema")
    run = Path(plan["run_root"]).resolve()
    if run not in bundle.parents or any(base == bundle or base in bundle.parents
                                       for base in (run / "state", run / "data", run / "work")):
        raise ValueError("Repair bundle/run path mismatch")
    backup = bundle / "backup"
    if backup.is_symlink() or (backup.exists() and not backup.is_dir()) or backup.resolve().parent != bundle:
        raise ValueError("Unsafe repair backup directory")
    work = run / "work/preparation"
    with ExitStack() as locks:
        for path in (run / "pipeline.lock", run / "data/.preparation.lock", work / ".preparation.lock"):
            locks.enter_context(held_lock(path))
        locks.enter_context(held_lock(bundle / ".apply.lock", create=True))
        context = load_context(run)
        locks.enter_context(held_lock(context["source"] / "logs/.metadata_repair.lock", create=True))
        if context["source"] != Path(plan["source_root"]) or context["binding"] != plan["preparation_binding"]:
            raise ValueError("Preparation binding changed after verification")
        if fingerprint_without_ctime(context["prepare"]["fingerprint"]) != fingerprint_without_ctime(plan["prepare_fingerprint"]):
            raise ValueError("Preparation fingerprint changed after verification")
        if digest(ROOT / "scripts/audit_horizyn1_reconstruction.py")["sha256"] != plan["auditor_sha256"]:
            raise ValueError("Auditor implementation changed after verification")
        audit_bytes = (bundle / "reconstruction_audit.json").read_bytes()
        if sha_bytes(audit_bytes) != plan["audit_candidate_sha256"]:
            raise ValueError("Verified audit bundle changed")
        audit = json.loads(audit_bytes)
        validate_audit(audit, context["source"])
        audit_signatures_current(audit)
        wanted_proofs = {r["path"] for r in context["binding"]["inputs"].values()} | set(audit["artifact_signatures"])
        if set(plan["input_proofs"]) != wanted_proofs:
            raise ValueError("Incomplete input proof coverage")
        for path, expected in plan["input_proofs"].items():
            progress(f"Rechecking content before apply: {Path(path).name}")
            if digest(path) != expected:
                raise ValueError(f"Content or metadata changed after verification: {path}")
        # Historical content hashes must remain evidence, not a mutable plan's assertion.
        for expected in context["binding"]["inputs"].values():
            if plan["input_proofs"][expected["path"]]["sha256"] != expected["sha256"]:
                raise ValueError("Repair proof conflicts with historical input hash")
        targets = {"prepare": run / "state/prepare.json", "audit": context["audit_path"]}
        transaction_path = bundle / "transaction.json"
        if transaction_path.exists():
            transaction = read_json(transaction_path)
        else:
            if digest(targets["audit"])["sha256"] != plan["audit_original_sha256"]:
                raise ValueError("Original audit changed; create a fresh verification bundle")
            if context["prepare"]["fingerprint"] != plan["prepare_fingerprint"]:
                raise ValueError("Original preparation fingerprint changed")
            updated = json.loads(json.dumps(context["prepare"]))
            for record in updated["fingerprint"]["inputs"]:
                only_ctime_changed(record, signature(record["path"]))
                record["ctime_ns"] = signature(record["path"])["ctime_ns"]
            desired = {"prepare": payload(updated), "audit": audit_bytes}
            backup.mkdir(exist_ok=True)
            transaction = {"targets": {}}
            for name, path in targets.items():
                before = path.read_bytes()
                publish_bytes(backup / f"{name}.json", before)
                publish_bytes(bundle / f"publish_{name}.json", desired[name])
                transaction["targets"][name] = {"path": str(path), "before_sha256": sha_bytes(before),
                                                 "after_sha256": sha_bytes(desired[name])}
            if (run / "state/preflight.json").is_file():
                publish_bytes(backup / "preflight.json", (run / "state/preflight.json").read_bytes())
            publish_bytes(transaction_path, payload(transaction))
        if set(transaction["targets"]) != set(targets):
            raise ValueError("Unexpected repair publication targets")
        # A resumed transaction must still describe precisely this verified
        # repair, not merely contain internally consistent replacement hashes.
        original_prepare = read_json(bundle / "backup/prepare.json")
        if original_prepare["fingerprint"] != plan["prepare_fingerprint"]:
            raise ValueError("Preparation backup disagrees with verified fingerprint")
        expected_prepare = json.loads(json.dumps(original_prepare))
        for record in expected_prepare["fingerprint"]["inputs"]:
            only_ctime_changed(record, signature(record["path"]))
            record["ctime_ns"] = signature(record["path"])["ctime_ns"]
        if (bundle / "publish_prepare.json").read_bytes() != payload(expected_prepare):
            raise ValueError("Staged preparation repair changes more than input ctimes")
        if (bundle / "publish_audit.json").read_bytes() != audit_bytes:
            raise ValueError("Staged audit differs from verified candidate")
        if sha_bytes((bundle / "backup/audit.json").read_bytes()) != plan["audit_original_sha256"]:
            raise ValueError("Audit backup differs from verified original")
        # Validate every publication before the first write. Interrupted apply
        # accepts only exact old/new bytes and completes the same transaction.
        for name, path in targets.items():
            record = transaction["targets"][name]
            if record["path"] != str(path) or path.is_symlink():
                raise ValueError("Unsafe repair target path")
            if sha_bytes((bundle / f"publish_{name}.json").read_bytes()) != record["after_sha256"]:
                raise ValueError("Staged repair publication changed")
            if sha_bytes((bundle / "backup" / f"{name}.json").read_bytes()) != record["before_sha256"]:
                raise ValueError("Repair backup changed")
            if digest(path)["sha256"] not in {record["before_sha256"], record["after_sha256"]}:
                raise ValueError("Repair target changed outside this transaction")
        for path, proof in plan["input_proofs"].items():
            if signature(path) != {k: v for k, v in proof.items() if k != "sha256"}:
                raise ValueError(f"Input changed before publication: {path}")
        audit_signatures_current(audit)
        # Publish audit first: a partial apply remains fail-closed on resume.
        for name in ("audit", "prepare"):
            path = targets[name]
            if digest(path)["sha256"] != transaction["targets"][name]["after_sha256"]:
                publish_bytes(path, (bundle / f"publish_{name}.json").read_bytes())
        receipt = {"status": "applied", "run_root": str(run), "backup_dir": str(bundle / "backup"),
                   "next_step": "Relaunch all so normal preflight regenerates its own records. No pipeline code, data, database or preflight records were modified."}
        publish_bytes(bundle / "applied.json", payload(receipt))
        progress("Repair applied. Original receipts backed up; normal preflight is still required.")
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "apply"))
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "verify":
        if args.run_root is None:
            parser.error("verify requires --run-root")
        result = verify(args.run_root, args.bundle_dir)
    else:
        result = apply(args.bundle_dir)
    print(json.dumps({k: result[k] for k in ("status", "run_root", "next_step") if k in result}, indent=2))


if __name__ == "__main__":
    main()
