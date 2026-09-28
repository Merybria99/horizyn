#!/usr/bin/env python3
"""Repair one SHA-verified clustering mtime in a stopped preparation checkpoint.

Does not change clustering, splits, source files, pipeline code or resume guards.
Backs up the complete database and any inactive rollback journal before a single
transactional stages.groups update. A nonempty unfinished bundle is not replayed:
inspect its backups and use a fresh bundle if another attempt is necessary.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import repair_horizyn1_run_metadata as metadata


def check_sidecars(path):
    for suffix in ("-wal", "-shm"):
        sidecar = path.with_name(path.name + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            raise ValueError("WAL/SHM sidecar exists; this repair supports stopped rollback-journal databases only")
    journal = path.with_name(path.name + "-journal")
    if journal.is_symlink():
        raise ValueError("Unsafe rollback journal")
    if journal.exists():
        metadata.signature(journal)
        with journal.open("rb") as handle:
            if any(handle.read(28)):
                raise ValueError("Nonzero rollback journal header; recover/inspect the database separately first")
        return journal
    return None


def stages(database):
    if database.execute("SELECT 1 FROM sqlite_master WHERE type='trigger' LIMIT 1").fetchone():
        raise ValueError("Unexpected database triggers")
    columns = database.execute("PRAGMA table_info(stages)").fetchall()
    if [(row[1], row[2], row[3], row[5]) for row in columns] != [
            ("name", "TEXT", 0, 1), ("value", "TEXT", 1, 0)]:
        raise ValueError("Unexpected stages table schema")
    values = dict(database.execute("SELECT name,value FROM stages"))
    if set(values) != {"entities", "groups"}:
        raise ValueError("This repair requires exactly entities/groups; later or missing stages need separate review")
    return values


def content_evidence(path):
    proof = metadata.digest(path)
    return {key: proof[key] for key in ("path", "size", "mtime_ns", "sha256")}, proof


def copy_backup(source, destination):
    before = metadata.signature(source)
    digest = hashlib.sha256()
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as output, source.open("rb") as input_file:
        for block in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
            output.write(block)
            digest.update(block)
        output.flush()
        os.fsync(output.fileno())
    metadata.sync_directory(destination.parent)
    if metadata.signature(source) != before:
        raise ValueError(f"Source changed while backing up: {source}")
    result = {**before, "sha256": digest.hexdigest(), "backup_path": str(destination)}
    if metadata.digest(destination)["sha256"] != result["sha256"]:
        raise ValueError("Backup failed independent checksum verification")
    return result


def update_groups(database, before, after):
    cursor = database.execute("UPDATE stages SET value=? WHERE name='groups' AND value=?", (after, before))
    if cursor.rowcount != 1:
        raise ValueError("Groups checkpoint changed before update")


def repair(run_root: Path, bundle_dir: Path) -> dict:
    run = Path(run_root).resolve()
    bundle = Path(bundle_dir).resolve()
    if run not in bundle.parents or any(base == bundle or base in bundle.parents
                                       for base in (run / "state", run / "data", run / "work")):
        raise ValueError("Repair bundle must be beneath RUN_ROOT, outside state/data/work")
    work = run / "work/preparation"
    with ExitStack() as locks:
        for path in (run / "pipeline.lock", run / "data/.preparation.lock", work / ".preparation.lock"):
            locks.enter_context(metadata.held_lock(path))
        context = metadata.load_context(run)
        if context["prepare"].get("status") not in {"failed", "running"}:
            raise ValueError("This repair is for an incomplete stopped preparation only")
        for record in context["prepare"]["fingerprint"]["inputs"]:
            if metadata.signature(record["path"]) != record:
                raise ValueError("Controller input metadata is stale; this repair does not rebase controller receipts")
        if (run / "data/preparation_manifest.json").exists():
            raise ValueError("Completed preparation needs separate review")
        input_proofs = metadata.verify_historical_inputs(context)
        database_path = work / "preparation.sqlite"
        if database_path.is_symlink():
            raise ValueError("Unsafe preparation database path")
        journal = check_sidecars(database_path)
        journal_before = metadata.signature(journal) if journal else None
        database_before = metadata.signature(database_path)
        reader = sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True, timeout=2)
        try:
            reader.execute("PRAGMA query_only=ON")
            before = stages(reader)
        finally:
            reader.close()
        groups = json.loads(before["groups"])
        paths = {"protein_cluster_file": work / "protein_clusters_cluster.tsv",
                 "reaction_similarity_edges": run / "data/reaction_similarity_edges.tsv"}
        proofs = {}
        updated = json.loads(before["groups"])
        for name, path in paths.items():
            expected = groups[name]
            if path.is_symlink() or expected.get("path") != str(path):
                raise ValueError(f"Unexpected artifact path: {name}")
            current, proof = content_evidence(path)
            allowed = {"mtime_ns"} if name == "protein_cluster_file" else set()
            if set(expected) != set(current) or any(expected[key] != current[key] for key in current if key not in allowed):
                raise ValueError(f"Artifact content/binding differs beyond the authorized cluster mtime: {name}")
            updated[name] = current
            proofs[str(path)] = proof
        if updated == groups:
            return {"status": "already_current", "run_root": str(run), "message": "Both committed group evidence records match; no writes performed"}
        if metadata.signature(database_path) != database_before:
            raise ValueError("Database changed during read-only verification")
        if bundle.exists():
            if not bundle.is_dir() or any(bundle.iterdir()):
                raise ValueError("Bundle is not empty; inspect any previous attempt and choose a new bundle")
        bundle.mkdir(parents=True, exist_ok=True, mode=0o700)
        needed = database_path.stat().st_size + (journal.stat().st_size if journal else 0) + 64 * 1024 * 1024
        if shutil.disk_usage(bundle).free < needed:
            raise ValueError("Insufficient space for full stopped-database backups")
        metadata.progress("Backing up complete stopped preparation database")
        database_backup = copy_backup(database_path, bundle / "backup_preparation.sqlite")
        journal_backup = copy_backup(journal, bundle / "backup_preparation.sqlite-journal") if journal else None
        after = {**before, "groups": json.dumps(updated, sort_keys=True)}
        plan = {"schema": "horizyn1_group_evidence_mtime_repair_v1", "run_root": str(run),
                "database_backup": database_backup, "journal_backup": journal_backup,
                "stages_before": before, "stages_after": after, "artifact_proofs": proofs,
                "input_proofs": input_proofs, "preparation_binding": context["binding"],
                "prepare_receipt": context["prepare"],
                "change": "Only groups.protein_cluster_file.mtime_ns; historical SHA-256 retained"}
        metadata.publish_bytes(bundle / "plan.json", metadata.payload(plan))
        for proof in [*input_proofs.values(), *proofs.values()]:
            if metadata.signature(proof["path"]) != {k: v for k, v in proof.items() if k != "sha256"}:
                raise ValueError("Verified input changed before database update")
        if metadata.signature(database_path) != database_before or check_sidecars(database_path) != journal:
            raise ValueError("Database or sidecars changed before update")
        if journal is not None and metadata.signature(journal) != journal_before:
            raise ValueError("Rollback journal changed during verification/backup")
        final_context = metadata.load_context(run)
        if final_context["binding"] != context["binding"] or final_context["prepare"] != context["prepare"]:
            raise ValueError("Run binding changed before update")
        writer = sqlite3.connect(database_path.as_uri() + "?mode=rw", uri=True, timeout=2)
        try:
            writer.execute("PRAGMA locking_mode=EXCLUSIVE")
            if writer.execute("PRAGMA journal_mode=DELETE").fetchone()[0] != "delete":
                raise ValueError("Could not establish rollback journaling")
            writer.execute("PRAGMA synchronous=FULL")
            writer.execute("BEGIN EXCLUSIVE")
            if stages(writer) != before:
                raise ValueError("Committed stages changed before transaction")
            update_groups(writer, before["groups"], after["groups"])
            if stages(writer) != after:
                raise ValueError("Unexpected checkpoint change during repair")
            if writer.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ValueError("SQLite integrity check failed; rolling back metadata repair")
            for path, expected in proofs.items():
                if metadata.signature(path) != {k: v for k, v in expected.items() if k != "sha256"}:
                    raise ValueError("Artifact changed during metadata transaction")
            writer.commit()
        except BaseException:
            writer.rollback()
            raise
        finally:
            writer.close()
        metadata.sync_directory(work)
        reader = sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True, timeout=2)
        try:
            if stages(reader) != after:
                raise ValueError("Post-commit checkpoint verification failed; retain backups")
        finally:
            reader.close()
        for name, path in paths.items():
            if content_evidence(path)[0] != updated[name]:
                raise ValueError("Post-commit group evidence mismatch; retain backups")
        result = {"status": "applied", "run_root": str(run), "bundle_dir": str(bundle),
                  "database_after": metadata.digest(database_path), "change": plan["change"],
                  "integrity_check": "ok", "next_step": "Relaunch the same all command on the GPU node"}
        metadata.publish_bytes(bundle / "applied.json", metadata.payload(result))
        metadata.progress("Internal group evidence repaired and rechecked; cluster bytes and entity/split rows were not edited")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    args = parser.parse_args()
    result = repair(args.run_root, args.bundle_dir)
    print(json.dumps({key: result[key] for key in ("status", "run_root", "next_step") if key in result}, indent=2))


if __name__ == "__main__":
    main()
