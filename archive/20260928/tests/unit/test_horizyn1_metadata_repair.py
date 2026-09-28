"""Metadata recovery must not turn timestamp repair into stale-data acceptance."""

import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest


PROJECT = Path(__file__).resolve().parents[2]
SCHEMA = "horizyn1_circe_v2_training_preparation_v1"
REJECTIONS = (ValueError, RuntimeError, OSError)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def write_json(path, value):
    return write(path, json.dumps(value, sort_keys=True, indent=2) + "\n")


def read_json(path):
    return json.loads(path.read_text())


def stat_record(path):
    info = path.stat()
    return {"path": str(path.resolve()), "inode": info.st_ino, "size": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}


def hash_record(path):
    record = stat_record(path)
    return {key: record[key] for key in ("path", "size", "mtime_ns")} | {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def change_content_preserving_size_and_mtime(path):
    before = path.stat()
    original = path.read_bytes()
    assert original
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns


@pytest.fixture
def case(tmp_path, monkeypatch):
    repair = load_script("repair_horizyn1_run_metadata")
    audit = load_script("audit_horizyn1_reconstruction")
    project = tmp_path / "project"
    source = project / "data/reconstructed/horizyn1_2023_05"
    run = project / "runs/horizyn1_circe_v2_h200"
    work = run / "work/preparation"
    contents = {
        "raw/raw_proteins.fasta": ">P1 description\nMA\nAA\n>P2\nMAAT\n>P3\nMCCC\n",
        "raw/raw_reactions.tsv": "reaction_id\treaction_smiles\nEm_1\tC>>CO\nRh_1\tCO>>C=O\n",
        "raw/raw_pairs.tsv": "reaction_id\tprotein_id\tsources\nEm_1\tP3\tEnzymeMap_v2\nRh_1\tP1\tdevelopment_release\nRh_1\tP2\tUniProtKB_TrEMBL_2023_05,development_release\n",
        "clustered/proteins.fasta": ">P1\nMAAA\n>P3\nMCCC\n",
        "clustered/clusters.tsv": "P1\tP1\nP1\tP2\nP3\tP3\n",
        "clustered/pairs.tsv": "reaction_id\tprotein_id\tsources\nEm_1\tP3\tEnzymeMap_v2\nRh_1\tP1\tdevelopment_release,UniProtKB_TrEMBL_2023_05,development_release\n",
    }
    for relative, value in contents.items():
        write(source / relative, value)
    write_json(source / "raw/raw_manifest.json", {"raw_proteins": 3, "raw_reactions": 2, "raw_pairs": 3})
    write_json(source / "clustered/clustered_manifest.json", {
        "clustered_proteins": 2, "clustered_pairs": 2, "clustered_members": 3})
    controller = write(project / "scripts/horizyn1_circe_v2_pipeline.py", "# fixed controller\n")
    write(project / "scripts/audit_horizyn1_reconstruction.py",
          (PROJECT / "scripts/audit_horizyn1_reconstruction.py").read_text())
    hashed_inputs = {
        "builder": write(project / "scripts/prepare_horizyn1_training.py", "# fixed builder\n"),
        "implementation": write(project / "horizyn/datasets/horizyn1_training.py", "# fixed implementation\n"),
        "mmseqs": write(project / ".deps/mmseqs/bin/mmseqs", "# fixed executable\n"),
        "representative_fasta": source / "clustered/proteins.fasta",
        "clustered_pairs": source / "clustered/pairs.tsv",
        "raw_pairs": source / "raw/raw_pairs.tsv",
        "reactions": source / "raw/raw_reactions.tsv",
    }
    owner = write_json(work / "preparation_owner.json", {
        "schema_version": SCHEMA, "output_dir": str(run / "data"), "work_dir": str(work)})
    with sqlite3.connect(work / "preparation.sqlite") as database:
        database.execute("CREATE TABLE stages(name TEXT PRIMARY KEY,value TEXT NOT NULL)")
    binding = write_json(run / "state/work_directory.json", {
        "schema": 1, "run_root": str(run), "profile": "h200", "storage": "shared", "work_dir": str(work)})
    parameters = {"coverage": .8, "min_seq_id": .5, "reaction_similarity_threshold": .8,
                  "seed": 42, "sqlite_cache_mib": 512, "test_fraction": .05,
                  "threads": 64, "validation_fraction": .05}
    preparation_state = write_json(run / "data/preparation_state.json", {
        "schema_version": SCHEMA, "inputs": {key: hash_record(path) for key, path in hashed_inputs.items()},
        "parameters": parameters, "work_dir": str(work), "log_dir": str(run / "logs/preparation"),
        "storage_policy": {"filesystem": "nfs4", "journal_mode": "delete", "locking_mode": "exclusive", "mount_point": "/datastor2"},
        "versions": {"python": sys.version, "sqlite": sqlite3.sqlite_version, "rdkit": "test", "mmseqs": "test"},
    })
    previous_audit = audit.Audit(source, "clustered").run()
    assert previous_audit["status"] == "passed"
    for values in previous_audit["artifact_signatures"].values():
        values[0] += 99  # Device IDs legitimately differ between mounts/nodes.
        values[4] -= 1  # Historical ctime, with unchanged bytes/inode/mtime.
    audit_path = write_json(source / "logs/clustered_integrity_audit.json", previous_audit)
    records = [stat_record(path) for path in [*hashed_inputs.values(), binding]]
    for record in records:
        record["ctime_ns"] -= 1
    prepare_path = write_json(run / "state/prepare.json", {
        "status": "running", "started": 1234,
        "fingerprint": {"controller_sha256": hashlib.sha256(controller.read_bytes()).hexdigest(),
                        "inputs": records, "parameters": parameters | {"profile": "h200", "work_dir": str(work)}},
    })
    preflight = write_json(run / "state/preflight.json", {
        "status": "complete", "profile": "h200", "work_dir": str(work), "time": 1233,
        "required_inputs": [stat_record(audit_path)],
    })
    for lock in (run / "pipeline.lock", run / "data/.preparation.lock", work / ".preparation.lock"):
        write(lock, "")
    monkeypatch.setattr(repair, "ROOT", project)
    return SimpleNamespace(repair=repair, audit=audit, project=project, source=source, run=run,
                           work=work, owner=owner, binding=binding, controller=controller,
                           hashed_inputs=hashed_inputs, prepare=prepare_path, preflight=preflight,
                           preparation_state=preparation_state, audit_path=audit_path,
                           bundle=run / "repairs/verification_bundle")


def snapshot(case):
    return {path: path.read_bytes() for path in [case.prepare, case.preflight, case.preparation_state,
            case.audit_path, case.owner, case.binding, *case.hashed_inputs.values()]}


def test_verify_writes_proof_without_refreshing_active_state(case):
    before = snapshot(case)
    result = case.repair.verify(case.run, case.bundle)
    assert isinstance(result, dict)
    assert (case.bundle / "bundle.json").is_file()
    assert read_json(case.bundle / "reconstruction_audit.json")["status"] == "passed"
    assert snapshot(case) == before


def test_read_only_verification_can_run_while_controller_is_locked(case):
    before = snapshot(case)
    with (case.run / "pipeline.lock").open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        case.repair.verify(case.run, case.bundle)
    assert snapshot(case) == before


def test_apply_repairs_only_input_ctimes_and_reaudits_without_touching_preflight(case):
    before = snapshot(case)
    original = read_json(case.prepare)
    case.repair.verify(case.run, case.bundle)
    result = case.repair.apply(case.bundle)
    assert isinstance(result, dict)
    expected = copy.deepcopy(original)
    for record in expected["fingerprint"]["inputs"]:
        record["ctime_ns"] = Path(record["path"]).stat().st_ctime_ns
    assert read_json(case.prepare) == expected
    for path, value in before.items():
        if path not in {case.prepare, case.audit_path}:
            assert path.read_bytes() == value
    assert read_json(case.audit_path)["status"] == "passed"
    for name, record in read_json(case.audit_path)["artifact_signatures"].items():
        actual = list(case.audit.signature(Path(name)))
        assert record[1:] == actual[1:]


def test_completed_apply_is_idempotent(case):
    case.repair.verify(case.run, case.bundle)
    case.repair.apply(case.bundle)
    before = snapshot(case)
    case.repair.apply(case.bundle)
    assert snapshot(case) == before


def test_verified_stop_may_change_running_to_failed_without_changing_fingerprint(case):
    case.repair.verify(case.run, case.bundle)
    stopped = read_json(case.prepare)
    stopped.update(status="failed", error="143")
    write_json(case.prepare, stopped)
    case.repair.apply(case.bundle)
    result = read_json(case.prepare)
    assert result["status"] == "failed"
    assert result["error"] == "143"
    assert result["started"] == stopped["started"]


def test_apply_refuses_tampered_verified_audit(case):
    case.repair.verify(case.run, case.bundle)
    verified = case.bundle / "reconstruction_audit.json"
    report = read_json(verified)
    report["observed_counts"]["raw_proteins"] += 1
    write_json(verified, report)
    before = snapshot(case)
    with pytest.raises(REJECTIONS):
        case.repair.apply(case.bundle)
    assert snapshot(case) == before


@pytest.mark.parametrize("key", ["builder", "implementation", "mmseqs", "representative_fasta", "clustered_pairs", "raw_pairs", "reactions"])
def test_verify_rejects_same_size_same_mtime_content_change(case, key):
    change_content_preserving_size_and_mtime(case.hashed_inputs[key])
    before = snapshot(case)
    with pytest.raises(REJECTIONS):
        case.repair.verify(case.run, case.bundle)
    assert snapshot(case) == before


@pytest.mark.parametrize("change", ["inode", "mtime", "size"])
def test_verify_rejects_non_ctime_stat_changes(case, change):
    path = case.hashed_inputs["raw_pairs"]
    before_stat = path.stat()
    if change == "inode":
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(path.read_bytes())
        os.utime(replacement, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
        replacement.replace(path)
    elif change == "mtime":
        os.utime(path, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns + 1000))
    else:
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(REJECTIONS):
        case.repair.verify(case.run, case.bundle)


@pytest.mark.parametrize("target", ["owner", "binding"])
def test_verify_rejects_foreign_work_binding(case, target):
    path = getattr(case, target)
    value = read_json(path)
    value["work_dir"] = str(case.work.parent / "someone_else")
    write_json(path, value)
    with pytest.raises(REJECTIONS):
        case.repair.verify(case.run, case.bundle)


@pytest.mark.parametrize("change", ["controller", "parameters", "missing_hash"])
def test_verify_rejects_changed_immutable_run_contract(case, change):
    if change == "controller":
        change_content_preserving_size_and_mtime(case.controller)
    elif change == "parameters":
        saved = read_json(case.prepare)
        saved["fingerprint"]["parameters"]["seed"] += 1
        write_json(case.prepare, saved)
    else:
        saved = read_json(case.preparation_state)
        del saved["inputs"]["raw_pairs"]["sha256"]
        write_json(case.preparation_state, saved)
    with pytest.raises(REJECTIONS):
        case.repair.verify(case.run, case.bundle)


@pytest.mark.parametrize("stage", ["labels", "index", "pilot_selection", "train"])
def test_verify_rejects_downstream_stages(case, stage):
    write_json(case.run / "state" / f"{stage}.json", {"status": "running"})
    with pytest.raises(REJECTIONS):
        case.repair.verify(case.run, case.bundle)


@pytest.mark.parametrize("which", ["pipeline", "output", "work"])
def test_apply_requires_all_writer_locks(case, which):
    case.repair.verify(case.run, case.bundle)
    path = {"pipeline": case.run / "pipeline.lock", "output": case.run / "data/.preparation.lock",
            "work": case.work / ".preparation.lock"}[which]
    before = snapshot(case)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(REJECTIONS):
            case.repair.apply(case.bundle)
    assert snapshot(case) == before


@pytest.mark.parametrize("change", ["source", "audit_source", "controller", "arguments", "downstream"])
def test_apply_rechecks_proof_after_verification(case, change):
    case.repair.verify(case.run, case.bundle)
    if change == "source":
        change_content_preserving_size_and_mtime(case.hashed_inputs["raw_pairs"])
    elif change == "audit_source":
        # This file lacks a historical preparation SHA but is bound by fresh audit stats.
        change_content_preserving_size_and_mtime(case.source / "raw/raw_proteins.fasta")
    elif change == "controller":
        change_content_preserving_size_and_mtime(case.controller)
    elif change == "arguments":
        saved = read_json(case.prepare)
        saved["fingerprint"]["parameters"]["seed"] += 1
        write_json(case.prepare, saved)
    else:
        write_json(case.run / "state/labels.json", {"status": "running"})
    before = snapshot(case)
    with pytest.raises(REJECTIONS):
        case.repair.apply(case.bundle)
    assert snapshot(case) == before


def test_verify_refuses_foreign_nonempty_bundle(case):
    sentinel = write(case.bundle / "unrelated.txt", "preserve this user file\n")
    with pytest.raises(REJECTIONS):
        case.repair.verify(case.run, case.bundle)
    assert sentinel.read_text() == "preserve this user file\n"


@pytest.mark.parametrize("change", ["failed", "missing_artifact", "foreign_path", "counts", "checks"])
def test_verify_does_not_replace_an_invalid_or_semantically_different_old_audit(case, change):
    report = read_json(case.audit_path)
    if change == "failed":
        report["status"] = "failed"
        report["errors"] = [{"check": "completed_inputs", "count": 1, "examples": ["bad input"]}]
    elif change == "missing_artifact":
        report["artifact_signatures"].pop(str(case.source / "raw/raw_proteins.fasta"))
    elif change == "foreign_path":
        original = str(case.source / "raw/raw_proteins.fasta")
        report["artifact_signatures"][str(case.source / "foreign.fasta")] = report["artifact_signatures"].pop(original)
    elif change == "counts":
        report["observed_counts"]["raw_proteins"] += 1
    else:
        report["checks"]["foreign_check"] = {"status": "passed", "error_count": 0, "examples": []}
    write_json(case.audit_path, report)
    before = snapshot(case)
    with pytest.raises(REJECTIONS):
        case.repair.verify(case.run, case.bundle)
    assert snapshot(case) == before


def test_new_reconstruction_audit_must_actually_pass(case):
    path = case.source / "raw/raw_proteins.fasta"
    original = path.stat()
    path.write_text(path.read_text().replace("MAAA", "MAAT").replace("MA\nAA", "MA\nAT"))
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    before = snapshot(case)
    with pytest.raises(REJECTIONS):
        case.repair.verify(case.run, case.bundle)
    assert snapshot(case) == before


@pytest.mark.parametrize("target_name", ["prepare", "audit_path"])
@pytest.mark.parametrize("timing", ["before", "after"])
def test_interrupted_publication_can_be_retried_without_losing_originals(case, monkeypatch, target_name, timing):
    case.repair.verify(case.run, case.bundle)
    before = snapshot(case)
    target = getattr(case, target_name).resolve()
    original_replace = os.replace
    fired = False

    def interrupted_replace(source, destination, *args, **kwargs):
        nonlocal fired
        if Path(destination).resolve() == target and not fired:
            fired = True
            if timing == "after":
                original_replace(source, destination, *args, **kwargs)
            raise OSError("simulated interruption at publication")
        return original_replace(source, destination, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupted_replace)
        with pytest.raises(REJECTIONS):
            case.repair.apply(case.bundle)
    assert fired, "Repair publications must use atomic replacement"
    case.repair.apply(case.bundle)
    repaired = read_json(case.prepare)
    for record in repaired["fingerprint"]["inputs"]:
        assert record["ctime_ns"] == Path(record["path"]).stat().st_ctime_ns
    for path, value in before.items():
        if path not in {case.prepare, case.audit_path}:
            assert path.read_bytes() == value
    # Both pre-repair artifacts must remain recoverable after partial success.
    backups = [path.read_bytes() for path in case.bundle.rglob("*") if path.is_file()]
    assert before[case.prepare] in backups
    assert before[case.audit_path] in backups


@pytest.mark.parametrize("interrupted_target", ["prepare", "audit_path"])
@pytest.mark.parametrize("tampered_publication", ["prepare", "audit"])
def test_coherently_tampered_transaction_cannot_authorize_another_repair(
        case, monkeypatch, interrupted_target, tampered_publication):
    case.repair.verify(case.run, case.bundle)
    original_publish = case.repair.publish_bytes
    interrupted_path = getattr(case, interrupted_target)

    def interrupt_target(path, contents):
        if Path(path) == interrupted_path:
            raise OSError("simulated stopped transaction")
        return original_publish(path, contents)

    with monkeypatch.context() as patch:
        patch.setattr(case.repair, "publish_bytes", interrupt_target)
        with pytest.raises(OSError, match="simulated stopped transaction"):
            case.repair.apply(case.bundle)
    transaction_path = case.bundle / "transaction.json"
    assert transaction_path.is_file()
    publication_path = case.bundle / f"publish_{tampered_publication}.json"
    publication = read_json(publication_path)
    if tampered_publication == "prepare":
        publication["fingerprint"]["parameters"]["seed"] += 1
    else:
        publication["observed_counts"]["raw_proteins"] += 1
    write_json(publication_path, publication)
    transaction = read_json(transaction_path)
    transaction["targets"][tampered_publication]["after_sha256"] = hashlib.sha256(publication_path.read_bytes()).hexdigest()
    write_json(transaction_path, transaction)
    before = snapshot(case)
    with pytest.raises(ValueError, match="Staged preparation repair|Staged audit differs"):
        case.repair.apply(case.bundle)
    assert snapshot(case) == before


def test_auditor_implementation_cannot_change_during_fresh_audit(case, monkeypatch):
    original_fresh_audit = case.repair.fresh_audit
    auditor_path = case.project / "scripts/audit_horizyn1_reconstruction.py"

    def change_auditor_after_scan(source):
        report = original_fresh_audit(source)
        auditor_path.write_text(auditor_path.read_text() + "\n# concurrent implementation change\n")
        return report

    before = snapshot(case)
    monkeypatch.setattr(case.repair, "fresh_audit", change_auditor_after_scan)
    with pytest.raises(ValueError, match="Auditor|auditor"):
        case.repair.verify(case.run, case.bundle)
    assert snapshot(case) == before
    assert not (case.bundle / "bundle.json").exists()


@pytest.mark.parametrize("kind", ["symlink", "file"])
def test_backup_path_cannot_escape_bundle_or_overwrite_foreign_data(case, kind):
    case.repair.verify(case.run, case.bundle)
    outside = case.project.parent / "unrelated_user_files"
    outside_contents = {}
    for name in ("prepare.json", "audit.json", "preflight.json", "notes.txt"):
        path = write(outside / name, f"preserve unrelated {name}\n")
        outside_contents[path] = path.read_bytes()
    backup = case.bundle / "backup"
    if kind == "symlink":
        backup.symlink_to(outside, target_is_directory=True)
    else:
        write(backup, "preserve existing backup-named user file\n")
    before = snapshot(case)
    with pytest.raises(ValueError, match="Unsafe repair backup directory"):
        case.repair.apply(case.bundle)
    assert snapshot(case) == before
    assert {path: path.read_bytes() for path in outside.iterdir()} == outside_contents
    if kind == "symlink":
        assert backup.is_symlink()
    else:
        assert backup.read_text() == "preserve existing backup-named user file\n"
    assert not (case.bundle / "transaction.json").exists()


@pytest.mark.parametrize("forbidden_parent", ["state", "data", "work"])
def test_apply_rechecks_bundle_location_even_after_successful_verification(case, forbidden_parent):
    case.repair.verify(case.run, case.bundle)
    forbidden = case.run / forbidden_parent / "misplaced_repair_bundle"
    case.bundle.rename(forbidden)
    before = snapshot(case)
    with pytest.raises(ValueError, match="Repair bundle/run path mismatch"):
        case.repair.apply(forbidden)
    assert snapshot(case) == before
    assert not (forbidden / "transaction.json").exists()
    assert not (forbidden / ".apply.lock").exists()
