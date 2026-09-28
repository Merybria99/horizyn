"""Stopped-run evidence repair must alter metadata, never grouping decisions."""

import copy
import fcntl
import json
import os
from pathlib import Path
import sqlite3

import pytest

from tests.unit.test_horizyn1_metadata_repair import (
    REJECTIONS,
    case,
    change_content_preserving_size_and_mtime,
    hash_record,
    load_script,
    read_json,
    snapshot,
    stat_record,
    write,
    write_json,
)


@pytest.fixture
def grouped(case, monkeypatch):
    module = load_script("repair_horizyn1_group_evidence")
    monkeypatch.setattr(module, "metadata", case.repair)
    case.group_repair = module
    case.cluster = write(case.work / "protein_clusters_cluster.tsv", "P1\tP1\nP3\tP3\n")
    case.edges = write(case.run / "data/reaction_similarity_edges.tsv",
                       "reaction_id_a\treaction_id_b\tevidence\tsimilarity\n")
    case.database = case.work / "preparation.sqlite"
    parameters = read_json(case.preparation_state)["parameters"]
    groups = {
        "protein_cluster_file": hash_record(case.cluster),
        "protein_cluster_command": [
            str(case.hashed_inputs["mmseqs"]), "easy-linclust",
            str(case.hashed_inputs["representative_fasta"]),
            str(case.work / "protein_clusters"), str(case.work / "cluster_tmp"),
            "--min-seq-id", str(parameters["min_seq_id"]), "-c", str(parameters["coverage"]),
            "--cov-mode", "0", "--threads", str(parameters["threads"]),
        ],
        "reaction_method": "canonical_reversed_alias_plus_signed_morgan_radius2_difference_set_jaccard_connected_components",
        "reaction_similarity_edges": hash_record(case.edges),
    }
    # The historical byte hash still agrees; only this saved mtime is obsolete.
    groups["protein_cluster_file"]["mtime_ns"] -= 1_000_000
    case.old_groups = groups
    with sqlite3.connect(case.database) as connection:
        connection.executescript("""
            CREATE TABLE proteins(id TEXT PRIMARY KEY, sequence_sha256 TEXT NOT NULL,
                length INTEGER NOT NULL, group_id TEXT, split TEXT, sample_key TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE reactions(id TEXT PRIMARY KEY, smiles TEXT NOT NULL,
                group_id TEXT, split TEXT, sample_key TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE pairs(protein_id TEXT, reaction_id TEXT,
                PRIMARY KEY(protein_id,reaction_id)) WITHOUT ROWID;
        """)
        connection.executemany("INSERT INTO proteins VALUES (?,?,?,?,?,?)", [
            ("P1", "a" * 64, 4, "P1", "train", "first"),
            ("P3", "b" * 64, 4, "P3", "test", "second"),
        ])
        connection.executemany("INSERT INTO reactions VALUES (?,?,?,?,?)", [
            ("Em_1", "C>>CO", "Em_1", "train", "reaction_first"),
            ("Rh_1", "CO>>C=O", "Rh_1", "validation", "reaction_second"),
        ])
        connection.execute("INSERT INTO pairs VALUES (?,?)", ("P1", "Em_1"))
        connection.executemany("INSERT INTO stages VALUES (?,?)", [
            ("entities", json.dumps({"proteins": 2, "residues": 8, "reactions": 2})),
            ("groups", json.dumps(groups, sort_keys=True)),
        ])
    prepare = read_json(case.prepare)
    prepare.update(status="failed", error="Committed protein_cluster_file artifact changed")
    prepare["fingerprint"]["inputs"] = [stat_record(Path(value["path"]))
                                        for value in prepare["fingerprint"]["inputs"]]
    write_json(case.prepare, prepare)
    case.bundle = case.run / "metadata_repairs/group_evidence"
    return case


def database_snapshot(case):
    with sqlite3.connect(f"file:{case.database}?mode=ro", uri=True) as connection:
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {name: connection.execute(f'SELECT * FROM "{name}" ORDER BY 1').fetchall()
                for name in tables}


def all_source_bytes(case):
    return snapshot(case) | {path: path.read_bytes()
                            for path in (case.cluster, case.edges, case.database)}


def set_groups(case, groups):
    with sqlite3.connect(case.database) as connection:
        connection.execute("UPDATE stages SET value=? WHERE name='groups'",
                           (json.dumps(groups, sort_keys=True),))


def test_repair_changes_only_cluster_evidence_mtime_and_preserves_all_splits(grouped):
    before = database_snapshot(grouped)
    files = all_source_bytes(grouped)
    result = grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert isinstance(result, dict)
    after = database_snapshot(grouped)
    assert {k: v for k, v in after.items() if k != "stages"} == {
        k: v for k, v in before.items() if k != "stages"}
    old_stages, new_stages = dict(before["stages"]), dict(after["stages"])
    assert old_stages["entities"] == new_stages["entities"]
    expected = copy.deepcopy(grouped.old_groups)
    expected["protein_cluster_file"]["mtime_ns"] = grouped.cluster.stat().st_mtime_ns
    assert json.loads(new_stages["groups"]) == expected
    assert set(old_stages) == set(new_stages) == {"entities", "groups"}
    for path, value in files.items():
        if path != grouped.database:
            assert path.read_bytes() == value
    assert grouped.bundle.is_dir()
    assert (grouped.bundle / "backup_preparation.sqlite").read_bytes() == files[grouped.database]


def test_completed_repair_is_idempotent_and_does_not_rewrite_database(grouped):
    grouped.group_repair.repair(grouped.run, grouped.bundle)
    before = all_source_bytes(grouped)
    original_stat = grouped.database.stat()
    grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert all_source_bytes(grouped) == before
    assert grouped.database.stat().st_mtime_ns == original_stat.st_mtime_ns


@pytest.mark.parametrize("target", ["cluster", "edges"])
def test_same_size_same_mtime_content_change_is_rejected_without_writes(grouped, target):
    change_content_preserving_size_and_mtime(getattr(grouped, target))
    before = all_source_bytes(grouped)
    with pytest.raises(REJECTIONS):
        grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert all_source_bytes(grouped) == before


@pytest.mark.parametrize("which", ["pipeline", "output", "work"])
def test_repair_requires_all_three_writer_locks(grouped, which):
    path = {"pipeline": grouped.run / "pipeline.lock", "output": grouped.run / "data/.preparation.lock",
            "work": grouped.work / ".preparation.lock"}[which]
    before = all_source_bytes(grouped)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(REJECTIONS):
            grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert all_source_bytes(grouped) == before


@pytest.mark.parametrize("stage", ["labels", "pilot_selection", "train"])
def test_repair_refuses_downstream_controller_stages(grouped, stage):
    write_json(grouped.run / "state" / f"{stage}.json", {"status": "running"})
    before = all_source_bytes(grouped)
    with pytest.raises(REJECTIONS):
        grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert all_source_bytes(grouped) == before


@pytest.mark.parametrize("stage", ["protein_audit", "own_pairs", "uncertain_transfers"])
def test_repair_refuses_later_committed_database_stages(grouped, stage):
    with sqlite3.connect(grouped.database) as connection:
        connection.execute("INSERT INTO stages VALUES (?,?)", (stage, "{}"))
    before = all_source_bytes(grouped)
    with pytest.raises(REJECTIONS):
        grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert all_source_bytes(grouped) == before


def test_repair_refuses_missing_entity_stage(grouped):
    with sqlite3.connect(grouped.database) as connection:
        connection.execute("DELETE FROM stages WHERE name='entities'")
    before = all_source_bytes(grouped)
    with pytest.raises(REJECTIONS):
        grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert all_source_bytes(grouped) == before


@pytest.mark.parametrize("status", ["complete", "pending", "unknown"])
def test_repair_refuses_non_interrupted_preparation_status(grouped, status):
    record = read_json(grouped.prepare)
    record["status"] = status
    write_json(grouped.prepare, record)
    before = all_source_bytes(grouped)
    with pytest.raises(REJECTIONS):
        grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert all_source_bytes(grouped) == before


def test_repair_does_not_expand_scope_to_reaction_mtime(grouped):
    info = grouped.edges.stat()
    os.utime(grouped.edges, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
    before = all_source_bytes(grouped)
    with pytest.raises(REJECTIONS):
        grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert all_source_bytes(grouped) == before


@pytest.mark.parametrize("suffix", ["-journal", "-wal", "-shm"])
def test_repair_does_not_open_database_with_nonempty_recovery_sidecars(grouped, suffix):
    sidecar = write(Path(str(grouped.database) + suffix), "unresolved SQLite state")
    before = all_source_bytes(grouped)
    with pytest.raises(REJECTIONS):
        grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert all_source_bytes(grouped) == before
    assert sidecar.read_text() == "unresolved SQLite state"


@pytest.mark.parametrize("change", ["foreign_path", "missing_sha", "unexpected_key"])
def test_repair_refuses_unknown_or_unbound_cluster_evidence(grouped, change):
    groups = copy.deepcopy(grouped.old_groups)
    if change == "foreign_path":
        path = write(grouped.work / "other_clusters.tsv", grouped.cluster.read_text())
        groups["protein_cluster_file"] = hash_record(path)
    elif change == "missing_sha":
        del groups["protein_cluster_file"]["sha256"]
    else:
        groups["protein_cluster_file"]["other"] = "unexpected"
    set_groups(grouped, groups)
    before = all_source_bytes(grouped)
    with pytest.raises(REJECTIONS):
        grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert all_source_bytes(grouped) == before


def test_repair_refuses_a_controller_input_that_has_not_been_repaired(grouped):
    record = read_json(grouped.prepare)
    record["fingerprint"]["inputs"][0]["ctime_ns"] -= 1
    write_json(grouped.prepare, record)
    before = all_source_bytes(grouped)
    with pytest.raises(REJECTIONS):
        grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert all_source_bytes(grouped) == before


@pytest.mark.parametrize("name", ["builder", "implementation", "mmseqs", "representative_fasta",
                                  "raw_pairs", "reactions", "clustered_pairs"])
def test_refreshed_stat_record_cannot_hide_a_changed_historical_input(grouped, name):
    path = grouped.hashed_inputs[name]
    change_content_preserving_size_and_mtime(path)
    record = read_json(grouped.prepare)
    for value in record["fingerprint"]["inputs"]:
        if value["path"] == str(path):
            value.update(stat_record(path))
    write_json(grouped.prepare, record)
    before = all_source_bytes(grouped)
    with pytest.raises(REJECTIONS):
        grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert all_source_bytes(grouped) == before


def test_rollback_restores_original_checkpoint_after_update_fault(grouped, monkeypatch):
    before = database_snapshot(grouped)
    original_update = grouped.group_repair.update_groups
    called = False

    def fault_after_update(connection, previous, desired):
        nonlocal called
        called = True
        original_update(connection, previous, desired)
        raise OSError("simulated fault after UPDATE before COMMIT")

    monkeypatch.setattr(grouped.group_repair, "update_groups", fault_after_update)
    with pytest.raises(OSError, match="simulated fault"):
        grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert called
    assert database_snapshot(grouped) == before
    assert not (grouped.bundle / "applied.json").exists()


def test_failed_plan_publication_never_updates_checkpoint(grouped, monkeypatch):
    before = database_snapshot(grouped)
    original_publish = grouped.repair.publish_bytes
    called = False

    def fault_plan(path, contents):
        nonlocal called
        if Path(path).name == "plan.json":
            called = True
            raise OSError("simulated plan publication fault")
        return original_publish(path, contents)

    monkeypatch.setattr(grouped.repair, "publish_bytes", fault_plan)
    with pytest.raises(OSError, match="simulated plan"):
        grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert called
    assert database_snapshot(grouped) == before


def test_missing_receipt_after_commit_is_a_safe_noop_on_retry(grouped, monkeypatch):
    original_publish = grouped.repair.publish_bytes
    called = False

    def fault_receipt(path, contents):
        nonlocal called
        if Path(path).name == "applied.json":
            called = True
            raise OSError("simulated receipt publication fault")
        return original_publish(path, contents)

    with monkeypatch.context() as patch:
        patch.setattr(grouped.repair, "publish_bytes", fault_receipt)
        with pytest.raises(OSError, match="simulated receipt"):
            grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert called
    result = database_snapshot(grouped)
    actual = json.loads(dict(result["stages"])["groups"])
    assert actual["protein_cluster_file"] == hash_record(grouped.cluster)
    before_bytes = grouped.database.read_bytes()
    grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert grouped.database.read_bytes() == before_bytes


def test_interrupted_plan_is_not_replayed_against_mutated_database(grouped, monkeypatch):
    original_update = grouped.group_repair.update_groups

    def fault_after_update(connection, previous, desired):
        original_update(connection, previous, desired)
        raise OSError("stop before commit")

    with monkeypatch.context() as patch:
        patch.setattr(grouped.group_repair, "update_groups", fault_after_update)
        with pytest.raises(OSError, match="stop before commit"):
            grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert (grouped.bundle / "plan.json").is_file()
    groups = copy.deepcopy(grouped.old_groups)
    groups["protein_cluster_file"]["mtime_ns"] -= 1_000_000
    set_groups(grouped, groups)
    before = all_source_bytes(grouped)
    with pytest.raises(REJECTIONS):
        grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert all_source_bytes(grouped) == before


def test_foreign_bundle_data_is_not_overwritten(grouped):
    sentinel = write(grouped.bundle / "notes.txt", "this belongs to the user\n")
    before = all_source_bytes(grouped)
    with pytest.raises(REJECTIONS):
        grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert all_source_bytes(grouped) == before
    assert sentinel.read_text() == "this belongs to the user\n"


def test_inactive_zero_header_journal_is_backed_up_before_repair(grouped):
    journal = Path(str(grouped.database) + "-journal")
    journal.write_bytes(b"\0" * 512 + b"inactive previous rollback pages")
    previous_journal = journal.read_bytes()
    previous_database = grouped.database.read_bytes()
    grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert (grouped.bundle / "backup_preparation.sqlite").read_bytes() == previous_database
    assert (grouped.bundle / "backup_preparation.sqlite-journal").read_bytes() == previous_journal
    after = json.loads(dict(database_snapshot(grouped)["stages"])["groups"])
    assert after["protein_cluster_file"] == hash_record(grouped.cluster)


def test_database_triggers_cannot_turn_checkpoint_update_into_data_edits(grouped):
    with sqlite3.connect(grouped.database) as connection:
        connection.execute("""CREATE TRIGGER surprise AFTER UPDATE ON stages
                              BEGIN UPDATE proteins SET split='train'; END""")
    before = all_source_bytes(grouped)
    with pytest.raises(REJECTIONS):
        grouped.group_repair.repair(grouped.run, grouped.bundle)
    assert all_source_bytes(grouped) == before


@pytest.mark.parametrize("directory", ["state", "data", "work"])
def test_bundle_must_not_be_created_inside_pipeline_owned_directories(grouped, directory):
    bundle = grouped.run / directory / "misplaced_bundle"
    before = all_source_bytes(grouped)
    with pytest.raises(REJECTIONS):
        grouped.group_repair.repair(grouped.run, bundle)
    assert all_source_bytes(grouped) == before
    assert not bundle.exists()
