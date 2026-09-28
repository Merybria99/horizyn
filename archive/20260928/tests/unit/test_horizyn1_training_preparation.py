import csv
import argparse
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("horizyn1_training_test_impl", Path(__file__).resolve().parents[2] / "horizyn/datasets/horizyn1_training.py")
preparation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preparation)


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def pair_set(path: Path) -> set[tuple[str, str]]:
    return {(row["protein_id"], row["reaction_id"]) for row in rows(path)}


def identifiers_by_split(axis: str) -> dict[str, list[str]]:
    result = {split: [] for split in preparation.SPLITS}
    for index in range(10_000):
        identifier = f"{axis}_{index:05d}"
        split = preparation.split_for_group(identifier, axis, 20260909, 0.05, 0.05)
        if len(result[split]) < 2:
            result[split].append(identifier)
        if all(len(values) == 2 for values in result.values()):
            return result
    raise AssertionError("Could not make deterministic fixture")


@pytest.fixture
def fixture(tmp_path):
    proteins = identifiers_by_split("protein")
    reactions = identifiers_by_split("reaction")
    protein_ids = sum(proteins.values(), [])
    reaction_ids = sum(reactions.values(), [])
    fasta = write(tmp_path / "representatives.fasta", "".join(f">{identifier}\nMACDEFGHIK\n" for identifier in protein_ids))
    reaction_table = write(tmp_path / "reactions.tsv", "reaction_id\treaction_smiles\n" + "".join(f"{identifier}\tCCO>>CC=O\n" for identifier in reaction_ids))
    protein_clusters = write(tmp_path / "protein_clusters.tsv", "".join(f"{identifier}\t{identifier}\n" for identifier in protein_ids))
    reaction_clusters = write(tmp_path / "reaction_clusters.tsv", "".join(f"{identifier}\t{identifier}\n" for identifier in reaction_ids))
    hits = write(tmp_path / "hits.tsv", "query\ttarget\tfident\tqcov\ttcov\n")
    absent = (proteins["train"][1], reactions["train"][1])
    own = {(protein, reaction) for protein in protein_ids for reaction in reaction_ids} - {absent}
    raw = write(tmp_path / "raw.tsv", "reaction_id\tprotein_id\tsource\n" + "".join(f"{reaction}\t{protein}\town-source\n" for protein, reaction in sorted(own)) + f"{reactions['train'][0]}\tnot_a_representative\tmember-source\n" + f"{reactions['train'][0]}\t{proteins['train'][0]}\tsecond-provenance\n")
    collapsed = write(tmp_path / "collapsed.tsv", "reaction_id\tprotein_id\n" + "".join(f"{reaction}\t{protein}\n" for protein, reaction in sorted(own | {absent})))
    args = preparation.make_parser().parse_args([
        "--representative-fasta", str(fasta), "--raw-pairs", str(raw),
        "--reactions", str(reaction_table), "--output-dir", str(tmp_path / "prepared"),
        "--protein-clusters", str(protein_clusters), "--reaction-clusters", str(reaction_clusters),
        "--protein-cross-split-hits", str(hits), "--clustered-pairs", str(collapsed),
        "--panel-queries", "1", "--panel-protein-candidates", "1",
    ])
    return args, proteins, reactions, own, absent


def test_split_own_gold_source_preservation_and_uncertainty_exclusion(fixture):
    args, proteins, reactions, own, absent = fixture
    manifest = preparation.prepare(args)
    assert manifest["status"] == "complete"
    for split in preparation.SPLITS:
        expected = {(p, r) for p, r in own if p in proteins[split] and r in reactions[split]}
        assert pair_set(args.output_dir / f"{split}_pairs.csv") == expected
        assert set((args.output_dir / f"{split}_protein_ids.txt").read_text().splitlines()) == set(proteins[split])
        assert {row["reaction_id"] for row in rows(args.output_dir / f"{split}_rxns.csv")} == set(reactions[split])
    own_raw = rows(args.output_dir / "train_own_raw_associations.csv")
    assert len(own_raw) == 4  # Three unique train edges, two provenance rows for one.
    assert {row["source"] for row in own_raw} == {"own-source", "second-provenance"}
    assert all(row["protein_id"] in proteins["train"] and row["reaction_id"] in reactions["train"] for row in own_raw)
    assert pair_set(args.output_dir / "train_transferred_uncertain_pairs.csv") == {absent}
    assert absent not in pair_set(args.output_dir / "train_pairs.csv")
    assert manifest["stages"]["own_pairs"]["nonrepresentative_raw_rows_ignored"] == 1
    assert manifest["stages"]["groups"]["reaction_method"].startswith("precomputed_external")
    for split in preparation.SPLITS:
        pairs = rows(args.output_dir / f"{split}_pairs.csv")
        assert [row["pr_id"] for row in pairs] == [str(index) for index in range(len(pairs))]
        assert "rs_id" in rows(args.output_dir / f"{split}_rxns.csv")[0]


def test_full_catalog_test_gold_has_complete_adjacency_and_query_filters(fixture):
    args, proteins, reactions, own, _ = fixture
    manifest = preparation.prepare(args)
    expected = {(p, r) for p, r in own if p in proteins["test"] or r in reactions["test"]}
    assert pair_set(args.output_dir / "test_query_gold.csv") == expected
    assert set((args.output_dir / "test_enzyme_query_ids.txt").read_text().splitlines()) == set(proteins["test"])
    assert set((args.output_dir / "test_reaction_query_ids.txt").read_text().splitlines()) == set(reactions["test"])
    assert len((args.output_dir / "all_candidate_ids.txt").read_text().splitlines()) == 6
    assert len((args.output_dir / "all_reaction_ids.txt").read_text().splitlines()) == 6
    assert "Separate held-out query filters" in manifest["full_catalog_test"]["semantics"]


def test_frozen_panels_keep_positives_and_axis_specific_candidates(fixture):
    args, proteins, reactions, own, _ = fixture
    preparation.prepare(args)
    panels = json.loads((args.output_dir / "panels/panel_manifest.json").read_text())
    assert len(panels["panels"]) == 12
    for panel in panels["panels"].values():
        assert panel["query_count"] <= 1
        gold = pair_set(Path(panel["pairs"]))
        candidates = set(Path(panel["protein_candidate_ids"]).read_text().splitlines())
        reaction_candidates = set(Path(panel["reaction_candidate_ids"]).read_text().splitlines())
        assert {p for p, r in gold} <= candidates <= set(proteins[panel["protein_axis"]])
        assert reaction_candidates == set(reactions[panel["reaction_axis"]])
        assert all(p in proteins[panel["protein_axis"]] and r in reactions[panel["reaction_axis"]] for p, r in gold)
        queries = set(Path(panel["query_ids"]).read_text().splitlines())
        expected = {(p, r) for p, r in own if p in proteins[panel["protein_axis"]] and r in reactions[panel["reaction_axis"]] and (r if panel["direction"] == "reaction_to_enzyme" else p) in queries}
        assert gold == expected
    combined = panels["combined_panels"]["validation_both_cold"]
    assert combined["query_counts"] == {"enzyme_to_reaction": 1, "reaction_to_enzyme": 1}
    # Mandatory positives exceed requested candidate count; never drop truth.
    assert combined["protein_candidate_count"] == 2
    assert "incidental" in combined["warning"]


def test_seeded_assignment_is_deterministic_and_group_atomic():
    assert preparation.split_for_group("same", "protein", 1, 0.05, 0.05) == preparation.split_for_group("same", "protein", 1, 0.05, 0.05)
    assert preparation.make_parser().get_default("validation_fraction") == 0.05
    assert preparation.make_parser().get_default("test_fraction") == 0.05


def test_completed_resume_checks_source_and_output_hashes(fixture):
    args, _, _, _, _ = fixture
    first = preparation.prepare(args)
    with pytest.raises(ValueError, match="--resume"):
        preparation.prepare(args)
    args.resume = True
    assert preparation.prepare(args) == first
    output = args.output_dir / "train_pairs.csv"
    output.write_text(output.read_text() + "unknown,unknown\n")
    with pytest.raises(ValueError, match="Completed output changed"):
        preparation.prepare(args)


def test_changed_input_cannot_resume(fixture):
    args, _, _, _, _ = fixture
    preparation.prepare(args)
    args.resume = True
    args.raw_pairs.write_text(args.raw_pairs.read_text() + "R\tP\tnew\n")
    with pytest.raises(ValueError, match="Stale preparation"):
        preparation.prepare(args)


def test_interrupted_stage_resumes_without_reprocessing_completed_stages(fixture, monkeypatch):
    args, _, _, _, _ = fixture
    original = preparation.export_panels

    def interrupt(*_):
        raise RuntimeError("injected interruption")

    monkeypatch.setattr(preparation, "export_panels", interrupt)
    with pytest.raises(RuntimeError, match="injected"):
        preparation.prepare(args)
    assert not (args.output_dir / "preparation_manifest.json").exists()
    monkeypatch.setattr(preparation, "export_panels", original)
    args.resume = True
    assert preparation.prepare(args)["status"] == "complete"


def test_cross_split_search_threshold_violation_fails_closed(fixture):
    args, proteins, _, _, _ = fixture
    args.protein_cross_split_hits.write_text(f"{proteins['test'][0]}\t{proteins['train'][0]}\t0.5\t0.8\t0.8\n")
    with pytest.raises(ValueError, match="merge all offending similarity groups"):
        preparation.prepare(args)
    assert not (args.output_dir / "preparation_manifest.json").exists()
    assert not (args.output_dir / "train_own_raw_associations.csv").exists()
    assert not (args.output_dir / "train_pairs.csv").exists()
    report = json.loads((args.output_dir / "protein_similarity_audit.json").read_text())
    assert report["cross_split_violations"] == 1


@pytest.mark.parametrize("scores", ["50\t80\t80", "nan\t0.8\t0.8", "0.5\t-0.1\t0.8"])
def test_search_rejects_percentages_nonfinite_or_invalid_scores(fixture, scores):
    args, proteins, _, _, _ = fixture
    args.protein_cross_split_hits.write_text(f"{proteins['test'][0]}\t{proteins['train'][0]}\t{scores}\n")
    with pytest.raises(ValueError, match="fractions"):
        preparation.prepare(args)


def test_below_threshold_search_hit_is_not_prohibited(fixture):
    args, proteins, _, _, _ = fixture
    args.protein_cross_split_hits.write_text(f"{proteins['test'][0]}\t{proteins['train'][0]}\t0.9\t0.79\t0.99\n")
    assert preparation.prepare(args)["protein_similarity_audit"]["cross_split_violations"] == 0


def test_cluster_membership_requires_complete_inventory_and_self_roots(fixture):
    args, _, _, _, _ = fixture
    args.protein_clusters.write_text("not_a_root\tunknown\n")
    with pytest.raises(ValueError, match="cover each inventory ID"):
        preparation.prepare(args)


def test_duplicate_fasta_and_nonempty_output_are_rejected(fixture):
    args, proteins, _, _, _ = fixture
    args.representative_fasta.write_text(args.representative_fasta.read_text() + f">{proteins['train'][0]}\nAAAA\n")
    with pytest.raises(preparation.sqlite3.IntegrityError):
        preparation.prepare(args)


def test_nonrepresentative_members_cannot_fabricate_missing_reaction_gold(fixture):
    args, _, _, _, _ = fixture
    args.raw_pairs.write_text(args.raw_pairs.read_text() + "missing_rxn\tnot_a_representative\tmember\n")
    manifest = preparation.prepare(args)
    assert manifest["stages"]["own_pairs"]["nonrepresentative_raw_rows_ignored"] == 2


def test_own_association_with_missing_reaction_fails(fixture):
    args, proteins, _, _, _ = fixture
    args.raw_pairs.write_text(args.raw_pairs.read_text() + f"missing_rxn\t{proteins['train'][0]}\town\n")
    with pytest.raises(ValueError, match="missing reaction"):
        preparation.prepare(args)


def test_optional_uncertainty_exclusions_are_explicitly_absent(fixture):
    args, _, _, _, _ = fixture
    args.clustered_pairs = None
    manifest = preparation.prepare(args)
    assert not manifest["training_transfer_uncertainty_exclusions"]["provided"]
    assert pair_set(args.output_dir / "train_transferred_uncertain_pairs.csv") == set()


def test_sparse_prefix_join_matches_bruteforce_including_reverse():
    rng = random.Random(182)
    tokens = [(bit, sign) for bit in range(30) for sign in (-1, 1)]
    descriptors = {str(index): frozenset(rng.sample(tokens, rng.randrange(1, 25))) for index in range(60)}
    descriptors["copy"] = descriptors["0"]
    descriptors["reverse"] = frozenset((bit, -sign) for bit, sign in descriptors["1"])
    descriptors["empty"] = frozenset()
    for threshold in (0.3, 0.5, 0.8, 1.0):
        observed = {frozenset((a, b)) for a, b, _ in preparation.similarity_edges(descriptors, threshold)}
        expected = set()
        for first, bits in descriptors.items():
            for second, other in descriptors.items():
                if first >= second or not bits or not other:
                    continue
                reverse = frozenset((bit, -sign) for bit, sign in other)
                best = max(len(bits & orientation) / len(bits | orientation) for orientation in (other, reverse))
                if best >= threshold:
                    expected.add(frozenset((first, second)))
        assert observed == expected


def test_reaction_aliases_maps_reverse_and_spectator_cancellation():
    pytest.importorskip("rdkit")
    exact, changed = preparation.reaction_descriptor("[CH3:1][CH2:2]O.O>>CC=O.O")
    reverse, reverse_changed = preparation.reaction_descriptor("O.CC=O>>O.OCC")
    no_spectator, no_spectator_changed = preparation.reaction_descriptor("CCO>>CC=O")
    assert exact == reverse
    assert changed == frozenset((bit, -sign) for bit, sign in reverse_changed)
    assert changed == no_spectator_changed
    assert exact != no_spectator
    assert preparation.reaction_descriptor("CCO>>CCO")[1] == frozenset()
    with pytest.raises(ValueError, match="canonicalize"):
        preparation.reaction_descriptor("not-valid-smiles>>CCO")


def test_group_merge_has_deterministic_minimum_root():
    union = preparation.UnionFind(["c", "b", "a"])
    union.union("c", "b")
    union.union("c", "a")
    assert {union.find(item) for item in ("a", "b", "c")} == {"a"}


def test_mmseqs_path_invokes_clustering_and_explicit_sensitive_audit(fixture, monkeypatch):
    args, _, _, _, _ = fixture
    cluster_text = args.protein_clusters.read_text()
    args.protein_clusters = None
    args.protein_cross_split_hits = None
    args.mmseqs = Path("true")
    args.work_dir = args.output_dir.parent / "mmseqs_scratch"
    args.log_dir = args.output_dir.parent / "shared_mmseqs_logs"
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[1] == "easy-linclust":
            Path(command[3] + "_cluster.tsv").write_text(cluster_text)
        elif command[1] == "easy-search":
            Path(command[4]).write_text("")
        return SimpleNamespace(stdout="test-mmseqs-version")

    monkeypatch.setattr(preparation.subprocess, "run", fake_run)
    manifest = preparation.prepare(args)
    assert manifest["protein_similarity_audit"]["method"] == "sensitive_mmseqs_heldout_against_all_search"
    cluster = next(command for command in commands if command[1] == "easy-linclust")
    search = next(command for command in commands if command[1] == "easy-search")
    assert cluster[cluster.index("--min-seq-id") + 1] == "0.5"
    assert cluster[cluster.index("-c") + 1] == "0.8"
    assert search[search.index("--format-output") + 1] == "query,target,fident,qcov,tcov"
    assert search[search.index("--alignment-mode") + 1] == "3"
    assert search[search.index("--cov-mode") + 1] == "0"
    assert search[search.index("-s") + 1] == "7.5"
    assert len(list(preparation.fasta_records(args.work_dir / "heldout.fasta"))) == 4
    assert (args.log_dir / "mmseqs.log").is_file()
    assert str(args.work_dir) in cluster[3]
    assert str(args.work_dir) in search[4]
    assert not (args.output_dir / "work").exists()


def test_computed_reaction_groups_keep_aliases_and_changed_transformations(fixture):
    pytest.importorskip("rdkit")
    args, _, _, _, _ = fixture
    args.reaction_clusters = None
    args.reactions.write_text("reaction_id\treaction_smiles\na\tCCO>>CC=O\nb\tCC=O>>OCC\nc\tCCCl>>CCO\nd\tCCO.O>>CC=O.O\n")
    args.output_dir.mkdir()
    database = preparation.connect(args.output_dir / "direct_test.sqlite")
    try:
        preparation.ingest_entities(database, args)
        preparation.group_entities(database, args)
        groups = dict(database.execute("SELECT id,group_id FROM reactions"))
        assert groups["a"] == groups["b"] == groups["d"]
        assert groups["a"] != groups["c"]
        assert "signed_morgan" in preparation.stage_value(database, "groups")["reaction_method"]
    finally:
        database.close()


def test_concurrent_preparation_is_rejected(fixture):
    args, _, _, _, _ = fixture
    args.output_dir.mkdir()
    with (args.output_dir / ".preparation.lock").open("a") as lock:
        preparation.fcntl.flock(lock.fileno(), preparation.fcntl.LOCK_EX | preparation.fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another preparation"):
            preparation.prepare(args)


def test_nonempty_unowned_output_is_rejected(fixture):
    args, _, _, _, _ = fixture
    args.output_dir.mkdir()
    sentinel = write(args.output_dir / "user_data.txt", "preserve me")
    with pytest.raises(ValueError, match="not empty"):
        preparation.prepare(args)
    assert sentinel.read_text() == "preserve me"


def test_filesystem_resolution_handles_nested_network_and_escaped_mounts(tmp_path):
    mountinfo = "1 0 0:1 / / rw - ext4 /dev/root rw\n2 1 0:2 / /shared rw - nfs4 host:/data rw\n3 1 0:3 / /local\\040disk rw - xfs /dev/local rw\n"
    assert preparation.filesystem_for_path(Path("/shared/run/work"), mountinfo)["filesystem"] == "nfs4"
    assert preparation.filesystem_for_path(Path("/local disk/job/work"), mountinfo) == {"filesystem": "xfs", "mount_point": "/local disk"}
    assert preparation.filesystem_for_path(Path("/shared-looking/local"), mountinfo)["filesystem"] == "ext4"


@pytest.mark.parametrize("filesystem,journal", [("nfs", "delete"), ("nfs4", "delete"), ("cifs", "delete"), ("fuse.sshfs", "delete"), ("ext4", "wal"), ("xfs", "wal")])
def test_storage_mode_exclusive_lock_durability_and_cache(tmp_path, monkeypatch, filesystem, journal):
    monkeypatch.setattr(preparation, "filesystem_for_path", lambda path: {"filesystem": filesystem, "mount_point": "/"})
    database = preparation.connect(tmp_path / "scratch.sqlite", cache_mib=64)
    try:
        assert database.execute("PRAGMA journal_mode").fetchone()[0] == journal
        assert database.execute("PRAGMA locking_mode").fetchone()[0] == "exclusive"
        assert database.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
        assert database.execute("PRAGMA cache_size").fetchone()[0] == -64 * 1024
    finally:
        database.close()


def test_explicit_work_and_shared_log_paths_are_bound_and_not_published(fixture):
    args, _, _, _, _ = fixture
    args.work_dir = args.output_dir.parent / "separate_work"
    args.log_dir = args.output_dir.parent / "shared_logs"
    manifest = preparation.prepare(args)
    assert (args.work_dir / "preparation.sqlite").is_file()
    assert (args.work_dir / "preparation_owner.json").is_file()
    assert not (args.output_dir / "work").exists()
    assert manifest["work_dir"] == str(args.work_dir)
    assert manifest["log_dir"] == str(args.log_dir)
    assert all(not path.startswith(str(args.work_dir) + "/") for path in manifest["output_signatures"])
    args.resume = True
    args.work_dir = args.output_dir.parent / "changed_work"
    with pytest.raises(ValueError, match="work-dir binding"):
        preparation.prepare(args)
    assert not args.work_dir.exists()


@pytest.mark.parametrize("missing", ["directory", "database", "owner"])
def test_missing_completed_scratch_cannot_silently_resume(fixture, missing):
    args, _, _, _, _ = fixture
    preparation.prepare(args)
    if missing == "directory":
        args.work_dir.rename(args.work_dir.with_name("preserved_work"))
    elif missing == "database":
        (args.work_dir / "preparation.sqlite").rename(args.work_dir / "preserved.sqlite")
    else:
        (args.work_dir / "preparation_owner.json").rename(args.work_dir / "preserved_owner.json")
    args.resume = True
    with pytest.raises(ValueError, match="scratch is missing or incomplete"):
        preparation.prepare(args)


def test_work_directory_cannot_be_shared_between_runs_or_relative(fixture):
    args, _, _, _, _ = fixture
    args.work_dir = Path("relative_work")
    with pytest.raises(ValueError, match="absolute"):
        preparation.prepare(args)
    args.work_dir = args.output_dir.parent / "owned_work"
    preparation.prepare(args)
    args.output_dir = args.output_dir.with_name("different_output")
    args.resume = False
    with pytest.raises(ValueError, match="Work directory is not empty"):
        preparation.prepare(args)


def test_network_rollback_and_local_wal_produce_byte_identical_gold(fixture, monkeypatch):
    args, _, _, _, _ = fixture
    first = preparation.prepare(args)
    original_output = args.output_dir
    second_args = argparse.Namespace(**vars(args))
    second_args.output_dir = original_output.with_name("network_prepared")
    second_args.work_dir = second_args.output_dir / "work"
    second_args.log_dir = second_args.output_dir / "logs"
    monkeypatch.setattr(preparation, "filesystem_for_path", lambda path: {"filesystem": "nfs4", "mount_point": "/"})
    second = preparation.prepare(second_args)
    assert second["storage_policy"]["journal_mode"] == "delete"
    assert first["counts"] == second["counts"]
    for file in original_output.rglob("*"):
        if file.is_file() and file.suffix in {".csv", ".tsv", ".txt"}:
            assert file.read_bytes() == (second_args.output_dir / file.relative_to(original_output)).read_bytes()


def test_small_batches_preserve_raw_order_duplicates_and_never_serialize_rows(fixture, monkeypatch):
    args, _, _, _, _ = fixture
    monkeypatch.setattr(preparation, "BATCH_SIZE", 2)
    original_dumps = preparation.json.dumps

    def reject_raw_serialization(value, *positional, **kwargs):
        if isinstance(value, dict) and {"protein_id", "reaction_id", "source"} <= value.keys():
            raise AssertionError("Raw CSV row serialized through JSON")
        return original_dumps(value, *positional, **kwargs)

    monkeypatch.setattr(preparation.json, "dumps", reject_raw_serialization)
    preparation.prepare(args)
    original = list(csv.DictReader(args.raw_pairs.open(), delimiter="\t"))
    train_ids = set((args.output_dir / "train_protein_ids.txt").read_text().splitlines())
    train_reactions = set((args.output_dir / "train_reaction_ids.txt").read_text().splitlines())
    expected = [row for row in original if row["protein_id"] in train_ids and row["reaction_id"] in train_reactions]
    assert rows(args.output_dir / "train_own_raw_associations.csv") == expected


def test_bulk_lookup_retains_order_duplicates_and_unknown_ids(fixture, monkeypatch):
    args, proteins, _, _, _ = fixture
    preparation.prepare(args)
    database = preparation.connect(args.work_dir / "preparation.sqlite")
    statements = []
    database.set_trace_callback(statements.append)
    try:
        ids = [proteins["test"][0], "missing", proteins["train"][0], proteins["test"][0]]
        assert preparation.lookup_protein_splits(database, ids) == ["test", None, "train", "test"]
        assert sum(statement.startswith("SELECT p.split") for statement in statements) == 1
        assert not any("SELECT split FROM proteins WHERE id=" in statement for statement in statements)
        monkeypatch.setattr(preparation, "BATCH_SIZE", 1)
        with pytest.raises(ValueError, match="bounded batch"):
            preparation.lookup_protein_splits(database, ids)
    finally:
        database.close()


def test_bulk_audit_preserves_threshold_boundary_hits_across_batches(fixture, monkeypatch):
    args, proteins, _, _, _ = fixture
    monkeypatch.setattr(preparation, "BATCH_SIZE", 2)
    first, second = proteins["test"][0], proteins["train"][0]
    args.protein_cross_split_hits.write_text(f"{first}\t{second}\t0.49\t1\t1\n{first}\t{first}\t1\t1\t1\n{first}\t{second}\t0.5\t0.8\t0.8\n{first}\t{second}\t0.9\t0.79\t1\n")
    with pytest.raises(ValueError, match="Found 1 prohibited"):
        preparation.prepare(args)
    report = json.loads((args.output_dir / "protein_similarity_audit.json").read_text())
    assert report["rows_checked"] == 4
    assert report["cross_split_violations"] == 1
    assert not (args.output_dir / "train_own_raw_associations.csv").exists()


def test_progress_messages_have_explicit_timestamp(capsys):
    preparation.progress("bounded progress")
    message = capsys.readouterr().err
    assert preparation.re.match(r"\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}\] bounded progress\n", message)


def test_heldout_fasta_bulk_batches_bound_payload_and_preserve_order(fixture, monkeypatch):
    args, proteins, _, _, _ = fixture
    preparation.prepare(args)
    database = preparation.connect(args.work_dir / "preparation.sqlite")
    calls = []
    original = preparation.lookup_protein_splits

    def lookup(connection, ids):
        calls.append(list(ids))
        return original(connection, ids)

    monkeypatch.setattr(preparation, "lookup_protein_splits", lookup)
    monkeypatch.setattr(preparation, "FASTA_LOOKUP_BATCH_RESIDUES", 15)
    destination = args.work_dir / "test_heldout.fasta"
    try:
        count = preparation.write_heldout_queries(database, args.representative_fasta, destination)
    finally:
        database.close()
    assert count == 4
    assert all(len(call) == 1 for call in calls)  # Each fixture sequence has 10 residues.
    expected = [(identifier, sequence) for identifier, sequence in preparation.fasta_records(args.representative_fasta) if identifier not in proteins["train"]]
    assert list(preparation.fasta_records(destination)) == expected


class InitializationInterrupted(RuntimeError):
    """Simulated process interruption, scoped to tiny fixture artifacts."""


def initialization_input_snapshot(args):
    return {path: path.read_bytes() for name in (
        "representative_fasta", "raw_pairs", "reactions", "protein_clusters",
        "reaction_clusters", "protein_cross_split_hits", "clustered_pairs",
    ) if (path := getattr(args, name)) is not None}


def configure_initialization_work(args, external):
    if external:
        args.work_dir = args.output_dir.parent / "initialization_external_work"


def inject_initialization_interruption(patch, point):
    """Close mocked connections just as process death releases real handles."""
    original_write = preparation.write_json
    original_connect = preparation.connect

    def interrupted_write(path, value):
        is_owner = path.name == "preparation_owner.json"
        is_state = path.name == "preparation_state.json"
        relevant = (is_owner and point.endswith("owner")) or (is_state and point.endswith("state"))
        if relevant:
            if point.startswith("after_"):
                original_write(path, value)
            elif point.startswith("partial_"):
                path.with_name(path.name + ".partial").write_text('{"interrupted":', encoding="utf-8")
            raise InitializationInterrupted(point)
        return original_write(path, value)

    def interrupted_connect(*args, **kwargs):
        if point == "after_schema":
            database = original_connect(*args, **kwargs)
            database.close()
        raise InitializationInterrupted(point)

    def interrupted_ingest(*args, **kwargs):
        raise InitializationInterrupted(point)

    if point.endswith(("owner", "state")):
        patch.setattr(preparation, "write_json", interrupted_write)
    elif point in {"before_connect", "after_schema"}:
        patch.setattr(preparation, "connect", interrupted_connect)
    elif point == "before_ingest":
        patch.setattr(preparation, "ingest_entities", interrupted_ingest)
    else:
        raise AssertionError(f"Unknown interruption point: {point}")


@pytest.mark.parametrize("external", [False, True], ids=["nested_work", "external_work"])
@pytest.mark.parametrize("network", [False, True], ids=["local_wal", "network_delete"])
@pytest.mark.parametrize("point", [
    "before_owner", "after_owner", "partial_owner", "before_state",
    "partial_state", "after_state", "before_connect", "after_schema", "before_ingest",
])
def test_initialization_interruption_retries_without_changing_gold(fixture, monkeypatch, external, network, point):
    args, proteins, reactions, own, _ = fixture
    configure_initialization_work(args, external)
    if network:
        monkeypatch.setattr(preparation, "filesystem_for_path", lambda path: {"filesystem": "nfs4", "mount_point": "/"})
    inputs = initialization_input_snapshot(args)
    with monkeypatch.context() as patch:
        inject_initialization_interruption(patch, point)
        with pytest.raises(InitializationInterrupted, match=point):
            preparation.prepare(args)
    marker = preparation.initialization_directory(args)
    assert marker.parent == args.output_dir
    assert marker.name.startswith(".preparation_initializing_")
    assert marker.is_dir() == (point != "before_ingest")
    if marker.exists():
        assert not any(marker.iterdir())
    assert not (args.output_dir / "train_pairs.csv").exists()
    assert initialization_input_snapshot(args) == inputs

    args.resume = True
    manifest = preparation.prepare(args)
    assert manifest["status"] == "complete"
    assert manifest["storage_policy"]["journal_mode"] == ("delete" if network else "wal")
    assert not marker.exists()
    assert not (args.output_dir / "preparation_state.json.partial").exists()
    assert not (args.work_dir / "preparation_owner.json.partial").exists()
    assert initialization_input_snapshot(args) == inputs
    for split in preparation.SPLITS:
        assert pair_set(args.output_dir / f"{split}_pairs.csv") == {
            (protein, reaction) for protein, reaction in own
            if protein in proteins[split] and reaction in reactions[split]
        }


@pytest.mark.parametrize("external", [False, True], ids=["nested_work", "external_work"])
@pytest.mark.parametrize("foreign_location", ["output", "work", "marker"])
def test_pending_initialization_preserves_and_rejects_foreign_artifacts(fixture, monkeypatch, external, foreign_location):
    args, _, _, _, _ = fixture
    configure_initialization_work(args, external)
    inputs = initialization_input_snapshot(args)
    with monkeypatch.context() as patch:
        inject_initialization_interruption(patch, "after_owner")
        with pytest.raises(InitializationInterrupted):
            preparation.prepare(args)
    directory = {"output": args.output_dir, "work": args.work_dir,
                 "marker": preparation.initialization_directory(args)}[foreign_location]
    sentinel = write(directory / "foreign_user_data.bin", "preserve this unrelated file\n")
    saved_owner = (args.work_dir / "preparation_owner.json").read_bytes()
    args.resume = True
    with pytest.raises(ValueError):
        preparation.prepare(args)
    assert sentinel.read_text() == "preserve this unrelated file\n"
    assert (args.work_dir / "preparation_owner.json").read_bytes() == saved_owner
    assert initialization_input_snapshot(args) == inputs
    assert not (args.output_dir / "train_pairs.csv").exists()


@pytest.mark.parametrize("external", [False, True], ids=["nested_work", "external_work"])
@pytest.mark.parametrize("table", ["proteins", "reactions", "pairs", "stages"])
def test_pending_initialization_rejects_any_populated_database(fixture, monkeypatch, external, table):
    args, _, _, _, _ = fixture
    configure_initialization_work(args, external)
    inputs = initialization_input_snapshot(args)
    with monkeypatch.context() as patch:
        inject_initialization_interruption(patch, "after_schema")
        with pytest.raises(InitializationInterrupted):
            preparation.prepare(args)
    database = preparation.sqlite3.connect(args.work_dir / "preparation.sqlite")
    insert = {
        "proteins": ("INSERT INTO proteins(id,sequence_sha256,length,sample_key) VALUES (?,?,?,?)", ("foreign_protein", "0" * 64, 10, "0" * 16)),
        "reactions": ("INSERT INTO reactions(id,smiles,sample_key) VALUES (?,?,?)", ("foreign_reaction", "CC>>CO", "0" * 16)),
        "pairs": ("INSERT INTO pairs VALUES (?,?)", ("foreign_protein", "foreign_reaction")),
        "stages": ("INSERT INTO stages VALUES (?,?)", ("foreign_stage", "{}")),
    }[table]
    try:
        database.execute(*insert)
        database.commit()
    finally:
        database.close()
    saved_state = (args.output_dir / "preparation_state.json").read_bytes()
    args.resume = True
    with pytest.raises(ValueError):
        preparation.prepare(args)
    database = preparation.sqlite3.connect(args.work_dir / "preparation.sqlite")
    try:
        assert database.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 1
    finally:
        database.close()
    assert (args.output_dir / "preparation_state.json").read_bytes() == saved_state
    assert initialization_input_snapshot(args) == inputs
    assert not (args.output_dir / "train_pairs.csv").exists()


@pytest.mark.parametrize("external", [False, True], ids=["nested_work", "external_work"])
def test_pending_initialization_rejects_changed_committed_arguments(fixture, monkeypatch, external):
    args, _, _, _, _ = fixture
    configure_initialization_work(args, external)
    with monkeypatch.context() as patch:
        inject_initialization_interruption(patch, "before_connect")
        with pytest.raises(InitializationInterrupted):
            preparation.prepare(args)
    saved_state = (args.output_dir / "preparation_state.json").read_bytes()
    args.resume = True
    args.seed += 1
    with pytest.raises(ValueError):
        preparation.prepare(args)
    assert (args.output_dir / "preparation_state.json").read_bytes() == saved_state
    assert not (args.work_dir / "preparation.sqlite").exists()


@pytest.mark.parametrize("external", [False, True], ids=["nested_work", "external_work"])
def test_ready_initialization_missing_database_is_not_reclassified_as_pending(fixture, monkeypatch, external):
    args, _, _, _, _ = fixture
    configure_initialization_work(args, external)
    with monkeypatch.context() as patch:
        inject_initialization_interruption(patch, "before_ingest")
        with pytest.raises(InitializationInterrupted):
            preparation.prepare(args)
    assert not preparation.initialization_directory(args).exists()
    database_path = args.work_dir / "preparation.sqlite"
    preserved_path = args.work_dir / "preserved_ready.sqlite"
    database_path.rename(preserved_path)
    saved = preserved_path.read_bytes()
    saved_state = (args.output_dir / "preparation_state.json").read_bytes()
    args.resume = True
    with pytest.raises(ValueError):
        preparation.prepare(args)
    assert not database_path.exists()
    assert preserved_path.read_bytes() == saved
    assert (args.output_dir / "preparation_state.json").read_bytes() == saved_state
    assert not preparation.initialization_directory(args).exists()


@pytest.mark.parametrize("journal", ["delete", "wal"])
@pytest.mark.parametrize("point", ["after_schema", "partial_schema"])
def test_initialization_recovers_after_real_process_death_with_unclosed_database(fixture, monkeypatch, journal, point):
    args, proteins, reactions, own, _ = fixture
    filesystem = "nfs4" if journal == "delete" else "ext4"
    monkeypatch.setattr(preparation, "filesystem_for_path", lambda path: {"filesystem": filesystem, "mount_point": "/"})
    inputs = initialization_input_snapshot(args)
    original_connect = preparation.connect

    def run_child():
        def die_during_connect(path, cache_mib=512):
            if point == "after_schema":
                database = original_connect(path, cache_mib)
            else:
                database = preparation.sqlite3.connect(path)
                database.execute("PRAGMA locking_mode=EXCLUSIVE")
                database.execute(f"PRAGMA journal_mode={journal}")
                database.execute("PRAGMA synchronous=FULL")
                database.execute("PRAGMA cache_size=1")
                database.execute("PRAGMA cache_spill=ON")
                database.execute("CREATE TABLE stages(name TEXT PRIMARY KEY,value TEXT NOT NULL)")
                database.execute("BEGIN IMMEDIATE")
                database.execute("CREATE TABLE proteins(id TEXT PRIMARY KEY,sequence_sha256 TEXT NOT NULL,length INTEGER NOT NULL,group_id TEXT,split TEXT,sample_key TEXT NOT NULL) WITHOUT ROWID")
                database.execute("CREATE TABLE reactions(id TEXT PRIMARY KEY,smiles TEXT NOT NULL,group_id TEXT,split TEXT,sample_key TEXT NOT NULL) WITHOUT ROWID")
            # Keep the connection alive until _exit: no close, Python finally,
            # connection destructor, rollback, or clean process shutdown runs.
            assert database is not None
            os._exit(73)

        preparation.connect = die_during_connect
        preparation.prepare(args)
        os._exit(99)  # The injected death point must have been reached.

    process = multiprocessing.get_context("fork").Process(target=run_child)
    process.start()
    try:
        process.join(timeout=20)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
            pytest.fail("Tiny initialization child did not reach its injected exit")
        assert process.exitcode == 73
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        process.close()

    marker = preparation.initialization_directory(args)
    database_path = preparation.work_directory(args) / "preparation.sqlite"
    assert marker.is_dir()
    assert (args.output_dir / "preparation_state.json").is_file()
    assert database_path.is_file()
    if point == "partial_schema" or journal == "wal":
        suffix = "-journal" if journal == "delete" else "-wal"
        sidecar = database_path.with_name(database_path.name + suffix)
        assert sidecar.stat().st_size > 0
        if point == "partial_schema" and journal == "delete":
            # Tiny-cache spill must leave a substantive rollback journal with
            # a synced (nonzero) header, not merely an unused empty sidecar.
            assert sidecar.stat().st_size > 512
            with sidecar.open("rb") as handle:
                assert handle.read(8) != b"\x00" * 8
    assert not (args.output_dir / "train_pairs.csv").exists()
    assert initialization_input_snapshot(args) == inputs

    args.resume = True
    manifest = preparation.prepare(args)
    assert manifest["status"] == "complete"
    assert manifest["storage_policy"]["journal_mode"] == journal
    assert not marker.exists()
    assert initialization_input_snapshot(args) == inputs
    for split in preparation.SPLITS:
        assert pair_set(args.output_dir / f"{split}_pairs.csv") == {
            (protein, reaction) for protein, reaction in own
            if protein in proteins[split] and reaction in reactions[split]
        }


@pytest.mark.parametrize("external", [False, True], ids=["nested_work", "external_work"])
def test_pending_initialization_can_retry_a_zero_byte_database_without_sidecars(fixture, monkeypatch, external):
    args, _, _, _, _ = fixture
    configure_initialization_work(args, external)
    inputs = initialization_input_snapshot(args)
    with monkeypatch.context() as patch:
        inject_initialization_interruption(patch, "before_connect")
        with pytest.raises(InitializationInterrupted):
            preparation.prepare(args)
    database_path = args.work_dir / "preparation.sqlite"
    database_path.touch()
    assert database_path.stat().st_size == 0
    assert preparation.initialization_directory(args).is_dir()
    args.resume = True
    assert preparation.prepare(args)["status"] == "complete"
    assert not preparation.initialization_directory(args).exists()
    assert initialization_input_snapshot(args) == inputs


@pytest.mark.parametrize("database_state", ["missing", "zero_byte"])
@pytest.mark.parametrize("suffix", ["-journal", "-wal", "-shm"])
def test_pending_missing_or_zero_database_with_sidecars_fails_closed(fixture, monkeypatch, database_state, suffix):
    args, _, _, _, _ = fixture
    inputs = initialization_input_snapshot(args)
    with monkeypatch.context() as patch:
        inject_initialization_interruption(patch, "before_connect")
        with pytest.raises(InitializationInterrupted):
            preparation.prepare(args)
    database_path = args.work_dir / "preparation.sqlite"
    if database_state == "zero_byte":
        database_path.touch()
    sidecar = write(database_path.with_name(database_path.name + suffix), "untrusted surviving sidecar; preserve\n")
    saved_state = (args.output_dir / "preparation_state.json").read_bytes()
    args.resume = True
    with pytest.raises(ValueError, match="sidecars"):
        preparation.prepare(args)
    assert database_path.exists() == (database_state == "zero_byte")
    if database_path.exists():
        assert database_path.stat().st_size == 0
    assert sidecar.read_text() == "untrusted surviving sidecar; preserve\n"
    assert (args.output_dir / "preparation_state.json").read_bytes() == saved_state
    assert preparation.initialization_directory(args).is_dir()
    assert initialization_input_snapshot(args) == inputs
    assert not (args.output_dir / "train_pairs.csv").exists()
