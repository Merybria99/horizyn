"""Disk-backed, representative-own CIRCE-v2 split preparation.

Protein groups are MMseqs similarity groups, not the earlier source-collapse
clusters. Reaction groups combine undirected exact chemistry with a *heuristic*
signed Morgan-difference similarity. Neither grouping certifies biochemical
equivalence or a mathematical sequence-similarity cutoff.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Iterable, Iterator


SCHEMA = "horizyn1_circe_v2_training_preparation_v1"
SPLITS = ("train", "validation", "test")
BATCH_SIZE = 50_000
FASTA_LOOKUP_BATCH_RESIDUES = 8_000_000


def progress(message: str) -> None:
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", file=sys.stderr, flush=True)


NETWORK_FILESYSTEMS = frozenset({"nfs", "nfs4", "cifs", "smbfs", "sshfs", "9p", "ceph", "glusterfs", "afs", "lustre", "gpfs", "panfs", "beegfs"})


def filesystem_for_path(path: Path, mountinfo: str | None = None) -> dict[str, str]:
    """Resolve the most specific Linux mount, including escaped bind paths."""
    path = path.resolve()
    if mountinfo is None:
        try:
            mountinfo = Path("/proc/self/mountinfo").read_text()
        except OSError as error:
            raise ValueError("Cannot determine safe SQLite journal policy without /proc/self/mountinfo") from error
    candidates = []
    for line in mountinfo.splitlines():
        before, separator, after = line.partition(" - ")
        fields, details = before.split(), after.split()
        if not separator or len(fields) < 5 or not details:
            continue
        mount = Path(re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), fields[4]))
        if path == mount or mount in path.parents:
            candidates.append((len(mount.parts), {"mount_point": str(mount), "filesystem": details[0]}))
    if not candidates:
        raise ValueError(f"Cannot identify filesystem for work directory: {path}")
    return max(candidates, key=lambda item: item[0])[1]


def work_storage_policy(path: Path) -> dict[str, str]:
    filesystem = filesystem_for_path(path)
    kind = filesystem["filesystem"]
    network = kind in NETWORK_FILESYSTEMS or kind.startswith("fuse.")
    return {**filesystem, "journal_mode": "delete" if network else "wal", "locking_mode": "exclusive"}


def work_directory(args: argparse.Namespace) -> Path:
    return args.work_dir if args.work_dir is not None else args.output_dir / "work"


def log_directory(args: argparse.Namespace) -> Path:
    return args.log_dir if args.log_dir is not None else args.output_dir / "logs"


def work_owner(args: argparse.Namespace) -> dict:
    return {"schema_version": SCHEMA, "output_dir": str(args.output_dir), "work_dir": str(work_directory(args))}


def initialization_directory(args: argparse.Namespace) -> Path:
    # mkdir is atomic: unlike a first JSON write, a killed writer cannot leave
    # a truncated ownership claim. The name binds the two locked directories.
    identity = hashlib.sha256(json.dumps(work_owner(args), sort_keys=True).encode()).hexdigest()
    return args.output_dir / f".preparation_initializing_{identity}"


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def check_initialization_contents(args: argparse.Namespace, *, initializing: bool) -> None:
    """Only setup artifacts are allowed before the durable ready transition."""
    allowed_output = {args.output_dir / ".preparation.lock"}
    work = work_directory(args)
    if work.parent == args.output_dir:
        allowed_output.add(work)
    if initializing:
        allowed_output.update({initialization_directory(args), args.output_dir / "preparation_state.json",
                               args.output_dir / "preparation_state.json.partial"})
    if any(path not in allowed_output for path in args.output_dir.iterdir()):
        raise ValueError("Output directory is not empty and has no matching initialization state")
    allowed_work = {".preparation.lock"}
    if initializing:
        allowed_work.update({"preparation_owner.json", "preparation_owner.json.partial"})
        if (args.output_dir / "preparation_state.json").is_file():
            allowed_work.update({"preparation.sqlite", "preparation.sqlite-journal",
                                 "preparation.sqlite-wal", "preparation.sqlite-shm"})
    if any(path.name not in allowed_work or not path.is_file() or path.is_symlink() for path in work.iterdir()):
        raise ValueError(f"Work directory is not empty or contains non-initialization artifacts: {work}")
    for name in ("preparation_state.json", "preparation_state.json.partial"):
        path = args.output_dir / name
        if path.exists() and (not path.is_file() or path.is_symlink()):
            raise ValueError(f"Invalid initialization state artifact: {path}")


@contextmanager
def owned_work_directory(args: argparse.Namespace):
    """Lock scratch and distinguish unfinished setup from lost committed data."""
    work = work_directory(args)
    owner_path = work / "preparation_owner.json"
    state_path = args.output_dir / "preparation_state.json"
    state_exists = state_path.exists()
    initializing_path = initialization_directory(args)
    initializing = initializing_path.exists()
    owner = work_owner(args)
    if state_exists:
        previous = json.loads(state_path.read_text())
        if previous.get("work_dir") != str(work):
            raise ValueError("Stale preparation work-dir binding; use a NEW output/work directory, not live database migration")
        if not work.is_dir():
            raise ValueError("Bound preparation scratch is missing or incomplete. Restore the entire stopped scratch directory from a consistent backup, or rebuild with a NEW output/work directory; partial scratch will not be silently reused.")
    work.mkdir(parents=True, exist_ok=True)
    with (work / ".preparation.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another preparation owns this work directory") from error
        if initializing or not state_exists:
            check_initialization_contents(args, initializing=initializing)
            if initializing:
                if initializing_path.is_symlink() or not initializing_path.is_dir() or any(initializing_path.iterdir()):
                    raise ValueError("Invalid preparation initialization marker")
                if not args.resume:
                    raise ValueError("Preparation initialization already exists; use --resume to verify and continue")
            else:
                initializing_path.mkdir()
                fsync_directory(args.output_dir)
                initializing = True
            if owner_path.exists():
                if json.loads(owner_path.read_text()) != owner:
                    raise ValueError("Scratch owner does not match this output directory")
            elif state_exists:
                raise ValueError("Bound preparation scratch is missing or incomplete: missing owner")
            else:
                write_json(owner_path, owner)
        else:
            database_path = work / "preparation.sqlite"
            if not owner_path.is_file() or not database_path.is_file() or not database_path.stat().st_size:
                raise ValueError("Bound preparation scratch is missing or incomplete. Restore consistent stopped scratch, or rebuild with a NEW output/work directory.")
            if json.loads(owner_path.read_text()) != owner:
                raise ValueError("Scratch owner does not match this output directory")
        # SQLite consults SQLITE_TMPDIR for file-backed temporary tables/sorts.
        previous_tmp = os.environ.get("SQLITE_TMPDIR")
        os.environ["SQLITE_TMPDIR"] = str(work)
        try:
            yield initializing
        finally:
            if previous_tmp is None:
                os.environ.pop("SQLITE_TMPDIR", None)
            else:
                os.environ["SQLITE_TMPDIR"] = previous_tmp


def signature(path: Path) -> dict:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Input changed while hashing: {path}")
    return {"path": str(path.resolve()), "size": after.st_size,
            "mtime_ns": after.st_mtime_ns, "sha256": digest.hexdigest()}


def verify_stage_evidence(record: dict, label: str) -> None:
    path = Path(record["path"])
    if not path.is_file():
        raise ValueError(f"Committed {label} artifact is missing: {path}. Restore consistent stopped scratch/output backups or rebuild in a NEW output/work directory.")
    if signature(path) != record:
        raise ValueError(f"Committed {label} artifact changed: {path}; use a NEW output/work directory")


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open("r", encoding="utf-8", newline="")


@contextmanager
def atomic_text(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def write_json(path: Path, value: dict) -> None:
    with atomic_text(path) as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")


def fasta_records(path: Path) -> Iterator[tuple[str, str]]:
    identifier = None
    chunks: list[str] = []
    with open_text(path) as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    if not chunks:
                        raise ValueError(f"Empty sequence: {identifier}")
                    yield identifier, "".join(chunks)
                fields = line[1:].split()
                if not fields:
                    raise ValueError(f"Empty FASTA ID at line {number}")
                identifier, chunks = fields[0], []
            else:
                if identifier is None:
                    raise ValueError("Sequence precedes first FASTA header")
                sequence = "".join(line.split()).upper()
                if not sequence.isalpha():
                    raise ValueError(f"Non-alphabetic sequence at line {number}")
                chunks.append(sequence)
    if identifier is not None:
        if not chunks:
            raise ValueError(f"Empty sequence: {identifier}")
        yield identifier, "".join(chunks)


def table_reader(handle, path: Path) -> csv.DictReader:
    delimiter = "," if path.name.endswith((".csv", ".csv.gz")) else "\t"
    return csv.DictReader(handle, delimiter=delimiter)


def stable_key(seed: int, axis: str, identifier: str) -> str:
    return hashlib.sha256(f"{seed}\0{axis}\0{identifier}".encode()).hexdigest()[:16]


def split_for_group(group: str, axis: str, seed: int, validation: float, test: float) -> str:
    value = int(stable_key(seed, axis, group), 16) / 2**64
    if value < test:
        return "test"
    if value < test + validation:
        return "validation"
    return "train"


class UnionFind:
    def __init__(self, identifiers: Iterable[str]):
        self.parent = {identifier: identifier for identifier in identifiers}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: str, second: str) -> None:
        first, second = self.find(first), self.find(second)
        if first != second:
            self.parent[max(first, second)] = min(first, second)


def reaction_descriptor(smiles: str) -> tuple[tuple, frozenset[tuple[int, int]]]:
    """Map-free canonical components and signed changed Morgan environments.

    Equal reactant/product environments cancel, including unchanged spectator
    currency molecules. Counts define the sign; set Jaccard intentionally does
    not model stoichiometric magnitude. Empty transformations get exact grouping
    only. Reacting coenzymes can contribute: this is not a mechanism oracle.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    fields = smiles.split(">")
    if len(fields) != 3 or not fields[0] or not fields[2]:
        raise ValueError(f"Expected nonempty reactants>agents>products: {smiles!r}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    sides: list[tuple[str, ...]] = []
    difference: Counter[int] = Counter()
    for side_index, side in enumerate((fields[0], fields[2])):
        canonical = []
        for component in side.split("."):
            molecule = Chem.MolFromSmiles(component)
            if molecule is None or molecule.GetNumAtoms() == 0:
                raise ValueError(f"Cannot canonicalize reaction component: {component!r}")
            for atom in molecule.GetAtoms():
                atom.SetAtomMapNum(0)
            canonical.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
            for bit, count in generator.GetSparseCountFingerprint(molecule).GetNonzeroElements().items():
                difference[int(bit)] += count * (-1 if side_index == 0 else 1)
        sides.append(tuple(sorted(canonical)))
    exact = tuple(sorted(sides))  # Aliases and reversed reactions stay together.
    changed = frozenset((bit, 1 if count > 0 else -1) for bit, count in difference.items() if count)
    return exact, changed


def similarity_edges(descriptors: dict[str, frozenset], threshold: float) -> Iterator[tuple[str, str, float]]:
    """Exact threshold join for sparse signed sets, with reverse invariance.

    Globally ordered prefix filtering avoids an N-by-N dense matrix. Candidate
    work can still be large for highly similar collections; no silent candidate
    caps/approximate LSH recall are used. This is exact for these fingerprints,
    not for chemistry or mechanism. Empty sets never produce similarity edges.
    """
    if not 0 < threshold <= 1:
        raise ValueError("Reaction similarity threshold must be in (0, 1]")
    oriented = []
    frequency: Counter = Counter()
    for identifier, bits in descriptors.items():
        if bits:
            for orientation in (bits, frozenset((bit, -sign) for bit, sign in bits)):
                oriented.append((identifier, orientation))
                frequency.update(orientation)
    oriented.sort(key=lambda item: (len(item[1]), item[0], sorted(item[1])))
    postings: dict[tuple, list[int]] = defaultdict(list)
    last_progress = time.monotonic()
    comparisons = 0
    for index, (identifier, bits) in enumerate(oriented):
        ordered = sorted(bits, key=lambda bit: (frequency[bit], bit))
        prefix = ordered[:len(bits) - math.ceil(threshold * len(bits)) + 1]
        candidates: set[int] = set()
        for bit in prefix:
            candidates.update(postings[bit])
        for candidate in sorted(candidates):
            other, other_bits = oriented[candidate]
            if identifier == other or len(other_bits) < threshold * len(bits):
                continue
            comparisons += 1
            intersection = len(bits.intersection(other_bits))
            score = intersection / (len(bits) + len(other_bits) - intersection)
            if score >= threshold:
                yield identifier, other, score
        for bit in prefix:
            postings[bit].append(index)
        if (index + 1) % 10000 == 0 or time.monotonic() - last_progress >= 30:
            progress(f"Reaction similarity join: {index + 1:,}/{len(oriented):,} oriented fingerprints; {comparisons:,} sparse candidate comparisons")
            last_progress = time.monotonic()


def connect(path: Path, cache_mib: int = 512) -> sqlite3.Connection:
    policy = work_storage_policy(path)
    database = sqlite3.connect(path)
    database.execute("PRAGMA locking_mode=EXCLUSIVE")
    actual = database.execute(f"PRAGMA journal_mode={policy['journal_mode']}").fetchone()[0]
    if actual != policy["journal_mode"]:
        database.close()
        raise ValueError(f"Could not establish required SQLite journal mode {policy['journal_mode']}: {actual}")
    database.execute("PRAGMA synchronous=FULL")
    database.execute("PRAGMA temp_store=FILE")
    database.execute(f"PRAGMA cache_size={-cache_mib * 1024}")
    progress(f"SQLite storage: {policy['filesystem']}, journal={actual}, exclusive writer, cache={cache_mib} MiB, path={path}")
    database.executescript("""
        CREATE TABLE IF NOT EXISTS stages(name TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS proteins(id TEXT PRIMARY KEY, sequence_sha256 TEXT NOT NULL,
            length INTEGER NOT NULL, group_id TEXT, split TEXT, sample_key TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS reactions(id TEXT PRIMARY KEY, smiles TEXT NOT NULL,
            group_id TEXT, split TEXT, sample_key TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS pairs(protein_id TEXT, reaction_id TEXT,
            PRIMARY KEY(protein_id, reaction_id)) WITHOUT ROWID;
    """)
    return database


def verify_unstarted_database(path: Path) -> None:
    """An initialization retry may complete schema DDL, never erase data."""
    if not path.exists() or not path.stat().st_size:
        if any(path.with_name(path.name + suffix).exists() for suffix in ("-journal", "-wal", "-shm")):
            raise ValueError("Incomplete initialization database has journal sidecars; restore consistent scratch or use a NEW directory")
        return
    columns = {
        "stages": ["name", "value"],
        "proteins": ["id", "sequence_sha256", "length", "group_id", "split", "sample_key"],
        "reactions": ["id", "smiles", "group_id", "split", "sample_key"],
        "pairs": ["protein_id", "reaction_id"],
    }
    # mode=rw prevents a diagnostic check from creating a missing database.
    database = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)
    try:
        database.execute("PRAGMA locking_mode=EXCLUSIVE")
        objects = database.execute("SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchall()
        for kind, name in objects:
            if kind != "table" or name not in columns:
                raise ValueError("Initialization database contains unexpected schema; use a NEW directory")
            if [row[1] for row in database.execute(f'PRAGMA table_info("{name}")')] != columns[name]:
                raise ValueError("Initialization database schema changed; use a NEW directory")
            if database.execute(f'SELECT 1 FROM "{name}" LIMIT 1').fetchone() is not None:
                raise ValueError("Initialization database contains data or committed stages; refusing to reinitialize")
    finally:
        database.close()


def stage_value(database: sqlite3.Connection, name: str):
    row = database.execute("SELECT value FROM stages WHERE name=?", (name,)).fetchone()
    return json.loads(row[0]) if row else None


def finish_stage(database: sqlite3.Connection, name: str, value: dict) -> None:
    database.execute("INSERT OR REPLACE INTO stages VALUES (?,?)", (name, json.dumps(value, sort_keys=True)))
    database.commit()


def ingest_entities(database: sqlite3.Connection, args: argparse.Namespace) -> None:
    if stage_value(database, "entities") is not None:
        return
    progress("Indexing representative FASTA and reaction inventory (streaming)")
    database.execute("DELETE FROM proteins")
    batch = []
    proteins = residues = 0
    started = last_progress = time.monotonic()
    for identifier, sequence in fasta_records(args.representative_fasta):
        batch.append((identifier, hashlib.sha256(sequence.encode()).hexdigest(), len(sequence),
                      stable_key(args.seed, "protein_sample", identifier)))
        proteins += 1
        residues += len(sequence)
        if len(batch) == BATCH_SIZE:
            batch.sort(key=lambda row: row[0])
            database.executemany("INSERT INTO proteins(id,sequence_sha256,length,sample_key) VALUES (?,?,?,?)", batch)
            batch.clear()
            now = time.monotonic()
            if proteins % 500_000 == 0 or now - last_progress >= 30:
                progress(f"Indexed {proteins:,} representative proteins / {residues:,} residues ({proteins / max(now - started, 1e-6):,.0f} proteins/s elapsed average)")
                last_progress = now
    batch.sort(key=lambda row: row[0])
    database.executemany("INSERT INTO proteins(id,sequence_sha256,length,sample_key) VALUES (?,?,?,?)", batch)
    database.execute("DELETE FROM reactions")
    with open_text(args.reactions) as handle:
        reader = table_reader(handle, args.reactions)
        if not {"reaction_id", "reaction_smiles"}.issubset(reader.fieldnames or []):
            raise ValueError("Reaction table requires reaction_id and reaction_smiles")
        for row in reader:
            identifier, smiles = row["reaction_id"].strip(), row["reaction_smiles"].strip()
            if not identifier or not smiles:
                raise ValueError("Empty reaction ID or chemistry")
            previous = database.execute("SELECT smiles FROM reactions WHERE id=?", (identifier,)).fetchone()
            if previous and previous[0] != smiles:
                raise ValueError(f"Conflicting chemistry for reaction ID {identifier}")
            database.execute("INSERT OR IGNORE INTO reactions(id,smiles,sample_key) VALUES (?,?,?)",
                             (identifier, smiles, stable_key(args.seed, "reaction_sample", identifier)))
    reactions = database.execute("SELECT count(*) FROM reactions").fetchone()[0]
    if not proteins or not reactions:
        raise ValueError("Protein and reaction inventories must be nonempty")
    finish_stage(database, "entities", {"proteins": proteins, "residues": residues, "reactions": reactions})
    progress(f"Entity stage committed: {proteins:,} proteins, {reactions:,} reactions; {time.monotonic() - started:.1f}s")


def run_command(command: list[str], log: Path) -> None:
    progress("Running " + " ".join(command))
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(datetime.now().astimezone().isoformat(timespec="seconds") + " " + json.dumps(command) + "\n")
        handle.flush()
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)


def load_clusters(database: sqlite3.Connection, path: Path, table: str) -> None:
    if table not in {"proteins", "reactions"}:
        raise ValueError("Invalid cluster entity table")
    database.execute("DROP TABLE IF EXISTS incoming_clusters")
    database.execute("CREATE TEMP TABLE incoming_clusters(member TEXT PRIMARY KEY, representative TEXT NOT NULL) WITHOUT ROWID")
    batch = []
    number = 0
    last_progress = time.monotonic()
    with open_text(path) as handle:
        for number, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if len(row) != 2 or not all(row):
                raise ValueError(f"Cluster file must be headerless representative<TAB>member: {path}:{number}")
            batch.append((row[1], row[0]))
            if len(batch) == BATCH_SIZE:
                batch.sort(key=lambda record: record[0])
                database.executemany("INSERT INTO incoming_clusters VALUES (?,?)", batch)
                batch.clear()
                if number % 1_000_000 == 0 or time.monotonic() - last_progress >= 30:
                    progress(f"Imported {number:,} {table} cluster memberships")
                    last_progress = time.monotonic()
    batch.sort(key=lambda record: record[0])
    database.executemany("INSERT INTO incoming_clusters VALUES (?,?)", batch)
    missing = database.execute(f"SELECT id FROM {table} LEFT JOIN incoming_clusters ON id=member WHERE member IS NULL LIMIT 1").fetchone()
    extra = database.execute(f"SELECT member FROM incoming_clusters LEFT JOIN {table} ON id=member WHERE id IS NULL LIMIT 1").fetchone()
    invalid = database.execute("SELECT representative FROM incoming_clusters c WHERE NOT EXISTS (SELECT 1 FROM incoming_clusters r WHERE r.member=c.representative AND r.representative=c.representative) LIMIT 1").fetchone()
    if missing or extra or invalid:
        raise ValueError(f"Cluster membership must cover each inventory ID exactly once with self-member roots: missing={missing}, extra={extra}, invalid_root={invalid}")
    database.execute(f"UPDATE {table} SET group_id=(SELECT representative FROM incoming_clusters WHERE member=id)")
    database.execute("DROP TABLE incoming_clusters")
    progress(f"Validated and assigned {number:,} {table} cluster memberships")


def group_entities(database: sqlite3.Connection, args: argparse.Namespace) -> None:
    previous = stage_value(database, "groups")
    if previous is not None:
        for name in ("protein_cluster_file", "reaction_similarity_edges"):
            verify_stage_evidence(previous[name], name)
        return
    output = args.output_dir
    work = work_directory(args)
    if args.protein_clusters:
        cluster_file = args.protein_clusters
        cluster_command = None
    else:
        cluster_prefix = work / "protein_clusters"
        cluster_file = Path(str(cluster_prefix) + "_cluster.tsv")
        cluster_command = [str(args.mmseqs), "easy-linclust", str(args.representative_fasta),
                           str(cluster_prefix), str(work / "cluster_tmp"),
                           "--min-seq-id", str(args.min_seq_id), "-c", str(args.coverage),
                           "--cov-mode", "0", "--threads", str(args.threads)]
        run_command(cluster_command, log_directory(args) / "mmseqs.log")
    progress("Importing protein groups")
    load_clusters(database, cluster_file, "proteins")
    reaction_edges = output / "reaction_similarity_edges.tsv"
    if args.reaction_clusters:
        load_clusters(database, args.reaction_clusters, "reactions")
        reaction_method = "precomputed_external_groups_not_chemically_audited"
        with atomic_text(reaction_edges) as handle:
            handle.write("reaction_id_a\treaction_id_b\tevidence\tsimilarity\n")
    else:
        progress("Grouping canonical/reversed reaction aliases and sparse transformation similarities")
        descriptors = {}
        exact_seen = {}
        union = UnionFind(row[0] for row in database.execute("SELECT id FROM reactions ORDER BY id"))
        with atomic_text(reaction_edges) as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(("reaction_id_a", "reaction_id_b", "evidence", "similarity"))
            for identifier, smiles in database.execute("SELECT id,smiles FROM reactions ORDER BY id"):
                try:
                    exact, changed = reaction_descriptor(smiles)
                except ValueError as error:
                    raise ValueError(f"Reaction {identifier}: {error}") from error
                if exact in exact_seen:
                    union.union(identifier, exact_seen[exact])
                    writer.writerow((identifier, exact_seen[exact], "canonical_or_reversed_alias", "1"))
                else:
                    exact_seen[exact] = identifier
                descriptors[identifier] = changed
                if len(descriptors) % 1000 == 0:
                    progress(f"Built transformation descriptors for {len(descriptors):,} reactions")
            for first, second, score in similarity_edges(descriptors, args.reaction_similarity_threshold):
                if union.find(first) != union.find(second):
                    union.union(first, second)
                    writer.writerow((first, second, "signed_morgan_difference_jaccard", f"{score:.12g}"))
        database.executemany("UPDATE reactions SET group_id=? WHERE id=?",
                             ((union.find(identifier), identifier) for identifier in sorted(descriptors)))
        reaction_method = "canonical_reversed_alias_plus_signed_morgan_radius2_difference_set_jaccard_connected_components"
    for table, axis in (("proteins", "protein"), ("reactions", "reaction")):
        database.create_function("group_split", 1, lambda group, axis=axis: split_for_group(group, axis, args.seed, args.validation_fraction, args.test_fraction), deterministic=True)
        database.execute(f"UPDATE {table} SET split=group_split(group_id)")
        database.execute(f"CREATE INDEX IF NOT EXISTS {table}_split_idx ON {table}(split,sample_key)")
    finish_stage(database, "groups", {"protein_cluster_file": signature(cluster_file),
                 "protein_cluster_command": cluster_command, "reaction_method": reaction_method,
                 "reaction_similarity_edges": signature(reaction_edges)})
    progress("Similarity groups and deterministic entity splits committed")


def ingest_own_pairs(database: sqlite3.Connection, args: argparse.Namespace) -> None:
    previous = stage_value(database, "own_pairs")
    if previous is not None:
        if signature(args.output_dir / "train_own_raw_associations.csv") != previous["train_raw_signature"]:
            raise ValueError("Intermediate train-own association output changed; use a new output directory")
        return
    progress("Streaming raw associations; keeping representative OWN positives only")
    database.execute("DELETE FROM pairs")
    database.execute("CREATE TEMP TABLE IF NOT EXISTS incoming_pairs(n INTEGER PRIMARY KEY,r TEXT,p TEXT)")
    database.execute("CREATE TEMP TABLE IF NOT EXISTS matched_pairs(n INTEGER PRIMARY KEY,r TEXT,p TEXT,protein_split TEXT,reaction_split TEXT)")
    counts: Counter = Counter()
    train_raw = args.output_dir / "train_own_raw_associations.csv"
    with open_text(args.raw_pairs) as source, atomic_text(train_raw) as target:
        reader = table_reader(source, args.raw_pairs)
        if not {"reaction_id", "protein_id"}.issubset(reader.fieldnames or []):
            raise ValueError("Raw associations require reaction_id and protein_id")
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        batch = []

        def flush() -> None:
            if not batch:
                return
            database.execute("DELETE FROM incoming_pairs")
            database.executemany("INSERT INTO incoming_pairs VALUES (?,?,?)", ((index, row["reaction_id"], row["protein_id"]) for index, row in enumerate(batch)))
            # One inventory join per batch. Full CSV rows remain bounded in RAM;
            # SQLite stores only row indices/IDs/splits, never serialized JSON.
            database.execute("DELETE FROM matched_pairs")
            database.execute("INSERT INTO matched_pairs SELECT i.n,i.r,i.p,p.split,r.split FROM incoming_pairs i JOIN proteins p ON p.id=i.p LEFT JOIN reactions r ON r.id=i.r")
            invalid = database.execute("SELECT r FROM matched_pairs WHERE reaction_split IS NULL LIMIT 1").fetchone()
            if invalid:
                raise ValueError(f"Representative-own association has missing reaction: {invalid[0]}")
            for index, protein_split, reaction_split in database.execute("SELECT n,protein_split,reaction_split FROM matched_pairs ORDER BY n"):
                counts["representative_own_raw_rows"] += 1
                if protein_split == reaction_split == "train":
                    writer.writerow(batch[index])
                    counts["training_own_raw_rows"] += 1
            database.execute("INSERT OR IGNORE INTO pairs SELECT p,r FROM matched_pairs")
            batch.clear()

        for number, row in enumerate(reader, 1):
            if None in row or not row.get("reaction_id") or not row.get("protein_id"):
                raise ValueError(f"Malformed raw association row {number}")
            counts["raw_association_rows"] += 1
            batch.append(row)
            if len(batch) == BATCH_SIZE:
                flush()
                if number % 1_000_000 == 0:
                    progress(f"Read {number:,} raw association rows")
        flush()
    counts["unique_own_pairs"] = database.execute("SELECT count(*) FROM pairs").fetchone()[0]
    counts["nonrepresentative_raw_rows_ignored"] = counts["raw_association_rows"] - counts["representative_own_raw_rows"]
    database.execute("CREATE INDEX IF NOT EXISTS pairs_reaction_idx ON pairs(reaction_id,protein_id)")
    finish_stage(database, "own_pairs", {**counts, "train_raw_signature": signature(train_raw)})


def export_uncertain_transfers(database: sqlite3.Connection, args: argparse.Namespace) -> dict:
    """Cluster-member transfers are exclusions only, never representative gold."""
    previous = stage_value(database, "uncertain_transfers")
    output = args.output_dir / "train_transferred_uncertain_pairs.csv"
    if previous is not None:
        if signature(output) != previous["output"]:
            raise ValueError("Intermediate transfer-exclusion output changed")
        return previous
    database.execute("CREATE TABLE IF NOT EXISTS uncertain_pairs(protein_id TEXT,reaction_id TEXT, PRIMARY KEY(protein_id,reaction_id)) WITHOUT ROWID")
    database.execute("DELETE FROM uncertain_pairs")
    rows = 0
    if args.clustered_pairs:
        progress("Streaming source-collapsed pairs for exclusion-only member-transfer uncertainty")
        database.execute("CREATE TEMP TABLE IF NOT EXISTS incoming_exclusions(p TEXT,r TEXT)")
        batch = []

        def flush() -> None:
            if not batch:
                return
            database.execute("DELETE FROM incoming_exclusions")
            database.executemany("INSERT INTO incoming_exclusions VALUES (?,?)", batch)
            invalid = database.execute("SELECT i.p,i.r FROM incoming_exclusions i LEFT JOIN proteins p ON p.id=i.p LEFT JOIN reactions r ON r.id=i.r WHERE p.id IS NULL OR r.id IS NULL LIMIT 1").fetchone()
            if invalid:
                raise ValueError(f"Clustered exclusion pair has unknown representative or reaction: {invalid}")
            database.execute("INSERT OR IGNORE INTO uncertain_pairs SELECT i.p,i.r FROM incoming_exclusions i JOIN proteins p ON p.id=i.p JOIN reactions r ON r.id=i.r WHERE p.split='train' AND r.split='train' AND NOT EXISTS (SELECT 1 FROM pairs a WHERE a.protein_id=i.p AND a.reaction_id=i.r)")
            batch.clear()

        with open_text(args.clustered_pairs) as handle:
            reader = table_reader(handle, args.clustered_pairs)
            if not {"reaction_id", "protein_id"}.issubset(reader.fieldnames or []):
                raise ValueError("Clustered pairs require reaction_id and protein_id")
            for rows, row in enumerate(reader, 1):
                if None in row or not row.get("protein_id") or not row.get("reaction_id"):
                    raise ValueError(f"Malformed clustered pair row {rows}")
                batch.append((row["protein_id"], row["reaction_id"]))
                if len(batch) == BATCH_SIZE:
                    flush()
            flush()
    count = write_query(database, output, ("protein_id", "reaction_id"), "SELECT protein_id,reaction_id FROM uncertain_pairs ORDER BY protein_id,reaction_id")
    report = {"provided": args.clustered_pairs is not None, "clustered_rows": rows,
              "training_exclusion_pairs": count, "output": signature(output),
              "semantics": "Training-axis source-collapsed pairs absent from representative-own gold; uncertainty exclusions only, never gold or negative labels"}
    finish_stage(database, "uncertain_transfers", report)
    return report


def write_query(database: sqlite3.Connection, path: Path, header: tuple, query: str, parameters=(), *, delimiter=",") -> int:
    count = 0
    # Existing CSVDataset pair consumers require an explicit unique pr_id.
    # Stable sorted SQL order supplies reproducible row IDs in each artifact.
    index_column = "pr_id" if header == ("protein_id", "reaction_id") else "rs_id" if header == ("reaction_id", "reaction_smiles") else None
    with atomic_text(path) as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow((index_column, *header) if index_column else header)
        for row in database.execute(query, parameters):
            writer.writerow((count, *row) if index_column else row)
            count += 1
    return count


def write_ids(database: sqlite3.Connection, path: Path, query: str, parameters=()) -> int:
    count = 0
    with atomic_text(path) as handle:
        for row in database.execute(query, parameters):
            handle.write(row[0] + "\n")
            count += 1
    return count


PAIR_JOIN = "FROM pairs a JOIN proteins p ON a.protein_id=p.id JOIN reactions r ON a.reaction_id=r.id"


def export_splits(database: sqlite3.Connection, args: argparse.Namespace) -> dict:
    output = args.output_dir
    counts = {}
    for split in SPLITS:
        counts[split] = {
            "pairs": write_query(database, output / f"{split}_pairs.csv", ("protein_id", "reaction_id"),
                                 f"SELECT a.protein_id,a.reaction_id {PAIR_JOIN} WHERE p.split=? AND r.split=? ORDER BY a.protein_id,a.reaction_id", (split, split)),
            "proteins": write_ids(database, output / f"{split}_protein_ids.txt", "SELECT id FROM proteins WHERE split=? ORDER BY id", (split,)),
            "reactions": write_query(database, output / f"{split}_rxns.csv", ("reaction_id", "reaction_smiles"), "SELECT id,smiles FROM reactions WHERE split=? ORDER BY id", (split,)),
        }
        write_ids(database, output / f"{split}_reaction_ids.txt", "SELECT id FROM reactions WHERE split=? ORDER BY id", (split,))
    write_query(database, output / "reactions.csv", ("reaction_id", "reaction_smiles"), "SELECT id,smiles FROM reactions ORDER BY id")
    write_ids(database, output / "all_reaction_ids.txt", "SELECT id FROM reactions ORDER BY id")
    write_ids(database, output / "all_candidate_ids.txt", "SELECT id FROM proteins ORDER BY id")
    # Full-catalog test gold is the union of complete adjacency lists of heldout
    # enzyme and reaction queries; evaluation MUST use the separate query filters.
    counts["full_catalog_test_gold"] = write_query(database, output / "test_query_gold.csv", ("protein_id", "reaction_id"), f"SELECT a.protein_id,a.reaction_id {PAIR_JOIN} WHERE p.split='test' OR r.split='test' ORDER BY a.protein_id,a.reaction_id")
    write_ids(database, output / "test_enzyme_query_ids.txt", "SELECT id FROM proteins p WHERE split='test' AND EXISTS (SELECT 1 FROM pairs a WHERE a.protein_id=p.id) ORDER BY id")
    write_ids(database, output / "test_reaction_query_ids.txt", "SELECT id FROM reactions r WHERE split='test' AND EXISTS (SELECT 1 FROM pairs a WHERE a.reaction_id=r.id) ORDER BY id")
    write_query(database, output / "protein_groups.tsv", ("protein_id", "group_id", "split", "sequence_sha256", "length"), "SELECT id,group_id,split,sequence_sha256,length FROM proteins ORDER BY id", delimiter="\t")
    write_query(database, output / "reaction_groups.tsv", ("reaction_id", "group_id", "split"), "SELECT id,group_id,split FROM reactions ORDER BY id", delimiter="\t")
    counts["quadrants"] = {f"{p}/{r}": n for p, r, n in database.execute(f"SELECT p.split,r.split,count(*) {PAIR_JOIN} GROUP BY p.split,r.split")}
    for split in SPLITS:
        if not counts[split]["pairs"]:
            raise ValueError(f"No representative-own {split} pairs remain; inspect group sizes/allocation. A usable train/validation/test preparation requires nonempty both-axis subsets.")
    return counts


def lookup_protein_splits(database: sqlite3.Connection, identifiers: list[str]) -> list[str | None]:
    """One bounded SQL join, preserving duplicate IDs and input row order."""
    if len(identifiers) > 2 * BATCH_SIZE:
        raise ValueError("Protein split lookup exceeds its bounded batch size")
    database.execute("CREATE TEMP TABLE IF NOT EXISTS requested_proteins(n INTEGER PRIMARY KEY,id TEXT)")
    database.execute("DELETE FROM requested_proteins")
    database.executemany("INSERT INTO requested_proteins VALUES (?,?)", enumerate(identifiers))
    return [row[0] for row in database.execute("SELECT p.split FROM requested_proteins q LEFT JOIN proteins p ON p.id=q.id ORDER BY q.n")]


def write_heldout_queries(database: sqlite3.Connection, fasta: Path, destination: Path) -> int:
    """Bound both record count and sequence payload while bulk-joining IDs."""
    count = scanned = batch_residues = 0
    batch: list[tuple[str, str]] = []
    last_progress = time.monotonic()
    with atomic_text(destination) as handle:
        def flush() -> None:
            nonlocal count, batch_residues, last_progress
            if not batch:
                return
            splits = lookup_protein_splits(database, [identifier for identifier, _ in batch])
            for (identifier, sequence), split in zip(batch, splits):
                if split is None:
                    raise ValueError(f"FASTA protein outside the bound split inventory: {identifier}")
                if split != "train":
                    handle.write(f">{identifier}\n{sequence}\n")
                    count += 1
            batch.clear()
            batch_residues = 0
            if time.monotonic() - last_progress >= 30:
                progress(f"Audit query preparation: scanned {scanned:,} proteins, retained {count:,} held-out queries")
                last_progress = time.monotonic()

        for identifier, sequence in fasta_records(fasta):
            if batch and batch_residues + len(sequence) > FASTA_LOOKUP_BATCH_RESIDUES:
                flush()
            batch.append((identifier, sequence))
            scanned += 1
            batch_residues += len(sequence)
            if len(batch) >= BATCH_SIZE or batch_residues >= FASTA_LOOKUP_BATCH_RESIDUES:
                flush()
        flush()
    progress(f"Audit query preparation complete: {count:,} held-out sequences from {scanned:,} representatives")
    return count


def audit_protein_splits(database: sqlite3.Connection, args: argparse.Namespace) -> dict:
    completed = stage_value(database, "protein_audit")
    if completed:
        verify_stage_evidence(completed["hits"], "protein audit search")
        return completed
    output = args.output_dir
    work = work_directory(args)
    commands = []
    if args.protein_cross_split_hits:
        hits = args.protein_cross_split_hits
        method = "provided_search_results_external_completeness_not_verifiable"
    else:
        heldout = work / "heldout.fasta"
        progress("Writing held-out sequence queries for sensitive cross-split MMseqs search")
        write_heldout_queries(database, args.representative_fasta, heldout)
        # Release the temporary-table transaction before the external search.
        database.commit()
        hits = work / "protein_cross_split_hits.tsv"
        command = [str(args.mmseqs), "easy-search", str(heldout), str(args.representative_fasta),
                   str(hits), str(work / "audit_tmp"), "--min-seq-id", str(args.min_seq_id),
                   "-c", str(args.coverage), "--cov-mode", "0", "--alignment-mode", "3",
                   "-s", "7.5", "--max-seqs", "1000000", "--threads", str(args.threads),
                   "--format-output", "query,target,fident,qcov,tcov"]
        run_command(command, log_directory(args) / "mmseqs.log")
        commands.append(command)
        method = "sensitive_mmseqs_heldout_against_all_search"
    violations = total = 0
    last_progress = time.monotonic()
    progress(f"Validating protein search hits in bounded batches: {hits}")
    violation_path = output / "protein_cross_split_violations.tsv"
    with open_text(hits) as source, atomic_text(violation_path) as target:
        writer = csv.writer(target, delimiter="\t")
        writer.writerow(("query", "target", "fident", "qcov", "tcov", "query_split", "target_split"))
        batch = []

        def flush() -> None:
            nonlocal violations, total, last_progress
            if not batch:
                return
            splits = lookup_protein_splits(database, [identifier for row, *_ in batch for identifier in row[:2]])
            for index, (row, identity, qcov, tcov) in enumerate(batch):
                first, second = splits[2 * index:2 * index + 2]
                if first is None or second is None:
                    raise ValueError(f"Search hit references a protein outside the bound inventory: {row[:2]}")
                total += 1
                if first != second and identity >= args.min_seq_id and min(qcov, tcov) >= args.coverage:
                    violations += 1
                    writer.writerow((*row, first, second))
            batch.clear()
            if total % 1_000_000 == 0 or time.monotonic() - last_progress >= 30:
                progress(f"Checked {total:,} search hits; {violations:,} cross-split violations")
                last_progress = time.monotonic()

        for number, row in enumerate(csv.reader(source, delimiter="\t"), 1):
            if row == ["query", "target", "fident", "qcov", "tcov"]:
                continue
            if len(row) != 5:
                raise ValueError(f"Search hits require query,target,fident,qcov,tcov TSV: line {number}")
            try:
                identity, qcov, tcov = map(float, row[2:])
            except ValueError as error:
                raise ValueError(f"Invalid MMseqs numeric score at line {number}") from error
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in (identity, qcov, tcov)):
                raise ValueError("Use MMseqs fident/qcov/tcov fractions in [0,1], not percentages")
            batch.append((row, identity, qcov, tcov))
            if len(batch) == BATCH_SIZE:
                flush()
        flush()
    progress(f"Protein audit checked {total:,} hits; {violations:,} prohibited cross-split matches")
    report = {"method": method, "hits": signature(hits), "commands": commands,
              "rows_checked": total, "cross_split_violations": violations,
              "min_seq_id": args.min_seq_id, "coverage_both_sequences": args.coverage,
              "guarantee": "No qualifying cross-split hits detected in this search; heuristic search does not prove absence of every possible alignment."}
    write_json(output / "protein_similarity_audit.json", report)
    if violations:
        raise ValueError(f"Found {violations} prohibited cross-split protein matches. Inspect {violation_path}; merge all offending similarity groups into connected components and rerun preparation in a NEW output directory with that cluster map, then repeat the cross-split search. No completed manifest was written.")
    finish_stage(database, "protein_audit", report)
    return report


def export_panels(database: sqlite3.Connection, args: argparse.Namespace) -> dict:
    output = args.output_dir / "panels"
    metadata = {"schema_version": SCHEMA + "_panels", "panels": {},
                "candidate_policy": "Regime-specific entity axes; all positive candidate proteins are mandatory. Remaining proteins are deterministic sampled decoys; reactions are the complete regime-specific reaction axis.",
                "selection": "Hash(seed, entity ID) then lexical ID, independent of model scores; select query anchors and retain all own positives within the specified quadrant."}
    database.executescript("CREATE TEMP TABLE IF NOT EXISTS panel_queries(id TEXT PRIMARY KEY); CREATE TEMP TABLE IF NOT EXISTS panel_candidates(id TEXT PRIMARY KEY); CREATE TEMP TABLE IF NOT EXISTS combined_queries(axis TEXT,id TEXT,PRIMARY KEY(axis,id));")
    for split in ("validation", "test"):
        database.execute("DELETE FROM combined_queries")
        for regime, protein_split, reaction_split in (("cold_protein", split, "train"), ("cold_reaction", "train", split), ("both_cold", split, split)):
            for direction in ("reaction_to_enzyme", "enzyme_to_reaction"):
                name = f"{split}_{regime}_{direction}"
                query_alias = "r" if direction == "reaction_to_enzyme" else "p"
                database.execute("DELETE FROM panel_queries")
                database.execute(f"INSERT INTO panel_queries SELECT DISTINCT {query_alias}.id {PAIR_JOIN} WHERE p.split=? AND r.split=? ORDER BY {query_alias}.sample_key,{query_alias}.id LIMIT ?", (protein_split, reaction_split, args.panel_queries))
                select = f"SELECT a.protein_id,a.reaction_id {PAIR_JOIN} JOIN panel_queries q ON q.id={query_alias}.id WHERE p.split=? AND r.split=?"
                pair_count = write_query(database, output / f"{name}_pairs.csv", ("protein_id", "reaction_id"), select + " ORDER BY a.protein_id,a.reaction_id", (protein_split, reaction_split))
                query_count = write_ids(database, output / f"{name}_query_ids.txt", "SELECT id FROM panel_queries ORDER BY id")
                if regime == "both_cold":
                    database.execute("INSERT INTO combined_queries SELECT ?,id FROM panel_queries", (direction,))
                database.execute("DELETE FROM panel_candidates")
                database.execute(f"INSERT OR IGNORE INTO panel_candidates SELECT a.protein_id {PAIR_JOIN} JOIN panel_queries q ON q.id={query_alias}.id WHERE p.split=? AND r.split=?", (protein_split, reaction_split))
                positives = database.execute("SELECT count(*) FROM panel_candidates").fetchone()[0]
                database.execute("INSERT OR IGNORE INTO panel_candidates SELECT id FROM proteins WHERE split=? AND id NOT IN (SELECT id FROM panel_candidates) ORDER BY sample_key,id LIMIT ?", (protein_split, max(0, args.panel_protein_candidates - positives)))
                protein_count = write_ids(database, output / f"{name}_protein_ids.txt", "SELECT id FROM panel_candidates ORDER BY id")
                reaction_path = args.output_dir / f"{reaction_split}_reaction_ids.txt"
                metadata["panels"][name] = {"split": split, "regime": regime, "direction": direction,
                    "protein_axis": protein_split, "reaction_axis": reaction_split,
                    "pairs": str((output / f"{name}_pairs.csv").resolve()),
                    "query_ids": str((output / f"{name}_query_ids.txt").resolve()),
                    "protein_candidate_ids": str((output / f"{name}_protein_ids.txt").resolve()),
                    "reaction_candidate_ids": str(reaction_path.resolve()),
                    "query_count": query_count, "pair_count": pair_count,
                    "protein_candidate_count": protein_count, "mandatory_positive_proteins": positives,
                    "empty_panel": query_count == 0}
        combined_name = f"{split}_both_cold"
        condition = "p.split=? AND r.split=? AND (EXISTS (SELECT 1 FROM combined_queries q WHERE q.axis='enzyme_to_reaction' AND q.id=p.id) OR EXISTS (SELECT 1 FROM combined_queries q WHERE q.axis='reaction_to_enzyme' AND q.id=r.id))"
        combined_pairs = output / f"{combined_name}_pairs.csv"
        combined_count = write_query(database, combined_pairs, ("protein_id", "reaction_id"), f"SELECT a.protein_id,a.reaction_id {PAIR_JOIN} WHERE {condition} ORDER BY a.protein_id,a.reaction_id", (split, split))
        database.execute("DELETE FROM panel_candidates")
        database.execute(f"INSERT OR IGNORE INTO panel_candidates SELECT a.protein_id {PAIR_JOIN} WHERE {condition}", (split, split))
        positives = database.execute("SELECT count(*) FROM panel_candidates").fetchone()[0]
        database.execute("INSERT OR IGNORE INTO panel_candidates SELECT id FROM proteins WHERE split=? AND id NOT IN (SELECT id FROM panel_candidates) ORDER BY sample_key,id LIMIT ?", (split, max(0, args.panel_protein_candidates - positives)))
        combined_candidates = output / f"{combined_name}_protein_ids.txt"
        candidate_count = write_ids(database, combined_candidates, "SELECT id FROM panel_candidates ORDER BY id")
        filters = {direction: [row[0] for row in database.execute("SELECT id FROM combined_queries WHERE axis=? ORDER BY id", (direction,))] for direction in ("reaction_to_enzyme", "enzyme_to_reaction")}
        write_json(output / f"{combined_name}_query_ids.json", filters)
        metadata.setdefault("combined_panels", {})[combined_name] = {
            "pairs": str(combined_pairs.resolve()), "protein_candidate_ids": str(combined_candidates.resolve()),
            "reaction_candidate_ids": str((args.output_dir / f"{split}_reaction_ids.txt").resolve()),
            "query_ids": str((output / f"{combined_name}_query_ids.json").resolve()),
            "pair_count": combined_count, "protein_candidate_count": candidate_count,
            "query_counts": {direction: len(values) for direction, values in filters.items()},
            "warning": "Use direction-specific query filters for the capped benchmark. Complete gold introduces incidental opposite-axis entities; unfiltered evaluation does not retain the stated anchor cap."}
    write_json(output / "panel_manifest.json", metadata)
    return metadata


def check_args(args: argparse.Namespace) -> None:
    if not (0 < args.validation_fraction < 1 and 0 < args.test_fraction < 1 and args.validation_fraction + args.test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than one")
    if not (0 < args.min_seq_id <= 1 and 0 < args.coverage <= 1 and 0 < args.reaction_similarity_threshold <= 1):
        raise ValueError("Identity, coverage and reaction similarity thresholds must be in (0,1]")
    if min(args.threads, args.panel_queries, args.panel_protein_candidates) < 1:
        raise ValueError("Threads and panel sizes must be positive")
    if not 1 <= args.sqlite_cache_mib <= 65536:
        raise ValueError("SQLite cache must be between 1 and 65536 MiB")
    if args.work_dir is not None and not args.work_dir.is_absolute():
        raise ValueError("--work-dir must be an absolute, dedicated scratch directory")
    work = work_directory(args).resolve()
    if work == args.output_dir.resolve() or work in args.output_dir.resolve().parents:
        raise ValueError("Work directory must not equal or contain the published output directory")
    if not args.mmseqs and not (args.protein_clusters and args.protein_cross_split_hits):
        raise ValueError("Provide MMseqs, or both precomputed protein clusters and cross-split search hits")
    for name in ("representative_fasta", "raw_pairs", "reactions", "protein_clusters", "reaction_clusters", "protein_cross_split_hits", "clustered_pairs"):
        path = getattr(args, name)
        if path and (path.resolve() == args.output_dir.resolve() or args.output_dir.resolve() in path.resolve().parents):
            raise ValueError(f"Input {name} must be outside the dedicated output directory")
        if path and (path.resolve() == work or work in path.resolve().parents):
            raise ValueError(f"Input {name} must be outside the dedicated work directory")


def _prepare_locked(args: argparse.Namespace, *, initializing: bool = False) -> dict:
    check_args(args)
    args.output_dir = args.output_dir.resolve()
    for name in ("representative_fasta", "raw_pairs", "reactions", "protein_clusters", "reaction_clusters", "protein_cross_split_hits", "clustered_pairs"):
        if getattr(args, name):
            setattr(args, name, getattr(args, name).resolve())
    inputs = {name: signature(getattr(args, name)) for name in ("representative_fasta", "raw_pairs", "reactions", "protein_clusters", "reaction_clusters", "protein_cross_split_hits", "clustered_pairs") if getattr(args, name)}
    inputs["implementation"] = signature(Path(__file__))
    inputs["builder"] = signature(Path(__file__).resolve().parents[2] / "scripts" / "prepare_horizyn1_training.py")
    mmseqs_version = None
    if args.mmseqs:
        binary = shutil.which(str(args.mmseqs))
        if not binary:
            raise ValueError(f"MMseqs executable not found: {args.mmseqs}")
        args.mmseqs = Path(binary).resolve()
        inputs["mmseqs"] = signature(args.mmseqs)
        mmseqs_version = subprocess.run([str(args.mmseqs), "version"], check=True, capture_output=True, text=True).stdout.strip()
    rdkit_version = None
    if not args.reaction_clusters:
        try:
            import rdkit
        except ImportError as error:
            raise ValueError("RDKit is required for reaction grouping; use the project's chemistry environment") from error
        rdkit_version = rdkit.__version__
    parameters = {key: value for key, value in vars(args).items() if key not in {"resume", "output_dir"} and not isinstance(value, Path)}
    binding = {"schema_version": SCHEMA, "inputs": inputs, "parameters": parameters,
               "work_dir": str(work_directory(args)), "log_dir": str(log_directory(args)),
               "storage_policy": work_storage_policy(work_directory(args)),
               "versions": {"python": sys.version, "sqlite": sqlite3.sqlite_version, "rdkit": rdkit_version, "mmseqs": mmseqs_version}}
    state_path = args.output_dir / "preparation_state.json"
    manifest_path = args.output_dir / "preparation_manifest.json"
    if state_path.exists():
        if not args.resume:
            raise ValueError("Preparation already exists; use --resume to verify and continue")
        if json.loads(state_path.read_text()) != binding:
            raise ValueError("Stale preparation: input content/stat, implementation, tools, or parameters changed; use a NEW output directory")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            for evidence in (manifest["stages"]["groups"]["protein_cluster_file"], manifest["protein_similarity_audit"]["hits"]):
                if not Path(evidence["path"]).is_file():
                    raise ValueError("Completed preparation scratch evidence is missing; restore a consistent backup or rebuild in a NEW output/work directory")
                if signature(Path(evidence["path"])) != evidence:
                    raise ValueError(f"Completed preparation scratch evidence changed: {evidence['path']}")
            for name, expected in manifest["output_signatures"].items():
                if signature(Path(name)) != expected:
                    raise ValueError(f"Completed output changed: {name}")
            return manifest
    else:
        check_initialization_contents(args, initializing=initializing)
        write_json(state_path, binding)
    database_path = work_directory(args) / "preparation.sqlite"
    if initializing:
        verify_unstarted_database(database_path)
    database = connect(database_path, args.sqlite_cache_mib)
    try:
        if initializing:
            # The FULL-synchronous schema must be durable before leaving setup.
            # No entity/pair writes are allowed until the marker deletion is
            # durable, so only the explicit setup phase can recreate a DB.
            database.commit()
            if database.execute("PRAGMA journal_mode").fetchone()[0] == "wal":
                checkpoint = database.execute("PRAGMA wal_checkpoint(FULL)").fetchone()
                if checkpoint[0] != 0:
                    raise ValueError("Could not checkpoint initialization schema")
            fsync_directory(work_directory(args))
            initialization_directory(args).rmdir()
            fsync_directory(args.output_dir)
        ingest_entities(database, args)
        group_entities(database, args)
        # The audit only needs entity groups. Fail before parsing/indexing 30M
        # raw pairs when a split still has prohibited sequence similarities.
        audit = audit_protein_splits(database, args)
        ingest_own_pairs(database, args)
        uncertain_transfers = export_uncertain_transfers(database, args)
        split_counts = export_splits(database, args)
        panels = export_panels(database, args)
        stages = {name: json.loads(value) for name, value in database.execute("SELECT name,value FROM stages")}
        group_counts = {table: {split: count for split, count in database.execute(f"SELECT split,count(DISTINCT group_id) FROM {table} GROUP BY split")} for table in ("proteins", "reactions")}
        # Detect source mutation during a long build before publishing completion.
        for name, expected in inputs.items():
            actual = Path(expected["path"]).stat()
            if (actual.st_size, actual.st_mtime_ns) != (expected["size"], expected["mtime_ns"]):
                raise ValueError(f"Bound input changed during preparation: {name}")
        outputs = sorted(path for directory in (args.output_dir, args.output_dir / "panels") for path in directory.iterdir() if path.is_file() and path != manifest_path and not path.name.endswith(".partial") and not path.name.startswith("."))
        manifest = {**binding, "status": "complete", "counts": split_counts, "group_counts": group_counts,
            "stages": stages, "protein_similarity_audit": audit,
            "representative_fasta": str(args.representative_fasta),
            "all_reactions": str(args.output_dir / "reactions.csv"),
            "training_own_raw_associations": str(args.output_dir / "train_own_raw_associations.csv"),
            "training_transfer_uncertainty_exclusions": uncertain_transfers,
            "full_catalog_test": {"gold": str(args.output_dir / "test_query_gold.csv"),
                "enzyme_query_ids": str(args.output_dir / "test_enzyme_query_ids.txt"),
                "reaction_query_ids": str(args.output_dir / "test_reaction_query_ids.txt"),
                "protein_candidate_ids": str(args.output_dir / "all_candidate_ids.txt"),
                "reaction_candidate_ids": str(args.output_dir / "all_reaction_ids.txt"),
                "semantics": "Separate held-out query filters, complete representative-own adjacency across all opposite-axis splits. Test gold is their union; never evaluate incidental non-test entities as test queries."},
            "panels_manifest": str(args.output_dir / "panels" / "panel_manifest.json"),
            "panel_count": len(panels["panels"]),
            "split_semantics": {"training": "E_train x R_train representative-own positives only; no edge holdout inside these axes",
                "validation_test_pairs": "E_split x R_split (both cold); other cold quadrants are separate panels",
                "allocation": "Independent seeded hash allocation of whole similarity groups, expected 90/5/5 with defaults; realized entity/pair fractions may differ, especially for large connected components",
                "sampling_exclusions": "Training samplers must use train_protein_ids.txt and train_rxns.csv only; all heldout entity axes are excluded",
                "raw_associations": "Original columns/values retained only for representative-own E_train x R_train raw rows, with duplicates/provenance preserved; export labels with --pair-scope train",
                "source_collapse": "No member-to-representative reaction transfer; raw nonrepresentative associations ignored",
                "reaction_caveat": "Fingerprint similarity and transitive groups are leakage-control heuristics, not enzyme/mechanism equivalence; unchanged environments cancel, reacting currency may still contribute",
                "external_groups": "Precomputed cluster/search files are content-bound but their algorithm/completeness cannot be established by this script"},
            "output_signatures": {str(path): signature(path) for path in outputs}}
        write_json(manifest_path, manifest)
        return manifest
    finally:
        database.close()


def prepare(args: argparse.Namespace) -> dict:
    """Serialize writers; never silently race two resumable preparations."""
    check_args(args)
    args.output_dir = args.output_dir.resolve()
    args.work_dir = work_directory(args).resolve()
    args.log_dir = log_directory(args).resolve()
    work_storage_policy(args.work_dir)  # Verify the journal policy before writes.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".preparation.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another preparation owns this output directory") from error
        with owned_work_directory(args) as initializing:
            return _prepare_locked(args, initializing=initializing)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for argument in ("representative-fasta", "raw-pairs", "reactions", "output-dir"):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, help="Absolute dedicated directory for SQLite and MMseqs scratch; defaults to output-dir/work. Network filesystems use DELETE rollback journaling, local filesystems use WAL; never migrate a live database.")
    parser.add_argument("--log-dir", type=Path, help="Directory for persistent MMseqs logs, may be shared; defaults to output-dir/logs")
    parser.add_argument("--sqlite-cache-mib", type=int, default=512, help="Bounded SQLite page cache in MiB (1..65536); exclusive writer and FULL synchronous durability remain enabled")
    parser.add_argument("--mmseqs", type=Path)
    parser.add_argument("--protein-clusters", type=Path, help="Headerless representative<TAB>member, covering representative inventory exactly")
    parser.add_argument("--protein-cross-split-hits", type=Path, help="Complete external search results: query,target,fident,qcov,tcov TSV, scores in [0,1]")
    parser.add_argument("--reaction-clusters", type=Path, help="External complete headerless representative<TAB>member map; bypasses RDKit grouping and declares external unverified chemistry")
    parser.add_argument("--clustered-pairs", type=Path, help="Optional source-collapsed inventory pairs, used ONLY for training negative uncertainty exclusions")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--min-seq-id", type=float, default=0.5)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--reaction-similarity-threshold", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--panel-queries", type=int, default=1000)
    parser.add_argument("--panel-protein-candidates", type=int, default=100000)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    manifest = prepare(args)
    print(json.dumps({"status": manifest["status"], "manifest": str(args.output_dir / "preparation_manifest.json"), "counts": manifest["counts"]}, sort_keys=True, indent=2))
