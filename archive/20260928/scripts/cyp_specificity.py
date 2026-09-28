#!/usr/bin/env python3
"""Data contract and strict evaluation for the released CYP within-family task."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile

import numpy as np

RELEASE_REVISION = "a5c15469c6caae1328ef777ba141215a8c7c14b5"
RELEASE_REPO = "lizmahood/cyp_pred_repos"
RELEASE_FILE = "fusionesp_data_dir.tar.gz"
POOL_PREFIX = "data_dir/eval/inputs/rxn_files/"
SCHEMA = "cyp_specificity_v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_rows(path):
    with Path(path).open(newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t" if str(path).endswith(".tsv") else ",")


def write_rows(path, rows, fields=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def canonical_reaction(smiles):
    """Remove atom maps/order only; preserve stereo, charge, multiplicity, direction."""
    from rdkit import Chem

    # A single '>' can belong to an RDKit dative bond (N->[Fe]), not a delimiter.
    parts = smiles.strip().split(">>")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Expected explicit substrates>>products: {smiles[:80]}")
    sides = []
    for side in parts:
        molecules = []
        for text in side.split("."):
            mol = Chem.MolFromSmiles(text)
            if mol is None or any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
                raise ValueError(f"Invalid or unspecified molecule: {text}")
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            molecules.append(Chem.MolToSmiles(mol, isomericSmiles=True))
        sides.append(".".join(sorted(molecules)))
    return ">>".join(sides)


def reaction_key(smiles):
    forward = canonical_reaction(smiles)
    return min(forward, ">>".join(reversed(forward.split(">>"))))


def sequence_hash(sequence):
    sequence = "".join(sequence.split()).upper()
    if not sequence or not re.fullmatch("[ACDEFGHIKLMNPQRSTVWYBXZOUJ]+", sequence):
        raise ValueError("Invalid or empty protein sequence")
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def read_fasta(path):
    name, sequence = None, []
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(sequence).upper()
                name, sequence = line[1:].split()[0], []
            else:
                sequence.append(line.strip())
        if name is not None:
            yield name, "".join(sequence).upper()


def prepare(archive, output, expected_queries=45):
    """Read CSV members only. Never extract paths, links, pickles, or executables."""
    output = Path(output)
    archive_hash = digest(archive)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = verify_bundle(output)
        if manifest["archive_sha256"] != archive_hash:
            raise ValueError("Release changed; use a new output directory")
        return manifest
    proteins, queries, candidates, members = {}, [], [], []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.name.startswith(POOL_PREFIX) or not member.name.endswith(".csv"):
                continue
            filename = PurePosixPath(member.name).name
            match = re.fullmatch(r"r_([A-Za-z0-9]+)_[0-9]+_[0-9]+\.csv", filename)
            if (not member.isfile() or not match or member.name != POOL_PREFIX + filename
                    or member.size > 16 * 1024**2):
                raise ValueError(f"Unexpected release member: {member.name}")
            raw = tar.extractfile(member).read()
            rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
            if not rows or not {"reaction", "sequence", "protein_id", "cif"} <= set(rows[0]):
                raise ValueError(f"Invalid candidate table: {member.name}")
            qid, positive = filename[:-4], match[1]
            reactions = {canonical_reaction(row["reaction"]) for row in rows}
            ids = [row["protein_id"] for row in rows]
            if len(reactions) != 1 or len(ids) != len(set(ids)) or positive not in ids:
                raise ValueError(f"Inconsistent query, duplicate ID, or missing positive: {qid}")
            organisms = {PurePosixPath(row["cif"]).parent.name for row in rows}
            if len(organisms) != 1:
                raise ValueError(f"Mixed source organisms in {qid}")
            for row in rows:
                pid, seq = row["protein_id"], "".join(row["sequence"].split()).upper()
                sha = sequence_hash(seq)
                record = dict(protein_id=pid, sequence=seq, sha256=sha)
                if pid in proteins and proteins[pid] != record:
                    raise ValueError(f"Conflicting sequences for {pid}")
                proteins[pid] = record
                candidates.append(dict(query_id=qid, protein_id=pid,
                                       designated_positive=int(pid == positive)))
            queries.append(dict(reaction_id=qid, rxn=reactions.pop(),
                                positive_id=positive, organism=organisms.pop(),
                                candidate_count=len(ids),
                                unique_sequences=len({proteins[p]["sha256"] for p in ids})))
            members.append(dict(member=member.name, sha256=hashlib.sha256(raw).hexdigest()))
    if len(queries) != expected_queries or len({q["reaction_id"] for q in queries}) != len(queries):
        raise ValueError(f"Expected {expected_queries} unique queries, got {len(queries)}")
    queries.sort(key=lambda q: q["reaction_id"])
    write_rows(output / "queries.csv", queries)
    write_rows(output / "reactions.csv", [dict(reaction_id=q["reaction_id"], rxn=q["rxn"]) for q in queries])
    write_rows(output / "proteins.csv", [proteins[p] for p in sorted(proteins)])
    write_rows(output / "candidates.csv", sorted(candidates, key=lambda r: (r["query_id"], r["protein_id"])))
    # All candidates, not just designated positives, must receive functional tokens.
    write_rows(output / "encoding_pairs.csv", [dict(reaction_id="candidates", protein_id=p) for p in sorted(proteins)])
    fasta = output / "proteins.fasta"
    with fasta.open("w") as stream:
        for pid in sorted(proteins):
            stream.write(f">{pid}\n{proteins[pid]['sequence']}\n")
    manifest = dict(schema=SCHEMA, archive_sha256=archive_hash,
                    source_repo=RELEASE_REPO, source_revision=RELEASE_REVISION,
                    source_archive=RELEASE_FILE, members=members,
                    query_count=len(queries), candidate_count=len(proteins),
                    candidate_pairs=len(candidates),
                    files={p.name: digest(p) for p in output.iterdir() if p.suffix in {".csv", ".fasta"}},
                    labels="One published designated catalyst per query; other candidates are UNKNOWN, not assayed negatives",
                    protocol="Published candidate pools; canonicalized map-free stereo-preserving full reactions; no EC inputs")
    write_json(manifest_path, manifest)
    return manifest


def verify_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unknown benchmark schema")
    for name, sha in manifest["files"].items():
        if Path(name).name != name or digest(directory / name) != sha:
            raise ValueError(f"Benchmark input changed: {name}")
    return manifest


def evaluate_scores(directory, predictions):
    """Require exactly one finite score for every published query-candidate pair."""
    directory = Path(directory)
    verify_bundle(directory)
    queries = {r["reaction_id"]: r for r in read_rows(directory / "queries.csv")}
    proteins = {r["protein_id"]: r for r in read_rows(directory / "proteins.csv")}
    expected = {(r["query_id"], r["protein_id"]) for r in read_rows(directory / "candidates.csv")}
    scores = {}
    for row in predictions:
        key = row["query_id"], row["protein_id"]
        score = float(row["score"])
        if key in scores or key not in expected or not math.isfinite(score):
            raise ValueError(f"Duplicate, unexpected, or non-finite prediction: {key}")
        scores[key] = score
    if set(scores) != expected:
        raise ValueError(f"Missing {len(expected - set(scores))} candidate scores; pools cannot shrink")
    per_query, rankings = [], []
    for qid, query in sorted(queries.items()):
        pool = sorted((p for q, p in expected if q == qid), key=lambda p: (-scores[qid, p], p))
        positive = query["positive_id"]
        value = scores[qid, positive]
        rank = pool.index(positive) + 1
        best_rank = 1 + sum(scores[qid, p] > value for p in pool)
        worst_rank = sum(scores[qid, p] >= value for p in pool)
        hashes = list(dict.fromkeys(proteins[p]["sha256"] for p in pool))
        sequence_rank = hashes.index(proteins[positive]["sha256"]) + 1
        result = dict(query_id=qid, organism=query["organism"], n_candidates=len(pool),
                      unique_sequences=len(hashes), first_positive_rank=rank,
                      first_positive_mrr=1 / rank, optimistic_mrr=1 / best_rank,
                      pessimistic_mrr=1 / worst_rank, percentile=rank / len(pool),
                      positive_tie_size=worst_rank - best_rank + 1,
                      sequence_rank=sequence_rank, sequence_mrr=1 / sequence_rank)
        result.update({f"hit_at_{k}": int(rank <= k) for k in (1, 5, 10)})
        result.update({f"hit_at_{p}pct": int(rank / len(pool) <= p / 100) for p in (10, 20, 50)})
        per_query.append(result)
        rankings.extend(dict(query_id=qid, protein_id=p, rank=i + 1, score=scores[qid, p],
                             designated_positive=int(p == positive),
                             positive_sequence_match=int(proteins[p]["sha256"] == proteins[positive]["sha256"]))
                        for i, p in enumerate(pool))
    metrics = {k: float(np.mean([r[k] for r in per_query])) for k in per_query[0]
               if k not in {"query_id", "organism"}}
    metrics["queries"] = len(per_query)
    return metrics, per_query, rankings


def compare_reports(directory, predictions, output, bootstrap_samples=2000):
    """All methods evaluated together on exactly the same data, with paired CIs."""
    if not predictions or bootstrap_samples < 1:
        raise ValueError("At least one method and one bootstrap sample are required")
    if any(not name.replace("_", "").isalnum() for name in predictions):
        raise ValueError("Method names must be alphanumeric identifiers")
    output = Path(output)
    summary, per_model = {}, {}
    for name, path in predictions.items():
        metrics, rows, rankings = evaluate_scores(directory, read_rows(path))
        summary[name], per_model[name] = metrics, rows
        write_rows(output / f"{name}.per_query.csv", rows)
        write_rows(output / f"{name}.rankings.csv", rankings)
    differences = []
    if "f3" in summary and "residual" in summary:
        base, full = per_model["f3"], per_model["residual"]
        groups = sorted({r["organism"] for r in base})
        delta = np.array([b["first_positive_mrr"] - a["first_positive_mrr"] for a, b in zip(base, full)])
        by_group = [np.array([i for i, r in enumerate(base) if r["organism"] == g]) for g in groups]
        rng = np.random.default_rng(42)
        boot = [float(delta[np.concatenate([by_group[i] for i in rng.integers(len(groups), size=len(groups))])].mean())
                for _ in range(bootstrap_samples)]
        differences.append(dict(comparison="residual_minus_f3", metric="first_positive_mrr",
                                delta=float(delta.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(),
                                method="paired organism-cluster bootstrap, conditional on fixed checkpoints",
                                organism_clusters=len(groups)))
    write_json(output / "summary.json", dict(metrics=summary, differences=differences,
                                             benchmark_sha256=digest(Path(directory) / "manifest.json")))
    lines = ["# CYP within-family specificity retrieval", "",
             "Same released candidate pool for every method; unknown candidates are not biological negatives.",
             "MRR is reciprocal rank of the published designated catalyst, NOT all-positive ReactZyme MRR.",
             "Ties: descending score, ascending protein ID. Optimistic/pessimistic tie bounds and sequence-deduplicated diagnostics are in JSON/CSV.", "",
             "| Model | Queries | Hit@1 | Hit@5 | Hit@10 | MRR | Mean screening depth |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in summary.items():
        lines.append(f"| {name} | {m['queries']} | {m['hit_at_1']:.4f} | {m['hit_at_5']:.4f} | {m['hit_at_10']:.4f} | {m['first_positive_mrr']:.4f} | {m['first_positive_rank']:.2f} |")
    for comparison in differences:
        low, high = comparison["ci95"]
        lines += ["", f"Residual − F3 MRR: {comparison['delta']:+.4f}; paired organism-cluster "
                  f"bootstrap 95% CI [{low:+.4f}, {high:+.4f}] (fixed checkpoints)."]
    lines += ["", "These are retrospective within-family scores, not proof of activity or leakage-free generalization.",
              "See audit outputs for specified training sources; untraced upstream supervision remains UNKNOWN."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return summary
