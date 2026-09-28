"""Strict, tie-aware evaluation and CPU controls for the released CYP pools.

Separate from cyp_specificity.py so completed feature/score receipts stay valid.
"""
import csv
import io
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np

try:
    from .cyp_specificity import (
        POOL_PREFIX, canonical_reaction, digest, evaluate_scores, read_rows,
        sequence_hash, verify_bundle, write_json, write_rows,
    )
except ImportError:
    from cyp_specificity import (
        POOL_PREFIX, canonical_reaction, digest, evaluate_scores, read_rows,
        sequence_hash, verify_bundle, write_json, write_rows,
    )

DATA_SHA = "51bdb5f18b5ecf708dab1370be25ede27861db907dd30dd051f0ea63606e97d8"
MODELS_SHA = "db08ce67008328e7bbf5fcac40b76b3ef0910d0d202b7811737fa7fd298fdc83"
RELEASE_REVISION = "a5c15469c6caae1328ef777ba141215a8c7c14b5"
ALIGNMENT_SOURCE = "https://gitlab.com/cyp_pred_repos/boltzcyp/-/blob/a1a110a7bf850af7650f12219e8facf04039b659/src/boltzcyp/runscripts/evaluation_baseline.py"
METRICS = ("mrr", "hit_at_1", "hit_at_5", "hit_at_10", "hit_at_10pct",
           "hit_at_20pct", "hit_at_50pct", "mean_rank", "percentile")


def tie_metrics(first, last, size):
    """Exact expectation under a uniformly random ordering of equal scores.

    Top p% uses floor(p*N), matching rank/N <= p, even for very small pools.
    E[1/rank] is NOT 1/E[rank]. Percentile is on a 0..100 scale.
    """
    if not 1 <= first <= last <= size:
        raise ValueError("Invalid tie interval")
    width = last - first + 1
    hit = lambda cutoff: max(0, min(last, cutoff) - first + 1) / width
    result = dict(mrr=sum(1 / i for i in range(first, last + 1)) / width,
                  mean_rank=(first + last) / 2,
                  percentile=100 * (first + last) / (2 * size))
    result.update({f"hit_at_{k}": hit(k) for k in (1, 5, 10)})
    result.update({f"hit_at_{p}pct": hit(p * size // 100) for p in (10, 20, 50)})
    return result


def evaluate(directory, predictions):
    _, old_rows, rankings = evaluate_scores(directory, predictions)
    rows = []
    for old in old_rows:
        first = round(1 / old["optimistic_mrr"])
        last = round(1 / old["pessimistic_mrr"])
        rows.append(dict(query_id=old["query_id"], organism=old["organism"],
                         n_candidates=old["n_candidates"], tie_first=first, tie_last=last,
                         positive_tie_size=last - first + 1,
                         legacy_id_tiebreak_mrr=old["first_positive_mrr"],
                         **tie_metrics(first, last, old["n_candidates"])))
    metrics = {key: float(np.mean([r[key] for r in rows])) for key in METRICS}
    metrics.update(queries=len(rows), positive_ties=sum(r["positive_tie_size"] > 1 for r in rows))
    return metrics, rows, rankings


def paired_intervals(reference, challenger, samples=10000, seed=42):
    """Query-macro deltas; resample organisms with replacement, paired by query ID."""
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    a = {r["query_id"]: r for r in reference}
    b = {r["query_id"]: r for r in challenger}
    if len(a) != len(reference) or len(b) != len(challenger) or a.keys() != b.keys():
        raise ValueError("Bootstrap requires identical, unique query IDs")
    ids = sorted(a)
    if any((a[q]["organism"], a[q]["n_candidates"]) !=
           (b[q]["organism"], b[q]["n_candidates"]) for q in ids):
        raise ValueError("Bootstrap pools/organisms do not match")
    groups = sorted({a[q]["organism"] for q in ids})
    if len(groups) < 2:
        raise ValueError("At least two organism clusters are needed for uncertainty")
    delta = np.array([[b[q][k] - a[q][k] for k in METRICS] for q in ids])
    group_ids = [np.array([i for i, q in enumerate(ids) if a[q]["organism"] == g]) for g in groups]
    # The same resampling weights apply to every metric and model comparison.
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(groups), size=(samples, len(groups)))
    counts = np.zeros((samples, len(groups)), dtype=np.int64)
    np.add.at(counts, (np.arange(samples)[:, None], draws), 1)
    sums = np.stack([delta[index].sum(axis=0) for index in group_ids])
    sizes = np.array([len(index) for index in group_ids])
    boot = counts @ sums / (counts @ sizes)[:, None]
    bounds = np.quantile(boot, [.025, .975], axis=0)
    return {key: dict(delta=float(delta[:, i].mean()), ci95=bounds[:, i].tolist(),
                      organism_clusters=len(groups), samples=samples, seed=seed)
            for i, key in enumerate(METRICS)}


def archive_members(path, names=None, prefix=None, expected_sha=None):
    """Read selected regular files only; never extract links/paths or unpickle."""
    if expected_sha and digest(path) != expected_sha:
        raise ValueError(f"Archive checksum mismatch: {path}")
    found = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not ((names is not None and member.name in names) or
                    (prefix is not None and member.name.startswith(prefix))):
                continue
            if (member.name in found or not member.isfile() or
                    ".." in PurePosixPath(member.name).parts or
                    member.name.startswith("/") or member.size > 128 * 1024**2):
                raise ValueError(f"Unsafe/duplicate/oversized member: {member.name}")
            found.add(member.name)
            yield member.name, archive.extractfile(member).read()
    if names is not None and not set(names) <= found:
        raise ValueError(f"Missing archive members: {set(names) - found}")


def prepare_inputs(archive, bundle, output, expected_sha=DATA_SHA):
    """Recover original molecule order, mapped reactions and train-positive records."""
    from rdkit import Chem

    manifest = verify_bundle(bundle)
    if manifest["archive_sha256"] != digest(archive):
        raise ValueError("Benchmark does not belong to this release")
    output = Path(output)
    queries = {r["reaction_id"]: r for r in read_rows(Path(bundle) / "queries.csv")}
    proteins = {r["protein_id"]: r for r in read_rows(Path(bundle) / "proteins.csv")}
    expected = {(r["query_id"], r["protein_id"]) for r in read_rows(Path(bundle) / "candidates.csv")}
    references, mapped, substrates, seen = [], [], [], set()
    train_prefix = "data_dir/retrain/train_inputs/train_pos/"
    for name, raw in archive_members(archive, prefix=(POOL_PREFIX, train_prefix), expected_sha=expected_sha):
        if name.startswith(POOL_PREFIX) and name.endswith(".csv"):
            qid = PurePosixPath(name).stem
            if qid not in queries:
                raise ValueError(f"Unexpected query {qid}")
            rows = list(csv.DictReader(io.StringIO(raw.decode())))
            raw_rxns = {r["reaction"] for r in rows}
            if len(raw_rxns) != 1:
                raise ValueError(f"Inconsistent mapped reaction: {qid}")
            reaction = raw_rxns.pop()
            if canonical_reaction(reaction) != queries[qid]["rxn"]:
                raise ValueError(f"Changed reaction: {qid}")
            # The authors take the LAST molecule in the ORIGINAL substrate side.
            substrate = reaction.split(">>")[0].split(".")[-1]
            mol = Chem.MolFromSmiles(substrate)
            if mol is None:
                raise ValueError(f"Invalid substrate in {qid}")
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            substrates.append(dict(query_id=qid, substrate=substrate,
                                   unmapped_canonical=Chem.MolToSmiles(mol, canonical=True)))
            for row in rows:
                key = qid, row["protein_id"]
                if key not in expected or key in seen or row["sequence"] != proteins[key[1]]["sequence"]:
                    raise ValueError(f"Changed/duplicate candidate: {key}")
                seen.add(key)
                mapped.append(dict(query_id=qid, **{k: row[k] for k in ("protein_id", "sequence", "reaction", "cif")}))
        elif name.startswith(train_prefix) and name.endswith(".fasta"):
            fields, header = {}, None
            for line in raw.decode().splitlines():
                if line.startswith(">"):
                    header = line[1:].split("|")[:2]
                    header = tuple(header)
                    if header in fields:
                        raise ValueError(f"Repeated training FASTA chain: {name}")
                    fields[header] = ""
                elif line.strip():
                    if header is None:
                        raise ValueError(f"Missing FASTA header: {name}")
                    fields[header] += line.strip()
            sequence, substrate = fields[("A", "protein")], fields[("C", "smiles")]
            sequence_hash(sequence)
            references.append(dict(sample_id=PurePosixPath(name).stem,
                                   protein_id=PurePosixPath(name).stem.split("_")[0],
                                   sequence=sequence, substrate=substrate))
    if seen != expected or len(substrates) != len(queries) or not references:
        raise ValueError("Incomplete query pools or empty CYP train-positive references")
    references.sort(key=lambda r: r["sample_id"])
    for filename, rows in (("references.csv", references), ("mapped_inputs.csv", mapped),
                           ("substrates.csv", substrates)):
        write_rows(output / filename, rows)
    return dict(reference_samples=len(references), reference_proteins=len({r["sequence"] for r in references}),
                reference_scope="Released FusionESP CYP train_pos only; excludes val_pos and train_neg",
                reference_order="sample_id ascending; original BoltzCYP train DataFrame order unavailable",
                structures="cif paths are author-supplied references, NOT locally verified structures")


def make_aligner():
    from Bio.Align import PairwiseAligner, substitution_matrices
    aligner = PairwiseAligner(mode="global")
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11.0
    aligner.extend_gap_score = -1.0
    return aligner


def nearest_substrates(queries, references):
    """Paper fingerprint (RDK maxPath=2, NOT Morgan) and first maximum."""
    from rdkit import Chem, DataStructs
    substrates = list(dict.fromkeys(r["substrate"] for r in references))
    def fingerprint(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid substrate: {smiles}")
        return Chem.RDKFingerprint(mol, maxPath=2)
    fps = [fingerprint(s) for s in substrates]
    result = []
    for row in queries:
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint(row["substrate"]), fps)
        index = int(np.argmax(similarities))
        result.append(dict(query_id=row["query_id"], substrate=substrates[index],
                           tanimoto=similarities[index],
                           tied_nearest_substrates=sum(v == similarities[index] for v in similarities)))
    return result


def _alignment_worker_init(sequences):
    global _ALIGNER, _REFERENCES
    _ALIGNER, _REFERENCES = make_aligner(), sequences


def _align_candidate(sequence):
    return [_ALIGNER.score(sequence, reference) for reference in _REFERENCES]


def alignment_controls(bundle, prepared, output, workers=4):
    """Compute each unique sequence pair once, on CPU; preserve record weighting."""
    if workers < 1:
        raise ValueError("workers must be positive")
    prepared, output = Path(prepared), Path(output)
    references = list(read_rows(prepared / "references.csv"))
    proteins = list(read_rows(Path(bundle) / "proteins.csv"))
    train_sequences = list(dict.fromkeys(r["sequence"] for r in references))
    candidate_sequences = list(dict.fromkeys(r["sequence"] for r in proteins))
    # Check dependencies/alphabet before dispatch so worker initialization cannot
    # hide a missing package or invalid sequence behind BrokenProcessPool.
    alphabet = set(make_aligner().substitution_matrix.alphabet)
    if any(set(s) - alphabet for s in train_sequences + candidate_sequences):
        raise ValueError("A sequence contains residues unsupported by BLOSUM62; no candidates were dropped")
    candidates_by_id = {r["protein_id"]: i for i, r in enumerate(proteins)}
    seq_index = {s: i for i, s in enumerate(candidate_sequences)}
    train_index = {s: i for i, s in enumerate(train_sequences)}
    nearest = nearest_substrates(read_rows(prepared / "substrates.csv"), references)
    write_rows(output / "nearest_substrates.csv", nearest)
    selections = {r["query_id"]: [train_index[t["sequence"]] for t in references
                                 if t["substrate"] == r["substrate"]] for r in nearest}
    matrix = np.empty((len(candidate_sequences), len(train_sequences)), dtype=np.float64)
    with ProcessPoolExecutor(max_workers=workers, initializer=_alignment_worker_init,
                             initargs=(train_sequences,)) as pool:
        for i, values in enumerate(pool.map(_align_candidate, candidate_sequences, chunksize=4)):
            matrix[i] = values
            if (i + 1) % 100 == 0 or i + 1 == len(matrix):
                print(f"Alignment: {i + 1}/{len(matrix)} unique candidates", flush=True)
    rows = {"alignment_sequence": [], "alignment_substrate": []}
    for pair in read_rows(Path(bundle) / "candidates.csv"):
        qid, pid = pair["query_id"], pair["protein_id"]
        values = matrix[seq_index[proteins[candidates_by_id[pid]]["sequence"]]]
        for name, score in (("alignment_sequence", values.max()),
                            ("alignment_substrate", values[selections[qid]].mean())):
            rows[name].append(dict(query_id=qid, protein_id=pid, score=float(score)))
    return rows
