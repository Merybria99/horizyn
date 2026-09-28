"""Validation-only retrieval diagnostics; no model fitting or test-set selection."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("reactzyme_mrr", "first_positive_mrr", "recall_10", "top_10")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_pairs(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    pairs = [(r["reaction_id"], r["protein_id"]) for r in rows]
    if not pairs or len(set(pairs)) != len(pairs) or any(not q or not p for q, p in pairs):
        raise ValueError(f"Empty IDs/pairs or duplicate associations: {path}")
    return pairs


def associations(pairs):
    r2e, e2r = defaultdict(set), defaultdict(set)
    for reaction, enzyme in pairs:
        r2e[reaction].add(enzyme)
        e2r[enzyme].add(reaction)
    return dict(r2e), dict(e2r)


def novelty(queries, training_queries):
    seen = [q in training_queries for q in queries]
    return "seen" if all(seen) else "unseen" if not any(seen) else "mixed"


def reaction_components(pairs):
    """Cluster shared-reaction enzyme queries; bridge multi-activity enzymes too."""
    r2e, e2r = associations(pairs)
    parent = {r: r for r in r2e}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for reactions in e2r.values():
        ordered = sorted(reactions)
        for r in ordered[1:]:
            a, b = sorted((find(ordered[0]), find(r)))
            parent[b] = a
    return ({r: find(r) for r in r2e},
            {p: find(next(iter(rs))) for p, rs in e2r.items()})


def cluster_interval(values, groups, *, seed=42, repeats=500):
    """Query-weighted mean; resample reaction components, not dependent proteins."""
    values = np.asarray(values, dtype=np.float64)
    groups = list(groups)
    if not len(values) or len(values) != len(groups) or not np.isfinite(values).all():
        raise ValueError("Empty, mismatched or nonfinite bootstrap data")
    _, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=values)
    result = dict(mean=float(values.mean()), queries=len(values), clusters=len(counts))
    if len(counts) < 2:
        return dict(result, ci95=None)
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, repeats, 64):
        draws = rng.integers(0, len(counts), size=(min(64, repeats-start), len(counts)))
        estimates.extend((sums[draws].sum(1) / counts[draws].sum(1)).tolist())
    return dict(result, ci95=np.quantile(estimates, [.025, .975]).tolist())


def rank_metrics(scores, positive_indices):
    """Stable candidate-ID order breaks ties; retain all-zero/no-hit queries."""
    scores = np.asarray(scores)
    positive_indices = np.unique(positive_indices)
    if not len(positive_indices) or not np.isfinite(scores).all():
        raise ValueError("Missing positives or nonfinite scores")
    order = np.argsort(-scores, kind="stable")
    relevant = np.zeros(len(scores), dtype=bool)
    relevant[positive_indices] = True
    ranks = np.flatnonzero(relevant[order]) + 1
    return dict(reactzyme_mrr=float(np.mean(1 / ranks)),
                first_positive_mrr=float(1 / ranks[0]),
                recall_10=float(np.mean(ranks <= 10)), top_10=float(ranks[0] <= 10),
                positive_count=len(ranks), candidate_count=len(scores),
                all_scores_tied=bool(np.ptp(scores) == 0))


def score_matrix_rows(matrix, reactions, enzymes, pairs):
    r2e, e2r = associations(pairs)
    qi, pi = {x:i for i,x in enumerate(reactions)}, {x:i for i,x in enumerate(enzymes)}
    rows = []
    for q in sorted(r2e):
        rows.append(dict(direction="reaction_to_enzyme", query_id=q,
                         **rank_metrics(matrix[qi[q]], [pi[p] for p in r2e[q]])))
    for p in sorted(e2r):
        rows.append(dict(direction="enzyme_to_reaction", query_id=p,
                         **rank_metrics(matrix[:, pi[p]], [qi[q] for q in e2r[p]])))
    return rows


def fingerprints(smiles):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048,
                                                          includeChirality=True)
    result = []
    for value in smiles:
        # ReactZyme releases unordered molecule sets. Do not invent reaction sides.
        molecule = Chem.MolFromSmiles(value)
        result.append(generator.GetFingerprint(molecule) if molecule is not None else None)
    return result


def chemistry_similarity(training_smiles, validation_smiles):
    from rdkit import DataStructs
    train, valid = fingerprints(training_smiles), fingerprints(validation_smiles)
    result = np.zeros((len(valid), len(train)), dtype=np.float32)
    usable = [i for i, fp in enumerate(train) if fp is not None]
    if not usable:
        raise ValueError("No parseable training chemistry")
    refs = [train[i] for i in usable]
    for i, fp in enumerate(valid):
        if fp is not None:
            result[i, usable] = DataStructs.BulkTanimotoSimilarity(fp, refs)
    return result, np.array([x is not None for x in train]), np.array([x is not None for x in valid])


def read_sequence_hits(path, train_proteins, validation_proteins, min_coverage=.8):
    """Best *retrieved* identity, not an exhaustive global-alignment guarantee."""
    hits = defaultdict(list)
    with Path(path).open() as handle:
        for line in handle:
            query, target, ident, qcov, tcov, bits = line.rstrip().split("\t")
            if query not in validation_proteins or target not in train_proteins:
                raise ValueError("Sequence search contains IDs outside validation/train")
            ident, qcov, tcov, bits = map(float, (ident, qcov, tcov, bits))
            if not all(np.isfinite([ident, qcov, tcov, bits])) or not 0 <= ident <= 1:
                raise ValueError("Expected finite fractional MMseqs fident")
            if qcov >= min_coverage and tcov >= min_coverage:
                hits[query].append((target, ident, bits))
    return {p: sorted(v, key=lambda x: (-x[2], x[0])) for p,v in hits.items()}


def transfer_baselines(similarity, train_reactions, reactions, enzymes, train_pairs, hits,
                       valid_chemistry=None):
    """Two opposite-direction association-transfer heuristics; training edges only.

    Reaction-first picks nearest training chemistry then uses retrieved sequence
    identities to its annotated enzymes. Enzyme-first picks the highest-bit-score
    training enzyme then uses chemistry similarity to its annotated reactions.
    No search hit yields all-zero scores, never an invented identity or dropped query.
    """
    _, e2r = associations(train_pairs)
    ti = {r:i for i,r in enumerate(train_reactions)}
    nearest = np.argmax(similarity, axis=1)
    reaction_first = np.zeros((len(reactions),len(enzymes)), dtype=np.float32)
    enzyme_first = np.zeros_like(reaction_first)
    for j, p in enumerate(enzymes):
        phits = hits.get(p, [])
        if not phits:
            continue
        best = phits[0][0]
        enzyme_first[:,j] = similarity[:,[ti[r] for r in e2r[best]]].max(1)
        supported = {}
        for target, ident, _ in phits:
            for r in e2r[target]:
                idx = ti[r]
                supported[idx] = max(supported.get(idx,0), ident)
        reaction_first[:,j] = [supported.get(int(i),0) for i in nearest]
    # With no chemical overlap there is no defensible nearest association.
    reaction_first[similarity.max(axis=1) == 0] = 0
    if valid_chemistry is not None:
        reaction_first[~valid_chemistry] = 0
        enzyme_first[~valid_chemistry] = 0
    return {"reaction_neighbor_transfer":reaction_first, "enzyme_neighbor_transfer":enzyme_first}


def minimum_known(values):
    values = [v for v in values if v is not None]
    return min(values) if values else None


def decorate_rows(rows, train_pairs, validation_pairs, reaction_metadata, protein_metadata):
    train_r2e, _ = associations(train_pairs)
    r2e, e2r = associations(validation_pairs)
    rc, pc = reaction_components(validation_pairs)
    output = []
    for row in rows:
        row = dict(row)
        q = row["query_id"]
        if row["direction"] == "reaction_to_enzyme":
            if q not in r2e and q.endswith("_f"):
                q = q[:-2]
            qs, ps, cluster = [q], sorted(r2e[q]), rc[q]
        else:
            qs, ps, cluster = sorted(e2r[q]), [q], pc[q]
        row.update(query_id=q, cluster=cluster, novelty=novelty(qs, train_r2e),
                   chemistry_similarity=minimum_known([reaction_metadata[r].get("similarity") for r in qs]),
                   sequence_identity=minimum_known([protein_metadata.get(p,{}).get("identity") for p in ps]),
                   sequence_hit_coverage=sum(protein_metadata.get(p,{}).get("identity") is not None for p in ps)/len(ps),
                   auxiliary_target_coverage=sum(protein_metadata.get(p,{}).get("auxiliary",False) for p in ps)/len(ps),
                   missing_modality=any(reaction_metadata[r].get("missing_modality",False) for r in qs))
        # For R->E these are ground-truth-set descriptors, never scoring inputs.
        output.append(row)
    return output


def band(value, edges):
    if value is None:
        return "unknown"
    for lower, upper in zip(edges, edges[1:]):
        if lower <= value < upper:
            return f"{lower:g}-{min(upper,1):g}"
    raise ValueError(f"Value outside bands: {value}")


def row_strata(row):
    chem = band(row["chemistry_similarity"], [0,.4,.6,.8,1.000001])
    seq = band(row["sequence_identity"], [0,.3,.5,.8,1.000001])
    n = row["positive_count"]
    return ["all", "reaction/"+row["novelty"], "chemistry/"+chem,
            "sequence/"+seq, "joint/"+chem+"/"+seq,
            "positives/"+("1" if n==1 else "2-10" if n<=10 else ">10"),
            "auxiliary/"+("none" if row["auxiliary_target_coverage"]==0 else "some"),
            "modalities/"+("missing" if row["missing_modality"] else "complete"),
            "sequence_hits/"+("complete" if row["sequence_hit_coverage"]==1 else "incomplete")]


def summarize(method_rows, repeats=500):
    """Report per-direction strata and paired cluster-bootstrap method deltas."""
    summaries, differences = [], []
    indexed = {}
    for method, rows in method_rows.items():
        indexed[method] = {(r['direction'],r['query_id']):r for r in rows}
        if len(indexed[method]) != len(rows):
            raise ValueError("Duplicate per-query records")
        groups = defaultdict(list)
        for row in rows:
            for stratum in row_strata(row):
                groups[(row['direction'],stratum)].append(row)
        for (direction,stratum), selected in sorted(groups.items()):
            for metric in METRICS:
                summaries.append(dict(method=method,direction=direction,stratum=stratum,metric=metric,
                    **cluster_interval([r[metric] for r in selected], [r['cluster'] for r in selected], repeats=repeats),
                    all_tied_queries=sum(r.get('all_scores_tied',False) for r in selected)))
    references = ['none'] if 'none' in indexed else []
    if 'cls002' in indexed and 'cls005' in indexed:
        references.append('cls002')
    for reference in references:
        for method, mapping in indexed.items():
            if method==reference or (reference=='cls002' and method!='cls005'): continue
            if mapping.keys()!=indexed[reference].keys():
                raise ValueError(f"Query coverage differs: {method}")
            for direction in ('reaction_to_enzyme','enzyme_to_reaction'):
                for stratum in ('all','reaction/seen','reaction/unseen','reaction/mixed'):
                    keys=[k for k,r in mapping.items() if k[0]==direction and stratum in row_strata(r)]
                    if not keys:continue
                    for metric in METRICS:
                        differences.append(dict(method=method,reference=reference,direction=direction,stratum=stratum,metric=metric,
                            **cluster_interval([mapping[k][metric]-indexed[reference][k][metric] for k in keys],
                                               [mapping[k]['cluster'] for k in keys],repeats=repeats)))
    return dict(summaries=summaries,paired_differences=differences,
                uncertainty='Percentile reaction-component cluster bootstrap; fixed trained models/candidate pools; not training-seed uncertainty')
