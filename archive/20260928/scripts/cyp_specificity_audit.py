"""Training-relative exposure audit and reaction-conditioned MMseqs baseline.

No-hit and untraced upstream supervision are UNKNOWN, never proof of novelty.
The baseline only uses the explicitly configured reference source's positive edges.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from scripts.cyp_specificity import (canonical_reaction, digest, read_fasta, read_rows,
                                     reaction_key, sequence_hash, write_json, write_rows)


def protein_key(value):
    value = str(value).strip()
    if value.startswith(("sp|", "tr|")):
        value = value.split("|")[1]
    for prefix in ("prot_", "uprot_", "nr90_"):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def positive_pairs(path):
    for row in read_rows(path):
        label = row.get("label", row.get("y", "1"))
        if str(label).strip() in {"0", "0.0", "False", "false"}:
            continue
        if str(label).strip() not in {"1", "1.0", "True", "true"}:
            raise ValueError(f"Unknown pair label {label!r} in {path}")
        yield row["reaction_id"], protein_key(row["protein_id"])


def sequences_from(path):
    """Use a FASTA or the sequences actually recorded in training-pair tables."""
    if Path(path).suffix == ".csv":
        for row in read_rows(path):
            yield row["protein_id"], "".join(row["protein_sequence"].split()).upper()
    else:
        yield from read_fasta(path)


def fingerprint(smiles):
    """Direction-aware side-concatenated OR Morgan-2 proxy; NOT reaction mechanism."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    joined = DataStructs.ExplicitBitVect(4096)
    for side, text in enumerate(smiles.split(">>")):
        for molecule in text.split("."):
            for bit in generator.GetFingerprint(Chem.MolFromSmiles(molecule)).GetOnBits():
                joined.SetBit(side * 2048 + bit)
    return joined


def nearest_reactions(queries, reactions):
    from rdkit import DataStructs

    ids = sorted(reactions)
    fps = [fingerprint(reactions[r]) for r in ids]
    result = {}
    for q in queries:
        similarities = np.array(DataStructs.BulkTanimotoSimilarity(fingerprint(q["rxn"]), fps))
        maximum = float(similarities.max())
        result[q["reaction_id"]] = dict(similarity=maximum,
                                       reaction_ids=[ids[i] for i in np.flatnonzero(similarities == maximum)])
    return result


def align(mmseqs, query, reference, output, threads):
    output = Path(output)
    if not output.exists():
        # Interrupted MMseqs output must never be mistaken for a completed search.
        with tempfile.TemporaryDirectory(prefix="mmseqs_", dir=output.parent) as temporary:
            result = Path(temporary) / "hits.tsv"
            subprocess.run([str(mmseqs), "easy-search", str(query), str(reference), str(result),
                            str(Path(temporary) / "work"), "--threads", str(threads), "-s", "7.5",
                            "-e", "0.001", "--max-seqs", "10000", "--alignment-mode", "3",
                            "--format-output", "query,target,fident,qcov,tcov,bits"], check=True)
            result.replace(output)
    hits = defaultdict(list)
    with output.open() as stream:
        for line in stream:
            q, t, identity, qcov, tcov, bits = line.rstrip().split("\t")
            hits[q].append(dict(target=t, identity=float(identity), qcov=float(qcov),
                                tcov=float(tcov), bits=float(bits)))
    return hits


def audit_source(bundle, spec, directory, mmseqs, threads=4, homology=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    inputs = {key: dict(path=str(Path(spec[key]).resolve()), sha256=digest(spec[key]))
              for key in ("pairs", "reactions", "proteins")}
    signature = dict(inputs=inputs, benchmark_sha256=digest(Path(bundle) / "manifest.json"),
                     homology=homology, mmseqs=str(Path(mmseqs).resolve()),
                     audit_code_sha256=digest(__file__), mmseqs_sha256=digest(mmseqs),
                     chemistry_code_sha256=digest(Path(__file__).with_name("cyp_specificity.py")),
                     algorithm="cyp_audit_v1_morgan2_chiral4096_mmseqs_s7p5_e1e-3_max10000")
    receipt = directory / "complete.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous["signature"] != signature:
            raise ValueError(f"Audit inputs changed; use a new run directory: {directory}")
        for name, sha in previous["outputs"].items():
            if digest(directory / name) != sha:
                raise ValueError(f"Audit artifact changed: {name}")
        return previous
    pending = directory / "inputs.json"
    if pending.exists() and json.loads(pending.read_text()) != signature:
        raise ValueError("Cannot resume an audit with changed inputs")
    write_json(pending, signature)
    queries = list(read_rows(Path(bundle) / "queries.csv"))
    proteins = {r["protein_id"]: r for r in read_rows(Path(bundle) / "proteins.csv")}
    graph = defaultdict(set)
    for reaction, protein in positive_pairs(spec["pairs"]):
        graph[reaction].add(protein)
    reactions, invalid, representations = {}, [], set()
    for row in read_rows(spec["reactions"]):
        rid = row["reaction_id"]
        if rid not in graph:
            continue
        text = row.get("rxn", row.get("reaction_smiles", ""))
        directed = ">>" in text
        representations.add("directed" if directed else "participant_set")
        try:
            canonical = canonical_reaction(text if directed else f"{text}>>{text}")
        except ValueError as exc:
            invalid.append(dict(reaction_id=rid, error=str(exc)))
            continue
        if rid in reactions and reactions[rid] != canonical:
            raise ValueError(f"Conflicting source reaction: {rid}")
        reactions[rid] = canonical
    if not reactions:
        raise ValueError("No parseable training reactions")
    if len(representations) != 1:
        raise ValueError("Mixed directed/participant-set source; define separate reference sources")
    representation = representations.pop()
    reference_queries = queries
    if representation == "participant_set":
        # ReactZyme lacks side assignments. Compare like with like, without
        # falsely interpreting participant equality as transformation equality.
        reference_queries = [dict(q, rxn=canonical_reaction(
            f"{q['rxn'].replace('>>', '.')}>>{q['rxn'].replace('>>', '.')}")) for q in queries]
    nearest = nearest_reactions(reference_queries, reactions)
    baseline_ids = {p for hit in nearest.values() for rid in hit["reaction_ids"] for p in graph[rid]}
    wanted = {p for values in graph.values() for p in values}
    query_hashes = {r["sha256"] for r in proteins.values()}
    exact = defaultdict(set)
    seen, baseline_seen, sequences = set(), set(), {}
    with (directory / "training.fasta").open("w") as train, (directory / "baseline_reference.fasta").open("w") as baseline:
        for raw_id, seq in sequences_from(spec["proteins"]):
            pid = protein_key(raw_id)
            if pid not in wanted:
                continue
            sha = sequence_hash(seq)
            if pid in seen:
                if sequences[pid] != sha:
                    raise ValueError(f"Conflicting training sequences for {pid}")
                continue
            seen.add(pid)
            sequences[pid] = sha
            train.write(f">{pid}\n{seq}\n")
            if pid in baseline_ids:
                baseline_seen.add(pid)
                baseline.write(f">{pid}\n{seq}\n")
            if sha in query_hashes:
                exact[sha].add(pid)
    if wanted - seen:
        raise ValueError(f"Missing {len(wanted - seen)} training sequences, e.g. {sorted(wanted - seen)[:5]}")
    if baseline_ids != baseline_seen:
        raise ValueError("Baseline reference enzymes missing")
    baseline_hits = align(mmseqs, Path(bundle) / "proteins.fasta", directory / "baseline_reference.fasta",
                          directory / "baseline_hits.tsv", threads)
    all_hits = align(mmseqs, Path(bundle) / "proteins.fasta", directory / "training.fasta",
                     directory / "homology.tsv", threads) if homology else {}
    audit_proteins = []
    for pid, record in proteins.items():
        eligible = [h for h in all_hits.get(pid, []) if h["qcov"] >= .8 and h["tcov"] >= .8]
        hit = max(eligible, key=lambda h: h["identity"], default=None)
        audit_proteins.append(dict(protein_id=pid, exact_sequence_matches=";".join(sorted(exact[record["sha256"]])),
                                   best_reported_identity=hit["identity"] if hit else "",
                                   homology_status="reported_coverage80_hit" if hit else ("no_reported_coverage80_hit" if homology else "not_run")))
    keys = defaultdict(list)
    for rid, smiles in reactions.items():
        keys[reaction_key(smiles)].append(rid)
    q_audit = []
    for q in reference_queries:
        qid = q["reaction_id"]
        matching = keys[reaction_key(q["rxn"])]
        gold_hash = proteins[q["positive_id"]]["sha256"]
        known_pair = any(graph[rid] & exact[gold_hash] for rid in matching)
        q_audit.append(dict(query_id=qid, reaction_representation=representation,
                            exact_reaction_matches=";".join(sorted(matching)) if representation == "directed" else "",
                            participant_set_matches=";".join(sorted(matching)) if representation == "participant_set" else "",
                            positive_exact_sequence_match=bool(exact[gold_hash]),
                            exact_pair_match=known_pair if representation == "directed" else "UNKNOWN",
                            participant_set_pair_match=known_pair if representation == "participant_set" else "",
                            nearest_reaction_similarity=nearest[qid]["similarity"],
                            nearest_reaction_ids=";".join(nearest[qid]["reaction_ids"])))
    predictions = []
    for r in read_rows(Path(bundle) / "candidates.csv"):
        refs = {p for rid in nearest[r["query_id"]]["reaction_ids"] for p in graph[rid]}
        hits = [h for h in baseline_hits.get(r["protein_id"], []) if h["target"] in refs]
        predictions.append(dict(query_id=r["query_id"], protein_id=r["protein_id"],
                                score=max((h["bits"] for h in hits), default=0.0),
                                reported_hit=bool(hits)))
    write_rows(directory / "proteins.csv", audit_proteins)
    write_rows(directory / "queries.csv", q_audit)
    write_rows(directory / "homology_baseline.csv", predictions)
    write_rows(directory / "invalid_reactions.csv", invalid, ["reaction_id", "error"])
    result = dict(signature=signature, training_proteins=len(seen), valid_reactions=len(reactions),
                  reaction_representation=representation,
                  invalid_reactions=len(invalid), upstream_supervised_exposure="UNKNOWN",
                  caveats=["No reported homology hit is not proof of low identity",
                           "Canonical exact matches retain all participants; cofactor/proton differences can hide equivalent transformations",
                           "Nearest chemistry uses a participant fingerprint proxy, not mechanistic equivalence",
                           "Participant-set sources have no side assignments: exact transformation/pair exposure is UNKNOWN",
                           "Baseline uses maximum reported MMseqs bitscore to enzymes of the nearest training reaction(s); no hit scores zero",
                           "Explicit sources do not certify absence from upstream SLEEC or other supervised pretraining"],
                  outputs={name: digest(directory / name) for name in
                           ("proteins.csv", "queries.csv", "homology_baseline.csv", "invalid_reactions.csv")})
    write_json(receipt, result)
    return result
