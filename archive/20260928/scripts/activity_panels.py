"""Measured within-family activity panels: immutable inputs and tie-exact metrics.

No assay labels are used to construct features or choose checkpoints. Nitrilase
products are template hypotheses, not experimentally resolved product labels.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import urllib.request

import numpy as np

SCHEMA = "measured_activity_panel_v1"
REVISION = "627556e265e2a52d39753e684c05b550d53a9be4"
SOURCES = {
    "data/processed/nitrilase_binary.csv": "879af177bc6820eb0c6f6a7192ddfa0c46b177c6ee57ff143a2d68f48ddf5887",
    "data/raw/nitrilase/nitrilase_data.xlsx": "b49d1f997d29d5c86bf58c00f329b1bdcca08ec28aba2c02007b0e3501b66f5c",
    "bin/reformat_nitrilase.py": "99908ddf4c43614fb52d58591e4c9d1cfb126ba7c9a3495a34b38fd6c4c76763",
    "README.md": "e5fe08d5010bde301815781d4484024289d19d4614a121d53e82207a40ebe500",
}
NITRILASE_SMARTS = "[C,c:1][C:2]#[N:3].[OH2:4].[OH2:5]>>[C,c:1][C:2](=[O:4])[OH:5].[NH3:3]"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, records, fields=None):
    records = list(records); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(records[0]) if records else None)
    if not fields: raise ValueError("Empty table needs an explicit schema")
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    tmp.replace(path)


def sequence_hash(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


def canonical_molecule(smiles):
    from rdkit import Chem
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None: raise ValueError(f"Invalid molecule: {smiles}")
    for atom in molecule.GetAtoms(): atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def nitrilase_reactions(substrate):
    """Enumerate distinct single-nitrile hydrolyses without any activity labels."""
    from rdkit import Chem
    from rdkit.Chem import rdChemReactions
    substrate = canonical_molecule(substrate)
    rxn = rdChemReactions.ReactionFromSmarts(NITRILASE_SMARTS)
    outputs = set()
    for products in rxn.RunReactants((Chem.MolFromSmiles(substrate), Chem.MolFromSmiles("O"), Chem.MolFromSmiles("O"))):
        for molecule in products: Chem.SanitizeMol(molecule)
        product = ".".join(sorted(canonical_molecule(Chem.MolToSmiles(m)) for m in products))
        outputs.add(f"{substrate}.O.O>>{product}")
    if not outputs: raise ValueError(f"No nitrile hydrolysis can be constructed: {substrate}")
    return sorted(outputs)


def verify_bundle(bundle):
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA: raise ValueError("Wrong activity-panel schema")
    for name, sha in manifest["files"].items():
        if Path(name).name != name or digest(bundle / name) != sha:
            raise ValueError(f"Changed panel input: {name}")
    return manifest


def prepare_nitrilase(release, bundle):
    import openpyxl
    release, bundle = Path(release), Path(bundle)
    release.mkdir(parents=True, exist_ok=True); bundle.mkdir(parents=True, exist_ok=True)
    for filename, expected in SOURCES.items():
        path = release / Path(filename).name
        if not path.exists():
            url = f"https://raw.githubusercontent.com/samgoldman97/enzyme-datasets/{REVISION}/{filename}"
            data = urllib.request.urlopen(url, timeout=60).read()
            if hashlib.sha256(data).hexdigest() != expected: raise ValueError(f"Wrong download: {filename}")
            tmp = path.with_suffix(path.suffix + ".partial"); tmp.write_bytes(data); tmp.replace(path)
        if digest(path) != expected: raise ValueError(f"Source checksum mismatch: {filename}")
    signature = dict(revision=REVISION, sources=SOURCES, code_sha256=digest(__file__))
    if (bundle / "manifest.json").exists():
        old = verify_bundle(bundle)
        if old["preparation"] != signature: raise ValueError("Preparation changed; use a new bundle")
        return old
    raw = rows(release / "nitrilase_binary.csv")
    if len(raw) != 684: raise ValueError("Unexpected source pair count")
    proteins = sorted({r["SEQ"] for r in raw}, key=sequence_hash)
    if len(proteins) != 18 or any(not s or set(s) - set("ACDEFGHIKLMNPQRSTVWYUXBZJO") for s in proteins):
        raise ValueError("Invalid source protein catalog")
    pids = {s: "nit_p_" + sequence_hash(s)[:16] for s in proteins}
    substrates = sorted({canonical_molecule(r["SUBSTRATES"]) for r in raw})
    if len(substrates) != 38: raise ValueError("Unexpected canonical substrate count")
    qids = {s: "nit_r_" + hashlib.sha256(s.encode()).hexdigest()[:16] for s in substrates}
    labels = {}; source_profiles = defaultdict(dict)
    for r in raw:
        if r["Conversion"] not in {"0", "1"}: raise ValueError("Unknown labels are not negatives")
        substrate = canonical_molecule(r["SUBSTRATES"]); label = int(r["Conversion"])
        key = qids[substrate], pids[r["SEQ"]]
        if key in labels: raise ValueError("Duplicate source pair")
        labels[key] = label; source_profiles[r["SEQ"]][substrate] = label
    if set(labels) != {(q, p) for q in qids.values() for p in pids.values()}:
        raise ValueError("Panel is not fully measured")
    # The public parser maps any value not >0, including NaN, to zero. Verify the
    # underlying curated workbook independently before accepting its zeros.
    workbook = openpyxl.load_workbook(release / "nitrilase_data.xlsx", data_only=True, read_only=True)
    values = list(workbook.active.values); workbook.close()
    accessions = values[0][3:]; workbook_profiles = {a: {} for a in accessions}; names = {}
    for row in values[1:]:
        if not isinstance(row[1], str): continue
        substrate = canonical_molecule(row[2]); names[substrate] = row[1]
        for accession, value in zip(accessions, row[3:]):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Missing/invalid assay value; cannot classify as inactive")
            workbook_profiles[accession][substrate] = int(value > 0)
    profile = lambda d: tuple(d[s] for s in substrates)
    if Counter(map(profile, source_profiles.values())) != Counter(map(profile, workbook_profiles.values())):
        raise ValueError("Released labels disagree with underlying workbook")
    # Profiles can be identical; never infer accession-to-sequence IDs from labels.
    protein_records = [dict(protein_id=pids[s], sequence=s, sha256=sequence_hash(s), length=len(s)) for s in proteins]
    queries, variants = [], []
    for substrate in substrates:
        qid = qids[substrate]; reactions = nitrilase_reactions(substrate)
        queries.append(dict(query_id=qid, substrate_smiles=substrate, substrate_name=names[substrate], variant_count=len(reactions)))
        for i, reaction in enumerate(reactions, 1):
            variants.append(dict(reaction_id=f"{qid}_v{i:02d}", query_id=qid, reaction_smiles=reaction))
    candidates = [dict(query_id=q, protein_id=p, label=v) for (q, p), v in sorted(labels.items())]
    write_rows(bundle / "proteins.csv", protein_records); write_rows(bundle / "queries.csv", queries)
    write_rows(bundle / "reactions.csv", variants); write_rows(bundle / "candidates.csv", candidates)
    # These rows define encoder coverage only; they do not train the model.
    write_rows(bundle / "encoding_pairs.csv", [dict(reaction_id=r["reaction_id"], protein_id=p["protein_id"]) for r in variants for p in protein_records])
    (bundle / "proteins.fasta").write_text("".join(f">{r['protein_id']}\n{r['sequence']}\n" for r in protein_records))
    manifest = dict(schema=SCHEMA, panel="nitrilase", preparation=signature, query_count=len(queries),
        protein_count=len(proteins), candidate_pairs=len(candidates), active_pairs=sum(labels.values()),
        inactive_pairs=len(labels)-sum(labels.values()), reaction_variant_count=len(variants),
        all_inactive_reactions=sum(not any(v for (q,p),v in labels.items() if q==qid) for qid in qids.values()),
        all_inactive_enzymes=sum(not any(v for (q,p),v in labels.items() if p==pid) for pid in pids.values()),
        missing_measurements=0, source_doi="10.1039/C4CC06021K",
        label_definition="Released Conversion 1 iff curated ammonia signal >0; zero is no detectable activity under assay conditions, not universal inactivity.",
        reaction_policy=dict(smarts=NITRILASE_SMARTS, aggregation="maximum score across fixed single-hydrolysis variants", products="template-derived; not product-selectivity ground truth"),
        accession_policy="Stable sequence hashes; ambiguous activity profiles are never used to infer accession IDs",
        files={p.name:digest(p) for p in sorted(bundle.iterdir()) if p.suffix in {".csv", ".fasta"}})
    write_json(bundle / "manifest.json", manifest)
    return manifest


def validate_scores(bundle, predictions):
    verify_bundle(bundle)
    expected = {(r["query_id"], r["protein_id"]) for r in rows(Path(bundle)/"candidates.csv")}
    scores = {}
    for r in predictions:
        key = r["query_id"], r["protein_id"]
        if key in scores: raise ValueError(f"Duplicate score: {key}")
        score = float(r["score"])
        if not math.isfinite(score): raise ValueError("Nonfinite score")
        scores[key] = score
    if set(scores) != expected: raise ValueError(f"Score coverage mismatch: missing={len(expected-set(scores))}, extra={len(set(scores)-expected)}")
    return scores


def ranking_metrics(labels, scores, budgets=(1,3,5,8,24)):
    """Expected metrics under uniform random ordering within exact-score ties.

    AP is expectation over item-level permutations, not sklearn's grouped PR AP.
    Queries with no positives have AP/Hit/recall zero and are explicitly counted.
    """
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or len(labels)==0 or labels.shape != scores.shape: raise ValueError("Invalid metric shapes")
    if not np.isin(labels, [0,1]).all() or not np.isfinite(scores).all(): raise ValueError("Invalid metric values")
    n, positives = len(labels), int(labels.sum()); ap = mrr = 0.0
    active_at = {min(int(k),n):0.0 for k in budgets}; hit_at = dict.fromkeys(active_at,0.0)
    if min(budgets) < 1: raise ValueError("Positive budgets required")
    before = positive_before = 0
    for score in sorted(set(scores), reverse=True):
        selected = labels[scores == score]; m, p = len(selected), int(selected.sum())
        for j in range(1,m+1):
            expected_prior = (j-1)*(p-1)/(m-1) if m>1 and p else 0.0
            ap += (p/m)*(positive_before+1+expected_prior)/(before+j)
        if positive_before==0 and p:
            denom=math.comb(m,p)
            mrr = sum(math.comb(m-j,p-1)/denom/(before+j) for j in range(1,m-p+2))
        for k in active_at:
            take=max(0,min(m,k-before)); active_at[k] += take*p/m
            if positive_before and k>before: hit_at[k]=1.0
            elif take and p:
                hit_at[k] = 1.0 if take>m-p else 1-math.comb(m-p,take)/math.comb(m,take)
        before += m; positive_before += p
    result = dict(candidates=n, positives=positives, prevalence=positives/n,
                  ap=ap/positives if positives else 0.0, mrr_first=mrr,
                  no_positives=int(positives==0), all_positive=int(positives==n))
    for requested in budgets:
        k=min(requested,n)
        result.update({f"k_used_{requested}":k, f"hit_at_{requested}":hit_at[k],
                       f"active_at_{requested}":active_at[k], f"precision_at_{requested}":active_at[k]/k,
                       f"recall_at_{requested}":active_at[k]/positives if positives else 0.0})
    return result


def evaluate(bundle, predictions):
    scores=validate_scores(bundle,predictions); records=rows(Path(bundle)/"candidates.csv")
    ever_active={r["protein_id"] for r in records if int(r["label"])}
    result=[]
    for direction,key,subset in [("r2e","query_id","full"),("e2r","protein_id","full"),
                                 ("r2e","query_id","ever_active_enzymes_diagnostic")]:
        groups=defaultdict(list)
        for r in records:
            if subset!="full" and r["protein_id"] not in ever_active: continue
            groups[r[key]].append(r)
        for query,group in sorted(groups.items()):
            metrics=ranking_metrics([int(r["label"]) for r in group], [scores[r["query_id"],r["protein_id"]] for r in group])
            result.append(dict(direction=direction,subset=subset,query_id=query,**metrics))
    return result


def summarize(per_query):
    result={}
    for direction,subset in sorted({(r["direction"],r["subset"]) for r in per_query}):
        group=[r for r in per_query if (r["direction"],r["subset"])==(direction,subset)]
        eligible=[r for r in group if r["positives"]>0]
        keys=[k for k in group[0] if k not in {"direction","subset","query_id"}]
        result[f"{direction}/{subset}"]=dict(query_count=len(group),positive_query_count=len(eligible),
            no_positive_query_count=len(group)-len(eligible),
            all_queries={k:float(np.mean([r[k] for r in group])) for k in keys},
            positive_queries_only={k:float(np.mean([r[k] for r in eligible])) for k in keys} if eligible else None)
    return result
