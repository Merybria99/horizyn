"""Exact supervised exposure and training-only chemistry/sequence transfer.

The large sequence catalog is streamed; only panel matches and nearest-reaction
reference proteins are retained. General homology novelty is not asserted.
"""
from collections import defaultdict
from pathlib import Path
import json

from scripts.activity_panels import digest, rows, write_json, write_rows, sequence_hash, verify_bundle
from scripts.cyp_specificity import canonical_reaction, reaction_key, read_fasta, read_rows
from scripts.cyp_specificity_audit import fingerprint, positive_pairs, sequences_from, protein_key, align


def audit(bundle, spec, output, mmseqs, threads=4):
    from rdkit import DataStructs
    bundle, output = Path(bundle), Path(output); output.mkdir(parents=True,exist_ok=True)
    verify_bundle(bundle)
    signature=dict(bundle=digest(bundle/"manifest.json"),source={k:dict(path=str(v),sha256=digest(v)) for k,v in spec.items()},
                   audit_code=digest(__file__),mmseqs=digest(mmseqs),
                   helpers={p:digest(Path(__file__).with_name(p)) for p in ["activity_panels.py","cyp_specificity.py","cyp_specificity_audit.py"]})
    if (output/"complete.json").exists():
        old=json.loads((output/"complete.json").read_text())
        if old["signature"]!=signature: raise ValueError("Changed audit inputs; use a new run root")
        for name,sha in old["files"].items():
            if digest(output/name)!=sha: raise ValueError("Changed audit output")
        return old
    variants=rows(bundle/"reactions.csv"); proteins=rows(bundle/"proteins.csv")
    chemical={}; invalid=[]; representations=set()
    for row in read_rows(spec["reactions"]):
        text=row.get("rxn",row.get("reaction_smiles","")); directed=">>" in text
        representations.add("directed" if directed else "participant_set")
        try: chemical[row["reaction_id"]]=canonical_reaction(text if directed else f"{text}>>{text}")
        except Exception as e: invalid.append(dict(reaction_id=row["reaction_id"],error=str(e)))
    if len(representations)!=1 or not chemical: raise ValueError("Invalid or mixed source chemistry")
    representation=representations.pop(); chemical_ids=sorted(chemical)
    fps=[fingerprint(chemical[r]) for r in chemical_ids]
    key_ids=defaultdict(set)
    for rid,text in chemical.items(): key_ids[reaction_key(text)].add(rid)
    closest={}; exact_rxns=defaultdict(set)
    for variant in variants:
        text=variant["reaction_smiles"]
        if representation=="participant_set":
            text=text.replace(">>","."); text=f"{text}>>{text}"
        canonical=canonical_reaction(text); q=variant["query_id"]
        values=DataStructs.BulkTanimotoSimilarity(fingerprint(canonical),fps); best=max(values)
        tied={rid for rid,v in zip(chemical_ids,values) if v==best}
        if q not in closest or best>closest[q]["similarity"]: closest[q]=dict(similarity=best,reaction_ids=tied)
        elif best==closest[q]["similarity"]: closest[q]["reaction_ids"].update(tied)
        exact_rxns[q].update(key_ids[reaction_key(canonical)])
    needed={rid for c in closest.values() for rid in c["reaction_ids"]}
    graph=defaultdict(set)
    print("Collecting training-only nearest-reaction associations",flush=True)
    for rid,pid in positive_pairs(spec["pairs"]):
        if rid in needed: graph[rid].add(pid)
    for q,c in closest.items():
        if any(not graph[rid] for rid in c["reaction_ids"]):
            raise ValueError(f"Nearest reaction lacks positive training edges: {q}")
    reference_ids={pid for ids in graph.values() for pid in ids}
    panel_hashes={p["sha256"] for p in proteins}; exact=defaultdict(set); ref_seen=set()
    print("Streaming source sequences for exact exposure and reference extraction",flush=True)
    with (output/"reference.fasta").open("w") as dest:
        for raw_id,seq in sequences_from(spec["proteins"]):
            pid=protein_key(raw_id); sha=sequence_hash(seq)
            if sha in panel_hashes: exact[sha].add(pid)
            if pid in reference_ids and pid not in ref_seen:
                dest.write(f">{pid}\n{seq}\n"); ref_seen.add(pid)
    if ref_seen!=reference_ids: raise ValueError("Missing baseline reference proteins")
    matching_ids={p for ids in exact.values() for p in ids}; observed=defaultdict(set)
    for rid,pid in positive_pairs(spec["pairs"]):
        if pid in matching_ids: observed[pid].add(rid)
    exact={sha:{p for p in ids if p in observed} for sha,ids in exact.items()}
    hits=align(mmseqs,bundle/"proteins.fasta",output/"reference.fasta",output/"alignment.tsv",threads)
    pair_audits=[]; predictions=[]; hashes={r["protein_id"]:r["sha256"] for r in proteins}
    for pair in rows(bundle/"candidates.csv"):
        q,p=pair["query_id"],pair["protein_id"]
        matching=exact.get(hashes[p],set())
        matched_pair=any(observed[pid]&exact_rxns[q] for pid in matching)
        refs={pid for rid in closest[q]["reaction_ids"] for pid in graph[rid]}
        relevant=[h for h in hits.get(p,[]) if h["target"] in refs]
        predictions.append(dict(query_id=q,protein_id=p,score=max((h["bits"] for h in relevant),default=0.0)))
        pair_audits.append(dict(query_id=q,protein_id=p,label=int(pair["label"]),
            exact_training_sequence=bool(matching),exact_training_reaction=bool(exact_rxns[q]) if representation=="directed" else "UNKNOWN",
            exact_training_pair=matched_pair if representation=="directed" else "UNKNOWN",
            participant_set_pair=matched_pair if representation=="participant_set" else "NOT_APPLICABLE",
            nearest_reaction_similarity=closest[q]["similarity"],homology_status="NOT_RUN",
            reaction_representation=representation))
    write_rows(output/"pairs.csv",pair_audits); write_rows(output/"scores.csv",predictions)
    write_rows(output/"nearest_reactions.csv",[dict(query_id=q,similarity=c["similarity"],reaction_ids=";".join(sorted(c["reaction_ids"]))) for q,c in sorted(closest.items())])
    write_rows(output/"invalid_reactions.csv",invalid,["reaction_id","error"])
    result=dict(signature=signature,reaction_representation=representation,reference_proteins=len(ref_seen),
        exact_exposed_panel_proteins=sum(bool(exact.get(p["sha256"])) for p in proteins),
        exact_exposed_active_pairs=sum(r["label"]==1 and r["exact_training_pair"] is True for r in pair_audits) if representation=="directed" else None,
        participant_set_exposed_active_pairs=sum(r["label"]==1 and r["participant_set_pair"] is True for r in pair_audits) if representation=="participant_set" else None,
        upstream_supervision="UNKNOWN; foundation-model and SLEEC exposure not audited",
        limitations=["Exact identity retains participants/charges; changed cofactors/protons can hide equivalent chemistry",
                    "Full training-relative homology not run; baseline alignments do not establish remoteness",
                    "Participant sets do not establish exact transformation exposure"],
        files={name:digest(output/name) for name in ["pairs.csv","scores.csv","nearest_reactions.csv","invalid_reactions.csv","alignment.tsv","reference.fasta"]})
    write_json(output/"complete.json",result)
    return result
