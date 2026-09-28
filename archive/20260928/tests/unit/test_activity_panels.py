"""Check scientific metric semantics and input integrity independently."""
import itertools
import json
import math

import numpy as np
import pytest

from scripts.activity_panels import (ranking_metrics,nitrilase_reactions,canonical_molecule,
    digest,write_rows,write_json,validate_scores,evaluate,summarize,SCHEMA)


def explicit_metrics(labels,order,k):
    y=[labels[i] for i in order];p=sum(y);ranks=[i+1 for i,v in enumerate(y) if v]
    return dict(ap=sum(sum(y[:r])/r for r in ranks)/p if p else 0,
                mrr_first=1/min(ranks) if ranks else 0,
                hit=float(any(y[:k])),active=float(sum(y[:k])))


@pytest.mark.parametrize("labels,scores",[
    ([1,0,1,0],[0,0,0,0]),([1,0,1,0],[3,2,2,1]),([0,1,1,0,1],[2,2,1,1,1]),
    ([0,0,0],[1,0,0]),([1,1,1],[1,1,1]),([1],[0]),([0],[0]),
])
def test_exact_ties_against_all_permutations(labels,scores):
    orders=[p for p in itertools.permutations(range(len(labels))) if all(scores[p[i]]>=scores[p[i+1]] for i in range(len(p)-1))]
    for k in [1,2,5]:
        actual=ranking_metrics(labels,scores,budgets=(k,))
        expected=[explicit_metrics(labels,p,min(k,len(labels))) for p in orders]
        for key in ["ap","mrr_first"]:assert actual[key]==pytest.approx(np.mean([x[key] for x in expected]))
        assert actual[f"hit_at_{k}"]==pytest.approx(np.mean([x["hit"] for x in expected]))
        assert actual[f"active_at_{k}"]==pytest.approx(np.mean([x["active"] for x in expected]))


@pytest.mark.parametrize("labels,scores",[([],[]),([1],[math.nan]),([2],[0]),([1,0],[1])])
def test_invalid_metrics_fail(labels,scores):
    with pytest.raises(ValueError):ranking_metrics(labels,scores)


def test_acetonitrile_reaction_is_balanced():
    from rdkit import Chem
    from collections import Counter
    rxn=nitrilase_reactions("CC#N")
    assert len(rxn)==1
    left,right=rxn[0].split(">>")
    def atoms(side):
        counts=Counter()
        for text in side.split("."):
            counts.update(a.GetSymbol() for a in Chem.AddHs(Chem.MolFromSmiles(text)).GetAtoms())
        return counts
    assert atoms(left)==atoms(right)
    assert set(right.split("."))=={"N",canonical_molecule("CC(=O)O")}


def test_ambiguous_products_are_deduplicated_and_order_invariant():
    from rdkit import Chem
    substrate="N#CCC(C)C#N"
    rxns=nitrilase_reactions(substrate)
    alternative=Chem.MolToSmiles(Chem.MolFromSmiles(substrate),doRandom=True)
    assert rxns==nitrilase_reactions(alternative)
    assert len(rxns)==2
    assert len(nitrilase_reactions("N#Cc1cccc(C#N)c1"))==1


def test_non_nitrile_rejected():
    with pytest.raises(ValueError):nitrilase_reactions("CCO")


@pytest.fixture
def bundle(tmp_path):
    records=[dict(query_id=q,protein_id=p,label=int(q=="q1" and p=="p1")) for q in ["q1","q2"] for p in ["p1","p2"]]
    write_rows(tmp_path/"candidates.csv",records)
    write_json(tmp_path/"manifest.json",dict(schema=SCHEMA,files={"candidates.csv":digest(tmp_path/"candidates.csv")}))
    return tmp_path


def good_scores():
    return [dict(query_id=q,protein_id=p,score=1 if p=="p1" else 0) for q in ["q1","q2"] for p in ["p1","p2"]]


@pytest.mark.parametrize("mutation",["missing","duplicate","extra","nonfinite"])
def test_score_contract_rejects_incomplete_or_invalid(bundle,mutation):
    pred=good_scores()
    if mutation=="missing":pred.pop()
    elif mutation=="duplicate":pred.append(pred[0])
    elif mutation=="extra":pred.append(dict(query_id="other",protein_id="p1",score=0))
    else:pred[0]["score"]=float("inf")
    with pytest.raises(ValueError):validate_scores(bundle,pred)


def test_input_tampering_rejected(bundle):
    with (bundle/"candidates.csv").open("a") as f:f.write("q3,p1,0\n")
    with pytest.raises(ValueError,match="Changed panel"):validate_scores(bundle,good_scores())


def test_no_positive_queries_retained_and_label_conditioned_subset_named(bundle):
    result=summarize(evaluate(bundle,good_scores()))
    for direction in ["r2e","e2r"]:
        s=result[f"{direction}/full"]
        assert s["query_count"]==2 and s["no_positive_query_count"]==1
        assert s["all_queries"]["ap"]==s["positive_queries_only"]["ap"]/2
    assert "r2e/ever_active_enzymes_diagnostic" in result


def test_worker_keeps_virtual_environment_python_path(tmp_path,monkeypatch):
    from scripts import run_activity_panels as runner
    executable=tmp_path/"system-python"; executable.touch()
    venv=tmp_path/"venv/bin"; venv.mkdir(parents=True)
    link=venv/"python"; link.symlink_to(executable)
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    assert runner.executable_path("venv/bin/python")==link
    assert runner.executable_path(link)==link
    assert runner.executable_path(link)!=link.resolve()


@pytest.mark.parametrize("participant_set",[False,True])
def test_audit_separates_catalog_from_training_and_does_not_use_assay_labels(tmp_path,monkeypatch,participant_set):
    from scripts import activity_panel_audit as module
    from scripts.activity_panels import sequence_hash,rows
    bundle=tmp_path/"panel"; bundle.mkdir()
    reaction=nitrilase_reactions("CC#N")[0]
    sequences={"p1":"ACDEFGHIK","p2":"LMNPQRSTV"}
    write_rows(bundle/"proteins.csv",[dict(protein_id=p,sequence=s,sha256=sequence_hash(s)) for p,s in sequences.items()])
    write_rows(bundle/"reactions.csv",[dict(query_id="q",reaction_id="v",reaction_smiles=reaction)])
    (bundle/"proteins.fasta").write_text("".join(f">{p}\n{s}\n" for p,s in sequences.items()))
    spec={k:tmp_path/f"{k}.csv" for k in ["pairs","reactions"]}
    spec["proteins"]=tmp_path/"catalog.fasta"
    # p2 exists in the source catalog but has only a negative training edge.
    spec["proteins"].write_text(">train1\nACDEFGHIK\n>catalog_only\nLMNPQRSTV\n")
    write_rows(spec["pairs"],[dict(protein_id="train1",reaction_id="train_r",label=1),
                              dict(protein_id="catalog_only",reaction_id="train_r",label=0)])
    write_rows(spec["reactions"],[dict(reaction_id="train_r",rxn=reaction.replace(">>",".") if participant_set else reaction)])
    mmseqs=tmp_path/"mmseqs";mmseqs.write_text("mock executable")
    def alignment(executable,query,reference,output,threads):
        assert reference.read_text()==">train1\nACDEFGHIK\n"
        output.write_text("p1\ttrain1\t1\t1\t1\t90\n")
        return {"p1":[dict(target="train1",bits=90)]}
    monkeypatch.setattr(module,"align",alignment)
    predictions=[]
    for i in [0,1]:
        write_rows(bundle/"candidates.csv",[dict(query_id="q",protein_id=p,label=i) for p in sequences])
        write_json(bundle/"manifest.json",dict(schema=SCHEMA,files={p.name:digest(p) for p in bundle.iterdir() if p.name!="manifest.json"}))
        output=tmp_path/f"audit{i}"
        result=module.audit(bundle,spec,output,mmseqs)
        exposure={r["protein_id"]:r for r in rows(output/"pairs.csv")}
        assert exposure["p1"]["exact_training_sequence"]=="True"
        assert exposure["p2"]["exact_training_sequence"]=="False"
        assert exposure["p1"]["exact_training_pair"]==("UNKNOWN" if participant_set else "True")
        assert result["exact_exposed_active_pairs"]==(None if participant_set else i)
        predictions.append(rows(output/"scores.csv"))
    assert predictions[0]==predictions[1]
