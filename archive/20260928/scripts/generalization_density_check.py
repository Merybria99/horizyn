#!/usr/bin/env python3
"""Production-equivalence checks on training/validation density artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from horizyn.generalization_density import DensityGatedEncoder
from scripts.generalization_export import atomic_json, identity
from scripts.generalization_full_graph import validation_data
from scripts.generalization_metrics import evaluate_scores


def difference(first, second):
    return dict(exact=bool(torch.equal(first, second)), maximum_absolute=float((first - second).abs().max()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--expected", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a=p.parse_args()
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    model,state=DensityGatedEncoder.from_artifact(a.gate,a.features,a.device)
    catalog=state["feature_catalog"]
    with np.load(a.features/"pairs.npz") as source: pairs={key:source[key] for key in ("train","validation")}
    vr,ve,truth=validation_data(catalog,pairs)
    with np.load(a.features/"f3_features.npz") as source:
        base_e=torch.tensor(source["proteins"][ve],device=a.device)
        base_r=torch.tensor(source["reactions"][vr],device=a.device)
    with h5py.File(a.features/"protein_mean.h5") as source: means=torch.tensor(source["vectors"][:][ve],device=a.device)
    with np.load(a.features/"reaction_features.npz") as source:
        blocks={key:torch.tensor(source[key][vr],device=a.device) for key in model.modalities}
        masks={key:torch.tensor(source[key+"_mask"][vr],device=a.device) for key in model.modalities}
    with np.load(a.expected) as source:
        expected_e=torch.tensor(source["proteins"],device=a.device)
        expected_r=torch.tensor(source["reactions"],device=a.device)
    encoded_e, diagnostics_e=model.encode_enzymes(base_e,means,512,True)
    encoded_r, diagnostics_r=model.encode_reactions(base_r,blocks,masks,512,True)
    checks=dict(saved_enzymes=difference(encoded_e,expected_e),saved_reactions=difference(encoded_r,expected_r))
    checks["reference_batch_enzymes"]=difference(encoded_e,model.encode_enzymes(base_e,means,31))
    checks["reference_batch_reactions"]=difference(encoded_r,model.encode_reactions(base_r,blocks,masks,31))
    rng=np.random.default_rng(20260920)
    pi=torch.tensor(rng.choice(len(base_e),193,False),device=a.device)
    qi=torch.tensor(rng.choice(len(base_r),193,False),device=a.device)
    checks["subset_enzymes"]=difference(encoded_e[pi],model.encode_enzymes(base_e[pi],means[pi],31))
    checks["subset_reactions"]=difference(encoded_r[qi],model.encode_reactions(base_r[qi],{k:v[qi] for k,v in blocks.items()},{k:v[qi] for k,v in masks.items()},31))
    checks["singleton_enzyme"]=difference(encoded_e[pi[:1]],model.encode_enzymes(base_e[pi[:1]],means[pi[:1]],31))
    checks["singleton_reaction"]=difference(encoded_r[qi[:1]],model.encode_reactions(base_r[qi[:1]],{k:v[qi[:1]] for k,v in blocks.items()},{k:v[qi[:1]] for k,v in masks.items()},31))
    scores=(encoded_r.double()@encoded_e.double().T).float()
    expected_scores=(expected_r.double()@expected_e.double().T).float()
    checks["saved_scores"]=difference(scores,expected_scores)
    evaluated=evaluate_scores(scores,truth)
    expected_evaluation=evaluate_scores(expected_scores,truth)
    checks["saved_metrics_exact"]=evaluated["summary"]==expected_evaluation["summary"]
    checks["saved_positive_ranks_exact"]=all(np.array_equal(evaluated["per_positive"][d]["rank"],expected_evaluation["per_positive"][d]["rank"])for d in evaluated["per_positive"])
    atomic_json(a.output,dict(checks=checks,test_used=False,source=identity(__file__,True),module=identity(ROOT/"horizyn/generalization_density.py",True),gate=identity(a.gate,True),expected=identity(a.expected,True)))
    np.savez_compressed(a.output.with_suffix(".diagnostics.npz"),**{f"{endpoint}_{key}":value.cpu().numpy() for endpoint,mapping in (("enzyme",diagnostics_e),("reaction",diagnostics_r)) for key,value in mapping.items() if value is not None})
    print(json.dumps(checks,indent=2),flush=True)
    if not checks["saved_metrics_exact"] or not checks["saved_positive_ranks_exact"]:
        raise ValueError("Production density encoder does not reproduce selected validation")


if __name__=="__main__": main()
