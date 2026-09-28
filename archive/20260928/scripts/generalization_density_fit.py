#!/usr/bin/env python3
"""Transfer a selected density recipe using each split's training data only.

No validation scores or labels are consulted. Each graph checkpoint is one of
the previously fixed seed replicates; each receives the same fitted density map.
"""
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
from horizyn.semantic_anchors import fit_center, centered_unit, reaction_features, row_unit
from horizyn.generalization_retrieval import checked_artifact
from scripts.generalization_density_gate import nearest_support
from scripts.generalization_export import atomic_json, identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--graph-checkpoint", action="append", required=True, help="SEED=PATH, repeat for fixed replicates")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()): raise ValueError("Use a new train-only density fit directory")
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    template_receipt=json.loads((a.template.parent/"selected_validation_receipt.json").read_text())
    if checked_artifact(template_receipt["gate"]).resolve()!=a.template.resolve(): raise ValueError("Template gate hash/path mismatch")
    checked_artifact(template_receipt["selection"])
    template=torch.load(a.template,map_location="cpu",weights_only=False)
    if template.get("inference_precision")!="stable_fp64": raise ValueError("Phase2 transfer requires stable FP64 postprocessing")
    selected=template["selected_gate"]
    if selected["space"] not in ("raw","f3"): raise ValueError("This transfer fitter expects a selected density gate")
    feature_names=("catalog.json","manifest.json","pairs.npz","f3_features.npz","reaction_features.npz","protein_mean.h5")
    sources={name:identity(a.features/name,True)for name in feature_names}
    fixed_graphs={}
    for value in a.graph_checkpoint:
        seed,path=value.split("=",1)
        if int(seed) not in (42,17,73) or seed in fixed_graphs: raise ValueError("Expected unique predeclared seeds42/17/73")
        checkpoint=Path(path).resolve()
        snapshot=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if snapshot["registry"].get("feature_manifest_sha256")!=sources["manifest.json"]["sha256"] or snapshot["registry"].get("test_used") is not False:
            raise ValueError("Graph checkpoint was not fitted on this training source")
        fixed_graphs[seed]=identity(checkpoint,True)
    with np.load(a.features/"pairs.npz")as source:
        tr,te=np.unique(source["train"][:,0]),np.unique(source["train"][:,1])
    with h5py.File(a.features/"protein_mean.h5")as source:
        if not source["complete"][:].all(): raise ValueError("Incomplete protein mean source")
        raw_e=torch.tensor(source["vectors"][:],device=a.device)
    center_e=fit_center(raw_e,te)
    with np.load(a.features/"reaction_features.npz")as source:
        modalities=tuple(template["reaction_centers"])
        blocks={key:torch.tensor(source[key],device=a.device)for key in modalities}
        masks={key:torch.tensor(source[key+"_mask"],device=a.device)for key in modalities}
    centers_r={key:fit_center(blocks[key],tr,masks[key])for key in modalities}
    if selected["space"]=="raw":
        anchors=dict(enzyme=centered_unit(raw_e,center_e)[te],reaction=reaction_features(blocks,centers_r,masks,modalities)[tr])
    else:
        with np.load(a.features/"f3_features.npz")as source:
            anchors=dict(enzyme=row_unit(torch.tensor(source["proteins"][te],device=a.device)),reaction=row_unit(torch.tensor(source["train_reactions"],device=a.device)))
    rng=np.random.default_rng(20260920)
    calibration=dict(reaction=np.arange(len(tr)),enzyme=np.sort(rng.choice(len(te),min(4096,len(te)),replace=False)))
    thresholds={selected["space"]:{}}
    for endpoint,values in anchors.items():
        chosen=torch.tensor(calibration[endpoint],device=a.device)
        support=nearest_support(values[chosen],values,exclude=chosen)
        thresholds[selected["space"]][endpoint]={str(q):float(torch.quantile(support,q))for q in (.25,.5,.95)}
    for seed,graph in fixed_graphs.items():
        directory=a.output/f"seed{seed}"
        directory.mkdir()
        registry=dict(schema="phase2_density_train_only_transfer_v1",test_used=False,validation_used=False,
            sources=sources,graph_checkpoint=graph,template=identity(a.template,True),
            template_receipt=identity(a.template.parent/"selected_validation_receipt.json",True),
            selected_recipe=selected,calibration_seed=20260920,calibration_proteins=len(calibration["enzyme"]),
            script=identity(__file__,True),note="Thresholds and centers refitted on this split's training rows only; gate rule and graph checkpoints remain fixed")
        atomic_json(directory/"registry.json",registry)
        atomic_json(directory/"fit_recipe.json",dict(selected=dict(candidate=selected),test_used=False,validation_used=False))
        torch.save(dict(selected_gate=selected,thresholds=thresholds,inference_precision="stable_fp64",
            raw_protein_center=center_e.cpu(),reaction_centers={key:value.cpu()for key,value in centers_r.items()},
            calibration_indices=calibration,graph_checkpoint=graph),directory/"selected_gate.pt")
        atomic_json(directory/"artifact_receipt.json",dict(schema="phase2_train_only_density_receipt_v1",test_used=False,
            gate=identity(directory/"selected_gate.pt",True),selection=identity(directory/"fit_recipe.json",True),
            registry=identity(directory/"registry.json",True)))
        atomic_json(directory/"density_bundle.json",dict(schema="phase2_density_bundle_v1",test_used=False,seed=int(seed),
            gate=identity(directory/"selected_gate.pt",True),registry=identity(directory/"registry.json",True),
            artifact_receipt=identity(directory/"artifact_receipt.json",True),features=str(a.features.resolve()),
            feature_manifest=sources["manifest.json"],feature_manifest_sha256=sources["manifest.json"]["sha256"],
            graph_checkpoint=graph,inference_precision="stable_fp64",gate_recipe=selected,
            module=identity(ROOT/"horizyn/generalization_density.py",True)))
    atomic_json(a.output/"complete.json",dict(test_used=False,validation_used=False,thresholds=thresholds,
        seeds=list(fixed_graphs),template=identity(a.template,True),sources=sources))
    print(json.dumps(dict(thresholds=thresholds,seeds=list(fixed_graphs)),indent=2),flush=True)


if __name__=="__main__": main()
