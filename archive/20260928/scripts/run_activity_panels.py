#!/usr/bin/env python3
"""Frozen activity-panel comparison. All stages are resumable and provenance-bound."""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import yaml
from scripts.activity_panels import (digest, rows, write_rows, write_json, prepare_nitrilase,
    verify_bundle, validate_scores, evaluate, summarize)


def resolve(path):
    path=Path(path); return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def executable_path(path):
    # Resolving a venv's python symlink loses pyvenv.cfg and its dependencies.
    path=Path(path); return path if path.is_absolute() else ROOT/path


def code_identity():
    files=[*sorted((ROOT/"scripts").glob("activity_panel*.py")),Path(__file__),
           ROOT/"scripts/run_cyp_specificity.py",ROOT/"scripts/cyp_specificity_audit.py",ROOT/"scripts/cyp_specificity.py"]
    return {str(p.relative_to(ROOT)):digest(p) for p in files}


def preflight(settings,args):
    from scripts.run_cyp_specificity import check_transformer_checkpoint
    from importlib.metadata import version
    paths=[resolve(settings["mmseqs"]),resolve(settings["horizyn_python"])]
    paths.extend(resolve(v) for v in settings["features"].values())
    paths.extend(ROOT/p for p in [".deps/ChIRo/paper_results/RS_experiment/ChIRo/params_RS_ChIRo.json",
                                ".deps/ChIRo/paper_results/RS_experiment/ChIRo/results_RS_ChIRo_seed1/best_model.pt"])
    for spec in settings["models"].values():
        paths.extend(resolve(spec[k]) for k in ("checkpoint","config","chemistry_schema","upstream") if k in spec)
    for source in settings["sources"].values(): paths.extend(resolve(v) for v in source.values())
    missing=[str(p) for p in paths if not p.exists()]
    if missing: raise FileNotFoundError("Missing inputs:\n"+"\n".join(missing))
    subprocess.run([str(executable_path(settings["horizyn_python"])),"-c",
                    "import h5py, numpy, torch, lightning, rdkit"],check=True,cwd=ROOT)
    for k in ("prott5","reaction_t5"): check_transformer_checkpoint(resolve(settings["features"][k]),version("torch"))
    print("Preflight passed: both CIRCE variants, F3, official Horizyn, frozen feature models and training sources",flush=True)


def feature_stage(settings,args):
    from scripts.run_cyp_specificity import stage, identity
    from importlib.metadata import version
    bundle=resolve(settings["benchmark"]); feat=args.run_root/"features"; feat.mkdir(exist_ok=True)
    cfg=settings["features"]; verify_bundle(bundle)
    scripts=["extract_prott5_residue_embeddings.py","extract_reaction_t5v2_embeddings.py",
             "extract_unimol2_reaction_embeddings.py","extract_chiro_reaction_embeddings.py"]
    signature=dict(bundle=digest(bundle/"manifest.json"),features=cfg,device=args.device,batch_size=args.batch_size,
        code={name:digest(ROOT/"scripts"/name) for name in scripts},runner=digest(__file__),
        runtime={p:version(p) for p in ("torch","transformers","rdkit","numpy")},
        backbones={k:{p.name:identity(p) for p in sorted(resolve(cfg[k]).iterdir()) if p.suffix in {".json",".safetensors",".bin",".model"}} for k in ("prott5","reaction_t5")})
    base=[sys.executable]; cpu={"CUDA_VISIBLE_DEVICES":""} if args.device=="cpu" else {}
    cmd=[*base,ROOT/"scripts/extract_prott5_residue_embeddings.py","--fasta",bundle/"proteins.fasta",
         "--output",feat/"proteins.h5","--model-name",resolve(cfg["prott5"]),"--device",args.device,
         "--batch-size",str(args.batch_size),"--max-tokens-per-batch","8192","--max-sequence-length","1022",
         "--length-sort","--padded-token-budget","--resume"]
    merge=[*cmd,"--merge-only","--device","cpu","--merge-order","shard","--merge-storage","copy"]
    stage(cmd,[feat/"proteins.h5"],signature,feat/"prott5.complete.json",cpu,after_command=merge)
    common=["--reactions",bundle/"reactions.csv","--no-bidirectional","--no-allow-pseudo-reactions"]
    stage([*base,ROOT/"scripts/extract_reaction_t5v2_embeddings.py",*common,"--output",feat/"reactiont5.h5",
        "--model-name",resolve(cfg["reaction_t5"]),"--device",args.device,"--batch-size","8","--max-length","512","--pooling","mean","--force"],
        [feat/"reactiont5.h5"],signature,feat/"reactiont5.complete.json",cpu)
    stage([*base,ROOT/"scripts/extract_unimol2_reaction_embeddings.py",*common,"--output",feat/"unimol2.h5",
        "--batch-size","16","--dtype","float16","--skip-invalid-molecules","--no-skip-invalid-reactions","--compression","none","--force"],
        [feat/"unimol2.h5"],signature,feat/"unimol2.complete.json",
        dict(cpu,PYTHONPATH=os.pathsep.join(map(str,[ROOT/".deps/unimol_tools",ROOT.parent/"env/unimol2_site",ROOT])),UNIMOL_WEIGHT_DIR=str(resolve(cfg["unimol_weights"]))))
    stage([*base,ROOT/"scripts/extract_chiro_reaction_embeddings.py",*common,"--output",feat/"chiro.h5",
        "--device",args.device,"--num-workers","1","--skip-invalid-molecules","--no-skip-invalid-reactions","--log-skipped-molecules","--batch-size","16","--force"],
        [feat/"chiro.h5"],signature,feat/"chiro.complete.json",
        dict(cpu,PYTHONPATH=os.pathsep.join(map(str,[ROOT/".deps/python",ROOT/".deps/ChIRo",ROOT]))))
    for name,spec in settings["models"].items():
        if spec["type"]!="circe":continue
        schema=resolve(spec["chemistry_schema"]); dictionary=resolve(cfg["cofactor_dictionary"])
        stage([*base,ROOT/"wet_lab/materialize_chemistry.py","--reactions",bundle/"reactions.csv","--schema",schema,
               "--cofactor-dictionary",dictionary,"--output",feat/f"chemistry_{name}.npz"],
              [feat/f"chemistry_{name}.npz"],dict(signature,schema=digest(schema),dictionary=digest(dictionary),
              materializer=digest(ROOT/"horizyn/capability/reaction_set_features.py")),feat/f"chemistry_{name}.complete.json",cpu)


def score_signature(settings,args,name):
    spec=settings["models"][name]; feat=args.run_root/"features"
    files=["proteins.h5"] if spec["type"]=="horizyn" else ["proteins.h5","reactiont5.h5","unimol2.h5","chiro.h5",f"chemistry_{name}.npz"]
    return dict(bundle=digest(resolve(settings["benchmark"])/"manifest.json"),method=spec,
        checkpoint_sha256=digest(resolve(spec["checkpoint"])),config_sha256=digest(resolve(spec["config"])),
        features={f:digest(feat/f) for f in files},code=code_identity(),device=args.device,
        model_code={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/"horizyn").rglob("*.py"))} if spec["type"]=="circe" else None)


def publish_scores(settings,args,name,variant_records,signature):
    bundle=resolve(settings["benchmark"]); predictions=defaultdict(list)
    variants=rows(bundle/"reactions.csv"); proteins=rows(bundle/"proteins.csv")
    expected={(r["reaction_id"],p["protein_id"]) for r in variants for p in proteins}
    seen=set(); qmap={r["reaction_id"]:r["query_id"] for r in variants}
    for r in variant_records:
        key=r["reaction_id"],r["protein_id"]
        if key in seen or key not in expected or r["query_id"]!=qmap[r["reaction_id"]]: raise ValueError("Invalid variant score coverage")
        seen.add(key); predictions[r["query_id"],r["protein_id"]].append(float(r["score"]))
    if seen!=expected: raise ValueError("Missing variant scores")
    records=[dict(query_id=q,protein_id=p,score=max(v)) for (q,p),v in sorted(predictions.items())]
    validate_scores(bundle,records)
    dest=args.run_root/"scores"/f"{name}.csv"; write_rows(dest,records)
    variant_path=dest.with_suffix(".variants.csv"); write_rows(variant_path,variant_records)
    write_json(dest.with_suffix(".json"),dict(signature=signature,scores_sha256=digest(dest),variants_sha256=digest(variant_path),
        checkpoint_selection="fixed before activity-panel evaluation; no label fitting",aggregation="max over fixed template variants"))


def score_model(settings,args,name):
    import torch
    spec=settings["models"][name]; bundle=resolve(settings["benchmark"]); feat=args.run_root/"features"
    dest=args.run_root/"scores"/f"{name}.csv"; signature=score_signature(settings,args,name)
    if dest.with_suffix(".json").exists():
        old=json.loads(dest.with_suffix(".json").read_text())
        if old["signature"]!=signature or old["scores_sha256"]!=digest(dest) or old["variants_sha256"]!=digest(dest.with_suffix(".variants.csv")):
            raise ValueError(f"Changed predictions: {name}")
        validate_scores(bundle,rows(dest)); print(f"Reusing {name}",flush=True); return
    print(f"Scoring frozen {name}",flush=True)
    if spec["type"]=="horizyn":
        from scripts.run_cyp_specificity import run_command
        worker=args.run_root/"horizyn_worker.json"; out=args.run_root/"horizyn_variants.csv"
        write_json(worker,dict(bundle=str(bundle),upstream=str(resolve(spec["upstream"])),revision=spec["revision"],
            checkpoint=str(resolve(spec["checkpoint"])),checkpoint_sha256=signature["checkpoint_sha256"],
            features=str(feat/"proteins.h5"),device=args.device,output=str(out)))
        run_command([executable_path(settings["horizyn_python"]),ROOT/"scripts/activity_panel_horizyn.py","--spec",worker],args.run_root/"horizyn.log")
        publish_scores(settings,args,name,rows(out),signature); return
    from horizyn.benchmarks.retrieval import BenchmarkTask,build_reaction_inputs,encode_reactions,encode_residue_targets,load_repo_checkpoint,cosine_scores
    from horizyn.config import load_config
    from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
    config=load_config(str(resolve(spec["config"]))); config.data.reaction_chemistry_vectors_path=str(feat/f"chemistry_{name}.npz")
    for m in ("unimol2","chiro","chemistry"): config.data[f"reaction_allow_missing_{m}"]=False
    task=BenchmarkTask(name="nitrilase_activity",task_type="retrieval",dataset="measured_activity",task_label="within_family",
        split="external",pairs=bundle/"encoding_pairs.csv",reactions=bundle/"reactions.csv",
        reaction_model_embeds_h5=feat/"reactiont5.h5",reaction_unimol2_embeds_h5=feat/"unimol2.h5",reaction_chiro_embeds_h5=feat/"chiro.h5")
    reactions=build_reaction_inputs(task,config); variants=rows(bundle/"reactions.csv"); proteins=rows(bundle/"proteins.csv")
    qids=[r["reaction_id"] for r in variants]; pids=[r["protein_id"] for r in proteins]
    if set(reactions.keys)!=set(qids): raise ValueError("Reaction coverage mismatch")
    for q in qids:
        sample=reactions[q]
        flags={k:bool(sample[k]) for k in ("has_unimol2","has_chiro","has_chirality","has_reaction_chemistry") if k in sample}
        if not all(flags.values()): raise ValueError(f"Missing complete reaction modality: {q}: {flags}")
    module,kind=load_repo_checkpoint(resolve(spec["checkpoint"]),config,args.device)
    if kind!="residue": raise ValueError("Unexpected CIRCE model kind")
    dataset=ResidueEmbedDataset(str(feat/"proteins.h5"),max_tokens=1022,truncation="ends_center")
    if set(dataset.keys)!=set(pids):raise ValueError("Protein coverage mismatch")
    try:
        with torch.inference_mode():
            targets=encode_residue_targets(module,dataset,pids,args.device,args.batch_size,store_on_device=False)
            queries=encode_reactions(module,reactions,qids,args.device,8).cpu()
            matrix=cosine_scores(queries,targets).cpu()
        if not torch.isfinite(matrix).all(): raise ValueError("Nonfinite model scores")
        records=[dict(reaction_id=r["reaction_id"],query_id=r["query_id"],protein_id=p["protein_id"],score=float(matrix[i,j])) for i,r in enumerate(variants) for j,p in enumerate(proteins)]
        publish_scores(settings,args,name,records,signature)
    finally:
        dataset.close(); del module; gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()


def report(settings,args):
    import numpy as np
    bundle=resolve(settings["benchmark"]); manifest=verify_bundle(bundle); out=args.run_root/"report"; out.mkdir(exist_ok=True)
    expected=dict(settings["models"])
    expected["random"]=dict(label="Random (exact expectation)")
    for source in settings["sources"]: expected[f"chemistry_sequence_{source}"]=dict(label=f"Chemistry + sequence ({source})")
    all_metrics={}; summaries={}; pending=[]; prediction_tables={}; provenance={}
    for name,spec in expected.items():
        if name=="random":pred=[dict(query_id=r["query_id"],protein_id=r["protein_id"],score=0.0) for r in rows(bundle/"candidates.csv")]
        elif name.startswith("chemistry_sequence_"):
            directory=args.run_root/"audit"/name.removeprefix("chemistry_sequence_")
            if not (directory/"complete.json").exists():pending.append(name);continue
            receipt=json.loads((directory/"complete.json").read_text())
            if digest(directory/"scores.csv")!=receipt["files"]["scores.csv"]:raise ValueError("Changed baseline scores")
            pred=rows(directory/"scores.csv");provenance[name]=receipt
        else:
            path=args.run_root/"scores"/f"{name}.csv"
            if not path.with_suffix(".json").exists():pending.append(name);continue
            receipt=json.loads(path.with_suffix(".json").read_text())
            if receipt["scores_sha256"]!=digest(path) or receipt["signature"]["bundle"]!=digest(bundle/"manifest.json"):
                raise ValueError("Changed score inputs")
            pred=rows(path);provenance[name]=receipt
        metrics=evaluate(bundle,pred);all_metrics[name]=metrics;summaries[name]=summarize(metrics);prediction_tables[name]=pred
        write_rows(out/f"{name}_per_query.csv",metrics)
    # Fixed-model, paired query bootstrap; these are exploratory intervals, not
    # independent-family replication or multiple-comparison-adjusted inference.
    intervals=[]; rng=np.random.default_rng(settings["seed"])
    for primary in [m for m in settings["models"] if m.startswith("circe_v2") and m in all_metrics]:
        for other in all_metrics:
            if primary==other:continue
            for direction in ("r2e","e2r"):
                a={r["query_id"]:r["ap"] for r in all_metrics[primary] if r["direction"]==direction and r["subset"]=="full"}
                b={r["query_id"]:r["ap"] for r in all_metrics[other] if r["direction"]==direction and r["subset"]=="full"}
                if a.keys()!=b.keys():raise ValueError("Paired bootstrap query mismatch")
                delta=np.array([a[q]-b[q] for q in sorted(a)])
                boot=delta[rng.integers(0,len(delta),size=(settings["bootstrap_samples"],len(delta)))].mean(1)
                intervals.append(dict(method=primary,reference=other,direction=direction,metric="all-query mean AP",difference=float(delta.mean()),
                                      low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975))))
    write_rows(out/"paired_differences.csv",intervals,["method","reference","direction","metric","difference","low","high"])
    result=dict(complete=not pending,pending=pending,panel=manifest,metrics=summaries,paired_ap_differences=intervals,
                provenance=provenance,uncertainty="Exploratory paired query bootstrap, correlated chemistry/proteins; no independent-family or training-seed replication")
    write_json(out/"summary.json",result)
    lines=["# Nitrilase activity-panel comparison", "",f"Frozen models; {manifest['candidate_pairs']} measured pairs, {manifest['active_pairs']} active and {manifest['inactive_pairs']} assay-inactive. No activity-label fitting.","",
           "Products are template-derived single-nitrile hydrolysis hypotheses. This evaluates substrate activity, not product selectivity or kinetic rates.","",
           "Different model training corpora prevent architecture-only conclusions. Upstream SLEEC/backbone exposure is unknown. Exact source audits accompany this report; general homology novelty is not established.",""]
    for direction in ("r2e","e2r"):
        lines += [f"## {'Reaction → enzyme' if direction=='r2e' else 'Enzyme → reaction'}", "",
                  "| Model | AP, all queries | AP, queries with actives | Hit@1 | Hit@5 | Active hits@5 |", "|---|---:|---:|---:|---:|---:|"]
        for name,summary in summaries.items():
            s=summary[f"{direction}/full"];a=s["all_queries"];c=s["positive_queries_only"]
            lines.append(f"| {expected[name]['label']} | {a['ap']:.4f} | {c['ap']:.4f} | {a['hit_at_1']:.4f} | {a['hit_at_5']:.4f} | {a['active_at_5']:.3f} |")
        lines += ["",f"Queries: {manifest['query_count'] if direction=='r2e' else manifest['protein_count']}; no-positive queries: {manifest['all_inactive_reactions'] if direction=='r2e' else manifest['all_inactive_enzymes']}. AP/Hit/recall are zero for these queries in the all-query averages.",""]
    lines += ["## Specificity diagnostic", "", "The secondary R→E analysis keeps only enzymes active on at least one panel substrate. It conditions on assay labels and is not a deployable selection rule. The full-panel comparison above is primary.","",
              "| Model | AP among ever-active enzymes |", "|---|---:|"]
    for name,s in summaries.items():lines.append(f"| {expected[name]['label']} | {s['r2e/ever_active_enzymes_diagnostic']['all_queries']['ap']:.4f} |")
    lines += ["","AP and Hit@K integrate exactly over random ordering of tied scores. AP is expected item-level AP, not grouped-threshold PR area. Requested budgets exceeding a pool are capped and recorded per query.","",
              "Paired intervals are exploratory, conditional on fixed checkpoints and this one family. They do not establish prospective performance or independence of related substrates.","",
              f"Completion: {'all configured comparisons complete' if not pending else 'PENDING: '+', '.join(pending)}.","",
              "Sources: [released panel](https://github.com/samgoldman97/enzyme-datasets/tree/"+manifest['preparation']['revision']+") and [Black et al.](https://doi.org/10.1039/C4CC06021K)."]
    (out/"summary.md").write_text("\n".join(lines)+"\n")
    if summaries:
        import matplotlib;matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names=list(summaries);fig,axes=plt.subplots(1,2,figsize=(13,5),layout="constrained")
        for ax,direction,title in zip(axes,["r2e","e2r"],["Reaction → enzyme","Enzyme → reaction"]):
            ax.barh([expected[n]["label"] for n in names],[summaries[n][f"{direction}/full"]["all_queries"]["ap"] for n in names])
            ax.set(xlim=(0,1),xlabel="Mean AP (all queries; exact tie expectation)",title=title);ax.invert_yaxis()
        fig.savefig(out/"comparison.png",dpi=180);fig.savefig(out/"comparison.svg");plt.close(fig)
    print(f"Report: {out/'summary.md'}; pending={pending}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["prepare","preflight","features","score","audit","report","all"])
    p.add_argument("--config",type=Path,default=ROOT/"configs/benchmarks/activity_panels_nitrilase.yaml")
    p.add_argument("--run-root",type=Path,default=ROOT/"runs/activity_panels_nitrilase_v2")
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--batch-size",type=int,default=8);p.add_argument("--threads",type=int,default=4)
    args=p.parse_args()
    if min(args.batch_size,args.threads)<1:p.error("Positive batch size and threads required")
    args.run_root=args.run_root.resolve();args.run_root.mkdir(parents=True,exist_ok=True)
    os.chdir(ROOT);settings=yaml.safe_load(args.config.read_text())
    os.environ.update(OMP_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),WANDB_MODE="disabled")
    with (args.run_root/"controller.lock").open("a") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit("This run is already active")
        if args.stage in {"prepare","all"}:print(prepare_nitrilase(resolve(settings["release"]),resolve(settings["benchmark"])),flush=True)
        verify_bundle(resolve(settings["benchmark"]))
        binding=dict(config=settings,code=code_identity(),benchmark=digest(resolve(settings["benchmark"])/"manifest.json"))
        path=args.run_root/"protocol.json"
        if path.exists() and json.loads(path.read_text())!=binding:raise ValueError("Protocol/code changed; use a new run root")
        if not path.exists():write_json(path,binding)
        if args.stage in {"preflight","all"}:preflight(settings,args)
        if args.stage in {"features","all"}:feature_stage(settings,args)
        if args.stage in {"score","all"}:
            for name in settings["models"]:score_model(settings,args,name)
        if args.stage in {"audit","all"}:
            from scripts.activity_panel_audit import audit
            for name,spec in settings["sources"].items():
                print(f"Auditing {name}",flush=True)
                audit(resolve(settings["benchmark"]),{k:resolve(v) for k,v in spec.items()},args.run_root/"audit"/name,resolve(settings["mmseqs"]),args.threads)
        if args.stage in {"report","all"}:report(settings,args)


if __name__=="__main__":main()
