#!/usr/bin/env python3
"""Validate phase-2 residual/semantic feature composition on the full library.

The score equals a dot product of independently encoded, sqrt-weighted
concatenated endpoint features. The dictionary contains training edges only.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horizyn.generalization_residual import FrozenGeometryResidual
from horizyn.semantic_smooth import SmoothAnchorDualEncoder
from generalization_clipzyme_phase2_screen import library, queries
from generalization_clipzyme_f3_validation import read_csv
from generalization_clipzyme_screening_evaluate import evaluate_query, sha256
from generalization_clipzyme_screening_protocol import identifier
from generalization_full_graph import atomic_json


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fixed-alpha", type=float, choices=(0., .1, .25, .4, .5))
    p.add_argument("--fixed-cap", type=float, choices=(1., .5, .25), default=1.)
    p.add_argument("--output-name", default="composition_v1")
    p.add_argument("--selection-metric", choices=("bedroc85", "geometric_mean"), default="bedroc85")
    p.add_argument("--cross-root", type=Path)
    p.add_argument("--validate-only", action="store_true", help="Freeze validation results without opening test queries or labels")
    a = p.parse_args()
    campaign = a.campaign.resolve()
    if Path(a.output_name).name != a.output_name:
        raise ValueError("Output name must be a single directory component")
    out = campaign / a.output_name
    out.mkdir(exist_ok=False)
    cross = a.cross_root.resolve() if a.cross_root else campaign.parent
    parent = json.loads((campaign / "protocol.json").read_text())
    selection = json.loads((campaign / "validation_selected.json").read_text())
    parent_validation = json.loads(Path(selection["selected"]["evaluation"]).read_text())
    refiner = parent_validation["refiner"]
    if sha256(Path(refiner["path"])) != refiner["sha256"]:
        raise ValueError("Selected residual has changed")
    grid = [dict(cap=cap, alpha=alpha) for cap in (1., .5) for alpha in (0., .1, .25)]
    if a.fixed_alpha is not None:
        grid = [dict(cap=1., alpha=0.)]
        fixed = dict(cap=a.fixed_cap, alpha=a.fixed_alpha)
        if fixed != grid[0]:
            grid.append(fixed)
    registry = dict(created_utc=datetime.now(timezone.utc).isoformat(), grid=grid,
        parent_selection_sha256=sha256(campaign / "validation_selected.json"),
        refiner=refiner, dictionary_sha256=sha256(campaign / "anchors.pt"),
        feature_manifest_sha256=sha256(campaign / "features/manifest.json"),
        selection="Full-library validation BEDROC85", test_used_for_selection=False,
        score="FP32 dot of residual coordinates plus FP32 sparse semantic dot; equivalent to sqrt-weight concatenation",
        density_gate="Constant residual scale; no candidate-pool normalization or hubness",
        source_sha256=sha256(Path(__file__)))
    registry["validation_only"] = a.validate_only
    if a.fixed_alpha is not None:
        registry["selection"] = "Fixed replication recipe; parent parity control is not a selection candidate"
        registry["fixed_recipe"] = fixed
    elif a.selection_metric == "geometric_mean":
        registry["selection"] = "Maximum validation geometric mean of BEDROC85, BEDROC20, EF5/20, EF10/10"
        registry["secondary_exploratory_selector_after_primary_test"] = True
    atomic_json(out / "registry.json", registry)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    a.catalog = cross / "clipzyme_f3_catalog_v1"
    a.protocol = cross / "clipzyme_screening_evaluation_protocol_v2"
    a.features = campaign / "features"
    base_run = Path(parent["base_config"]).parents[1]
    screen_root = Path(parent["validation_summary"]).parent.parent
    if screen_root.parent != base_run:
        raise ValueError("Validation summary does not belong to the selected base run")
    a.screen = campaign / "selected_test/test_embeddings"
    if not a.screen.exists():
        a.screen = screen_root / "requested_test/test_embeddings"
    means_path = cross / "clipzyme_phase2_enzymemap_v1/screen_protein_mean.h5"
    registry["screen_means_sha256"] = sha256(means_path)
    for rank in range(4):
        stem = f"protein_rank{rank:02d}_of_04"
        receipt = json.loads((a.screen / (stem + ".receipt.json")).read_text())
        if (receipt["checkpoint_sha256"] != parent["base_checkpoint_sha256"] or
                receipt["embedding_sha256"] != sha256(a.screen / (stem + ".npy")) or
                receipt["ids_sha256"] != sha256(a.screen / (stem + ".ids.txt"))):
            raise ValueError("Screening protein lineage changed")
    candidate_ids, protein_ids, expanded, base_e, means = library(a.catalog, a.screen, means_path)
    feature_catalog = json.loads((a.features / "catalog.json").read_text())
    dictionary = torch.load(campaign / "anchors.pt", map_location=a.device, weights_only=False)
    if dictionary["feature_manifest_sha256"] != registry["feature_manifest_sha256"]:
        raise ValueError("Training dictionary belongs to different input features")
    saved = torch.load(refiner["path"], map_location=a.device, weights_only=False)
    if (saved["registry"].get("test_used") is not False or
            saved["registry"]["feature_manifest_sha256"] != registry["feature_manifest_sha256"]):
        raise ValueError("Residual and semantic dictionary training inputs disagree")
    residual = FrozenGeometryResidual(**saved["model_config"]).to(a.device).eval().requires_grad_(False)
    residual.load_state_dict(saved["state_dict"], strict=True)
    smooth = SmoothAnchorDualEncoder(dictionary, dictionary["train_reactions"], dict(
        kernel="exponential", reaction_neighbors=None, reaction_temperature=.03,
        protein_neighbors=32, enzyme_temperature=.03, alpha=1.)).to(a.device)
    base_e = torch.from_numpy(base_e).to(a.device)
    caps = sorted({recipe["cap"] for recipe in grid})
    dense_e = {cap: torch.cat([F.normalize(b + cap * residual.scale * residual.enzyme(b), dim=-1)
                              for b in base_e.split(8192)]) for cap in caps}
    pointers = [torch.zeros(1, dtype=torch.int64, device=a.device)]
    columns, weights, total = [], [], 0
    for start in range(0, len(means), 512):
        semantic = smooth.encode_semantic_enzymes(torch.as_tensor(means[start:start+512], device=a.device), 512)
        sparse = semantic.to_sparse_csr()
        pointers.append(sparse.crow_indices()[1:] + total)
        columns.append(sparse.col_indices()); weights.append(sparse.values())
        total += sparse.values().numel()
    semantic_e = torch.sparse_csr_tensor(torch.cat(pointers), torch.cat(columns), torch.cat(weights),
        size=(len(means), len(dictionary["train_reactions"])), device=a.device)
    del means, base_e, columns, weights, pointers, semantic, sparse
    atomic_json(out / "registry.json", registry)
    release = json.loads((cross / "clipzyme_manifests_v2/manifest.json").read_text())
    rows = {}
    for split in ("train", "dev"):
        record = release["associations"][split]
        if sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("Training/validation association split changed")
        rows[split] = read_csv(Path(record["path"]))
    seen_r = {r["reaction"] for r in rows["train"]}
    seen_e = {r["protein_id"] for r in rows["train"]}
    truth = defaultdict(set)
    for row in rows["dev"]:
        if row["reaction"] not in seen_r:
            truth[identifier(row["reaction"])].add(row["protein_id"])
    index = {key:i for i,key in enumerate(candidate_ids)}
    kept = np.asarray([i for i,k in enumerate(candidate_ids) if k not in seen_e], dtype=np.int32)

    def query_features(scope):
        a.scope = scope
        ids, br, blocks, masks = queries(a, feature_catalog)
        # Reuse precisely the raw query vectors used to validate the parent.
        if scope == "validation":
            directory = screen_root / "validation_embeddings"
            keys = (directory / "query_ids.txt").read_text().splitlines()
            original = np.load(directory / "reaction_embeddings.npy")
            lookup = {key:i for i,key in enumerate(keys)}
            br = original[[lookup[k] for k in ids]]
        br = torch.from_numpy(br).to(a.device)
        dense = {cap:F.normalize(br + cap * residual.scale * residual.reaction(br),dim=-1)
                 for cap in caps}
        semantic = smooth.encode_semantic_reactions(
            {k:torch.as_tensor(v,device=a.device) for k,v in blocks.items()},
            {k:torch.as_tensor(v,device=a.device) for k,v in masks.items()})
        return ids, dense, semantic

    ids, dense_r, semantic_r = query_features("validation")
    positive_indices = [np.asarray([index[k] for k in truth.get(key,set()) if k in index],dtype=np.int64) for key in ids]
    records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for recipe in grid:
            result = []
            cap, alpha = recipe["cap"], recipe["alpha"]
            for start in range(0,len(ids),64):
                values = (1-alpha) * (dense_r[cap][start:start+64] @ dense_e[cap].T)
                if alpha:
                    values += alpha * torch.sparse.mm(semantic_e, semantic_r[start:start+64].T.contiguous()).T
                values = values.cpu().numpy()[:,expanded]
                args = [(start+i,ids[start+i],v,positive_indices[start+i],kept) for i,v in enumerate(values)]
                result.extend(r for r in pool.map(evaluate_query,args) if r is not None)
            summary = {}
            for table,count in (("table1",261907),("table2",252113)):
                values = [r[table] for r in result if r[table] is not None]
                summary[table] = dict(queries=len(values),candidate_ids=count,
                    **{m:float(np.mean([v[m] for v in values])) for m in ("bedroc85","bedroc20","ef0.05","ef0.1")})
            if summary["table1"]["queries"]!=2652 or summary["table2"]["queries"]!=2216:
                raise ValueError("Full-library validation denominator changed")
            validation = summary["table1"]
            value = validation["bedroc85"]
            if a.selection_metric == "geometric_mean":
                value = float(np.exp(np.mean(np.log([
                    validation["bedroc85"], validation["bedroc20"],
                    validation["ef0.05"] / 20, validation["ef0.1"] / 10]))))
            record=dict(recipe=recipe,summary=summary,value=value)
            records.append(record)
            atomic_json(out/"validation.json",dict(records=records))
            print(json.dumps(record),flush=True)
    for table in ("table1","table2"):
        for metric in ("bedroc85","bedroc20","ef0.05","ef0.1"):
            if abs(records[0]["summary"][table][metric]-parent_validation["summary"][table][metric])>1e-4:
                raise ValueError("Zero-semantic composition does not reproduce the parent validation")
    best=records[-1] if a.fixed_alpha is not None else max(records,key=lambda r:r["value"])
    atomic_json(out/"selection.json",dict(selected=best,selected_utc=datetime.now(timezone.utc).isoformat(),
        registry_sha256=sha256(out/"registry.json"),test_used_for_selection=False))
    if a.validate_only:
        atomic_json(out/"complete.json",dict(validation_only=True,test_queries_opened=False,test_labels_read=False))
        return
    # Test features and labels are opened only after freezing this selection.
    ids,dense_r,semantic_r=query_features("test")
    cap,alpha=best["recipe"]["cap"],best["recipe"]["alpha"]
    scores=np.empty((len(ids),len(candidate_ids)),dtype=np.float32)
    for start in range(0,len(ids),64):
        values=(1-alpha)*(dense_r[cap][start:start+64]@dense_e[cap].T)
        if alpha:values+=alpha*torch.sparse.mm(semantic_e,semantic_r[start:start+64].T.contiguous()).T
        scores[start:start+len(values)]=values.cpu().numpy()[:,expanded]
    np.save(out/"scores.npy",scores)
    (out/"query_ids.txt").write_text("\n".join(ids)+"\n")
    (out/"candidate_ids.txt").write_text("\n".join(candidate_ids)+"\n")
    atomic_json(out/"prediction_receipt.json",dict(selection_sha256=sha256(out/"selection.json"),
        scores_sha256=sha256(out/"scores.npy"),score_shape=list(scores.shape),test_labels_read=False))
    subprocess.run([sys.executable,str(ROOT/"scripts/generalization_clipzyme_screening_evaluate.py"),
        "--scores",str(out/"scores.npy"),"--query-ids",str(out/"query_ids.txt"),
        "--candidate-ids",str(out/"candidate_ids.txt"),"--protocol",str(a.protocol),
        "--output",str(out/"test_evaluation")],check=True)


if __name__=="__main__":main()
