"""Isolated official Horizyn development-model worker for activity panels."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.activity_panels import rows, write_rows, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec",required=True); args=p.parse_args(); spec=json.loads(Path(args.spec).read_text())
    upstream=Path(spec["upstream"])
    if subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip()!=spec["revision"]:
        raise ValueError("Official source revision changed")
    if subprocess.check_output(["git","-C",str(upstream),"diff","HEAD","--"],text=True).strip():
        raise ValueError("Modified official source")
    if digest(spec["checkpoint"])!=spec["checkpoint_sha256"]: raise ValueError("Checkpoint changed")
    sys.path.insert(0,str(upstream))
    import h5py
    import numpy as np
    import torch
    from horizyn.lightning_module import HorizynLitModule
    from horizyn.config import load_config
    torch.serialization.add_safe_globals([argparse.Namespace])
    model=HorizynLitModule.load_from_checkpoint(spec["checkpoint"],map_location="cpu",weights_only=True).eval().to(spec["device"])
    module_spec=importlib.util.spec_from_file_location("official_activity_predict",upstream/"scripts/predict.py")
    native=importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(native)
    config=load_config(upstream/"configs/sota.yaml")
    proteins=rows(Path(spec["bundle"])/"proteins.csv"); variants=rows(Path(spec["bundle"])/"reactions.csv")
    with h5py.File(spec["features"],"r") as handle:
        ids=list(handle["ids"].asstr()[:]); idx={v:i for i,v in enumerate(ids)}; offsets=handle["offsets"][:]
        means=[]
        for row in proteins:
            i=idx[row["protein_id"]]; start,end=int(offsets[i]),int(offsets[i+1])
            if end-start!=len(row["sequence"]): raise ValueError("Official model requires full-sequence mean; truncated cache is not accepted")
            means.append(np.asarray(handle["vectors"][start:end],dtype=np.float32).mean(0))
    with torch.inference_mode():
        targets=model.model.target_encoder(torch.from_numpy(np.stack(means)).to(spec["device"]))
        queries=torch.cat([model.model.query_encoder(native.build_reaction_fingerprint(r["reaction_smiles"],config)[0].to(spec["device"])) for r in variants])
        matrix=(queries@targets.T).cpu().numpy()
    if not np.isfinite(matrix).all(): raise ValueError("Nonfinite official model scores")
    write_rows(spec["output"],[dict(reaction_id=r["reaction_id"],query_id=r["query_id"],protein_id=p["protein_id"],score=float(matrix[i,j])) for i,r in enumerate(variants) for j,p in enumerate(proteins)])


if __name__=="__main__":main()
