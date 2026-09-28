"""Fixed-pool inputs and auditable outputs for native CYP baseline inference."""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external/cyp_specificity_2026"
BENCH = DATA / "benchmark"
RUN = ROOT / "runs/cyp_external_v1"
ASSETS = RUN / "assets"
SOURCES = {
    "horizyn1_dev": ("horizyn_official", "6944198303f2f0946d448a259ab788589cfd27b3"),
    "clipzyme_pretrained": ("clipzyme_official", "6e48ae05e2cf705af16368afc579246d80767326"),
    "enzymecage_pretrained": ("enzymecage_cyp", "c6c7dc64b7fffa5b6b265fb52f7b61a566123286"),
}


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def inputs():
    from cyp_specificity import verify_bundle
    verify_bundle(BENCH)
    mapped = rows(ROOT / "runs/cyp_baselines_v1/inputs/mapped_inputs.csv")
    pairs = rows(BENCH / "candidates.csv")
    expected = {(r["query_id"], r["protein_id"]) for r in pairs}
    if len(mapped) != len(expected) or {(r["query_id"], r["protein_id"]) for r in mapped} != expected:
        raise ValueError("Mapped inputs do not cover the frozen pool exactly")
    proteins = {r["protein_id"]: r["sequence"] for r in rows(BENCH / "proteins.csv")}
    if any(proteins[r["protein_id"]] != r["sequence"] for r in mapped):
        raise ValueError("Sequence mismatch")
    return mapped, proteins


def source(method):
    directory, revision = SOURCES[method]
    path = DATA / directory
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(path), "diff", "HEAD", "--"], text=True)
    if actual != revision or dirty:
        raise ValueError(f"Changed upstream source: {path}")
    sys.path.insert(0, str(path))
    return path


def safe_torch():
    import torch
    import numpy as np
    from torch_geometric.data import Data, HeteroData
    from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage, GlobalStorage
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    torch.serialization.add_safe_globals([
        argparse.Namespace, Data, HeteroData, BaseStorage, NodeStorage, EdgeStorage,
        GlobalStorage, DataEdgeAttr, DataTensorAttr, np.ndarray, np.dtype,
        np.core.multiarray._reconstruct, np.core.multiarray.scalar,
        type(np.dtype("float32")), type(np.dtype("float64")), type(np.dtype("int64")),
    ])
    return torch


def finish(method, predictions, checkpoint, scope, **extra):
    mapped, _ = inputs()
    expected = {(r["query_id"], r["protein_id"]) for r in mapped}
    if predictions.keys() != expected or not all(math.isfinite(float(v)) for v in predictions.values()):
        raise ValueError("Incomplete or nonfinite scores; no results published")
    output = RUN / method
    output.mkdir(parents=True, exist_ok=True)
    dest = output / "scores.csv"
    if dest.exists():
        raise FileExistsError(f"Preserving previous scores: {dest}")
    temp = dest.with_suffix(".partial")
    with temp.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["query_id", "protein_id", "score"])
        w.writerows((q, p, float(predictions[q, p])) for q, p in sorted(expected))
    metadata = dict(method=method, variant="pretrained", checkpoint_sha256=digest(checkpoint),
        source_revision=SOURCES[method][1], input_manifest_sha256=digest(BENCH / "manifest.json"),
        input_scope=scope, upstream_exposure="UNKNOWN: upstream training overlap has not been excluded",
        checkpoint_selection="fixed_without_cyp_selection", generator_command=" ".join(sys.argv),
        scores_sha256=digest(temp), score_direction="higher_is_better",
        runner_files={p.name: digest(p) for p in sorted((ROOT / "scripts").glob("cyp_native_*.py"))},
        common_sha256=digest(__file__), **extra)
    metadata_temp = dest.with_suffix(".json.partial")
    metadata_temp.write_text(json.dumps(metadata, indent=2) + "\n")
    metadata_temp.replace(dest.with_suffix(".json"))
    temp.replace(dest)
    print(f"Complete: {method}, {len(predictions)} scores", flush=True)
