"""Score the fixed measured esterase panel with the official Horizyn-1 dev model.

This is a retrospective, frozen-checkpoint external comparison. It never trains
or selects a model on assay labels and retains all 86 x 145 measured pairs.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np

from cyp_external_common import DATA, ROOT, safe_torch, source
from generalization_nitrilase_evaluate import evaluate_matrix, paired_bootstrap


PANEL = ROOT / "runs/generalization_20260919_2251/esterase_audit"
OUTPUT = PANEL / "public_horizyn1_dev"
CHECKPOINT = DATA / "release/horizyn_v1_0_dev.ckpt.part"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_panel():
    catalog_path = PANEL / "features/catalog.json"
    catalog = json.loads(catalog_path.read_text())
    reaction_rows = read_csv(PANEL / "inputs/reactions.csv")
    protein_rows = read_csv(PANEL / "inputs/proteins.csv")
    reactions = {row["reaction_id"]: row["reaction_smiles"] for row in reaction_rows}
    proteins = {row["protein_id"]: row["sequence"] for row in protein_rows}
    qids, pids = catalog["reactions"], catalog["proteins"]
    if (len(qids), len(pids)) != (86, 145) or list(reactions) != qids or list(proteins) != pids:
        raise ValueError("Changed esterase query/candidate pool")
    label_path = PANEL / "evaluation_v3/all_assay_scores.csv"
    indices = {(q, p): (i, j) for i, q in enumerate(qids) for j, p in enumerate(pids)}
    labels = np.full((86, 145), -1, dtype=np.int8)
    prior = {key: np.full((86, 145), np.nan, dtype=np.float64) for key in ("F3_native", "phase2", "morgan")}
    for row in read_csv(label_path):
        method = row["method"]
        if method not in prior:
            continue
        pair = (row["reaction_id"], row["protein_id"])
        if pair not in indices:
            raise ValueError("Unknown assay pair")
        i, j = indices[pair]
        value = int(row["detected"])
        if value not in (0, 1) or (labels[i, j] >= 0 and labels[i, j] != value):
            raise ValueError("Conflicting assay label")
        labels[i, j] = value
        if np.isfinite(prior[method][i, j]):
            raise ValueError("Duplicate prior score")
        prior[method][i, j] = float(row["score"])
    if (labels < 0).any() or int(labels.sum()) != 2565 or any(not np.isfinite(s).all() for s in prior.values()):
        raise ValueError("Incomplete measured panel or prior score matrix")
    return qids, pids, reactions, proteins, labels, prior, catalog_path, label_path


def load_protein_means(pids: list[str], proteins: dict[str, str]) -> np.ndarray:
    path = PANEL / "features/proteins.h5"
    with h5py.File(path, "r") as handle:
        ids = list(handle["ids"].asstr()[:])
        offsets = handle["offsets"][:]
        if ids != pids or handle["vectors"].shape[1] != 1024 or handle.attrs["embedding_model_type"] != "prott5":
            raise ValueError("Wrong residue cache")
        means = []
        for i, pid in enumerate(ids):
            if offsets[i + 1] - offsets[i] != len(proteins[pid]):
                raise ValueError(f"Wrong residue count: {pid}")
            means.append(np.asarray(handle["vectors"][offsets[i]:offsets[i + 1]], dtype=np.float32).mean(0))
    return np.stack(means)


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise ValueError("Preserve existing public-checkpoint result")
    qids, pids, reactions, proteins, labels, prior, catalog_path, label_path = load_panel()
    upstream = source("horizyn1_dev")
    torch = safe_torch()
    from horizyn.config import load_config
    from horizyn.lightning_module import HorizynLitModule

    if hashlib.file_digest(CHECKPOINT.open("rb"), "md5").hexdigest() != "5b1f938f8b0a82fbe91892a3b4e2bf2c":
        raise ValueError("Official Horizyn-1 dev checkpoint checksum mismatch")
    model = HorizynLitModule.load_from_checkpoint(str(CHECKPOINT), map_location="cpu", weights_only=True).eval()
    spec = importlib.util.spec_from_file_location("official_predict", upstream / "scripts/predict.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    config = load_config(upstream / "configs/sota.yaml")
    means = load_protein_means(pids, proteins)
    fingerprints = [native.build_reaction_fingerprint(reactions[q], config)[0] for q in qids]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    with torch.inference_mode():
        p = torch.cat([
            model.model.target_encoder(torch.from_numpy(means[i:i + 128]).to(device)).cpu()
            for i in range(0, len(pids), 128)
        ])
        q = torch.cat([model.model.query_encoder(fp.to(device)).cpu() for fp in fingerprints])
        scores = (q @ p.T).numpy().astype(np.float64)
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Missing or nonfinite score")
    metrics = evaluate_matrix(scores, labels)
    references = {name: evaluate_matrix(array, labels) for name, array in prior.items()}
    differences = {name: paired_bootstrap(metrics, result, replicates=10000, seed=20260920)
                   for name, result in references.items()}
    OUTPUT.mkdir(parents=True)
    score_path = OUTPUT / "scores.csv"
    with score_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("reaction_id", "protein_id", "detected", "score"))
        for i, rid in enumerate(qids):
            for j, pid in enumerate(pids):
                writer.writerow((rid, pid, int(labels[i, j]), float(scores[i, j])))
    result = {
        "schema": "esterase_public_horizyn1_dev_comparison_v1",
        "model": "official Horizyn-1 train-split development checkpoint",
        "input_scope": "physical forward reaction; complete ProtT5 residue mean; all 86 x 145 measured pairs",
        "device": device,
        "checkpoint_sha256": sha256(CHECKPOINT),
        "upstream_revision": __import__("subprocess").check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in (
            catalog_path, PANEL / "inputs/reactions.csv", PANEL / "inputs/proteins.csv",
            PANEL / "features/proteins.h5", label_path)},
        "implementation_sha256": sha256(Path(__file__)),
        "scores_sha256": sha256(score_path),
        "assayed_pairs": int(labels.size), "detected_pairs": int(labels.sum()),
        "metrics": metrics, "prior_metrics": references,
        "paired_query_bootstrap_95pct": differences,
        "test_labels_used_for_model_selection": False,
        "limitations": ["retrospective, already inspected esterase panel", "training exposure not fully excluded",
                        "physical reactions and wet-cell activity do not isolate product specificity",
                        "paired intervals condition on fixed checkpoints and this one enzyme family"],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "output": str(OUTPUT), "checkpoint_sha256": result["checkpoint_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
