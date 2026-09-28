"""Export V4 phase-1 vectors and the training graph for residual refinement."""
import csv
import json
from pathlib import Path

import numpy as np

from horizyn.benchmarks.retrieval import BenchmarkTask, build_reaction_inputs
from .checkpoints import model_from_checkpoint
from .inference import RetrievalPipeline, sha256


def export_refinement(config_path, checkpoint, output, *, device="cpu", batch_size=128):
    """Read only training associations/features; use uncalibrated base vectors."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a fresh export directory")
    model, config = model_from_checkpoint(Path(config_path), Path(checkpoint), device)
    pipeline = RetrievalPipeline(model, config, fusion_multiplier=1.0, device=device)
    with Path(config.data.train_pairs_path).open(newline="") as stream:
        pairs = sorted({(row["reaction_id"], row["protein_id"]) for row in csv.DictReader(stream)})
    if not pairs:
        raise ValueError("Training associations must be nonempty")
    r_ids, e_ids = sorted({r for r, _ in pairs}), sorted({e for _, e in pairs})
    ri, ei = {key: i for i, key in enumerate(r_ids)}, {key: i for i, key in enumerate(e_ids)}
    edges = np.asarray([(ri[r], ei[e]) for r, e in pairs], dtype=np.int64)
    config.data.reaction_chemistry_vectors_path = config.data.train_reaction_chemistry_vectors_path
    task = BenchmarkTask(name="training", task_type="retrieval", dataset="training",
                         task_label="phase1 export", split="train",
                         pairs=Path(config.data.train_pairs_path),
                         reactions=Path(config.data.train_reactions_path),
                         reaction_model_embeds_h5=Path(config.data.train_reaction_t5v2_embeds_path),
                         reaction_unimol2_embeds_h5=Path(config.data.train_reaction_unimol2_embeds_path),
                         reaction_chiro_embeds_h5=Path(config.data.train_reaction_chiro_embeds_path))
    inputs = build_reaction_inputs(task, config)
    try:
        r = pipeline.encode_reactions(inputs, r_ids, batch_size).cpu().numpy()
        e = pipeline.encode_enzymes(e_ids, batch_size).cpu().numpy()
    finally:
        if hasattr(inputs, "close"):
            inputs.close()
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "pairs.npz", train=edges)
    np.savez(output / "f3_features.npz", proteins=e, reactions=r, train_reactions=r)
    (output / "catalog.json").write_text(json.dumps({"reactions": r_ids, "proteins": e_ids,
                                                    "train_reactions": r_ids}, indent=2) + "\n")
    def record(path):
        return {"path": str(Path(path).resolve()), "sha256": sha256(path)}
    manifest = {"schema": "v4_training_features_v1", "training_edges_only": True,
                "test_used": False, "checkpoint": record(checkpoint),
                "config": record(config_path), "training_graph": record(output / "pairs.npz"),
                "training_pairs": record(config.data.train_pairs_path),
                "training_reactions": record(config.data.train_reactions_path),
                "fusion_multiplier": 1.0}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
