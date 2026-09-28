"""V4 phase 2: uniform-positive full-graph alignment and identity regularization.

No dictionaries, biological-label auxiliary objectives, test data, or variant
sweeps are part of this trainer. Inputs use the existing frozen-feature schema.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from horizyn.generalization_residual import FrozenGeometryResidual, full_graph_contrastive_loss
from .inference import sha256


def train_refinement(features, output, *, device="cpu", steps=100, hidden=1024,
                     scale=0.2, temperature=0.2, identity_weight=10.0,
                     learning_rate=1e-4, weight_decay=0.01, seed=42):
    """Fit heads to training nodes only and save a fixed-step checkpoint.

    f3_features.npz holds `proteins` in catalog order and `train_reactions`
    in compact training-reaction order. pairs.npz's `train` array contains
    unique global (reaction, enzyme) index pairs. Other split arrays are unused.
    """
    if steps < 1 or temperature <= 0 or identity_weight < 0:
        raise ValueError("Require steps>=1, temperature>0, and identity_weight>=0")
    features, output = Path(features), Path(output)
    manifest = json.loads((features / "manifest.json").read_text())
    reactzyme_train_only = manifest.get("test_used") is False and manifest.get("training_edges_only") is True
    enzymemap_train_only = (manifest.get("schema") == "clipzyme_enzymemap_phase2_features_v1"
                           and manifest.get("test_labels_read") is False
                           and manifest.get("train_only_fit") is True)
    if not (reactzyme_train_only or enzymemap_train_only):
        raise ValueError("Refinement requires an explicit training-only feature manifest")
    graph_hash = manifest.get("training_graph", {}).get("sha256")
    if graph_hash and sha256(features / "pairs.npz") != graph_hash:
        raise ValueError("Training graph hash disagrees with its frozen manifest")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a fresh refinement output directory")
    catalog = json.loads((features / "catalog.json").read_text())
    with np.load(features / "pairs.npz", allow_pickle=False) as data:
        edges = data["train"]
    if edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("train must contain integer (reaction, enzyme) indices")
    if not len(edges) or len(np.unique(edges, axis=0)) != len(edges):
        raise ValueError("Training edges must be nonempty and unique")
    tr, te = np.unique(edges[:, 0]), np.unique(edges[:, 1])
    if tr[0] < 0 or te[0] < 0 or tr[-1] >= len(catalog["reactions"]) or te[-1] >= len(catalog["proteins"]):
        raise ValueError("Training edges are outside the catalog")
    lookup = {key: i for i, key in enumerate(catalog["reactions"])}
    if not np.array_equal(tr, [lookup[key] for key in catalog["train_reactions"]]):
        raise ValueError("Compact training-reaction order disagrees with catalog")
    torch.manual_seed(seed)
    with np.load(features / "f3_features.npz", allow_pickle=False) as data:
        e = torch.as_tensor(data["proteins"][te], device=device).float()
        r = torch.as_tensor(data["train_reactions"], device=device).float()
    if r.shape != (len(tr), e.shape[1]) or not torch.isfinite(e).all() or not torch.isfinite(r).all():
        raise ValueError("Training embeddings must be finite and match the graph")
    edge_r = torch.as_tensor(np.searchsorted(tr, edges[:, 0]), device=device)
    edge_e = torch.as_tensor(np.searchsorted(te, edges[:, 1]), device=device)
    model = FrozenGeometryResidual(e.shape[1], hidden, scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    r_reference, e_reference = F.normalize(r, dim=-1), F.normalize(e, dim=-1)
    history = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        u_r, u_e = model.encode_reactions(r), model.encode_enzymes(e)
        alignment, _, _ = full_graph_contrastive_loss(u_r @ u_e.T / temperature, edge_r, edge_e)
        identity = ((1 - (u_r * r_reference).sum(-1)).mean()
                    + (1 - (u_e * e_reference).sum(-1)).mean()) / 2
        loss = alignment + identity_weight * identity
        loss.backward()
        optimizer.step()
        history.append({"step": step, "loss": float(loss.detach()),
                        "alignment": float(alignment.detach()), "identity": float(identity.detach())})
    output.mkdir(parents=True, exist_ok=True)
    registry = {"schema": "v4_full_graph_v1", "test_used": False,
                "feature_manifest_sha256": sha256(features / "manifest.json"),
                "base_checkpoint": manifest.get("checkpoint"),
                "selection": "fixed final step; no validation or test selection",
                "seed": seed, "steps": steps, "temperature": temperature,
                "identity_weight": identity_weight, "learning_rate": learning_rate,
                "weight_decay": weight_decay}
    torch.save({"state_dict": model.state_dict(), "registry": registry, "fixed_step": steps,
                "model_config": {"dimension": e.shape[1], "hidden": hidden, "scale": scale}},
               output / f"step{steps:04d}.pt")
    (output / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    (output / "training.json").write_text(json.dumps(history, indent=2) + "\n")
    return model
