"""Independent support-gated frozen-F3 residual encoder for phase2 research."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from horizyn.generalization_residual import FrozenGeometryResidual
from horizyn.generalization_retrieval import checked_artifact
from horizyn.semantic_anchors import centered_unit, reaction_features, row_unit


@torch.inference_mode()
def nearest_support(query, anchors, batch_size=512):
    if batch_size < 1 or not len(anchors):
        raise ValueError("Positive batch size and nonempty training anchors required")
    precise = anchors.double()
    values = [(query[start:start + batch_size].double() @ precise.T).amax(1).float().clamp(-1, 1)
              for start in range(0, len(query), batch_size)]
    return torch.cat(values) if values else query.new_empty((0,))


def support_gate(similarity, low, high, cap):
    if high < low:
        raise ValueError("Reversed training support quantiles")
    return ((similarity >= high).to(similarity.dtype) * cap if high == low
            else ((similarity - low) / (high - low)).clamp(0, 1) * cap)


class DensityGatedEncoder(nn.Module):
    """Each endpoint gate depends on itself and a fixed training reference only."""

    def __init__(self, residual, selected_gate, thresholds, protein_center,
                 reaction_centers, protein_anchors, reaction_anchors, modalities,
                 inference_precision="legacy_fp32"):
        super().__init__()
        self.residual = residual.eval()
        self.inference_precision = inference_precision
        if inference_precision == "stable_fp64":
            self.residual.double()
        elif inference_precision != "legacy_fp32":
            raise ValueError("Unknown density numerical contract")
        self.selected_gate, self.thresholds = selected_gate, thresholds
        self.modalities = tuple(modalities)
        self.register_buffer("protein_center", protein_center)
        self.register_buffer("protein_anchors", protein_anchors)
        self.register_buffer("reaction_anchors", reaction_anchors)
        for key, value in reaction_centers.items():
            self.register_buffer(f"reaction_center_{key}", value)

    @property
    def reaction_centers(self):
        return {key: getattr(self, f"reaction_center_{key}") for key in self.modalities}

    def _gate(self, endpoint, base, delta, raw_encoded, batch_size):
        specification = self.selected_gate
        space, cap = specification["space"], specification["cap"]
        if space == "pointwise_norm":
            return (cap * base.norm(dim=1) / delta.norm(dim=1).clamp_min(1e-12)).clamp(max=1), None
        if space == "constant" or specification["endpoints"] not in ("both", endpoint):
            return base.new_full((len(base),), cap), None
        encoded = raw_encoded if space == "raw" else row_unit(base)
        anchors = self.protein_anchors if endpoint == "enzyme" else self.reaction_anchors
        similarity = nearest_support(encoded, anchors, batch_size)
        bounds = self.thresholds[space][endpoint]
        return support_gate(similarity, bounds[str(specification["lower_quantile"])],
                            bounds[str(specification["upper_quantile"])], cap), similarity

    def _residual_delta(self, base, endpoint):
        inputs = base.double() if self.inference_precision == "stable_fp64" else base
        return self.residual.scale * getattr(self.residual, endpoint)(inputs)

    def _normalize(self, base, delta, gates):
        if self.inference_precision == "stable_fp64":
            return F.normalize(base.double() + gates.double()[:, None] * delta.double(), dim=-1).to(base.dtype)
        return F.normalize(base + gates[:, None] * delta, dim=-1)

    @torch.inference_mode()
    def encode_enzymes(self, native_f3, raw_mean, batch_size=2048, return_diagnostics=False):
        if native_f3.ndim != 2 or native_f3.shape[1] != self.residual.dimension or len(raw_mean) != len(native_f3):
            raise ValueError("Protein base/raw feature dimensions disagree")
        encoded = centered_unit(raw_mean, self.protein_center)
        # The tower executes once for the submitted endpoint catalog, exactly
        # matching the validation screen. batch_size bounds reference lookup.
        delta = self._residual_delta(native_f3, "enzyme")
        gates, support = self._gate("enzyme", native_f3, delta, encoded, batch_size)
        result = self._normalize(native_f3, delta, gates)
        return (result, dict(support_cosine=support, gate_scale=gates)) if return_diagnostics else result

    @torch.inference_mode()
    def encode_reactions(self, native_f3, blocks, masks, batch_size=2048, return_diagnostics=False):
        if any(len(value) != len(native_f3) for value in blocks.values()):
            raise ValueError("Reaction base/raw feature dimensions disagree")
        encoded = reaction_features(blocks, self.reaction_centers, masks, self.modalities)
        delta = self._residual_delta(native_f3, "reaction")
        gates, support = self._gate("reaction", native_f3, delta, encoded, batch_size)
        result = self._normalize(native_f3, delta, gates)
        return (result, dict(support_cosine=support, gate_scale=gates)) if return_diagnostics else result

    @classmethod
    def from_artifact(cls, gate_path, features_dir, device="cpu"):
        gate_path, features_dir = Path(gate_path).resolve(), Path(features_dir).resolve()
        registry_path = gate_path.parent / "registry.json"
        registry = json.loads(registry_path.read_text())
        if registry.get("test_used") is not False:
            raise ValueError("Density fitting artifact must use training/validation only")
        receipt_path = gate_path.parent / "artifact_receipt.json"
        if not receipt_path.exists():
            receipt_path = gate_path.parent / "selected_validation_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        if checked_artifact(receipt["gate"], gate_path.parent).resolve() != gate_path:
            raise ValueError("Selected gate identity mismatch")
        checked_artifact(receipt["selection"], gate_path.parent)
        for name in ("catalog.json", "manifest.json", "pairs.npz", "f3_features.npz", "reaction_features.npz", "protein_mean.h5"):
            if checked_artifact(registry["sources"][name], gate_path.parent).resolve() != features_dir / name:
                raise ValueError(f"Density feature source identity mismatch: {name}")
        graph_path = checked_artifact(registry["graph_checkpoint"], gate_path.parent)
        gate = torch.load(gate_path, map_location="cpu", weights_only=False)
        if checked_artifact(gate["graph_checkpoint"], gate_path.parent).resolve() != graph_path:
            raise ValueError("Density gate and graph checkpoint disagree")
        results = json.loads(Path(receipt["selection"]["path"]).read_text())
        if results["selected"]["candidate"] != gate["selected_gate"]:
            raise ValueError("Gate configuration differs from recorded validation selection")
        catalog = json.loads((features_dir / "catalog.json").read_text())
        with np.load(features_dir / "pairs.npz") as source:
            tr, te = np.unique(source["train"][:, 0]), np.unique(source["train"][:, 1])
        snapshot = torch.load(graph_path, map_location=device, weights_only=False)
        graph_registry = snapshot.get("registry", {})
        if (graph_registry.get("feature_manifest_sha256") != registry["sources"]["manifest.json"]["sha256"]
                or graph_registry.get("test_used") is not False):
            raise ValueError("Residual and density training feature provenance disagree")
        residual = FrozenGeometryResidual(**snapshot["model_config"]).to(device)
        residual.load_state_dict(snapshot["state_dict"])
        center_e = gate["raw_protein_center"].to(device)
        centers_r = {key: value.to(device) for key, value in gate["reaction_centers"].items()}
        modalities = tuple(centers_r)
        if gate["selected_gate"]["space"] == "f3":
            with np.load(features_dir / "f3_features.npz") as source:
                anchors_e = row_unit(torch.tensor(source["proteins"][te], device=device))
                anchors_r = row_unit(torch.tensor(source["train_reactions"], device=device))
        else:
            with h5py.File(features_dir / "protein_mean.h5") as source:
                if source["ids"].asstr()[:].tolist() != catalog["proteins"] or not source["complete"][:].all():
                    raise ValueError("Incomplete or reordered protein means")
                anchors_e = centered_unit(torch.tensor(source["vectors"][:][te], device=device), center_e)
            with np.load(features_dir / "reaction_features.npz") as source:
                blocks = {key: torch.tensor(source[key][tr], device=device) for key in modalities}
                masks = {key: torch.tensor(source[key + "_mask"][tr], device=device) for key in modalities}
            anchors_r = reaction_features(blocks, centers_r, masks, modalities)
        model = cls(residual, gate["selected_gate"], gate["thresholds"], center_e,
                    centers_r, anchors_e, anchors_r, modalities,
                    gate.get("inference_precision", "legacy_fp32")).to(device).eval()
        return model, dict(gate=gate, registry=registry, feature_catalog=catalog,
                          provenance_receipt=receipt, graph_checkpoint=snapshot)

    @classmethod
    def from_bundle(cls, bundle_path, device="cpu"):
        bundle_path = Path(bundle_path).resolve()
        bundle = json.loads(bundle_path.read_text())
        if bundle.get("schema") != "phase2_density_bundle_v1" or bundle.get("test_used") is not False:
            raise ValueError("Unrecognized or test-fitted density bundle")
        gate = checked_artifact(bundle["gate"], bundle_path.parent)
        registry = checked_artifact(bundle["registry"], bundle_path.parent)
        receipt = checked_artifact(bundle["artifact_receipt"], bundle_path.parent)
        if registry.resolve() != gate.parent / "registry.json" or receipt.resolve() != gate.parent / "artifact_receipt.json":
            raise ValueError("Density bundle points to unrelated gate provenance")
        graph = checked_artifact(bundle["graph_checkpoint"], bundle_path.parent)
        features = Path(bundle["features"]).resolve()
        feature_manifest = checked_artifact(bundle["feature_manifest"], bundle_path.parent)
        if feature_manifest.resolve() != features / "manifest.json":
            raise ValueError("Density bundle feature manifest identity mismatch")
        model, state = cls.from_artifact(gate, features, device)
        if Path(state["gate"]["graph_checkpoint"]["path"]).resolve() != graph.resolve():
            raise ValueError("Bundle selects another residual checkpoint")
        if (state["gate"].get("inference_precision") != bundle["inference_precision"]
                or state["gate"]["selected_gate"] != bundle["gate_recipe"]
                or state["graph_checkpoint"]["registry"]["arguments"]["seed"] != bundle["seed"]):
            raise ValueError("Density bundle recipe, numerical contract, or seed changed")
        state["bundle"] = bundle
        return model, state
