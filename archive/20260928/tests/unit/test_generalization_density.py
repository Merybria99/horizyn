import hashlib
import json
import pytest
import torch
from torch.nn import functional as F

from horizyn.generalization_density import DensityGatedEncoder, nearest_support
from horizyn.generalization_residual import FrozenGeometryResidual
from horizyn.semantic_anchors import row_unit, centered_unit, reaction_features


def synthetic_model():
    torch.manual_seed(19)
    residual = FrozenGeometryResidual(4, 8, .2)
    for tower in (residual.enzyme, residual.reaction):
        torch.nn.init.normal_(tower[-1].weight, std=.1)
    center = torch.zeros(6)
    centers = dict(t5v2=torch.zeros(5), chemistry=torch.zeros(3))
    anchors_p = row_unit(torch.randn(7, 6))
    anchors_q = row_unit(torch.randn(7, 8))
    selected = dict(space="raw", endpoints="both", cap=1., lower_quantile=.25, upper_quantile=.95)
    thresholds = dict(raw={endpoint:{"0.25":.2,"0.95":.9} for endpoint in ("reaction", "enzyme")})
    return DensityGatedEncoder(residual, selected, thresholds, center, centers, anchors_p, anchors_q, tuple(centers))


def test_density_endpoints_support_subset_and_lookup_batching():
    model = synthetic_model()
    proteins = F.normalize(torch.randn(13, 4), dim=1)
    means = torch.randn(13, 6)
    reactions = F.normalize(torch.randn(11, 4), dim=1)
    blocks = dict(t5v2=torch.randn(11, 5), chemistry=torch.randn(11, 3))
    masks = {key: torch.ones(11, dtype=torch.bool) for key in blocks}
    ep, diagnostics = model.encode_enzymes(proteins, means, batch_size=2, return_diagnostics=True)
    eq = model.encode_reactions(reactions, blocks, masks, batch_size=3)
    assert torch.equal(ep, model.encode_enzymes(proteins, means, batch_size=7))
    assert torch.equal(eq, model.encode_reactions(reactions, blocks, masks, batch_size=11))
    indices = torch.tensor([9, 1, 4])
    assert torch.allclose(ep[indices], model.encode_enzymes(proteins[indices], means[indices]), atol=1e-7, rtol=1e-6)
    assert torch.allclose(eq[indices], model.encode_reactions(reactions[indices], {k:v[indices] for k,v in blocks.items()},
        {k:v[indices] for k,v in masks.items()}), atol=1e-7, rtol=1e-6)
    assert torch.all((diagnostics["gate_scale"] >= 0) & (diagnostics["gate_scale"] <= 1))
    assert torch.allclose(ep.norm(dim=1), torch.ones(len(ep)), atol=1e-6)


def test_far_endpoint_reduces_exactly_to_f3_without_opposing_batch():
    model = synthetic_model()
    model.thresholds["raw"]["enzyme"] = {"0.25":1., "0.95":1.}
    base, means = torch.randn(3, 4), torch.randn(3, 6)
    encoded, diagnostics = model.encode_enzymes(base, means, return_diagnostics=True)
    assert torch.equal(diagnostics["gate_scale"], torch.zeros(3))
    assert torch.equal(encoded, F.normalize(base, dim=1))


def test_stable_postprocessor_is_bitwise_invariant_for_subsets_and_singletons():
    model = synthetic_model()
    model.inference_precision = "stable_fp64"
    model.residual.double()
    base, means = torch.randn(13, 4), torch.randn(13, 6)
    full = model.encode_enzymes(base, means)
    assert torch.equal(full, torch.cat([model.encode_enzymes(base[i:i+1], means[i:i+1]) for i in range(len(base))]))
    subset = torch.tensor([12, 1, 9, 3])
    assert torch.equal(full[subset], model.encode_enzymes(base[subset], means[subset]))


def test_bundle_rejects_tampered_gate_before_loading(tmp_path):
    gate = tmp_path / "gate.pt"
    gate.write_bytes(b"modified gate")
    bundle = tmp_path / "density_bundle.json"
    bundle.write_text(json.dumps(dict(schema="phase2_density_bundle_v1",test_used=False,
        gate=dict(path=str(gate),sha256="0"*64))))
    with pytest.raises(ValueError,match="hash mismatch"):
        DensityGatedEncoder.from_bundle(bundle)


def test_artifact_rejects_changed_feature_source_before_loading(tmp_path):
    def record(path):
        return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    gate = tmp_path / "selected_gate.pt"
    torch.save({},gate)
    selected = tmp_path / "results.json"
    selected.write_text("{}")
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}")
    (tmp_path / "registry.json").write_text(json.dumps(dict(test_used=False,sources={
        "catalog.json":dict(path=str(catalog),sha256="0"*64)})))
    (tmp_path / "artifact_receipt.json").write_text(json.dumps(dict(gate=record(gate),selection=record(selected))))
    with pytest.raises(ValueError,match="hash mismatch"):
        DensityGatedEncoder.from_artifact(gate,tmp_path)
