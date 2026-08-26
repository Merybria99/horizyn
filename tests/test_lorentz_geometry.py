from __future__ import annotations

import torch

from horizyn.lorentz import (
    expmap0_lorentz,
    logmap0_lorentz,
    lorentz_constraint_error,
    lorentz_distance,
    pairwise_lorentz_distance,
)


def test_lorentz_expmap_logmap_are_finite_and_on_manifold() -> None:
    torch.manual_seed(0)
    curvature = 0.25
    tangent = torch.randn(8, 512) * 0.1

    hyp = expmap0_lorentz(tangent, kappa=curvature)
    assert hyp.shape == (8, 513)
    assert torch.isfinite(hyp).all()
    assert torch.all(lorentz_constraint_error(hyp, kappa=curvature) < 1e-4)

    recovered = logmap0_lorentz(hyp, kappa=curvature)
    assert recovered.shape == (8, 512)
    assert torch.isfinite(recovered).all()

    distances = lorentz_distance(hyp, hyp, kappa=curvature)
    assert distances.shape == (8,)
    assert torch.isfinite(distances).all()
    assert torch.all(distances >= 0.0)


def test_pairwise_lorentz_distance_shape_and_values() -> None:
    torch.manual_seed(1)
    curvature = 0.25
    x = expmap0_lorentz(torch.randn(8, 512) * 0.1, kappa=curvature)
    y = expmap0_lorentz(torch.randn(8, 512) * 0.1, kappa=curvature)

    distances = pairwise_lorentz_distance(x, y, kappa=curvature)
    assert distances.shape == (8, 8)
    assert torch.isfinite(distances).all()
    assert torch.all(distances >= 0.0)
