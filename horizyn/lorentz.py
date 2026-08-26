"""Lorentz-model hyperbolic geometry utilities."""

from __future__ import annotations

import torch


def _kappa_like(kappa: float, ref: torch.Tensor) -> torch.Tensor:
    if kappa <= 0:
        raise ValueError("curvature kappa must be positive")
    return torch.as_tensor(kappa, dtype=ref.dtype, device=ref.device)


def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Lorentz inner product with time coordinate in the first column."""
    if x.shape[-1] != y.shape[-1]:
        raise ValueError(f"dimension mismatch: x={tuple(x.shape)}, y={tuple(y.shape)}")
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)


def lorentz_origin(
    dim: int,
    kappa: float = 0.25,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the Lorentz origin in R^{dim+1}."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    if kappa <= 0:
        raise ValueError("curvature kappa must be positive")
    output = torch.zeros(dim + 1, dtype=dtype, device=device)
    output[0] = 1.0 / (kappa**0.5)
    return output


def expmap0_lorentz(
    u: torch.Tensor,
    kappa: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Exponential map at the Lorentz origin for tangent vectors [*, D]."""
    if u.ndim < 1:
        raise ValueError("u must have at least one dimension")
    k_t = _kappa_like(kappa, u)
    sqrt_k = torch.sqrt(k_t)
    norm = u.norm(dim=-1, keepdim=True)
    scaled_norm = sqrt_k * norm
    spatial_factor = torch.where(
        scaled_norm > eps,
        torch.sinh(scaled_norm) / scaled_norm.clamp_min(eps),
        torch.ones_like(scaled_norm),
    )
    spatial = spatial_factor * u
    time = torch.cosh(scaled_norm) / sqrt_k
    return torch.cat([time, spatial], dim=-1)


def logmap0_lorentz(
    x: torch.Tensor,
    kappa: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Logarithmic map at the Lorentz origin, returning tangent vectors [*, D]."""
    if x.ndim < 1 or x.shape[-1] < 2:
        raise ValueError("x must have shape [*, D+1]")
    k_t = _kappa_like(kappa, x)
    sqrt_k = torch.sqrt(k_t)
    time = x[..., :1]
    spatial = x[..., 1:]
    acosh_arg = (sqrt_k * time).clamp_min(1.0 + eps)
    tangent_norm = torch.acosh(acosh_arg) / sqrt_k
    spatial_norm = spatial.norm(dim=-1, keepdim=True)
    direction = spatial / spatial_norm.clamp_min(eps)
    tangent = tangent_norm * direction
    return torch.where(spatial_norm > eps, tangent, torch.zeros_like(spatial))


def lorentz_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    kappa: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Lorentz geodesic distance between corresponding points."""
    k_t = _kappa_like(kappa, x)
    sqrt_k = torch.sqrt(k_t)
    acosh_arg = (-k_t * lorentz_inner(x, y)).clamp_min(1.0 + eps)
    return torch.acosh(acosh_arg) / sqrt_k


def pairwise_lorentz_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    kappa: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pairwise Lorentz distances for x [N, D+1] and y [M, D+1]."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("pairwise_lorentz_distance expects rank-2 tensors")
    if x.shape[-1] != y.shape[-1]:
        raise ValueError(f"dimension mismatch: x={tuple(x.shape)}, y={tuple(y.shape)}")
    return lorentz_distance(x[:, None, :], y[None, :, :], kappa=kappa, eps=eps)


def lorentz_constraint_error(
    x: torch.Tensor,
    kappa: float = 0.25,
) -> torch.Tensor:
    """Absolute error from <x,x>_L = -1/kappa."""
    if kappa <= 0:
        raise ValueError("curvature kappa must be positive")
    return (lorentz_inner(x, x) + 1.0 / kappa).abs()


__all__ = [
    "lorentz_inner",
    "lorentz_origin",
    "expmap0_lorentz",
    "logmap0_lorentz",
    "lorentz_distance",
    "pairwise_lorentz_distance",
    "lorentz_constraint_error",
]
