"""Train-only linear semantic bridges with independently encodable endpoints."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def fit_normalizer(vectors, train_rows, mask=None):
    rows = train_rows if mask is None else train_rows[mask[train_rows]]
    if not len(rows):
        raise ValueError("No observed training vectors for normalizer")
    return F.normalize(vectors[rows], dim=1).mean(0)


def apply_normalizer(vectors, mean, mask=None):
    output = F.normalize(F.normalize(vectors, dim=1) - mean, dim=1)
    return output if mask is None else output * mask.unsqueeze(1)


def paired_statistics(queries, enzymes, reaction_index, enzyme_index):
    """Weighted centered moments: uniform reactions, then their positive edges."""
    x, y = queries.double(), enzymes.double()
    r, e = reaction_index, enzyme_index
    nr, ne = len(x), len(y)
    degree = torch.bincount(r, minlength=nr).double()
    if bool((degree == 0).any()):
        raise ValueError("Each reaction needs at least one training positive")
    edge_weight = degree[r].reciprocal()
    enzyme_weight = torch.zeros(ne, device=y.device, dtype=y.dtype).scatter_add_(0, e, edge_weight)
    if bool((enzyme_weight == 0).any()):
        raise ValueError("Each enzyme needs at least one training positive")
    y_per_reaction = torch.zeros((nr, y.shape[1]), device=y.device, dtype=y.dtype)
    y_per_reaction.index_add_(0, r, y[e] * edge_weight.unsqueeze(1))
    x_mean, y_mean = x.mean(0), (enzyme_weight @ y) / nr
    xc, yc = x - x_mean, y - y_mean
    xx = xc.T @ xc / nr
    yy = yc.T @ (yc * enzyme_weight.unsqueeze(1)) / nr
    xy = xc.T @ (y_per_reaction - y_mean) / nr
    dx, ux = torch.linalg.eigh(xx)
    dy, uy = torch.linalg.eigh(yy)
    return dict(x_mean=x_mean, y_mean=y_mean, xx=xx, yy=yy, xy=xy,
                x_eigenvalues=dx.clamp_min(0), y_eigenvalues=dy.clamp_min(0),
                x_eigenvectors=ux, y_eigenvectors=uy)


def fit_bridges(statistics, ridge_relative, cca_rank=256):
    if ridge_relative <= 0:
        raise ValueError("ridge_relative must be positive")
    s = statistics
    dx, dy = s["x_eigenvalues"], s["y_eigenvalues"]
    ux, uy = s["x_eigenvectors"], s["y_eigenvectors"]
    lx = ridge_relative * s["xx"].trace() / len(dx)
    ly = ridge_relative * s["yy"].trace() / len(dy)
    if float(lx) <= 0 or float(ly) <= 0:
        raise ValueError("Degenerate covariance")
    ix = (ux * (dx + lx).reciprocal()) @ ux.T
    iy = (uy * (dy + ly).reciprocal()) @ uy.T
    wx = (ux * (dx + lx).rsqrt()) @ ux.T
    wy = (uy * (dy + ly).rsqrt()) @ uy.T
    u, singular, vh = torch.linalg.svd(wx @ s["xy"] @ wy, full_matrices=False)
    rank = min(cca_rank, len(singular))
    common = dict(x_mean=s["x_mean"].float(), y_mean=s["y_mean"].float(),
                  ridge_relative=float(ridge_relative), ridge_x=float(lx), ridge_y=float(ly))
    return {
        "reaction_to_mean_enzyme": dict(common, kind="reaction_to_mean_enzyme", weight=(ix @ s["xy"]).float()),
        "enzyme_to_reaction": dict(common, kind="enzyme_to_reaction", weight=(iy @ s["xy"].T).float()),
        "cca": dict(common, kind="cca", query_weight=(wx @ u[:, :rank]).float(),
                    enzyme_weight=(wy @ vh[:rank].T).float(), singular_values=singular[:rank].float()),
    }


def encode_bridge(state, queries=None, enzymes=None):
    """Each endpoint can be encoded alone, without seeing the other endpoint."""
    q, e = queries, enzymes
    if state["kind"] == "reaction_to_mean_enzyme":
        if q is not None:
            q = (q - state["x_mean"]) @ state["weight"] + state["y_mean"]
    elif state["kind"] == "enzyme_to_reaction":
        if e is not None:
            e = (e - state["y_mean"]) @ state["weight"] + state["x_mean"]
    elif state["kind"] == "cca":
        if q is not None:
            q = (q - state["x_mean"]) @ state["query_weight"]
        if e is not None:
            e = (e - state["y_mean"]) @ state["enzyme_weight"]
    else:
        raise ValueError(state["kind"])
    return (None if q is None else F.normalize(q, dim=1),
            None if e is None else F.normalize(e, dim=1))
