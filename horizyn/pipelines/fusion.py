"""V4 enzyme fusion, with the original fitted-checkpoint arithmetic."""

import torch
from torch.nn import functional as F


@torch.inference_mode()
def components(model, config, keys, device, batch_size=128):
    """Encode global/site views and verify that they reconstruct the base output."""
    from horizyn.benchmarks.retrieval import encode_residue_targets
    from horizyn.training_io import StoragePrecisionResidues

    encoder = model.model.multiview_encoder
    if encoder is None:
        raise ValueError("Fusion calibration requires the V4 multiview encoder")
    if not keys:
        raise ValueError("At least one enzyme ID is required")
    values = {"global": [], "fused": []}

    def capture(name):
        def hook(module, inputs, output):
            values[name].append(F.normalize(output.float(), dim=-1, eps=1e-6).cpu())

        return hook

    residues = StoragePrecisionResidues(
        config.data.protein_residue_embeds_path,
        max_tokens=config.data.max_protein_tokens,
        truncation=config.data.protein_truncation,
    )
    handles = [
        encoder.global_output.register_forward_hook(capture("global")),
        encoder.fused_output.register_forward_hook(capture("fused")),
    ]
    try:
        reference = encode_residue_targets(
            model, residues, keys, device, batch_size, False
        ).float()
    finally:
        for handle in handles:
            handle.remove()
        residues.close()
    g = torch.cat(values["global"]).to(device)
    f = torch.cat(values["fused"]).to(device)
    scale = float(encoder.residual_scale)
    reproduced = F.normalize(g + scale * f, dim=-1, eps=1e-6)
    error = float((reproduced.cpu() - reference).abs().max())
    if error > 1e-6:
        raise ValueError(f"Fusion reconstruction failed: {error}")
    return g, f, scale, reference.to(device), error


def adjusted(g, f, scale, multiplier, original):
    """Apply the multiplier once; calibrated checkpoints use multiplier=1."""
    if multiplier == 1:
        return original
    if not 0 < scale * multiplier < 0.5:
        raise ValueError("Fusion must retain a positive bounded site/view contribution")
    return F.normalize(g + (scale * multiplier) * f, dim=-1, eps=1e-6)
