"""Masked reaction fusion: competitive mixing and the legacy residual block."""
import math

import torch
from torch import nn


class CompetitiveReactionFusion(nn.Module):
    """Jointly conditioned softmax mixing over available modality tokens.

    ``featurewise=True`` learns a separate competition for each feature channel;
    the scalar control broadcasts one weight per modality across all channels.
    Both variants receive the same flattened, already normalized tokens and
    availability mask. Zero final logits initialize an available-modality mean.
    """

    def __init__(self, dim: int, modalities: int, featurewise: bool = True):
        super().__init__()
        if dim <= 0 or modalities <= 0:
            raise ValueError("dim and modalities must be positive")
        self.dim = dim
        self.modalities = modalities
        self.featurewise = featurewise
        self.gate = nn.Sequential(
            nn.Linear(modalities * dim + modalities, dim),
            nn.ReLU(),
            nn.Linear(dim, modalities * dim if featurewise else modalities),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, tokens, valid, return_details=False):
        if tokens.ndim != 3 or tokens.shape[1:] != (self.modalities, self.dim):
            raise ValueError("Expected [batch, modalities, dim] tokens")
        if valid.shape != tokens.shape[:2]:
            raise ValueError("Invalid modality mask shape")
        valid = valid.to(device=tokens.device, dtype=torch.bool)
        clean = tokens.masked_fill(~valid[..., None], 0)
        gate_inputs = torch.cat((clean.flatten(start_dim=1), valid.to(tokens.dtype)), dim=-1)
        logits = self.gate(gate_inputs).reshape(tokens.shape[0], self.modalities, -1)
        if not self.featurewise:
            logits = logits.expand(-1, -1, self.dim)
        masked_logits = logits.masked_fill(~valid[..., None], torch.finfo(logits.dtype).min)
        # A finite sentinel gives a defined softmax even for fully missing rows.
        # Clearing its weights afterwards yields exactly zero output and gradient.
        weights = masked_logits.softmax(dim=1).masked_fill(~valid[..., None], 0)
        fused = (weights * clean).sum(dim=1)
        details = {}
        if return_details:
            # Compute entropy in float32 so clamp/log remain finite under fp16.
            safe_weights = weights.float()
            entropy = -(safe_weights.clamp_min(1e-12).log() * safe_weights).sum(dim=1)
            details = {
                "modality_feature_weights": weights,
                "modality_feature_logits": logits.masked_fill(~valid[..., None], 0),
                "modality_feature_entropy": entropy,
            }
        return fused, details


class GatedReactionFusion(nn.Module):
    def __init__(self, dim: int, modalities: int, heads: int = 4, gate_init: float = 0.1):
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads or modalities <= 0:
            raise ValueError('Positive dimensions required; dim must be divisible by heads')
        if not 0 < gate_init < 1:
            raise ValueError('gate_init must be in (0, 1)')
        self.modality_embedding = nn.Parameter(torch.zeros(modalities, dim))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.gates = nn.ModuleList([nn.Linear(2 * dim, dim) for _ in range(modalities)])
        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, math.log(gate_init / (1 - gate_init)))

    def forward(self, tokens, valid, return_details=False):
        if tokens.ndim != 3 or tokens.shape[1:] != self.modality_embedding.shape:
            raise ValueError('Expected [batch, modalities, dim] tokens')
        if valid.shape != tokens.shape[:2]:
            raise ValueError('Invalid modality mask shape')
        valid = valid.to(device=tokens.device, dtype=torch.bool)
        clean = tokens.masked_fill(~valid[..., None], 0)
        typed = self.norm(clean + self.modality_embedding).masked_fill(~valid[..., None], 0)
        # A null key is visible ONLY for fully missing rows; masked queries are
        # zeroed afterwards. This avoids all-masked softmax NaNs.
        keys = torch.cat([typed, typed.new_zeros(typed.shape[0], 1, typed.shape[2])], dim=1)
        key_valid = torch.cat([valid, ~valid.any(dim=1, keepdim=True)], dim=1)
        delta, attention = self.attention(typed, keys, keys, key_padding_mask=~key_valid,
                                          need_weights=return_details, average_attn_weights=False)
        delta = delta.masked_fill(~valid[..., None], 0)
        gates = torch.stack([torch.sigmoid(gate(torch.cat([clean[:, i], delta[:, i]], dim=-1)))
                             for i, gate in enumerate(self.gates)], dim=1)
        gates = gates.masked_fill(~valid[..., None], 0)
        fused = (clean + gates * delta).masked_fill(~valid[..., None], 0)
        details = {}
        if return_details:
            attention = attention[..., :-1].masked_fill(~valid[:, None, :, None], 0)
            details = {'cross_modal_gates': gates, 'cross_modal_attention': attention,
                       'cross_modal_update_norms': (gates * delta).norm(dim=-1)}
        return fused, details
