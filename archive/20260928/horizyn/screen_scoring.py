"""Cache single-model enzyme prototypes for full-library screening."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F


def export_scoring_head(model, output: Path):
    head = getattr(model, "enzyme_prototype_head", None)
    if head is None or head.prototype_count == 1:
        return {"kind": "cosine", "prototype_count": 1}
    path = output / "screen_head.pt"
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in head.state_dict().items()},
                    config=dict(embedding_dim=head.embedding_dim,
                                prototype_count=head.prototype_count,
                                bottleneck_dim=head.bottleneck_dim,
                                aggregation_temperature=head.aggregation_temperature)), path)
    return {"kind": "enzyme_prototypes", "prototype_count": head.prototype_count,
            "artifact": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class ScreenScorer:
    """Evaluate exactly the trained head; precompute sequence-only states once."""
    def __init__(self, candidates, *, head=None, block_size=8192, already_normalized=False):
        self.temperature = None
        self.already_normalized = already_normalized
        with torch.inference_mode():
            if head is None or head.prototype_count == 1:
                self.candidates = candidates.float() if already_normalized else F.normalize(candidates.float(), dim=-1)
            else:
                prototypes, priors = [], []
                for block in candidates.split(block_size):
                    p, d = head(block, return_details=True)
                    # head.score normalizes prototype features after forward.
                    prototypes.append(F.normalize(p.float(), dim=-1))
                    priors.append(d["log_priors"].float())
                self.candidates = torch.cat(prototypes)
                self.log_priors = torch.cat(priors)
                self.temperature = head.aggregation_temperature

    @classmethod
    def from_export(cls, candidates, directory, reaction_receipt):
        spec = reaction_receipt.get("scoring", {"kind": "cosine"})
        if spec["kind"] == "cosine":
            return cls(candidates)
        if spec["kind"] != "enzyme_prototypes":
            raise ValueError("Unknown screening score head")
        path = directory / spec["artifact"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError("Screening head artifact changed")
        from horizyn.model import ResidualEnzymePrototypeHead
        saved = torch.load(path, map_location=candidates.device, weights_only=True)
        head = ResidualEnzymePrototypeHead(**saved["config"]).to(candidates.device).eval()
        head.load_state_dict(saved["state_dict"], strict=True)
        return cls(candidates, head=head)

    def __call__(self, queries):
        q = queries.float() if self.already_normalized else F.normalize(queries.float(), dim=-1)
        if self.temperature is None:
            return q @ self.candidates.T
        scores = torch.einsum("qd,ekd->qek", q, self.candidates)
        return self.temperature * torch.logsumexp(
            scores / self.temperature + self.log_priors.unsqueeze(0), dim=-1)
