"""A single inference interface for all supported pipelines."""

import hashlib
import json
from pathlib import Path

import torch

from .checkpoints import load_head, model_from_checkpoint
from .fusion import adjusted, components
from .registry import PipelineName, validate_pipeline
from .scoring import canonical_dot, refine


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


class RetrievalPipeline:
    """Independently encode reactions/enzymes and score their normalized vectors.

    V4 uses the frozen no-dictionary recipe. F3 and CIRCEv2 use their base
    embeddings. Feature extraction is a separate, explicit preprocessing step.
    """

    def __init__(self, model, config, *, pipeline="v4", head=None,
                 fusion_multiplier=3.0, residual_cap=0.5, device="cpu"):
        self.name = validate_pipeline(config, pipeline)
        if self.name != PipelineName.V4 and head is not None:
            raise ValueError("Residual refinement belongs to V4")
        if not 0 <= residual_cap <= 1:
            raise ValueError("residual_cap must be in [0, 1]")
        self.model, self.config, self.head = model, config, head
        self.fusion_multiplier, self.residual_cap = fusion_multiplier, residual_cap
        self.device = str(device)

    @classmethod
    def from_manifest(cls, path, device="cpu"):
        """Load either a clean model manifest or the original frozen V4 protocol.

        Relative artifact paths are resolved against the manifest, never cwd.
        Declared artifact hashes are checked before loading any weights.
        """
        path = Path(path).resolve()
        protocol = json.loads(path.read_text())
        record = dict(protocol.get("model", protocol))
        recipe = protocol.get("recipe", {})
        if recipe.get("semantic_alpha", 0) != 0 or record.get("semantic_alpha", 0) != 0:
            raise ValueError("Supported V4 inference does not use dictionary scoring")
        for key in ("config", "checkpoint", "phase2_checkpoint"):
            if not record.get(key):
                continue
            artifact = Path(record[key])
            artifact = artifact if artifact.is_absolute() else path.parent / artifact
            if record.get(key + "_sha256") and sha256(artifact) != record[key + "_sha256"]:
                raise ValueError(f"Artifact hash mismatch: {artifact}")
            record[key] = artifact
        model, config = model_from_checkpoint(record["config"], record["checkpoint"], device)
        name = record.get("pipeline", "v4")
        head = None
        if record.get("phase2_checkpoint"):
            head = load_head(record["phase2_checkpoint"], record["feature_manifest_sha256"], device)
            feature_manifest = record.get("feature_manifest")
            if not feature_manifest and record.get("source_phase2"):
                feature_manifest = str(Path(record["source_phase2"]) / "features/manifest.json")
            if feature_manifest:
                feature_path = Path(feature_manifest)
                feature_path = feature_path if feature_path.is_absolute() else path.parent / feature_path
                if sha256(feature_path) != record["feature_manifest_sha256"]:
                    raise ValueError("Refinement feature manifest hash mismatch")
                source = json.loads(feature_path.read_text())
                training_base = source.get("checkpoint", {}).get("sha256")
            else:
                training_base = (head.training_registry.get("base_checkpoint") or {}).get("sha256")
            if training_base:
                from horizyn.inference_fusion_lineage import verify_refiner_base

                receipt = record.get("fusion_receipt", record.get("calibration_receipt"))
                if receipt and not Path(receipt).is_absolute():
                    receipt = path.parent / receipt
                verify_refiner_base(training_base, sha256(record["checkpoint"]), receipt,
                                    record["checkpoint"])
        multiplier = record.get("fusion_multiplier", recipe.get("inference_fusion_multiplier", 3.0))
        if record.get("checkpoint_already_calibrated", False):
            multiplier = 1.0
        return cls(model, config, pipeline=name, head=head, fusion_multiplier=multiplier,
                   residual_cap=recipe.get("residual_cap", record.get("residual_cap", 0.5)), device=device)

    @torch.inference_mode()
    def encode_enzymes(self, ids, batch_size=128):
        if self.name == PipelineName.V4:
            g, f, scale, base, _ = components(self.model, self.config, ids, self.device, batch_size)
            base = adjusted(g, f, scale, self.fusion_multiplier, base)
        else:
            from horizyn.benchmarks.retrieval import encode_residue_targets
            from horizyn.training_io import StoragePrecisionResidues

            residues = StoragePrecisionResidues(
                self.config.data.protein_residue_embeds_path,
                max_tokens=self.config.data.max_protein_tokens,
                truncation=self.config.data.protein_truncation,
            )
            try:
                base = encode_residue_targets(
                    self.model, residues, ids, self.device, batch_size, True
                ).float()
            finally:
                residues.close()
        return refine(base, self.head, "enzyme", self.residual_cap)

    @torch.inference_mode()
    def encode_reactions(self, inputs, ids, batch_size=128):
        from horizyn.benchmarks.retrieval import encode_reactions

        base = encode_reactions(self.model, inputs, ids, self.device, batch_size).float()
        return refine(base, self.head, "reaction", self.residual_cap)

    score = staticmethod(canonical_dot)
