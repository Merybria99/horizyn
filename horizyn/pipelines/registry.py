"""One registry for the three maintained pipelines."""

from enum import Enum


class PipelineName(str, Enum):
    V4 = "v4"
    F3 = "f3"
    CIRCEV2 = "circev2"


PIPELINES = {
    PipelineName.V4: "Learned residue views, anchor-balanced alignment, residual refinement",
    PipelineName.F3: "SLEEC-guided residue pooling and multimodal reaction encoding",
    PipelineName.CIRCEV2: "ProtT5/SLEEC encoding with annotation-derived training negatives",
}


def validate_pipeline(config, pipeline="v4"):
    """Reject unrelated architectures at maintained training/inference entry points.

    Legacy F3 and CIRCEv2 configs use the biological-factorized encoder mode;
    those historical implementations are deliberately retained unchanged.
    """
    name = PipelineName(pipeline)
    if config.model.get("name") != "ProteinPooledDualModel":
        raise ValueError("Supported pipelines require model.name=ProteinPooledDualModel")
    if config.model.get("query_encoder_type") != "multimodal_reaction_attention":
        raise ValueError("Supported pipelines require the multimodal reaction encoder")
    mode = config.model.get("enzyme_input_mode", "standard")
    allowed = {
        PipelineName.V4: {"raw_mean_sleec_multiview"},
        PipelineName.F3: {"standard", "raw_mean_sleec_biological_factorized"},
        PipelineName.CIRCEV2: {"raw_mean_sleec_biological_factorized"},
    }
    if mode not in allowed[name]:
        raise ValueError(f"{name.value} does not support enzyme_input_mode={mode!r}")
    for component in ("e2r_adapter", "r2e_adapter", "biological_residual"):
        if config.model.get(component, {}).get("enabled", False):
            raise ValueError(f"{component} is not part of the supported pipelines")
    if config.model.get("enzyme_prototypes", {}).get("count", 1) != 1:
        raise ValueError("Enzyme prototype scoring is not a supported pipeline")
    return name
