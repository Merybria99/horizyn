"""Supported enzyme–reaction pipelines: V4 (default), F3, and CIRCEv2.

Training implementations retain historical module names for checkpoint
compatibility. New applications should import this API rather than scripts.
"""

from .registry import PIPELINES, PipelineName, validate_pipeline

__all__ = ["PIPELINES", "PipelineName", "validate_pipeline", "RetrievalPipeline"]


def __getattr__(name):
    if name == "RetrievalPipeline":
        from .inference import RetrievalPipeline

        return RetrievalPipeline
    raise AttributeError(name)
