"""Enzyme–reaction retrieval: V4, F3, and CIRCEv2.

Derived from Horizyn (Dayhoff Labs); original license and attribution retained.
"""
__version__ = "1.0.0"
__author__ = "Dayhoff Labs"
__license__ = "PolyForm-Noncommercial-1.0.0"

from .config import DotDict, load_config, parse_overrides
from .pipelines import PIPELINES, PipelineName, validate_pipeline

__all__ = ["DotDict", "load_config", "parse_overrides", "PIPELINES",
           "PipelineName", "validate_pipeline", "RetrievalPipeline"]


def __getattr__(name):
    if name == "RetrievalPipeline":
        from .pipelines.inference import RetrievalPipeline
        return RetrievalPipeline
    raise AttributeError(name)
