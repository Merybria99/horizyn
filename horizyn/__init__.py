"""
Horizyn: Contrastive Learning for Enzyme-Reaction Matching

Official implementation of the Horizyn SOTA model for matching enzymatic
reactions with their catalyzing proteins using contrastive learning.

License: PolyForm Noncommercial License 1.0.0
Copyright (c) 2025 Dayhoff Labs
"""

__version__ = "0.1.0"
__author__ = "Dayhoff Labs"
__license__ = "PolyForm-Noncommercial-1.0.0"

from horizyn.config import DotDict, load_config, parse_overrides


_LAZY_EXPORTS = {
    "HorizynDataModule": ("horizyn.data_module", "HorizynDataModule"),
    "HorizynLitModule": ("horizyn.lightning_module", "HorizynLitModule"),
    "DualContrastiveModel": ("horizyn.model", "DualContrastiveModel"),
    "FunctionalResidueScorer": ("horizyn.model", "FunctionalResidueScorer"),
    "HybridReactionEncoder": ("horizyn.model", "HybridReactionEncoder"),
    "MLP": ("horizyn.model", "MLP"),
    "MoleculeSetMeanPooling": ("horizyn.model", "MoleculeSetMeanPooling"),
    "MultimodalReactionAttentionEncoder": (
        "horizyn.model",
        "MultimodalReactionAttentionEncoder",
    ),
    "ProteinAttentionPooling": ("horizyn.model", "ProteinAttentionPooling"),
    "ProteinMeanPooling": ("horizyn.model", "ProteinMeanPooling"),
    "ProteinPooledDualModel": ("horizyn.model", "ProteinPooledDualModel"),
    "ReactionFingerprintAttentionPool": (
        "horizyn.model",
        "ReactionFingerprintAttentionPool",
    ),
    "SLEECFunctionalPool": ("horizyn.model", "SLEECFunctionalPool"),
    "ProteinPooledLitModule": (
        "horizyn.protein_pooling_lightning_module",
        "ProteinPooledLitModule",
    ),
    "SLEECStage1Classifier": ("horizyn.sleec_stage1", "SLEECStage1Classifier"),
    "FullBatchNCELoss": ("horizyn.losses", "FullBatchNCELoss"),
    "FullBatchMLNCELoss": ("horizyn.losses", "FullBatchMLNCELoss"),
    "BidirectionalAnchorBalancedSupConLoss": (
        "horizyn.losses",
        "BidirectionalAnchorBalancedSupConLoss",
    ),
    "HorizynFGWLoss": ("horizyn.losses", "HorizynFGWLoss"),
    "MultiAlignmentRetrievalLoss": ("horizyn.losses", "MultiAlignmentRetrievalLoss"),
    "build_horizyn_loss": ("horizyn.losses", "build_horizyn_loss"),
    "create_retrieval_metrics": ("horizyn.metrics", "create_retrieval_metrics"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'horizyn' has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

# Package exports
__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "DotDict",
    "load_config",
    "parse_overrides",
    "HorizynDataModule",
    "HorizynLitModule",
    "DualContrastiveModel",
    "FunctionalResidueScorer",
    "HybridReactionEncoder",
    "MLP",
    "MoleculeSetMeanPooling",
    "MultimodalReactionAttentionEncoder",
    "ProteinAttentionPooling",
    "ProteinMeanPooling",
    "ProteinPooledDualModel",
    "ReactionFingerprintAttentionPool",
    "SLEECFunctionalPool",
    "ProteinPooledLitModule",
    "SLEECStage1Classifier",
    "FullBatchNCELoss",
    "FullBatchMLNCELoss",
    "BidirectionalAnchorBalancedSupConLoss",
    "HorizynFGWLoss",
    "MultiAlignmentRetrievalLoss",
    "build_horizyn_loss",
    "create_retrieval_metrics",
]
