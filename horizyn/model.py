"""Historical model imports; implementations are organized under horizyn.models.

Keep this module path for old Lightning/pickled checkpoints and external code.
Maintained pipeline selection and inference live in horizyn.pipelines.
"""

from .models.common import (
    BaseModel,
    MLP,
    NormalizeLayer,
    NormalizedReactionBlockProjection,
    ResidualMLPProjection,
    _stage1_config,
    _torch_load_checkpoint,
    signed_power_transform
)

from .models.enzyme import (
    EnzymeBiologicalFactorizedEncoder,
    FunctionalResidueScorer,
    ProteinAttentionPooling,
    ProteinMeanPooling,
    SLEECFunctionalPool,
    SLEECGuidedAttentionPool
)

from .models.reaction import (
    MoleculeSetAttentionPooling,
    MoleculeSetInteractionPooling,
    MoleculeSetMeanPooling,
    MultimodalReactionAttentionEncoder
)

from .models.dual import (
    ProteinPooledDualModel
)

from .models.compat import (
    BlockwiseEnzymeFeatureFusion,
    DualContrastiveModel,
    E2RReactionAdapter,
    EnzymeBioFPSplitEncoder,
    GatedEnzymeFeatureFusion,
    HybridReactionEncoder,
    R2EEnzymeAdapter,
    ReactionConditionedAttentionPooling,
    ReactionConditionedDualModel,
    ReactionFingerprintAttentionPool,
    ResidualEnzymePrototypeHead,
    TigerTextGatedFusion,
    UniMol2ReactionAttentionEncoder
)

# Keep fully-qualified class/function names compatible with historical pickles.
for _name in ['BaseModel', 'BlockwiseEnzymeFeatureFusion', 'DualContrastiveModel', 'E2RReactionAdapter', 'EnzymeBioFPSplitEncoder', 'EnzymeBiologicalFactorizedEncoder', 'FunctionalResidueScorer', 'GatedEnzymeFeatureFusion', 'HybridReactionEncoder', 'MLP', 'MoleculeSetAttentionPooling', 'MoleculeSetInteractionPooling', 'MoleculeSetMeanPooling', 'MultimodalReactionAttentionEncoder', 'NormalizeLayer', 'NormalizedReactionBlockProjection', 'ProteinAttentionPooling', 'ProteinMeanPooling', 'ProteinPooledDualModel', 'R2EEnzymeAdapter', 'ReactionConditionedAttentionPooling', 'ReactionConditionedDualModel', 'ReactionFingerprintAttentionPool', 'ResidualEnzymePrototypeHead', 'ResidualMLPProjection', 'SLEECFunctionalPool', 'SLEECGuidedAttentionPool', 'TigerTextGatedFusion', 'UniMol2ReactionAttentionEncoder', '_stage1_config', '_torch_load_checkpoint', 'signed_power_transform']:
    globals()[_name].__module__ = __name__
del _name
