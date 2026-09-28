"""Reaction representation shared by feature extraction and inference."""
from horizyn.capability.reaction_set_features import molecule_set_components
from horizyn.chemistry.reaction_recovery import canonical_molecule

def reaction_smiles_for_model(
    reaction_smiles: str, *, normalize_molecule_sets_as_self_reactions: bool
) -> str:
    """Apply the training representation before every reaction extractor.

    A participant model reads one molecular bank, so it must receive products
    there as well as reactants. An existing S>>S representation contributes S
    once, preserving stoichiometric multiplicities and making this idempotent.
    """
    if not normalize_molecule_sets_as_self_reactions:
        return reaction_smiles
    participants = sorted(canonical_molecule(value) for value in molecule_set_components(reaction_smiles))
    if not participants:
        raise ValueError("The participant collection is empty")
    side = ".".join(participants)
    return f"{side}>>{side}"
