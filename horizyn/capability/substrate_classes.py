"""Rule-based substrate and product class extraction."""

from __future__ import annotations

from collections.abc import Iterable

from rdkit import Chem


SMARTS: dict[str, str] = {
    "carboxylate": "C(=O)[O-,OH]",
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "amide": "C(=O)N",
    "ester": "[#6][CX3](=O)[OX2][#6]",
    "ether": "[OD2]([#6])[#6]",
    "phosphate_containing": "P(=O)(O)(O)",
    "phosphate_ester": "[PX4](=O)([OX2H,OX1-])([OX2H,OX1-])[OX2][#6]",
    "phosphonate": "[PX4](=O)([OX2H,OX1-])([OX2H,OX1-])[#6]",
    "pyrophosphate": "[PX4](=O)([OX2H,OX1-])O[PX4](=O)([OX2H,OX1-])[OX2H,OX1-]",
    "acyl_phosphate": "[CX3](=O)O[PX4](=O)",
    "sulfate_containing": "S(=O)(=O)(O)",
    "sulfonate": "[SX4](=O)(=O)[OX2H,OX1-]",
    "sulfoxide": "[SX3](=O)([#6])[#6]",
    "sulfone": "[SX4](=O)(=O)([#6])[#6]",
    "thiol": "[SH]",
    "thioether": "[SX2]([#6])[#6]",
    "disulfide": "[SX2][SX2]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "imine": "[CX3]=[NX2]",
    "nitrile": "[CX2]#N",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "aromatic": "a",
    "phenol": "c[OH]",
    "coa_thioester": "C(=O)S",
    "alcohol": "[CX4][OX2H]",
    "alkene": "C=C",
    "alkyne": "C#C",
    "organohalide": "[#6][F,Cl,Br,I]",
    "epoxide": "[OX2r3]1[#6r3][#6r3]1",
    "lactone": "[CX3;R](=O)[OX2;R][#6]",
    "lactam": "[CX3;R](=O)[NX3;R][#6]",
    "carbonate": "[OX2][CX3](=O)[OX2]",
    "carbamate": "[NX3][CX3](=O)[OX2]",
    "urea": "[NX3][CX3](=O)[NX3]",
    "guanidine": "[NX3][CX3](=[NX2,NX3+])[NX3]",
    "acetal": "[CX4]([OX2][#6])([OX2][#6])",
    "purine_like": "n1cnc2ncnc12",
    "pyrimidine_like": "n1ccnc(=O)[nH]1",
}

_SMARTS_MOLS = {label: Chem.MolFromSmarts(pattern) for label, pattern in SMARTS.items()}
_COFACTOR_SMILES_LABEL_CACHE: dict[tuple[str, int, int], tuple[str, ...]] = {}

CURRENCY_SMILES = {
    "O",
    "[H+]",
    "[OH-]",
    "O=P(O)(O)O",
    "O=P([O-])([O-])[O-]",
    "O=C=O",
    "N",
    "O=O",
    "[HH]",
}


def split_reaction_sides(reaction_smiles: str) -> tuple[list[str], list[str]]:
    parts = str(reaction_smiles).split(">")
    if len(parts) == 2:
        left, right = parts
        middle = ""
    elif len(parts) == 3:
        left, middle, right = parts
    else:
        return [], []
    reactants = [value for side in (left, middle) for value in side.split(".") if value]
    products = [value for value in right.split(".") if value]
    return reactants, products


def _is_currency_mol(mol: Chem.Mol) -> bool:
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if smiles in CURRENCY_SMILES:
        return True
    heavy_atoms = mol.GetNumHeavyAtoms()
    return heavy_atoms <= 1


def classify_molecule(mol: Chem.Mol) -> list[str]:
    labels = {
        label
        for label, pattern in _SMARTS_MOLS.items()
        if pattern is not None and mol.HasSubstructMatch(pattern)
    }
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    hetero_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6})
    hydroxyl_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[OX2H]")))
    amide_count = len(mol.GetSubstructMatches(_SMARTS_MOLS["amide"])) if _SMARTS_MOLS["amide"] else 0
    ring_info = mol.GetRingInfo()
    ring_count = ring_info.NumRings() if ring_info is not None else 0
    if {"amine", "carboxylate"}.issubset(labels):
        labels.add("amino_acid_like")
    if {"ketone", "carboxylate"}.issubset(labels):
        labels.add("keto_acid_like")
    if {"alcohol", "carboxylate"}.issubset(labels):
        labels.add("hydroxy_acid_like")
    if labels & {"purine_like", "pyrimidine_like"}:
        labels.add("nucleobase_like")
    if hydroxyl_count >= 3 and carbon_count >= 4:
        labels.add("sugar_like")
    if labels & {"purine_like", "pyrimidine_like"}:
        labels.add("nucleobase_like")
    if (
        labels & {"purine_like", "pyrimidine_like", "heteroaromatic"}
        and "sugar_like" in labels
        and sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7) >= 2
    ):
        labels.add("nucleoside_like")
    if "nucleoside_like" in labels and "phosphate_containing" in labels:
        labels.add("nucleotide_like")
    if amide_count >= 2 or ("amino_acid_like" in labels and "amide" in labels):
        labels.add("peptide_like")
    if labels & {"carboxylate"}:
        labels.add("organic_acid")
    if carbon_count >= 10 and hetero_count <= max(2, carbon_count // 8):
        labels.add("large_hydrophobic")
    if carbon_count >= 8 and "carboxylate" in labels:
        labels.add("fatty_acid_like")
    if carbon_count >= 10 and (labels & {"ester", "coa_thioester", "fatty_acid_like"}):
        labels.add("lipid_like")
    if carbon_count >= 10 and "alkene" in labels and hetero_count <= 2:
        labels.add("terpene_like")
    if carbon_count >= 17 and ring_count >= 4 and hetero_count <= 4:
        labels.add("steroid_like")
    if carbon_count <= 4 and "aromatic" not in labels:
        labels.add("small_aliphatic")
    if "aromatic" in labels and any(atom.GetIsAromatic() and atom.GetAtomicNum() != 6 for atom in mol.GetAtoms()):
        labels.add("heteroaromatic")
    if (
        labels & {"purine_like", "pyrimidine_like", "heteroaromatic"}
        and "sugar_like" in labels
        and sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7) >= 2
    ):
        labels.add("nucleoside_like")
    if "nucleoside_like" in labels and "phosphate_containing" in labels:
        labels.add("nucleotide_like")
    if "organohalide" in labels:
        labels.add("halogenated")
    if "acetal" in labels and "sugar_like" in labels:
        labels.add("glycoside")
    return sorted(labels)


def _is_skipped_cofactor_molecule(
    smiles: str,
    *,
    cofactor_aliases: dict[str, list[str]] | None,
    skip_cofactor_labels: set[str],
) -> bool:
    if not skip_cofactor_labels:
        return False
    from horizyn.capability.cofactors import extract_cofactor_labels_from_smiles

    alias_count = sum(len(values) for values in cofactor_aliases.values()) if cofactor_aliases else 0
    cache_key = (str(smiles), id(cofactor_aliases), alias_count)
    cached = _COFACTOR_SMILES_LABEL_CACHE.get(cache_key)
    if cached is None:
        cached = tuple(
            extract_cofactor_labels_from_smiles(
                str(smiles),
                cofactor_aliases=cofactor_aliases,
            )
        )
        _COFACTOR_SMILES_LABEL_CACHE[cache_key] = cached
    labels = list(cached)
    return bool(set(labels) & skip_cofactor_labels)


def classify_smiles_list(
    smiles_values: Iterable[str],
    *,
    cofactor_aliases: dict[str, list[str]] | None = None,
    skip_cofactor_labels: Iterable[str] | None = None,
) -> tuple[list[str], list[str]]:
    labels: set[str] = set()
    flags: set[str] = set()
    skip_labels = {str(label) for label in skip_cofactor_labels or []}
    for smiles in smiles_values:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            flags.add("substrate_class_invalid_molecule")
            continue
        if _is_currency_mol(mol):
            continue
        if _is_skipped_cofactor_molecule(
            str(smiles),
            cofactor_aliases=cofactor_aliases,
            skip_cofactor_labels=skip_labels,
        ):
            flags.add("substrate_class_removed_cofactor_molecule")
            continue
        molecule_labels = classify_molecule(mol)
        if molecule_labels:
            flags.add("substrate_class_from_smarts")
        labels.update(molecule_labels)
    return sorted(labels), sorted(flags)


def classify_reaction_substrates_products(
    reaction_smiles: str,
    *,
    cofactor_aliases: dict[str, list[str]] | None = None,
    skip_cofactor_labels: Iterable[str] | None = None,
) -> tuple[list[str], list[str], list[str], str]:
    reactants, products = split_reaction_sides(reaction_smiles)
    substrate_labels, substrate_flags = classify_smiles_list(
        reactants,
        cofactor_aliases=cofactor_aliases,
        skip_cofactor_labels=skip_cofactor_labels,
    )
    product_labels, product_flags = classify_smiles_list(
        products,
        cofactor_aliases=cofactor_aliases,
        skip_cofactor_labels=skip_cofactor_labels,
    )
    flags = sorted(set(substrate_flags) | set(product_flags))
    if substrate_labels or product_labels:
        status = "weak_smarts" if flags else "ok"
    else:
        status = "unknown"
    return substrate_labels, product_labels, flags, status


TRANSITION_RULES: tuple[tuple[str, str, str], ...] = (
    ("alcohol", "ketone", "alcohol_to_ketone"),
    ("alcohol", "aldehyde", "alcohol_to_aldehyde"),
    ("aldehyde", "alcohol", "aldehyde_to_alcohol"),
    ("aldehyde", "carboxylate", "aldehyde_to_carboxylate"),
    ("ketone", "alcohol", "ketone_to_alcohol"),
    ("ketone", "imine", "ketone_to_imine"),
    ("imine", "amine", "imine_to_amine"),
    ("amine", "imine", "amine_to_imine"),
    ("nitrile", "amide", "nitrile_to_amide"),
    ("nitrile", "carboxylate", "nitrile_to_carboxylate"),
    ("hydroxy_acid_like", "keto_acid_like", "hydroxy_acid_to_keto_acid"),
    ("keto_acid_like", "hydroxy_acid_like", "keto_acid_to_hydroxy_acid"),
    ("amino_acid_like", "keto_acid_like", "amino_acid_to_keto_acid"),
    ("keto_acid_like", "amino_acid_like", "keto_acid_to_amino_acid"),
    ("coa_thioester", "carboxylate", "coa_thioester_to_carboxylate"),
    ("carboxylate", "coa_thioester", "carboxylate_to_coa_thioester"),
    ("ester", "carboxylate", "ester_to_carboxylate"),
    ("carboxylate", "ester", "carboxylate_to_ester"),
    ("amide", "carboxylate", "amide_to_carboxylate"),
    ("carboxylate", "amide", "carboxylate_to_amide"),
    ("disulfide", "thiol", "disulfide_to_thiol"),
    ("thiol", "disulfide", "thiol_to_disulfide"),
    ("alcohol", "phosphate_ester", "alcohol_to_phosphate_ester"),
    ("phosphate_ester", "alcohol", "phosphate_ester_to_alcohol"),
)


def substrate_product_transition_labels(
    substrate_labels: Iterable[str],
    product_labels: Iterable[str],
) -> list[str]:
    """Create directional broad-chemotype transitions from substrate to product."""

    substrates = {str(label) for label in substrate_labels if str(label)}
    products = {str(label) for label in product_labels if str(label)}
    labels: set[str] = set()
    for left, right, transition in TRANSITION_RULES:
        if left in substrates and right in products:
            labels.add(transition)
    if "phosphate_containing" in substrates and "phosphate_containing" not in products:
        labels.add("loss_phosphate_containing")
        labels.add("dephosphorylation_like")
    if "phosphate_containing" in products and "phosphate_containing" not in substrates:
        labels.add("gain_phosphate_containing")
        labels.add("phosphorylation_like")
    for label in sorted((substrates | products) - {"small_aliphatic"}):
        if label in substrates and label not in products:
            labels.add(f"loss_{label}")
        elif label in products and label not in substrates:
            labels.add(f"gain_{label}")
    return sorted(labels)
