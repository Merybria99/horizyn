#!/usr/bin/env python3
"""Input-only, single-site carboxylic ester hydrolysis for the literature panel.

Products are the conventional neutral hydrolysis products, not evidence that
each product was identified experimentally. No activity values are accepted.
"""
from collections import Counter
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


class IneligibleChemistry(ValueError):
    pass


def atom_inventory(mol):
    explicit = Chem.AddHs(mol)
    return Counter((a.GetAtomicNum(), a.GetIsotope()) for a in explicit.GetAtoms())


def single_ester_hydrolysis(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise IneligibleChemistry('invalid SMILES')
    if len(Chem.GetMolFrags(mol)) != 1:
        raise IneligibleChemistry('multiple substrate components')
    if any(a.HasQuery() or a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise IneligibleChemistry('nonfinite/query structure')
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    # Include carbonate/carbamate acyl-O sites in the ambiguity count. The
    # carbonyl carbon's remaining substituent must subsequently be C or H.
    broad = Chem.MolFromSmarts('[CX3](=[OX1])-[OX2]-[#6]')
    matches = mol.GetSubstructMatches(broad, uniquify=True)
    sites = sorted({(match[0], match[2]) for match in matches})
    if len(sites) != 1:
        raise IneligibleChemistry(f'expected one acyl-O ester site; found {len(sites)}')
    carbon, oxygen = sites[0]
    ca = mol.GetAtomWithIdx(carbon)
    remaining = [a for a in ca.GetNeighbors() if a.GetIdx() != oxygen and
                 mol.GetBondBetweenAtoms(carbon, a.GetIdx()).GetBondType() != Chem.BondType.DOUBLE]
    if any(a.GetAtomicNum() not in (1, 6) for a in remaining):
        raise IneligibleChemistry('carbonate/carbamate or other noncarboxylic acyl group')
    if ca.GetFormalCharge() or mol.GetAtomWithIdx(oxygen).GetFormalCharge():
        raise IneligibleChemistry('charged ester reaction center')
    original_atoms = [(a.GetAtomicNum(), a.GetIsotope(), a.GetFormalCharge(), a.GetChiralTag())
                      for a in mol.GetAtoms()]
    original_bonds = {(b.GetBeginAtomIdx(), b.GetEndAtomIdx()):
                      (b.GetBondType(), b.GetStereo(), b.GetBondDir()) for b in mol.GetBonds()
                      if {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} != {carbon, oxygen}}
    edited = Chem.RWMol(mol)
    edited.RemoveBond(carbon, oxygen)
    new_oxygen = edited.AddAtom(Chem.Atom(8))
    edited.AddBond(carbon, new_oxygen, Chem.BondType.SINGLE)
    edited.GetAtomWithIdx(oxygen).SetNoImplicit(False)
    product = edited.GetMol()
    Chem.SanitizeMol(product)
    # Existing reaction-center atoms are achiral; every pre-existing stereo
    # atom and every uncut bond retain their original structural attributes.
    assert original_atoms == [(a.GetAtomicNum(), a.GetIsotope(), a.GetFormalCharge(), a.GetChiralTag())
                              for a in list(product.GetAtoms())[:-1]]
    for (left, right), attributes in original_bonds.items():
        bond = product.GetBondBetweenAtoms(left, right)
        assert (bond.GetBondType(), bond.GetStereo(), bond.GetBondDir()) == attributes
    water = Chem.MolFromSmiles('O')
    assert atom_inventory(product) == atom_inventory(mol) + atom_inventory(water)
    assert Chem.GetFormalCharge(product) == Chem.GetFormalCharge(mol)
    substrate = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    products = sorted(Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
                      for fragment in Chem.GetMolFrags(product, asMols=True, sanitizeFrags=True))
    canonical_product = Chem.MolFromSmiles('.'.join(products))
    assert atom_inventory(canonical_product) == atom_inventory(product)
    return dict(substrate_smiles=substrate, water_smiles='O', product_smiles='.'.join(products),
                reaction_smiles=substrate + '.O>>' + '.'.join(products),
                substrate_formula=rdMolDescriptors.CalcMolFormula(mol),
                product_formula=rdMolDescriptors.CalcMolFormula(product),
                ring_opening=len(products) == 1, ester_carbon_index=carbon,
                ester_oxygen_index=oxygen, atom_and_charge_balance=True,
                product_evidence='Conventional single-ester hydrolysis mapping; assay measures acid release.')
