import sys
from pathlib import Path
import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from generalization_esterase_chemistry import single_ester_hydrolysis, IneligibleChemistry


@pytest.mark.parametrize('substrate,expected', [
    ('CC(=O)OC', 'CC(=O)O.CO'),
    ('COC=O', 'CO.O=CO'),
    ('O=C1OCCC1', 'O=C(O)CCCO'),
])
def test_chemically_known_products(substrate, expected):
    result = single_ester_hydrolysis(substrate)
    assert Chem.MolToSmiles(Chem.MolFromSmiles(result['product_smiles'])) == Chem.MolToSmiles(Chem.MolFromSmiles(expected))
    assert result['atom_and_charge_balance']


@pytest.mark.parametrize('substrate', ['COC(=O)CC(=O)OC', 'COC(=O)OC', 'COC(=O)N', 'CCO', 'CC(=O)OC.O'])
def test_ambiguous_or_unsupported_inputs_rejected(substrate):
    with pytest.raises(IneligibleChemistry):
        single_ester_hydrolysis(substrate)


def test_chiral_and_isotopic_centers_survive():
    result = single_ester_hydrolysis('COC(=O)[C@H](O)[13CH3]')
    product = Chem.MolFromSmiles(result['product_smiles'])
    assert any(atom.GetIsotope() == 13 for atom in product.GetAtoms())
    assert [label for _, label in Chem.FindMolChiralCenters(product)] == ['S']
