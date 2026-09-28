"""Chiral Morgan participant counts as an independent reaction anchor view."""
from __future__ import annotations

from functools import lru_cache
import math
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from .semantic_smooth import reaction_responses


def canonical_components(value):
    """Retain every participant; accept only sets or identical self-reactions."""
    if not isinstance(value,str) or not value.strip():
        raise ValueError('Nonempty participant string required')
    sides=value.strip().split('>')
    if len(sides) not in (1,3) or (len(sides)==3 and sides[1]):
        raise ValueError('Expected participant collection or agent-free self-reaction')
    def parse(side):
        if not side or any(not item for item in side.split('.')):
            raise ValueError('Empty participant component')
        output=[]
        for item in side.split('.'):
            mol=Chem.MolFromSmiles(item,sanitize=True)
            if mol is None or mol.GetNumAtoms()==0:
                raise ValueError('Invalid molecular component: '+item)
            for atom in mol.GetAtoms():atom.SetAtomMapNum(0)
            output.append(Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True))
        return tuple(sorted(output))
    result=parse(sides[0])
    if len(sides)==3 and result!=parse(sides[2]):
        raise ValueError('Directional reactions must not be treated as self-reactions')
    return result


@lru_cache(maxsize=4)
def _generator(radius):
    if radius not in (2,3):raise ValueError('Fixed radii are 2 and 3')
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=4096,includeChirality=True)


@lru_cache(maxsize=100000)
def _component_unit(smiles,radius):
    mol=Chem.MolFromSmiles(smiles,sanitize=True)
    if mol is None or mol.GetNumAtoms()==0:raise ValueError('Invalid canonical component')
    count=_generator(radius).GetCountFingerprintAsNumPy(mol).astype(np.float64)
    norm=np.linalg.norm(count)
    if not np.isfinite(count).all() or norm<=0:raise ValueError('Invalid/zero Morgan counts')
    result=count/norm;result.flags.writeable=False
    return result


def participant_morgan(value,radius):
    """Component-unit sum in sorted canonical order, then unit FP32 output.

    Standard Morgan invariants may ignore isotope differences. Canonical inputs
    retain isotope labels; descriptor equality is never molecular identity.
    """
    components=canonical_components(value)
    result=np.zeros(4096,dtype=np.float64)
    for component in components:result+=_component_unit(component,radius)
    norm=np.linalg.norm(result)
    if not np.isfinite(norm) or norm<=0:raise ValueError('Invalid participant fingerprint')
    return (result/norm).astype(np.float32)


class MorganReactionAnchor(torch.nn.Module):
    """Only the reaction-side semantic map changes; enzyme index is unchanged."""
    def __init__(self,train_raw,train_morgan,eta):
        super().__init__()
        if eta not in (.5,1.):raise ValueError('Fixed Morgan weights are .5 and 1')
        self.eta=float(eta)
        self._check(train_raw,train_morgan)
        self.register_buffer('train_raw',train_raw)
        self.register_buffer('train_morgan',train_morgan)
        self.requires_grad_(False).eval()

    @staticmethod
    def _check(raw,morgan):
        if raw.ndim!=2 or morgan.ndim!=2 or len(raw)!=len(morgan) or morgan.shape[1]!=4096:
            raise ValueError('Aligned raw and 4096-dimensional Morgan matrices required')
        if raw.dtype!=torch.float32 or morgan.dtype!=torch.float32 or not torch.isfinite(raw).all() or not torch.isfinite(morgan).all():
            raise ValueError('Finite FP32 geometry required')
        for block in (raw,morgan):
            if not torch.allclose(torch.linalg.vector_norm(block.double(),dim=1),torch.ones(len(block),device=block.device,dtype=torch.float64),atol=2e-6,rtol=0):
                raise ValueError('Both reaction views must be unit-normalized independently')

    def view(self,raw,morgan):
        self._check(raw,morgan)
        return torch.cat((math.sqrt(1-self.eta)*raw,math.sqrt(self.eta)*morgan),1)

    @torch.inference_mode()
    def encode_reactions(self,raw,morgan):
        return reaction_responses(self.view(raw,morgan),self.view(self.train_raw,self.train_morgan),
                                  kernel='exponential',temperature=.03,neighbors=None)
