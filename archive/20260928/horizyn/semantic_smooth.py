"""Smooth training-reaction responses for independent endpoint retrieval.

These are exploratory additions; the phase-one frozen encoder is unchanged.
"""
from __future__ import annotations

import torch

from .semantic_anchors import row_unit, stable_topk


def stable_neighbor_order(values, indices):
    """Sort by descending value, then anchor index, permitting nested prefixes."""
    order = torch.argsort(indices, dim=1, stable=True)
    values, indices = values.gather(1, order), indices.gather(1, order)
    order = torch.argsort(values, dim=1, descending=True, stable=True)
    return values.gather(1, order), indices.gather(1, order)


def reaction_responses(encoded, anchors, kernel="exponential", temperature=.1, neighbors=None):
    """Return unit vectors indexed by a fixed training-reaction dictionary.

    Full exponential and shifted-cosine responses have full support (strictly
    positive after a numerical floor). Centered cosine is a signed smooth map.
    Truncated exponential variants are diagnostic approximations only.
    """
    if temperature <= 0 or (neighbors is not None and neighbors <= 0):
        raise ValueError("Positive temperature and neighbor count required")
    similarity = (encoded.double() @ anchors.double().T).float()
    if kernel == "exponential":
        if neighbors is None:
            response = torch.exp((similarity - similarity.amax(1, keepdim=True)) / temperature)
        else:
            values, indices = stable_topk(similarity, neighbors)
            weights = torch.exp((values - values[:, :1]) / temperature)
            response = torch.zeros_like(similarity).scatter_(1, indices, weights)
    elif kernel == "shifted_cosine":
        if neighbors is not None:
            raise ValueError("Shifted cosine requires full support")
        response = ((similarity + 1) / 2).clamp_min(torch.finfo(similarity.dtype).eps)
    elif kernel == "centered_cosine":
        if neighbors is not None:
            raise ValueError("Centered cosine requires full support")
        # Double centering against TRAINING anchors only. No inference-pool fit.
        center = anchors.double().mean(0)
        column_mean = (center @ anchors.double().T).float()
        query_mean = (encoded.double() @ center).float()
        response = similarity - query_mean[:, None] - column_mean[None, :] + column_mean.mean()
    else:
        raise ValueError(f"Unknown reaction response kernel: {kernel}")
    if kernel == "exponential" and neighbors is None:
        response = response.clamp_min(torch.finfo(response.dtype).tiny)
    return row_unit(response)


class SmoothAnchorDualEncoder(torch.nn.Module):
    """Native frozen F3 plus smooth training-dictionary endpoint features."""

    def __init__(self, dictionary, train_reactions, config):
        super().__init__()
        if not 0 <= config['alpha'] <= 1:
            raise ValueError('Mixture alpha must lie in [0, 1]')
        self.config = dict(config)
        self.modalities = list(dictionary['modalities'])
        for key in ('protein_center', 'train_proteins', 'adjacency'):
            self.register_buffer(key, dictionary[key])
        self.register_buffer('train_reactions', train_reactions)
        for key in self.modalities:
            self.register_buffer('reaction_center_' + key, dictionary['reaction_centers'][key])
        self.requires_grad_(False).eval()

    def _compose(self, base, semantic):
        import math
        alpha = self.config['alpha']
        return torch.cat((math.sqrt(1-alpha)*torch.nn.functional.normalize(base,dim=-1),
                          math.sqrt(alpha)*semantic),dim=1)

    @torch.inference_mode()
    def encode_semantic_enzymes(self, protein_mean, batch_size=512):
        from .semantic_anchors import centered_unit, nearest_training_proteins, enzyme_anchor_features
        encoded = centered_unit(protein_mean, self.protein_center)
        values, indices = nearest_training_proteins(encoded, self.train_proteins,
            self.config['protein_neighbors'], batch_size)
        return enzyme_anchor_features(values, indices, self.adjacency,
            len(self.train_reactions), self.config['enzyme_temperature'])

    @torch.inference_mode()
    def encode_semantic_reactions(self, blocks, masks):
        from .semantic_anchors import reaction_features
        centers = {key:getattr(self,'reaction_center_'+key) for key in self.modalities}
        raw = reaction_features(blocks,centers,masks,self.modalities)
        return reaction_responses(raw,self.train_reactions,self.config['kernel'],
            self.config['reaction_temperature'],self.config['reaction_neighbors'])

    @torch.inference_mode()
    def encode_enzymes(self, base, protein_mean, batch_size=512):
        if len(base) != len(protein_mean):
            raise ValueError('Protein means and base rows disagree')
        return self._compose(base,self.encode_semantic_enzymes(protein_mean,batch_size))

    @torch.inference_mode()
    def encode_reactions(self, base, blocks, masks):
        semantic = self.encode_semantic_reactions(blocks,masks)
        if len(base) != len(semantic):
            raise ValueError('Reaction raw and base rows disagree')
        return self._compose(base,semantic)

    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        from .generalization_retrieval import checked_artifact
        state = torch.load(path,map_location=device,weights_only=False)
        dictionary_path = checked_artifact(dict(path=state['dictionary_path'],sha256=state['dictionary_sha256']))
        dictionary = torch.load(dictionary_path,map_location=device,weights_only=False)
        if dictionary['feature_manifest_sha256'] != state['feature_manifest_sha256']:
            raise ValueError('Smooth state and dictionary feature manifests disagree')
        return cls(dictionary,state['train_reactions'],state['config']).to(device), state
