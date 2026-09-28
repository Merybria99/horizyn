"""B1/B2: frozen residue features, molecular Graphormer, and shared token banks.

This module deliberately has no dependency on the CIRCE fusion architecture.
Both retrieval directions use the same score. Molecular components have no
positional or artificial substrate/product labels.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint


@dataclass
class TokenModelConfig:
    residue_dim: int = 1152
    width: int = 256
    heads: int = 4
    graph_layers: int = 4
    latent_layers: int = 2
    ffn_dim: int = 1024
    protein_tokens: int = 32
    reaction_tokens: int = 16
    local_dim: int = 128
    global_dim: int = 256
    dropout: float = 0.1
    local_temperature: float = 0.1
    protein_chunk: int = 32
    molecule_chunk: int = 32
    reaction_chunk: int = 16
    checkpoint_encoders: bool = True
    # Legacy default keeps existing scratch checkpoints loadable. New campaign
    # configs explicitly select the pinned pretrained molecular encoder.
    graph_backbone: str = "venus_scratch"
    graph_checkpoint_dir: str | None = None


class TokenPool(nn.Module):
    def __init__(self, input_dim: int, count: int, cfg: TokenModelConfig):
        super().__init__()
        self.input_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, cfg.width))
        self.queries = nn.Parameter(torch.randn(count, cfg.width) * 0.02)
        self.cross_attention = nn.MultiheadAttention(cfg.width, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(cfg.width)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(cfg.width, cfg.heads, cfg.ffn_dim, cfg.dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(cfg.latent_layers)
        ])
        self.local_projection = nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, cfg.local_dim))
        self.global_projection = nn.Linear(cfg.local_dim, cfg.global_dim)

    def forward(self, values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        if values.shape[:2] != valid.shape or not bool(valid.any(dim=1).all()):
            raise ValueError("Every token bank needs at least one valid input")
        values = self.input_projection(values)
        queries = self.queries.unsqueeze(0).expand(values.shape[0], -1, -1)
        attended, _ = self.cross_attention(queries, values, values, key_padding_mask=~valid, need_weights=False)
        tokens = self.cross_norm(queries + attended)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.local_projection(tokens)
        pooled = F.normalize(self.global_projection(tokens.mean(dim=1)), dim=-1)
        return pooled, F.normalize(tokens, dim=-1)


def local_pair_scores(reaction: Tensor, enzyme: Tensor, temperature: float = 0.1) -> Tensor:
    """Corresponding pairs [N, K, D], with full, fixed-size token banks."""
    similarities = torch.bmm(reaction, enzyme.transpose(1, 2))
    return (temperature * (torch.logsumexp(similarities / temperature, dim=-1)
                           - math.log(enzyme.shape[1]))).mean(dim=-1)


class TokenRetrievalModel(nn.Module):
    def __init__(self, cfg: TokenModelConfig, variant: str):
        super().__init__()
        if variant not in {"B1", "B2"}:
            raise ValueError("variant must be B1 or B2")
        self.cfg, self.variant = cfg, variant
        self.protein_pool = TokenPool(cfg.residue_dim, cfg.protein_tokens, cfg)
        if cfg.graph_backbone == "pcqm4mv2_pretrained":
            from horizyn.pretrained_graphormer import PretrainedMolecularGraphormer
            self.graph_encoder = PretrainedMolecularGraphormer(cfg.graph_checkpoint_dir)
            graph_width = self.graph_encoder.output_dim
        elif cfg.graph_backbone == "venus_scratch":
            from rxnzyme.data.reaction import x_dim, e_dim
            from rxnzyme.models.modules.graphormer_graph_encoder import GraphormerGraphEncoder
            self.graph_encoder = GraphormerGraphEncoder(
                num_spatial=24, num_edge_dis=20, edge_type="single_hop",
                feature_dim=x_dim, edge_attr_dim=e_dim,
                num_encoder_layers=cfg.graph_layers, embedding_dim=cfg.width,
                ffn_embedding_dim=cfg.ffn_dim, num_attention_heads=cfg.heads,
                dropout=cfg.dropout, attention_dropout=cfg.dropout,
                activation_dropout=cfg.dropout, pooling="mean",
            )
            graph_width = cfg.width
        else:
            raise ValueError(f"Unknown molecular backbone: {cfg.graph_backbone}")
        self.reaction_pool = TokenPool(graph_width, cfg.reaction_tokens, cfg)

    @property
    def device(self):
        return self.protein_pool.queries.device

    def _pool_ragged(self, values: Sequence[Tensor], pool: TokenPool, chunk_size: int):
        outputs = []
        for start in range(0, len(values), chunk_size):
            part = values[start:start + chunk_size]
            padded = pad_sequence(part, batch_first=True).to(self.device, dtype=torch.float32)
            lengths = torch.tensor([len(x) for x in part], device=self.device)
            valid = torch.arange(padded.shape[1], device=self.device)[None, :] < lengths[:, None]
            if self.training and self.cfg.checkpoint_encoders:
                outputs.append(checkpoint(pool, padded, valid, use_reentrant=False))
            else:
                outputs.append(pool(padded, valid))
        return tuple(torch.cat([part[i] for part in outputs]) for i in range(2))

    def encode_proteins(self, residues: Sequence[Tensor]):
        return self._pool_ragged(residues, self.protein_pool, self.cfg.protein_chunk)

    def encode_reactions(self, reactions: Sequence[Sequence[object]]):
        # Deduplicate molecules in this call, then restore multiplicity when
        # assembling each reaction. No component is discarded or truncated.
        molecules, indices, reaction_indices = [], {}, []
        for reaction in reactions:
            rows = []
            for graph in reaction:
                key = graph.canonical_smiles
                if key not in indices:
                    indices[key] = len(molecules)
                    molecules.append(graph)
                rows.append(indices[key])
            if not rows:
                raise ValueError("A reaction must contain a molecular component")
            reaction_indices.append(rows)
        # Sort by size only for padding efficiency; retain original identities.
        order = sorted(range(len(molecules)), key=lambda i: molecules[i].num_nodes)
        encoded = [None] * len(molecules)
        for start in range(0, len(order), self.cfg.molecule_chunk):
            ids = order[start:start + self.cfg.molecule_chunk]
            if self.cfg.graph_backbone == "pcqm4mv2_pretrained":
                from horizyn.pretrained_graphormer import collate_pretrained_graphs
                batch = collate_pretrained_graphs([molecules[i] for i in ids], self.graph_encoder.config)
            else:
                from rxnzyme.data.graph_utils.collator import collate
                # VenusRXN's collator modifies spatial_pos, so give it copies.
                batch = collate([molecules[i].clone() for i in ids])
            batch = {k: v.to(self.device) for k, v in batch.items()}
            graph_trainable = any(p.requires_grad for p in self.graph_encoder.parameters())
            if self.training and self.cfg.checkpoint_encoders and graph_trainable:
                atoms, _ = checkpoint(self.graph_encoder, batch, use_reentrant=False)
            else:
                atoms, _ = self.graph_encoder(batch)
            for j, i in enumerate(ids):
                encoded[i] = atoms[j, :molecules[i].num_nodes]
        participants = [torch.cat([encoded[i] for i in ids]) for ids in reaction_indices]
        return self._pool_ragged(participants, self.reaction_pool, self.cfg.reaction_chunk)

    def forward(self, residues, reactions):
        return self.encode_reactions(reactions), self.encode_proteins(residues)

    def score_pairs(self, reaction, enzyme, pair_indices: Tensor | None = None,
                    query_chunk: int = 16, target_chunk: int = 256):
        """Exact scores, either for specified pairs or a complete candidate grid."""
        rg, rt = reaction
        eg, et = enzyme
        if pair_indices is not None:
            chunks = []
            for pair in pair_indices.split(1024):
                q, e = pair.unbind(dim=1)
                scores = (rg[q] * eg[e]).sum(dim=-1)
                if self.variant == "B2":
                    local = local_pair_scores(rt[q], et[e], self.cfg.local_temperature)
                    scores = 0.5 * scores + 0.5 * local
                chunks.append(scores)
            return torch.cat(chunks)
        scores = rg @ eg.T
        if self.variant == "B1":
            return scores
        # This full-grid path is for no-grad validation and screening. Training
        # requests only the pairs admitted by the existing masked objective.
        for qs in range(0, len(rg), query_chunk):
            for es in range(0, len(eg), target_chunk):
                r = rt[qs:qs + query_chunk]
                e = et[es:es + target_chunk]
                sim = torch.einsum("qid,ejd->qeij", r, e)
                local = (self.cfg.local_temperature * (
                    torch.logsumexp(sim / self.cfg.local_temperature, dim=-1) - math.log(e.shape[1])
                )).mean(dim=-1)
                scores[qs:qs + len(r), es:es + len(e)] = (
                    0.5 * scores[qs:qs + len(r), es:es + len(e)] + 0.5 * local)
        return scores

    def description(self):
        result = {"variant": self.variant, **asdict(self.cfg)}
        if self.cfg.graph_backbone == "pcqm4mv2_pretrained":
            result["graph_initialization"] = self.graph_encoder.provenance
        return result
