"""Promiscuity-aware biological residual for reaction--enzyme retrieval.

The residual deliberately leaves the reusable CIRCE embeddings untouched.  It
scores a reaction's molecule set against a compact set of SLEEC-selected
protein residues and is fused only at score level::

    score = circe_cosine + alpha * tanh(biological_score)

Setting ``alpha`` to zero is therefore an exact CIRCE recovery path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

from horizyn.datasets.base import BaseDataset


FUNCTIONAL_TOKEN_SCHEMA_VERSION = "sleec_functional_tokens_v1"


class FunctionalTokenH5Dataset(BaseDataset[str]):
    """Read fixed-width SLEEC-selected residue tokens from HDF5.

    Required datasets are ``ids``, ``token_vectors``, ``token_mask``, and
    ``sleec_scores``.  ``token_mask`` uses True for real tokens (unlike the
    padding masks used by the legacy residue collator).
    """

    def __init__(self, file_path: str | Path) -> None:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Functional-token HDF5 not found: {path}")
        self.file_path = str(path)
        self.file: h5py.File | None = None
        self._file_pid: int | None = None
        with h5py.File(path, "r") as handle:
            required = {"ids", "token_vectors", "token_mask", "sleec_scores"}
            missing = sorted(required - set(handle.keys()))
            if missing:
                raise KeyError(f"Functional-token HDF5 is missing datasets: {missing}")
            ids = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in handle["ids"][:]
            ]
            vectors = handle["token_vectors"]
            mask = handle["token_mask"]
            scores = handle["sleec_scores"]
            if vectors.ndim != 3:
                raise ValueError("token_vectors must have shape [N, K, D]")
            if mask.shape != vectors.shape[:2] or scores.shape != vectors.shape[:2]:
                raise ValueError("token_mask and sleec_scores must have shape [N, K]")
            if len(ids) != vectors.shape[0] or len(ids) != len(set(ids)):
                raise ValueError("Functional-token ids must be unique and align with token_vectors")
            if any(not value.strip() for value in ids):
                raise ValueError("Functional-token ids must not be empty")
            schema = handle.attrs.get("schema_version", "")
            if isinstance(schema, bytes):
                schema = schema.decode("utf-8")
            if schema and str(schema) != FUNCTIONAL_TOKEN_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported functional-token schema {schema!r}; "
                    f"expected {FUNCTIONAL_TOKEN_SCHEMA_VERSION!r}"
                )
            self.token_count = int(vectors.shape[1])
            self.vec_dim = int(vectors.shape[2])
            self.has_positions = "token_positions" in handle
        super().__init__(keys=ids, use_key_to_idx=True)

    def _ensure_file(self) -> h5py.File:
        pid = os.getpid()
        if self.file is not None and self._file_pid != pid:
            self.close()
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")
            self._file_pid = pid
        return self.file

    def __getitem__(self, key: str | int) -> dict[str, torch.Tensor]:
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            actual_key = self.keys[key]
            idx = key
        else:
            actual_key = key
            idx = self._get_idx(actual_key)
        handle = self._ensure_file()
        sample = {
            "biological_residue_tokens": torch.from_numpy(handle["token_vectors"][idx]).float(),
            "biological_residue_mask": torch.from_numpy(handle["token_mask"][idx]).bool(),
            "biological_sleec_scores": torch.from_numpy(handle["sleec_scores"][idx]).float(),
        }
        if self.has_positions:
            sample["biological_residue_positions"] = torch.from_numpy(
                handle["token_positions"][idx]
            ).long()
        return self._apply_transforms(actual_key, sample)

    def close(self) -> None:
        if self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass
        self.file = None
        self._file_pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["file"] = None
        state["_file_pid"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def select_functional_token_indices(
    sleec_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    top_k: int,
    context_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select high-SLEEC residues plus sequence-stratified context residues.

    This helper is deterministic and intentionally has no learned query.  It is
    used by the one-time cache builder and is exposed here for testing.
    """

    if sleec_scores.ndim != 2 or valid_mask.shape != sleec_scores.shape:
        raise ValueError("sleec_scores and valid_mask must both have shape [B, L]")
    if top_k < 0 or context_k < 0 or top_k + context_k <= 0:
        raise ValueError("top_k and context_k must be non-negative with a positive sum")
    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask.bool()
    batch_size, _seq_len = sleec_scores.shape
    width = top_k + context_k
    indices = torch.zeros(batch_size, width, dtype=torch.long, device=sleec_scores.device)
    selected_mask = torch.zeros(batch_size, width, dtype=torch.bool, device=sleec_scores.device)
    for row in range(batch_size):
        valid = torch.nonzero(valid_mask[row], as_tuple=False).flatten()
        if valid.numel() == 0:
            raise ValueError("Every protein must contain at least one valid residue")
        ranked = valid[torch.argsort(sleec_scores[row, valid], descending=True, stable=True)]
        chosen = ranked[:top_k].tolist()
        chosen_set = set(chosen)
        remaining = [int(value) for value in valid.tolist() if int(value) not in chosen_set]
        if context_k > 0 and remaining:
            # Midpoints of equal-width sequence bins avoid an N-terminus bias.
            positions = torch.linspace(
                0,
                len(remaining) - 1,
                steps=min(context_k, len(remaining)),
                device=sleec_scores.device,
            ).round().long().tolist()
            chosen.extend(remaining[position] for position in positions)
        count = min(len(chosen), width)
        if count:
            indices[row, :count] = torch.tensor(
                chosen[:count], dtype=torch.long, device=sleec_scores.device
            )
            selected_mask[row, :count] = True
    return indices, selected_mask


class _TokenTransformer(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or layers < 0 or dim % heads != 0:
            raise ValueError("token dim must be positive/divisible by heads and layers non-negative")
        if layers == 0:
            self.encoder = nn.Identity()
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=4 * dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or valid_mask.shape != tokens.shape[:2]:
            raise ValueError("tokens must be [B, K, D] and valid_mask [B, K]")
        if valid_mask.sum(dim=1).eq(0).any():
            raise ValueError("Every token set must contain at least one valid token")
        if isinstance(self.encoder, nn.Identity):
            encoded = tokens
        else:
            encoded = self.encoder(tokens, src_key_padding_mask=~valid_mask.bool())
        return self.norm(encoded).masked_fill(~valid_mask.unsqueeze(-1), 0.0)


class SLEECFunctionalTokenEncoder(nn.Module):
    """Project and contextualize SLEEC-selected ProtT5 residue tokens."""

    def __init__(
        self,
        input_dim: int = 1024,
        token_dim: int = 128,
        heads: int = 4,
        layers: int = 1,
        dropout: float = 0.1,
        sleec_threshold: float = 0.34,
        sleec_pool_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if not 0.0 < sleec_threshold < 1.0:
            raise ValueError("sleec_threshold must lie in (0, 1)")
        self.input_dim = int(input_dim)
        self.token_dim = int(token_dim)
        self.sleec_threshold = float(sleec_threshold)
        self.sleec_pool_scale = float(sleec_pool_scale)
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.contextualizer = _TokenTransformer(token_dim, heads, layers, dropout)

    def forward(
        self,
        residue_tokens: torch.Tensor,
        residue_mask: torch.Tensor,
        sleec_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residue_tokens.ndim != 3 or residue_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"residue_tokens must have shape [B, K, {self.input_dim}], "
                f"got {tuple(residue_tokens.shape)}"
            )
        if residue_mask.shape != residue_tokens.shape[:2]:
            raise ValueError("residue_mask must match residue_tokens [B, K]")
        if sleec_scores.shape != residue_tokens.shape[:2]:
            raise ValueError("sleec_scores must match residue_tokens [B, K]")
        valid = residue_mask.bool()
        tokens = self.contextualizer(self.projection(residue_tokens.float()), valid)
        centered = sleec_scores.float() - self.sleec_threshold
        logits = self.sleec_pool_scale * centered
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1).masked_fill(~valid, 0.0)
        pooled = torch.einsum("bk,bkd->bd", weights, tokens)
        return F.normalize(tokens, dim=-1, eps=1e-12), F.normalize(
            pooled, dim=-1, eps=1e-12
        )


class MoleculeSetTokenEncoder(nn.Module):
    """Permutation-equivariant UniMol2+ChIRo molecule-set encoder."""

    def __init__(
        self,
        unimol_dim: int = 768,
        chiro_dim: int = 256,
        token_dim: int = 128,
        heads: int = 4,
        layers: int = 1,
        dropout: float = 0.1,
        max_molecules: int = 32,
        deduplicate_self_reactions: bool = True,
    ) -> None:
        super().__init__()
        if max_molecules <= 0:
            raise ValueError("max_molecules must be positive")
        self.unimol_dim = int(unimol_dim)
        self.chiro_dim = int(chiro_dim)
        self.token_dim = int(token_dim)
        self.max_molecules = int(max_molecules)
        self.deduplicate_self_reactions = bool(deduplicate_self_reactions)
        self.unimol_projection = nn.Sequential(
            nn.LayerNorm(unimol_dim), nn.Linear(unimol_dim, token_dim)
        )
        self.chiro_projection = nn.Sequential(
            nn.LayerNorm(chiro_dim), nn.Linear(chiro_dim, token_dim)
        )
        self.modality_embeddings = nn.Parameter(torch.zeros(2, token_dim))
        nn.init.normal_(self.modality_embeddings, std=0.02)
        self.contextualizer = _TokenTransformer(token_dim, heads, layers, dropout)
        # The same learned content score is used for the hard memory cap and
        # final set pooling.  A separate top-k-only scorer would be a dead
        # parameter because gradients cannot flow through integer indices.
        self.pool_score = nn.Linear(token_dim, 1, bias=False)

    def _limit_token_count(
        self, tokens: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.shape[1] <= self.max_molecules:
            return tokens, valid
        logits = self.pool_score(tokens).squeeze(-1)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        indices = torch.topk(
            logits, k=self.max_molecules, dim=1, largest=True, sorted=True
        ).indices
        tokens = torch.gather(
            tokens, 1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        )
        valid = torch.gather(valid, 1, indices)
        return tokens, valid

    @staticmethod
    def _valid_mask(values: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        if padding_mask is None:
            return torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        if padding_mask.shape != values.shape[:2]:
            raise ValueError("molecule padding mask must match [B, M]")
        return ~padding_mask.bool()

    @staticmethod
    def _same_set_rows(
        reactants: torch.Tensor,
        products: torch.Tensor,
        reactant_valid: torch.Tensor,
        product_valid: torch.Tensor,
    ) -> torch.Tensor:
        if reactants.shape != products.shape or reactant_valid.shape != product_valid.shape:
            return torch.zeros(reactants.shape[0], dtype=torch.bool, device=reactants.device)
        same_validity = (reactant_valid == product_valid).all(dim=1)
        difference = (reactants.float() - products.float()).abs().amax(dim=-1)
        same_values = (difference.masked_fill(~reactant_valid, 0.0) <= 1e-6).all(dim=1)
        return same_validity & same_values

    def forward(
        self,
        reactant_embeddings: torch.Tensor,
        product_embeddings: torch.Tensor,
        reactant_padding_mask: torch.Tensor | None = None,
        product_padding_mask: torch.Tensor | None = None,
        reactant_chirality_embeddings: torch.Tensor | None = None,
        product_chirality_embeddings: torch.Tensor | None = None,
        reactant_chirality_padding_mask: torch.Tensor | None = None,
        product_chirality_padding_mask: torch.Tensor | None = None,
        has_chirality: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for name, value in (
            ("reactant_embeddings", reactant_embeddings),
            ("product_embeddings", product_embeddings),
        ):
            if value.ndim != 3 or value.shape[-1] != self.unimol_dim:
                raise ValueError(
                    f"{name} must have shape [B, M, {self.unimol_dim}], got {tuple(value.shape)}"
                )
        if reactant_embeddings.shape[0] != product_embeddings.shape[0]:
            raise ValueError("reactant and product batches must have equal size")
        reactant_valid = self._valid_mask(reactant_embeddings, reactant_padding_mask)
        product_valid = self._valid_mask(product_embeddings, product_padding_mask)
        if self.deduplicate_self_reactions:
            self_rows = self._same_set_rows(
                reactant_embeddings, product_embeddings, reactant_valid, product_valid
            )
            product_valid = product_valid & ~self_rows.unsqueeze(1)

        unimol = torch.cat((reactant_embeddings, product_embeddings), dim=1).float()
        unimol_valid = torch.cat((reactant_valid, product_valid), dim=1)
        token_groups = [
            self.unimol_projection(unimol) + self.modality_embeddings[0]
        ]
        valid_groups = [unimol_valid]

        chiro_present = (
            reactant_chirality_embeddings is not None
            and product_chirality_embeddings is not None
        )
        if chiro_present:
            if (
                reactant_chirality_embeddings.ndim != 3
                or product_chirality_embeddings.ndim != 3
                or reactant_chirality_embeddings.shape[0] != reactant_embeddings.shape[0]
                or product_chirality_embeddings.shape[0] != product_embeddings.shape[0]
                or reactant_chirality_embeddings.shape[-1] != self.chiro_dim
                or product_chirality_embeddings.shape[-1] != self.chiro_dim
            ):
                raise ValueError(
                    "ChIRo tensors must share the reaction batch and configured feature dim"
                )
            reactant_chiro_valid = self._valid_mask(
                reactant_chirality_embeddings, reactant_chirality_padding_mask
            )
            product_chiro_valid = self._valid_mask(
                product_chirality_embeddings, product_chirality_padding_mask
            )
            if self.deduplicate_self_reactions:
                chiro_self_rows = self._same_set_rows(
                    reactant_chirality_embeddings,
                    product_chirality_embeddings,
                    reactant_chiro_valid,
                    product_chiro_valid,
                )
                product_chiro_valid = product_chiro_valid & ~chiro_self_rows.unsqueeze(1)
            chiro = torch.cat(
                (reactant_chirality_embeddings, product_chirality_embeddings), dim=1
            ).float()
            chiro_valid = torch.cat((reactant_chiro_valid, product_chiro_valid), dim=1)
            if has_chirality is not None:
                chiro_valid = chiro_valid & has_chirality.bool().unsqueeze(1)
            token_groups.append(
                self.chiro_projection(chiro) + self.modality_embeddings[1]
            )
            valid_groups.append(chiro_valid)

        # UniMol2 and ChIRo featurizers can retain different numbers of
        # molecules. They are independent typed sets, not aligned views to add.
        tokens = torch.cat(token_groups, dim=1)
        valid = torch.cat(valid_groups, dim=1)
        tokens, valid = self._limit_token_count(tokens, valid)
        if valid.sum(dim=1).eq(0).any():
            raise ValueError("Every reaction must retain at least one molecule token")
        tokens = self.contextualizer(tokens, valid)
        logits = self.pool_score(tokens).squeeze(-1)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1).masked_fill(~valid, 0.0)
        pooled = torch.einsum("bm,bmd->bd", weights, tokens)
        return (
            F.normalize(tokens, dim=-1, eps=1e-12),
            valid,
            F.normalize(pooled, dim=-1, eps=1e-12),
        )


class ResidueMoleculeScorer(nn.Module):
    """Memory-bounded symmetric late interaction between molecules and residues."""

    def __init__(
        self,
        sleec_threshold: float = 0.34,
        sleec_bias: float = 0.25,
        lse_temperature: float = 0.10,
        target_chunk_size: int = 128,
    ) -> None:
        super().__init__()
        if lse_temperature <= 0.0 or target_chunk_size <= 0:
            raise ValueError("lse_temperature and target_chunk_size must be positive")
        self.sleec_threshold = float(sleec_threshold)
        self.sleec_bias = float(sleec_bias)
        self.lse_temperature = float(lse_temperature)
        self.target_chunk_size = int(target_chunk_size)

    def _normalized_lse(self, values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        temperature = self.lse_temperature
        negative = torch.finfo(values.dtype).min
        masked = values.masked_fill(~mask, negative)
        counts = mask.sum(dim=dim).clamp_min(1).to(values.dtype)
        return temperature * (
            torch.logsumexp(masked / temperature, dim=dim) - counts.log()
        )

    def forward(
        self,
        reaction_tokens: torch.Tensor,
        reaction_mask: torch.Tensor,
        reaction_pooled: torch.Tensor,
        residue_tokens: torch.Tensor,
        residue_mask: torch.Tensor,
        residue_pooled: torch.Tensor,
        sleec_scores: torch.Tensor,
    ) -> torch.Tensor:
        if reaction_tokens.shape[-1] != residue_tokens.shape[-1]:
            raise ValueError("reaction and residue token dimensions must match")
        outputs: list[torch.Tensor] = []
        for start in range(0, residue_tokens.shape[0], self.target_chunk_size):
            end = min(start + self.target_chunk_size, residue_tokens.shape[0])
            enzyme_tokens = residue_tokens[start:end]
            enzyme_mask = residue_mask[start:end].bool()
            enzyme_pooled = residue_pooled[start:end]
            enzyme_scores = sleec_scores[start:end].float()

            # Reaction summary against every candidate functional residue.
            site_values = torch.einsum("qd,ekd->qek", reaction_pooled, enzyme_tokens)
            site_values = site_values + self.sleec_bias * (
                enzyme_scores - self.sleec_threshold
            ).unsqueeze(0)
            site_mask = enzyme_mask.unsqueeze(0).expand(site_values.shape)
            site_score = self._normalized_lse(site_values, site_mask, dim=2)

            # Every contextualized molecule against the SLEEC-weighted enzyme summary.
            molecule_values = torch.einsum("qmd,ed->qem", reaction_tokens, enzyme_pooled)
            molecule_mask = reaction_mask[:, None, :].expand(molecule_values.shape)
            molecule_score = self._normalized_lse(molecule_values, molecule_mask, dim=2)
            outputs.append(0.5 * (site_score + molecule_score))
        return torch.cat(outputs, dim=1)


class ProtectedScoreFusion(nn.Module):
    """Bounded score residual with an exact alpha=0 recovery path."""

    def __init__(self, alpha: float = 0.0, max_alpha: float = 0.2) -> None:
        super().__init__()
        if max_alpha <= 0.0 or abs(alpha) > max_alpha:
            raise ValueError("fusion alpha must satisfy abs(alpha) <= max_alpha")
        self.max_alpha = float(max_alpha)
        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))

    def set_alpha(self, alpha: float) -> None:
        if abs(float(alpha)) > self.max_alpha:
            raise ValueError(f"abs(alpha) must be <= {self.max_alpha}")
        self.alpha.fill_(float(alpha))

    def forward(self, base_score: torch.Tensor, biological_score: torch.Tensor) -> torch.Tensor:
        if base_score.shape != biological_score.shape:
            raise ValueError("base_score and biological_score must have identical shapes")
        if float(self.alpha.item()) == 0.0:
            return base_score
        return base_score + self.alpha.to(base_score.dtype) * torch.tanh(biological_score)


class PromiscuityAwareBiologicalResidual(nn.Module):
    """End-to-end molecule-set/SLEEC-residue residual scorer."""

    def __init__(
        self,
        residue_dim: int = 1024,
        unimol_dim: int = 768,
        chiro_dim: int = 256,
        token_dim: int = 128,
        heads: int = 4,
        layers: int = 1,
        dropout: float = 0.1,
        max_molecules: int = 32,
        sleec_threshold: float = 0.34,
        sleec_bias: float = 0.25,
        sleec_pool_scale: float = 2.0,
        lse_temperature: float = 0.10,
        target_chunk_size: int = 128,
        fusion_alpha: float = 0.0,
        max_fusion_alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.enzyme_encoder = SLEECFunctionalTokenEncoder(
            input_dim=residue_dim,
            token_dim=token_dim,
            heads=heads,
            layers=layers,
            dropout=dropout,
            sleec_threshold=sleec_threshold,
            sleec_pool_scale=sleec_pool_scale,
        )
        self.reaction_encoder = MoleculeSetTokenEncoder(
            unimol_dim=unimol_dim,
            chiro_dim=chiro_dim,
            token_dim=token_dim,
            heads=heads,
            layers=layers,
            dropout=dropout,
            max_molecules=max_molecules,
        )
        self.scorer = ResidueMoleculeScorer(
            sleec_threshold=sleec_threshold,
            sleec_bias=sleec_bias,
            lse_temperature=lse_temperature,
            target_chunk_size=target_chunk_size,
        )
        self.fusion = ProtectedScoreFusion(fusion_alpha, max_fusion_alpha)

    def forward(
        self,
        reaction_inputs: dict[str, torch.Tensor],
        residue_tokens: torch.Tensor,
        residue_mask: torch.Tensor,
        sleec_scores: torch.Tensor,
        *,
        base_score: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        reaction_tokens, reaction_mask, reaction_pooled = self.reaction_encoder(
            reactant_embeddings=reaction_inputs["reactant_embeddings"],
            product_embeddings=reaction_inputs["product_embeddings"],
            reactant_padding_mask=reaction_inputs.get("reactant_padding_mask"),
            product_padding_mask=reaction_inputs.get("product_padding_mask"),
            reactant_chirality_embeddings=reaction_inputs.get(
                "reactant_chirality_embeddings"
            ),
            product_chirality_embeddings=reaction_inputs.get(
                "product_chirality_embeddings"
            ),
            reactant_chirality_padding_mask=reaction_inputs.get(
                "reactant_chirality_padding_mask"
            ),
            product_chirality_padding_mask=reaction_inputs.get(
                "product_chirality_padding_mask"
            ),
            has_chirality=reaction_inputs.get(
                "has_chirality",
                reaction_inputs.get("has_chiro", reaction_inputs.get("has_chienn")),
            ),
        )
        encoded_residues, residue_pooled = self.enzyme_encoder(
            residue_tokens, residue_mask, sleec_scores
        )
        local = self.scorer(
            reaction_tokens,
            reaction_mask,
            reaction_pooled,
            encoded_residues,
            residue_mask,
            residue_pooled,
            sleec_scores,
        )
        # Missing UniMol2 rows must not acquire a dataset/missingness shortcut
        # through projection biases. Their residual is exactly zero, so scoring
        # falls back to CIRCE for those queries.
        has_unimol2 = reaction_inputs.get("has_unimol2")
        if has_unimol2 is not None:
            if has_unimol2.ndim != 1 or has_unimol2.shape[0] != local.shape[0]:
                raise ValueError("has_unimol2 must have shape [reaction_batch]")
            local = local * has_unimol2.to(device=local.device, dtype=local.dtype).unsqueeze(1)
        output = {"local_score": local, "bounded_local_score": torch.tanh(local)}
        if base_score is not None:
            output["fused_score"] = self.fusion(base_score, local)
        return output


__all__ = [
    "FUNCTIONAL_TOKEN_SCHEMA_VERSION",
    "FunctionalTokenH5Dataset",
    "MoleculeSetTokenEncoder",
    "PromiscuityAwareBiologicalResidual",
    "ProtectedScoreFusion",
    "ResidueMoleculeScorer",
    "SLEECFunctionalTokenEncoder",
    "select_functional_token_indices",
]
