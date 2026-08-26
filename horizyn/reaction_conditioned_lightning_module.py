"""
Lightning module for reaction-conditioned residue-pooling Horizyn.
"""

from typing import Any, Dict, List

import lightning.pytorch as pl
import torch
import torch.distributed as dist

from horizyn.checkpoint_utils import load_query_encoder_checkpoint
from horizyn.losses import HorizynFGWLoss, build_horizyn_loss
from horizyn.model import (
    HybridReactionEncoder,
    MultimodalReactionAttentionEncoder,
    ReactionConditionedDualModel,
    UniMol2ReactionAttentionEncoder,
)
from horizyn.utils import residue_collate_fn


class ReactionConditionedLitModule(pl.LightningModule):
    """
    Train reaction-conditioned attention pooling with full-batch MLNCE.
    """

    def __init__(
        self,
        query_encoder_dims: List[int],
        target_encoder_dims: List[int],
        embedding_dim: int = 512,
        residue_dim: int = 1024,
        attention_rank: int | None = None,
        projection_bias: bool = False,
        score_mode: str = "target_mlp",
        value_projection_bias: bool = False,
        normalize_pooled_values: bool = True,
        return_attention: bool = False,
        beta: float = 10.0,
        learn_beta: bool = False,
        loss_name: str = "FullBatchMLNCELoss",
        lambda_r: float = 0.05,
        lambda_e: float = 0.05,
        lambda_g: float = 0.01,
        tau_r: float = 0.1,
        tau_e: float = 0.1,
        tau_t: float = 0.1,
        delta_r: float = 0.5,
        delta_e: float = 0.5,
        symmetric_gw: bool = True,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        query_encoder_type: str = "mlp",
        reaction_model_dim: int = 1024,
        reaction_unimol_dim: int = 768,
        reaction_chienn_dim: int = 256,
        reaction_use_chienn: bool = True,
        reaction_pooling: str = "attention",
        reaction_attention_bias: bool = True,
        reaction_separate_side_poolers: bool = True,
        reaction_modality_attention_hidden_dim: int | None = None,
        reaction_modality_attention_dropout: float = 0.0,
        reaction_modality_dropout: float = 0.0,
        reaction_modality_token_layer_norm: bool = False,
        reaction_modality_encoder_num_layers: int | None = None,
        reaction_modality_encoder_widths: int | list[int] | None = None,
        reaction_modality_encoder_use_layer_norm: bool = False,
        reaction_modality_encoder_dropout: float = 0.0,
        reaction_modality_encoder_normalise_output: bool = False,
        query_encoder_checkpoint_path: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if query_encoder_dims[-1] != embedding_dim:
            raise ValueError("query_encoder_dims final element must equal embedding_dim")
        if target_encoder_dims[-1] != embedding_dim:
            raise ValueError("target_encoder_dims final element must equal embedding_dim")
        if target_encoder_dims[0] != residue_dim:
            raise ValueError("target_encoder_dims first element must equal residue_dim")
        if query_encoder_type not in {
            "mlp",
            "unimol2_reaction_attention",
            "hybrid_reaction",
            "multimodal_reaction_attention",
        }:
            raise ValueError(
                "query_encoder_type must be one of: mlp, unimol2_reaction_attention, "
                "hybrid_reaction, multimodal_reaction_attention"
            )
        if reaction_pooling not in {"attention", "mean"}:
            raise ValueError("reaction_pooling must be one of: attention, mean")
        if (
            query_encoder_type == "unimol2_reaction_attention"
            and query_encoder_dims[0] != 4 * reaction_unimol_dim
        ):
            raise ValueError("Uni-Mol2 reaction query input dim must equal 4 * reaction_unimol_dim")

        query_hidden_widths = query_encoder_dims[1:-1]
        target_hidden_widths = target_encoder_dims[1:-1]
        query_encoder_class = None
        if query_encoder_type == "unimol2_reaction_attention":
            query_encoder_class = UniMol2ReactionAttentionEncoder
        elif query_encoder_type == "hybrid_reaction":
            query_encoder_class = HybridReactionEncoder
        elif query_encoder_type == "multimodal_reaction_attention":
            query_encoder_class = MultimodalReactionAttentionEncoder

        query_encoder_kwargs = {
            "input_dim": query_encoder_dims[0],
            "output_dim": embedding_dim,
            "num_layers": len(query_hidden_widths),
            "widths": query_hidden_widths,
            "normalise_output": True,
        }
        if query_encoder_type in {"unimol2_reaction_attention", "hybrid_reaction"}:
            query_encoder_kwargs.update(
                {
                    "unimol_dim": reaction_unimol_dim,
                    "reaction_pooling": reaction_pooling,
                    "attention_bias": reaction_attention_bias,
                    "separate_side_poolers": reaction_separate_side_poolers,
                }
            )
        elif query_encoder_type == "multimodal_reaction_attention":
            query_encoder_kwargs.update(
                {
                    "reaction_model_dim": reaction_model_dim,
                    "unimol_dim": reaction_unimol_dim,
                    "chienn_dim": reaction_chienn_dim,
                    "use_chienn": reaction_use_chienn,
                    "reaction_pooling": reaction_pooling,
                    "attention_bias": reaction_attention_bias,
                    "separate_side_poolers": reaction_separate_side_poolers,
                    "modality_attention_hidden_dim": reaction_modality_attention_hidden_dim,
                    "modality_attention_dropout": reaction_modality_attention_dropout,
                    "modality_dropout": reaction_modality_dropout,
                    "modality_token_layer_norm": reaction_modality_token_layer_norm,
                    "modality_encoder_num_layers": reaction_modality_encoder_num_layers,
                    "modality_encoder_widths": reaction_modality_encoder_widths,
                    "modality_encoder_use_layer_norm": reaction_modality_encoder_use_layer_norm,
                    "modality_encoder_dropout": reaction_modality_encoder_dropout,
                    "modality_encoder_normalise_output": (
                        reaction_modality_encoder_normalise_output
                    ),
                }
            )

        model_kwargs = {}
        if query_encoder_class is not None:
            model_kwargs["query_encoder"] = query_encoder_class

        self.model = ReactionConditionedDualModel(
            query_encoder_kwargs=query_encoder_kwargs,
            target_encoder_kwargs={
                "input_dim": target_encoder_dims[0],
                "output_dim": embedding_dim,
                "num_layers": len(target_hidden_widths),
                "widths": target_hidden_widths,
                "normalise_output": True,
            },
            residue_dim=residue_dim,
            embedding_dim=embedding_dim,
            attention_rank=attention_rank,
            projection_bias=projection_bias,
            score_mode=score_mode,
            value_projection_bias=value_projection_bias,
            normalize_pooled_values=normalize_pooled_values,
            return_attention=return_attention,
            **model_kwargs,
        )
        load_query_encoder_checkpoint(
            self.model.query_encoder,
            query_encoder_checkpoint_path,
        )
        self.loss_fn = build_horizyn_loss(
            name=loss_name,
            beta=beta,
            learn_beta=learn_beta,
            lambda_r=lambda_r,
            lambda_e=lambda_e,
            lambda_g=lambda_g,
            tau_r=tau_r,
            tau_e=tau_e,
            tau_t=tau_t,
            delta_r=delta_r,
            delta_e=delta_e,
            symmetric_gw=symmetric_gw,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def _loss_with_components(
        self,
        dists: torch.Tensor,
        query_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if isinstance(self.loss_fn, HorizynFGWLoss):
            loss, components = self.loss_fn(
                dists,
                query_idx,
                target_idx,
                return_components=True,
            )
            return loss, components
        return self.loss_fn(dists, query_idx, target_idx), {}

    def _log_loss_components(
        self,
        prefix: str,
        components: dict[str, torch.Tensor],
        batch_size: int,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        for name, value in components.items():
            self.log(
                f"{prefix}/loss_{name}",
                value,
                on_epoch=on_epoch,
                on_step=on_step,
                batch_size=batch_size,
                sync_dist=True,
            )

    def _distributed_enabled(self) -> bool:
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    def _batch_to_cpu_samples(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        query_vecs = batch["query_vec"]
        residues = batch["residue_embeddings"].detach().cpu()
        masks = batch["residue_padding_mask"].detach().cpu()
        query_ids = list(batch["query_id"])
        target_ids = list(batch["target_id"])

        samples = []
        for idx, (query_id, target_id) in enumerate(zip(query_ids, target_ids)):
            length = int((~masks[idx]).sum().item())
            samples.append(
                {
                    "query_id": query_id,
                    "target_id": target_id,
                    "query_vec": {
                        key: value[idx].detach().cpu()
                        if torch.is_tensor(value) and value.shape[0] == len(query_ids)
                        else value
                        for key, value in query_vecs.items()
                    }
                    if isinstance(query_vecs, dict)
                    else query_vecs[idx].detach().cpu(),
                    "residue_embeddings": residues[idx, :length].contiguous(),
                }
            )
        return samples

    def _gather_global_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        local_samples = self._batch_to_cpu_samples(batch)
        if self._distributed_enabled():
            gathered_samples = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_samples, local_samples)
            samples = [sample for rank_samples in gathered_samples for sample in rank_samples]
        else:
            samples = local_samples

        global_batch = residue_collate_fn(samples)
        for key, value in list(global_batch.items()):
            if isinstance(value, torch.Tensor):
                global_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                global_batch[key] = {
                    nested_key: nested_value.to(self.device)
                    if isinstance(nested_value, torch.Tensor)
                    else nested_value
                    for nested_key, nested_value in value.items()
                }
        return global_batch

    @staticmethod
    def _deduplicate_query_vectors(
        query_vecs: torch.Tensor | dict[str, torch.Tensor],
        query_ids: List[str],
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor, List[str]]:
        id_to_idx: dict[str, int] = {}
        keep_indices = []
        inverse_indices = []
        unique_ids = []
        for row_idx, query_id in enumerate(query_ids):
            if query_id not in id_to_idx:
                id_to_idx[query_id] = len(keep_indices)
                keep_indices.append(row_idx)
                unique_ids.append(query_id)
            inverse_indices.append(id_to_idx[query_id])

        first_tensor = (
            next(value for value in query_vecs.values() if torch.is_tensor(value))
            if isinstance(query_vecs, dict)
            else query_vecs
        )
        device = first_tensor.device
        keep = torch.tensor(keep_indices, dtype=torch.long, device=device)
        inverse = torch.tensor(inverse_indices, dtype=torch.long, device=device)
        if isinstance(query_vecs, dict):
            unique_query_vecs = {
                key: value[keep] if torch.is_tensor(value) and value.shape[0] == len(query_ids) else value
                for key, value in query_vecs.items()
            }
        else:
            unique_query_vecs = query_vecs[keep]
        return unique_query_vecs, inverse, unique_ids

    @staticmethod
    def _deduplicate_residue_tensors(
        residue_embeddings: torch.Tensor,
        residue_padding_mask: torch.Tensor,
        target_ids: List[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        id_to_idx: dict[str, int] = {}
        keep_indices = []
        inverse_indices = []
        unique_ids = []
        for row_idx, target_id in enumerate(target_ids):
            if target_id not in id_to_idx:
                id_to_idx[target_id] = len(keep_indices)
                keep_indices.append(row_idx)
                unique_ids.append(target_id)
            inverse_indices.append(id_to_idx[target_id])

        device = residue_embeddings.device
        keep = torch.tensor(keep_indices, dtype=torch.long, device=device)
        inverse = torch.tensor(inverse_indices, dtype=torch.long, device=device)
        return residue_embeddings[keep], residue_padding_mask[keep], inverse, unique_ids

    def _compute_full_batch_loss(
        self, batch: Dict[str, Any]
    ) -> tuple[torch.Tensor, int, dict[str, torch.Tensor]]:
        global_batch = self._gather_global_batch(batch)
        query_vecs = global_batch["query_vec"]
        residue_embeddings = global_batch["residue_embeddings"]
        residue_padding_mask = global_batch["residue_padding_mask"]
        query_ids = list(global_batch["query_id"])
        target_ids = list(global_batch["target_id"])

        unique_query_vecs, query_idx, _ = self._deduplicate_query_vectors(query_vecs, query_ids)
        unique_residues, unique_masks, target_idx, _ = self._deduplicate_residue_tensors(
            residue_embeddings,
            residue_padding_mask,
            target_ids,
        )

        scores = self.model.score_matrix(unique_query_vecs, unique_residues, unique_masks)
        dists = 1.0 - scores
        loss, components = self._loss_with_components(dists, query_idx, target_idx)
        return loss, len(query_ids), components

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, batch_size, loss_components = self._compute_full_batch_loss(batch)
        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        if self.loss_fn.learn_beta:
            self.log(
                "train/beta",
                self.loss_fn.beta,
                on_epoch=True,
                on_step=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        self._log_loss_components(
            "train",
            loss_components,
            batch_size=batch_size,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        loss, batch_size, loss_components = self._compute_full_batch_loss(batch)
        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self._log_loss_components(
            "val",
            loss_components,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
