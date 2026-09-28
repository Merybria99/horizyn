"""
Lightning module for Horizyn contrastive learning.

This module implements the training and validation loop for the dual-encoder
contrastive learning model using Maximum Likelihood Noise Contrastive Estimation (MLNCE).
"""

from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from horizyn.checkpoint_utils import load_query_encoder_checkpoint
from horizyn.losses import HorizynFGWLoss, build_horizyn_loss
from horizyn.metrics import create_retrieval_metrics
from horizyn.model import (
    DualContrastiveModel,
    HybridReactionEncoder,
    MultimodalReactionAttentionEncoder,
    UniMol2ReactionAttentionEncoder,
)
from horizyn.structural_similarity import (
    flatten_feature_matrix,
    pairwise_cosine_similarity,
    reaction_feature_matrix,
)


class HorizynLitModule(pl.LightningModule):
    """
    Lightning module for Horizyn SOTA contrastive learning.

    This module orchestrates training and validation for the dual-encoder
    contrastive model. It computes MLNCE loss during training, and evaluates
    retrieval metrics during validation.

    SOTA Configuration:
        - Model: DualContrastiveModel with MLP encoders
        - Loss: FullBatchMLNCELoss with beta=10.0
        - Optimizer: AdamW with lr=1e-4
        - Distance: Cosine distance

    Args:
        query_encoder_dims: Dimensions for query (reaction) encoder MLP.
        target_encoder_dims: Dimensions for target (protein) encoder MLP.
        embedding_dim: Output embedding dimension (default: 512).
        beta: Inverse temperature for MLNCE loss (default: 10.0).
        learn_beta: Whether to learn beta parameter (default: False).
        learning_rate: Learning rate for AdamW optimizer (default: 1e-4).
        weight_decay: Weight decay for AdamW optimizer (default: 0.01).
        metric_ks: List of k values for top-k metrics (default: [1, 5, 10, 50]).

    Example:
        >>> lit_module = HorizynLitModule(
        ...     query_encoder_dims=[2048, 4096, 512],
        ...     target_encoder_dims=[1024, 4096, 512],
        ...     embedding_dim=512,
        ...     beta=10.0,
        ...     learn_beta=False,
        ...     learning_rate=1e-4,
        ... )
        >>> trainer = pl.Trainer(max_epochs=100)
        >>> trainer.fit(lit_module, datamodule=data_module)
    """

    def __init__(
        self,
        query_encoder_dims: List[int],
        target_encoder_dims: List[int],
        embedding_dim: int = 512,
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
        metric_ks: List[int] = [1, 5, 10, 50],
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

        # Validate and map encoder dims to MLP kwargs
        if len(query_encoder_dims) < 2 or len(target_encoder_dims) < 2:
            raise ValueError(
                "Encoder dims must include at least [input_dim, output_dim]. "
                f"Got query_encoder_dims={query_encoder_dims}, target_encoder_dims={target_encoder_dims}"
            )
        # Fail fast if the provided dims' output differs from embedding_dim
        if query_encoder_dims[-1] != embedding_dim:
            raise ValueError(
                f"query_encoder_dims final element must equal embedding_dim. "
                f"Got query_encoder_dims[-1]={query_encoder_dims[-1]} vs embedding_dim={embedding_dim}"
            )
        if target_encoder_dims[-1] != embedding_dim:
            raise ValueError(
                f"target_encoder_dims final element must equal embedding_dim. "
                f"Got target_encoder_dims[-1]={target_encoder_dims[-1]} vs embedding_dim={embedding_dim}"
            )
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

        # Build encoder kwargs from dims: [in, h1, ..., hN, out]
        # widths correspond to hidden layers only (exclude input and output dims)
        query_hidden_widths = query_encoder_dims[1:-1]
        target_hidden_widths = target_encoder_dims[1:-1]

        query_encoder_kwargs = {
            "input_dim": query_encoder_dims[0],
            "output_dim": embedding_dim,
            "num_layers": max(len(query_hidden_widths), 0),
            "widths": query_hidden_widths if len(query_hidden_widths) > 0 else [],
            "normalise_output": True,
        }
        query_encoder_class = None
        if query_encoder_type == "unimol2_reaction_attention":
            query_encoder_class = UniMol2ReactionAttentionEncoder
        elif query_encoder_type == "hybrid_reaction":
            query_encoder_class = HybridReactionEncoder
        elif query_encoder_type == "multimodal_reaction_attention":
            query_encoder_class = MultimodalReactionAttentionEncoder
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
        if query_encoder_type == "unimol2_reaction_attention":
            query_encoder_kwargs.update(
                {
                    "unimol_dim": reaction_unimol_dim,
                    "reaction_pooling": reaction_pooling,
                    "attention_bias": reaction_attention_bias,
                    "separate_side_poolers": reaction_separate_side_poolers,
                }
            )
        elif query_encoder_type == "hybrid_reaction":
            query_encoder_kwargs.update(
                {
                    "unimol_dim": reaction_unimol_dim,
                    "reaction_pooling": reaction_pooling,
                    "attention_bias": reaction_attention_bias,
                    "separate_side_poolers": reaction_separate_side_poolers,
                }
            )

        target_encoder_kwargs = {
            "input_dim": target_encoder_dims[0],
            "output_dim": embedding_dim,
            "num_layers": max(len(target_hidden_widths), 0),
            "widths": target_hidden_widths if len(target_hidden_widths) > 0 else [],
            "normalise_output": True,
        }

        # Model
        model_kwargs = {
            "query_encoder_kwargs": query_encoder_kwargs,
            "target_encoder_kwargs": target_encoder_kwargs,
        }
        if query_encoder_class is not None:
            model_kwargs["query_encoder"] = query_encoder_class
        self.model = DualContrastiveModel(**model_kwargs)
        load_query_encoder_checkpoint(
            self.model.query_encoder,
            query_encoder_checkpoint_path,
        )

        # Loss function
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

        # Optimizer configuration
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Metrics
        self.metric_functionals = create_retrieval_metrics(top_k=metric_ks)

        # Target lookup table for validation retrieval metrics
        self.target_lookup_table = None
        self.num_targets = None

    def forward(
        self, query_vec: torch.Tensor, target_vec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the dual encoder model.

        Args:
            query_vec: Query input vectors (e.g., reaction fingerprints).
            target_vec: Target input vectors (e.g., protein embeddings).

        Returns:
            Tuple of (query_embeddings, target_embeddings).
        """
        return self.model(query_vec, target_vec)

    def _compute_cosine_distances(
        self, query_vecs: torch.Tensor, target_vecs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine distances between query and target embeddings.

        Args:
            query_vecs: Query embeddings of shape (num_queries, embedding_dim).
            target_vecs: Target embeddings of shape (num_targets, embedding_dim).

        Returns:
            Distance matrix of shape (num_queries, num_targets).
        """
        # Cosine distance = 1 - cosine_similarity.  The model is configured to
        # emit normalized embeddings, but normalize defensively here so loss and
        # validation retrieval remain cosine-based even if a future encoder
        # config accidentally omits its final NormalizeLayer.
        query_vecs = F.normalize(query_vecs, p=2, dim=-1, eps=1e-12)
        target_vecs = F.normalize(target_vecs, p=2, dim=-1, eps=1e-12)
        return 1.0 - torch.matmul(query_vecs, target_vecs.T)

    def _loss_with_components(
        self,
        dists: torch.Tensor,
        query_idx: torch.Tensor,
        target_idx: torch.Tensor,
        query_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
        reaction_similarity: Optional[torch.Tensor] = None,
        enzyme_similarity: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        query_embeds = F.normalize(query_embeds, p=2, dim=-1, eps=1e-12)
        target_embeds = F.normalize(target_embeds, p=2, dim=-1, eps=1e-12)
        if isinstance(self.loss_fn, HorizynFGWLoss):
            loss, components = self.loss_fn(
                dists,
                query_idx,
                target_idx,
                query_embeds=query_embeds,
                target_embeds=target_embeds,
                reaction_similarity=reaction_similarity,
                enzyme_similarity=enzyme_similarity,
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
                add_dataloader_idx=False,
            )

    def _deduplicate_inputs_with_ids(
        self, vecs: torch.Tensor | dict[str, torch.Tensor], ids: List[str]
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor, List[str]]:
        """
        Deduplicate input vectors based on unique IDs.

        This is used to avoid redundant forward passes when the same
        query or target appears multiple times in a batch.

        Args:
            vecs: Input vectors of shape (batch_size, vec_dim).
            ids: List of IDs corresponding to each vector.

        Returns:
            Tuple of (deduplicated_vectors, inverse_indices, unique_ids).
            - deduplicated_vectors: Unique vectors.
            - inverse_indices: Indices to reconstruct original batch.
            - unique_ids: IDs corresponding to deduplicated_vectors.
        """
        _, unique_idx, unique_inverse = np.unique(ids, return_index=True, return_inverse=True)

        # Convert to tensors
        first_tensor = (
            next(value for value in vecs.values() if torch.is_tensor(value))
            if isinstance(vecs, dict)
            else vecs
        )
        unique_idx_tensor = torch.from_numpy(unique_idx).to(
            dtype=torch.long,
            device=first_tensor.device,
        )
        unique_inverse_tensor = torch.from_numpy(unique_inverse).to(
            dtype=torch.long, device=first_tensor.device
        )

        # Get unique vectors
        if isinstance(vecs, dict):
            unique_vecs = {
                key: value[unique_idx_tensor]
                if torch.is_tensor(value) and value.shape[0] == len(ids)
                else value
                for key, value in vecs.items()
            }
        else:
            unique_vecs = vecs[unique_idx_tensor]
        unique_ids = [ids[int(idx)] for idx in unique_idx]

        return unique_vecs, unique_inverse_tensor, unique_ids

    def _deduplicate_inputs(
        self, vecs: torch.Tensor, ids: List[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Deduplicate input vectors based on unique IDs.

        This keeps the public helper shape used by existing tests and callers.
        """
        unique_vecs, unique_inverse, _ = self._deduplicate_inputs_with_ids(vecs, ids)
        return unique_vecs, unique_inverse

    def _distributed_enabled(self) -> bool:
        """Return True when running under initialized multi-process DDP."""
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    def _gather_objects(self, obj: Any) -> List[Any]:
        """Gather a Python object from all DDP ranks."""
        if not self._distributed_enabled():
            return [obj]

        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, obj)
        return gathered

    def _gather_variable_tensor(
        self, tensor: torch.Tensor, sync_grads: bool = False
    ) -> List[torch.Tensor]:
        """
        Gather tensors whose leading dimension can differ across DDP ranks.

        Lightning's all_gather requires identical tensor shapes, so we pad to the
        maximum local length, gather, then trim each rank back to its true size.
        """
        if not self._distributed_enabled():
            return [tensor]

        local_size = torch.tensor([tensor.shape[0]], dtype=torch.long, device=self.device)
        gathered_sizes = self.all_gather(local_size)
        if gathered_sizes.dim() == local_size.dim():
            gathered_sizes = gathered_sizes.unsqueeze(0)
        gathered_sizes = gathered_sizes.reshape(-1)
        max_size = int(gathered_sizes.max().item())

        if tensor.shape[0] < max_size:
            pad_shape = (max_size - tensor.shape[0], *tensor.shape[1:])
            padding = tensor.new_zeros(pad_shape)
            padded = torch.cat([tensor, padding], dim=0)
        else:
            padded = tensor

        gathered = self.all_gather(padded, sync_grads=sync_grads)
        if gathered.dim() == padded.dim():
            gathered = gathered.unsqueeze(0)

        return [
            gathered[rank, : int(gathered_sizes[rank].item())]
            for rank in range(gathered_sizes.numel())
        ]

    def _global_full_batch_loss(
        self,
        query_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
        query_similarity_features: Optional[torch.Tensor],
        target_similarity_features: Optional[torch.Tensor],
        unique_query_ids: List[str],
        unique_target_ids: List[str],
        pair_query_ids: List[str],
        pair_target_ids: List[str],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute MLNCE over the global DDP batch.

        The SOTA objective uses the full batch as the negative pool. Under DDP,
        each process only sees a shard of the dataloader batch, so this gathers
        embeddings and pair IDs from all ranks before applying the same full-batch
        loss used for single-GPU training.
        """
        gathered_query_embeds = self._gather_variable_tensor(query_embeds, sync_grads=True)
        gathered_target_embeds = self._gather_variable_tensor(target_embeds, sync_grads=True)
        gathered_query_features = (
            self._gather_variable_tensor(query_similarity_features, sync_grads=False)
            if query_similarity_features is not None
            else None
        )
        gathered_target_features = (
            self._gather_variable_tensor(target_similarity_features, sync_grads=False)
            if target_similarity_features is not None
            else None
        )
        gathered_unique_query_ids = self._gather_objects(unique_query_ids)
        gathered_unique_target_ids = self._gather_objects(unique_target_ids)
        gathered_pair_query_ids = self._gather_objects(list(pair_query_ids))
        gathered_pair_target_ids = self._gather_objects(list(pair_target_ids))

        all_query_embeds = torch.cat(gathered_query_embeds, dim=0)
        all_target_embeds = torch.cat(gathered_target_embeds, dim=0)
        all_query_features = (
            torch.cat(gathered_query_features, dim=0) if gathered_query_features is not None else None
        )
        all_target_features = (
            torch.cat(gathered_target_features, dim=0)
            if gathered_target_features is not None
            else None
        )
        all_unique_query_ids = [
            query_id for rank_ids in gathered_unique_query_ids for query_id in rank_ids
        ]
        all_unique_target_ids = [
            target_id for rank_ids in gathered_unique_target_ids for target_id in rank_ids
        ]

        query_id_to_global_idx: dict[str, int] = {}
        query_keep_indices = []
        for idx, query_id in enumerate(all_unique_query_ids):
            if query_id not in query_id_to_global_idx:
                query_id_to_global_idx[query_id] = len(query_keep_indices)
                query_keep_indices.append(idx)

        target_id_to_global_idx: dict[str, int] = {}
        target_keep_indices = []
        for idx, target_id in enumerate(all_unique_target_ids):
            if target_id not in target_id_to_global_idx:
                target_id_to_global_idx[target_id] = len(target_keep_indices)
                target_keep_indices.append(idx)

        query_keep_tensor = torch.tensor(query_keep_indices, dtype=torch.long, device=self.device)
        target_keep_tensor = torch.tensor(target_keep_indices, dtype=torch.long, device=self.device)
        global_query_embeds = all_query_embeds[query_keep_tensor]
        global_target_embeds = all_target_embeds[target_keep_tensor]
        global_query_features = (
            all_query_features[query_keep_tensor] if all_query_features is not None else None
        )
        global_target_features = (
            all_target_features[target_keep_tensor] if all_target_features is not None else None
        )

        all_pair_query_ids = [
            query_id for rank_ids in gathered_pair_query_ids for query_id in rank_ids
        ]
        all_pair_target_ids = [
            target_id for rank_ids in gathered_pair_target_ids for target_id in rank_ids
        ]

        global_query_idx = torch.tensor(
            [query_id_to_global_idx[query_id] for query_id in all_pair_query_ids],
            dtype=torch.long,
            device=self.device,
        )
        global_target_idx = torch.tensor(
            [target_id_to_global_idx[target_id] for target_id in all_pair_target_ids],
            dtype=torch.long,
            device=self.device,
        )

        dists = self._compute_cosine_distances(global_query_embeds, global_target_embeds)
        return self._loss_with_components(
            dists,
            global_query_idx,
            global_target_idx,
            global_query_embeds,
            global_target_embeds,
            reaction_similarity=pairwise_cosine_similarity(global_query_features),
            enzyme_similarity=pairwise_cosine_similarity(global_target_features),
        )

    def _compute_full_batch_loss(
        self, batch: Dict[str, Any]
    ) -> tuple[torch.Tensor, int, dict[str, torch.Tensor]]:
        """Compute the MLNCE loss for a local or distributed full batch."""
        query_vecs = batch["query_vec"]
        target_vecs = batch["target_vec"]
        query_ids = batch["query_id"]
        target_ids = batch["target_id"]

        batch_size = len(query_ids)

        # Deduplicate inputs (same query/target may appear multiple times)
        unique_query_vecs, unique_inverse_query_ids, unique_query_ids = (
            self._deduplicate_inputs_with_ids(query_vecs, query_ids)
        )
        unique_target_vecs, unique_inverse_target_ids, unique_target_ids = (
            self._deduplicate_inputs_with_ids(target_vecs, target_ids)
        )
        query_similarity_features = reaction_feature_matrix(unique_query_vecs)
        target_similarity_features = flatten_feature_matrix(unique_target_vecs)

        # Encode
        query_embeds, target_embeds = self.model(unique_query_vecs, unique_target_vecs)

        if self._distributed_enabled():
            loss = self._global_full_batch_loss(
                query_embeds=query_embeds,
                target_embeds=target_embeds,
                query_similarity_features=query_similarity_features,
                target_similarity_features=target_similarity_features,
                unique_query_ids=unique_query_ids,
                unique_target_ids=unique_target_ids,
                pair_query_ids=query_ids,
                pair_target_ids=target_ids,
            )
        else:
            # Compute distances
            dists = self._compute_cosine_distances(query_embeds, target_embeds)

            # Compute loss
            loss = self._loss_with_components(
                dists,
                unique_inverse_query_ids,
                unique_inverse_target_ids,
                query_embeds,
                target_embeds,
                reaction_similarity=pairwise_cosine_similarity(query_similarity_features),
                enzyme_similarity=pairwise_cosine_similarity(target_similarity_features),
            )

        loss_value, loss_components = loss
        return loss_value, batch_size, loss_components

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Training step: compute embeddings and MLNCE loss.

        Args:
            batch: Batch dict containing:
                - query_vec: Reaction fingerprints (batch_size, query_dim)
                - target_vec: Protein embeddings (batch_size, target_dim)
                - query_id: List of query IDs
                - target_id: List of target IDs
            batch_idx: Index of the batch.

        Returns:
            Loss value.
        """
        loss, batch_size, loss_components = self._compute_full_batch_loss(batch)

        # Log
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

    def on_validation_epoch_start(self):
        """Initialize target lookup table at the start of validation."""
        # Get datamodule to determine number of targets
        datamodule = self.trainer.datamodule

        # Preallocate target lookup table using FULL screening set (train + val proteins)
        # CRITICAL: Must use _screening_target_data, not _target_data
        self.num_targets = len(datamodule._screening_target_data)
        vec_dim = self.model.target_encoder.output_dim

        self.target_lookup_table = torch.zeros(
            (self.num_targets, vec_dim), dtype=torch.float32, device=self.device
        )

        # Create mapping from target IDs to lookup table indices
        self.target_id_to_idx = {
            target_id: idx for idx, target_id in enumerate(datamodule._screening_target_data.keys)
        }

    def _update_target_lookup_table(self, batch: Dict[str, Any], target_embeds: torch.Tensor):
        """
        Update the target lookup table with encoded target embeddings.

        Args:
            batch: Batch dict containing target_lookup_row_idx.
            target_embeds: Encoded target embeddings.
        """
        row_indices = batch["target_lookup_row_idx"]  # CPU tensor

        # Gather across DDP ranks
        world_size = self.trainer.world_size

        gathered_embeds = self.all_gather(target_embeds)
        if world_size == 1 and gathered_embeds.dim() == target_embeds.dim():
            gathered_embeds = gathered_embeds.unsqueeze(0)

        row_indices_device = row_indices.to(self.device)
        gathered_row_indices = self.all_gather(row_indices_device)
        if world_size == 1 and gathered_row_indices.dim() == row_indices_device.dim():
            gathered_row_indices = gathered_row_indices.unsqueeze(0)

        # Update lookup table from all ranks
        for rank in range(world_size):
            embeds_from_rank = F.normalize(
                gathered_embeds[rank].float(),
                p=2,
                dim=-1,
                eps=1e-12,
            )
            indices_from_rank = gathered_row_indices[rank]

            if indices_from_rank.numel() > 0:
                self.target_lookup_table[indices_from_rank] = embeds_from_rank

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor | None:
        """
        Validation step: compute loss or retrieval metrics.

        The datamodule returns 3 dataloaders:
            0. Validation loss dataloader
            1. Target lookup table dataloader
            2. Query retrieval metrics dataloader

        Args:
            batch: Batch dict (contents vary by dataloader_idx).
            batch_idx: Index of the batch.
            dataloader_idx: Index of the dataloader (0, 1, or 2).

        Returns:
            Loss value for dataloader 0, None otherwise.
        """
        if dataloader_idx == 0:
            # Compute validation loss
            return self._validation_loss_step(batch, batch_idx)

        elif dataloader_idx == 1:
            # Build target lookup table
            self._validation_lookup_step(batch, batch_idx)
            return None

        else:
            # Compute retrieval metrics
            self._validation_retrieval_step(batch, batch_idx)
            return None

    def _validation_loss_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Compute validation loss.

        Args:
            batch: Batch dict containing query_vec, target_vec, query_id, target_id.
            batch_idx: Index of the batch.

        Returns:
            Loss value.
        """
        loss, batch_size, loss_components = self._compute_full_batch_loss(batch)

        # Log
        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            add_dataloader_idx=False,
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

    def _validation_lookup_step(self, batch: torch.Tensor | Dict[str, Any], batch_idx: int):
        """
        Build target lookup table by encoding all target embeddings.

        Args:
            batch: Either a tensor (target vectors) or batch dict containing
                   target_vec and target_lookup_row_idx.
            batch_idx: Index of the batch.
        """
        # Get datamodule for batch size info
        datamodule = self.trainer.datamodule

        # Handle both tensor and dict inputs
        if isinstance(batch, torch.Tensor):
            # Batch is just the target vectors from EmbedDataset
            target_vecs = batch
            # Compute row indices based on batch_idx and batch_size
            batch_size = target_vecs.shape[0]
            # Use train_batch_size since val loader 1 uses that
            start_idx = batch_idx * datamodule.train_batch_size
            row_indices = torch.arange(start_idx, start_idx + batch_size, dtype=torch.long)
            # Create a dict for _update_target_lookup_table
            batch_dict = {"target_lookup_row_idx": row_indices}
        else:
            # Batch is a dict (for backward compatibility with tests)
            target_vecs = batch["target_vec"]
            batch_dict = batch

        # Encode targets
        target_embeds = self.model.target_encoder(target_vecs)

        # Update lookup table
        self._update_target_lookup_table(batch_dict, target_embeds)

        # Synchronize across DDP ranks
        self.trainer.strategy.barrier()

    def _validation_retrieval_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Compute retrieval metrics using the target lookup table.

        Each query can have multiple valid targets (multi-label retrieval).
        Metrics check if ANY valid target appears in top-K.

        Args:
            batch: Batch dict containing:
                - query_vec: Query vectors (batch_size, query_dim)
                - query_id: List of query IDs (used to get target lists)
            batch_idx: Index of the batch.
        """
        query_vecs = batch["query_vec"]
        query_ids = batch["query_id"]

        # Encode queries
        query_embeds = (
            self.model.query_encoder(**query_vecs)
            if isinstance(query_vecs, dict)
            else self.model.query_encoder(query_vecs)
        )

        # Compute distances to all targets in lookup table
        dists = self._compute_cosine_distances(query_embeds, self.target_lookup_table)

        # Convert distances to scores (higher is better)
        scores = -dists

        # Get datamodule to access target lists
        datamodule = self.trainer.datamodule

        # Compute metrics for each query
        batch_size = query_embeds.shape[0]

        for metric_name, metric_func in self.metric_functionals.items():
            metric_values = []
            for idx in range(batch_size):
                # Get query ID and its list of valid target IDs
                query_id = query_ids[idx]

                # Get all valid target IDs for this query from the retrieval dataset
                valid_target_ids = datamodule._val_retrieval_targets[query_id]

                # Convert target IDs to lookup table indices
                target_indices = []
                for target_id in valid_target_ids:
                    if target_id in self.target_id_to_idx:
                        target_indices.append(self.target_id_to_idx[target_id])

                # Create tensor of target indices (padded with -1 for metric functions)
                if len(target_indices) == 0:
                    # No valid targets found - skip this query
                    continue

                target_idx_tensor = torch.tensor(
                    target_indices, dtype=torch.long, device=self.device
                )

                # Compute metric (metrics handle multiple targets via torch.isin)
                metric_value = metric_func(scores[idx], target_idx_tensor)
                metric_values.append(metric_value)

            # Log mean metric
            if len(metric_values) > 0:
                metric_tensor = torch.stack(metric_values)
                self.log(
                    f"val/{metric_name}",
                    metric_tensor.mean(),
                    on_epoch=True,
                    on_step=False,
                    prog_bar=(metric_name == "top_1_hit_rate"),  # Show Top-1 in progress bar
                    add_dataloader_idx=False,
                    batch_size=batch_size,
                    sync_dist=True,
                )

    def configure_optimizers(self):
        """
        Configure AdamW optimizer.

        Returns:
            Optimizer instance.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
