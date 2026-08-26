"""
Loss functions for contrastive learning.

This module implements Maximum Likelihood Noise Contrastive Estimation (MLNCE) and related
contrastive loss functions for dual-encoder architectures.
"""

from typing import Any, Optional

import torch
from torch import nn
import torch.nn.functional as F


class FullBatchNCELoss(nn.Module):
    """
    Base class for Noise Contrastive Estimation (NCE) losses that utilize an entire
    batch to construct negative examples.

    This loss function supports a learnable inverse temperature parameter (beta) that
    controls the scale of the distance metric. Beta is stored in log space for
    numerical stability and can be constrained to a specified range during training.

    The loss operates on pairwise distances between query and target embeddings,
    where each query-target pair can be labeled as positive (matching) or negative
    (non-matching). The full batch of targets serves as a pool of negatives for
    each query.

    Attributes:
        beta_init (float): Initial value for the beta parameter.
        learn_beta (bool): Whether beta is a learnable parameter.
        beta_min (float): Minimum allowed value for beta (when learned).
        beta_max (float): Maximum allowed value for beta (when learned).
        logbeta (nn.Parameter): The beta parameter in log space.
    """

    def __init__(
        self,
        beta: float = 1.0,
        learn_beta: bool = False,
        beta_min: float = -float("inf"),
        beta_max: float = float("inf"),
        *args,
        **kwargs,
    ):
        """
        Initialize the FullBatchNCELoss.

        Args:
            beta (float, optional): Initial inverse temperature parameter that controls
                the scale of distances. Higher values make the loss more sensitive to
                small distance differences. Defaults to 1.0.
            learn_beta (bool, optional): Whether to learn beta during training.
                Defaults to False.
            beta_min (float, optional): Minimum value for beta when learning. Only
                enforced if learn_beta is True. Defaults to -inf (no constraint).
            beta_max (float, optional): Maximum value for beta when learning. Only
                enforced if learn_beta is True. Defaults to inf (no constraint).
            *args: Additional positional arguments passed to nn.Module.
            **kwargs: Additional keyword arguments passed to nn.Module.
        """
        super().__init__(*args, **kwargs)

        self.beta_init = beta
        self.learn_beta = learn_beta
        self.beta_min = beta_min
        self.beta_max = beta_max

        self._setup_beta(self.beta_init, self.learn_beta)

    def _setup_beta(self, beta_init: float, learn_beta: bool) -> None:
        """
        Set up the beta parameter in log space.

        Beta is stored as log(beta) for numerical stability, particularly when
        learning beta. This prevents beta from becoming negative and improves
        gradient behavior.

        Args:
            beta_init (float): Initial value for beta.
            learn_beta (bool): Whether to learn beta during training.
        """
        self.logbeta = nn.Parameter(torch.log(torch.tensor(beta_init)), requires_grad=learn_beta)

        if learn_beta:
            self.register_forward_pre_hook(self._clip_beta)

    @property
    def beta(self) -> torch.Tensor:
        """
        Get the beta parameter in linear space.

        Returns:
            torch.Tensor: The inverse temperature parameter (scalar).
        """
        return torch.exp(self.logbeta)

    def _clip_beta(self, module: nn.Module, input: tuple) -> None:
        """
        Clip beta to the specified range [beta_min, beta_max].

        This method is registered as a forward pre-hook when learn_beta is True,
        ensuring beta stays within the specified bounds during training.

        Args:
            module (nn.Module): The current module (self). Required for hook signature.
            input (tuple): Input to the forward pass. Required for hook signature.
        """
        # Use no_grad to safely update parameter without autograd side-effects
        with torch.no_grad():
            clamped = torch.clamp(self.beta, self.beta_min, self.beta_max)
            self.logbeta.copy_(torch.log(clamped))

    def _validate_inputs(
        self, dists: torch.Tensor, query_idx: torch.Tensor, target_idx: torch.Tensor
    ) -> None:
        """Validate inputs for NCE-style losses.

        Args:
            dists: Pairwise distance matrix of shape (num_queries, num_targets).
            query_idx: Query indices for positive pairs (shape: (num_pairs,)).
            target_idx: Target indices for positive pairs (shape: (num_pairs,)).

        Raises:
            ValueError: If shapes, dtypes, or index ranges are invalid.
        """
        if dists.ndim != 2:
            raise ValueError(
                f"dists must be rank-2 (num_queries, num_targets), got shape={tuple(dists.shape)}"
            )
        if query_idx.dtype != torch.long or target_idx.dtype != torch.long:
            raise ValueError("query_idx and target_idx must have dtype torch.long")
        if query_idx.numel() == 0 or target_idx.numel() == 0:
            raise ValueError("query_idx and target_idx must be non-empty")
        if query_idx.numel() != target_idx.numel():
            raise ValueError(
                f"query_idx and target_idx must have same length, got {query_idx.numel()} and {target_idx.numel()}"
            )
        num_queries, num_targets = dists.shape
        qmin = int(query_idx.min().item())
        tmin = int(target_idx.min().item())
        qmax = int(query_idx.max().item())
        tmax = int(target_idx.max().item())
        if qmin < 0 or tmin < 0:
            raise ValueError("query_idx and target_idx must be non-negative")
        if qmax >= num_queries:
            raise ValueError(f"query_idx out of range: max={qmax} >= num_queries={num_queries}")
        if tmax >= num_targets:
            raise ValueError(f"target_idx out of range: max={tmax} >= num_targets={num_targets}")
        if not torch.isfinite(dists).all():
            raise ValueError("dists must be finite (no NaN/Inf values)")

    def forward(
        self, dists: torch.Tensor, query_idx: torch.Tensor, target_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the NCE loss for a batch of data.

        This method must be implemented by subclasses to define the specific loss
        computation strategy.

        Args:
            dists (torch.Tensor): Pairwise distances of shape (num_queries, num_targets).
                Each entry dists[i, j] is the distance between query i and target j.
                For cosine similarity, this is typically 1 - cosine_similarity.
            query_idx (torch.Tensor): Indices of queries in positive pairs, shape (num_pairs,).
                Each entry is a row index into the dists tensor.
            target_idx (torch.Tensor): Indices of targets in positive pairs, shape (num_pairs,).
                Each entry is a column index into the dists tensor.

        Returns:
            torch.Tensor: Scalar loss value for the batch.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class FullBatchMLNCELoss(FullBatchNCELoss):
    """
    Maximum Likelihood Noise Contrastive Estimation (MLNCE) loss.

    MLNCE is a contrastive loss function that handles the case where each query can
    have multiple correct targets (multi-label), unlike standard InfoNCE which assumes
    one-to-one query-target relationships.

    The loss maximizes the likelihood of positive pairs relative to all possible pairs
    in the batch. Specifically, it treats the problem as learning a probability
    distribution over all query-target pairs, where positive pairs should have high
    probability.

    Mathematical Formulation:
        For a batch with Q queries and T targets, let:
        - d_ij be the distance between query i and target j
        - P be the set of positive (query, target) pairs
        - β (beta) be the inverse temperature parameter

        The MLNCE loss is:
            L = (β / |P|) * Σ_{(i,j) ∈ P} d_ij + log(Σ_i Σ_j exp(-β * d_ij))

        where:
        - The first term encourages small distances for positive pairs
        - The second term (partition function) normalizes over all pairs

    Key Properties:
        - Handles multiple positive targets per query naturally
        - Symmetric with respect to queries and targets
        - Scales to large batch sizes (full-batch contrastive learning)
        - Temperature parameter β controls the concentration of the distribution

    Usage Example:
        >>> loss_fn = FullBatchMLNCELoss(beta=10.0, learn_beta=False)
        >>> # Compute distances (e.g., 1 - cosine_similarity for normalized embeddings)
        >>> dists = 1 - torch.mm(query_embeds, target_embeds.t())  # (Q, T)
        >>> # Define positive pairs
        >>> query_idx = torch.tensor([0, 0, 1, 2])  # Query 0 has 2 targets
        >>> target_idx = torch.tensor([0, 3, 1, 2])
        >>> loss = loss_fn(dists, query_idx, target_idx)
    """

    def forward(
        self, dists: torch.Tensor, query_idx: torch.Tensor, target_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the MLNCE loss for a batch of data.

        Args:
            dists (torch.Tensor): Pairwise distances of shape (num_queries, num_targets).
                For normalized embeddings with cosine similarity, use:
                dists = 1 - torch.mm(query_embeds, target_embeds.t())
            query_idx (torch.Tensor): Indices of queries in positive pairs, shape (num_pairs,).
                Each entry is a row index into the dists tensor.
            target_idx (torch.Tensor): Indices of targets in positive pairs, shape (num_pairs,).
                Each entry is a column index into the dists tensor.

        Returns:
            torch.Tensor: Scalar loss value for the batch.

        Note:
            The loss is computed as:
            1. Extract distances of positive pairs: pos_dists = dists[query_idx, target_idx]
            2. Compute global partition function: logZ = logsumexp(-beta * dists)
            3. Return: mean(beta * pos_dists) + logZ
        """
        # Validate inputs
        self._validate_inputs(dists, query_idx, target_idx)

        # Extract the distances of the positive pairs (num_pairs,)
        pos_dists = dists[query_idx, target_idx]

        # Compute the partition function over all pairs in the batch
        # logZ = log(Σ_i Σ_j exp(-β * d_ij))
        logZ = torch.logsumexp(-self.beta * dists, dim=(0, 1))

        # MLNCE loss: mean distance of positives + global partition function
        return self.beta * pos_dists.mean() + logZ


class DegreeTemperedFullBatchMLNCELoss(FullBatchNCELoss):
    """MLNCE with inverse reaction-degree weighting of positive pairs.

    ``query_degrees`` should contain the full training-graph degree for every
    reaction represented by a row of ``dists``. When it is omitted, the loss
    falls back to the positive degree visible in the current matrix.
    """

    def __init__(self, degree_alpha: float = 0.5, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if degree_alpha < 0:
            raise ValueError("degree_alpha must be non-negative")
        self.degree_alpha = float(degree_alpha)

    def forward(
        self,
        dists: torch.Tensor,
        query_idx: torch.Tensor,
        target_idx: torch.Tensor,
        query_degrees: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_inputs(dists, query_idx, target_idx)
        if query_degrees is None:
            query_degrees = torch.bincount(query_idx, minlength=dists.shape[0])
        if query_degrees.ndim != 1 or query_degrees.numel() != dists.shape[0]:
            raise ValueError("query_degrees must have one value per row of dists")
        query_degrees = query_degrees.to(device=dists.device, dtype=dists.dtype)
        if bool((query_degrees[query_idx] <= 0).any()):
            raise ValueError("positive-pair query degrees must be positive")

        pair_weights = query_degrees[query_idx].pow(-self.degree_alpha)
        pair_weights = pair_weights / pair_weights.sum()
        positive_term = self.beta * torch.sum(pair_weights * dists[query_idx, target_idx])
        log_partition = torch.logsumexp(-self.beta * dists, dim=(0, 1))
        total = positive_term + log_partition
        if not return_components:
            return total
        return total, {
            "positive": positive_term,
            "log_partition": log_partition,
            "mean_pair_weight": pair_weights.mean(),
        }


class DecoupledAllPositiveInfoNCELoss(FullBatchNCELoss):
    """Bidirectional, anchor-balanced InfoNCE with positives decoupled.

    Each known positive is contrasted only with negatives for its anchor. The
    positive losses are averaged within an anchor before anchors are averaged,
    avoiding both false-negative competition and degree-dependent anchor mass.
    """

    def __init__(
        self,
        lambda_r2e: float = 0.5,
        lambda_e2r: float = 0.5,
        separate_direction_temperatures: bool = False,
        beta_r2e: float = 10.0,
        beta_e2r: float = 10.0,
        temperature_regularization_weight: float = 0.0,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if lambda_r2e < 0 or lambda_e2r < 0 or lambda_r2e + lambda_e2r <= 0:
            raise ValueError("lambda_r2e and lambda_e2r must have a positive sum")
        if beta_r2e <= 0 or beta_e2r <= 0:
            raise ValueError("directional beta values must be positive")
        if temperature_regularization_weight < 0:
            raise ValueError("temperature_regularization_weight must be non-negative")
        direction_sum = float(lambda_r2e + lambda_e2r)
        self.lambda_r2e = float(lambda_r2e) / direction_sum
        self.lambda_e2r = float(lambda_e2r) / direction_sum
        self.separate_direction_temperatures = bool(separate_direction_temperatures)
        self.temperature_regularization_weight = float(temperature_regularization_weight)
        self.register_buffer("initial_logbeta_r2e", torch.log(torch.tensor(float(beta_r2e))))
        self.register_buffer("initial_logbeta_e2r", torch.log(torch.tensor(float(beta_e2r))))
        self.logbeta_r2e = nn.Parameter(
            self.initial_logbeta_r2e.clone(),
            requires_grad=self.separate_direction_temperatures,
        )
        self.logbeta_e2r = nn.Parameter(
            self.initial_logbeta_e2r.clone(),
            requires_grad=self.separate_direction_temperatures,
        )

    @property
    def beta_r2e(self) -> torch.Tensor:
        if not self.separate_direction_temperatures:
            return self.beta
        return torch.exp(self.logbeta_r2e)

    @property
    def beta_e2r(self) -> torch.Tensor:
        if not self.separate_direction_temperatures:
            return self.beta
        return torch.exp(self.logbeta_e2r)

    def _clamped_directional_betas(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.separate_direction_temperatures:
            return self.beta, self.beta
        with torch.no_grad():
            for parameter in (self.logbeta_r2e, self.logbeta_e2r):
                clamped = torch.clamp(torch.exp(parameter), self.beta_min, self.beta_max)
                parameter.copy_(torch.log(clamped))
        return self.beta_r2e, self.beta_e2r

    @staticmethod
    def _positive_mask(
        dists: torch.Tensor,
        query_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        positive_mask = torch.zeros_like(dists, dtype=torch.bool)
        positive_mask[query_idx, target_idx] = True
        return positive_mask

    @staticmethod
    def _decoupled_anchor_loss(
        dists: torch.Tensor,
        positive_mask: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        anchor_losses: list[torch.Tensor] = []
        for anchor_idx in range(dists.shape[0]):
            positive_distances = dists[anchor_idx, positive_mask[anchor_idx]]
            negative_distances = dists[anchor_idx, ~positive_mask[anchor_idx]]
            if positive_distances.numel() == 0 or negative_distances.numel() == 0:
                continue
            positive_logits = -beta * positive_distances
            negative_logits = -beta * negative_distances
            log_negative_mass = torch.logsumexp(negative_logits, dim=0)
            positive_losses = F.softplus(log_negative_mass - positive_logits)
            anchor_losses.append(positive_losses.mean())
        if not anchor_losses:
            return dists.new_zeros(())
        return torch.stack(anchor_losses).mean()

    def _temperature_regularization(self) -> torch.Tensor:
        if not self.separate_direction_temperatures:
            return self.logbeta.new_zeros(())
        return self.temperature_regularization_weight * (
            (self.logbeta_r2e - self.initial_logbeta_r2e).square()
            + (self.logbeta_e2r - self.initial_logbeta_e2r).square()
        )

    def _cardinality_components(
        self,
        dists: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        beta_r2e, beta_e2r = self._clamped_directional_betas()
        r2e = self._decoupled_anchor_loss(dists, positive_mask, beta_r2e)
        e2r = self._decoupled_anchor_loss(dists.t(), positive_mask.t(), beta_e2r)
        temperature_regularization = self._temperature_regularization()
        cardinality = self.lambda_r2e * r2e + self.lambda_e2r * e2r
        total = cardinality + temperature_regularization
        return total, {
            "r2e": r2e,
            "e2r": e2r,
            "cardinality": cardinality,
            "temperature_regularization": temperature_regularization,
            "beta_r2e": beta_r2e,
            "beta_e2r": beta_e2r,
            "r2e_weight": dists.new_tensor(self.lambda_r2e),
            "e2r_weight": dists.new_tensor(self.lambda_e2r),
        }

    def forward(
        self,
        dists: torch.Tensor,
        query_idx: torch.Tensor,
        target_idx: torch.Tensor,
        return_components: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del kwargs
        self._validate_inputs(dists, query_idx, target_idx)
        positive_mask = self._positive_mask(dists, query_idx, target_idx)
        total, components = self._cardinality_components(dists, positive_mask)
        if return_components:
            return total, components
        return total


class HybridCardinalityRetrievalLoss(DecoupledAllPositiveInfoNCELoss):
    """MLNCE/cardinality hybrid with optional bidirectional soft-rank loss."""

    def __init__(
        self,
        cardinality_weight: float = 0.3,
        cardinality_warmup_epochs: int = 5,
        soft_rank_weight: float = 0.0,
        soft_rank_tau: float = 0.1,
        soft_rank_top_k: int = 128,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if not 0 <= cardinality_weight <= 1:
            raise ValueError("cardinality_weight must be in [0, 1]")
        if cardinality_warmup_epochs < 0:
            raise ValueError("cardinality_warmup_epochs must be non-negative")
        if soft_rank_weight < 0:
            raise ValueError("soft_rank_weight must be non-negative")
        if soft_rank_tau <= 0:
            raise ValueError("soft_rank_tau must be positive")
        if soft_rank_top_k <= 0:
            raise ValueError("soft_rank_top_k must be positive")
        self.cardinality_weight = float(cardinality_weight)
        self.cardinality_warmup_epochs = int(cardinality_warmup_epochs)
        self.soft_rank_weight = float(soft_rank_weight)
        self.soft_rank_tau = float(soft_rank_tau)
        self.soft_rank_top_k = int(soft_rank_top_k)
        self._current_epoch = 0

    def set_training_progress(self, epoch: int) -> None:
        self._current_epoch = max(0, int(epoch))

    @property
    def effective_cardinality_weight(self) -> float:
        if self.cardinality_warmup_epochs == 0:
            return self.cardinality_weight
        progress = min(1.0, self._current_epoch / self.cardinality_warmup_epochs)
        return self.cardinality_weight * progress

    def _soft_rank_anchor_loss(
        self,
        dists: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        anchor_losses: list[torch.Tensor] = []
        for anchor_idx in range(dists.shape[0]):
            positive_distances = dists[anchor_idx, positive_mask[anchor_idx]]
            negative_distances = dists[anchor_idx, ~positive_mask[anchor_idx]]
            if positive_distances.numel() == 0 or negative_distances.numel() == 0:
                continue
            hard_negative_distances = torch.topk(
                negative_distances,
                k=min(self.soft_rank_top_k, negative_distances.numel()),
                largest=False,
            ).values
            outranking_mass = torch.sigmoid(
                (positive_distances[:, None] - hard_negative_distances[None, :])
                / self.soft_rank_tau
            ).sum(dim=1)
            anchor_losses.append(torch.log1p(outranking_mass).mean())
        if not anchor_losses:
            return dists.new_zeros(())
        return torch.stack(anchor_losses).mean()

    def forward(
        self,
        dists: torch.Tensor,
        query_idx: torch.Tensor,
        target_idx: torch.Tensor,
        return_components: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del kwargs
        self._validate_inputs(dists, query_idx, target_idx)
        positive_mask = self._positive_mask(dists, query_idx, target_idx)
        cardinality_with_reg, components = self._cardinality_components(dists, positive_mask)
        temperature_regularization = components["temperature_regularization"]
        cardinality = cardinality_with_reg - temperature_regularization

        positive_distances = dists[query_idx, target_idx]
        mlnce = self.beta * positive_distances.mean() + torch.logsumexp(
            -self.beta * dists,
            dim=(0, 1),
        )
        current_weight = self.effective_cardinality_weight
        if self.soft_rank_weight > 0:
            soft_rank_r2e = self._soft_rank_anchor_loss(dists, positive_mask)
            soft_rank_e2r = self._soft_rank_anchor_loss(dists.t(), positive_mask.t())
            soft_rank = self.lambda_r2e * soft_rank_r2e + self.lambda_e2r * soft_rank_e2r
        else:
            soft_rank_r2e = dists.new_zeros(())
            soft_rank_e2r = dists.new_zeros(())
            soft_rank = dists.new_zeros(())
        weighted_soft_rank = self.soft_rank_weight * soft_rank
        total = (
            (1.0 - current_weight) * mlnce
            + current_weight * cardinality
            + temperature_regularization
            + weighted_soft_rank
        )
        components.update(
            {
                "mlnce": mlnce,
                "effective_cardinality_weight": dists.new_tensor(current_weight),
                "soft_rank_r2e": soft_rank_r2e,
                "soft_rank_e2r": soft_rank_e2r,
                "soft_rank": soft_rank,
                "weighted_soft_rank": weighted_soft_rank,
            }
        )
        if return_components:
            return total, components
        return total


class BalancedSigmoidEBMLoss(FullBatchNCELoss):
    """Balanced pairwise sigmoid energy loss over all positive/negative pairs."""

    def __init__(
        self,
        sigmoid_bias_init: float = 0.0,
        sigmoid_learn_bias: bool = True,
        sigmoid_negative_weight: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if sigmoid_negative_weight < 0:
            raise ValueError("sigmoid_negative_weight must be non-negative")
        self.bias = nn.Parameter(
            torch.tensor(float(sigmoid_bias_init)),
            requires_grad=bool(sigmoid_learn_bias),
        )
        self.sigmoid_negative_weight = float(sigmoid_negative_weight)

    def forward(
        self,
        dists: torch.Tensor,
        query_idx: torch.Tensor,
        target_idx: torch.Tensor,
        return_components: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del kwargs
        self._validate_inputs(dists, query_idx, target_idx)
        positive_mask = torch.zeros_like(dists, dtype=torch.bool)
        positive_mask[query_idx, target_idx] = True
        logits = self.beta * (1.0 - dists) + self.bias
        positive = F.softplus(-logits[positive_mask]).mean()
        if bool((~positive_mask).any()):
            negative = F.softplus(logits[~positive_mask]).mean()
        else:
            negative = dists.new_zeros(())
        weighted_negative = self.sigmoid_negative_weight * negative
        total = positive + weighted_negative
        if not return_components:
            return total
        return total, {
            "positive": positive,
            "negative": negative,
            "weighted_negative": weighted_negative,
            "bias": self.bias,
        }


class BidirectionalAnchorBalancedSupConLoss(FullBatchNCELoss):
    """
    Bidirectional supervised contrastive loss for reaction-enzyme retrieval.

    Positive indices define a sparse multi-positive matrix over the DDP-global
    reaction and enzyme embeddings. The loss averages one supervised
    contrastive term per reaction anchor and one per enzyme anchor, then
    averages the two directions so neither side is implicitly weighted by the
    number of observed positive pairs.
    """

    def __init__(
        self,
        beta: float = 10.0,
        learn_beta: bool = False,
        beta_min: float = -float("inf"),
        beta_max: float = float("inf"),
        direction_balance_weight: float = 0.0,
        lambda_r2e: float = 0.5,
        lambda_e2r: float = 0.5,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            beta=beta,
            learn_beta=learn_beta,
            beta_min=beta_min,
            beta_max=beta_max,
            *args,
            **kwargs,
        )
        if direction_balance_weight < 0:
            raise ValueError("direction_balance_weight must be non-negative")
        if lambda_r2e < 0 or lambda_e2r < 0 or lambda_r2e + lambda_e2r <= 0:
            raise ValueError("lambda_r2e and lambda_e2r must be non-negative with positive sum")
        self.direction_balance_weight = float(direction_balance_weight)
        direction_sum = float(lambda_r2e + lambda_e2r)
        self.lambda_r2e = float(lambda_r2e) / direction_sum
        self.lambda_e2r = float(lambda_e2r) / direction_sum

    @staticmethod
    def _positive_mask(
        dists: torch.Tensor,
        query_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        mask = torch.zeros_like(dists, dtype=torch.bool)
        mask[query_idx, target_idx] = True
        return mask

    @staticmethod
    def _anchor_loss(
        logits: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_anchors = positive_mask.any(dim=1)
        if not bool(valid_anchors.any()):
            return logits.new_zeros(())

        masked_logits = logits.masked_fill(~positive_mask, -torch.inf)
        positive_logsumexp = torch.logsumexp(masked_logits, dim=1)
        denominator_logsumexp = torch.logsumexp(logits, dim=1)
        anchor_losses = denominator_logsumexp - positive_logsumexp
        return anchor_losses[valid_anchors].mean()

    def forward(
        self,
        dists: torch.Tensor,
        query_idx: torch.Tensor,
        target_idx: torch.Tensor,
        query_embeds: Optional[torch.Tensor] = None,
        target_embeds: Optional[torch.Tensor] = None,
        reaction_similarity: Optional[torch.Tensor] = None,
        enzyme_similarity: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del query_embeds, target_embeds, reaction_similarity, enzyme_similarity
        self._validate_inputs(dists, query_idx, target_idx)

        logits = -self.beta * dists
        positive_mask = self._positive_mask(dists, query_idx, target_idx)

        r2e = self._anchor_loss(logits, positive_mask)
        e2r = self._anchor_loss(logits.t(), positive_mask.t())
        direction_gap = (r2e - e2r).abs()
        weighted_direction_gap = direction_gap * self.direction_balance_weight
        total = self.lambda_r2e * r2e + self.lambda_e2r * e2r + weighted_direction_gap

        if not return_components:
            return total

        return total, {
            "r2e": r2e,
            "e2r": e2r,
            "direction_gap": direction_gap,
            "weighted_direction_gap": weighted_direction_gap,
            "r2e_weight": dists.new_tensor(self.lambda_r2e),
            "e2r_weight": dists.new_tensor(self.lambda_e2r),
        }


class MultiAlignmentRetrievalLoss(FullBatchNCELoss):
    """
    Anchor-balanced multi-positive retrieval loss with optional structure terms.

    The cross-domain term averages reaction->enzyme and enzyme->reaction
    supervised contrastive losses by anchor, matching bidirectional retrieval
    metrics better than pair-averaged MLNCE. Optional EC-weighted RR/EE terms
    and detached GW geometry regularization can be enabled by config.
    """

    def __init__(
        self,
        beta: float = 10.0,
        learn_beta: bool = False,
        beta_min: float = -float("inf"),
        beta_max: float = float("inf"),
        lambda_rr: float = 0.0,
        lambda_ee: float = 0.0,
        lambda_gw: float = 0.0,
        lambda_direction_gap: float = 0.0,
        lambda_r2e: float = 0.5,
        lambda_e2r: float = 0.5,
        lambda_r2e_hard_neg: float = 0.0,
        r2e_hard_neg_top_k: int = 0,
        r2e_hard_neg_margin: float = 0.0,
        lambda_e2r_hard_neg: float = 0.0,
        e2r_hard_neg_top_k: int = 0,
        e2r_hard_neg_margin: float = 0.0,
        tau_rr: float = 0.1,
        tau_ee: float = 0.1,
        tau_gw: float = 0.1,
        gw_max_anchors: int | None = 512,
        symmetric_gw: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            beta=beta,
            learn_beta=learn_beta,
            beta_min=beta_min,
            beta_max=beta_max,
            *args,
            **kwargs,
        )
        for name, value in {
            "lambda_rr": lambda_rr,
            "lambda_ee": lambda_ee,
            "lambda_gw": lambda_gw,
            "lambda_direction_gap": lambda_direction_gap,
            "lambda_r2e": lambda_r2e,
            "lambda_e2r": lambda_e2r,
            "lambda_r2e_hard_neg": lambda_r2e_hard_neg,
            "r2e_hard_neg_margin": r2e_hard_neg_margin,
            "lambda_e2r_hard_neg": lambda_e2r_hard_neg,
            "e2r_hard_neg_margin": e2r_hard_neg_margin,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if lambda_r2e + lambda_e2r <= 0:
            raise ValueError("lambda_r2e + lambda_e2r must be positive")
        if r2e_hard_neg_top_k < 0:
            raise ValueError("r2e_hard_neg_top_k must be non-negative")
        if e2r_hard_neg_top_k < 0:
            raise ValueError("e2r_hard_neg_top_k must be non-negative")
        for name, value in {"tau_rr": tau_rr, "tau_ee": tau_ee, "tau_gw": tau_gw}.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if gw_max_anchors is not None and gw_max_anchors <= 0:
            raise ValueError("gw_max_anchors must be positive or None")

        self.lambda_rr = float(lambda_rr)
        self.lambda_ee = float(lambda_ee)
        self.lambda_gw = float(lambda_gw)
        self.lambda_direction_gap = float(lambda_direction_gap)
        direction_weight_sum = float(lambda_r2e + lambda_e2r)
        self.lambda_r2e = float(lambda_r2e) / direction_weight_sum
        self.lambda_e2r = float(lambda_e2r) / direction_weight_sum
        self.lambda_r2e_hard_neg = float(lambda_r2e_hard_neg)
        self.r2e_hard_neg_top_k = int(r2e_hard_neg_top_k)
        self.r2e_hard_neg_margin = float(r2e_hard_neg_margin)
        self.lambda_e2r_hard_neg = float(lambda_e2r_hard_neg)
        self.e2r_hard_neg_top_k = int(e2r_hard_neg_top_k)
        self.e2r_hard_neg_margin = float(e2r_hard_neg_margin)
        self.tau_rr = float(tau_rr)
        self.tau_ee = float(tau_ee)
        self.tau_gw = float(tau_gw)
        self.gw_max_anchors = None if gw_max_anchors is None else int(gw_max_anchors)
        self.symmetric_gw = bool(symmetric_gw)

    @staticmethod
    def _positive_mask(
        dists: torch.Tensor,
        query_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        mask = torch.zeros_like(dists, dtype=torch.bool)
        mask[query_idx, target_idx] = True
        return mask

    @staticmethod
    def _anchor_loss(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        valid_anchors = positive_mask.any(dim=1)
        if not bool(valid_anchors.any()):
            return logits.new_zeros(())

        positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
        positive_logsumexp = torch.logsumexp(positive_logits, dim=1)
        denominator_logsumexp = torch.logsumexp(logits, dim=1)
        anchor_losses = denominator_logsumexp - positive_logsumexp
        return anchor_losses[valid_anchors].mean()

    @staticmethod
    def _r2e_hard_negative_loss(
        logits: torch.Tensor,
        positive_mask: torch.Tensor,
        top_k: int,
        margin: float,
    ) -> torch.Tensor:
        if top_k <= 0:
            return logits.new_zeros(())

        valid_positive = positive_mask.any(dim=1)
        if not bool(valid_positive.any()):
            return logits.new_zeros(())

        negative_mask = ~positive_mask
        valid_negative = negative_mask.any(dim=1)
        valid_anchors = valid_positive & valid_negative
        if not bool(valid_anchors.any()):
            return logits.new_zeros(())

        positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
        positive_logsumexp = torch.logsumexp(positive_logits, dim=1)
        negative_logits = logits.masked_fill(~negative_mask, -torch.inf)
        k = min(int(top_k), logits.shape[1])
        hard_negative_logits = torch.topk(negative_logits, k=k, dim=1).values
        has_finite_hard_negative = torch.isfinite(hard_negative_logits).any(dim=1)
        valid_anchors = valid_anchors & has_finite_hard_negative
        if not bool(valid_anchors.any()):
            return logits.new_zeros(())

        hard_negative_logsumexp = torch.logsumexp(hard_negative_logits, dim=1)
        hard_losses = torch.nn.functional.softplus(
            hard_negative_logsumexp - positive_logsumexp + float(margin)
        )
        return hard_losses[valid_anchors].mean()

    @staticmethod
    def _weighted_supcon_loss(
        embeds: torch.Tensor,
        positive_weights: torch.Tensor | None,
        temperature: float,
        name: str,
    ) -> torch.Tensor:
        if positive_weights is None:
            return embeds.new_zeros(())
        if embeds.ndim != 2:
            raise ValueError(f"{name} embeds must be rank-2, got shape={tuple(embeds.shape)}")
        num_items = embeds.shape[0]
        if positive_weights.shape != (num_items, num_items):
            raise ValueError(
                f"{name} weights must have shape {(num_items, num_items)}, "
                f"got {tuple(positive_weights.shape)}"
            )
        if num_items < 2:
            return embeds.new_zeros(())

        weights = positive_weights.to(device=embeds.device, dtype=embeds.dtype)
        eye = torch.eye(num_items, dtype=torch.bool, device=embeds.device)
        weights = weights.masked_fill(eye, 0.0)
        positives_per_anchor = weights.sum(dim=1)
        valid_anchors = positives_per_anchor > 0
        if not bool(valid_anchors.any()):
            return embeds.new_zeros(())

        embeds = torch.nn.functional.normalize(embeds, p=2, dim=-1, eps=1e-12)
        logits = embeds @ embeds.t()
        logits = logits / temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        logits = logits.masked_fill(eye, -torch.inf)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        log_prob = log_prob.masked_fill(eye, 0.0)

        anchor_loss = -(weights * log_prob).sum(dim=1)
        anchor_loss = anchor_loss / positives_per_anchor.clamp_min(1e-12)
        return anchor_loss[valid_anchors].mean()

    def _maybe_cap_gw_anchors(
        self,
        query_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gw_max_anchors is None:
            return query_embeds, target_embeds
        return query_embeds[: self.gw_max_anchors], target_embeds[: self.gw_max_anchors]

    def _detached_gw_loss(
        self,
        query_embeds: torch.Tensor | None,
        target_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        if query_embeds is None or target_embeds is None:
            raise ValueError("query_embeds and target_embeds are required for GW loss")
        if query_embeds.ndim != 2 or target_embeds.ndim != 2:
            raise ValueError(
                "query_embeds and target_embeds must be rank-2 for GW loss, "
                f"got {tuple(query_embeds.shape)} and {tuple(target_embeds.shape)}"
            )
        if query_embeds.shape[1] != target_embeds.shape[1]:
            raise ValueError(
                "query_embeds and target_embeds must have the same embedding dimension, "
                f"got {query_embeds.shape[1]} and {target_embeds.shape[1]}"
            )
        if query_embeds.shape[0] == 0 or target_embeds.shape[0] == 0:
            return query_embeds.new_zeros(())

        query_embeds, target_embeds = self._maybe_cap_gw_anchors(query_embeds, target_embeds)
        query_embeds = torch.nn.functional.normalize(query_embeds, p=2, dim=-1, eps=1e-12)
        target_embeds = torch.nn.functional.normalize(target_embeds, p=2, dim=-1, eps=1e-12)

        reaction_geometry = (query_embeds @ query_embeds.t()).detach()
        enzyme_geometry = (target_embeds @ target_embeds.t()).detach()
        cross_similarity = query_embeds @ target_embeds.t()

        transport_r2e = torch.softmax(cross_similarity / self.tau_gw, dim=1)
        reaction_from_enzymes = transport_r2e @ enzyme_geometry @ transport_r2e.t()
        reaction_loss = torch.mean((reaction_geometry - reaction_from_enzymes) ** 2)

        if not self.symmetric_gw:
            return reaction_loss

        transport_e2r = torch.softmax(cross_similarity.t() / self.tau_gw, dim=1)
        enzyme_from_reactions = transport_e2r @ reaction_geometry @ transport_e2r.t()
        enzyme_loss = torch.mean((enzyme_geometry - enzyme_from_reactions) ** 2)
        return 0.5 * (reaction_loss + enzyme_loss)

    def forward(
        self,
        dists: torch.Tensor,
        query_idx: torch.Tensor,
        target_idx: torch.Tensor,
        query_embeds: Optional[torch.Tensor] = None,
        target_embeds: Optional[torch.Tensor] = None,
        reaction_similarity: Optional[torch.Tensor] = None,
        enzyme_similarity: Optional[torch.Tensor] = None,
        reaction_ec_weights: Optional[torch.Tensor] = None,
        enzyme_ec_weights: Optional[torch.Tensor] = None,
        e2r_dists: Optional[torch.Tensor] = None,
        structure_terms_enabled: bool = True,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del reaction_similarity, enzyme_similarity
        self._validate_inputs(dists, query_idx, target_idx)
        if e2r_dists is not None:
            self._validate_inputs(e2r_dists, query_idx, target_idx)
            if e2r_dists.shape != dists.shape:
                raise ValueError(
                    "e2r_dists must have the same reaction-by-enzyme shape as dists, "
                    f"got {tuple(e2r_dists.shape)} and {tuple(dists.shape)}"
                )

        r2e_logits = -self.beta * dists
        e2r_logits = -self.beta * (dists if e2r_dists is None else e2r_dists)
        positive_mask = self._positive_mask(dists, query_idx, target_idx)
        r2e = self._anchor_loss(r2e_logits, positive_mask)
        e2r = self._anchor_loss(e2r_logits.t(), positive_mask.t())
        cross = self.lambda_r2e * r2e + self.lambda_e2r * e2r
        direction_gap = (r2e - e2r).abs()
        weighted_direction_gap = direction_gap * self.lambda_direction_gap
        r2e_hard_neg = self._r2e_hard_negative_loss(
            r2e_logits,
            positive_mask,
            top_k=self.r2e_hard_neg_top_k,
            margin=self.r2e_hard_neg_margin,
        )
        e2r_hard_neg = self._r2e_hard_negative_loss(
            e2r_logits.t(),
            positive_mask.t(),
            top_k=self.e2r_hard_neg_top_k,
            margin=self.e2r_hard_neg_margin,
        )

        zero = dists.new_zeros(())
        rr = zero
        ee = zero
        gw = zero
        if structure_terms_enabled:
            if self.lambda_rr > 0 and query_embeds is not None:
                rr = self._weighted_supcon_loss(
                    query_embeds,
                    reaction_ec_weights,
                    temperature=self.tau_rr,
                    name="reaction_ec",
                )
            if self.lambda_ee > 0 and target_embeds is not None:
                ee = self._weighted_supcon_loss(
                    target_embeds,
                    enzyme_ec_weights,
                    temperature=self.tau_ee,
                    name="enzyme_ec",
                )
            if self.lambda_gw > 0:
                gw = self._detached_gw_loss(query_embeds, target_embeds)

        weighted_rr = rr * self.lambda_rr
        weighted_ee = ee * self.lambda_ee
        weighted_gw = gw * self.lambda_gw
        weighted_r2e_hard_neg = r2e_hard_neg * self.lambda_r2e_hard_neg
        weighted_e2r_hard_neg = e2r_hard_neg * self.lambda_e2r_hard_neg
        total = (
            cross
            + weighted_direction_gap
            + weighted_r2e_hard_neg
            + weighted_e2r_hard_neg
            + weighted_rr
            + weighted_ee
            + weighted_gw
        )

        if not return_components:
            return total

        return total, {
            "r2e": r2e,
            "e2r": e2r,
            "cross": cross,
            "r2e_weight": dists.new_tensor(self.lambda_r2e),
            "e2r_weight": dists.new_tensor(self.lambda_e2r),
            "direction_gap": direction_gap,
            "weighted_direction_gap": weighted_direction_gap,
            "r2e_hard_neg": r2e_hard_neg,
            "weighted_r2e_hard_neg": weighted_r2e_hard_neg,
            "e2r_hard_neg": e2r_hard_neg,
            "weighted_e2r_hard_neg": weighted_e2r_hard_neg,
            "rr": rr,
            "ee": ee,
            "gw": gw,
            "weighted_rr": weighted_rr,
            "weighted_ee": weighted_ee,
            "weighted_gw": weighted_gw,
        }


class HorizynFGWLoss(FullBatchMLNCELoss):
    """
    Horizyn MLNCE loss with optional structure-aware regularizers.

    The original reaction-enzyme retrieval objective is kept unchanged by reusing
    FullBatchMLNCELoss for the MLNCE term. Optional regularizers add:
        - reaction-reaction supervised contrastive structure (RR)
        - enzyme-enzyme supervised contrastive structure (EE)
        - symmetric GW-style geometry alignment between learned spaces (GW)

    If the external similarity matrices or learned embeddings required by a
    regularizer are not provided, that regularizer contributes zero. This keeps
    existing training paths compatible while allowing FGW-style training where the
    modules can provide the extra inputs.
    """

    def __init__(
        self,
        beta: float = 10.0,
        learn_beta: bool = False,
        beta_min: float = -float("inf"),
        beta_max: float = float("inf"),
        lambda_r: float = 0.05,
        lambda_e: float = 0.05,
        lambda_g: float = 0.01,
        tau_r: float = 0.1,
        tau_e: float = 0.1,
        tau_t: float = 0.1,
        delta_r: float = 0.5,
        delta_e: float = 0.5,
        symmetric_gw: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            beta=beta,
            learn_beta=learn_beta,
            beta_min=beta_min,
            beta_max=beta_max,
            *args,
            **kwargs,
        )

        if lambda_r < 0 or lambda_e < 0 or lambda_g < 0:
            raise ValueError("FGW regularizer weights must be non-negative")
        if tau_r <= 0 or tau_e <= 0 or tau_t <= 0:
            raise ValueError("FGW temperatures must be positive")

        self.lambda_r = float(lambda_r)
        self.lambda_e = float(lambda_e)
        self.lambda_g = float(lambda_g)
        self.tau_r = float(tau_r)
        self.tau_e = float(tau_e)
        self.tau_t = float(tau_t)
        self.delta_r = float(delta_r)
        self.delta_e = float(delta_e)
        self.symmetric_gw = bool(symmetric_gw)

    def forward(
        self,
        dists: torch.Tensor,
        query_idx: torch.Tensor,
        target_idx: torch.Tensor,
        query_embeds: Optional[torch.Tensor] = None,
        target_embeds: Optional[torch.Tensor] = None,
        reaction_similarity: Optional[torch.Tensor] = None,
        enzyme_similarity: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute MLNCE plus optional RR, EE, and GW regularizers.

        Args:
            dists: Pairwise reaction-enzyme distances, shape (num_reactions, num_enzymes).
            query_idx: Reaction indices for positive pairs.
            target_idx: Enzyme indices for positive pairs.
            query_embeds: Learned normalized reaction embeddings, shape (num_reactions, dim).
            target_embeds: Learned normalized enzyme embeddings, shape (num_enzymes, dim).
            reaction_similarity: External reaction similarity matrix, shape (num_reactions, num_reactions).
            enzyme_similarity: External enzyme similarity matrix, shape (num_enzymes, num_enzymes).
            return_components: If True, return the total loss and a dict of components.

        Returns:
            Scalar total loss, or (loss, components) when return_components=True.
        """
        mlnce = super().forward(dists, query_idx, target_idx)
        zero = dists.new_zeros(())

        rr = zero
        if self.lambda_r > 0 and query_embeds is not None and reaction_similarity is not None:
            rr = self._supervised_contrastive_structure_loss(
                query_embeds,
                reaction_similarity,
                threshold=self.delta_r,
                temperature=self.tau_r,
                name="reaction_similarity",
            )

        ee = zero
        if self.lambda_e > 0 and target_embeds is not None and enzyme_similarity is not None:
            ee = self._supervised_contrastive_structure_loss(
                target_embeds,
                enzyme_similarity,
                threshold=self.delta_e,
                temperature=self.tau_e,
                name="enzyme_similarity",
            )

        gw = zero
        if self.lambda_g > 0 and query_embeds is not None and target_embeds is not None:
            gw = self._gw_geometry_loss(query_embeds, target_embeds)

        weighted_rr = rr * self.lambda_r
        weighted_ee = ee * self.lambda_e
        weighted_gw = gw * self.lambda_g
        total = mlnce + weighted_rr + weighted_ee + weighted_gw

        if not return_components:
            return total

        components = {
            "mlnce": mlnce,
            "rr": rr,
            "ee": ee,
            "gw": gw,
            "weighted_rr": weighted_rr,
            "weighted_ee": weighted_ee,
            "weighted_gw": weighted_gw,
        }
        return total, components

    def _supervised_contrastive_structure_loss(
        self,
        embeds: torch.Tensor,
        external_similarity: torch.Tensor,
        threshold: float,
        temperature: float,
        name: str,
    ) -> torch.Tensor:
        if embeds.ndim != 2:
            raise ValueError(f"embeds must be rank-2, got shape={tuple(embeds.shape)}")
        num_items = embeds.shape[0]
        if external_similarity.shape != (num_items, num_items):
            raise ValueError(
                f"{name} must have shape {(num_items, num_items)}, "
                f"got {tuple(external_similarity.shape)}"
            )
        if num_items < 2:
            return embeds.new_zeros(())

        external_similarity = external_similarity.to(device=embeds.device)
        eye = torch.eye(num_items, dtype=torch.bool, device=embeds.device)
        positive_mask = (external_similarity > threshold) & ~eye
        positives_per_anchor = positive_mask.sum(dim=1)
        valid_anchors = positives_per_anchor > 0
        if not bool(valid_anchors.any()):
            return embeds.new_zeros(())

        logits = embeds @ embeds.t()
        logits = logits / temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        logits = logits.masked_fill(eye, -torch.inf)

        log_denominator = torch.logsumexp(logits, dim=1, keepdim=True)
        log_prob = logits - log_denominator
        log_prob = log_prob.masked_fill(eye, 0.0)

        positive_mask_float = positive_mask.to(dtype=log_prob.dtype)
        anchor_loss = -(positive_mask_float * log_prob).sum(dim=1)
        anchor_loss = anchor_loss / positives_per_anchor.clamp_min(1).to(dtype=log_prob.dtype)
        return anchor_loss[valid_anchors].mean()

    def _gw_geometry_loss(
        self, query_embeds: torch.Tensor, target_embeds: torch.Tensor
    ) -> torch.Tensor:
        if query_embeds.ndim != 2 or target_embeds.ndim != 2:
            raise ValueError(
                "query_embeds and target_embeds must be rank-2 for GW loss, "
                f"got {tuple(query_embeds.shape)} and {tuple(target_embeds.shape)}"
            )
        if query_embeds.shape[1] != target_embeds.shape[1]:
            raise ValueError(
                "query_embeds and target_embeds must have the same embedding dimension, "
                f"got {query_embeds.shape[1]} and {target_embeds.shape[1]}"
            )
        if query_embeds.shape[0] == 0 or target_embeds.shape[0] == 0:
            return query_embeds.new_zeros(())

        reaction_geometry = query_embeds @ query_embeds.t()
        enzyme_geometry = target_embeds @ target_embeds.t()
        transport = torch.softmax((query_embeds @ target_embeds.t()) / self.tau_t, dim=1)

        reaction_from_enzymes = transport @ enzyme_geometry @ transport.t()
        reaction_loss = torch.mean((reaction_geometry - reaction_from_enzymes) ** 2)

        if not self.symmetric_gw:
            return reaction_loss

        enzyme_from_reactions = transport.t() @ reaction_geometry @ transport
        enzyme_loss = torch.mean((enzyme_geometry - enzyme_from_reactions) ** 2)
        return 0.5 * (reaction_loss + enzyme_loss)


def build_horizyn_loss(
    name: str = "FullBatchMLNCELoss",
    beta: float = 10.0,
    learn_beta: bool = False,
    beta_min: float = -float("inf"),
    beta_max: float = float("inf"),
    **kwargs: Any,
) -> FullBatchNCELoss:
    """Build a Horizyn loss from config-compatible names."""
    normalized_name = name.lower()
    if normalized_name in {"fullbatchmlnceloss", "mlnce"}:
        return FullBatchMLNCELoss(
            beta=beta,
            learn_beta=learn_beta,
            beta_min=beta_min,
            beta_max=beta_max,
        )
    if normalized_name in {
        "degreetemperedfullbatchmlnceloss",
        "degree_tempered_mlnce",
    }:
        return DegreeTemperedFullBatchMLNCELoss(
            beta=beta,
            learn_beta=learn_beta,
            beta_min=beta_min,
            beta_max=beta_max,
            degree_alpha=kwargs.get("degree_alpha", 0.5),
        )
    if normalized_name in {
        "decoupledallpositiveinfonceloss",
        "decoupled_all_positive_infonce",
    }:
        return DecoupledAllPositiveInfoNCELoss(
            beta=beta,
            learn_beta=learn_beta,
            beta_min=beta_min,
            beta_max=beta_max,
            lambda_r2e=kwargs.get("lambda_r2e", 0.5),
            lambda_e2r=kwargs.get("lambda_e2r", 0.5),
            separate_direction_temperatures=kwargs.get("separate_direction_temperatures", False),
            beta_r2e=kwargs.get("beta_r2e", beta),
            beta_e2r=kwargs.get("beta_e2r", beta),
            temperature_regularization_weight=kwargs.get("temperature_regularization_weight", 0.0),
        )
    if normalized_name in {
        "hybridcardinalityretrievalloss",
        "hybrid_cardinality_retrieval",
    }:
        return HybridCardinalityRetrievalLoss(
            beta=beta,
            learn_beta=learn_beta,
            beta_min=beta_min,
            beta_max=beta_max,
            lambda_r2e=kwargs.get("lambda_r2e", 0.5),
            lambda_e2r=kwargs.get("lambda_e2r", 0.5),
            cardinality_weight=kwargs.get("cardinality_weight", 0.3),
            cardinality_warmup_epochs=kwargs.get("cardinality_warmup_epochs", 5),
            separate_direction_temperatures=kwargs.get("separate_direction_temperatures", False),
            beta_r2e=kwargs.get("beta_r2e", beta),
            beta_e2r=kwargs.get("beta_e2r", beta),
            temperature_regularization_weight=kwargs.get("temperature_regularization_weight", 0.0),
            soft_rank_weight=kwargs.get("soft_rank_weight", 0.0),
            soft_rank_tau=kwargs.get("soft_rank_tau", 0.1),
            soft_rank_top_k=kwargs.get("soft_rank_top_k", 128),
        )
    if normalized_name in {"balancedsigmoidebmloss", "balanced_sigmoid_ebm"}:
        return BalancedSigmoidEBMLoss(
            beta=beta,
            learn_beta=learn_beta,
            beta_min=beta_min,
            beta_max=beta_max,
            sigmoid_bias_init=kwargs.get("sigmoid_bias_init", 0.0),
            sigmoid_learn_bias=kwargs.get("sigmoid_learn_bias", True),
            sigmoid_negative_weight=kwargs.get("sigmoid_negative_weight", 1.0),
        )
    if normalized_name in {"horizynfgwloss", "fgw", "horizyn_fgw"}:
        return HorizynFGWLoss(
            beta=beta,
            learn_beta=learn_beta,
            beta_min=beta_min,
            beta_max=beta_max,
            lambda_r=kwargs.get("lambda_r", 0.05),
            lambda_e=kwargs.get("lambda_e", 0.05),
            lambda_g=kwargs.get("lambda_g", 0.01),
            tau_r=kwargs.get("tau_r", 0.1),
            tau_e=kwargs.get("tau_e", 0.1),
            tau_t=kwargs.get("tau_t", 0.1),
            delta_r=kwargs.get("delta_r", 0.5),
            delta_e=kwargs.get("delta_e", 0.5),
            symmetric_gw=kwargs.get("symmetric_gw", True),
        )
    if normalized_name in {
        "bidirectionalanchorbalancedsupconloss",
        "bidirectional_anchor_balanced_supcon",
        "anchor_balanced_supcon",
    }:
        return BidirectionalAnchorBalancedSupConLoss(
            beta=beta,
            learn_beta=learn_beta,
            beta_min=beta_min,
            beta_max=beta_max,
            direction_balance_weight=kwargs.get("direction_balance_weight", 0.0),
            lambda_r2e=kwargs.get("lambda_r2e", 0.5),
            lambda_e2r=kwargs.get("lambda_e2r", 0.5),
        )
    if normalized_name in {
        "multialignmentretrievalloss",
        "multi_alignment_retrieval",
        "multi_alignment",
    }:
        return MultiAlignmentRetrievalLoss(
            beta=beta,
            learn_beta=learn_beta,
            beta_min=beta_min,
            beta_max=beta_max,
            lambda_rr=kwargs.get("lambda_rr", 0.0),
            lambda_ee=kwargs.get("lambda_ee", 0.0),
            lambda_gw=kwargs.get("lambda_gw", 0.0),
            lambda_direction_gap=kwargs.get(
                "lambda_direction_gap",
                kwargs.get("direction_balance_weight", 0.0),
            ),
            lambda_r2e=kwargs.get("lambda_r2e", 0.5),
            lambda_e2r=kwargs.get("lambda_e2r", 0.5),
            lambda_r2e_hard_neg=kwargs.get("lambda_r2e_hard_neg", 0.0),
            r2e_hard_neg_top_k=kwargs.get("r2e_hard_neg_top_k", 0),
            r2e_hard_neg_margin=kwargs.get("r2e_hard_neg_margin", 0.0),
            lambda_e2r_hard_neg=kwargs.get("lambda_e2r_hard_neg", 0.0),
            e2r_hard_neg_top_k=kwargs.get("e2r_hard_neg_top_k", 0),
            e2r_hard_neg_margin=kwargs.get("e2r_hard_neg_margin", 0.0),
            tau_rr=kwargs.get("tau_rr", kwargs.get("tau_r", 0.1)),
            tau_ee=kwargs.get("tau_ee", kwargs.get("tau_e", 0.1)),
            tau_gw=kwargs.get("tau_gw", kwargs.get("tau_t", 0.1)),
            gw_max_anchors=kwargs.get("gw_max_anchors", 512),
            symmetric_gw=kwargs.get("symmetric_gw", True),
        )
    raise ValueError(f"Unsupported loss name: {name}")
