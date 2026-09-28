"""
Unit tests for loss functions.
"""

import pytest
import torch
import torch.nn.functional as F

from horizyn.losses import (
    BalancedSigmoidEBMLoss,
    BidirectionalAnchorBalancedSupConLoss,
    DecoupledAllPositiveInfoNCELoss,
    DegreeTemperedFullBatchMLNCELoss,
    FullBatchMLNCELoss,
    FullBatchNCELoss,
    HorizynFGWLoss,
    HybridCardinalityRetrievalLoss,
    MultiAlignmentRetrievalLoss,
    SampledMultiPositiveInfoNCELoss,
    build_horizyn_loss,
)


class TestFullBatchNCELoss:
    """Tests for the FullBatchNCELoss base class."""

    def test_initialization_default(self):
        """Test default initialization."""
        loss_fn = FullBatchNCELoss()
        assert loss_fn.beta_init == 1.0
        assert loss_fn.learn_beta is False
        assert loss_fn.beta_min == -float("inf")
        assert loss_fn.beta_max == float("inf")
        assert torch.allclose(loss_fn.beta, torch.tensor(1.0))


class TestSampledMultiPositiveInfoNCELoss:
    def test_uses_all_positives_and_only_explicit_negatives(self):
        loss_fn = SampledMultiPositiveInfoNCELoss(beta=2.0)
        query_idx = torch.tensor([0, 0, 1], dtype=torch.long)
        target_idx = torch.tensor([0, 1, 2], dtype=torch.long)
        biological = torch.tensor([[False, False, True, False], [True, False, False, False]])
        random_negative = torch.tensor([[False, False, False, True], [False, True, False, False]])
        dists = torch.tensor([[0.1, 0.2, 0.7, 0.8], [0.9, 0.8, 0.1, 0.0]], requires_grad=True)
        baseline = loss_fn(
            dists,
            query_idx,
            target_idx,
            biological,
            random_negative,
        )
        changed_unlabelled = dists.detach().clone()
        changed_unlabelled[1, 3] = -100.0
        assert torch.allclose(
            baseline,
            loss_fn(
                changed_unlabelled,
                query_idx,
                target_idx,
                biological,
                random_negative,
            ),
        )
        harder_negative = dists.detach().clone()
        harder_negative[0, 2] = 0.0
        assert (
            loss_fn(
                harder_negative,
                query_idx,
                target_idx,
                biological,
                random_negative,
            )
            > baseline
        )
        baseline.backward()
        assert torch.isfinite(dists.grad).all()

    def test_rejects_positive_negative_overlap(self):
        loss_fn = SampledMultiPositiveInfoNCELoss()
        dists = torch.ones(1, 2)
        with pytest.raises(ValueError, match="must not overlap"):
            loss_fn(
                dists,
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([[True, False]]),
                torch.zeros(1, 2, dtype=torch.bool),
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
    def test_rejects_negative_masks_on_a_different_device(self):
        loss_fn = SampledMultiPositiveInfoNCELoss()
        with pytest.raises(ValueError, match="same device"):
            loss_fn(
                torch.ones(1, 2, device="cuda"),
                torch.tensor([0], device="cuda"),
                torch.tensor([0], device="cuda"),
                torch.tensor([[False, True]]),
                torch.zeros(1, 2, dtype=torch.bool),
            )

    def test_factory_alias(self):
        assert isinstance(
            build_horizyn_loss("sampled_multi_positive_infonce"),
            SampledMultiPositiveInfoNCELoss,
        )


class TestFullBatchNCELossContinued:
    def test_initialization_custom_beta(self):
        """Test initialization with custom beta value."""
        loss_fn = FullBatchNCELoss(beta=10.0)
        assert loss_fn.beta_init == 10.0
        assert torch.allclose(loss_fn.beta, torch.tensor(10.0), rtol=1e-5)

    def test_beta_stored_in_log_space(self):
        """Test that beta is stored in log space."""
        loss_fn = FullBatchNCELoss(beta=5.0)
        expected_logbeta = torch.log(torch.tensor(5.0))
        assert torch.allclose(loss_fn.logbeta, expected_logbeta)

    def test_beta_property(self):
        """Test beta property converts from log space."""
        loss_fn = FullBatchNCELoss(beta=3.0)
        assert torch.allclose(loss_fn.beta, torch.tensor(3.0))

    def test_learn_beta_false(self):
        """Test that beta is not learnable when learn_beta is False."""
        loss_fn = FullBatchNCELoss(beta=2.0, learn_beta=False)
        assert loss_fn.logbeta.requires_grad is False

    def test_learn_beta_true(self):
        """Test that beta is learnable when learn_beta is True."""
        loss_fn = FullBatchNCELoss(beta=2.0, learn_beta=True)
        assert loss_fn.logbeta.requires_grad is True

    def test_beta_clipping_hook_registered(self):
        """Test that beta clipping hook is registered when learn_beta is True."""
        loss_fn = FullBatchNCELoss(beta=5.0, learn_beta=True, beta_min=1.0, beta_max=10.0)
        # Check that a forward pre-hook is registered
        assert len(loss_fn._forward_pre_hooks) > 0

    def test_beta_clipping_hook_not_registered(self):
        """Test that beta clipping hook is not registered when learn_beta is False."""
        loss_fn = FullBatchNCELoss(beta=5.0, learn_beta=False, beta_min=1.0, beta_max=10.0)
        assert len(loss_fn._forward_pre_hooks) == 0

    def test_beta_clipping_min(self):
        """Test that beta is clipped to minimum value."""
        loss_fn = FullBatchNCELoss(beta=5.0, learn_beta=True, beta_min=2.0, beta_max=10.0)
        # Manually set beta below minimum
        loss_fn.logbeta.data = torch.log(torch.tensor(0.5))

        # Trigger the clipping hook via a forward pass
        dists = torch.randn(2, 2)
        query_idx = torch.tensor([0])
        target_idx = torch.tensor([0])

        try:
            loss_fn(dists, query_idx, target_idx)
        except NotImplementedError:
            pass  # Expected since forward is not implemented

        # Beta should be clipped to beta_min
        assert torch.allclose(loss_fn.beta, torch.tensor(2.0), atol=1e-5)

    def test_beta_clipping_max(self):
        """Test that beta is clipped to maximum value."""
        loss_fn = FullBatchNCELoss(beta=5.0, learn_beta=True, beta_min=1.0, beta_max=8.0)
        # Manually set beta above maximum
        loss_fn.logbeta.data = torch.log(torch.tensor(20.0))

        # Trigger the clipping hook via a forward pass
        dists = torch.randn(2, 2)
        query_idx = torch.tensor([0])
        target_idx = torch.tensor([0])

        try:
            loss_fn(dists, query_idx, target_idx)
        except NotImplementedError:
            pass  # Expected since forward is not implemented

        # Beta should be clipped to beta_max
        assert torch.allclose(loss_fn.beta, torch.tensor(8.0), atol=1e-5)

    def test_forward_not_implemented(self):
        """Test that forward raises NotImplementedError."""
        loss_fn = FullBatchNCELoss()
        dists = torch.randn(2, 2)
        query_idx = torch.tensor([0])
        target_idx = torch.tensor([0])

        with pytest.raises(NotImplementedError):
            loss_fn(dists, query_idx, target_idx)


class TestFullBatchMLNCELoss:
    """Tests for the FullBatchMLNCELoss class."""

    def test_initialization(self):
        """Test basic initialization."""
        loss_fn = FullBatchMLNCELoss(beta=10.0, learn_beta=False)
        assert torch.allclose(loss_fn.beta, torch.tensor(10.0))
        assert loss_fn.learn_beta is False

    def test_forward_single_pair(self):
        """Test forward pass with a single positive pair."""
        loss_fn = FullBatchMLNCELoss(beta=1.0, learn_beta=False)

        # Simple 2x2 distance matrix
        dists = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
        query_idx = torch.tensor([0])
        target_idx = torch.tensor([0])

        loss = loss_fn(dists, query_idx, target_idx)

        # Loss should be a scalar
        assert loss.dim() == 0
        # Loss should be positive (we're minimizing distance + partition)
        assert loss.item() > 0

    def test_forward_multiple_pairs(self):
        """Test forward pass with multiple positive pairs."""
        loss_fn = FullBatchMLNCELoss(beta=1.0, learn_beta=False)

        # 3x3 distance matrix
        dists = torch.tensor([[0.1, 0.9, 0.9], [0.9, 0.1, 0.9], [0.9, 0.9, 0.1]])
        query_idx = torch.tensor([0, 1, 2])
        target_idx = torch.tensor([0, 1, 2])

        loss = loss_fn(dists, query_idx, target_idx)

        assert loss.dim() == 0
        assert loss.item() > 0

    def test_forward_multi_label(self):
        """Test forward pass with multiple targets per query (multi-label)."""
        loss_fn = FullBatchMLNCELoss(beta=1.0, learn_beta=False)

        # 2x3 distance matrix
        dists = torch.tensor([[0.1, 0.2, 0.9], [0.9, 0.9, 0.1]])
        # Query 0 has two positive targets (0 and 1)
        query_idx = torch.tensor([0, 0, 1])
        target_idx = torch.tensor([0, 1, 2])

        loss = loss_fn(dists, query_idx, target_idx)

        assert loss.dim() == 0
        assert loss.item() > 0

    def test_forward_shape_consistency(self):
        """Test that forward works with different batch sizes."""
        loss_fn = FullBatchMLNCELoss(beta=1.0)

        # Test with various sizes
        for num_queries, num_targets, num_pairs in [(5, 10, 5), (10, 5, 10), (20, 20, 30)]:
            dists = torch.randn(num_queries, num_targets).abs()
            query_idx = torch.randint(0, num_queries, (num_pairs,))
            target_idx = torch.randint(0, num_targets, (num_pairs,))

            loss = loss_fn(dists, query_idx, target_idx)
            assert loss.dim() == 0

    def test_invalid_dists_rank_raises(self):
        """dists must be 2D."""
        loss_fn = FullBatchMLNCELoss(beta=1.0)
        dists = torch.randn(3, 4, 5)
        qi = torch.tensor([0], dtype=torch.long)
        ti = torch.tensor([0], dtype=torch.long)
        with pytest.raises(ValueError, match="rank-2"):
            loss_fn(dists, qi, ti)

    def test_wrong_index_dtype_raises(self):
        """Indices must be torch.long."""
        loss_fn = FullBatchMLNCELoss(beta=1.0)
        dists = torch.randn(3, 3)
        qi = torch.tensor([0.0])  # float
        ti = torch.tensor([0])
        with pytest.raises(ValueError, match="dtype torch.long"):
            loss_fn(dists, qi, ti)

    def test_empty_indices_raise(self):
        """Empty positive pairs should raise ValueError."""
        loss_fn = FullBatchMLNCELoss(beta=1.0)
        dists = torch.randn(2, 2)
        qi = torch.tensor([], dtype=torch.long)
        ti = torch.tensor([], dtype=torch.long)
        with pytest.raises(ValueError, match="non-empty"):
            loss_fn(dists, qi, ti)

    def test_mismatched_index_lengths_raise(self):
        """Lengths of query_idx and target_idx must match."""
        loss_fn = FullBatchMLNCELoss(beta=1.0)
        dists = torch.randn(3, 3)
        qi = torch.tensor([0, 1], dtype=torch.long)
        ti = torch.tensor([0], dtype=torch.long)
        with pytest.raises(ValueError, match="same length"):
            loss_fn(dists, qi, ti)

    def test_negative_indices_raise(self):
        """Negative indices should raise ValueError."""
        loss_fn = FullBatchMLNCELoss(beta=1.0)
        dists = torch.randn(3, 3)
        qi = torch.tensor([-1], dtype=torch.long)
        ti = torch.tensor([0], dtype=torch.long)
        with pytest.raises(ValueError, match="non-negative"):
            loss_fn(dists, qi, ti)

    def test_out_of_bounds_indices_raise(self):
        """Out-of-range indices should raise ValueError."""
        loss_fn = FullBatchMLNCELoss(beta=1.0)
        dists = torch.randn(2, 2)
        qi = torch.tensor([1, 2], dtype=torch.long)
        ti = torch.tensor([0, 1], dtype=torch.long)
        with pytest.raises(ValueError, match="out of range"):
            loss_fn(dists, qi, ti)

    def test_non_finite_dists_raise(self):
        """NaN/Inf in dists should raise ValueError."""
        loss_fn = FullBatchMLNCELoss(beta=1.0)
        dists = torch.tensor([[0.1, float("inf")], [float("nan"), 0.2]])
        qi = torch.tensor([0], dtype=torch.long)
        ti = torch.tensor([0], dtype=torch.long)
        with pytest.raises(ValueError, match="finite"):
            loss_fn(dists, qi, ti)

    def test_gradient_flow_through_dists(self):
        """Test that gradients flow through the distance matrix."""
        loss_fn = FullBatchMLNCELoss(beta=1.0, learn_beta=False)

        dists = torch.tensor([[0.1, 0.9], [0.9, 0.1]], requires_grad=True)
        query_idx = torch.tensor([0, 1])
        target_idx = torch.tensor([0, 1])

        loss = loss_fn(dists, query_idx, target_idx)
        loss.backward()

        assert dists.grad is not None
        assert not torch.allclose(dists.grad, torch.zeros_like(dists))

    def test_gradient_flow_through_beta(self):
        """Test that gradients flow through beta when learnable."""
        loss_fn = FullBatchMLNCELoss(beta=5.0, learn_beta=True)

        dists = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
        query_idx = torch.tensor([0, 1])
        target_idx = torch.tensor([0, 1])

        loss = loss_fn(dists, query_idx, target_idx)
        loss.backward()

        assert loss_fn.logbeta.grad is not None
        assert not torch.allclose(loss_fn.logbeta.grad, torch.tensor(0.0))

    def test_beta_effect_on_loss(self):
        """Test that beta affects the loss (not necessarily monotonically)."""
        dists = torch.tensor([[0.2, 0.8], [0.8, 0.2]])
        query_idx = torch.tensor([0, 1])
        target_idx = torch.tensor([0, 1])

        # Different beta values should produce different losses
        loss_fn_low = FullBatchMLNCELoss(beta=1.0)
        loss_fn_high = FullBatchMLNCELoss(beta=10.0)

        loss_low = loss_fn_low(dists, query_idx, target_idx)
        loss_high = loss_fn_high(dists, query_idx, target_idx)

        # Both should be positive
        assert loss_low.item() > 0
        assert loss_high.item() > 0
        # Losses should be different (beta changes the scale)
        assert not torch.allclose(loss_low, loss_high, rtol=0.1)

    def test_learned_beta_updates(self):
        """Test that learned beta can be updated via gradient descent."""
        loss_fn = FullBatchMLNCELoss(beta=1.0, learn_beta=True)
        optimizer = torch.optim.SGD(loss_fn.parameters(), lr=0.1)

        initial_beta = loss_fn.beta.item()

        # Perform a few optimization steps
        for _ in range(5):
            dists = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
            query_idx = torch.tensor([0, 1])
            target_idx = torch.tensor([0, 1])

            optimizer.zero_grad()
            loss = loss_fn(dists, query_idx, target_idx)
            loss.backward()
            optimizer.step()

        final_beta = loss_fn.beta.item()

        # Beta should have changed
        assert abs(final_beta - initial_beta) > 1e-3

    def test_beta_clipping_during_training(self):
        """Test that beta is clipped during training when constraints are set."""
        loss_fn = FullBatchMLNCELoss(beta=5.0, learn_beta=True, beta_min=1.0, beta_max=10.0)

        # Manually set beta to violate constraints
        loss_fn.logbeta.data = torch.log(torch.tensor(0.1))  # Below min

        dists = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
        query_idx = torch.tensor([0, 1])
        target_idx = torch.tensor([0, 1])

        loss_fn(dists, query_idx, target_idx)

        # Beta should be clipped to minimum
        assert loss_fn.beta.item() >= 1.0 - 1e-5

    def test_device_placement(self, device):
        """Test that loss function works on both CPU and GPU."""
        loss_fn = FullBatchMLNCELoss(beta=1.0).to(device)

        dists = torch.tensor([[0.1, 0.9], [0.9, 0.1]]).to(device)
        query_idx = torch.tensor([0, 1]).to(device)
        target_idx = torch.tensor([0, 1]).to(device)

        loss = loss_fn(dists, query_idx, target_idx)

        assert loss.device.type == device.type

    def test_loss_decreases_with_better_distances(self):
        """Test that loss is lower when positive pairs have smaller distances."""
        loss_fn = FullBatchMLNCELoss(beta=10.0)

        query_idx = torch.tensor([0, 1])
        target_idx = torch.tensor([0, 1])

        # Good distances: positive pairs are close
        dists_good = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
        loss_good = loss_fn(dists_good, query_idx, target_idx)

        # Bad distances: positive pairs are far
        dists_bad = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        loss_bad = loss_fn(dists_bad, query_idx, target_idx)

        # Loss should be lower for good distances
        assert loss_good.item() < loss_bad.item()

    def test_all_pairs_positive(self):
        """Test edge case where all pairs are positive."""
        loss_fn = FullBatchMLNCELoss(beta=1.0)

        dists = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        # All 4 pairs are positive
        query_idx = torch.tensor([0, 0, 1, 1])
        target_idx = torch.tensor([0, 1, 0, 1])

        loss = loss_fn(dists, query_idx, target_idx)

        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_numerical_stability_extreme_distances(self):
        """Test numerical stability with very large and small distances."""
        loss_fn = FullBatchMLNCELoss(beta=100.0)

        # Mix of very small and large distances
        dists = torch.tensor([[1e-6, 10.0], [10.0, 1e-6]])
        query_idx = torch.tensor([0, 1])
        target_idx = torch.tensor([0, 1])

        loss = loss_fn(dists, query_idx, target_idx)

        # Should not be NaN or inf
        assert torch.isfinite(loss)

    def test_consistency_with_cosine_distance(self):
        """Test that loss works with cosine distance (1 - cosine_similarity)."""
        loss_fn = FullBatchMLNCELoss(beta=10.0)

        # Normalized embeddings
        query_embeds = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
        target_embeds = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)

        # Cosine distance
        cosine_sim = torch.mm(query_embeds, target_embeds.t())
        dists = 1.0 - cosine_sim

        query_idx = torch.tensor([0, 1, 2])
        target_idx = torch.tensor([0, 1, 2])

        loss = loss_fn(dists, query_idx, target_idx)

        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_sota_configuration(self):
        """Test with SOTA paper configuration."""
        # From sota.yaml
        loss_fn = FullBatchMLNCELoss(
            beta=10.0,
            learn_beta=False,
            beta_min=0.01,
            beta_max=100.0,
        )

        # Simulate a batch with query and target encoders
        batch_size = 16
        embed_dim = 512

        # Mock embeddings (normalized)
        query_embeds = torch.nn.functional.normalize(torch.randn(batch_size, embed_dim), dim=1)
        target_embeds = torch.nn.functional.normalize(torch.randn(batch_size, embed_dim), dim=1)

        # Cosine distance
        dists = 1.0 - torch.mm(query_embeds, target_embeds.t())

        # Assume diagonal pairs are positive (simple case)
        query_idx = torch.arange(batch_size)
        target_idx = torch.arange(batch_size)

        loss = loss_fn(dists, query_idx, target_idx)

        assert torch.isfinite(loss)
        assert loss.item() > 0
        # Verify beta is as configured
        assert torch.allclose(loss_fn.beta, torch.tensor(10.0))


class TestBidirectionalAnchorBalancedSupConLoss:
    """Tests for bidirectional anchor-balanced supervised contrastive loss."""

    def test_single_positive_matches_bidirectional_cross_entropy(self):
        dists = torch.tensor(
            [
                [0.1, 0.8, 0.7],
                [0.9, 0.2, 0.6],
                [0.7, 0.6, 0.3],
            ]
        )
        query_idx = torch.tensor([0, 1, 2], dtype=torch.long)
        target_idx = torch.tensor([0, 1, 2], dtype=torch.long)
        loss_fn = BidirectionalAnchorBalancedSupConLoss(beta=2.0)

        loss, components = loss_fn(dists, query_idx, target_idx, return_components=True)
        logits = -2.0 * dists
        expected_r2e = F.cross_entropy(logits, target_idx)
        expected_e2r = F.cross_entropy(logits.t(), query_idx)
        expected = 0.5 * (expected_r2e + expected_e2r)

        assert torch.allclose(loss, expected)
        assert torch.allclose(components["r2e"], expected_r2e)
        assert torch.allclose(components["e2r"], expected_e2r)
        assert torch.allclose(
            components["direction_gap"],
            (expected_r2e - expected_e2r).abs(),
            atol=1e-6,
        )

    def test_multi_positive_averages_per_anchor(self):
        dists = torch.tensor(
            [
                [0.1, 0.2, 1.0],
                [0.8, 0.9, 0.1],
            ]
        )
        query_idx = torch.tensor([0, 0, 1], dtype=torch.long)
        target_idx = torch.tensor([0, 1, 2], dtype=torch.long)
        loss_fn = BidirectionalAnchorBalancedSupConLoss(beta=1.0)

        loss, components = loss_fn(dists, query_idx, target_idx, return_components=True)
        logits = -dists

        r2e_q0 = torch.logsumexp(logits[0], dim=0) - torch.logsumexp(logits[0, [0, 1]], dim=0)
        r2e_q1 = torch.logsumexp(logits[1], dim=0) - logits[1, 2]
        expected_r2e = torch.stack([r2e_q0, r2e_q1]).mean()

        e2r_t0 = torch.logsumexp(logits[:, 0], dim=0) - logits[0, 0]
        e2r_t1 = torch.logsumexp(logits[:, 1], dim=0) - logits[0, 1]
        e2r_t2 = torch.logsumexp(logits[:, 2], dim=0) - logits[1, 2]
        expected_e2r = torch.stack([e2r_t0, e2r_t1, e2r_t2]).mean()

        assert torch.allclose(components["r2e"], expected_r2e)
        assert torch.allclose(components["e2r"], expected_e2r)
        assert torch.allclose(loss, 0.5 * (expected_r2e + expected_e2r))

    def test_gradients_are_finite(self):
        raw_q = torch.randn(3, 8, requires_grad=True)
        raw_e = torch.randn(4, 8, requires_grad=True)
        q = F.normalize(raw_q, dim=-1)
        e = F.normalize(raw_e, dim=-1)
        dists = 1.0 - q @ e.t()
        query_idx = torch.tensor([0, 1, 2, 2], dtype=torch.long)
        target_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        loss = BidirectionalAnchorBalancedSupConLoss(beta=5.0)(dists, query_idx, target_idx)
        loss.backward()

        assert torch.isfinite(loss)
        assert raw_q.grad is not None
        assert raw_e.grad is not None
        assert torch.isfinite(raw_q.grad).all()
        assert torch.isfinite(raw_e.grad).all()

    def test_build_horizyn_loss_accepts_alias(self):
        loss_fn = build_horizyn_loss(
            "anchor_balanced_supcon",
            beta=7.0,
            direction_balance_weight=0.1,
        )
        assert isinstance(loss_fn, BidirectionalAnchorBalancedSupConLoss)
        assert torch.allclose(loss_fn.beta, torch.tensor(7.0))
        assert loss_fn.direction_balance_weight == 0.1


class TestHorizynFGWLoss:
    """Tests for the FGW-regularized Horizyn loss."""

    def test_zero_regularizers_match_mlnce(self):
        q = F.normalize(torch.randn(4, 8), dim=-1)
        e = F.normalize(torch.randn(5, 8), dim=-1)
        dists = 1.0 - q @ e.t()
        query_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        target_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        mlnce = FullBatchMLNCELoss(beta=3.0)(dists, query_idx, target_idx)
        fgw = HorizynFGWLoss(
            beta=3.0,
            lambda_r=0.0,
            lambda_e=0.0,
            lambda_g=0.0,
        )(dists, query_idx, target_idx)

        assert torch.allclose(fgw, mlnce)

    def test_structural_components_are_finite_and_differentiable(self):
        raw_q = torch.randn(4, 8, requires_grad=True)
        raw_e = torch.randn(4, 8, requires_grad=True)
        q = F.normalize(raw_q, dim=-1)
        e = F.normalize(raw_e, dim=-1)
        dists = 1.0 - q @ e.t()
        query_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        target_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        reaction_similarity = torch.eye(4)
        reaction_similarity[0, 1] = reaction_similarity[1, 0] = 0.9
        enzyme_similarity = torch.eye(4)
        enzyme_similarity[2, 3] = enzyme_similarity[3, 2] = 0.9

        loss_fn = HorizynFGWLoss(
            beta=2.0,
            lambda_r=0.05,
            lambda_e=0.05,
            lambda_g=0.01,
            delta_r=0.5,
            delta_e=0.5,
        )
        loss, components = loss_fn(
            dists,
            query_idx,
            target_idx,
            query_embeds=q,
            target_embeds=e,
            reaction_similarity=reaction_similarity,
            enzyme_similarity=enzyme_similarity,
            return_components=True,
        )

        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert set(components) == {
            "mlnce",
            "rr",
            "ee",
            "gw",
            "weighted_rr",
            "weighted_ee",
            "weighted_gw",
        }
        assert components["rr"] > 0
        assert components["ee"] > 0
        assert components["gw"] >= 0

        loss.backward()
        assert raw_q.grad is not None
        assert raw_e.grad is not None
        assert torch.isfinite(raw_q.grad).all()
        assert torch.isfinite(raw_e.grad).all()

    def test_structure_terms_skip_anchors_without_positives(self):
        q = F.normalize(torch.randn(3, 8), dim=-1)
        e = F.normalize(torch.randn(3, 8), dim=-1)
        dists = 1.0 - q @ e.t()
        query_idx = torch.tensor([0, 1, 2], dtype=torch.long)
        target_idx = torch.tensor([0, 1, 2], dtype=torch.long)

        loss_fn = HorizynFGWLoss(
            lambda_r=1.0,
            lambda_e=1.0,
            lambda_g=0.0,
            delta_r=0.5,
            delta_e=0.5,
        )
        _, components = loss_fn(
            dists,
            query_idx,
            target_idx,
            query_embeds=q,
            target_embeds=e,
            reaction_similarity=torch.eye(3),
            enzyme_similarity=torch.eye(3),
            return_components=True,
        )

        assert torch.allclose(components["rr"], torch.tensor(0.0))
        assert torch.allclose(components["ee"], torch.tensor(0.0))

    def test_build_horizyn_loss_accepts_fgw_alias(self):
        loss_fn = build_horizyn_loss("horizyn_fgw", beta=5.0, lambda_g=0.02)
        assert isinstance(loss_fn, HorizynFGWLoss)
        assert loss_fn.lambda_g == 0.02


class TestMultiAlignmentRetrievalLoss:
    def test_anchor_balanced_loss_logs_components_and_backprops(self):
        raw_q = torch.randn(3, 8, requires_grad=True)
        raw_e = torch.randn(3, 8, requires_grad=True)
        q = F.normalize(raw_q, dim=-1)
        e = F.normalize(raw_e, dim=-1)
        dists = 1.0 - q @ e.t()
        query_idx = torch.tensor([0, 0, 1, 2], dtype=torch.long)
        target_idx = torch.tensor([0, 1, 1, 2], dtype=torch.long)

        loss_fn = MultiAlignmentRetrievalLoss(
            beta=2.0,
            lambda_direction_gap=0.1,
            lambda_rr=0.0,
            lambda_ee=0.0,
            lambda_gw=0.0,
        )
        loss, components = loss_fn(
            dists,
            query_idx,
            target_idx,
            query_embeds=q,
            target_embeds=e,
            return_components=True,
        )

        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert components["r2e"] > 0
        assert components["e2r"] > 0
        assert components["cross"] > 0
        loss.backward()
        assert raw_q.grad is not None
        assert raw_e.grad is not None
        assert torch.isfinite(raw_q.grad).all()
        assert torch.isfinite(raw_e.grad).all()

    def test_weighted_rr_ee_terms_return_zero_without_valid_positives(self):
        q = F.normalize(torch.randn(3, 8), dim=-1)
        e = F.normalize(torch.randn(3, 8), dim=-1)
        dists = 1.0 - q @ e.t()
        query_idx = torch.tensor([0, 1, 2], dtype=torch.long)
        target_idx = torch.tensor([0, 1, 2], dtype=torch.long)

        loss_fn = MultiAlignmentRetrievalLoss(
            lambda_rr=1.0,
            lambda_ee=1.0,
            lambda_gw=0.0,
        )
        _, components = loss_fn(
            dists,
            query_idx,
            target_idx,
            query_embeds=q,
            target_embeds=e,
            reaction_ec_weights=torch.eye(3),
            enzyme_ec_weights=torch.eye(3),
            return_components=True,
        )

        assert torch.allclose(components["rr"], torch.tensor(0.0))
        assert torch.allclose(components["ee"], torch.tensor(0.0))

    def test_r2e_weights_and_hard_negatives_are_configurable(self):
        dists = torch.tensor(
            [
                [0.10, 0.12, 0.50],
                [0.55, 0.20, 0.21],
            ],
            dtype=torch.float32,
        )
        query_idx = torch.tensor([0, 1], dtype=torch.long)
        target_idx = torch.tensor([0, 1], dtype=torch.long)

        balanced = MultiAlignmentRetrievalLoss(beta=2.0)
        focused = MultiAlignmentRetrievalLoss(
            beta=2.0,
            lambda_r2e=0.8,
            lambda_e2r=0.2,
            lambda_r2e_hard_neg=0.5,
            r2e_hard_neg_top_k=1,
        )
        balanced_loss, balanced_components = balanced(
            dists,
            query_idx,
            target_idx,
            return_components=True,
        )
        focused_loss, focused_components = focused(
            dists,
            query_idx,
            target_idx,
            return_components=True,
        )

        assert torch.allclose(focused_components["r2e_weight"], torch.tensor(0.8))
        assert torch.allclose(focused_components["e2r_weight"], torch.tensor(0.2))
        assert focused_components["r2e_hard_neg"] > 0
        assert focused_components["weighted_r2e_hard_neg"] > 0
        assert not torch.allclose(focused_loss, balanced_loss)
        assert torch.allclose(balanced_components["weighted_r2e_hard_neg"], torch.tensor(0.0))

    def test_detached_gw_keeps_geometry_matrices_out_of_gradient_path(self):
        raw_q = torch.randn(4, 8, requires_grad=True)
        raw_e = torch.randn(4, 8, requires_grad=True)
        q = F.normalize(raw_q, dim=-1)
        e = F.normalize(raw_e, dim=-1)
        dists = 1.0 - q @ e.t()
        query_idx = torch.arange(4, dtype=torch.long)
        target_idx = torch.arange(4, dtype=torch.long)

        loss_fn = MultiAlignmentRetrievalLoss(
            lambda_gw=1.0,
            gw_max_anchors=4,
            tau_gw=0.5,
        )
        loss, components = loss_fn(
            dists,
            query_idx,
            target_idx,
            query_embeds=q,
            target_embeds=e,
            return_components=True,
        )

        assert components["gw"] >= 0
        loss.backward()
        assert raw_q.grad is not None
        assert raw_e.grad is not None
        assert torch.isfinite(raw_q.grad).all()
        assert torch.isfinite(raw_e.grad).all()

    def test_build_horizyn_loss_accepts_multi_alignment_alias(self):
        loss_fn = build_horizyn_loss(
            "multi_alignment_retrieval",
            beta=3.0,
            lambda_rr=0.02,
            lambda_ee=0.03,
            lambda_gw=0.004,
        )

        assert isinstance(loss_fn, MultiAlignmentRetrievalLoss)
        assert loss_fn.lambda_rr == 0.02
        assert loss_fn.lambda_ee == 0.03
        assert loss_fn.lambda_gw == 0.004

    def test_build_horizyn_loss_passes_r2e_focus_parameters(self):
        loss_fn = build_horizyn_loss(
            "multi_alignment_retrieval",
            lambda_r2e=0.85,
            lambda_e2r=0.15,
            lambda_r2e_hard_neg=0.25,
            r2e_hard_neg_top_k=32,
            r2e_hard_neg_margin=0.1,
        )

        assert isinstance(loss_fn, MultiAlignmentRetrievalLoss)
        assert loss_fn.lambda_r2e == pytest.approx(0.85)
        assert loss_fn.lambda_e2r == pytest.approx(0.15)
        assert loss_fn.lambda_r2e_hard_neg == 0.25
        assert loss_fn.r2e_hard_neg_top_k == 32
        assert loss_fn.r2e_hard_neg_margin == 0.1


def test_anchor_balanced_direction_weights_change_only_direction_mix():
    q = F.normalize(torch.randn(3, 8), dim=-1)
    e = F.normalize(torch.randn(4, 8), dim=-1)
    dists = 1.0 - q @ e.t()
    query_idx = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    target_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    loss_fn = BidirectionalAnchorBalancedSupConLoss(
        beta=2.0,
        lambda_r2e=0.4,
        lambda_e2r=0.6,
    )

    loss, components = loss_fn(
        dists,
        query_idx,
        target_idx,
        return_components=True,
    )

    expected = 0.4 * components["r2e"] + 0.6 * components["e2r"]
    assert torch.allclose(loss, expected)
    assert components["r2e_weight"].item() == pytest.approx(0.4)
    assert components["e2r_weight"].item() == pytest.approx(0.6)


class TestF3LossAblations:
    @staticmethod
    def _inputs():
        dists = torch.tensor(
            [[0.1, 0.2, 0.9], [0.8, 0.1, 0.3]],
            dtype=torch.float32,
            requires_grad=True,
        )
        query_idx = torch.tensor([0, 0, 1], dtype=torch.long)
        target_idx = torch.tensor([0, 1, 2], dtype=torch.long)
        return dists, query_idx, target_idx

    def test_degree_alpha_zero_matches_full_batch_mlnce(self):
        dists, query_idx, target_idx = self._inputs()
        baseline = FullBatchMLNCELoss(beta=3.0)(dists, query_idx, target_idx)
        tempered = DegreeTemperedFullBatchMLNCELoss(beta=3.0, degree_alpha=0.0)(
            dists,
            query_idx,
            target_idx,
            query_degrees=torch.tensor([7.0, 1.0]),
        )
        assert torch.allclose(tempered, baseline)

    def test_degree_tempering_reduces_high_degree_pair_mass(self):
        dists, query_idx, target_idx = self._inputs()
        loss_fn = DegreeTemperedFullBatchMLNCELoss(beta=2.0, degree_alpha=0.5)
        loss = loss_fn(
            dists,
            query_idx,
            target_idx,
            query_degrees=torch.tensor([4.0, 1.0]),
        )
        weights = torch.tensor([0.5, 0.5, 1.0])
        weights = weights / weights.sum()
        expected = 2.0 * (weights * dists[query_idx, target_idx]).sum()
        expected = expected + torch.logsumexp(-2.0 * dists, dim=(0, 1))
        assert torch.allclose(loss, expected)

    def test_decoupled_loss_is_finite_and_backpropagates(self):
        dists, query_idx, target_idx = self._inputs()
        loss_fn = DecoupledAllPositiveInfoNCELoss(beta=3.0)
        loss, components = loss_fn(
            dists,
            query_idx,
            target_idx,
            return_components=True,
        )
        assert torch.isfinite(loss)
        assert components["r2e"] >= 0
        assert components["e2r"] >= 0
        loss.backward()
        assert dists.grad is not None
        assert torch.isfinite(dists.grad).all()

    def test_decoupled_loss_collapses_duplicate_positive_indices(self):
        dists, query_idx, target_idx = self._inputs()
        loss_fn = DecoupledAllPositiveInfoNCELoss(beta=3.0)
        original = loss_fn(dists, query_idx, target_idx)
        duplicated = loss_fn(
            dists,
            torch.cat([query_idx, query_idx[:1]]),
            torch.cat([target_idx, target_idx[:1]]),
        )
        assert torch.allclose(original, duplicated)

    def test_hybrid_warmup_starts_at_mlnce_and_reaches_requested_mix(self):
        dists, query_idx, target_idx = self._inputs()
        loss_fn = HybridCardinalityRetrievalLoss(
            beta=3.0,
            cardinality_weight=0.3,
            cardinality_warmup_epochs=5,
        )
        loss_fn.set_training_progress(0)
        epoch_zero, zero_components = loss_fn(dists, query_idx, target_idx, return_components=True)
        assert torch.allclose(epoch_zero, zero_components["mlnce"])
        assert zero_components["effective_cardinality_weight"].item() == 0.0

        loss_fn.set_training_progress(5)
        warmed, warmed_components = loss_fn(dists, query_idx, target_idx, return_components=True)
        expected = 0.7 * warmed_components["mlnce"] + 0.3 * warmed_components["cardinality"]
        assert torch.allclose(warmed, expected)

    def test_separate_direction_temperatures_are_learnable_and_clamped(self):
        dists, query_idx, target_idx = self._inputs()
        loss_fn = HybridCardinalityRetrievalLoss(
            beta=10.0,
            beta_min=3.0,
            beta_max=30.0,
            separate_direction_temperatures=True,
            beta_r2e=10.0,
            beta_e2r=10.0,
            temperature_regularization_weight=1e-3,
            cardinality_warmup_epochs=0,
        )
        loss = loss_fn(dists, query_idx, target_idx)
        loss.backward()
        assert loss_fn.logbeta_r2e.grad is not None
        assert loss_fn.logbeta_e2r.grad is not None
        with torch.no_grad():
            loss_fn.logbeta_r2e.copy_(torch.log(torch.tensor(100.0)))
        _, components = loss_fn(dists.detach(), query_idx, target_idx, return_components=True)
        assert components["beta_r2e"].item() == pytest.approx(30.0)

    def test_soft_rank_penalizes_an_outranking_negative(self):
        positive_mask = torch.tensor([[True, False, False]])
        loss_fn = HybridCardinalityRetrievalLoss(soft_rank_weight=0.05)
        good = loss_fn._soft_rank_anchor_loss(torch.tensor([[0.1, 0.5, 0.7]]), positive_mask)
        bad = loss_fn._soft_rank_anchor_loss(torch.tensor([[0.5, 0.1, 0.7]]), positive_mask)
        assert bad > good

    def test_balanced_sigmoid_uses_equal_class_means_and_learns_bias(self):
        dists, query_idx, target_idx = self._inputs()
        loss_fn = BalancedSigmoidEBMLoss(beta=2.0, sigmoid_bias_init=0.0)
        loss, components = loss_fn(dists, query_idx, target_idx, return_components=True)
        assert torch.allclose(loss, components["positive"] + components["negative"])
        loss.backward()
        assert loss_fn.bias.grad is not None

    @pytest.mark.parametrize(
        ("name", "expected_type"),
        [
            ("degree_tempered_mlnce", DegreeTemperedFullBatchMLNCELoss),
            ("decoupled_all_positive_infonce", DecoupledAllPositiveInfoNCELoss),
            ("hybrid_cardinality_retrieval", HybridCardinalityRetrievalLoss),
            ("balanced_sigmoid_ebm", BalancedSigmoidEBMLoss),
        ],
    )
    def test_factory_supports_ablation_losses(self, name, expected_type):
        assert isinstance(build_horizyn_loss(name), expected_type)
