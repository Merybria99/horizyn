from unittest.mock import patch

import pytest
import torch

from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule


def _small_module(**kwargs) -> ProteinPooledLitModule:
    return ProteinPooledLitModule(
        query_encoder_dims=[4, 4],
        target_encoder_dims=[4, 4],
        embedding_dim=4,
        residue_dim=4,
        pooling="mean",
        validation_retrieval_metrics=kwargs.pop("validation_retrieval_metrics", False),
        **kwargs,
    )


def test_observed_positive_pair_indices_match_batch_pairs():
    module = _small_module()

    query_idx, target_idx = module._positive_pair_indices(
        unique_query_ids=["q1", "q2"],
        unique_target_ids=["t1", "t2"],
        pair_query_ids=["q1", "q1", "q2"],
        pair_target_ids=["t2", "t1", "t2"],
        positive_pair_source="observed_pairs",
    )

    assert query_idx.tolist() == [0, 0, 1]
    assert target_idx.tolist() == [1, 0, 1]


def test_all_known_positive_pair_indices_expand_to_in_batch_known_pairs():
    module = _small_module(positive_pair_source="all_known_in_batch")

    class MockDataModule:
        _train_query_to_targets = {
            "q1": ["t1", "t2", "missing_target"],
            "q2": ["t3"],
            "missing_query": ["t1"],
        }

    class MockTrainer:
        datamodule = MockDataModule()

    module.trainer = MockTrainer()

    query_idx, target_idx = module._positive_pair_indices(
        unique_query_ids=["q1", "q2"],
        unique_target_ids=["t1", "t2", "t3"],
        pair_query_ids=["q1"],
        pair_target_ids=["t1"],
    )

    assert query_idx.tolist() == [0, 0, 1]
    assert target_idx.tolist() == [0, 1, 2]


def test_all_known_positive_pair_indices_raises_when_no_known_pair_in_batch():
    module = _small_module(positive_pair_source="all_known_in_batch")

    class MockDataModule:
        _train_query_to_targets = {"q1": ["missing_target"]}

    class MockTrainer:
        datamodule = MockDataModule()

    module.trainer = MockTrainer()

    with pytest.raises(RuntimeError, match="No positive pairs"):
        module._positive_pair_indices(
            unique_query_ids=["q1"],
            unique_target_ids=["t1"],
            pair_query_ids=["q1"],
            pair_target_ids=["t1"],
        )


def test_typed_negative_masks_preserve_sources_and_leave_other_cells_unlabelled():
    module = _small_module(loss_name="SampledMultiPositiveInfoNCELoss")
    query_idx = torch.tensor([0, 1])
    target_idx = torch.tensor([0, 1])
    biological, random_negative = module._typed_negative_masks(
        unique_query_ids=["q1", "q2"],
        unique_target_ids=["t1", "t2", "t3"],
        pair_query_ids=["q1", "q1", "q2", "q2"],
        pair_target_ids=["t1", "t2", "t2", "t3"],
        pair_types=["positive", "biological_negative", "positive", "random_negative"],
        query_idx=query_idx,
        target_idx=target_idx,
    )

    assert biological.tolist() == [[False, True, False], [False, False, False]]
    assert random_negative.tolist() == [[False, False, False], [False, False, True]]


def test_typed_negative_masks_include_all_verified_pairs_in_global_batch():
    module = _small_module(loss_name="SampledMultiPositiveInfoNCELoss")

    class MockDataModule:
        _train_typed_negative_targets = {
            "q1": {
                "biological": frozenset({"t2", "t3"}),
                "random": frozenset({"t4"}),
            }
        }

    class MockTrainer:
        datamodule = MockDataModule()

    module.trainer = MockTrainer()
    biological, random_negative = module._typed_negative_masks(
        unique_query_ids=["q1", "q2"],
        unique_target_ids=["t1", "t2", "t3", "t4"],
        pair_query_ids=["q1", "q1", "q2"],
        pair_target_ids=["t1", "t2", "t4"],
        pair_types=["positive", "biological_negative", "positive"],
        query_idx=torch.tensor([0, 1]),
        target_idx=torch.tensor([0, 3]),
    )

    assert biological.tolist() == [
        [False, True, True, False],
        [False, False, False, False],
    ]
    assert random_negative.tolist() == [
        [False, False, False, True],
        [False, False, False, False],
    ]


def test_typed_negative_masks_reject_a_known_positive_negative():
    module = _small_module(loss_name="SampledMultiPositiveInfoNCELoss")
    with pytest.raises(RuntimeError, match="known positive"):
        module._typed_negative_masks(
            unique_query_ids=["q1"],
            unique_target_ids=["t1"],
            pair_query_ids=["q1"],
            pair_target_ids=["t1"],
            pair_types=["biological_negative"],
            query_idx=torch.tensor([0]),
            target_idx=torch.tensor([0]),
        )


def test_observed_positive_pair_ids_exclude_typed_negatives():
    query_ids, target_ids = ProteinPooledLitModule._observed_positive_pair_ids(
        ["q1", "q1", "q2"],
        ["p1", "p2", "p3"],
        ["positive", "biological_negative", "random_negative"],
    )

    assert query_ids == ["q1"]
    assert target_ids == ["p1"]


def test_degree_tempered_loss_uses_full_train_graph_degrees():
    module = _small_module(loss_name="DegreeTemperedFullBatchMLNCELoss")

    class MockDataModule:
        _train_query_to_targets = {
            "q1": ["t1", "t2", "t2"],
            "q2": ["t3"],
        }

    class MockTrainer:
        datamodule = MockDataModule()

    module.trainer = MockTrainer()
    degrees = module._full_train_query_degrees(
        ["q1", "q2", "missing"],
        torch.empty(3, 4),
    )

    assert degrees.tolist() == [2.0, 1.0, 1.0]


def test_balanced_validation_metrics_are_harmonic_means():
    module = _small_module(validation_retrieval_metrics=True)
    module._reset_balanced_metric_state()
    module._accumulate_balanced_metric_batch(
        "reaction_to_enzyme",
        "mrr",
        [torch.tensor(0.5), torch.tensor(1.0)],
        torch.device("cpu"),
    )
    module._accumulate_balanced_metric_batch(
        "enzyme_to_reaction",
        "mrr",
        [torch.tensor(0.25), torch.tensor(0.25)],
        torch.device("cpu"),
    )
    module._accumulate_balanced_metric_batch(
        "reaction_to_enzyme",
        "top_1",
        [torch.tensor(1.0), torch.tensor(0.0)],
        torch.device("cpu"),
    )
    module._accumulate_balanced_metric_batch(
        "enzyme_to_reaction",
        "top_1",
        [torch.tensor(1.0), torch.tensor(1.0)],
        torch.device("cpu"),
    )

    with patch.object(module, "log") as log_mock:
        module.on_validation_epoch_end()

    logged = {call.args[0]: call.args[1] for call in log_mock.call_args_list}
    assert torch.isclose(logged["val/balanced_mrr"], torch.tensor(0.375))
    assert torch.isclose(logged["val/balanced_top_1"], torch.tensor(2.0 / 3.0))


def test_validation_reports_first_positive_and_reactzyme_mrr_separately():
    module = _small_module(validation_retrieval_metrics=True)

    num_valid, metrics = module._batched_retrieval_metric_values(
        torch.tensor([[0.9, 0.8, 0.7, 0.6]]),
        anchor_ids=["query"],
        positive_id_lookup={"query": ["target_0", "target_3"]},
        candidate_id_to_idx={
            "target_0": 0,
            "target_1": 1,
            "target_2": 2,
            "target_3": 3,
        },
    )

    assert num_valid == 1
    assert metrics["mrr"].item() == pytest.approx(1.0)
    assert metrics["reactzyme_mrr"].item() == pytest.approx((1.0 + 0.25) / 2.0)


@pytest.mark.parametrize("scores,positives", [
    ([0.5, 0.5, 0.5], [1, 2]),
    ([1.0, 1.0, 0.0, -1.0], [1, 3, 3]),
    ([0.9, 0.8, 0.7, 0.6], [0, 3]),
])
def test_validation_metrics_match_stable_candidate_order(scores, positives):
    from horizyn.benchmarks.chunked_retrieval import chunked_positive_ranks, rank_metrics

    module = _small_module(validation_retrieval_metrics=True)
    module.metric_functionals = {"r_precision": None, "avg_precision": None}
    score_tensor = torch.tensor(scores)
    candidate_ids = [f"target_{i}" for i in range(len(scores))]
    num_valid, metrics = module._batched_retrieval_metric_values(
        score_tensor.unsqueeze(0), ["query"],
        {"query": [candidate_ids[i] for i in positives]},
        {identifier: i for i, identifier in enumerate(candidate_ids)},
    )
    ranked = torch.argsort(score_tensor, descending=True, stable=True).tolist()
    ranks = sorted(ranked.index(i) + 1 for i in set(positives))
    assert num_valid == 1
    assert metrics["mean_rank"].item() == ranks[0]
    assert metrics["mrr"].item() == pytest.approx(1.0 / ranks[0])
    assert metrics["reactzyme_mrr"].item() == pytest.approx(
        sum(1.0 / rank for rank in ranks) / len(ranks)
    )
    assert metrics["avg_precision"].item() == pytest.approx(
        sum(i / rank for i, rank in enumerate(ranks, 1)) / len(ranks)
    )
    assert metrics["avg_precision"].item() <= 1.0
    assert metrics["r_precision"].item() == pytest.approx(
        sum(rank <= len(ranks) for rank in ranks) / len(ranks)
    )
    assert metrics["top_1"].item() == float(ranks[0] == 1)
    chunked_metrics = rank_metrics(
        chunked_positive_ranks(torch.ones(1, 1), score_tensor.unsqueeze(1),
                               [positives], chunk_size=2),
        module.retrieval_metric_top_k,
    )
    for name, value in metrics.items():
        assert torch.allclose(value, chunked_metrics[name]), name


def test_ec_weight_matrix_uses_hierarchical_weights():
    module = _small_module(ec_min_shared_depth=2)

    weights = module._ec_weight_matrix(
        ["rxn1_f", "rxn2_f", "rxn3_f"],
        {
            "rxn1_f": ("1.2.3.4",),
            "rxn2_f": ("1.2.3.9",),
            "rxn3_f": ("1.2.8.9",),
        },
    )

    assert weights.shape == (3, 3)
    assert torch.allclose(torch.diag(weights), torch.zeros(3))
    assert torch.isclose(weights[0, 1], torch.tensor(0.5))
    assert torch.isclose(weights[0, 2], torch.tensor(0.25))


def test_multi_alignment_ec_weight_flags_follow_enabled_terms():
    anchor_only = _small_module(
        loss_name="MultiAlignmentRetrievalLoss",
        lambda_rr=0.0,
        lambda_ee=0.0,
        lambda_gw=0.0,
    )
    assert not anchor_only._needs_reaction_ec_weights()
    assert not anchor_only._needs_enzyme_ec_weights()

    reaction_structure = _small_module(
        loss_name="MultiAlignmentRetrievalLoss",
        lambda_rr=0.02,
        lambda_ee=0.0,
    )
    assert reaction_structure._needs_reaction_ec_weights()
    assert not reaction_structure._needs_enzyme_ec_weights()

    enzyme_structure = _small_module(
        loss_name="MultiAlignmentRetrievalLoss",
        lambda_rr=0.0,
        lambda_ee=0.02,
    )
    assert not enzyme_structure._needs_reaction_ec_weights()
    assert enzyme_structure._needs_enzyme_ec_weights()


def test_reaction_chemistry_consistency_uses_retained_rows_and_detached_target():
    module = _small_module(reaction_chemistry_consistency_weight=0.03)
    query_embeds = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        requires_grad=True,
    )
    chemistry_free = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        requires_grad=True,
    )
    query_inputs = {
        "reaction_chemistry_vector": torch.ones(3, 2),
        "has_reaction_chemistry": torch.ones(3, dtype=torch.bool),
    }
    attention = {
        "modality_names": ("unimol2", "reaction_chemistry"),
        "modality_mask": torch.tensor(
            [[True, True], [True, False], [True, True]],
        ),
    }

    with patch.object(module.model, "encode_queries", return_value=chemistry_free) as encode:
        loss, active = module._reaction_chemistry_consistency_loss(
            query_embeds=query_embeds,
            query_inputs=query_inputs,
            query_attention_details=attention,
            retrieval_direction="reaction_to_enzyme",
        )

    assert loss.item() == pytest.approx(0.5)
    assert active.item() == 2
    assert not encode.call_args.args[0]["has_reaction_chemistry"].any()
    loss.backward()
    assert query_embeds.grad is not None
    assert chemistry_free.grad is None


def test_reaction_chemistry_consistency_is_zero_without_retained_rows():
    module = _small_module(reaction_chemistry_consistency_weight=0.03)
    query_embeds = torch.randn(2, 4, requires_grad=True)
    attention = {
        "modality_names": ("unimol2", "reaction_chemistry"),
        "modality_mask": torch.tensor([[True, False], [True, False]]),
    }

    with patch.object(module.model, "encode_queries") as encode:
        loss, active = module._reaction_chemistry_consistency_loss(
            query_embeds=query_embeds,
            query_inputs={"reaction_chemistry_vector": torch.ones(2, 2)},
            query_attention_details=attention,
            retrieval_direction="reaction_to_enzyme",
        )

    assert loss.item() == 0.0
    assert active.item() == 0
    encode.assert_not_called()


def test_reaction_residual_identity_uses_projection_base_cosine():
    module = _small_module(reaction_residual_identity_weight=0.02)
    query_embeds = torch.randn(2, 4, requires_grad=True)
    base_cosine = torch.tensor([0.9, 0.7], requires_grad=True)

    loss = module._reaction_residual_identity_loss(
        query_embeds=query_embeds,
        query_attention_details={"residual_base_cosine": base_cosine},
    )

    assert loss.item() == pytest.approx(0.2)
    loss.backward()
    assert base_cosine.grad is not None


def test_reaction_residual_identity_requires_residual_projection_details():
    module = _small_module(reaction_residual_identity_weight=0.02)

    with pytest.raises(RuntimeError, match="output_projection=residual_mlp"):
        module._reaction_residual_identity_loss(
            query_embeds=torch.randn(2, 4),
            query_attention_details={},
        )
