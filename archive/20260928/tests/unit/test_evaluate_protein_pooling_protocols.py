from collections import defaultdict

import pytest
import torch
import torch.nn.functional as F

from scripts.evaluate_protein_pooling import (
    CONFIGURED_FORWARD_CANDIDATES,
    HIT_RATE_CUTOFFS,
    LEGACY_ALL_CANDIDATES,
    PAPER_TEST_CANDIDATES,
    append_retrieval_metrics,
    encode_targets,
    format_results_table,
    load_target_embedding_cache,
    score_enzyme_to_reaction,
    score_reaction_to_enzyme,
    select_candidate_keys,
)


def test_encode_targets_forwards_external_score_residues():
    class TargetEncoder:
        output_dim = 2

    class Model:
        target_encoder = TargetEncoder()
        enzyme_input_mode = "raw_mean_sleec_biological_factorized"

        def encode_targets(
            self,
            residue_embeddings,
            *,
            residue_padding_mask,
            score_residue_embeddings,
            score_residue_padding_mask,
            **kwargs,
        ):
            del kwargs
            assert residue_embeddings.shape == (1, 2, 3)
            assert score_residue_embeddings.shape == (1, 2, 4)
            assert torch.equal(residue_padding_mask, score_residue_padding_mask)
            return torch.tensor([[0.25, 0.75]])

    class Module:
        model = Model()

    class Dataset:
        keys = ["p1"]

        def __len__(self):
            return 1

        def __getitem__(self, key):
            assert key == "p1"
            return {
                "residue_embeddings": torch.ones(2, 3),
                "score_residue_embeddings": torch.ones(2, 4),
            }

    result = encode_targets(
        module=Module(),
        residue_dataset=Dataset(),
        device="cpu",
        target_batch_size=1,
        store_on_device=False,
    )

    assert result.tolist() == [[0.25, 0.75]]


def test_standalone_evaluation_uses_checkpoint_retrieval_scorers():
    class Module:
        def _compute_retrieval_scores(self, reaction_embeds, enzyme_embeds):
            assert reaction_embeds.shape == (2, 3)
            assert enzyme_embeds.shape == (4, 3)
            return torch.full((2, 4), 7.0)

        def _compute_enzyme_to_reaction_scores(self, enzyme_embeds, reaction_embeds):
            assert enzyme_embeds.shape == (4, 3)
            assert reaction_embeds.shape == (2, 3)
            return torch.full((4, 2), 11.0)

    module = Module()
    reaction_embeds = torch.zeros(2, 3)
    enzyme_embeds = torch.zeros(4, 3)

    r2e_scores = score_reaction_to_enzyme(module, reaction_embeds, enzyme_embeds)
    e2r_scores = score_enzyme_to_reaction(module, enzyme_embeds, reaction_embeds)

    assert r2e_scores.shape == (2, 4)
    assert torch.all(r2e_scores == 7.0)
    assert e2r_scores.shape == (4, 2)
    assert torch.all(e2r_scores == 11.0)


def test_standalone_signed_square_root_cosine_for_single_prototype():
    class Model:
        enzyme_prototype_count = 1

    class Module:
        validation_similarity = "cosine"
        model = Model()

    reaction_embeds = torch.tensor([[4.0, -1.0]])
    enzyme_embeds = torch.tensor([[1.0, -4.0], [4.0, 1.0]])

    actual = score_reaction_to_enzyme(
        Module(),
        reaction_embeds,
        enzyme_embeds,
        cosine_feature_power=0.5,
    )
    transformed_reactions = torch.tensor([[2.0, -1.0]])
    transformed_enzymes = torch.tensor([[1.0, -2.0], [2.0, 1.0]])
    expected = F.normalize(transformed_reactions, dim=-1) @ F.normalize(
        transformed_enzymes,
        dim=-1,
    ).t()

    assert torch.allclose(actual, expected)


def test_hit_rate_metrics_include_tiger_appendix_cutoffs():
    metrics = defaultdict(list)
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    target_idx = torch.tensor([4])

    append_retrieval_metrics(metrics, scores, target_idx)

    assert HIT_RATE_CUTOFFS[:7] == (1, 2, 3, 4, 5, 10, 20)
    assert metrics["top_4"] == [0.0]
    assert metrics["top_5"] == [1.0]
    assert metrics["top_10"] == [1.0]


def test_legacy_protocol_uses_configured_candidate_pool():
    keys, stats = select_candidate_keys(
        evaluation_protocol=LEGACY_ALL_CANDIDATES,
        available_keys=["p1", "p2", "p3"],
        configured_candidate_ids=["p3", "missing", "p1"],
        test_target_ids=["p2"],
    )

    assert keys == ["p3", "p1"]
    assert stats["missing_count"] == 1
    assert stats["test_count"] == 1


def test_configured_forward_protocol_uses_configured_candidate_pool():
    keys, stats = select_candidate_keys(
        evaluation_protocol=CONFIGURED_FORWARD_CANDIDATES,
        available_keys=["p1", "p2", "p3"],
        configured_candidate_ids=["p3", "missing", "p1"],
        test_target_ids=["p2"],
    )

    assert keys == ["p3", "p1"]
    assert stats["missing_count"] == 1


def test_paper_protocol_uses_unique_test_proteins_in_pair_order():
    keys, stats = select_candidate_keys(
        evaluation_protocol=PAPER_TEST_CANDIDATES,
        available_keys=["p1", "p2", "p3"],
        configured_candidate_ids=["p3", "p1", "p2"],
        test_target_ids=["p2", "p1", "p2"],
    )

    assert keys == ["p2", "p1"]
    assert stats["selected_count"] == 2
    assert stats["missing_count"] == 0


def test_paper_protocol_rejects_missing_test_proteins():
    with pytest.raises(ValueError, match="every unique test protein"):
        select_candidate_keys(
            evaluation_protocol=PAPER_TEST_CANDIDATES,
            available_keys=["p1"],
            configured_candidate_ids=[],
            test_target_ids=["p1", "p2"],
        )


def test_target_cache_can_supply_ordered_subset(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    config = tmp_path / "config.yaml"
    checkpoint.touch()
    config.touch()
    cache = tmp_path / "targets.pt"
    torch.save(
        {
            "checkpoint": str(checkpoint.resolve()),
            "config": str(config.resolve()),
            "target_keys": ["p1", "p2", "p3"],
            "target_embeds": torch.tensor([[1.0], [2.0], [3.0]]),
        },
        cache,
    )

    result = load_target_embedding_cache(
        cache_path=cache,
        expected_target_keys=["p3", "p1"],
        checkpoint_path=checkpoint,
        config_path=config,
        device="cpu",
        store_on_device=False,
    )

    assert result is not None
    assert result.tolist() == [[3.0], [1.0]]


def test_results_table_formats_both_directions():
    table = format_results_table(
        {
            "direction": "both",
            "evaluation_protocol": PAPER_TEST_CANDIDATES,
            "reaction_to_enzyme/num_queries": 2,
            "reaction_to_enzyme/reactzyme_mrr": 0.25,
            "reaction_to_enzyme/first_positive_mrr": 0.3,
            "enzyme_to_reaction/num_queries": 3,
            "enzyme_to_reaction/reactzyme_mrr": 0.5,
            "enzyme_to_reaction/first_positive_mrr": 0.6,
        }
    )

    assert "REACTION TO ENZYME" in table
    assert "ENZYME TO REACTION" in table
    assert "Queries evaluated: 2" in table
    assert "Queries evaluated: 3" in table
    assert "0.2500" in table
    assert "0.3000" in table
    assert "0.5000" in table
    assert "0.6000" in table
