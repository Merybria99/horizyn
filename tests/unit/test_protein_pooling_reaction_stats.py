import torch

from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule


def test_reaction_stats_include_factorized_blocks_and_directional_gate():
    attention = {
        "modality": torch.tensor([[0.25, 0.50, 0.25], [0.25, 0.50, 0.25]]),
        "modality_entropy": torch.tensor([1.0, 1.0]),
        "modality_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "modality_names": ("reaction_model", "unimol2", "chiro"),
        "modality_block_norms": torch.tensor([[0.5, 0.7, 0.5], [0.5, 0.7, 0.0]]),
        "directional_gate": torch.tensor([0.12, 0.12]),
        "directional_mask": torch.tensor([True, False]),
    }

    stats = ProteinPooledLitModule._reaction_multimodal_attention_stats(attention)

    assert torch.isclose(stats["reaction_factorized/block_norm_unimol2"], torch.tensor(0.7))
    assert torch.isclose(stats["reaction_factorized/block_norm_chiro"], torch.tensor(0.25))
    assert torch.isclose(stats["reaction_directional/gate"], torch.tensor(0.12))
    assert torch.isclose(stats["reaction_directional/active_fraction"], torch.tensor(0.5))


def test_reaction_stats_include_symmetric_output_blocks():
    attention = {
        "modality": torch.tensor([[0.5, 0.5]]),
        "reaction_block_norm_core": torch.tensor([1.0]),
        "reaction_block_weighted_norm_core": torch.tensor([0.75]),
        "reaction_block_weight_core": torch.tensor([0.5625]),
    }

    stats = ProteinPooledLitModule._reaction_multimodal_attention_stats(attention)

    assert stats["reaction_block/norm_core"] == torch.tensor(1.0)
    assert stats["reaction_block/weighted_norm_core"] == torch.tensor(0.75)
    assert stats["reaction_block/weight_core"] == torch.tensor(0.5625)
