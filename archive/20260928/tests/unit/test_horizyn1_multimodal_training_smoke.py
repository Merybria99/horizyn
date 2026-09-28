"""CPU integration smoke: real indexed batches, both towers and optimizer steps."""
import csv

import h5py
import lightning.pytorch as pl
import numpy as np
import pytest
import torch

from horizyn.datasets.indexed_pairs import SEMANTICS, write_indexed_pairs
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from horizyn.reaction_conditioned_data_module import ReactionConditionedDataModule


def make_multimodal_fixture(tmp_path, materialized_directions, *, model_options=None):
    rng = np.random.default_rng(42)
    qids = [f"r{i}" for i in range(4)]
    if materialized_directions:
        qids = [f"{q}__horizyn_{'forward' if i % 2 == 0 else 'reverse'}" for i, q in enumerate(qids)]
    pids = [f"p{i}" for i in range(4)]
    index = tmp_path / "index"
    write_indexed_pairs(
        index, query_ids=np.asarray(qids), protein_ids=np.asarray(pids),
        pairs=np.column_stack((np.arange(4), np.arange(4))),
        protein_ec=np.arange(4), ec_prefix=np.array([0, 0, 1, 1]),
        query_ec_indptr=np.arange(5), query_ec=np.arange(4),
        mechanism_bits=np.ones(4, dtype=int), native_cofactor_bits=np.ones(4, dtype=int),
        reaction_cofactor_bits=np.zeros(4, dtype=int),
        ec_eligible=np.ones(4, dtype=bool), biological_eligible=np.ones(4, dtype=bool),
        query_signature=np.arange(4), provenance={"pair_scope": "train", "annotation_semantics": SEMANTICS},
    )
    pairs, reactions = tmp_path / "pairs.csv", tmp_path / "reactions.csv"
    with pairs.open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pr_id", "reaction_id", "protein_id"])
        writer.writerows((i, q, p) for i, (q, p) in enumerate(zip(qids, pids)))
    with reactions.open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reaction_id", "reaction_smiles"])
        writer.writerows((q, "CCO>>CC=O") for q in qids)
    with h5py.File(tmp_path / "proteins.h5", "w") as handle:
        handle["ids"] = np.array(pids, dtype="S")
        handle["vectors"] = rng.normal(size=(16, 8)).astype("float16")
        handle["offsets"] = np.arange(0, 17, 4)
    # Extraction stores unsuffixed forward IDs; datamodule adds _f. This exercises
    # the real feature alias resolution instead of hand-building trainer batches.
    with h5py.File(tmp_path / "t5.h5", "w") as handle:
        handle["ids"] = np.array(qids, dtype="S")
        handle["vectors"] = rng.normal(size=(4, 6)).astype("float32")
    for name, dim in (("unimol", 5), ("chiro", 3)):
        with h5py.File(tmp_path / f"{name}.h5", "w") as handle:
            handle["ids"] = np.array(qids, dtype="S")
            for side in ("reactant", "product"):
                handle[f"{side}_vectors"] = rng.normal(size=(4, dim)).astype("float32")
                handle[f"{side}_offsets"] = np.arange(5)
    np.savez(tmp_path / "chem.npz", ids=qids, vectors=rng.normal(size=(4, 617)).astype("float32"), mask=np.ones(4, dtype=bool))
    data = ReactionConditionedDataModule(
        indexed_pairs_dir=str(index), train_pairs_path=str(pairs), test_pairs_path=str(pairs),
        train_reactions_path=str(reactions), test_reactions_path=str(reactions),
        protein_residue_embeds_path=str(tmp_path / "proteins.h5"), residue_dim=8,
        reaction_representation="multimodal_reaction_attention", reaction_direction_mode="forward_only",
        reaction_t5v2_embeds_path=str(tmp_path / "t5.h5"), reaction_unimol2_embeds_path=str(tmp_path / "unimol.h5"),
        reaction_chiro_embeds_path=str(tmp_path / "chiro.h5"), reaction_chemistry_vectors_path=str(tmp_path / "chem.npz"),
        reaction_model_dim=6, reaction_unimol_dim=5, reaction_chiro_dim=3, reaction_chemistry_dim=617,
        reaction_use_chiro=True, reaction_use_chemistry=True, standardize_reactions=False,
        train_batch_size=20, typed_negative_positive_fraction=.85, validation_enabled=False, num_workers=0,
    )
    model = ProteinPooledLitModule(
        query_encoder_dims=[16, 32, 16], target_encoder_dims=[16, 16], embedding_dim=16, residue_dim=8,
        pooling="sleec_guided_attention", sleec_scorer_hidden_dim=8, sleec_score_hidden_dim=8,
        sleec_freeze_scorer=True, enzyme_input_mode="raw_mean_sleec_biological_factorized",
        enzyme_block_dims={"core": 4, "site": 4, "mechanism": 4, "cofactor": 2, "ec": 2},
        enzyme_block_weights={"core": .55, "site": .20, "mechanism": .12, "cofactor": .08, "ec": .05},
        biofp_family_dims={"mechanism": 8, "cofactor": 32}, biofp_hidden_dim=16, hyperbolic_hyp_dim=4,
        query_encoder_type="multimodal_reaction_attention", reaction_model_dim=6,
        reaction_unimol_dim=5, reaction_chienn_dim=3, reaction_chemistry_dim=617,
        reaction_use_model=True, reaction_use_chienn=True, reaction_use_chemistry=True,
        reaction_modality_encoder_widths=[32], reaction_side_composition="directional_delta",
        loss_name="BidirectionalSampledMultiPositiveInfoNCELoss", positive_pair_source="all_known_in_batch",
        sampled_require_both_directions=True, sampled_share_indexed_negatives=True,
        biofp_aux_weight=0, log_attention_stats=False, **(model_options or {}),
    )
    return data, model


@pytest.mark.parametrize("materialized_directions", [False, True])
def test_indexed_multimodal_factorized_training_two_steps(tmp_path, materialized_directions):
    data, model = make_multimodal_fixture(tmp_path, materialized_directions)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=2, max_epochs=2,
                         logger=False, enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, use_distributed_sampler=False, limit_val_batches=0)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        trainer.fit(model, datamodule=data)
    finally:
        torch.set_num_threads(previous_threads)
    assert trainer.global_step == 2
    assert torch.isfinite(trainer.callback_metrics["train/loss"])
