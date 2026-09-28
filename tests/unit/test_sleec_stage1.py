import h5py
import numpy as np
import pytest
import torch

from horizyn.sleec_stage1 import (
    FastaRecord,
    ResidueLabelRecord,
    ResidueLabelDataset,
    SLEECStage1Classifier,
    balance_binary_records,
    binary_classification_metrics,
    confidence_aware_stage1_loss,
    low_entropy_labels,
    msa_column_entropies,
    read_fasta,
    residue_label_collate,
)


def test_msa_entropy_and_low_entropy_labels():
    records = [
        FastaRecord("query", "SVADGH"),
        FastaRecord("h1", "SYADLH"),
        FastaRecord("h2", "SQGEAY"),
        FastaRecord("h3", "SNFDKH"),
        FastaRecord("h4", "SLAEAY"),
        FastaRecord("h5", "STTDEY"),
        FastaRecord("h6", "SIWDGY"),
    ]

    entropies = msa_column_entropies(records)
    labels = low_entropy_labels(entropies, positive_fraction=0.10)

    assert len(entropies) == 6
    assert labels.count(1) == 1
    assert labels[0] == 1


def test_a3m_reader_removes_lowercase_insertions(tmp_path):
    a3m_path = tmp_path / "query.a3m"
    a3m_path.write_text(">query\nACDefGH\n>h1\nACDqqGH\n", encoding="utf-8")

    records = read_fasta(a3m_path, a3m=True)

    assert [record.sequence for record in records] == ["ACDGH", "ACDGH"]


def test_confidence_aware_loss_filters_low_confidence_pseudo_examples():
    supervised_logits = torch.tensor([0.0, 2.0])
    supervised_labels = torch.tensor([0.0, 1.0])
    pseudo_logits = torch.tensor([0.0, 3.0, -3.0])
    pseudo_labels = torch.tensor([1.0, 1.0, 0.0])

    loss, components = confidence_aware_stage1_loss(
        supervised_logits,
        supervised_labels,
        pseudo_logits,
        pseudo_labels,
        lambda_pseudo=1.0,
        confidence_threshold=0.9,
    )

    assert torch.isfinite(loss)
    assert components["pseudo_selected"].item() == 2.0
    assert components["pseudo_coverage"].item() == pytest.approx(2 / 3)


def test_confidence_aware_loss_supports_supervised_pos_weight():
    supervised_logits = torch.tensor([0.0, 0.0])
    supervised_labels = torch.tensor([1.0, 0.0])

    unweighted, _ = confidence_aware_stage1_loss(supervised_logits, supervised_labels)
    weighted, components = confidence_aware_stage1_loss(
        supervised_logits,
        supervised_labels,
        supervised_pos_weight=3.0,
    )

    assert weighted > unweighted
    assert float(components["supervised_loss"]) == pytest.approx(float(weighted))


def test_balance_binary_records_makes_one_to_one_negative_sample():
    records = [
        ResidueLabelRecord("p", 0, 1),
        ResidueLabelRecord("p", 1, 1),
        ResidueLabelRecord("p", 2, 0),
        ResidueLabelRecord("p", 3, 0),
        ResidueLabelRecord("p", 4, 0),
        ResidueLabelRecord("p", 5, 0),
    ]

    balanced = balance_binary_records(records, seed=7)

    assert len(balanced) == 4
    assert sum(record.label == 1 for record in balanced) == 2
    assert sum(record.label == 0 for record in balanced) == 2


def test_residue_label_dataset_reads_single_residue_vectors(tmp_path):
    h5_path = tmp_path / "residues.h5"
    vectors = np.arange(15, dtype=np.float32).reshape(5, 3)
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("ids", data=np.array([b"p1", b"p2"]))
        h5.create_dataset("offsets", data=np.array([0, 2, 5], dtype=np.int64))
        h5.create_dataset("vectors", data=vectors)

    records = [
        ResidueLabelRecord("p1", 1, 1),
        ResidueLabelRecord("p2", 2, 0),
    ]
    dataset = ResidueLabelDataset(records, h5_path)
    batch = residue_label_collate([dataset[0], dataset[1]])

    assert torch.equal(batch["embeddings"][0], torch.from_numpy(vectors[1]))
    assert torch.equal(batch["embeddings"][1], torch.from_numpy(vectors[4]))
    assert batch["labels"].tolist() == [1.0, 0.0]


def test_stage1_classifier_variants_and_metrics_smoke():
    torch.manual_seed(0)
    embeddings = torch.randn(5, 4)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])

    paper_model = SLEECStage1Classifier(input_dim=4, hidden_dim=3)
    regularized_model = SLEECStage1Classifier(
        input_dim=4,
        hidden_dim=3,
        variant="regularized",
        dropout=0.1,
        layer_norm=True,
    )
    paper_logits = paper_model(embeddings)
    regularized_logits = regularized_model(embeddings)
    metrics = binary_classification_metrics(paper_logits, labels)

    assert paper_logits.shape == (5,)
    assert regularized_logits.shape == (5,)
    assert paper_model.variant == "paper"
    assert regularized_model.variant == "regularized"
    assert any(isinstance(module, torch.nn.LayerNorm) for module in regularized_model.net)
    assert any(isinstance(module, torch.nn.Dropout) for module in regularized_model.net)
    assert set(metrics) >= {"precision", "recall", "f1", "accuracy"}


def test_stage1_classifier_rejects_unknown_variant():
    with pytest.raises(ValueError, match="variant"):
        SLEECStage1Classifier(input_dim=4, hidden_dim=3, variant="wide")
