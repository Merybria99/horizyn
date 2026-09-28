import csv
import hashlib
import torch

from horizyn.data_module import HorizynDataModule
from horizyn.datasets.base import BaseDataset
from scripts.build_reactzyme_paper_protocols import build_protocol
from horizyn.benchmarks.reactzyme_protocol import (
    split_train_rows,
    split_train_rows_by_reaction_smiles,
)


PAIR_FIELDS = ["pr_id", "reaction_id", "protein_id", "reaction_smiles", "protein_sequence"]
REACTION_FIELDS = ["reaction_id", "reaction_smiles"]


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_source(root):
    source = root / "time"
    train_pairs = [
        {
            "pr_id": str(index),
            "reaction_id": f"r{index}",
            "protein_id": f"p{index}",
            "reaction_smiles": f"C{index}>>N{index}",
            "protein_sequence": "ACDE",
        }
        for index in range(10)
    ]
    train_rxns = [
        {"reaction_id": row["reaction_id"], "reaction_smiles": row["reaction_smiles"]}
        for row in train_pairs
    ]
    test_pairs = [
        {
            "pr_id": "0",
            "reaction_id": "rt",
            "protein_id": "pt",
            "reaction_smiles": "C>>O",
            "protein_sequence": "ACDE",
        }
    ]
    write_csv(source / "train_pairs.csv", PAIR_FIELDS, train_pairs)
    write_csv(source / "train_rxns.csv", REACTION_FIELDS, train_rxns)
    write_csv(source / "test_pairs.csv", PAIR_FIELDS, test_pairs)
    write_csv(
        source / "test_rxns.csv",
        REACTION_FIELDS,
        [{"reaction_id": "rt", "reaction_smiles": "C>>O"}],
    )
    return source


def test_build_protocol_creates_deterministic_train_validation_and_untouched_test(tmp_path):
    source_root = tmp_path / "source"
    source = make_source(source_root)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first = build_protocol(
        "time",
        source_root=source_root,
        out_root=first_root,
        validation_fraction=0.1,
        seed=42,
    )
    build_protocol(
        "time",
        source_root=source_root,
        out_root=second_root,
        validation_fraction=0.1,
        seed=42,
    )

    assert first["counts"]["train_pairs"] == 9
    assert first["counts"]["validation_pairs"] == 1
    assert first["audits"]["train_validation_exact_pair_overlap"] == 0
    assert (first_root / "time/train_pairs.csv").read_bytes() == (
        second_root / "time/train_pairs.csv"
    ).read_bytes()
    assert digest(source / "test_pairs.csv") == digest(first_root / "time/test_pairs.csv")
    assert digest(source / "test_rxns.csv") == digest(first_root / "time/test_rxns.csv")
    assert (first_root / "time/test_candidate_ids.txt").read_text() == "pt\n"
    assert first["original_validation_indices_recoverable"] is False


def test_positive_row_split_uses_seeded_torch_random_split_permutation():
    rows = [{"row": str(index)} for index in range(10)]
    expected_indices = torch.randperm(10, generator=torch.Generator().manual_seed(42)).tolist()

    train, validation = split_train_rows(rows, validation_fraction=0.2, seed=42)

    assert [int(row["row"]) for row in train] == expected_indices[:8]
    assert [int(row["row"]) for row in validation] == expected_indices[8:]


def test_reaction_smiles_split_is_deterministic_and_group_disjoint():
    rows = [
        {
            "reaction_id": f"r{index}",
            "reaction_smiles": smiles,
            "protein_id": f"p{index}",
        }
        for index, smiles in enumerate(
            ["A.B", "A.B", "C.D", "E.F", "G.H", "I.J", "K.L", "M.N", "O.P", "Q.R"]
        )
    ]

    train, validation = split_train_rows_by_reaction_smiles(
        rows,
        validation_fraction=0.2,
        seed=42,
    )
    train_again, validation_again = split_train_rows_by_reaction_smiles(
        rows,
        validation_fraction=0.2,
        seed=42,
    )

    assert train == train_again
    assert validation == validation_again
    assert {row["reaction_smiles"] for row in train}.isdisjoint(
        {row["reaction_smiles"] for row in validation}
    )
    assert len(train) + len(validation) == len(rows)


def test_build_protocol_supports_reaction_smiles_disjoint_validation(tmp_path):
    source_root = tmp_path / "source"
    source = make_source(source_root)
    rows, fields = [], PAIR_FIELDS
    with (source / "train_pairs.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[1]["reaction_smiles"] = rows[0]["reaction_smiles"]
    write_csv(source / "train_pairs.csv", fields, rows)
    write_csv(
        source / "train_rxns.csv",
        REACTION_FIELDS,
        [
            {"reaction_id": row["reaction_id"], "reaction_smiles": row["reaction_smiles"]}
            for row in rows
        ],
    )

    manifest = build_protocol(
        "time",
        source_root=source_root,
        out_root=tmp_path / "out",
        validation_fraction=0.2,
        seed=7,
        split_method="reaction_smiles_disjoint",
    )

    assert manifest["audits"]["train_validation_reaction_id_overlap"] == 0
    assert manifest["audits"]["train_validation_reaction_smiles_overlap"] == 0
    assert manifest["audits"]["test_files_byte_identical_to_release"] is True


def test_forward_only_pair_mode_retains_only_canonical_feature_keys():
    pairs = BaseDataset(
        keys=["0", "1"],
        array_data=[
            {"query_id": "r1", "target_id": "p1"},
            {"query_id": "r2", "target_id": "p2"},
        ],
    )
    data_module = HorizynDataModule.__new__(HorizynDataModule)
    data_module.reaction_direction_mode = "forward_only"

    prepared = data_module._augment_pairs_for_direction_mode(pairs)

    assert prepared.keys == ["0_f", "1_f"]
    assert prepared["0_f"] == {"query_id": "r1_f", "target_id": "p1"}
    assert prepared["1_f"] == {"query_id": "r2_f", "target_id": "p2"}


def test_bidirectional_mode_remains_backward_compatible():
    pairs = BaseDataset(
        keys=["0"],
        array_data=[{"query_id": "r1", "target_id": "p1"}],
    )
    data_module = HorizynDataModule.__new__(HorizynDataModule)
    data_module.reaction_direction_mode = "bidirectional"

    prepared = data_module._augment_pairs_for_direction_mode(pairs)

    assert prepared.keys == ["0_f", "0_r"]
