import csv
from pathlib import Path

from scripts.prepare_enzymecage_f3 import convert


def test_positive_only_dedup_and_validation_overlap(tmp_path: Path):
    columns = ["Label", "sequence", "CANO_RXN_SMILES"]
    train = tmp_path / "train.csv"
    valid = tmp_path / "valid.csv"
    for path, rows in (
        (train, [("1", "AAAA", "C>>O"), ("1", "AAAA", "C>>O"),
                 ("0", "BBBB", "C>>O")]),
        (valid, [("1", "AAAA", "C>>O"), ("1", "CCCC", "C>>O")]),
    ):
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
    proteins, reactions, train_pairs = {}, {}, set()
    train_stats = convert(train, "train", tmp_path, proteins, reactions, train_pairs)
    valid_stats = convert(valid, "validation", tmp_path, proteins, reactions, train_pairs)
    assert train_stats["positive_pairs"] == 1
    assert train_stats["duplicate_pairs"] == 1
    assert valid_stats["positive_pairs"] == 1
    assert valid_stats["train_overlap_pairs"] == 1
    assert len(proteins) == 2
    assert len(reactions) == 1
