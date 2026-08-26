from scripts.audit_reactzyme_pretraining_overlap import (
    _mcsa_accessions,
    _parse_fasta,
    normalize_sequence,
    truncate_ends_center,
)


def test_mcsa_accessions_include_reference_and_sequence_records():
    entries = [
        {
            "reference_uniprot_id": "P00001, Q00003",
            "protein": {"sequences": [{"uniprot_id": "P00001"}, {"uniprot_id": "Q00002"}]},
        }
    ]

    assert _mcsa_accessions(entries) == {"P00001", "Q00002", "Q00003"}


def test_parse_fasta_and_normalize_sequence():
    records = _parse_fasta(">sp|P00001|TEST name\nACDUOB-*\n>Q00002 description\nGGG\n")

    assert records == {"P00001": "ACDUOB-*", "Q00002": "GGG"}
    assert normalize_sequence(records["P00001"]) == "ACDXXXXX"


def test_ends_center_truncation_is_stable_and_fixed_length():
    sequence = "A" * 300 + "C" * 300 + "D" * 300 + "E" * 300
    truncated = truncate_ends_center(sequence)

    assert len(truncated) == 1022
    assert truncated == truncate_ends_center(sequence)
    assert truncated[:255] == "A" * 255
    assert truncated[-255:] == "E" * 255
