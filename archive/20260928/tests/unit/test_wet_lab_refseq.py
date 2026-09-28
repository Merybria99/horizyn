import csv
import gzip

import pytest

from wet_lab.prepare_refseq import (
    discover_current_release,
    normalize_refseq_fastas,
    parse_refseq_header,
    parse_release_files,
)


def test_discover_current_refseq_release_uses_newest_catalog():
    index = (
        '<a href="release235.files.installed">old</a>'
        '<a href="release236.files.installed">current</a>'
        '<a href="release236.removed-records.gz">removed</a>'
    )

    assert discover_current_release(index) == 236


def test_parse_release_files_selects_wp_fastas_in_numeric_order():
    catalog = "\n".join(
        [
            f"{'a' * 32}\tbacteria.wp_protein.10.protein.faa.gz",
            f"{'b' * 32}\tarchaea.wp_protein.2.protein.faa.gz",
            f"{'c' * 32}\tbacteria.2.protein.faa.gz",
            f"{'d' * 32}\tbacteria.wp_protein.2.protein.faa.gz",
            f"{'e' * 32}\tbacteria.wp_protein.1.protein.faa.gz",
            f"{'f' * 32}\tarchaea.wp_protein.1.protein.faa.gz",
        ]
    )

    selected = parse_release_files(
        catalog,
        divisions=("bacteria", "archaea"),
        max_files_per_division=2,
    )

    assert [item.filename for item in selected] == [
        "bacteria.wp_protein.1.protein.faa.gz",
        "bacteria.wp_protein.2.protein.faa.gz",
        "archaea.wp_protein.1.protein.faa.gz",
        "archaea.wp_protein.2.protein.faa.gz",
    ]
    assert selected[0].url.endswith("/bacteria/bacteria.wp_protein.1.protein.faa.gz")


def test_parse_refseq_header_preserves_version_and_extracts_organism():
    assert parse_refseq_header("WP_012345678.2 D-tagatose epimerase [Example archaeon]") == (
        "WP_012345678.2",
        "D-tagatose epimerase",
        "Example archaeon",
    )

    with pytest.raises(ValueError, match="versioned RefSeq"):
        parse_refseq_header("WP_012345678 unversioned protein [Example]")


def test_normalize_refseq_fastas_writes_query_compatible_artifacts(tmp_path):
    archaea = tmp_path / "archaea.wp_protein.1.protein.faa.gz"
    bacteria = tmp_path / "bacteria.wp_protein.1.protein.faa.gz"
    with gzip.open(archaea, "wt", encoding="utf-8") as handle:
        handle.write(
            ">WP_000000001.1 Epimerase alpha [Archaeon one]\n"
            "ACDU\n"
            ">WP_000000002.1 short protein [Archaeon two]\n"
            "GG\n"
        )
    with gzip.open(bacteria, "wt", encoding="utf-8") as handle:
        handle.write(
            ">WP_000000002.1 short protein [Archaeon two]\n"
            "GG\n"
            ">WP_000000003.4 hypothetical protein [Bacterium one]\n"
            "TTXX\n"
        )

    output = tmp_path / "proteins.fasta"
    ids = tmp_path / "candidate_ids.txt"
    metadata = tmp_path / "proteins.csv"
    stats = normalize_refseq_fastas(
        [archaea, bacteria],
        output,
        ids,
        metadata,
        min_length=4,
    )

    assert output.read_text(encoding="utf-8") == (">WP_000000001.1\nACDU\n>WP_000000003.4\nTTXX\n")
    assert ids.read_text(encoding="utf-8") == "WP_000000001.1\nWP_000000003.4\n"
    with metadata.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "protein_id": "WP_000000001.1",
            "accession": "WP_000000001",
            "description": "Epimerase alpha",
            "organism": "Archaeon one",
            "sequence_length": "4",
            "refseq_division": "archaea",
            "source_file": archaea.name,
        },
        {
            "protein_id": "WP_000000003.4",
            "accession": "WP_000000003",
            "description": "hypothetical protein",
            "organism": "Bacterium one",
            "sequence_length": "4",
            "refseq_division": "bacteria",
            "source_file": bacteria.name,
        },
    ]
    assert stats == {
        "source_record_count": 4,
        "protein_count": 2,
        "duplicate_accessions_skipped": 1,
        "below_minimum_length_skipped": 1,
        "above_maximum_length_skipped": 0,
        "total_residues": 8,
        "minimum_sequence_length": 4,
        "maximum_sequence_length": 4,
        "nonstandard_residue_counts": {"U": 1, "X": 2},
    }
