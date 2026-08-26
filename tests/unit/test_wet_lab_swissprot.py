import gzip
import json

from wet_lab.prepare_swissprot import normalize_swissprot_fasta


def test_normalize_swissprot_fasta_uses_accessions(tmp_path):
    source = tmp_path / "uniprot_sprot.fasta.gz"
    output = tmp_path / "proteins.fasta"
    ids = tmp_path / "candidate_ids.txt"
    metadata = tmp_path / "proteins.csv"
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write(
            ">sp|P00001|ONE_TEST First protein OS=Test\n"
            "ACDU\n"
            ">sp|Q00002|TWO_TEST Second protein OS=Test\n"
            "GGTT\n"
        )

    stats = normalize_swissprot_fasta(source, output, ids, metadata)

    assert output.read_text(encoding="utf-8") == (
        ">P00001 swissprot_entry=ONE_TEST\nACDU\n" ">Q00002 swissprot_entry=TWO_TEST\nGGTT\n"
    )
    assert ids.read_text(encoding="utf-8") == "P00001\nQ00002\n"
    assert stats["protein_count"] == 2
    assert stats["total_residues"] == 8
    assert stats["nonstandard_residue_counts"] == {"U": 1}


def test_swissprot_manifest_schema_constant_is_json_serializable():
    # Guards the manifest-facing statistics types against numpy scalar leakage.
    assert json.dumps({"protein_count": 2, "nonstandard_residue_counts": {"U": 1}})
