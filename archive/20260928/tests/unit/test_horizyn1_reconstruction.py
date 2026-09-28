import csv
import gzip
from pathlib import Path

from horizyn.datasets.horizyn1_reconstruction import (
    build_rhea_direction_map,
    cluster_and_collapse,
    finalize_enzymemap_pairs,
    load_rhea_direction_map,
    merge_sources,
    parse_uniprot_release,
    prepare_enzymemap,
)


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def read_gzip(path: Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return handle.read()


def test_streaming_uniprot_selection_and_enzymemap_alias_resolution(tmp_path: Path) -> None:
    directions = write(
        tmp_path / "rhea-directions.tsv",
        "RHEA_ID_MASTER\tRHEA_ID_LR\tRHEA_ID_RL\tRHEA_ID_BI\n" "10000\t10001\t10002\t10003\n",
    )
    direction_map = tmp_path / "rhea_map.tsv"
    stats = build_rhea_direction_map(directions, direction_map)
    assert stats["mapped_ids"] == 4
    assert load_rhea_direction_map(direction_map)["10002"] == "10000"

    uniprot = write(
        tmp_path / "uniprot_trembl.dat",
        "ID   FIRST\n"
        "AC   T00001; SECONDARY1;\n"
        "CC   -!- CATALYTIC ACTIVITY: Reaction=x; Xref=Rhea:RHEA:10001;\n"
        "SQ   SEQUENCE   6 AA;\n"
        "     MAA AA1\n"
        "//\n"
        "ID   EXTRA\n"
        "AC   T00002;\n"
        "SQ   SEQUENCE   4 AA;\n"
        "     MCCC\n"
        "//\n"
        "ID   INVALID\n"
        "AC   T00003;\n"
        "DR   Rhea; RHEA:99999;\n"
        "SQ   SEQUENCE   4 AA;\n"
        "     MDDD\n"
        "//\n",
    )
    pair_output = tmp_path / "trembl_pairs.tsv.gz"
    selected_fasta = tmp_path / "selected.fasta.gz"
    extra_fasta = tmp_path / "extra.fasta.gz"
    accession_map = tmp_path / "accession_map.tsv"
    stats = parse_uniprot_release(
        input_path=uniprot,
        rhea_map_path=direction_map,
        valid_master_ids={"10000"},
        wanted_accessions={"SECONDARY1", "T00002", "MISSING"},
        existing_accessions=set(),
        pair_output=pair_output,
        selected_fasta_output=selected_fasta,
        extra_fasta_output=extra_fasta,
        accession_map_output=accession_map,
        source_kind="trembl",
    )
    assert stats["records"] == 3
    assert stats["selected_trembl_proteins"] == 1
    assert stats["trembl_pairs"] == 1
    assert stats["wanted_accessions_resolved"] == 2
    assert stats["wanted_accessions_missing"] == 1
    assert "Rh_10000\tT00001" in read_gzip(pair_output)
    assert ">T00001\nMAAAA\n" in read_gzip(selected_fasta)
    assert ">T00002\nMCCC\n" in read_gzip(extra_fasta)
    assert "SECONDARY1\tT00001" in accession_map.read_text(encoding="utf-8")


def test_enzymemap_preparation_and_finalization(tmp_path: Path) -> None:
    processed = tmp_path / "processed.csv.gz"
    with gzip.open(processed, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["unmapped", "rxn_idx", "ec_num", "protein_refs", "protein_db"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "unmapped": "A>>B",
                "rxn_idx": "7",
                "ec_num": "1.2.3.4",
                "protein_refs": "['SECONDARY1', 'T00002']",
                "protein_db": "uniprot",
            }
        )
        writer.writerow(
            {
                "unmapped": "A>>B",
                "rxn_idx": "8",
                "ec_num": "1.2.3.4",
                "protein_refs": "['SECONDARY1']",
                "protein_db": "uniprot",
            }
        )
    raw_pairs = tmp_path / "raw_pairs.tsv.gz"
    reactions = tmp_path / "reactions.tsv.gz"
    wanted = tmp_path / "wanted.txt"
    stats = prepare_enzymemap(processed, raw_pairs, reactions, wanted)
    assert stats["unique_reactions"] == 1
    assert stats["unique_pairs_before_sequence_resolution"] == 2

    accession_map = write(
        tmp_path / "map.tsv",
        "requested_accession\tprimary_accession\nSECONDARY1\tT00001\nT00002\tT00002\n",
    )
    final_pairs = tmp_path / "final.tsv.gz"
    stats = finalize_enzymemap_pairs(
        raw_pairs,
        accession_map,
        {"T00001", "T00002"},
        final_pairs,
    )
    assert stats["resolved_unique_pairs"] == 2
    contents = read_gzip(final_pairs)
    assert "\tT00001\tEnzymeMap_v2" in contents
    assert "\tT00002\tEnzymeMap_v2" in contents


def test_merge_deduplicates_edges_and_preserves_provenance(tmp_path: Path) -> None:
    dev_pairs = write(
        tmp_path / "dev_pairs.csv",
        "pr_id,reaction_id,protein_id\n0,Rh_10000,P1\n",
    )
    dev_reactions = write(
        tmp_path / "dev_rxns.csv",
        "rs_id,reaction_id,reaction_smiles\n0,Rh_10000,A>>B\n",
    )
    dev_fasta = write(tmp_path / "dev.fasta", ">P1\nMAAA\n")
    trembl_fasta = tmp_path / "trembl.fasta.gz"
    with gzip.open(trembl_fasta, "wt", encoding="utf-8") as handle:
        handle.write(">P2\nMBBB\n")
    extra_fasta = tmp_path / "extra.fasta.gz"
    with gzip.open(extra_fasta, "wt", encoding="utf-8") as handle:
        handle.write(">P3\nMCCC\n")
    trembl_pairs = tmp_path / "trembl.tsv.gz"
    with gzip.open(trembl_pairs, "wt", encoding="utf-8") as handle:
        handle.write(
            "reaction_id\tprotein_id\tsource\n"
            "Rh_10000\tP1\tUniProtKB_TrEMBL_2023_05\n"
            "Rh_10000\tP2\tUniProtKB_TrEMBL_2023_05\n"
        )
    enzymemap_pairs = tmp_path / "em.tsv.gz"
    with gzip.open(enzymemap_pairs, "wt", encoding="utf-8") as handle:
        handle.write("reaction_id\tprotein_id\tsource\nEm_x\tP3\tEnzymeMap_v2\n")
    enzymemap_reactions = tmp_path / "em_rxns.tsv.gz"
    with gzip.open(enzymemap_reactions, "wt", encoding="utf-8") as handle:
        handle.write(
            "reaction_id\treaction_smiles\tsource_reaction_id\tec\nEm_x\tC>>D\t1\t1.1.1.1\n"
        )

    output = tmp_path / "merged"
    stats = merge_sources(
        [dev_pairs],
        [dev_reactions],
        [dev_fasta, trembl_fasta, extra_fasta],
        trembl_pairs,
        enzymemap_pairs,
        enzymemap_reactions,
        output,
        sort_threads=1,
    )
    assert stats["raw_proteins"] == 3
    assert stats["raw_reactions"] == 2
    assert stats["raw_pairs"] == 3
    pairs = (output / "raw_pairs.tsv").read_text(encoding="utf-8")
    assert "UniProtKB_TrEMBL_2023_05,development_release" in pairs


def test_mmseqs_cluster_and_pair_collapse(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    mmseqs = project_root / "tools/mmseqs/bin/mmseqs"
    if not mmseqs.exists():
        return
    fasta = write(
        tmp_path / "raw.fasta",
        ">P1\nMKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP\n"
        ">P2\nMKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP\n"
        ">P3\nGCPVNITLDEMRKAFWQEHGVSYTPCLNIRDMKQAFGV\n",
    )
    pairs = write(
        tmp_path / "raw_pairs.tsv",
        "reaction_id\tprotein_id\tsources\n"
        "Rh_1\tP1\tdevelopment_release\n"
        "Rh_1\tP2\tUniProtKB_TrEMBL_2023_05\n"
        "Em_x\tP3\tEnzymeMap_v2\n",
    )
    stats = cluster_and_collapse(
        raw_fasta=fasta,
        raw_pairs=pairs,
        output_dir=tmp_path / "clustered",
        mmseqs=mmseqs,
        threads=2,
    )
    assert stats["clustered_members"] == 3
    assert stats["clustered_proteins"] == 2
    assert stats["clustered_pairs"] == 2
