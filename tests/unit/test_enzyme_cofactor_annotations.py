import pytest

pd = pytest.importorskip("pandas")

from horizyn.capability.cofactors import (
    cofactor_architecture_bins,
    cofactor_chemistry_bins,
)
from horizyn.capability.enzyme_cofactor_annotations import (
    enhance_enzyme_cofactor_annotations,
    extract_cofactor_labels_from_uniprotkb_comment,
    extract_uniprot_accessions_from_source_entries,
    protein_uid,
)


def test_extract_uniprot_accessions_from_source_entries():
    entries = (
        "horizyn_train:Rh_1:P12345:0|"
        "reactzyme_time_train:rxn_x:prot_x:7:RHEA:10008|"
        "horizyn_train:Rh_2:A0A024B7W1-2:1"
    )
    assert extract_uniprot_accessions_from_source_entries(entries) == ["A0A024B7W1", "P12345"]


def test_cofactor_bin_mapping_for_representative_labels():
    assert "rossmann_NAD_NADP" in cofactor_architecture_bins(["NAD"])
    assert "redox" in cofactor_chemistry_bins(["NAD"])
    assert "flavin_FAD_FMN" in cofactor_architecture_bins(["FAD"])
    assert "PLP_dependent" in cofactor_architecture_bins(["PLP"])
    assert "TPP_dependent" in cofactor_architecture_bins(["TPP"])
    assert "P_loop_NTP_kinase" in cofactor_architecture_bins(["ATP"])
    assert "phosphoryl_transfer" in cofactor_chemistry_bins(["ATP"])
    assert "sugar_nucleotide_GT" in cofactor_architecture_bins(["sugar_nucleotide"])
    assert "group_transfer" in cofactor_chemistry_bins(["sugar_nucleotide"])
    assert "thiol_or_protein_redox" in cofactor_architecture_bins(["glutathione"])
    assert "conjugation" in cofactor_chemistry_bins(["glutathione"])
    assert "quinone_PQQ_redox" in cofactor_architecture_bins(["PQQ"])
    assert "biotin_carboxylation" in cofactor_architecture_bins(["biotin"])
    assert "molybdopterin_redox" in cofactor_architecture_bins(["molybdopterin"])
    assert "ascorbate_pterin_redox" in cofactor_architecture_bins(["ascorbate"])
    assert "carotenoid_redox" in cofactor_architecture_bins(["carotenoid"])
    assert "simple_metal_binding" in cofactor_architecture_bins(["Zn2+"])
    assert "metal_catalysis" in cofactor_chemistry_bins(["Zn2+"])
    assert cofactor_architecture_bins([]) == ["none_or_unknown"]
    assert cofactor_chemistry_bins([]) == ["unknown"]


def test_enhance_enzyme_cofactor_annotations_from_source_entries(tmp_path):
    cap_dir = tmp_path / "cap"
    cap_dir.mkdir()
    train_pairs = tmp_path / "train_pairs.csv"
    train_pairs.write_text(
        "pr_id,reaction_id,protein_id,source_entries\n"
        "0,r1,e1,horizyn_train:Rh_1:P12345:0\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "reaction_id": "r1",
                "cofactor_labels": ["NAD"],
                "core_cofactor_labels": ["NAD"],
                "metal_ion_labels": [],
                "auxiliary_participant_labels": [],
            }
        ]
    ).to_parquet(cap_dir / "reaction_features.parquet", index=False)
    pd.DataFrame(
        [
            {
                "enzyme_id": "e1",
                "cofactor_labels_train": [],
                "core_cofactor_labels_train": [],
                "metal_ion_labels_train": [],
                "auxiliary_participant_labels_train": [],
                "quality_flags": ["train_only_labels"],
            }
        ]
    ).to_parquet(cap_dir / "enzyme_capability_labels.parquet", index=False)
    pd.DataFrame(
        [
            {
                "enzyme_id": "e1",
                "reaction_id": "r1",
                "is_positive": 1,
                "source_split": "train",
            }
        ]
    ).to_parquet(cap_dir / "pair_capability_training.parquet", index=False)
    (cap_dir / "enzyme_label_vocabs.json").write_text("{}", encoding="utf-8")
    molecules = tmp_path / "uniprot_molecules.tsv"
    molecules.write_text("uniprot_id\tmolecules\nP12345\tNADH\n", encoding="utf-8")

    labels, pairs, report = enhance_enzyme_cofactor_annotations(
        capability_dir=cap_dir,
        train_pairs_path=train_pairs,
        uniprot_molecules_path=molecules,
        write_csv=False,
    )

    row = labels.iloc[0]
    assert row["enzyme_derived_core_cofactor_labels_train"] == ["NAD"]
    assert row["combined_core_cofactor_labels_train"] == ["NAD"]
    assert row["cofactor_architecture_bins_train"] == ["rossmann_NAD_NADP"]
    assert row["cofactor_chemistry_bins_train"] == ["redox"]
    assert pairs.iloc[0]["combined_core_cofactor_overlap"] == 1
    assert report["new_core_cofactor_enzymes_from_enzyme_side"] == 1


def test_enhance_enzyme_cofactor_annotations_from_sequence_hash(tmp_path):
    cap_dir = tmp_path / "cap"
    cap_dir.mkdir()
    sequence = "ACDEFGHIK"
    enzyme_id = protein_uid(sequence)
    train_pairs = tmp_path / "train_pairs.csv"
    train_pairs.write_text(
        f"pr_id,reaction_id,protein_id,source_entries\n0,r1,{enzyme_id},reactzyme_time_train:rxn_x:prot_x:0:RHEA:1\n",
        encoding="utf-8",
    )
    cleaned = tmp_path / "cleaned_uniprot_rhea.tsv"
    cleaned.write_text(
        f"Entry\tEC number\tRhea ID\tDate of creation\tSequence\nQ9TES1\t1.1.1.1\tRHEA:1\t20200101\t{sequence}\n",
        encoding="utf-8",
    )
    molecules = tmp_path / "uniprot_molecules.tsv"
    molecules.write_text("uniprot_id\tmolecules\nQ9TES1\tFAD\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "reaction_id": "r1",
                "cofactor_labels": ["FAD"],
                "core_cofactor_labels": ["FAD"],
                "metal_ion_labels": [],
                "auxiliary_participant_labels": [],
            }
        ]
    ).to_parquet(cap_dir / "reaction_features.parquet", index=False)
    pd.DataFrame(
        [
            {
                "enzyme_id": enzyme_id,
                "cofactor_labels_train": [],
                "core_cofactor_labels_train": [],
                "metal_ion_labels_train": [],
                "auxiliary_participant_labels_train": [],
                "quality_flags": [],
            }
        ]
    ).to_parquet(cap_dir / "enzyme_capability_labels.parquet", index=False)
    pd.DataFrame(
        [{"enzyme_id": enzyme_id, "reaction_id": "r1", "is_positive": 1, "source_split": "train"}]
    ).to_parquet(cap_dir / "pair_capability_training.parquet", index=False)
    (cap_dir / "enzyme_label_vocabs.json").write_text("{}", encoding="utf-8")

    labels, pairs, _report = enhance_enzyme_cofactor_annotations(
        capability_dir=cap_dir,
        train_pairs_path=train_pairs,
        uniprot_molecules_path=molecules,
        cleaned_uniprot_rhea_path=cleaned,
        write_csv=False,
    )

    row = labels.iloc[0]
    assert row["enzyme_cofactor_sequence_accessions"] == ["Q9TES1"]
    assert row["enzyme_derived_core_cofactor_labels_train"] == ["FAD"]
    assert pairs.iloc[0]["enzyme_derived_core_cofactor_overlap"] == 1


def test_enhance_enzyme_cofactor_annotations_uses_reaction_only_labels_as_fallback(tmp_path):
    cap_dir = tmp_path / "cap"
    cap_dir.mkdir()
    train_pairs = tmp_path / "train_pairs.csv"
    train_pairs.write_text(
        "pr_id,reaction_id,protein_id,source_entries\n"
        "0,r1,e1,horizyn_train:Rh_1:P12345:0\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "reaction_id": "r1",
                "cofactor_labels": ["NAD"],
                "core_cofactor_labels": ["NAD"],
                "metal_ion_labels": [],
                "auxiliary_participant_labels": [],
            }
        ]
    ).to_parquet(cap_dir / "reaction_features.parquet", index=False)
    pd.DataFrame(
        [
            {
                "enzyme_id": "e1",
                "cofactor_labels_train": ["NAD"],
                "core_cofactor_labels_train": ["NAD"],
                "metal_ion_labels_train": [],
                "auxiliary_participant_labels_train": [],
                "quality_flags": ["train_only_labels"],
            }
        ]
    ).to_parquet(cap_dir / "enzyme_capability_labels.parquet", index=False)
    pd.DataFrame(
        [{"enzyme_id": "e1", "reaction_id": "r1", "is_positive": 1, "source_split": "train"}]
    ).to_parquet(cap_dir / "pair_capability_training.parquet", index=False)
    (cap_dir / "enzyme_label_vocabs.json").write_text("{}", encoding="utf-8")
    molecules = tmp_path / "uniprot_molecules.tsv"
    molecules.write_text("uniprot_id\tmolecules\nP12345\t\n", encoding="utf-8")

    labels, pairs, report = enhance_enzyme_cofactor_annotations(
        capability_dir=cap_dir,
        train_pairs_path=train_pairs,
        uniprot_molecules_path=molecules,
        write_csv=False,
    )

    row = labels.iloc[0]
    assert row["combined_core_cofactor_labels_train"] == ["NAD"]
    assert row["reaction_fallback_core_cofactor_labels_train"] == ["NAD"]
    assert row["reaction_plus_uniprot_core_cofactor_labels_train"] == ["NAD"]
    assert row["enzyme_cofactor_label_source"] == "reaction_train_pairs_fallback"
    assert "enzyme_core_cofactor_from_reaction_train_pair_fallback" in row["quality_flags"]
    assert pairs.iloc[0]["combined_core_cofactor_overlap"] == 1
    assert pairs.iloc[0]["reaction_fallback_core_cofactor_overlap"] == 1
    assert pairs.iloc[0]["reaction_plus_uniprot_core_cofactor_overlap"] == 1
    assert report["combined_core_cofactor_enzymes"] == 1
    assert report["reaction_fallback_core_cofactor_enzymes"] == 1
    assert report["reaction_plus_uniprot_core_cofactor_enzymes"] == 1


def test_enhance_enzyme_cofactor_annotations_does_not_pool_reaction_when_uniprot_exists(tmp_path):
    cap_dir = tmp_path / "cap"
    cap_dir.mkdir()
    train_pairs = tmp_path / "train_pairs.csv"
    train_pairs.write_text(
        "pr_id,reaction_id,protein_id,source_entries\n"
        "0,r1,e1,horizyn_train:Rh_1:P12345:0\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "reaction_id": "r1",
                "cofactor_labels": ["NAD"],
                "core_cofactor_labels": ["NAD"],
                "metal_ion_labels": [],
                "auxiliary_participant_labels": [],
            }
        ]
    ).to_parquet(cap_dir / "reaction_features.parquet", index=False)
    pd.DataFrame(
        [
            {
                "enzyme_id": "e1",
                "cofactor_labels_train": ["NAD"],
                "core_cofactor_labels_train": ["NAD"],
                "metal_ion_labels_train": [],
                "auxiliary_participant_labels_train": [],
                "quality_flags": ["train_only_labels"],
            }
        ]
    ).to_parquet(cap_dir / "enzyme_capability_labels.parquet", index=False)
    pd.DataFrame(
        [{"enzyme_id": "e1", "reaction_id": "r1", "is_positive": 1, "source_split": "train"}]
    ).to_parquet(cap_dir / "pair_capability_training.parquet", index=False)
    (cap_dir / "enzyme_label_vocabs.json").write_text("{}", encoding="utf-8")
    molecules = tmp_path / "uniprot_molecules.tsv"
    molecules.write_text("uniprot_id\tmolecules\nP12345\tFAD\n", encoding="utf-8")

    labels, pairs, report = enhance_enzyme_cofactor_annotations(
        capability_dir=cap_dir,
        train_pairs_path=train_pairs,
        uniprot_molecules_path=molecules,
        write_csv=False,
    )

    row = labels.iloc[0]
    assert row["combined_core_cofactor_labels_train"] == ["FAD"]
    assert row["reaction_fallback_core_cofactor_labels_train"] == []
    assert row["reaction_plus_uniprot_core_cofactor_labels_train"] == ["FAD", "NAD"]
    assert row["enzyme_cofactor_label_source"] == "uniprot_molecules"
    assert pairs.iloc[0]["combined_core_cofactor_overlap"] == 0
    assert pairs.iloc[0]["reaction_fallback_core_cofactor_overlap"] == 0
    assert pairs.iloc[0]["reaction_plus_uniprot_core_cofactor_overlap"] == 1
    assert report["combined_core_cofactor_enzymes"] == 1
    assert report["reaction_fallback_core_cofactor_enzymes"] == 0
    assert report["reaction_plus_uniprot_core_cofactor_enzymes"] == 1


def test_extract_uniprotkb_cofactor_comment_maps_chebi_metal():
    labels, flags = extract_cofactor_labels_from_uniprotkb_comment(
        "COFACTOR: Name=Fe(2+); Xref=ChEBI:CHEBI:29033; "
        "Evidence={ECO:0000250|UniProtKB:Q8GUI6}; "
        "Note=Binds 1 Fe(2+) ion per subunit. {ECO:0000250|UniProtKB:Q8GUI6};"
    )

    assert "Fe2+" in labels
    assert "metal" in labels
    assert "enzyme_cofactor_from_uniprotkb_comment" in flags
    assert "enzyme_cofactor_from_uniprotkb_by_similarity" in flags


def test_extract_uniprotkb_cofactor_comment_maps_explicit_absence():
    labels, flags = extract_cofactor_labels_from_uniprotkb_comment(
        "COFACTOR: Note=None. Contrary to most other dioxygenases, "
        "this enzyme does not require a cofactor for catalysis.;"
    )

    assert labels == ["no_cofactor"]
    assert "enzyme_cofactor_explicit_no_cofactor" in flags
    assert "enzyme_no_cofactor_from_uniprotkb_comment" in flags


def test_enhance_enzyme_cofactor_annotations_from_uniprotkb_cache(tmp_path):
    cap_dir = tmp_path / "cap"
    cap_dir.mkdir()
    train_pairs = tmp_path / "train_pairs.csv"
    train_pairs.write_text(
        "pr_id,reaction_id,protein_id,source_entries\n"
        "0,r1,e1,horizyn_train:Rh_1:Q67ZB6:0\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "reaction_id": "r1",
                "cofactor_labels": ["Fe2+", "metal"],
                "core_cofactor_labels": [],
                "metal_ion_labels": ["Fe2+", "metal"],
                "auxiliary_participant_labels": [],
            }
        ]
    ).to_parquet(cap_dir / "reaction_features.parquet", index=False)
    pd.DataFrame(
        [
            {
                "enzyme_id": "e1",
                "cofactor_labels_train": [],
                "core_cofactor_labels_train": [],
                "metal_ion_labels_train": [],
                "auxiliary_participant_labels_train": [],
                "quality_flags": ["train_only_labels", "enzyme_cofactor_no_label_from_molecules"],
            }
        ]
    ).to_parquet(cap_dir / "enzyme_capability_labels.parquet", index=False)
    pd.DataFrame(
        [{"enzyme_id": "e1", "reaction_id": "r1", "is_positive": 1, "source_split": "train"}]
    ).to_parquet(cap_dir / "pair_capability_training.parquet", index=False)
    (cap_dir / "enzyme_label_vocabs.json").write_text("{}", encoding="utf-8")
    molecules = tmp_path / "uniprot_molecules.tsv"
    molecules.write_text("uniprot_id\tmolecules\nQ67ZB6\t\n", encoding="utf-8")
    uniprotkb = tmp_path / "uniprotkb_cofactor_comments.tsv"
    uniprotkb.write_text(
        "Entry\tCofactor\n"
        "Q67ZB6\tCOFACTOR: Name=Fe(2+); Xref=ChEBI:CHEBI:29033; "
        "Evidence={ECO:0000250|UniProtKB:Q8GUI6};\n",
        encoding="utf-8",
    )

    labels, pairs, report = enhance_enzyme_cofactor_annotations(
        capability_dir=cap_dir,
        train_pairs_path=train_pairs,
        uniprot_molecules_path=molecules,
        uniprotkb_cofactor_path=uniprotkb,
        write_csv=False,
    )

    row = labels.iloc[0]
    assert row["enzyme_cofactor_uniprotkb_accessions"] == ["Q67ZB6"]
    assert "Fe2+" in row["enzyme_uniprotkb_cofactor_labels_train"]
    assert "Fe2+" in row["combined_cofactor_labels_train"]
    assert "Fe2+" in row["combined_metal_ion_labels_train"]
    assert row["enzyme_cofactor_label_source"] == "uniprotkb_cofactor_comments"
    assert "enzyme_side_uniprotkb_cofactor_labels" in row["quality_flags"]
    assert pairs.iloc[0]["combined_metal_ion_overlap"] == 2
    assert report["new_any_cofactor_enzymes_from_uniprotkb"] == 1


def test_enhance_enzyme_cofactor_annotations_from_uniprotkb_no_cofactor_cache(tmp_path):
    cap_dir = tmp_path / "cap"
    cap_dir.mkdir()
    train_pairs = tmp_path / "train_pairs.csv"
    train_pairs.write_text(
        "pr_id,reaction_id,protein_id,source_entries\n"
        "0,r1,e1,horizyn_train:Rh_1:Q9N0C0:0\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "reaction_id": "r1",
                "cofactor_labels": [],
                "core_cofactor_labels": [],
                "metal_ion_labels": [],
                "auxiliary_participant_labels": [],
            }
        ]
    ).to_parquet(cap_dir / "reaction_features.parquet", index=False)
    pd.DataFrame(
        [
            {
                "enzyme_id": "e1",
                "cofactor_labels_train": [],
                "core_cofactor_labels_train": [],
                "metal_ion_labels_train": [],
                "auxiliary_participant_labels_train": [],
                "quality_flags": ["train_only_labels", "enzyme_has_no_cofactor_labels"],
            }
        ]
    ).to_parquet(cap_dir / "enzyme_capability_labels.parquet", index=False)
    pd.DataFrame(
        [{"enzyme_id": "e1", "reaction_id": "r1", "is_positive": 1, "source_split": "train"}]
    ).to_parquet(cap_dir / "pair_capability_training.parquet", index=False)
    (cap_dir / "enzyme_label_vocabs.json").write_text("{}", encoding="utf-8")
    molecules = tmp_path / "uniprot_molecules.tsv"
    molecules.write_text("uniprot_id\tmolecules\nQ9N0C0\t\n", encoding="utf-8")
    uniprotkb = tmp_path / "uniprotkb_cofactor_comments.tsv"
    uniprotkb.write_text(
        "Entry\tCofactor\n"
        "Q9N0C0\tCOFACTOR: Note=None. This enzyme does not require a cofactor for catalysis.;\n",
        encoding="utf-8",
    )

    labels, _pairs, report = enhance_enzyme_cofactor_annotations(
        capability_dir=cap_dir,
        train_pairs_path=train_pairs,
        uniprot_molecules_path=molecules,
        uniprotkb_cofactor_path=uniprotkb,
        write_csv=False,
    )

    row = labels.iloc[0]
    assert row["enzyme_uniprotkb_cofactor_labels_train"] == ["no_cofactor"]
    assert row["combined_cofactor_labels_train"] == ["no_cofactor"]
    assert row["combined_core_cofactor_labels_train"] == ["no_cofactor"]
    assert row["cofactor_architecture_bins_train"] == ["none_or_unknown"]
    assert row["cofactor_chemistry_bins_train"] == ["unknown"]
    assert row["enzyme_cofactor_label_source"] == "uniprotkb_cofactor_comments"
    assert "enzyme_has_no_cofactor_labels" not in row["quality_flags"]
    assert "enzyme_cofactor_explicit_no_cofactor" in row["quality_flags"]
    assert report["num_enzymes_with_explicit_no_cofactor"] == 1
