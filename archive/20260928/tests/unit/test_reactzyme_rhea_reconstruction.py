import pytest

pd = pytest.importorskip("pandas")

from horizyn.capability.reactzyme_rhea import (
    combine_source_collapse_variant,
    reconstruct_reactzyme_train_rhea,
)


def _write_split(root, split, rows):
    split_dir = root / split
    split_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(split_dir / "train_pairs.csv", index=False)


def test_reconstruct_reactzyme_rhea_train_excludes_unmapped_rows(tmp_path):
    eval_root = tmp_path / "reactzyme" / "eval"
    seq_a = "MSEQAAA"
    seq_b = "MSEQBBB"
    row_a = {
        "reaction_id": "rxn_set_a",
        "protein_id": "prot_a",
        "reaction_smiles": "CCO.CC=O",
        "protein_sequence": seq_a,
    }
    row_b = {
        "reaction_id": "rxn_set_b",
        "protein_id": "prot_b",
        "reaction_smiles": "CCC.CCO",
        "protein_sequence": seq_b,
    }
    _write_split(eval_root, "time", [row_a, row_b])
    _write_split(eval_root, "enzyme_smi", [row_a])
    _write_split(eval_root, "reaction_smi", [row_a])

    cleaned = tmp_path / "cleaned_uniprot_rhea.tsv"
    cleaned.write_text(
        "Entry\tEC number\tRhea ID\tDate of creation\tSequence\n"
        f"P1\t1.1.1.1\tRHEA:1;RHEA:2\t20200101\t{seq_a}\n",
        encoding="utf-8",
    )
    rhea = tmp_path / "rhea_molecules.tsv"
    rhea.write_text(
        "Rhea ID\tsubstrate\tproduct\n"
        "RHEA:1\tCCO\tCC=O\n"
        "RHEA:2\tCCC\tCC=C\n",
        encoding="utf-8",
    )

    pairs, reactions, members, report = reconstruct_reactzyme_train_rhea(
        reactzyme_eval_root=eval_root,
        cleaned_uniprot_rhea_path=cleaned,
        rhea_molecules_path=rhea,
    )

    assert len(pairs) == 1
    assert len(reactions) == 1
    assert len(members) == 3
    assert set(reactions["reaction_smiles"]) == {"CCO>>CC=O"}
    assert set(members["rhea_id"]) == {"RHEA:1"}
    assert set(members["molecule_match_status"]) == {"exact_molecule_multiset"}
    assert report["splits"]["time"]["rows_without_sequence_match"] == 1
    assert all(">>" in value for value in reactions["reaction_smiles"])


def test_combine_variant_removes_reactzyme_molecule_sets_and_adds_directional():
    base_pairs = pd.DataFrame(
        [
            {
                "pr_id": 0,
                "reaction_id": "r_keep",
                "protein_id": "p_keep",
                "member_count": 1,
                "protein_uid": "p_keep",
                "source_datasets": "horizyn",
                "source_splits": "train",
                "source_entries": "horizyn_train:Rh_1:P1:0",
            },
            {
                "pr_id": 1,
                "reaction_id": "r_old_reactzyme",
                "protein_id": "p_rz",
                "member_count": 1,
                "protein_uid": "p_rz",
                "source_datasets": "reactzyme",
                "source_splits": "time_train",
                "source_entries": "reactzyme_time_train:rxn_set:prot:0",
            },
        ]
    )
    base_reactions = pd.DataFrame(
        [
            {
                "reaction_id": "r_keep",
                "reaction_smiles": "CCO>>CC=O",
                "source_entries": "horizyn_train",
                "source_reaction_ids": "Rh_1",
            },
            {
                "reaction_id": "r_old_reactzyme",
                "reaction_smiles": "CCO.CC=O",
                "source_entries": "reactzyme_time_train",
                "source_reaction_ids": "rxn_set",
            },
        ]
    )
    reconstructed_pairs = pd.DataFrame(
        [
            {
                "pr_id": 0,
                "reaction_id": "r_new",
                "protein_id": "p_rz",
                "member_count": 1,
                "protein_uid": "p_rz",
                "source_datasets": "reactzyme",
                "source_splits": "time_train",
                "source_entries": "reactzyme_time_train:rxn_set:prot:0:RHEA:1",
            }
        ]
    )
    reconstructed_reactions = pd.DataFrame(
        [
            {
                "reaction_id": "r_new",
                "reaction_smiles": "CCO>>CC=O",
                "source_entries": "reactzyme_rhea_reconstructed",
                "source_reaction_ids": "RHEA:1",
            }
        ]
    )

    pairs, reactions = combine_source_collapse_variant(
        base_pairs=base_pairs,
        base_reactions=base_reactions,
        reconstructed_pairs=reconstructed_pairs,
        reconstructed_reactions=reconstructed_reactions,
    )

    assert set(pairs["reaction_id"]) == {"r_keep", "r_new"}
    assert "r_old_reactzyme" not in set(reactions["reaction_id"])
    assert all(">>" in value for value in reactions["reaction_smiles"])
