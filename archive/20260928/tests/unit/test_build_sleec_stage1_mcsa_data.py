from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_sleec_stage1_mcsa_data.py"


def load_builder_module():
    spec = importlib.util.spec_from_file_location("build_sleec_stage1_mcsa_data", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mcsa_reference_positive_parser_reads_top_level_residue_sequences(tmp_path):
    module = load_builder_module()
    mcsa_json = tmp_path / "mcsa.json"
    mcsa_json.write_text(
        json.dumps(
            [
                {
                    "residue_sequences": [
                        {"uniprot_id": "P11111", "resid": 5, "is_reference": True},
                        {"uniprot_id": "P11111", "resid": 5, "is_reference": True},
                        {"uniprot_id": "P22222", "resid": 8, "is_reference": False},
                        {"uniprot_id": "", "resid": 1, "is_reference": True},
                        {"uniprot_id": "P33333", "resid": "bad", "is_reference": True},
                    ]
                }
            ]
        )
    )

    positives, stats = module.load_mcsa_reference_positives(mcsa_json)

    assert positives == {"P11111": {4}}
    assert stats.raw_reference_proteins == 1
    assert stats.raw_reference_positive_residues == 1
    assert stats.raw_all_homologue_uniprot_ids == 3
    assert stats.raw_empty_uniprot_entries == 1
    assert stats.raw_invalid_residue_entries == 1


def test_truncate_sequence_preserves_esm2_ends_center_index_map():
    module = load_builder_module()
    sequence = "ABCDEFGHIJKL"

    truncated, index_map = module.truncate_sequence(
        sequence,
        max_sequence_length=8,
        strategy="ends_center",
    )

    assert truncated == "ABEFGHKL"
    assert index_map == {0: 0, 1: 1, 4: 2, 5: 3, 6: 4, 7: 5, 10: 6, 11: 7}


def test_build_outputs_uses_cached_sequences_and_protein_level_split(tmp_path):
    module = load_builder_module()
    mcsa_json = tmp_path / "mcsa.json"
    mcsa_json.write_text(
        json.dumps(
            [
                {
                    "residue_sequences": [
                        {"uniprot_id": "P11111", "resid": 2, "is_reference": True},
                        {"uniprot_id": "P22222", "resid": 4, "is_reference": True},
                    ]
                }
            ]
        )
    )
    cache = tmp_path / "cache.tsv"
    cache.write_text("protein_id\tsequence\nP11111\tACDE\nP22222\tFGHI\n")
    output_dir = tmp_path / "out"

    manifest = module.build_outputs(
        argparse.Namespace(
            mcsa_json=mcsa_json,
            curated_csv=tmp_path / "missing_curated.csv",
            output_dir=output_dir,
            uniprot_cache=cache,
            split_csv=None,
            seed=42,
            uniprot_batch_size=100,
            request_timeout=1,
            request_retries=1,
            skip_uniprot_fetch=True,
            max_sequence_length=1022,
            sequence_truncation="ends_center",
            strict_paper_counts=False,
        )
    )

    labels = (output_dir / "mcsa_residue_labels.csv").read_text()
    assert "P11111,1,1," in labels
    assert "P22222,3,1," in labels
    assert manifest["generated"]["total_proteins"] == 2
    assert manifest["generated"]["positive_residues"] == 2
    assert manifest["generated"]["train_proteins"] == 1
    assert manifest["generated"]["val_proteins"] == 1
