import csv
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "horizyn/datasets/horizyn1_uniprot_annotations.py"
SPEC = importlib.util.spec_from_file_location("native_annotation_test_module", MODULE)
native = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = native
SPEC.loader.exec_module(native)


def record(accessions="P1", sequence="MAAA", annotations=""):
    accession_text = "; ".join(accessions.split(","))
    return (f"ID   ENTRY Unreviewed; {len(sequence)} AA.\nAC   {accession_text};\n"
            f"{annotations}SQ   SEQUENCE   {len(sequence)} AA;\n     {sequence}\n//\n")


def run_extraction(tmp_path, flatfile, fasta=">P1\nMAAA\n", **kwargs):
    source = tmp_path / "uniprot_trembl.dat"
    target = tmp_path / "targets.fasta"
    source.write_text(flatfile, encoding="utf-8")
    target.write_text(fasta, encoding="utf-8")
    args = dict(input_path=source, target_fasta=target,
                output_path=tmp_path / "annotations.tsv.gz",
                manifest_path=tmp_path / "annotation_manifest.json",
                source_kind="trembl")
    args.update(kwargs)
    result = native.extract_uniprot_annotations(**args)
    with gzip.open(args["output_path"], "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        for field in native.LABEL_FIELDS + ("candidate_accessions",):
            row[field] = json.loads(row[field])
    return result, {row["protein_id"]: row for row in rows}, args


def test_multiline_multi_partial_ec_and_cofactor_evidence(tmp_path):
    annotations = (
        "DE   RecName: Full=Example;\n"
        "DE            EC=1.2.\n"
        "DE            3.- {ECO:0000256|RuleBase:RU001,\n"
        "DE            ECO:0000250|UniProtKB:Q12345};\n"
        "DE            EC=2.7.1.1;\n"
        "CC   -!- CATALYTIC ACTIVITY:\n"
        "CC       Reaction=Example; Xref=Rhea:RHEA:12345; EC=3.1.1.1;\n"
        "CC       Evidence={ECO:0000269|PubMed:123,\n"
        "CC       ECO:0000255|HAMAP-Rule:MF_00001};\n"
        "CC   -!- COFACTOR:\n"
        "CC       Name=pyridoxal 5'-\n"
        "CC       phosphate; Xref=ChEBI:CHEBI:597326;\n"
        "CC       Evidence={ECO:0000269|PubMed:456};\n"
        "CC       Name=Mg(2+); Xref=ChEBI:CHEBI:18420;\n"
        "CC       Evidence={ECO:0000250|UniProtKB:P12345};\n"
        "CC       Note=General note; Evidence={ECO:0000269|PubMed:999};\n"
        "CC   -!- SIMILARITY: A family formerly assigned EC=9.9.9.9;\n"
        "DR   Rhea; RHEA:10000;\n"
    )
    result, rows, args = run_extraction(tmp_path, record(annotations=annotations))
    row = rows["P1"]
    assert row["annotation_status"] == "matched"
    assert row["ec_numbers"] == ["1.2.3.-", "2.7.1.1", "3.1.1.1"]
    ec = {value["ec_number"]: value for value in row["ec_evidence"]}
    assert ec["1.2.3.-"]["evidence"] == ["ECO:0000250|UniProtKB:Q12345", "ECO:0000256|RuleBase:RU001"]
    assert ec["2.7.1.1"]["evidence"] == []
    assert ec["3.1.1.1"]["context"] == "CC CATALYTIC ACTIVITY"
    assert "ECO:0000269|PubMed:123" in ec["3.1.1.1"]["evidence"]
    assert row["cofactor_names"] == ["Mg(2+)", "pyridoxal 5'- phosphate"]
    cofactors = {value["name"]: value["evidence"] for value in row["cofactor_evidence"]}
    assert cofactors["Mg(2+)"] == ["ECO:0000250|UniProtKB:P12345"]
    assert cofactors["pyridoxal 5'- phosphate"] == ["ECO:0000269|PubMed:456"]
    assert row["rhea_ids"] == ["10000", "12345"]
    assert row["source_kind"] == "trembl"
    assert row["source_release"] == "2023_05"
    assert result["counts"]["output_rows"] == 1
    assert result["output"]["sha256"] == hashlib.sha256(args["output_path"].read_bytes()).hexdigest()
    assert json.loads(args["manifest_path"].read_text())["status"] == "complete"


def test_exact_primary_secondary_ids_case_and_missingness(tmp_path):
    fasta = ">P1 original\nmaaa\n>SECONDARY\nMAAA\n>DIFFERENT\nMCCC\n>MISSING\nMDDD\n>prot_P1\nMAAA\n>P2\nMEEE\n"
    source = (record("P1,SECONDARY,DIFFERENT", annotations="DE   RecName: Full=X; EC=1.1.1.1;\n")
              + record("P2", "MEEE") + record("NOT_SELECTED", "MFFF"))
    result, rows, _ = run_extraction(tmp_path, source, fasta)
    assert len(rows) == 6
    assert rows["SECONDARY"]["uniprot_accession"] == "P1"
    assert rows["SECONDARY"]["ec_numbers"] == ["1.1.1.1"]
    assert rows["P1"]["sequence_sha256"] == hashlib.sha256(b"MAAA").hexdigest()
    assert rows["DIFFERENT"]["annotation_status"] == "sequence_mismatch"
    assert rows["MISSING"]["annotation_status"] == "unresolved"
    assert rows["prot_P1"]["annotation_status"] == "unresolved"
    assert rows["P2"]["annotation_status"] == "matched_unannotated"
    for protein in ("DIFFERENT", "MISSING", "prot_P1", "P2"):
        assert all(rows[protein][field] == [] for field in native.LABEL_FIELDS)
    assert result["counts"]["records_scanned"] == 3
    assert result["counts"]["selected_native_records"] == 2
    assert result["annotation_status_counts"] == {"matched": 2, "sequence_mismatch": 1, "unresolved": 2, "matched_unannotated": 1}


def test_secondary_accession_on_later_ac_line(tmp_path):
    source = record("P1", annotations="DE   EC=1.1.-.-;\n").replace("AC   P1;\n", "AC   P1; OLD1;\nAC   TARGET;\n")
    _, rows, _ = run_extraction(tmp_path, source, ">TARGET\nMAAA\n")
    assert rows["TARGET"]["uniprot_accession"] == "P1"
    assert rows["TARGET"]["ec_numbers"] == ["1.1.-.-"]


@pytest.mark.parametrize("second_sequence", ["MAAA", "MCCC"])
def test_ambiguous_secondary_accession_never_gets_labels(tmp_path, second_sequence):
    source = (record("P1,OLD", annotations="DE   EC=1.1.1.1;\n")
              + record("P2,OLD", second_sequence, "DE   EC=2.2.2.2;\n"))
    result, rows, _ = run_extraction(tmp_path, source, ">OLD\nMAAA\n")
    assert result["counts"]["output_rows"] == 1
    row = rows["OLD"]
    assert row["annotation_status"] == "ambiguous_accession"
    assert row["uniprot_accession"] == ""
    assert row["candidate_accessions"] == ["P1", "P2"]
    assert all(row[field] == [] for field in native.LABEL_FIELDS)


def test_unrelated_ec_mentions_and_no_label_imputation(tmp_path):
    source = record(annotations=("DE   RecName: Full=An EC=6.6.6.6 binding protein;\n"
                                 "CC   -!- FUNCTION: Similar to EC=9.9.9.9;\n"
                                 "CC   -!- COFACTOR: Name=FAD; Note=Some EC=8.8.8.8 function; Name=not a real cofactor;\n"
                                 "CC   -!- SIMILARITY: EC=7.7.7.7;\n"
                                 "CC   -!- CATALYTIC ACTIVITY: Reaction=substrate resembles EC=5.5.5.5 activity;\n"))
    _, rows, _ = run_extraction(tmp_path, source)
    assert rows["P1"]["ec_numbers"] == []
    assert rows["P1"]["cofactor_names"] == ["FAD"]
    assert rows["P1"]["cofactor_evidence"][0]["evidence"] == []


@pytest.mark.parametrize("source", [
    record()[:-3],
    record()[:-3] + record("P2"),
    record("NOT_SELECTED")[:-3],
    record().replace("4 AA;", "5 AA;"),
    "ID   X\nAC   P1;\n//\n",
])
def test_truncation_or_invalid_selected_sequence_does_not_publish(tmp_path, source):
    with pytest.raises(ValueError):
        run_extraction(tmp_path, source)
    assert not (tmp_path / "annotations.tsv.gz").exists()
    assert not (tmp_path / "annotation_manifest.json").exists()
    assert not list(tmp_path.glob(".annotations.tsv.gz.stage-*"))


@pytest.mark.parametrize("fasta", ["", ">P1\n", ">P1\nMAAA\n>P1\nMAAA\n", "MAAA\n", ">P1\nM*AA\n", ">\nMAAA\n"])
def test_invalid_target_fasta_rejected(tmp_path, fasta):
    with pytest.raises(ValueError):
        run_extraction(tmp_path, record(), fasta)
    assert not (tmp_path / "annotations.tsv.gz").exists()


def test_safe_resume_verifies_without_rescanning_archive(tmp_path, monkeypatch):
    result, _, args = run_extraction(tmp_path, record())
    before = args["output_path"].stat()
    monkeypatch.setattr(native, "index_target_fasta", lambda *_: pytest.fail("should resume without indexing"))
    resumed = native.extract_uniprot_annotations(**args)
    assert resumed == result
    assert args["output_path"].stat().st_mtime_ns == before.st_mtime_ns


def test_resume_rejects_changed_input_and_preserves_output(tmp_path):
    _, _, args = run_extraction(tmp_path, record())
    original = args["output_path"].read_bytes()
    args["input_path"].write_text(record("P2"))
    with pytest.raises(FileExistsError, match="inputs changed"):
        native.extract_uniprot_annotations(**args)
    assert args["output_path"].read_bytes() == original


def test_resume_rejects_wrong_release(tmp_path):
    _, _, args = run_extraction(tmp_path, record())
    with pytest.raises(FileExistsError, match="request"):
        native.extract_uniprot_annotations(**args, release="2024_01")


def test_resume_checks_checksum_not_only_manifest_stat(tmp_path):
    _, _, args = run_extraction(tmp_path, record())
    args["output_path"].write_bytes(b"corrupted")
    saved = json.loads(args["manifest_path"].read_text())
    saved["output"]["signature"] = native._signature(args["output_path"])
    args["manifest_path"].write_text(json.dumps(saved))
    with pytest.raises(FileExistsError, match="checksum"):
        native.extract_uniprot_annotations(**args)
    assert args["output_path"].read_bytes() == b"corrupted"


def test_resume_rechecks_input_after_output_hashing(tmp_path, monkeypatch):
    _, _, args = run_extraction(tmp_path, record())
    actual_hash = native._sha256

    def changing_hash(path):
        value = actual_hash(path)
        if path == args["output_path"]:
            args["input_path"].write_text(record("P2"))
        return value

    monkeypatch.setattr(native, "_sha256", changing_hash)
    with pytest.raises(FileExistsError, match="during resume"):
        native.extract_uniprot_annotations(**args)


def test_resume_rejects_implementation_change(tmp_path, monkeypatch):
    _, _, args = run_extraction(tmp_path, record())
    monkeypatch.setattr(native, "_implementation_signatures", lambda: {"changed": "yes"})
    with pytest.raises(FileExistsError, match="implementation changed"):
        native.extract_uniprot_annotations(**args)


@pytest.mark.parametrize("orphan", ["annotations.tsv.gz", "annotation_manifest.json"])
def test_orphan_existing_output_is_preserved(tmp_path, orphan):
    original = b"existing data"
    (tmp_path / orphan).write_bytes(original)
    with pytest.raises(FileExistsError, match="counterpart"):
        run_extraction(tmp_path, record())
    assert (tmp_path / orphan).read_bytes() == original


@pytest.mark.parametrize("alias_kind", ["direct", "symlink", "hardlink", "manifest_input", "same_outputs", "dangling_symlink"])
def test_output_alias_safety(tmp_path, alias_kind):
    source = tmp_path / "uniprot_trembl.dat"
    target = tmp_path / "targets.fasta"
    source.write_text(record())
    target.write_text(">P1\nMAAA\n")
    output = tmp_path / "annotations.tsv.gz"
    manifest = tmp_path / "manifest.json"
    if alias_kind == "direct":
        output = source
    elif alias_kind == "symlink":
        output.symlink_to(target)
    elif alias_kind == "hardlink":
        os.link(source, output)
    elif alias_kind == "manifest_input":
        manifest = target
    elif alias_kind == "same_outputs":
        manifest = output
    else:
        output.symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError):
        native.extract_uniprot_annotations(input_path=source, target_fasta=target,
                                          output_path=output, manifest_path=manifest, source_kind="trembl")
    assert source.read_text() == record()
    assert target.read_text() == ">P1\nMAAA\n"


def test_input_mutation_during_scan_prevents_publication(tmp_path, monkeypatch):
    actual_iterator = native._source_iterator()

    def changing_iterator(path, source_kind):
        yield from actual_iterator(path, source_kind=source_kind)
        path.write_text(path.read_text() + "\n")

    monkeypatch.setattr(native, "_source_iterator", lambda: changing_iterator)
    with pytest.raises(RuntimeError, match="inputs changed"):
        run_extraction(tmp_path, record())
    assert not (tmp_path / "annotations.tsv.gz").exists()
    assert not (tmp_path / "annotation_manifest.json").exists()


def test_nested_tar_gzip_auto_and_kind_filter(tmp_path):
    source = tmp_path / "release.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        for kind, accession in (("sprot", "P1"), ("trembl", "P2")):
            payload = gzip.compress(record(accession, annotations="DE   EC=1.1.1.1;\n").encode())
            info = tarfile.TarInfo(f"knowledgebase/uniprot_{kind}.dat.gz")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    target = tmp_path / "target.fasta"
    target.write_text(">P1\nMAAA\n>P2\nMAAA\n")
    for kind in ("auto", "sprot"):
        output = tmp_path / f"{kind}.tsv.gz"
        result = native.extract_uniprot_annotations(input_path=source, target_fasta=target,
                                                   output_path=output, manifest_path=tmp_path / f"{kind}.json", source_kind=kind)
        assert result["counts"]["output_rows"] == 2
        assert result["annotation_status_counts"]["matched"] == (2 if kind == "auto" else 1)
        assert result["native_records_by_kind"] == ({"sprot": 1, "trembl": 1} if kind == "auto" else {"sprot": 1})


def test_truncated_gzip_not_published(tmp_path):
    source = tmp_path / "uniprot_trembl.dat.gz"
    source.write_bytes(gzip.compress(record().encode())[:-8])
    target = tmp_path / "target.fasta"
    target.write_text(">P1\nMAAA\n")
    output = tmp_path / "annotations.tsv.gz"
    with pytest.raises((EOFError, OSError)):
        native.extract_uniprot_annotations(input_path=source, target_fasta=target,
                                          output_path=output, manifest_path=tmp_path / "manifest.json", source_kind="trembl")
    assert not output.exists()


def test_cli_stdlib_only(tmp_path):
    source = tmp_path / "uniprot_trembl.dat"
    target = tmp_path / "target.fasta"
    source.write_text(record(annotations="DE   EC=1.1.1.1;\n"))
    target.write_text(">P1\nMAAA\n")
    process = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/extract_horizyn1_uniprot_annotations.py"),
                              "--input", str(source), "--target-fasta", str(target),
                              "--output", str(tmp_path / "annotations.tsv.gz"), "--manifest", str(tmp_path / "manifest.json"),
                              "--source-kind", "trembl"], capture_output=True, text=True, check=True)
    assert json.loads(process.stdout)["counts"]["output_rows"] == 1


def test_progress_interval_emits_counts(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(native, "PROGRESS_INTERVAL", 1)
    run_extraction(tmp_path, record())
    output = capsys.readouterr().err
    assert "Indexed 1 target sequences" in output
    assert "Scanned 1 UniProt records" in output
    assert "Finalized 1 / 1" in output
