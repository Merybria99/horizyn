import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/audit_horizyn1_reconstruction.py"
SPEC = importlib.util.spec_from_file_location("reconstruction_audit", SCRIPT)
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


@pytest.fixture
def run_root(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "clustered").mkdir()
    contents = {
        "raw/raw_proteins.fasta": ">P1 description\nMA\nAA\n>P2\nMAAT\n>P3\nMCCC\n",
        "raw/raw_reactions.tsv": "reaction_id\treaction_smiles\nEm_1\tC>>CO\nRh_1\tCO>>C=O\n",
        "raw/raw_pairs.tsv": (
            "reaction_id\tprotein_id\tsources\n"
            "Em_1\tP3\tEnzymeMap_v2\n"
            "Rh_1\tP1\tdevelopment_release\n"
            "Rh_1\tP2\tUniProtKB_TrEMBL_2023_05,development_release\n"
        ),
        "clustered/proteins.fasta": ">P1\nMAAA\n>P3\nMCCC\n",
        "clustered/clusters.tsv": "P1\tP1\nP1\tP2\nP3\tP3\n",
        "clustered/pairs.tsv": (
            "reaction_id\tprotein_id\tsources\n"
            "Em_1\tP3\tEnzymeMap_v2\n"
            "Rh_1\tP1\tdevelopment_release,UniProtKB_TrEMBL_2023_05,development_release\n"
        ),
    }
    for relative, text in contents.items():
        (tmp_path / relative).write_text(text, encoding="utf-8")
    (tmp_path / "raw/raw_manifest.json").write_text(
        json.dumps(
            {
                "raw_proteins": 3,
                "raw_reactions": 2,
                "raw_pairs": 3,
            }
        )
    )
    (tmp_path / "clustered/clustered_manifest.json").write_text(
        json.dumps(
            {
                "clustered_proteins": 2,
                "clustered_pairs": 2,
                "clustered_members": 3,
            }
        )
    )
    return tmp_path


def run_audit(root, stage="clustered"):
    return audit_module.Audit(root, stage).run()


def checks_failed(report):
    return {item["check"] for item in report["errors"]}


def replace(root, relative, old, new):
    path = root / relative
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))


def test_complete_graph_and_source_union_pass(run_root):
    report = run_audit(run_root)
    assert report["status"] == "passed"
    assert report["errors"] == []
    assert report["observed_counts"]["raw_proteins"] == 3
    assert report["observed_counts"]["expected_collapsed_pairs"] == 2
    assert report["observed_counts"]["clustered_members"] == 3
    assert report["observed_counts"]["missing_collapsed_pairs"] == 0
    assert report["paper_deltas"]["clustered_pairs"] == 2 - 8_897_870


def test_raw_stage_does_not_require_clustered_outputs(run_root):
    (run_root / "clustered/clustered_manifest.json").unlink()
    report = run_audit(run_root, "raw")
    assert report["status"] == "passed"
    assert "clustered_pairs" not in report["observed_counts"]


def test_passed_audit_binds_every_checked_artifact_version(run_root):
    report = run_audit(run_root)
    assert report["status"] == "passed"
    assert report["artifact_signatures"] == {
        path: list(audit_module.signature(Path(path))) for path in report["paths"].values()
    }
    original = dict(report["artifact_signatures"])
    pairs = run_root / "clustered/pairs.tsv"
    pairs.write_text(pairs.read_text().replace("Em_1", "Em_2"))
    assert original[str(pairs)] != list(audit_module.signature(pairs))


def test_raw_audit_records_only_artifacts_it_checked(run_root):
    report = run_audit(run_root, "raw")
    assert set(report["artifact_signatures"]) == {
        path for name, path in report["paths"].items() if name.startswith("raw_")
    }


@pytest.mark.parametrize("relative", ["raw/raw_pairs.tsv", "clustered/clustered_manifest.json"])
def test_missing_completed_outputs_are_incomplete(run_root, relative):
    (run_root / relative).unlink()
    report = run_audit(run_root)
    assert report["status"] == "incomplete"
    assert report["observed_counts"] == {}


def test_partial_output_is_never_accepted_as_complete(run_root):
    (run_root / "raw/raw_pairs.tsv.partial").write_text("unfinished\n")
    assert run_audit(run_root)["status"] == "incomplete"


@pytest.mark.parametrize(
    "relative,old,new,check",
    [
        ("raw/raw_pairs.tsv", "Em_1\tP3", "Em_1\tMISSING", "raw_pairs"),
        ("raw/raw_pairs.tsv", "Em_1\tP3", "Em_MISSING\tP3", "raw_pairs"),
        ("clustered/pairs.tsv", "Em_1\tP3", "Em_1\tP2", "clustered_pairs"),
        ("raw/raw_proteins.fasta", ">P2\n", ">P1\n", "raw_proteins"),
        ("raw/raw_reactions.tsv", "Rh_1\t", "Em_1\t", "raw_reactions"),
        ("raw/raw_reactions.tsv", "C>>CO", "", "raw_reactions"),
        ("raw/raw_proteins.fasta", ">P2\nMAAT\n", ">P2\n", "raw_proteins"),
        ("raw/raw_proteins.fasta", "MAAT", "MA*T", "raw_proteins"),
        ("clustered/proteins.fasta", "MAAA", "MAAT", "representative_sequences"),
        ("clustered/clusters.tsv", "P1\tP2\n", "", "cluster_membership"),
        ("clustered/clusters.tsv", "P1\tP2\n", "P1\tP1\n", "cluster_membership"),
        ("clustered/clusters.tsv", "P1\tP1\n", "P3\tP1\n", "cluster_membership"),
        ("clustered/clusters.tsv", "P1\tP2\n", "P1\tMISSING\n", "cluster_membership"),
        ("raw/raw_pairs.tsv", "\tsources\n", "\tsource\n", "raw_pairs"),
        ("raw/raw_pairs.tsv", "Em_1\tP3\tEnzymeMap_v2\n", "Em_1\tP3\t\n", "raw_pairs"),
        ("clustered/pairs.tsv", "Em_1\tP3\tEnzymeMap_v2\n", "", "collapsed_edge_union"),
        (
            "clustered/pairs.tsv",
            "Em_1\tP3\tEnzymeMap_v2\n",
            "Em_1\tP1\tEnzymeMap_v2\nEm_1\tP3\tEnzymeMap_v2\n",
            "collapsed_edge_union",
        ),
        (
            "clustered/pairs.tsv",
            "development_release,UniProtKB_TrEMBL_2023_05,development_release",
            "development_release",
            "collapsed_source_union",
        ),
    ],
)
def test_integrity_errors_are_detected(run_root, relative, old, new, check):
    replace(run_root, relative, old, new)
    report = run_audit(run_root)
    assert report["status"] == "failed"
    assert check in checks_failed(report)


@pytest.mark.parametrize(
    "relative,key",
    [
        ("raw/raw_manifest.json", "raw_proteins"),
        ("raw/raw_manifest.json", "raw_reactions"),
        ("raw/raw_manifest.json", "raw_pairs"),
        ("clustered/clustered_manifest.json", "clustered_proteins"),
        ("clustered/clustered_manifest.json", "clustered_pairs"),
        ("clustered/clustered_manifest.json", "clustered_members"),
    ],
)
def test_manifest_counts_are_measured_independently(run_root, relative, key):
    path = run_root / relative
    manifest = json.loads(path.read_text())
    manifest[key] += 1
    path.write_text(json.dumps(manifest))
    report = run_audit(run_root)
    assert report["status"] == "failed"
    assert Path(relative).stem in checks_failed(report)


@pytest.mark.parametrize("name", ["raw/raw_pairs.tsv", "clustered/pairs.tsv"])
def test_duplicate_edges_fail_even_if_manifest_count_matches(run_root, name):
    path = run_root / name
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join([lines[0], lines[1], lines[1], *lines[2:]]))
    manifest_path = run_root / (
        "raw/raw_manifest.json" if name.startswith("raw/") else "clustered/clustered_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    count_name = "raw_pairs" if name.startswith("raw/") else "clustered_pairs"
    manifest[count_name] += 1
    manifest_path.write_text(json.dumps(manifest))
    report = run_audit(run_root)
    assert count_name in checks_failed(report)


def test_file_change_during_audit_is_incomplete(run_root, monkeypatch):
    original = audit_module.Audit.reactions

    def mutate(self):
        result = original(self)
        path = self.paths["raw_proteins"]
        path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(audit_module.Audit, "reactions", mutate)
    report = run_audit(run_root)
    assert report["status"] == "incomplete"
    assert "input_stability" in checks_failed(report)
    changed = run_root / "raw/raw_proteins.fasta"
    assert report["artifact_signatures"][str(changed)] != list(audit_module.signature(changed))


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_cli_refuses_to_overwrite_an_input(run_root, capsys, alias):
    original = run_root / "raw/raw_pairs.tsv"
    before = original.read_bytes()
    output = original
    if alias != "direct":
        output = run_root / "report.json"
        if alias == "symlink":
            output.symlink_to(original)
        else:
            output.hardlink_to(original)
    code = audit_module.main(["--run-root", str(run_root), "--output", str(output)])
    assert code == 2
    assert original.read_bytes() == before
    assert "report_output" in checks_failed(json.loads(capsys.readouterr().out))


def test_cli_writes_report_and_returns_failure_for_incomplete(run_root, capsys):
    output = run_root / "audit.json"
    assert audit_module.main(["--run-root", str(run_root), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["status"] == "passed"
    capsys.readouterr()
    (run_root / "clustered/clusters.tsv").unlink()
    assert audit_module.main(["--run-root", str(run_root)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "incomplete"
