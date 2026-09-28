#!/usr/bin/env python3
"""Package campaign source changes relative to the saved dirty-workspace baseline.

This deliberately does not stage or commit unrelated pre-existing user changes.
"""
from __future__ import annotations
import argparse
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile


def sha(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    campaign = args.campaign.resolve()
    initial = json.loads((campaign / "initial_source_manifest.json").read_text())
    baseline = {}
    with tarfile.open(campaign / "initial_source.tar.gz", "r:gz") as archive:
        for row in initial:
            payload = archive.extractfile(row["path"]).read()
            if sha(payload) != row["sha256"]:
                raise ValueError(f"Initial snapshot mismatch: {row['path']}")
            baseline[row["path"]] = payload
    candidates = set(baseline)
    # New files belong to explicit campaign namespaces. Other untracked files
    # and existing documentation absent from the snapshot are not adopted.
    patterns = ["horizyn/generalization_*.py", "horizyn/semantic_*.py",
        "scripts/generalization_*.py", "tests/unit/test_generalization_*.py",
        "tests/test_generalization_*.py",
        "tests/unit/test_semantic_*.py", "tests/unit/test_wet_lab_generalization.py",
        "documents/generalization_*.md", "documents/code_review_coverage.md",
        "docs/CROSS_PAPER_RETRAINING.md",
        "wet_lab/generalization.py", "wet_lab/GENERALIZATION.md",
        "wet_lab/configs/phase2_generalization.yaml", "wet_lab/configs/phase4_generalization.yaml", "findings.md"]
    for pattern in patterns:
        candidates.update(str(path.relative_to(repo)) for path in repo.glob(pattern) if path.is_file())
    changed, patch = [], []
    for name in sorted(candidates):
        path = repo / name
        before = baseline.get(name)
        after = path.read_bytes() if path.is_file() else None
        if before == after:
            continue
        old_text = [] if before is None else before.decode("utf-8").splitlines(keepends=True)
        new_text = [] if after is None else after.decode("utf-8").splitlines(keepends=True)
        patch.append(f"diff --git a/{name} b/{name}\n")
        if before is None:
            mode = "100755" if path.stat().st_mode & 0o111 else "100644"
            patch.append(f"new file mode {mode}\n")
        elif after is None:
            patch.append("deleted file mode 100644\n")
        for line in difflib.unified_diff(old_text, new_text,
            fromfile="/dev/null" if before is None else f"a/{name}",
            tofile="/dev/null" if after is None else f"b/{name}"):
            patch.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
        changed.append(dict(path=name, status="added" if before is None else "removed" if after is None else "modified",
            before_sha256=None if before is None else sha(before),
            after_sha256=None if after is None else sha(after),
            bytes=None if after is None else len(after)))
    out = campaign / "source_changes"
    out.mkdir(exist_ok=True)
    (out / "campaign.patch").write_text("".join(patch))
    with tarfile.open(out / "changed_source.tar.gz", "w:gz") as archive:
        for row in changed:
            if row["status"] != "removed":
                archive.add(repo / row["path"], arcname=row["path"], recursive=False)
    manifest = dict(schema="campaign_source_delta_v1",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        branch="research/f3-circev2-generalization-20260919",
        reference="Saved initial working tree, including pre-existing user changes; not clean git HEAD.",
        initial_manifest_sha256=sha((campaign / "initial_source_manifest.json").read_bytes()),
        initial_snapshot_sha256=sha((campaign / "initial_source.tar.gz").read_bytes()),
        patch_sha256=sha((out / "campaign.patch").read_bytes()),
        changed_archive_sha256=sha((out / "changed_source.tar.gz").read_bytes()),
        counts={state:sum(row["status"]==state for row in changed) for state in ("added","modified","removed")},
        files=changed,
        outside_nested_repo={"workspace_findings":str(repo.parent / "findings.md"),
            "sha256":sha((repo.parent / "findings.md").read_bytes())},
        limitations="Artifacts, data, model checkpoints, downloaded primary sources and campaign-local scripts remain in the campaign directory, outside this product-source delta. Existing files not in the initial source snapshot are not attributed as campaign edits.")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (out / "README.md").write_text(
        "# Campaign source changes\n\n"
        "`campaign.patch` compares the final campaign source against the saved initial working tree. "
        "That baseline includes substantial user work which predates this campaign. Applying the patch "
        "directly to clean git HEAD is not supported. `changed_source.tar.gz` contains final versions of "
        "only added or changed campaign source files; `manifest.json` records before/after hashes. "
        "No unrelated user changes were staged, reset, stashed or committed. Both repositories remain "
        "on the dedicated research branch. Model/data/evaluation artifacts are preserved separately in "
        "the campaign directory. Regenerate with `scripts/generalization_snapshot_changes.py` after "
        "further campaign edits. The generator validates a patch round trip in a temporary directory; "
        "`roundtrip.json` binds that check to this exact patch and manifest.\n")
    # Check the deliverable itself, without touching the actual working tree.
    with tempfile.TemporaryDirectory(prefix="enzyme-campaign-patch-") as temporary:
        destination = Path(temporary)
        with tarfile.open(campaign / "initial_source.tar.gz", "r:gz") as archive:
            archive.extractall(destination, filter="data")
        for command in (["git", "apply", "--check", str(out / "campaign.patch")],
                        ["git", "apply", str(out / "campaign.patch")]):
            subprocess.run(command, cwd=destination, check=True, capture_output=True, text=True)
        expected = {name: sha(value) for name, value in baseline.items()}
        for row in changed:
            if row["status"] == "removed":
                expected.pop(row["path"], None)
                if (destination / row["path"]).exists():
                    raise ValueError(f"Round-trip deletion failed: {row['path']}")
            else:
                expected[row["path"]] = row["after_sha256"]
        for name, checksum in expected.items():
            if sha((destination / name).read_bytes()) != checksum:
                raise ValueError(f"Round-trip content mismatch: {name}")
    (out / "roundtrip.json").write_text(json.dumps(dict(
        schema="campaign_patch_roundtrip_v1", all_pass=True,
        patch_sha256=manifest["patch_sha256"],
        manifest_sha256=sha((out / "manifest.json").read_bytes()),
        changed_files_verified=len(changed), total_files_verified=len(expected),
        procedure="Extract initial working-tree archive into temporary directory; git apply --check; git apply; verify all changed and unchanged expected content hashes. Actual repository untouched."), indent=2)+"\n")
    print(json.dumps(dict(output=str(out), counts=manifest["counts"],
        patch_sha256=manifest["patch_sha256"], roundtrip_pass=True,
        roundtrip_files=len(expected))))


if __name__ == "__main__":
    main()
