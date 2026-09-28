#!/usr/bin/env python3
"""Frozen CYP comparison: CPU controls, FusionESP, existing CIRCE, strict score import."""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
# Scoped optional dependencies; never install into the shared training env.
sys.path.insert(0, str(ROOT / ".deps/cyp-baseline-deps"))

try:
    from . import cyp_baselines as core
except ImportError:
    import cyp_baselines as core

def read_json(path):
    return json.loads(Path(path).read_text())


def load_config(path):
    settings = yaml.safe_load(Path(path).read_text())
    for key in ("benchmark", "archive", "fusion_checkpoints", "fusion_python", "existing_run"):
        path = ROOT / settings[key]
        # Resolving a venv's python symlink selects the bare system interpreter
        # and loses its installed packages. Keep the executable's venv path.
        settings[key] = str(path.absolute() if key == "fusion_python" else path.resolve())
    return settings


def freeze(settings, run):
    core.verify_bundle(settings["benchmark"])
    protocol = dict(
        version=1, settings=settings,
        benchmark_sha256=core.digest(Path(settings["benchmark"]) / "manifest.json"),
        code_sha256={name: core.digest(ROOT / "scripts" / name) for name in
                     ("run_cyp_baselines.py", "cyp_baselines.py", "cyp_fusionesp.py", "cyp_specificity.py")},
        direction="reaction-to-enzyme only", aggregation="query macro, designated positive",
        ties="exact score equality; uniform random ordering within ties; E[1/rank]",
        percentiles="100*rank/N; top p% means rank <= floor(p*N/100)",
        uncertainty="paired organism-cluster bootstrap, fixed checkpoints; exploratory unadjusted 95% intervals",
        test_status="RETROSPECTIVE: initial F3/residual CYP results were inspected before this protocol",
        selection="No CYP-based checkpoint, alpha, threshold, or hyperparameter selection",
        labels="Other candidates are UNKNOWN, not experimentally established negatives",
        alignment_source=core.ALIGNMENT_SOURCE,
        alignment_deviation="Author score formula; references reconstructed from released FusionESP train_pos. "
                            "Original BoltzCYP DataFrame membership/order not independently verified. "
                            "First nearest-substrate tie uses ascending sample_id, recorded explicitly. "
                            "These are paper-style controls, NOT a numerically exact paper reproduction.",
    )
    destination = run / "protocol.json"
    if destination.exists():
        if read_json(destination) != protocol:
            raise ValueError("Protocol/code/inputs changed: use a NEW comparison run root")
    else:
        core.write_json(destination, protocol)
    return core.digest(destination)


def existing_score(run, method, protocol_sha):
    csv_path, receipt = run / "scores" / f"{method}.csv", run / "scores" / f"{method}.json"
    if not receipt.exists():
        if csv_path.exists():
            raise ValueError(f"Uncommitted scores: {csv_path}; inspect before using a new run root")
        return None
    metadata = read_json(receipt)
    if metadata["protocol_sha256"] != protocol_sha or core.digest(csv_path) != metadata["scores_sha256"]:
        raise ValueError(f"Stale/tampered scores: {method}")
    return metadata


def save_scores(settings, run, method, rows, provenance, protocol_sha):
    rows = sorted((dict(query_id=r["query_id"], protein_id=r["protein_id"], score=float(r["score"]))
                   for r in rows), key=lambda r: (r["query_id"], r["protein_id"]))
    core.evaluate(settings["benchmark"], rows)  # strict coverage, duplicate and finite checks
    old = existing_score(run, method, protocol_sha)
    if old:
        if old["provenance"] != provenance:
            raise ValueError(f"Changed provenance for {method}; use a new run root")
        previous = [dict(query_id=r["query_id"], protein_id=r["protein_id"], score=float(r["score"]))
                    for r in core.read_rows(run / "scores" / f"{method}.csv")]
        if previous != rows:
            raise ValueError(f"Changed scores for {method}; use a new run root")
        return
    destination = run / "scores" / f"{method}.csv"
    core.write_rows(destination, rows,
                    ["query_id", "protein_id", "score"])
    core.write_json(destination.with_suffix(".json"), dict(
        protocol_sha256=protocol_sha, scores_sha256=core.digest(destination), provenance=provenance))


def prepare(settings, run, protocol_sha):
    output = run / "inputs"
    receipt = output / "complete.json"
    if receipt.exists():
        previous = read_json(receipt)
        if previous["protocol_sha256"] != protocol_sha or any(
                core.digest(output / name) != sha for name, sha in previous["files"].items()):
            raise ValueError("Prepared input content changed")
        return
    info = core.prepare_inputs(settings["archive"], settings["benchmark"], output)
    core.write_json(receipt, dict(protocol_sha256=protocol_sha, reference=info,
                                 files={p.name: core.digest(p) for p in sorted(output.glob("*.csv"))}))
    print(f"Prepared {info['reference_samples']} CYP train-positive records; no feature extraction", flush=True)


def internal_and_random(settings, run, protocol_sha):
    rows = [dict(query_id=r["query_id"], protein_id=r["protein_id"], score=0.)
            for r in core.read_rows(Path(settings["benchmark"]) / "candidates.csv")]
    save_scores(settings, run, "random", rows, {"formula": "Exact uniform-random rank expectation"}, protocol_sha)
    for method in ("f3", "residual"):
        score_file = Path(settings["existing_run"]) / "scores" / f"{method}.csv"
        receipt = read_json(score_file.with_suffix(".json"))
        if (receipt.get("status") != "frozen external evaluation; no test-time fitting" or
                receipt["scores_sha256"] != core.digest(score_file) or
                receipt["signature"]["benchmark_sha256"] != core.digest(Path(settings["benchmark"]) / "manifest.json")):
            raise ValueError(f"Invalid original score receipt: {method}")
        save_scores(settings, run, method, core.read_rows(score_file),
                    {"original_score_file": str(score_file), "original_receipt": receipt}, protocol_sha)


def controls(settings, run, protocol_sha, workers):
    import Bio
    names = ("alignment_sequence", "alignment_substrate")
    if all(existing_score(run, name, protocol_sha) for name in names):
        return
    values = core.alignment_controls(settings["benchmark"], run / "inputs", run / "controls", workers)
    provenance = dict(source=core.ALIGNMENT_SOURCE, mode="global", matrix="BLOSUM62",
                      gap_open=-11., gap_extend=-1., fingerprint="RDKFingerprint(maxPath=2)",
                      sequence_score="maximum over unique training sequences",
                      substrate_score="mean over train-positive records for first nearest substrate",
                      references_sha256=core.digest(run / "inputs/references.csv"),
                      nearest_substrates_sha256=core.digest(run / "controls/nearest_substrates.csv"),
                      reference_scope="Reconstructed FusionESP train_pos; not verified identical to BoltzCYP split/order")
    provenance["biopython_version"] = Bio.__version__
    for name, rows in values.items():
        save_scores(settings, run, name, rows, provenance, protocol_sha)


def fusion(settings, run, protocol_sha, workers):
    names = ("fusionesp_pretrained", "fusionesp_cyp_adapted")
    if all(existing_score(run, name, protocol_sha) for name in names):
        return
    if not Path(settings["fusion_checkpoints"]).is_file():
        raise FileNotFoundError("Download fusionesp_model_ckpts.tar.gz with the pinned hf command in docs/cyp_baselines.md")
    output = run / "fusion_native"
    command = [settings["fusion_python"], str(ROOT / "scripts/cyp_fusionesp.py"),
               "--data-archive", settings["archive"], "--model-archive", settings["fusion_checkpoints"],
               "--prepared", str(run / "inputs"), "--bundle", settings["benchmark"],
               "--output", str(output), "--threads", str(workers)]
    subprocess.run(command, check=True, cwd=ROOT)
    for name in names:
        score_file = output / f"{name}.csv"
        provenance = read_json(score_file.with_suffix(".json"))
        if provenance["scores_sha256"] != core.digest(score_file):
            raise ValueError(f"FusionESP output checksum mismatch: {name}")
        save_scores(settings, run, name, core.read_rows(score_file), provenance, protocol_sha)
    # Precomputed release output has no checkpoint receipt. Compare numerically,
    # but never turn similarity alone into verified training/checkpoint provenance.
    member = "data_dir/eval/outputs/ranked_predictions.csv"
    raw = dict(core.archive_members(settings["archive"], names=[member], expected_sha=core.DATA_SHA))[member]
    released = [dict(query_id=r["eval_set"].removesuffix(".csv"), protein_id=r["protein_id"], score=float(r["y_pred"]))
                for r in csv.DictReader(io.StringIO(raw.decode()))]
    core.evaluate(settings["benchmark"], released)
    release_map = {(r["query_id"], r["protein_id"]): r["score"] for r in released}
    comparisons = {}
    for name in names:
        differences = [abs(float(r["score"]) - release_map[r["query_id"], r["protein_id"]])
                       for r in core.read_rows(output / f"{name}.csv")]
        comparisons[name] = {"max_abs_difference": max(differences), "mean_abs_difference": sum(differences) / len(differences)}
    core.write_json(output / "release_prediction_audit.json", dict(
        release_member=member, release_member_sha256=hashlib.sha256(raw).hexdigest(),
        checkpoint_provenance="UNKNOWN; raw release predictions excluded from primary comparison",
        comparisons=comparisons))


def import_scores(settings, run, method, path, provenance_path, protocol_sha):
    spec = settings["methods"].get(method, {})
    if not spec.get("import_only"):
        raise ValueError("Use an explicitly planned external import-only method")
    provenance = read_json(provenance_path)
    required = ("method", "variant", "checkpoint_sha256", "source_revision", "input_manifest_sha256",
                "input_scope", "upstream_exposure", "checkpoint_selection", "generator_command", "scores_sha256")
    if any(not isinstance(provenance.get(key), str) or not provenance[key].strip() for key in required):
        raise ValueError(f"External provenance needs nonempty strings: {required}")
    if provenance["method"] != method or provenance["variant"] != spec["group"]:
        raise ValueError("External model variant does not match the frozen protocol")
    if not re.fullmatch(r"[0-9a-f]{64}", provenance["checkpoint_sha256"]):
        raise ValueError("A concrete checkpoint SHA256 is required")
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", provenance["source_revision"]):
        raise ValueError("An immutable source revision hash is required")
    if provenance["scores_sha256"] != core.digest(path):
        raise ValueError("External scores checksum mismatch")
    if provenance["input_manifest_sha256"] != core.digest(Path(settings["benchmark"]) / "manifest.json"):
        raise ValueError("External predictions refer to different benchmark inputs")
    if provenance.get("score_direction") != "higher_is_better":
        raise ValueError("Normalize to higher-is-better scores before importing")
    if provenance["checkpoint_selection"] != "fixed_without_cyp_selection":
        raise ValueError("CYP-selected checkpoints are not part of this protocol")
    save_scores(settings, run, method, core.read_rows(path), provenance, protocol_sha)


def exposure_summary(settings):
    out = {}
    expected = {r["reaction_id"] for r in core.read_rows(Path(settings["benchmark"]) / "queries.csv")}
    for source in ("reactzyme", "residual_source"):
        directory = Path(settings["existing_run"]) / "audit" / source
        if not (directory / "complete.json").exists():
            out[source] = {"status": "UNKNOWN: no completed audit"}
            continue
        receipt = read_json(directory / "complete.json")
        path = directory / "queries.csv"
        if core.digest(path) != receipt["outputs"]["queries.csv"]:
            raise ValueError(f"Changed exposure audit: {path}")
        rows = list(core.read_rows(path))
        if len(rows) != len(expected) or {r["query_id"] for r in rows} != expected:
            raise ValueError("Exposure audit query IDs do not match")
        out[source] = dict(queries=len(rows), positive_exact_sequence_matches=sum(
            r["positive_exact_sequence_match"] == "True" for r in rows),
            exact_reaction_matches="UNKNOWN" if any(r["exact_reaction_matches"] == "" for r in rows)
            else sum(int(r["exact_reaction_matches"]) > 0 for r in rows),
            audit_path=str(path), audit_sha256=core.digest(path),
            upstream_exposure=receipt["upstream_supervised_exposure"])
    return out


def report(settings, run, protocol_sha):
    manifest = core.verify_bundle(settings["benchmark"])
    output = run / "report"
    summary, per_query, pending = {}, {}, []
    for name, spec in settings["methods"].items():
        receipt = existing_score(run, name, protocol_sha)
        if not receipt:
            pending.append(name)
            continue
        metrics, rows, ranks = core.evaluate(settings["benchmark"], core.read_rows(run / "scores" / f"{name}.csv"))
        summary[name] = dict(**spec, metrics=metrics, score_receipt_sha256=core.digest(run / "scores" / f"{name}.json"))
        per_query[name] = rows
        core.write_rows(output / f"{name}.per_query.csv", rows)
        core.write_rows(output / f"{name}.rankings.csv", ranks)
    differences = {}
    if "residual" in per_query:
        for name, rows in per_query.items():
            if name != "residual":
                differences[f"residual_minus_{name}"] = core.paired_intervals(
                    rows, per_query["residual"], settings["bootstrap_samples"], settings["seed"])
    exposure = exposure_summary(settings)
    core.write_json(output / "summary.json", dict(protocol_sha256=protocol_sha, methods=summary,
        differences=differences, pending=pending, exposure=exposure, complete=not pending))
    lines = ["# CYP within-family retrieval — retrospective comparison", "",
             f"Same {manifest['query_count']} organism-specific query pools; all {manifest['candidate_pairs']:,} pairs required for every method. R→E only.",
             "Unknown candidates are not assayed negatives. Metrics recover one designated catalyst per query.",
             "Ties use exact expected ranks/MRR/Hit rates under random tie order; no alphabetical advantage.",
             "Top p% uses floor(p*N). CI: paired organism-cluster bootstrap, fixed checkpoints, exploratory/unadjusted.", "",
             "| Model | Group | MRR | Hit@1 | Hit@5 | Hit@10 | Top10% | Top20% | Top50% | Mean percentile ↓ |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for entry in summary.values():
        m = entry["metrics"]
        values = [m[k] for k in ("mrr", "hit_at_1", "hit_at_5", "hit_at_10", "hit_at_10pct", "hit_at_20pct", "hit_at_50pct", "percentile")]
        lines.append(f"| {entry['label']} | {entry['group']} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", "## Paired MRR differences", ""]
    for name, metrics in differences.items():
        m = metrics["mrr"]
        lines.append(f"- {name}: {m['delta']:+.4f}, 95% CI [{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}].")
    lines += ["", "Pending (NOT evaluated): " + (", ".join(pending) or "none"), "",
              "Alignment controls use the authors' global BLOSUM62/RDKFingerprint formula, with references reconstructed "
              "from released FusionESP train_pos. Exact BoltzCYP split/order equivalence is unverified; these are not exact paper-number reproductions.",
              "Original and CYP-adapted models are separated. Different training data prevent architecture-only causal claims.",
              "CLIPZyme/Horizyn-1/EnzymeCAGE require native external predictions; this runner does not execute those models.", "",
              "## Training exposure", ""]
    for source, audit in exposure.items():
        lines.append(f"- {source}: {audit.get('positive_exact_sequence_matches', 'UNKNOWN')}/{audit.get('queries', '?')} "
                     "positive sequences match the recorded training source. See summary.json and source audits.")
    lines += ["", "Backbone/SLEEC and untraced external pretraining exposure remain UNKNOWN. "
              "No leakage-free or experimentally validated activity claim is supported by this table."]
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "summary.md.partial"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(output / "summary.md")
    print(f"Report: {output / 'summary.md'}; {len(summary)} evaluated, {len(pending)} pending", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "prepare", "controls", "fusion", "report", "import"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmarks/cyp_baselines.yaml")
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/cyp_baselines_v1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--method")
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.stage == "import" and not all((args.method, args.scores, args.provenance)):
        parser.error("import needs --method, --scores and --provenance")
    settings = load_config(args.config)
    run = args.run_root.resolve()
    if run == Path(settings["existing_run"]) or Path(settings["existing_run"]) in run.parents:
        parser.error("Use a separate comparison run root; preserve the completed CYP run")
    run.mkdir(parents=True, exist_ok=True)
    with (run / "controller.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another comparison process owns this run root")
        protocol_sha = freeze(settings, run)
        if args.stage in ("all", "prepare", "controls", "fusion"):
            prepare(settings, run, protocol_sha)
            internal_and_random(settings, run, protocol_sha)
        if args.stage in ("all", "controls"):
            controls(settings, run, protocol_sha, args.workers)
        if args.stage in ("all", "fusion"):
            fusion(settings, run, protocol_sha, args.workers)
        if args.stage == "import":
            import_scores(settings, run, args.method, args.scores, args.provenance, protocol_sha)
        report(settings, run, protocol_sha)


if __name__ == "__main__":
    main()
