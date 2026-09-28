#!/usr/bin/env python3
"""Repair only the known disabled-pilot candidate-set error, preserving caches.

Default is read-only verification. --apply backs up config/state, verifies all
saved artifact signatures, and migrates only the audited two-line controller
change plus the failed pilot's config binding. Never marks a failed stage passed.
Run on the execution node after the verified stop helper, before relaunching.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.horizyn1_circe_v2_pipeline import signature

SCHEMA = "horizyn1_disabled_pilot_candidate_set_repair_v1"
OLD_CONTROLLER_SHA = "96b5889f41f42b695f25d3f91e0166f3428fce0f0fe4ab3f050b46eb457a384a"
CONTROLLER = ROOT / "scripts/horizyn1_circe_v2_pipeline.py"
COMPLETE = {"preflight", "prepare", "labels", "index", "pilot_selection", "pilot_prott5",
            "pilot_reactiont5v2", "pilot_unimol2", "pilot_chiro", "pilot_chemistry", "pilot_index"}
SIG_KEYS = {"path", "size", "inode", "mtime_ns", "ctime_ns"}
ERROR = "validation_retrieval_candidate_set must be one of: validation, screening, custom"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def payload(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def publish(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as out:
        temporary = Path(out.name)
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)


def controller_change():
    current = CONTROLLER.read_bytes()
    original = current
    replacements = (
        (b'data["validation_retrieval_candidate_set"] = "validation"',
         b'data["validation_retrieval_candidate_set"] = "all"'),
        (b'validation_retrieval_candidate_set="validation", validation_retrieval_candidate_ids_path=None,',
         b'validation_retrieval_candidate_set="all", validation_retrieval_candidate_ids_path=None,'),
    )
    for fixed, old in replacements:
        if original.count(fixed) != 1:
            raise ValueError("Controller is not the exact audited two-line pilot fix")
        original = original.replace(fixed, old, 1)
    if sha(original) != OLD_CONTROLLER_SHA:
        raise ValueError("Controller contains changes beyond the audited pilot fix; refusing cache migration")
    return original, current


def artifact_records(value):
    if isinstance(value, dict):
        if SIG_KEYS <= value.keys():
            yield value
        else:
            for child in value.values():
                yield from artifact_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from artifact_records(child)


def validate_artifacts(documents, *, except_path=None):
    seen = {}
    for value in documents.values():
        for record in artifact_records(value):
            path = record["path"]
            if path == except_path:
                continue
            expected = {key: record[key] for key in SIG_KEYS}
            actual = seen.setdefault(path, signature(Path(path)))
            if expected != actual:
                raise ValueError(f"Saved artifact changed; refusing repair: {path}")


def corrected_config(data):
    import yaml
    value = yaml.safe_load(data)
    if value["training"].get("validation_enabled") is not False or value["training"].get("validation_retrieval_metrics") is not False:
        raise ValueError("Repair is only for a validation-disabled pilot")
    for section in ("data", "training"):
        if value[section].get("validation_retrieval_candidate_set") != "all":
            raise ValueError("Expected exactly the failed pilot candidate-set configuration")
        value[section]["validation_retrieval_candidate_set"] = "validation"
    return yaml.safe_dump(value, sort_keys=False).encode()


def verify_regenerated_config(run, documents, fixed):
    # A deterministic reconstruction can bind the current config even when its
    # timestamps drifted. Do not infer historical byte identity from its stats.
    from scripts.horizyn1_circe_v2_reaction_holdout import Pipeline, parse_args
    profile = documents["state/preflight.json"]["profile"]
    binding = documents["state/dataset_protocol.json"]
    args = parse_args(["--profile", profile, "--run-root", str(run), "--seed", str(binding["seed"])])
    with tempfile.TemporaryDirectory(prefix="verify-pilot-config-") as temporary:
        path = Path(temporary) / "train.yaml"
        Pipeline(args).config(path, pilot=run / "pilot")
        if path.read_bytes() != fixed:
            raise ValueError("Pilot config differs from the regenerated launch configuration beyond the two approved values")


def inspect_run(run):
    original_controller, current_controller = controller_change()
    files = {f"state/{p.name}": p.read_bytes() for p in (run / "state").glob("*.json")}
    expected = {f"state/{name}.json" for name in COMPLETE | {"dataset_protocol", "pilot_training"}}
    if set(files) != expected:
        raise ValueError("Expected only completed preparation/pilot features and failed pilot training; unexpected stages")
    documents = {name: json.loads(data) for name, data in files.items()}
    binding = documents["state/dataset_protocol.json"]
    if binding.get("dataset_protocol") != "horizyn80_clustered_reaction_holdout_v1":
        raise ValueError("Not a reaction-held-out run")
    for name in COMPLETE:
        if documents[f"state/{name}.json"].get("status") != "complete":
            raise ValueError(f"Expected completed {name}")
    failed = documents["state/pilot_training.json"]
    if failed.get("status") != "failed" or "pilot_training failed with exit code 1" not in failed.get("error", ""):
        raise ValueError("Expected the failed pilot training stage")
    log = run / "logs/pilot_training.log"
    if ERROR not in log.read_text() or any((run / "pilot/checkpoints").rglob("*.ckpt")):
        raise ValueError("Expected initialization failure before any pilot checkpoint")
    for name, value in documents.items():
        if "fingerprint" in value and value["fingerprint"].get("controller_sha256") != OLD_CONTROLLER_SHA:
            raise ValueError(f"Unexpected historical controller binding: {name}")
    config_path = run / "pilot/train.yaml"
    if sum(r["path"] == str(config_path) for r in failed["fingerprint"]["inputs"]) != 1:
        raise ValueError("Pilot config must have exactly one saved input binding")
    validate_artifacts(documents, except_path=str(config_path))
    saved_config = next(r for r in failed["fingerprint"]["inputs"] if r["path"] == str(config_path))
    actual_config = signature(config_path)
    if any(actual_config[key] != saved_config[key] for key in ("path", "inode", "size")):
        raise ValueError("Pilot config identity/size changed; refusing automatic repair")
    files["pilot/train.yaml"] = config_path.read_bytes()
    fixed = corrected_config(files["pilot/train.yaml"])
    verify_regenerated_config(run, documents, fixed)
    return files, fixed, original_controller, current_controller, signature(log)


@contextmanager
def stopped_run(run):
    if not (run / "pipeline.lock").is_file():
        raise ValueError("Missing existing run lock")
    with (run / "pipeline.lock").open("r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Pipeline is running; stop it on its execution node before repair") from None
        yield


def repair(run_root, *, apply=False):
    run = Path(run_root).resolve()
    bundle = run / "metadata_repairs/pilot_candidate_set_v1"
    with stopped_run(run):
        original_controller, current_controller = controller_change()
        plan_path = bundle / "plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if (plan.get("schema") != SCHEMA or plan.get("run_root") != str(run)
                    or plan.get("controller_after_sha256") != sha(current_controller)):
                raise ValueError("Existing repair bundle does not match this run/controller")
            if (bundle / "completed.json").exists():
                return {"status": "already_applied", "backup": str(bundle)}
            files = {}
            for name, digest in plan["original_sha256"].items():
                if name not in {f"state/{s}.json" for s in COMPLETE | {"dataset_protocol", "pilot_training"}} | {"pilot/train.yaml"}:
                    raise ValueError("Unexpected repair target")
                data = (bundle / "original" / name).read_bytes()
                if sha(data) != digest:
                    raise ValueError(f"Repair backup changed: {name}")
                files[name] = data
            fixed = corrected_config(files["pilot/train.yaml"])
        else:
            files, fixed, original_controller, current_controller, log_signature = inspect_run(run)
            plan = {"schema": SCHEMA, "run_root": str(run),
                    "controller_before_sha256": sha(original_controller),
                    "controller_after_sha256": sha(current_controller),
                    "original_sha256": {name: sha(data) for name, data in files.items()},
                    "failure_log": log_signature,
                    "config_before_signature": signature(run / "pilot/train.yaml"),
                    "config_verification": "Current bytes match deterministic regeneration except the two approved values; timestamp drift is not historical byte-identity evidence.",
                    "policy": "Only two disabled-pilot candidate values and controller/config bindings change; all artifacts and stage statuses are preserved."}
            if not apply:
                return {"status": "verified", "completed_stages_preserved": len(COMPLETE), "backup": str(bundle)}
            if bundle.exists():
                raise ValueError("Incomplete/unrecognized repair bundle; inspect it before proceeding")
            bundle.mkdir(parents=True)
            for name, data in files.items():
                publish(bundle / "original" / name, data)
            publish(bundle / "controller_before.py", original_controller)
            publish(bundle / "controller_after.py", current_controller)
            publish(plan_path, payload(plan))

        if not apply:
            return {"status": "pending_repair", "backup": str(bundle)}
        documents = {name: json.loads(data) for name, data in files.items() if name.startswith("state/")}
        config_path = run / "pilot/train.yaml"
        current_config = config_path.read_bytes()
        if current_config not in (files["pilot/train.yaml"], fixed):
            raise ValueError("Pilot config changed outside this repair")
        validate_artifacts(documents, except_path=str(config_path))
        if signature(run / "logs/pilot_training.log") != plan["failure_log"]:
            raise ValueError("Pilot training log changed since repair verification")
        if any((run / "pilot/checkpoints").rglob("*.ckpt")):
            raise ValueError("Pilot checkpoints appeared during repair")

        def updates():
            result = copy.deepcopy(documents)
            for value in result.values():
                if "fingerprint" in value:
                    value["fingerprint"]["controller_sha256"] = sha(current_controller)
            inputs = result["state/pilot_training.json"]["fingerprint"]["inputs"]
            for i, record in enumerate(inputs):
                if record["path"] == str(config_path):
                    inputs[i] = signature(config_path)
            return {name: payload(value) for name, value in result.items()}

        # Validate the entire write set first. If interrupted, accept only bytes
        # from the original backup or this same migration, never arbitrary edits.
        candidates = updates() if current_config == fixed else files
        for name in documents:
            if (run / name).read_bytes() not in (files[name], candidates[name]):
                raise ValueError(f"Run state changed outside this repair: {name}")
        if current_config != fixed:
            if signature(config_path) != plan["config_before_signature"]:
                raise ValueError("Original pilot config metadata changed")
            publish(config_path, fixed)
        updated = updates()
        for name, data in updated.items():
            if (run / name).read_bytes() != data:
                publish(run / name, data)
        validate_artifacts({name: json.loads(data) for name, data in updated.items()})
        result = {"status": "applied", "backup": str(bundle), "completed_stages_preserved": len(COMPLETE),
                  "pilot_training_status": "failed; will retry normally", "config": signature(config_path)}
        publish(bundle / "completed.json", payload(result))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(repair(args.run_root, apply=args.apply), indent=2))


if __name__ == "__main__":
    main()
