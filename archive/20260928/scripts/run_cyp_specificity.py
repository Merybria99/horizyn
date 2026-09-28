#!/usr/bin/env python3
"""Frozen-checkpoint CYP benchmark. No training, test-time selection, or job stopping."""
from __future__ import annotations

import argparse
import fcntl
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.cyp_specificity import (RELEASE_FILE, RELEASE_REPO, RELEASE_REVISION,
                                    compare_reports, digest, evaluate_scores, prepare,
                                    read_rows, verify_bundle, write_json, write_rows)


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def run_command(command, log, extra_env=None):
    print("Running:", " ".join(map(str, command)), flush=True)
    environment = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1",
                       TOKENIZERS_PARALLELISM="false")
    environment.update(extra_env or {})
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open("a") as stream:
        subprocess.run(list(map(str, command)), cwd=ROOT, env=environment,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def stage(command, outputs, signature, receipt, env=None, after_command=None):
    """A result is reusable only after successful exit and input/output verification."""
    receipt = Path(receipt)
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old["signature"] != signature or old["outputs"] != [identity(p) for p in outputs]:
            raise ValueError(f"Changed stage inputs/outputs: {receipt}; use a new run root")
        print(f"Reusing {receipt.stem}", flush=True)
        return
    pending = receipt.with_suffix(".inputs.json")
    if pending.exists() and json.loads(pending.read_text()) != signature:
        raise ValueError(f"Changed inputs to interrupted stage: {pending}")
    write_json(pending, signature)
    run_command(command, receipt.with_suffix(".log"), env)
    if after_command is not None:
        run_command(after_command, receipt.with_suffix(".log"), env)
    write_json(receipt, dict(signature=signature, outputs=[identity(p) for p in outputs]))


def preparation(settings):
    bundle = resolve(settings["benchmark"])
    archive = resolve(settings["archive"])
    if not archive.exists():
        hf = Path(sys.executable).parent / "hf"
        subprocess.run([str(hf), "download", RELEASE_REPO, RELEASE_FILE, "--repo-type", "dataset",
                        "--revision", RELEASE_REVISION, "--local-dir", str(archive.parent)], check=True)
    # Fixed public release; a valid but different tarball cannot silently redefine this test.
    if digest(archive) != "51bdb5f18b5ecf708dab1370be25ede27861db907dd30dd051f0ea63606e97d8":
        raise ValueError("CYP release archive checksum mismatch")
    manifest = prepare(archive, bundle)
    print(f"CYP prepared: {manifest['query_count']} queries; {manifest['candidate_count']} proteins; "
          f"{manifest['candidate_pairs']} query-candidate scores per method", flush=True)


def check_transformer_checkpoint(path, torch_version):
    path = Path(path)
    safe = (path / "model.safetensors").is_file() or (path / "model.safetensors.index.json").is_file()
    legacy = (path / "pytorch_model.bin").is_file() or (path / "pytorch_model.bin.index.json").is_file()
    if not safe and not legacy:
        raise FileNotFoundError(f"No Transformers model weights in {path}")
    if not safe and tuple(int(p) for p in torch_version.split(".")[:2]) < (2, 6):
        raise RuntimeError(f"{path} has only .bin weights, incompatible with safe loading on "
                           f"PyTorch {torch_version}. Use a verified safetensors copy or PyTorch >=2.6; "
                           "do not downgrade the safety check.")


def preflight(settings, args):
    paths = [resolve(settings["mmseqs"])]
    paths.extend(resolve(value) for key, value in settings["features"].items()
                 if key != "preserve_reaction_charges")
    paths.extend(ROOT / path for path in (
        ".deps/ChIRo/paper_results/RS_experiment/ChIRo/params_RS_ChIRo.json",
        ".deps/ChIRo/paper_results/RS_experiment/ChIRo/results_RS_ChIRo_seed1/best_model.pt"))
    for name in args.models:
        spec = settings["models"][name]
        paths.extend(resolve(spec[key]) for key in ("checkpoint", "config", "chemistry_schema"))
        for source in spec["sources"]:
            paths.extend(resolve(value) for value in settings["sources"][source].values())
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing benchmark inputs:\n" + "\n".join(missing))
    for key in ("prott5", "reaction_t5"):
        check_transformer_checkpoint(resolve(settings["features"][key]), version("torch"))
    if "residual" in args.models:
        from scripts.evaluate_biological_residual import fixed_test_alpha
        from scripts.run_biological_residual_paper_analysis import validate_parent_weights
        spec = settings["models"]["residual"]
        alpha = fixed_test_alpha(resolve(spec["selection"]), resolve(spec["checkpoint"]), "reactzyme_mrr")
        validate_parent_weights(resolve(settings["models"]["f3"]["checkpoint"]), resolve(spec["checkpoint"]))
        print(f"Verified frozen F3 parent; fixed validation-selected alpha={alpha:g}", flush=True)
    print("Input/checkpoint preflight passed. No training or label fitting is performed.", flush=True)


def feature_signature(settings, args):
    bundle = resolve(settings["benchmark"])
    config = settings["features"]
    signature = dict(benchmark=digest(bundle / "manifest.json"), settings=config, device=args.device,
                     runner_sha256=digest(__file__), batch_size=args.batch_size,
                     code={s: digest(ROOT / "scripts" / s) for s in (
                         "extract_prott5_residue_embeddings.py", "extract_reaction_t5v2_embeddings.py",
                         "extract_unimol2_reaction_embeddings.py", "extract_chiro_reaction_embeddings.py",
                         "cache_sleec_functional_tokens.py")})
    signature["feature_helpers"] = {path: digest(ROOT / path) for path in (
        "horizyn/biological_residual.py", "horizyn/chemistry/standardizer.py",
        "horizyn/capability/reaction_set_features.py", "wet_lab/materialize_chemistry.py")}
    signature["runtime"] = {package: version(package) for package in ("torch", "transformers", "safetensors")}
    signature["backbone_files"] = {
        key: {path.name: identity(path) for path in sorted(resolve(config[key]).iterdir())
              if path.suffix in {".json", ".safetensors", ".bin", ".model"}}
        for key in ("prott5", "reaction_t5")
    }
    return signature


def validate_prott5_shard(bundle, model, output, batch_size):
    """Check saved IDs, sequence lengths, provenance and committed completion."""
    import h5py
    import numpy as np
    from scripts import extract_prott5_residue_embeddings as extractor

    options = argparse.Namespace(
        fasta=str(bundle / "proteins.fasta"), model_name=str(model),
        max_sequence_length=1022, sequence_truncation="ends_center", rank=0, world_size=1,
        length_sort=True, padded_token_budget=True, dtype="float16", compression="none")
    records = [extractor.ProteinRecord(r.index, r.protein_id, extractor.truncate_sequence(r.sequence, 1022))
               for r in extractor.iter_fasta_records(options.fasta)]
    records.sort(key=lambda r: (len(r.sequence), r.index))
    offsets = np.cumsum([0] + [len(r.sequence) for r in records], dtype=np.int64)
    batches = extractor.iter_batches(records, batch_size, 8192, padded_token_budget=True)
    shard = extractor.get_shard_path(output / "proteins.h5", None, 0, 1)
    with h5py.File(shard, "r") as handle:
        count = extractor.validate_partial_shard(handle, options, records, offsets, batches,
                                                  expected_residue_dim=1024)
    if count != len(records):
        raise ValueError(f"Shard is not complete: {count}/{len(records)} proteins")
    return shard


def recover_missing_prott5_merge(output, signature, bundle, model, batch_size):
    """Migrate only the known pre-merge bug; changed extraction inputs still fail."""
    pending = output / "prott5.complete.inputs.json"
    if not pending.exists() or (output / "prott5.complete.json").exists():
        return
    previous = json.loads(pending.read_text())
    if previous == signature:
        return
    old_runner = "58864d5d2aacbc54e08764a131f4bdd350429942b8f88b72793da62c7d05d377"
    if previous != dict(signature, runner_sha256=old_runner):
        raise ValueError("Cannot recover ProtT5: extraction inputs or runtime changed")
    if (output / "proteins.h5").exists():
        raise ValueError("Pre-merge recovery expected a missing final protein file")
    shard = validate_prott5_shard(bundle, model, output, batch_size)
    backup = pending.with_name("prott5.complete.inputs.before-merge-fix.json")
    if backup.exists():
        raise FileExistsError(backup)
    pending.rename(backup)
    write_json(pending, signature)
    write_json(output / "prott5.merge_recovery.json", dict(
        previous_inputs=str(backup), previous_inputs_sha256=digest(backup),
        verified_completed_shard=identity(shard), recovery="missing merge; no protein re-encoding"))
    print("Recovered completed ProtT5 shard; all extraction inputs match. No re-encoding needed.", flush=True)


def reuse_protein_cache(source, output, signature, bundle):
    """Reuse only a completed cache for the same benchmark and ProtT5 weights."""
    import h5py
    receipt = json.loads((source.parent / "prott5.complete.json").read_text())
    old = receipt["signature"]
    if (receipt["outputs"] != [identity(source)] or old["benchmark"] != signature["benchmark"]
            or old["backbone_files"]["prott5"] != signature["backbone_files"]["prott5"]
            or old["code"]["extract_prott5_residue_embeddings.py"]
            != signature["code"]["extract_prott5_residue_embeddings.py"]):
        raise ValueError("Protein cache provenance does not match this benchmark/model")
    with h5py.File(source, "r") as handle:
        ids = handle["ids"].asstr()[:].tolist()
        expected = {row["protein_id"] for row in read_rows(bundle / "proteins.csv")}
        if set(ids) != expected or len(ids) != len(expected) or handle["vectors"].shape[1] != 1024:
            raise ValueError("Protein cache coverage/dimension mismatch")
    target = output / "proteins.h5"
    if target.exists() and target.resolve() != source:
        raise ValueError(f"Different protein cache already exists: {target}")
    if not target.exists():
        target.symlink_to(source)
    write_json(output / "prott5.reused.json", dict(source=identity(source), receipt=receipt))
    print(f"Reusing verified CYP protein features: {source}", flush=True)


def features(settings, args, merge_only=False):
    bundle, output = resolve(settings["benchmark"]), args.run_root / "features"
    output.mkdir(parents=True, exist_ok=True)
    verify_bundle(bundle)
    rows = [dict(reaction_id=r["reaction_id"], reaction_smiles=r["rxn"])
            for r in read_rows(bundle / "queries.csv")]
    config = settings["features"]
    preserve_charges = config.get("preserve_reaction_charges", False)
    if preserve_charges:
        from horizyn.chemistry.standardizer import Standardizer
        standardizer = Standardizer(standardize_uncharge=False)
        rows = [dict(row, reaction_smiles=standardizer.standardize_reaction(row["reaction_smiles"]))
                for row in rows]
    reaction_csv = output / "reactions.csv"
    if not reaction_csv.exists():
        write_rows(reaction_csv, rows)
    elif list(read_rows(reaction_csv)) != rows:
        raise ValueError("Feature reaction table changed")
    signature = feature_signature(settings, args)
    child_env = {"CUDA_VISIBLE_DEVICES": ""} if args.device == "cpu" else {}
    common = ["--reactions", reaction_csv, "--no-bidirectional", "--no-allow-pseudo-reactions"]
    if preserve_charges:
        common.append("--no-standardize")
    prott5_command = [sys.executable, ROOT / "scripts/extract_prott5_residue_embeddings.py",
           "--fasta", bundle / "proteins.fasta", "--output", output / "proteins.h5",
           "--model-name", resolve(config["prott5"]), "--device", args.device,
           "--batch-size", str(args.batch_size), "--max-tokens-per-batch", "8192",
           "--max-sequence-length", "1022", "--length-sort", "--padded-token-budget", "--resume"]
    # Length-sorted shards cannot use the merger's contiguous-copy FASTA order.
    # Consumers look up protein IDs; shard order preserves every ID/vector pair.
    merge_command = [*prott5_command, "--merge-only", "--device", "cpu",
                     "--merge-order", "shard", "--merge-storage", "copy"]
    if config.get("protein_cache"):
        reuse_protein_cache(resolve(config["protein_cache"]), output, signature, bundle)
    else:
        recover_missing_prott5_merge(output, signature, bundle, resolve(config["prott5"]), args.batch_size)
    if not config.get("protein_cache"):
        if merge_only or (not (output / "prott5.complete.json").exists()
                         and (output / "proteins_shards/proteins.shard00-of-01.h5").exists()):
            validate_prott5_shard(bundle, resolve(config["prott5"]), output, args.batch_size)
        stage(merge_command if merge_only else prott5_command,
              [output / "proteins.h5"], signature, output / "prott5.complete.json", child_env,
              after_command=None if merge_only else merge_command)
    if merge_only:
        return
    stage([sys.executable, ROOT / "scripts/extract_reaction_t5v2_embeddings.py", *common,
           "--output", output / "reactiont5.h5", "--model-name", resolve(config["reaction_t5"]),
           "--device", args.device, "--batch-size", "8", "--max-length", "512", "--pooling", "mean", "--force"],
          [output / "reactiont5.h5"], signature, output / "reactiont5.complete.json", child_env)
    stage([sys.executable, ROOT / "scripts/extract_unimol2_reaction_embeddings.py", *common,
           "--output", output / "unimol2.h5", "--batch-size", "16", "--dtype", "float16",
           "--skip-invalid-molecules", "--no-skip-invalid-reactions",
           "--compression", "none", "--force"], [output / "unimol2.h5"], signature,
          output / "unimol2.complete.json",
          dict(child_env, PYTHONPATH=os.pathsep.join(map(str, [ROOT / ".deps/unimol_tools", ROOT.parent / "env/unimol2_site", ROOT])),
               UNIMOL_WEIGHT_DIR=str(resolve(config["unimol_weights"]))))
    stage([sys.executable, ROOT / "scripts/extract_chiro_reaction_embeddings.py", *common,
           "--output", output / "chiro.h5", "--device", args.device, "--num-workers", "1",
           "--skip-invalid-molecules", "--no-skip-invalid-reactions", "--log-skipped-molecules",
           "--batch-size", "16", "--force"], [output / "chiro.h5"], signature,
          output / "chiro.complete.json",
          dict(child_env, PYTHONPATH=os.pathsep.join(map(str, [ROOT / ".deps/python", ROOT / ".deps/ChIRo", ROOT]))))
    for name in args.models:
        spec = settings["models"][name]
        schema = resolve(spec["chemistry_schema"])
        dictionary = resolve(config["cofactor_dictionary"])
        stage([sys.executable, ROOT / "wet_lab/materialize_chemistry.py", "--reactions", reaction_csv,
               "--schema", schema, "--cofactor-dictionary", dictionary,
               "--output", output / f"chemistry_{name}.npz"], [output / f"chemistry_{name}.npz"],
              dict(signature, schema_sha256=digest(schema), dictionary_sha256=digest(dictionary)),
              output / f"chemistry_{name}.complete.json", child_env)
    if "residual" in args.models:
        sleec = resolve(config["sleec_checkpoint"])
        stage([sys.executable, ROOT / "scripts/cache_sleec_functional_tokens.py",
               "--residue-h5", output / "proteins.h5", "--sleec-checkpoint", sleec,
               "--pairs", bundle / "encoding_pairs.csv", "--output", output / "functional_tokens.h5",
               "--top-k", "48", "--context-k", "16", "--max-tokens", "1022",
               "--batch-size", str(args.batch_size), *(["--bf16"] if args.device == "cuda" else [])],
              [output / "functional_tokens.h5"],
              dict(signature, sleec_sha256=digest(sleec), residue=identity(output / "proteins.h5")),
              output / "functional_tokens.complete.json", child_env)


def prediction_signature(settings, args, name):
    spec = settings["models"][name]
    result = dict(benchmark_sha256=digest(resolve(settings["benchmark"]) / "manifest.json"),
                  checkpoint_sha256=digest(resolve(spec["checkpoint"])),
                  config_sha256=digest(resolve(spec["config"])),
                  runner_sha256=digest(__file__),
                  device=args.device, batch_size=args.batch_size,
                  features={filename: identity(args.run_root / "features" / filename) for filename in
                            ("proteins.h5", "reactiont5.h5", "unimol2.h5", "chiro.h5")
                            + (("functional_tokens.h5",) if name == "residual" else ())},
                  chemistry=identity(args.run_root / "features" / f"chemistry_{name}.npz"))
    result["model_code_sha256"] = {str(path.relative_to(ROOT)): digest(path)
                                  for path in sorted((ROOT / "horizyn").rglob("*.py"))}
    if "selection" in spec:
        result["selection_sha256"] = digest(resolve(spec["selection"]))
        result["parent_sha256"] = digest(resolve(settings["models"]["f3"]["checkpoint"]))
    return result


def score_model(settings, args, name):
    import torch
    from horizyn.biological_residual import FunctionalTokenH5Dataset
    from horizyn.benchmarks.retrieval import (BenchmarkTask, build_reaction_inputs, cosine_scores,
                                            encode_reactions, encode_residue_targets, load_repo_checkpoint)
    from horizyn.config import load_config
    from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset

    spec = settings["models"][name]
    bundle, feat = resolve(settings["benchmark"]), args.run_root / "features"
    output = args.run_root / "scores" / f"{name}.csv"
    receipt = output.with_suffix(".json")
    signature = prediction_signature(settings, args, name)
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old["signature"] != signature or old["scores_sha256"] != digest(output):
            raise ValueError(f"Changed prediction artifacts: {name}; use a new run root")
        evaluate_scores(bundle, read_rows(output))
        print(f"Reusing predictions: {name}", flush=True)
        return
    config = load_config(str(resolve(spec["config"])))
    config.data.reaction_chemistry_vectors_path = str(feat / f"chemistry_{name}.npz")
    # Never use a training cache as a missing-feature fallback on external queries.
    for modality in ("unimol2", "chiro", "chemistry"):
        config.data[f"reaction_allow_missing_{modality}"] = False
    task = BenchmarkTask(name="CYP45", task_type="retrieval", dataset="CYP", task_label="within_family",
                         split="external", pairs=bundle / "encoding_pairs.csv", reactions=feat / "reactions.csv",
                         reaction_model_embeds_h5=feat / "reactiont5.h5",
                         reaction_unimol2_embeds_h5=feat / "unimol2.h5", reaction_chiro_embeds_h5=feat / "chiro.h5")
    reaction_data = build_reaction_inputs(task, config)
    qids = [r["reaction_id"] for r in read_rows(bundle / "queries.csv")]
    pids = [r["protein_id"] for r in read_rows(bundle / "proteins.csv")]
    if set(reaction_data.keys) != set(qids):
        raise ValueError("Reaction feature coverage differs from benchmark")
    # Report all availability masks rather than allowing silent query-specific modality loss.
    availability = {}
    for q in qids:
        sample = reaction_data[q]
        availability[q] = {key: bool(sample[key]) for key in ("has_unimol2", "has_chiro", "has_chirality", "has_reaction_chemistry") if key in sample}
        if not all(availability[q].values()):
            raise ValueError(f"Missing reaction modality for {q}: {availability[q]}")
    print(f"Loading frozen {name}; scoring {len(qids)} queries against shared candidates", flush=True)
    module, kind = load_repo_checkpoint(resolve(spec["checkpoint"]), config, args.device)
    if kind != "residue":
        raise ValueError(f"Expected residue-pooled CIRCE checkpoint, got {kind}")
    target_data = ResidueEmbedDataset(str(feat / "proteins.h5"), max_tokens=1022, truncation="ends_center")
    if set(target_data.keys) != set(pids):
        raise ValueError("Protein feature coverage differs from benchmark")
    try:
        with torch.inference_mode():
            targets = encode_residue_targets(module, target_data, pids, args.device, args.batch_size,
                                             store_on_device=False, progress_every_batches=10)
            queries = encode_reactions(module, reaction_data, qids, args.device, 8).cpu()
            scores = cosine_scores(queries, targets).cpu()
            alpha = None
            if name == "residual":
                from scripts.evaluate_biological_residual import (_biological_scores, _encode_functional_targets,
                                                                 fixed_test_alpha)
                from scripts.run_biological_residual_paper_analysis import validate_parent_weights
                validate_parent_weights(resolve(settings["models"]["f3"]["checkpoint"]), resolve(spec["checkpoint"]))
                alpha = fixed_test_alpha(resolve(spec["selection"]), resolve(spec["checkpoint"]), "reactzyme_mrr")
                residual = module.model.biological_residual
                residual.fusion.set_alpha(alpha)
                functional = FunctionalTokenH5Dataset(feat / "functional_tokens.h5")
                try:
                    encoded = _encode_functional_targets(module, functional, pids, torch.device(args.device), args.batch_size)
                    local = _biological_scores(module, reaction_data, qids, encoded, torch.device(args.device), 8, args.batch_size)
                    scores = residual.fusion.cpu()(scores, local)
                finally:
                    functional.close()
            qidx, pidx = {q: i for i, q in enumerate(qids)}, {p: i for i, p in enumerate(pids)}
            predictions = [dict(query_id=r["query_id"], protein_id=r["protein_id"],
                                score=float(scores[qidx[r["query_id"]], pidx[r["protein_id"]]]))
                           for r in read_rows(bundle / "candidates.csv")]
            evaluate_scores(bundle, predictions)
            write_rows(output, predictions)
            write_json(receipt, dict(signature=signature, scores_sha256=digest(output),
                                     alpha=alpha, availability=availability,
                                     status="frozen external evaluation; no test-time fitting"))
    finally:
        target_data.close()
        del module
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def report(settings, args):
    predictions, model_receipts, audit_receipts = {}, {}, {}
    for name in args.models:
        path = args.run_root / "scores" / f"{name}.csv"
        receipt = json.loads(path.with_suffix(".json").read_text())
        signature = prediction_signature(settings, args, name)
        # Reporting can run on CPU after GPU inference, without changing scores.
        for key in ("device", "batch_size"):
            signature[key] = receipt["signature"][key]
        if signature != receipt["signature"] or digest(path) != receipt["scores_sha256"]:
            raise ValueError(f"Stale or modified model scores: {name}")
        model_receipts[name] = receipt
        predictions[name] = path
    bundle = resolve(settings["benchmark"])
    source_names = sorted({s for m in args.models for s in settings["models"][m]["sources"]})
    exposures = {}
    for name in source_names:
        directory = args.run_root / "audit" / name
        if not (directory / "complete.json").exists():
            exposures[name] = {"status": "NOT RUN; exposure UNKNOWN"}
            continue
        receipt = json.loads((directory / "complete.json").read_text())
        if receipt["signature"]["benchmark_sha256"] != digest(bundle / "manifest.json"):
            raise ValueError(f"Different benchmark in audit {name}")
        for filename, sha in receipt["outputs"].items():
            if digest(directory / filename) != sha:
                raise ValueError(f"Modified audit output: {name}/{filename}")
        audit_receipts[name] = receipt
        predictions[f"homology_{name}"] = directory / "homology_baseline.csv"
        rows = list(read_rows(directory / "queries.csv"))
        exposures[name] = dict(queries=len(rows),
            positive_exact_sequence_matches=sum(r["positive_exact_sequence_match"] == "True" for r in rows),
            exact_reaction_matches=sum(bool(r["exact_reaction_matches"]) for r in rows),
            participant_set_matches=sum(bool(r["participant_set_matches"]) for r in rows),
            exact_pair_matches=sum(r["exact_pair_match"] == "True" for r in rows),
            representation=receipt["reaction_representation"],
            upstream_supervision="UNKNOWN")
    external = {}
    for value in args.external_scores:
        name, path = value.split("=", 1)
        if not name.replace("_", "").isalnum() or name in predictions:
            raise ValueError("External method names must be distinct alphanumeric identifiers")
        predictions[name] = resolve(path)
        external[name] = dict(path=str(predictions[name]), sha256=digest(predictions[name]),
                              provenance="User-supplied scores; model/training provenance not verified")
    output = args.run_root / "reports"
    compare_reports(bundle, predictions, output)
    write_json(output / "provenance.json", dict(models=model_receipts, audits=audit_receipts,
                                               exposure_summary=exposures, external=external))
    with (output / "summary.md").open("a") as stream:
        stream.write("\n## Training exposure\n\n")
        for name, values in exposures.items():
            stream.write(f"- {name}: {json.dumps(values, sort_keys=True)}\n")
        stream.write("\nParticipant-set matches are not exact transformation matches. "
                     "Upstream SLEEC/backbone exposure is untraced; these counts do not certify novelty.\n")
    print(f"Report: {output / 'summary.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "preflight", "audit", "merge-proteins", "features", "score", "report", "all"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmarks/cyp_specificity.yaml")
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/cyp_specificity_v1")
    parser.add_argument("--models", nargs="+", choices=("f3", "residual", "circe_v2"), default=["f3", "residual"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-homology-audit", action="store_true", help="Exact audit remains; homology is explicitly UNKNOWN")
    parser.add_argument("--external-scores", action="append", default=[], metavar="NAME=CSV",
                        help="External method adapter: query_id,protein_id,score; exact full pool required")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("Batch size and threads must be positive")
    args.run_root = args.run_root.resolve()
    os.chdir(ROOT)
    settings = yaml.safe_load(args.config.read_text())
    args.run_root.mkdir(parents=True, exist_ok=True)
    with (args.run_root / "controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This benchmark run root is already active")
        os.environ.update(OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
                          MKL_NUM_THREADS=str(args.threads), WANDB_MODE="disabled")
        if args.stage in {"all", "prepare"}:
            preparation(settings)
        verify_bundle(resolve(settings["benchmark"]))
        if args.stage in {"all", "preflight"}:
            preflight(settings, args)
        if args.stage in {"all", "audit"}:
            from scripts.cyp_specificity_audit import audit_source
            sources = sorted({s for m in args.models for s in settings["models"][m]["sources"]})
            for name in sources:
                print(f"Auditing specified training source: {name}", flush=True)
                source = {k: resolve(v) for k, v in settings["sources"][name].items()}
                result = audit_source(resolve(settings["benchmark"]), source, args.run_root / "audit" / name,
                                      resolve(settings["mmseqs"]), args.threads, not args.skip_homology_audit)
                print(f"Audit complete: {name}; {result['training_proteins']} proteins, "
                      f"{result['valid_reactions']} valid/{result['invalid_reactions']} invalid reactions", flush=True)
        if args.stage in {"all", "features", "merge-proteins"}:
            features(settings, args, merge_only=args.stage == "merge-proteins")
        if args.stage in {"all", "score"}:
            import torch
            torch.set_num_threads(args.threads)
            torch.set_float32_matmul_precision("highest")
            for name in args.models:
                score_model(settings, args, name)
        if args.stage in {"all", "report"}:
            report(settings, args)


if __name__ == "__main__":
    main()
