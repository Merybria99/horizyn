#!/usr/bin/env python3
"""Run the complete Case1 restricted-setting experiment with F3."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from compare_ranked_candidates import compare_rankings
from prepare_candidate_pool import build_candidate_pool


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
POOL_DIR = SCRIPT_DIR / "candidate_pool"
RESIDUE_H5 = POOL_DIR / "proteins_prott5_residue.h5"
CONFIG_PATH = SCRIPT_DIR / "query_f3.yaml"
SOURCE_TABLE = SCRIPT_DIR.parent / "sequence_pool/final_entry_sequences.csv"
PROTT5_SNAPSHOT = (
    REPOSITORY_ROOT / "hf_cache/hub/models--Rostlab--prot_t5_xl_half_uniref50-enc/snapshots/"
    "94a6abc029ae13029317b140b7424e012bf8dfbf"
)


def _runtime_environment(gpu: int) -> dict[str, str]:
    runtime = SCRIPT_DIR / "runtime"
    directories = {
        "HOME": runtime / "home",
        "HF_HOME": runtime / "cache/huggingface",
        "TORCH_HOME": runtime / "cache/torch",
        "XDG_CACHE_HOME": runtime / "cache/xdg",
        "XDG_CONFIG_HOME": runtime / "config/xdg",
        "XDG_STATE_HOME": runtime / "state/xdg",
        "MPLCONFIGDIR": runtime / "config/matplotlib",
        "TMPDIR": runtime / "tmp",
    }
    environment = os.environ.copy()
    for variable, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        environment[variable] = str(directory)
    environment["TRANSFORMERS_CACHE"] = str(directories["HF_HOME"] / "transformers")
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    return environment


def _run_logged(command: list[str], log_path: Path, environment: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}; log: {log_path}\n"
            + "\n".join(tail)
        )


def _extract_residue_embeddings(environment: dict[str, str], *, force: bool, python: Path) -> None:
    if RESIDUE_H5.exists() and not force:
        print(f"Using existing residue embeddings: {RESIDUE_H5}", flush=True)
        return
    if not PROTT5_SNAPSHOT.exists():
        raise FileNotFoundError(f"Local ProtT5 snapshot is missing: {PROTT5_SNAPSHOT}")
    if force:
        RESIDUE_H5.unlink(missing_ok=True)
        shutil.rmtree(POOL_DIR / "prott5_shards", ignore_errors=True)

    command = [
        str(python),
        "scripts/extract_prott5_residue_embeddings.py",
        "--fasta",
        str(POOL_DIR / "proteins.fasta"),
        "--output",
        str(RESIDUE_H5),
        "--tmp-dir",
        str(POOL_DIR / "prott5_shards"),
        "--model-name",
        str(PROTT5_SNAPSHOT),
        "--device",
        "cuda:0",
        "--batch-size",
        "16",
        "--max-tokens-per-batch",
        "12000",
        "--dtype",
        "float16",
        "--compression",
        "none",
        "--progress-every",
        "10",
        "--checkpoint-every",
        "10",
        "--merged-ids-output",
        str(POOL_DIR / "candidate_ids_prott5_order.txt"),
        "--cleanup-shards",
        "--resume",
    ]
    _run_logged(command, SCRIPT_DIR / "logs/prott5_extraction.log", environment)
    _run_logged(
        command + ["--merge-only"],
        SCRIPT_DIR / "logs/prott5_merge.log",
        environment,
    )
    if not RESIDUE_H5.exists():
        raise RuntimeError(f"ProtT5 merge did not create the expected HDF5: {RESIDUE_H5}")


def run(*, gpu: int, force_embeddings: bool, force_features: bool) -> Path:
    manifest = build_candidate_pool(SOURCE_TABLE, POOL_DIR, expected_count=144)
    print(
        f"Prepared {manifest['candidate_rows']} Homolog rows "
        f"({manifest['unique_sequences']} unique sequences).",
        flush=True,
    )

    environment = _runtime_environment(gpu)
    query_python = REPOSITORY_ROOT / ".capability-run-py/bin/python"
    if not query_python.exists():
        query_python = Path(sys.executable)
    extractor_python = REPOSITORY_ROOT / "env/bin/python"
    if not extractor_python.exists():
        extractor_python = query_python
    _extract_residue_embeddings(
        environment,
        force=force_embeddings,
        python=extractor_python,
    )

    command = [
        str(query_python),
        "-m",
        "wet_lab.query",
        "--config",
        str(CONFIG_PATH),
    ]
    if force_features:
        command.append("--force-features")
    query_log = SCRIPT_DIR / "logs/f3_query.log"
    _run_logged(command, query_log, environment)

    result_candidates = sorted(
        (SCRIPT_DIR / "runs").glob("tagatose_4_epimerase_*/results.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not result_candidates:
        raise RuntimeError(f"Query completed but no results.json was found; inspect {query_log}")
    result_path = result_candidates[0]
    summary, _ = compare_rankings(
        result_path,
        SOURCE_TABLE,
        SCRIPT_DIR / "results",
        top_k=25,
    )
    experiment = {
        "schema_version": "case1_restricted_experiment_v1",
        "gpu": gpu,
        "candidate_manifest": manifest,
        "query_results": str(result_path),
        "comparison_report": str(SCRIPT_DIR / "results/report.md"),
        "comparison_summary": summary,
    }
    experiment_path = SCRIPT_DIR / "experiment_manifest.json"
    with experiment_path.open("w", encoding="utf-8") as handle:
        json.dump(experiment, handle, indent=2)
    print(f"Comparison report: {SCRIPT_DIR / 'results/report.md'}", flush=True)
    return experiment_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index")
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = run(
        gpu=args.gpu,
        force_embeddings=args.force_embeddings,
        force_features=args.force_features,
    )
    print(f"Experiment manifest: {path}")


if __name__ == "__main__":
    main()
