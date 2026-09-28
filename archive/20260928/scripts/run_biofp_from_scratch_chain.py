#!/usr/bin/env python3
"""Run the fresh BioFP enzyme/reaction retrieval chain end to end."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CAP_PY = Path("/datastor2/deep-proteins/EnzymeDiscovery/.capability-run-py/bin/python")
ENV_PY = Path("/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python")
ENV_SITE = Path("/datastor2/deep-proteins/EnzymeDiscovery/env/lib/python3.12/site-packages")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-root", default="outputs/biofp_from_scratch_chains")
    parser.add_argument("--wandb-project", default="horizyn-biofp-from-scratch-chain")
    parser.add_argument("--wandb-entity", default="omnai")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument(
        "--start-stage",
        type=int,
        default=1,
        help="Resume from this numeric stage prefix, e.g. 11 resumes at 11_extract_prott5_residue_embeddings.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-fetch-uniprotkb-cofactors", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_name = args.run_name or f"biofp-from-scratch-{timestamp()}"
        self.run_dir = (ROOT / args.run_root / self.run_name).resolve()
        self.data_root = self.run_dir / "data/standardized/retrieval_training_source_collapse"
        self.cap_dir = self.run_dir / "data/processed/capability_features/train_exact_rhea_reconstructed"
        self.config_dir = self.run_dir / "configs"
        self.log_dir = self.run_dir / "logs"
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.results_dir = self.run_dir / "results"
        self.cache_dir = self.run_dir / "cache"
        self.home_dir = self.run_dir / "home"
        run_hash = hashlib.sha1(self.run_name.encode("utf-8")).hexdigest()[:10]
        self.tmp_dir = Path("/datastor2/deep-proteins/tmp") / f"hz_{run_hash}"
        self.status_path = self.run_dir / "stage_status.jsonl"
        for path in (
            self.data_root,
            self.cap_dir,
            self.config_dir,
            self.log_dir,
            self.ckpt_dir,
            self.results_dir,
            self.cache_dir,
            self.home_dir,
            self.tmp_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(ROOT)
        env["HOME"] = str(self.home_dir)
        env["TMPDIR"] = str(self.tmp_dir)
        env["TEMP"] = str(self.tmp_dir)
        env["TMP"] = str(self.tmp_dir)
        env["HF_HOME"] = str(self.cache_dir / "huggingface")
        env["HF_DATASETS_CACHE"] = str(self.cache_dir / "huggingface/datasets")
        env["TRANSFORMERS_CACHE"] = str(self.cache_dir / "huggingface/transformers")
        env["TORCH_HOME"] = str(self.cache_dir / "torch")
        env["PIP_CACHE_DIR"] = str(self.cache_dir / "pip")
        env["MPLCONFIGDIR"] = str(self.cache_dir / "matplotlib")
        env["XDG_CACHE_HOME"] = str(self.cache_dir / "xdg/cache")
        env["XDG_CONFIG_HOME"] = str(self.cache_dir / "xdg/config")
        env["XDG_STATE_HOME"] = str(self.cache_dir / "xdg/state")
        env["WANDB_ROOT"] = str(self.run_dir / "wandb")
        env["WANDB_DIR"] = str(self.run_dir / "wandb")
        env["WANDB_DATA_DIR"] = str(self.run_dir / "wandb/data")
        env["WANDB_CACHE_DIR"] = str(self.run_dir / "wandb/cache")
        env["WANDB_CONFIG_DIR"] = str(self.run_dir / "wandb/config")
        env["TOKENIZERS_PARALLELISM"] = "false"
        if extra:
            env.update(extra)
        return env

    def env_with_site(self, *paths: Path) -> dict[str, str]:
        parts = [str(path) for path in paths if path.exists()]
        parts.extend([str(ENV_SITE), str(ROOT)])
        env = self.env()
        env["PYTHONPATH"] = ":".join(parts)
        return env

    def record(self, payload: dict[str, Any]) -> None:
        payload = {"time": datetime.now().isoformat(timespec="seconds"), **payload}
        with self.status_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def should_run_stage(self, name: str) -> bool:
        try:
            stage_num = int(name.split("_", 1)[0])
        except ValueError:
            return True
        return stage_num >= self.args.start_stage

    def run(self, name: str, cmd: list[str], *, env: dict[str, str] | None = None) -> None:
        log_path = self.log_dir / f"{name}.log"
        if not self.should_run_stage(name):
            self.record({"stage": name, "event": "skipped_resume", "start_stage": self.args.start_stage})
            print(f"[{datetime.now().isoformat(timespec='seconds')}] SKIP {name} start_stage={self.args.start_stage}", flush=True)
            return
        self.record({"stage": name, "event": "start", "cmd": cmd, "log": str(log_path)})
        print(f"[{datetime.now().isoformat(timespec='seconds')}] START {name}", flush=True)
        if self.args.dry_run:
            print(" ".join(cmd), flush=True)
            self.record({"stage": name, "event": "dry_run"})
            return
        start = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env or self.env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        elapsed = time.time() - start
        self.record({"stage": name, "event": "end", "returncode": proc.returncode, "elapsed_sec": elapsed})
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] END {name} rc={proc.returncode} elapsed={elapsed:.1f}s",
            flush=True,
        )
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

    def run_parallel(self, name: str, commands: list[tuple[list[str], dict[str, str], Path]]) -> None:
        if not self.should_run_stage(name):
            self.record({"stage": name, "event": "skipped_resume", "start_stage": self.args.start_stage})
            print(f"[{datetime.now().isoformat(timespec='seconds')}] SKIP {name} start_stage={self.args.start_stage}", flush=True)
            return
        self.record({"stage": name, "event": "start", "workers": len(commands)})
        print(f"[{datetime.now().isoformat(timespec='seconds')}] START {name} workers={len(commands)}", flush=True)
        if self.args.dry_run:
            for cmd, _, _ in commands:
                print(" ".join(cmd), flush=True)
            self.record({"stage": name, "event": "dry_run"})
            return
        processes = []
        for cmd, env, log_path in commands:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((proc, log, cmd, log_path))
        failed: list[tuple[int, list[str], Path]] = []
        for proc, log, cmd, log_path in processes:
            rc = proc.wait()
            log.close()
            if rc != 0:
                failed.append((rc, cmd, log_path))
        if failed:
            self.record({"stage": name, "event": "failed", "failed": [(rc, str(log)) for rc, _, log in failed]})
            rc, cmd, log_path = failed[0]
            raise subprocess.CalledProcessError(rc, cmd, output=f"see {log_path}")
        self.record({"stage": name, "event": "end", "returncode": 0})
        print(f"[{datetime.now().isoformat(timespec='seconds')}] END {name}", flush=True)

    def write_yaml(self, path: Path, payload: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def latest_checkpoint(self, checkpoint_dir: Path) -> Path:
        candidates: dict[Path, Path] = {}
        for pattern in ("last*.ckpt", "*.ckpt"):
            for path in checkpoint_dir.glob(pattern):
                if path.is_file():
                    candidates[path.resolve()] = path
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
        return max(candidates.values(), key=lambda path: (path.stat().st_mtime, path.name))

    def latest_checkpoint_for_command(self, checkpoint_dir: Path, default_name: str = "last.ckpt") -> Path:
        try:
            return self.latest_checkpoint(checkpoint_dir)
        except FileNotFoundError:
            if self.args.dry_run:
                return checkpoint_dir / default_name
            raise

    def update_yaml_value(self, path: Path, keys: tuple[str, ...], value: Any) -> None:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        cursor = payload
        for key in keys[:-1]:
            cursor = cursor[key]
        cursor[keys[-1]] = value
        self.write_yaml(path, payload)

    def load_template(self, path: str) -> dict[str, Any]:
        with (ROOT / path).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    @property
    def train_pairs(self) -> Path:
        return self.data_root / "train_exact/train_pairs_valid_rxn_rhea_reconstructed.csv"

    @property
    def train_rxns(self) -> Path:
        return self.data_root / "train_exact/train_rxns_valid_rxn_rhea_reconstructed.csv"

    @property
    def val_pairs(self) -> Path:
        return self.data_root / "validation/clipzyme_eval/pairs_horizyn.csv"

    @property
    def val_rxns(self) -> Path:
        return self.data_root / "validation/clipzyme_eval/reactions.csv"

    @property
    def prot5_h5(self) -> Path:
        return self.data_root / "train_exact/fit_proteins_with_clipzyme_eval_prott5_residue.h5"

    @property
    def reactiont5_h5(self) -> Path:
        return self.data_root / "train_exact/rxns_reactiont5v2_forward_768_exact_rhea_plus_clipzyme_eval.h5"

    @property
    def unimol_h5(self) -> Path:
        return self.data_root / "train_exact/rxns_unimol2_84m_exact_rhea_plus_clipzyme_eval.h5"

    @property
    def chiro_h5(self) -> Path:
        return self.data_root / "train_exact/rxns_chiro_256_exact_rhea_plus_clipzyme_eval.h5"

    def build_configs(self) -> tuple[Path, Path, Path]:
        biofp_targets = self.cap_dir / "enzyme_biofp_soft_targets.npz"
        biofp_vocab = self.cap_dir / "enzyme_biofp_vocab.json"

        pretrain = {
            "seed": 42,
            "logging": {
                "log_dir": str(self.log_dir / "enzyme_biofp_pretrain"),
                "checkpoint_dir": str(self.ckpt_dir / "enzyme_biofp_pretrain"),
                "wandb": {
                    "enabled": True,
                    "project": self.args.wandb_project,
                    "entity": self.args.wandb_entity,
                    "run_name": f"{self.run_name}-enzyme-biofp-pretrain",
                    "mode": self.args.wandb_mode,
                    "tags": ["from-scratch-chain", "enzyme-biofp-pretrain", "prott5", "sleec"],
                    "log_model": False,
                },
            },
            "wandb_args": {
                "wandb": True,
                "wandb_project": self.args.wandb_project,
                "wandb_entity": self.args.wandb_entity,
                "wandb_run_name": f"{self.run_name}-enzyme-biofp-pretrain",
                "wandb_mode": self.args.wandb_mode,
                "wandb_tags": ["from-scratch-chain", "enzyme-biofp-pretrain", "prott5", "sleec"],
                "wandb_log_model": False,
            },
            "data": {
                "protein_residue_embeds_path": str(self.prot5_h5),
                "protein_biofp_targets_path": str(biofp_targets),
                "biofp_missing_policy": "zero_with_mask",
                "residue_dim": 1024,
                "max_protein_tokens": 1022,
                "protein_truncation": "ends_center",
                "residue_in_memory": False,
            },
            "model": {
                "name": "ProteinPooledDualModel",
                "pooling": "sleec_guided_attention",
                "query_encoder_dims": [512, 512],
                "target_encoder_dims": [512, 512],
                "embedding_dim": 512,
                "biofp": {
                    "center_dim": 8,
                    "cofactor_dim": 16,
                    "transition_dim": 16,
                    "seq_dim": 384,
                    "dim": 128,
                    "hidden_dim": 512,
                    "seq_weight": 0.75,
                    "dropout": 0.10,
                },
                "sleec_pooling": {
                    "threshold": 0.34,
                    "scorer_hidden_dim": 256,
                    "checkpoint_path": str(ROOT / "checkpoints/SLEEC/sleec_stage1_prott5_uniref90_msa_4gpu_20260601_235157/best.ckpt"),
                    "freeze_scorer": True,
                    "initial_bias_scale": 1.0,
                    "train_bias_scale": False,
                },
            },
            "training": {
                "max_epochs": 12,
                "batch_size": 96,
                "validation_fraction": 0.05,
                "learning_rate": 1.0e-4,
                "weight_decay": 0.01,
                "num_workers": 4,
                "pin_memory": False,
                "precision": "32-true",
                "accelerator": "gpu",
                "devices": len(self.args.devices.split(",")),
                "strategy": "ddp_find_unused_parameters_true",
                "gradient_clip_val": 1.0,
                "checkpoint_monitor": "val/loss",
                "checkpoint_mode": "min",
                "log_every_n_steps": 10,
                "enable_progress_bar": True,
                "biofp_center_weight": 0.45,
                "biofp_transition_weight": 0.35,
                "biofp_cofactor_weight": 0.20,
                "biofp_confidence_cap": 8.0,
            },
        }
        pretrain_path = self.write_yaml(self.config_dir / "enzyme_biofp_pretrain.yaml", pretrain)

        joint = self.load_template("configs/retrieval_prott5_sleec_biofp_split_multimodal_allknown_4gpu.yaml")
        joint["logging"]["log_dir"] = str(self.log_dir / "joint_biofp_retrieval")
        joint["logging"]["checkpoint_dir"] = str(self.ckpt_dir / "joint_biofp_retrieval")
        joint["logging"]["wandb"]["project"] = self.args.wandb_project
        joint["logging"]["wandb"]["entity"] = self.args.wandb_entity
        joint["logging"]["wandb"]["run_name"] = f"{self.run_name}-joint-biofp-retrieval"
        joint["logging"]["wandb"]["mode"] = self.args.wandb_mode
        joint["logging"]["wandb"]["tags"] = list(joint["logging"]["wandb"].get("tags", [])) + ["from-scratch-chain", "biofp-pretrained"]
        joint["data"]["train_pairs_path"] = str(self.train_pairs)
        joint["data"]["train_reactions_path"] = str(self.train_rxns)
        joint["data"]["test_pairs_path"] = str(self.val_pairs)
        joint["data"]["test_reactions_path"] = str(self.val_rxns)
        joint["data"]["protein_residue_embeds_path"] = str(self.prot5_h5)
        joint["data"]["protein_biofp_targets_path"] = str(biofp_targets)
        joint["data"]["protein_biofp_vocab_path"] = str(biofp_vocab)
        joint["data"]["reaction_t5v2_embeds_path"] = str(self.reactiont5_h5)
        joint["data"]["reaction_unimol2_embeds_path"] = str(self.unimol_h5)
        joint["data"]["reaction_chiro_embeds_path"] = str(self.chiro_h5)
        joint["training"]["biofp_pretrain_checkpoint"] = str(self.ckpt_dir / "enzyme_biofp_pretrain/last.ckpt")
        joint["training"]["devices"] = len(self.args.devices.split(","))
        joint_path = self.write_yaml(self.config_dir / "joint_biofp_retrieval.yaml", joint)

        tune = self.load_template("configs/retrieval_prott5_sleec_biofp_split_multimodal_enzyme_tune_4gpu.yaml")
        tune["logging"]["log_dir"] = str(self.log_dir / "enzyme_only_tuning")
        tune["logging"]["checkpoint_dir"] = str(self.ckpt_dir / "enzyme_only_tuning")
        tune["logging"]["wandb"]["project"] = self.args.wandb_project
        tune["logging"]["wandb"]["entity"] = self.args.wandb_entity
        tune["logging"]["wandb"]["run_name"] = f"{self.run_name}-enzyme-only-tuning"
        tune["logging"]["wandb"]["mode"] = self.args.wandb_mode
        tune["logging"]["wandb"]["tags"] = list(tune["logging"]["wandb"].get("tags", [])) + ["from-scratch-chain", "biofp-pretrained"]
        tune["data"] = dict(joint["data"])
        tune["training"]["init_from_checkpoint"] = str(self.ckpt_dir / "joint_biofp_retrieval/last.ckpt")
        tune["training"]["devices"] = len(self.args.devices.split(","))
        tune_path = self.write_yaml(self.config_dir / "enzyme_only_tuning.yaml", tune)
        return pretrain_path, joint_path, tune_path

    def run_all(self) -> None:
        self.record(
            {
                "event": "chain_start",
                "run_name": self.run_name,
                "run_dir": str(self.run_dir),
                "wandb_project": self.args.wandb_project,
            }
        )
        (self.run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run_name": self.run_name,
                    "run_dir": str(self.run_dir),
                    "wandb_project": self.args.wandb_project,
                    "wandb_entity": self.args.wandb_entity,
                    "devices": self.args.devices,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        self.run(
            "01_build_source_collapse",
            [
                str(CAP_PY),
                "scripts/build_retrieval_training_source_collapse.py",
                "--output-root",
                str(self.data_root),
                "--mmseqs-bin",
                str(ROOT / "tools/mmseqs/bin/mmseqs"),
                "--threads",
                str(self.args.threads),
            ],
        )
        self.run(
            "02_filter_valid_pseudo_reactions",
            [
                str(CAP_PY),
                "scripts/filter_standardized_retrieval_reaction_smiles.py",
                "--root",
                str(self.data_root),
                "--policies",
                "exact",
                "nr90",
                "nr50",
                "--policy-dir-template",
                "train_{policy}",
                "--splits",
                "train",
                "--suffix",
                "_valid_rxn_pseudo",
                "--allow-pseudo-reactions",
            ],
        )
        self.run(
            "03_prepare_fit_inputs",
            [
                str(CAP_PY),
                "scripts/prepare_retrieval_source_collapse_pipeline_inputs.py",
                "--root",
                str(self.data_root),
                "--variants",
                "exact",
                "nr90",
                "nr50",
                "--shared-eval-root",
                str(ROOT / "data/standardized/horizyn_reactzyme_eval"),
            ],
        )
        self.run(
            "04_reconstruct_rhea_train",
            [
                str(CAP_PY),
                "scripts/reconstruct_reactzyme_rhea_directional_train.py",
                "--reactzyme-eval-root",
                str(ROOT / "data/paper/reactzyme/eval"),
                "--cleaned-uniprot-rhea",
                str(ROOT / "data/paper/reactzyme/raw/cleaned_uniprot_rhea.tsv"),
                "--rhea-molecules",
                str(ROOT / "data/paper/reactzyme/raw/rhea_molecules.tsv"),
                "--base-train-pairs",
                str(self.data_root / "train_exact/train_pairs_valid_rxn_pseudo.csv"),
                "--base-train-rxns",
                str(self.data_root / "train_exact/train_rxns_valid_rxn_pseudo.csv"),
                "--out-dir",
                str(self.data_root / "train_exact"),
                "--audit-dir",
                str(self.run_dir / "data/paper/reactzyme/reconstructed_rhea_train"),
                "--suffix",
                "_valid_rxn_rhea_reconstructed",
            ],
        )
        cofactor_dict = self.run_dir / "data/processed/capability_features/chebi_cofactor_dictionary.tsv"
        self.run(
            "05_build_chebi_cofactor_dictionary",
            [
                str(CAP_PY),
                "scripts/build_chebi_cofactor_dictionary.py",
                "--out",
                str(cofactor_dict),
            ],
        )
        self.run(
            "06_build_reaction_features",
            [
                str(CAP_PY),
                "scripts/build_reaction_features.py",
                "--reaction-smiles",
                str(self.train_rxns),
                "--train-pairs",
                str(self.train_pairs),
                "--rhea2ec",
                str(ROOT / "data/paper/reactzyme/raw/cleaned_uniprot_rhea.tsv"),
                "--rhea-chebi-smiles",
                str(ROOT / "data/paper/reactzyme/raw/rhea_molecules.tsv"),
                "--cofactor-dictionary",
                str(cofactor_dict),
                "--out-dir",
                str(self.cap_dir),
                "--drfp-bits",
                "2048",
                "--use-rxnmapper",
                "true",
                "--rxnmapper-batch-size",
                "64",
            ],
            env=self.env_with_site(),
        )
        self.run(
            "07_build_reaction_demand_vectors",
            [
                str(CAP_PY),
                "scripts/build_reaction_demand_vectors.py",
                "--reaction-features",
                str(self.cap_dir / "reaction_features.parquet"),
                "--reaction-drfp",
                str(self.cap_dir / "reaction_drfp.npz"),
                "--out-dir",
                str(self.cap_dir),
            ],
        )
        self.run(
            "08_build_enzyme_capability_labels",
            [
                str(CAP_PY),
                "scripts/build_enzyme_capability_labels.py",
                "--train-pairs",
                str(self.train_pairs),
                "--reaction-features",
                str(self.cap_dir / "reaction_features.parquet"),
                "--out-dir",
                str(self.cap_dir),
            ],
        )
        enhance_cmd = [
            str(CAP_PY),
            "scripts/enhance_enzyme_cofactor_annotations.py",
            "--capability-dir",
            str(self.cap_dir),
            "--train-pairs",
            str(self.train_pairs),
            "--uniprot-molecules",
            str(ROOT / "data/paper/reactzyme/raw/uniprot_molecules.tsv"),
            "--cleaned-uniprot-rhea",
            str(ROOT / "data/paper/reactzyme/raw/cleaned_uniprot_rhea.tsv"),
            "--reaction-features",
            str(self.cap_dir / "reaction_features.parquet"),
            "--cofactor-dictionary",
            str(cofactor_dict),
        ]
        if not self.args.skip_fetch_uniprotkb_cofactors:
            enhance_cmd.extend(["--fetch-uniprotkb-cofactors", "--uniprotkb-fetch-scope", "missing_cofactor"])
        self.run("09_enhance_enzyme_cofactors", enhance_cmd)
        self.run(
            "10_build_biofp_soft_targets",
            [
                str(CAP_PY),
                "scripts/build_enzyme_biofp_soft_targets.py",
                "--train-pairs",
                str(self.train_pairs),
                "--reaction-features",
                str(self.cap_dir / "reaction_features.parquet"),
                "--enzyme-cofactor-labels",
                str(self.cap_dir / "enzyme_cofactor_labels_enhanced.csv"),
                "--glycosyl-transfer-cofactor-rule",
                "type_or_transition",
                "--cofactor-label-set",
                "expanded",
                "--out-npz",
                str(self.cap_dir / "enzyme_biofp_soft_targets.npz"),
                "--out-vocab",
                str(self.cap_dir / "enzyme_biofp_vocab.json"),
            ],
        )

        prott5_tmp = self.data_root / "train_exact/prott5_residue_shards"
        common_prott5 = [
            str(ENV_PY),
            "scripts/extract_prott5_residue_embeddings.py",
            "--fasta",
            str(self.data_root / "train_exact/fit_proteins_with_clipzyme_eval.fasta"),
            "--output",
            str(self.prot5_h5),
            "--tmp-dir",
            str(prott5_tmp),
            "--model-name",
            "Rostlab/prot_t5_xl_half_uniref50-enc",
            "--world-size",
            str(len(self.args.devices.split(","))),
            "--batch-size",
            "8",
            "--max-tokens-per-batch",
            "4096",
            "--max-sequence-length",
            "1022",
            "--dtype",
            "float16",
            "--compression",
            "none",
            "--resume",
            "--force",
        ]
        gpu_ids = self.args.devices.split(",")
        workers = []
        for rank, gpu in enumerate(gpu_ids):
            workers.append(
                (
                    common_prott5 + ["--rank", str(rank), "--device", "cuda"],
                    self.env({"CUDA_VISIBLE_DEVICES": gpu}),
                    self.log_dir / f"11_extract_prott5_rank{rank}.log",
                )
            )
        self.run_parallel("11_extract_prott5_residue_embeddings", workers)
        self.run(
            "12_merge_prott5_residue_embeddings",
            common_prott5 + ["--merge-only", "--cleanup-shards"],
        )
        self.run(
            "13_extract_reactiont5v2",
            [
                str(CAP_PY),
                "scripts/extract_reaction_t5v2_embeddings.py",
                "--reactions",
                str(self.train_rxns),
                str(self.val_rxns),
                "--output",
                str(self.reactiont5_h5),
                "--model-name",
                "sagawa/ReactionT5v2-forward",
                "--batch-size",
                "64",
                "--max-length",
                "512",
                "--pooling",
                "mean",
                "--device",
                "cuda",
                "--dtype",
                "float16",
                "--allow-pseudo-reactions",
                "--force",
            ],
            env=self.env({"CUDA_VISIBLE_DEVICES": gpu_ids[0]}),
        )
        self.run(
            "14_extract_unimol2",
            [
                str(ENV_PY),
                "scripts/extract_unimol2_reaction_embeddings.py",
                "--reactions",
                str(self.train_rxns),
                str(self.val_rxns),
                "--output",
                str(self.unimol_h5),
                "--batch-size",
                "64",
                "--dtype",
                "float16",
                "--compression",
                "none",
                "--allow-pseudo-reactions",
                "--skip-invalid-molecules",
                "--skip-invalid-reactions",
                "--force",
            ],
            env=self.env_with_site(ROOT / ".deps/unimol_tools", Path("/datastor2/deep-proteins/EnzymeDiscovery/env/unimol2_site")),
        )
        self.run(
            "15_extract_chiro",
            [
                str(ENV_PY),
                "scripts/extract_chiro_reaction_embeddings.py",
                "--reactions",
                str(self.train_rxns),
                str(self.val_rxns),
                "--output",
                str(self.chiro_h5),
                "--device",
                "cuda",
                "--num-workers",
                "4",
                "--batch-size",
                "64",
                "--allow-pseudo-reactions",
                "--force",
            ],
            env=self.env_with_site(ROOT / ".deps/python", ROOT / ".deps/ChIRo"),
        )

        pretrain_config, joint_config, tune_config = self.build_configs()
        train_env = self.env_with_site()
        train_env["CUDA_VISIBLE_DEVICES"] = self.args.devices
        self.run(
            "16_pretrain_enzyme_biofp_split",
            [str(CAP_PY), "scripts/pretrain_enzyme_biofp_split.py", "--config", str(pretrain_config)],
            env=train_env,
        )
        if self.should_run_stage("17_train_joint_biofp_retrieval"):
            self.update_yaml_value(
                joint_config,
                ("training", "biofp_pretrain_checkpoint"),
                str(self.latest_checkpoint_for_command(self.ckpt_dir / "enzyme_biofp_pretrain")),
            )
        self.run(
            "17_train_joint_biofp_retrieval",
            [
                str(CAP_PY),
                "scripts/train_protein_pooling.py",
                "--config",
                str(joint_config),
                "--wandb",
                "--wandb-project",
                self.args.wandb_project,
                "--wandb-entity",
                self.args.wandb_entity,
                "--wandb-run-name",
                f"{self.run_name}-joint-biofp-retrieval",
                "--wandb-mode",
                self.args.wandb_mode,
                "--wandb-tags",
                "from-scratch-chain",
                "joint-biofp-retrieval",
            ],
            env=train_env,
        )
        if self.should_run_stage("18_train_enzyme_only_tuning"):
            self.update_yaml_value(
                tune_config,
                ("training", "init_from_checkpoint"),
                str(self.latest_checkpoint_for_command(self.ckpt_dir / "joint_biofp_retrieval")),
            )
        self.run(
            "18_train_enzyme_only_tuning",
            [
                str(CAP_PY),
                "scripts/train_protein_pooling.py",
                "--config",
                str(tune_config),
                "--wandb",
                "--wandb-project",
                self.args.wandb_project,
                "--wandb-entity",
                self.args.wandb_entity,
                "--wandb-run-name",
                f"{self.run_name}-enzyme-only-tuning",
                "--wandb-mode",
                self.args.wandb_mode,
                "--wandb-tags",
                "from-scratch-chain",
                "enzyme-only-tuning",
            ],
            env=train_env,
        )
        benchmark_checkpoint = self.ckpt_dir / "enzyme_only_tuning/last.ckpt"
        if self.should_run_stage("19_unified_test_benchmark"):
            benchmark_checkpoint = self.latest_checkpoint_for_command(
                self.ckpt_dir / "enzyme_only_tuning"
            )
        self.run(
            "19_unified_test_benchmark",
            [
                str(CAP_PY),
                "scripts/run_unified_retrieval_benchmark.py",
                "--checkpoint",
                str(benchmark_checkpoint),
                "--config",
                str(tune_config),
                "--suite",
                str(ROOT / "configs/benchmarks/enzyme_retrieval_unified.yaml"),
                "--tasks",
                "all",
                "--protein-embedding",
                "prott5",
                "--score-protein-embedding",
                "prott5",
                "--output-dir",
                str(self.results_dir / "unified_test_benchmark"),
                "--device",
                "cuda",
                "--store-targets-on-cpu",
                "--target-cache-dir",
                str(self.results_dir / "target_cache"),
            ],
            env=train_env,
        )
        self.record({"event": "chain_end", "run_name": self.run_name})


def main() -> None:
    args = parse_args()
    runner = Runner(args)
    print(f"Run directory: {runner.run_dir}")
    print(f"W&B project: {args.wandb_project}")
    try:
        runner.run_all()
    except Exception as exc:
        runner.record({"event": "chain_failed", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
