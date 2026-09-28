#!/usr/bin/env python3
"""
Evaluate a Horizyn protein-pooling checkpoint under paper-style benchmark settings.

This script intentionally evaluates the user's Horizyn model, not the paper's
FGW-CLIP method. It only mirrors the experimental settings:

* EnzymeMap-style virtual screening: reaction queries against a large enzyme
  candidate pool with BEDROC85, BEDROC20, EF@5%, and EF@10%.
* ReactZyme-style retrieval: reaction-to-enzyme and/or enzyme-to-reaction
  retrieval with Top-k, Top-k-N, Mean Rank, and MRR.

Expected pair CSV columns default to the Horizyn names:
    reaction_id, protein_id

Expected reaction CSV columns default to:
    reaction_id, reaction_smiles

Candidate protein pools should be represented by the same ragged residue HDF5
schema used by the attention-pooling models. To reproduce the paper setting,
pass the paper's protein library/split files via --candidate-ids, --pairs, and
--reactions. The model architecture and checkpoint remain Horizyn's.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from horizyn.config import load_config
from horizyn.benchmarks.retrieval import (
    evaluate_retrieval as unified_evaluate_retrieval,
    evaluate_screening as unified_evaluate_screening,
    rank_metrics_for_query as unified_rank_metrics_for_query,
    screening_metrics_for_query as unified_screening_metrics_for_query,
)
from horizyn.datasets.base import BaseDataset
from horizyn.datasets.csv import CSVDataset
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from horizyn.reaction_features import build_reaction_feature_dataset
from horizyn.utils import residue_collate_fn, unimol2_reaction_collate_fn


PAPER_BENCHMARKS = {
    "reactzyme_time": {
        "setting": "reactzyme",
        "split": "time",
        "pairs": "reactzyme/eval/time/test_pairs.csv",
        "train_pairs": "reactzyme/eval/time/train_pairs.csv",
        "reactions": "reactzyme/eval/time/reactions.csv",
        "candidate_ids": "reactzyme/eval/time/candidate_ids.txt",
        "candidate_h5": {
            "prott5": "reactzyme/eval/all_proteins_prott5_residue.h5",
            "esm2": "reactzyme/eval/all_proteins_esm2_650m_residue.h5",
        },
        "reaction_embeds_h5": "reactzyme/eval/reactzyme_unimol2_reaction_sets.h5",
        "candidates_from_test_positives": True,
    },
    "reactzyme_enzyme_smi": {
        "setting": "reactzyme",
        "split": "enzyme_smi",
        "pairs": "reactzyme/eval/enzyme_smi/test_pairs.csv",
        "train_pairs": "reactzyme/eval/enzyme_smi/train_pairs.csv",
        "reactions": "reactzyme/eval/enzyme_smi/reactions.csv",
        "candidate_ids": "reactzyme/eval/enzyme_smi/candidate_ids.txt",
        "candidate_h5": {
            "prott5": "reactzyme/eval/all_proteins_prott5_residue.h5",
            "esm2": "reactzyme/eval/all_proteins_esm2_650m_residue.h5",
        },
        "reaction_embeds_h5": "reactzyme/eval/reactzyme_unimol2_reaction_sets.h5",
        "candidates_from_test_positives": True,
    },
    "reactzyme_reaction_smi": {
        "setting": "reactzyme",
        "split": "reaction_smi",
        "pairs": "reactzyme/eval/reaction_smi/test_pairs.csv",
        "train_pairs": "reactzyme/eval/reaction_smi/train_pairs.csv",
        "reactions": "reactzyme/eval/reaction_smi/reactions.csv",
        "candidate_ids": "reactzyme/eval/reaction_smi/candidate_ids.txt",
        "candidate_h5": {
            "prott5": "reactzyme/eval/all_proteins_prott5_residue.h5",
            "esm2": "reactzyme/eval/all_proteins_esm2_650m_residue.h5",
        },
        "reaction_embeds_h5": "reactzyme/eval/reactzyme_unimol2_reaction_sets.h5",
        "candidates_from_test_positives": True,
    },
    "clipzyme_enzymemap": {
        "setting": "enzymemap",
        "split": "enzymemap",
        "pairs": "clipzyme/eval/enzymemap/pairs.csv",
        "reactions": "clipzyme/eval/enzymemap/reactions.csv",
        "candidate_ids": "clipzyme/eval/enzymemap/candidate_ids.txt",
        "candidate_h5": {
            "prott5": "clipzyme/eval/enzymemap/proteins_prott5_residue.h5",
            "esm2": "clipzyme/eval/enzymemap/proteins_esm2_650m_residue.h5",
        },
        "reaction_embeds_h5": "clipzyme/eval/enzymemap/unimol2_reaction_sets.h5",
    },
}


class TargetEncoderWrapper(torch.nn.Module):
    """Forward wrapper so DataParallel can shard encode_targets batches."""

    def __init__(
        self,
        model: torch.nn.Module,
        retrieval_direction: str = "reaction_to_enzyme",
    ):
        super().__init__()
        self.model = model
        self.retrieval_direction = retrieval_direction

    def forward(
        self,
        residue_embeddings: torch.Tensor,
        residue_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.encode_targets(
            residue_embeddings,
            residue_padding_mask=residue_padding_mask,
            retrieval_direction=self.retrieval_direction,
        )


def read_id_list(path: str | None, column: str | None = None) -> list[str] | None:
    if path is None:
        return None

    id_path = Path(path)
    if not id_path.exists():
        raise FileNotFoundError(f"ID file not found: {id_path}")

    if column is not None:
        with id_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{column}' not found in {id_path}; columns={reader.fieldnames}"
                )
            ids = []
            for row_number, row in enumerate(reader, start=2):
                raw_id = row.get(column)
                if raw_id is None:
                    raise ValueError(
                        f"Candidate ID file {id_path} has a missing cell at row {row_number}"
                    )
                identifier = raw_id.strip()
                if identifier:
                    ids.append(identifier)
            if len(ids) != len(set(ids)):
                raise ValueError(f"Candidate ID file {id_path} contains duplicate IDs")
            return ids

    ids = []
    with id_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line.split(",")[0].split()[0])
    if len(ids) != len(set(ids)):
        raise ValueError(f"Candidate ID file {id_path} contains duplicate IDs")
    return ids


def read_pairs(
    path: str,
    reaction_id_col: str,
    protein_id_col: str,
) -> list[tuple[str, str]]:
    pairs_path = Path(path)
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pair file not found: {pairs_path}")

    pairs: list[tuple[str, str]] = []
    with pairs_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Pair file has no header: {pairs_path}")
        missing = [col for col in (reaction_id_col, protein_id_col) if col not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Missing columns in {pairs_path}: {missing}; available={reader.fieldnames}"
            )
        seen_pairs = set()
        for row_number, row in enumerate(reader, start=2):
            raw_reaction_id = row.get(reaction_id_col)
            raw_protein_id = row.get(protein_id_col)
            reaction_id = "" if raw_reaction_id is None else raw_reaction_id.strip()
            protein_id = "" if raw_protein_id is None else raw_protein_id.strip()
            if not reaction_id or not protein_id:
                raise ValueError(f"Pair file {pairs_path} has an empty ID at row {row_number}")
            pair = (reaction_id, protein_id)
            if pair in seen_pairs:
                raise ValueError(f"Pair file {pairs_path} contains duplicate pair {pair}")
            seen_pairs.add(pair)
            pairs.append(pair)
    if not pairs:
        raise ValueError(f"Pair file contains no positive pairs: {pairs_path}")
    return pairs


def group_pairs(
    pairs: Iterable[tuple[str, str]],
    allowed_reactions: set[str] | None = None,
    allowed_proteins: set[str] | None = None,
    excluded_proteins: set[str] | None = None,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    reaction_to_proteins: dict[str, set[str]] = defaultdict(set)
    protein_to_reactions: dict[str, set[str]] = defaultdict(set)

    for reaction_id, protein_id in pairs:
        if allowed_reactions is not None and reaction_id not in allowed_reactions:
            continue
        if allowed_proteins is not None and protein_id not in allowed_proteins:
            continue
        if excluded_proteins is not None and protein_id in excluded_proteins:
            continue
        reaction_to_proteins[reaction_id].add(protein_id)
        protein_to_reactions[protein_id].add(reaction_id)

    return (
        {key: sorted(values) for key, values in reaction_to_proteins.items()},
        {key: sorted(values) for key, values in protein_to_reactions.items()},
    )


def resolve_paper_benchmark_args(args: argparse.Namespace) -> None:
    if args.benchmark is None:
        return
    if args.benchmark == "all":
        raise ValueError(
            "--benchmark all is only supported by scripts/run_paper_setting_eval_4gpu.sh; "
            "choose one concrete benchmark for this evaluator"
        )
    if args.benchmark not in PAPER_BENCHMARKS:
        raise ValueError(f"Unsupported paper benchmark preset: {args.benchmark}")

    preset = PAPER_BENCHMARKS[args.benchmark]
    data_root = Path(args.data_root)
    args.setting = args.setting or preset["setting"]
    args.pairs = args.pairs or str(data_root / preset["pairs"])
    args.reactions = args.reactions or str(data_root / preset["reactions"])
    args.candidate_ids = args.candidate_ids or str(data_root / preset["candidate_ids"])
    args.paper_split = args.paper_split or preset["split"]
    args.candidates_from_test_positives = bool(
        args.candidates_from_test_positives or preset.get("candidates_from_test_positives", False)
    )
    if args.train_pairs is None and preset.get("train_pairs") is not None:
        args.train_pairs = str(data_root / preset["train_pairs"])
    if args.candidate_residue_h5 is None:
        candidate_h5_by_embedding = preset["candidate_h5"]
        args.candidate_residue_h5 = str(
            data_root / candidate_h5_by_embedding[args.protein_embedding]
        )
    if args.reaction_embeds_h5 is None and preset.get("reaction_embeds_h5") is not None:
        candidate_reaction_h5 = data_root / preset["reaction_embeds_h5"]
        if candidate_reaction_h5.exists():
            args.reaction_embeds_h5 = str(candidate_reaction_h5)


def normalize_paper_reaction_smiles(_key: str, sample: dict) -> dict:
    smiles = sample.get("reaction_smiles", "")
    if isinstance(smiles, str) and smiles.count(">") < 2:
        sample = dict(sample)
        sample["reaction_smiles"] = f"{smiles}>>{smiles}"
    return sample


def reaction_input_mode(config) -> str:
    representation = config.data.get("reaction_representation", "fingerprint")
    query_encoder_type = config.model.get("query_encoder_type", "mlp")
    if representation == "hybrid_fingerprint_unimol2" or query_encoder_type == "hybrid_reaction":
        return "hybrid_fingerprint_unimol2"
    if representation == "unimol2_attention" or query_encoder_type == "unimol2_reaction_attention":
        return "unimol2_attention"
    return "fingerprint"


def build_reaction_inputs(
    reactions_path: str,
    reaction_id_col: str,
    reaction_smiles_col: str,
    reaction_embeds_h5: str | None,
    config,
) -> BaseDataset:
    mode = reaction_input_mode(config)
    if mode in {"hybrid_fingerprint_unimol2", "unimol2_attention"}:
        if reaction_embeds_h5 is None:
            raise FileNotFoundError(
                "This checkpoint/config expects UniMol2 reaction inputs, but no paper "
                "reaction embedding HDF5 was provided. Pass --reaction-embeds-h5 "
                "(expected paper paths include data/paper/reactzyme/eval/"
                "reactzyme_unimol2_reaction_sets.h5 and data/paper/clipzyme/eval/"
                "enzymemap/unimol2_reaction_sets.h5)."
            )
        if not Path(reaction_embeds_h5).exists():
            raise FileNotFoundError(
                f"Paper reaction embedding HDF5 not found: {reaction_embeds_h5}"
            )
        config = copy.deepcopy(config)
        config.data.reaction_embeds_path = reaction_embeds_h5

    raw_reactions = CSVDataset(
        file_path=reactions_path,
        key_column=reaction_id_col,
        columns=[reaction_smiles_col],
        rename_map={reaction_smiles_col: "reaction_smiles"},
    )
    pseudo_reaction_count = sum(
        1
        for reaction_id in raw_reactions.keys
        if str(raw_reactions._data[reaction_id]["reaction_smiles"]).count(">") < 2
    )
    if pseudo_reaction_count:
        print(
            f"Using molecule-set pseudo reactions for {pseudo_reaction_count} "
            f"records in {reactions_path}",
            flush=True,
        )
    return build_reaction_feature_dataset(
        reactions_path=reactions_path,
        config=config,
        key_column=reaction_id_col,
        smiles_column=reaction_smiles_col,
        bidirectional=False,
        transforms=normalize_paper_reaction_smiles,
    )


def load_candidate_keys(
    residue_dataset: ResidueEmbedDataset,
    candidate_ids_path: str | None,
    candidate_id_col: str | None,
) -> list[str]:
    length_by_key = residue_dataset.length_by_key
    hdf5_keys = set(residue_dataset.keys)
    requested_ids = read_id_list(candidate_ids_path, candidate_id_col)
    if requested_ids is None:
        candidate_keys = [
            protein_id
            for protein_id in residue_dataset.keys
            if length_by_key.get(protein_id, 0) > 0
        ]
        skipped_empty = len(residue_dataset.keys) - len(candidate_keys)
        if skipped_empty:
            print(
                f"Warning: {skipped_empty} candidates have zero residue embeddings and were skipped",
                flush=True,
            )
        if not candidate_keys:
            raise ValueError("No candidate IDs have non-empty residue embeddings")
        return candidate_keys

    candidate_keys = [
        protein_id
        for protein_id in requested_ids
        if protein_id in hdf5_keys and length_by_key.get(protein_id, 0) > 0
    ]
    missing = len(requested_ids) - len(candidate_keys)
    if missing:
        missing_ids = sum(protein_id not in hdf5_keys for protein_id in requested_ids)
        empty_ids = sum(
            protein_id in hdf5_keys and length_by_key.get(protein_id, 0) <= 0
            for protein_id in requested_ids
        )
        raise ValueError(
            f"Candidate manifest cannot be represented exactly: {missing} invalid IDs "
            f"({missing_ids} missing, {empty_ids} zero-length)"
        )
    if not candidate_keys:
        raise ValueError("No candidate IDs overlap with non-empty residue HDF5 IDs")
    return candidate_keys


def validate_paper_inputs(
    reaction_inputs: BaseDataset,
    residue_dataset: ResidueEmbedDataset,
    candidate_keys: list[str],
    eval_pairs: list[tuple[str, str]],
    reaction_id_col: str,
    protein_id_col: str,
) -> dict[str, int | str]:
    residue_key_set = set(residue_dataset.keys)
    candidate_key_set = set(candidate_keys)
    pair_reactions = {reaction_id for reaction_id, _protein_id in eval_pairs}
    pair_proteins = {protein_id for _reaction_id, protein_id in eval_pairs}
    missing_reactions = pair_reactions - set(reaction_inputs.keys)
    missing_candidate_proteins = pair_proteins - candidate_key_set
    missing_hdf5_proteins = pair_proteins - residue_key_set
    candidate_hdf5_overlap = len(candidate_key_set & residue_key_set)

    length_by_key = residue_dataset.length_by_key
    zero_length_candidates = sum(
        length_by_key.get(protein_id, 0) <= 0 for protein_id in candidate_keys
    )

    stats: dict[str, int | str] = {
        "pair_count": len(eval_pairs),
        "unique_pair_reactions": len(pair_reactions),
        "unique_pair_proteins": len(pair_proteins),
        "candidate_count": len(candidate_keys),
        "candidate_hdf5_overlap": candidate_hdf5_overlap,
        "missing_reaction_count": len(missing_reactions),
        "missing_candidate_positive_count": len(missing_candidate_proteins),
        "missing_hdf5_positive_count": len(missing_hdf5_proteins),
        "zero_length_candidate_count": zero_length_candidates,
        "reaction_id_col": reaction_id_col,
        "protein_id_col": protein_id_col,
    }
    print(
        "Paper input validation: "
        f"pairs={stats['pair_count']}, "
        f"unique_reactions={stats['unique_pair_reactions']}, "
        f"unique_positive_proteins={stats['unique_pair_proteins']}, "
        f"candidates={stats['candidate_count']}, "
        f"missing_reactions={stats['missing_reaction_count']}, "
        f"missing_candidate_positives={stats['missing_candidate_positive_count']}, "
        f"missing_hdf5_positives={stats['missing_hdf5_positive_count']}",
        flush=True,
    )
    if candidate_hdf5_overlap != len(candidate_keys):
        raise ValueError("Candidate IDs are not fully represented by the residue HDF5")
    if missing_reactions:
        raise ValueError(
            f"{len(missing_reactions)} pair reactions are missing from reaction inputs: "
            f"{sorted(missing_reactions)[:5]}"
        )
    if missing_candidate_proteins:
        raise ValueError(
            f"{len(missing_candidate_proteins)} positive proteins are outside the "
            f"candidate pool: {sorted(missing_candidate_proteins)[:5]}"
        )
    if missing_hdf5_proteins:
        raise ValueError(
            f"{len(missing_hdf5_proteins)} positive proteins are missing from the "
            f"candidate residue HDF5: {sorted(missing_hdf5_proteins)[:5]}"
        )
    if zero_length_candidates:
        raise ValueError(f"Candidate pool contains {zero_length_candidates} zero-length proteins")
    return stats


def encode_targets(
    module: ProteinPooledLitModule,
    residue_dataset: ResidueEmbedDataset,
    target_keys: list[str],
    device: str,
    target_batch_size: int,
    store_on_device: bool,
    target_encoder: torch.nn.Module | None = None,
    retrieval_direction: str = "reaction_to_enzyme",
) -> torch.Tensor:
    output_dim = module.model.target_encoder.output_dim
    storage_device = device if store_on_device else "cpu"
    target_encoder = target_encoder or TargetEncoderWrapper(
        module.model,
        retrieval_direction=retrieval_direction,
    )
    target_encoder.eval()
    target_embeds = torch.empty(
        len(target_keys),
        output_dim,
        dtype=torch.float32,
        device=storage_device,
    )

    for batch_start in tqdm(
        range(0, len(target_keys), target_batch_size),
        desc="Encoding candidate proteins",
    ):
        batch_end = min(batch_start + target_batch_size, len(target_keys))
        samples = []
        for target_id in target_keys[batch_start:batch_end]:
            sample = residue_dataset[target_id]
            sample["target_id"] = target_id
            samples.append(sample)
        batch = residue_collate_fn(samples)
        residues = batch["residue_embeddings"].to(device, non_blocking=True)
        mask = batch["residue_padding_mask"].to(device, non_blocking=True)
        encoded = target_encoder(residues, mask)
        target_embeds[batch_start:batch_end] = encoded.to(storage_device)
    return target_embeds


def _encode_target_chunk_on_device(
    target_encoder: torch.nn.Module,
    device_id: int,
    residue_embeddings: torch.Tensor,
    residue_padding_mask: torch.Tensor,
) -> torch.Tensor:
    device = f"cuda:{device_id}"
    torch.cuda.set_device(device_id)
    with torch.inference_mode(), torch.cuda.device(device_id):
        encoded = target_encoder(
            residue_embeddings.to(device, non_blocking=True),
            residue_padding_mask.to(device, non_blocking=True),
        )
        return encoded.detach().cpu()


def build_target_encoder_replicas(
    module: ProteinPooledLitModule,
    device_ids: list[int],
    retrieval_direction: str = "reaction_to_enzyme",
) -> list[tuple[int, torch.nn.Module]]:
    replicas = []
    for replica_idx, device_id in enumerate(device_ids):
        model = module.model if replica_idx == 0 else copy.deepcopy(module.model)
        device = f"cuda:{device_id}"
        model.to(device)
        model.eval()
        replicas.append(
            (
                device_id,
                TargetEncoderWrapper(
                    model,
                    retrieval_direction=retrieval_direction,
                )
                .to(device)
                .eval(),
            )
        )
    return replicas


def encode_targets_multigpu_batch(
    module: ProteinPooledLitModule,
    residue_dataset: ResidueEmbedDataset,
    target_keys: list[str],
    target_batch_size: int,
    device_ids: list[int],
    retrieval_direction: str = "reaction_to_enzyme",
) -> torch.Tensor:
    output_dim = module.model.target_encoder.output_dim
    target_embeds = torch.empty(
        len(target_keys),
        output_dim,
        dtype=torch.float32,
        device="cpu",
    )
    replicas = build_target_encoder_replicas(
        module,
        device_ids,
        retrieval_direction=retrieval_direction,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(replicas)) as executor:
        for batch_start in tqdm(
            range(0, len(target_keys), target_batch_size),
            desc="Encoding candidate proteins",
        ):
            batch_end = min(batch_start + target_batch_size, len(target_keys))
            samples = []
            for target_id in target_keys[batch_start:batch_end]:
                sample = residue_dataset[target_id]
                sample["target_id"] = target_id
                samples.append(sample)
            batch = residue_collate_fn(samples)
            residues = batch["residue_embeddings"]
            mask = batch["residue_padding_mask"]
            residue_chunks = torch.chunk(residues, len(replicas), dim=0)
            mask_chunks = torch.chunk(mask, len(replicas), dim=0)

            futures = []
            for (device_id, target_encoder), residue_chunk, mask_chunk in zip(
                replicas,
                residue_chunks,
                mask_chunks,
            ):
                if residue_chunk.shape[0] == 0:
                    continue
                futures.append(
                    executor.submit(
                        _encode_target_chunk_on_device,
                        target_encoder,
                        device_id,
                        residue_chunk,
                        mask_chunk,
                    )
                )
            encoded = torch.cat([future.result() for future in futures], dim=0)
            target_embeds[batch_start:batch_end] = encoded
    return target_embeds


def _encode_target_shard_worker(
    rank: int,
    device_id: int,
    checkpoint: str,
    residue_h5: str,
    target_keys: list[str],
    max_tokens: int | None,
    truncation: str,
    target_batch_size: int,
    output_path: str,
) -> None:
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device_id)
    module = ProteinPooledLitModule.load_from_checkpoint(checkpoint, map_location=device)
    module.eval()
    module.to(device)
    residue_dataset = ResidueEmbedDataset(
        file_path=residue_h5,
        in_memory=False,
        max_tokens=max_tokens,
        truncation=truncation,
    )
    with torch.inference_mode():
        embeddings = encode_targets(
            module=module,
            residue_dataset=residue_dataset,
            target_keys=target_keys,
            device=device,
            target_batch_size=target_batch_size,
            store_on_device=False,
            target_encoder=TargetEncoderWrapper(module.model),
        )
    torch.save(
        {
            "rank": rank,
            "target_keys": target_keys,
            "embeddings": embeddings,
        },
        output_path,
    )


def parse_target_encode_devices(device_arg: str, requested: str) -> list[int]:
    if not str(device_arg).startswith("cuda") or not torch.cuda.is_available():
        return []
    if requested == "none":
        return []
    if requested == "auto":
        return list(range(torch.cuda.device_count()))
    devices = []
    for value in requested.split(","):
        value = value.strip()
        if not value:
            continue
        devices.append(int(value))
    return devices


def encode_targets_sharded(
    checkpoint: str,
    residue_h5: str,
    target_keys: list[str],
    max_tokens: int | None,
    truncation: str,
    target_batch_size: int,
    device_ids: list[int],
    output_dir: Path | None = None,
) -> torch.Tensor:
    if len(device_ids) <= 1:
        raise ValueError("encode_targets_sharded requires at least two device IDs")
    if not target_keys:
        return torch.empty(0, 0)

    output_dir = output_dir or Path(tempfile.gettempdir())
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="target_encode_shards_", dir=str(output_dir)
    ) as tmp_dir:
        tmp_path = Path(tmp_dir)
        shard_size = math.ceil(len(target_keys) / len(device_ids))
        ctx = torch.multiprocessing.get_context("spawn")
        processes = []
        shard_paths: list[Path] = []
        for rank, device_id in enumerate(device_ids):
            start = rank * shard_size
            end = min(start + shard_size, len(target_keys))
            shard_keys = target_keys[start:end]
            if not shard_keys:
                continue
            shard_path = tmp_path / f"targets_rank{rank}.pt"
            shard_paths.append(shard_path)
            print(
                f"Starting target encoding shard {rank} on cuda:{device_id} "
                f"with {len(shard_keys)} proteins",
                flush=True,
            )
            process = ctx.Process(
                target=_encode_target_shard_worker,
                args=(
                    rank,
                    device_id,
                    checkpoint,
                    residue_h5,
                    shard_keys,
                    max_tokens,
                    truncation,
                    target_batch_size,
                    str(shard_path),
                ),
            )
            process.start()
            processes.append((rank, process))

        failed = []
        for rank, process in processes:
            process.join()
            if process.exitcode != 0:
                failed.append((rank, process.exitcode))
        if failed:
            raise RuntimeError(f"Target encoding shards failed: {failed}")

        shard_outputs = [
            torch.load(shard_path, map_location="cpu", weights_only=False)
            for shard_path in shard_paths
        ]
        shard_outputs.sort(key=lambda item: item["rank"])
        return torch.cat([item["embeddings"] for item in shard_outputs], dim=0)


def encode_reactions(
    module: ProteinPooledLitModule,
    reaction_inputs: BaseDataset,
    reaction_ids: list[str],
    device: str,
    batch_size: int,
    retrieval_direction: str = "reaction_to_enzyme",
) -> torch.Tensor:
    output_dim = module.model.query_encoder.output_dim
    query_embeds = torch.empty(len(reaction_ids), output_dim, dtype=torch.float32, device=device)
    for batch_start in tqdm(
        range(0, len(reaction_ids), batch_size),
        desc="Encoding reaction queries",
    ):
        batch_end = min(batch_start + batch_size, len(reaction_ids))
        query_vecs = build_query_inputs(
            reaction_inputs,
            reaction_ids[batch_start:batch_end],
            device,
        )
        query_embeds[batch_start:batch_end] = module.model.encode_queries(
            query_vecs,
            retrieval_direction=retrieval_direction,
        )
    return query_embeds


def build_query_inputs(
    reaction_inputs: BaseDataset,
    reaction_ids: list[str],
    device: str,
) -> torch.Tensor | dict[str, torch.Tensor]:
    samples = [reaction_inputs[reaction_id] for reaction_id in reaction_ids]
    if isinstance(samples[0], dict):
        batch = unimol2_reaction_collate_fn(samples)
        return {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
    return torch.stack(samples).to(device, non_blocking=True)


def screening_metrics_for_query(
    scores: torch.Tensor,
    positive_indices: list[int],
    bedroc_alphas: list[float],
    ef_fractions: list[float],
) -> dict[str, float]:
    return unified_screening_metrics_for_query(
        scores,
        positive_indices,
        bedroc_alphas,
        ef_fractions,
    )


def rank_metrics_for_query(
    scores: torch.Tensor,
    positive_indices: list[int],
    top_k_values: list[int],
) -> dict[str, float]:
    return unified_rank_metrics_for_query(
        scores,
        positive_indices,
        top_k_values,
        metric_protocol="reactzyme",
    )


def mean_dict(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = sorted({key for row in metric_rows for key in row})
    return {key: sum(row.get(key, 0.0) for row in metric_rows) / len(metric_rows) for key in keys}


def evaluate_enzymemap(
    module: ProteinPooledLitModule,
    reaction_inputs: BaseDataset,
    target_embeds: torch.Tensor,
    candidate_keys: list[str],
    reaction_to_proteins: dict[str, list[str]],
    device: str,
    batch_size: int,
    bedroc_alphas: list[float],
    ef_fractions: list[float],
) -> dict:
    results = unified_evaluate_screening(
        module,
        reaction_inputs,
        target_embeds,
        candidate_keys,
        reaction_to_proteins,
        device,
        batch_size,
        tuple(bedroc_alphas),
        tuple(ef_fractions),
    )
    results["setting"] = "enzymemap_screening"
    return results


def evaluate_reactzyme_direction(
    query_ids: list[str],
    candidate_ids: list[str],
    score_matrix: torch.Tensor,
    query_to_candidates: dict[str, list[str]],
    top_k_values: list[int],
) -> dict:
    candidate_to_idx = {candidate_id: idx for idx, candidate_id in enumerate(candidate_ids)}
    metric_rows: list[dict[str, float]] = []
    for row_idx, query_id in enumerate(tqdm(query_ids, desc="Computing retrieval metrics")):
        declared_positives = query_to_candidates.get(query_id, [])
        missing = [item for item in declared_positives if item not in candidate_to_idx]
        if missing:
            raise ValueError(
                f"Query {query_id!r} has positives outside the candidate IDs: {missing[:10]}"
            )
        positives = [candidate_to_idx[candidate_id] for candidate_id in declared_positives]
        if not positives:
            raise ValueError(f"Query {query_id!r} has no declared positive candidates")
        metric_rows.append(
            rank_metrics_for_query(
                score_matrix[row_idx],
                positives,
                top_k_values=top_k_values,
            )
        )
    results = mean_dict(metric_rows)
    results["num_queries"] = len(metric_rows)
    results["num_targets"] = len(candidate_ids)
    return results


def evaluate_reactzyme(
    module: ProteinPooledLitModule,
    reaction_inputs: BaseDataset,
    target_embeds: torch.Tensor,
    candidate_keys: list[str],
    reaction_to_proteins: dict[str, list[str]],
    protein_to_reactions: dict[str, list[str]],
    direction: str,
    device: str,
    batch_size: int,
    top_k_values: list[int],
    e2r_target_embeds: torch.Tensor | None = None,
) -> dict:
    directions = (
        ("reaction_to_enzyme", "enzyme_to_reaction") if direction == "both" else (direction,)
    )
    results = unified_evaluate_retrieval(
        module,
        reaction_inputs,
        target_embeds,
        candidate_keys,
        reaction_to_proteins,
        protein_to_reactions,
        directions,
        tuple(top_k_values),
        device,
        batch_size,
        e2r_target_embeds=e2r_target_embeds,
        metric_protocol="reactzyme",
    )
    results["setting"] = "reactzyme_retrieval"
    return results


def parse_float_list(values: list[str]) -> list[float]:
    return [float(value) for value in values]


def parse_int_list(values: list[str]) -> list[int]:
    return [int(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Horizyn under paper-style benchmark settings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--benchmark",
        choices=[*PAPER_BENCHMARKS.keys(), "all"],
        default=None,
        help="Paper benchmark preset. Use the wrapper script for --benchmark all.",
    )
    parser.add_argument("--data-root", default="data/paper")
    parser.add_argument(
        "--protein-embedding",
        choices=["prott5", "esm2"],
        default="prott5",
        help="Which paper candidate residue HDF5 to select when using --benchmark.",
    )
    parser.add_argument("--paper-split", default=None, help="Optional split name for metadata")
    parser.add_argument("--setting", choices=["enzymemap", "reactzyme"], default=None)
    parser.add_argument("--pairs", default=None, help="Evaluation pair CSV for the chosen split")
    parser.add_argument("--reactions", default=None, help="Reaction CSV for the chosen split")
    parser.add_argument("--candidate-residue-h5", default=None)
    parser.add_argument(
        "--candidate-ids", default=None, help="Optional paper candidate pool ID file"
    )
    parser.add_argument(
        "--candidates-from-test-positives",
        action="store_true",
        help="Restrict candidates to proteins occurring in the evaluation positives",
    )
    parser.add_argument("--candidate-id-col", default=None)
    parser.add_argument("--reaction-embeds-h5", default=None)
    parser.add_argument(
        "--train-pairs", default=None, help="Training pair CSV for seen-enzyme exclusion"
    )
    parser.add_argument(
        "--exclude-train-proteins",
        action="store_true",
        help="Remove train-seen enzymes from candidates/positives for novel-enzyme screening",
    )
    parser.add_argument("--reaction-id-col", default="reaction_id")
    parser.add_argument("--reaction-smiles-col", default="reaction_smiles")
    parser.add_argument("--protein-id-col", default="protein_id")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-batch-size", type=int, default=512)
    parser.add_argument(
        "--target-encode-devices",
        default="auto",
        help="Comma-separated visible CUDA device IDs for sharded target encoding, 'auto', or 'none'",
    )
    parser.add_argument("--store-targets-on-cpu", action="store_true")
    parser.add_argument("--bedroc-alpha", nargs="+", default=["85", "20"])
    parser.add_argument("--ef-fraction", nargs="+", default=["0.05", "0.1"])
    parser.add_argument("--top-k", nargs="+", default=["1", "2", "3", "4", "5", "10", "20", "50"])
    parser.add_argument(
        "--reactzyme-direction",
        choices=["reaction_to_enzyme", "enzyme_to_reaction", "both"],
        default="both",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate paper files and exit before model target encoding/scoring.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    resolve_paper_benchmark_args(args)
    if args.setting is None or args.pairs is None or args.reactions is None:
        parser.error(
            "Either pass --benchmark or provide --setting, --pairs, and --reactions explicitly"
        )

    config = load_config(args.config)
    residue_h5 = args.candidate_residue_h5 or config.data.protein_residue_embeds_path
    reaction_mode = reaction_input_mode(config)

    print(f"Loading checkpoint: {args.checkpoint}")
    module = ProteinPooledLitModule.load_from_checkpoint(args.checkpoint, map_location=args.device)
    module.eval()
    module.to(args.device)
    target_encode_devices = parse_target_encode_devices(args.device, args.target_encode_devices)
    if len(target_encode_devices) > 1:
        print(
            f"Using multi-GPU batched target encoding on {len(target_encode_devices)} visible GPUs: "
            f"{target_encode_devices}",
            flush=True,
        )

    print(f"Loading reactions: {args.reactions}")
    reaction_inputs = build_reaction_inputs(
        reactions_path=args.reactions,
        reaction_id_col=args.reaction_id_col,
        reaction_smiles_col=args.reaction_smiles_col,
        reaction_embeds_h5=args.reaction_embeds_h5,
        config=config,
    )

    print(f"Loading residue candidate HDF5: {residue_h5}")
    residue_dataset = ResidueEmbedDataset(
        file_path=residue_h5,
        in_memory=False,
        max_tokens=config.data.get("max_protein_tokens", 1024),
        truncation=config.data.get("protein_truncation", "ends_center"),
    )
    candidate_keys = load_candidate_keys(
        residue_dataset,
        candidate_ids_path=args.candidate_ids,
        candidate_id_col=args.candidate_id_col,
    )
    eval_pairs = read_pairs(args.pairs, args.reaction_id_col, args.protein_id_col)
    train_pairs = (
        read_pairs(args.train_pairs, args.reaction_id_col, args.protein_id_col)
        if args.train_pairs is not None
        else None
    )
    train_eval_overlap = set(train_pairs or []) & set(eval_pairs)
    if train_eval_overlap:
        raise ValueError(
            f"Evaluation split leaks {len(train_eval_overlap)} exact positive training pairs"
        )
    if args.candidates_from_test_positives:
        positive_proteins = {protein_id for _reaction_id, protein_id in eval_pairs}
        candidate_set = set(candidate_keys)
        missing_positive_candidates = sorted(positive_proteins - candidate_set)
        if missing_positive_candidates:
            raise ValueError(
                f"Published test-positive candidate pool is missing "
                f"{len(missing_positive_candidates)} proteins: "
                f"{missing_positive_candidates[:10]}"
            )
        candidate_keys = [key for key in candidate_keys if key in positive_proteins]

    excluded_proteins: set[str] | None = None
    if args.exclude_train_proteins:
        if train_pairs is None:
            raise ValueError("--exclude-train-proteins requires --train-pairs")
        excluded_proteins = {protein_id for _reaction_id, protein_id in train_pairs}
        candidate_keys = [
            protein_id for protein_id in candidate_keys if protein_id not in excluded_proteins
        ]
        original_eval_pair_count = len(eval_pairs)
        eval_pairs = [pair for pair in eval_pairs if pair[1] not in excluded_proteins]
        if not eval_pairs:
            raise ValueError("No evaluation pairs remain after excluding train-seen proteins")
        print(
            f"Excluded {len(excluded_proteins)} train-seen proteins and "
            f"{original_eval_pair_count - len(eval_pairs)} affected evaluation pairs"
        )

    print(f"Candidate proteins: {len(candidate_keys)}")
    validation_stats = validate_paper_inputs(
        reaction_inputs=reaction_inputs,
        residue_dataset=residue_dataset,
        candidate_keys=candidate_keys,
        eval_pairs=eval_pairs,
        reaction_id_col=args.reaction_id_col,
        protein_id_col=args.protein_id_col,
    )
    validation_stats["train_eval_pair_overlap_count"] = len(train_eval_overlap)
    candidate_pool_policy = (
        "train_unseen_test_positive_entities"
        if args.candidates_from_test_positives and args.exclude_train_proteins
        else (
            "published_test_positive_entities"
            if args.candidates_from_test_positives
            else ("explicit_candidate_manifest" if args.candidate_ids else "full_embedding_store")
        )
    )
    reaction_to_proteins, protein_to_reactions = group_pairs(
        eval_pairs,
        allowed_reactions=set(reaction_inputs.keys),
        allowed_proteins=set(candidate_keys),
        excluded_proteins=excluded_proteins,
    )

    if args.validate_only:
        results = {
            "benchmark": args.benchmark,
            "setting": args.setting,
            "split": args.paper_split,
            "validation": validation_stats,
            "checkpoint": args.checkpoint,
            "config": args.config,
            "pairs": args.pairs,
            "reactions": args.reactions,
            "candidate_residue_h5": residue_h5,
            "candidate_ids": args.candidate_ids,
            "candidate_pool_size": len(candidate_keys),
            "candidate_pool_policy": candidate_pool_policy,
            "excluded_train_proteins": bool(args.exclude_train_proteins),
            "reaction_input_mode": reaction_mode,
            "reaction_embeds_h5": args.reaction_embeds_h5,
        }
        print(json.dumps(results, indent=2))
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, indent=2))
            print(f"Saved results to: {output_path}")
        return

    with torch.inference_mode():
        target_retrieval_direction = (
            "reaction_to_enzyme"
            if args.setting == "enzymemap"
            or args.reactzyme_direction in {"reaction_to_enzyme", "both"}
            else "enzyme_to_reaction"
        )
        if len(target_encode_devices) > 1:
            target_embeds = encode_targets_multigpu_batch(
                module=module,
                residue_dataset=residue_dataset,
                target_keys=candidate_keys,
                target_batch_size=args.target_batch_size,
                device_ids=target_encode_devices,
                retrieval_direction=target_retrieval_direction,
            )
            if not args.store_targets_on_cpu:
                target_embeds = target_embeds.to(args.device)
        else:
            target_embeds = encode_targets(
                module=module,
                residue_dataset=residue_dataset,
                target_keys=candidate_keys,
                device=args.device,
                target_batch_size=args.target_batch_size,
                store_on_device=not args.store_targets_on_cpu,
                target_encoder=TargetEncoderWrapper(
                    module.model,
                    retrieval_direction=target_retrieval_direction,
                ),
                retrieval_direction=target_retrieval_direction,
            )

        e2r_target_embeds = None
        if (
            args.setting != "enzymemap"
            and args.reactzyme_direction == "both"
            and getattr(module.model, "r2e_adapter", None) is not None
        ):
            if len(target_encode_devices) > 1:
                e2r_target_embeds = encode_targets_multigpu_batch(
                    module=module,
                    residue_dataset=residue_dataset,
                    target_keys=candidate_keys,
                    target_batch_size=args.target_batch_size,
                    device_ids=target_encode_devices,
                    retrieval_direction="enzyme_to_reaction",
                )
                if not args.store_targets_on_cpu:
                    e2r_target_embeds = e2r_target_embeds.to(args.device)
            else:
                e2r_target_embeds = encode_targets(
                    module=module,
                    residue_dataset=residue_dataset,
                    target_keys=candidate_keys,
                    device=args.device,
                    target_batch_size=args.target_batch_size,
                    store_on_device=not args.store_targets_on_cpu,
                    target_encoder=TargetEncoderWrapper(
                        module.model,
                        retrieval_direction="enzyme_to_reaction",
                    ),
                    retrieval_direction="enzyme_to_reaction",
                )

        if args.setting == "enzymemap":
            results = evaluate_enzymemap(
                module=module,
                reaction_inputs=reaction_inputs,
                target_embeds=target_embeds,
                candidate_keys=candidate_keys,
                reaction_to_proteins=reaction_to_proteins,
                device=args.device,
                batch_size=args.batch_size,
                bedroc_alphas=parse_float_list(args.bedroc_alpha),
                ef_fractions=parse_float_list(args.ef_fraction),
            )
        else:
            results = evaluate_reactzyme(
                module=module,
                reaction_inputs=reaction_inputs,
                target_embeds=target_embeds,
                candidate_keys=candidate_keys,
                reaction_to_proteins=reaction_to_proteins,
                protein_to_reactions=protein_to_reactions,
                direction=args.reactzyme_direction,
                device=args.device,
                batch_size=args.batch_size,
                top_k_values=parse_int_list(args.top_k),
                e2r_target_embeds=e2r_target_embeds,
            )

    results["benchmark"] = args.benchmark
    results["split"] = args.paper_split
    results["checkpoint"] = args.checkpoint
    results["config"] = args.config
    results["pairs"] = args.pairs
    results["reactions"] = args.reactions
    results["candidate_residue_h5"] = residue_h5
    results["candidate_ids"] = args.candidate_ids
    results["candidate_pool_policy"] = candidate_pool_policy
    results["candidate_pool_size"] = len(candidate_keys)
    results["excluded_train_proteins"] = bool(args.exclude_train_proteins)
    results["reaction_input_mode"] = reaction_mode
    results["reaction_embeds_h5"] = args.reaction_embeds_h5
    results["validation"] = validation_stats
    results["experimental_setting_only"] = True

    print(json.dumps(results, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
