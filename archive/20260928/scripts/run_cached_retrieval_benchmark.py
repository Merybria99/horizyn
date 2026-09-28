#!/usr/bin/env python3
"""Run retrieval benchmark suites with a per-model candidate-embedding cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from horizyn.capability.enzyme_capability_dataset import (
    CapabilityVectorDataset,
    TargetWithCapabilityDataset,
    TargetWithTextDataset,
    TextVectorDataset,
)
from horizyn.benchmarks.retrieval import (
    BenchmarkTask,
    _target_cache_base_metadata,
    attach_benchmark_artifact_manifest,
    benchmark_artifact_inputs,
    build_reaction_inputs,
    candidate_pool_policy,
    evaluate_retrieval,
    evaluate_screening,
    expand_bidirectional_pairs,
    filter_candidate_keys_by_score_residue,
    group_pairs,
    load_benchmark_suite,
    load_candidate_keys_from_embedding,
    load_candidate_keys_from_residue,
    load_repo_checkpoint,
    model_kind_from_config,
    needs_score_residue_embeddings,
    reaction_input_mode,
    read_pairs,
    restrict_candidates_to_test_positives,
    select_residue_h5,
    select_score_residue_h5,
    task_to_dict,
    validate_task_inputs,
    write_grouped_summary_tables,
    write_json,
    write_summary_csv,
    write_summary_markdown,
)
from horizyn.config import load_config
from horizyn.datasets.hdf5 import EmbedDataset
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from horizyn.utils import residue_collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one or more retrieval suites while encoding the union of candidate "
            "proteins only once for a checkpoint."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Repo Lightning checkpoint")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--suites", nargs="+", required=True, help="Benchmark suite YAML files")
    parser.add_argument(
        "--suite-labels",
        nargs="+",
        default=None,
        help="Output subdirectory labels matching --suites order",
    )
    parser.add_argument("--tasks", nargs="+", default=["all"], help="Task names or 'all'")
    parser.add_argument(
        "--protein-embedding",
        choices=["prott5", "esm2", "esmc"],
        default="prott5",
    )
    parser.add_argument(
        "--score-protein-embedding",
        choices=["prott5", "esm2", "esmc"],
        default="esm2",
    )
    parser.add_argument("--output-dir", required=True, help="Root directory for outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--target-batch-size", type=int, default=1024)
    parser.add_argument("--store-targets-on-cpu", action="store_true")
    parser.add_argument("--torch-num-threads", type=int, default=None)
    parser.add_argument("--progress-every-batches", type=int, default=10)
    return parser.parse_args()


def select_tasks(tasks: list[BenchmarkTask], requested_names: list[str]) -> list[BenchmarkTask]:
    if requested_names == ["all"] or "all" in requested_names:
        return tasks
    by_name = {task.name: task for task in tasks}
    missing = [name for name in requested_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown benchmark task(s): {missing}; available={sorted(by_name)}")
    return [by_name[name] for name in requested_names]


def candidate_h5_key(task: BenchmarkTask, kind: str, protein_embedding: str) -> Path:
    if kind == "pooled":
        if task.candidate_embedding_h5 is None:
            raise ValueError(f"Task {task.name} does not define candidate_embedding_h5")
        return task.candidate_embedding_h5.resolve()
    return select_residue_h5(task, protein_embedding).resolve()


def score_h5_key(
    task: BenchmarkTask,
    config: Any,
    kind: str,
    score_protein_embedding: str,
) -> Path | None:
    if kind == "pooled" or not needs_score_residue_embeddings(config):
        return None
    return select_score_residue_h5(task, score_protein_embedding).resolve()


def cached_target_encoding_metadata(
    *,
    task: BenchmarkTask,
    config: Any,
    kind: str,
    checkpoint: str,
    config_path: str,
    protein_embedding: str,
    score_protein_embedding: str,
    retrieval_direction: str,
) -> dict[str, Any]:
    """Describe the source inputs used for one cached-union target encoding."""

    if kind == "pooled":
        return _target_cache_base_metadata(
            kind=kind,
            checkpoint=checkpoint,
            config_path=config_path,
            protein_embedding=protein_embedding,
            score_protein_embedding=score_protein_embedding,
            candidate_embedding_h5=candidate_h5_key(task, kind, protein_embedding),
            retrieval_direction=retrieval_direction,
        )
    capability_path = config.data.get("protein_capability_vectors_path", None)
    text_path = config.data.get("protein_text_vectors_path", None)
    return _target_cache_base_metadata(
        kind=kind,
        checkpoint=checkpoint,
        config_path=config_path,
        protein_embedding=protein_embedding,
        score_protein_embedding=score_protein_embedding,
        residue_h5=candidate_h5_key(task, kind, protein_embedding),
        score_residue_h5=score_h5_key(
            task,
            config,
            kind,
            score_protein_embedding,
        ),
        capability_vectors_path=capability_path,
        capability_missing_policy=(
            config.data.get("capability_missing_policy", "zero_with_mask")
            if capability_path
            else None
        ),
        text_vectors_path=text_path,
        text_vector_missing_policy=(
            config.data.get("text_vector_missing_policy", "zero_with_mask") if text_path else None
        ),
        max_tokens=config.data.get("max_protein_tokens", 1024),
        truncation=config.data.get("protein_truncation", "ends_center"),
        retrieval_direction=retrieval_direction,
    )


def maybe_attach_capability_vectors(target_dataset, config: Any):
    capability_path = config.data.get("protein_capability_vectors_path", None)
    if not capability_path:
        return target_dataset
    capability_path = Path(capability_path)
    if not capability_path.is_absolute():
        capability_path = Path.cwd() / capability_path
    if not capability_path.exists():
        raise FileNotFoundError(f"Protein capability vector file not found: {capability_path}")
    capability_dataset = CapabilityVectorDataset(capability_path)
    merged = TargetWithCapabilityDataset(
        target_dataset=target_dataset,
        capability_dataset=capability_dataset,
        capability_key="capability_vec",
        mask_key="capability_mask",
        missing_policy=config.data.get("capability_missing_policy", "zero_with_mask"),
    )
    missing = merged.missing_count
    if missing:
        print(
            "Capability vectors missing for "
            f"{missing}/{len(target_dataset.keys)} benchmark candidate proteins; "
            "using zero vectors with capability_mask=False",
            flush=True,
        )
    return merged


def maybe_attach_text_vectors(target_dataset, config: Any):
    text_path = config.data.get("protein_text_vectors_path", None)
    if not text_path:
        return target_dataset
    text_path = Path(text_path)
    if not text_path.is_absolute():
        text_path = Path.cwd() / text_path
    if not text_path.exists():
        raise FileNotFoundError(f"Protein text vector file not found: {text_path}")
    text_dataset = TextVectorDataset(text_path)
    merged = TargetWithTextDataset(
        target_dataset=target_dataset,
        text_dataset=text_dataset,
        text_key="text_vec",
        mask_key="text_mask",
        missing_policy=config.data.get("text_vector_missing_policy", "zero_with_mask"),
    )
    missing = merged.missing_count
    if missing:
        print(
            "Text vectors missing for "
            f"{missing}/{len(target_dataset.keys)} benchmark candidate proteins; "
            "using zero vectors with text_mask=False",
            flush=True,
        )
    return merged


def build_candidate_cache(
    *,
    tasks_by_suite: list[tuple[str, Path, list[BenchmarkTask]]],
    config: Any,
    kind: str,
    protein_embedding: str,
    score_protein_embedding: str,
    module: torch.nn.Module,
    device: str,
    target_batch_size: int,
    store_targets_on_cpu: bool,
    progress_every_batches: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    dict[str, int],
    dict[tuple[str, str], list[str]],
    dict[tuple[str, str], dict[str, int]],
    object,
    list[str],
]:
    all_tasks = [task for _label, _suite, tasks in tasks_by_suite for task in tasks]
    needs_directional_e2r = getattr(module.model, "r2e_adapter", None) is not None and any(
        "enzyme_to_reaction" in task.directions for task in all_tasks
    )
    if kind == "pooled" and needs_directional_e2r:
        raise ValueError("R2E adapters require residue-level enzyme inputs")
    candidate_paths = {candidate_h5_key(task, kind, protein_embedding) for task in all_tasks}
    if len(candidate_paths) != 1:
        raise ValueError(f"Cached benchmark expects one candidate HDF5; got {candidate_paths}")
    candidate_path = next(iter(candidate_paths))

    score_paths = {
        path
        for task in all_tasks
        if (path := score_h5_key(task, config, kind, score_protein_embedding)) is not None
    }
    if len(score_paths) > 1:
        raise ValueError(f"Cached benchmark expects one score HDF5; got {score_paths}")

    task_candidate_keys: dict[tuple[str, str], list[str]] = {}
    task_candidate_stats: dict[tuple[str, str], dict[str, int]] = {}

    if kind == "pooled":
        target_dataset = EmbedDataset(str(candidate_path), in_memory=False)
        score_dataset = None
        target_store_keys = list(target_dataset.keys)
        for suite_label, _suite_path, tasks in tasks_by_suite:
            for task in tasks:
                keys, stats = load_candidate_keys_from_embedding(
                    target_dataset,
                    task.candidate_ids,
                )
                eval_pairs = expand_bidirectional_pairs(
                    task,
                    read_pairs(task.pairs, task.reaction_id_col, task.protein_id_col),
                )
                keys, stats = restrict_candidates_to_test_positives(task, eval_pairs, keys, stats)
                task_candidate_keys[(suite_label, task.name)] = keys
                task_candidate_stats[(suite_label, task.name)] = stats
    else:
        max_tokens = config.data.get("max_protein_tokens", 1024)
        truncation = config.data.get("protein_truncation", "ends_center")
        target_dataset = ResidueEmbedDataset(
            str(candidate_path),
            in_memory=False,
            max_tokens=max_tokens,
            truncation=truncation,
        )
        score_dataset = None
        if score_paths:
            score_dataset = ResidueEmbedDataset(
                str(next(iter(score_paths))),
                in_memory=False,
                max_tokens=max_tokens,
                truncation=truncation,
            )
        target_store_keys = list(target_dataset.keys)
        for suite_label, _suite_path, tasks in tasks_by_suite:
            for task in tasks:
                keys, stats = load_candidate_keys_from_residue(target_dataset, task.candidate_ids)
                eval_pairs = expand_bidirectional_pairs(
                    task,
                    read_pairs(task.pairs, task.reaction_id_col, task.protein_id_col),
                )
                keys, stats = restrict_candidates_to_test_positives(task, eval_pairs, keys, stats)
                if score_dataset is not None:
                    keys, score_stats = filter_candidate_keys_by_score_residue(keys, score_dataset)
                    stats.update(score_stats)
                task_candidate_keys[(suite_label, task.name)] = keys
                task_candidate_stats[(suite_label, task.name)] = stats

    union_keys: list[str] = []
    seen: set[str] = set()
    for _suite_label, _suite_path, tasks in tasks_by_suite:
        for task in tasks:
            for key in task_candidate_keys[(_suite_label, task.name)]:
                if key not in seen:
                    seen.add(key)
                    union_keys.append(key)
    storage_rank = {key: idx for idx, key in enumerate(target_store_keys)}
    union_keys.sort(key=lambda key: storage_rank.get(key, len(storage_rank)))
    if kind != "pooled":
        target_dataset = maybe_attach_capability_vectors(target_dataset, config)
        target_dataset = maybe_attach_text_vectors(target_dataset, config)

    print(
        f"Encoding candidate cache in HDF5 order: {len(union_keys)} unique proteins "
        f"from {candidate_path}",
        flush=True,
    )
    with torch.inference_mode():
        if kind == "pooled":
            target_embeds = encode_pooled_targets_with_progress(
                module,
                target_dataset,
                union_keys,
                device,
                target_batch_size,
                store_on_device=not store_targets_on_cpu,
                progress_every_batches=progress_every_batches,
            )
            e2r_target_embeds = None
        else:
            target_embeds = encode_residue_targets_with_progress(
                module,
                target_dataset,
                union_keys,
                device,
                target_batch_size,
                store_on_device=not store_targets_on_cpu,
                score_dataset=score_dataset,
                progress_every_batches=progress_every_batches,
                retrieval_direction="reaction_to_enzyme",
            )
            e2r_target_embeds = (
                encode_residue_targets_with_progress(
                    module,
                    target_dataset,
                    union_keys,
                    device,
                    target_batch_size,
                    store_on_device=not store_targets_on_cpu,
                    score_dataset=score_dataset,
                    progress_every_batches=progress_every_batches,
                    retrieval_direction="enzyme_to_reaction",
                )
                if needs_directional_e2r
                else None
            )
    key_to_idx = {key: idx for idx, key in enumerate(union_keys)}
    return (
        target_embeds,
        e2r_target_embeds,
        key_to_idx,
        task_candidate_keys,
        task_candidate_stats,
        target_dataset,
        target_store_keys,
    )


def subset_target_embeds(
    union_embeds: torch.Tensor,
    key_to_idx: dict[str, int],
    candidate_keys: list[str],
) -> torch.Tensor:
    index_device = union_embeds.device
    rows = torch.as_tensor(
        [key_to_idx[key] for key in candidate_keys],
        dtype=torch.long,
        device=index_device,
    )
    return union_embeds.index_select(0, rows)


def encode_pooled_targets_with_progress(
    module: torch.nn.Module,
    dataset: EmbedDataset,
    target_keys: list[str],
    device: str,
    batch_size: int,
    store_on_device: bool,
    progress_every_batches: int,
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    storage_device = device if store_on_device else "cpu"
    total_batches = (len(target_keys) + batch_size - 1) // batch_size
    with torch.inference_mode():
        for batch_idx, batch_start in enumerate(range(0, len(target_keys), batch_size), start=1):
            batch_end = min(batch_start + batch_size, len(target_keys))
            vectors = torch.stack(
                [dataset[target_id] for target_id in target_keys[batch_start:batch_end]]
            )
            encoded = module.model.target_encoder(vectors.to(device, non_blocking=True))
            output.append(encoded.detach().to(storage_device))
            if progress_every_batches > 0 and (
                batch_idx == 1
                or batch_idx == total_batches
                or batch_idx % progress_every_batches == 0
            ):
                print(
                    f"Encoded candidate cache batch {batch_idx}/{total_batches} "
                    f"({batch_end}/{len(target_keys)} proteins)",
                    flush=True,
                )
    if not output:
        return torch.empty(0, 0, dtype=torch.float32, device=storage_device)
    return torch.cat(output, dim=0)


def encode_residue_targets_with_progress(
    module: torch.nn.Module,
    dataset: ResidueEmbedDataset,
    target_keys: list[str],
    device: str,
    batch_size: int,
    store_on_device: bool,
    score_dataset: ResidueEmbedDataset | None,
    progress_every_batches: int,
    retrieval_direction: str = "reaction_to_enzyme",
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    storage_device = device if store_on_device else "cpu"
    model_dtype = next(
        (
            parameter.dtype
            for parameter in module.model.parameters()
            if parameter.is_floating_point()
        ),
        torch.float32,
    )
    total_batches = (len(target_keys) + batch_size - 1) // batch_size
    with torch.inference_mode():
        for batch_idx, batch_start in enumerate(range(0, len(target_keys), batch_size), start=1):
            batch_end = min(batch_start + batch_size, len(target_keys))
            samples = []
            for target_id in target_keys[batch_start:batch_end]:
                sample = dict(dataset[target_id])
                if score_dataset is not None:
                    score_sample = score_dataset[target_id]
                    sample["score_residue_embeddings"] = score_sample["residue_embeddings"]
                sample["target_id"] = target_id
                samples.append(sample)
            batch = residue_collate_fn(samples)
            residues = batch["residue_embeddings"].to(
                device=device,
                dtype=model_dtype,
                non_blocking=True,
            )
            mask = batch["residue_padding_mask"].to(device, non_blocking=True)
            score_residues = batch.get("score_residue_embeddings")
            score_mask = batch.get("score_residue_padding_mask")
            if score_residues is not None:
                score_residues = score_residues.to(
                    device=device,
                    dtype=model_dtype,
                    non_blocking=True,
                )
                if score_mask is not None:
                    score_mask = score_mask.to(device, non_blocking=True)
            capability_vectors = batch.get("capability_vec")
            capability_mask = batch.get("capability_mask")
            text_vectors = batch.get("text_vec")
            text_mask = batch.get("text_mask")
            if torch.is_tensor(capability_vectors):
                capability_vectors = capability_vectors.to(device, non_blocking=True)
            else:
                capability_vectors = None
            if torch.is_tensor(capability_mask):
                capability_mask = capability_mask.to(device, non_blocking=True, dtype=torch.bool)
            else:
                capability_mask = None
            if torch.is_tensor(text_vectors):
                text_vectors = text_vectors.to(device, non_blocking=True)
            else:
                text_vectors = None
            if torch.is_tensor(text_mask):
                text_mask = text_mask.to(device, non_blocking=True, dtype=torch.bool)
            else:
                text_mask = None
            encoded = module.model.encode_targets(
                residues,
                residue_padding_mask=mask,
                score_residue_embeddings=score_residues,
                score_residue_padding_mask=score_mask,
                capability_vectors=capability_vectors,
                capability_mask=capability_mask,
                text_vectors=text_vectors,
                text_mask=text_mask,
                retrieval_direction=retrieval_direction,
            )
            output.append(encoded.detach().to(storage_device))
            if progress_every_batches > 0 and (
                batch_idx == 1
                or batch_idx == total_batches
                or batch_idx % progress_every_batches == 0
            ):
                print(
                    f"Encoded candidate cache batch {batch_idx}/{total_batches} "
                    f"({batch_end}/{len(target_keys)} proteins)",
                    flush=True,
                )
    if not output:
        return torch.empty(0, 0, dtype=torch.float32, device=storage_device)
    return torch.cat(output, dim=0)


def main() -> None:
    args = parse_args()
    if args.suite_labels is not None and len(args.suite_labels) != len(args.suites):
        raise ValueError("--suite-labels length must match --suites length")
    if args.torch_num_threads is not None:
        torch.set_num_threads(args.torch_num_threads)

    config = load_config(args.config)
    kind = model_kind_from_config(config)
    module, kind = load_repo_checkpoint(args.checkpoint, config, args.device)

    project_root = Path.cwd()
    labels = args.suite_labels or [Path(path).stem for path in args.suites]
    tasks_by_suite: list[tuple[str, Path, list[BenchmarkTask]]] = []
    for label, suite in zip(labels, args.suites):
        suite_path = Path(suite)
        tasks = select_tasks(load_benchmark_suite(suite_path, project_root), args.tasks)
        tasks_by_suite.append((label, suite_path, tasks))

    (
        union_embeds,
        e2r_union_embeds,
        key_to_idx,
        task_candidate_keys,
        task_candidate_stats,
        _target_dataset,
        target_store_keys,
    ) = build_candidate_cache(
        tasks_by_suite=tasks_by_suite,
        config=config,
        kind=kind,
        protein_embedding=args.protein_embedding,
        score_protein_embedding=args.score_protein_embedding,
        module=module,
        device=args.device,
        target_batch_size=args.target_batch_size,
        store_targets_on_cpu=args.store_targets_on_cpu,
        progress_every_batches=args.progress_every_batches,
    )
    first_task = tasks_by_suite[0][2][0]
    primary_target_metadata = cached_target_encoding_metadata(
        task=first_task,
        config=config,
        kind=kind,
        checkpoint=args.checkpoint,
        config_path=args.config,
        protein_embedding=args.protein_embedding,
        score_protein_embedding=args.score_protein_embedding,
        retrieval_direction="reaction_to_enzyme",
    )
    target_encoding_metadata: dict[str, Any]
    if e2r_union_embeds is None:
        target_encoding_metadata = primary_target_metadata
    else:
        target_encoding_metadata = {
            "reaction_to_enzyme": primary_target_metadata,
            "enzyme_to_reaction": cached_target_encoding_metadata(
                task=first_task,
                config=config,
                kind=kind,
                checkpoint=args.checkpoint,
                config_path=args.config,
                protein_embedding=args.protein_embedding,
                score_protein_embedding=args.score_protein_embedding,
                retrieval_direction="enzyme_to_reaction",
            ),
        }
    config_fingerprint = primary_target_metadata.get("config")
    if not isinstance(config_fingerprint, dict) or "sha256" not in config_fingerprint:
        raise ValueError("Benchmark provenance requires a content-hashed config")
    config_sha256 = str(config_fingerprint["sha256"])

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for suite_label, suite_path, tasks in tasks_by_suite:
        output_dir = output_root / suite_label
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_suite = {
            "suite": str(suite_path),
            "checkpoint": args.checkpoint,
            "config": args.config,
            "protein_embedding": args.protein_embedding,
            "score_protein_embedding": args.score_protein_embedding,
            "device": args.device,
            "query_batch_size": args.query_batch_size,
            "target_batch_size": args.target_batch_size,
            "store_targets_on_cpu": bool(args.store_targets_on_cpu),
            "cached_candidate_union_size": len(key_to_idx),
            "tasks": [task_to_dict(task) for task in tasks],
        }
        write_json(output_dir / "resolved_suite.json", resolved_suite)

        results = []
        for task in tasks:
            print(f"Running benchmark task: {suite_label}/{task.name}", flush=True)
            reaction_inputs = build_reaction_inputs(task, config)
            eval_pairs = expand_bidirectional_pairs(
                task,
                read_pairs(task.pairs, task.reaction_id_col, task.protein_id_col),
            )
            candidate_keys = task_candidate_keys[(suite_label, task.name)]
            candidate_stats = task_candidate_stats[(suite_label, task.name)]
            validation_stats = validate_task_inputs(
                task,
                reaction_inputs,
                candidate_keys,
                target_store_keys,
                eval_pairs,
                candidate_stats,
            )
            reaction_to_proteins, protein_to_reactions = group_pairs(
                eval_pairs,
                allowed_reactions=set(reaction_inputs.keys),
                allowed_proteins=set(candidate_keys),
            )
            target_embeds = subset_target_embeds(union_embeds, key_to_idx, candidate_keys)
            e2r_target_embeds = (
                None
                if e2r_union_embeds is None
                else subset_target_embeds(e2r_union_embeds, key_to_idx, candidate_keys)
            )
            artifact_inputs = benchmark_artifact_inputs(task, target_encoding_metadata)
            result: dict[str, Any] = {
                "task": task.name,
                "dataset": task.dataset,
                "task_label": task.task_label,
                "setting": task.task_type,
                "split": task.split,
                "checkpoint": str(args.checkpoint),
                "config": str(args.config),
                "protein_embedding": args.protein_embedding,
                "score_protein_embedding": args.score_protein_embedding,
                "candidate_pool_policy": candidate_pool_policy(task, candidate_keys, eval_pairs),
                "metric_protocol": task.metric_protocol,
                "pairs": str(task.pairs),
                "reactions": str(task.reactions),
                "candidate_ids": None if task.candidate_ids is None else str(task.candidate_ids),
                "candidate_pool_size": len(candidate_keys),
                "reaction_input_mode": reaction_input_mode(config),
                "validation": validation_stats,
                "validate_only": False,
                "cached_candidate_union_size": len(key_to_idx),
            }
            if kind == "pooled":
                result["candidate_embedding_h5"] = str(task.candidate_embedding_h5)
            else:
                result["candidate_residue_h5"] = str(
                    select_residue_h5(task, args.protein_embedding)
                )
                if needs_score_residue_embeddings(config):
                    result["candidate_score_residue_h5"] = str(
                        select_score_residue_h5(task, args.score_protein_embedding)
                    )

            with torch.inference_mode():
                if task.is_screening:
                    metrics = evaluate_screening(
                        module,
                        reaction_inputs,
                        target_embeds,
                        candidate_keys,
                        reaction_to_proteins,
                        args.device,
                        args.query_batch_size,
                        task.bedroc_alphas,
                        task.ef_fractions,
                    )
                else:
                    metrics = evaluate_retrieval(
                        module,
                        reaction_inputs,
                        target_embeds,
                        candidate_keys,
                        reaction_to_proteins,
                        protein_to_reactions,
                        task.directions,
                        task.top_k,
                        args.device,
                        args.query_batch_size,
                        score_dump_task_name=f"{suite_label}__{task.name}",
                        e2r_target_embeds=e2r_target_embeds,
                        artifact_inputs=artifact_inputs,
                        config_sha256=config_sha256,
                        metric_protocol=task.metric_protocol,
                    )
            result.update(metrics)
            attach_benchmark_artifact_manifest(
                result,
                task=task,
                candidate_keys=candidate_keys,
                artifact_inputs=artifact_inputs,
                config_sha256=config_sha256,
                validate_only=False,
            )
            results.append(result)
            write_json(output_dir / f"{task.name}.json", result)
            print(json.dumps(result, indent=2)[:4000], flush=True)

        write_summary_csv(results, output_dir / "summary.csv")
        write_summary_markdown(results, output_dir / "summary.md")
        write_grouped_summary_tables(results, output_dir)
        print(f"Saved benchmark outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
