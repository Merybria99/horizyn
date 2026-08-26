#!/usr/bin/env python3
"""Run one reaction through multiple model query configurations."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from wet_lab.query import PROJECT_ROOT, _resolve_path, load_query_config, run_query


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty campaign table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    path: Path,
    *,
    reaction: dict[str, str],
    model_results: list[dict[str, Any]],
    consensus_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Wet-Lab Query Campaign",
        "",
        f"- Reaction: `{reaction['id']}`",
        f"- Models: {len(model_results)}",
        "- Scores are cosine retrieval similarities, not calibrated probabilities.",
        "- Consensus uses reciprocal-rank fusion and does not average raw model scores.",
        "",
        "## Per-Model Top 10",
        "",
    ]
    for result in model_results:
        lines.extend(
            [
                f"### {result['model']['name']}",
                "",
                "| Rank | Protein | Cosine similarity |",
                "|---:|:--|---:|",
            ]
        )
        for row in result["rankings"][:10]:
            lines.append(
                f"| {row['rank']} | {row['protein_id']} | " f"{row['cosine_similarity']:.6f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Consensus Top 20",
            "",
            "| Consensus rank | Protein | RRF score | Model support | Best rank |",
            "|---:|:--|---:|---:|---:|",
        ]
    )
    for row in consensus_rows[:20]:
        lines.append(
            f"| {row['consensus_rank']} | {row['protein_id']} | "
            f"{row['rrf_score']:.8f} | {row['model_support']} | "
            f"{row['best_rank']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def consolidate_results(
    *,
    reaction: dict[str, str],
    result_paths: list[Path],
    output_dir: Path,
    rrf_constant: float,
    max_rank: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    model_names = [str(result["model"]["name"]) for result in model_results]
    if len(model_names) != len(set(model_names)):
        raise ValueError("Campaign model names must be unique")

    long_rows: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "rrf_score": 0.0,
            "model_support": 0,
            "best_rank": max_rank + 1,
            "models": [],
        }
    )
    for result in model_results:
        model_name = str(result["model"]["name"])
        for row in result["rankings"]:
            rank = int(row["rank"])
            if rank > max_rank:
                continue
            protein_id = str(row["protein_id"])
            score = float(row["cosine_similarity"])
            long_rows.append(
                {
                    "model": model_name,
                    "rank": rank,
                    "protein_id": protein_id,
                    "cosine_similarity": score,
                }
            )
            item = aggregate[protein_id]
            item["rrf_score"] += 1.0 / (rrf_constant + rank)
            item["model_support"] += 1
            item["best_rank"] = min(item["best_rank"], rank)
            item["models"].append(model_name)

    ordered = sorted(
        aggregate.items(),
        key=lambda item: (
            -float(item[1]["rrf_score"]),
            -int(item[1]["model_support"]),
            int(item[1]["best_rank"]),
            item[0],
        ),
    )
    consensus_rows = [
        {
            "consensus_rank": rank,
            "protein_id": protein_id,
            "rrf_score": values["rrf_score"],
            "model_support": values["model_support"],
            "best_rank": values["best_rank"],
            "models": "; ".join(values["models"]),
        }
        for rank, (protein_id, values) in enumerate(ordered, start=1)
    ]
    _write_csv(output_dir / "per_model_rankings.csv", long_rows)
    _write_csv(output_dir / "consensus.csv", consensus_rows)
    _write_summary(
        output_dir / "summary.md",
        reaction=reaction,
        model_results=model_results,
        consensus_rows=consensus_rows,
    )

    campaign_result = {
        "schema_version": "horizyn_wet_lab_f4_campaign_v1",
        "reaction": reaction,
        "models": [
            {
                "name": result["model"]["name"],
                "checkpoint": result["model"]["checkpoint"],
                "result": str(path),
                "top_1": result["rankings"][0],
            }
            for result, path in zip(model_results, result_paths)
        ],
        "consensus": {
            "method": "reciprocal_rank_fusion",
            "rrf_constant": rrf_constant,
            "max_rank": max_rank,
            "rankings": consensus_rows,
        },
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(campaign_result, indent=2) + "\n", encoding="utf-8")
    return result_path


def run_campaign(config_path: str | Path) -> Path:
    path = Path(config_path).expanduser().resolve()
    campaign = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    reaction = campaign.get("reaction")
    model_configs = campaign.get("models")
    if not isinstance(reaction, dict) or not reaction.get("id") or not reaction.get("smiles"):
        raise ValueError("Campaign config requires reaction.id and reaction.smiles")
    if not isinstance(model_configs, list) or not model_configs:
        raise ValueError("Campaign config requires a non-empty models list")

    result_paths: list[Path] = []
    for index, model_config_path in enumerate(model_configs, start=1):
        query_config = load_query_config(_resolve_path(model_config_path))
        query_config["reaction"] = dict(reaction)
        print(
            f"\n[{index}/{len(model_configs)}] "
            f"{query_config['model'].get('name', model_config_path)}",
            flush=True,
        )
        result_paths.append(run_query(query_config))

    consensus = campaign.get("consensus", {})
    output_dir = _resolve_path(
        campaign.get("output", {}).get(
            "directory",
            f"wet_lab/reports/{reaction['id']}_all_f4",
        ),
        must_exist=False,
    )
    result_path = consolidate_results(
        reaction=dict(reaction),
        result_paths=result_paths,
        output_dir=output_dir,
        rrf_constant=float(consensus.get("rrf_constant", 60.0)),
        max_rank=int(consensus.get("max_rank", 100)),
    )
    print(f"\nSaved F4 campaign results to: {result_path}", flush=True)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    run_campaign(args.config)


if __name__ == "__main__":
    main()
