"""F3 checkpoint evaluation and selection on reaction-cluster validation panels."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

from horizyn.benchmarks.reactzyme_cluster_validation import (
    load_npz_vector_union,
    read_csv_rows,
)


SCHEMA_VERSION = "reactzyme_f3_cluster_checkpoint_audit_v1"
EPOCH_PATTERN = re.compile(r"protein-pooling-epoch=(\d+)\.ckpt$")
E2R_KEY = "enzyme_to_reaction/reactzyme_mrr"
R2E_KEY = "reaction_to_enzyme/reactzyme_mrr"


def discover_epoch_checkpoints(
    checkpoint_dir: Path,
    *,
    minimum_epoch: int = 10,
    maximum_epoch: int = 30,
) -> tuple[dict[int, Path], list[int]]:
    """Discover literal checkpoint epoch labels and report gaps in the requested span."""

    checkpoints: dict[int, Path] = {}
    for path in checkpoint_dir.glob("protein-pooling-epoch=*.ckpt"):
        match = EPOCH_PATTERN.match(path.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        if minimum_epoch <= epoch <= maximum_epoch:
            if epoch in checkpoints:
                raise ValueError(f"Duplicate checkpoint label for epoch {epoch}")
            checkpoints[epoch] = path.resolve()
    missing = [
        epoch
        for epoch in range(minimum_epoch, maximum_epoch + 1)
        if epoch not in checkpoints
    ]
    return dict(sorted(checkpoints.items())), missing


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_chemistry_subset(
    *,
    source_npz_paths: Sequence[Path],
    reactions_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write exact 617-d checkpoint-compatible vectors for a reaction subset."""

    _, reaction_rows = read_csv_rows(reactions_path)
    reaction_ids = [row["reaction_id"] for row in reaction_rows]
    vectors, mask = load_npz_vector_union(source_npz_paths, reaction_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ids=np.asarray(reaction_ids, dtype=str),
        vectors=vectors,
        mask=mask,
    )
    return {
        "path": str(output_path.resolve()),
        "num_reactions": len(reaction_ids),
        "vector_dim": int(vectors.shape[1]),
        "valid_masks": int(mask.sum()),
    }


def build_f3_panel_config(
    *,
    base_config_path: Path,
    panel_dir: Path,
    chemistry_path: Path,
    official_train_feature_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build an evaluator config without changing the saved F3 architecture."""

    config = copy.deepcopy(_load_yaml(base_config_path))
    data = config["data"]
    data["test_pairs_path"] = str((panel_dir / "validation_pairs.csv").resolve())
    data["test_reactions_path"] = str((panel_dir / "validation_rxns.csv").resolve())
    data["validation_retrieval_candidate_set"] = "custom"
    data["validation_retrieval_candidate_ids_path"] = str(
        (panel_dir / "validation_candidate_ids.txt").resolve()
    )
    modality_paths = {
        "t5v2": official_train_feature_dir / "reactiont5v2.h5",
        "unimol2": official_train_feature_dir / "unimol2.h5",
        "chiro": official_train_feature_dir / "chiro.h5",
    }
    for config_name, path in modality_paths.items():
        resolved = str(path.resolve())
        data[f"reaction_{config_name}_embeds_path"] = resolved
        data[f"validation_reaction_{config_name}_embeds_path"] = resolved
    data["reaction_chemistry_vectors_path"] = str(chemistry_path.resolve())
    data["validation_reaction_chemistry_vectors_path"] = str(chemistry_path.resolve())
    data["protein_residue_embeds_path"] = str(
        (Path(__file__).resolve().parents[2]
         / data["protein_residue_embeds_path"]).resolve()
        if not Path(data["protein_residue_embeds_path"]).is_absolute()
        else Path(data["protein_residue_embeds_path"]).resolve()
    )
    training = config.setdefault("training", {})
    training["validation_retrieval_candidate_set"] = "custom"
    training["validation_retrieval_candidate_ids_path"] = data[
        "validation_retrieval_candidate_ids_path"
    ]
    ablation = config.setdefault("ablation", {})
    ablation.update(
        {
            "evaluation_subset": "reaction_cluster_validation",
            "evaluation_protocol": "configured_forward_candidates",
            "reaction_cluster_panel": panel_dir.name,
            "checkpoint_compatible_f3_chemistry": True,
        }
    )
    _write_yaml(output_path, config)
    return config


def build_f3_cluster_training_config(
    *,
    base_config_path: Path,
    panel_dir: Path,
    chemistry_dir: Path,
    official_train_feature_dir: Path,
    run_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Create a clean F3 training config whose validation monitor is all-positive E->R."""

    config = copy.deepcopy(_load_yaml(base_config_path))
    data = config["data"]
    data.update(
        {
            "train_pairs_path": str((panel_dir / "train_pairs.csv").resolve()),
            "train_reactions_path": str((panel_dir / "train_rxns.csv").resolve()),
            "validation_pairs_path": str(
                (panel_dir / "validation_pairs.csv").resolve()
            ),
            "validation_reactions_path": str(
                (panel_dir / "validation_rxns.csv").resolve()
            ),
            "validation_retrieval_candidate_set": "custom",
            "validation_retrieval_candidate_ids_path": str(
                (panel_dir / "validation_candidate_ids.txt").resolve()
            ),
            "train_reaction_chemistry_vectors_path": str(
                (chemistry_dir / "train_reaction_set_features.npz").resolve()
            ),
            "validation_reaction_chemistry_vectors_path": str(
                (chemistry_dir / "validation_reaction_set_features.npz").resolve()
            ),
        }
    )
    for config_name, filename in (
        ("t5v2", "reactiont5v2.h5"),
        ("unimol2", "unimol2.h5"),
        ("chiro", "chiro.h5"),
    ):
        feature_path = str((official_train_feature_dir / filename).resolve())
        data[f"train_reaction_{config_name}_embeds_path"] = feature_path
        data[f"validation_reaction_{config_name}_embeds_path"] = feature_path
    protein_path = Path(data["protein_residue_embeds_path"])
    if not protein_path.is_absolute():
        protein_path = Path(__file__).resolve().parents[2] / protein_path
    data["protein_residue_embeds_path"] = str(protein_path.resolve())

    logging = config["logging"]
    logging.update(
        {
            "log_dir": str((run_root / "logs").resolve()),
            "checkpoint_dir": str((run_root / "checkpoints").resolve()),
            "checkpoint_monitor": "val/enzyme_to_reaction/reactzyme_mrr",
            "checkpoint_mode": "max",
            "checkpoint_on_validation_end": True,
            "save_top_k": -1,
        }
    )
    wandb = logging.setdefault("wandb", {})
    wandb.update(
        {
            "project": "horizyn-reactzyme-reaction-cluster-proxy-v1",
            "run_name": "f3-reaction-smi-cluster-0p85-seed42",
            "tags": sorted(
                set(
                    list(wandb.get("tags", []))
                    + [
                        "reaction-cluster-held-out",
                        "similarity-0p85",
                        "e2r-all-positive-monitor",
                        "retain-all-epochs",
                    ]
                )
            ),
        }
    )
    training = config["training"]
    training.update(
        {
            "validation_retrieval_candidate_set": "custom",
            "validation_retrieval_candidate_ids_path": data[
                "validation_retrieval_candidate_ids_path"
            ],
            "validation_retrieval_directions": [
                "reaction_to_enzyme",
                "enzyme_to_reaction",
            ],
            "check_val_every_n_epoch": 1,
        }
    )
    training.setdefault("early_stopping", {})["enabled"] = False
    config.setdefault("ablation", {}).update(
        {
            "run_id": "reaction_smi_F3_cluster_proxy_0p85",
            "data_protocol": "reactzyme-reaction-cluster-validation-v1-similarity-0p85",
            "evaluation_subset": "reaction_cluster_validation",
            "evaluation_protocol": "configured_forward_candidates",
            "checkpoint_selection": {
                "primary": "enzyme_to_reaction/reactzyme_mrr",
                "constraint": "reaction_to_enzyme/reactzyme_mrr >= f3_baseline - 0.015",
                "secondary": "harmonic_all_positive_mrr",
                "application": "offline because every validation epoch is retained",
            },
        }
    )
    _write_yaml(output_path, config)
    return config


def build_f3_cluster_test_config(
    *,
    base_config_path: Path,
    test_protocol_dir: Path,
    chemistry_path: Path,
    official_test_feature_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Create the untouched Reaction-Sim test config for a clean cluster-trained F3."""

    config = copy.deepcopy(_load_yaml(base_config_path))
    data = config["data"]
    data.update(
        {
            "test_pairs_path": str((test_protocol_dir / "test_pairs.csv").resolve()),
            "test_reactions_path": str((test_protocol_dir / "test_rxns.csv").resolve()),
            "validation_retrieval_candidate_set": "custom",
            "validation_retrieval_candidate_ids_path": str(
                (test_protocol_dir / "test_candidate_ids.txt").resolve()
            ),
            "reaction_chemistry_vectors_path": str(chemistry_path.resolve()),
            "validation_reaction_chemistry_vectors_path": str(
                chemistry_path.resolve()
            ),
        }
    )
    for config_name, filename in (
        ("t5v2", "reactiont5v2.h5"),
        ("unimol2", "unimol2.h5"),
        ("chiro", "chiro.h5"),
    ):
        feature_path = str((official_test_feature_dir / filename).resolve())
        data[f"reaction_{config_name}_embeds_path"] = feature_path
        data[f"validation_reaction_{config_name}_embeds_path"] = feature_path
    protein_path = Path(data["protein_residue_embeds_path"])
    if not protein_path.is_absolute():
        protein_path = Path(__file__).resolve().parents[2] / protein_path
    data["protein_residue_embeds_path"] = str(protein_path.resolve())
    training = config.setdefault("training", {})
    training["validation_retrieval_candidate_set"] = "custom"
    training["validation_retrieval_candidate_ids_path"] = data[
        "validation_retrieval_candidate_ids_path"
    ]
    config.setdefault("ablation", {}).update(
        {
            "evaluation_subset": "released_test",
            "evaluation_protocol": "paper_test_candidates",
            "data_protocol": "reactzyme-official-reaction-smi-test",
            "cluster_train_fitted_chemistry": True,
        }
    )
    _write_yaml(output_path, config)
    return config


def harmonic_mrr(e2r: float, r2e: float) -> float:
    return 0.0 if e2r + r2e == 0.0 else 2.0 * e2r * r2e / (e2r + r2e)


def audit_checkpoint_training_overlap(
    *,
    panel_pairs_path: Path,
    checkpoint_training_pairs_path: Path,
) -> dict[str, Any]:
    """Measure whether a retrospective panel was already seen by a checkpoint."""

    _, panel_rows = read_csv_rows(panel_pairs_path)
    _, training_rows = read_csv_rows(checkpoint_training_pairs_path)
    panel_reactions = {row["reaction_id"] for row in panel_rows}
    training_reactions = {row["reaction_id"] for row in training_rows}
    panel_pairs = {(row["reaction_id"], row["protein_id"]) for row in panel_rows}
    training_pairs = {(row["reaction_id"], row["protein_id"]) for row in training_rows}
    panel_proteins = {row["protein_id"] for row in panel_rows}
    training_proteins = {row["protein_id"] for row in training_rows}
    reaction_overlap = len(panel_reactions & training_reactions)
    pair_overlap = len(panel_pairs & training_pairs)
    protein_overlap = len(panel_proteins & training_proteins)
    return {
        "checkpoint_training_pairs_path": str(checkpoint_training_pairs_path.resolve()),
        "panel_reactions": len(panel_reactions),
        "overlapping_reactions": reaction_overlap,
        "reaction_overlap_fraction": reaction_overlap / max(len(panel_reactions), 1),
        "panel_pair_keys": len(panel_pairs),
        "overlapping_pair_keys": pair_overlap,
        "pair_overlap_fraction": pair_overlap / max(len(panel_pairs), 1),
        "panel_proteins": len(panel_proteins),
        "overlapping_proteins": protein_overlap,
        "protein_overlap_fraction": protein_overlap / max(len(panel_proteins), 1),
        "retrospective_selection_valid": reaction_overlap == 0 and pair_overlap == 0,
    }


def result_row(payload: dict[str, Any], *, epoch: int, path: Path) -> dict[str, Any]:
    e2r = float(payload[E2R_KEY])
    r2e = float(payload[R2E_KEY])
    return {
        "epoch": epoch,
        "checkpoint": payload.get("checkpoint"),
        "result_path": str(path.resolve()),
        "e2r_all_positive_mrr": e2r,
        "r2e_all_positive_mrr": r2e,
        "balanced_all_positive_mrr": (e2r + r2e) / 2.0,
        "harmonic_all_positive_mrr": harmonic_mrr(e2r, r2e),
        "e2r_first_positive_mrr": float(
            payload.get("enzyme_to_reaction/first_positive_mrr", 0.0)
        ),
        "r2e_first_positive_mrr": float(
            payload.get("reaction_to_enzyme/first_positive_mrr", 0.0)
        ),
        "num_enzyme_queries": int(payload.get("enzyme_to_reaction/num_queries", 0)),
        "num_reaction_queries": int(payload.get("reaction_to_enzyme/num_queries", 0)),
        "num_enzyme_candidates": int(payload.get("num_enzyme_candidates", 0)),
        "num_reaction_candidates": int(payload.get("num_reaction_candidates", 0)),
    }


def load_panel_results(result_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(result_dir.glob("epoch_*.json")):
        match = re.fullmatch(r"epoch_(\d+)\.json", path.name)
        if match is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(result_row(payload, epoch=int(match.group(1)), path=path))
    return sorted(rows, key=lambda row: int(row["epoch"]))


def select_checkpoint(
    rows: Sequence[dict[str, Any]],
    *,
    baseline_epoch: int,
    r2e_tolerance: float,
) -> dict[str, Any]:
    """Maximize E->R subject to an epoch-baseline R->E floor."""

    if not 0.0 <= r2e_tolerance <= 1.0:
        raise ValueError("r2e_tolerance must be between 0 and 1")
    baseline = next(
        (row for row in rows if int(row["epoch"]) == baseline_epoch), None
    )
    if baseline is None:
        raise ValueError(f"Baseline epoch {baseline_epoch} has no evaluation result")
    floor = float(baseline["r2e_all_positive_mrr"]) - r2e_tolerance
    annotated = [
        {**row, "r2e_eligible": float(row["r2e_all_positive_mrr"]) >= floor}
        for row in rows
    ]
    eligible = [row for row in annotated if row["r2e_eligible"]]
    if not eligible:
        raise ValueError("No checkpoint satisfies the R->E constraint")
    selected = max(
        eligible,
        key=lambda row: (
            float(row["e2r_all_positive_mrr"]),
            float(row["harmonic_all_positive_mrr"]),
            float(row["r2e_all_positive_mrr"]),
            -int(row["epoch"]),
        ),
    )
    return {
        "primary_metric": "enzyme_to_reaction/reactzyme_mrr",
        "primary_metric_definition": "enzyme-anchor all-positive mean reciprocal rank",
        "baseline_epoch": baseline_epoch,
        "baseline_r2e_all_positive_mrr": float(baseline["r2e_all_positive_mrr"]),
        "r2e_tolerance": r2e_tolerance,
        "r2e_floor": floor,
        "secondary_metric": "harmonic_all_positive_mrr",
        "selected": selected,
        "rows": annotated,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    available = ", ".join(str(value) for value in report["available_epochs"]) or "none"
    missing = ", ".join(str(value) for value in report["missing_epochs"]) or "none"
    lines = [
        "# F3 Reaction-Cluster Checkpoint Audit",
        "",
        f"Panel: `{report['panel']}`",
        "",
        f"Requested literal checkpoint labels: {report['minimum_epoch']}–{report['maximum_epoch']}",
        "",
        f"Available: {available}",
        "",
        f"Missing: {missing}",
        "",
        "The primary metric is benchmark-compatible enzyme-anchor, all-positive E→R MRR. "
        "A checkpoint is eligible only if all-positive R→E MRR is within the configured "
        "tolerance of the epoch baseline. Harmonic all-positive MRR is the secondary tie-breaker.",
        "",
    ]
    overlap = report.get("checkpoint_training_overlap")
    if overlap and not overlap["retrospective_selection_valid"]:
        lines.extend(
            [
                "**Invalid for retrospective checkpoint selection:** "
                f"{overlap['overlapping_reactions']}/{overlap['panel_reactions']} panel "
                "reactions and "
                f"{overlap['overlapping_pair_keys']}/{overlap['panel_pair_keys']} exact pair "
                "keys were already in the checkpoints' training set. Values below are "
                "contamination diagnostics only.",
                "",
            ]
        )
    lines.extend(
        [
        "| Epoch | E→R all-positive MRR | R→E all-positive MRR | Balanced | Harmonic | Eligible |",
        "|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    selection = report.get("selection") or report.get("diagnostic_selection")
    selected_epoch = (
        int(selection["selected"]["epoch"])
        if selection is not None
        else None
    )
    rows: Iterable[dict[str, Any]] = (
        selection["rows"] if selection is not None else report["rows"]
    )
    for row in rows:
        marker_label = "selected" if report.get("selection") is not None else "diagnostic best"
        marker = f" **{marker_label}**" if int(row["epoch"]) == selected_epoch else ""
        eligible = row.get("r2e_eligible")
        lines.append(
            f"| {row['epoch']}{marker} | {row['e2r_all_positive_mrr']:.6f} | "
            f"{row['r2e_all_positive_mrr']:.6f} | "
            f"{row['balanced_all_positive_mrr']:.6f} | "
            f"{row['harmonic_all_positive_mrr']:.6f} | "
            f"{'yes' if eligible else 'no' if eligible is not None else 'pending'} |"
        )
    if report.get("selection") is not None:
        lines.extend(
            [
                "",
                f"R→E floor: `{selection['r2e_floor']:.6f}` "
                f"(epoch {selection['baseline_epoch']} minus {selection['r2e_tolerance']:.3f}).",
                "",
                f"Selected checkpoint: epoch `{selection['selected']['epoch']}`.",
            ]
        )
    elif report.get("diagnostic_selection") is not None:
        lines.extend(
            [
                "",
                "No checkpoint is promoted from this retrospective audit because the "
                "panel overlaps checkpoint training data.",
            ]
        )
    return "\n".join(lines) + "\n"


def write_panel_report(
    *,
    panel: str,
    result_dir: Path,
    output_dir: Path,
    available_epochs: Sequence[int],
    missing_epochs: Sequence[int],
    minimum_epoch: int,
    maximum_epoch: int,
    baseline_epoch: int,
    r2e_tolerance: float,
    panel_pairs_path: Path,
    checkpoint_training_pairs_path: Path,
) -> dict[str, Any]:
    rows = load_panel_results(result_dir)
    diagnostic_selection = (
        select_checkpoint(
            rows,
            baseline_epoch=baseline_epoch,
            r2e_tolerance=r2e_tolerance,
        )
        if rows and any(int(row["epoch"]) == baseline_epoch for row in rows)
        else None
    )
    overlap = audit_checkpoint_training_overlap(
        panel_pairs_path=panel_pairs_path,
        checkpoint_training_pairs_path=checkpoint_training_pairs_path,
    )
    selection = (
        diagnostic_selection if overlap["retrospective_selection_valid"] else None
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "panel": panel,
        "minimum_epoch": minimum_epoch,
        "maximum_epoch": maximum_epoch,
        "available_epochs": list(available_epochs),
        "missing_epochs": list(missing_epochs),
        "rows": rows,
        "selection": selection,
        "diagnostic_selection": (
            diagnostic_selection if selection is None else None
        ),
        "checkpoint_training_overlap": overlap,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{panel}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / f"{panel}.md").write_text(
        render_markdown_report(report), encoding="utf-8"
    )
    return report
