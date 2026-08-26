#!/usr/bin/env python3
"""Build the ReactZyme/TIGER comparison and TIGER-style Hit@k tables."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("time", "enzyme_smi", "reaction_smi")
DIRECTIONS = ("enzyme_to_reaction", "reaction_to_enzyme")
METRICS = ("top_1", "top_5", "top_10", "reactzyme_mrr")


@dataclass(frozen=True)
class Model:
    key: str
    label: str
    definition: str


BASE_MODELS = (
    Model("Q5_combined", "Q5 Combined", "Combined Q1, Q2, Q3, and Q4 changes."),
    Model(
        "Q4_factorized_modalities",
        "Q4 Factorized modalities",
        "Q1 plus factorized raw ReactionT5v2, UniMol2, and ChIRo inputs.",
    ),
    Model("Q1_e2r_adapter", "Q1 E->R adapter", "Q0 plus a frozen-base E->R residual reaction adapter."),
    Model("Q0_f3", "Q0 Fresh F3 baseline", "Freshly trained F3 baseline."),
    Model("Q3_dense_transform", "Q3 Dense chemistry", "Q1 plus dense directional and reaction-center chemistry."),
    Model(
        "F3_set_chemistry",
        "F3 Set chemistry",
        "F1 plus train-fitted molecule-set descriptors, fingerprints, and cofactors.",
    ),
    Model("Q2_e2r_hardneg", "Q2 Hard negatives", "Q1 plus mined E->R hard-negative training."),
    Model(
        "F0_directional_delta_control",
        "F0 Directional-delta control",
        "Directional-delta reaction composition control.",
    ),
    Model(
        "F1_molecule_set",
        "F1 Unordered molecule set",
        "UniMol2 and ChIRo participants treated as one unordered molecule set.",
    ),
    Model(
        "F6_rhea_center",
        "F6 Rhea reaction center",
        "Factorized reaction modalities plus Rhea direction and reaction-center features.",
    ),
)

SHORTCUT_MODELS = (
    Model("S0", "S0 F3 control", "Exact F3 single-head attention control on the audited protocol."),
    Model("S1", "S1 Normalized F3", "S0 with scale-preserving modality L2 normalization."),
    Model("S2", "S2 Bounded F3", "S1 with prior-bounded adaptive single-head attention."),
)
PARTIAL_MODELS = (
    Model(
        "S3",
        "S3 Bounded + chemistry dropout",
        "S2 with chemistry-only dropout at probability `0.25`.",
    ),
)
COMPLETE_MODELS = BASE_MODELS + SHORTCUT_MODELS
ALL_MODELS = COMPLETE_MODELS + PARTIAL_MODELS
DEFAULT_SHORTCUT_RUN_ROOT = ROOT / "runs" / "reactzyme_f3_shortcut_ablation_v1"

# TIGER Table 1. These remain separate from the appendix values below.
MAIN_REFERENCES = (
    (
        "ReactZyme (UniMol-3D + ESM)",
        {
            "time": (0.410, 0.140),
            "enzyme_smi": (0.811, 0.293),
            "reaction_smi": (0.201, 0.134),
        },
    ),
    (
        "TIGER ESM2Text",
        {
            "time": (0.690, 0.366),
            "enzyme_smi": (0.956, 0.592),
            "reaction_smi": (0.518, 0.319),
        },
    ),
    (
        "TIGER ProtT3",
        {
            "time": (0.683, 0.372),
            "enzyme_smi": (0.940, 0.579),
            "reaction_smi": (0.472, 0.337),
        },
    ),
)

# TIGER Appendix Tables 4, 6, 8, 10, 12, and 14. Each tuple is
# (H@1, H@5, H@10, MRR). The paper does not provide ProtT3 Hit@5/Hit@10.
APPENDIX_REFERENCES = {
    ("time", "enzyme_to_reaction"): (
        ("ReactZyme (UniMol-3D + ESM)", (0.2905, 0.5365, 0.6586, 0.4104)),
        ("TIGER (ESM2Text)", (0.5810, 0.8190, 0.8740, 0.6902)),
    ),
    ("time", "reaction_to_enzyme"): (
        ("ReactZyme (UniMol-3D + ESM)", (0.1678, 0.3155, 0.3960, 0.1400)),
        ("TIGER (ESM2Text)", (0.4536, 0.6708, 0.7676, 0.3658)),
    ),
    ("enzyme_smi", "enzyme_to_reaction"): (
        ("ReactZyme (UniMol-3D + ESM)", (0.7267, 0.9062, 0.9487, 0.8112)),
        ("TIGER (ESM2Text)", (0.9308, 0.9850, 0.9916, 0.9561)),
    ),
    ("enzyme_smi", "reaction_to_enzyme"): (
        ("ReactZyme (UniMol-3D + ESM)", (0.4088, 0.6892, 0.7953, 0.2930)),
        ("TIGER (ESM2Text)", (0.7921, 0.9408, 0.9688, 0.5921)),
    ),
    ("reaction_smi", "enzyme_to_reaction"): (
        ("ReactZyme (UniMol-3D + ESM)", (0.0912, 0.2580, 0.4213, 0.1856)),
        ("TIGER (ESM2Text)", (0.4155, 0.6416, 0.6827, 0.5180)),
    ),
    ("reaction_smi", "reaction_to_enzyme"): (
        ("ReactZyme (UniMol-3D + ESM)", (0.0924, 0.1332, 0.1790, 0.0943)),
        ("TIGER (ESM2Text)", (0.4305, 0.6113, 0.6994, 0.3185)),
    ),
}

SPLIT_LABELS = {
    "time": "Time",
    "enzyme_smi": "Enzyme Similarity",
    "reaction_smi": "Reaction Similarity",
}
DIRECTION_LABELS = {
    "enzyme_to_reaction": "E->R",
    "reaction_to_enzyme": "R->E",
}


def _load_result(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        f"{direction}/{metric}"
        for direction in DIRECTIONS
        for metric in METRICS
    }
    absent = sorted(required - payload.keys())
    if absent:
        raise ValueError(f"{path}: missing {', '.join(absent)}")
    return payload


def load_results(
    run_root: Path,
    shortcut_run_root: Path = DEFAULT_SHORTCUT_RUN_ROOT,
) -> dict[str, dict[str, dict]]:
    results: dict[str, dict[str, dict]] = {}
    missing = []
    for model in BASE_MODELS:
        results[model.key] = {}
        for split in SPLITS:
            path = run_root / "eval" / model.key / f"{split}.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            try:
                results[model.key][split] = _load_result(path)
            except ValueError as exc:
                missing.append(str(exc))

    for model in SHORTCUT_MODELS:
        results[model.key] = {}
        for split in SPLITS:
            path = (
                shortcut_run_root
                / "results"
                / "exploratory_test_finished"
                / model.key
                / split
                / "seed42"
                / "result.json"
            )
            if not path.is_file():
                missing.append(str(path))
                continue
            try:
                results[model.key][split] = _load_result(path)
            except ValueError as exc:
                missing.append(str(exc))

    for model in PARTIAL_MODELS:
        results[model.key] = {}
        for split in SPLITS:
            path = (
                shortcut_run_root
                / "results"
                / "exploratory_test_finished"
                / model.key
                / split
                / "seed42"
                / "result.json"
            )
            if path.is_file():
                results[model.key][split] = _load_result(path)
    if missing:
        raise FileNotFoundError("Incomplete Hit@k evaluation:\n" + "\n".join(missing))
    return results


def value(results: dict, model: str, split: str, direction: str, metric: str) -> float:
    return float(results[model][split][f"{direction}/{metric}"])


def macro_mrr(results: dict, model: str) -> float:
    return sum(
        value(results, model, split, direction, "reactzyme_mrr")
        for split in SPLITS
        for direction in DIRECTIONS
    ) / 6.0


def bold_if_max(number: float, maximum: float) -> str:
    formatted = f"{number:.4f}"
    return f"**{formatted}**" if abs(number - maximum) < 5e-8 else formatted


def build_main_table(results: dict, ranked: list[Model]) -> list[str]:
    rows: list[tuple[str, str, list[float | None]]] = []
    for name, cells in MAIN_REFERENCES:
        values = [number for split in SPLITS for number in cells[split]]
        rows.append(("Ref.", name, values + [sum(values) / len(values)]))
    for rank, model in enumerate(ranked, start=1):
        values = [
            value(results, model.key, split, direction, "reactzyme_mrr")
            for split in SPLITS
            for direction in DIRECTIONS
        ]
        rows.append((str(rank), model.label, values + [sum(values) / len(values)]))
    for model in PARTIAL_MODELS:
        values = [
            (
                value(results, model.key, split, direction, "reactzyme_mrr")
                if split in results[model.key]
                else None
            )
            for split in SPLITS
            for direction in DIRECTIONS
        ]
        rows.append(("Partial", model.label, values + [None]))
    maxima = [
        max(float(row[2][index]) for row in rows if row[2][index] is not None)
        for index in range(7)
    ]
    lines = [
        "| Rank | Method | Time E->R | Time R->E | Enzyme-Sim E->R | Enzyme-Sim R->E | Reaction-Sim E->R | Reaction-Sim R->E | Macro |",
        "|:--:|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for rank, name, values in rows:
        rendered = [
            bold_if_max(float(number), maxima[index]) if number is not None else "-"
            for index, number in enumerate(values)
        ]
        lines.append(f"| {rank} | {name} | " + " | ".join(rendered) + " |")
    return lines


def build_hitk_table(results: dict, ranked: list[Model], split: str, direction: str) -> list[str]:
    rows: list[tuple[str, str, tuple[float, ...]]] = [
        ("Ref.", name, numbers)
        for name, numbers in APPENDIX_REFERENCES[(split, direction)]
    ]
    for rank, model in enumerate(ranked, start=1):
        numbers = tuple(value(results, model.key, split, direction, metric) for metric in METRICS)
        rows.append((str(rank), model.label, numbers))
    for model in PARTIAL_MODELS:
        if split not in results[model.key]:
            continue
        numbers = tuple(value(results, model.key, split, direction, metric) for metric in METRICS)
        rows.append(("Partial", model.label, numbers))
    maxima = [max(row[2][index] for row in rows) for index in range(len(METRICS))]
    lines = [
        "| Rank | Method | H@1 | H@5 | H@10 | MRR |",
        "|:--:|:--|--:|--:|--:|--:|",
    ]
    for rank, name, numbers in rows:
        rendered = [bold_if_max(number, maxima[index]) for index, number in enumerate(numbers)]
        lines.append(f"| {rank} | {name} | " + " | ".join(rendered) + " |")
    return lines


def build_report(results: dict) -> str:
    ranked = sorted(
        COMPLETE_MODELS,
        key=lambda model: macro_mrr(results, model.key),
        reverse=True,
    )[:10]
    best = ranked[0]
    shortcut_ranked = sorted(
        SHORTCUT_MODELS,
        key=lambda model: macro_mrr(results, model.key),
        reverse=True,
    )
    lines = [
        "# ReactZyme and TIGER vs Horizyn Top 10",
        "",
        f"Last updated: {date.today().isoformat()}",
        "",
        "## Comparison Protocol",
        "",
        "- Results are held-out ReactZyme test metrics.",
        "- `E->R` means enzyme-to-reaction retrieval; `R->E` means reaction-to-enzyme retrieval.",
        "- Horizyn uses the official paper test candidate pools and canonical forward reactions.",
        "- Horizyn MRR is the all-positive ReactZyme definition; H@k records whether any known positive is in the top k.",
        "- Horizyn models are ranked by the arithmetic mean of the six split/direction MRR cells.",
        "- Horizyn results are single-seed results from seed 42.",
        "- Horizyn cells were freshly recomputed from the checkpoint/config paths recorded by the historical official tests. This report uses those current-code results rather than copying the older aggregate JSON.",
        "- S0-S3 are exploratory paper-test evaluations of checkpoints selected only by validation. S0-S2 have all six cells; S3 currently has Time-Sim only and is excluded from the ranked top ten and six-cell macro comparison.",
        "- The summary table uses TIGER Table 1. The six H@k tables use TIGER Appendix Tables 4, 6, 8, 10, 12, and 14 without reconciling paper-internal discrepancies.",
        "- The appendix reports H@5/H@10 only for `TIGER (Ours)`, corresponding to the paper's primary ESM2Text configuration; ProtT3 is therefore absent from the H@k tables.",
        "- The L-series loss campaign is excluded because only validation results were retained.",
        "",
        "## Test Leaderboard",
        "",
        "`Rank` orders only Horizyn models. Published methods are marked `Ref.`. Bold marks the best result in each column.",
        "",
        *build_main_table(results, ranked),
        "",
        "## TIGER-Style Hit@k Tables",
        "",
        "These tables reproduce the split-by-direction layout of the TIGER appendix while adding the Horizyn top ten.",
        "",
    ]
    for split in SPLITS:
        for direction in DIRECTIONS:
            lines.extend(
                [
                    f"### {SPLIT_LABELS[split]} Split, {DIRECTION_LABELS[direction]}",
                    "",
                    *build_hitk_table(results, ranked, split, direction),
                    "",
                ]
            )
    lines.extend(
        [
            "## Main Results",
            "",
            f"1. {best.label} has the strongest six-cell macro MRR among the evaluated Horizyn models at `{macro_mrr(results, best.key):.4f}`.",
            f"2. {shortcut_ranked[0].key} is the strongest complete shortcut-control variant at `{macro_mrr(results, shortcut_ranked[0].key):.4f}`, followed by {shortcut_ranked[1].key} at `{macro_mrr(results, shortcut_ranked[1].key):.4f}` and {shortcut_ranked[2].key} at `{macro_mrr(results, shortcut_ranked[2].key):.4f}`.",
            "3. S1 sets the best Enzyme-Sim R->E MRR in this table at `0.6840`, but its Reaction-Sim R->E MRR remains below Q/F-series results.",
            "4. S3 currently underperforms S0-S2 on both Time-Sim directions; its other splits are not yet represented.",
            "5. Hit@5 and Hit@10 are measured directly from checkpoint rankings; they are not interpolated from Hit@1 or MRR.",
            "6. The appendix reference rows are transcribed independently from the Table 1 summary because the paper reports different ReactZyme values in a few cells.",
            "",
            "## Model Definitions",
            "",
            "| ID | Definition |",
            "|:--|:--|",
        ]
    )
    for model in sorted(ALL_MODELS, key=lambda item: item.key):
        lines.append(f"| {model.key.split('_', 1)[0]} | {model.definition} |")
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "- Recomputed Hit@k artifacts: `horizyn/runs/reactzyme_tiger_top10_hitk_v1/eval/`",
            "- Recomputed evaluation logs: `horizyn/runs/reactzyme_tiger_top10_hitk_v1/logs/`",
            "- Complete ReactZyme ablation ledger: `documents/README_reactzyme_all_ablations.md`",
            "- F-series source tests: `horizyn/runs/reactzyme_reaction_features_v1/eval/`",
            "- Q-series source tests: `horizyn/runs/reactzyme_e2r_pareto_v1/eval/seed42/`",
            "- S-series exploratory tests: `horizyn/runs/reactzyme_f3_shortcut_ablation_v1/results/exploratory_test_finished/`",
            "- S-series validation-selected checkpoints: `horizyn/runs/reactzyme_f3_shortcut_ablation_v1/results/validation/`",
            "- TIGER paper: <https://arxiv.org/pdf/2605.24489>",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "runs" / "reactzyme_tiger_top10_hitk_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "documents" / "README_reactzyme_tiger_top10.md",
    )
    parser.add_argument(
        "--shortcut-run-root",
        type=Path,
        default=DEFAULT_SHORTCUT_RUN_ROOT,
    )
    args = parser.parse_args()
    results = load_results(
        args.run_root.resolve(),
        args.shortcut_run_root.resolve(),
    )
    report = build_report(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
