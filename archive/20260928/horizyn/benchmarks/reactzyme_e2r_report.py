"""Report and select ReactZyme E2R Pareto variants using validation guardrails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from horizyn.benchmarks.reactzyme_e2r_pareto import SPLITS, VARIANTS


R2E_KEY = "reaction_to_enzyme/reactzyme_mrr"
E2R_KEY = "enzyme_to_reaction/reactzyme_mrr"
R2E_TOLERANCE = 1e-6
NON_PRIMARY_E2R_DROP_TOLERANCE = 0.01


def _result_path(
    run_root: Path,
    seed: int,
    split: str,
    variant: str,
    subset: str,
) -> Path:
    return run_root / "eval" / f"seed{seed}" / split / variant / f"{subset}.json"


def _load_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if R2E_KEY not in payload or E2R_KEY not in payload:
        raise ValueError(f"Missing paper MRR fields in {path}")
    return payload


def _variant_summary(
    run_root: Path,
    *,
    seed: int,
    variant: str,
    subset: str,
) -> dict[str, Any] | None:
    baseline = {
        split: _load_result(
            _result_path(run_root, seed, split, "Q0_f3", subset)
        )
        for split in SPLITS
    }
    candidate = {
        split: _load_result(_result_path(run_root, seed, split, variant, subset))
        for split in SPLITS
    }
    if any(value is None for value in baseline.values()):
        return None
    if variant != "Q0_f3" and any(value is None for value in candidate.values()):
        return None

    gates: dict[str, dict[str, Any]] = {}
    effective: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        base = baseline[split]
        current = base if variant == "Q0_f3" else candidate[split]
        r2e_delta = float(current[R2E_KEY]) - float(base[R2E_KEY])
        e2r_delta = float(current[E2R_KEY]) - float(base[E2R_KEY])
        r2e_ok = abs(r2e_delta) <= R2E_TOLERANCE
        e2r_ok = (
            True
            if split == "reaction_smi"
            else e2r_delta >= -NON_PRIMARY_E2R_DROP_TOLERANCE
        )
        accepted = variant == "Q0_f3" or (r2e_ok and e2r_ok)
        gates[split] = {
            "accepted": accepted,
            "r2e_delta": r2e_delta,
            "e2r_delta": e2r_delta,
            "r2e_ok": r2e_ok,
            "non_primary_e2r_ok": e2r_ok,
        }
        effective[split] = current if accepted else base

    return {
        "seed": seed,
        "variant": variant,
        "subset": subset,
        "complete": True,
        "gates": gates,
        "all_validation_guardrails_pass": all(
            gate["accepted"] for gate in gates.values()
        ),
        "raw": candidate if variant != "Q0_f3" else baseline,
        "effective": effective,
        "primary_e2r": float(effective["reaction_smi"][E2R_KEY]),
        "macro_mrr": sum(
            float(effective[split][R2E_KEY]) + float(effective[split][E2R_KEY])
            for split in SPLITS
        )
        / 6.0,
    }


def build_report(run_root: Path, *, seed: int = 42) -> dict[str, Any]:
    validation = {}
    test = {}
    for variant in VARIANTS:
        validation_summary = _variant_summary(
            run_root,
            seed=seed,
            variant=variant,
            subset="validation",
        )
        if validation_summary is not None:
            validation[variant] = validation_summary
        test_summary = _variant_summary(
            run_root,
            seed=seed,
            variant=variant,
            subset="test",
        )
        if test_summary is not None:
            test[variant] = test_summary

    ranked = sorted(
        (
            summary
            for variant, summary in validation.items()
            if variant != "Q0_f3"
            and summary["all_validation_guardrails_pass"]
        ),
        key=lambda row: (row["primary_e2r"], row["macro_mrr"]),
        reverse=True,
    )
    improving = []
    baseline_primary = validation.get("Q0_f3", {}).get("primary_e2r")
    if baseline_primary is not None:
        improving = [
            row for row in ranked if row["primary_e2r"] > baseline_primary
        ]
    promotion_rows = (improving + [row for row in ranked if row not in improving])[:2]
    promoted = [row["variant"] for row in promotion_rows]

    return {
        "schema_version": "reactzyme_e2r_pareto_report_v1",
        "seed": seed,
        "selection_subset": "validation",
        "test_used_for_selection": False,
        "r2e_absolute_tolerance": R2E_TOLERANCE,
        "non_primary_e2r_drop_tolerance": NON_PRIMARY_E2R_DROP_TOLERANCE,
        "validation": validation,
        "test": test,
        "promoted_variants": promoted,
    }


def _markdown_table(
    summaries: dict[str, dict[str, Any]],
    *,
    effective: bool,
) -> list[str]:
    headers = [
        "Variant",
        "Time E->R",
        "Time R->E",
        "Enzyme Smi E->R",
        "Enzyme Smi R->E",
        "Reaction Smi E->R",
        "Reaction Smi R->E",
        "Macro",
        "Guard",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :-- | " + " | ".join(["--:"] * 7) + " | :-- |",
    ]
    for variant in VARIANTS:
        summary = summaries.get(variant)
        if summary is None:
            continue
        rows = summary["effective" if effective else "raw"]
        macro_mrr = sum(
            float(rows[split][R2E_KEY]) + float(rows[split][E2R_KEY])
            for split in SPLITS
        ) / 6.0
        values = [VARIANTS[variant]["label"]]
        for split in SPLITS:
            values.extend(
                [
                    f"{float(rows[split][E2R_KEY]):.4f}",
                    f"{float(rows[split][R2E_KEY]):.4f}",
                ]
            )
        values.extend(
            [
                f"{macro_mrr:.4f}",
                (
                    "pass"
                    if effective and summary["all_validation_guardrails_pass"]
                    else "fallback"
                    if effective
                    else "raw"
                ),
            ]
        )
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_outputs(run_root: Path, report: dict[str, Any]) -> None:
    out_dir = (
        run_root / "reports"
        if int(report["seed"]) == 42
        else run_root / "reports" / f"seed{report['seed']}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "promotion.json").write_text(
        json.dumps(
            {
                "seed": report["seed"],
                "selection_subset": "validation",
                "variants": report["promoted_variants"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# ReactZyme E2R Pareto campaign",
        "",
        "Model selection uses validation only. Test metrics are reported but never "
        "used by the guardrails or seed promotion.",
        "",
        "## Validation, guarded",
        "",
        *_markdown_table(report["validation"], effective=True),
        "",
        "## Paper test, raw",
        "",
        *_markdown_table(report["test"], effective=False),
        "",
        "## Promotion",
        "",
        "Promoted variants: "
        + (
            ", ".join(report["promoted_variants"])
            if report["promoted_variants"]
            else "pending"
        ),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = build_report(args.run_root.resolve(), seed=args.seed)
    write_outputs(args.run_root.resolve(), report)
    print(json.dumps(report["promoted_variants"]))


if __name__ == "__main__":
    main()
