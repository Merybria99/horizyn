#!/usr/bin/env python3
"""Render the frozen phase2 evidence, including failed external evaluations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign.resolve()
    phase = root / "phase2"
    sources = {name: phase / path for name, path in {
        "official": "official_evaluation/summary.json",
        "external": "external_evaluation/summary.json",
        "nitrilase": "nitrilase_evaluation/summary.json",
        "freeze": "frozen_recipe.json",
    }.items()}
    data = {name: json.loads(path.read_text()) for name, path in sources.items()}
    out = phase / "figures"
    out.mkdir(exist_ok=True)
    primary, reference = "phase2_seed42", "F3_fp64"
    colors = {reference: "#6c7a89", primary: "#007f86", "smooth_only": "#b68a36"}
    labels = {reference: "F3 (precision matched)", primary: "Frozen primary", "smooth_only": "Smooth-only diagnostic"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none",
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.6))
    ax = axes[0, 0]
    cells = [(s, d) for s in ("reaction_smi", "enzyme_smi", "time")
             for d in ("reaction_to_enzyme", "enzyme_to_reaction")]
    x = np.arange(6)
    for offset, method in [(-.18, reference), (.18, primary)]:
        vals = [data["official"]["methods"][method][s][d]["reactzyme_mrr"] for s, d in cells]
        ax.bar(x + offset, vals, width=.34, color=colors[method], label=labels[method])
    # Published values are comparisons to the paper, not a matched rerun.
    tiger = [.3185, .5180, .5921, .9561, .3658, .6902]
    ax.scatter(x, tiger, marker="_", s=190, color="#b34457", linewidths=2,
               label="TIGER ESM2Text (reported)", zorder=4)
    ax.set_xticks(x, ["Reaction\nR→E", "Reaction\nE→R", "Enzyme\nR→E", "Enzyme\nE→R", "Time\nR→E", "Time\nE→R"])
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("All-positive MRR")
    ax.set_title("A  Official ReactZyme test pools", loc="left", weight="bold")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[0, 1]
    for method in (reference, primary):
        result = data["external"]["case1"][method]["primary_papers"]
        values = [result[f"recovered_at_{k}"] for k in (5, 10, 25)]
        ax.plot([5, 10, 25], values, marker="o", lw=2, color=colors[method], label=labels[method])
        for k, value in zip((5, 10, 25), values):
            if k == 25:
                ax.annotate(f"{value}/12", (k, value), xytext=(6, 0), textcoords="offset points", va="center")
    ax.plot([0, 25], [0, 25 * 12 / 123], ls="--", color="#a5a5a5", label="Random expectation")
    ax.set_xlim(0, 29)
    ax.set_ylim(-.25, 12.5)
    ax.set_xticks([5, 10, 25])
    ax.set_yticks([0, 4, 8, 12])
    ax.set_xlabel("Candidates considered")
    ax.set_ylabel("Paper-supported catalysts recovered")
    ax.set_title("B  Case 1: 123 unique candidate sequences", loc="left", weight="bold")
    ax.text(.04, .95, "Three added catalysts are 96.8–99.1% identical\nto a supervised training enzyme.", transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(fontsize=8, loc="center left")

    methods = [reference, primary, "smooth_only"]
    ax = axes[1, 0]
    for i, method in enumerate(methods):
        result = data["external"]["p450_reaction_permutation"][method]["both_absent"]["reactzyme_mrr"]
        ax.bar(i, result["correct"], .58, color=colors[method])
        ax.errorbar(i, result["null_mean"], yerr=[[result["null_mean"] - result["null_lower_95"]],
                    [result["null_upper_95"] - result["null_mean"]]], fmt="o", color="#313131",
                    capsize=5, markersize=4, label="Reaction-permutation null (95% range)" if i == 0 else None)
    ax.set_xticks(range(3), ["F3", "Primary", "Smooth-only\n(diagnostic)"])
    ax.set_ylabel("All-positive MRR")
    ax.set_ylim(bottom=0)
    ax.set_title("C  P450: 126 queries with both overlaps absent", loc="left", weight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.text(.02, .80, "Primary does not exceed its reaction-permutation null.\nUnlisted associations are not measured negatives.", transform=ax.transAxes, fontsize=9, va="top")

    ax = axes[1, 1]
    for offset, direction, hatch in [(-.19, "reaction_to_enzyme", None), (.19, "enzyme_to_reaction", "//")]:
        vals = [data["nitrilase"]["metrics"][method][direction]["mixed_class_queries_only"]["auroc"] for method in methods]
        ax.bar(np.arange(3) + offset, vals, .36, color=[colors[m] for m in methods], hatch=hatch,
               edgecolor="white", label="R→E (35 queries)" if hatch is None else "E→R (10 queries)")
    ax.axhline(.5, color="#555555", linestyle="--", lw=1)
    ax.set_xticks(range(3), ["F3", "Primary", "Smooth-only\n(diagnostic)"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean within-query AUROC")
    ax.set_title("D  Nitrilase: complete 18 × 38 assay panel", loc="left", weight="bold")
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1, .83))
    ax.text(.03, .97, "Primary R→E declines; E→R remains below chance.\nOnly mixed-class queries enter AUROC.", transform=ax.transAxes, fontsize=9, va="top")
    fig.suptitle("Benchmark gains do not establish broad catalytic generalization", fontsize=15, weight="bold", y=.98)
    fig.text(.06, .015, "Phase2 was designed after the first frozen model failed. All diagnostics are retained; no external winner replaces the frozen primary.\nCase 1 is retrospective literature evidence. Bar heights are point estimates; uncertainty and query dependence are detailed in the reports.", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .95), h_pad=2.8, w_pad=2)
    for extension in ("svg", "png", "pdf"):
        fig.savefig(out / f"phase2_evidence.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    (out / "provenance.json").write_text(json.dumps({
        "schema": "frozen_phase2_evidence_figure_v1",
        "sources": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in sources.items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection": "Frozen primary and prespecified controls; no outcome-based replacement.",
        "tiger_source": "https://aclanthology.org/2026.acl-long.1643.pdf",
    }, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
