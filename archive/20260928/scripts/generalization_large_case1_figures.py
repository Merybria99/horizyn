#!/usr/bin/env python3
"""Plot every fixed method/tier in the completed Case1 pool-size sensitivity."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


STYLES = {
    "f3_native": ("F3 native", "#66717D", "x", "-"),
    "f3_fp64": ("F3 precision matched", "#222222", "+", "--"),
    "circev2": ("CIRCEv2", "#B15D9C", "s", "-."),
    "phase4": ("Phase 4", "#D55E00", "^", "-"),
    "phase2": ("Phase 2", "#0072B2", "o", "-"),
}
TIERS = (("primary_papers_12", "Paper-supported catalysts", 12),
         ("secondary_papers_and_patents_24", "Paper- and patent-supported catalysts", 24))
CUTS = (25, 1000)
BACKGROUNDS = (1000, 10000, 100000, 1044645)


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(item):
    path = Path(item["path"])
    if identity(path)["sha256"] != item["sha256"]:
        raise ValueError("Figure source changed: " + str(path))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    study = args.campaign / "large_case1_background_sensitivity"
    complete_path = study / "complete.json"
    complete = json.loads(complete_path.read_text())
    if (complete.get("schema") != "large_case1_nested_background_sensitivity_complete_v1"
        or complete.get("all_full_pool_metrics_exactly_replay_primary") is not True
        or complete.get("primary_protocol_unchanged") is not True):
        raise ValueError("Require the successful unchanged-primary sensitivity")
    for item in (complete["plan"], complete["primary_evaluation"], complete["completion_audit"]):
        checked(item)
    for item in complete["outputs"].values():
        checked(item)
    metrics_path = checked(complete["outputs"]["metrics.csv"])
    with metrics_path.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5 * 2 * 3 * 4:
        raise ValueError("Incomplete fixed method/tier/cutoff/background grid")
    keys = {(r["method"], r["tier"], int(r["cutoff"]), int(r["background_count"])) for r in rows}
    expected = {(m, t[0], k, b) for m in STYLES for t in TIERS for k in (25, 100, 1000) for b in BACKGROUNDS}
    if keys != expected:
        raise ValueError("Figure may not choose favorable methods or background sizes")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.titlepad": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), sharex=True, sharey=True)
    for row_index, (tier, title, denominator) in enumerate(TIERS):
        for col_index, cutoff in enumerate(CUTS):
            ax = axes[row_index, col_index]
            for method, (label, color, marker, linestyle) in STYLES.items():
                selected = sorted((r for r in rows if r["method"] == method and r["tier"] == tier
                                   and int(r["cutoff"]) == cutoff), key=lambda r: int(r["background_count"]))
                x = np.array([int(r["candidate_count"]) for r in selected])
                y = np.array([float(r["uniform_tie_expected_recall"]) for r in selected])
                low = np.array([int(r["guaranteed_recovered"]) / denominator for r in selected])
                high = np.array([int(r["possible_recovered"]) / denominator for r in selected])
                if (any(int(r["known_positive_count"]) != denominator for r in selected)
                    or not np.array_equal(x, np.array(BACKGROUNDS) + 123)
                    or not np.all((low <= y) & (y <= high))):
                    raise ValueError("Invalid tier denominator, candidate count or tie interval")
                ax.vlines(x, low, high, color=color, alpha=.35, linewidth=3)
                ax.plot(x, y, label=label, color=color, marker=marker, linestyle=linestyle,
                        linewidth=1.8, markersize=6, markerfacecolor="white", markeredgewidth=1.4)
            ax.plot(x, cutoff / x, color="#999999", linestyle=":", linewidth=1.4,
                    label="Uniform random ordering")
            ax.set_xscale("log")
            ax.set_ylim(-.035, 1.055)
            ax.set_xlim(850, 1400000)
            ax.set_xticks(np.array(BACKGROUNDS) + 123, ["1,123", "10,123", "100,123", "1,044,768"])
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.set_yticks(np.linspace(0, 1, 5))
            ax.grid(axis="y", alpha=.18)
            ax.set_title(f"{title} (n={denominator})\nRecovery in the top {cutoff:,}", fontsize=11)
            if col_index == 0:
                ax.set_ylabel("Fraction of known catalysts recovered")
            if row_index == 1:
                ax.set_xlabel("Total candidate sequences (log scale)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .09), ncol=3,
               frameon=False, fontsize=10, columnspacing=2)
    fig.suptitle("Case 1 recovery as the candidate pool grows", fontsize=16, y=.975)
    fig.text(.5, .025,
        "All 123 literature sequences are retained; background proteins are unassayed. Nested subsets use fixed score-blind hashes.\n"
        "Vertical ranges are conservative per-catalyst tie bounds, not confidence intervals. Coincident curves overlap; no prospective assays.",
        ha="center", va="bottom", fontsize=9, color="#444444")
    fig.subplots_adjust(left=.09, right=.98, bottom=.24, top=.86, hspace=.32, wspace=.16)
    out = args.campaign / "figures"
    out.mkdir(exist_ok=True)
    paths = []
    for extension in ("png", "pdf", "svg"):
        path = out / ("large_case1_candidate_pools." + extension)
        fig.savefig(path, dpi=180, facecolor="white")
        paths.append(path)
    plt.close(fig)
    receipt = dict(schema="large_case1_candidate_pool_figure_v1",
        sources=[identity(complete_path), identity(metrics_path), complete["plan"],
                 complete["primary_evaluation"], complete["completion_audit"]],
        implementation=identity(Path(__file__)), outputs=[identity(p) for p in paths],
        plot_contract="Both fixed evidence tiers, all five fixed methods and all four nested sizes; @25 and @1000 shown, @100 retained in source table. Full candidate pools and labels unchanged.",
        uncertainty="Uniform exact-tie expected recovery with conservative summed per-catalyst possible/guaranteed bounds; endpoints need not be jointly attainable within a shared boundary tie. No query/population confidence interval or biological specificity estimate.")
    (out / "large_case1_candidate_pools_provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"outputs": [str(p) for p in paths]}))


if __name__ == "__main__":
    main()
