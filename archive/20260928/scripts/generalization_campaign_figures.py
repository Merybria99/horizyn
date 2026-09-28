#!/usr/bin/env python3
"""Export the completed frozen-model evidence as a scientific figure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign.resolve()
    out = root / "figures"
    contract_path = out / "chart_contract.json"
    contract = json.loads(contract_path.read_text())
    sources = {key: root / "phase4" / value for key, value in {
        "official": "official_evaluation/summary.json",
        "external": "external_evaluation/summary.json",
        "nitrilase": "nitrilase_evaluation/summary.json",
        "aminotransferase": "aminotransferase_evaluation/summary.json",
    }.items()}
    sources["paper_tables"] = root / "tiger_protocol_audit/arxiv_tables.json"
    data = {key: json.loads(path.read_text()) for key, path in sources.items()}
    table3 = next(table for table in data["paper_tables"]
                  if table["attrs"].get("id") == "S4.T3.2.1")
    mlp_reaction_e2r = float(next(row for row in table3["rows"] if row[0] == "2-layer MLP")[10])
    methods = ["F3_fp64", "phase2_seed42", "phase4_seed42"]
    labels = ["F3 (precision matched)", "Frozen phase 2", "Frozen phase 4"]
    palette = contract["palette"]
    hatches = [None, None, "///"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "text.color": "#272A2E",
                         "axes.labelcolor": "#272A2E", "axes.axisbelow": True})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.2))
    for ax in axes.flat:
        ax.grid(axis="y", color="#E8EAEC", linewidth=.7)

    def bars(ax, x, values, method_index, width=.24):
        return ax.bar(np.asarray(x) + (method_index - 1) * width, values,
                      width=width * .93, color=palette[methods[method_index]],
                      hatch=hatches[method_index], edgecolor="#41464C", linewidth=.45)

    ax = axes[0, 0]
    cells = [(s, d) for s in ("reaction_smi", "enzyme_smi", "time")
             for d in ("reaction_to_enzyme", "enzyme_to_reaction")]
    for i, method in enumerate(methods):
        vals = [data["official"]["methods"][method][s][d]["reactzyme_mrr"] for s, d in cells]
        bars(ax, np.arange(6), vals, i)
    ax.scatter(np.arange(6), [.3185, .5180, .5921, .9561, .3658, .6902],
               marker="_", s=260, color="#24272B", linewidths=2,
               label="TIGER Table 1: ESM2Text (reported)", zorder=5)
    ax.scatter([1], [mlp_reaction_e2r], marker="D", s=35, facecolors="white",
               edgecolors="#24272B", linewidths=1.2, zorder=6,
               label="TIGER Table 3: MLP, Reaction E→R only")
    ax.set_xticks(np.arange(6), ["Reaction\nR→E", "Reaction\nE→R", "Enzyme\nR→E", "Enzyme\nE→R", "Time\nR→E", "Time\nE→R"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("MRR (our scores: all-positive)")
    ax.set_title("A  Official ReactZyme test pools", loc="left", weight="bold", pad=30)
    ax.text(0, 1.025, "Three splits × two directions; full released candidate pools", transform=ax.transAxes, fontsize=9)
    ax.legend(fontsize=8, loc="upper left", frameon=False)

    ax = axes[0, 1]
    tiers = ["primary_papers", "primary_papers_and_patents"]
    for i, method in enumerate(methods):
        vals = [data["external"]["case1"][method][tier]["recovered_at_25"] for tier in tiers]
        marks = bars(ax, np.arange(2), vals, i, .25)
        ax.bar_label(marks, padding=4, fontsize=10)
    ax.set_xticks([0, 1], ["Paper-supported\n12 known catalysts", "Paper + patent\n24 known catalysts"])
    ax.set_ylim(0, 12.5)
    ax.set_yticks([0, 3, 6, 9, 12])
    ax.set_ylabel("Known catalysts recovered in top 25")
    ax.set_title("B  Literature Case 1 recovery", loc="left", weight="bold", pad=30)
    ax.text(0, 1.025, "123 unique candidate sequences; heterogeneous assay conditions", transform=ax.transAxes, fontsize=9)
    ax.text(.04, .95, "Three added paper hits have 96.8–99.1% identity\nto a supervised training enzyme.", transform=ax.transAxes, fontsize=9, va="top")

    ax = axes[1, 0]
    directions = ["reaction_to_enzyme", "enzyme_to_reaction"]
    for i, method in enumerate(methods):
        vals = [data["nitrilase"]["metrics"][method][d]["mixed_class_queries_only"]["auroc"] for d in directions]
        vals += [data["aminotransferase"]["universes"]["full_25_primary"]["metrics"][method][d]["mixed_class_queries_only"]["auroc"] for d in directions]
        bars(ax, np.arange(4), vals, i)
    ax.axhline(.5, color="#41464C", ls="--", linewidth=1)
    ax.set_xticks(np.arange(4), ["Nitrilase\nR→E, n=35", "Nitrilase\nE→R, n=10", "Aminotransferase\nR→E, n=18", "Aminotransferase\nE→R, n=25"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean within-query AUROC")
    ax.set_title("C  Complete measured assay panels", loc="left", weight="bold", pad=30)
    ax.text(0, 1.025, "684 nitrilase and 450 aminotransferase cells; mixed-class queries only", transform=ax.transAxes, fontsize=9)
    ax.text(.025, .95, "Non-detects are conditional on the source assay.\nDashed line: chance discrimination.", transform=ax.transAxes, fontsize=9, va="top")

    ax = axes[1, 1]
    for i, method in enumerate(methods):
        record = data["external"]["p450_reaction_permutation"][method]["both_absent"]["reactzyme_mrr"]
        ax.bar(i, record["correct"], .53, color=palette[method], hatch=hatches[i],
               edgecolor="#41464C", linewidth=.45)
        ax.errorbar(i, record["null_mean"],
                    yerr=[[record["null_mean"] - record["null_lower_95"]],
                          [record["null_upper_95"] - record["null_mean"]]],
                    fmt="o", color="#24272B", capsize=5, markersize=4, linewidth=1.1,
                    label="Permutation null: mean and 95% range" if i == 0 else None)
    ax.set_xticks(range(3), ["F3", "Phase 2", "Phase 4"])
    ax.set_ylim(0, .045)
    ax.set_ylabel("All-positive MRR")
    ax.set_title("D  P450 reaction-conditioning check", loc="left", weight="bold", pad=30)
    ax.text(0, 1.025, "126 queries with both specified training overlaps absent; 490 candidates", transform=ax.transAxes, fontsize=9)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.text(.025, .83, "Bars: correct-query retrieval. Null ranges are not CIs.\nUnlisted associations are not measured negatives.", transform=ax.transAxes, fontsize=9, va="top")

    handles = [Patch(facecolor=palette[m], edgecolor="#41464C", hatch=h,
                     label=label) for m, label, h in zip(methods, labels, hatches)]
    fig.suptitle("Frozen-model benchmark and catalytic evidence", fontsize=16, weight="bold", y=.994)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .969),
               ncol=3, frameon=False, fontsize=10)
    fig.text(.055, .018,
             "Campaign: 19–20 September 2026. Point estimates; paired intervals and dependence limits are in findings.md and linked readouts.\n"
             "TIGER references are published values, not controlled reruns. Its MLP ablation exceeds phase 2 in Reaction E→R; SwissProt adds annotations.\n"
             "Later phases are exploratory; no diagnostic replaces a frozen primary. Complete source tables and additional paper variants are in findings.md.",
             fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .935), h_pad=3, w_pad=2.8)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"campaign_evidence.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    provenance = {"schema": "campaign_evidence_figure_v1",
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
                  "sources": {key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                              for key, path in sources.items()},
                  "tiger_source": "https://aclanthology.org/2026.acl-long.1643.pdf"}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
