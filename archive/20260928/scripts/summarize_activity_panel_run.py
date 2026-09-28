#!/usr/bin/env python3
"""Create an audit-aware readout and activity-recovery figure from a finished run."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.activity_panels import digest,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,default=ROOT/"runs/activity_panels_nitrilase_v2")
    args=parser.parse_args(); run=args.run_root.resolve(); out=run/"report"
    summary=json.loads((out/"summary.json").read_text())
    if not summary["complete"]:raise ValueError("Finish every configured comparison before making the readout")
    settings=json.loads((run/"protocol.json").read_text())["config"]
    names=list(summary["metrics"])
    labels={name:settings["models"][name]["label"] for name in settings["models"]}
    labels["random"]="Random expectation"
    for source in settings["sources"]:labels[f"chemistry_sequence_{source}"]=f"Chemistry + sequence ({source})"
    def metrics(name,direction):return summary["metrics"][name][f"{direction}/full"]["all_queries"]
    lines=["# Experiment 1: first measured activity panel", "",
           "The completed nitrilase run compares both CIRCE v2 checkpoints separately against F3, official Horizyn-1 development, two training-only transfer controls, and exact random ranking. It contains 18 enzymes, 38 substrates and 684 measured pairs (85 active).", "",
           "| Frozen system | Reaction → enzyme mean AP | Enzyme → reaction mean AP |",
           "|---|---:|---:|"]
    for name in names:
        lines.append(f"| {labels[name]} | {metrics(name,'r2e')['ap']:.4f} | {metrics(name,'e2r')['ap']:.4f} |")
    lines += ["", "All-query means retain three substrates and eight enzymes with no active candidates. AP integrates over uniform ordering of ties; the random baseline therefore includes the finite size of the candidate pools.", "",
              "## CIRCE comparison", ""]
    for name in [n for n in settings["models"] if n.startswith("circe_v2")]:
        differences=[metrics(name,d)["ap"]-metrics("random",d)["ap"] for d in ["r2e","e2r"]]
        lines.append(f"- {labels[name]}: mean AP differs from random by {differences[0]:+.4f} for reaction → enzyme and {differences[1]:+.4f} for enzyme → reaction.")
    lines += ["", "| CIRCE checkpoint | Direction | AP difference versus official Horizyn | Exploratory 95% interval |",
              "|---|---|---:|---:|"]
    for row in summary["paired_ap_differences"]:
        if row["reference"]=="horizyn1_dev":
            direction="Reaction → enzyme" if row["direction"]=="r2e" else "Enzyme → reaction"
            lines.append(f"| {labels[row['method']]} | {direction} | {row['difference']:+.4f} | [{row['low']:+.4f}, {row['high']:+.4f}] |")
    lines += ["", "These are descriptive comparisons for one family. Different training corpora prevent attributing a difference solely to architecture. The paired query-bootstrap intervals in `paired_differences.csv` are exploratory and do not account for independent family or training-seed variation.", "",
              "## Training exposure", "",
              "| Audited source | Exact panel sequences in positive training edges | Exact exposed active pairs | Participant-set exposed active pairs |",
              "|---|---:|---:|---:|"]
    audits={}
    for source in settings["sources"]:
        path=run/"audit"/source/"complete.json"; audits[source]=digest(path)
        a=json.loads(path.read_text())
        exact=a["exact_exposed_active_pairs"]; participants=a["participant_set_exposed_active_pairs"]
        lines.append(f"| {source} | {a['exact_exposed_panel_proteins']} / 18 | {exact if exact is not None else 'Unknown (no reaction sides)'} | {participants if participants is not None else 'Not applicable'} |")
    lines += ["", "ReactZyme participant sets cannot establish exact transformation exposure. Exact-match searches may miss equivalent chemistry represented with different cofactors or charges. General training-relative homology, upstream SLEEC/backbone exposure, and official Horizyn supervised exposure are not established by these audits. Consequently, this is not a leakage-free novelty benchmark.", "",
              "## Activity recovery", "", "![Measured active hits at fixed testing budgets](activity_recovery.png)", "",
              "Each curve shows the mean number of measured actives recovered per query when testing the top K candidates. No-positive queries remain included. The reaction → enzyme budget stops at all 18 enzymes; enzyme → reaction is shown through 24 substrates.", "",
              "## Scope and reproducibility", "",
              "This first implementation covers the binary nitrilase panel. Aminotransferase/OleA and other panels, continuous-activity correlations, full homology strata, and CLIPZyme/EnzymeCAGE evaluation on this panel remain separate extensions. CYP retrospective retrieval is not included in this run.", "",
              "Reaction products were constructed with a fixed hydrolysis template. These results evaluate measured substrate activity, not experimentally established product selectivity. The secondary ever-active-enzyme diagnostic in `summary.md` is conditioned on assay labels and cannot serve as a deployable selection rule.", "",
              "The run contains complete pair scores, per-query metrics, audits and model/input checksums. The accompanying setup instructions document all stages and their provenance checks. No training or activity-label fitting was performed.", "",
              "Sources: [pinned released panel](https://github.com/samgoldman97/enzyme-datasets/tree/627556e265e2a52d39753e684c05b550d53a9be4) and [Black et al., Chemical Communications](https://doi.org/10.1039/C4CC06021K). See `summary.md` for all hit-rate and diagnostic tables.", ""]
    (out/"readout.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    colors=plt.get_cmap("tab10")(np.linspace(0,.8,len(names)))
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,direction,pool,title in zip(axes,["r2e","e2r"],[18,38],["Reaction → enzyme","Enzyme → reaction"]):
        for name,color in zip(names,colors):
            m=metrics(name,direction); points={min(k,pool):m[f"active_at_{k}"] for k in [1,3,5,8,24]}
            ax.plot(list(points),list(points.values()),label=labels[name],color="0.45" if name=="random" else color,
                    linestyle="--" if name=="random" else "-",marker="o",markersize=4,
                    linewidth=2.2 if name.startswith("circe_v2") else 1.5)
        ax.set(title=title,xlabel="Candidates tested (K)",ylabel="Mean measured active hits",ylim=(0,None))
        ax.set_xticks([1,3,5,8,min(24,pool)]); ax.grid(alpha=.2)
    handles,legend=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc="lower center",ncol=2,frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,.2,1,1))
    fig.savefig(out/"activity_recovery.png",dpi=180);fig.savefig(out/"activity_recovery.svg");plt.close(fig)
    write_json(out/"readout.provenance.json",dict(code_sha256=digest(__file__),summary_sha256=digest(out/"summary.json"),
        protocol_sha256=digest(run/"protocol.json"),audit_sha256=audits,
        outputs={name:digest(out/name) for name in ["readout.md","activity_recovery.png","activity_recovery.svg"]}))
    print(out/"readout.md")


if __name__=="__main__":main()
