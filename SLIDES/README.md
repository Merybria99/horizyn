# EnzymeDiscovery presentation

This folder contains the project presentation in two forms:

- `index.html`: self-contained presentation for a browser. It works offline and
  supports keyboard navigation and browser printing to PDF.
- `presentation.md`: editable Marp-compatible slide source.

Open `index.html` directly in a browser. The deck separates corrected,
paper-comparable ReactZyme results from historical or legacy evaluations and
places the comprehensive method inventory in the appendix.

Regenerate the HTML after editing the Markdown source:

```bash
python SLIDES/build.py
```

## Evidence sources

The presentation was assembled from the repository's retained reports and
artifacts as of 2026-08-14:

- `documents/README_reactzyme_all_ablations.md`
- `documents/README_training_protocol_top5_analysis.md`
- `documents/README_B0_configuration.md`
- `documents/README_F3_loss_improvement_proposals.md`
- `documents/README_benchmark_methods.md`
- `documents/README_data.md`
- `horizyn/docs/reactzyme_split_biology_report.md`
- `horizyn/runs/reactzyme_reaction_features_v1/eval/tiger_comparison.md`
- `horizyn/runs/reactzyme_e2r_pareto_v1/reports/summary.md`

TIGER values are Table 1 point estimates from
<https://arxiv.org/pdf/2605.24489>. F- and Q-series values are retained seed-42
test results under the official ReactZyme candidate protocol and corrected
all-positive MRR. Legacy tables are labeled explicitly and must not be used for
leaderboard claims.
