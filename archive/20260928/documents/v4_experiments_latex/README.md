# Editable V4 experimental section

**For conference submission, use the revised [ICLR 2027 single-column section](../v4_experiments_iclr2027/README.md): seven pages including references.** This original 25-page package is retained as the detailed evidence and audit companion.

This package accompanies [the V4 methods section](../v4_methods_latex/README.md). It follows FGW-CLIP's organization: enzyme screening, enzyme–reaction retrieval, and ablations, with datasets, baselines, metrics, and results described for each benchmark. Case1 has its own retrospective case-study section.

Start with [the compiled PDF](main.pdf) or edit [experiments.tex](experiments.tex). Upload the complete package to Overleaf and select `main.tex` as the main document.

## Contents

- `experiments.tex`: main experimental section, benchmark and Case1 tables, and all eight existing ablation figures with explanatory captions.
- `supplementary.tex`: exact recipe, baseline adaptations and budgets, metric definitions, retrieval diagnostics, all target-trained Case1 results, and every completed public-baseline result in the snapshot.
- `main.tex`, `macros.tex`: standalone wrapper and method-name/figure-path commands.
- `references.bib`: benchmark, comparator, and primary Case1 references.
- `figures/`: eight vector PDF figures, unchanged from the approved SciencePlots blue–orange exports. Figure 04 retains its shared symmetric logarithmic color scale.
- `tables/`: editable table fragments generated from saved evidence.
- `evidence_snapshot.json`: frozen results and source hashes, captured **2026-09-22 00:04:47 UTC** (2026-09-21 in Chicago).
- `prepare_tables.py`, `table_values.csv`: reproducible table builder and numerical audit ledger.
- `source_map.json`, `bibliography_metadata.json`, `validation.json`: provenance, DOI metadata, and final verification.
- `author_notes.md`: interpretation and reporting boundaries to retain while editing.

## Compile

From this directory, use either:

```bash
latexmk -pdf main.tex
```

or:

```bash
tectonic --keep-logs main.tex
```

No model weights, GPU, Python package installation, or access to the experiment repository is needed to compile the package. The standalone wrapper starts the experimental section at Section 5; remove `\setcounter{section}{4}` if that numbering is unsuitable. Landscape pages keep the eight-panel figures readable without changing their original exports.

## Integrate with the manuscript

Copy `experiments.tex`, `supplementary.tex`, `macros.tex`, `tables/`, and `figures/` into the manuscript project. Load the relevant packages from `main.tex`; notably `booktabs`, `tabularx`, `longtable`, `array`, `graphicx`, `pdflscape`, `placeins`, and `natbib` are used in addition to the mathematics packages. Load `macros.tex` and `tables/snapshot_macros.tex` in the preamble. Insert `\input{experiments}` in the main text and `\input{supplementary}` after the manuscript's existing `\appendix`.

The method name is shared through `\methodname`; the experiment macros use `\providecommand` and can coexist with the methods package. If the figure directory moves, redefine `\expFigPath` with a trailing slash. The table input paths are relative to the main document. Merge bibliography entries by citation key instead of creating duplicates. Prefer the version-specific FGW-CLIP entry supplied here for the reported results.

## Rebuild tables without new experiments

```bash
python3 prepare_tables.py
```

This uses the bundled frozen snapshot and is portable. It does not inspect live jobs or refresh results. The eight figure PDFs are already included.

Only inside the original repository, `python3 prepare_tables.py --refresh` intentionally captures newly completed receipts and the current figure exports. This also changes the evidence snapshot. Re-audit prose, completion counts, conclusions, and the provenance/validation files before distributing a refreshed manuscript. Background jobs can continue after a snapshot without changing its tables.

## Evidence scope

The snapshot contains 126 completed public-baseline runs, including all three original Horizyn fits. It separates reported literature scores, local matched-data retraining, and the released CLIPZyme checkpoint evaluation. Original Horizyn exceeds V4 and V4+Bio in all six locally evaluated ReactZyme MRR cells; that result is retained explicitly.

The requested Case1 catalogue has **145 entries**, of which **144 have resolved sequences**, representing **123 unique sequences**. H017 has an incomplete sequence definition and was not scored. The case study reports both entry-level and sequence-level recovery, distinguishes source-backed evidence from broader workbook labels, and describes retrospective prioritization rather than prospective wet-lab validation.
