# ICLR 2027 experimental section

**Eight pages including references**, in the supplied ICLR 2027 single-column template. This is an experimental-section manuscript fragment, covering EnzymeMap, ReactZyme, controlled ablations, and the Case1 literature panel. It replaces the long report as the submission-oriented version.

The companion [paper-style method](../v4_methods_iclr2027/README.md) now occupies less than 2.5 pages of main text, with its detailed architecture, losses, and implementation in an appendix. Its standalone numbering places the method at Section 3 before this Section 4.

- [main.pdf](main.pdf): compiled paper-style preview.
- [experiments.tex](experiments.tex): editable section to insert after the method.
- [main.tex](main.tex): standalone ICLR wrapper, with no title page, contents, or appendix.
- `tables/`: five editable tables generated from frozen results, including the phase-1 / phase-2 / phase-2+Bio ablation.
- `figures/`: four readable, column-width scientific figures adapted from the previous ablation figures 01, 03, 04, and 08. Blue–orange SciencePlots styling is retained. Other original figures remain in the detailed evidence package.
- `references.bib`: cited literature, using the supplied ICLR bibliography style.
- `evidence_snapshot.json`, `stage_comparison.json`, `table_values.csv`, `figure_values.json`, `source_map.json`, `validation.json`: numerical provenance and checks; these are not printed in the manuscript.
- `prepare_assets.py`: rebuild tables and figures from the bundled evidence, without model execution.

The ICLR `.sty` and `.bst` files, together with their supplied `natbib.sty` and `fancyhdr.sty`, are copied unchanged from the existing `latex-src/iclr2027-tenzyme` project. The wrapper does not change the template's font size, margins, text width, or line spacing. Review line numbering remains enabled. The section starts at Section 4 in the standalone preview; the fragment itself does not reset section numbering.

## Compile

Upload the folder contents to Overleaf and select `main.tex`, or run:

```bash
latexmk -pdf main.tex
```

Alternatively, `tectonic --keep-logs main.tex` builds the same source. The compiled section is eight pages including its bibliography, below the requested ten-page limit. Pagination may change when integrated into the complete manuscript.

## Integrate into the ICLR paper

Copy the section, tables, figures, and macros into the paper project. Include `experiments.tex` after the method, adapt its relative asset paths, and merge the bibliography by citation key. The required additional packages are listed in `main.tex`. Do not copy the standalone `\setcounter{section}{3}` into the manuscript. No separate title, author block, or submission-status claim is introduced by this fragment.

The current method name remains V4, matching the companion methods package. Update the existing paper's method naming consistently if it differs. The original paper scaffold is not overwritten by this standalone package.

## Evidence and presentation

The original evidence snapshot remains frozen at 2026-09-22 00:04:47 UTC. The added `stage_comparison.json` records the completed matched stage comparison from 2026-09-22. Main tables now present V4 without added biological-label supervision; V4+Bio appears only as an ablation. This manuscript revision reuses saved results without retraining or changing any recorded metric. The stage comparison jointly adds residual refinement and dictionary scoring; it does not isolate the contribution of the refinement loss. Reported literature comparisons and local matched-data retraining are separated. The stronger original Horizyn comparator, mixed biological effects, repeated-test development history, and weaker EnzymeMap-to-Case1 transfer remain in the paper.

Case1 uses the original 145-entry catalogue, with 144 resolved entries representing 123 unique sequences. H017 has no complete recoverable sequence. Both candidate views are retained. The primary V4 result is 10/12 unique literature catalysts and 10/15 original paper entries at rank 25; 11/12 belongs only to the biological ablation.

Figures are recomposed at the ICLR column width, with 7–8 point labels, instead of shrinking the wide report exports. Figure 3 retains the shared symmetric logarithmic color scale used in original Figure 04. The temperature figure shows all 24 paired contrasts; the Case1 figure shows the two recovery views while its conditional AUROC is reported in the table and text.

The full audit and completed-run inventory remain in [the detailed evidence package](../v4_experiments_latex/README.md); they are not part of this eight-page manuscript. Rebuilding the assets requires NumPy, Matplotlib, and SciencePlots; compiling LaTeX requires no Python dependencies or experiment repository access.
