# Editable V4 methods section

**For the paper, use the [new ICLR 2027 methods package](../v4_methods_iclr2027/README.md): a main section within 2.5 pages with the detailed formulation moved to the appendix.** This folder retains the earlier extended explanation and its original validation record.

This package documents the implemented **original V4** and **V4 with relative biological supervision only in phase 2 (`all_1`)**. It follows the saved 2026-09-21 recipe and the actual forward/loss implementations. It does not describe fresh-F3 biological retraining as the recommended configuration.

The companion [ICLR 2027 experimental section](../v4_experiments_iclr2027/README.md) covers ReactZyme, EnzymeMap screening, the small Case1 literature panel, and selected ablation figures in seven single-column pages including references. The [extended evidence package](../v4_experiments_latex/README.md) retains the full run inventory and all eight original figures.

## Files

- `main.tex`: standalone document; select this as the main file in Overleaf.
- `methods.tex`: complete methods section, with architecture and loss equations.
- `implementation_details.tex`: exact layer table, preprocessing, efficient loss derivation, annotation provenance, hyperparameters, and execution order.
- `architecture.tex`: editable TikZ diagram, included by the methods section.
- `macros.tex`: mathematical commands and the editable method name.
- `references.bib`: resource citations compatible with the keys in the existing manuscript.
- `source_map.json`: source files, symbols, hashes, and frozen experimental recipe used for this draft.
- `author_notes.md`: reporting boundaries and implementation points to retain while editing.
- `main.pdf`: compiled preview of the complete draft.
- `validation.json`: compilation, reference, checkpoint-specification, and numerical equation checks.

## Compile or edit

Upload the directory contents to Overleaf and compile `main.tex`. Alternatively, use either:

```bash
latexmk -pdf main.tex
```

or:

```bash
tectonic --keep-logs main.tex
```

For integration into an existing manuscript, copy the fragments into the manuscript project and adapt their `\input{...}` paths. Include the packages used by `main.tex`, load `macros.tex` in the preamble, insert `methods.tex` in the main text, and insert `implementation_details.tex` after the manuscript's existing `\appendix`. Merge the bibliography entries instead of duplicating citation keys. The TikZ figure needs the `arrows.meta`, `positioning`, and `calc` libraries.

The draft is intentionally extensive. The main method is separated from implementation details so the latter can become supplementary material. The architecture is represented explicitly; no experiments are rerun or model code changed by this documentation task.

## What is distinguished explicitly

- The 512-dimensional neural representation and the **512 + number of training reactions** final representation.
- The phase-1 decoupled multi-positive objective and the different phase-2 uniform-positive cross-entropy.
- Raw attention, smoothed pooling attention, and the two active attention regularizers.
- Family coefficient, category weight, membership confidence, and the fixed biological averaging divisor.
- The native training computation and the factor-three protein fusion / factor-one-half residual adjustments used at inference.
- Chemistry features already present in V4 versus newly added biological loss terms.
- Target-specific association training versus the frozen pretrained and auxiliary annotation resources.

The bibliography retains the local manuscript's publication metadata where appropriate and adds the official ReactionT5v2 repository so the v2 resource is not identified solely by the original ReactionT5 paper. The SLEEC publisher page returned HTTP 403 during this documentation check; its entry is retained from the repository bibliography, with its DOI, rather than treated as newly verified publisher metadata.
