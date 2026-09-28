# ICLR 2027 V4 method: main text within 2.5 pages and appendix

The expanded main method occupies **two full pages and less than half of a third** in the supplied ICLR 2027 single-column template, with its font size, margins, and spacing unchanged. This is the paper-style replacement for the earlier extended methods document.

**V4 without added biological-label supervision is the sole proposed method.** Phase 1 learns the encoders with anchor-balanced multi-positive alignment and the existing attention regularizers. Phase 2 optimizes full-graph association cross-entropy plus the identity penalty, `L_CE^(2) + 10 L_id`. The EC, cofactor-category, and bond-change auxiliary losses are excluded because their recorded benefits are mixed. SLEEC residue features and reaction chemistry features remain part of the evaluated architecture; these are distinct from the removed auxiliary supervision.

Main result tables and checkpoint provenance use the existing no-biology V4 checkpoints. V4+Bio is retained only as an ablation, with its original measurements and historical provenance. No checkpoint or saved evaluation has been modified or retrained. The primary Case1 result is 10/12 unique literature catalysts at rank 25 (10/15 original paper entries); the biological ablation's 11/12 is not attributed to the proposed method.

The prose now follows the author's supplied writing sample: formal definitions before architectural details, connected explanatory paragraphs, and explicit transitions and section references. The problem formulation distinguishes the overall association graph, the training graph, and the two query-dependent candidate pools. Its normalized embeddings denote the complete neural-plus-dictionary representation used by the final score.

The reaction subsection now explains the fixed chemistry descriptor: its
617 coordinates, chemical motivation, training-only preprocessing, and learned
projection into modality attention. Appendix D gives its exact construction
and clarifies that only the 64 aggregated molecular-property coordinates are
standardized; the descriptor summarizes participants irrespective of side.
Its numbered property table lists all 16 RDKit descriptors in implementation
order, with definitions and the sum/mean/maximum/population-standard-deviation
aggregation convention that gives 64 coordinates.

The reaction encoder also distinguishes molecule attention within each side
from attention over the four feature sources. It defines the signed and absolute
contrasts and explains why ReactZyme can produce different pooled vectors from
identical participant sets without encoding physical reaction direction.

The enzyme subsection defines the learned-view count $K$ and retains $K=4$
as the common default. Its query/key/value description distinguishes shared
trainable queries, normalized residue keys for attention scoring, and residue
values for feature aggregation. The attention equation is explained in terms
of cosine scores, a learned concentration scale, 95% attention-weighted plus
5% uniform pooling, and caching after the encoder is frozen.
The main text and appendix now share one notation: bold vectors, scalar
attention weights, raw features `x_e^v`, and projected features `y_e^v`.
The author-selected labels are `P` for the global ProtT5 mean, `S` for the
SLEEC-weighted mean, and `k=1,...,K` for learned features. Final retrieval
vectors use `z` throughout; phase-specific scalar logits use `s^(1), s^(2)`.
The SLEEC paragraph explicitly defines `x_e^S` as a 1024-dimensional
whole-enzyme weighted mean. The global feature `x_e^P` is defined before it.
The expanded notation table records dimensions and distinguishes participant
molecules from available modalities.
The fusion paragraph follows the author's opening, “We combine the extracted
enzyme features ... by projecting,” using formal definitions of the projected
features, gate weights, and final embedding. The paper calls this
**feature-wise gated fusion** and explicitly defines it
as an input-dependent weighted average with separate normalized view weights
for each feature coordinate. Fusion follows the computation in order: projection to a common
size, a coordinate-wise weighted average, and combination with the global
sequence branch. The main text defines the two output vectors before using
them; the appendix distinguishes per-view gate weights from the shared scalar
that controls the whole fused contribution.
Appendix J motivates the capacity choice and reports
the completed $K=1,2,4,8$ ablation: 16 fresh seed-42 fits, all 56 benchmark
metric cells, and the exact multiview-module parameter counts. It explains
the metric trade-offs and the single-seed, non-parameter-matched scope.

- [method_only.pdf](method_only.pdf): main-text preview, occupying about 2.27 pages of content across three PDF pages.
- [methods.tex](methods.tex): editable main-paper section.
- [enzyme_representation.tex](enzyme_representation.tex): enzyme subsection with the motivation and interpretation of learned residue views.
- [training_and_scoring.tex](training_and_scoring.tex): the main alignment loss, refinement objective without biological labels, and final scoring.
- [biological_supervision.tex](biological_supervision.tex): unused historical ablation fragment; the complete ablation specification is in the appendix.
- [main.pdf](main.pdf): complete 21-page preview: three method pages (the third less than half full), one reference page, then 17 appendix pages.
- [appendix.tex](appendix.tex): appendix entry point, including the detailed formulation and implementation sections.
- [appendix_specification.tex](appendix_specification.tex): architecture diagram, exact encoder computations, both training objectives, biological-loss definition, dictionary, and inference adjustments.
- [appendix_implementation.tex](appendix_implementation.tex): layer specifications, preprocessing, efficient biological-loss derivation, annotation provenance, hyperparameters, and reproducible inference procedure.
- [residue_view_ablation.tex](residue_view_ablation.tex): Appendix J, including the matched K protocol, full ReactZyme/EnzymeMap tables, and interpretation of K=4.
- [k_ablation_evidence.json](k_ablation_evidence.json): exact source metrics, source hashes, and parameter counts behind the ablation tables.
- [main.tex](main.tex), [preamble.tex](preamble.tex), [macros.tex](macros.tex), and [references.bib](references.bib): standalone wrapper, packages, editable method name, and citations.
- `source_map.json`, `biological_ablation_source_map.json`, `validation.json`, `prior_presentation_validation.json`, `prior_formula_validation.json`, and `author_notes.md`: implementation provenance and documentation checks, kept outside the manuscript.

The main-text preview retains the full document's resolved appendix references and citation text; use `main.pdf` to follow references into the appendix or bibliography. No title page, contents page, or operational progress report is printed. The method is numbered Section 3 in the standalone wrapper, immediately before the companion [experimental section](../v4_experiments_iclr2027/README.md), which starts at Section 4.

## Main-text organization

1. Problem formulation: many-to-many associations, compatibility, candidate pools, and bidirectional retrieval.
2. Enzyme representation: explicit SLEEC pooling, learned residue attention, and feature-wise fusion.
3. Reaction encoder: molecular sides, pretrained views, and chemistry features.
4. Anchor-balanced multi-positive alignment: the decoupled minibatch loss and its averaging convention.
5. Full-graph refinement: uniform-positive cross-entropy and geometry preservation.
6. Bidirectional scoring: training-reaction coordinates, inference adjustments, and the complete shared representation.

The displayed equations retain the core model and objectives. Full layer widths, attention regularizers, biological confidence weighting, numerical details, and optimization settings are in the appendix. The sole method is V4; added biological supervision is excluded from both stages. The manuscript preserves the participant-only ReactZyme input semantics and the distinction between frozen pretrained resources and newly fitted retrieval weights.

The query and fusion equations now expose K explicitly; substituting K=4
recovers the previous equations. The K study uses the same association-only objective, with biological supervision disabled. Its freshly retrained K=4 row does not
replace the historical primary benchmark checkpoints or their reported values.

## Compile and integrate

Upload this folder to Overleaf and select `main.tex`, or run:

```bash
latexmk -pdf main.tex
```

Alternatively, use `tectonic --keep-logs main.tex`. The supplied ICLR `.sty` and `.bst` files and their `natbib.sty` and `fancyhdr.sty` dependencies are unchanged copies of the local conference template.

For the complete manuscript, load the packages and macros from `preamble.tex` once, insert `methods.tex` before the experimental section, and insert `appendix.tex` after the paper's existing `\appendix`. Adapt the relative input paths or copy the fragment files to the manuscript root. Merge bibliography entries by key. Do not copy the standalone `\setcounter{section}{2}`, `\clearpage`, or a second `\appendix` into the main text. Both fragments use compatible method-name macros. The method's labels use a separate prefix from the experimental section.

The enzyme fragment uses the author's `\myparagraph{}` command, with a standard paragraph fallback provided in the wrapper. Copy `enzyme_representation.tex` and `training_and_scoring.tex` with `methods.tex` when integrating the source. The historical biology fragment is not included in the main method.

The main section uses the author's `\method{}` command, provided as an alias for `\methodname` in the standalone package. Set `\methodname` in `macros.tex` to rename the method throughout. If the destination manuscript already defines `\method`, the alias leaves that definition intact; keep the appendix's `\methodname` consistent. The formulation uses `f_R,f_E` for complete encoder functions and `f_e` for the fused protein branch, avoiding a function/vector notation collision.

The 2.5-page content limit is verified in this template with all references resolved; final pagination can change with surrounding manuscript content. The previous [extended methods package](../v4_methods_latex/README.md) remains available as an audit companion. This revision selects the existing no-biology V4 recipe as the proposed method and updates the primary results accordingly. It does not rerun training or alter any experiment.
