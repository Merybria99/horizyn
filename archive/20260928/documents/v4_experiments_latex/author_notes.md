# Author notes for the experimental section

The document is an extensive editable draft with a separate supplement. It uses saved results; this writing task did not retrain models or compute new test predictions.

## Claims supported by the current evidence

V4 improves all four recorded EnzymeMap screening metrics over the released CLIPZyme checkpoint and the displayed FGW-CLIP results in both screening pools. FGW-CLIP remains a reported comparison; the released CLIPZyme checkpoint was evaluated locally under the shared released protocol. This distinction remains visible in the tables.

On ReactZyme, V4 exceeds the displayed main-method literature point estimates. The new original-Horizyn matched-data comparator exceeds V4 and V4+Bio in **all six** split/direction MRR cells. Thus neither variant currently supports a claim of winning every retrained competitor. Different budgets, representations, and objectives prevent interpreting this ordering as an isolated architecture or loss effect.

The phase-2 biological controls do not establish a consistent benchmark benefit. In Case1, biology moves one primary-paper catalyst from unique-sequence rank 26 to 25, while the 144-entry primary-paper recovery stays 10/15. Preserve both views. The EnzymeMap-trained model's weaker Case1 transfer is included in the supplement.

## Protocol details that affect interpretation

1. **Same downstream data, distinct pretrained resources.** F3 retrieval layers are freshly initialized for each benchmark target. Frozen feature extractors and auxiliary annotations are additional resources, and should not be called target-only training from scratch.
2. **EnzymeMap queries and pools.** The 4,642 test association rows produce 1,521 eligible unique reaction queries under the released notebook procedure. Full screening ranks 261,907 identifiers; excluding training identifiers leaves 252,113 candidates and 1,337 eligible queries. Identifier exclusion is not proof of sequence or homology disjointness.
3. **Screening metrics.** BEDROC85, BEDROC20, EF5, and EF10 are the endpoints. Tables express BEDROC as percentages; figures use raw units. The supplement retains the released zero-based rank calculation and the EF floor convention. MRR is not the screening endpoint.
4. **ReactZyme MRR.** All local comparisons use query-averaged all-positive reciprocal ranks. First-positive MRR and Hit@1/5/10 are separate diagnostics. Published wording does not establish equivalence to local metric code; literature rows stay separate from reproduced rows.
5. **Recipe selection.** Final epoch 10/update 100 is the recorded recipe. Repeated exploratory inspection of benchmark tests occurred during development. Do not rewrite this history as a single untouched held-out evaluation or validation-only selection of V4. Public-baseline checkpoints were selected on validation.
6. **Public baselines.** Preserve sampled-negative, SaProt fallback, corrected-contrastive-label, participant-adapter, EnzGFM preprocessing, and budget disclosures. The main table uses one fixed feature slice; the supplement retains every completed combination. Unfinished or structurally unsupported competitors have no fabricated result.
7. **Snapshot.** The paper tables are frozen at 2026-09-22 00:04:47 UTC. The 126 completed public runs comprise 95 official-family configurations, 24 corrected-contrastive controls, three Horizyn runs, and four EnzGFM controls. Live status JSONs may subsequently change. A new status is not a reason to silently replace this snapshot.
8. **Case1 denominator.** The original catalogue has 145 entries; H017 (KoT4E M6) lacks a complete recoverable sequence. The scored population is 144 entries / 123 unique sequences. The conservative primary-paper positives are 15 entries / 12 sequences. Broader workbook labels are heterogeneous conditional assay evidence, not universal inactivity labels.
9. **Case1 scope.** This is one retrospective reaction, not new wet-lab validation. Related constructs, training homology, training chemistry overlap, and unknown pretrained-backbone exposure constrain novelty claims. Candidate-IID confidence intervals would overstate independence.
10. **Ablation interpretation.** Paired temperature and fresh-F3 biology studies have their own recipes. They are not seed replications of final V4. Loss, dictionary removal, residual removal, native fusion, and learned-query architecture still lack a complete matched final-recipe comparison on both benchmarks. No missing ablation is invented.
11. **Figure 04.** Its colors are symmetric logarithmic with a small linear region; annotations remain the exact untransformed differences. Figure 01 uses linear colors. Visible color in Figure 04 does not make tiny changes large or statistically established.
12. **Near ties.** The 1e-6 score-band intervals describe numerical ordering sensitivity, not sampling uncertainty. The larger Enzyme-Sim/Time E→R changes require this qualification.

## Bibliography and integration

FGW-CLIP is explicitly pinned to arXiv:2512.08508v2 (25 May 2026); its task organization and Tables 1–3, 7–8 were checked. TIGER uses its main Table 1, not its different-data supplementary conditions. Case1 and EnzymeMap publication metadata were resolved through DOI/Crossref records and are bundled in `bibliography_metadata.json`. The full published titles of Case1 sources S02 and S16 replace their shortened local audit labels; the source IDs and DOIs are unchanged. Metadata verification does not constitute a new construct-level activity audit.

For a shorter submission, move detailed adaptation/metric discussion into the supplement while keeping the comparison labels, denominators, selection history, and materially limiting results in the main text. The main section and supplement are separate precisely to support that editing workflow.
