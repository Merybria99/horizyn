# F3 / CIRCEv2 generalization investigation

> **Historical experiment log.** Entries below retain their original dates,
> recipes, and conclusions, including superseded variants and launch-time status
> reports. Current dictionary-free V4, F3, and CIRCEv2 usage is documented in
> [PIPELINES.md](docs/PIPELINES.md); comparator status and source locations are
> indexed in [BASELINES.md](docs/BASELINES.md). This log is not the current method
> specification.

**Latest benchmark update, 2026-09-21, 11:54 UTC:** 6 shared SLEEC residue-view configurations now exceed all **14 primary benchmark point estimates** on ReactZyme and EnzymeMap. Each has completed Case1 scoring on **144 entries / 123 unique sequences**, and each has its own detailed method section below. The predeclared Reaction-Sim checkpoint recovers **4/12 to 10/12** unique paper catalysts at 25, depending on the version; EnzymeMap-trained checkpoints recover **6–7/12** and have conditional AUROC near **0.47**. The reports distinguish training families from inference-only variants. All use seed 42, with exploratory benchmark comparisons; broad catalytic generalization is not established. Pending studies are not counted as winners. [All versions, exact test metrics, and Case1 results](archive/20260928/documents/shared_winning_methods_20260921.md). **User preference: V4**, recorded after the completed benchmark and Case1 results; [method details](archive/20260928/documents/shared_recipe_alpha04_cap05_v1.md).

The following opening summary describes the earlier campaign; dated updates below record subsequent experiments.

Branch: `research/f3-circev2-generalization-20260919`. Campaign began **2026-09-19 22:51 UTC**; the planned eight-hour window ended **2026-09-20 06:51 UTC**. The frozen phase 1/2/4 evaluations, fixed large-background Case 1 stress test, a complete measured esterase-panel transfer, fresh fixed-architecture EnzymeCAGE-data F3-base/phase-2 retrains, and an end-to-end all-label BCE F3 retrain/P450 evaluation are complete. The matched EnzymeMap F3 60-epoch and EnzymeMap-trained phase-2 full-pool screens are complete and below the released CLIPZyme checkpoint; the 100-epoch F3 continuation was subsequently stopped when the architecture pilots replaced it. Global calibration failed benchmark transfer; support-dependent affinity and semantic-scale controls were rejected on validation. Late representation studies tested local stereochemical fingerprints and bounded residuals with access to cached raw features. Morgan fingerprints improved the benchmark mean but failed the large Case 1 test; the raw residual was not promoted. The original large-background methods remain fixed.

**The benchmark improved, but the joint goal of beating every reported competitor and establishing broad catalytic generalization has not been achieved.** The frozen phase 2 model exceeds all 12 methods in TIGER's main Table 1 on each of the six reported MRR cells. TIGER's main Reaction-Sim E→R result is **0.518**, below our **0.523981**. The separate two-layer-MLP ablation reports **0.543**; following the user's clarification on 2026-09-21, that ablation is secondary context rather than the main-model completion threshold. Phase 2's six-cell mean is **0.674219**. The late frozen Morgan extension raises it to **0.677253**, versus **0.653639** for precision-matched F3, while reducing Reaction-Sim E→R to **0.519291**. Case 1 paper-catalyst recovery improves **5/12 → 8/12 at 25**, but the three additions are close training homologs; paper-plus-patent recovery falls **10/24 → 9/24**. In a new complete measured esterase panel, Morgan gives a small R→E AUROC gain over F3, while E→R remains below chance. In the expanded **1,044,768-candidate Case 1 pool, all five original models recover zero of the 12 paper catalysts or 24 paper-plus-patent catalysts through rank 1,000**. The final validation-selected phase 4 extension regresses on Reaction-Sim E→R and is retained as a failed extension.

The architecture remains an independent enzyme/reaction dual encoder with precomputable candidate representations. No external-panel winner was substituted for a frozen primary. Case 1 is **retrospective evidence from literature experiments**, as clarified by the user; this campaign performed no new wet-lab assays.

## What changed

1. **Corrected training/inference input consistency.** The historical ReactZyme Case 1 path supplied a physical reactant/product pair, while the F3 checkpoint was trained on complete participant sets on both sides of a self-reaction. Because its molecular set encoder reads only the reactant bank, historical inference omitted tagatose from two modalities. The pipeline now infers the policy from the actual training catalog or an explicit override. Directional checkpoints retain directional inputs, and physical display reactions remain separate from encoded inputs. Incompatible caches are preserved before replacement.
2. **Added full-graph residual training and independent semantic anchors.** Both existing F3 towers remain frozen. Small residual heads train on all recorded training associations, while a second independent representation transfers protein/reaction similarity through a fixed training-reaction dictionary.
3. **Added support-dependent residual gates and smooth reaction anchors.** Phase 2 suppresses residual corrections where F3 training support is low and removes the first model's disjoint-anchor zero-score failure. All statistics and dictionaries fit training data only.
4. **Added deployable sparse indexing and an optional wet-lab adapter.** Frozen bundles, input contracts, candidate order and source hashes are checked. Both phase 2 and phase 4 can run through the existing CLI. Scores remain retrieval affinities, not activity probabilities.
5. **Added auditable evaluation.** Stable benchmark ties, unique positive IDs, full official candidate pools, paired query intervals, reaction-group sensitivity, fixed external panels, assay non-detect handling, permutation controls and failed experiments are preserved.

The starting workspace already contained substantial changes. Work was performed on a separate branch without resetting or staging those changes. The initial source snapshot and worktree diff are under [the campaign directory](runs/generalization_20260919_2251). [Code-review coverage](archive/20260928/documents/code_review_coverage.md) states the readable inventory and review depth; inaccessible directories, environments, binary assets and vendor code are not falsely described as exhaustively reviewed line by line.

## Official ReactZyme results

Our entries below use **all-positive MRR**: average reciprocal rank over all
labeled positives within each query, then average across queries. This differs
from first-positive MRR. FGW-CLIP's Appendix B.2 describes first-positive MRR,
but its perfect-ranking table rows are below 1 on multi-positive R→E queries.
Our all-positive perfect-ranking oracle reproduces all **six** published
perfect-ranking MRR values to four decimals: Reaction-Sim **0.6715/1.0000**,
Enzyme-Sim **0.7321/0.9999**, and Time **0.7497/0.9998** (R→E/E→R).
The [metric parity audit](runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_fgw_metric_parity_v1.json)
therefore supports all-positive as the reduction underlying FGW-CLIP's reported
numbers, despite its prose definition. FGW-CLIP's code and predictions remain
unavailable, so exact scoring parity is not independently proven. TIGER entries
are its reported MRR; its exact reduction remains unverified. Our full released
candidate pools are retained.

| Split | Direction | F3 native | F3 precision matched | Phase 2 | Phase 4 | Late Morgan | TIGER ESM2Text reported | FGW-CLIP best reported |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Reaction-Sim | R→E | 0.396318 | 0.396318 | **0.415174** | 0.407318 | 0.414891 | 0.3185 | 0.3181 |
| Reaction-Sim | E→R | 0.490416 | 0.490416 | **0.523981** | 0.489450 | 0.519291 | 0.5180 | 0.3804 |
| Enzyme-Sim | R→E | 0.682638 | 0.682638 | 0.697468 | 0.697938 | **0.700621** | 0.5921 | 0.5300 |
| Enzyme-Sim | E→R | 0.970555 | **0.975459** | 0.973676 | 0.973050 | 0.974404 | 0.9561 | 0.8683 |
| Time | R→E | 0.568773 | 0.568773 | 0.599849 | 0.606054 | **0.610047** | 0.3658 | 0.3394 |
| Time | E→R | 0.809533 | 0.808228 | 0.835167 | 0.842712 | **0.844265** | 0.6902 | 0.5229 |
| Six-cell mean | | 0.653039 | 0.653639 | 0.674219 | 0.669420 | **0.677253** | 0.573450 | 0.493183 |

Phase 2 exceeds TIGER ESM2Text's reported MRR, Hit@1 and Hit@10 in every cell. It also exceeds all 12 main-Table-1 methods on all six MRR and Hit@1 cells, including ProtT3's stronger Time and Reaction-Sim R→E values. **This does not cover every variant in the paper.** Table 3 replaces TIGER's projector with a two-layer MLP on the same task and reports Reaction-Sim E→R **MRR 0.543, Hit@1 0.446**. Table 2 adds human-reviewed SwissProt descriptions and reports **MRR 0.632, Hit@1 0.543** in that cell. Both exceed phase 2 there; the SwissProt variant also uses additional annotation input.

These are **literature point-estimate comparisons**, not controlled competitor retraining or paired significance tests. The FGW-CLIP column takes the larger of its reported EC Mode and EC Max MRR in each cell; all six are below phase 2. Exact TIGER/FGW metric implementations and candidate assets were not independently reproduced, and TIGER's main/appendix baseline tables have inconsistencies. “Beats every competitor” is not supported. See [the TIGER paper, Tables 1–3](https://aclanthology.org/2026.acl-long.1643.pdf), [FGW-CLIP v2, Tables 3 and 7–8](https://arxiv.org/pdf/2512.08508), [transcribed FGW values](runs/generalization_20260919_2251/cross_paper_retraining/fgw_reactzyme_paper_values.json), [phase 2 complete results](runs/generalization_20260919_2251/phase2/official_evaluation/report.md), and [phase 4 complete results](runs/generalization_20260919_2251/phase4/official_evaluation/report.md). The additional paper variants were checked after phase 4 model selection closed. A later, separately recorded calibration follow-up uses training/validation selection and is explicitly post-evaluation exploration.

A separate **local top-10 hubness correction** was selected using only
Reaction-Sim F3 validation: subtract half of a candidate enzyme's average
similarity to its ten nearest training reactions for R→E, and the analogous
training-enzyme neighborhood score for E→R. Its fixed setting improved the
Enzyme-Sim and Time validation means without retuning. On the six official
ReactZyme F3 test cells, mean all-positive MRR moved **0.653015→0.656609**;
Reaction-Sim E→R fell **0.490427→0.488654**, while the other five cells
improved. This small late exploratory gain remains below phase 2's **0.674219**
mean and is not evidence of catalytic generalization. The [validation
selection](runs/generalization_20260919_2251/reactzyme_local_hubness_validation_v1/selection.json),
[fixed Enzyme-Sim check](runs/generalization_20260919_2251/reactzyme_local_hubness_enzyme_smi_v1/selection.json),
[fixed Time check](runs/generalization_20260919_2251/reactzyme_local_hubness_time_v1/selection.json),
and [official test evaluation](runs/generalization_20260919_2251/reactzyme_local_hubness_official_test_v1/summary.json)
retain the separate baseline/corrected results. The official test had been
consulted in earlier campaign phases, so this is not an independent confirmation.
No new RefSeq experiment was run; the user explicitly excluded it.

The underlying ReactZyme release was independently verified: all three local
original archives match official Zenodo checksums, and every released
participant-string/full-sequence test pair matches the campaign, with **zero
missing or extra pairs**. Three duplicate Reaction-Sim rows reduce to 14,689
distinct edges, as in the released binary label matrix. The upstream metric
function also agrees with the campaign on a synthetic multi-positive check.
Candidate ordering and exact-tie handling differ explicitly from upstream's
set/random behavior. [Released-data and comparator audit](runs/generalization_20260919_2251/tiger_protocol_audit/README.md).

An input-only endpoint audit finds exact canonical complete-participant exposure
in **33/386 Reaction-Sim**, **1,564/1,573 Enzyme-Sim**, and **2,540/2,634 Time**
test reactions. Exact full-sequence exposure is only **10/14,688**, **3/8,734**,
and **32/12,277**, respectively. No exact sequence-plus-canonical-participant
training/test edge overlaps were found. Canonicalization preserves component
multiplicity, stereo, charge and isotope; it removes only atom-map labels and
serialization differences. Thus high Enzyme-Sim/Time performance mainly tests
new enzymes for familiar reaction endpoints, not new reaction chemistry.
[Input-only exposure audit](runs/generalization_20260919_2251/official_exposure_audit/README.md).

This does not explain away phase 2's Reaction-Sim gain. With unchanged full
candidate pools, the **353 canonical-training-absent reaction queries** improve
R→E MRR **0.374554→0.395819**. Their **13,937 enzyme queries** improve E→R MRR
**0.463991→0.499450**. The 33 seen reactions slightly decline. This post-hoc
stratification supports transfer within the benchmark distribution, while exact
absence alone does not establish chemical distance or assay specificity.
[All exposure strata and fixed-pool definitions](runs/generalization_20260919_2251/official_exposure_strata/README.md).

Phase 2 improves all six native-F3 cells and five of six precision-matched cells. The precision distinction matters on saturated Enzyme-Sim E→R, where tiny normalization changes alter near-tied ranks. Its Reaction-Sim E→R improvement over precision-matched F3 is +0.033565: query-bootstrap 95% interval **[0.030702, 0.036491]**, but reaction-group sensitivity **[−0.001172, 0.079323]**. The latter crosses zero. The small Enzyme-Sim E→R decline has intervals crossing zero. All three R→E query-bootstrap intervals are positive. Seeds 17 and 73 give six-cell means 0.674079 and 0.673384; these vary the added heads, not the frozen F3 backbones.

Phase 4 improves validation yet loses **0.034531** Reaction-Sim E→R MRR relative to phase 2, with reaction-group interval **[−0.078074, −0.002327]**. It misses TIGER's reported Reaction-Sim E→R MRR, Hit@1 and Hit@10. Time improves, but the overall gain does not transfer. All results remain under their originally frozen identities; phase 4 was not retuned after this outcome. [Interpretation and dependence limits](runs/generalization_20260919_2251/phase4/official_interpretation.md).

The late Morgan model has the strongest completed six-cell benchmark mean,
**0.677253**, with seed-17/73 means **0.676928/0.676886**. It improves four
phase 2 cells, but its Reaction-Sim E→R decline is **−0.004690**, query-bootstrap
95% interval **[−0.006389, −0.002972]**. It still misses TIGER's 0.543 same-task
MLP variant. The recipe was selected from five fixed validation configurations
and frozen before these scores; earlier campaign outcomes were already exposed.
[Complete Morgan official evaluation](runs/generalization_20260919_2251/post_evaluation_morgan/transfer/official_evaluation/report.md).

## Literature Case 1

The workbook's `Ranked Candidates` is a dietary-priority projection, not the activity ground truth. The audit resolves **123 unique sequences**, including **12 primary-paper-supported catalysts** and **12 additional patent-supported catalysts**. Broad workbook labels contain 81 sequences with some reported activity and 42 labeled non-detection only. Seven sequences have positive evidence in one assay and non-detection in another; they are excluded from the negative class. Conditions differ, and some negative labels reflect approximately-zero figure bars rather than explicit source censoring. These broad labels remain a **workbook proxy**, not a uniformly verified activity screen. [Sources, sequences, conditions and evidence tiers](runs/generalization_20260919_2251/case1_audit/README.md).

| Model / input | Paper catalysts @25, of 12 | Paper + patent @25, of 24 | Workbook-proxy AUROC |
|---|---:|---:|---:|
| F3 epoch 29, historical physical input | 5 | 7 | 0.5817 |
| F3 epoch 29, training-matched input | 5 | 10 | 0.5570 |
| CIRCEv2 epoch 23, historical input | 0 | 2 | 0.4518 |
| CIRCEv2 epoch 23, training-matched input | 0 | 3 | 0.4568 |
| Frozen phase 1 | 0 | 0 | See retained failure audit |
| Frozen phase 2 | **8** | 9 | 0.5926 |
| Frozen phase 4 | **8** | 9 | 0.5902 |

Correcting the participant input is necessary for a valid deployment test, but is insufficient to solve activity ranking. The frozen F3 historical/corrected query cosine is approximately 0.465, yet paper recovery at 25 does not improve. CIRCEv2 remains an independently retained native-checkpoint control; new learned adapters use F3.

A source-only follow-up found no sufficiently verified, matched-condition
Case 1 subset containing both explicit measured positives and sequence-resolved
non-detects. It retained every potentially eligible study and documented why
its negative labels or assay comparability were insufficient. No favorable
within-study subset was assembled from model scores. Accordingly, the proxy
AUROCs above do not establish measured specificity.
[Within-study eligibility and all exclusions](runs/generalization_20260919_2251/case1_within_study_audit/README.md).

Phase 2 paper recovery at **5/10/25** is **0/2/8**, versus F3's **0/1/5**. Its extra hits are TpetWT, TneWT and TneM3, with ranks **77→15, 80→17 and 68→21**. Their identities to supervised training enzyme TM_0440 are **99.1%, 97.5% and 96.8%**. This demonstrates better recovery of close homologs for this reaction, not remote-family discovery. The four paper catalysts outside the top 25 remain at ranks 29, 94, 111 and 112. All 123 phase 2 scores are distinct.

The prespecified density-only control also retrieves 8/12 paper and 9/24
paper-plus-patent catalysts at 25. The smooth-anchor-only control retrieves
0/12 and 4/24. Thus the observed Case 1 recovery gain is already present in
the gated residual branch; the smooth anchors do not explain the added paper
hits by themselves. These ablations remain diagnostics, not replacement models.

Case 1 has no exact complete stereochemical participant-set match in the supervised reaction training set, but chemically adjacent training sets and protein-family exposure remain. The five catalysts retrieved by native F3 lacked a qualifying MMseqs training hit under the declared coverage/search settings; that does not prove absence of remote homology. These repeated Case 1 evaluations are exploratory, not untouched confirmation.

## Case 1 in a million-candidate search

The completed stress test contains **1,044,645 RefSeq background sequences plus
all 123 unique literature sequences: 1,044,768 candidates**. The sample selected
256 physical residue-feature tiles without replacement, before any scores were
inspected. This is a cluster sample, not a completed full 3.94-million-protein
search. Known literature references are deliberately included; no natural
positive prevalence or full-catalog rank extrapolation follows.

| Fixed method | Best paper-catalyst rank | Paper recovered @1,000 / @10,000 | Paper + patent recovered @1,000 / @10,000 |
|---|---:|---:|---:|
| Phase 2 | 10,339 | 0/12 / 0/12 | 0/24 / 0/24 |
| Phase 4 | 10,352 | 0/12 / 0/12 | 0/24 / 0/24 |
| F3 native | 11,432 | 0/12 / 0/12 | 0/24 / 0/24 |
| F3 precision matched | 11,432 | 0/12 / 0/12 | 0/24 / 0/24 |
| CIRCEv2 epoch 23 | 8,777 | 0/12 / 2/12 | 0/24 / 5/24 |

None of these positive ranks is tied. Better curated-pool recovery therefore
does not establish practical retrieval of the known catalysts in this expanded
search. Top-ranked background proteins are **unassayed**, so these data cannot
measure precision, specificity or inactivity. They could include undiscovered
active enzymes. [Complete original primary and diagnostics](runs/generalization_20260919_2251/large_case1_interpretation/readout.md).

![Fixed Case 1 recovery across candidate-pool sizes](runs/generalization_20260919_2251/figures/large_case1_candidate_pools.png)

The nested background-size sensitivity uses fixed hash priorities, retains all
123 literature sequences at each size and changes no model. Lines show expected
recovery under uniform exact-tie ordering; vertical ranges are conservative sums
of individual tie possibilities, not confidence intervals or jointly attainable
extremes. [Vector PDF](runs/generalization_20260919_2251/figures/large_case1_candidate_pools.pdf).

Two prespecified diagnostics narrow the explanation. Every method's top 1,000
contains **zero exact positive-training-endpoint sequences**, despite 1,971 such
background matches in the full pool. None is exactly 50 amino acids or stored
at the 1,024-residue boundary. Phase 2's top-1,000 mean length is 292.9 aa, versus
340.6 aa in the background. These facts reject exact training-sequence repetition
and concentration at those length boundaries as explanations for the top lists;
they do not rule out homologous exposure or other length effects.

A separate generic training-reaction mean-query control also recovers no verified
catalysts through rank 1,000. At 10,000 it retrieves two paper catalysts for both
phase 2 and phase 4, while the actual Case 1 queries retrieve none. However, the
actual queries have higher reciprocal-rank averages across all paper catalysts.
Thus generic affinity alone is not an established explanation.
[Mean-query ablation](runs/generalization_20260919_2251/large_case1_mean_query/README.md).

The scan took **8,018.74 seconds**. A separate audit reproduced fresh native
embeddings for 96 proteins across three prespecified tiles, whole-chunk score
parity for those tiles and all 123 prior reference scores. An independent count
oracle reproduced 480 integer rank fields and 60 tier/cutoff comparisons.
[Annotated top candidates and all verified reference ranks](runs/generalization_20260919_2251/large_case1_annotated_candidates/README.md)
retain source accessions, descriptions and organisms without changing rankings;
annotations are not measured activity evidence.

The later Morgan supplement uses exactly the same million-candidate pool and
unchanged phase 2 enzyme index. It recovers **0/12 and 0/24 through rank 10,000**;
the best verified rank is **10,306**. Its small-pool recovery remains 8/12 and
9/24 at 25. The original five-method primary is preserved, with all five metric
summaries reproduced exactly in the supplemental evaluator. A separately pinned
source-closure correction fixed an evaluation guard before Morgan large-pool
outcomes were opened; it changed no score or metric calculation.
[Morgan supplemental ranks](runs/generalization_20260919_2251/post_evaluation_morgan/large_case1_evaluation/readout.md).

## External measured and literature panels

| Panel / endpoint | F3 precision matched | Phase 2 | Phase 4 | Late Morgan |
|---|---:|---:|---:|---:|
| P450 all-query R→E MRR | 0.0175 | 0.0251 | 0.0261 | 0.0563 |
| P450 strict 126-query R→E MRR | 0.0103 | 0.0130 | 0.0134 | 0.0152 |
| Nitrilase R→E MRR | 0.2354 | 0.2341 | 0.2716 | 0.2197 |
| Nitrilase R→E AUROC | **0.6783** | 0.6266 | 0.6240 | 0.6476 |
| Nitrilase E→R AUROC | 0.4160 | 0.4145 | 0.4079 | 0.4180 |
| Aminotransferase R→E MRR | **0.2575** | 0.2446 | 0.2420 | 0.2445 |
| Aminotransferase R→E AUROC | **0.7278** | 0.6847 | 0.6941 | 0.6894 |
| Aminotransferase E→R AUROC | 0.6911 | 0.6855 | 0.6969 | 0.6861 |

**P450:** 191 reaction queries, 490 candidate IDs and 487 unique sequences; unlisted pairs are not assayed inactivity. The fixed 126-query subset lacks both exact positive-enzyme sequences and training reactions containing the full query molecule pair. Phase 4's strict MRR **0.01338** is below its reaction-permutation null mean **0.02031**, with one-sided **p=0.896**. Phase 2 similarly fails this conditioning check. Better retrieval than a weak F3 baseline is insufficient here. Strong anchor-only diagnostics are retained and were not promoted after seeing outcomes.

**Nitrilase:** the complete [primary-literature panel](https://doi.org/10.1039/C4CC06021K) contains **18 enzymes × 38 reactions = 684 measured pairs**, with 85 positives and 599 assay-conditional non-detects. Three reactions and eight enzymes have no positive assay. The table’s R→E MRR averages all **38 queries**, assigning zero to the three without a positive; AUROC averages only the **35 R→E** and **10 E→R** mixed-class queries. Phase 2 R→E AUROC declines by 0.05173, with paired 95% interval **[−0.09635, −0.00591]**. Phase 4's MRR gain over F3 has interval **[−0.01646, 0.09887]**, and reaction-permutation tests fail in both directions. The panel had prior use elsewhere in this workspace; its first label opening in this campaign does not make it globally untouched. [Preregistered phase 2 evaluation](runs/generalization_20260919_2251/phase2/nitrilase_evaluation/readout.md).

**New aminotransferase panel:** [Li et al., ACS Catalysis 2020](https://doi.org/10.1021/acscatal.0c01895), [SI Table S7](https://doi.org/10.1021/acscatal.0c01895.s001), supplies **25 enzymes × 18 amino-acid reactions = 450 measured cells**. All cells are retained: 186 detectable activities and 264 observations censored below **0.01 U/mg**. These are conditional non-detects, not numerical zero rates or universal inactivity. The assay uses 2-ketoglutarate as acceptor and PLP as catalyst. The catalyst and auxiliary enzymes from a separate cascade are not inserted as consumed reaction participants.

Both models were frozen before opening this panel's outcomes. The source curator saw part of the assay table while checking the detection limit; this is disclosed, and model-selection agents did not receive its pairwise values. The phase 4 primary performs worse than F3 on R→E AUROC: **0.69414 vs 0.72784**, paired difference interval **[−0.07513, −0.00169]**. AP also declines. E→R AUROC changes **0.69110→0.69689**, with an interval crossing zero. Permutation tests show reaction-conditioned ranking in this panel (**p=0.012 R→E, 0.007 E→R**), but native F3 also has conditioning signal. This supports limited task information, not superiority of the new method.

Exposure materially limits novelty: 5/25 reference sequences exactly match reaction-split training, all 25 have substantial-coverage retrieved homologs, and 15/18 reactions overlap training participant pairs after charge normalization. Canonical-tautomer sensitivity increases representation overlap to 17/18; this is broader representation collision evidence, not proof of identical chemical assay states. One sequence has 39 unknown residues; the complete-sequence 24-enzyme sensitivity is reported alongside the primary panel. Accession references are not independently verified tagged experimental constructs. All 11 methods, five fixed candidate rectangles, rates, censoring and source discrepancies remain in [the complete aminotransferase readout](runs/generalization_20260919_2251/phase4/external_readout/aminotransferase_readout.md).

**Late Morgan transfer:** all-query P450 R→E MRR rises to **0.05630**, with a
reaction-conditioning permutation p-value **0.000999**. The strict 126-query
subset still has MRR **0.01515**, below its null mean **0.01905** (p=**0.731**).
This gain therefore does not establish the desired transfer beyond reaction and
sequence exposure. Nitrilase R→E AUROC improves over phase 2 by **0.02107**
(interval **[0.00181, 0.04965]**) but remains below F3. Aminotransferase R→E
AUROC is **0.68943 vs F3 0.72784**; its paired interval crosses zero. E→R AUROC
is **0.68610 vs 0.69110**, also inconclusive. These repeatedly used panels are
exploratory for Morgan. [All fixed Morgan external outcomes and intervals](runs/generalization_20260919_2251/post_evaluation_morgan/readout/readout.md).

**New measured esterase panel:** A separate literature screen contains **145
whole-cell esterases × 86 chemistry-qualified substrates = 12,470 measured
pairs**, with 2,565 detected rates and 9,905 assay-conditional nondetects.
All five frozen score matrices were authenticated before the outcomes were
opened. Morgan R→E AUROC is **0.62048** versus F3 **0.61364**, a paired
query-bootstrap difference of **+0.00684 [0.00230, 0.01214]**; AP is
**0.31926** versus **0.30889**. Phase 2 is slightly below F3. Morgan E→R
AUROC is **0.44892**, below the 0.5 chance reference, with no significant
enzyme-endpoint conditioning (permutation p=**0.638**). This is limited R→E
signal, not bidirectional activity generalization. Whole-cell zeros are
conditional nondetection, and product SMILES are chemically inferred; assay
conditions and expression affect rates. Input-only sensitivity finds **0/86**
exact or containing training organic participant sets after removing water;
there is still no proof of remote catalytic novelty. An independent
sklearn/count audit reproduces all 30 headline MRR/AUROC/AP values. [Source, protocol, exact metrics and
exposure](runs/generalization_20260919_2251/esterase_audit/README_results.md).

**Released-model esterase comparison:** The official Horizyn-1 development
checkpoint was subsequently scored on the identical 86 × 145 measured assay
rectangle, using its native forward reaction fingerprints and complete-residue
ProtT5 means. Its R→E/E→R AUROC is **0.72047/0.56246**, versus F3
**0.61364/0.44771**, phase 2 **0.61310/0.44809**, and Morgan
**0.62048/0.44892**. All 12,470 pairs are retained. A checksum-verified audit
of the official supervised training release finds **zero exact sequence
matches among 145 proteins** and **zero literal physical-reaction matches among
86 queries**; homologous or chemically equivalent exposure remains possible.
This makes the real-world generalization gap more concrete, while not proving
an architecture-only cause. [Full same-panel report and paired intervals](runs/generalization_20260919_2251/esterase_audit/public_horizyn1_dev/README.md).

**Cross-paper comparability:** The [standalone comparison](docs/CROSS_PAPER_COMPARISON.md)
separates same-test-panel transfer diagnostics from the still-pending
training-controlled experiments. **No completed external result controls
training supervision across our model and the released competitor.** The
frozen-checkpoint rows use different training datasets; the EnzymeCAGE P450
F3 retrain uses only unique positive associations from the official training
CSV, whereas EnzymeCAGE's training configuration also uses its labeled
negative rows. The source CSV contains 1,445,915 rows: 243,890 positive rows
(207,173 unique positive pairs) and 1,202,025 negative rows. These comparisons
cannot establish an architecture win or loss. Two earlier public-checkpoint
transfer experiments are now surfaced there:
the [measured nitrilase panel](runs/activity_panels_nitrilase_v2/report/readout.md)
against official Horizyn-1 (CIRCEv2 ReactZyme R→E/E→R AP **0.3785/0.1459**
versus **0.3163/0.2357**) and the
[CYP within-family pool](runs/cyp_baselines_v1/report/summary.md) against
official CLIPZyme, Horizyn-1, EnzymeCAGE and FusionESP (F3 MRR **0.1373**
versus **0.0999/0.2081/0.3176/0.0915** for those pretrained comparators).
CLIPZyme's native run now covers all **10,105** CYP pairs, using exact-sequence
recovered structures where the paper's files were missing
([score receipt](runs/cyp_external_v1/clipzyme_pretrained/scores.json)). The CYP pool
recovers designated catalysts among unassayed candidates, not measured
inactive pairs. None of these is a new phase-2 matched retrain.
ReactZyme-trained P450/EnzymeMap transfer is
not a direct EnzymeCAGE/FGW-CLIP comparison. The [fixed-architecture retraining
protocol](docs/CROSS_PAPER_RETRAINING.md) locks the 512-dimensional F3 dual
encoder and added residual/anchor geometry, target-specific training data,
held-out splits, candidate IDs and metrics. A fresh EnzymeCAGE-data F3 base
run on **207,173 training positives** finished on four GPUs with early stopping
after epoch 11. The run selected epoch 6 by its own validation mean
bidirectional MRR, **0.405478**; the P450 test was not used for selection.
Four stalled network residue shards were byte-verified and staged locally
without changing input values. The new epoch-6 checkpoint's **92/92 state
tensors match the earlier pilot exactly** ([state replay](runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_epoch06_state_replay.json)).
This is a deterministic reproduction and completion of that fixed F3 recipe,
not an independent seed or a new learned improvement. The 93,590 P450 raw
scores differ from the pilot by at most 3.28×10⁻⁷ numerically, while all 191
first-positive ranks are identical
([inference replay](runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_p450_inference_replay.json)).

The [completion-gated official P450 evaluation](runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_p450_evaluation/summary.json)
uses **191 physical reaction queries and exactly 490 candidate IDs**. It
agrees with an independent Top-4/14/24 rank count. On raw ranking:

| Target-trained P450 arm | Top-4 (1%) | Top-14 (3%) | Top-24 (5%) |
|---|---:|---:|---:|
| Fresh fixed F3 base | 0.02618 | 0.09948 | 0.15707 |
| EnzymeCAGE pretrained, same panel | 0.02618 | 0.09948 | **0.19895** |
| EnzymeCAGE P450-finetuned, extra P450 labels | **0.07330** | **0.13613** | 0.18848 |

The pretrained-arm Top-24 difference is **−0.04188**, with paired-query 95%
bootstrap interval **[−0.10995, 0.02618]**. The separate official
homology/reaction-prior arm gives fresh F3 **0.01571/0.06806/0.10995** versus
EnzymeCAGE pretrained **0.06806/0.12565/0.17277** at Top-4/14/24; that
prior uses extra information and is not part of raw F3. The
[paired comparison](runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_p450_comparison/summary.json)
and [completion receipt](runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_finalize_status.json)
pin the scores and official evaluator. **On this shared P450 panel, the F3 base
scores below the pretrained EnzymeCAGE checkpoint at Top-24; the training
supervision differs, so this is not a fair model-method comparison.**

**Target-trained phase 2:** The fixed residual/anchor extension was then
trained on the EnzymeCAGE positive graph, preserving the 512-dimensional
independent F3 towers and using original validation for residual checkpoint
selection. Its R→E/E→R all-positive validation MRR improves from the same
target F3 base's **0.27928/0.36332** to **0.30803/0.36907**. A label-free
physical P450 re-encoding reproduces the earlier F3 score matrix exactly.
Primary seed 42 yields official P450 Top-4/14/24 **0.01571/0.12042/0.20942**
versus pretrained EnzymeCAGE **0.02618/0.09948/0.19895**; first-positive
MRR is **0.02968** versus **0.03710**. The paired Top-24 difference
**+0.01047** has a query-bootstrap 95% interval **[−0.05236, 0.07330]**.
Residual seed 17 also reaches 0.20942 Top-24, while seed 73 falls to
0.10995. Thus this is a mixed, seed-sensitive retrospective result, not a
training-controlled win. [Full target phase-2 report](runs/generalization_20260919_2251/cross_paper_retraining/phase2_target/README.md).

**All-label loss-alignment diagnostic:** To test the supervision mismatch,
all **1,445,915** EnzymeCAGE labeled training rows and **67,677** original
validation rows were retained, including explicit negatives, duplicates and
180 conflicting sequence/reaction pairs. Features were extracted for 34
additional proteins, giving complete row coverage. The positive-trained F3
base stayed frozen while residual towers were trained with unweighted BCE,
selecting on original-validation AUROC. The unfitted F3 baseline is **0.53050**;
best BCE residuals are **0.45445/0.46410/0.44110** for seeds 42/17/73,
all selected at epoch zero. This failed adaptation was not promoted for P450
scoring. It matches labeled rows and loss family in the residual stage, but
does **not** train the full F3 base end-to-end with BCE. [All-label audit and histories](runs/generalization_20260919_2251/cross_paper_retraining/labeled_parity_v1/README.md).

**End-to-end all-label F3 on the matched EnzymeCAGE training split:** Both
512-dimensional F3 towers were trained with unweighted BCE on every original
training row; no P450 labels entered training or checkpoint selection. The
primary fresh-tower run peaked at original-validation AUROC **0.61718** after
epoch 1 (zero-indexed), versus **0.53050** for the frozen positive-trained F3
baseline. Twelve epochs ran on four H200 GPUs before validation early stopping;
the best epoch took about **29 seconds** and the peak per-GPU allocation was
**76.47 GiB**. The [training completion receipt](runs/generalization_20260919_2251/cross_paper_retraining/labeled_parity_v1/full_f3_bce_scratch_seed42_gpu_cache/complete.json)
pins the selected checkpoint. Despite that validation gain, the fixed physical
P450 panel yields official Top-4/14/24 **0.00524/0.03141/0.06806** and
first-positive MRR **0.01469**, versus pretrained EnzymeCAGE
**0.02618/0.09948/0.19895** and **0.03710**. The paired Top-24 difference is
**−0.13089**, 95% query-bootstrap interval **[−0.19895, −0.06806]**.
All **191** P450 reaction queries choose the same top protein in this F3
checkpoint's score matrix, a direct query-conditioning collapse. The
[label-free score receipt](runs/generalization_20260919_2251/cross_paper_retraining/labeled_parity_v1/full_f3_bce_scratch_p450_scores/receipt.json)
and [official evaluation](runs/generalization_20260919_2251/cross_paper_retraining/labeled_parity_v1/full_f3_bce_scratch_p450_evaluation/summary.json)
record the result. A secondary positive-trained warm start also fails P450
Top-24 at **0.06806** after all-label BCE, with validation AUROC **0.57314**
([evaluation](runs/generalization_20260919_2251/cross_paper_retraining/labeled_parity_v1/full_f3_bce_warmstart_p450_evaluation/summary.json)).
This is a same-labeled-data comparison to EnzymeCAGE but does not isolate
architecture: foundation features and architecture differ, and the P450 panel
was opened earlier in the campaign. Subsequent objective changes on this panel
must be labeled retrospective exploration.

An exploratory same-row, fresh-tower BCE plus bidirectional observed-label
retrieval-loss arm (weight **0.2**, temperature **0.1**) also failed. Its
best original-validation AUROC was **0.56048**; official P450 Top-4/14/24
fell to **0/0.01047/0.03665** and all 191 queries still chose the same top
protein ([completion receipt](runs/generalization_20260919_2251/cross_paper_retraining/labeled_parity_v1/full_f3_bce_retrieval_scratch_seed42_batch2048/complete.json),
[evaluation](runs/generalization_20260919_2251/cross_paper_retraining/labeled_parity_v1/full_f3_bce_retrieval_p450_evaluation/summary.json)).
That single loss weight did not solve the collapse and was not promoted.
The exact [official CLIPZyme EnzymeMap release](https://zenodo.org/records/15161343)
and rule split were subsequently acquired and checksum-verified. The
[release audit](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_release_audit.json)
matches **34,427/7,287/4,642** association entries, disjoint rule IDs and
**261,907** screening IDs. Its sequence map has six proteins longer than the
paper's stated 650-residue limit and 72 empty sequences; a fixed-architecture
FGW-CLIP comparison must preserve the released candidate IDs and state any
F3 feature-coverage gap. The [source-order retraining manifest](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_manifests_v2/manifest.json)
pins separate train/dev/test associations and candidate order. Despite rule
disjointness, train/test share **183 protein IDs and 30 exact reaction strings**,
but no exact UniProt-ID/reaction pair. A stricter
[sequence-level audit](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_sequence_overlap.json)
finds **two identical protein-sequence/reaction pairs** across train/test under
different IDs. At the test-row level, **552/4,642** reuse a training protein
ID, **569** reuse an exact training sequence and **98** reuse an exact reaction
string ([input-only exposure strata](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_input_exposure/summary.json)).
This matters when interpreting held-out generalization. The
existing F3 residue cache covers only **4,764/9,794**
unique training protein IDs and **173,745/261,907** screening IDs; the
[feature audit](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_existing_f3_feature_coverage_v2.json)
identified **78,221 distinct nonempty sequences** lacking existing ProtT5
features. The released associations also contain **162 train, two validation
and four test rows with empty sequences**. All **72** missing screening IDs
were subsequently restored from the UniProt archive: **50** exact 2022_01
sequences and **22** identical sequences bracketed by adjacent archived
versions ([rescue receipt](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_unisave_2022_01_rescue_v3/receipt.json)).
The now-complete [F3 catalog](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_catalog_v1/preparation.json)
preserves all **34,427/7,287/4,642** association rows and **261,907**
screening IDs, representing **222,985** unique screening sequences. After
rescue, **78,287** distinct sequences required new ProtT5 extraction
([feature receipt](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_missing_prott5_v1/receipt.json));
their four-GPU extraction is complete. ReactionT5v2 and UniMol2 cover all
**16,776** unique reactions. The ChIRo molecule cache stalled on a difficult
molecule; a read-only cache export covers **16,753/16,776** reactions and omits
63 uncached molecules. F3's configured missing-ChIRo handling retains the
other reaction modalities. The exact train/validation loader preflight passed
**34,427/7,287** association rows. Target-specific train-only chemistry
features are complete. The fresh four-GPU F3 fit completed 60 epochs; its
100-epoch continuation is running to audit the FGW-CLIP paper's stated
last-checkpoint EnzymeMap schedule. Frozen-F3 phase-2 training on the same
EnzymeMap training graph is being evaluated separately.
The [released CLIPZyme evaluation notebook](data/external/cyp_specificity_2026/clipzyme_official/analysis/Results.ipynb)
does not average over all 4,642 test association rows: it deduplicates test
reactions and skips reactions also found in training. The
[pinned evaluation protocol](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_screening_evaluation_protocol_v2/receipt.json)
therefore has **1,521** distinct train-unseen test reaction queries and
**4,544** known positive candidate-ID labels across the unchanged 261,907-ID
library. Its separate Table-2 training-enzyme exclusion leaves **1,337**
queries with a known positive. The BEDROC/EF implementation was checked
numerically against the notebook's own functions on a synthetic ranking.
The [released CLIPZyme checkpoint replay](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_released_screen_evaluation_v1/summary.json)
then reproduced its published Table-1 screening values to displayed
precision: BEDROC85/20 **0.44694/0.62982**, EF5/10 **14.085/8.060**.
After removing training enzymes, its Table-2 values were **0.39133/0.58862**
and **13.397/7.808**. This validates the released-query/candidate and metric
pipeline, but is not an F3 result. The FGW-CLIP Table-1 paper values are
**0.4866/0.6669** and **14.91/8.18**; its code/checkpoint have not been
independently rerun.

An **interim, validation-selected epoch-23 F3 checkpoint** was screened against
all **261,907** released candidate IDs for all **1,521** eligible test
reactions. The [checkpoint export and label-free score receipt](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_interim_epoch23/score_receipt.json)
pin the axes and source hashes; the [official-metric evaluation](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_interim_epoch23_evaluation/summary.json)
uses the same notebook-equivalent reductions as the CLIPZyme replay. F3 loses
to the released checkpoint on both screening arms:

| Screening arm | Model | BEDROC85 | BEDROC20 | EF5% | EF10% |
|---|---|---:|---:|---:|---:|
| Table 1, 1,521 queries | Interim F3 epoch 23 | 0.43254 | 0.57827 | 12.531 | 7.313 |
| Table 1, 1,521 queries | Released CLIPZyme | 0.44694 | 0.62982 | 14.085 | 8.060 |
| Table 2, 1,337 queries | Interim F3 epoch 23 | 0.37254 | 0.52686 | 11.569 | 6.931 |
| Table 2, 1,337 queries | Released CLIPZyme | 0.39133 | 0.58862 | 13.397 | 7.808 |

This interim checkpoint was selected before screening by validation
bidirectional MRR **0.12211**, not by test performance. Later validation checks
improved, but the [60-epoch run's validation-selected epoch-53 checkpoint](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_final_epoch53_evaluation/summary.json)
screened worse on the same official library: Table-1 BEDROC85/20
**0.42199/0.55662**, EF5/10 **11.995/6.923**; Table-2 BEDROC85/20
**0.35804/0.50197**, EF5/10 **10.940/6.514**. The selection metric is a
small-pool bidirectional MRR diagnostic and is evidently misaligned with
full-library screening. The [EnzymeMap-trained phase-2 epoch-53 evaluation](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_phase2_enzymemap_v1/test_evaluation/summary.json)
uses the same fixed residual-plus-smooth architecture and EnzymeMap-only
training associations. It raises Table-1 BEDROC85/20 to **0.43755/0.56763**
and EF5/10 to **12.170/6.980**, and Table-2 BEDROC85/20 to
**0.37099/0.51195** and EF5/10 to **11.055/6.572**. It still loses to the
released CLIPZyme checkpoint on every official screening metric. On a
separate [full-library EnzymeMap validation screen](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_phase2_enzymemap_v1/validation_evaluation/summary.json),
F3-to-phase-2 Table-1 BEDROC85 moves only **0.49380→0.49438**. The
100-epoch last-checkpoint run is under way. The interim result is
already evidence against claiming an EnzymeMap screening win; it is not an
independent wet-lab generalization result. The released CLIPZyme checkpoint
and F3 share the official association split and candidate library, while their
foundation features and pretraining differ.

The ReactZyme-motivated inference-time hubness penalty was tested separately
on this interim checkpoint. A train-reference top-50 penalty at strength 0.5
was selected using EnzymeMap validation only: R→E MRR moved
**0.10456→0.10497** and E→R **0.14024→0.14263**
([selection](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_interim_epoch23_hubness_validation/selection.json)).
Applying that fixed setting to the entire screening pool lowered Table-1
BEDROC85 **0.43254→0.42826** and Table-2 **0.37254→0.37241**; BEDROC20 and
EF5/10 also declined ([corrected evaluation](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_interim_epoch23_hubness_evaluation/summary.json)).
The raw score remains the primary result. This is a post-interim exploratory
check, not a screening-label-selected gain.

A [paired query bootstrap](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_interim_epoch23_evaluation/paired_comparison.json)
puts the interim F3-minus-CLIPZyme Table-1 BEDROC85 difference at **−0.01441**
with a 95% interval **[−0.03147, 0.00252]**. The interval crosses zero,
while BEDROC20 and both enrichment-factor differences are negative with
intervals below zero. Query resampling does not account for related reaction
families and should not be treated as a separate validation set.

[Combined external interpretation and paired phase 4/phase 2 comparisons](runs/generalization_20260919_2251/phase4/external_readout/readout.md).

![Phase 2 and phase 4 benchmark and catalytic evidence](runs/generalization_20260919_2251/figures/campaign_evidence.png)

This figure covers the phase 2 and phase 4 studies above. Its [PDF](runs/generalization_20260919_2251/figures/campaign_evidence.pdf)
and [SVG](runs/generalization_20260919_2251/figures/campaign_evidence.svg) are
standalone exports with recorded source hashes. Error ranges in the P450 panel
are permutation-null ranges, not confidence intervals.

## Why benchmark success did not ensure transfer

The evidence identifies several separate mechanisms; it does not establish one universal biological cause.

- **Input inconsistency:** historical Case 1 features did not follow the actual ReactZyme participant-set training policy. This is a verified pipeline defect; its correction alone does not improve the main paper-recovery endpoint.
- **Validation chemistry is familiar:** ordinary validation is dominated by training-seen reactions. Even among unseen reaction IDs, median nearest-training raw-feature cosine is **0.960659**, versus **0.606478** for Case 1. Four unseen IDs have byte-identical raw features to training reactions. ID novelty is a weak proxy for chemical novelty.
- **Unsupported learned corrections:** a mean identity penalty constrains average training behavior, not every unseen endpoint. The first frozen residual moves all five F3-retrieved paper catalysts far down the list. Training-density gating limits this failure, but similarity is not calibrated confidence.
- **Sparse-anchor degeneracy:** the first anchor-only Case 1 scores are all exactly zero because reaction and enzyme supports do not intersect. Apparent perfect recovery then comes entirely from catalog tie order; AUROC is 0.5. Smooth reaction anchors remove this numerical/support failure without guaranteeing useful biology.
- **A mixture coefficient does not measure ranking influence:** in phase 2, weighted neural-score standard deviation exceeds weighted semantic-score standard deviation by a median **279×** for P450 R→E, **90.8×** for Case 1, and **38.8×** for official Reaction-Sim. P450 top-25 overlap is **99.2%** with the neural branch and **9.6%** with the semantic branch. This label-free, post-hoc decomposition explains why a nominal 25% semantic coefficient can have little ranking influence; it does not show that increasing it will improve activity prediction. Small adjustments can still change near-tied top ranks. [Authenticated branch-dispersion audit](runs/generalization_20260919_2251/posthoc_branch_dominance/README.md).
- **Training association retrieval differs from activity discrimination:** labels are mostly recorded positives with unlisted pairs, while real activity depends on assay conditions and fine specificity. False-negative supervision and insufficient functional information in frozen features are plausible contributors. The experiments do not prove either as the sole cause.
- **Measured specificity is asymmetric:** in the new complete esterase matrix, the Morgan R→E AUROC is 0.6205 but its reaction-endpoint permutation null mean is already 0.5984; E→R AUROC is 0.4489 and does not beat its enzyme-endpoint null. A large part of R→E discrimination can therefore persist without the correct reaction endpoint, while the model fails to rank substrates for a given enzyme. This is descriptive, not a causal isolation of protein promiscuity, assay format, or input chemistry.

The input-only support audit finds **84/191 P450** reactions and **37/38 nitrilase** reactions below training's leave-one-out fifth-percentile support. A post-evaluation stratification with unchanged official candidate pools shows Reaction-Sim R→E gains concentrated in the high-support band: MRR **0.406379→0.435419**. Its 20 low-support queries slightly decline **0.372741→0.370949**. This pattern is not universal: 69 low-support Time queries improve **0.511218→0.547049**. [Support distribution](runs/generalization_20260919_2251/phase2/support_distribution/README.md) and [fixed-pool performance strata](runs/generalization_20260919_2251/phase2/support_performance/README.md).

## Experiments and selection history

| Stage | Change / test | Selection and result |
|---|---|---|
| Initial controls | Training-mean correction, linear CCA/ridge, full-graph residual and sparse anchors | Training/validation only; complete records retained |
| Phase 1 | 75% residual + 25% top-16 reaction anchors | Frozen 23:42 UTC; mean benchmark MRR 0.667416; Case 1 paper recovery **0/12** |
| Phase 2 | Density-gated residual; full-dictionary smooth reaction anchors; balanced seen/unseen validation | Frozen 00:36 UTC; mean benchmark MRR **0.674219**; limited Case 1 gain, external assay failures |
| Bounded raw-geometry controls | Six CCA residual configurations | Best balanced validation 0.607953 versus phase 2's 0.635186; not externally evaluated |
| Phase 3 | Six bounded positive-only alignment/moment configurations | Best balanced validation **0.596172**; rejected before any external prediction |
| Phase 4 | Nine raw/native semantic-geometry mixtures | Validation chooses raw enzyme / half-native reaction geometry; balanced MRR **0.639512**, but official mean falls to **0.669420** |
| Final stress test | Fixed 1,044,768-candidate RefSeq + Case 1 pool | All five original methods recover zero verified catalysts through top 1,000; complete |
| Post-evaluation global control | Training-only mean-score calibration of frozen phase 2 | Balanced validation 0.639560; official mean **0.672205**, below phase 2; no Case 1 rank changes; retained failure |
| Bounded affinity control | Support-dependent, mean-centered generic enzyme-affinity correction | All 15 validation configurations retained; selects the unchanged phase 2 fallback, no external predictions |
| Late semantic-scale exploration | Query scaling from a fixed training-enzyme reference bank, preserving training-mean score | All 12 nonidentity settings worse than phase 2 and fail validation guards; no external predictions |
| Late molecular-fingerprint exploration | Local stereochemical count features as a reaction-anchor view | Five settings; validation **0.643794**, official mean **0.677253**; large Case 1 fails, measured panels do not establish superiority |
| Late raw-feature exploration | Bounded residual towers read native F3 plus cached protein/reaction features | Four configurations; best balanced validation **0.636868**, below Morgan/P4; no external transfer |

On the Reaction-Sim training split, full-graph training uses **6,977 reactions, 147,299 proteins and 147,393 unique observed edges**. Both residual heads start at identity; F3 remains frozen. Phase 2's score is a weighted concatenation giving **75% density-gated F3 residual similarity + 25% smooth semantic-anchor similarity**. Enzyme anchors use 32 training protein neighbors; reaction anchors use all training reactions, with temperature 0.03. Phase 4 changes only the reaction-anchor geometry to equal raw/native cosine weight; its enzyme index is identical to phase 2. [Equations and numerical contract](archive/20260928/documents/generalization_mathematical_method.md).

The positive-only control uses known-positive cosine alignment, identity preservation and mean/covariance regularization, with pointwise residual caps 0.1/0.3 and moment weights 0.1/1/10. Five of six runs select the identity fallback. This rejects that adapter, not the general false-negative hypothesis: the underlying F3 was already contrastively trained and moment regularization adds its own assumptions. [Phase 3 report](runs/generalization_20260919_2251/phase3/positive_geometry/README.md).

Frozen recipes are timestamped and hashed: [phase 1](runs/generalization_20260919_2251/frozen_recipe.json), [phase 2](runs/generalization_20260919_2251/phase2/frozen_recipe.json), [phase 4](runs/generalization_20260919_2251/phase4/frozen_recipe.json). Later phases follow earlier held-out observations and repeated validation use, so their evidence is exploratory. The new aminotransferase results were opened only after phase 4 model selection closed and all declared predictions were authenticated. A later training-mean calibration control is explicitly post-evaluation exploration: its coefficients use training/validation only, but these already-opened external panels cannot serve as new independent confirmation for it.

## Post-evaluation calibration follow-up

A fixed training/validation screen tested whether subtracting mean training
endpoint affinity reduces generic high-scoring candidates. It selected a
reaction-candidate bias correction for E→R, implemented by two extra independent
endpoint coordinates. Balanced validation MRR improved **0.635186→0.639560**,
but the nine split/seed transfer models did not improve the benchmark primary.
Seed-42 E→R MRR is **0.519626 Reaction-Sim**, **0.975929 Enzyme-Sim**, and
**0.825182 Time**; the six-cell mean is **0.672205**, below phase 2's 0.674219.
The Time decline relative to phase 2 has reaction-group 95% interval
**[−0.018286, −0.001166]**. It still misses TIGER's stronger MLP ablation.
[Complete official follow-up](runs/generalization_20260919_2251/post_evaluation_calibration/transfer/official_evaluation/report.md).

For R→E, the selected correction is mathematically only a query constant.
All three seeds preserve the exact rankings and tie partitions on Case 1 and
both small measured panels. A few one-place positive-rank changes on the official
pools are final-FP32-rounding effects; the reported primary R→E MRRs are unchanged.
This supplies no additional Case 1 recovery result.

Nitrilase E→R AUROC improves **0.414507→0.431804** versus phase 2, but remains
below 0.5 and fails the reaction-conditioning control. Aminotransferase E→R
AUROC becomes **0.699540**, versus F3's 0.691099, with paired difference interval
**[−0.009477, 0.025183]**. These results do not justify replacing phase 2.
[All measured-panel results and invariance checks](runs/generalization_20260919_2251/post_evaluation_calibration/transfer/external_readout/readout.md).
All previous outcomes were already available during this follow-up's development;
its coefficients use training/validation only, but the repeated panels are
explicitly exploratory. The original freezes, predictions and failed evaluation
log remain intact. An evaluation-only amendment accommodates two exact, preapproved reusable
feature receipts that omit an optional freeze field. It changes no scores,
labels, metrics, candidate sets or model parameters. A provenance erratum
corrects the amendment's explanatory timing claim; the guards use authenticated
receipt, checkpoint and input hashes, not that timing claim.
[Preserved amendment and provenance correction](runs/generalization_20260919_2251/post_evaluation_calibration/transfer/evaluation_guard_provenance_erratum.json).

The final bounded control subtracts mean-centered generic enzyme affinity only
where reaction support is low. Its **15 prespecified validation configurations**
combine five coefficients with an F3-support gate, a raw-support gate, or an
ungated control. It selects the unchanged phase 2 fallback. The best nonzero
balanced value is **0.634709**, below 0.635186; both gated variants fail the
unseen-E→R safeguard even at coefficient 0.25. All fixed endpoint/batch checks
pass, and numerical tie changes from query-constant shifts are recorded.
The rejected raw-support ramp has one-FP32-ULP CPU/GPU differences; its GPU
batch checks do not imply cross-device bitwise parity.
**No external predictions were generated.** This is a rejected hypothesis,
not evidence that support-dependent calibration can never work. After this
rejection, a new branch-dispersion diagnostic motivated one separate semantic-scale
validation study. Its late development and repeated validation exposure are
explicit; no prior frozen primary is replaced.
[Final validation-only control](runs/generalization_20260919_2251/post_evaluation_centered_affinity/README.md).

The subsequent semantic-scale control uses a fixed **4,096-enzyme training
reference bank** to balance neural and semantic score variation separately for
each reaction. An extra independent coordinate preserves average affinity over
all training enzymes in exact arithmetic. It never fits the searched candidate
pool. All **12 nonidentity settings** are worse than unchanged phase 2 and fail
the validation guards. The best has balanced MRR **0.626206**, versus **0.635186**;
all endpoint/subset/singleton and sparse/dense checks pass. Eight new focused
tests and an independent NumPy/count-based review pass. No external predictions
were generated. Thus the observed branch-dispersion imbalance does not, by
itself, provide a successful correction.
[Complete semantic-scale rejection](runs/generalization_20260919_2251/post_evaluation_semantic_scale/README.md).

A separate five-setting control tests chiral Morgan count fingerprints as a
local molecular view for reaction anchors, with protein endpoints and the
learned score branch fixed. A second control trains four pointwise-bounded
residual configurations that can read the original cached raw protein/reaction
features as well as F3 vectors. Earlier learned adapters only saw compressed F3
vectors. Both studies were [authorized before the completed large-background
outcomes opened](runs/generalization_20260919_2251/late_representation_authorization.json),
with those outcomes withheld from the agents performing model selection.
They remain explicitly late explorations using the repeatedly consulted
training/validation split; they cannot turn previously examined external panels
into independent confirmation.

The Morgan selector improves both unseen-reaction validation directions: R→E
**0.449081→0.466042** and E→R **0.446683→0.463287**. The selected radius-3,
4,096-bin chiral count view replaces only reaction semantic coordinates; it
retains the phase 2 learned branch and enzyme index. Finite hashing can collide,
and RDKit's standard Morgan invariants do not guarantee isotope discrimination.
The transfer recipe was frozen before any new official/external scores.
[Validation and transfer records](runs/generalization_20260919_2251/post_evaluation_morgan/README.md).

The raw-feature study's best cap-0.3/identity-weight-5 checkpoint is step 100.
Its balanced gain is **0.001682** over exact phase 2 and **0.001674** over a
zero-weight normalization control. All 84 checkpoint decisions and 692,392
selected-state cap rows were checked. A final serialization-key guard failed
after the first two runs; a separately recorded amendment replayed their saved
checkpoints and changed no training math. All four runs completed. The lower
validation objective was the reason to withhold transfer.
[Raw-feature ablation](runs/generalization_20260919_2251/post_evaluation_raw_residual/README.md).

## Deployment, reproducibility and resource use

The optional [wet-lab adapter](wet_lab/GENERALIZATION.md) checks the bundle, native checkpoint, participant policy, chemistry schema and modality lineage. It stores a dense learned block plus sparse semantic coordinates. Example configurations are [phase 2](wet_lab/configs/phase2_generalization.yaml) and [phase 4](wet_lab/configs/phase4_generalization.yaml). Both run through the existing query CLI.

For reproducing the strongest benchmark result from this campaign, use the
**Morgan seed-42 recipe** in [its transfer freeze](runs/generalization_20260919_2251/post_evaluation_morgan/transfer/frozen_recipe.json). Its encoder retains the phase 2 enzyme index and exposes an independent reaction tower. Phase 2 remains stronger on Reaction-Sim E→R and is available through the existing optional wet-lab adapter; phase 4 is a retained experimental extension. The
measured-panel results do not justify replacing F3 universally with either
adapter, and Case 1 scores do not predict activity under a chosen temperature,
pH or assay protocol absent from the model inputs.

Prepared Case 1 inputs reproduce frozen scores and ranks exactly, including chunked sparse indexes. Actual CLI runs and same-cache reruns also pass. Freshly regenerated molecular features introduce small differences from historical pinned caches: maximum score change **0.000256** for phase 2 and **0.000111** for phase 4. Top-10 order and top-25 membership remain unchanged for both. Atom order and historical software-version effects were not separately isolated; exact replay requires the pinned features, not only checkpoint and seed. [Phase 2 real CLI check](runs/generalization_20260919_2251/phase2/wet_lab_adapter_check/cli_integration.md).

Canonical final dot products use FP64 accumulation and FP32 stored vectors/scores,
with exact score ties handled by the declared policy. Native neural towers run
in FP32. Both native-FP32-normalized and precision-matched F3 controls are retained.
All split/seed models have full/subset/singleton endpoint checks and sparse/dense
score parity. These concern the composed encoder; upstream native neural kernels
can have tiny batch-shape differences, which are recorded separately.

Four H200 NVL GPUs were used. Three full-graph jobs per card reached approximately **84–91 GiB resident memory** and full GPU utilization; useful batch/cache sizes were chosen rather than allocating unused tensors. Residue-feature scans have a different bottleneck: the RefSeq VDS represents **2.746 TB** of uncompressed features across three physical shards, while the available network link is 1 Gbit/s. A measured 9,216-protein pilot gives a read-only full-scan estimate of about 6.8 hours, before all computation. The fixed sampled stress test is therefore explicitly separate from a completed full 3.94-million-protein scan.

Runtime dependencies were installed in isolated `/tmp` overlays; the shared environment was not upgraded. [Original reproduction guide](archive/20260928/documents/generalization_reproduction.md), [phase 2 reproduction guide](archive/20260928/documents/generalization_phase2_reproduction.md), [phase 4 reproduction guide](archive/20260928/documents/generalization_phase4_reproduction.md), and [phase 4 protocol](archive/20260928/documents/generalization_phase4_protocol.md) contain commands and provenance.

## Verification and remaining limits

Focused verification passes **421 tests across 50 files**, plus **six disjoint
new centered-affinity tests**: **427 distinct passes**, with **one CUDA-only
skip**. It covers the changed configuration/loss paths, protein pooling, token
retrieval, generalization and semantic modules, wet-lab integration, external
evaluators and the receipt compatibility amendment. Commands, logs, XML,
nonoverlap checks and source identities are in [the final verification receipt](runs/generalization_20260919_2251/final_verification/centered_affinity/complete.json).
All 17 phase 2 and 30 phase 4 frozen implementation hashes remain unchanged.
Supplemental large-pool audit tests and independent numerical oracles are
recorded with those studies. This is focused coverage, not a claim that every
test in the large workspace ran.

All four documented phase 4 evaluation commands were independently rerun on CPU
in fresh output directories. **53 deterministic output files matched exactly**,
including 336 arrays across 24 official per-query archives, all bootstrap and
permutation results, assay summaries and ordinal associations.
[Reproduction verification](runs/generalization_20260919_2251/final_verification/reproduction/complete.json).

The two newly documented global-calibration evaluation commands were likewise
rerun unchanged on CPU in fresh directories. **40 deterministic files matched
exactly**, including another 336 per-query arrays and all panel/bootstrap/
permutation outputs. Execution metadata are recorded separately.
[Calibration reproduction verification](runs/generalization_20260919_2251/final_verification/calibration_reproduction/complete.json).

Independent count-based calculations pass **32 headline checks**, including all
six-cell means, full Reaction-Sim positive ranks for F3/phase 2/phase 4, and the
aminotransferase MRR, AUROC and AP in both directions. A separate **20-check**
count-based audit verifies the global calibration means, all six official MRR
cells and complete aminotransferase metrics; it also passes. [Calibration calculation audit](runs/generalization_20260919_2251/claim_validation/calibration_independent_calculations.json).
Score authentication,
stable-tie checks, source audits, exact fixed-input sparse/dense parity and real
CLI integration checks are retained. [Claim validation and calculation sources](runs/generalization_20260919_2251/claim_validation/README.md).
The completed large-background results are reported above. The final source
package will be refreshed after the remaining fixed transfers and report edits.

The available evidence supports a reproducible benchmark improvement and a narrower close-homolog recovery result. It does **not** establish a broadly superior enzyme-discovery model, calibrated activity probabilities, prospective wet-lab success, or controlled superiority over every competitor. A convincing next confirmation would require frozen evaluation on additional chemically distant, fully measured panels and prospective assays; selecting a favorable diagnostic after viewing these outcomes would not provide that evidence.

## EnzymeMap full-library validation and directional F3 follow-up (2026-09-20)

The CLIPZyme comparison now selects F3 checkpoints by **validation Table-1
BEDROC85 across the complete 261,907-ID enzyme screen**. Cheap training-time
R→E and E→R MRR on 1,937 validation candidates remain diagnostics. This
change is motivated by the 60-epoch ordinary F3 run: epoch 53 improved
small-pool bidirectional MRR relative to epoch 23, but its official full-screen
test Table-1 BEDROC85 was lower (0.42199 versus 0.43254). The selection
protocol uses only EnzymeMap train/dev associations, 2,652 dev queries with
library positives, and no test labels. Its independent epoch-53 replay gives
Table-1 BEDROC85 **0.493803** versus the previous **0.493801**; EF5/EF10 are
identical. [Method and comparison](docs/CROSS_PAPER_COMPARISON.md).
The exact epoch-99 checkpoint is lower on validation Table-1 BEDROC85,
**0.47645**, and official test Table-1 BEDROC85, **0.39937**. Epoch 53 gave
**0.42199** on that test. A 100-epoch budget without endpoint-aligned
checkpoint selection therefore worsens the observed screening result.

The directional F3 variant uses separate reactant/product molecular pooling
and their signed and absolute difference, preserving the dual-tower interface,
protein encoder, EnzymeMap positives, loss, seed and global batch. Relative to
ordinary F3 it adds about **12.58 million trainable parameters**. At matched
epoch 5 its small-pool R→E MRR is **0.08874 versus 0.09372**, and E→R is
**0.12578 versus 0.11791**. The 0.10726 versus 0.10582 bidirectional mean is
only an early diagnostic and does not establish better screening. A fresh
100-epoch directional run records fixed snapshots at epochs 19/39/59/79/99
for full-library validation selection. Compact train/dev ProtT5 storage was
verified byte-identical on sampled residues; it changes data access rather
than model inputs. The full-pool comparison will determine whether the
directional change helps R→E.

An exploratory phase-2 weight of 0.1 selected on full-library validation
Table-1 BEDROC85 (0.49511 versus 0.49438 at weight 0.25) transferred worse
to the official test (0.43062 versus 0.43755). This is a caution against
treating a roughly 0.0007 validation gain as a robust improvement. No
alpha=0.1 result is promoted as a benchmark win.

The next controlled F3 ablations should change one component at a time on
the same downstream training associations and cached backbone inputs:

1. Replace the current pair-global `FullBatchMLNCELoss` with the repository's
   anchor-balanced bidirectional multi-positive loss, keeping the F3 towers
   fixed. The present objective normalizes over the entire batch matrix;
   screening and ReactZyme retrieval rank candidates per query. A prior
   `all_known_in_batch` ReactZyme run alone was worse than F3 and used a
   different global batch, so it does not settle the loss comparison.
2. Add a small reaction–reaction geometry objective on training reactions
   only. This can use input chemistry similarity or training-only functional
   annotations; no held-out rule labels. FGW-CLIP reports a 45.94→48.08
   Table-1 BEDROC85 improvement for its reaction–reaction ablation, which is
   motivation rather than evidence of an F3 gain.
3. Test a parameter-controlled, zero-initialized signed product-minus-reactant
   residual if the present 12.58-million-parameter directional variant fails
   full screening. Its early small-pool bidirectional MRR is inconclusive.

Full-library validation BEDROC85 chooses EnzymeMap checkpoints, while official
ReactZyme validation MRR in each direction chooses ReactZyme checkpoints.
Only after those choices are locked should score averaging of independent
checkpoints or the existing train-only phase-2 adapter be tested as separate
inference arms. Monotone score scaling cannot improve retrieval ranks.

### First directional screening and matched loss signals (2026-09-20, 22:08 UTC)

The directional epoch-19 checkpoint has now completed full-library validation.
Against ordinary F3 epoch 53, Table-1 BEDROC85 improves **0.493803→0.511433**,
BEDROC20 **0.658460→0.692208**, EF5 **14.66768→15.22718**, and EF10
**8.00704→8.65281**. With training enzymes excluded (Table 2), BEDROC85
improves **0.431694→0.452205**. This supports further evaluation of the
directional representation on held-out rules; the checkpoints have different
training ages and this is validation, not evidence of a test-set or wet-lab win.
Sources: `clipzyme_f3_directional_epoch19_validation_evaluation/summary.json`
and `clipzyme_f3_validation_epoch53_evaluation/summary.json`, under the
cross-paper retraining run directory above. Later snapshots still need full
screening before checkpoint selection is locked.

Three matched fresh F3 runs are active in `clipzyme_f3_fast_loss_v1`, sharing
the original architecture, seed 42, training associations, global batch 2,048,
BF16 forward passes with FP32 contrastive reductions, and a 100-epoch budget.
At the first diagnostic validation, after three epochs (zero-based epoch 2):

| Loss / positive map | R→E MRR | E→R MRR | Mean MRR |
| --- | ---: | ---: | ---: |
| MLNCE / observed pairs | 0.062330 | 0.071208 | 0.066769 |
| MLNCE / all training-known batch positives | 0.050587 | 0.067628 | 0.059107 |
| Decoupled per-anchor InfoNCE / all training-known batch positives | 0.076324 | 0.108015 | 0.092169 |

These are training-time small-pool diagnostics from each arm's
`logs/train/protein_pooling_training/version_0/metrics.csv`, not the official
screening endpoint. The per-anchor objective starts better in both directions;
expanding positives alone does not. This is one seed at a very early epoch,
so no loss is selected yet. Fixed snapshots at epochs 19/39/59/79/99 will be
compared using full-library validation BEDROC85. Absolute training losses
across the two objective definitions are not directly comparable.

### Directional F3 with the per-anchor loss (2026-09-20)

At the user's request, the original F3 + observed-pair MLNCE fast control
(`clipzyme_f3_fast_loss_v1/mlnce_observed`) was stopped during zero-based
epoch 5. Its logs and a `stopped.json` receipt remain available. Its GPU 0
slot was reassigned to a fresh `directional_anchor_allknown` run. The other
MLNCE/all-known and original-F3/per-anchor runs, and the existing directional
MLNCE run, continue.

The combination keeps the per-anchor arm's training data, frozen features,
seed, batch, loss, optimizer, precision and validation grid. Its model config
is identical to the existing directional F3: `side_composition:
directional_delta`, with reactants/products preserved rather than normalized
to self-reactions. No previous trained F3 checkpoint is loaded. The loss is
`DecoupledAllPositiveInfoNCELoss`, beta 10, equal R→E/E→R weights, and
`all_known_in_batch` positives from training associations only. The budget is
100 epochs, batch 2,048 on one H200, using local feature storage, BF16 neural
forward passes, FP32 contrastive reductions and fused AdamW.

The combination versus the original-F3/per-anchor arm isolates the
directional representation, including its additional parameters. The older
four-GPU directional MLNCE run uses FP32 and is not an otherwise identical
loss control. Validation Table-1 BEDROC85 over 261,907 candidates selects
among epochs 19/39/59/79/99. This is a newly launched hypothesis test; additive
gains from the two components have not been established. Configuration,
comparison constraints and hashes are recorded in
`clipzyme_f3_fast_loss_v1/directional_anchor_allknown/protocol.json`.

### Directional F3 intermediate official test (2026-09-20, 22:20 UTC)

After reporting validation, the fixed directional epoch-19 checkpoint was
evaluated on the official test screen using its already exported protein
embeddings. This is **directional F3 + original MLNCE**, before phase 2;
the newly launched directional/per-anchor model is still training. The
checkpoint was fixed before this test scoring and is an intermediate report,
not the winner of the yet-incomplete validation grid.

| Test setting | Model | BEDROC85 | BEDROC20 | EF5 | EF10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Full library | Released CLIPZyme replay | 0.446942 | 0.629817 | 14.085125 | 8.060006 |
| Full library | Directional F3 epoch 19 | 0.450989 | 0.593656 | 13.014822 | 7.199572 |
| Excluding training enzymes | Released CLIPZyme replay | 0.391328 | 0.588619 | 13.396710 | 7.808135 |
| Excluding training enzymes | Directional F3 epoch 19 | 0.403222 | 0.551603 | 12.204755 | 6.863887 |

Full-library evaluation covers 1,521 unique eligible test reactions and
261,907 candidate IDs. The training-enzyme exclusion setting covers 1,337
queries with a positive and 252,113 IDs. Directional F3 has a small nominal
BEDROC85 advantage over the released CLIPZyme replay in both settings, but
loses on BEDROC20 and both EF metrics. It remains below the published
FGW-CLIP values on all four metrics in both settings. No statistical
significance or overall superiority is established. The same downstream
training associations are used, while frozen pretrained inputs differ across
architectures.

The evidence is recorded in
`clipzyme_f3_directional_epoch19_test_evaluation/summary.json`; the fixed
checkpoint and evaluation rationale are in
`clipzyme_f3_directional_epoch19_test_embeddings/intermediate_test_request.json`,
under the cross-paper retraining directory. The existing validation-only
checkpoint-selection rule remains in place. This intermediate test inspection
joins the previous exploratory tests and must be disclosed in any final
generalization claim. The [comparison document](docs/CROSS_PAPER_COMPARISON.md)
includes the new rows.

### Combined directional/per-anchor screening started (2026-09-20, 22:38 UTC)

At the matched 18-epoch diagnostic validation (zero-based epoch 17), the
directional/per-anchor combination scores R→E MRR **0.104987**, E→R MRR
**0.136283**, and mean MRR **0.120635**. Original F3 with the same per-anchor
loss scores **0.107971 / 0.140785 / 0.124378** at that epoch. The combination
is therefore not ahead on these diagnostics, and the earlier component gains
cannot be assumed additive.

The combination's fixed 20-epoch snapshot (`screen-epoch=19.ckpt`) has been
saved. Its independent protein export and validation plus intermediate test
screening were launched on all four GPUs. The run records its checkpoint
hash and scope decision before the snapshot's full-library metrics are seen
in `clipzyme_f3_fast_loss_v1/directional_anchor_allknown/screen_epoch19/request.json`.
The pipeline writes `status.json`, then separate `validation_evaluation` and
`test_evaluation` directories. It recomputes the protein encoder outputs for
this checkpoint, keeps all 261,907 candidate IDs, and uses the same published
BEDROC/EF metric implementation. These are base-model evaluations before
phase 2. The final snapshot-selection criterion remains validation BEDROC85
on the declared epoch grid; the intermediate test is exploratory and is not
used to select the checkpoint.

### Combined directional/per-anchor screening complete (2026-09-20, 22:47 UTC)

The fixed epoch-19 combination finished full-library validation and the
intermediate official test in approximately 9 minutes 21 seconds, including
four-GPU protein export. The exact protocol, candidate-ID axis, query axis,
and metric implementation match the released CLIPZyme replay.

| Full-library test model | BEDROC85 | BEDROC20 | EF5 | EF10 |
| --- | ---: | ---: | ---: | ---: |
| Released CLIPZyme replay | 0.446942 | 0.629817 | 14.085125 | 8.060006 |
| Directional F3 + MLNCE, epoch 19 | 0.450989 | 0.593656 | 13.014822 | 7.199572 |
| Directional F3 + per-anchor loss, epoch 19 | 0.468593 | 0.615931 | 13.366311 | 7.561644 |

With training enzymes excluded, the combination scores **0.414928 /
0.574010 / 12.614199 / 7.260830** (BEDROC85 / BEDROC20 / EF5 / EF10).
The combination improves all four test metrics over the earlier directional
MLNCE snapshot, and exceeds the released CLIPZyme replay on BEDROC85 in
both screening settings. It is still below released CLIPZyme on BEDROC20 and
both enrichment factors and below published FGW-CLIP on all four metrics.
No overall benchmark win or statistical significance is claimed. These
results are before phase 2; BF16/single-GPU versus FP32/four-GPU training is
an additional difference between the two directional runs.

Validation Table-1 scores are **0.502495 / 0.682210 / 15.215817 / 8.432689**;
Table-2 scores are **0.443733 / 0.644174 / 14.649542 / 8.249243**. These are
below directional MLNCE epoch 19, despite the combination's better test
scores. The differing validation/test order is explicitly retained rather
than using this test to revise the checkpoint criterion. The training budget
and fixed validation-selection grid continue unchanged.

Sources under `clipzyme_f3_fast_loss_v1/directional_anchor_allknown/screen_epoch19/`:
`validation_evaluation/summary.json`, `test_evaluation/summary.json`,
`request.json`, and `status.json`. The [comparison document](docs/CROSS_PAPER_COMPARISON.md)
contains the new test rows and comparison limitations.

### Single-model training review: sampling and positive coverage (2026-09-20)

The user excludes ensembling; no score-ensemble experiment was launched.
Proposed follow-ups should retain one model at inference and the same
downstream training associations. Hubness correction and RefSeq experiments
remain excluded.

The current combined run uses the default shuffled-pair DataLoader. The
per-anchor loss balances the distinct anchors present within a batch, but
does not equalize their inclusion frequency across batches. In its training
CSV there are 34,427 rows, 12,603 reaction IDs and 9,666 protein IDs. The
median reaction row-degree is 1, 60.03% of reactions have one row, and the
most frequent 1% of reactions account for 14.22% of training rows. Maximum
reaction/protein row-degrees are 133/187. This establishes a sampling mismatch
with uniformly averaged query metrics, not proof that balanced sampling will
improve held-out performance.

A training-only illustrative shuffle (`random.Random(42).shuffle`, consecutive
batches of 2,048 CSV rows) was used to inspect positive coverage. This is not
a replay of the actual training DataLoader RNG. Across 20,148 observations of
multi-positive reaction anchors, an average 46.56% of their training-known
protein positives were present in the candidate batch (median 42.86%). For
reactions with at least ten distinct positives, average coverage was 29.48%.
Such reactions appeared in 10.78 batches on average, compared with 1.00 for
single-positive reactions. `all_known_in_batch` recognizes the positives
present in the batch; it does not bring in the missing training positives.

This motivates separate controlled trials of balanced reaction/enzyme anchor
sampling, grouped positives including singleton anchors, and a short
single-model residual refinement stage using the full training embedding
bank. The existing grouped hypergraph sampler only groups degree>=2 anchors;
it should not be described as uniformly sampling all reactions unchanged.
Chemistry/modality dropout is another inexpensive trial: both are currently
zero, while the code already supports chemistry masking and a chemistry-view
consistency term. A multi-vector protein representation is a larger,
unvalidated architectural proposal; it would retain independent encoders but
change scoring and increase the cached index size.

For inference diagnostics, a fixed chemistry-channel mask could be checked
on validation with only reaction re-encoding. This is not an established gain
and changes the input distribution of a model trained without modality
dropout. Temperature scaling of a single score cannot change ranks. Reranking
only the top 1,000 candidates cannot change EF5/EF10 membership for a 261,907-ID
library, whose cutoffs are 13,095 and 26,190. Any refinement intended to improve
those metrics must change membership across the corresponding cutoffs.

### Single-model ablations: first completed validation results (2026-09-20, 23:29 UTC)

All five authorized ablations were implemented under
`cross_paper_retraining/clipzyme_f3_single_model_v2/`. No ensemble, hubness
correction, RefSeq run, or new test-based selection was introduced. The
directional F3 + anchor-balanced epoch-19 checkpoint is the fixed base for
the inexpensive inference and phase-2 comparisons. Its trainable towers
were initialized fresh on the official EnzymeMap training split; frozen
feature backbones and the pretrained frozen SLEEC scorer are unchanged.

The following are **validation** results, not test comparisons to published
FGW-CLIP values. Table 1 averages 2,652 eligible validation reactions against
261,907 candidate IDs. Table 2 excludes training enzyme IDs and retains 2,216
queries against 252,113 candidates.

| Validation Table-1 variant | BEDROC85 | BEDROC20 | EF5 | EF10 |
| --- | ---: | ---: | ---: | ---: |
| Fixed directional + anchor-balanced base, epoch 19 | **0.502495** | **0.682210** | **15.215817** | **8.432689** |
| Chemistry channel disabled at inference | 0.202627 | 0.325245 | 7.299273 | 4.549585 |
| Full-graph decoupled phase 2, 100 steps | 0.501529 | 0.671285 | 14.808041 | 8.193607 |
| Full-graph decoupled + Smooth-AP, 100 steps | 0.501538 | 0.671259 | 14.817081 | 8.192365 |
| Full-graph decoupled phase 2, 200 steps | 0.475838 | 0.639394 | 14.151262 | 7.817336 |
| Full-graph decoupled + Smooth-AP, 200 steps | 0.477948 | 0.641236 | 14.200702 | 7.818693 |

The chemistry mask is rejected: this checkpoint relies on chemistry, and
removing that channel causes a major performance loss. This does not test
the separate chemistry-dropout training hypothesis. Neither full-graph
phase-2 candidate beats the unmodified base on the selection metric. The
100-step BEDROC85 difference is small and is not a significance claim;
200-step performance is clearly lower descriptively. The base is retained,
and these candidates were initially held back from the test set. The user
subsequently explicitly requested their test outcomes; see the registered
test follow-up below.

Both phase-2 runs fit independent identity-initialized residual towers to
the **same complete training graph**, with 12,603 reaction nodes and 9,666
enzyme nodes. Decoupled InfoNCE excludes all known positives from each
negative denominator and averages first within, then across anchors, matching
the phase-1 anchor-balanced objective (beta 10, equally weighted directions).
The ranking arm adds weight 0.1 Smooth-AP at temperature 0.05, using 128
uniformly sampled anchors per direction while retaining all their positives
and the entire training candidate bank. Identity weight is 2. The fixed
100/200-step grid is compared using full-library validation BEDROC85;
small-pool MRR logs remain diagnostics. Nonedges remain unannotated
associations, not experimentally established negatives. Optimization itself
took 7.8 seconds for the control and 9.0 seconds for the ranking arm,
excluding feature export, startup, and full-library validation.

Three fresh F3 training arms are still running: singleton-inclusive balanced
reaction/enzyme sampling (up to four positives per sampled anchor), chemistry
dropout 0.1, and four learned enzyme prototypes. Each changes one component
relative to directional + anchor-balanced training; seed 42, batch 2,048,
100 epochs, BF16 with FP32 contrastive reductions, and the exact training
associations are shared. Diagnostic validation runs every three epochs.
Snapshots at zero-based epochs 19/39/59/79/99 are automatically exported and
evaluated on the full library. Uniform anchor draws do not imply uniform
incidental neighbor exposure, and a sampled epoch need not visit every edge.

The prototype screening path now caches its learned prototype vectors and
sequence-derived priors, then uses the trained log-sum-exp scoring rule.
It is one model with independent encoders. It is not evaluated by silently
replacing that head with raw cosine. Tests compare cached scores with direct
head predictions and preserve legacy cosine exactly. Sampler, wiring,
prototype, positive-pair, directional, residual, and ranking checks passed
(62 tests plus 10 subtests).

Evidence: [campaign protocol](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_single_model_v2/protocol.json),
[execution receipt](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_single_model_v2/execution.json),
[chemistry-mask validation](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_single_model_v2/chemistry_mask/validation_evaluation/summary.json),
[phase-2 validation selection](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_single_model_v2/full_graph_phase2/validation_selected.json),
and [live evaluation queue](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_single_model_v2/watch/status.json).

### User-requested intermediate test follow-up (2026-09-20, 23:37 UTC)

After reviewing the validation results, the user explicitly requested actual
test BEDROC/EF. The [fixed test queue](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_single_model_v2/requested_tests_v1/plan.json)
records the decisions before any of these new test outcomes: evaluate the
100-step phase-2 control and ranking arm (each selected over its own 200-step
snapshot by validation), the fixed inference chemistry mask, the directional
+ anchor-balanced epoch-39 base, and the first epoch-19 checkpoints of all
three fresh training arms. Protein exports are reused across validation and
test; model fitting still uses only target training associations. Test scores
cannot be substituted for validation in the existing final checkpoint selector.

These are requested exploratory intermediate tests. The test set has already
been inspected in this campaign, and reporting these results does not make
it a newly untouched final evaluation. The exact official Table-1 axes remain
1,521 reaction queries and 261,907 IDs; Table 2 retains 1,337 queries after
excluding training enzyme IDs, with 252,113 IDs. All four metrics use the
released notebook definitions and NumPy ranking convention.

The test score exporter now buffers the 1.48-GiB score matrix in RAM and
writes one atomic sequential NPY file for subsequent jobs. Running workers
were observed waiting on NFS page I/O under the earlier per-block mmap
flushes. The arithmetic and float32 saved scores are unchanged; this is an
I/O change, not an inference adjustment. The phase-2 score path is tested
against the same normalized residual geometry used during training.

The two requested phase-2 tests completed at approximately **23:41 UTC**.
Protocol, query-axis, candidate-axis and notebook hashes match the existing
base-model test and released CLIPZyme replay.

| Official test Table-1 model | BEDROC85 | BEDROC20 | EF5 | EF10 |
| --- | ---: | ---: | ---: | ---: |
| Released CLIPZyme replay | 0.446942 | 0.629817 | 14.085125 | 8.060006 |
| Directional + anchor-balanced base, epoch 19 | 0.468593 | 0.615931 | 13.366311 | 7.561644 |
| Full-graph decoupled phase 2, 100 steps | 0.465320 | 0.607185 | 13.125065 | 7.418361 |
| Full-graph decoupled + Smooth-AP, 100 steps | 0.465624 | 0.607742 | 13.137031 | 7.430695 |
| Chemistry channel disabled at inference | 0.214392 | 0.289848 | 6.295633 | 3.566865 |

With training enzymes excluded (Table 2), the phase-2 control scores
**0.408870 / 0.561513 / 12.256305 / 7.066652** and the ranking arm scores
**0.409338 / 0.562317 / 12.303821 / 7.088992**. Both are below the base on
all four metrics in both test settings. Both still exceed released CLIPZyme
on BEDROC85, but neither improves the remaining metrics or establishes an
overall win. No significance claim is made. These are actual test results,
not extrapolations from validation. [Completed test comparison and sources](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_single_model_v2/requested_tests_v1/results.md).

The chemistry-mask test also completed: Table-1 BEDROC85 falls to **0.214392**
from **0.468593**, with every reported test metric lower. Table-2 scores are
**0.180758 / 0.255112 / 5.564842 / 3.222989**. This independently confirms
that disabling the chemistry channel of this fixed checkpoint is harmful
in the requested test setting. It does not decide the still-training
chemistry-dropout model's outcome.

The first tests of balanced sampling, chemistry dropout, and four prototypes
remain queued behind their epoch-19 snapshots and full-library encoding.
An exact local screening-residue cache is being materialized to reduce
repeated NFS reads. It preserves source order, residue values, dtype and
candidate identities; contiguous source ranges are copied in blocks and
sampled rows are checked against the source. A separate ragged-row test
verifies noncontiguous selection, offset reconstruction and cache publication.


### Architecture search redirected after stopping current runs (2026-09-21)

The user explicitly stopped the current training and screening jobs. The six
trainers, four-shard export and automatic validation/test launchers were stopped;
existing snapshots, completed scores and logs were preserved. Cache preparation
continues. The stopped balanced-sampling, chemistry-dropout and prototype arms
had reached approximately epoch33, but no full-library test had completed for
those arms. Stopping them is not evidence that they failed.

The last completed old-control test, epoch40, is below epoch20 on all four
metrics. Table1: **0.448663 / 0.588527 / 12.760290 / 7.235093**. Table2:
**0.391717 / 0.543263 / 11.948055 / 6.909061**, in BEDROC85 / BEDROC20 /
EF5 / EF10 order. Score axes and protocol hashes match the earlier tests.

Two previously proposed local F3 changes were implemented: a rank64,
zero-output-initialized signed side residual, and a training-only chemistry
neighborhood KL penalty at weight0.1. The residual adds131,072 parameters
for the768-dimensional UniMol2 and256-dimensional ChIRo features. The
geometry teacher uses fixed input descriptors, not EC/rule labels or held-out
examples. These configurations were prepared but **not launched**, after the
user raised concern about restricting the search to F3.

The replacement architecture pilots are:

| Arm | Reaction encoder | Protein encoder |
| --- | --- | --- |
| Small reaction / F3 protein | Modality widths1024/512; output MLP hidden1024 | Original F3 |
| Lean mean | Same smaller reaction encoder | Masked ProtT5 mean and1024-hidden MLP; no SLEEC or biological blocks |
| Residue views | Same smaller reaction encoder | Global, frozen SLEEC, and four learned residue-attention views; fused into one512-vector |
| Reference | Original large reaction encoder | Original F3 |

The residue views operate before sequence pooling and are distinct from the
stopped post-pooling prototype head. They remain latent learned views, not
experimentally verified active sites. None is an ensemble. The reference's
EnzymeMap epoch20 result is reused at the same training budget; ReactZyme's
reference is trained fresh. All trainable weights are fresh per target dataset,
with seed42, anchor-balanced all-known-in-batch loss, BF16 forward and FP32
contrastive reductions. Initial EnzymeMap pilots run20 epochs at batch2048;
ReactZyme reaction_smi pilots run10 epochs at batch512, with fixed tests at5/10.
These are explicitly **before phase2**. Phase2 and other ReactZyme splits remain
required follow-ups before any claim that a final recipe wins both benchmarks.

Input audit: all12,603 EnzymeMap training reactions have explicit reactant/product
separation. None of the6,977 ReactZyme reaction_smi training reactions has that
separation; the2448 validation and386 test reactions are also participant sets.
The ReactZyme pilot therefore preserves the existing pseudo-self-reaction input
policy. No physical reaction direction is inferred or claimed for that benchmark.
A reaction-center/bond-change architecture needs a separate input-coverage audit.

The changes and relevant legacy behaviors passed107 tests plus10 subtests
(78 geometry/directional/options/positive-pair/cache cases and29 residue-view
cases). A separate GPU forward/backward preflight and launch receipt distinguish
prepared configurations from running training.

Artifacts: [replacement protocol](runs/generalization_20260919_2251/cross_paper_retraining/dual_encoder_architecture_pilots_v1/protocol.json),
[stop receipt](runs/generalization_20260919_2251/cross_paper_retraining/stopped_current_runs_20260921.json),
and [completed old-test results](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_single_model_v2/requested_tests_v1/results.md).

The replacement EnzymeMap trainers launched at **2026-09-21 00:25 UTC**: small reaction/F3 protein PID2592271 on GPU0, mean protein PID2592272 on GPU1, residue views PID2592273 on GPU2. All three passed an actual CUDA BF16 forward / FP32 contrastive backward preflight. ReactZyme trainers are queued behind the exact local residue-cache materialization. [Execution receipt](runs/generalization_20260919_2251/cross_paper_retraining/dual_encoder_architecture_pilots_v1/execution.json).


GPU3 was assigned to a fourth EnzymeMap architecture at **00:30 UTC**:
**RDKit+/DRFP reaction fingerprints + a masked-mean ProtT5 protein MLP**.
The reaction input is4096 bits: chirality-aware Morgan reactants/products
(1024 bits each) and differential reaction fingerprints (2048 bits). Both
MLPs have one1024-wide hidden layer and512-dimensional output. This removes
the F3 reaction multimodal tower, SLEEC pooler and biological factorization.
It has **6,294,528 trainable parameters**. A CUDA BF16 forward / FP32-loss
backward check passed using32 real training-reaction fingerprints.

This arm uses the same34,427 EnzymeMap training pairs, fresh trainable weights,
seed42, batch2048, anchor-balanced all-known-in-batch loss, and20-epoch pilot
budget. It changes representation as well as architecture; it is not a pure
loss ablation. Input reaction SMILES are fingerprinted without added labels
or automatic chemical standardization. The existing pipeline also exposes
reverse-reaction features, but the association loader stays forward-only.
The full-library test is queued at epoch20. Phase2 is still pending.
ReactZyme fingerprint training is not scheduled until the participant-set
semantics are separately audited; this fourth arm alone cannot establish a
shared recipe win on both benchmarks.

[GPU3 protocol and execution](runs/generalization_20260919_2251/cross_paper_retraining/dual_encoder_architecture_pilots_v1/fingerprint_followup/protocol.json).

At the first matched epoch3 diagnostic validation (1937 protein candidates),
R→E MRR is **0.077843** for the existing large F3 control, **0.061137** for
the smaller reaction/F3-protein arm, **0.051069** for the mean-protein MLP,
and **0.063086** for learned residue views. None leads the control at this
early budget. These are small-pool validation diagnostics, not BEDROC/EF on
the261,907-ID screening test; they neither establish a win nor conclusively
reject an undertrained architecture. No new full-library test has completed.
[Exact epoch3 diagnostic snapshot](runs/generalization_20260919_2251/cross_paper_retraining/dual_encoder_architecture_pilots_v1/early_validation_epoch3.json).

### Screening monitoring correction (2026-09-21,00:41 UTC)

The user requested BEDROC-based feedback instead of MRR. EnzymeMap selection
already uses full-library validation BEDROC85; the recent epoch3 comparison
was only a small-pool MRR diagnostic and does not establish which architecture
is better at screening. The live monitoring report now leads with BEDROC85,
BEDROC20, EF5 and EF10, separating validation from test and checking their
exact candidate/query denominators. Missing results are shown as pending.
No MRR-based promotion or rejection is made. ReactZyme retains its retrieval
metrics. The training loss is unchanged.

The active jobs have no saved intermediate weights before epoch20. Their
in-memory callback cadence cannot be changed by editing YAML on disk.
Training is preserved, with full-library BEDROC/EF queued for that first
snapshot; no earlier BEDROC result is claimed. The live report is
[screening_metrics.md](runs/generalization_20260919_2251/cross_paper_retraining/dual_encoder_architecture_pilots_v1/screening_metrics.md).


### Architecture pilots: matched epoch-12 diagnostics (2026-09-21)

At epoch 12, R→E / E→R small-pool validation MRR is 0.096374 / 0.139216 for the earlier Directional F3 + anchor-balanced model, 0.075497 / 0.116159 for the smaller reaction/F3-protein model, 0.082158 / 0.106314 for the mean-protein model, 0.086473 / 0.120822 for residue views, and 0.090427 / 0.121515 for reaction fingerprints plus mean-protein encoding. No new variant leads this matched-epoch diagnostic. These are not full-library BEDROC/EF results and do not select or reject screening checkpoints.

The fingerprint model completed its 20-epoch training budget and began full-library validation/test evaluation. Its diagnostic R→E MRR fell from 0.090427 at epoch 12 to 0.087217 at epoch 18, while validation loss rose from 5.732293 to 5.763954. This is consistent with possible overfitting but does not establish the optimum screening epoch; no earlier fingerprint checkpoint was saved.

[Exact diagnostic snapshot](runs/generalization_20260919_2251/cross_paper_retraining/dual_encoder_architecture_pilots_v1/early_validation_epoch12.json).


### Fingerprint architecture: completed full-library test (2026-09-21, 00:59 UTC)

The fresh reaction-fingerprint + mean-ProtT5 model completed its fixed 20-epoch EnzymeMap pilot. Its full-library test results are: Table 1 BEDROC85 **0.417348**, BEDROC20 **0.566199**, EF5 **12.453180**, EF10 **6.990138**; Table 2 **0.375336 / 0.530953 / 11.803647 / 6.718378**, respectively. It underperforms released CLIPZyme and the earlier Directional F3 + anchor-balanced epoch-20 checkpoint on all four metrics in both settings. This fixed-budget result is not a benchmark win.

Protocol receipt, query/candidate axes, and notebook hashes match all three evaluations. These are exploratory tests after repeated test inspection; full-library validation BEDROC85 remains the checkpoint selector. The new arm is a fresh-base architecture/representation pilot before phase 2. Other architecture screening results are still pending.

[Completed test comparison](runs/generalization_20260919_2251/cross_paper_retraining/dual_encoder_architecture_pilots_v1/completed_test_results.md).

Full-library validation also completed: fingerprint Table 1 BEDROC85 **0.446840**, BEDROC20 **0.618057**, EF5 **13.669094**, EF10 **7.813933**; Table 2 **0.388698 / 0.575796 / 12.951162 / 7.561823**. The same-epoch Directional F3 + anchor-balanced reference has validation BEDROC85 **0.502495**, so the fingerprint model is **0.055655 lower** at the matched 20-epoch budget. Association-manifest hashes and validation denominators match. This provides validation-only evidence against extending the current fingerprint pilot; its fixed-budget training has already ended. It does not establish the performance of unsaved earlier checkpoints.


### SLEEC-preserving early-checkpoint study (2026-09-21, 01:12 UTC)

Four fresh EnzymeMap runs retain SLEEC pooling and the exact 34,427 target training pairs: replay of Directional F3 + anchor-balanced; chemistry-neighborhood regularization (weight 0.1); low-rank signed directional residual plus the same regularization; and contrastive beta 20 (versus control beta 10). All preserve the frozen pretrained feature banks, train for at most 18 epochs, and save full checkpoints every 3 epochs. Eight loader workers per trainer replace two/four to improve throughput. The reaction geometry teacher uses input chemistry descriptors only.

Full-library validation reports BEDROC85, BEDROC20, EF5 and EF10, using 2,652 queries and 261,907 IDs, plus the training-enzyme-excluded setting (2,216 / 252,113). BEDROC85 selects among epochs 3/6/9/12/15/18; small-pool MRR never promotes, rejects, or selects a screening model. One validation-selected checkpoint per arm will receive an exploratory test evaluation. Phase 2 remains required for the final recipe. No ensemble, hubness correction, or RefSeq run is introduced.

The pending mean-only ReactZyme arm had launched during a tool interruption; it was terminated at 01:09 UTC because it removes SLEEC. Its cancelled run is not counted as completed. The other three ReactZyme trainers retain SLEEC.

[Protocol](runs/generalization_20260919_2251/cross_paper_retraining/sleec_early_geometry_v1/protocol.json) and [live screening validation/test metrics](runs/generalization_20260919_2251/cross_paper_retraining/sleec_early_geometry_v1/screening_metrics.md).


### Completed SLEEC architecture and neighbor-refinement results (2026-09-21, 01:44 UTC)

The residue-view enzyme encoder retains SLEEC, a global protein view, and four learned residue views. It leads the completed architecture pilots on full-library validation BEDROC85: **0.535856**, with BEDROC20 **0.724884**, EF5 **16.247021**, and EF10 **8.719741**. This validation evidence supports the next controlled training study.

Its fixed epoch-20 test gives Table 1 **0.486984 / 0.644680 / 14.075036 / 7.890832** and Table 2 **0.448227 / 0.615981 / 13.615824 / 7.737826**, in BEDROC85 / BEDROC20 / EF5 / EF10 order. It beats released CLIPZyme on two Table-1 metrics and three Table-2 metrics, but not all four in either setting. Table-1 BEDROC85 is slightly above the published FGW-CLIP value 0.4866; the 0.000384 margin is not a significance claim or an all-metric win.

[All four architecture tests and aligned references](runs/generalization_20260919_2251/cross_paper_retraining/dual_encoder_architecture_pilots_v1/completed_test_results.md). The smaller-reaction F3 and mean-only controls do not establish the requested win. Mean-only models are not eligible under the retained-SLEEC requirement.

Train-neighbor refinement of the existing SLEEC phase-2 model completed on all three ReactZyme splits. Each recipe was selected on validation before reading its test labels; five of six test metrics became worse. It is retained as a negative experiment, not promoted. [Exact six-cell comparison](runs/generalization_20260919_2251/phase2_training_neighbor_v1/comparison.md).

A full-library export briefly failed from overlapping GPU evaluators. Only the missing shard was regenerated; the three successful shards were preserved. Exports now coordinate per physical GPU, stage optimizer-containing checkpoints on CPU, and choose export microbatches from available VRAM. Exact NumPy ranking and BEDROC/EF definitions are preserved while query metric calculations run in eight CPU threads. Tests cover ranking ties, training-candidate exclusions, and invalid scores.

### Next SLEEC multiview training study (prepared 2026-09-21, 01:44 UTC)

Four fresh EnzymeMap runs use the same 34,427 training pairs and the residue-view architecture: reaction-neighborhood preservation (0.1), protein-neighborhood preservation (0.1), both (0.1 each), and contrastive beta 5. Protein neighborhoods use detached mean ProtT5 features; reaction neighborhoods use fixed chemistry descriptors. No new association labels or EC supervision are introduced. New auxiliary weights default to zero for old checkpoints. The earlier running trainers retain their already-loaded implementation.

Each run has a 24-epoch maximum, with the predeclared full-library validation grid **3/6/12/18/24**. All four screening metrics are reported; BEDROC85 alone selects the checkpoint. Small-pool MRR computation is disabled in these new EnzymeMap runs, and no in-trainer best-checkpoint or early-stopping selection is active. Eight workers per trainer and GPU export coordination preserve capacity for ongoing ReactZyme training. Launch waits for the preceding four trainers to finish, to avoid exhausting host RAM. Phase 2 remains pending for the final selected model.

[Protocol](runs/generalization_20260919_2251/cross_paper_retraining/sleec_multiview_geometry_v1/protocol.json).


### First all-four-metric CLIPZyme test win (2026-09-21, 01:56 UTC)

SLEEC residue views with validation-selected phase-2 residual refinement exceeds released CLIPZyme on all four test point estimates in both official settings. Table 1 gives **BEDROC85 0.514886, BEDROC20 0.673510, EF5 14.712503, EF10 8.236496**, versus CLIPZyme **0.446942 / 0.629817 / 14.085125 / 8.060006**. Table 2 gives **0.477094 / 0.645176 / 14.302808 / 8.023259**, versus **0.391328 / 0.588619 / 13.396710 / 7.808135**.

The residual uses full-graph bidirectional uniform-positive cross entropy, temperature 0.2, identity penalty 10, and 100 updates. Its 34,180 unique sequence/reaction edges derive from the same 34,427 original training rows. Validation BEDROC85 selected it from the predeclared four-objective, four-snapshot grid, including the unmodified base. Validation Table 1 is **0.544781 / 0.733617 / 16.348631 / 8.828459**. No MRR selection is used. This result applies the learned residual component of phase 2; semantic-anchor composition and density gating remain pending.

It exceeds published FGW-CLIP on three Table-1 metrics (EF5 remains below 14.91) and all four Table-2 metrics. Paired query bootstrap intervals exclude zero for both BEDROC differences and EF5 against CLIPZyme; EF10 intervals include zero in both settings. All four point estimates win, but an all-four statistically conclusive win is not established. Repeated test inspection makes this exploratory.

[Method, benchmark tables, uncertainty and artifact links](runs/generalization_20260919_2251/cross_paper_retraining/sleec_multiview_phase2_v1/comparison.md). ReactZyme transfer of this new recipe and full phase-2 composition remain pending.

### Completed phase-2 composition and replication plan (2026-09-21, 02:40 UTC)

Full-library validation selects semantic weight **alpha 0.1** with full residual scale from the predeclared six-point composition grid. The semantic dictionary contains training associations only. Its weighted coordinates are concatenated with the independent learned residual representations, retaining a dual encoder and SLEEC pooling. Validation Table 1 is **0.546203 / 0.734269 / 16.351725 / 8.826858**. BEDROC85 is the selection criterion; BEDROC20, EF5 and EF10 are reported alongside it. No MRR selection, ensemble, hubness correction or candidate-pool normalization is used.

The selected composed model's actual test gives Table 1 **0.517299 / 0.677777 / 14.838323 / 8.274966** and Table 2 **0.477487 / 0.645705 / 14.319221 / 8.030113**, in BEDROC85 / BEDROC20 / EF5 / EF10 order. All eight point estimates exceed the released CLIPZyme replay. Seven exceed the published FGW-CLIP table: Table-1 EF5 remains **0.071677 below 14.91**. This is not an all-metric FGW-CLIP win. Unadjusted paired-query bootstrap intervals exclude zero for all four Table-1 differences, but Table-2 EF10 still crosses zero. Reaction-rule clustering and repeated test inspection limit the inference.

[Updated comparison, uncertainty, and exact artifacts](runs/generalization_20260919_2251/cross_paper_retraining/sleec_multiview_phase2_v1/comparison.md).

Two new seeds, **17 and 73**, started fresh EnzymeMap F3 training at 02:29 UTC with the original residue-view architecture, exact target training associations and a fixed 20-epoch budget. Both will receive the same fixed residual-head recipe (temperature 0.2, identity weight 10, 100 updates) and report their tests regardless of which seed scores better. There is no seed selection or ensemble. These replications initially assess the residual component; semantic composition results will be identified separately. [Replication protocol](runs/generalization_20260919_2251/cross_paper_retraining/sleec_multiview_replication_v1/protocol.json).

### Negative ReactZyme follow-ups and resource recovery (2026-09-21, 02:40 UTC)

The new soft-temperature phase-2 head was tested independently on all three ReactZyme training splits, then inserted into each frozen parent composition while keeping its semantic dictionary and support gates fixed. The predeclared snapshots were 5/20/50/100 updates. All three validation selections retained the original parent. A second campaign warm-started the existing target-specific residual head, reduced the learning rate to 1e-5, and applied identity weight 50 relative to its initial outputs. It also retained the parent on all three splits. Neither campaign establishes a ReactZyme improvement; repeated parent test values are not counted as new wins.

Artifacts: [fresh-head protocol](runs/generalization_20260919_2251/phase2_soft_ce_v1/protocol.json), [warm-start protocol](runs/generalization_20260919_2251/phase2_soft_ce_warm_v1/protocol.json). Each split directory includes `composition_validation.json`, `composition_selected.json`, and `test_summary.json`.

All four multiview geometry trainers were killed with exit -9 at approximately 01:54 UTC after a host-RAM spike. Kernel OOM attribution could not be confirmed because kernel logs are inaccessible. Owned orphan loader processes were cleaned up. Training resumed from complete epoch-3 checkpoints, including optimizer state, with four workers instead of eight and storage-precision residue transport. All four resumed runs completed the planned 24 epochs by **02:28:49 UTC**. This is ordinary checkpoint recovery, not a claim of an uninterrupted identical random trajectory. Their fixed full-library validation grid and validation-selected tests continue. The monitor now reads resumed logs instead of displaying a stale epoch 5.

An exact-input I/O benchmark measured **3.060 versus 1.744 seconds per step** over 15 post-warmup steps on the same ReactZyme batch-512 configuration (observed 1.75x). The stored FP16 residue values are restored to the original FP32 model input after GPU transfer; this is not input quantization. Logged losses matched, and a separate 512-protein export check gave bitwise identical embeddings. The sequential benchmark ran on a shared host, so cache and load effects prevent an isolated hardware speed claim. The threaded full-library metric implementation also reproduced the previous full-test summary and per-query hash exactly.

[I/O benchmark](runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_residue_io_v1/comparison.json), [export parity](runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_residue_io_v1/export_parity.json), [recovery status](runs/generalization_20260919_2251/cross_paper_retraining/sleec_multiview_geometry_v1/resume_status.json).

### Seed stability and additional ReactZyme probes (2026-09-21, 02:56 UTC)

The two fresh-seed EnzymeMap replications completed the fixed 20-epoch base plus 100-update phase-2 residual recipe. They do **not** reproduce a consistent all-four-metric win individually:

| Seed | Table | BEDROC85 | BEDROC20 | EF5 | EF10 |
| --- | --- | ---: | ---: | ---: | ---: |
| 42 | 1 | 0.514886 | 0.673510 | 14.712503 | 8.236496 |
| 17 | 1 | 0.471129 | 0.644877 | 14.274048 | 8.058571 |
| 73 | 1 | 0.440861 | 0.615569 | 13.746328 | 7.914820 |
| 42 | 2 | 0.477094 | 0.645176 | 14.302808 | 8.023259 |
| 17 | 2 | 0.420624 | 0.602273 | 13.448291 | 7.775959 |
| 73 | 2 | 0.392385 | 0.575811 | 12.973943 | 7.648647 |

Seed 17 wins both BEDROC metrics and EF5 against CLIPZyme in both tables, but misses EF10. Seed 73 loses all four Table-1 metrics and wins only BEDROC85 in Table 2. These runs retain the original recipe while using the verified storage-precision loader with four workers. The seed-42 result is a real point-estimate win, but stability remains a limitation. The fixed alpha-0.1 semantic composition chosen at 02:21 UTC is being applied to both replications without per-seed tuning or seed selection.

A new train-only graph-alignment residual was evaluated on the three existing ReactZyme phase-2 models. It fits anchor-balanced positive-partner mean squared error with ridge regularization toward identity, retaining the semantic coordinates exactly. Validation considered ridge 0.1/1/10 and residual strength 0.1/0.25/0.5/1, including the parent. Reaction-similarity and time splits retained the parent. The enzyme-similarity split selected ridge 1, strength 0.25: test R→E **0.698019** versus **0.697468**, E→R **0.973632** versus **0.973676**. This tiny mixed change does not establish an improvement in both directions; it is not promoted as a new winning recipe. [Protocol and per-split artifacts](runs/generalization_20260919_2251/phase2_graph_alignment_v1/protocol.json).

The three fresh ReactZyme architecture pilots have a fixed continuation to 20 epochs after their 10-epoch checkpoint. Optimizer state is resumed and the same target associations, batch size, architecture and loss are retained; four workers and exact storage-precision transport improve throughput. Separate predeclared phase-2 studies at epochs 10 and 20 use the successful EnzymeMap residual recipe (temperature 0.2, identity 10, 100 updates), snapshots 0/5/20/50/100 and semantic weights 0/0.1/0.25. Selection compares balanced seen/unseen **validation** all-positive MRR with direction-specific guards against the existing ReactZyme parent; test values do not choose the architecture. The selected model alone receives a new phase-2 test. ReactZyme uses its own retrieval metric; EnzymeMap continues to use full-library BEDROC85/20 and EF5/10.

[Continuation protocol](runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_architecture_extension_v1/protocol.json), [epoch-10 phase-2 protocol](runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_architecture_phase2_epoch10_v1/protocol.json), [epoch-20 phase-2 protocol](runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_architecture_phase2_epoch20_v1/protocol.json).

### Joint-metric screening selection: first all-cell FGW-CLIP point-estimate win (2026-09-21, 03:09 UTC)

A separately declared secondary selector uses the **geometric mean of validation BEDROC85, BEDROC20, EF5/20 and EF10/10**, giving each requested screening metric equal log weight. The existing four-objective residual grid still selects soft-temperature CE at step 100. The six-point composition grid selects full residual scale and **alpha 0.25**. The original BEDROC85-only selection remains unchanged and separately reported. This secondary selection was designed after inspecting the original primary test, so it is explicitly exploratory, not a claim of untouched-test confirmatory validation.

The newly selected test results exceed both the released CLIPZyme replay and all four published FGW-CLIP point estimates in both settings:

| Setting / model | BEDROC85 | BEDROC20 | EF5 | EF10 |
| --- | ---: | ---: | ---: | ---: |
| Table 1, joint-metric selection | **0.523889** | **0.683692** | **14.973095** | **8.301971** |
| Table 1, released CLIPZyme | 0.446942 | 0.629817 | 14.085125 | 8.060006 |
| Table 1, published FGW-CLIP | 0.486600 | 0.666900 | 14.910000 | 8.180000 |
| Table 2, joint-metric selection | **0.476145** | **0.646876** | **14.374697** | **8.062667** |
| Table 2, released CLIPZyme | 0.391328 | 0.588619 | 13.396710 | 7.808135 |
| Table 2, published FGW-CLIP | 0.451400 | 0.614300 | 13.570000 | 7.610000 |

The Table-1 FGW EF5 margin is only **0.063095**. This is a seed-42 point-estimate win, not established superiority across seeds or measured catalytic activity. FGW remains a published-number comparison without a released checkpoint. The fixed alpha-0.25 recipe is being replicated on seeds 17 and 73 without per-seed tuning. The alpha-0.1 replication already completed: seed 17 wins all eight CLIPZyme cells, whereas seed 73 still loses all four Table-1 cells.

[Secondary selector protocol](runs/generalization_20260919_2251/cross_paper_retraining/sleec_multiview_phase2_v1/all_metrics_selector_protocol.json), [validation selection](runs/generalization_20260919_2251/cross_paper_retraining/sleec_multiview_phase2_v1/composition_all_metrics_v1/selection.json), [actual test](runs/generalization_20260919_2251/cross_paper_retraining/sleec_multiview_phase2_v1/composition_all_metrics_v1/test_evaluation/summary.json).

Fresh residue-view F3 training also started for the Enzyme-Sim and Time ReactZyme splits, using the same architecture, fresh trainable weights, seed 42, batch 512, and each split's own target training data. Their 10/20-epoch phase-2 follow-ups are declared in separate protocols. This is necessary to assess a shared architecture across benchmarks; the earlier positive EnzymeMap result alone does not establish that transfer. A same-seed EnzymeMap replay checks the optimized loader: its first recorded step and epoch losses exactly match the original seed-42 run so far.

The earlier ReactZyme parent used a longer phase-2 fit (temperature 0.07, identity weight 2, up to 1,000 updates; selected step 550 for Reaction-Sim). Separate follow-ups of the new architectures now test that historical training scale, with fixed snapshots 100/250/500/1000 and validation-only composition selection. The successful EnzymeMap 100-update recipe remains separately reported. No additional association data, ensemble or hubness correction is introduced.

### Fixed phase 2 across the residue-view geometry family (2026-09-21, 03:42 UTC)

All four completed geometry arms use their BEDROC85-selected base checkpoint and exactly the same phase-2 recipe: positive CE at temperature 0.2, identity weight 10, 100 updates, residual cap 1 and training-only semantic weight 0.25. Every arm is reported, including negative controls elsewhere. In BEDROC85 / BEDROC20 / EF5 / EF10 order:

| Base variant | Test Table 1 | Test Table 2 |
| --- | --- | --- |
| Reaction geometry | 0.527465 / 0.690038 / 15.205107 / 8.342150 | 0.477950 / 0.652098 / 14.544802 / 8.100486 |
| Protein geometry | 0.537194 / 0.707801 / 15.576570 / 8.591632 | 0.491104 / 0.673550 / 14.989642 / 8.400875 |
| Both geometry penalties | 0.540318 / 0.702669 / 15.478101 / 8.391733 | 0.492877 / 0.666414 / 14.843120 / 8.174575 |
| Softer F3 logits, beta 5 | **0.580104 / 0.743666 / 16.404273 / 8.759964** | **0.541624 / 0.713875 / 15.917810 / 8.534267** |

All four exceed the released CLIPZyme and published FGW-CLIP point estimates in every cell. The beta-5 base was selected at epoch 18. Its composed validation Table 1 is **0.547120 / 0.738288 / 16.404597 / 9.006311**, exceeding the earlier beta-10 residue-view composition on all four validation metrics. Its wider test margin is encouraging, but these are still exploratory, recovered seed-42 runs after repeated benchmark inspection. The stronger result does not erase the earlier seed-instability finding.

[Detailed beta-5 comparison and provenance](runs/generalization_20260919_2251/cross_paper_retraining/sleec_multiview_temperature5_fixed_phase2_v1/comparison.md). Paired reaction-rule-component bootstrap intervals against released CLIPZyme stay above zero for BEDROC85, BEDROC20 and EF5 in both settings; **EF10 intervals include zero** under this more conservative dependence check. Query-level intervals exclude zero for all eight cells. These are unadjusted descriptive intervals, not confirmatory activity evidence.

Three uninterrupted fresh beta-5 trainings use seeds 17, 73 and 42, a fixed 18-epoch budget, the same target associations, and the same fixed residual/semantic phase 2. No seed will be selected or ensembled. The launch protocol inherited stale descriptive fields saying 20 epochs and two residual-only seeds; these were corrected in a separately recorded metadata revision. The actual three task configs, epoch-18 checkpoints and alpha-0.25 composition were already fixed at launch and are unchanged. [Replication protocol and correction receipt](runs/generalization_20260919_2251/cross_paper_retraining/sleec_beta5_fresh_replication_v1/protocol.json).

A separate globally highest-validation large-F3 beta-20 model illustrates the selection problem: its composed validation BEDROC85 is **0.560179**, but test Table 1 is only **0.474812 / 0.633992 / 14.095749 / 7.850042**. We do not silently replace that validation-selected outcome with a test-selected architecture or claim one global selection protocol proves the new architecture superior.

### Validation-metric guard and reproducibility limits (2026-09-21, 03:42 UTC)

Active EnzymeMap F3 runs disable retrieval-MRR validation, retain fixed epoch snapshots, and use external full-library BEDROC85, BEDROC20, EF5 and EF10 reports. Phase-2 training uses `external_screening`: a regression check makes small-pool validation and MRR evaluation raise if called. That check and full-pool tie/parallel metric checks passed (**4 tests**). Primary checkpoint selection uses validation BEDROC85; the separately declared four-metric geometric selector and fixed-recipe replications remain distinct studies.

The completed beta-10 seed-42 optimized-loader replay matched all five logged training losses through step 49 but diverged from step 59, after the first validation boundary. Its final model weights are not bitwise identical. The replay also disables the original run's diagnostic retrieval validation and changes loader-worker count, so this comparison does not isolate I/O transport from RNG consumption. Exact stored input and 512-protein embedding-export parity remain valid; full-run identity is not established. Replay residual-only test Table 1 is **0.527691 / 0.687947 / 15.077103 / 8.337472**, Table 2 **0.492605 / 0.660516 / 14.590256 / 8.109948**. [Replay audit](runs/generalization_20260919_2251/cross_paper_retraining/sleec_multiview_seed42_io_replay_v1/checkpoint_parity.json).

### Fresh beta-5 replication and matched controls (2026-09-21, 04:08 UTC)

The first two uninterrupted 18-epoch beta-5 base trainings and their fixed phase-2 evaluations are complete. Every seed is reported:

| Seed / setting | BEDROC85 | BEDROC20 | EF5 | EF10 |
| --- | ---: | ---: | ---: | ---: |
| 17 / Table 1 | 0.516611 | 0.704690 | 15.851767 | 8.680139 |
| 17 / Table 2 | 0.467917 | 0.669474 | 15.212468 | 8.504379 |
| 73 / Table 1 | 0.486885 | 0.661899 | 14.773096 | 8.075776 |
| 73 / Table 2 | 0.440750 | 0.624457 | 14.054898 | 7.786950 |

Seed 17 exceeds both comparators in all eight cells. Seed 73 exceeds released CLIPZyme in seven cells (Table-2 EF10 is slightly lower) and published FGW-CLIP in four. This is **not a consistent all-seed, all-metric win**. Fresh seed 42 remains in training, and the three-seed mean will include it regardless of outcome. These independent model summaries are not ensemble predictions.

A matched beta-10 control uses exactly the same three seeds, fixed 18 epochs, architecture, loader settings, training associations and fixed phase 2. Only the base inverse temperature changes. This addresses the earlier beta-10 control's different 20-epoch budget. [Matched control protocol](runs/generalization_20260919_2251/cross_paper_retraining/sleec_beta10_matched18_replication_v1/protocol.json).

### ReactZyme follow-ups and evaluation scheduling (2026-09-21, 04:08 UTC)

The historical training-support gate was transferred to the new ReactZyme architectures: nearest training F3 support controls each endpoint independently, with thresholds fitted from training leave-one-out support (25th/95th percentiles, deterministic 4,096-protein sample). The initial epoch-10 grid improved balanced validation scores but failed a direction-specific generalization guard. Expanding semantic weights from a boundary optimum at 0.25 to 0.5/0.75 still retained the existing parent. The parent test has not been relabeled as a new win.

The compact reaction and residue-view models completed epoch 20 and continue to epoch 30 with full optimizer-state recovery and the same fixed learning rate. Their phase-2 heads reuse completed F3 features. Fresh beta-5 residue-view training now covers all three ReactZyme splits with their own target training associations; the Enzyme-Sim and Time pilots have fixed epoch-10 and epoch-20 follow-ups. [Thirty-epoch continuation](runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_architecture_extension30_v1/protocol.json), [beta-5 remaining splits](runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/protocol.json), [epoch-20 support-gate protocol](runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_architecture_support_phase2_epoch20_expanded_v1/protocol.json).

Resource recovery is explicit: seed 73's EnzymeMap training completed normally, but a subsequent protein-export shard failed at CUDA initialization during a memory peak. Evaluation reused the verified checkpoint and three completed shards; it now completed. Separately, overlapping protein exports and paired soft/legacy ReactZyme residual fits caused four residual-fit OOMs across GPUs 0 and 1. Their completed F3 features/checkpoints were retained; failed residual directories were preserved, and the short heads are retrained from the same initialization. Full-graph fits now share the physical-GPU export lock and wait for sufficient free VRAM before initializing CUDA. EnzymeMap still uses external BEDROC/EF validation; its no-MRR regression check passed again after this scheduling change. The additional beta-5 split launcher initially hit a Python module-name collision before any trainer started; renaming the launcher resolved it.

### Three fresh beta-5 seeds complete (2026-09-21, 04:22 UTC)

Fresh seed 42 completes the fixed 18-epoch recipe with test Table 1 **0.561472 / 0.735602 / 16.330732 / 8.792757** and Table 2 **0.522387 / 0.706061 / 15.802751 / 8.600392**. Combined with seeds 17 and 73, the mean exceeds both released CLIPZyme and published FGW-CLIP in all four metrics in both settings. This summarizes separate trained models, not ensemble inference. Per-seed outcomes and sample standard deviations are retained in the [replication comparison](runs/generalization_20260919_2251/cross_paper_retraining/sleec_beta5_fresh_replication_v1/comparison.md); the weaker seed 73 is not discarded.

A secondary evaluation removes 2,285 additional candidate aliases whose exact sequence appears in the target training data. It leaves **249,828 candidate IDs and 1,333 queries**, and compares the same frozen model score matrices with released CLIPZyme under the same stricter mask. All original Table-2 per-query metrics reproduce before the mask changes. The fresh-seed mean remains above CLIPZyme on all four metrics. [Exact-sequence-disjoint secondary results](runs/generalization_20260919_2251/cross_paper_retraining/sleec_sequence_disjoint_screening_v1/comparison.md). These are additional retrospective screening results, not FGW-CLIP's official table, remote-homology separation, or new activity measurements.

A source/configuration audit confirms identical model sections across the beta-5 EnzymeMap and three ReactZyme target configurations. The global-positive lookup is constructed only from filtered training rows; it does not import validation/test positives. EnzymeMap has 34,427 original training rows and 34,180 unique internal reaction/sequence edges; the three ReactZyme training files contain 147,393, 152,748 and 149,554 unique edges. Frozen encoder/SLEEC pretraining remains a difference from the published competitors. [Audit](runs/generalization_20260919_2251/cross_paper_retraining/sleec_beta5_fresh_replication_v1/training_lineage_audit.json).

The Reaction-Sim beta-5 trainer also encountered a GPU-memory collision after epoch 7. It resumes the complete epoch-5 optimizer checkpoint on GPU 1, losing approximately two epochs of unsaved progress. This ReactZyme run is a recovered trajectory, not an uninterrupted reproducibility run. GPU memory polling now retries temporary driver timeouts, and interrupted support-gate validations are being retried with their original model/selection recipes. No EnzymeMap training or completed screening result changed.

### Validation coverage explains a model-selection risk (2026-09-21, 04:31 UTC)

Rechecking the original checksum-verified CLIPZyme split confirms **321 training rules, 5 validation rules and 68 test rules**, consistent with the earlier release audit. Rule 279 supplies **99.22% of validation associations** and **98.53% of evaluated validation queries**. Query-averaged validation BEDROC/EF therefore mostly measures one rule family. The large beta-20 model's higher score on that family outweighed worse scores on small validation rules, despite its inferior test performance. This is a plausible explanation for EnzymeMap selection instability, not causal proof or an explanation of every ReactZyme/Case1 failure. [Rule-coverage diagnostic and exact sources](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_validation_rule_coverage.md).

The official split and metrics are unchanged. A future training-internal rule-diverse validation study would require a fresh final refit on all official training associations to maintain the matched-training comparison. No test labels are proposed for that selection.

### Actual Reaction-Sim test of the new epoch-20 residue-view model (2026-09-21, 04:31 UTC)

The validation-eligible support-gated residue-view model selected step 1,000, residual cap 1, semantic alpha 0.25, and the training-fitted F3 support gate. Its actual test is **R→E 0.412466 / E→R 0.477566** (all-positive MRR). The existing phase-2 reference is **0.415174 / 0.523981**. The new variant therefore does not improve the reference, despite its better balanced validation score. This negative result is retained. [Selection and test artifacts](runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_residue_views_phase2_epoch20_fast_v1/selected_test_summary.json). Beta-5 and epoch-30 follow-ups remain pending; the EnzymeMap mean win does not establish a shared winning model across both benchmarks.

### Validation confirmation and paired temperature control (2026-09-21, 04:40 UTC)

All six fresh EnzymeMap replication configurations disable retrieval-MRR validation and early stopping; fixed epoch snapshots are evaluated against the full screening library with **BEDROC85, BEDROC20, EF5 and EF10**. The no-MRR phase-2 regression and full-pool metric checks passed again (**4 tests**). Primary selection studies continue to use validation BEDROC85; fixed-budget replications do not select checkpoints or seeds by their test outcomes.

The first two matched 18-epoch beta-10 controls are complete. Against each corresponding beta-10 seed, beta 5 improves all eight test measurements for seed 17 and six of eight for seed 73 (both EF10 values are slightly lower for seed 73). The third beta-10 seed remains in training, so no completed three-seed paired estimate is claimed yet. A configuration audit confirms that only the inverse temperature changes in training mathematics. [Automatically updated paired comparison](runs/generalization_20260919_2251/cross_paper_retraining/sleec_beta10_matched18_replication_v1/comparison.md).

The Enzyme-Sim epoch-10 phase-2 evaluation had stopped on a temporary NVML timeout before its residual fit. It was recovered on GPU 3 with verified completed F3 features and the same model/selection recipe. The original error log and execution receipt are preserved.

### Fixed EnzymeMap recipe transferred to ReactZyme (2026-09-21, 04:50 UTC)

To avoid reporting only validation-eligible variants, a separate fixed-recipe study declares all twelve combinations of beta 5/10, epochs 10/20, and three ReactZyme target splits before reading their new test results. It reuses each target-trained base and its 100-update positive-CE head, fixes residual cap 1 and semantic alpha 0.25, and reports every outcome without a validation gate or parent fallback. This preserves the model architecture and phase-2 recipe; target training budgets differ explicitly (ReactZyme batch 512 and 10/20 epochs, EnzymeMap batch 2,048 and 18 epochs). No new base training was launched by this evaluation study. These are exploratory comparisons after earlier test inspection.

| Configuration | R→E all-positive MRR | E→R all-positive MRR |
| --- | ---: | ---: |
| enzyme_smi_beta10_epoch10 | 0.680091 | 0.971884 |
| reaction_smi_beta10_epoch10 | 0.402786 | 0.470564 |
| reaction_smi_beta10_epoch20 | 0.405844 | 0.484152 |
| time_beta10_epoch10 | 0.556108 | 0.803804 |

The completed beta-10 entries remain below the retained phase-2 reference in both directions. Reaction-Sim E→R improves from epoch 10 to 20, but this is not a win over the reference or TIGER’s stronger variants. Pending beta-5 outcomes are not imputed. The full table updates as the existing checkpoints and heads finish: [fixed-transfer comparison](runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_fixed_enzymemap_recipe_transfer_v1/comparison.md), [predeclared protocol](runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_fixed_enzymemap_recipe_transfer_v1/protocol.json).

### Residue-view fusion diagnostic (2026-09-21, 05:00 UTC)

The current multiview encoder computes `normalize(global_unit + scale * fused_unit)`, where `scale = 0.5 * sigmoid(parameter)`. Its learned scales are about **0.103** for both EnzymeMap temperature settings, **0.123** for Reaction-Sim beta 5 at epoch 10, and **0.143** for Reaction-Sim beta 10 at epoch 20. Consequently the immediate fused representation stays within roughly 6–8 degrees of its learned global branch. This is an algebraic limitation on the final fusion, not evidence that SLEEC has no training effect or that increasing its contribution will help. A future validation-only fusion-strength calibration could test this hypothesis without ensembles or hubness correction. No such inference change has been applied to the reported results. [Checkpoint diagnostic](runs/generalization_20260919_2251/cross_paper_retraining/sleec_multiview_fusion_scale_diagnostic.json).

### Four-hour benchmark checkpoint (2026-09-21T05:07:28.083901+00:00)

The EnzymeMap mean win is replicated across three fresh seeds, but a shared model that improves all six ReactZyme cells is not established. The fixed beta-5 epoch-10 Reaction-Sim test is **0.389372 / 0.515102** (R→E/E→R): better E→R and worse R→E than matched beta 10 at epoch 10 (**0.402786 / 0.470564**). Beta-10 Enzyme-Sim epoch 20 is **0.688714 / 0.976363**, and Time epoch 20 is **0.574441 / 0.832144**. All outcomes remain in the automatically updated fixed-recipe table. [Concise four-hour report](archive/20260928/documents/benchmark_results_20260921_0506.md).

### Completed matched inverse-temperature control (2026-09-21T05:08:58.058340+00:00)

All three matched 18-epoch beta-10 controls and fixed phase-2 tests are complete. Beta 5 improves **22/24 paired seed-by-metric test cells**, with the two exceptions being seed 73 EF10. Its three-seed mean improves every metric in both official settings. These correlated cells are descriptive comparisons, not 24 independent statistical samples.

| Setting / recipe | BEDROC85 ↑ | BEDROC20 ↑ | EF5 ↑ | EF10 ↑ |
| --- | ---: | ---: | ---: | ---: |
| table1 / beta 5 | 0.521656 | 0.700731 | 15.651865 | 8.516224 |
| table1 / beta 10 | 0.494040 | 0.664343 | 14.685863 | 8.168627 |
| table2 / beta 5 | 0.477018 | 0.666664 | 15.023372 | 8.297240 |
| table2 / beta 10 | 0.439892 | 0.623451 | 13.953732 | 7.907565 |

The paired configuration audit confirms identical model, data, optimizer, 18-epoch budget and phase-2 recipe; only base inverse temperature changes in training mathematics. [Complete paired comparison and exact sources](runs/generalization_20260919_2251/cross_paper_retraining/sleec_beta10_matched18_replication_v1/comparison.md).

### Export optimization benchmark (2026-09-21T05:08:58.058340+00:00)

An ordered four-worker CPU residue exporter produced bitwise-identical embeddings on all 8,192 proteins at the same batch size 128 and FP32 inference. The CPU-fork implementation averaged **12.643 seconds**, versus **11.369 seconds** for the serial exporter across reversed-order repeats. A spawn-worker variant was substantially slower due to startup/serialization overhead. The optimization therefore **was not enabled** in production exports. [Exact parity and timing receipt](runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_parallel_export_check_v3/parity.json).

### CLIPZyme validation metric enforcement (2026-09-21T05:33:17.032055+00:00)

CLIPZyme/EnzymeMap validation reports **BEDROC85 ↑, BEDROC20 ↑, EF5 ↑ and EF10 ↑** against the full released screening library. The primary validation-selected checkpoint maximizes Table 1 BEDROC85; all four metrics are retained for both Table 1 (261,907 candidates) and Table 2 (252,113 candidates after training-ID exclusion). Fixed-epoch replications keep their predeclared budgets and do not select checkpoints or seeds. MRR is not used for this benchmark's validation or selection.

A fresh audit verified all six beta-5/beta-10 replication configurations: retrieval-MRR validation, small-pool early stopping and small-pool top-k checkpoint selection are disabled. Their full-library validation artifacts contain exactly the four screening metrics plus query/candidate counts. The checkpoint selector and watcher status now retain the complete metric quartet for both settings. The selector also rejects nonfinite metrics and a stored selection value inconsistent with BEDROC85. Historical artifacts and training recipes were not rewritten.

Validation: **7 tests passed**, including a conflicting-MRR/BEDROC ranking test, malformed-score rejection, the phase-2 no-MRR/no-validation-edge regression, and full-pool metric parity with ties and exclusions. The updated selector also accepted an actual full-library validation artifact. [Six-run audit and exact source hashes](runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_validation_metric_audit_20260921.json).

### Resumed shared-method search and corrected primary targets (2026-09-21T06:14:51.288832+00:00)

The primary competitor set is the main ReactZyme comparison table plus FGW-CLIP, and released CLIPZyme plus published FGW-CLIP for EnzymeMap. TIGER's main Reaction-Sim E→R MRR is **0.518**; the previously emphasized **0.543** belongs to its separate two-layer-MLP ablation. That stronger ablation remains disclosed but is not substituted for the main model. [Primary target registry](runs/generalization_20260919_2251/cross_paper_retraining/goal_primary_comparators_20260921.json).

The existing beta-5, epoch-10 SLEEC multiview checkpoints with a threefold inference fusion scalar and fixed 100-update phase 2 exceed every primary ReactZyme MRR target: Reaction-Sim **0.399995 / 0.523804**, Enzyme-Sim **0.667677 / 0.962493**, Time **0.540884 / 0.781538** (R→E / E→R). This candidate is now prospectively transferred to a new EnzymeMap training using the same seed 42, batch 512, 10 epochs, loss, optimizer, model and inference recipe; no EnzymeMap validation fallback or test-driven parameter change is allowed within that experiment. This remains exploratory development because the ReactZyme tests were already inspected. [Exact shared recipe](runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/protocol.json).

A separate four-GPU experiment starts the internal fused residue contribution at 0.3 instead of 0.1, with all four target datasets using fresh training, beta 5, batch 1,024 and 20 epochs. All model configurations match exactly. The configurable initialization preserves the default model and all other seed-matched initial tensors; 54 multiview/integration tests passed, including checkpoint/retrieval parity. [Fresh stronger-fusion protocol](runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/protocol.json).

The user requests multiple winning versions, each evaluated on both benchmarks and then Case 1. Case 1 is the restricted literature panel of **144 entries / 123 unique sequences**, with no RefSeq expansion. Every winning version will be frozen before new Case1 scoring and reported separately; Case1 is not used for fitting or selecting versions. Literature evidence tiers and duplicate-aware recovery will remain separate from measured-activity claims.

### Winning configuration: shared_recipe_beta5_b512_e10_fusion3_v2

Recorded 2026-09-21T07:38:56.694021+00:00. This configuration exceeds all **14 primary benchmark point estimates** (six ReactZyme cells and eight EnzymeMap screening cells). The comparison is exploratory after repeated benchmark inspection; it does not establish statistical superiority or control for differing pretrained resources.

**Architecture and training.** One dual encoder retains the frozen SLEEC scorer, a global protein view and four learned residue views. The compact reaction tower uses the same model configuration across targets. Each benchmark/split has its own freshly trained F3 weights and its own training-only semantic dictionary. Frozen protein/reaction encoder and SLEEC pretraining differ from competitors; matched downstream associations do not eliminate that pretraining difference.

**Stage 1.** Anchor-balanced all-positive decoupled InfoNCE, beta 5, equal R→E/E→R weights, seed 42, batch 512, 10-epoch checkpoint, AdamW learning rate 0.0001 and weight decay 0.01. Reaction-Sim recovered its optimizer state after a memory collision; it is not described as uninterrupted training.

**Stage 2.** A residual dual encoder fits the full target training graph for 100 positive-CE updates, temperature 0.2, identity weight 10 and learning rate 0.0001. The semantic dictionary uses training associations only. Inference uses residual cap **1.0**, semantic weight **0.25**, and a **3×** multiplier on the internal fused residue contribution. Endpoint embeddings remain independently encodable; there is no model ensemble, candidate-pool hubness correction, or RefSeq search.

**Protocol.** ReactZyme uses the official three target splits and all-positive MRR in both directions. EnzymeMap uses 34,427/7,287/4,642 original train/dev/test associations, with the released 261,907-ID screening pool; Table 2 excludes training enzyme IDs, leaving 252,113 candidates. Validation reports BEDROC85, BEDROC20, EF5 and EF10; MRR is disabled for EnzymeMap. This fixed candidate has no EnzymeMap validation fallback or seed selection. ReactZyme reaction inputs follow its participant-set representation; EnzymeMap retains physical reactant/product sides.

| Benchmark | Setting | Metric | Method ↑ | Comparison target | Difference |
| --- | --- | --- | ---: | ---: | ---: |
| ReactZyme | reaction_smi | enzyme_to_reaction | 0.523804 | 0.518000 | +0.005804 |
| ReactZyme | reaction_smi | reaction_to_enzyme | 0.399995 | 0.337000 | +0.062995 |
| ReactZyme | enzyme_smi | enzyme_to_reaction | 0.962493 | 0.956000 | +0.006493 |
| ReactZyme | enzyme_smi | reaction_to_enzyme | 0.667677 | 0.592000 | +0.075677 |
| ReactZyme | time | enzyme_to_reaction | 0.781538 | 0.690000 | +0.091538 |
| ReactZyme | time | reaction_to_enzyme | 0.540884 | 0.372000 | +0.168884 |
| EnzymeMap | table1 | bedroc85 | 0.577817 | 0.486600 | +0.091217 |
| EnzymeMap | table1 | bedroc20 | 0.755572 | 0.666900 | +0.088672 |
| EnzymeMap | table1 | ef0.05 | 16.811991 | 14.910000 | +1.901991 |
| EnzymeMap | table1 | ef0.1 | 8.909144 | 8.180000 | +0.729144 |
| EnzymeMap | table2 | bedroc85 | 0.536819 | 0.451400 | +0.085419 |
| EnzymeMap | table2 | bedroc20 | 0.727165 | 0.614300 | +0.112865 |
| EnzymeMap | table2 | ef0.05 | 16.355816 | 13.570000 | +2.785816 |
| EnzymeMap | table2 | ef0.1 | 8.736791 | 7.810000 | +0.926791 |

For Table-2 EF10, the comparison above uses the published CLIPZyme value **7.81** ([FGW-CLIP Table 2](https://arxiv.org/html/2512.08508v1#S5.T2)); the immutable original qualification used the locally reproduced **7.80813533427353**. This configuration exceeds both.

**Separate TIGER ablation context.** The primary main-model Reaction-Sim E→R target is 0.518. This configuration's 0.523804 MRR is below the two-layer-MLP ablation’s 0.543 and below the SwissProt+DGN condition’s 0.632. These are separate published conditions; this statement concerns that MRR cell only. [TIGER Tables 1–3](https://arxiv.org/html/2605.24489v1).

**Case 1.** Choices were frozen before scoring this version. All 144 literature-panel entries are retained, representing 123 unique sequences. The table below counts unique-sequence recovery; the companion CSV also ranks every original entry. Each target-trained checkpoint is reported independently, alongside its native encoder before phase 2.

| Target model / score | Paper catalysts @25 / 12 | Papers + patents @25 / 24 | Workbook actives @25 / 81 | Conditional AUROC |
| --- | ---: | ---: | ---: | ---: |
| reaction_smi/selected | 9 | 11 | 23 | 0.622869 |
| reaction_smi/native_before_phase2 | 9 | 9 | 21 | 0.600823 |
| enzyme_smi/selected | 5 | 8 | 20 | 0.576426 |
| enzyme_smi/native_before_phase2 | 5 | 8 | 20 | 0.594944 |
| time/selected | 5 | 10 | 20 | 0.606408 |
| time/native_before_phase2 | 5 | 8 | 19 | 0.635215 |
| enzymemap/selected | 6 | 6 | 19 | 0.469724 |
| enzymemap/native_before_phase2 | 6 | 6 | 19 | 0.471193 |

**Predeclared deployment result.** Reaction-Sim paper-catalyst recovery changes from **9/12 before phase 2 to 9/12 after phase 2** at 25. The other target-trained checkpoints are reported without substituting one based on Case1 performance.

**Early literature-catalyst recovery.** All three recorded cutoffs are shown to make screening-budget tradeoffs visible. The final column counts primary papers represented by at least one recovered catalyst; it is descriptive source coverage, not a count of independent validation experiments. These results do not alter the frozen model choices.

| Target model / score | Paper catalysts @5 / 12 | @10 / 12 | @25 / 12 | Primary papers represented @25 |
| --- | ---: | ---: | ---: | ---: |
| reaction_smi/selected | 0 | 4 | 9 | 6/8 |
| reaction_smi/native_before_phase2 | 1 | 3 | 9 | 6/8 |
| enzyme_smi/selected | 3 | 4 | 5 | 4/8 |
| enzyme_smi/native_before_phase2 | 3 | 4 | 5 | 4/8 |
| time/selected | 1 | 3 | 5 | 4/8 |
| time/native_before_phase2 | 1 | 4 | 5 | 4/8 |
| enzymemap/selected | 1 | 5 | 6 | 3/8 |
| enzymemap/native_before_phase2 | 1 | 4 | 6 | 3/8 |

**Requested 144-entry panel.** The same sequence can represent several literature entries; these counts are not independent confirmations. Rankings retain all entries and use the recorded deterministic tie order.

| Target model / score | Paper entries @25 / 15 | Papers + patents @25 / 36 | Workbook active entries @25 / 102 |
| --- | ---: | ---: | ---: |
| reaction_smi/selected | 11 | 11 | 23 |
| reaction_smi/native_before_phase2 | 10 | 10 | 23 |
| enzyme_smi/selected | 8 | 11 | 21 |
| enzyme_smi/native_before_phase2 | 8 | 11 | 21 |
| time/selected | 7 | 12 | 21 |
| time/native_before_phase2 | 8 | 11 | 21 |
| enzymemap/selected | 6 | 6 | 19 |
| enzymemap/native_before_phase2 | 6 | 6 | 19 |

**Reaction-Sim training homology.** Of 9 paper catalysts recovered at 25, 4 have a retrieved training hit with at least 90% sequence identity. The audit covers supervised Reaction-Sim training only, not frozen protein/SLEEC pretraining. Both query molecules also occur together within larger training participant sets. No qualifying MMseqs hit means unknown similarity, not established absence of a homolog. The panel therefore cannot establish distant catalytic generalization.

| Paper catalyst | Selected rank | Native rank | Best retrieved training identity |
| --- | ---: | ---: | ---: |
| H007 | 6 | 10 | unknown (no qualifying hit) |
| H012 | 7 | 5 | unknown (no qualifying hit) |
| H011 | 8 | 6 | unknown (no qualifying hit) |
| H005 | 10 | 16 | unknown (no qualifying hit) |
| H003 | 13 | 13 | 97.5% |
| H004 | 14 | 14 | 96.8% |
| H001 | 15 | 15 | 99.1% |
| H006 | 16 | 33 | unknown (no qualifying hit) |
| H002 | 20 | 18 | 98.1% |
| H008 | 26 | 28 | 50.2% |
| H009 | 28 | 24 | 50.0% |
| H013 | 52 | 66 | 44.4% |

Case 1 is one previously examined reaction, with dependent constructs and heterogeneous literature assays. The 42 non-detect-only sequences are conditional assay observations, not universally inactive enzymes. These results are retrospective evidence, not new wet-lab validation or proof of broad catalytic generalization. No model is selected by Case1 results.

**Frozen artifacts.**

- reaction_smi: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_multiview_beta5_v1/reaction_smi/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_multiview_beta5_v1/reaction_smi/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_phase2_epoch10_soft_v1/residue_views/training/step0100.pt).
- enzyme_smi: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/enzyme_smi/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/enzyme_smi/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_enzyme_smi_beta5_phase2_epoch10_soft_v1/residue_views/training/step0100.pt).
- time: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/time/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/time/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_time_beta5_phase2_epoch10_soft_v1/residue_views/training/step0100.pt).
- enzymemap: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/enzymemap_seed42/calibrated/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/enzymemap_seed42/calibrated/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/enzymemap_seed42/phase2/training/step0100.pt).

[Exact protocol](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/protocol.json) · [Benchmark source records](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/qualification.json) · [Case1 metrics](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/case1/summary.json) · [All 144 rankings](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/case1/all_144_entry_rankings.csv) · [123 unique-sequence rankings](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/case1/unique_sequence_rankings.csv).

### Winning configuration: shared_recipe_alpha04_cap1_v1

Recorded 2026-09-21T07:38:56.797125+00:00. This configuration exceeds all **14 primary benchmark point estimates** (six ReactZyme cells and eight EnzymeMap screening cells). The comparison is exploratory after repeated benchmark inspection; it does not establish statistical superiority or control for differing pretrained resources.

**Architecture and training.** One dual encoder retains the frozen SLEEC scorer, a global protein view and four learned residue views. The compact reaction tower uses the same model configuration across targets. Each benchmark/split has its own freshly trained F3 weights and its own training-only semantic dictionary. Frozen protein/reaction encoder and SLEEC pretraining differ from competitors; matched downstream associations do not eliminate that pretraining difference.

**Stage 1.** Anchor-balanced all-positive decoupled InfoNCE, beta 5, equal R→E/E→R weights, seed 42, batch 512, 10-epoch checkpoint, AdamW learning rate 0.0001 and weight decay 0.01. Reaction-Sim recovered its optimizer state after a memory collision; it is not described as uninterrupted training.

**Stage 2.** A residual dual encoder fits the full target training graph for 100 positive-CE updates, temperature 0.2, identity weight 10 and learning rate 0.0001. The semantic dictionary uses training associations only. Inference uses residual cap **1.0**, semantic weight **0.4**, and a **3×** multiplier on the internal fused residue contribution. Endpoint embeddings remain independently encodable; there is no model ensemble, candidate-pool hubness correction, or RefSeq search.

**Protocol.** ReactZyme uses the official three target splits and all-positive MRR in both directions. EnzymeMap uses 34,427/7,287/4,642 original train/dev/test associations, with the released 261,907-ID screening pool; Table 2 excludes training enzyme IDs, leaving 252,113 candidates. Validation reports BEDROC85, BEDROC20, EF5 and EF10; MRR is disabled for EnzymeMap. This fixed candidate has no EnzymeMap validation fallback or seed selection. ReactZyme reaction inputs follow its participant-set representation; EnzymeMap retains physical reactant/product sides.

| Benchmark | Setting | Metric | Method ↑ | Comparison target | Difference |
| --- | --- | --- | ---: | ---: | ---: |
| ReactZyme | reaction_smi | enzyme_to_reaction | 0.537749 | 0.518000 | +0.019749 |
| ReactZyme | reaction_smi | reaction_to_enzyme | 0.401450 | 0.337000 | +0.064450 |
| ReactZyme | enzyme_smi | enzyme_to_reaction | 0.963482 | 0.956000 | +0.007482 |
| ReactZyme | enzyme_smi | reaction_to_enzyme | 0.666656 | 0.592000 | +0.074656 |
| ReactZyme | time | enzyme_to_reaction | 0.782770 | 0.690000 | +0.092770 |
| ReactZyme | time | reaction_to_enzyme | 0.538339 | 0.372000 | +0.166339 |
| EnzymeMap | table1 | bedroc85 | 0.577945 | 0.486600 | +0.091345 |
| EnzymeMap | table1 | bedroc20 | 0.757111 | 0.666900 | +0.090211 |
| EnzymeMap | table1 | ef0.05 | 16.864295 | 14.910000 | +1.954295 |
| EnzymeMap | table1 | ef0.1 | 8.953017 | 8.180000 | +0.773017 |
| EnzymeMap | table2 | bedroc85 | 0.534129 | 0.451400 | +0.082729 |
| EnzymeMap | table2 | bedroc20 | 0.727833 | 0.614300 | +0.113533 |
| EnzymeMap | table2 | ef0.05 | 16.408538 | 13.570000 | +2.838538 |
| EnzymeMap | table2 | ef0.1 | 8.779122 | 7.810000 | +0.969122 |

For Table-2 EF10, the comparison above uses the published CLIPZyme value **7.81** ([FGW-CLIP Table 2](https://arxiv.org/html/2512.08508v1#S5.T2)); the immutable original qualification used the locally reproduced **7.80813533427353**. This configuration exceeds both.

**Separate TIGER ablation context.** The primary main-model Reaction-Sim E→R target is 0.518. This configuration's 0.537749 MRR is below the two-layer-MLP ablation’s 0.543 and below the SwissProt+DGN condition’s 0.632. These are separate published conditions; this statement concerns that MRR cell only. [TIGER Tables 1–3](https://arxiv.org/html/2605.24489v1).

**Case 1.** Choices were frozen before scoring this version. All 144 literature-panel entries are retained, representing 123 unique sequences. The table below counts unique-sequence recovery; the companion CSV also ranks every original entry. Each target-trained checkpoint is reported independently, alongside its native encoder before phase 2.

| Target model / score | Paper catalysts @25 / 12 | Papers + patents @25 / 24 | Workbook actives @25 / 81 | Conditional AUROC |
| --- | ---: | ---: | ---: | ---: |
| reaction_smi/selected | 9 | 11 | 23 | 0.623751 |
| reaction_smi/native_before_phase2 | 9 | 9 | 21 | 0.600823 |
| enzyme_smi/selected | 5 | 8 | 20 | 0.576426 |
| enzyme_smi/native_before_phase2 | 5 | 8 | 20 | 0.594944 |
| time/selected | 5 | 10 | 20 | 0.606408 |
| time/native_before_phase2 | 5 | 8 | 19 | 0.635215 |
| enzymemap/selected | 6 | 6 | 19 | 0.470899 |
| enzymemap/native_before_phase2 | 6 | 6 | 19 | 0.471193 |

**Predeclared deployment result.** Reaction-Sim paper-catalyst recovery changes from **9/12 before phase 2 to 9/12 after phase 2** at 25. The other target-trained checkpoints are reported without substituting one based on Case1 performance.

**Early literature-catalyst recovery.** All three recorded cutoffs are shown to make screening-budget tradeoffs visible. The final column counts primary papers represented by at least one recovered catalyst; it is descriptive source coverage, not a count of independent validation experiments. These results do not alter the frozen model choices.

| Target model / score | Paper catalysts @5 / 12 | @10 / 12 | @25 / 12 | Primary papers represented @25 |
| --- | ---: | ---: | ---: | ---: |
| reaction_smi/selected | 0 | 4 | 9 | 6/8 |
| reaction_smi/native_before_phase2 | 1 | 3 | 9 | 6/8 |
| enzyme_smi/selected | 3 | 4 | 5 | 4/8 |
| enzyme_smi/native_before_phase2 | 3 | 4 | 5 | 4/8 |
| time/selected | 1 | 3 | 5 | 4/8 |
| time/native_before_phase2 | 1 | 4 | 5 | 4/8 |
| enzymemap/selected | 1 | 5 | 6 | 3/8 |
| enzymemap/native_before_phase2 | 1 | 4 | 6 | 3/8 |

**Requested 144-entry panel.** The same sequence can represent several literature entries; these counts are not independent confirmations. Rankings retain all entries and use the recorded deterministic tie order.

| Target model / score | Paper entries @25 / 15 | Papers + patents @25 / 36 | Workbook active entries @25 / 102 |
| --- | ---: | ---: | ---: |
| reaction_smi/selected | 12 | 12 | 23 |
| reaction_smi/native_before_phase2 | 10 | 10 | 23 |
| enzyme_smi/selected | 8 | 11 | 21 |
| enzyme_smi/native_before_phase2 | 8 | 11 | 21 |
| time/selected | 7 | 12 | 21 |
| time/native_before_phase2 | 8 | 11 | 21 |
| enzymemap/selected | 6 | 6 | 19 |
| enzymemap/native_before_phase2 | 6 | 6 | 19 |

**Reaction-Sim training homology.** Of 9 paper catalysts recovered at 25, 4 have a retrieved training hit with at least 90% sequence identity. The audit covers supervised Reaction-Sim training only, not frozen protein/SLEEC pretraining. Both query molecules also occur together within larger training participant sets. No qualifying MMseqs hit means unknown similarity, not established absence of a homolog. The panel therefore cannot establish distant catalytic generalization.

| Paper catalyst | Selected rank | Native rank | Best retrieved training identity |
| --- | ---: | ---: | ---: |
| H007 | 6 | 10 | unknown (no qualifying hit) |
| H012 | 7 | 5 | unknown (no qualifying hit) |
| H011 | 8 | 6 | unknown (no qualifying hit) |
| H005 | 10 | 16 | unknown (no qualifying hit) |
| H003 | 13 | 13 | 97.5% |
| H004 | 14 | 14 | 96.8% |
| H001 | 15 | 15 | 99.1% |
| H006 | 17 | 33 | unknown (no qualifying hit) |
| H002 | 19 | 18 | 98.1% |
| H008 | 26 | 28 | 50.2% |
| H009 | 28 | 24 | 50.0% |
| H013 | 52 | 66 | 44.4% |

Case 1 is one previously examined reaction, with dependent constructs and heterogeneous literature assays. The 42 non-detect-only sequences are conditional assay observations, not universally inactive enzymes. These results are retrospective evidence, not new wet-lab validation or proof of broad catalytic generalization. No model is selected by Case1 results.

**Frozen artifacts.**

- reaction_smi: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_multiview_beta5_v1/reaction_smi/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_multiview_beta5_v1/reaction_smi/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_phase2_epoch10_soft_v1/residue_views/training/step0100.pt).
- enzyme_smi: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/enzyme_smi/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/enzyme_smi/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_enzyme_smi_beta5_phase2_epoch10_soft_v1/residue_views/training/step0100.pt).
- time: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/time/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/time/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_time_beta5_phase2_epoch10_soft_v1/residue_views/training/step0100.pt).
- enzymemap: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/enzymemap_seed42/calibrated/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/enzymemap_seed42/calibrated/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/enzymemap_seed42/phase2/training/step0100.pt).

[Exact protocol](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_alpha04_cap1_v1/protocol.json) · [Benchmark source records](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_alpha04_cap1_v1/qualification.json) · [Case1 metrics](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_alpha04_cap1_v1/case1/summary.json) · [All 144 rankings](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_alpha04_cap1_v1/case1/all_144_entry_rankings.csv) · [123 unique-sequence rankings](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_alpha04_cap1_v1/case1/unique_sequence_rankings.csv).

### Winning configuration: shared_recipe_alpha05_cap1_v1

Recorded 2026-09-21T07:38:56.903976+00:00. This configuration exceeds all **14 primary benchmark point estimates** (six ReactZyme cells and eight EnzymeMap screening cells). The comparison is exploratory after repeated benchmark inspection; it does not establish statistical superiority or control for differing pretrained resources.

**Architecture and training.** One dual encoder retains the frozen SLEEC scorer, a global protein view and four learned residue views. The compact reaction tower uses the same model configuration across targets. Each benchmark/split has its own freshly trained F3 weights and its own training-only semantic dictionary. Frozen protein/reaction encoder and SLEEC pretraining differ from competitors; matched downstream associations do not eliminate that pretraining difference.

**Stage 1.** Anchor-balanced all-positive decoupled InfoNCE, beta 5, equal R→E/E→R weights, seed 42, batch 512, 10-epoch checkpoint, AdamW learning rate 0.0001 and weight decay 0.01. Reaction-Sim recovered its optimizer state after a memory collision; it is not described as uninterrupted training.

**Stage 2.** A residual dual encoder fits the full target training graph for 100 positive-CE updates, temperature 0.2, identity weight 10 and learning rate 0.0001. The semantic dictionary uses training associations only. Inference uses residual cap **1.0**, semantic weight **0.5**, and a **3×** multiplier on the internal fused residue contribution. Endpoint embeddings remain independently encodable; there is no model ensemble, candidate-pool hubness correction, or RefSeq search.

**Protocol.** ReactZyme uses the official three target splits and all-positive MRR in both directions. EnzymeMap uses 34,427/7,287/4,642 original train/dev/test associations, with the released 261,907-ID screening pool; Table 2 excludes training enzyme IDs, leaving 252,113 candidates. Validation reports BEDROC85, BEDROC20, EF5 and EF10; MRR is disabled for EnzymeMap. This fixed candidate has no EnzymeMap validation fallback or seed selection. ReactZyme reaction inputs follow its participant-set representation; EnzymeMap retains physical reactant/product sides.

| Benchmark | Setting | Metric | Method ↑ | Comparison target | Difference |
| --- | --- | --- | ---: | ---: | ---: |
| ReactZyme | reaction_smi | enzyme_to_reaction | 0.537550 | 0.518000 | +0.019550 |
| ReactZyme | reaction_smi | reaction_to_enzyme | 0.394653 | 0.337000 | +0.057653 |
| ReactZyme | enzyme_smi | enzyme_to_reaction | 0.961178 | 0.956000 | +0.005178 |
| ReactZyme | enzyme_smi | reaction_to_enzyme | 0.662794 | 0.592000 | +0.070794 |
| ReactZyme | time | enzyme_to_reaction | 0.774713 | 0.690000 | +0.084713 |
| ReactZyme | time | reaction_to_enzyme | 0.531367 | 0.372000 | +0.159367 |
| EnzymeMap | table1 | bedroc85 | 0.574570 | 0.486600 | +0.087970 |
| EnzymeMap | table1 | bedroc20 | 0.757460 | 0.666900 | +0.090560 |
| EnzymeMap | table1 | ef0.05 | 16.935198 | 14.910000 | +2.025198 |
| EnzymeMap | table1 | ef0.1 | 9.001075 | 8.180000 | +0.821075 |
| EnzymeMap | table2 | bedroc85 | 0.529326 | 0.451400 | +0.077926 |
| EnzymeMap | table2 | bedroc20 | 0.727885 | 0.614300 | +0.113585 |
| EnzymeMap | table2 | ef0.05 | 16.488139 | 13.570000 | +2.918139 |
| EnzymeMap | table2 | ef0.1 | 8.826313 | 7.810000 | +1.016313 |

For Table-2 EF10, the comparison above uses the published CLIPZyme value **7.81** ([FGW-CLIP Table 2](https://arxiv.org/html/2512.08508v1#S5.T2)); the immutable original qualification used the locally reproduced **7.80813533427353**. This configuration exceeds both.

**Separate TIGER ablation context.** The primary main-model Reaction-Sim E→R target is 0.518. This configuration's 0.537550 MRR is below the two-layer-MLP ablation’s 0.543 and below the SwissProt+DGN condition’s 0.632. These are separate published conditions; this statement concerns that MRR cell only. [TIGER Tables 1–3](https://arxiv.org/html/2605.24489v1).

**Case 1.** Choices were frozen before scoring this version. All 144 literature-panel entries are retained, representing 123 unique sequences. The table below counts unique-sequence recovery; the companion CSV also ranks every original entry. Each target-trained checkpoint is reported independently, alongside its native encoder before phase 2.

| Target model / score | Paper catalysts @25 / 12 | Papers + patents @25 / 24 | Workbook actives @25 / 81 | Conditional AUROC |
| --- | ---: | ---: | ---: | ---: |
| reaction_smi/selected | 9 | 11 | 23 | 0.624633 |
| reaction_smi/native_before_phase2 | 9 | 9 | 21 | 0.600823 |
| enzyme_smi/selected | 5 | 8 | 20 | 0.576132 |
| enzyme_smi/native_before_phase2 | 5 | 8 | 20 | 0.594944 |
| time/selected | 5 | 10 | 20 | 0.606408 |
| time/native_before_phase2 | 5 | 8 | 19 | 0.635215 |
| enzymemap/selected | 6 | 6 | 19 | 0.470606 |
| enzymemap/native_before_phase2 | 6 | 6 | 19 | 0.471193 |

**Predeclared deployment result.** Reaction-Sim paper-catalyst recovery changes from **9/12 before phase 2 to 9/12 after phase 2** at 25. The other target-trained checkpoints are reported without substituting one based on Case1 performance.

**Early literature-catalyst recovery.** All three recorded cutoffs are shown to make screening-budget tradeoffs visible. The final column counts primary papers represented by at least one recovered catalyst; it is descriptive source coverage, not a count of independent validation experiments. These results do not alter the frozen model choices.

| Target model / score | Paper catalysts @5 / 12 | @10 / 12 | @25 / 12 | Primary papers represented @25 |
| --- | ---: | ---: | ---: | ---: |
| reaction_smi/selected | 0 | 4 | 9 | 6/8 |
| reaction_smi/native_before_phase2 | 1 | 3 | 9 | 6/8 |
| enzyme_smi/selected | 3 | 4 | 5 | 4/8 |
| enzyme_smi/native_before_phase2 | 3 | 4 | 5 | 4/8 |
| time/selected | 1 | 3 | 5 | 4/8 |
| time/native_before_phase2 | 1 | 4 | 5 | 4/8 |
| enzymemap/selected | 1 | 5 | 6 | 3/8 |
| enzymemap/native_before_phase2 | 1 | 4 | 6 | 3/8 |

**Requested 144-entry panel.** The same sequence can represent several literature entries; these counts are not independent confirmations. Rankings retain all entries and use the recorded deterministic tie order.

| Target model / score | Paper entries @25 / 15 | Papers + patents @25 / 36 | Workbook active entries @25 / 102 |
| --- | ---: | ---: | ---: |
| reaction_smi/selected | 12 | 12 | 23 |
| reaction_smi/native_before_phase2 | 10 | 10 | 23 |
| enzyme_smi/selected | 8 | 11 | 21 |
| enzyme_smi/native_before_phase2 | 8 | 11 | 21 |
| time/selected | 7 | 12 | 21 |
| time/native_before_phase2 | 8 | 11 | 21 |
| enzymemap/selected | 6 | 6 | 19 |
| enzymemap/native_before_phase2 | 6 | 6 | 19 |

**Reaction-Sim training homology.** Of 9 paper catalysts recovered at 25, 4 have a retrieved training hit with at least 90% sequence identity. The audit covers supervised Reaction-Sim training only, not frozen protein/SLEEC pretraining. Both query molecules also occur together within larger training participant sets. No qualifying MMseqs hit means unknown similarity, not established absence of a homolog. The panel therefore cannot establish distant catalytic generalization.

| Paper catalyst | Selected rank | Native rank | Best retrieved training identity |
| --- | ---: | ---: | ---: |
| H007 | 6 | 10 | unknown (no qualifying hit) |
| H012 | 7 | 5 | unknown (no qualifying hit) |
| H011 | 8 | 6 | unknown (no qualifying hit) |
| H005 | 10 | 16 | unknown (no qualifying hit) |
| H003 | 13 | 13 | 97.5% |
| H004 | 14 | 14 | 96.8% |
| H001 | 15 | 15 | 99.1% |
| H006 | 17 | 33 | unknown (no qualifying hit) |
| H002 | 19 | 18 | 98.1% |
| H008 | 26 | 28 | 50.2% |
| H009 | 28 | 24 | 50.0% |
| H013 | 52 | 66 | 44.4% |

Case 1 is one previously examined reaction, with dependent constructs and heterogeneous literature assays. The 42 non-detect-only sequences are conditional assay observations, not universally inactive enzymes. These results are retrospective evidence, not new wet-lab validation or proof of broad catalytic generalization. No model is selected by Case1 results.

**Frozen artifacts.**

- reaction_smi: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_multiview_beta5_v1/reaction_smi/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_multiview_beta5_v1/reaction_smi/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_phase2_epoch10_soft_v1/residue_views/training/step0100.pt).
- enzyme_smi: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/enzyme_smi/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/enzyme_smi/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_enzyme_smi_beta5_phase2_epoch10_soft_v1/residue_views/training/step0100.pt).
- time: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/time/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/time/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_time_beta5_phase2_epoch10_soft_v1/residue_views/training/step0100.pt).
- enzymemap: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/enzymemap_seed42/calibrated/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/enzymemap_seed42/calibrated/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/enzymemap_seed42/phase2/training/step0100.pt).

[Exact protocol](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_alpha05_cap1_v1/protocol.json) · [Benchmark source records](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_alpha05_cap1_v1/qualification.json) · [Case1 metrics](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_alpha05_cap1_v1/case1/summary.json) · [All 144 rankings](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_alpha05_cap1_v1/case1/all_144_entry_rankings.csv) · [123 unique-sequence rankings](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_alpha05_cap1_v1/case1/unique_sequence_rankings.csv).

### Winning configuration: shared_recipe_alpha04_cap05_v1

Recorded 2026-09-21T07:38:57.013223+00:00. This configuration exceeds all **14 primary benchmark point estimates** (six ReactZyme cells and eight EnzymeMap screening cells). The comparison is exploratory after repeated benchmark inspection; it does not establish statistical superiority or control for differing pretrained resources.

**Architecture and training.** One dual encoder retains the frozen SLEEC scorer, a global protein view and four learned residue views. The compact reaction tower uses the same model configuration across targets. Each benchmark/split has its own freshly trained F3 weights and its own training-only semantic dictionary. Frozen protein/reaction encoder and SLEEC pretraining differ from competitors; matched downstream associations do not eliminate that pretraining difference.

**Stage 1.** Anchor-balanced all-positive decoupled InfoNCE, beta 5, equal R→E/E→R weights, seed 42, batch 512, 10-epoch checkpoint, AdamW learning rate 0.0001 and weight decay 0.01. Reaction-Sim recovered its optimizer state after a memory collision; it is not described as uninterrupted training.

**Stage 2.** A residual dual encoder fits the full target training graph for 100 positive-CE updates, temperature 0.2, identity weight 10 and learning rate 0.0001. The semantic dictionary uses training associations only. Inference uses residual cap **0.5**, semantic weight **0.4**, and a **3×** multiplier on the internal fused residue contribution. Endpoint embeddings remain independently encodable; there is no model ensemble, candidate-pool hubness correction, or RefSeq search.

**Protocol.** ReactZyme uses the official three target splits and all-positive MRR in both directions. EnzymeMap uses 34,427/7,287/4,642 original train/dev/test associations, with the released 261,907-ID screening pool; Table 2 excludes training enzyme IDs, leaving 252,113 candidates. Validation reports BEDROC85, BEDROC20, EF5 and EF10; MRR is disabled for EnzymeMap. This fixed candidate has no EnzymeMap validation fallback or seed selection. ReactZyme reaction inputs follow its participant-set representation; EnzymeMap retains physical reactant/product sides.

| Benchmark | Setting | Metric | Method ↑ | Comparison target | Difference |
| --- | --- | --- | ---: | ---: | ---: |
| ReactZyme | reaction_smi | enzyme_to_reaction | 0.534366 | 0.518000 | +0.016366 |
| ReactZyme | reaction_smi | reaction_to_enzyme | 0.400381 | 0.337000 | +0.063381 |
| ReactZyme | enzyme_smi | enzyme_to_reaction | 0.971355 | 0.956000 | +0.015355 |
| ReactZyme | enzyme_smi | reaction_to_enzyme | 0.666249 | 0.592000 | +0.074249 |
| ReactZyme | time | enzyme_to_reaction | 0.780669 | 0.690000 | +0.090669 |
| ReactZyme | time | reaction_to_enzyme | 0.538285 | 0.372000 | +0.166285 |
| EnzymeMap | table1 | bedroc85 | 0.572337 | 0.486600 | +0.085737 |
| EnzymeMap | table1 | bedroc20 | 0.755433 | 0.666900 | +0.088533 |
| EnzymeMap | table1 | ef0.05 | 16.859042 | 14.910000 | +1.949042 |
| EnzymeMap | table1 | ef0.1 | 8.958692 | 8.180000 | +0.778692 |
| EnzymeMap | table2 | bedroc85 | 0.528734 | 0.451400 | +0.077334 |
| EnzymeMap | table2 | bedroc20 | 0.726403 | 0.614300 | +0.112103 |
| EnzymeMap | table2 | ef0.05 | 16.418633 | 13.570000 | +2.848633 |
| EnzymeMap | table2 | ef0.1 | 8.793103 | 7.810000 | +0.983103 |

For Table-2 EF10, the comparison above uses the published CLIPZyme value **7.81** ([FGW-CLIP Table 2](https://arxiv.org/html/2512.08508v1#S5.T2)); the immutable original qualification used the locally reproduced **7.80813533427353**. This configuration exceeds both.

**Separate TIGER ablation context.** The primary main-model Reaction-Sim E→R target is 0.518. This configuration's 0.534366 MRR is below the two-layer-MLP ablation’s 0.543 and below the SwissProt+DGN condition’s 0.632. These are separate published conditions; this statement concerns that MRR cell only. [TIGER Tables 1–3](https://arxiv.org/html/2605.24489v1).

**Case 1.** Choices were frozen before scoring this version. All 144 literature-panel entries are retained, representing 123 unique sequences. The table below counts unique-sequence recovery; the companion CSV also ranks every original entry. Each target-trained checkpoint is reported independently, alongside its native encoder before phase 2.

| Target model / score | Paper catalysts @25 / 12 | Papers + patents @25 / 24 | Workbook actives @25 / 81 | Conditional AUROC |
| --- | ---: | ---: | ---: | ---: |
| reaction_smi/selected | 10 | 10 | 22 | 0.612875 |
| reaction_smi/native_before_phase2 | 9 | 9 | 21 | 0.600823 |
| enzyme_smi/selected | 5 | 8 | 20 | 0.584656 |
| enzyme_smi/native_before_phase2 | 5 | 8 | 20 | 0.594944 |
| time/selected | 5 | 10 | 20 | 0.634333 |
| time/native_before_phase2 | 5 | 8 | 19 | 0.635215 |
| enzymemap/selected | 6 | 6 | 19 | 0.470606 |
| enzymemap/native_before_phase2 | 6 | 6 | 19 | 0.471193 |

**Predeclared deployment result.** Reaction-Sim paper-catalyst recovery changes from **9/12 before phase 2 to 10/12 after phase 2** at 25. The other target-trained checkpoints are reported without substituting one based on Case1 performance.

**Early literature-catalyst recovery.** All three recorded cutoffs are shown to make screening-budget tradeoffs visible. The final column counts primary papers represented by at least one recovered catalyst; it is descriptive source coverage, not a count of independent validation experiments. These results do not alter the frozen model choices.

| Target model / score | Paper catalysts @5 / 12 | @10 / 12 | @25 / 12 | Primary papers represented @25 |
| --- | ---: | ---: | ---: | ---: |
| reaction_smi/selected | 0 | 3 | 10 | 7/8 |
| reaction_smi/native_before_phase2 | 1 | 3 | 9 | 6/8 |
| enzyme_smi/selected | 3 | 4 | 5 | 4/8 |
| enzyme_smi/native_before_phase2 | 3 | 4 | 5 | 4/8 |
| time/selected | 1 | 4 | 5 | 4/8 |
| time/native_before_phase2 | 1 | 4 | 5 | 4/8 |
| enzymemap/selected | 1 | 4 | 6 | 3/8 |
| enzymemap/native_before_phase2 | 1 | 4 | 6 | 3/8 |

**Requested 144-entry panel.** The same sequence can represent several literature entries; these counts are not independent confirmations. Rankings retain all entries and use the recorded deterministic tie order.

| Target model / score | Paper entries @25 / 15 | Papers + patents @25 / 36 | Workbook active entries @25 / 102 |
| --- | ---: | ---: | ---: |
| reaction_smi/selected | 10 | 10 | 23 |
| reaction_smi/native_before_phase2 | 10 | 10 | 23 |
| enzyme_smi/selected | 8 | 11 | 21 |
| enzyme_smi/native_before_phase2 | 8 | 11 | 21 |
| time/selected | 7 | 12 | 21 |
| time/native_before_phase2 | 8 | 11 | 21 |
| enzymemap/selected | 6 | 6 | 19 |
| enzymemap/native_before_phase2 | 6 | 6 | 19 |

**Reaction-Sim training homology.** Of 10 paper catalysts recovered at 25, 4 have a retrieved training hit with at least 90% sequence identity. The audit covers supervised Reaction-Sim training only, not frozen protein/SLEEC pretraining. Both query molecules also occur together within larger training participant sets. No qualifying MMseqs hit means unknown similarity, not established absence of a homolog. The panel therefore cannot establish distant catalytic generalization.

| Paper catalyst | Selected rank | Native rank | Best retrieved training identity |
| --- | ---: | ---: | ---: |
| H012 | 6 | 5 | unknown (no qualifying hit) |
| H011 | 7 | 6 | unknown (no qualifying hit) |
| H007 | 8 | 10 | unknown (no qualifying hit) |
| H003 | 11 | 13 | 97.5% |
| H005 | 12 | 16 | unknown (no qualifying hit) |
| H004 | 13 | 14 | 96.8% |
| H001 | 15 | 15 | 99.1% |
| H002 | 18 | 18 | 98.1% |
| H006 | 23 | 33 | unknown (no qualifying hit) |
| H009 | 24 | 24 | 50.0% |
| H008 | 26 | 28 | 50.2% |
| H013 | 62 | 66 | 44.4% |

Case 1 is one previously examined reaction, with dependent constructs and heterogeneous literature assays. The 42 non-detect-only sequences are conditional assay observations, not universally inactive enzymes. These results are retrospective evidence, not new wet-lab validation or proof of broad catalytic generalization. No model is selected by Case1 results.

**Frozen artifacts.**

- reaction_smi: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_multiview_beta5_v1/reaction_smi/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_multiview_beta5_v1/reaction_smi/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_phase2_epoch10_soft_v1/residue_views/training/step0100.pt).
- enzyme_smi: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/enzyme_smi/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/enzyme_smi/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_enzyme_smi_beta5_phase2_epoch10_soft_v1/residue_views/training/step0100.pt).
- time: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/time/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/time/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_time_beta5_phase2_epoch10_soft_v1/residue_views/training/step0100.pt).
- enzymemap: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/enzymemap_seed42/calibrated/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/enzymemap_seed42/calibrated/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/enzymemap_seed42/phase2/training/step0100.pt).

[Exact protocol](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_alpha04_cap05_v1/protocol.json) · [Benchmark source records](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_alpha04_cap05_v1/qualification.json) · [Case1 metrics](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_alpha04_cap05_v1/case1/summary.json) · [All 144 rankings](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_alpha04_cap05_v1/case1/all_144_entry_rankings.csv) · [123 unique-sequence rankings](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_alpha04_cap05_v1/case1/unique_sequence_rankings.csv).

### 2026-09-21: ranking-augmented phase 2 did not win both benchmarks

Added SmoothAP ranking weight 1, temperature 0.05 and 128 sampled anchors to the fixed 100-update positive-CE phase-2 objective. Identity weight 10, contrastive temperature 0.2, residual cap 1 and semantic alpha 0.25 were held fixed. Fresh beta-5 F3 sources were ReactZyme epoch 10 / batch 512 and EnzymeMap epoch 18 / batch 2048; thus this pilot was not an exact shared base-training recipe. No Case1 test was promoted from this failed joint candidate.

| ReactZyme split | R→E MRR | E→R MRR |
| --- | ---: | ---: |
| reaction_smi | 0.387496 | 0.511017 |
| enzyme_smi | 0.664028 | 0.968344 |
| time | 0.531834 | 0.784674 |

Reaction-Sim E→R **0.511017** remains below TIGER’s primary **0.518** threshold. This fails the joint criterion even though EnzymeMap exceeds all eight primary targets.

| EnzymeMap setting | BEDROC85 | BEDROC20 | EF5 | EF10 |
| --- | ---: | ---: | ---: | ---: |
| table1 | 0.559381 | 0.734138 | 16.294853 | 8.767176 |
| table2 | 0.521203 | 0.705270 | 15.795259 | 8.583583 |

[Frozen protocol](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_phase2_ranking_v1/protocol.json) · [EnzymeMap test](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_phase2_ranking_v1/enzymemap/composition_v1/test_evaluation/summary.json).

### 2026-09-21: inference fusion is not a universal improvement across EnzymeMap seeds

The separate beta-5 / batch-2048 / epoch-18 replication tested a 3× internal fusion contribution against the unchanged 1× parent. The predeclared rule required improved validation Table-1 BEDROC85 and non-decreasing BEDROC20, EF5 and EF10. Seeds 42 and 17 retained 1×; seed 73 selected 3×. This does not contradict the newer batch-512 / epoch-10 shared recipe: the checkpoints and training trajectory differ.

| Seed | Selected multiplier | Validation BEDROC85, 1× | Validation BEDROC85, 3× |
| --- | ---: | ---: | ---: |
| 42 | 1.0 | 0.539268 | 0.537652 |
| 17 | 1.0 | 0.539241 | 0.533292 |
| 73 | 3.0 | 0.521110 | 0.528342 |

Seed 73’s selected test results (BEDROC85 / BEDROC20 / EF5 / EF10):
- table1: **0.496553 / 0.671109 / 14.959539 / 8.192457**.
- table2: **0.449594 / 0.633793 / 14.335832 / 7.918422**.

Table-2 BEDROC85 **0.449594** still misses the primary **0.451400** target. This is not an eight-cell winner. The other two seeds reuse their unchanged, hashed reference test results. Failed import and lineage-check attempts were preserved; resumed evaluation required an explicit receipt proving that only the scalar fusion parameter differed from the phase-2 training checkpoint.
[Selection and completed results](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/enzymemap_multiview_fusion_transfer_v1).

### 2026-09-21: stronger-fusion fresh-training follow-up

Four benchmark-specific fresh F3 fits use initial residual scale 0.3, beta 5, batch 1024, seed 42 and a maximum of 20 epochs. The predeclared 5/10/15/20-epoch snapshots each receive the fixed 100-update positive-CE phase 2, residual cap 1 and semantic alpha 0.25. No inference fusion multiplier is applied. ReactZyme selects on the mean of its two directions and seen/unseen validation reaction groups; EnzymeMap selects on full-library Table-1 BEDROC85 while reporting all four BEDROC/EF metrics in both screening settings. One selected test is run per target after that validation grid completes. These runs are still in progress and are not among the four completed winners.
[Protocol](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/protocol.json) · [Follower implementation](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/scripts/generalization_fresh_phase2_follow.py).

### 2026-09-21: completed shared-winner evidence audit

The four completed versions pass a direct audit of all 14 metric cells against their source results, frozen protocol and Case1 hashes, identical model/base-loss/optimizer settings across all four target models, and identical phase-2 optimization parameters. The phase-2 positive edge sets equal the unique target training CSV edges exactly: 147,393 Reaction-Sim, 152,748 Enzyme-Sim, 149,554 Time, and 34,180 internal EnzymeMap edges from 34,427 original rows. The semantic dictionaries contain exactly the target-training enzyme and reaction IDs. Archived phase-2 source/model hashes agree with the trained checkpoints. All 144 entry rankings and 123 sequence rankings are present for every target's selected and native score modes, and each detailed report is present in this findings file.

This verifies downstream association matching and reported artifacts. It does not erase the different pretrained feature resources: the retained frozen SLEEC run uses M-CSA/CATH residue supervision and pseudo-label training, as recorded in its run manifest. These comparisons must not be described as isolating the architecture under identical pretraining resources. The variants share the same fitted weights and seed, and the repeated-test and restricted-Case1 qualifications remain unchanged.

[Machine-readable evidence audit](runs/generalization_20260919_2251/cross_paper_retraining/shared_winners_evidence_audit_20260921.json) · [Reproducible audit script](scripts/generalization_shared_winners_audit.py) · [SLEEC pretraining manifest](checkpoints/SLEEC/sleec_stage1_prott5_uniref90_msa_4gpu_20260601_235157/run_manifest.json).

### 2026-09-21: stronger-fusion EnzymeMap selected test completed

The predeclared 5/10/15/20-epoch validation grid selected epoch 15. Initial fused residue scale 0.3, beta 5, batch 1024, fixed positive-CE phase 2 (100 updates), alpha 0.25 and cap 1 produce the following completed EnzymeMap tests. All eight primary targets, including the published 7.81 Table-2 EF10 value, are exceeded. ReactZyme tests for this validation-selected study remain pending; it is not yet a shared winner.

| Setting | BEDROC85 | BEDROC20 | EF5 | EF10 |
| --- | ---: | ---: | ---: | ---: |
| table1 | 0.545234 | 0.717676 | 15.817309 | 8.685418 |
| table2 | 0.502070 | 0.685032 | 15.238332 | 8.469625 |

[Validation selection](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/enzymemap/phase2_followup/selection.json) · [Test scores and metrics](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/enzymemap/phase2_followup/epoch15/composition_v1/test_evaluation/summary.json).

An additional fixed epoch-10 test of all four target models was declared after inspecting epoch-10 validation and this EnzymeMap epoch-15 test. It uses the exact same epoch and hyperparameters on every target, is reported without fallback, and does not change the separate validation-selected study. It is explicitly exploratory. The now-free GPU 3 performs these tests while ReactZyme training continues.
[Fixed epoch-10 protocol](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_epoch10_fixed_test_v1/protocol.json).

### Fresh stronger-fusion study: shared_fusion03_beta5_b1024_epoch10_fixed_test_v1

Completed 2026-09-21T07:07:03.976678+00:00. **14/14** primary benchmark targets exceeded. Joint winner.

The fixed epoch-10 snapshots of four fresh target-specific F3 fits use initial fused contribution 0.3, beta 5.0, batch 1024 and seed 42. Phase 2 uses 100 positive-CE updates, temperature 0.2, identity weight 10, semantic alpha 0.25 and residual cap 1. All four targets use the same fixed epoch; no checkpoint or multiplier selection. This intermediate test was declared after the epoch-10 validation and EnzymeMap epoch-15 test had been inspected; it is exploratory and does not alter the separate predeclared 5/10/15/20-epoch validation selector.

Selected epochs: reaction_smi=10, enzyme_smi=10, time=10, enzymemap=10.

| Benchmark | Setting | Metric | Method | Strongest primary target | Pass |
| --- | --- | --- | ---: | ---: | --- |
| ReactZyme | reaction_smi | enzyme_to_reaction | 0.524770 | 0.518000 | yes |
| ReactZyme | reaction_smi | reaction_to_enzyme | 0.393927 | 0.337000 | yes |
| ReactZyme | enzyme_smi | enzyme_to_reaction | 0.967982 | 0.956000 | yes |
| ReactZyme | enzyme_smi | reaction_to_enzyme | 0.662023 | 0.592000 | yes |
| ReactZyme | time | enzyme_to_reaction | 0.775257 | 0.690000 | yes |
| ReactZyme | time | reaction_to_enzyme | 0.527831 | 0.372000 | yes |
| EnzymeMap | table1 | bedroc85 | 0.547147 | 0.486600 | yes |
| EnzymeMap | table1 | bedroc20 | 0.708884 | 0.666900 | yes |
| EnzymeMap | table1 | ef0.05 | 15.457319 | 14.910000 | yes |
| EnzymeMap | table1 | ef0.1 | 8.433043 | 8.180000 | yes |
| EnzymeMap | table2 | bedroc85 | 0.507029 | 0.451400 | yes |
| EnzymeMap | table2 | bedroc20 | 0.675522 | 0.614300 | yes |
| EnzymeMap | table2 | ef0.05 | 14.840457 | 13.570000 | yes |
| EnzymeMap | table2 | ef0.1 | 8.175664 | 7.808135 | yes |

[Protocol](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_epoch10_fixed_test_v1/protocol.json) · [Exact source records](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_epoch10_fixed_test_v1/qualification.json).

### Winning configuration: shared_fusion03_beta5_b1024_epoch10_fixed_test_v1

Recorded 2026-09-21T07:38:57.118696+00:00. This configuration exceeds all **14 primary benchmark point estimates** (six ReactZyme cells and eight EnzymeMap screening cells). The comparison is exploratory after repeated benchmark inspection; it does not establish statistical superiority or control for differing pretrained resources.

**Architecture and training.** One dual encoder retains the frozen SLEEC scorer, a global protein view and four learned residue views. The compact reaction tower uses the same model configuration across targets. Each benchmark/split has its own freshly trained F3 weights and its own training-only semantic dictionary. Frozen protein/reaction encoder and SLEEC pretraining differ from competitors; matched downstream associations do not eliminate that pretraining difference.

**Stage 1.** Anchor-balanced all-positive decoupled InfoNCE, beta 5.0, equal R→E/E→R weights, seed 42, batch 1024, fixed epoch 10 snapshot and initial fused residue scale 0.3. AdamW learning rate 0.0001 and weight decay 0.01. Snapshots come from independent fresh target fits with constant learning rate; later training cannot alter these saved weights.

**Stage 2.** A residual dual encoder fits the full target training graph for 100 positive-CE updates, temperature 0.2, identity weight 10 and learning rate 0.0001. The semantic dictionary uses training associations only. Inference uses residual cap **1.0**, semantic weight **0.25**, and a **1×** multiplier on the internal fused residue contribution. Endpoint embeddings remain independently encodable; there is no model ensemble, candidate-pool hubness correction, or RefSeq search.

**Protocol.** ReactZyme uses the official three target splits and all-positive MRR in both directions. EnzymeMap uses 34,427/7,287/4,642 original train/dev/test associations, with the released 261,907-ID screening pool; Table 2 excludes training enzyme IDs, leaving 252,113 candidates. Validation reports BEDROC85, BEDROC20, EF5 and EF10; MRR is disabled for EnzymeMap. All four targets use the same fixed epoch, with no fallback within this evaluation. This epoch-10 experiment was declared after inspecting epoch-10 validation and the separate EnzymeMap epoch-15 test. It is an exploratory follow-up, not an independent prospective confirmation. It does not alter the separate predeclared 5/10/15/20-epoch validation selector. ReactZyme reaction inputs follow its participant-set representation; EnzymeMap retains physical reactant/product sides.

| Benchmark | Setting | Metric | Method ↑ | Comparison target | Difference |
| --- | --- | --- | ---: | ---: | ---: |
| ReactZyme | reaction_smi | enzyme_to_reaction | 0.524770 | 0.518000 | +0.006770 |
| ReactZyme | reaction_smi | reaction_to_enzyme | 0.393927 | 0.337000 | +0.056927 |
| ReactZyme | enzyme_smi | enzyme_to_reaction | 0.967982 | 0.956000 | +0.011982 |
| ReactZyme | enzyme_smi | reaction_to_enzyme | 0.662023 | 0.592000 | +0.070023 |
| ReactZyme | time | enzyme_to_reaction | 0.775257 | 0.690000 | +0.085257 |
| ReactZyme | time | reaction_to_enzyme | 0.527831 | 0.372000 | +0.155831 |
| EnzymeMap | table1 | bedroc85 | 0.547147 | 0.486600 | +0.060547 |
| EnzymeMap | table1 | bedroc20 | 0.708884 | 0.666900 | +0.041984 |
| EnzymeMap | table1 | ef0.05 | 15.457319 | 14.910000 | +0.547319 |
| EnzymeMap | table1 | ef0.1 | 8.433043 | 8.180000 | +0.253043 |
| EnzymeMap | table2 | bedroc85 | 0.507029 | 0.451400 | +0.055629 |
| EnzymeMap | table2 | bedroc20 | 0.675522 | 0.614300 | +0.061222 |
| EnzymeMap | table2 | ef0.05 | 14.840457 | 13.570000 | +1.270457 |
| EnzymeMap | table2 | ef0.1 | 8.175664 | 7.810000 | +0.365664 |

For Table-2 EF10, the comparison above uses the published CLIPZyme value **7.81** ([FGW-CLIP Table 2](https://arxiv.org/html/2512.08508v1#S5.T2)); the immutable original qualification used the locally reproduced **7.80813533427353**. This configuration exceeds both.

**Separate TIGER ablation context.** The primary main-model Reaction-Sim E→R target is 0.518. This configuration's 0.524770 MRR is below the two-layer-MLP ablation’s 0.543 and below the SwissProt+DGN condition’s 0.632. These are separate published conditions; this statement concerns that MRR cell only. [TIGER Tables 1–3](https://arxiv.org/html/2605.24489v1).

**Case 1.** Choices were frozen before scoring this version. All 144 literature-panel entries are retained, representing 123 unique sequences. The table below counts unique-sequence recovery; the companion CSV also ranks every original entry. Each target-trained checkpoint is reported independently, alongside its native encoder before phase 2.

| Target model / score | Paper catalysts @25 / 12 | Papers + patents @25 / 24 | Workbook actives @25 / 81 | Conditional AUROC |
| --- | ---: | ---: | ---: | ---: |
| reaction_smi/selected | 7 | 9 | 21 | 0.598471 |
| reaction_smi/native_before_phase2 | 6 | 6 | 17 | 0.534685 |
| enzyme_smi/selected | 5 | 10 | 21 | 0.577895 |
| enzyme_smi/native_before_phase2 | 5 | 9 | 20 | 0.616990 |
| time/selected | 5 | 8 | 18 | 0.526161 |
| time/native_before_phase2 | 5 | 5 | 18 | 0.572898 |
| enzymemap/selected | 7 | 7 | 17 | 0.471193 |
| enzymemap/native_before_phase2 | 7 | 7 | 17 | 0.465902 |

**Predeclared deployment result.** Reaction-Sim paper-catalyst recovery changes from **6/12 before phase 2 to 7/12 after phase 2** at 25. The other target-trained checkpoints are reported without substituting one based on Case1 performance.

**Early literature-catalyst recovery.** All three recorded cutoffs are shown to make screening-budget tradeoffs visible. The final column counts primary papers represented by at least one recovered catalyst; it is descriptive source coverage, not a count of independent validation experiments. These results do not alter the frozen model choices.

| Target model / score | Paper catalysts @5 / 12 | @10 / 12 | @25 / 12 | Primary papers represented @25 |
| --- | ---: | ---: | ---: | ---: |
| reaction_smi/selected | 2 | 2 | 7 | 5/8 |
| reaction_smi/native_before_phase2 | 0 | 1 | 6 | 3/8 |
| enzyme_smi/selected | 1 | 3 | 5 | 4/8 |
| enzyme_smi/native_before_phase2 | 1 | 4 | 5 | 4/8 |
| time/selected | 1 | 4 | 5 | 4/8 |
| time/native_before_phase2 | 0 | 4 | 5 | 4/8 |
| enzymemap/selected | 0 | 0 | 7 | 4/8 |
| enzymemap/native_before_phase2 | 0 | 0 | 7 | 4/8 |

**Requested 144-entry panel.** The same sequence can represent several literature entries; these counts are not independent confirmations. Rankings retain all entries and use the recorded deterministic tie order.

| Target model / score | Paper entries @25 / 15 | Papers + patents @25 / 36 | Workbook active entries @25 / 102 |
| --- | ---: | ---: | ---: |
| reaction_smi/selected | 7 | 11 | 21 |
| reaction_smi/native_before_phase2 | 4 | 4 | 18 |
| enzyme_smi/selected | 8 | 14 | 24 |
| enzyme_smi/native_before_phase2 | 6 | 10 | 22 |
| time/selected | 8 | 12 | 22 |
| time/native_before_phase2 | 8 | 8 | 21 |
| enzymemap/selected | 7 | 7 | 18 |
| enzymemap/native_before_phase2 | 6 | 6 | 18 |

**Reaction-Sim training homology.** Of 7 paper catalysts recovered at 25, 3 have a retrieved training hit with at least 90% sequence identity. The audit covers supervised Reaction-Sim training only, not frozen protein/SLEEC pretraining. Both query molecules also occur together within larger training participant sets. No qualifying MMseqs hit means unknown similarity, not established absence of a homolog. The panel does not establish distant catalytic generalization.

| Paper catalyst | Selected rank | Native rank | Best retrieved training identity |
| --- | ---: | ---: | ---: |
| H012 | 3 | 22 | unknown (no qualifying hit) |
| H011 | 4 | 23 | unknown (no qualifying hit) |
| H007 | 12 | 72 | unknown (no qualifying hit) |
| H005 | 16 | 58 | unknown (no qualifying hit) |
| H004 | 17 | 8 | 96.8% |
| H003 | 22 | 14 | 97.5% |
| H001 | 24 | 13 | 99.1% |
| H002 | 28 | 24 | 98.1% |
| H006 | 53 | 90 | unknown (no qualifying hit) |
| H008 | 57 | 38 | 50.2% |
| H009 | 60 | 32 | 50.0% |
| H013 | 68 | 57 | 44.4% |

Case 1 is one previously examined reaction, with dependent constructs and heterogeneous literature assays. The 42 non-detect-only sequences are conditional assay observations, not universally inactive enzymes. These results are retrospective evidence, not new wet-lab validation or proof of broad catalytic generalization. No model is selected by Case1 results.

**Frozen artifacts.**

- reaction_smi: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/reaction_smi/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/reaction_smi/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/reaction_smi/phase2_followup/epoch10/training/step0100.pt).
- enzyme_smi: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/enzyme_smi/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/enzyme_smi/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/enzyme_smi/phase2_followup/epoch10/training/step0100.pt).
- time: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/time/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/time/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/time/phase2_followup/epoch10/training/step0100.pt).
- enzymemap: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/enzymemap/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/enzymemap/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/enzymemap/phase2_followup/epoch10/training/step0100.pt).

[Exact protocol](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_epoch10_fixed_test_v1/protocol.json) · [Benchmark source records](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_epoch10_fixed_test_v1/qualification.json) · [Case1 metrics](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_epoch10_fixed_test_v1/case1/summary.json) · [All 144 rankings](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_epoch10_fixed_test_v1/case1/all_144_entry_rankings.csv) · [123 unique-sequence rankings](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_epoch10_fixed_test_v1/case1/unique_sequence_rankings.csv).

[Architecture, training-edge and artifact audit](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_epoch10_fixed_test_v1/evidence_audit.json).

### Fresh stronger-fusion study: shared_fusion03_beta5_b1024_v1

Completed 2026-09-21T07:36:50.732633+00:00. **14/14** primary benchmark targets exceeded. Joint winner.

Fresh target-specific SLEEC residue-view F3 fits use initial fused contribution 0.3, beta 5, batch 1024 and seed 42. Checkpoints at 5/10/15/20 epochs receive identical positive-CE phase 2 (100 updates, temperature 0.2, identity weight 10), semantic alpha 0.25 and residual cap 1. No inference fusion adjustment is applied. Checkpoint selection uses target validation only; EnzymeMap uses full-library BEDROC85 and reports all BEDROC/EF cells.

Selected epochs: reaction_smi=20, enzyme_smi=20, time=20, enzymemap=15.

| Benchmark | Setting | Metric | Method | Strongest primary target | Pass |
| --- | --- | --- | ---: | ---: | --- |
| ReactZyme | reaction_smi | enzyme_to_reaction | 0.549547 | 0.518000 | yes |
| ReactZyme | reaction_smi | reaction_to_enzyme | 0.412479 | 0.337000 | yes |
| ReactZyme | enzyme_smi | enzyme_to_reaction | 0.978006 | 0.956000 | yes |
| ReactZyme | enzyme_smi | reaction_to_enzyme | 0.672143 | 0.592000 | yes |
| ReactZyme | time | enzyme_to_reaction | 0.799581 | 0.690000 | yes |
| ReactZyme | time | reaction_to_enzyme | 0.547569 | 0.372000 | yes |
| EnzymeMap | table1 | bedroc85 | 0.545234 | 0.486600 | yes |
| EnzymeMap | table1 | bedroc20 | 0.717676 | 0.666900 | yes |
| EnzymeMap | table1 | ef0.05 | 15.817309 | 14.910000 | yes |
| EnzymeMap | table1 | ef0.1 | 8.685418 | 8.180000 | yes |
| EnzymeMap | table2 | bedroc85 | 0.502070 | 0.451400 | yes |
| EnzymeMap | table2 | bedroc20 | 0.685032 | 0.614300 | yes |
| EnzymeMap | table2 | ef0.05 | 15.238332 | 13.570000 | yes |
| EnzymeMap | table2 | ef0.1 | 8.469625 | 7.808135 | yes |

[Protocol](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/protocol.json) · [Exact source records](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/qualification.json).

### Winning configuration: shared_fusion03_beta5_b1024_v1

Recorded 2026-09-21T07:38:57.220141+00:00. This configuration exceeds all **14 primary benchmark point estimates** (six ReactZyme cells and eight EnzymeMap screening cells). The comparison is exploratory after repeated benchmark inspection; it does not establish statistical superiority or control for differing pretrained resources.

**Architecture and training.** One dual encoder retains the frozen SLEEC scorer, a global protein view and four learned residue views. The compact reaction tower uses the same model configuration across targets. Each benchmark/split has its own freshly trained F3 weights and its own training-only semantic dictionary. Frozen protein/reaction encoder and SLEEC pretraining differ from competitors; matched downstream associations do not eliminate that pretraining difference.

**Stage 1.** Anchor-balanced all-positive decoupled InfoNCE, beta 5.0, equal R→E/E→R weights, seed 42, batch 1024, maximum 20 epochs, initial fused residue scale 0.3. AdamW learning rate 0.0001 and weight decay 0.01. Each benchmark is independently trained from initialization. Checkpoints at epochs 5/10/15/20 are selected using the predeclared target validation criterion, with all four phase-2 fits reported. Selected epochs: reaction_smi=20, enzyme_smi=20, time=20, enzymemap=15.

**Stage 2.** A residual dual encoder fits the full target training graph for 100 positive-CE updates, temperature 0.2, identity weight 10 and learning rate 0.0001. The semantic dictionary uses training associations only. Inference uses residual cap **1.0**, semantic weight **0.25**, and a **1×** multiplier on the internal fused residue contribution. Endpoint embeddings remain independently encodable; there is no model ensemble, candidate-pool hubness correction, or RefSeq search.

**Protocol.** ReactZyme uses the official three target splits and all-positive MRR in both directions. EnzymeMap uses 34,427/7,287/4,642 original train/dev/test associations, with the released 261,907-ID screening pool; Table 2 excludes training enzyme IDs, leaving 252,113 candidates. Validation reports BEDROC85, BEDROC20, EF5 and EF10; MRR is disabled for EnzymeMap. The checkpoint grid is selected by ReactZyme mean bidirectional seen/unseen validation MRR or EnzymeMap full-library Table-1 BEDROC85. The phase-2 and semantic composition recipe is fixed. ReactZyme reaction inputs follow its participant-set representation; EnzymeMap retains physical reactant/product sides.

| Benchmark | Setting | Metric | Method ↑ | Comparison target | Difference |
| --- | --- | --- | ---: | ---: | ---: |
| ReactZyme | reaction_smi | enzyme_to_reaction | 0.549547 | 0.518000 | +0.031547 |
| ReactZyme | reaction_smi | reaction_to_enzyme | 0.412479 | 0.337000 | +0.075479 |
| ReactZyme | enzyme_smi | enzyme_to_reaction | 0.978006 | 0.956000 | +0.022006 |
| ReactZyme | enzyme_smi | reaction_to_enzyme | 0.672143 | 0.592000 | +0.080143 |
| ReactZyme | time | enzyme_to_reaction | 0.799581 | 0.690000 | +0.109581 |
| ReactZyme | time | reaction_to_enzyme | 0.547569 | 0.372000 | +0.175569 |
| EnzymeMap | table1 | bedroc85 | 0.545234 | 0.486600 | +0.058634 |
| EnzymeMap | table1 | bedroc20 | 0.717676 | 0.666900 | +0.050776 |
| EnzymeMap | table1 | ef0.05 | 15.817309 | 14.910000 | +0.907309 |
| EnzymeMap | table1 | ef0.1 | 8.685418 | 8.180000 | +0.505418 |
| EnzymeMap | table2 | bedroc85 | 0.502070 | 0.451400 | +0.050670 |
| EnzymeMap | table2 | bedroc20 | 0.685032 | 0.614300 | +0.070732 |
| EnzymeMap | table2 | ef0.05 | 15.238332 | 13.570000 | +1.668332 |
| EnzymeMap | table2 | ef0.1 | 8.469625 | 7.810000 | +0.659625 |

For Table-2 EF10, the comparison above uses the published CLIPZyme value **7.81** ([FGW-CLIP Table 2](https://arxiv.org/html/2512.08508v1#S5.T2)); the immutable original qualification used the locally reproduced **7.80813533427353**. This configuration exceeds both.

**Separate TIGER ablation context.** The primary main-model Reaction-Sim E→R target is 0.518. This configuration's 0.549547 MRR is above the two-layer-MLP ablation’s 0.543 and below the SwissProt+DGN condition’s 0.632. These are separate published conditions; this statement concerns that MRR cell only. [TIGER Tables 1–3](https://arxiv.org/html/2605.24489v1).

**Case 1.** Choices were frozen before scoring this version. All 144 literature-panel entries are retained, representing 123 unique sequences. The table below counts unique-sequence recovery; the companion CSV also ranks every original entry. Each target-trained checkpoint is reported independently, alongside its native encoder before phase 2.

| Target model / score | Paper catalysts @25 / 12 | Papers + patents @25 / 24 | Workbook actives @25 / 81 | Conditional AUROC |
| --- | ---: | ---: | ---: | ---: |
| reaction_smi/selected | 4 | 6 | 18 | 0.564668 |
| reaction_smi/native_before_phase2 | 6 | 6 | 20 | 0.557319 |
| enzyme_smi/selected | 4 | 11 | 21 | 0.607878 |
| enzyme_smi/native_before_phase2 | 4 | 10 | 22 | 0.639330 |
| time/selected | 5 | 8 | 19 | 0.616990 |
| time/native_before_phase2 | 5 | 10 | 22 | 0.624339 |
| enzymemap/selected | 6 | 6 | 18 | 0.467372 |
| enzymemap/native_before_phase2 | 6 | 6 | 19 | 0.456202 |

**Predeclared deployment result.** Reaction-Sim paper-catalyst recovery changes from **6/12 before phase 2 to 4/12 after phase 2** at 25. This is a negative transfer result despite the benchmark win. The other target-trained checkpoints are reported without substituting one based on Case1 performance.

**Early literature-catalyst recovery.** All three recorded cutoffs are shown to make screening-budget tradeoffs visible. The final column counts primary papers represented by at least one recovered catalyst; it is descriptive source coverage, not a count of independent validation experiments. These results do not alter the frozen model choices.

| Target model / score | Paper catalysts @5 / 12 | @10 / 12 | @25 / 12 | Primary papers represented @25 |
| --- | ---: | ---: | ---: | ---: |
| reaction_smi/selected | 0 | 1 | 4 | 2/8 |
| reaction_smi/native_before_phase2 | 1 | 3 | 6 | 3/8 |
| enzyme_smi/selected | 0 | 1 | 4 | 3/8 |
| enzyme_smi/native_before_phase2 | 3 | 4 | 4 | 3/8 |
| time/selected | 0 | 1 | 5 | 4/8 |
| time/native_before_phase2 | 0 | 0 | 5 | 4/8 |
| enzymemap/selected | 0 | 2 | 6 | 3/8 |
| enzymemap/native_before_phase2 | 0 | 1 | 6 | 3/8 |

**Requested 144-entry panel.** The same sequence can represent several literature entries; these counts are not independent confirmations. Rankings retain all entries and use the recorded deterministic tie order.

| Target model / score | Paper entries @25 / 15 | Papers + patents @25 / 36 | Workbook active entries @25 / 102 |
| --- | ---: | ---: | ---: |
| reaction_smi/selected | 4 | 8 | 21 |
| reaction_smi/native_before_phase2 | 5 | 5 | 20 |
| enzyme_smi/selected | 3 | 14 | 24 |
| enzyme_smi/native_before_phase2 | 6 | 12 | 23 |
| time/selected | 8 | 13 | 23 |
| time/native_before_phase2 | 6 | 11 | 22 |
| enzymemap/selected | 6 | 6 | 18 |
| enzymemap/native_before_phase2 | 6 | 6 | 19 |

**Reaction-Sim training homology.** Of 4 paper catalysts recovered at 25, 4 have a retrieved training hit with at least 90% sequence identity. The audit covers supervised Reaction-Sim training only, not frozen protein/SLEEC pretraining. Both query molecules also occur together within larger training participant sets. No qualifying MMseqs hit means unknown similarity, not established absence of a homolog. The panel does not establish distant catalytic generalization.

| Paper catalyst | Selected rank | Native rank | Best retrieved training identity |
| --- | ---: | ---: | ---: |
| H004 | 8 | 4 | 96.8% |
| H003 | 14 | 9 | 97.5% |
| H001 | 15 | 10 | 99.1% |
| H002 | 16 | 11 | 98.1% |
| H013 | 39 | 48 | 44.4% |
| H008 | 43 | 21 | 50.2% |
| H009 | 51 | 25 | 50.0% |
| H011 | 68 | 66 | unknown (no qualifying hit) |
| H012 | 74 | 67 | unknown (no qualifying hit) |
| H007 | 81 | 87 | unknown (no qualifying hit) |
| H005 | 87 | 84 | unknown (no qualifying hit) |
| H006 | 90 | 88 | unknown (no qualifying hit) |

Case 1 is one previously examined reaction, with dependent constructs and heterogeneous literature assays. The 42 non-detect-only sequences are conditional assay observations, not universally inactive enzymes. These results are retrospective evidence, not new wet-lab validation or proof of broad catalytic generalization. No model is selected by Case1 results.

**Frozen artifacts.**

- reaction_smi: F3 [screen-epoch=19.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/reaction_smi/checkpoints/screen_selection/screen-epoch=19.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/reaction_smi/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/reaction_smi/phase2_followup/epoch20/training/step0100.pt).
- enzyme_smi: F3 [screen-epoch=19.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/enzyme_smi/checkpoints/screen_selection/screen-epoch=19.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/enzyme_smi/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/enzyme_smi/phase2_followup/epoch20/training/step0100.pt).
- time: F3 [screen-epoch=19.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/time/checkpoints/screen_selection/screen-epoch=19.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/time/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/time/phase2_followup/epoch20/training/step0100.pt).
- enzymemap: F3 [screen-epoch=14.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/enzymemap/checkpoints/screen_selection/screen-epoch=14.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/enzymemap/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/enzymemap/phase2_followup/epoch15/training/step0100.pt).

[Exact protocol](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/protocol.json) · [Benchmark source records](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/qualification.json) · [Case1 metrics](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/case1/summary.json) · [All 144 rankings](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/case1/all_144_entry_rankings.csv) · [123 unique-sequence rankings](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/case1/unique_sequence_rankings.csv).

[Architecture, training-edge and artifact audit](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/evidence_audit.json).

### Completed shared-method campaign, 2026-09-21

Six configurations now exceed all 14 primary benchmark point estimates, with completed frozen Case1 predictions for all 144 entries / 123 unique sequences and separate detailed reports. V1–V4 are inference variants of one training family; V5–V6 use two checkpoint policies from a second fresh training family. All use seed 42. The final 5/10/15/20-epoch study is complete: validation selects epoch 20 on all three ReactZyme splits and epoch 15 on EnzymeMap.

V6 has the strongest six ReactZyme MRR results among these six configurations, including Reaction-Sim R→E/E→R 0.412479/0.549547, but its predeclared Case1 deployment recovers only 4/12 paper catalysts at 25, below V5’s 7/12 and its own native encoder’s 6/12. All four recovered V6 paper catalysts have retrieved training homologs at ≥90% identity. V4 has the highest paper recovery at 25 (10/12, covering seven of eight primary papers), while V5 recovers two related S16 constructs in its top five. These tradeoffs are retained without selecting or combining models on Case1.

The requested benchmark comparison and Case1 reporting work is complete. The comparison remains exploratory after repeated test inspection; pretrained resources differ and exact unpublished competitor metric implementations remain unverified. Broad catalytic generalization and new wet-lab success are not established. All training and evaluation workers in this final study have exited; the campaign monitor was stopped after its final refresh. No new RefSeq experiment, ensemble or hubness correction was run.

[All six methods and exact results](archive/20260928/documents/shared_winning_methods_20260921.md) · [Completion evidence](runs/generalization_20260919_2251/cross_paper_retraining/shared_methods_completion_audit_20260921.json).

### User preference: V4, 2026-09-21

The user selected V4 (`shared_recipe_alpha04_cap05_v1`) as the preferred configuration after reviewing the completed results. Follow-up work should use this configuration as its reference. Its existing target-specific checkpoint bundles and evaluation records remain frozen. This is a post-results user preference, separate from the original evaluation selection rules.

The recipe retains SLEEC, anchor-balanced all-positive decoupled InfoNCE with beta 5, batch 512 and the epoch-10 F3 checkpoint. Phase 2 uses 100 positive-CE updates, temperature 0.2 and identity weight 10. Inference uses semantic weight 0.4, residual cap 0.5 and the recorded 3× internal residue-fusion calibration. The EnzymeMap checkpoint already contains that calibration; its manifest prevents applying it twice.

V4 exceeds all 14 primary benchmark comparison targets. Its predeclared Reaction-Sim deployment recovers 10/12 unique paper catalysts at 25 in Case1, across seven of eight primary papers, with conditional AUROC 0.612875. All 144 entries remain ranked.

[V4 detailed report and checkpoint links](archive/20260928/documents/shared_recipe_alpha04_cap05_v1.md) · [Preference record](runs/generalization_20260919_2251/cross_paper_retraining/preferred_method.json).

### V4 extensive implementation explanation, 2026-09-21

Added [the extensive V4 explanation](archive/20260928/documents/v4_extensive_explanation.md), covering the six protein views, reaction inputs, phase-1 decoupled all-positive objective, phase-2 positive-CE objective, frozen training dictionary, score equation, exact benchmark results and Case1 interpretation. This is a documentation pass; no new experiments or parameter changes were made.

Three implementation clarifications: the phase-2 setting called cap 0.5 is a multiplier on the residual output, not a hard norm bound; phase 2 was fitted on native F3 vectors before the 3× fusion adjustment on every target; the semantic block transfers the maximum neighbor contribution per training reaction, not summed neighbor frequency. V4’s final score is 0.6 times the neural dot product plus 0.4 times the semantic dot product.

The extra primary-paper hit relative to V2 is H009 (TsT4Ease M4-4), rank 28→24 when halving the residual update. Relative to V4’s native encoder, H006 moves 33→23 and paper recovery rises 9/12→10/12 at 25. Other cutoffs do not uniformly improve: primary-paper recovery at 5 falls 1/12→0/12. The later user preference for V4 was recorded after reviewing these outcomes and is separate from the original frozen per-variant evaluation choices.


### V4 biological supervision with unchanged architecture, 2026-09-21

The user requested EC/cofactor/mechanism bias while explicitly rejecting extra architecture components. A training-only category-balanced cosine-distance regularizer now supervises existing embeddings. No heads, encoders, slots, inference inputs or learned parameters were added. SLEEC, the V4 dual encoder, the existing phase-2 heads and the existing dictionary score are retained. EC prefixes encode functional ancestry; these are not positional encodings, and individual learned coordinates do not have assigned biological meanings.

A predeclared phase-2 study compared weight 0, 0.03, 0.1 and 0.3, leave-one-family-out ablations at 0.1, and a shuffled-label control. All eight configurations exceed the 14 primary benchmark point estimates. All four unchanged controls reproduce the original V4 phase-2 parameters exactly. This reuses each benchmark's independently fresh target-specific F3 fit; no ReactZyme-trained F3 is transferred into EnzymeMap. Annotations add supervision, so only downstream associations and architecture are controlled relative to the no-biology model.

The complete-signal 0.03, 0.1 and 0.3 variants were frozen and evaluated on Case1. Each retains 10/12 primary-paper catalysts at 25 for the Reaction-Sim deployment; all four target-trained models and all 144 entries /123 unique sequences are reported. Conditional AUROC stays about 0.613 for that deployment and about 0.47 for the EnzymeMap-trained checkpoint. This preserves the earlier result, not a demonstrated wet-lab generalization gain.

The numerical audit is consequential: Time E→R rises from 0.780669 to 0.815391 at weight 0.3, but its diagnostic 1e-6 pessimistic rank bound changes only 0.771579→0.771696. Most of that headline gain is sensitive to almost-tied scores. Enzyme-Sim E→R also has substantial near-tie sensitivity. Official-style fixed score ranking is retained and the sensitivity is reported separately. The shuffled-label control also retains all benchmark wins; those wins by themselves do not show that biological supervision helped. EnzymeMap differences over V4 are mostly tiny.

Mechanism annotations are coarse atom-mapped bond-change descriptors, not experimentally established catalytic mechanisms. Cofactor labels indicate recognized reaction participants, not proven cofactor dependence. Unknown labels produce no negative biological targets. Weak reaction maps below confidence 0.5 were excluded, leaving much lower mechanism coverage on ReactZyme than on EnzymeMap. Native ReactZyme EC labels are restricted to training sequences; they may describe multiple activities.

A separate fresh-F3 study is active: the identical training-only regularizer reaches the existing residue views, without adding model components. Ten epochs, batch 512, seed 42, beta 5 and fixed snapshot policy are declared for each target. An initial launcher guard rejected disabled validation with screening snapshots before any optimizer updates; those failed preflight outputs are retained. The corrected launch preserves validation-loss checks every three epochs while retrieval MRR remains disabled, with full official metric evaluation to follow. No live run will be stopped merely at the three-hour boundary.

[Detailed tables, ablations, coverage, numerical audit and Case1](archive/20260928/documents/v4_biological_geometry_20260921.md) · [Loss-only study](runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_geometry_20260921_v1/protocol.json) · [Fresh F3 study](runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_f3_20260921_v1/protocol.json).

### Fresh biological F3 and contribution controls, 2026-09-21

Status at 13:44 UTC: the EnzymeMap epoch-10 evaluation is complete; the three ReactZyme fresh fits and their queued evaluations remain in progress. The architecture and original target-specific association data are unchanged. The existing SLEEC scorer is bit-identical to V4, state tensor names/shapes match, and no biological head is added. New tests demonstrate gradient flow to the existing residue queries and content-based pooling invariance after the pretrained residue features are fixed.

With biology in F3 and phase 2, EnzymeMap Table 1 is BEDROC85 **0.586748**, BEDROC20 **0.760990**, EF5 **16.824537**, EF10 **8.982937**; Table 2 is **0.544864 / 0.732691 / 16.421220 / 8.836806**. All eight primary screening targets are exceeded. Relative to V4, BEDROC85 gains are +0.014411 and +0.016130. Paired rule-cluster bootstrap 95% intervals are [0.000371, 0.024082] and [0.000491, 0.027101]. These describe fixed-model query/rule variation, not training-seed variation or confirmatory significance after repeated experiments. Table 1 EF5 declines slightly.

The contribution audit prevents a stronger causal claim: removing cofactors improves BEDROC85 to **0.595922 / 0.555082**, and shuffled annotations achieve **0.591985 / 0.551120**, above the correctly annotated all-signal model on this metric. Removing mechanism supervision improves BEDROC20 and enrichment, with a small BEDROC85 reduction. Correct biological assignments have therefore not been shown to explain the full gain. All family effects and eight-metric tables are in `documents/v4_biology_contributions_20260921.md`.

A post-fit validation diagnostic also finds no improved EC or mechanism category separation in the learned reaction vectors. Only five of the 2,652 eligible validation reactions have recognized cofactor categories, so the cofactor diagnostic is underpowered. Do not claim that individual learned slots have become EC, cofactor or catalytic-mechanism detectors.

Follow-up: two additional paired EnzymeMap seeds (43/44, biological vs unannotated F3) are running. A new **relative biological loss**, using the same vectors and no extra parameters, is training on all four benchmark targets; matched EnzymeMap family-removal and shuffled controls are also scheduled/running. It compares within-category distance with annotated background distance, using a confidence-scaled margin. This responds to the shuffled-label result and is explicitly exploratory. Default attraction behavior remains unchanged and the expanded checks pass (50 tests, 10 subtests).

Detailed vector motivation and the distinction from positional encoding: `documents/v4_biological_signal_architecture.md`. Fresh-F3 results: `documents/v4_biological_f3_20260921.md`. Branch: `research/v4-biological-loss-20260921`. No new Case1 claim is made before all 14 primary comparisons qualify and the new models are frozen.


### Biological supervision without architectural additions: complete first fresh-F3 test, 2026-09-21

Status recorded at 2026-09-21T14:25:39.034617+00:00. Branch: `research/v4-biological-loss-20260921`. V4/SLEEC, four residue queries, embedding dimensions, phase2 modules and inference recipe are unchanged. Biological signals enter training losses on existing vectors. An actual completed checkpoint encoded synthetic residue inputs with the annotation path deliberately absent; no annotation helper was initialized. The architecture audit verifies identical state tensor names/shapes and frozen SLEEC values. The runtime check is `v4_biological_f3_20260921_v1/inference_without_annotations.json` under the cross-paper campaign directory.

The first fresh-F3 attraction experiment is fully evaluated on both benchmarks. Both predeclared phase2 variants pass **13/14**, not all 14, primary point comparisons. Biology in F3 gives ReactZyme R→E/E→R: Reaction-Sim **0.402602/0.532584**, Enzyme-Sim **0.664804/0.954584**, Time **0.539845/0.794002**. Enzyme-Sim E→R misses the **0.956** target. Adding biology in phase2 gives **0.954336** there. These two configurations are not promoted to the qualified Case1 follow-up. Earlier phase2-only biological configurations retain their separate 14/14 comparisons and completed Case1 results.

Fresh score sensitivity checks show Enzyme-Sim E→R bounds approximately **0.948825–0.979525** when competitors within 1e-6 are treated as potentially tied. These are diagnostic per-positive bounds, not replacement official metrics; they do not turn a failed comparison into a win. The first fresh-F3 report now includes the bounds and explicit failed-target status.

Relative biological loss, weight0.1, has completed EnzymeMap testing. With biology in both stages, Table1 BEDROC85/BEDROC20/EF5/EF10 is **0.587095/0.765618/16.996782/9.057582**; Table2 is **0.544910/0.737364/16.556981/8.912204**. All eight primary screening targets are exceeded. ReactZyme results for this loss and the stronger weight1 experiment remain pending.

The relative-loss leave-one-family-out results show positive conditional effects from retaining cofactors and mechanism on all eight screening metrics at seed42; EC is mostly neutral or slightly adverse. However, shuffled profiles score **0.589697/0.548546** on Table1/Table2 BEDROC85, higher than correctly assigned labels. Correct labels give stronger EF10 than the shuffled control. Thus the controls show mixed effects and do not establish that the full BEDROC85 gain requires correct biological assignments. Validation reaction geometry also shows slightly weaker EC/mechanism separation than V4; the cofactor diagnostic covers only five reactions.

The attraction-loss seed43/44 matched controls are complete. Three-seed mean BEDROC85 changes **0.514163→0.519046** for Table1 and **0.462274→0.467760** for Table2; it improves in two seeds and declines in seed44. BEDROC20 improves in all three. Variation across seeds is much larger than the mean biological gain. Seeds affect initialization, batch order and dropout; this is not an isolated initialization ablation. All four new seed-control checkpoints pass the architecture/data/SLEEC audit. The mean exceeds the primary screening targets, but individual seeds do not all win. No seed or score ensemble is used.

At 14:21 UTC, two additional seeds for each already-declared relative-loss weight (0.1 and1.0) were predeclared and launched across four GPUs in `v4_relative_biology_seed_replication_20260921_v1`. They reuse the matching unannotated seed43/44 controls and report every result. They add no new model configuration beyond replication of the existing loss settings. Startup experienced shared-filesystem waits; jobs and existing ReactZyme training remain active. The goal remains open until pending tests and qualified Case1 follow-ups finish.

- [Complete first fresh-F3 report](archive/20260928/documents/v4_biological_f3_20260921.md)
- [Relative-loss family contributions](archive/20260928/documents/v4_relative_biology_contributions_20260921.md)
- [Three-seed audit and standalone figure](archive/20260928/documents/v4_biology_seed_sensitivity_20260921.md)
- [Architecture and learned-vector interpretation](archive/20260928/documents/v4_biological_signal_architecture.md)


### Stronger relative biological loss: EnzymeMap and complete family controls, 2026-09-21 14:27 UTC

F3 relative-loss weight1.0, unchanged V4, seed42, epoch10, with existing phase2 biological weight0.1, exceeds all eight primary EnzymeMap targets. Table1 BEDROC85/BEDROC20/EF5/EF10: **0.591840/0.762491/16.685533/8.962760**. Table2: **0.550281/0.733446/16.201370/8.807355**. Compared with original V4, BEDROC85 improves by **0.019504/0.021546**; EF5 declines by **0.173509/0.217263** while remaining above the published comparators.

Correct-minus-shuffled BEDROC85 is **+0.006400/+0.006349**. Unlike the weaker loss, this favors true assignments, but descriptive reaction-rule bootstrap95% intervals are **[-0.009031,+0.021678]** and **[-0.012712,+0.022735]**, so the observed advantage is not conclusive. Correct labels also score lower than shuffled labels on EF5. All outcomes are retained.

All-minus-without-mechanism BEDROC85 is **+0.016321/+0.018747**, and all-minus-without-EC is **+0.007132/+0.007879**. EC has an opposing enrichment tradeoff. Cofactor removal is nearly neutral for BEDROC and improves enrichment. Consequently, the evidence does not support claiming that all three signals help every metric. Post-fit validation category separation improves slightly for EC (0.741956→0.742609) and mechanism (0.566445→0.569171); cofactor evidence remains limited to five validation reactions.

All twelve completed family-control checkpoints across attraction and both relative-loss weights pass architecture, downstream-data and frozen-SLEEC checks. ReactZyme fits and the newly launched relative-loss seed43/44 replications continue. No configuration is promoted on the basis of EnzymeMap alone.

[Detailed stronger-loss contributions and uncertainty](archive/20260928/documents/v4_relative_biology_weight1_contributions_20260921.md) · [Benchmark report](archive/20260928/documents/v4_relative_biology_weight1_20260921.md)


### What the learned queries actually do, and why some ReactZyme ranks are fragile

Diagnostic completed 2026-09-21T14:47:13.496975+00:00. Four completed EnzymeMap checkpoints (V4, attraction0.1, relative0.1, relative1) were evaluated on the same 128 validation proteins absent from training by exact protein ID, selected by sorted identifiers without test/Case1 outcomes. All use native F3 before phase2 or the fixed inference fusion multiplier.

Mean normalized raw query-attention entropy is **0.943–0.945**; effective support is **73.5–74.1%** of protein length. Centered inter-query attention cosine is **0.549–0.559**. These learned queries are broad, overlapping content selectors, not demonstrated catalytic-site detectors. Mean fused-branch gate mass is approximately **18% global,33% SLEEC,49% collectively across four learned views**. Gate mass is not a causal importance percentage or a share of the final score. Reversing already-contextualized residue arrays changes the tested pooled vector by only **2.2e-8–4.5e-8**. This confirms the implemented content-pooling interpretation; reversing the amino-acid input to ProtT5 would be a different operation. The loss adds no parameter or view.

The first fresh-F3 near-tie chemistry audit now identifies the relevant input ambiguity. In Enzyme-Sim, **all542/542** positive associations with a competitor within1e-6 involve a different reaction ID with identical canonical molecular participants; there are12 such near-tied reaction pairs. Native F3 reaction vectors for these pairs differ by at most **8.2e-8**. In Time, the analogous count is **1281/1286**, involving42 canonical-alias pairs. In Reaction-Sim it is only2/14. Canonicalization preserves stereochemistry and molecule multiplicity while removing atom mapping and component order. The released ReactZyme inputs lack physical reaction arrows and use the established participant-set adapter. The audit does not alter the benchmark, labels, official metrics or qualification gate. It explains why some E→R point-score changes cannot be interpreted as improved biochemical discrimination.

Cofactor annotation coverage also has a marked shift: **4915/12603** EnzymeMap training reactions versus **5/2652** eligible validation reactions. The five observed training categories are NAD/NADP, CoA, quinone, FAD/FMN and generic metal; only two training reactions receive the generic-metal label. This participant recognizer cannot recover enzyme-bound cofactors omitted from the reaction string. The mechanism vocabulary is four coarse transformation groups, not a detailed catalytic mechanism model.

The relative-loss ReactZyme fits, feature exports and additional EnzymeMap seeds remain in progress. Seed reports and final tie/chemistry diagnostics are automated, and qualified methods retain automatic frozen Case1 evaluation.

- [Query measurements and interpretation](archive/20260928/documents/v4_biological_query_audit_20260921.md)
- [Architecture, annotation scope and mathematical limits](archive/20260928/documents/v4_biological_signal_architecture.md)
- [ReactZyme tie chemistry with raw reaction pairs](runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_f3_20260921_v1/tie_chemistry_audit.json)
- [Relative-loss seed replication](archive/20260928/documents/v4_relative_biology_seed_sensitivity_20260921.md)


### Two fresh V4 relative-biological methods exceed all primary benchmark targets (2026-09-21T14:58:14.233106+00:00)

Both methods retain the full V4 architecture, frozen SLEEC, four existing residue queries, phase2 model and fixed semantic scoring. F3 is trained fresh per target dataset for10epochs, batch512, seed42, using the existing decoupled anchor-balanced multi-positive objective plus relative biological geometry at weight0.1 and margin0.1. EC, cofactor and coarse mechanism terms act on the existing512D vectors; there are zero added parameter tensors and no inference annotation inputs. The first method uses the original phase2 loss; the second also applies biological geometry in phase2 at weight0.1 per family, with fixed divisor3. Both use100phase2steps, residual multiplier0.5, semantic weight0.4 and3× inference fusion. No ensemble or hubness correction.

All14 primary targets are exceeded for each method. These are fixed-recipe exploratory point comparisons after repeated testing, not confirmatory claims against every differently supervised result in the literature.

| Setting / metric | Comparator | Biology in F3 | Biology in F3 + phase2 |
| --- | ---: | ---: | ---: |
| reaction_smi / enzyme_to_reaction | 0.518000 | 0.546721 | 0.546759 |
| reaction_smi / reaction_to_enzyme | 0.337000 | 0.398129 | 0.398132 |
| enzyme_smi / enzyme_to_reaction | 0.956000 | 0.971280 | 0.971723 |
| enzyme_smi / reaction_to_enzyme | 0.592000 | 0.665099 | 0.665099 |
| time / enzyme_to_reaction | 0.690000 | 0.796441 | 0.796079 |
| time / reaction_to_enzyme | 0.372000 | 0.539334 | 0.539316 |
| table1 / bedroc85 | 0.486600 | 0.587057 | 0.587095 |
| table1 / bedroc20 | 0.666900 | 0.765590 | 0.765618 |
| table1 / ef0.05 | 14.910000 | 16.983633 | 16.996782 |
| table1 / ef0.1 | 8.180000 | 9.057582 | 9.057582 |
| table2 / bedroc85 | 0.451400 | 0.544873 | 0.544910 |
| table2 / bedroc20 | 0.614300 | 0.737335 | 0.737364 |
| table2 / ef0.05 | 13.570000 | 16.558643 | 16.556981 |
| table2 / ef0.1 | 7.810000 | 8.912204 | 8.912204 |

Case1 was frozen after benchmark qualification, scored across all144entries, and separately evaluated on123unique sequences. All target-trained checkpoints are reported; none was selected using Case1.

| Method / target-trained checkpoint | Paper catalysts @25 /12 | Conditional AUROC |
| --- | ---: | ---: |
| f3_biology / reaction_smi | 7 | 0.551146 |
| f3_biology / enzyme_smi | 5 | 0.610229 |
| f3_biology / time | 5 | 0.632863 |
| f3_biology / enzymemap | 6 | 0.478248 |
| f3_phase2_biology / reaction_smi | 7 | 0.551146 |
| f3_phase2_biology / enzyme_smi | 5 | 0.610229 |
| f3_phase2_biology / time | 5 | 0.633157 |
| f3_phase2_biology / enzymemap | 6 | 0.478542 |

**Generalization result:** the predesignated Reaction-Sim model recovers7/12 paper catalysts at25 and has conditional AUROC0.551146, worse than original V4's10/12 and0.612875. Thus these new benchmark-winning configurations do not improve the primary real-case assessment. The panel is retrospective, contains related constructs and heterogeneous literature assays, and has no new measured activity.

**Contribution limits:** in EnzymeMap seed42, dropping cofactors or mechanism harms several metrics, but shuffled labels score higher on BEDROC85. Across three seeds, relative0.1 improves BEDROC85 only at seed42; the paired mean change is+0.002480/+0.002651 for Tables1/2. Relative1 has a larger mean change of+0.010256/+0.011009 and improves2/3seeds, with EF5 tradeoffs. The semantic query/protein scoring inputs and candidate axes are exactly identical in the audited seed controls, so differences in that frozen component do not explain the large seed variation.

**Additional controls:** fresh unannotated EnzymeMap seed42, relative1 shuffled profiles at seeds43/44, and mechanism removal at seed43 are running in`v4_relative_biology_additional_controls_20260921_v1`. Fresh unannotated ReactZyme controls with identical fast validation cadence are running in`v4_matched_reactzyme_control_20260921_v1`; these isolate a known random-number-consumption confound and may extend beyond the3hour return time. No existing run was stopped.

[Full method report](archive/20260928/documents/v4_relative_biology_f3_20260921.md) · [Three-seed comparisons](archive/20260928/documents/v4_relative_biology_seed_sensitivity_20260921.md) · [Architecture explanation](archive/20260928/documents/v4_biological_signal_architecture.md)


### Two stronger relative-biological methods also exceed all primary benchmarks (2026-09-21T15:06:39.692631+00:00)

These use the identical unchanged V4 recipe and target-specific fresh F3 training documented above, with F3 relative biological weight1.0 instead of0.1. Both retain frozen SLEEC, four existing residue queries,512D embeddings,100-step existing phase2, residual multiplier0.5, semantic weight0.4 and3× inference fusion. The first uses unannotated phase2; the second adds the same relative biology at phase2 weight0.1 per family. No new parameters, heads, input branches or inference annotations. Seed42,10epochs,batch512 are fixed.

| Setting / metric | Comparator | Biology in F3 | Biology in F3 + phase2 |
| --- | ---: | ---: | ---: |
| reaction_smi / enzyme_to_reaction | 0.518000 | 0.541618 | 0.541619 |
| reaction_smi / reaction_to_enzyme | 0.337000 | 0.401638 | 0.401637 |
| enzyme_smi / enzyme_to_reaction | 0.956000 | 0.957043 | 0.957807 |
| enzyme_smi / reaction_to_enzyme | 0.592000 | 0.664423 | 0.664423 |
| time / enzyme_to_reaction | 0.690000 | 0.792256 | 0.803627 |
| time / reaction_to_enzyme | 0.372000 | 0.539148 | 0.539152 |
| table1 / bedroc85 | 0.486600 | 0.591894 | 0.591840 |
| table1 / bedroc20 | 0.666900 | 0.762516 | 0.762491 |
| table1 / ef0.05 | 14.910000 | 16.678959 | 16.685533 |
| table1 / ef0.1 | 8.180000 | 8.962760 | 8.962760 |
| table2 / bedroc85 | 0.451400 | 0.550344 | 0.550281 |
| table2 / bedroc20 | 0.614300 | 0.733475 | 0.733446 |
| table2 / ef0.05 | 13.570000 | 16.186411 | 16.201370 |
| table2 / ef0.1 | 7.810000 | 8.807355 | 8.807355 |

All14 primary targets are exceeded, but Enzyme-Sim E→R is numerically fragile because of the documented canonical reaction aliases. These are exploratory point comparisons, not independent confirmatory superiority.

| Method / target-trained checkpoint | Case1 paper catalysts @25 /12 | Conditional AUROC |
| --- | ---: | ---: |
| f3_biology / reaction_smi | 6 | 0.509112 |
| f3_biology / enzyme_smi | 5 | 0.595826 |
| f3_biology / time | 5 | 0.616990 |
| f3_biology / enzymemap | 6 | 0.485009 |
| f3_phase2_biology / reaction_smi | 6 | 0.509112 |
| f3_phase2_biology / enzyme_smi | 5 | 0.596120 |
| f3_phase2_biology / time | 5 | 0.616990 |
| f3_phase2_biology / enzymemap | 6 | 0.485009 |

The primary Reaction-Sim Case1 model recovers6/12 at25 and has conditional AUROC0.509112, worse than original V4 and the weaker relative loss. Thus the stronger biological loss has **not** solved real-case generalization, despite winning the benchmark point comparisons. All144entries and123unique-sequence rankings are preserved, with literature/assay caveats unchanged.

Across three seeds, stronger relative F3 gives mean BEDROC85 **0.524419/0.473283**, versus unannotated **0.514163/0.462274**. Mean paired gains are+0.010256/+0.011009, with2/3seeds improved. Mean EF5 changes are-0.151444/-0.222346. Annotation assignment and mechanism-specific replication controls are running; the completed seed42 controls and bootstrap intervals are reported without claiming significance.

A new fixed-base phase2 study was declared at15:04UTC in`v4_relative_biology_phase2_20260921_v1`: relative biological weights1,3,10 on the existing phase2 heads, leaving original target-trained V4 F3 fixed. A zero-biology control is reused with its exact reproduction evidence; leave-EC/cofactor/mechanism-out and shuffled-profile controls use weight10. All recipes are predeclared and evaluated on both benchmarks; qualifying complete-signal methods receive frozen Case1 evaluation. Existing native embeddings and semantic caches are reused with source checks. This tests whether biological supervision can preserve V4's better real-case behavior with the same architecture. Because it follows observed Case1 outcomes, it remains retrospective exploration.

[Stronger-method report](archive/20260928/documents/v4_relative_biology_weight1_20260921.md) · [Contributions](archive/20260928/documents/v4_relative_biology_weight1_contributions_20260921.md) · [Three-seed results](archive/20260928/documents/v4_relative_biology_seed_sensitivity_20260921.md) · [Fixed-base phase2 follow-up](archive/20260928/documents/v4_relative_biology_phase2_20260921.md)


## Relative biology only in existing phase2: three qualified methods (2026-09-21T15:25:06.737852+00:00)

V4 architecture, SLEEC, F3 checkpoints, native feature caches, dictionary and inference remain unchanged. Existing phase2 heads receive relative category supervision with margin0.1. All families have the same weight (1,3,10); family averaging retains divisor3. Same 100 updates, identity penalty10 and existing contrastive objective. These are separate methods, never an ensemble. The studies are exploratory after repeated test and Case1 inspection.

| Setting / metric | V4 | Weight1 | Weight3 | Weight10 |
| --- | ---: | ---: | ---: | ---: |
| ReactZyme / reaction_smi / enzyme_to_reaction | 0.534366 | 0.534777 | 0.534908 | 0.535844 |
| ReactZyme / reaction_smi / reaction_to_enzyme | 0.400381 | 0.400412 | 0.400456 | 0.400608 |
| ReactZyme / enzyme_smi / enzyme_to_reaction | 0.971355 | 0.967373 | 0.968651 | 0.967817 |
| ReactZyme / enzyme_smi / reaction_to_enzyme | 0.666249 | 0.666234 | 0.666172 | 0.666156 |
| ReactZyme / time / enzyme_to_reaction | 0.780669 | 0.800614 | 0.789883 | 0.790308 |
| ReactZyme / time / reaction_to_enzyme | 0.538285 | 0.538317 | 0.538237 | 0.538722 |
| EnzymeMap / table1 / bedroc85 | 0.572337 | 0.572096 | 0.571237 | 0.567451 |
| EnzymeMap / table1 / bedroc20 | 0.755433 | 0.755322 | 0.755160 | 0.752611 |
| EnzymeMap / table1 / ef0.05 | 16.859042 | 16.860921 | 16.872308 | 16.847596 |
| EnzymeMap / table1 / ef0.1 | 8.958692 | 8.956501 | 8.980329 | 8.992382 |
| EnzymeMap / table2 / bedroc85 | 0.528734 | 0.528464 | 0.527528 | 0.523364 |
| EnzymeMap / table2 / bedroc20 | 0.726403 | 0.726275 | 0.726107 | 0.723300 |
| EnzymeMap / table2 / ef0.05 | 16.418633 | 16.435718 | 16.452814 | 16.442394 |
| EnzymeMap / table2 / ef0.1 | 8.793103 | 8.803076 | 8.824366 | 8.826859 |

All three full-signal methods exceed all14 primary comparator cells. This preserves benchmark wins; it does not mean each exceeds V4. EnzymeMap BEDROC85 decreases with these phase2 weights. The original F3 had already been trained from scratch on each target-specific training split.

| Variant / model | Unique literature catalysts @25 /12, among123 sequences | Literature entries @25 /15, among144 entries | Conditional AUROC on123 sequences |
| --- | ---: | ---: | ---: |
| all_1 / reaction_smi | 11 | 10 | 0.613757 |
| all_1 / enzyme_smi | 5 | 8 | 0.585538 |
| all_1 / time | 5 | 7 | 0.636390 |
| all_1 / enzymemap | 6 | 6 | 0.470018 |
| all_3 / reaction_smi | 11 | 10 | 0.615520 |
| all_3 / enzyme_smi | 5 | 8 | 0.584950 |
| all_3 / time | 5 | 8 | 0.636978 |
| all_3 / enzymemap | 6 | 6 | 0.470606 |
| all_10 / reaction_smi | 10 | 10 | 0.611993 |
| all_10 / enzyme_smi | 5 | 8 | 0.584068 |
| all_10 / time | 5 | 6 | 0.642563 |
| all_10 / enzymemap | 6 | 6 | 0.472663 |

Original V4 Reaction-Sim recovers10/12 unique catalysts and10/15 paper-supported entries at25, AUROC0.612875. Weights1/3 increase unique catalyst recovery to11/12 but leave entry-level recovery at10/15. This is one retrospective reaction with related constructs and incompletely independently verified workbook assays, not prospective activity validation or broad generalization. No candidate-IID significance claim is made.

The completed weight10 removal/shuffle controls have mixed or adverse screening effects. Additional controls at exactly weights1 and3 are now running, so their family contributions are not inferred from weight10. All protocols and predictions are preserved.

[Full phase2 method report](archive/20260928/documents/v4_relative_biology_phase2_20260921.md) · [Architecture and learned vectors](archive/20260928/documents/v4_biological_signal_architecture.md)

## Additional matched EnzymeMap controls completed

Fresh unannotated seed42 F3 reproduces all120 original state tensors bit for bit, maximum difference0. Small downstream ranking differences remain (BEDROC85 about7e-5). True relative annotations beat shuffled assignments on BEDROC85 for seeds42/43, lose for seed44, and reduce mean EF5. The seed43 mechanism-removal control supports a BEDROC85 benefit but an EF5 cost. This is mixed evidence about biological assignment, not universal family gains. [All metrics and paired differences](archive/20260928/documents/v4_biology_additional_controls_20260921.md).

The Case1 unique-sequence gain at phase2 weights1/3 is a cutoff crossing: H008 (TsT4Ease WT) moves from rank26 to25. Original144-entry paper recovery is unchanged. [All12 literature-catalyst ranks and source hashes](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_phase2_20260921_v1/case1_rank_change_audit.json).


## Phase2 family controls: benchmark and Case1 follow-up complete

Matched family-removal and shuffled controls at weights1/3 are complete. All qualified controls receive the same retrospective144-entry Case1 follow-up, with the123-sequence view kept separate. No architecture changes or additional training data were introduced. [All Case1 contributions and source hashes](archive/20260928/documents/v4_relative_biology_phase2_case1_controls_20260921.md); the report links both full benchmark contribution tables.

Case1 conditional contributions for the unchanged original Reaction-Sim F3:

| Phase2 weight / annotations | Unique paper catalysts @25 /12 | Original paper entries @25 /15 | Conditional AUROC |
| --- | ---: | ---: | ---: |
| 1 / all | 11 | 10 | 0.613757 |
| 1 / without_ec | 10 | 10 | 0.613757 |
| 1 / without_cofactor | 11 | 10 | 0.614345 |
| 1 / without_mechanism | 10 | 10 | 0.613169 |
| 1 / shuffled_1 | 10 | 10 | 0.611111 |
| 3 / all | 11 | 10 | 0.615520 |
| 3 / without_ec | 11 | 10 | 0.615814 |
| 3 / without_cofactor | 11 | 10 | 0.616108 |
| 3 / without_mechanism | 10 | 10 | 0.612875 |
| 3 / shuffled_3 | 10 | 10 | 0.609347 |

At both weights, removing mechanism or shuffling labels loses the one additional unique catalyst at25. EC removal loses it only at weight1; cofactor removal preserves it and slightly improves AUROC. This is a conditional effect on one retrospective cutoff crossing. The144-entry top25 recovery remains unchanged, and screening contributions have different tradeoffs.


## Local gradient audit: biological signal without new components

At the saved phase2 weight3 step100 heads, the gradient of each applied biological term is measured against the existing retrieval gradient. No optimizer step, model change, test selection or new data is used. These are pre-clipping parameter gradients, not Adam update magnitudes or causal importance scores.

| Target | EC norm / retrieval | Cofactor norm / retrieval | Mechanism norm / retrieval |
| --- | ---: | ---: | ---: |
| reaction_smi | 0.011143 | 0.004931 | 0.043250 |
| enzyme_smi | 0.012462 | 0.000000 | 0.042015 |
| time | 0.015661 | 0.004302 | 0.020454 |
| enzymemap | 0.002464 | 0.034979 | 0.253576 |

Mechanism has the largest local biological gradient in these four fits. The existing identity-preservation gradient opposes the retrieval gradient strongly, especially on ReactZyme; this is consistent with conservative refinement. Equal family coefficients do not provide equal local gradients. Outcome contributions remain the separately reported removal/shuffle experiments. [Diagnostic methods and source hashes](archive/20260928/documents/v4_biology_gradient_audit_20260921.md).


## Matched fresh ReactZyme controls completed

All three unannotated controls now have fixed-epoch test results and architecture/data audits. The primary comparison uses unannotated phase2 throughout, isolating the F3 loss change under the same validation cadence. Full absolute metrics, signed differences and the separate phase2-only controls are in [the matched-control report](archive/20260928/documents/v4_matched_reactzyme_controls_20260921.md). These controls are exploratory and single-seed; no Case1 improvement is inferred.


## Matched fresh controls complete: benchmark wins do not isolate biological benefit

Both the unannotated F3 control and its relative0.1 phase2-only variant exceed all14 benchmark comparator cells. Their F3, batch, seed and validation cadence match the fresh biological runs. These controls remove the known cadence confound from the biological-loss comparison; they do not identify validation cadence as the sole cause of original V4 versus fresh-training differences.

| Control / target | Unique paper catalysts @25 /12 | Original paper entries @25 /15 | Conditional AUROC |
| --- | ---: | ---: | ---: |
| f3_biology / reaction_smi | 8 | 9 | 0.568783 |
| f3_biology / enzyme_smi | 5 | 8 | 0.582598 |
| f3_biology / time | 4 | 6 | 0.638154 |
| f3_biology / enzymemap | 6 | 6 | 0.471193 |
| f3_phase2_biology / reaction_smi | 8 | 9 | 0.568783 |
| f3_phase2_biology / enzyme_smi | 5 | 8 | 0.582598 |
| f3_phase2_biology / time | 4 | 6 | 0.637860 |
| f3_phase2_biology / enzymemap | 6 | 6 | 0.470606 |

Here `f3_biology` is only a pipeline directory name: the F3 biological coefficient is zero. Matched unannotated Reaction-Sim yields8/12 unique catalysts and AUROC0.568783, versus original V4's10/12 and0.612875. Fresh relative biology yields7/12 and0.551146 at0.1, or6/12 and0.509112 at1.0. Thus the original-versus-fresh regression cannot all be assigned to the biological loss, but a further decline remains against the correct matched control.

The original-F3 phase2-only biological variants retain the better Case1 behavior. Their best reported unique top25 count is11/12, with unchanged144-entry paper recovery10/15 and modest conditional AUROC. All positive, neutral and adverse contrasts are preserved.

[Matched benchmark metrics](archive/20260928/documents/v4_matched_reactzyme_controls_20260921.md) · [All matched control Case1 results](archive/20260928/documents/v4_matched_control_case1_20260921.md) · [Updated architecture explanation](archive/20260928/documents/v4_biological_signal_architecture.md).


## Biological V4 experiment completion audit

All planned training, benchmark, annotation-control, seed, learned-query and Case1 follow-ups have finished. The final audit rechecked42 primary comparison cells across the three phase2 full-signal variants, source/qualification hashes, original144 versus unique123 candidate counts, and independently recomputed unique-sequence catalyst recovery and conditional AUROC for all12 main variant/target combinations. Screening validation usesBEDROC/EF. Saved phase2 architectures have no additional parameters. [Full completion audit and limits](archive/20260928/documents/v4_biology_final_audit_20260921.md).


## 2026-09-21 — Public competitors retrained on the exact ReactZyme splits

The user expanded the baseline campaign to include all public competitors. The work is on `research/reactzyme-public-baselines-20260921`; inherited unrelated changes were preserved.

The durable protocol is [documents/reactzyme_public_baselines_protocol_20260921.md](archive/20260928/documents/reactzyme_public_baselines_protocol_20260921.md). The [live comparison](runs/reactzyme_public_baselines_20260921/comparison.md) records test results, running jobs, dependencies, and failures. This section is an initial status snapshot, not a claim that the campaign is complete.

- Official ReactZyme matrix: 96 released-family configurations plus 24 separately labeled corrected-contrastive-loss sensitivities; 30/120 complete at 2026-09-21T21:00:52+00:00, eight active, 0 failed.
- Original Horizyn architecture/MLNCE: three completed matched-data runs with an explicit participant-set reaction adapter. Test R→E/E→R all-positive MRR: reaction similarity **0.431815 / 0.534799**, enzyme similarity **0.699701 / 0.976990**, time **0.598203 / 0.840621**. Epochs were selected on validation only. This parent model is a strong comparator; these rows must remain in the comparison.
- CREEP/CARE: all three splits training from public unimodal initialization, with exact-edge sampling and no EC Cartesian-product positives or description text. Full encoder fine-tuning is much slower than the fixed-feature heads. The two later runs use tested fused T5 attention and FFN recomputation, consuming about 130 GB GPU memory including neighboring jobs.
- EnzGFM-650M: twelve backbone-control runs queued behind frozen-feature pooling. The existing residue cache has a documented preprocessing difference from the paper (special tokens and truncation), so this is not presented as exact paper reproduction.
- Full CLIPZyme / EnzymeCAGE / VenusRXN: tracked in scope; no full native-model training claimed yet. Complete structural inputs and reaction representation adapters require further preparation. TIGER and FGW-CLIP remain paper-only references until a reproducible official release is verified.

All methods use the exact existing V4 `reactzyme_paper` training and validation partitions. The test queries, complete candidate pools, and their ordering match all three V4 test catalogs exactly. Shared scoring preserves the all-positive MRR definition in both directions. Frozen test-input feature extraction does not fit retrieval parameters or select checkpoints. Fresh downstream models never warm-start from the modified F3/V4 retrieval checkpoints.

The released classifier code has an unseeded validation split, checkpoint selection that follows training loss despite its naming, and a reversed positive convention in its contrastive branch. The campaign fixes validation selection, preserves original model classes and the original contrastive row, and labels its corrected-loss sensitivity separately. Missing released negative pair files require a documented deterministic corruption protocol; results are matched-data retraining, not exact copied-paper-score reproductions.

Twelve fidelity tests passed, covering native/cached pair scoring, known-positive filtering and deterministic negatives, metric semantics, and T5 output/gradient parity on CPU/CUDA. One trained Bi-RNN exceeded the tight cached-scoring numerical tolerance; it was recovered through the untouched pair forward from its already selected checkpoint. No retraining or test-based checkpoint switch was used for that recovery.

The local public EnzymeCAGE collection sequence-matches 160,241/178,327 proteins, with 18,086 missing. The user confirmed no additional prepared collection. Every missing sequence maps to public UniProt accessions in the original ReactZyme metadata. An AlphaFold coordinate-download pilot recovered 28/32 with exact whole-chain sequence matches; four remained unresolved. Public coordinate acquisition is now running, prioritizing these missing pockets before the remaining full CLIPZyme structure inputs. Missing proteins are never silently removed. Coordinates, predicted pockets, aligned residue features, graph construction, and model training are separate stages.

Matching associations controls downstream supervision, but differences in pretrained features, structural inputs, objectives, adapters, and tuning budgets remain. Strict architecture-only superiority requires additional common-input/common-loss controls and multi-seed uncertainty; no such superiority claim is made from this partial campaign.


### Public baseline monitoring — 2026-09-21T21:58:38+00:00

The official-family/sensitivity matrix reached 61/120 completed tests, with 8 active jobs and 0 current failures. All 64 completed result files have finite, in-range retrieval metrics. EnzGFM frozen features now cover all 178,327 proteins, and the first EnzGFM Transformer control is training. CREEP remains in epoch 1 on all three splits; decreasing training loss is not yet generalization evidence.

Public pocket recovery completed its priority pass: 16,143 exact public structures were available among the 18,086 previously unmatched proteins; P2Rank produced 14,555 pockets and no usable top-ranked pocket for 1,588. This leaves 3,531 proteins unresolved (1,943 without an accepted full structure plus 1,588 without a recovered pocket). The combined 174,796 mapped pocket files remain subject to residue-alignment validation. Full CLIPZyme/EnzymeCAGE training has not started; no candidate filtering was introduced. Full-pool coordinate acquisition continues. See the live comparison and `monitor/latest.json` for subsequent status.


### Public baseline monitoring — first completed EnzGFM control (2026-09-21T22:26:07+00:00)

The campaign has 77/120 official-family/sensitivity results and no current training failures. The first EnzGFM-650M residue-mean + released ReactZyme Transformer control, with MAT-2D reaction features on the reaction-similarity split, selected epoch 53 using validation BCE and achieved test all-positive MRR **0.135390 R→E / 0.296464 E→R**. This is the previously disclosed backbone control with different residue pooling/truncation from the EnzGFM paper, not an exact paper reproduction. Its same-architecture ESM2 control scores 0.122008 / 0.231432; the representation and preprocessing differences prevent attributing any difference to architecture. The EnzGFM enzyme-similarity run has started. All CREEP runs remain in their first epoch, with no validation/test result yet.


## Recorded-run ablation audit — 2026-09-21

Audited 260 benchmark measurements from 89 saved result/configuration files without launching, stopping or modifying model runs. The complete study is in [the ablation report](archive/20260928/documents/ablation_study_20260921/report.md), with [an executed audit notebook](archive/20260928/documents/ablation_study_20260921/audit.ipynb), source hashes, and a long-form CSV.

- Controlled final-recipe inference sensitivities: dictionary weight 0.25/0.40/0.50 at residual multiplier 1, and residual multiplier 0.5 versus 1 at weight 0.40. Same recorded checkpoint lineage and evaluation axes across all three ReactZyme splits and both EnzymeMap screening settings; effects trade off across metrics.
- Phase-2 biological supervision: no-added-label, correct-label, shuffled-label and EC/cofactor/mechanism removal controls. The unannotated control exactly reproduces V4 metrics. Table 1 BEDROC85 is 0.572337 without biology, 0.572096 with weight-1 biology, and 0.572093 with shuffled labels. These results do not establish added biological supervision as the cause of the benchmark wins.
- Matched 18-epoch EnzymeMap temperature study: beta5 beats beta10 in 22/24 seed-by-metric cells; mean Table 1 BEDROC85 0.521656 versus 0.494040 across seeds 17/42/73. This is a component study, not the final 10-epoch V4 seed study.
- F3 relative biology across seeds 42/43/44: weight1 mean Table 1 BEDROC85 effect +0.010256 (2/3 seeds), EF5 effect -0.151444. Keep this separate from phase-2-only biology.
- Reaction-Sim Case1 phase-2 biology follow-up: unique-paper recovery at25 changes 10/12 to 11/12; original144-entry recovery stays10/15. Retrospective literature evidence, not prospective wet-lab validation.

A complete final-V4 architectural ablation still needs matched dictionary/phase2 removals and loss/query/direction controls. Historical best checkpoints must not be mixed into a causal component table. All existing tests are exploratory after repeated inspection; preserve near-tie sensitivity and additional-annotation supervision caveats. The portable HTML builder rejects honest non-SQL source provenance, so the full report is delivered as Markdown; the blocker is recorded beside the artifact.


## Ablation figures — SciencePlots and blue–orange palette

Added eight interpreted figures to the [ablation report](archive/20260928/documents/ablation_study_20260921/report.md): inference sensitivity on both benchmarks, paired temperature effects, biological-loss controls, ReactZyme near-tie sensitivity, F3 seed variability, and Case1 recovery. All figures use SciencePlots science/no-latex styling with the user-selected blue–orange palette and a neutral midpoint for signed heatmaps. [PNG, SVG and PDF exports](archive/20260928/documents/ablation_study_20260921/figures/README.md) and reproducible plotting code are included.

The figures now use compact layouts without titles, subtitles, panel titles or footnotes. Essential axes, legends and data labels remain; the manuscript captions contain the explanations, evaluation scope and interpretation. All eight figures and their 24 exports were regenerated and visually checked, with numerical outputs unchanged.

Figure 04 uses a shared symmetric logarithmic color scale, linear within ±0.001 ΔMRR×100, to keep smaller effects visible alongside the large Time E→R changes. Cell values remain unchanged, and the manuscript caption explains the nonlinear color intensity. Figure 01 retains linear colors.

Verified 184 plotted signed contrasts against the existing source measurements, preserved all original tables and prose, and checked three color-vision deficiency simulations with redundant marker shapes, signed labels and hatching. This update changes presentation only; no additional model training or test evaluation was performed.


## Experimental section in LaTeX — snapshot 2026-09-22 00:04:47 UTC

Added the [editable experimental-section package](archive/20260928/documents/v4_experiments_latex/README.md), [compiled PDF](archive/20260928/documents/v4_experiments_latex/main.pdf), and [portable ZIP](archive/20260928/documents/v4_experiments_latex.zip), accompanying the existing methods document. The section follows FGW-CLIP's task organization with datasets, baselines, evaluation metrics, and results for EnzymeMap screening and ReactZyme retrieval, followed by Case1 and the recorded ablations. All eight approved SciencePlots blue–orange PDF figures are included with manuscript captions; their data and minimal styling are preserved.

The supplement includes the exact V4 recipe, public-baseline adaptations and budgets, explicit BEDROC/EF and all-positive MRR conventions, first-positive retrieval diagnostics, every target-trained Case1 result, and all **126 completed public-baseline receipts** at the frozen snapshot. This is 95 official-family configurations, 24 corrected-contrastive sensitivities, three original Horizyn fits, and four EnzGFM controls. Subsequent background progress does not silently change the manuscript tables.

The newer matched-data evidence is retained prominently: original Horizyn exceeds V4 and V4+Bio on all six ReactZyme split/direction MRR cells. V4 exceeds the released CLIPZyme screening checkpoint and the displayed FGW-CLIP screening point estimates, but the document does not claim superiority over every retrained competitor. Published references and local results are separated; different budgets, pretrained resources, and unresolved published MRR conventions remain explicit.

Audited the user's 145-entry Case1 catalogue against the sequence manifest: **145 catalogue entries → 144 resolved/scored entries → 123 unique sequences**. H017 (KoT4E M6) lacks a fully recoverable sequence. The main case study reports the Reaction-Sim model's 10/12 unique primary-paper recovery at 25, 11/12 with phase-2 biology, and unchanged 10/15 recovery in the original-entry view. It includes weaker EnzymeMap-trained transfer and distinguishes retrospective literature evidence from prospective wet-lab validation.

Verification: warning-free LaTeX compilation, resolved citations/references, all eight figure checksums preserved, 650 numerical table cells logged, all completed baseline receipts checked, deterministic table regeneration, and visual PDF review. Source provenance, a frozen evidence snapshot, and author notes are included. No model training or new test predictions were run for this writing task.


## ICLR 2027 experimental-section revision

Rewrote the extended experimental report as a [conference-paper section](archive/20260928/documents/v4_experiments_iclr2027/README.md), using the supplied **ICLR 2027 single-column template without modifying its font size, margins, or spacing**. The [compiled PDF](archive/20260928/documents/v4_experiments_iclr2027/main.pdf) is **seven pages including references**, within the user's ten-page budget for the experimental section. [Editable LaTeX](archive/20260928/documents/v4_experiments_iclr2027/experiments.tex) and a [portable source ZIP](archive/20260928/documents/v4_experiments_iclr2027.zip) are provided.

The manuscript contains EnzymeMap and ReactZyme protocols/results, four selected ablation figures, and the small Case1 literature panel. Four compact tables replace the extended run inventory; the report title, contents, operational status discussion, and implementation appendix are omitted. Full provenance and the 126-run inventory remain in the separate evidence package. The stronger original Horizyn result, published/local metric distinction, repeated-test development history, mixed biological effects, and Case1 transfer limitations remain in the paper.

Adapted original figures 01, 03, 04, and 08 to readable ICLR-width layouts with the approved SciencePlots blue–orange palette. Verified all 72 retained numerical contrasts, 12 Case1 recovery counts, and 153 table cells against the previous evidence. Case1 retains 145 catalogue entries, 144 resolved entries, and 123 unique sequences. Compilation has no unresolved references or overflow; one benign underfull vertical box from the unchanged template was visually checked. No additional model runs or test predictions were performed.


## ICLR 2027 method revision — two-page main section

Rewrote the method as a [paper-style LaTeX section](archive/20260928/documents/v4_methods_iclr2027/methods.tex), with a [two-page PDF preview](archive/20260928/documents/v4_methods_iclr2027/method_only.pdf) in the unmodified ICLR 2027 single-column template. The [complete PDF](archive/20260928/documents/v4_methods_iclr2027/main.pdf) contains the two main pages, one reference page, and 13 appendix pages. The [portable source ZIP](archive/20260928/documents/v4_methods_iclr2027.zip) and [integration instructions](archive/20260928/documents/v4_methods_iclr2027/README.md) accompany the section. The previous extensive document remains available as an audit companion.

The main text explains the independent protein/reaction encoders, anchor-balanced multi-positive alignment, full-graph residual refinement, optional phase-2 biological supervision, and dictionary-augmented bidirectional scoring. Five displayed equations retain the core formulation. The detailed architecture diagram, all layer specifications, attention safeguards, exact biological confidence weighting and efficient derivations, preprocessing, annotation provenance, optimizer settings, and checkpoint-specific inference procedure are in the appendix. The distinction between the two association losses and the post-training fusion/residual adjustments remains explicit.

Verified the two-page main-text length, unchanged conference template files, all 18 implementation-source hashes, and verbatim preservation of all 39 prior displayed equation environments in the appendix. Citations and cross-references resolve, there are no text overflows, and labels do not collide with the experimental section. One benign bibliography underfull horizontal box was visually checked. The earlier numerical loss/gradient checks are retained as provenance and were not rerun; this task changes neither the model nor experimental results.

### Method prose aligned with the author's writing sample

Revised the same [main method](archive/20260928/documents/v4_methods_iclr2027/methods.tex) around the author's supplied problem-formulation passage. The section now begins with explicit graph and compatibility definitions, defines both query-dependent candidate pools, and develops the enzyme encoder, reaction encoder, alignment, refinement, and scoring in separate subsections with connected prose and forward references. The opening cosine score and final score consistently use the complete neural-plus-dictionary vectors. The author's `\method{}` command is supported, while encoder functions `f_R,f_E` remain distinct from the fused enzyme vector `f_e`.

The compiled main text remains two pages in the unchanged ICLR template. Exact appendix equations, training losses, inference settings, and experimental results are unchanged. Refreshed the PDF previews, source archive, and document validation record.

### Expanded enzyme representation and biological supervision

Expanded the [enzyme representation](archive/20260928/documents/v4_methods_iclr2027/enzyme_representation.tex) in the author's requested style, with functional-residue motivation and explicit paragraphs/equations for SLEEC pooling, four learned residue summaries, and feature-wise fusion. Added an editable [biological-supervision fragment](archive/20260928/documents/v4_methods_iclr2027/biological_supervision.tex) to the phase-2 section, including the confidence-weighted relative-category hinge and the complete refinement objective.

Checked the SLEEC computation directly against `EnzymeMultiviewEncoder.forward`: the evaluated model uses a separate fixed view from centered/clipped SLEEC logits, smoothed with a uniform residue distribution. The author's proposed additive learned-logit bias is not this implementation and has been corrected in the manuscript. Biological supervision shapes the refinement heads after F3 is frozen, rather than directly specializing the learned residue queries. No architecture, checkpoint, or experiment was changed.

The expanded method now occupies two full pages and less than half a third (approximately 2.48 pages of content), within the requested 2.5-page budget in the unchanged ICLR template. The standalone preview therefore has three PDF pages; the complete document has 17 pages including references and appendix. Earlier two-page entries above describe previous document versions. Refreshed the LaTeX package, PDFs, and source archive.

### Paper-style alignment and V4+Bio as the sole proposed method

Rewrote the [alignment subsection](archive/20260928/documents/v4_methods_iclr2027/methods.tex) around the many-to-many association setting, the decoupled positive comparisons, and the role of per-anchor and directional averaging. The phase-1 mathematical objective is unchanged. The appendix retains the exact minibatch sampler, all-known-positive mask, annotation caveats, and entropy/diversity terms.

The current method package now presents **V4+Bio (`all_1`) as its sole proposed method**. Biological supervision is integral to phase-2 training; both main and appendix objectives use all three family losses with unit coefficients before averaging. Removed optional-variant wording, the alternate zero-biology method definition, and the historical attraction-only objective from the printed method. Updated the method name, architecture caption, training procedure, and documentation consistently. Association-only K experiments remain explicitly labelled ablation controls; their metrics have not been relabelled as V4+Bio results.

The selected paper method should not be called uniformly best across benchmarks: the recorded biological effects are mixed and later matched-data Horizyn results are stronger on ReactZyme. This presentation change makes no new empirical superiority claim and changes no training run, checkpoint, or reported measurement. The main text occupies approximately 2.44 pages in the unchanged ICLR template; the full PDF has 20 pages including references and appendices.

### Matched phase-1, phase-2 and phase-2+Bio test comparison (2026-09-22)

Completed the missing phase-1 evaluations using authenticated cached embeddings from the exact historical V4 backbones (seed42, epoch10, batch512, beta5). The deployed factor-three fusion adjustment is held fixed in all columns. Phase1 uses backbone cosine similarity alone; phase2 adds both residual refinement and the training-only dictionary (residual multiplier0.5, dictionary weight0.4); phase2+Bio uses the existing `all_1` heads. No retraining or selection was performed. This is a stage comparison, not an isolated full-graph-loss or residual-head ablation.

Phase2 exceeds phase1 in all14 benchmark metric cells. Reaction-Sim E→R/R→E MRR changes0.493650/0.356631→0.534366/0.400381. EnzymeMap full-pool BEDROC85 changes0.545263→0.572337; training-enzyme-excluded BEDROC85 changes0.518774→0.528734. Biology produces mixed changes (7/14 cells increase), including small BEDROC decreases in both pools. Case1 unique literature-catalyst recovery at25 changes9/12→10/12→11/12, whereas original144-entry recovery remains10/15 throughout. These exploratory single-seed outcomes retain near-tie and retrospective Case1 limitations.

The [complete17-row comparison](runs/v4_stage_comparison_20260922/comparison.md), [CSV](runs/v4_stage_comparison_20260922/comparison.csv), [exact metrics and35 source hashes](runs/v4_stage_comparison_20260922/comparison.json), and [evaluation script](runs/v4_stage_comparison_20260922/evaluate.py) are saved. EnzymeMap was evaluated with the existing released-protocol BEDROC/EF evaluator, retaining1,521/1,337 eligible queries and261,907/252,113 candidates. ReactZyme used the existing all-positive MRR evaluator; query and positive-edge counts match the recorded phase2 and Bio evaluations in every split.


## 2026-09-22 — Proposed method excludes added biological-label supervision

At the author's request, the current ICLR manuscript proposes **V4 without the added biological-label losses**. Phase 2 optimizes `L_CE^(2) + 10 L_id`; EC, cofactor-category, and bond-change supervision are excluded. SLEEC residue features, reaction chemistry inputs, the phase-1 attention penalties, residual heads, and the training-reaction dictionary remain part of the evaluated recipe. The biological variants are retained as clearly labeled ablations and historical experiments.

The completed [matched stage comparison](runs/v4_stage_comparison_20260922/comparison.md) supports the simpler choice: the complete phase-2 pipeline improves all 14 benchmark endpoints over phase 1, while phase-2 biological supervision improves seven and worsens seven. This comparison jointly adds residual refinement and dictionary scoring, with protein fusion held fixed; it does not isolate either component's causal contribution. The biology effects do not establish a consistent improvement, and the larger ReactZyme E→R movements require the documented near-tie qualification. This decision is not a claim that biology can never help.

Primary tables and model provenance now use the existing `shared_recipe_alpha04_cap05_v1` checkpoints. All four backbone and residual-head checkpoint hashes and four config hashes match the frozen manifest. No training, checkpoint mutation, or new model evaluation was needed for this manuscript revision. The main Case1 result is **10/12 unique literature catalysts** and **10/15 original paper entries** recovered at rank 25. The 11/12 unique recovery of V4+Bio remains an ablation result.

Updated deliverables:
- [Main method and appendix](archive/20260928/documents/v4_methods_iclr2027/main.pdf), with the [editable training/scoring fragment](archive/20260928/documents/v4_methods_iclr2027/training_and_scoring.tex).
- [Experimental section](archive/20260928/documents/v4_experiments_iclr2027/main.pdf), including the phase-1 / phase-2 / phase-2+Bio table and existing ablation figures.

Both compile without undefined references/citations or overfull boxes using the unmodified single-column ICLR template. Main method content occupies approximately 2.27 pages across three physical pages; the experimental section is eight pages including references. The appendix is separate from the main-method page budget. Earlier findings remain an unchanged historical record; this entry supersedes earlier selections of V4+Bio as the proposed method.

## 2026-09-24 — Released CLIPZyme checkpoint: matched embedding diagnostics

At the user's request, the existing embedding-space analysis was applied to the **released CLIPZyme checkpoint** and refreshed for the **current dictionary-free CERSEI manuscript checkpoint**. The earlier September 23 geometry package used older checkpoints/fusion settings and is retained as historical evidence. This new comparison uses CERSEI fusion multiplier 2, residual cap 0.5, and dictionary weight 0; it performs no training or model selection.

Both methods use the same official EnzymeMap source split (34,427 training associations); CERSEI collapses accession aliases with identical sequences to 34,180 unique training pairs. The analysis reuses the fixed 500-reaction query subset and all 1,357 test-positive unique sequences. Retrieval preserves the complete 261,907-accession enzyme bank and 1,521-reaction bank. Model-specific dimensions and pretrained inputs remain different, so this is a matched downstream-data method comparison, not an architecture-only causal experiment.

| Diagnostic | Released CLIPZyme | Current CERSEI | Paired difference [95% family-bootstrap interval] |
|---|---:|---:|---:|
| EC3 agreement@10 after close sequence-component exclusion (%) | 25.91 | 28.71 | +2.80 pp [0.33, 4.29] |
| Reaction-rule agreement@10, participant Tanimoto < 0.5 (%) | 16.94 | 19.26 | +2.32 pp [0.19, 4.21] |
| R→E known-positive coverage@50 (%) | 27.58 | 27.96 | +0.38 pp [−5.42, 5.53] |
| E→R known-positive coverage@50 (%) | 59.19 | 56.91 | −2.28 pp [−9.34, 8.02] |
| Same-rule cosine gap after participant matching | 0.247 | 0.279 | +0.031 [−0.029, 0.092] |

The strongest comparative evidence is improved local functional and transformation-related neighborhood agreement under the stated exclusions. Alignment coverage and the tightly matched transformation contrast do not establish a consistent advantage; difficult distractors remain problematic for both models. Cosine-margin magnitudes are not calibrated across the two embedding spaces. The intervals are unadjusted, conditional on one checkpoint per model, and exploratory after repeated benchmark inspection. Native EC3 annotations cover only 720/1,357 protein sequences; no missing labels are imputed.

The [complete report](runs/clipzyme_embedding_comparison_20260924/README.md) includes the full protocol, SciencePlots figures, query-level CSVs, every evaluated metric, checkpoint hashes, and a companion notebook. [Independent validation](runs/clipzyme_embedding_comparison_20260924/validation.json) checks source-data parity, nearest neighbors, macro means, and rank-based tie handling. The work is isolated on branch `research/embedding-public-checkpoints-20260924`; no original checkpoint, old analysis, or manuscript table was overwritten.

## 2026-09-24 — Public checkpoint embedding comparisons


**Status: completed scoped analyses for Horizyn, CREEP, CLEAN, and EnzymeCAGE, alongside released CLIPZyme and current manuscript CERSEI.** No new training or checkpoint selection. Work is on `research/embedding-public-checkpoints-20260924`.

[Full analysis, tables, captions, and limitations](runs/public_embedding_comparison_20260924/README.md) · [Reproducible notebook](runs/public_embedding_comparison_20260924/analysis.ipynb) · [Validation](runs/public_embedding_comparison_20260924/validation.json)

### Matched downstream-data comparisons

Original Horizyn was analysed on all three ReactZyme splits and EnzymeMap; the two-modality CREEP refits were analysed on all three ReactZyme splits. The graph audit confirms identical downstream training association sets for CERSEI, Horizyn, and CREEP on each ReactZyme split. These are our refits of public methods, not the authors’ originally released downstream weights. CREEP retains the disclosed exact-edge and participant-set adapters. Complete methods still differ in pretrained inputs, loss, capacity, and optimization budget.

All CERSEI geometry below is from the **current manuscript recipe** (fusion multiplier 2, residual cap 0.5, dictionary weight 0, no added biological-label supervision). It replaces use of the older configuration for these comparisons. Current score coordinates reproduce within 6e−7; CREEP reproduces its original full matrices exactly after replaying its native batch order and attention kernel.

EC3 neighborhood agreement@10 (%) after excluding same 50%-identity components, on identical full candidate banks and the fixed diagnostic query subsets:

| Partition | CERSEI | Horizyn | CREEP | CLIPZyme |
| --- | ---: | ---: | ---: | ---: |
| Time | 51.61 | 51.27 | 51.28 | — |
| Enzyme split | 57.86 | 53.51 | 54.23 | — |
| Reaction split | 60.52 | 55.12 | 55.75 | — |
| EnzymeMap | 28.71 | 27.53 | — | 25.91 |

CERSEI has stronger local functional neighborhoods on the enzyme and reaction splits; the time split is close. **This does not imply superior exact-pair retrieval: Horizyn still exceeds current CERSEI in all six full-test ReactZyme MRR cells.** Both directions, positive coverage@10/50, random and hard-unannotated margins, reaction organization, and uncertainty are reported. The EnzymeMap chemistry-matched analysis additionally uses the same 427 queries and 55,704 comparisons for every model.

![Matched-data functional neighborhoods](runs/public_embedding_comparison_20260924/01_matched_functional_neighborhoods.png)

Figure: EC3 agreement@10, after excluding close sequence components. Bars show family-weighted uncertainty, not training-seed variability. See the report for paired differences and candidate/query counts.

![Matched-data alignment](runs/public_embedding_comparison_20260924/02_matched_alignment.png)

Figure: known-positive coverage@10 in both directions with full candidate pools. Coverage differs from MRR; favorable functional clustering does not guarantee favorable exact ranks.

### CLEAN transfer control

The CARE-released CLEAN checkpoint was evaluated with its native frozen ESM-1b means and strict-loaded 128-dimensional head. The primary control removes exact CLEAN training-sequence matches from both query and candidate banks; native Euclidean geometry is also reported. CERSEI has higher observed functional-neighborhood agreement on these common banks. This is an unmatched-training-data transfer control, not an architectural superiority test.

![CLEAN transfer control](runs/public_embedding_comparison_20260924/03_clean_transfer_control.png)

Figure: EC3 agreement after removing CLEAN training-sequence overlap and close query–candidate sequence components. These fixed-subset banks are smaller than the full banks above. Three empty legacy EnzymeMap sequence strings were recovered from the original screening FASTA and verified against SHA256-derived protein IDs; none has an EC annotation. The original metadata artifact was preserved.

### EnzymeCAGE scope and native-panel result

Existing pockets cover only 37 / 10 / 13 / 0 proteins in Time / enzyme / reaction / EnzymeMap, with extensive exact training overlap. A full shared-pool EnzymeCAGE benchmark comparison is therefore **not available**. Undefined metrics remain missing.

To provide a substantive released-model diagnostic, the complete native P450 panel was analysed: **490 enzymes, 191 reactions, 93,590 pairs, and 318 listed positives**. We extracted the independent pocket branch before cross-attention and separately analysed the verified pretrained seed42 pair scores. Raw R→E positive coverage is 4.62% at 10 and 29.13% at 50; E→R is 10.70% and 35.78%. These omit the external prior and fine-tuning and are not replacements for the paper’s official filtered metric. The pooled pocket branch has weaker shared-reaction neighborhoods than input ESM-C pocket means, which does not characterize the complete pair-conditioned model.

The native panel contains 315/490 exact training-sequence matches. None of its 318 positive pairs matches a training-positive or training-negative pair by exact sequence plus canonical reaction string; chemically equivalent aliases are not ruled out. This is not evidence of generalization to unseen sequences.

![EnzymeCAGE native diagnostic](runs/public_embedding_comparison_20260924/04_enzymecage_native_panel.png)

Figure: left, shared-positive-reaction neighborhood agreement for the pocket branch and ESM-C control; right, native pair-score positive coverage. This is a standalone native-panel diagnostic, not a matched-data CERSEI comparison.

### Interpretation and reproducibility

The defensible finding is stronger local functional organization on selected held-out partitions, with mixed exact-pair retrieval. The analysis does not prove wet-lab applicability or a win over every competitor. Main uncertainty intervals retain observed annotation classes using 1,000 family-level exponential-weight draws. Original whole-family resampling intervals are preserved as a sensitivity analysis; weak EnzymeMap differences are sensitive to that convention. All comparisons remain exploratory and one-seed.

Four SciencePlots figures are available as PDF/SVG/PNG with the requested palette and no titles, subtitles, or footnotes. Nine validation groups passed, including independent table arithmetic, training-graph identity, finite vectors, coverage, and score parity. The notebook’s four read-only code cells executed successfully. The manuscript was not edited for this analysis.


## 2026-09-24 — CREEP EnzymeMap preparation and four-GPU launch

Prepared and launched the two-modality CREEP refit on the official EnzymeMap split. [Protocol, commands, runtime checks, and progress locations](runs/enzymemap_public_creep_20260924_seed42/README.md). This entry records setup and launch, **not completed performance**.

Training retains all 34,427 official training rows, native ProtT5/rxnfp backbones with fresh 256-dimensional heads, padded-mean/CLS pooling, and symmetric EBM-NCE (temperature 0.1, one cyclic negative, no training-vector normalization). Unlike the ReactZyme participant adapter, EnzymeMap supplies directed reaction strings. No EC-derived positive pairs, EC text, generative loss, or supervised checkpoint transfer is added. The budget is 40 epochs with Adam 1e-5; the global batch is 256 across four H200s, a disclosed throughput adaptation from the native batch 16 and prior ReactZyme batch 64.

Validation every three epochs, plus epoch 40, evaluates BEDROC85, BEDROC20, EF5 and EF10 against the complete screening library. Full-library validation BEDROC85 selects the checkpoint. A separate evaluator reads test labels only after selection; both full-library and training-ID-excluded test settings use that same checkpoint. No MRR selection or test-driven tuning is implemented.

Native-loss/gradient parity passed with maximum gradient error below 1.2e-7, including uneven batches without duplicated training rows. Training probes measured about 1.5 seconds per update and 125 GiB allocated on each GPU. Inference batch 256 was slightly faster than 512 or 768 and is retained. All 222,985 unique protein sequences and 16,776 reaction inputs are tokenized with zero unknown reaction tokens; native 512-token truncation affects 26,854 proteins and 116 reactions, without dropping candidates. Nine ineligible validation input rows are ignored as in the existing screening protocol; eligible validation and test reaction IDs are disjoint from training reactions.

The implementation is committed as `424f67c` on branch `research/creep-enzymemap-20260924`. Existing ReactZyme CREEP results remain unchanged and are available for all three splits. Final EnzymeMap results are pending.

## 2026-09-24 — Functional neighborhood curves across k=1–50

Completed two additional SciencePlots figures using the current dictionary-free manuscript CERSEI and the previously audited comparator embeddings. Analysis ran on CPU without interrupting the four-GPU CREEP EnzymeMap training. [Figures, captions, full tables and interpretation](runs/public_embedding_comparison_20260924/neighborhood_curves/README.md). Sources, per-query neighbors, per-class agreement, and all 1,600 plotted values are saved beside the figures.

![EnzymeMap neighborhood curves](runs/public_embedding_comparison_20260924/neighborhood_curves/05_enzymemap_neighborhood_curves.png)

Figure: EnzymeMap macro functional agreement across neighborhood sizes, with detected close sequence components excluded on the enzyme side and participant Tanimoto ≥0.5 excluded on the reaction side. Same queries and banks for CERSEI, Horizyn and CLIPZyme. Frozen ProtT5 and ReactionT5 are input controls. Protein geometry uses 1,357 test-positive unique sequences (720 EC3-eligible), not the complete screening library; reaction geometry uses 500 queries against 1,521 reactions. Curves are descriptive; no significance claim is made.

CERSEI leads both comparators in enzyme agreement at every tested k. At k=10: **28.71% CERSEI / 27.53% Horizyn / 25.91% CLIPZyme**. In reaction space, CERSEI leads both from k=4 onward; Horizyn leads CERSEI at k=1–2 and CLIPZyme at k=1–3. At k=10: **19.26% / 17.36% / 16.94%**. Thus the reaction advantage concerns broader local neighborhoods, not the closest neighbor in every case. CREEP EnzymeMap is not included while its refit is pending.

![ReactZyme neighborhood curves](runs/public_embedding_comparison_20260924/neighborhood_curves/06_reactzyme_neighborhood_curves.png)

Figure: matched functional-neighborhood comparison across Time, Enzyme-Similarity and Reaction-Similarity splits. Top: enzyme EC3. Bottom: reaction EC3 inherited from held-out associations, not independently annotated mechanism. All plots use original embedding-space cosine, with no UMAP projection, new fit, or checkpoint selection.

CERSEI exceeds Horizyn and CREEP in enzyme-space agreement at **all 50 neighborhood sizes on the enzyme and reaction splits**. At k=10 these are **57.86% / 53.51% / 54.23%** and **60.52% / 55.12% / 55.75%**, respectively. Time-split enzyme neighborhoods are close. Reaction-space results remain mixed: CREEP exceeds CERSEI at **46/50 k values on the reaction split**, and at k=45–50 on the enzyme split. The nested k values are correlated; the counts describe persistence rather than independent statistical tests.

**Supported conclusion:** stronger local functional organization, especially in enzyme embeddings, under these specific benchmark splits. **Unsupported conclusion:** universal superiority of both embedding spaces, every functional class, exact-pair ranking, or wet-lab activity. The existing full ReactZyme all-positive MRR results still favor the local Horizyn refit over current CERSEI in all six cells. Matched downstream data do not equalize backbones, objectives or optimization budgets.

All **64** k=10/50 comparisons match saved summary values (maximum absolute difference **3.33e−16**), with fixed query counts across k and exclusion masks checked. The prior fixed-class family-weighted bootstrap is degenerate for EnzymeMap reaction-rule labels; these figures therefore show no confidence bands. PDF/SVG/PNG exports and `scripts/cersei_neighborhood_curves.py` accompany the analysis.

### Protocol and scientific questions clarified

The [expanded explanation](runs/public_embedding_comparison_20260924/neighborhood_curves/README.md#purpose-what-are-we-looking-for) states what each diagnostic is intended to test, and defines the full procedure from fixed checkpoints to plotted agreement: query versus bank, deterministic query sampling, exact eligible counts, label provenance, connected-component sequence exclusions, participant-fingerprint chemistry exclusions, cosine neighbor ranking, multilabel agreement, and macro averaging. It includes equations, a worked example, curve-reading guidance, and the relation to retrieval, full-library screening and Case 1. Future figure regeneration incorporates the same prose from `protocol_explanation.md`.

Two interpretation limits are now explicit: the similarity filters apply within the diagnostic query/candidate pools, not as an additional training-set decontamination step; and fixed-k agreement is constrained by how many function-sharing candidates remain, particularly for rare classes. These plots do not fit global clusters, establish a chance-adjusted score, or isolate which architectural component causes an advantage. This documentation update changes no embeddings, metric values or figures and launches no additional experiments.

## 2026-09-24 — Enzyme functional organization across EC1–EC4

Completed the [all-level enzyme analysis](runs/public_embedding_comparison_20260924/ec_levels/README.md) using the same frozen CERSEI, Horizyn, CREEP/CLIPZyme and ProtT5 exports. The primary analysis evaluates EC1, EC2, EC3 and EC4 separately on each partition, using the existing fixed query samples and same-component sequence exclusion. A second analysis holds the EC4-annotated query/candidate cohort and neighbor rankings fixed across hierarchy levels. No model was retrained; all computations ran on CPU while CREEP EnzymeMap training continued on four GPUs.

![EC1–EC4 enzyme neighborhoods](runs/public_embedding_comparison_20260924/ec_levels/07_enzyme_ec1_ec4_agreement.png)

Figure: class-balanced agreement@10, with model comparisons on identical eligible pools within each cell. Annotation coverage is reported per level; the finer-level populations can be smaller. The report includes the complete 16-panel k=1–50 curve matrix and the common-cohort query-mean control as separate SciencePlots figures, each in PDF/SVG/PNG.

| Partition | CERSEI EC1 (%) | CERSEI EC2 (%) | CERSEI EC3 (%) | CERSEI EC4 (%) | EC4 interpretation at k=10 |
| --- | ---: | ---: | ---: | ---: | --- |
| Time | 86.35 | 57.84 | 51.61 | 27.11 | Horizyn higher: 27.94 |
| Enzyme split | 88.97 | 66.51 | 57.86 | 38.07 | Essentially tied with Horizyn: 38.04 |
| Reaction split | 94.98 | 75.55 | 60.52 | 34.50 | Higher than Horizyn 33.27 and CREEP 32.53 |
| EnzymeMap | 69.54 | 34.51 | 28.71 | 8.99 | Essentially tied with Horizyn: 8.97; CLIPZyme 8.62 |

CERSEI has the highest observed EC1 macro agreement@10 in all four partitions. At EC2 it leads in three; CREEP leads on Time (58.35% versus 57.84%). At EC3 the earlier results reproduce exactly. **The larger coarse/intermediate functional-neighborhood gains do not extend uniformly to complete EC-number agreement.** The EC4 differences of +0.03 pp on the enzyme split and +0.02 pp on EnzymeMap are not persuasive evidence of a meaningful advantage.

The common-cohort control keeps 1,746 / 1,891 / 1,609 / 720 fully annotated queries for Time / enzyme / reaction / EnzymeMap. Complete EC4 labels are truncated to form EC1–EC3, so query identities, candidate identities, rankings and exclusions are identical across levels. Both macro and ordinary query means are saved. The query-mean EC4 comparison also favors Horizyn on Time and slightly on the enzyme split; changing annotation coverage alone therefore does not explain away the fine-resolution limitation. Macro and query-mean rankings differ in some cells, and neither is substituted silently for the other.

A separate table extracts all EC1–EC4 results for CERSEI versus CLEAN and ESM-1b on the smaller CLEAN training-overlap-filtered transfer banks. These unmatched-training controls remain separate. CREEP EnzymeMap and full common-pool EnzymeCAGE results are not fabricated or replaced by interim/partial models.

All **128** primary k=10/50 checks passed against the existing summaries (maximum absolute difference **5.55e−16**). Every EC3 per-query curve reproduces the previous analysis exactly. The checks also confirm fixed query denominators across k, sequence/self exclusions, at least 50 eligible neighbors, and pointwise nonincreasing agreement with hierarchy depth on the common complete-annotation cohort. Sources, coverage, per-query/per-class arrays and all **6,400** curve rows are saved under `runs/public_embedding_comparison_20260924/ec_levels/`.

### Consolidated report in the original Markdown

The [original embedding-analysis README](runs/public_embedding_comparison_20260924/README.md) now contains all completed analyses directly, with all nine figures, the detailed protocol, complete EC1–EC4 and common-cohort tables, both-modality neighborhood curves, alignment/MRR context, chemistry-matched reactions, CLEAN transfer controls, and the EnzymeCAGE coverage/native-panel diagnostics. It no longer requires switching documents to read the completed experiments. CREEP EnzymeMap final testing and export are explicitly pending in the experiment inventory; incomplete common-pool EnzymeCAGE coverage is not presented as a completed baseline comparison.

The original report generator now preserves this consolidation on regeneration. Companion documents remain source artifacts, and `consolidated_report_receipt.json` records their hashes, verifies all local links, and confirms that the nine embedded figure paths are distinct. This step consolidates documentation only; it does not rerun experiments or change any metric.


## 2026-09-25 — Cross-modal EC3 recovery beyond recorded partners

Completed the requested reaction-to-enzyme diagnostic on every EC3-eligible test query, using the unchanged dictionary-free manuscript CERSEI, local Horizyn/CREEP refits and released EnzymeMap CLIPZyme. The [full analysis](runs/cersei_crossmodal_functional_recovery_20260925/README.md) is also incorporated into the [original consolidated report](runs/public_embedding_comparison_20260924/README.md), now with eleven figures. No training or checkpoint selection used these diagnostic outcomes.

The primary exclusion removes partners in the union of all locally released association records and exact-sequence aliases. A stronger control also removes every detected 50%-identity partner component (80% bidirectional coverage, E≤1e−3). Reaction EC3 labels are the union over recorded partners, including partners outside the bank. Missing EC3 candidates are excluded. We class-macro-average any-shared-label recovery, retain zero-alternative queries, and show all k=1–50 with exact random expectations and attainable ceilings. The EnzymeMap annotated bank contains 131,409 of 261,907 accessions; accessions remain the candidate unit.

| Partition | CERSEI recovery@10 (%) | Strongest evaluated comparator (%) | CERSEI with component exclusion (%) | Strongest comparator with component exclusion (%) |
| --- | ---: | --- | ---: | --- |
| Time | 31.70 | CREEP 31.14 | 29.82 | CREEP 29.64 |
| Enzyme-Sim | 29.34 | CREEP 26.90 | 27.57 | CREEP 25.34 |
| Reaction-Sim | 24.32 | Horizyn 23.60 | 23.67 | Horizyn 22.94 |
| EnzymeMap | 31.69 | CLIPZyme 27.55 | 30.11 | CLIPZyme 25.46 |

The clearest conditional evidence is on Enzyme-Sim and EnzymeMap: paired reaction-family multiplier intervals against every evaluated comparator are positive under both filters. Time versus CREEP and Reaction-Sim versus either baseline remain uncertain. CERSEI leads all k on EnzymeMap, but CREEP overtakes it on Time from k=18 under partner exclusion. The sequence control affects only 28 of 381 Reaction-Sim queries, versus 894 of 1,067 EnzymeMap queries. Six EnzymeMap queries are omitted from the stronger control because a recorded partner sequence is unavailable; their IDs are still excluded from primary ranking. Exact aliases of unavailable sequences cannot be audited in the primary control.

**Stage insight:** functional organization is already strong in phase 1. Native refinement changes primary recovery by −0.50/−0.63/−0.83/+0.59 percentage points in Time/Enzyme-Sim/Reaction-Sim/EnzymeMap order. Applying the deployed residual heads at fixed fusion 2 changes it by −0.40/−0.26/−0.46/+1.03 points. These interventions use the same fitted encoders/heads and the actual manuscript coefficients; they are not retraining ablations. The complete five-stage comparison and uncertainty plot are in the report and paper appendix.

This supports EC3-compatible candidates beyond locally recorded partner identities, not newly validated catalytic associations, exact substrate specificity, or absence of training exposure. Most eligible Time/Enzyme-Sim reaction queries occurred in training; Reaction-Sim/EnzymeMap queries did not under exact benchmark identifiers. Existing unfavorable exact-association retrieval/coverage results remain documented.

All eight panels and 58 method/stage evaluations passed independent checks. Maximum independent macro-aggregation discrepancy is 8.88e−16; hash-selected full-sort tie checks have zero error. Every saved top-50 list passes label, partner/alias and component exclusions. Final CERSEI reproduces all four original benchmark score matrices exactly. The new SciencePlots figures use blue/orange/pink/green and contain no plot titles or footnotes. The paper compiles with the added section, protocol tables, stage appendix and cross-references; pre-existing bibliography warnings remain. Reproducible scripts, score/cohort hashes, paired intervals, per-query and per-class results are saved with the analysis.


## 2026-09-25 — Main-paper embedding-map and functional-recovery figure

Built the requested FGW-CLIP Figure-2-inspired [composite](runs/cersei_joint_space_figure_20260925/joint_space_composite.pdf), using the same 720 EC-annotated test-associated EnzymeMap sequences for Horizyn, CREEP, CLIPZyme and current manuscript CERSEI. All six observed EC1 classes and 28 multiclass proteins are retained. The adjacent panel shows the existing component-excluded EC3 recovery@10: 25.25% Horizyn, 21.49% CREEP, 25.46% CLIPZyme and 30.11% CERSEI; random is 1.62%. The paired CERSEI−CLIPZyme interval is [+3.04,+6.87] percentage points around a +4.65-point difference.

The projections are illustrative enzyme-only views, while the measured recovery is cross-modal. The maps show functional structure in several models, including CLIPZyme; they are not used to claim a universal superiority of CERSEI's two-dimensional separation. No seed or sample was chosen for appearance. The primary seed and two sensitivity seeds were fixed before fitting and all 12 maps are retained. Broad EC1 colors are distinct from the finer EC3 labels used by the quantitative panel.

The map pool contains 720 unique sequences; the retrieval pool contains 1,067 reaction queries and initially 131,409 annotated accessions. Query-specific partner/component exclusion applies only to retrieval. All source identities were checked and reconstructed score discrepancies are below 7e−7. An independent rerun of the primary CERSEI projection reproduces its coordinates exactly. t-SNE retains roughly 70–74% of the original cosine top-10 neighborhoods, underscoring that it is an illustration rather than the evidential metric.

The main analysis is condensed to one composite figure and one functional-recovery analysis. Full k=1–50 curves, both exclusions, phase interventions, EC-level/neighborhood controls, projection seed sensitivity, and unfavorable exact-association comparisons remain in the appendix. Other main-paper sections were left unchanged. The [full report](runs/cersei_joint_space_figure_20260925/README.md) is included in the original consolidated Markdown, now with 13 figures. All plots use SciencePlots, minimal formatting and the requested color family. This was a CPU-only analysis of existing fixed models; it launches no new training.


## 2026-09-25 — Clearer SciencePlots rendering and auditable figure data

Revised the [main analysis figure](runs/cersei_joint_space_figure_20260925/joint_space_composite.pdf) with native SciencePlots serif styling, tighter square map windows, clearer markers, capped intervals and direct percentages. All five panels remain in one horizontal row as explicitly requested. The 720 sequence identities, all saved t-SNE coordinates, projection seeds and existing EC3 recovery values are unchanged. The figure remains illustrative on the enzyme side and quantitative on the cross-modal side; no new performance claim is inferred from the styling.

The [source-data ZIP](runs/cersei_joint_space_figure_20260925/figure_source_data.zip) includes normalized original embeddings, all three projection arrays, exact coordinate CSVs and per-query recovery contributions. The 8,640 coordinate rows reproduce the arrays exactly; sums over 5,335 method/query rows reproduce the plotted class-macro percentages to within 4e-14 percentage points. A data guide explains units, weighting and distinct visualization/retrieval cohorts. The original consolidated report includes these links and still retains all thirteen figures. The manuscript compiles, and its revised main and appendix figures were checked in the actual ICLR layout.


## 2026-09-25 — Quantitative enzyme organization on the exact map cohort

Completed the [matched-cohort clustering and neighborhood analysis](runs/cersei_enzyme_cluster_metrics_20260925/README.md) for CREEP, CLIPZyme and manuscript CERSEI using the same 720 enzymes as the maps. Original-space class-macro EC1 silhouette is 0.0704/0.0668/0.1001 in that order, computed on the identical 692 single-EC1 enzymes. CERSEI's paired silhouette differences are +0.0296 [0.0139, 0.0458] versus CREEP and +0.0333 [0.0183, 0.0459] versus CLIPZyme. EC3 neighbor agreement@10 after same-50%-identity-component exclusion is 25.64%/25.91%/28.71%; paired gains are +3.07 [2.32, 4.07] and +2.80 [2.12, 3.61] percentage points. All k=1–50, per-class and ordinary-query averages, exclusion controls and projection seeds are retained.

The primary EC1 neighbor result is more modest: 72.69%/72.49%/74.27%, with CERSEI–CREEP interval [−0.42, 3.91] percentage points. Six-cluster K-means does not favor CERSEI: mean ARI across all three preset seeds is 0.194/0.161/0.154, and adjusted mutual information also favors CREEP. These counterexamples remain in the report. A defensible claim is greater average EC1 separation and stronger class-balanced EC3 neighborhoods on this cohort, not universal clustering superiority. The EC3 advantage is smaller with ordinary query averaging. Sequence-component filtering does not establish absence of homology or training exposure.

This enzyme-to-enzyme diagnostic is separate from the reaction-to-enzyme recovery experiment and its 30.11% CERSEI score. Conditional family-multiplier intervals do not include model-training uncertainty; the study is exploratory. Source hashes, matched IDs, independent aggregation, exact ties and all 2,076 per-enzyme original-space silhouette values passed verification. Added the complete report to the original consolidated Markdown; its thirteen figures are retained. No retrieval models were retrained and no manuscript files were changed for this diagnostic.


## 2026-09-25 — Fresh manuscript CIRCE inference on the 144-entry Case 1 panel

Completed fresh inference for all four benchmark-trained current CIRCE checkpoints, using fusion multiplier 2, residual coefficient 0.1, and no dictionary scoring or biological-label supervision. All 144 sequence-resolved entries are retained; they comprise 123 unique sequences. The [report](runs/circe_case1_144_20260925/report.md) contains both pool definitions, complete rankings, all positive ranks, frozen provenance and reproduction commands. Raw pretrained features were reused, but every learned encoder and residual-head output was recomputed. No manuscript files were changed.

At 25 unique sequences, paper-supported recovery is 4/12 (Reaction-Sim), 1/12 (Enzyme-Sim), 3/12 (Time), and 7/12 (EnzymeMap), against 2.44 expected randomly. The corresponding 144-entry results are 4/15, 0/15, 2/15 and 6/15 after exact-sequence label propagation. EnzymeMap recovers no paper-supported catalyst in its top 10. Broad assay-conditional AUROC is respectively 0.5141, 0.5835, 0.6220 and 0.4706. All four models are disclosed; none is selected by these Case 1 outcomes.

The Reaction-Sim top-25 paper hits are Tpet WT/5V and Tne WT/M3, all 96.8–99.1% identical to a supervised training sequence in the existing homolog search. This is modest, family-dependent recovery and does not establish broad catalytic generalization. Historical Case 1 results from dictionary scoring or rank fusion must not be attributed to the current recipe. Independent score reconstruction, ranking counts, AUC/AP and candidate/hash checks passed.


## 2026-09-25 — Corrected requested model: CERCEI V4 reproduces 10/12 with and without dictionary

The user clarified that the requested Case 1 model was the original V4 `shared_recipe_alpha04_cap05_v1` Reaction-Sim epoch-10 model, not the later epoch-20 manuscript model used in the preceding run. Fresh inference reproduces all 123 historical selected scores exactly (maximum absolute difference 0), including 10/12 paper-supported catalysts in the top 25 distinct sequences. [Corrected report and full rankings](runs/cercei_v4_case1_144_reproduction_20260925/report.md).

A controlled inference-only ablation retains the exact encoder, residual heads, fusion multiplier 3 and residual cap 0.5 while changing semantic dictionary weight 0.4 to 0. It also recovers 10/12 at 25; paper recovery at 10 improves from 3/12 to 4/12, conditional AUROC changes 0.612875 to 0.611111, and AP changes 0.768252 to 0.767438. Native encoder scores are bit-identical. Thus the earlier 4/12 result cannot be attributed to removal of dictionary scoring: it used a different checkpoint. No new benchmark-wide dictionary ablation was performed.

Both V4 variants retrieve 10/15 paper-supported entries among the first 25 of the original 144-entry ranking, representing eight distinct paper-supported sequences. The 10/12 headline instead uses a shortlist of 25 distinct sequences from the 123-sequence pool. Both evaluation units are reported explicitly. This is a retrospective literature-panel reproduction, not new wet-lab validation. The earlier report is retained with a prominent model-selection correction; no manuscript files were changed.

## 2026-09-25 — Dictionary-free V4 Case 1 in CLIPZyme and Swiss-Prot EC 5 banks

The frozen Reaction-Sim V4 model was screened against 91,050 CLIPZyme and 5,176 reviewed Swiss-Prot EC 5 background sequences after excluding exact matches to its 147,299 unique training protein sequences. We added the 23 distinct resolved sequences represented by the Excel top-25 list to each bank before scoring. The resulting banks contain 91,073 and 5,199 unique sequences. **The model top 100 overlaps the Excel top 25 by 0/25 in both banks**; the best Excel candidate ranks 2,002 in CLIPZyme and 222 in Swiss-Prot EC 5. This fails to extend the 144-entry panel's 7/25 exact-ID priority-list agreement to broader pools. The top Swiss-Prot hit is annotated as a tagatose C3 epimerase, whereas the query is a C4 conversion; the participant-set `A.B>>A.B` query encoding is a plausible, untested contributor. The Excel list is not a measured activity set, so higher-ranked background enzymes are not labelled negatives. Exact sequence exclusion does not establish homology-disjointness; the query's two molecules co-occur in seven V4 training molecule sets. Full protocol, ranked outputs, sequence-annotated top hits, and numerical checks: [Case 1 CLIPZyme/Swiss-Prot report](runs/cercei_v4_case1_clipzyme_swissprot_20260925/report.md).

## 2026-09-25 — Rv1430 substrate specificity with dictionary-free CERCEI V4

The fixed Reaction-Sim V4 encoder and phase-2 head, with fusion multiplier 3 and no semantic dictionary, ranked eight Rv1430 p-nitrophenyl ester hydrolysis reactions. The model top two were pNPC8 (71% measured full-length activity) and pNPC10 (11%), whereas the measured top two are pNPC6 (100%) and pNPC8 (71%). pNPC6 ranked fourth. Spearman correlation across eight substrates is 0.708, matching the older Horizyn checkpoint only after rounding; the older checkpoint's top-two ranking is different and must not be attributed to V4. Rv1430 is an exact match to one enzyme in V4 supervised training, although none of the eight p-nitrophenyl ester substrate molecules occurs exactly in its training molecule sets. This is a seen-enzyme substrate-specificity check, not new-enzyme generalization. [Full V4 report and rankings](../case_studies/out/rv1430/v4_no_dictionary_20260925/report.md).

## 2026-09-25 — Original V4 without dictionary on Rv1430 across training partitions

The three additional original V4 checkpoints requested by the user have now been evaluated on the same full-length Rv1430 sequence and eight substrate hydrolysis reactions. No retraining or activity-based parameter selection was performed. Reaction-Sim remains strongest: Spearman 0.708, with the measured optimum pNPC6 ranked fourth. Enzyme-Sim obtains 0.000 (optimum fifth), Time −0.171 (seventh), and EnzymeMap −0.366 (fifth). Reaction-Sim retrieves one of the two measured top substrates in its top two; the other models retrieve none. These results do not show robust substrate-preference transfer across checkpoints.

Rv1430 is an exact supervised training-sequence match for all three ReactZyme models, but not EnzymeMap. None of the eight exact substrate molecules occurs in the ReactZyme training partitions; EnzymeMap includes pNPC2 and pNPC4. Homologue absence was not established. All models use dictionary weight zero and effective residual coefficient 0.1; fusion multiplier 3 is applied once, including the EnzymeMap checkpoint's baked calibration. Reaction preprocessing and chemistry schemas follow each checkpoint's training representation. Independent rank-correlation calculations agree, the earlier Reaction-Sim scores are reproduced exactly, and the smallest adjacent score gap across runs exceeds 0.00094. This is one retrospective enzyme panel, not evidence of broad superiority or failure. [Report, frozen registry, all scores, and numerical validation](../case_studies/out/rv1430/v4_other_training_data_20260925/report.md).

## 2026-09-25 — Original Reaction-Sim V4 without dictionary on phosphatase and dehydrogenase

The same frozen original V4 used for Rv1430 was evaluated on all 22 AtPAP15 substrates and all 25 HbADH2 transformations, with oxidation (19) and reduction (6) evaluated separately. Fusion multiplier 3, effective residual coefficient 0.1, and dictionary weight zero were retained; no retraining or activity-based selection was performed. Spearman correlations are 0.0949 (AtPAP15), 0.1326 (HbADH2 oxidation), and 0.0286 (HbADH2 reduction). Model-first substrates are PEP (92% measured activity), 1-propanol (64%), and propanal (20%); measured optima pNPP, 1-butanol, and butyraldehyde rank 6, 8, and 3. The results provide little evidence of accurate within-enzyme activity ranking.

AtPAP15 is an exact supervised training-sequence match; the supplied HbADH2 sequence is not. Exact substrate matches, preserving charge/stereochemistry, are 0/22 and 17/25, respectively; no homologue exclusion is claimed. HbADH2's 368-residue sequence was reconstructed from a thesis alignment, while the source text reports 367 residues. The supplied core reaction inputs omit NADP+/NADPH; diol products are inferred. AtPAP15 uses the full precursor and interpreted phosphate-release reactions, including net complete phytate hydrolysis. These uncertainties are documented rather than silently corrected.

Methanol lacks cached ChIRo features but remains in the 19-substrate oxidation evaluation using the checkpoint's trained missing-modality handling. Five forward/reverse HbADH2 pairs have exactly identical scores because the original Reaction-Sim representation uses unordered participant sets. Thus this checkpoint cannot model direction-dependent preferences for those pairs. All metadata values and sequences were checked against the local source workbooks; independent rank correlations agree and saved embedding dot products reproduce scores within 6e-8. [AtPAP15 full report and rankings](../case_studies/out/atpap15/v4_no_dictionary_20260925/report.md) and [HbADH2 full report and rankings](../case_studies/out/hbadh2/v4_no_dictionary_20260925/report.md).

## 2026-09-25 — Requested ex-novo feature rerun of AtPAP15 and HbADH2

At the user's request, both preceding case studies were repeated with all frozen features recomputed from their supplied sequences and reaction strings. ProtT5, ReactionT5v2, Uni-Mol2 conformers/embeddings, and ChIRo conformers/embeddings were extracted afresh across four GPUs; chemistry vectors were regenerated with the unchanged training-fitted schema, and SLEEC was evaluated on the new ProtT5 features. No previous case-study feature HDF5 or persistent molecule cache was used. The original Reaction-Sim V4 checkpoint, refinement head, and no-dictionary coefficients were unchanged. Extraction commands, timestamps, output hashes, logs, rankings, and independent validation are saved in the fresh output directories.

All three rankings are unchanged: AtPAP15 Spearman 0.094915 with pNPP at rank 6; HbADH2 oxidation 0.132572 with 1-butanol at rank 8; reduction 0.028571 with butyraldehyde at rank 3. Maximum absolute score changes relative to the earlier CSVs are 0.000864 and 0.000210, respectively. Thus the weak activity agreement is not resolved by recomputing the features. Fresh ChIRo skips formaldehyde (39/40 HbADH2 molecules encoded), but retains methanol's available signal in the participant-set representation; all 25 HbADH2 reactions remain scored. The sequence and reaction-construction limitations in the preceding entry still apply. The old reports are marked superseded. [Fresh AtPAP15 report](../case_studies/out/atpap15/v4_no_dictionary_fresh_20260925/report.md) · [Fresh HbADH2 report](../case_studies/out/hbadh2/v4_no_dictionary_fresh_20260925/report.md).

## 2026-09-25 — Testosterone panel, original Reaction-Sim V4, fresh features

The requested testosterone study is complete with all four feature models recomputed ex novo on four GPUs, followed by the same frozen Reaction-Sim V4 encoder/head, fusion multiplier 3, effective residual coefficient 0.1, and no dictionary. All 189 combinations of 21 distinct supplied P450 sequences and 9 transformations were scored. The workbook has 23 strain labels: AP11/AP10 and AP60/AP38 share their respective supplied sequences. Sequence-level aggregation yields 40 recorded enzyme–reaction pairs; the AP38 representative retains both M (AP38) and H (AP60) annotations at 6β without double-counting the pair.

Reaction-to-enzyme macro AUROC is 0.4562, mean AP 0.3002, and macro recall@10 0.4436. Recovery is selective: 6β AUROC 0.850, first recorded rank 1, 5/5 recorded enzymes at top10; 12β AUROC 0.8519, first recorded rank 3, 3/3 at top10. In contrast, 2α and 16α recover 0/1 and 0/2 at top10. For enzyme-to-reaction ranking, all 21 enzymes choose 15β first (minimum lead 0.002434), and macro AUROC is 0.4526. This broad common preference limits a claim of accurate enzyme-specific regioselectivity; the positive screening subsets should not obscure the weak overall result.

Three supplied sequences match supervised training exactly (AP10, AP95, BP113); homology-disjointness is not established. AP95 is truncated from 1049 to 1022 residues by the standard ends-and-center policy. All 10 unique organic molecules were freshly encoded successfully by Uni-Mol2 and ChIRo. The 7β product SMILES has unspecified stereochemistry, and the 17-position transformation is oxidation, not hydroxylation. H/M/L represent conversion bands, not exact activity values; unlisted associations are treated as background only for these retrieval diagnostics, not confirmed inactive pairs. All 189 CSV scores/ranks, positive counts, input workbook mappings, feature receipts, independent AUC/AP calculations, and embedding score reconstruction passed validation. [Full report and both ranking tables](../case_studies/out/testosterone/v4_no_dictionary_fresh_20260925/report.md).
