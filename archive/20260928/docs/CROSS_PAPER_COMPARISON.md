# Enzyme–reaction comparison: published results and released checkpoints

**Status: 2026-09-20. The new EnzymeCAGE comparison matches its labeled
training split, but F3 fails the external P450 panel.** The ReactZyme result
compares our scores with published numbers under incompletely reconciled
protocols. Earlier EnzymeCAGE P450 experiments trained F3 on positives only;
a later fresh F3 run uses all **1,445,915** original positive and negative
training rows and selects on original validation. Its P450 Top-24 is **0.0681**
versus **0.1990** for pretrained EnzymeCAGE. The CYP and measured
nitrilase/esterase experiments compare frozen systems trained on different
datasets. They are transfer diagnostics, not fair architecture comparisons.
The EnzymeCAGE full-label run controls downstream labeled data, while input
modalities and foundation pretraining still differ. The EnzymeMap F3
60-epoch retrain and EnzymeMap-trained phase-2 screen are complete and below
the released CLIPZyme checkpoint. The paper-style 100-epoch checkpoint is
complete and worse on the official screen. The evidence supports neither “beats all
competitors” nor broad real-world catalytic generalization. The complete
experiment record is in [findings.md](../findings.md), and the protocol is in
[CROSS_PAPER_RETRAINING.md](CROSS_PAPER_RETRAINING.md).

## What has been compared

| Test | Our training and model | Comparator | Status and conclusion |
|---|---|---|---|
| ReactZyme Reaction-Sim, Enzyme-Sim and Time; R→E and E→R | ReactZyme-trained F3, phase 2 and Morgan extension | Published TIGER and FGW-CLIP values | Completed paper-point comparison. Phase 2 beats FGW-CLIP's reported MRR in all six cells and all 12 TIGER main-Table-1 methods, but TIGER's Table-3 MLP beats phase 2 on Reaction-Sim E→R. The FGW perfect-ranking rows match our all-positive oracle to four decimals; its code and predictions remain unavailable. |
| EnzymeCAGE P450, 191 reaction queries × 490 candidate IDs | Fresh F3 base and target-trained phase 2 on EnzymeCAGE positives | EnzymeCAGE pretrained and separately P450-finetuned checkpoints | Same test panel, **different training supervision**: the official EnzymeCAGE recipe uses positive and negative rows; F3/phase 2 use positives only. Phase-2 Top-24 is 0.2094 versus pretrained EnzymeCAGE 0.1990, but Top-4 and MRR are lower, the paired interval crosses zero, and one residual seed regresses sharply. |
| EnzymeCAGE Enzyme-405, Orphan-335, terpene and phosphatase | Official test CSVs present locally; F3 test features and candidate protocols not yet completed | EnzymeCAGE | No target-trained evaluation yet. Do not treat the P450 panel as a replacement for these families. |
| FGW-CLIP/CLIPZyme EnzymeMap rule split | Fresh F3 towers trained on all 34,427 official training associations; phase 2 fitted on the same training graph | FGW-CLIP published values and released CLIPZyme checkpoint | The 60-epoch F3, phase-2 and exact 100-epoch final checkpoint are below released CLIPZyme on all official metrics. All 261,907 released candidate IDs are preserved; foundation pretraining differs. |
| Measured nitrilase panel, both directions | Frozen CIRCEv2 and F3 | Released official Horizyn-1 development checkpoint | Same 684 measured pairs, **different training data**. Descriptive transfer comparison only; CIRCEv2's R→E AP is higher, but its E→R AP is lower than both Horizyn-1 and random expectation. |
| Measured esterase panel, both directions | Frozen F3, phase 2 and Morgan | Released official Horizyn-1 development checkpoint | Same 12,470 measured pairs, **different training data**. Descriptive transfer comparison only; Horizyn-1 scores higher in both directions. |
| CYP within-family retrieval | Frozen F3 and biological residual | Released official CLIPZyme, Horizyn-1, EnzymeCAGE and FusionESP checkpoints | Same 45-query pool, **different training data**. Descriptive transfer comparison only; designated catalysts are known, while other candidates are unassayed. |
| EnzymeCAGE original labeled validation and external P450 | Fresh end-to-end F3 towers; all 1,445,915 labeled training rows, unweighted BCE; original-validation checkpoint selection | EnzymeCAGE's supervised-data recipe and pretrained P450 checkpoint | Original validation AUROC 0.6172 versus frozen-F3 0.5305, but P450 Top-24 0.0681 versus EnzymeCAGE 0.1990. All 191 queries choose one top enzyme: reaction-conditioning collapse. Same downstream labels, different architectures and foundation inputs. |
| Literature Case 1 | ReactZyme-trained frozen models | Known literature catalysts | Exploratory transfer evidence. Large-pool Case 1 recovery does not establish generalization. |

Sharing a test panel, ranking candidates and using a released checkpoint do
**not** make a training-controlled comparison. At best, these rows compare
whole deployed systems, including their training corpora, supervision, input
features and foundation-model pretraining. A downstream-data-controlled
comparison would give both methods the same labeled training rows, validation
rules, test queries and candidate pool, with independent seeds. Even that
would compare complete methods rather than isolate architecture unless input
features and backbone pretraining were also controlled.

The architecture for a target retrain is locked to the independent 512-dimensional
enzyme/reaction dual encoder, with the declared residual and anchor extension.
Each paper's training associations, validation selection, candidate IDs and
metric must remain separate. See the [architecture lock](../runs/generalization_20260919_2251/cross_paper_retraining/architecture_lock_v3.json)
and [retraining protocol](CROSS_PAPER_RETRAINING.md). A ReactZyme-trained model
evaluated zero-shot on another paper's test set answers a different question.

## ReactZyme: published-number comparison

The entries below are MRR. Our values use all-positive MRR over the released
candidate pools. FGW-CLIP's Appendix B.2 describes first-positive MRR, but
its published perfect-ranking rows below 1 on multi-positive queries match our
all-positive oracle on all six cells to four decimals
([metric parity audit](../runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_fgw_metric_parity_v1.json)).
This strongly supports metric and query-grouping comparability, although
FGW-CLIP's code and predictions remain unavailable. The FGW-CLIP column
chooses the better of its reported EC Mode and
EC Max values for each cell. The Morgan extension was a later exploration, not
the frozen phase-2 primary.

| Split | Direction | F3 precision matched | Phase 2 | Morgan extension | TIGER ESM2Text | FGW-CLIP best |
|---|---|---:|---:|---:|---:|---:|
| Reaction-Sim | R→E | 0.396318 | **0.415174** | 0.414891 | 0.3185 | 0.3181 |
| Reaction-Sim | E→R | 0.490416 | **0.523981** | 0.519291 | 0.5180 | 0.3804 |
| Enzyme-Sim | R→E | 0.682638 | 0.697468 | **0.700621** | 0.5921 | 0.5300 |
| Enzyme-Sim | E→R | **0.975459** | 0.973676 | 0.974404 | 0.9561 | 0.8683 |
| Time | R→E | 0.568773 | 0.599849 | **0.610047** | 0.3658 | 0.3394 |
| Time | E→R | 0.808228 | 0.835167 | **0.844265** | 0.6902 | 0.5229 |
| Six-cell mean | | 0.653639 | 0.674219 | **0.677253** | 0.573450 | 0.493183 |

The caveat to a universal win is concrete: [TIGER Table 3](https://aclanthology.org/2026.acl-long.1643.pdf)
reports **0.543** MRR for its two-layer-MLP Reaction-Sim E→R variant, higher
than phase 2's **0.523981**. The FGW values come from
[FGW-CLIP v2](https://arxiv.org/pdf/2512.08508); the
[full official evaluations](../findings.md#official-reactzyme-results) give
the other metric cells and protocol qualifications.

## EnzymeCAGE: fixed F3-base retrain on the P450 panel

F3 was retrained on **207,173 EnzymeCAGE training positives** using the
predeclared architecture and four GPUs. Its epoch-6 checkpoint was chosen by
target-validation bidirectional MRR (**0.405478**), without P450 test-based
selection. The run exactly reproduces the prior epoch-6 pilot weights, so it
is a completion and deterministic replay of one seed, not an independent new
seed. The official P450 evaluator uses **191 physical reaction queries** and
**490 candidate IDs**, including duplicate-sequence IDs. These are raw ranking
Top-k known-positive recovery rates; unlisted pairs are not known negatives.
The source training CSV contains **1,445,915 labeled rows**, including
**243,890 positive rows** (207,173 unique positive pairs) and **1,202,025
negative rows**. F3 used only unique positive pairs; the official EnzymeCAGE
training configuration points to this CSV with `Label` and trains with binary
cross-entropy. Thus the F3 run shares source positives and a test panel with
EnzymeCAGE, but **does not share its full supervision**. The released
checkpoint's exact training-run manifest has not been independently
authenticated. The P450 numbers below do not measure a controlled
architecture difference.

| P450 raw ranking | Top-4 (1%) | Top-14 (3%) | Top-24 (5%) |
|---|---:|---:|---:|
| Target-trained F3 base | 0.02618 | 0.09948 | 0.15707 |
| EnzymeCAGE pretrained | 0.02618 | 0.09948 | **0.19895** |
| EnzymeCAGE P450-finetuned, extra P450 supervision | **0.07330** | **0.13613** | 0.18848 |

Against the pretrained EnzymeCAGE arm, the F3 Top-24 difference is
**−0.04188** with a paired-query bootstrap 95% interval of
**[−0.10995, 0.02618]**. This interval reflects query variation, not training
seeds or new datasets. The [official evaluation](../runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_p450_evaluation/summary.json),
[paired comparison](../runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_p450_comparison/summary.json),
and [state replay](../runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_epoch06_state_replay.json)
pin the F3-base result. The phase-2 extension was subsequently retrained on
the same positive-only EnzymeCAGE data; its separate result follows.

## EnzymeCAGE: target-trained phase 2 and all-label diagnostic

The fixed phase-2 residual, train-only density gate and 75% learned/25%
smooth-anchor composition were fitted on the EnzymeCAGE positive training
graph. Primary seed 42 improved original-validation all-positive MRR from
**0.2793 to 0.3080 R→E** and **0.3633 to 0.3691 E→R**. A frozen, label-free
P450 export reproduced the F3-base score matrix exactly before applying the
extension. The official raw known-positive recovery is:

| P450 raw ranking | Top-4 (1%) | Top-14 (3%) | Top-24 (5%) | First-positive MRR |
|---|---:|---:|---:|---:|
| Target-trained F3 base | 0.02618 | 0.09948 | 0.15707 | 0.03028 |
| Target-trained phase 2, seed 42 | 0.01571 | 0.12042 | 0.20942 | 0.02968 |
| EnzymeCAGE pretrained | 0.02618 | 0.09948 | 0.19895 | 0.03710 |

The phase-2 minus EnzymeCAGE pretrained Top-24 difference is **+0.01047**
with paired-query 95% interval **[−0.05236, 0.07330]**. Top-4 and MRR are
lower. Residual seed 17 also gives 0.20942 Top-24, while seed 73 falls to
0.10995; these share the same F3 base and are not independent base-model
repetitions. This mixed retrospective result does **not** establish superiority
or catalytic generalization. The [complete phase-2 report](../runs/generalization_20260919_2251/cross_paper_retraining/phase2_target/README.md)
links the frozen recipe, all scores, official evaluator and seed sensitivity.

An all-label BCE diagnostic then retained all **1,445,915** EnzymeCAGE
training rows and both labels, extracting features for the 34 proteins missing
from the positive-only F3 catalog. The F3 base remained frozen and its
residual towers were trained with unweighted BCE and selected on the original
validation AUROC, matching the official *loss and selection metric*. The
unfitted F3 baseline had AUROC **0.5305**; the best BCE residuals gave
**0.4545/0.4641/0.4411** for seeds 42/17/73. All selected epoch 0 and were
rejected before external P450 scoring. This is a failed **frozen-base
adaptation**, not a complete end-to-end F3 retrain on all labels. The
[all-label audit](../runs/generalization_20260919_2251/cross_paper_retraining/labeled_parity_v1/README.md)
records row coverage, conflicting pairs and the three full histories.

The later **fresh end-to-end F3** run uses these same 1,445,915 labeled
training rows and unweighted BCE, selecting by original validation AUROC.
Its best validation AUROC is **0.61718**, but official P450 Top-4/14/24 is
**0.00524/0.03141/0.06806** versus EnzymeCAGE pretrained
**0.02618/0.09948/0.19895**. Every one of the 191 P450 queries ranks the
same enzyme first. The [completion receipt](../runs/generalization_20260919_2251/cross_paper_retraining/labeled_parity_v1/full_f3_bce_scratch_seed42_gpu_cache/complete.json)
and [official evaluation](../runs/generalization_20260919_2251/cross_paper_retraining/labeled_parity_v1/full_f3_bce_scratch_p450_evaluation/summary.json)
pin this failure. It is matched on downstream labeled rows and loss family,
not on foundation inputs or architecture; the P450 panel was already opened.

## Additional validation against released checkpoints

These comparisons use **actual public model weights and native inference**.
Their training datasets differ. They describe how the released systems transfer
to the same panel; a ranking gap cannot be attributed to architecture or
training method. None was a prospective wet-lab test.

The [measured nitrilase panel](../runs/activity_panels_nitrilase_v2/report/readout.md)
contains 18 enzymes × 38 substrate-derived reactions = **684 assayed pairs**,
including 85 active pairs. The comparator is the
[official Horizyn-1 development checkpoint](https://github.com/dayhofflabs/horizyn)
(the train-split-only evaluation checkpoint, not its full-data inference
checkpoint). The endpoint is query-macro average precision (AP):

| Frozen model | R→E AP | E→R AP |
|---|---:|---:|
| CIRCEv2 ReactZyme, epoch 23 | 0.3785 | 0.1459 |
| CIRCEv2 large corpus, step 16k | 0.3666 | 0.1466 |
| F3 ReactZyme, epoch 29 | 0.3061 | 0.1334 |
| Official Horizyn-1 development | 0.3163 | **0.2357** |
| Random expectation | 0.2412 | 0.1619 |

The apparent R→E CIRCEv2 advantage does not carry over to E→R. This is a
measured substrate-activity panel; products are generated from a fixed
hydrolysis template, so product selectivity is not measured. The full
[score and exposure audit](../runs/activity_panels_nitrilase_v2/report/summary.md)
records all no-positive queries and the remaining homology uncertainty.

The new [measured esterase comparison](../runs/generalization_20260919_2251/esterase_audit/public_horizyn1_dev/README.md)
uses **86 substrate-derived physical reactions × 145 enzymes = 12,470
assayed pairs** with 2,565 detected activities. It applies the same released
Horizyn-1 development checkpoint to every pair, with its native reaction
fingerprints and protein projection. The earlier five-score esterase campaign
used the same fixed labels; no model was selected on them.

| Frozen model | R→E AUROC | R→E AP | E→R AUROC | E→R AP |
|---|---:|---:|---:|---:|
| F3 native | 0.61364 | 0.30889 | 0.44771 | 0.21542 |
| Phase 2 | 0.61310 | 0.30821 | 0.44809 | 0.21514 |
| Morgan extension | 0.62048 | 0.31926 | 0.44892 | 0.21934 |
| **Official Horizyn-1 development** | **0.72047** | **0.41828** | **0.56246** | **0.27293** |

Horizyn-1 exceeds F3 by **+0.10683 R→E AUROC** (paired-query 95% interval
[0.08324, 0.13126]) and **+0.11475 E→R AUROC** ([0.09003, 0.14076]).
The [published training-source audit](../runs/generalization_20260919_2251/esterase_audit/public_horizyn1_dev/training_exposure.json)
finds **0/145 exact sequence** and **0/86 literal physical-reaction string**
matches to its supervised training release. This excludes only exact matches,
not homologs, equivalent chemistry or upstream pretraining exposure. A
[train-only MMseqs search](../runs/generalization_20260919_2251/esterase_audit/public_horizyn1_dev/homology/summary.json)
finds 70/145 proteins with at least 30% identity and 80% query coverage; the
[post-evaluation E→R stratum](../runs/generalization_20260919_2251/esterase_audit/public_horizyn1_dev/homology/performance_strata.json)
without such a hit still favors Horizyn-1, AUROC 0.5466 versus F3 0.4337.
This threshold does not prove novelty. The
large observed gap is a concrete external-generalization failure for the
current F3-derived methods on this one measured family.

The [CYP within-family report](../runs/cyp_baselines_v1/report/summary.md)
uses the same **45 organism-specific queries, 4,135 protein IDs and 10,105
query–candidate pairs** for every completed row. Its released comparators are
[CLIPZyme](https://github.com/pgmikhael/clipzyme),
[official Horizyn-1](https://github.com/dayhofflabs/horizyn),
[EnzymeCAGE](https://github.com/GENTEL-lab/EnzymeCAGE), and the original and
CYP-adapted FusionESP weights distributed by the
[BoltzCYP authors](https://gitlab.com/cyp_pred_repos/boltzcyp). Results are
designated-catalyst retrieval MRR, not activity classification:

| Frozen CYP arm | MRR |
|---|---:|
| F3 | 0.1373 |
| Biological residual, fixed alpha 0.1 | 0.1509 |
| FusionESP original, substrate-only | 0.0915 |
| FusionESP CYP-adapted, extra CYP labels | 0.1231 |
| CLIPZyme original pretrained | 0.0999 |
| Official Horizyn-1 development | 0.2081 |
| EnzymeCAGE original pretrained | **0.3176** |

The F3/residual checkpoint does not win this comparison. **31/45** designated
CYP positive sequences match the recorded ReactZyme training source, and
unassayed candidates are not verified negatives. EnzymeCAGE required generated
pocket inputs for missing released pairs; the
[native-inference provenance](cyp_external.md) distinguishes that recovered-input
setting from exact paper-feature reproduction. The separate
[CLIPZyme](https://github.com/pgmikhael/clipzyme) run is now complete on all
**10,105 pairs**. Its [native score receipt](../runs/cyp_external_v1/clipzyme_pretrained/scores.json)
pins the original checkpoint and code revision. Missing released CIFs were
replaced with exact-sequence AlphaFold or sequence-only ESMFold structures,
verified against the benchmark sequences. This is a recovered-input evaluation,
not a bitwise reconstruction of the paper's original structure features.

## FGW-CLIP: the missing target-data experiment

The [official CLIPZyme EnzymeMap release](https://zenodo.org/records/15161343)
and rule split have been acquired and audited: **34,427/7,287/4,642**
train/validation/test associations and **261,907** screening IDs. The
[manifest](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_manifests_v2/manifest.json)
preserves source candidate order. All **72** empty screening sequences were
restored from UniProt's archive: **50** exact 2022_01 records and **22**
unchanged sequences bracketed by archived records around that release
([rescue receipt](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_unisave_2022_01_rescue_v3/receipt.json)).
The [complete target F3 catalog](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_catalog_v1/preparation.json)
retains every train/validation/test row and **261,907** ordered screening IDs.
Its **222,985** unique screening sequences include **78,287** requiring new
ProtT5 extraction, which is complete. ReactionT5v2 and UniMol2 cover all
16,776 reactions; ChIRo covers 16,753 after a difficult-molecule cache stall.
The fresh F3 fit completed 60 epochs on the official training associations.
Its validation-selected epoch-53 full-pool result is available below, and a
100-epoch last-checkpoint continuation is complete to match the FGW-CLIP
paper's stated EnzymeMap schedule. Its exact epoch-99 checkpoint was saved
after training and scored on the full library. The archived
sequence reconstruction and existing train/test input overlap must be
disclosed in any later same-data comparison.
The [official CLIPZyme notebook](../data/external/cyp_specificity_2026/clipzyme_official/analysis/Results.ipynb)
deduplicates test reactions and excludes ones found in training before
averaging screening metrics. The [evaluation-only protocol receipt](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_screening_evaluation_protocol_v2/receipt.json)
pins **1,521** test queries, **4,544** positive candidate-ID labels, and
**1,337** queries with a positive after the Table-2 training-enzyme exclusion.

The [released CLIPZyme checkpoint replay](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_released_screen_evaluation_v1/summary.json)
reproduces the published baseline to its displayed precision on this exact
protocol. These are fractions for BEDROC and raw enrichment factors:

| Screening arm | BEDROC85 | BEDROC20 | EF5% | EF10% |
|---|---:|---:|---:|---:|
| Released CLIPZyme, Table 1 replay | 0.44694 | 0.62982 | 14.085 | 8.060 |
| Interim F3 epoch 23, Table 1 | 0.43254 | 0.57827 | 12.531 | 7.313 |
| F3 epoch 53, Table 1 | 0.42199 | 0.55662 | 11.995 | 6.923 |
| F3 epoch 99, Table 1 | 0.39937 | 0.52529 | 11.349 | 6.605 |
| EnzymeMap-trained phase 2 on F3 epoch 53, Table 1 | 0.43755 | 0.56763 | 12.170 | 6.980 |
| Directional F3 + MLNCE epoch 19, interim Table 1 | 0.45099 | 0.59366 | 13.015 | 7.200 |
| Directional F3 + per-anchor loss epoch 19, interim Table 1 | 0.46859 | 0.61593 | 13.366 | 7.562 |
| Released CLIPZyme, Table 2 replay | 0.39133 | 0.58862 | 13.397 | 7.808 |
| Interim F3 epoch 23, Table 2 | 0.37254 | 0.52686 | 11.569 | 6.931 |
| F3 epoch 53, Table 2 | 0.35804 | 0.50197 | 10.940 | 6.514 |
| F3 epoch 99, Table 2 | 0.33286 | 0.46732 | 10.267 | 6.161 |
| EnzymeMap-trained phase 2 on F3 epoch 53, Table 2 | 0.37099 | 0.51195 | 11.055 | 6.572 |
| Directional F3 + MLNCE epoch 19, interim Table 2 | 0.40322 | 0.55160 | 12.205 | 6.864 |
| Directional F3 + per-anchor loss epoch 19, interim Table 2 | 0.41493 | 0.57401 | 12.614 | 7.261 |
| FGW-CLIP Table 1, published only | 0.4866 | 0.6669 | 14.91 | 8.18 |
| FGW-CLIP Table 2, published only | 0.4514 | 0.6143 | 13.57 | 7.61 |

The FGW-CLIP rows are transcribed from its [paper](https://arxiv.org/pdf/2512.08508),
not independently rerun. The [interim F3 evaluation](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_interim_epoch23_evaluation/summary.json)
uses all 1,521 eligible queries and 261,907 candidate IDs. Epoch 23 was the
validation leader when this export began; screening results were not used to
select it. It loses to the released CLIPZyme checkpoint on every listed metric.
The [epoch-53 evaluation](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_final_epoch53_evaluation/summary.json)
is the validation-selected result of the 60-epoch run. It loses more than
epoch 23 on the full library despite a higher small-pool validation MRR,
showing that MRR selection is misaligned with this screening endpoint.
For subsequent EnzymeMap runs, the checkpoint-selection criterion is now the
**held-out rule-split validation Table-1 BEDROC85 over all 261,907 library IDs**.
The rule is fixed before considering additional test scores. Validation has
2,652 reaction queries with positives in the library; the training-enzyme
exclusion variant has 2,216 queries over 252,113 IDs and is reported as a
secondary metric. The training loop retains cheap 1,937-enzyme R→E and E→R
MRR only as diagnostics. Directional F3 saves a predeclared five-checkpoint
grid at epochs 19, 39, 59, 79, and 99; full-library validation selects from
that grid. The [validation scorer](../scripts/generalization_clipzyme_f3_validation.py)
reproduces the earlier epoch-53 F3 Table-1 BEDROC85 to within 0.000003:
0.493803 versus 0.493801. The small difference arises from a fresh
reaction-embedding export; all four EF values match exactly.
The epoch-99 full-library validation Table-1 BEDROC85 is **0.47645**, below
epoch 53's **0.49380**. Its
[official test result](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_paper100_final_evaluation/summary.json)
is also worse than epoch 53, **0.39937 versus 0.42199**. This is evidence of
late-epoch deterioration, not evidence that 100 epochs is universally harmful.

The [directional F3 epoch-19 intermediate test](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_directional_epoch19_test_evaluation/summary.json)
was run on 2026-09-20 after its full-library validation completed. That
validation gives Table-1 BEDROC85 **0.51143**, BEDROC20 **0.69221**, EF5
**15.227**, and EF10 **8.653**; validation Table-2 BEDROC85 is **0.45220**.
The fixed epoch-19 snapshot was scored as an intermediate test report before
the final validation-grid selection. This is directional F3 with the original
MLNCE loss, without phase 2 or hubness correction. It is not the newly launched
directional/per-anchor combination. On test, its Table-1 BEDROC85 is slightly
above released CLIPZyme (**0.45099 versus 0.44694**), and Table-2 BEDROC85 is
also higher (**0.40322 versus 0.39133**), but BEDROC20 and both enrichment
factors remain lower in each setting. All four metrics remain below the
published FGW-CLIP rows. These are nominal single-checkpoint differences,
not an established statistically significant or overall benchmark win.
The final epoch-grid criterion remains validation BEDROC85; this additional
test inspection and earlier exploratory test evaluations must be disclosed.

The [combined directional/per-anchor epoch-19 evaluation](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_fast_loss_v1/directional_anchor_allknown/screen_epoch19/test_evaluation/summary.json)
completed on 2026-09-20 at 22:47 UTC. Its **0.46859 / 0.61593 / 13.366 /
7.562** Table-1 test scores improve all four metrics over the directional
MLNCE epoch-19 snapshot. Table-2 scores are **0.41493 / 0.57401 / 12.614 /
7.261**, also higher on all four metrics than directional MLNCE. It exceeds
the released CLIPZyme replay on BEDROC85 in both settings, but remains below
CLIPZyme on BEDROC20 and both EF metrics, and below published FGW-CLIP on
all four metrics. These are intermediate base-F3 results, without phase 2.
The comparison does not isolate the loss alone: the new run also uses the
documented BF16/single-GPU execution recipe, while the earlier directional
MLNCE run used FP32/four-GPU training.

The combination's [full-library validation](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_fast_loss_v1/directional_anchor_allknown/screen_epoch19/validation_evaluation/summary.json)
has Table-1 BEDROC85 **0.50249**, below directional MLNCE's **0.51143** at
the same epoch; Table-2 BEDROC85 is **0.44373** versus **0.45220**. Thus
the validation and test ordering differ at this snapshot. This test inspection
does not change the declared validation criterion or establish additive gains
across datasets/seeds. The fixed snapshot and evaluation scopes were recorded
before its full-library metrics were seen in the run's `request.json`.

The [EnzymeMap-trained phase-2 evaluation](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_phase2_enzymemap_v1/test_evaluation/summary.json)
improves epoch-53 F3 on the official test but remains below released CLIPZyme
on every metric. Its [full-library validation comparison](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_phase2_enzymemap_v1/validation_evaluation/summary.json)
shows only a small Table-1 BEDROC85 gain, **0.49380→0.49438**.
An exploratory alpha=0.1 phase-2 score was marginally better than alpha=0.25
on validation Table-1 BEDROC85, **0.49511 versus 0.49438**, but worse on the
official test, **0.43062 versus 0.43755**. The
[alpha=0.1 test evaluation](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_phase2_enzymemap_v1/test_alpha0p1_evaluation/summary.json)
is recorded as a failed transfer and is not used for further selection.
The [validation-selected top-50 hubness correction](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_interim_epoch23_hubness_validation/selection.json)
gave a small validation MRR gain but lowered interim F3 Table-1 BEDROC85
**0.43254→0.42826** and Table-2 **0.37254→0.37241** on the full pool
([evaluation](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_interim_epoch23_hubness_evaluation/summary.json)).
It is not promoted for the screening comparison.

## Next public-checkpoint comparisons

The practical priority is to use published models whose **code, weights,
evaluation data and candidate definitions** can all be pinned. TIGER and
FGW-CLIP remain literature-only context in this audit: no official runnable
checkpoint was identified for either as of this document date. This is an
asset-availability statement, not a claim that reproduction is impossible.

1. **Complete F3 EnzymeMap training and screening, then run CLIPZyme's native
   terpene evaluation.** The
   [checkpoint, EnzymeMap data, precomputed screening embeddings and official
   evaluation code](https://github.com/pgmikhael/clipzyme) are released. Report
   the paper's R→E task first; a derived E→R ranking is a separate endpoint.
   Missing F3 input features must be resolved without dropping hard examples
   or changing candidate pools. Audit CLIPZyme's negative sampling and
   checkpoint provenance before describing a future result as
   training-controlled.
2. **Complete training-supervision control for EnzymeCAGE, then extend beyond P450**
   by training the full fixed F3 dual encoder with every labeled EnzymeCAGE
   row and BCE, or retraining EnzymeCAGE on the same positive-only graph.
   The first frozen-base BCE residual attempt failed original-validation
   AUROC and should not be promoted. Verify the released checkpoint's exact
   training provenance and report independent base-model seeds. Then use its
   [released pretrained checkpoint and official external-family evaluators](https://github.com/GENTEL-lab/EnzymeCAGE).
   Terpene synthase and phosphatase panels provide different chemistry and
   families. Verify local test labels, feature coverage and training overlap
   before scoring. The official Enzyme-405/Orphan-335 test CSVs are present locally, but
   complete F3 test features and protocol audits are pending. Keep pretrained
   and family-finetuned arms separate.
3. **Extend the measured-assay tests across public checkpoints.** Score the
   released CLIPZyme and EnzymeCAGE pretrained models on the unchanged
   nitrilase and esterase assay rectangles where their structure and reaction
   inputs can be generated without label-based filtering. Report complete
   coverage, R→E and E→R AP/AUROC, and the same input-exposure audit. This
   tests family transfer with actual assay negatives rather than only curated
   positives.

Two other public checkpoint candidates are
[eSFM](https://huggingface.co/SFM-BIIE-ETHZ/eSFM_VC-SFM), a substrate-only
dual encoder trained on ReactZyme, and
[EZHit](https://huggingface.co/deanluo/EzHit), an enzyme–reaction classifier
with released general and family-tuned weights. eSFM is a related but
nonidentical substrate-retrieval task; EZHit's model card still lists its
benchmark-dataset archive as forthcoming. Neither should be substituted for a
fully matched reaction-retrieval comparison without a separately defined,
source-audited test set.

## Generalization claim

The 2026-09-20 single-model follow-ups first added **validation**
negative evidence, followed by the requested test readouts below. On the fixed directional + anchor-balanced epoch-19 base,
disabling chemistry at inference reduces full-library validation BEDROC85
from **0.502495 to 0.202627**. Full-training-graph decoupled phase-2 refinement
scores **0.501529 at 100 steps / 0.475838 at 200**; adding Smooth-AP scores
**0.501538 / 0.477948**. All use the same EnzymeMap training graph. The
unchanged base wins this validation grid. These validation numbers must not
be compared directly to the published test values above. Fresh balanced-sampling, chemistry-dropout and
four-prototype F3 arms are still training, with full-library validation
selection scheduled. [Protocol and completed ablation evidence](../findings.md#single-model-ablations-first-completed-validation-results-2026-09-20-2329-utc).

Following the user's explicit test request, the validation-selected 100-step
phase-2 variants were also tested. The full-graph decoupled control scores
**0.465320 / 0.607185 / 13.125065 / 7.418361** on official test Table 1;
adding Smooth-AP scores **0.465624 / 0.607742 / 13.137031 / 7.430695**
(BEDROC85 / BEDROC20 / EF5 / EF10). Both underperform the fixed directional
+ anchor-balanced base on all four test metrics, including the training-enzyme
exclusion setting. Protocol and both score axes match the released CLIPZyme
replay. These requested intermediate results are exploratory after repeated
test inspection. [Exact test comparison and source receipts](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_single_model_v2/requested_tests_v1/results.md).

The fixed inference chemistry mask also fails on test: Table-1 BEDROC85 /
BEDROC20 / EF5 / EF10 are **0.214392 / 0.289848 / 6.295633 / 3.566865**,
well below the unmasked base. The separate chemistry-dropout training arm
still awaits its first test.

Case 1 is a literature-derived reaction with previously reported catalysts;
this campaign performed no new assay. In a 1,044,768-candidate search, none of
the five original models recovered any of the 12 paper catalysts or 24
paper-plus-patent catalysts within rank 1,000. The later Morgan extension
improved a smaller Case 1 pool's paper recovery from 5/12 to 8/12 at Top-25,
but its extra hits were close training homologs and it did not resolve the
large-pool failure. A complete measured esterase panel showed a small Morgan
R→E AUROC improvement, while E→R stayed below chance. The
[Case 1](../findings.md#literature-case-1),
[large-pool](../findings.md#case-1-in-a-million-candidate-search), and
[external-panel](../findings.md#external-measured-and-literature-panels)
sections give the denominators and limitations. These results do not justify
claiming broad real-world activity prediction.

To extend the controlled comparison, test the full-label F3 checkpoint on
the remaining EnzymeCAGE families and assess a retrieval-preserving objective
on validation before any further external-panel interpretation. Complete
FGW-CLIP's source-consistent input coverage and target retrain. Preserve each
paper's candidate IDs and official metrics, and report failed or uncovered
queries. The P450 phase-2 result fills a positive-only target-data row; the
later BCE run fills a same-labeled-data row and exposes a generalization
failure.


## Architecture pilots after the user-requested stop

The old training/screening queues were stopped and their artifacts preserved.
The final old-control epoch40 test declined from epoch20 across all four
metrics: Table1 BEDROC85 **0.448663**, BEDROC20 **0.588527**, EF5 **12.760290**,
EF10 **7.235093**. The three stopped fresh ablations have no complete test result.

A broader replacement comparison tests a smaller reaction tower with either
F3's protein tower, a plain mean-pooled protein MLP, or four learned residue
views. Each uses its target benchmark's training pairs and fresh trainable
weights. The reference is matched within each benchmark. Initial fixed-budget
readouts precede phase2 and cover only ReactZyme's reaction_smi split; they
cannot establish a win on the complete ReactZyme benchmark. ReactZyme supplies
participant sets without explicit reaction sides, so a direction-specific
chemistry claim is unavailable there without a new input audit.

[Architecture protocol and live artifacts](../runs/generalization_20260919_2251/cross_paper_retraining/dual_encoder_architecture_pilots_v1/protocol.json).

A fourth EnzymeMap pilot uses RDKit+/DRFP reaction fingerprints and a simple
mean-protein dual MLP (6.29M trainable parameters), with the same training
pairs, loss and20-epoch budget. GPU3 launched this arm at00:30 UTC. This is an
architecture-and-representation comparison; no corresponding ReactZyme
fingerprint experiment is yet scheduled.


### Fingerprint architecture: completed full-library test (2026-09-21, 00:59 UTC)

The fresh reaction-fingerprint + mean-ProtT5 model completed its fixed 20-epoch EnzymeMap pilot. Its full-library test results are: Table 1 BEDROC85 **0.417348**, BEDROC20 **0.566199**, EF5 **12.453180**, EF10 **6.990138**; Table 2 **0.375336 / 0.530953 / 11.803647 / 6.718378**, respectively. It underperforms released CLIPZyme and the earlier Directional F3 + anchor-balanced epoch-20 checkpoint on all four metrics in both settings. This fixed-budget result is not a benchmark win.

Protocol receipt, query/candidate axes, and notebook hashes match all three evaluations. These are exploratory tests after repeated test inspection; full-library validation BEDROC85 remains the checkpoint selector. The new arm is a fresh-base architecture/representation pilot before phase 2. Other architecture screening results are still pending.

[Completed test comparison](runs/generalization_20260919_2251/cross_paper_retraining/dual_encoder_architecture_pilots_v1/completed_test_results.md).

Full-library validation also completed: fingerprint Table 1 BEDROC85 **0.446840**, BEDROC20 **0.618057**, EF5 **13.669094**, EF10 **7.813933**; Table 2 **0.388698 / 0.575796 / 12.951162 / 7.561823**. The same-epoch Directional F3 + anchor-balanced reference has validation BEDROC85 **0.502495**, so the fingerprint model is **0.055655 lower** at the matched 20-epoch budget. Association-manifest hashes and validation denominators match. This provides validation-only evidence against extending the current fingerprint pilot; its fixed-budget training has already ended. It does not establish the performance of unsaved earlier checkpoints.
