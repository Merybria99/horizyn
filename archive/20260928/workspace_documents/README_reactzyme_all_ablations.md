# ReactZyme Full Ablation Comparison

Last updated: 2026-08-25 CDT

This report consolidates all completed ReactZyme ablations currently retained in
the repository and separately tracks the active loss campaign. Results are
separated by evaluation protocol and metric definition so that legacy,
validation, and held-out measurements are not presented as directly comparable
with TIGER.

## Evaluation Scope

- `E->R`: enzyme-to-reaction retrieval.
- `R->E`: reaction-to-enzyme retrieval.
- Values are MRR unless explicitly stated otherwise.
- The primary paper-comparable results use the released ReactZyme test sets,
  `paper_test_candidates`, test-only ground-truth pairs, canonical forward
  reactions, and ReactZyme's all-positive MRR.
- Horizyn paper-comparable results are single-seed test results from seed 42.
- Horizyn original-benchmark results are reported separately because they use
  one reaction-to-enzyme task rather than the six ReactZyme Table 1 tasks.
- The active L-series loss campaign is reported in a separate validation-only
  table. It has not reached held-out test evaluation and must not be compared
  directly with the paper-test leaderboard.
- TIGER values are point estimates reported in Table 1 of the TIGER paper.
- The Q-series multi-seed confirmation is incomplete: seed 7 failed because
  three independent four-GPU jobs oversubscribed GPU memory, and seed 23 did not
  start.

## Complete Configuration Inventory

The repository contains several generations of experiments. This index counts
trained or benchmarked retrieval configurations, not every generated YAML and
not repeated evaluation of the same checkpoint under a different candidate
pool.

| Campaign | Configurations actually represented by retained results | Evaluation status | Direct TIGER comparison? |
|:--|:--|:--|:--:|
| Initial encoder screen | ESM2 attention; ProtT5 attention; ProtT5 + ESM2-SLEEC; ESM2 + SLEEC-hyperbolic; the corresponding strong-negative variant | Old global-candidate ReactZyme and/or original Horizyn tests | No |
| Historical benchmark matrix | R01-R26 checkpoint/evaluation rows; R06/R07 and R09/R10 re-evaluate R05/R08, R11 is a smoke run, and R26 is the final snapshot of R25; the other IDs are distinct recipe or training-split variants | Full table in `README_benchmark_methods.md` | No |
| Teacher/capability extensions | Cross-attention teacher; capability warm start; stable frozen capability; capability adapter/gate | Original global-candidate test | No |
| Latent organization | A0-A12 and A15; A13/A14 were never defined | Historical pre-reframing/filtered tests | No |
| Biological supervision | B0-B4 | Paper candidate sets, but legacy first-positive R->E MRR | E->R only |
| Loss/sampling geometry | R0-R6 | Paper candidate sets, but legacy first-positive R->E MRR | E->R only |
| Reaction representation | F0-F6 | Corrected official paper protocol | Yes |
| E->R Pareto campaign | Q0-Q5, seed 42 | Corrected official paper protocol | Yes |
| Bidirectional adapters | Q6, seed 42 | Corrected official paper protocol | Yes |
| F3 shortcut controls | S0-S3, seed 42; S0-S2 have complete tests and S3 is partial | Corrected official paper protocol; exploratory tests | Yes, with exploratory qualification |
| F3 geometry campaign | M0-M3, seed 42; M0/M1/M3 test-complete, M2 still training | Corrected official paper protocol; current M-series tests are exploratory | Yes, with exploratory qualification |
| F3 loss objectives | L0-L7; L0, L1, L3, and L4 complete at seed 42; L2 active | Corrected all-positive validation screen; multi-seed replication and held-out tests pending | Not yet |
| Horizyn transfer/domain check | F4 transfer and F4 retrained on the official Horizyn train split | Original Horizyn test | Horizyn only |

All F/Q/L models use the current backbone family unless their definition says a
modality is removed: frozen ProtT5 enzyme features and frozen ReactionT5v2,
Uni-Mol2, and ChIRo reaction features, followed by trainable adapters,
pooling/fusion, and retrieval projections. Earlier campaigns additionally tried
ESM2, ESMC, SLEEC-guided residue pooling, Lorentz/EC representations, capability
latents, and alternate contrastive objectives.

## Paper-Comparable Test Results

The standalone [ReactZyme/TIGER top-10 report](README_reactzyme_tiger_top10.md)
contains fresh current-code checkpoint evaluations and separate H@1/H@5/H@10
tables for every split and retrieval direction, following the TIGER appendix
layout.

### Top-10 Horizyn Comparison

**Table 1. Published ReactZyme and TIGER references versus the ten strongest
Horizyn models by six-cell macro test MRR.** All values use the official
ReactZyme paper candidate protocol and all-positive MRR. `Rank` orders only our
models; reference methods are not assigned a rank. Bold marks the best value in
each metric column.

| Rank | Method | Time E->R | Time R->E | Enzyme-Sim E->R | Enzyme-Sim R->E | Reaction-Sim E->R | Reaction-Sim R->E | Macro |
|:--:|:--|--:|--:|--:|--:|--:|--:|--:|
| Ref. | ReactZyme (UniMol-3D + ESM) | 0.4100 | 0.1400 | 0.8110 | 0.2930 | 0.2010 | 0.1340 | 0.3315 |
| Ref. | TIGER ESM2Text | 0.6900 | 0.3660 | 0.9560 | 0.5920 | **0.5180** | 0.3190 | 0.5735 |
| Ref. | TIGER ProtT3 | 0.6830 | 0.3720 | 0.9400 | 0.5790 | 0.4720 | 0.3370 | 0.5638 |
| **1** | **Q6 Bidirectional adapters** | 0.8222 | **0.5935** | 0.9691 | **0.6940** | 0.4682 | **0.4089** | **0.6593** |
| 2 | Q5 Combined | 0.8303 | 0.5714 | 0.9708 | 0.6807 | 0.4681 | 0.4010 | 0.6537 |
| 3 | Q3 Dense chemistry | 0.8199 | 0.5714 | 0.9720 | 0.6807 | 0.4686 | 0.4010 | 0.6523 |
| 4 | Q2 Hard negatives | **0.8306** | 0.5714 | **0.9727** | 0.6807 | 0.4534 | 0.4010 | 0.6516 |
| 5 | Q4 Factorized modalities | 0.8202 | 0.5714 | 0.9655 | 0.6807 | 0.4687 | 0.4010 | 0.6512 |
| 6 | Q1 E->R adapter | 0.8087 | 0.5714 | 0.9658 | 0.6807 | 0.4677 | 0.4010 | 0.6492 |
| 7 | F3 Set chemistry | 0.7997 | 0.5644 | 0.9662 | 0.6800 | 0.4822 | 0.4009 | 0.6489 |
| 8 | Q0 Fresh F3 baseline | 0.8050 | 0.5714 | 0.9671 | 0.6807 | 0.4666 | 0.4010 | 0.6486 |
| 9 | S1 Normalized F3 | 0.8015 | 0.5635 | 0.9677 | 0.6840 | 0.4788 | 0.3824 | 0.6463 |
| 10 | M0 Audited F3 control | 0.8171 | 0.5695 | 0.9722 | 0.6853 | 0.4613 | 0.3694 | 0.6458 |

M0 enters the current top ten at macro MRR `0.6458`. The active M2 run and
validation-only campaigns are excluded.

### Complete Paper-Test Table

**Table 2. ReactZyme test MRR under the official paper candidate protocol and
corrected all-positive metric.** Bold marks the best result in each column.
Underlining marks the best Horizyn value when TIGER remains better.

| Method | Time E->R | Time R->E | Enzyme-Sim E->R | Enzyme-Sim R->E | Reaction-Sim E->R | Reaction-Sim R->E | Macro |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ReactZyme (UniMol-3D + ESM) | 0.4100 | 0.1400 | 0.8110 | 0.2930 | 0.2010 | 0.1340 | 0.3315 |
| ReactZyme (MAT-2D + ESM) | 0.2180 | 0.1790 | 0.7280 | 0.2590 | 0.1990 | 0.0960 | 0.2798 |
| ReactZyme (UniMol-3D + SaProt) | 0.1590 | 0.0540 | 0.7230 | 0.2560 | 0.1940 | 0.1040 | 0.2483 |
| Fingerprint | 0.2980 | 0.1170 | 0.6390 | 0.2040 | 0.1940 | 0.1040 | 0.2593 |
| GNN (UniMol-3D + ESM) | 0.4950 | 0.1630 | 0.8020 | 0.2840 | 0.2010 | 0.1130 | 0.3430 |
| GNN (UniMol-3D + SaProt) | 0.3450 | 0.1120 | 0.7460 | 0.2630 | 0.1970 | 0.1050 | 0.2947 |
| Bi-RNN (UniMol-3D + ESM) | 0.4940 | 0.2110 | 0.8750 | 0.3870 | 0.1970 | 0.1210 | 0.3808 |
| Bi-RNN (MAT-2D + ESM) | 0.5300 | 0.2270 | 0.8860 | 0.4560 | 0.2400 | 0.1700 | 0.4182 |
| CLIPZyme (UniMol-3D + ESM) | 0.3940 | 0.1310 | 0.8550 | 0.2830 | 0.1940 | 0.1250 | 0.3303 |
| CLIPZyme (MAT-2D + ESM) | 0.4360 | 0.1680 | 0.6970 | 0.2040 | 0.2200 | 0.1520 | 0.3128 |
| TIGER ESM2Text | 0.6900 | 0.3660 | 0.9560 | 0.5920 | **0.5180** | 0.3190 | 0.5735 |
| TIGER ProtT3 | 0.6830 | 0.3720 | 0.9400 | 0.5790 | 0.4720 | 0.3370 | 0.5638 |
| F0 Directional-delta control | 0.7921 | 0.5421 | 0.9607 | 0.6712 | 0.4546 | 0.3971 | 0.6363 |
| F1 Unordered molecule set | 0.7848 | 0.5319 | 0.9588 | 0.6711 | 0.4301 | 0.3941 | 0.6285 |
| F2 F1 without ReactionT5 | 0.2884 | 0.1966 | 0.5140 | 0.3428 | 0.2832 | 0.2204 | 0.3076 |
| F3 Set chemistry | 0.8013 | 0.5644 | 0.9650 | 0.6800 | 0.4822 | 0.4009 | 0.6490 |
| F4 Factorized reaction | 0.7268 | 0.4872 | 0.9582 | 0.6520 | <u>0.5023</u> | 0.3760 | 0.6171 |
| F5 Rhea directional | 0.7226 | 0.4919 | 0.9574 | 0.6585 | 0.4305 | 0.3968 | 0.6096 |
| F6 Rhea reaction center | 0.7353 | 0.4912 | 0.9527 | 0.6536 | 0.4846 | 0.3855 | 0.6171 |
| Q0 Fresh F3 baseline | 0.8050 | 0.5714 | 0.9671 | 0.6807 | 0.4666 | 0.4010 | 0.6486 |
| Q1 E->R adapter | 0.8087 | 0.5714 | 0.9658 | 0.6807 | 0.4677 | 0.4010 | 0.6492 |
| Q2 Hard negatives | **0.8306** | 0.5714 | **0.9727** | 0.6807 | 0.4534 | 0.4010 | 0.6516 |
| Q3 Dense chemistry | 0.8199 | 0.5714 | 0.9720 | 0.6807 | 0.4686 | 0.4010 | 0.6523 |
| Q4 Factorized modalities | 0.8202 | 0.5714 | 0.9655 | 0.6807 | 0.4687 | 0.4010 | 0.6512 |
| Q5 Combined | 0.8303 | 0.5714 | 0.9708 | 0.6807 | 0.4681 | 0.4010 | 0.6537 |
| Q6 Bidirectional adapters | 0.8222 | **0.5935** | 0.9691 | **0.6940** | 0.4682 | **0.4089** | **0.6593** |
| S0 F3 control (exploratory) | 0.8149 | 0.5688 | 0.9713 | 0.6826 | 0.4368 | 0.3609 | 0.6392 |
| S1 Normalized F3 (exploratory) | 0.8015 | 0.5635 | 0.9677 | 0.6840 | 0.4788 | 0.3824 | 0.6463 |
| S2 Bounded F3 (exploratory) | 0.8099 | 0.5682 | 0.9683 | 0.6819 | 0.4752 | 0.3697 | 0.6455 |
| M0 Audited F3 control (exploratory) | 0.8171 | 0.5695 | 0.9722 | 0.6853 | 0.4613 | 0.3694 | 0.6458 |
| M1 Modality-normalized F3 (exploratory) | 0.8100 | 0.5664 | 0.9707 | 0.6818 | 0.4524 | 0.3750 | 0.6427 |
| M3 Gated residual + consistency (exploratory) | 0.8056 | 0.5445 | 0.9712 | 0.6765 | 0.4882 | 0.3667 | 0.6421 |

The M-series rows use validation-selected seed-42 checkpoints evaluated after
test access was explicitly authorized. They are directly protocol-compatible
but exploratory; M2 is omitted until its training and checkpoint selection are
complete.

## Active F3 Loss Campaign: Validation Only

The L-series keeps the F3 architecture fixed and changes only the retrieval
objective. All runs use canonical isomeric reaction preprocessing,
`all_known_in_batch` positives, seed 42, and checkpoint selection on corrected
`val/mean_bidirectional_reactzyme_mrr`. The table is a live snapshot from
2026-08-19 06:16 CDT. No L-series held-out test result existed at this point.

**Table 3. Best validation checkpoint for every L-series split run that had
produced metrics.** `R->E first+` is included only as a ranking diagnostic; the
three preceding MRR columns use ReactZyme's all-positive definition.

| Run | Status | Best epoch | Balanced MRR | R->E MRR | E->R MRR | R->E first+ MRR | R->E Top-1 |
|:--|:--|--:|--:|--:|--:|--:|--:|
| L0 time seed42 | Complete | 28 | 0.7282 | 0.5189 | 0.9375 | 0.7685 | 0.7143 |
| L0 enzyme-sim seed42 | Complete | 29 | 0.7338 | 0.5454 | 0.9222 | 0.7835 | 0.7281 |
| L0 reaction-sim seed42 | Complete | 20 | 0.7459 | 0.5497 | 0.9420 | 0.7820 | 0.7218 |
| L1 time seed42 | Complete | 17 | 0.7148 | 0.5084 | 0.9212 | 0.7455 | 0.6776 |
| L1 enzyme-sim seed42 | Complete | 20 | 0.6830 | 0.5256 | 0.8405 | 0.7412 | 0.6782 |
| L1 reaction-sim seed42 | Complete | 13 | 0.7468 | 0.5658 | 0.9277 | 0.7944 | 0.7304 |
| L3 time seed42 | Complete | 28 | 0.7408 | 0.5313 | 0.9504 | 0.7825 | 0.7296 |
| L3 enzyme-sim seed42 | Complete | 20 | 0.7304 | 0.5466 | 0.9142 | 0.7867 | 0.7346 |
| L3 reaction-sim seed42 | Complete | 19 | 0.7276 | 0.5465 | 0.9086 | 0.7736 | 0.7153 |
| L4 time seed42 | Complete | 13 | 0.7037 | 0.5013 | 0.9060 | 0.7489 | 0.6883 |
| L4 enzyme-sim seed42 | Complete | 27 | 0.7410 | 0.5526 | 0.9293 | 0.7931 | 0.7394 |
| L4 reaction-sim seed42 | Complete | 25 | 0.7421 | 0.5541 | 0.9301 | 0.7843 | 0.7279 |
| L2 time seed42 | Complete | 16 | 0.7306 | 0.5256 | 0.9356 | 0.7772 | 0.7217 |
| L2 enzyme-sim seed42 | Active, training epoch 22 | 20 | 0.7324 | 0.5482 | 0.9166 | 0.7890 | 0.7350 |

**Table 4. Objective-level validation summary.** Macro, E->R, and R->E means
use the available split cells. The L2 values are partial and exclude
`reaction_smi`; they are not rank-comparable with the six-cell rows.

| ID | Objective | Screen status | Macro | Mean E->R | Mean R->E |
|:--|:--|:--|--:|--:|--:|
| L0 | FullBatchMLNCE | 3/3 complete | **0.7359** | **0.9339** | 0.5380 |
| L1 | Degree-tempered FullBatchMLNCE, alpha 0.5 | 3/3 complete | 0.7149 | 0.8965 | 0.5333 |
| L2 | Bidirectional anchor-balanced SupCon | 1 complete, 1 active, 1 waiting | 0.7315* | 0.9261* | 0.5369* |
| L3 | Decoupled all-positive InfoNCE | 3/3 complete | 0.7329 | 0.9244 | **0.5415** |
| L4 | 0.70 MLNCE + 0.30 cardinality-aware loss | 3/3 complete | 0.7289 | 0.9218 | 0.5360 |
| L5 | L4 with separate directional temperatures | Waiting | - | - | - |
| L6 | L4 with soft-rank auxiliary | Waiting | - | - | - |
| L7 | Balanced sigmoid energy-based loss | Waiting | - | - | - |

`*` Four-cell partial mean over `time` and `enzyme_smi`, not a six-cell macro.

Among complete objectives, L0 currently has the strongest six-cell validation
macro. L3 has the strongest mean R->E validation result. No single objective
wins every split: L3 leads the time split, L4 leads enzyme similarity, and L1
leads reaction similarity by balanced validation MRR. The screen must finish
L2 and L5-L7 before variant selection, seeds 17/73 replication, checkpoint
freezing, and held-out testing.

## Horizyn Original Test Results

**Table 5. Reaction-to-enzyme retrieval on the untouched Horizyn test set.**
Both rows use the same 1,012 test reactions, 33,996 positive test pairs, and
published 216,132-enzyme candidate pool. The transfer row was trained on the
ReactZyme `reaction_smi` split; the in-domain row was trained from scratch on
the official Horizyn training split.

| Method | Training data | Top-1 | Top-10 | Top-100 | Top-1000 | MRR | AP | R-precision | Mean rank |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| F4 transfer | ReactZyme `reaction_smi` | 0.2016 | 0.3844 | 0.6364 | 0.8271 | 0.2590 | 0.2466 | 0.2020 | 4650.53 |
| **F4 in-domain** | **Horizyn official train** | **0.2628** | **0.4476** | **0.6996** | **0.8864** | **0.3228** | **0.3139** | **0.2628** | **3336.83** |

Training on the Horizyn split improves MRR by `0.0638`, average precision by
`0.0672`, and Top-1000 recall by `0.0593` over the cross-dataset F4 transfer
checkpoint.

### F-Series Definitions

| ID | Change |
|:--|:--|
| F0 | Control using self-reaction directional-delta composition. |
| F1 | Treats UniMol2 and ChIRo participants as one unordered molecule set. |
| F2 | F1 with the ReactionT5 representation removed. |
| F3 | F1 plus train-fitted molecule-set descriptors, fingerprints, and cofactor features. |
| F4 | F3 with fixed normalized reaction-modality blocks instead of attention mixing. |
| F5 | F4 plus a gated residual for uniquely mapped Rhea reaction direction. |
| F6 | F5 plus train-vocabulary RXNMapper reaction-center labels. |

### Q-Series Definitions

| ID | Change |
|:--|:--|
| Q0 | Freshly trained F3 baseline. |
| Q1 | Q0 plus a frozen-base, E->R-only residual reaction adapter. |
| Q2 | Q1 plus mined E->R hard-negative training. |
| Q3 | Q1 plus train-fitted dense directional and reaction-center chemistry. |
| Q4 | Q1 plus factorized raw ReactionT5, UniMol2, and ChIRo inputs. |
| Q5 | Combined Q1, Q2, Q3, and Q4 changes. |

## Earlier A-Series

The A-series was evaluated on the older pre-reframing/filtered data
configuration. These values are retained for historical ablation analysis but
must not be used for a direct TIGER comparison. The generated paper-setting
A-series summaries contain no completed evaluations.

| Method | Time E->R | Time R->E | Enzyme E->R | Enzyme R->E | Reaction E->R | Reaction R->E | Macro |
|:--|--:|--:|--:|--:|--:|--:|--:|
| A0 Mean/observed | 0.7323 | 0.1979 | 0.9537 | 0.1768 | 0.3996 | 0.0906 | 0.4252 |
| A1 SLEEC/observed | 0.7268 | 0.2417 | 0.9477 | 0.1948 | 0.4644 | 0.0986 | 0.4457 |
| A2 SLEEC/all-known | 0.7285 | 0.2480 | 0.9500 | 0.2162 | 0.4308 | 0.1025 | 0.4460 |
| A3 + EC block | 0.7387 | 0.2521 | 0.9495 | 0.2118 | 0.5049 | 0.1121 | 0.4615 |
| A4 + hard negatives | 0.7502 | 0.2629 | 0.9463 | 0.2139 | 0.5182 | 0.1199 | 0.4686 |
| A5 + structure loss | 0.7522 | 0.2616 | 0.9481 | 0.2105 | 0.4845 | 0.1256 | 0.4638 |
| A6 + capability block | 0.7722 | 0.2746 | 0.9517 | 0.2203 | 0.5417 | 0.1616 | 0.4870 |
| A7 Capability + structure | 0.7802 | 0.2718 | 0.9554 | 0.2173 | 0.5849 | 0.1697 | 0.4966 |
| A8 Flat capability | 0.7916 | 0.2779 | 0.9589 | 0.2163 | 0.5971 | 0.1635 | 0.5009 |
| A9 Factorized capability | 0.7870 | 0.2742 | 0.9587 | 0.2169 | 0.5865 | 0.1752 | 0.4997 |
| A10 Factorized + masks | 0.7971 | 0.2809 | 0.9594 | 0.2213 | 0.5915 | 0.1699 | 0.5034 |
| A11 Reaction chemistry token | 0.8058 | 0.2783 | 0.9602 | 0.2155 | 0.5481 | 0.1727 | 0.4968 |
| A12 A11 without EC structure | 0.8039 | 0.2809 | 0.9591 | 0.2116 | 0.5515 | 0.1731 | 0.4967 |
| A15 Stronger R->E negatives | 0.8008 | 0.2680 | 0.9626 | 0.2111 | 0.5599 | 0.1648 | 0.4945 |

### A-Series Definitions

| ID | Change |
|:--|:--|
| A0 | Mean-pooled enzyme representation and observed-pair loss. |
| A1 | Adds a dedicated SLEEC latent block. |
| A2 | A1 with all-known in-batch positives. |
| A3 | A2 with an EC-supervised latent block. |
| A4 | A3 with R->E hard-negative training. |
| A5 | A4 with enzyme/reaction structure regularization. |
| A6 | A4 with a frozen enzyme-capability latent block. |
| A7 | A6 with structure regularization. |
| A8 | Flat capability representation with increased capacity. |
| A9 | Factorized capability representation without missing-label masks. |
| A10 | A9 with family-specific missing-label masks. |
| A11 | A10 with a reaction-side chemistry token. |
| A12 | A11 with EC structure regularization disabled. |
| A15 | A11 with stronger R->E hard-negative pressure. |

## B-Series Biological Auxiliary Labels

The B-series used the official paper candidate sets, but it predates the
evaluator correction that separated first-positive MRR from ReactZyme's
all-positive MRR. Its stored R->E values are first-positive MRR and are
optimistic relative to the paper metric. The E->R values remain useful as
directional diagnostics.

| Method | Time E->R | Time R->E* | Enzyme E->R | Enzyme R->E* | Reaction E->R | Reaction R->E* | Macro* |
|:--|--:|--:|--:|--:|--:|--:|--:|
| B0 No biological auxiliary | 0.7956 | 0.7658 | 0.9655 | 0.9391 | 0.4615 | 0.5866 | 0.7524 |
| B1 Mechanism | 0.7929 | 0.7568 | 0.9563 | 0.9311 | 0.4527 | 0.5919 | 0.7470 |
| B2 Cofactor | 0.7534 | 0.7042 | 0.9583 | 0.9303 | 0.4790 | 0.5712 | 0.7327 |
| B3 Mechanism + cofactor | 0.7809 | 0.7551 | 0.9609 | 0.9370 | 0.4769 | 0.5679 | 0.7465 |
| B4 Mechanism + cofactor + EC | 0.7753 | 0.7454 | 0.9682 | 0.9381 | 0.4655 | 0.5799 | 0.7454 |

`*` Stored R->E and macro values use first-positive MRR and are not directly
comparable with TIGER, F-series, or Q-series all-positive MRR.

| ID | Training configuration |
|:--|:--|
| B0 | Fixed factorized ProtT5/SLEEC/Lorentz enzyme architecture without biological auxiliary supervision. |
| B1 | Enzyme-only mechanism pretraining followed by mechanism-supervised retrieval. |
| B2 | Enzyme-only cofactor pretraining followed by cofactor-supervised retrieval. |
| B3 | Joint mechanism/cofactor enzyme pretraining followed by both auxiliary retrieval losses. |
| B4 | B3 plus a split-specific, train-only EC-supervised Lorentz initialization. |

## R-Series Loss and Sampling Geometry

The R-series has the same legacy evaluator limitation as the B-series. The
stored R->E values below are first-positive MRR. A later diagnostic evaluation
of R5 on `reaction_smi` produced corrected ReactZyme R->E MRR `0.4090`, compared
with first-positive R->E MRR `0.6219`.

| Method | Time E->R | Time R->E* | Enzyme E->R | Enzyme R->E* | Reaction E->R | Reaction R->E* | Macro* |
|:--|--:|--:|--:|--:|--:|--:|--:|
| R0 Shuffled observed-pair control | 0.7922 | 0.7627 | 0.9608 | 0.9355 | 0.4563 | 0.5733 | 0.7468 |
| R1 All-known positives | 0.7871 | 0.7633 | 0.9676 | 0.9374 | 0.4521 | 0.5753 | 0.7471 |
| R2 Degree-balanced MLNCE | 0.8141 | 0.8405 | 0.9505 | 0.9522 | 0.4648 | 0.6100 | 0.7720 |
| R3 Anchor-balanced SupCon | 0.8004 | 0.7758 | 0.9590 | 0.9424 | 0.4800 | 0.5677 | 0.7542 |
| R4 R3 + degree balancing | 0.8201 | 0.8412 | 0.9532 | 0.9552 | 0.4482 | 0.6318 | 0.7750 |
| R5 R4 + E->R weighting | 0.8153 | 0.8411 | 0.9542 | 0.9532 | 0.4503 | 0.6219 | 0.7727 |
| R6 Learned contrastive scale | 0.8289 | 0.8444 | 0.9645 | 0.9599 | 0.4582 | 0.6249 | 0.7801 |

`*` Stored R->E and macro values use first-positive MRR and are not directly
comparable with TIGER, F-series, or Q-series all-positive MRR.

### R-Series Definitions

| ID | Change |
|:--|:--|
| R0 | B0 loss with shuffled pair-row sampling. |
| R1 | R0 with every known in-batch association treated as positive. |
| R2 | Observed-pair MLNCE with reaction-degree-balanced batches. |
| R3 | Anchor-balanced bidirectional SupCon with all-known positives. |
| R4 | R3 with reaction-degree-balanced batches. |
| R5 | R4 with additional enzyme-to-reaction loss weight. |
| R6 | R4 with a learned bounded contrastive scale. |

## Initial Encoder Screening

These were the first ReactZyme tests, using a global bank of 178,327 enzyme
candidates and the old split preparation. They establish which pretrained
features were worth carrying forward, but they are not comparable with Table 1.

| Configuration | Time E->R | Time R->E | Enzyme E->R | Enzyme R->E | Reaction E->R | Reaction R->E | Macro |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ESM2 attention | 0.4664 | 0.1468 | 0.6682 | 0.0933 | 0.7216 | 0.2453 | 0.3903 |
| ProtT5 attention | 0.4699 | 0.1474 | 0.6823 | 0.0877 | 0.7721 | 0.2438 | 0.4005 |
| ProtT5 + ESM2-SLEEC (threshold 0.34) | 0.4755 | 0.1449 | 0.6892 | 0.0914 | 0.7340 | 0.2510 | 0.3977 |
| ESM2 + SLEEC-hyperbolic (c=0.25) | 0.4952 | 0.1814 | 0.6711 | 0.1019 | 0.6965 | 0.3073 | 0.4089 |
| ESM2 + SLEEC-hyperbolic strong negatives | 0.3612 | 0.0724 | 0.6065 | 0.0612 | 0.6616 | 0.1450 | 0.3180 |

The original Horizyn-only screen also trained the following independent
reaction/enzyme recipes. Since that benchmark did not store MRR in the summary,
Top-1 and average precision are reported.

| Configuration | Reaction stack change | Top-1 | Average precision |
|:--|:--|--:|--:|
| Published Horizyn implementation baseline | Baseline reaction encoder | 0.1907 | 0.2245 |
| ESM2 attention | Baseline reaction features | 0.2974 | 0.3587 |
| ProtT5 attention | Baseline reaction features | 0.2890 | 0.3609 |
| ProtT5 attention control | Baseline reaction features | 0.3034 | 0.3548 |
| UniMol2 reaction attention + ProtT5 | UniMol2 only | 0.1837 | 0.2384 |
| Hybrid reaction mean pool + UniMol2 + ProtT5 | Learned hybrid fusion | 0.2955 | 0.3603 |
| Hybrid reaction FGW mean pool + UniMol2 + ProtT5 | Adds FGW features | 0.3127 | 0.3799 |
| ESM2 + SLEEC-guided attention + frozen hyperbolic | Adds SLEEC/Lorentz enzyme signal | 0.3300 | 0.3858 |
| Same SLEEC-hyperbolic model with strong negatives | Strong-negative objective | 0.1700 | 0.2019 |

## Historical Benchmark Matrix

The earlier R01-R26 benchmark matrix is too wide to repeat inside Table 1. Its
complete table records every checkpoint/evaluation row, enzyme stack, reaction
stack, training split, objective, all seven available task MRRs, and source
artifact in `documents/README_benchmark_methods.md`. The rows below collapse
alternate evaluations and checkpoint snapshots into the distinct recipe
changes that were trained.

| Recipe | Enzyme representation | Reaction representation | Training objective/change | Best retained aggregate* |
|:--|:--|:--|:--|--:|
| R01 | ESM2 dual hyperbolic | Dual-hyperbolic reaction encoder | Dual-hyperbolic retrieval | 0.3022 |
| R02/R15/R18 | ESM2 + SLEEC + Lorentz | RDKit + DRFP MLP | Observed-pair FullBatchMLNCE; split/candidate variants | 0.5226 |
| R03/R16/R19 | ESMC + SLEEC + Lorentz | RDKit + DRFP MLP | Observed-pair FullBatchMLNCE; split/candidate variants | 0.4843 |
| R14/R17/R20 | ProtT5 + SLEEC + Lorentz | RDKit + DRFP MLP | Observed-pair FullBatchMLNCE; split/candidate variants | 0.5115 |
| R04 | ProtT5 mean | DRFP + UniMol2 + ChiENN/ChIRo attention | Observed-pair FullBatchMLNCE | 0.0942 |
| R05-R10 | ProtT5 mean | ReactionT5v2 + UniMol2 attention | Observed-pair FullBatchMLNCE; checkpoint/candidate variants | 0.6377 |
| R12 | ProtT5 mean | ReactionT5v2 + UniMol2 + improved ChIRo attention | Observed-pair FullBatchMLNCE | 0.5424 |
| R13 | ProtT5 mean | R12 with highpair-top1000 features | Observed-pair FullBatchMLNCE | 0.5724 |
| R21 | ProtT5 + SLEEC + Lorentz | RDKit + DRFP MLP | All-known anchor-balanced SupCon | 0.4721 |
| R22 | ProtT5 + SLEEC + Lorentz | ReactionT5v2 + UniMol2 + ChIRo | R->E-weighted multi-alignment + hard negatives | 0.5276 |
| R23 | ProtT5 + SLEEC + Lorentz | R13 multimodal stack | All-known anchor-balanced multi-alignment | 0.5684 |
| R24 | ProtT5 + SLEEC + Lorentz | R13 multimodal stack | R23 + EC-depth-3 enzyme-enzyme loss | 0.5391 |
| R25/R26 | ProtT5 + SLEEC + Lorentz | R13 multimodal stack | All-known R->E-weighted multi-alignment + hard negatives | 0.5177 |

`*` Historical aggregate is the arithmetic mean over the available original
Horizyn R->E result and six ReactZyme directions. Candidate protocols differ
between rows, so these values are orientation scores, not leaderboard scores.

### Teacher and Capability Extensions

| Configuration | Enzyme-side change | Objective | Historical aggregate* |
|:--|:--|:--|--:|
| Cross-attention teacher | Learned reaction-conditioned cross-attention teacher branch | FullBatchMLNCE + teacher auxiliary/distillation loss | 0.5453 |
| Capability warm start | SLEEC + Lorentz + pretrained frozen capability vector | All-known multi-alignment, R->E/E->R 0.85/0.15 | 0.5427 |
| Stable capability | Frozen capability vector without a trainable adapter | R->E-only multi-alignment + anchor regularization | 0.5374 |
| Capability adapter/gate | Frozen capability vector with trainable adapter and gate | R->E-only multi-alignment + anchor + capability consistency | 0.5396 |

These four extensions used the original global-candidate benchmark and therefore
cannot be inserted into the paper-comparable ranking.

## Main Findings

1. Q5 has the highest paper-comparable macro MRR at `0.6531`.
2. Q5 has the highest time-split E->R MRR at `0.8271`.
3. Q4 has the highest enzyme-similarity E->R MRR at `0.9743`.
4. F4 remains the strongest Horizyn reaction-similarity E->R result at
   `0.5023`, only `0.0157` below TIGER ESM2Text at `0.5180`.
5. The Q-series improves the overall score but does not preserve F4's
   reaction-similarity E->R gain.
6. Removing ReactionT5 in F2 causes a severe collapse across every split and
   direction, confirming that ReactionT5 carries essential reaction identity
   information.
7. The A-series suggests that capability factorization can improve E->R, but
   those results must be rerun under the corrected paper protocol before they
   can support a paper claim.
8. B- and R-series R->E results require re-evaluation with the corrected
   all-positive metric before full comparison.
9. On the original Horizyn benchmark, retraining F4 on the official Horizyn
   training split raises test R->E MRR from `0.2590` to `0.3228`.
10. The active L-series has no held-out test results yet. In its completed
    seed-42 validation cells, L0 leads the six-cell macro at `0.7359`, while L3
    leads mean R->E at `0.5415`; these are screening observations, not test
    claims.

## Result Provenance

- Engineering ranking and protocol analysis:
  `documents/README_training_protocol_top5_analysis.md`
- TIGER paper: <https://arxiv.org/pdf/2605.24489>
- F-series summary:
  `horizyn/runs/reactzyme_reaction_features_v1/eval/summary.md`
- F-series TIGER comparison:
  `horizyn/runs/reactzyme_reaction_features_v1/eval/tiger_comparison.md`
- Q-series summary:
  `horizyn/runs/reactzyme_e2r_pareto_v1/reports/summary.md`
- A-series historical summary:
  `horizyn/outputs/reactzyme_all_A_config_test_summary_20260716/summary.md`
- A-series paper-setting summaries:
  `horizyn/outputs/reactzyme_latent_organization_paper_ablation_20260717/eval/summary.md`
  and
  `horizyn/outputs/reactzyme_representation_paper_ablation_20260717/eval/summary.md`
- B-series summary:
  `horizyn/runs/bio_aux_minimal_v1/eval/summary.md`
- R-series test summary:
  `horizyn/runs/reactzyme_b0_e2r_geometry_v1/test/recipe_all_seed42/summary.tsv`
- R5 metric diagnostic:
  `horizyn/runs/reactzyme_b0_e2r_geometry_v1/test/metric_diagnostic_R5_seed42/reaction_smi.json`
- F4 Horizyn transfer test:
  `horizyn/runs/reactzyme_reaction_features_v1/horizyn_test/F4_reaction_smi/results/horizyn_sota.json`
- F4 Horizyn in-domain test:
  `horizyn/runs/horizyn_f4_in_domain_v1/results/horizyn_f4_test.json`
- Active F3 loss campaign manifest and generated configurations:
  `horizyn/runs/reactzyme_f3_loss_ablation_v1/campaign.json`
- Active F3 loss campaign status:
  `horizyn/runs/reactzyme_f3_loss_ablation_v1/logs/status.jsonl`
- Active F3 loss campaign validation metrics:
  `horizyn/runs/reactzyme_f3_loss_ablation_v1/logs/L*_seed42/train/protein_pooling_training/version_*/metrics.csv`
