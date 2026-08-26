---
marp: true
title: Enzyme-Reaction Retrieval in Horizyn
description: Task, implemented methods, ablations, and benchmark comparison
paginate: true
size: 16:9
---

<!-- _class: title -->

# Enzyme-Reaction Retrieval in Horizyn

## Task, implemented methods, ablations, and benchmark comparison

ReactZyme paper protocol | Corrected all-positive MRR | 2026-08-14

---

# Executive summary

- We learn a shared 512-D space for enzyme sequences and biochemical reactions.
- The strongest reliable recipe is F3/Q0: frozen ProtT5 on enzymes, frozen
  ReactionT5v2 + UniMol2 + ChIRo + set chemistry on reactions, and a single
  retrieval-training stage.
- F3 beats the best TIGER result in five of six ReactZyme MRR cells and reaches
  macro MRR 0.6490 versus TIGER ESM2Text at 0.5735.
- The remaining gap is reaction-similarity E->R: F4 reaches 0.5023 versus TIGER
  at 0.5180.
- Added biological labels, mapped reaction centers, hard negatives, and
  direction-specific adapters have not yet produced a replicated gain large
  enough to displace F3 as the default.

---

# The retrieval task

Two directions share one embedding space:

```text
reaction query  -> rank candidate enzymes     (R->E)
enzyme query    -> rank candidate reactions   (E->R)
```

For normalized embeddings `z_r` and `z_e`:

```text
similarity(r, e) = z_r dot z_e
distance(r, e)   = 1 - similarity(r, e)
```

The scientific target is compatibility, not EC classification alone. One
reaction can have many valid enzymes and an enzyme may catalyze multiple
reactions.

---

# ReactZyme is a bipartite graph

- 178,459 deduplicated enzyme-molecular-set pairs
- 178,327 enzymes and 7,726 released molecular sets
- Median 2 proteins per molecular set; maximum 5,489
- The released reaction strings are dot-separated molecular sets, not
  directional atom-mapped transformations: they contain no `>>` separator.
- A molecular set can summarize multiple Rhea IDs and EC annotations.

Consequence: multi-positive ranking, reaction hubs, wildcard chemistry, and
participant aggregation are part of the benchmark definition.

---

# Three protocols, three biological questions

| Split | Test pairs | Intended holdout | Biological audit |
|:--|--:|:--|:--|
| `time` | 12,287 | Later enzymes | Mostly later proteins for supported molecular sets |
| `enzyme_smi` | 8,739 | Sequence-novel enzymes | Exact-ID holdout; mostly close-homolog transfer |
| `reaction_smi` | 14,689 | Novel reactions | Coupled cold-reaction and cold-family shift |

`reaction_smi` is hardest: 386 unseen molecular-set IDs, median best enzyme hit
36.1% identity, only 64.8% with a significant train hit, and strong cofactor and
EC-family shifts.

---

# The hidden difficulty: degree imbalance

| Protocol | Mean proteins / reaction | Maximum | Pair share in top 10 | Gini |
|:--|--:|--:|--:|--:|
| `time` | 4.66 | 657 | 18.2% | 0.650 |
| `enzyme_smi` | 5.56 | 277 | 10.8% | 0.660 |
| `reaction_smi` | 38.05 | 2,249 | 56.7% | 0.901 |

The largest `reaction_smi` hub alone contains 15.3% of test pairs. Pair-weighted
losses and first-positive metrics can therefore tell a misleading story.

![Reaction-similarity positives per query](assets/reaction_smi_positive_degree.png)

---

# Evaluation protocol matters

Paper-comparable reporting uses:

- three independent split-specific models;
- a 90/10 train-derived validation split;
- untouched released tests;
- test-only positive pairs;
- candidates drawn from unique test entities;
- ReactZyme all-positive MRR, not first-positive MRR;
- both E->R and R->E, with arithmetic macro over six cells.

Legacy global-candidate or first-positive results remain useful diagnostics but
are not directly comparable with TIGER Table 1.

---

# Experimental program

| Generation | Main question | Status |
|:--|:--|:--|
| Encoder screens / R01-R26 | Which pretrained modalities and pooling? | Historical protocol |
| A0-A15 | How should enzyme latents be organized? | Historical filtered protocol |
| B0-B4 | Do mechanism, cofactor, and EC labels help? | Legacy R->E metric |
| R0-R6 | Which positive semantics and loss geometry? | Legacy R->E metric |
| F0-F6 | Which reaction representation works? | Correct paper protocol |
| Q0-Q5 | Can E->R-specific adaptation improve the Pareto frontier? | Correct paper protocol |
| L0-L7 | Can a better F3 loss improve robust ranking? | Implemented; results pending |

---

# Shared dual-tower design

```text
ENZYME                                      REACTION
amino-acid sequence                        reaction molecular set
       |                                          |
frozen protein LM features                 frozen chemistry encoders
       |                                          |
pooling + functional branches              modality MLPs + fusion
       |                                          |
       +--------- normalized 512-D space --------+
                              |
                    contrastive retrieval loss
```

Large pretrained encoders are frozen. Trainable capacity is concentrated in
adapters, pooling, fusion, and retrieval projections.

---

# Enzyme representations implemented

| Component | Implemented alternatives | What it contributes |
|:--|:--|:--|
| Protein LM | ESM2, ESMC, ProtT5 | Global sequence/function context |
| Pooling | Mean, learned attention, SLEEC-guided attention | Whole-protein vs active-site emphasis |
| Functional prior | Frozen SLEEC residue scorer | Site-biased residue weighting |
| Hierarchy | Lorentz/EC tangent branch | Hierarchical functional geometry |
| Capability | Flat, factorized, masked, gated | Predicted reaction-compatibility prior |
| BioFP | Mechanism, cofactor, transition heads | Enzyme-only auxiliary supervision |
| Text | TIGER-style gated text fusion | Implemented historically, excluded from current F/Q |

Current paper-comparable F/Q runs use no enzyme text.

---

# B0 enzyme tower: factorized biological layout

| Block | Source | Dim | Norm-squared weight |
|:--|:--|--:|--:|
| Core | Mean-pooled ProtT5 | 288 | 0.55 |
| Site | SLEEC-guided ProtT5 | 96 | 0.20 |
| Mechanism | Learned residue query | 64 | 0.12 |
| Cofactor | Learned residue query | 32 | 0.08 |
| EC | Lorentz tangent branch | 32 | 0.05 |

Each block is projected and normalized independently, scaled by the square root
of its allocation, concatenated, and normalized again. In B0 the mechanism and
cofactor slots receive retrieval gradients but no label loss.

---

# Reaction representations implemented

| Family | Inputs | Implemented fusion |
|:--|:--|:--|
| Fingerprints | RDKit, DRFP | Concatenated MLP |
| Language | ReactionT5v2 | Mean-pooled reaction embedding + MLP |
| 3D molecule | UniMol2 | Reactant/product set pooling |
| Chiral graph | ChIRo / ChiENN | Stereochemistry-aware molecule pooling |
| Set chemistry | descriptors, fingerprints, cofactors | Train-fitted structured token |
| Directional | Rhea substrate/product reconstruction | Gated residual when uniquely mapped |
| Reaction center | RXNMapper center labels | Train-vocabulary gated residual |

Canonical isomeric SMILES are enforced in the new F3 loss campaign.

---

# Reaction fusion strategies

**Attention fusion (F0/F1/F3)**

```text
each modality -> 4096 -> 4096 -> 512
normalized modality tokens -> learned attention -> reaction MLP -> 512-D
```

**Fixed factorized fusion (F4)**

```text
ReactionT5v2 0.250 | UniMol2 0.375 | ChIRo 0.1875 | chemistry 0.1875
```

Other implemented variants include unordered molecule-set pooling, directional
delta composition, masked missing modalities, gated residuals, and
direction-specific E->R adapters.

---

# Reference training recipe

| Setting | B0 / F-family reference |
|:--|:--|
| Optimizer | AdamW, learning rate 1e-4, weight decay 0.01 |
| Epochs | Up to 30 |
| Precision | 32-true |
| Distributed training | 4 GPUs, DDP global batch |
| Pair batch | 512 per rank, up to 2,048 rows globally |
| Similarity | Cosine |
| Temperature | Fixed beta = 10 unless ablated |
| Checkpoint selection | Validation bidirectional MRR |

Large encoders remain frozen; three ReactZyme split models are trained
independently.

---

# Retrieval objectives implemented

| Objective | Purpose |
|:--|:--|
| FullBatchMLNCE | Global pair partition; simple strong baseline |
| All-known-in-batch MLNCE | Recover every known positive among batch entities |
| Degree-tempered MLNCE | Reduce reaction-hub dominance |
| Bidirectional anchor-balanced SupCon | Equal mass per query in each direction |
| Decoupled all-positive InfoNCE | Positives compete only with negatives |
| Hybrid MLNCE + cardinality loss | Preserve global calibration and query balance |
| Soft-rank auxiliary | Directly penalize outranking negatives |
| Balanced sigmoid EBM | Equal positive/negative class means with learned bias |
| Multi-alignment / FGW | Add reaction/enzyme structural geometry |
| Hard-negative margins | Emphasize near misses in one or both directions |

---

# New F3 loss campaign: L0-L7

| ID | Loss variant | Key change |
|:--|:--|:--|
| L0 | Corrected FullBatchMLNCE | All-known positives, corrected monitor |
| L1 | Degree-tempered MLNCE | Full-graph degree exponent 0.5 |
| L2 | Anchor-balanced SupCon | Equal anchor mass in both directions |
| L3 | Decoupled all-positive InfoNCE | No positive-positive competition |
| L4 | Hybrid | 0.70 MLNCE + 0.30 cardinality, 5-epoch warmup |
| L5 | Hybrid + temperatures | Separate learned E->R/R->E beta in [3, 30] |
| L6 | Hybrid + soft rank | Weight 0.05, tau 0.1, top 128 negatives |
| L7 | Balanced sigmoid EBM | Learned global bias, balanced classes |

Screen: all 8 x 3 splits at seed 42. Replicate the best two at seeds 17 and
73. Results are not yet available and must not be mixed with completed tables.

---

# Biological labels implemented

**Enzyme-side priors**

- Mechanism core: redox, transfer, bond and stereochemical changes
- Cofactor core: NAD(P), flavin, PLP, NTP, CoA, SAM, sugar nucleotide, thiol
- Pruned substrate/product transition vocabulary
- Hierarchical EC labels and Lorentz embedding

Labels are train-only targets, not inference inputs. Missing labels use masks or
soft confidence targets. Reaction-side labels include set cofactors, Rhea
directional features, and RXNMapper centers.

Finding: biological priors can organize latents, but none has yet beaten the
clean no-auxiliary control under reliable corrected evaluation.

---

# A-series: latent organization

| Config | Change | Historical macro |
|:--|:--|--:|
| A0 | Mean enzyme, observed pairs | 0.4252 |
| A1 | + SLEEC block | 0.4457 |
| A3 | + EC block | 0.4615 |
| A4 | + hard negatives | 0.4686 |
| A6 | + capability block | 0.4870 |
| A10 | Factorized capability + masks | **0.5034** |
| A11 | + reaction chemistry token | 0.4968 |
| A15 | Stronger R->E negatives | 0.4945 |

Lesson: sequence-side functional structure helped historically, but too many
coupled changes and an obsolete data protocol prevent a paper claim.

---

# B-series: biological supervision

| Config | Added supervision | Legacy macro | Delta vs B0 |
|:--|:--|--:|--:|
| B0 | None beyond pair retrieval | **0.7524** | - |
| B1 | Mechanism | 0.7470 | -0.0054 |
| B2 | Cofactor | 0.7327 | -0.0197 |
| B3 | Mechanism + cofactor | 0.7465 | -0.0059 |
| B4 | B3 + EC Lorentz initialization | 0.7454 | -0.0070 |

The clean control won. These R->E values use legacy first-positive MRR; only
directional diagnostics are retained.

---

# R-series: loss and sampling geometry

| Config | Change | Legacy macro |
|:--|:--|--:|
| R0 | Shuffled observed-pair control | 0.7468 |
| R1 | All-known positives | 0.7471 |
| R2 | Degree-balanced MLNCE | 0.7720 |
| R3 | Anchor-balanced SupCon | 0.7542 |
| R4 | R3 + degree balancing | 0.7750 |
| R5 | R4 + E->R weighting | 0.7727 |
| R6 | Learned bounded scale | **0.7801** |

Corrected R5 reaction-similarity R->E fell from first-positive 0.6219 to
all-positive 0.4090. This motivated the corrected L0-L7 rerun.

---

# F-series: reaction representation ablation

| ID | Controlled change |
|:--|:--|
| F0 | Directional-delta multimodal control |
| F1 | Unordered UniMol2/ChIRo molecule set |
| F2 | F1 without ReactionT5v2 |
| F3 | F1 + train-fitted set chemistry |
| F4 | F3 with fixed factorized modality blocks |
| F5 | F4 + gated Rhea directional residual |
| F6 | F5 + RXNMapper reaction-center labels |

This is the strongest causal campaign because architecture changes were isolated
under the corrected official paper protocol.

---

# ReactZyme Table 1: MRR

| Method | Time E->R | Time R->E | Enz E->R | Enz R->E | Rxn E->R | Rxn R->E | Macro |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ReactZyme Bi-RNN MAT-2D+ESM | 0.530 | 0.227 | 0.886 | 0.456 | 0.240 | 0.170 | 0.418 |
| TIGER ESM2Text | 0.690 | 0.366 | 0.956 | 0.592 | **0.518** | 0.319 | 0.574 |
| TIGER ProtT3 | 0.683 | 0.372 | 0.940 | 0.579 | 0.472 | 0.337 | 0.564 |
| F0 | 0.792 | 0.542 | 0.961 | 0.671 | 0.455 | 0.397 | 0.636 |
| **F3** | 0.801 | 0.564 | 0.965 | 0.680 | 0.482 | **0.401** | **0.649** |
| F4 | 0.727 | 0.487 | 0.958 | 0.652 | 0.502 | 0.376 | 0.617 |
| Q5 | **0.827** | **0.571** | **0.971** | **0.681** | 0.468 | **0.401** | **0.653** |

Single-seed test values. F/Q models use no generated enzyme text.

---

# What the F-series established

1. ReactionT5v2 is essential: removing it in F2 collapses macro MRR from 0.6285
   to 0.3076.
2. Set chemistry is complementary: F3 improves F1 by 0.0205 macro MRR.
3. Learned fusion is best overall; fixed blocks in F4 specialize for
   reaction-similarity E->R.
4. Sparse mapped direction and center supervision did not justify their
   external preprocessing: F5/F6 trail F3.
5. The remaining TIGER gap is narrow and localized, not a general failure of
   the dual-tower architecture.

---

# Q-series: E->R-specific adaptation

| ID | Change after Q0 | Test macro |
|:--|:--|--:|
| Q0 | Fresh F3 baseline | 0.6504 |
| Q1 | Frozen-base residual adapter | 0.6512 |
| Q2 | Q1 + mined hard negatives | 0.6489 |
| Q3 | Q1 + dense chemistry | 0.6495 |
| Q4 | Q1 + factorized raw modalities | 0.6518 |
| Q5 | Combined Q1-Q4 machinery | **0.6531** |

Q5 is the raw winner, but its 0.0013 gain over Q4 is below observed rerun
variation and lacks completed multi-seed confirmation.

---

# Best engineering solutions

| Rank | Configuration | Why keep it |
|--:|:--|:--|
| 1 | F3 / Q0 | Best balance of score, protocol confidence, and one-stage training |
| 2 | F0 | Mandatory minimal control; no set schema or external labels |
| 3 | F4 | Interpretable specialist for reaction-similarity E->R |
| 4 | Q1 | Cheapest modular direction-specific adjustment |
| 5 | Q4 | Cleanest high-score E->R adapter |

Recommendation: F3/Q0 is the paper and wet-lab default. Keep F0 and F4 as
scientifically meaningful controls.

---

# Transfer to the Horizyn benchmark

| Method | Training data | Top-1 | Top-100 | MRR | AP |
|:--|:--|--:|--:|--:|--:|
| F4 transfer | ReactZyme `reaction_smi` | 0.2016 | 0.6364 | 0.2590 | 0.2466 |
| **F4 in-domain** | Horizyn official train | **0.2628** | **0.6996** | **0.3228** | **0.3139** |

The architecture transfers, but matching the training distribution improves MRR
by 0.0638. Data protocol matters more than a small fusion modification.

---

# From benchmark to wet lab

Implemented inference workflow:

```text
reaction SMILES config
        |
same reaction feature stack as checkpoint
        |
rank a cached enzyme catalog
        |
Top@1 / 5 / 10 / 20 / 50 / 100
        |
fold selected candidates with ESMFold / AF3 workflow
```

For discovery-scale use, the candidate bank can be expanded from benchmark
candidates to reviewed Swiss-Prot. Scores are retrieval similarities, not
calibrated catalytic probabilities; sequence identity, EC plausibility,
cofactors, localization, and assay feasibility remain downstream filters.

---

# Evidence boundaries

- F/Q comparisons are corrected paper-protocol, seed-42 test results.
- A-series and historical R01-R26 use older split or candidate definitions.
- B/R stored R->E values use legacy first-positive MRR.
- Q-series multi-seed confirmation is incomplete.
- Validation is pair-random and does not reproduce the coupled novelty of the
  `reaction_smi` test.
- ReactZyme inputs are participant sets; they do not expose a clean elementary
  reaction direction.
- The benchmark lacks taxonomy and organism metadata.

The deck reports these generations separately to avoid false leaderboard
comparisons.

---

# Current recommendation

**Default:** F3/Q0

- frozen ProtT5 + SLEEC/factorized enzyme tower;
- ReactionT5v2 + UniMol2 + ChIRo + set chemistry;
- learned multimodal attention;
- one-stage retrieval training;
- no enzyme text and no biological auxiliary loss.

**Controls:** F0 for simplicity; F4 for reaction-similarity E->R.

**Current experiment:** L0-L7 loss campaign using canonical isomeric SMILES,
all-known positives, corrected checkpoint selection, and multi-seed promotion.

---

<!-- _class: section -->

# Appendix

Comprehensive implemented-method inventory and detailed comparisons

---

# Appendix A: initial encoder screen

| Configuration | Time E->R | Time R->E | Enz E->R | Enz R->E | Rxn E->R | Rxn R->E | Macro |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ESM2 attention | 0.466 | 0.147 | 0.668 | 0.093 | 0.722 | 0.245 | 0.390 |
| ProtT5 attention | 0.470 | 0.147 | 0.682 | 0.088 | 0.772 | 0.244 | 0.401 |
| ProtT5 + ESM2-SLEEC | 0.476 | 0.145 | 0.689 | 0.091 | 0.734 | 0.251 | 0.398 |
| ESM2 + SLEEC-Lorentz | 0.495 | 0.181 | 0.671 | 0.102 | 0.697 | 0.307 | **0.409** |
| + strong negatives | 0.361 | 0.072 | 0.607 | 0.061 | 0.662 | 0.145 | 0.318 |

Historical global candidate protocol only.

---

# Appendix A2: complete A-series

| ID | Implemented change | Historical macro |
|:--|:--|--:|
| A0 | Mean enzyme, observed pairs | 0.4252 |
| A1 | + SLEEC block | 0.4457 |
| A2 | + all-known in-batch positives | 0.4460 |
| A3 | + EC block | 0.4615 |
| A4 | + R->E hard negatives | 0.4686 |
| A5 | + reaction/enzyme structure loss | 0.4638 |
| A6 | + frozen capability block | 0.4870 |
| A7 | Capability + structure | 0.4966 |
| A8 | Flat high-capacity capability | 0.5009 |
| A9 | Factorized capability, no masks | 0.4997 |
| A10 | Factorized capability + masks | **0.5034** |
| A11 | + reaction chemistry token | 0.4968 |
| A12 | A11 without EC structure | 0.4967 |
| A15 | Stronger R->E negatives | 0.4945 |

`A13` and `A14` were never defined.

---

# Appendix B: historical recipe matrix

| Recipes | Enzyme stack | Reaction stack | Objective / best aggregate* |
|:--|:--|:--|:--|
| R01 | ESM2 dual hyperbolic | Dual hyperbolic | Retrieval / 0.3022 |
| R02/R15/R18 | ESM2 + SLEEC + Lorentz | RDKit + DRFP | MLNCE / 0.5226 |
| R03/R16/R19 | ESMC + SLEEC + Lorentz | RDKit + DRFP | MLNCE / 0.4843 |
| R14/R17/R20 | ProtT5 + SLEEC + Lorentz | RDKit + DRFP | MLNCE / 0.5115 |
| R04 | ProtT5 mean | DRFP + UniMol2 + ChIRo | MLNCE / 0.0942 |
| R05-R10 | ProtT5 mean | ReactionT5v2 + UniMol2 | MLNCE / **0.6377** |
| R12/R13 | ProtT5 mean | + ChIRo / highpair | MLNCE / 0.5724 |
| R21-R26 | ProtT5/SLEEC/Lorentz | multimodal | SupCon, alignment, EC, negatives / 0.5684 |

`*` Mixed historical candidate protocols; orientation only.

---

# Appendix C: teacher and capability extensions

| Method | Added mechanism | Historical aggregate* |
|:--|:--|--:|
| Cross-attention teacher | Reaction-conditioned teacher + distillation | 0.5453 |
| Capability warm start | Frozen predicted capability vector | 0.5427 |
| Stable capability | Frozen capability without adapter | 0.5374 |
| Capability adapter/gate | Trainable adapter and gated residual | 0.5396 |

Implemented capability variants also include flat versus factorized vectors,
family masks, enzyme-only pretraining, joint retrieval, reaction-frozen enzyme
tuning, and capability consistency/anchor losses.

`*` Original global-candidate benchmark.

---

# Appendix D: complete F-series MRR

| Method | Time E->R | Time R->E | Enz E->R | Enz R->E | Rxn E->R | Rxn R->E | Macro |
|:--|--:|--:|--:|--:|--:|--:|--:|
| F0 | 0.7921 | 0.5421 | 0.9607 | 0.6712 | 0.4546 | 0.3971 | 0.6363 |
| F1 | 0.7848 | 0.5319 | 0.9588 | 0.6711 | 0.4301 | 0.3941 | 0.6285 |
| F2 | 0.2884 | 0.1966 | 0.5140 | 0.3428 | 0.2832 | 0.2204 | 0.3076 |
| F3 | **0.8013** | **0.5644** | **0.9650** | **0.6800** | 0.4822 | **0.4009** | **0.6490** |
| F4 | 0.7268 | 0.4872 | 0.9582 | 0.6520 | **0.5023** | 0.3760 | 0.6171 |
| F5 | 0.7226 | 0.4919 | 0.9574 | 0.6585 | 0.4305 | 0.3968 | 0.6096 |
| F6 | 0.7353 | 0.4912 | 0.9527 | 0.6536 | 0.4846 | 0.3855 | 0.6171 |

---

# Appendix E: complete Q-series MRR

| Method | Time E->R | Time R->E | Enz E->R | Enz R->E | Rxn E->R | Rxn R->E | Macro |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Q0 | 0.8128 | 0.5714 | 0.9699 | 0.6807 | 0.4666 | 0.4010 | 0.6504 |
| Q1 | 0.8140 | 0.5714 | 0.9721 | 0.6807 | 0.4677 | 0.4010 | 0.6512 |
| Q2 | 0.8162 | 0.5714 | 0.9709 | 0.6807 | 0.4534 | 0.4010 | 0.6489 |
| Q3 | 0.8049 | 0.5714 | 0.9702 | 0.6807 | 0.4686 | 0.4010 | 0.6495 |
| Q4 | 0.8149 | 0.5714 | **0.9743** | 0.6807 | **0.4687** | 0.4010 | 0.6518 |
| Q5 | **0.8271** | 0.5714 | 0.9705 | 0.6807 | 0.4681 | 0.4010 | **0.6531** |

Q adaptations alter E->R only; R->E remains the frozen Q0 geometry.

---

# Appendix F: full baseline comparison

| Method | Macro MRR | Best characteristic |
|:--|--:|:--|
| ReactZyme UniMol-3D + ESM | 0.3315 | Benchmark dual encoder |
| ReactZyme MAT-2D + ESM | 0.2798 | 2D reaction baseline |
| ReactZyme UniMol-3D + SaProt | 0.2483 | Structure-aware enzyme encoder |
| Fingerprint | 0.2593 | Non-neural chemistry control |
| GNN UniMol-3D + ESM | 0.3430 | Graph retrieval |
| GNN UniMol-3D + SaProt | 0.2947 | Graph + structure-aware enzyme |
| Bi-RNN UniMol-3D + ESM | 0.3808 | Sequence reaction baseline |
| Bi-RNN MAT-2D + ESM | 0.4182 | Best pre-TIGER listed baseline |
| CLIPZyme UniMol-3D + ESM | 0.3303 | Contrastive baseline |
| CLIPZyme MAT-2D + ESM | 0.3128 | Contrastive 2D baseline |
| TIGER ESM2Text | 0.5735 | Generated enzyme text fusion |
| TIGER ProtT3 | 0.5638 | Protein-text representation |
| Horizyn F3 | **0.6490** | Best reliable one-stage model |
| Horizyn Q5 | **0.6531** | Raw score winner; weak replication |

---

# Appendix G: loss definitions

For positive set `P`, distance matrix `D`, and inverse temperature `beta`:

```text
FullBatchMLNCE = beta * mean(D_ij for (i,j) in P)
                 + logsumexp(-beta * D)
```

- Degree tempering weights pair `(i,j)` by `degree(i)^(-alpha)`.
- Anchor-balanced losses average positives within a query, then average queries.
- Decoupling removes known positives from each anchor's negative partition.
- L4-L6 mix global MLNCE with cardinality-aware directional losses.
- The soft-rank term approximates how many hard negatives outrank a positive.
- Sigmoid EBM learns a global logit bias and balances positive/negative means.

---

# Appendix H: data and provenance controls

- Split-specific training; no merging of official ReactZyme protocols
- Train-derived validation and untouched released test rows
- Canonical isomeric SMILES for the L0-L7 campaign
- Source hashes and normalizer/extractor versions stamped into feature artifacts
- Feature coverage threshold before reuse
- Train-only fit for set descriptors, fingerprints, cofactor vocabularies, EC,
  mechanism, transition, and reaction-center labels
- Missing-label masks distinguish unknown from biological absence
- W&B run-per-split logging, immutable checkpoint hashes, and frozen test stage

---

# Sources

- `documents/README_reactzyme_all_ablations.md`
- `documents/README_training_protocol_top5_analysis.md`
- `documents/README_B0_configuration.md`
- `documents/README_F3_loss_improvement_proposals.md`
- `horizyn/docs/reactzyme_split_biology_report.md`
- `horizyn/runs/reactzyme_reaction_features_v1/eval/tiger_comparison.md`
- `horizyn/runs/reactzyme_e2r_pareto_v1/reports/summary.md`
- TIGER, arXiv:2605.24489, Table 1

All numbers are retained repository results as of 2026-08-14. Protocol caveats
shown in the deck are part of the result, not footnotes to ignore.
