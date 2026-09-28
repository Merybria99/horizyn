# Training Protocol Analysis and Top Five Solutions

Last updated: 2026-08-14

This report ranks the trained enzyme-reaction retrieval configurations by
engineering value, not test MRR alone. The complete numeric inventory is in
`documents/README_reactzyme_all_ablations.md`.

## Decision Criteria

The ranking uses four qualitative criteria:

1. **Performance:** corrected ReactZyme paper-test MRR across all six tasks,
   including the weak reaction-similarity E->R column.
2. **Protocol confidence:** official split-specific training, train-derived
   validation, untouched released test data, paper candidate sets, and corrected
   all-positive MRR.
3. **Simplicity:** number of training stages, learned branches, feature-generation
   dependencies, label sources, direction-specific behavior, and optimization
   machinery.
4. **Evidence strength:** causal ablation support and rerun/multi-seed evidence.

Simplicity is relative. Even F0 uses frozen ProtT5, ReactionT5v2, Uni-Mol2,
ChIRo, and a pretrained SLEEC scorer. It is simple only relative to the other
multimodal configurations in this repository.

## Protocol Generations

| Campaign | Main question | Best stored result | Protocol confidence | Complexity verdict |
|:--|:--|--:|:--|:--|
| Initial encoder screens | ESM2 vs ProtT5; attention; SLEEC/hyperbolic features | ESM2 + SLEEC-hyperbolic macro 0.4089 | Low: old global candidate bank | Useful architecture screening only |
| Historical R01-R26 | Mean pooling, SLEEC/Lorentz, ReactionT5, UniMol2, ChIRo, hard negatives | R07 historical aggregate 0.6377 | Low: mixed candidate and split protocols | R07 is a compelling simple recipe, but must be rerun before comparison |
| A0-A15 | SLEEC, EC, capability, structure, and chemistry latent organization | A10 macro 0.5034 | Low: pre-reframing filtered data | Too many coupled changes for a final protocol |
| B0-B4 | Mechanism, cofactor, and EC supervision | B0 legacy macro 0.7524 | Medium for E->R; invalid first-positive R->E | Auxiliary labels did not justify their stages |
| Geometry R0-R6 | Positive semantics, degree balancing, SupCon, learned temperature | R6 legacy macro 0.7801 | Medium for E->R; invalid first-positive R->E | Interesting loss study, not leaderboard evidence |
| F0-F6 | Reaction representation under the corrected paper protocol | F3 macro 0.6490 | High | Strongest causal campaign |
| Q0-Q5 | Frozen-base E->R adapters, mining, dense chemistry, factorization | Q5 macro 0.6531 | High metric semantics; weak replication | Gains are too small for their added machinery |
| Horizyn F4 | Cross-dataset transfer vs in-domain training | In-domain R->E MRR 0.3228 | High for Horizyn only | Shows that data matching matters more than a small architecture tweak |

## What Every Campaign Established

### Historical Architecture Runs

The robust historical result was not the most elaborate model. Mean-pooled
ProtT5 with ReactionT5v2 and UniMol2, trained with observed-pair
FullBatchMLNCE, produced the strongest old-protocol aggregate. Adding ChIRo
improved the original-test aggregate, but SLEEC/Lorentz, teacher, capability,
EC-structure, and hard-negative extensions did not consistently beat that
simple reference. These runs support the current frozen encoder choices, but
their scores cannot support a ReactZyme paper claim.

### A-Series: Latent Organization

| Change | Observed lesson |
|:--|:--|
| A0 -> A1 | SLEEC improved the historical macro by 0.0205. |
| A1 -> A2 | Switching to all-known in-batch positives added only 0.0003. |
| A2 -> A3 | An EC block added 0.0155 under the old protocol. |
| A3 -> A4 | Hard negatives added 0.0071; structure loss in A5 then lost 0.0048. |
| A4 -> A6 -> A10 | Capability blocks produced the main later gain, reaching 0.5034. |
| A10 -> A11/A12/A15 | Chemistry-token, EC-structure removal, and stronger negatives did not improve the macro. |

The A-series says that sequence-side functional structure can help, but it does
not identify a clean final recipe. Capability, EC, masks, structure, and hard
negatives were changed on top of an obsolete data protocol, and the winning
gain was concentrated in E->R.

### B-Series: Biological Supervision

| Config | Added stage/supervision | Legacy macro | Delta from B0 |
|:--|:--|--:|--:|
| B0 | Retrieval only | 0.7524 | - |
| B1 | Mechanism pretraining and auxiliary loss | 0.7470 | -0.0054 |
| B2 | Cofactor pretraining and auxiliary loss | 0.7327 | -0.0197 |
| B3 | Joint mechanism + cofactor supervision | 0.7465 | -0.0059 |
| B4 | B3 + EC-supervised Lorentz initialization | 0.7454 | -0.0070 |

The clean control won. Mechanism, cofactor, and EC labels increased data and
training complexity without improving the aggregate. B4 improved one
enzyme-similarity E->R cell, but that isolated gain does not justify three
coupled biological priors as the default protocol.

### Geometry R-Series

Observed-pair vs all-known positives was effectively a tie in R0/R1. Degree
balancing and learned temperature improved the legacy aggregate, but much of
the apparent gain came from the obsolete first-positive R->E metric. Mean E->R
improved only modestly, and a corrected R5 reaction-similarity R->E diagnostic
fell from 0.6219 first-positive MRR to 0.4090 all-positive MRR. These losses
deserve a corrected rerun, but none should replace FullBatchMLNCE today.

### F-Series: Reaction Representation

| Config | Main change | Corrected macro | Simplicity assessment |
|:--|:--|--:|:--|
| F0 | Directional side composition; no handcrafted chemistry | 0.6363 | Highest among strong models |
| F1 | Unordered molecule set | 0.6285 | Simple, but dominated by F0 |
| F2 | F1 without ReactionT5v2 | 0.3076 | Simple but incomplete; rejects removal of ReactionT5v2 |
| F3 | F1 + 617-D train-fitted RDKit set chemistry | 0.6490 | Best one-stage balance |
| F4 | F3 + fixed normalized modality blocks | 0.6171 | Interpretable and best reaction-sim E->R |
| F5 | F4 + uniquely matched Rhea direction | 0.6096 | External matching cost without a gain |
| F6 | F5 + RXNMapper center labels | 0.6171 | Highest preprocessing burden; no aggregate gain |

F3's 617-D vector is comparatively controlled: a 512-bit mean Morgan
fingerprint, 64 train-standardized molecular descriptor aggregates, four set
statistics, and up to 37 core-cofactor indicators. Its schema and vocabulary
are fitted on training reactions only. It does not require held-out labels,
Rhea matching, or RXNMapper.

### Q-Series: Direction-Specific Adapters

Q0 is configuration-identical to F3. Q1-Q5 first train/select Q0, freeze the
entire base model, and then train only an E->R reaction residual adapter for up
to 15 epochs. R->E therefore remains exactly the Q0 geometry.

| Config | Adapter inputs/objective | Macro | Delta from Q0 | Added complexity |
|:--|:--|--:|--:|:--|
| Q0 | Fresh F3 base | 0.6504 | - | One-stage reference |
| Q1 | Base embedding; E->R MLNCE + identity regularization | 0.6512 | +0.0008 | Second stage and direction-specific embedding |
| Q2 | Q1 + mined E->R hard negatives | 0.6489 | -0.0015 | Mining, custom batches, margin loss |
| Q3 | Q1 + 1,114-D directional/center chemistry | 0.6495 | -0.0009 | DRFP, signed Morgan deltas, mapped centers |
| Q4 | Q1 + factorized raw modalities | 0.6518 | +0.0014 | Extra projections, but no new external feature source |
| Q5 | Q4 + Q2 + Q3 | 0.6531 | +0.0027 | Maximum data, sampler, and objective complexity |

The exact F3/Q0 rerun changes macro MRR by only 0.0014, but changes
reaction-similarity E->R by 0.0156. Therefore every Q-series macro gain is
smaller than observed rerun variability. The planned Q multi-seed confirmation
did not complete. Q5 is the raw score winner, but it is not yet evidence of a
better training recipe.

For the `wet_lab` workflow, which queries a reaction and retrieves enzymes,
Q1-Q5 provide no benefit: their adapter executes only for E->R, while wet-lab
search is R->E.

## Top Five Engineering Solutions

### 1. F3 / Q0: Default Balanced Model

**Recommendation:** use this as the primary training and deployment protocol.

- Corrected macro MRR: 0.6490 in F3 and 0.6504 in the exact Q0 rerun.
- One retrieval-training stage, 30 epochs, observed-pair FullBatchMLNCE.
- No enzyme text, biological auxiliary loss, EC warm start, hard-negative
  mining, Rhea matching, or RXNMapper dependency.
- Deterministic train-fitted RDKit chemistry gives a meaningful 0.0127 macro
  gain over F0 in the original F campaign.
- It beats TIGER ESM2Text in five of six cells and by about 0.076 macro MRR.

The main caveat is that the shared enzyme/reaction fusion model is still large,
and the exact rerun exposes nontrivial split-level stochastic variation.

### 2. F0: Minimal Strong Protocol

**Recommendation:** retain as the mandatory control and use when preprocessing
or portability matters more than the last 0.013 macro MRR.

- Corrected macro MRR: 0.6363.
- Single-stage FullBatchMLNCE with the same frozen pretrained modalities.
- No RDKit set-feature schema, cofactor dictionary, biological labels, external
  reaction mapping, mining, or warm start.
- It exceeds TIGER ESM2Text macro MRR by 0.0628.

F0 is the cleanest answer to whether the pretrained modalities alone solve the
task. It is strictly worse than F3 on performance, but remains on the
performance/simplicity Pareto frontier.

### 3. F4: Interpretable Reaction-Similarity Specialist

**Recommendation:** use when reaction-similarity E->R is the priority or when
fixed modality allocation is preferred over learned fusion.

- Corrected macro MRR: 0.6171.
- Reaction-similarity E->R MRR: 0.5023, the best internal result and only
  0.0157 below TIGER ESM2Text.
- One training stage.
- Fixed modality blocks expose the allocation directly: ReactionT5v2 0.25,
  UniMol2 0.375, ChIRo 0.1875, and set chemistry 0.1875.
- Removes learned modality competition and the final reaction output MLP from
  the fusion path.

Its lower time and R->E scores make it a specialist, not the general default.

### 4. Q1: Cheapest Directional Fine-Tuning

**Recommendation:** use only when a separate E->R embedding is operationally
acceptable and a cheap post-training adjustment is needed.

- Corrected macro MRR: 0.6512.
- Freezes Q0 and trains a gated 512-D residual MLP with identity weight 0.05.
- No hard-negative mining or new chemistry features.
- Preserves all Q0 R->E behavior by construction.

The 0.0008 macro gain is below observed rerun variability. Q1 is ranked for its
modularity and low incremental compute, not for a proven accuracy improvement.

### 5. Q4: Cleanest High-Score Adapter

**Recommendation:** prefer Q4 over Q5 if an E->R-specific second stage is used.

- Corrected macro MRR: 0.6518.
- Best enzyme-similarity E->R MRR: 0.9743.
- Adds factorized raw ReactionT5v2, UniMol2, ChIRo, and set-chemistry blocks to
  the Q1 adapter.
- Does not require hard-negative mining, mapped reaction centers, or the dense
  directional feature pipeline.

Q4 gives most of the Q-series score with substantially fewer moving parts than
Q5, but still lacks multi-seed confirmation.

## Raw Winner Not Selected as Default

Q5 has the highest corrected macro MRR, 0.6531, and the best time-split E->R
MRR, 0.8271. It is excluded from the engineering top five because its 0.0013
gain over Q4 requires all of the following: a pretrained Q0 checkpoint, a
direction-specific adapter, mined hard-negative batches, a margin term,
factorized raw modalities, and a 1,114-D dense directional/center feature
pipeline. The gain is smaller than observed F3/Q0 rerun variability and has no
completed multi-seed support.

## Final Recommendation

Use **F3/Q0** as the main paper and wet-lab model, keep **F0** as the minimal
control, and retain **F4** as the reaction-similarity specialist. Treat Q1/Q4 as
optional E->R post-training experiments. Do not add BioFP/EC supervision,
Rhea/RXNMapper features, dense directional features, or hard-negative mining to
the default protocol until each produces a corrected, multi-seed gain larger
than the F3/Q0 rerun variation.

The most valuable unverified experiment is not another complex branch: rerun
the historical R07 recipe, ProtT5 mean pooling with ReactionT5v2 + UniMol2 and
observed-pair FullBatchMLNCE, under the current paper protocol. It is materially
simpler than the F/Q enzyme stack and is the cleanest test of whether SLEEC,
Lorentz, ChIRo, and biological-factorized enzyme blocks are actually necessary.
