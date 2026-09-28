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
