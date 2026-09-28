# ReactZyme Reaction-Cluster Validation

## Purpose

This protocol replaces random-row validation as the Reaction-Sim checkpoint proxy.
It asks whether an enzyme can rank reactions from chemical families excluded from
training while preserving the released ReactZyme test split for final evaluation.

## Cluster Construction

Only the 7,340 released Reaction-Sim training reactions are used. Each reaction is
represented by a concatenation of:

- the L2-normalized 512-bit, radius-2 F3 Morgan molecule-set mean fingerprint;
- the L2-normalized frozen forward ReactionT5v2 embedding;
- square-root block scaling with weights 0.70 and 0.30 respectively.

The ReactZyme paper describes Needleman-Wunsch reaction-SMILES similarity, but the
local release does not include its training similarity matrix or cluster assignments.
The hybrid representation is the stable fallback permitted by the protocol.

At each threshold, reactions are graph nodes and an edge joins two reactions when
their cosine similarity reaches the threshold. Connected components are indivisible
clusters. This is stricter than average linkage: no validation-to-training edge may
reach the panel threshold.

| Panel | Validation pairs | Validation reactions | Validation proteins | Maximum train similarity |
|---|---:|---:|---:|---:|
| 0.80 sensitivity | 8,969 (5.48%) | 621 | 8,968 | 0.799083 |
| **0.85 primary** | **16,426 (10.03%)** | **1,084** | **16,418** | **0.849611** |
| 0.90 sensitivity | 16,388 (10.01%) | 1,023 | 16,379 | 0.899967 |

The 0.80 graph has a 6,719-reaction giant component, so only 5.48% of pairs can be
held out without violating the edge constraint. All panels have zero reaction-ID,
cluster, exact-SMILES, and exact-pair overlap and copy released test CSVs byte-for-byte.

## Metric And Selection

The primary metric is enzyme-anchor ReactZyme MRR, which averages reciprocal rank
over every positive reaction for an enzyme:

```text
E->R all-positive MRR = mean_e(mean_{r in P(e)}(1 / rank_e(r)))
```

Every validation epoch is retained. Offline selection applies:

```text
primary:   maximize E->R all-positive MRR on similarity_0p85
constraint: R->E all-positive MRR >= clean F3 baseline - 0.015
secondary: maximize harmonic mean of E->R and R->E all-positive MRR
```

The generated training config monitors
`val/enzyme_to_reaction/reactzyme_mrr`, evaluates both directions every epoch,
sets `save_top_k: -1`, and disables early stopping. The R->E constraint is applied
offline because Lightning's single scalar checkpoint monitor cannot express it.

## Existing F3 Audit

Only literal checkpoint labels 26, 28, and 29 remain locally. Labels 10-25, 27,
and 30 are unavailable, and the original W&B run had model artifact logging disabled.

The available files were evaluated, but they predate the cluster split. The overlap
audit found that 1,031/1,084 primary-panel reactions and 14,747/16,426 exact pair keys
were in their original training data. Their approximately 0.974 E->R MRR values are
therefore contamination diagnostics and cannot select a checkpoint. The report
suppresses promotion accordingly.

## Artifacts

- Protocol: `data/revised_protocols/reactzyme_reaction_cluster_validation_v1/`
- Clean F3 config: `configs/benchmarks/reactzyme_f3_cluster_proxy_v1.yaml`
- Retrospective report: `runs/reactzyme_f3_cluster_validation_v1/reports/similarity_0p85.md`
- Train-fitted chemistry: `data/revised_protocols/reactzyme_reaction_cluster_validation_v1/similarity_0p85/features/reaction_set/`

Rebuild and evaluate with:

```bash
../env/bin/python scripts/build_reaction_cluster_validation.py --overwrite
../env/bin/python scripts/build_reaction_set_features.py \
  --train-reactions data/revised_protocols/reactzyme_reaction_cluster_validation_v1/similarity_0p85/train_rxns.csv \
  --validation-reactions data/revised_protocols/reactzyme_reaction_cluster_validation_v1/similarity_0p85/validation_rxns.csv \
  --test-reactions data/revised_protocols/reactzyme_reaction_cluster_validation_v1/similarity_0p85/test_rxns.csv \
  --cofactor-dictionary data/processed/capability_features/train_exact_rhea_reconstructed/cofactor_dictionary.csv \
  --out-dir data/revised_protocols/reactzyme_reaction_cluster_validation_v1/similarity_0p85/features/reaction_set
../env/bin/python scripts/create_f3_cluster_training_config.py
../env/bin/python scripts/evaluate_f3_reaction_cluster_checkpoints.py \
  --checkpoint-training-pairs \
  data/revised_protocols/reactzyme_reaction_cluster_validation_v1/similarity_0p85/train_pairs.csv
```

Do not use the last command's selection mode on old F3 checkpoints with the clean
training-pair argument. The argument must identify the actual training rows of the
evaluated checkpoint.
