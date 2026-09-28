# Enzyme Capability Training

This workflow adds an enzyme-only capability vector to the existing Horizyn
dual-tower retrieval model. The retrieval layout stays unchanged:

```text
reaction -> reaction embedding
enzyme   -> enzyme embedding
score    = dot/cosine similarity
```

The capability vector is computed once from enzyme-side inputs only. It is not a
reaction-conditioned encoder, cross-attention reranker, or pairwise inference
module.

## Leakage Rules

- Build enzyme capability labels from train pairs only.
- Exclude non-directional ReactZyme molecule-set rows from capability
  pretraining, enzyme-label propagation, hard-negative mining, and clean
  retrieval training.
- Export capability vectors for validation/test enzymes only by applying the
  enzyme-side encoder to enzyme-side features.
- Missing chemistry labels are unknown and are masked in auxiliary BCE losses.

## Label Construction

The capability labels are built from reactions first and enzymes second.

Reaction labels are deterministic annotations of directional reaction chemistry:
canonical reaction SMILES, DRFP, cofactor labels, cofactor tiers,
cofactor-filtered substrate/product classes, substrate-product transition
labels, reaction-center labels, reaction-type labels, and train-provenance EC
labels. Cofactors come from ChEBI participants, canonical ChEBI SMILES, curated
aliases, and structure/name rules. They are not inferred from EC alone.

The mixed `cofactor_labels` field is split into `core_cofactor_labels`,
`metal_ion_labels`, and `auxiliary_participant_labels`. For enzyme capability
pretraining, `core_cofactor_labels` should carry the main cofactor BCE weight;
metals/ions and auxiliary participants should be lower-weight context targets.
Substrate/product classes are computed after removing currency molecules and
detected cofactor molecules, with unfiltered audit columns retained. Reaction
centers come from atom mapping, using RXNMapper when a mapped reaction is not
already available.

Enzyme labels are train-only unions over observed train positives. For each
`enzyme_id`, the pipeline joins its train reaction IDs to
`reaction_features.parquet` and stores the union of EC, cofactor tiers,
reaction-center, substrate, product, transition, and reaction-type labels in
`enzyme_capability_labels.parquet`. This makes each enzyme row a train-derived
capability summary, not a complete curated protein annotation.

Missing label families are expected. A missing cofactor label may mean the
reaction has no cofactor, the source data did not list participants, or the
cofactor was outside the current dictionary. A missing reaction-center label
usually means atom mapping failed or the reaction was too long/complex. A
missing substrate/product label can mean all molecules were currency/cofactor
context or outside the SMARTS vocabulary. A missing EC label means the train
source provenance did not provide EC metadata. For training, missing means
unknown and should be masked; it should not be used as an all-zero negative
target.

## Data

The recommended clean source-collapse capability artifacts live under:

```text
data/processed/capability_features/train_exact_rhea_reconstructed/
```

This artifact contains 28,529 all-directional train reactions, 294,607 train
pairs, and 189,592 train enzymes. It removes the ReactZyme molecule-set
pseudo-reaction rows from the chemistry-supervised training table and replaces
recoverable train-side ReactZyme entries with unambiguous Rhea-backed
reactant/product directions.

The older pseudo-reaction artifact remains available under:

```text
data/processed/capability_features/train_exact/
```

That legacy inventory contains 35,219 reactions and 442,247 train pairs. Of
those reactions, 27,493 have valid directional reaction SMILES and 7,726 are
ReactZyme molecule-set rows without reaction arrows. Treat the molecule-set rows
as held-out/non-directional for clean ReactZyme evaluation unless a Rhea-backed
direction is recovered unambiguously.

ReactZyme train rows can also be reconstructed against the official ReactZyme
Rhea tables when the protein sequence has train-side Rhea annotations. This
does not use ReactZyme validation/test pairs; it replaces the unordered
ReactZyme molecule-set train rows with explicit Rhea substrate/product
directions:

```bash
python scripts/reconstruct_reactzyme_rhea_directional_train.py
python scripts/build_rhea_reconstructed_capability_artifacts.py
python scripts/fill_reaction_center_annotations.py \
  --capability-dir data/processed/capability_features/train_exact_rhea_reconstructed \
  --batch-size 32
python scripts/refresh_revised_capability_annotations.py \
  --capability-dir data/processed/capability_features/train_exact_rhea_reconstructed \
  --rebuild-demand \
  --rebuild-enzyme-labels
```

This writes:

```text
data/standardized/retrieval_training_source_collapse/train_exact/
  train_pairs_valid_rxn_rhea_reconstructed.csv
  train_rxns_valid_rxn_rhea_reconstructed.csv
  metadata_valid_rxn_rhea_reconstructed.json

data/paper/reactzyme/reconstructed_rhea_train/
  reactzyme_train_directional_rhea_pairs.csv
  reactzyme_train_directional_rhea_reactions.csv
  reactzyme_train_directional_rhea_members.csv
  reconstruction_report.json

data/processed/capability_features/train_exact_rhea_reconstructed/
  reaction_features.parquet
  reaction_drfp.npz
  reaction_demand_vectors.npz
  reaction_demand_metadata.parquet
  enzyme_capability_labels.parquet
  pair_capability_training.parquet
  rhea_reconstructed_capability_report.json
```

The current reconstruction has 294,607 train pairs, 28,529 all-directional
reactions, and 189,592 train enzymes. It reuses 27,493 existing annotated
directional reaction rows and annotates 1,036 newly recovered Rhea reactions
with RXNMapper enabled. Existing reused rows keep their prior reaction-center
annotations. The final RXNMapper completion pass leaves 28,152 reactions with
reaction-center status `ok`, 27,029 reactions with non-empty reaction-center
labels, and 377 very long or unmappable reactions with status `failed`. The
expanded focus refresh writes `annotation_dictionaries.json` version
`expanded_20260701` and yields demand vectors with shape `(28,529, 2,434)`,
13,245 reactions with core cofactor labels, 27,232 reactions with
cofactor-filtered substrate/product classes, and 24,078 reactions with
substrate-product transition labels. The dictionary snapshot supports 34 core
cofactor labels, 18 metal/ion labels, 30 reaction-center coarse families, 39
substrate/product SMARTS labels, and 24 directional transition rules; the
realized demand vocabulary contains only labels observed in this train split.

The enzyme-side cofactor enhancement can be rerun with:

```bash
python scripts/enhance_enzyme_cofactor_annotations.py \
  --capability-dir data/processed/capability_features/train_exact_rhea_reconstructed \
  --train-pairs data/standardized/retrieval_training_source_collapse/train_exact/train_pairs_valid_rxn_rhea_reconstructed.csv \
  --uniprot-molecules data/paper/reactzyme/raw/uniprot_molecules.tsv \
  --cleaned-uniprot-rhea data/paper/reactzyme/raw/cleaned_uniprot_rhea.tsv \
  --cofactor-dictionary data/processed/capability_features/chebi_cofactor_dictionary.tsv
```

This adds `enzyme_derived_*cofactor*_labels_train` and
`combined_*cofactor*_labels_train` columns to
`enzyme_capability_labels.parquet`, adds matching overlap diagnostics to
`pair_capability_training.parquet`, and writes
`enzyme_cofactor_annotation_report.json` plus
`enzyme_cofactor_labels_enhanced.csv`. On the current reconstructed train
split, 178,416 enzymes have a UniProt molecule record and 105,405 have
enzyme-side core cofactor labels, but combined core-cofactor coverage remains
108,922 enzymes. The local enzyme-side molecule source mostly confirms the
reaction-derived labels; it is not a complete UniProt cofactor-comment or
binding-site annotation source.

Build directional-only capability files with:

```bash
python scripts/build_directional_capability_split.py
```

This writes:

```text
train_pairs_directional.csv
data/processed/capability_features/train_exact_rhea_reconstructed/
  pair_capability_training.parquet
  enzyme_capability_labels.parquet
  enzyme_label_vocabs.json
  reaction_demand_vectors.npz
  annotation_dictionaries.json
```

## Static Enzyme Inputs

Capability pretraining consumes static enzyme-side branches:

```text
prot5_mean.npz
prot5_sleec.npz
lorentz_tangent.npz
```

They can be exported from an existing protein-pooling checkpoint/config:

```bash
python scripts/export_enzyme_capability_input_features.py \
  --checkpoint path/to/retrieval.ckpt \
  --config configs/retrieval_prott5_exact_mean_sleec_lorentz_gated_multimodal_r2e_allknown_hardneg_b512_4gpu.yaml \
  --out-dir data/processed/capability_features/train_exact_rhea_reconstructed/enzyme_features
```

## Stage 1: Capability Pretraining

Run:

```bash
python scripts/train_enzyme_capability_encoder.py \
  --config configs/enzyme_capability_pretrain.yaml
```

The default model now trains a factorized biological capability encoder:

- `ReactionDemandEncoder`: fixed reaction-demand vectors projected into global,
  cofactor, reaction-center, substrate, and product demand subspaces.
- `EnzymeCapabilityEncoder`: ProT5 mean + SLEEC pooled + optional frozen Lorentz
  tangent projected into matching enzyme capability subspaces.
- Auxiliary heads predict train-derived enzyme attributes from the matching
  enzyme subspace when labels are available.

The objective is:

```text
1.0 * R->E capability positive-set loss
+0.4 * E->R capability positive-set loss
+0.3 * cofactor-subspace contrastive loss
+0.3 * reaction-center-subspace contrastive loss
+0.2 * substrate-subspace contrastive loss
+0.2 * product-subspace contrastive loss
+0.10 * masked attribute BCE
+0.02 * weak EC BCE
```

The family-specific losses use the same train-positive maps as the global loss,
but they skip anchors and candidates whose corresponding label family is
missing. Missing cofactor, reaction-center, substrate, or product labels remain
unknowns rather than all-zero negatives.

The reaction-demand projector is frozen after the warmup epochs configured in
`reaction_demand_encoder.freeze_after_epochs`.

## Stage 1 v1: Biological Enzyme Stack Pretraining

The current recommended Stage 1 run is:

First verify the project runtime has the parquet dependencies:

```bash
PYTHONPATH=/datastor2/deep-proteins/EnzymeDiscovery/runtime_deps/enzyme_stage1 \
  /datastor2/deep-proteins/EnzymeDiscovery/env/bin/python - <<'PY'
import torch, lightning, pandas, pyarrow
print("runtime ok")
PY
```

```bash
/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python \
  scripts/train_enzyme_capability_encoder.py \
  --config configs/enzyme_bio_stack_pretrain_v1.yaml
```

On the current node the project venv has a broken `pip`, so pandas/pyarrow are
installed outside the git worktree in `runtime_deps/enzyme_stage1`. Launch with:

```bash
PYTHONPATH=/datastor2/deep-proteins/EnzymeDiscovery/runtime_deps/enzyme_stage1 \
  /datastor2/deep-proteins/EnzymeDiscovery/env/bin/python \
  scripts/train_enzyme_capability_encoder.py \
  --config configs/enzyme_bio_stack_pretrain_v1.yaml
```

This run trains an enzyme-only biological stack. It does not train against
reaction queries and does not use validation/test pairs. It uses the clean
Rhea-reconstructed directional train artifact and selected train-derived
enzyme labels:

```text
combined_core_cofactor_labels_train
substrate_product_transition_labels_train
substrate_class_labels_train
product_class_labels_train
```

Reaction-center labels are kept for diagnostics and optional hard-negative
context, but they are not allowed to make positives by themselves. EC labels
and broad reaction-type labels are excluded from the Stage 1 v1 objective.

The encoder uses:

```text
ProT5 residue embeddings
-> LayerNorm
-> shared residue adapter, 1024 -> 512
-> learned multi-query attention pools
   q_cofactor, q_transition, q_substrate, q_product

plus frozen ProT5 mean, SLEEC pooled ProT5, and Lorentz tangent branches
-> factorized biological projection
-> normalized enzyme biological vector
```

The composite biological similarity is fixed in config:

```text
0.40 * substrate/product transition overlap
0.35 * core cofactor overlap
0.15 * substrate class overlap
0.10 * product class overlap
```

A positive pair must have at least two known selected label families and must
share either a core cofactor or a substrate/product transition. Missing labels
are masked as unknown.

Before training, the script writes:

```text
outputs/enzyme_bio_pretrain_v1/label_quality_report.json
```

The configured gates require adequate cofactor, transition, substrate, product,
and usable-anchor coverage. If those gates fail, the v1 config aborts rather
than training on a weak supervision signal.

During validation, the run logs representation-placement diagnostics:

```text
val/placement/composite_positive_negative_cosine_gap
val/placement/composite_jaccard_at_10
val/placement/composite_enrichment_at_10
val/placement/cofactor_jaccard_at_10
val/placement/transition_jaccard_at_10
val/placement/substrate_jaccard_at_10
val/placement/product_jaccard_at_10
val/placement/effective_rank
val/placement/collapse_score
val/placement/attention_entropy_*
val/placement/query_cosine_similarity_*
```

The checkpoint monitor is
`val/placement/composite_positive_negative_cosine_gap`, because Stage 1 should
first prove that the enzyme embedding space separates biologically similar
enzymes from unrelated or weakly related enzymes.

## Stage 2: Export Capability Vectors

Run:

```bash
PYTHONPATH=/datastor2/deep-proteins/EnzymeDiscovery/runtime_deps/enzyme_stage1 \
  /datastor2/deep-proteins/EnzymeDiscovery/env/bin/python \
  scripts/export_enzyme_capability_vectors.py \
  --checkpoint outputs/enzyme_bio_pretrain_v1/best_enzyme_capability.ckpt \
  --enzyme-features data/processed/capability_features/train_exact/enzyme_features \
  --residue-h5 data/standardized/retrieval_training_source_collapse/train_exact/fit_proteins_with_clipzyme_eval_prott5_residue.h5 \
  --out-dir outputs/enzyme_bio_pretrain_v1 \
  --vector-filename enzyme_bio_vectors.npz
```

The NPZ contains vectors for enzymes with all required static feature branches.
The metadata also records requested enzymes that are missing vectors. Retrieval
uses zero vectors with `capability_mask = false` for those cases.

## Stage 3-5: Retrieval Integration

Use:

```text
enzyme_input_mode: raw_mean_sleec_hyperbolic_capability_gated
```

The enzyme stack becomes:

```text
ProT5 mean
+ SLEEC pooled ProT5
+ frozen Lorentz tangent
+ frozen pretrained capability vector
-> gated enzyme fusion
-> enzyme projection
```

First clean retrieval run:

```bash
python scripts/train_protein_pooling.py \
  --config configs/enzyme_only_capability_tuning.yaml
```

This freezes the reaction side, detaches reaction embeddings in the loss, trains
only enzyme-side fusion/projection components, and adds a small no-capability
anchor penalty.

The follow-up joint calibration run uses:

```bash
python scripts/train_protein_pooling.py \
  --config configs/joint_capability_retrieval.yaml
```

## Recommended Controls

- Baseline: ProT5 mean + SLEEC + Lorentz.
- Random capability branch.
- EC-only capability branch.
- Chemistry capability branch.
- Chemistry capability branch with residual adapter.
- Joint projection-head tuning.

The key comparison is whether the chemistry capability branch improves
reaction-to-enzyme retrieval beyond the random and EC-only controls.
