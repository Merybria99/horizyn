# Enzyme Capability Pretraining

This pipeline adds a static enzyme-only capability vector to the existing
Horizyn/TIGER-style dual tower. The retrieval model remains:

`reaction -> reaction embedding`, `enzyme -> enzyme embedding`, score by dot or
cosine similarity.

The capability vector is trained before retrieval and can be loaded as a fourth
enzyme branch beside ProT5 mean, SLEEC-pooled ProT5, and frozen Lorentz tangent
features.

## Leakage Rules

Enzyme labels are constructed only from training split enzyme-reaction pairs.
Reaction-level public features may be computed for all reactions, but validation
and test pairs must not contribute to `enzyme_capability_labels.parquet`.

The label builder accepts optional validation/test pair files and asserts that
all rows in `pair_capability_training.parquet` have `source_split == "train"`.

## Workflow

Build reaction features:

```bash
python scripts/build_reaction_features.py \
  --reaction-smiles data/reaction_smiles.csv \
  --train-pairs data/source_collapse/train_exact/train_pairs.csv \
  --rhea2ec data/rhea/rhea2ec.tsv \
  --out-dir data/processed/capability_features \
  --drfp-bits 2048 \
  --use-rxnmapper false
```

Build train-only enzyme labels:

```bash
python scripts/build_enzyme_capability_labels.py \
  --train-pairs data/source_collapse/train_exact/train_pairs.csv \
  --reaction-features data/processed/capability_features/reaction_features.parquet \
  --out-dir data/processed/capability_features
```

Build reaction demand vectors:

```bash
python scripts/build_reaction_demand_vectors.py \
  --reaction-features data/processed/capability_features/reaction_features.parquet \
  --reaction-drfp data/processed/capability_features/reaction_drfp.npz \
  --out-dir data/processed/capability_features
```

Pretrain and export:

```bash
python scripts/pretrain_enzyme_capability.py \
  --config configs/enzyme_capability_pretrain.yaml

python scripts/export_enzyme_capability_vectors.py \
  --checkpoint outputs/enzyme_capability_pretrain/best_enzyme_capability.ckpt \
  --enzyme-features data/features \
  --out-dir outputs/enzyme_capability_pretrain
```

## Feature Extraction

The first implementation is conservative and deterministic. It uses RDKit
canonical reaction SMILES, DRFP bits, ChEBI-backed cofactor aliases, SMARTS
substrate/product classes, and rule-based reaction types. Atom-mapped reaction
centers are used when supplied; RXNMapper is optional and disabled by default.

The built-in cofactor recognizer includes the incoming `has role cofactor`
terms listed under ChEBI `CHEBI:23357`, plus curated broad aliases such as
`NAD`, `NADP`, `FAD`, `FMN`, `PLP`, `TPP`, `PQQ`, `THF`, `cobalamin`,
`molybdopterin`, and metal-ion labels. A custom JSON/CSV/TSV dictionary can be
merged at build time with `scripts/build_reaction_features.py
--cofactor-dictionary`.

To refresh the exhaustive ChEBI dictionary with term synonyms and SMILES:

```bash
python scripts/build_chebi_cofactor_dictionary.py \
  --out data/processed/capability_features/chebi_cofactor_dictionary.tsv
```

Then pass that TSV to reaction annotation. Exact ChEBI-role SMILES matches are
recorded with `cofactor_from_chebi_smiles_match`. Some ChEBI cofactor-role
fillers are small or currency-like molecules, so downstream analyses should
interpret those labels as ChEBI-role evidence rather than always enzyme-specific
cofactor evidence.

Uncertain features are left empty and recorded in `quality_flags`.

## Pretraining Objective

The enzyme capability encoder consumes enzyme-side features only:

- ProT5 mean
- SLEEC pooled ProT5
- Lorentz tangent

It outputs a normalized 256-dimensional vector. A reaction demand encoder maps
reaction demand vectors into the same space.

The loss combines:

- reaction-to-enzyme multi-positive supervised contrastive loss
- enzyme-to-reaction multi-positive supervised contrastive loss
- low-weight BCE auxiliary heads for cofactor, reaction center,
  substrate/product, reaction type, and EC labels

EC is treated as useful but coarse auxiliary supervision and should not dominate
the capability representation.

## Retrieval Integration

Enable the static capability branch with:

```yaml
data:
  protein_capability_vectors_path: outputs/enzyme_capability_pretrain/enzyme_capability_vectors.npz
  protein_capability_metadata_path: outputs/enzyme_capability_pretrain/enzyme_capability_metadata.parquet

model:
  enzyme_input_mode: raw_mean_sleec_hyperbolic_capability_gated
  capability_vector:
    dim: 256
    freeze: true
    adapter: false
    dropout: 0.1
```

The enzyme embedding remains reusable and query-independent. No cross-attention
or pairwise reranking is added.

## Recommended Ablations

Compare the current baseline against:

- random capability vector
- EC-only pretrained capability vector
- DRFP-only capability vector
- DRFP + cofactor
- DRFP + cofactor + reaction center
- DRFP + cofactor + reaction center + substrate/product
- full capability vector frozen
- full capability vector with residual adapter

The key comparison is EC-only capability versus chemistry-rich capability. If
the latter improves reaction-to-enzyme retrieval more, the gain is likely from
catalytic capability signal rather than hierarchy alone.

## Limitations

The v1 path does not infer exact reaction centers without atom mapping, does not
use reaction-conditioned enzyme encoding, and does not use test annotations.
ChEBI ontology and SLEEC-guided residue pooling can be added later as stricter
feature sources without changing the retrieval layout.
