# Data And Annotation Inventory

Last updated: 2026-07-01.

This document summarizes the data currently available under
`horizyn/data`, the standardized retrieval splits used by the current
experiments, and the new enzyme-capability annotation artifacts.

The primary current clean capability annotation artifact is:

```text
horizyn/data/processed/capability_features/train_exact_rhea_reconstructed/
```

This is the preferred capability-pretraining input when directional reaction
chemistry is required. It replaces ReactZyme train molecule-set rows with
train-side, Rhea-backed directional substrate/product reactions where possible.

The older pseudo-reaction annotation artifact remains available at:

```text
horizyn/data/processed/capability_features/train_exact/
```

It was built from the source-collapse exact training split after removal of the
4 invalid SABIO rows. It keeps the ReactZyme molecule-set rows as
non-directional pseudo chemistry. Both artifact families use only training
pairs for enzyme-level capability labels.

## Key Points

- The active retrieval corpus is the source-collapse dataset:
  `horizyn/data/standardized/retrieval_training_source_collapse`.
- The recommended clean annotation run is
  `train_exact_rhea_reconstructed` with 28,529 all-directional reactions and
  294,607 train pairs.
- The older `train_exact` pseudo artifact covers 35,219 reactions and 442,247
  train pairs; 7,726 of those reactions are ReactZyme molecule-set rows without
  a reaction arrow.
- The ReactZyme reconstruction removes 178,440 non-directional molecule-set
  pairs and adds 240,024 Rhea-backed directional train pairs.
- Reaction centers are available for 27,029 reconstructed reactions. The
  remaining 377 directional reactions were attempted with RXNMapper and marked
  as explicit failures because they are too long or otherwise unmappable.
- Cofactors are detected from ChEBI/cofactor dictionary, structural, and alias
  rules, not from EC.
- EC annotations are populated from existing train-pair sources: Horizyn
  UniProt EC labels, CLIPZyme direct `ec` labels, and SABIO `ec_numbers_json`.
  ReactZyme molecule-set rows still do not carry EC in the official split files.
- Enzyme capability labels are train-only. Do not build enzyme labels from
  validation or test pairs.

## Directory Map

| Directory | Purpose |
|---|---|
| `horizyn/data/paper/` | Paper benchmark assets: ReactZyme, CLIPZyme/EnzymeMap, SABIO-RK. |
| `horizyn/data/standardized/retrieval_training_source_collapse/` | Main standardized retrieval training, validation, and benchmark data. |
| `horizyn/data/standardized/global_unique_retrieval/` | Standardized global unique train/test splits for stricter leakage-control experiments. |
| `horizyn/data/processed/capability_features/train_exact_rhea_reconstructed/` | Preferred all-directional reaction-feature, demand-vector, and enzyme-capability annotation artifacts. |
| `horizyn/data/processed/capability_features/train_exact/` | Older pseudo-reaction capability artifacts that retain ReactZyme molecule-set rows. |
| `horizyn/data/paper/reactzyme/reconstructed_rhea_train/` | ReactZyme train-to-Rhea reconstruction audit tables. |
| `horizyn/data/standardized/retrieval_training_source_collapse/test/reaction_features/` | Precomputed reaction representation HDF5 files for the held-out test suite. |

## Source-Collapse Training Splits

These are the main training splits used by the current Horizyn-style retrieval
experiments.

| Split | File | Pairs | Reactions | Proteins | Notes |
|---|---|---:|---:|---:|---|
| `train_exact` | `train_pairs.csv` | 442,251 | 35,223 | 189,593 | Exact sequence dedupe. |
| `train_exact` | `train_pairs_valid_rxn_pseudo.csv` | 442,247 | 35,219 | 189,592 | Used by the capability annotation run. Keeps ReactZyme molecule-set rows as pseudo/non-directional reactions. |
| `train_exact` | `train_pairs_valid_rxn_rhea_reconstructed.csv` | 294,607 | 28,529 | 189,592 | Preferred clean capability split. Replaces ReactZyme molecule-set train rows with Rhea-backed directional reactions. |
| `train_exact` | `train_pairs_valid_rxn_pseudo_nonempty_prott5.csv` | 442,247 | 35,219 | 189,592 | Same rows with non-empty ProT5 feature coverage. |
| `train_nr90` | `train_pairs.csv` | 312,046 | 35,223 | 129,400 | MMseqs 90 percent identity collapse. |
| `train_nr90` | `train_pairs_valid_rxn.csv` | 190,818 | 27,493 | 116,942 | Directional reaction-SMILES-only subset. |
| `train_nr90` | `train_pairs_valid_rxn_pseudo.csv` | 312,042 | 35,219 | 129,399 | Keeps the ReactZyme molecule-set rows. |
| `train_nr90` | `train_pairs_valid_rxn_nonempty_prott5.csv` | 190,818 | 27,493 | 116,942 | Directional-only subset with ProT5 feature coverage. |
| `train_nr90` | `train_pairs_valid_rxn_nonempty_esm2.csv` | 190,818 | 27,493 | 116,942 | Directional-only subset with ESM2 feature coverage. |
| `train_nr90` | `train_pairs_valid_rxn_nonempty_esmc.csv` | 190,818 | 27,493 | 116,942 | Directional-only subset with ESMC feature coverage. |
| `train_nr50` | `train_pairs.csv` | 148,180 | 35,223 | 52,563 | MMseqs 50 percent identity collapse. |

The source-collapse training reaction universe is shared across `exact`,
`nr90`, and `nr50` before protein clustering. Therefore the current
train-exact reaction-level annotations can be reused for matching reaction IDs
in `train_nr90` and `train_nr50`. Enzyme-level capability labels, however, are
split-specific and should be rebuilt if the collapsed protein IDs are used.

## Validation Split

The validation split is the CLIPZyme/EnzymeMap evaluation split converted to
Horizyn-compatible pair tables.

| File | Pairs | Reactions | Proteins | Notes |
|---|---:|---:|---:|---|
| `validation/clipzyme_eval/pairs.csv` | 48,074 | 17,266 | 13,121 | Original converted validation pairs. |
| `validation/clipzyme_eval/pairs_horizyn.csv` | 48,074 | 17,266 | 13,121 | Horizyn-compatible adapter. |
| `validation/clipzyme_eval/pairs_horizyn_nonempty_prott5.csv` | 47,904 | 17,232 | 13,078 | ProT5 feature-covered subset. |
| `validation/clipzyme_eval/pairs_horizyn_nonempty_esm2.csv` | 47,904 | 17,232 | 13,078 | ESM2 feature-covered subset. |
| `validation/clipzyme_eval/pairs_horizyn_nonempty_esmc.csv` | 47,904 | 17,232 | 13,078 | ESMC feature-covered subset. |
| `validation/clipzyme_eval/reactions.csv` | n/a | 17,266 | n/a | Validation reaction table. |

No enzyme-capability labels are currently generated for validation, and they
should not be generated from validation pairs. Reaction-level public features
may be computed separately for validation reactions if needed.

## Protein-Disjoint Test Splits

These are the currently used held-out benchmark splits after filtering test
proteins away from training proteins.

### Exact Protein-Disjoint Tests

| Task | Pairs | Reactions | Proteins | Candidate IDs |
|---|---:|---:|---:|---:|
| Horizyn | 26,429 | 715 | 25,551 | 94,288 |
| ReactZyme time | 3,089 | 951 | 3,089 | 70,353 |
| ReactZyme enzyme_smi | 3,536 | 886 | 3,534 | 70,353 |
| ReactZyme reaction_smi | 5,229 | 169 | 5,228 | 70,353 |
| Shared candidate pool | n/a | n/a | n/a | 164,641 |

Root:

```text
horizyn/data/standardized/retrieval_training_source_collapse/test/protein_disjoint_exact/
```

### NR90 Protein-Disjoint Tests

| Task | Pairs | Reactions | Proteins | Candidate IDs |
|---|---:|---:|---:|---:|
| Horizyn | 20,386 | 483 | 20,299 | 25,229 |
| ReactZyme time | 1,415 | 283 | 1,415 | 20,956 |
| ReactZyme enzyme_smi | 1,024 | 243 | 1,024 | 20,956 |
| ReactZyme reaction_smi | 704 | 64 | 704 | 20,956 |
| Shared candidate pool | n/a | n/a | n/a | 46,185 |

Root:

```text
horizyn/data/standardized/retrieval_training_source_collapse/test/protein_disjoint_nr90/
```

No enzyme-capability labels should be created from these test pairs. If a test
reaction is public and has a valid reaction SMILES, reaction-level features can
be computed for scoring or diagnostics, but test enzyme labels must remain
unused.

## Legacy Shared Test Suite

The older shared test-candidate folder is still available:

```text
horizyn/data/standardized/retrieval_training_source_collapse/test/horizyn_reactzyme_shared_candidates/
```

| Candidate file | Count |
|---|---:|
| `candidate_ids.txt` | 394,459 |
| `horizyn_sota_candidate_ids.txt` | 216,132 |
| `reactzyme_time_candidate_ids.txt` | 178,327 |
| `reactzyme_enzyme_smi_candidate_ids.txt` | 178,327 |
| `reactzyme_reaction_smi_candidate_ids.txt` | 178,327 |

## Paper Benchmark Assets

These are the original paper benchmark assets under `horizyn/data/paper`.

| Dataset | Split | Train pairs | Test pairs | Reactions | Candidate IDs | Notes |
|---|---|---:|---:|---:|---:|---|
| ReactZyme | `time` | 166,172 | 12,287 | 7,726 | 178,327 | Reactions are molecule sets, not directed reaction SMILES. |
| ReactZyme | `enzyme_smi` | 169,720 | 8,739 | 7,726 | 178,327 | Reactions are molecule sets, not directed reaction SMILES. |
| ReactZyme | `reaction_smi` | 163,770 | 14,689 | 7,726 | 178,327 | Reactions are molecule sets, not directed reaction SMILES. |
| CLIPZyme/EnzymeMap | `eval/enzymemap` | n/a | 48,074 | 17,266 | n/a | Used as validation in source-collapse retrieval. |
| SABIO-RK | `novelty90` | n/a | 2,339 | 1,570 | n/a | External SABIO novelty benchmark. |
| SABIO-RK | `novelty50` | n/a | 1,358 | 885 | n/a | Stricter SABIO novelty benchmark. |

The 7,726 ReactZyme reaction rows are non-directional molecule-set rows. They
do not contain `>` or `>>`, so reaction-center extraction and directional DRFP
fingerprints are not defined directly. For training-only capability
pretraining, we now materialize a Rhea-backed reconstruction using
`cleaned_uniprot_rhea.tsv` and `rhea_molecules.tsv`; validation/test ReactZyme
pairs are still not used for enzyme labels.

## Global Unique Retrieval Splits

These standardized splits live under:

```text
horizyn/data/standardized/global_unique_retrieval/
```

They are useful for stricter train/test leakage-control experiments.

| Policy | Train pairs | Train reactions | Train proteins | Test pairs | Test reactions | Test proteins |
|---|---:|---:|---:|---:|---:|---:|
| `exact` | 508,245 | 21,470 | 122,388 | 3,991 | 2,611 | 1,100 |
| `nr90` | 329,010 | 19,871 | 79,196 | 3,775 | 2,561 | 961 |
| `nr50` | 111,830 | 15,265 | 26,687 | 2,817 | 1,957 | 598 |

Strict directional-reaction subsets are also present:

| Policy | Train pairs valid-rxn | Train reactions valid-rxn | Test pairs valid-rxn | Test reactions valid-rxn |
|---|---:|---:|---:|---:|
| `exact` | 152,091 | 7,142 | 3,980 | 2,600 |
| `nr90` | 98,748 | 6,659 | 3,767 | 2,553 |
| `nr50` | 34,610 | 5,167 | 2,809 | 1,949 |

No capability annotations are currently materialized for the global-unique
splits. Rebuild enzyme labels split-by-split if these become the training
protocol.

## Precomputed Representation Files

### Enzyme / Protein Features

| Location | Available protein feature files |
|---|---|
| `retrieval_training_source_collapse/train_exact/` | `fit_proteins_with_clipzyme_eval_prott5_residue.h5` |
| `retrieval_training_source_collapse/train_nr90/` | `fit_proteins_with_clipzyme_eval_prott5_residue.h5`, `fit_proteins_with_clipzyme_eval_esm2_650m_residue.h5`, `fit_proteins_with_clipzyme_eval_esmc_6b_residue.h5` |
| `retrieval_training_source_collapse/test/horizyn_reactzyme_shared_candidates/` | `proteins_prott5_residue.h5`, `proteins_esm2_650m_residue.h5`, `proteins_esmc_6b_residue.h5` |

### Reaction Features

| Location | Available reaction representation files |
|---|---|
| `retrieval_training_source_collapse/train_nr90/` | ReactionT5v2, UniMol2, ChIRo, ChiENN, DRFP, RXNFP, SMi-TED, ChemBERTa2 HDF5 files. |
| `retrieval_training_source_collapse/test/reaction_features/` | ReactionT5v2, UniMol2, ChIRo, ChiENN, DRFP HDF5 files for the test suite. |
| `retrieval_training_source_collapse/test/` | `rxns_chemberta2_77m_mtr_test_suite.h5`. |

## Capability Annotation Outputs

Preferred clean annotation directory:

```text
horizyn/data/processed/capability_features/train_exact_rhea_reconstructed/
```

This directory is internally aligned:

```text
reaction_features.parquet:          28,529 rows
reaction_demand_vectors.npz:        shape (28,529, 2,434)
enzyme_capability_labels.parquet:   189,592 rows
pair_capability_training.parquet:   294,607 rows
```

Legacy pseudo-reaction annotation directory:

```text
horizyn/data/processed/capability_features/train_exact/
```

| File | Description |
|---|---|
| `reaction_features.parquet` | One row per train reaction with canonical SMILES, DRFP active bits, reaction-center labels, cofactor labels, cofactor tiers, cofactor-filtered substrate/product labels, substrate-product transition labels, reaction-type labels, and quality flags. |
| `reaction_drfp.npz` | Dense DRFP matrix. |
| `reaction_id_to_index.json` | Reaction ID to row index mapping. |
| `label_vocabs.json` | Reaction annotation vocabularies. |
| `reaction_demand_vectors.npz` | Dense demand vectors used for enzyme-capability pretraining. |
| `reaction_demand_metadata.parquet` | Row metadata aligned to `reaction_demand_vectors.npz`. |
| `reaction_demand_vocab.json` | Demand vector component vocabulary and slices. |
| `annotation_quality_report.json` | Reaction-level annotation coverage report. |
| `annotation_dictionaries.json` | Versioned dictionary snapshot used by the refresh; current version is `expanded_20260701`. |
| `reaction_feature_examples.csv` | Random examples for manual inspection. |
| `enzyme_capability_labels.parquet` | One row per train enzyme with train-only unioned capability labels. |
| `pair_capability_training.parquet` | One row per train enzyme-reaction pair for capability pretraining. |
| `enzyme_label_vocabs.json` | Enzyme label vocabularies. |
| `enzyme_capability_quality_report.json` | Enzyme-label coverage report. |
| `rhea_reconstructed_capability_report.json` | Present in `train_exact_rhea_reconstructed`; reports ReactZyme/Rhea reconstruction counts. |
| `reaction_center_fill_report.json` | Present in `train_exact_rhea_reconstructed`; reports RXNMapper completion/failure status for remaining missing centers. |
| `ec_annotation_report.json` | Coverage report for EC labels derived from existing train-pair sources. |
| `reaction_ec_from_train_pairs.csv` | Reaction-level EC unions from train-positive enzyme annotations. |
| `enzyme_ec_from_train_pairs.csv` | Enzyme-level EC unions from train-positive annotations. |
| `pair_ec_annotations.csv` | Per-train-pair EC labels and source provenance. |
| `revised_focus_annotation_report.json` | Coverage report for the biology-focused cofactor/reaction-center/substrate-product revision. |
| `enzyme_cofactor_annotation_report.json` | Coverage report for enzyme-side UniProt molecule cofactor enhancement. |
| `enzyme_cofactor_labels_enhanced.csv` | Human-readable enzyme cofactor audit table with reaction-derived, enzyme-derived, and combined labels. |

ReactZyme/Rhea reconstruction audit files:

```text
horizyn/data/paper/reactzyme/reconstructed_rhea_train/
  reactzyme_train_directional_rhea_pairs.csv
  reactzyme_train_directional_rhea_reactions.csv
  reactzyme_train_directional_rhea_members.csv
  reconstruction_report.json
```

Source-collapse reconstructed split files:

```text
horizyn/data/standardized/retrieval_training_source_collapse/train_exact/
  train_pairs_valid_rxn_rhea_reconstructed.csv
  train_rxns_valid_rxn_rhea_reconstructed.csv
  metadata_valid_rxn_rhea_reconstructed.json
```

### Label Construction Procedure

The annotation pipeline is deliberately conservative. It creates reaction-level
labels first, then creates enzyme-level labels by aggregating only the
train-positive reactions associated with each enzyme.

Reaction-level construction:

1. Canonicalize only directional reaction SMILES. Reactants, reagents, and
   products are parsed with RDKit, atom maps are removed for canonical
   fingerprints, stereochemistry is preserved, and each side is sorted into a
   deterministic `reactants>>products` string.
2. Exclude non-directional molecule-set rows from chemistry-supervised
   pretraining. ReactZyme train molecule sets are used only when they can be
   matched back to an unambiguous Rhea-backed substrate/product direction.
3. Compute DRFP from the canonical directional reaction. This is the dense
   reaction fingerprint part of the demand vector.
4. Assign EC labels from existing train-pair provenance, not from reaction
   chemistry. Horizyn contributes UniProt EC labels, CLIPZyme contributes direct
   `ec` labels, SABIO contributes `ec_numbers_json`, and reconstructed
   ReactZyme rows contribute sequence-to-Rhea metadata where available.
5. Assign cofactor labels from trusted chemistry evidence: ChEBI participants,
   exact/canonical ChEBI SMILES matches, curated cofactor aliases, and
   structure/name rules. The expanded dictionary covers 164 ChEBI
   cofactor-role terms, 94 alias labels, 34 core biological cofactor labels,
   and 18 metal/ion labels. The pipeline does not infer cofactors from EC alone.
6. Split mixed cofactor detections into biological tiers:
   `core_cofactor_labels`, `metal_ion_labels`, and
   `auxiliary_participant_labels`.
7. Assign substrate and product class labels from the non-currency molecules on
   each side. Currency molecules and detected core/metal/common auxiliary
   cofactors are removed first, then 39 SMARTS rules plus derived chemotype
   rules label functional groups and broad substrate/product classes. The
   unfiltered labels are retained in
   `substrate_class_labels_unfiltered` and `product_class_labels_unfiltered`
   for audit.
8. Assign substrate-product transition labels using 24 explicit directional
   transition rules plus gain/loss labels, including `alcohol_to_ketone`,
   `ketone_to_alcohol`, `alcohol_to_phosphate_ester`,
   `phosphate_ester_to_alcohol`, `nitrile_to_amide`, `ester_to_carboxylate`,
   `loss_phosphate_containing`, and `gain_carboxylate`.
9. Assign reaction-center labels from atom mapping. Existing mapped reactions
   are used when available; otherwise RXNMapper is attempted. Labels represent
   bond formation, bond cleavage, bond-order changes, charge changes, and
   chirality changes. The expanded coarse dictionary supports 30 families,
   including C-O, C-N, C-C, C-S, P-O, P-S, C-P, C-halogen, S-O, S-S,
   metal-ligand, redox-like, phosphate-transfer-like, halogenation-like, and
   disulfide-like changes. Reactions that are too long or unmappable are marked
   with an explicit failure status.
10. Assign reaction-type labels by deterministic rules over reaction-center
   labels, cofactors, substrate/product classes, and weak EC level-1 hints.

Enzyme-level construction:

1. Read the selected train pair table only.
2. Join every train reaction ID to `reaction_features.parquet`.
3. Group by `enzyme_id`.
4. Store the union of train reaction labels for each enzyme:
   EC labels, cofactor labels, cofactor tiers, reaction-center labels,
   substrate labels, product labels, substrate-product transition labels, and
   reaction-type labels.
5. Write one row per train enzyme in `enzyme_capability_labels.parquet`.
6. Write one row per train enzyme-reaction positive in
   `pair_capability_training.parquet`, including overlap diagnostics between
   the reaction labels and the enzyme's train-derived union labels.

Optional enzyme-side cofactor enhancement:

1. Read only the train pair table used for capability labels.
2. Recover UniProt accessions from collapsed train-pair `source_entries`.
3. Recover additional UniProt accessions by hashing sequences from
   `data/paper/reactzyme/raw/cleaned_uniprot_rhea.tsv` into active
   `uprot_<sha1>` enzyme IDs.
4. Join those accessions to `data/paper/reactzyme/raw/uniprot_molecules.tsv`.
5. Extract cofactors from the UniProt molecule strings with the same
   ChEBI/dictionary/structural rules used for reaction participants.
6. Add enzyme-side columns such as
   `enzyme_derived_core_cofactor_labels_train` and combined columns such as
   `combined_core_cofactor_labels_train`.
7. Add pair-level overlap diagnostics such as
   `enzyme_derived_core_cofactor_overlap` and
   `combined_core_cofactor_overlap`.

Current coverage for `train_exact_rhea_reconstructed`:

| Metric | Count |
|---|---:|
| train enzymes | 189,592 |
| enzymes with source-entry UniProt accession | 170,585 |
| enzymes with sequence-hash UniProt accession | 178,425 |
| enzymes with UniProt molecule record | 178,416 |
| reaction-derived core cofactor labels | 108,922 |
| enzyme-side molecule core cofactor labels | 105,405 |
| combined core cofactor labels | 108,922 |

This enzyme-side source mostly confirms reaction-derived cofactors in the
current reconstructed train set. It is not equivalent to UniProt curated
cofactor comments or binding-site features, so it does not rescue additional
core-cofactor coverage for enzymes whose train reactions lack cofactor
participants.

This means an enzyme label is a train-derived capability summary: "this enzyme
has been observed in training with reactions carrying these chemical demands."
It is not a curated complete mechanistic annotation of the protein, and it is
not allowed to use validation or test positives.

### Missing Label Semantics

Missing labels are expected and should be interpreted as unknown or not
confidently asserted. They are not reliable negative labels.

Common reasons labels are missing:

| Missing field | Reason |
|---|---|
| Reaction center | RXNMapper failed, the reaction is too long or chemically complex for stable atom mapping, or the mapped atoms do not yield a supported center label. |
| Cofactor labels | The reaction may genuinely not use a cofactor, the cofactor may be absent from source participants, or the evidence may not match the curated ChEBI/alias/structure dictionary. The pipeline does not fill this from EC because that would hallucinate cofactors. |
| EC labels | The source train provenance may not contain EC metadata for that enzyme or pair. ReactZyme official split rows do not carry EC fields, and Horizyn UniProt labels are enzyme-level rather than guaranteed pair-specific labels. |
| Substrate/product classes | Directional chemistry may be absent, the relevant molecules may be filtered as currency/cofactor molecules, or the remaining molecule may not match the current SMARTS vocabulary. |
| Reaction type | The upstream evidence may be absent or too weak. EC-only reaction type labels are marked as weak and should remain auxiliary. |
| Enzyme capability labels | The enzyme's train reactions may lack a specific label family, or the enzyme may only be linked to reactions whose annotations were intentionally left unknown. |

Training code should therefore mask missing label families for auxiliary BCE
losses. Treating missing labels as zeros would incorrectly teach the model that
an unannotated enzyme cannot use that cofactor, reaction center, or substrate
class.

### EC Join Step

The EC labels are not inferred from reaction chemistry. They are joined from
existing source annotations using the source provenance kept in
`train_pairs_valid_rxn_pseudo.csv`.

Reproducible command:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
python scripts/add_train_pair_ec_to_reaction_features.py \
  --train-pairs data/standardized/retrieval_training_source_collapse/train_exact/train_pairs_valid_rxn_pseudo.csv \
  --reaction-features data/processed/capability_features/train_exact/reaction_features.parquet \
  --out-dir data/processed/capability_features/train_exact
python scripts/build_reaction_demand_vectors.py \
  --reaction-features data/processed/capability_features/train_exact/reaction_features.parquet \
  --reaction-drfp data/processed/capability_features/train_exact/reaction_drfp.npz \
  --out-dir data/processed/capability_features/train_exact
python scripts/build_enzyme_capability_labels.py \
  --train-pairs data/standardized/retrieval_training_source_collapse/train_exact/train_pairs_valid_rxn_pseudo.csv \
  --reaction-features data/processed/capability_features/train_exact/reaction_features.parquet \
  --out-dir data/processed/capability_features/train_exact
```

For the all-directional Rhea-reconstructed capability artifacts, use:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
python scripts/reconstruct_reactzyme_rhea_directional_train.py
python scripts/build_rhea_reconstructed_capability_artifacts.py
python scripts/fill_reaction_center_annotations.py \
  --capability-dir data/processed/capability_features/train_exact_rhea_reconstructed \
  --batch-size 32
python scripts/build_reaction_demand_vectors.py \
  --reaction-features data/processed/capability_features/train_exact_rhea_reconstructed/reaction_features.parquet \
  --reaction-drfp data/processed/capability_features/train_exact_rhea_reconstructed/reaction_drfp.npz \
  --out-dir data/processed/capability_features/train_exact_rhea_reconstructed
python scripts/build_enzyme_capability_labels.py \
  --train-pairs data/processed/capability_features/train_exact_rhea_reconstructed/train_pairs_directional_rhea_reconstructed.csv \
  --reaction-features data/processed/capability_features/train_exact_rhea_reconstructed/reaction_features.parquet \
  --out-dir data/processed/capability_features/train_exact_rhea_reconstructed
python scripts/refresh_revised_capability_annotations.py \
  --capability-dir data/processed/capability_features/train_exact_rhea_reconstructed \
  --cofactor-dictionary data/processed/capability_features/chebi_cofactor_dictionary.tsv \
  --train-pairs data/standardized/retrieval_training_source_collapse/train_exact/train_pairs_valid_rxn_rhea_reconstructed.csv \
  --rebuild-demand \
  --rebuild-enzyme-labels
```

EC sources used by the join:

| Source | File / column | Coverage in train provenance |
|---|---|---:|
| Horizyn | `data/sota/uniprot_all_ec_labels.csv` | 248,039 EC-labeled source entries out of 257,733 Horizyn source entries. |
| CLIPZyme | `data/paper/clipzyme/train/enzymemap/pairs.csv`, column `ec` | 42,351 EC-labeled source entries out of 42,351 CLIPZyme source entries. |
| SABIO-RK | `data/paper/sabio_rk/eval/novelty90/pairs.csv`, column `ec_numbers_json` | 2,100 EC-labeled source entries out of 2,335 SABIO source entries. |
| ReactZyme | official train pair rows | No EC column in the official split rows. |

This produces `ec_annotation_report.json`,
`reaction_ec_from_train_pairs.csv`, `enzyme_ec_from_train_pairs.csv`, and
`pair_ec_annotations.csv`, then propagates EC into the demand vectors and
enzyme capability labels.

### Reaction Annotation Coverage: Rhea-Reconstructed Clean Artifacts

| Quantity | Count |
|---|---:|
| Total reactions | 28,529 |
| Valid directional reaction SMILES | 28,529 |
| ReactZyme non-directional molecule-set rows | 0 |
| DRFP vectors | 28,529 |
| RXNMapper atom mappings / reaction-center status `ok` | 28,152 |
| RXNMapper attempted but failed | 377 |
| Reaction-center labels | 27,029 |
| Cofactor labels | 15,295 |
| Core cofactor labels | 13,245 |
| Metal/ion labels | 725 |
| Auxiliary participant labels | 2,516 |
| Cofactor-filtered substrate/product class labels | 27,232 |
| Substrate-product transition labels | 24,078 |
| Reaction-type labels | 27,751 |
| EC-labeled reactions | 27,507 |
| Unique EC labels | 5,462 |
| Unique complete EC4 labels | 5,279 |

EC labels in `reaction_features.parquet` are train-pair-derived reaction EC
unions plus the Rhea-backed reconstruction where available. They come from
Horizyn UniProt labels, CLIPZyme direct `ec` labels, SABIO `ec_numbers_json`,
and ReactZyme sequence-to-Rhea metadata. ReactZyme validation/test pairs still
do not contribute enzyme labels.

The remaining 377 reaction-center failures are explicit RXNMapper failures,
not missing preprocessing. They are directional reactions, but they are too
long or chemically complex for the current mapper. Their median canonical
reaction-SMILES length is about 922 characters and the maximum is 9,332.

### Reaction Demand Vector

| Component | Dimension |
|---|---:|
| DRFP | 2,048 |
| Reaction-center coarse labels | 12 |
| Cofactor labels, mixed legacy view | 55 |
| Core cofactor labels | 22 |
| Metal/ion labels | 12 |
| Auxiliary participant labels | 21 |
| Substrate class labels | 54 |
| Product class labels | 54 |
| Substrate-product transition labels | 131 |
| Reaction type labels | 25 |
| Total demand vector | 2,434 |

Stored file:

```text
reaction_demand_vectors.npz
  ids:     shape (28529,)
  vectors: shape (28529, 2434), dtype float32
```

The DRFP-only matrix is:

```text
reaction_drfp.npz
  ids:     shape (28529,)
  vectors: shape (28529, 2048), dtype uint8
```

### Annotation Vocabularies

| Vocabulary | Size | Labels |
|---|---:|---|
| Cofactors, mixed legacy view | 55 | Expanded ChEBI/cofactor dictionary; see `reaction_demand_vocab.json`. This includes observed core cofactors, metals/ions, and auxiliary participants. |
| Core cofactors | 22 | Observed biology-focused cofactor targets such as `ATP`, `CoA`, `FAD`, `FMN`, `NAD`, `NADP`, `SAM`, `TPP`, `FeS_cluster`, glutathione, quinone, cobalamin, and related carrier cofactors. The dictionary supports 34 core labels, but 22 occur in this train split. |
| Metal/ion labels | 12 | `metal`, `Mg2+`, `Zn2+`, `Fe2+`, `Fe3+`, `Mn2+`, `Cu2+`, `Ni2+`, `Ca2+`, `Co2+`, `K+`, `Na+`. |
| Auxiliary participants | 21 | ChEBI cofactor-role or reaction-context labels kept separate from core cofactors, e.g. `hydrogenphosphate`, `hydrogen_peroxide`, `pyruvic_acid`, `ammonium`, `chloride`. |
| Reaction-center coarse | 12 | Observed labels: `bond_cleavage`, `bond_formation`, `bond_order_change`, `c_c_change`, `c_n_change`, `c_o_change`, `c_s_change`, `charge_change`, `chirality_change`, `p_o_change`, `phosphate_transfer_like`, `redox_like`. The dictionary supports 30 coarse families. |
| Reaction-center raw | 122 | Full atom/bond-change label set in `label_vocabs.json`. |
| Substrate classes | 54 | Expanded SMARTS-derived substrate class vocabulary after currency/cofactor filtering. |
| Product classes | 54 | Expanded SMARTS-derived product class vocabulary after currency/cofactor filtering. |
| Substrate-product transitions | 131 | Directional broad-chemotype changes such as `alcohol_to_ketone`, `ketone_to_alcohol`, `alcohol_to_phosphate_ester`, `phosphate_ester_to_alcohol`, `amide_to_carboxylate`, `gain_carboxylate`, and `loss_sugar_like`. |
| Reaction types | 25 | Rule-derived type vocabulary including redox, transfer, hydrolysis, phosphorylation/dephosphorylation, amide/ester/nitrile hydrolysis, halogenation/dehalogenation, and metal-dependent transformations. |
| EC labels | 5,462 | Train-pair-derived EC labels plus Rhea-backed ReactZyme train metadata. |

### Most Frequent Reaction Labels

| Label family | Top labels |
|---|---|
| Cofactors | `CoA` 3,401; `ATP` 3,184; `NAD` 2,889; `NADP` 2,869; `SAM` 842; `hydrogenphosphate` 717; `hydrogen_peroxide` 671; `FMN` 671; `metal` 473. |
| Core cofactors | `CoA` 3,401; `ATP` 3,184; `NAD` 2,889; `NADP` 2,869; `SAM` 842; `FMN` 671; `quinone` 217; `FeS_cluster` 191; `glutathione` 152; `FAD` 108. |
| Auxiliary participants | `hydrogenphosphate` 717; `hydrogen_peroxide` 671; `pyruvic_acid` 441; `ammonium` 314; `pyruvate` 155; `chloride` 123; `hydrogencarbonate` 67; `nicotinamide` 66. |
| Reaction-center coarse | `bond_cleavage` 20,898; `bond_formation` 20,667; `c_o_change` 19,477; `chirality_change` 12,221; `c_c_change` 12,110; `charge_change` 11,874; `c_n_change` 11,188; `bond_order_change` 11,162. |
| Substrate-product transitions | `amide_to_carboxylate` 3,291; `gain_alcohol` 3,191; `loss_alcohol` 3,010; `gain_carboxylate` 2,982; `gain_organic_acid` 2,982; `alcohol_to_ketone` 2,945; `alcohol_to_phosphate_ester` 2,773; `carboxylate_to_amide` 2,738; `ketone_to_alcohol` 2,496; `phosphate_ester_to_alcohol` 2,332. |
| Reaction types | `bond_cleavage` 21,523; `bond_formation` 20,667; `stereochemical_inversion` 12,221; `acid_base_or_charge_transfer` 11,874; `oxidoreduction` 9,113; `group_transfer` 8,592; `glycosyl_transfer` 7,290; `hydrolysis` 6,250; `hydride_transfer` 5,716; `phosphorylation` 4,583. |
| EC labels | `2.3.1.-` 639; `1.1.1.1` 348; `1.1.1.-` 338; `3.1.1.4` 238; `2.4.1.17` 219; `2.3.1.75` 217; `1.1.1.184` 201; `2.1.1.-` 199. |

### Quality Flags

| Quality flag | Count | Meaning |
|---|---:|---|
| `cofactor_from_structure_match` | 12,861 | Cofactor labels assigned from structural/alias detection. |
| `cofactor_from_chebi_smiles_match` | 11,399 | Cofactor labels assigned from exact/canonical ChEBI SMILES dictionary matches. |
| `substrate_class_from_smarts` | 27,232 | Cofactor-filtered substrate/product classes assigned using SMARTS fallback rules. |
| `substrate_class_removed_cofactor_molecule` | 13,697 | At least one molecule was removed from substrate/product class extraction because it matched a detected cofactor/core auxiliary filter. |
| `no_reaction_center` | 377 | Directional reactions where RXNMapper failed or center extraction did not yield labels. |

## Enzyme Capability Labels

The enzyme labels are built only from the selected source-collapse exact train
pairs. For the preferred Rhea-reconstructed artifacts, ReactZyme contributes
only train-side Rhea-backed directional pairs; validation/test ReactZyme pairs
are not used.

| Quantity | Count |
|---|---:|
| Train pairs in `pair_capability_training.parquet` | 294,607 |
| Source split values | `train` only |
| Enzyme label rows | 189,592 |
| Enzymes with reaction-union EC labels | 185,836 |
| Enzymes with at least one capability label | 189,103 |
| Enzymes with core cofactor labels | 108,922 |
| Enzymes with reaction-center labels | 181,189 |
| Enzymes with substrate labels | 161,782 |
| Enzymes with product labels | 168,812 |
| Enzymes with substrate-product transition labels | 156,656 |
| Enzymes with reaction-type labels | 186,328 |

Quality flags:

| Quality flag | Count | Meaning |
|---|---:|---|
| `train_only_labels` | 189,592 | Label row was constructed only from training pairs. |
| `enzyme_multifunctional` | 185,396 | Enzyme is linked to multiple train reactions or capability label groups. |
| `enzyme_has_no_cofactor_labels` | 55,643 | No cofactor labels were available across the enzyme's train reactions. |
| `enzyme_has_no_reaction_center_labels` | 8,403 | No reaction-center labels were available across the enzyme's train reactions. |
| `enzyme_has_no_ec_labels` | 3,756 | No train-derived EC labels were available across the enzyme's train reactions. |

## Annotation Availability By Split

| Split / data family | Reaction-level annotations | Enzyme-level capability labels | Current status |
|---|---|---|---|
| Source-collapse `train_exact_rhea_reconstructed` | Yes, materialized for 28,529 all-directional reactions. RXNMapper succeeded for 28,152 rows and failed explicitly for 377 long/complex rows. | Yes, materialized for 189,592 train enzymes and 294,607 train pairs using train pairs only. | Preferred current capability-pretraining target. |
| Source-collapse `train_exact` valid pseudo | Yes, materialized for 35,219 reactions. Directional chemistry is available for 27,493 valid reaction SMILES; 7,726 ReactZyme molecule-set rows remain non-directional. | Yes, materialized for 189,592 train enzymes using train pairs only. | Legacy pseudo-reaction artifact. |
| Source-collapse `train_nr90` valid pseudo | Reusable by reaction ID from `train_exact` because the reaction universe matches after pseudo filtering. | Not materialized. Must rebuild using `train_nr90` pairs and collapsed protein IDs. | Recommended if training the nr90 capability model. |
| Source-collapse `train_nr90` directional valid-rxn | Reusable subset of the 27,493 directional reactions. | Not materialized. Must rebuild if using this split. | Best split if the model should avoid ReactZyme molecule-set pseudo reactions. |
| Source-collapse `train_nr50` | Reusable by reaction ID for matching reactions. | Not materialized. Must rebuild using `train_nr50` collapsed protein IDs. | Available for stricter anti-homology experiments. |
| Validation `clipzyme_eval` | Not materialized in capability directory. Can be computed reaction-only if needed. | Not allowed from validation pairs. | Use for validation metrics only. |
| Protein-disjoint exact tests | Not materialized. Can be computed reaction-only for public reaction tables. | Not allowed from test pairs. | Use only for held-out evaluation. |
| Protein-disjoint nr90 tests | Not materialized. Can be computed reaction-only for public reaction tables. | Not allowed from test pairs. | Use only for held-out evaluation. |
| Paper ReactZyme eval splits | Not directionally annotatable as-is because reactions are molecule sets. Train rows are reconstructable through Rhea metadata; test rows must remain held out. | Not allowed from test pairs. ReactZyme train rows can contribute only via train-side Rhea reconstruction. | Explains why test molecule-set rows remain non-directional. |
| Paper CLIPZyme/EnzymeMap | Not materialized in capability directory. | Not allowed for source-collapse validation labels. | Used as validation in current retrieval. |
| Paper SABIO-RK | Not materialized in capability directory. | Not allowed for test labels. | External held-out benchmark. |
| Global unique retrieval splits | Not materialized. Some reaction IDs may overlap with source-collapse annotations. | Not materialized and must be rebuilt per policy. | Use for stricter protocol experiments. |

## Leakage Rules

- Enzyme-level labels must be constructed only from the selected training pair
  table.
- Current Rhea-reconstructed `pair_capability_training.parquet` has
  `source_split == "train"` for all 294,607 rows.
- Validation and test enzyme-reaction pairs must not contribute to
  `enzyme_capability_labels.parquet`.
- Reaction-level public annotations can be computed for validation/test
  reactions, but they are separate from train-derived enzyme capability labels.
- EC labels are available as auxiliary train-derived supervision, but should not
  be used as a substitute for reaction specificity.

## Practical Guidance

- For the next capability-pretraining run on directional chemistry, use
  `horizyn/data/processed/capability_features/train_exact_rhea_reconstructed/`.
- If the retrieval experiment uses `train_nr90` or `train_nr50`, rebuild
  `enzyme_capability_labels.parquet` and `pair_capability_training.parquet` from
  that split's pair table so the enzyme IDs and train-only labels match the
  actual training protocol.
- Use the legacy `train_exact/` capability artifacts only when an experiment
  intentionally keeps ReactZyme molecule-set pseudo reactions.
- Do not use ReactZyme validation/test molecule-set rows to create enzyme
  capability labels. They remain held-out evaluation data.
