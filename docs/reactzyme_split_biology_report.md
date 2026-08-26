# Biological audit of the ReactZyme splits

Generated from the local ReactZyme release and the current paper-protocol materialization on
2026-07-17.

## Executive interpretation

The three protocols partition the same 178,459 unique normalized enzyme-molecular-set pairs. They
are not three different biological datasets. Their test sets emphasize different parts of the same
bipartite graph:

- `time` is mostly a later-accession enzyme holdout against already supported molecular sets. It is
  not a pure cutoff split in the released data because 3,017 post-boundary training anchors retain
  molecular sets with no unequivocal pre-2010 support.
- `enzyme_smi` is almost exactly sequence-ID disjoint and chemically/functionally matched to its
  training set. It is not a remote-homology split under conventional sequence comparison: 94.8% of
  test proteins have a current-training neighbor proven to be less than 60% different by normalized
  global Levenshtein distance.
- `reaction_smi` is the genuinely hardest biological shift. It holds out all 386 test molecular-set
  IDs, but also shifts fine-grained EC functions, enzyme families, cofactors, and graph degree. Only
  64.8% of test proteins have a significant MMseqs2 train hit, and ten reaction hubs contain 56.7%
  of all test pairs.

The strongest practical conclusion is that low `reaction_smi` performance should not be diagnosed as
a reaction-tower problem alone. The protocol simultaneously tests chemistry novelty, fine-grained
functional novelty, enzyme-family novelty, and highly imbalanced multi-positive ranking.

## Audit coverage

Every one of the 535,377 materialized rows across protocol/subset combinations was mapped back to at
least one exact source accession using sequence and normalized molecular-set hashes. The full
`uniprot_rhea.tsv` table was used for provenance; restricting the audit to the smaller cleaned table
would have silently lost valid source aliases. EC, Rhea, date, sequence, participant chemistry, and
cached UniProtKB cofactor comments are included.

The release metadata do not include organism taxonomy, lineage, or protein names. Consequently this
report can characterize functional, chemical, temporal, and sequence-homology shifts, but it cannot
make a defensible claim about taxonomic or clade composition without an additional UniProt metadata
download.

## What one ReactZyme "reaction" is

The [ReactZyme paper](https://papers.nips.cc/paper_files/paper/2024/file/2e68b2367d2e0bc8dd6f0ff86e07c2eb-Paper-Datasets_and_Benchmarks_Track.pdf)
describes 178,463 positive pairs, 178,327 enzymes, and 7,726 reactions derived from SwissProt and
Rhea. The local release reveals an important representation detail:

- Every released reaction input is a dot-separated molecular set. None of the 7,726 strings contains
  a directional `>>` separator.
- The molecular set associated with one UniProt entry can summarize multiple Rhea IDs and EC
  annotations. It must not be interpreted as one atom-mapped elementary transformation.
- Upstream [ReactZyme code](https://github.com/WillHua127/ReactZyme) replaces every generic `*` atom
  with carbon before encoding: `mol.replace('*', 'C')`.
- All 7,726 normalized strings parse with RDKit, but that does not restore the chemical specificity
  lost by wildcard substitution or participant aggregation.

The local release dictionaries contain four duplicate normalized sequence-molecular-set rows. After
deduplication there are 178,459 unique pairs. This explains the small difference from the paper count.

Across all molecular sets:

- Median proteins per molecular set: 2; mean: 23.1; 95th percentile: 94; maximum: 5,489.
- Median associated complete/partial EC numbers: 1; 95th percentile: 3; maximum: 30.
- 5.89% of molecular sets are associated with more than one top-level EC class.
- Median Rhea IDs per pair: 1. The long tail reaches 50.

The largest molecular-set hub is linked to 5,489 proteins, 30 EC labels, and five top-level EC
classes. Such hubs are participant-pattern equivalence classes, not necessarily single biochemical
reactions.

## Materialized split sizes

The current protocol follows the paper's random 90/10 division of each official training partition.
The released test rows are unchanged.

| Protocol | Train pairs / proteins / molecular sets | Validation pairs / proteins / molecular sets | Test pairs / proteins / molecular sets |
|---|---:|---:|---:|
| `time` | 149,554 / 149,478 / 7,281 | 16,618 / 16,617 / 2,156 | 12,287 / 12,277 / 2,634 |
| `enzyme_smi` | 152,748 / 152,643 / 7,386 | 16,972 / 16,971 / 2,464 | 8,739 / 8,734 / 1,573 |
| `reaction_smi` | 147,393 / 147,299 / 6,977 | 16,377 / 16,374 / 2,448 | 14,689 / 14,688 / 386 |

The upstream evaluation constructs candidate matrices from unique entities in the test positives.
Therefore the paper-comparable R->E candidate pools are 12,277, 8,734, and 14,688 enzymes,
respectively, rather than the global 178,327-enzyme catalog.

## Shared universe and overlap

All protocols cover the same 178,459 unique pairs, but select mostly different test examples:

| Test intersection | Pairs |
|---|---:|
| `time` and `enzyme_smi` | 612 |
| `time` and `reaction_smi` | 743 |
| `enzyme_smi` and `reaction_smi` | 701 |
| All three | 19 |

Metrics across protocols consequently measure different populations even when architecture and loss
are identical.

## Validation biology

Validation is pair-random, not group-disjoint. Because almost every exact protein sequence has only
one molecular-set association, validation contains almost entirely new exact protein IDs, but mostly
seen molecular sets:

| Protocol | Validation proteins also in train | Validation molecular sets also in train |
|---|---:|---:|
| `time` | 11 / 16,617 | 1,711 / 2,156 |
| `enzyme_smi` | 18 / 16,971 | 2,124 / 2,464 |
| `reaction_smi` | 22 / 16,374 | 2,085 / 2,448 |

This validation set is useful for model selection on new exact sequences paired with largely familiar
chemistry, but it does not reproduce the coupled novelty of the `reaction_smi` test. It can therefore
be optimistic for selecting models intended to generalize to unseen reactions.

Randomly removing validation rows also removes the only optimization example for some molecular
sets. For example, only 2,535 of the 2,634 `time` test molecular sets remain in the actual 90% train
subset, despite all 2,634 being represented in the official pre-validation training partition.

## Functional composition

Pair-weighted top-level EC presence is shown below. Percentages can sum above 100 because one source
entry can carry multiple EC annotations.

| EC class | `time` test | `enzyme_smi` test | `reaction_smi` test |
|---|---:|---:|---:|
| 1 Oxidoreductases | 18.91% | 14.69% | 14.44% |
| 2 Transferases | 37.79% | 37.48% | 27.80% |
| 3 Hydrolases | 19.20% | 15.64% | 8.09% |
| 4 Lyases | 13.45% | 11.44% | 2.29% |
| 5 Isomerases | 5.60% | 6.73% | 14.47% |
| 6 Ligases | 6.67% | 13.74% | 17.90% |
| 7 Translocases | 2.66% | 3.87% | 15.52% |

`enzyme_smi` closely preserves the train functional mixture. `reaction_smi` does not: relative to its
train set, transferases fall from 38.62% to 27.80%, hydrolases from 16.09% to 8.09%, and lyases from
12.42% to 2.29%, while isomerases rise from 5.84% to 14.47% and translocases from 2.92% to 15.52%.

### EC novelty depth

| Protocol | Test pairs with complete EC4 | Complete EC4 labels all seen in train | EC3 labels all seen in train |
|---|---:|---:|---:|
| `time` | 88.52% | 98.36% of annotated pairs | 99.99% of annotated pairs |
| `enzyme_smi` | 95.03% | 99.94% of annotated pairs | 99.99% of annotated pairs |
| `reaction_smi` | 81.63% | 41.46% of annotated pairs | 99.98% of annotated pairs |

This means `reaction_smi` mostly retains familiar broad EC mechanisms through EC3 while holding out
substrate- or product-specific EC4 activities. It is a fine-grained functional novelty test, not an
entirely new top-level chemistry test.

## Time split

The paper states a 2010-12-31 cutoff. The released data are more nuanced when mapped back to every
compatible source accession:

- Of 166,172 official train/validation rows, 162,922 are unequivocally before 2010-01-01.
- 3,017 are unequivocally on or after 2010-01-01, and 233 have exact sequence/molecular-set aliases
  spanning that boundary.
- Each of the 3,017 unequivocally late train rows represents a different molecular set with no
  unequivocal pre-boundary train row.
- Of 12,287 test rows, 11,888 are unequivocally on or after 2010-01-01 and 399 have aliases spanning
  the boundary. None is unequivocally before it.

The most consistent interpretation is that the release uses an effective start-of-2010 boundary and
retains one later training anchor for molecular sets that otherwise have no historical support. This is
an inference from the released rows, not a documented algorithm in the paper.

Biologically, `time` is therefore dominated by later proteins for known participant patterns rather
than by new chemistry. Test proteins are also longer: median 393 residues versus 350 in train, with a
95th percentile of 1,139 versus 805.

Best-train sequence audit:

- 96.5% of test proteins have a significant MMseqs2 hit.
- Median best local identity among hits is 53.9%, with median query coverage 95.9%.
- 12.3% of all test proteins have a hit at least 90% identical.
- 64.6% of all test proteins are proven to have a train neighbor less than 60% different by normalized
  global Levenshtein distance.

## Enzyme-similarity split

The intended biological question is generalization to sequence-novel enzymes while reactions remain
known. Exact-ID behavior mostly matches that goal: only 3 of 8,734 test proteins overlap the current
90% train subset, while 1,563 of 1,573 test molecular sets are already present there. The chemistry
and EC composition closely match train.

However, exact-ID novelty does not imply remote homology:

- 8,706 of 8,734 test proteins, or 99.7%, have a significant train MMseqs2 hit.
- Median best local identity is 87.5%, with median query coverage 100%.
- 45.4% of all test proteins have a train hit at least 90% identical.
- For the MMseqs-selected neighbor, median normalized global Levenshtein difference is 13.35%.
- 94.8% of all test proteins have at least one train neighbor proven to be less than 60% different.

This directly conflicts with a literal reading of the paper's stated "at least 60% sequence difference"
criterion. The result is conservative because the neighbor was selected for local alignment, not to
minimize global Levenshtein distance. In practice, this protocol measures mostly close-homolog transfer
to unseen exact sequences.

## Reaction-similarity split

All 386 test molecular-set IDs are absent from train. Nevertheless, exact string novelty is not the same
as complete chemical novelty:

- Median best train molecular-set Morgan Tanimoto is 0.853.
- The 5th percentile is 0.489 and the 95th percentile is 1.000.
- A Tanimoto of 1 for different IDs can arise because the audit fingerprint is an order- and
  multiplicity-insensitive union over molecular components. It does not imply identical reactions.

The protocol also produces the strongest enzyme-family shift:

- Only 9,514 of 14,688 test proteins, or 64.8%, have a significant train MMseqs2 hit.
- The remaining 5,174 proteins have no detected hit at the configured sensitivity.
- Median best local identity among hits is 36.1%; median query coverage is 88.2%.
- Only 0.54% of all test proteins have a hit at least 90% identical.
- Median normalized global difference to the MMseqs neighbor is 71.7%.

Reaction novelty and protein novelty are biologically correlated because a reaction group is often
carried by related enzyme families. This is a coupled cold-reaction/cold-family benchmark.

### Degree imbalance

| Protocol | Median test proteins per molecular set | Mean | Maximum | Pair share in top 10 molecular sets | Gini |
|---|---:|---:|---:|---:|---:|
| `time` | 2 | 4.66 | 657 | 18.2% | 0.650 |
| `enzyme_smi` | 1 | 5.56 | 277 | 10.8% | 0.660 |
| `reaction_smi` | 2 | 38.05 | 2,249 | 56.7% | 0.901 |

The largest `reaction_smi` molecular set alone has 2,249 positive proteins, or 15.3% of all test pairs.
The top 20 contain 77.9%. Pair-averaged losses and metrics are therefore dominated by a small number
of broad biological groups unless reactions are explicitly balanced.

### Generic chemistry and cofactors

At the unique-molecular-set level, wildcard provenance is similar across tests: 30.6% for `time`,
35.3% for `enzyme_smi`, and 35.5% for `reaction_smi`. Pair weighting changes the `reaction_smi`
value to 62.8%, showing that its largest hubs disproportionately come from generic source chemistry.

Reaction-participant cofactor context is present in 62.2% of `reaction_smi` test pairs. Two strong
shifts are:

- Quinone context: 16.37% in test versus 0.74% in train.
- NAD context: 17.73% in test versus 5.74% in train.

UniProtKB enzyme-side comments tell a complementary story. An explicit cofactor label is available for
48.1% of `reaction_smi` test pairs, compared with 40.7% in train. Among core enzyme-side cofactors:

- Fe-S cluster annotations rise from 4.39% in train to 8.33% in test.
- Heme annotations rise from 1.30% to 3.19%.
- PLP annotations fall from 3.31% to 0.69%.

Together with the enrichment of EC class 7 and quinone/NAD participant patterns, this indicates that
the reaction holdout is enriched for electron-transfer, respiratory, and membrane-translocation biology.

## Consequences for modeling

1. Report macro-by-reaction metrics alongside pair-weighted metrics. Otherwise a few `reaction_smi`
   hubs dominate both optimization and evaluation.
2. Treat EC3 versus EC4 novelty separately. `reaction_smi` is largely familiar at EC3 and novel at
   EC4, which favors hierarchical biological representations.
3. Do not interpret reaction-side geometry as a directional transformation unless it comes from the
   reconstructed Rhea substrate/product records. The benchmark input itself is non-directional.
4. Preserve wildcard provenance as a confidence/mask feature. Replacing `*` with `C` makes generic
   groups look chemically concrete.
5. For `reaction_smi`, evaluate by sequence-homology strata and reaction degree. This separates
   chemistry failure from cold-family placement and multi-positive ranking failure.
6. Use a reaction-group-disjoint validation set when selecting for `reaction_smi`; the current random
   validation set does not represent its biological shift.
7. Interpret `enzyme_smi` as exact-sequence holdout unless the split is rebuilt and verified with a
   normalized global-distance or sequence-cluster threshold.

## Audit artifacts

- `results/reactzyme_split_biology/pair_annotations.parquet`: every materialized pair in every
  protocol/subset, with source accessions, EC/Rhea labels, dates, cofactor evidence, degrees, and
  train-overlap flags.
- `results/reactzyme_split_biology/enzyme_annotations.parquet`: 178,327 exact-sequence entities.
- `results/reactzyme_split_biology/reaction_annotations.parquet`: 7,726 molecular-set entities with
  chemistry, wildcard provenance, and associated biological diversity.
- `results/reactzyme_split_biology/protein_nearest_train_mmseqs.csv`: best train alignments and global
  edit-distance audits for held-out proteins.
- `results/reactzyme_split_biology/reaction_nearest_train.csv`: molecular-set fingerprint neighbors.
- `results/reactzyme_split_biology/label_distributions.csv`: EC and cofactor distributions.
- `results/reactzyme_split_biology/summary.json`: machine-readable report statistics.

The analysis is reproducible with `scripts/audit_reactzyme_split_biology.py`. MMseqs2 results are an
independent sequence audit and do not claim to reproduce the split's undocumented implementation of
the paper's Levenshtein criterion.
