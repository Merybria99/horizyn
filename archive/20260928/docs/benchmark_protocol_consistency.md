# Benchmark protocol consistency

This document records the source-level protocol audit used by the unified
retrieval evaluator. The audited upstream revisions are:

- Horizyn [`6944198`](https://github.com/dayhofflabs/horizyn/tree/6944198303f2f0946d448a259ab788589cfd27b3)
- ReactZyme [`c4d1554`](https://github.com/WillHua127/ReactZyme/tree/c4d1554640a01e5b8ae8e2de716b452ce93b1bc6)

## Horizyn

The published evaluator uses the 1,012 test reactions in both forward and
reverse directions, producing 2,024 reaction queries. It screens each query
against the complete protein embedding store and reports Hit@1/10/100/1000,
R-precision, and average precision. Scores are negative cosine distance, which
is ranking-equivalent to cosine similarity. The source-level references are
the upstream
[`scripts/evaluate.py`](https://github.com/dayhofflabs/horizyn/blob/6944198303f2f0946d448a259ab788589cfd27b3/scripts/evaluate.py)
and
[`horizyn/metrics.py`](https://github.com/dayhofflabs/horizyn/blob/6944198303f2f0946d448a259ab788589cfd27b3/horizyn/metrics.py).

The unified suite therefore sets `metric_protocol: horizyn`, enables
`bidirectional_reactions`, and uses exactly 216,132 released protein candidates.
When evaluation uses the larger 394,459-protein union HDF5, `data/sota/prots.fasta`
is the authoritative candidate manifest. Here "bidirectional" means forward and
reverse chemical-reaction queries; Horizyn reports reaction-to-enzyme retrieval,
not ReactZyme's separate enzyme-to-reaction task.

## ReactZyme

The published retrieval script evaluates the `time`, sequence-similarity, and
molecule-similarity test splits in both directions. Its candidate universes are
the unique positive entities occurring in each test split, not the combined
train/test protein FASTA. The local names `enzyme_smi` and `reaction_smi` map to
the sequence- and molecule-similarity protocols respectively.

ReactZyme's `MRR` is nonstandard: for each query it averages reciprocal rank
over every positive candidate, then averages over queries. The evaluator now
exports both `reactzyme_mrr` and `first_positive_mrr`; under
`metric_protocol: reactzyme`, the compatibility column `mrr` aliases
`reactzyme_mrr`. ReactZyme `Top-k Acc-N` is exported as `top_<k>_n` and divides
the number of positive hits by the requested `k`. Binary `Top-k Acc` is
exported as `top_<k>`. These definitions and the test-positive candidate
construction come directly from upstream
[`retrieval.py`](https://github.com/WillHua127/ReactZyme/blob/c4d1554640a01e5b8ae8e2de716b452ce93b1bc6/retrieval.py).

Official ReactZyme suite entries use `candidates_from_test_positives: true`, so
the candidate set is derived from the test pairs and checked against the
embedding store. Broader experimental candidate manifests remain supported,
but result metadata labels them `explicit_candidate_manifest`.

## Level-1 compliance contract

Published-suite tasks opt in with an explicit `benchmark_protocol`. The loader
checks metric settings, directionality, and cutoffs. Before evaluation, the
runner checks semantic digests and counts for train pairs, test pairs, reaction
rows, and the resolved candidate universe. A task that merely uses
`metric_protocol: horizyn` or `metric_protocol: reactzyme` is a custom task and
is not labeled as a published candidate pool.

The registered protocols are `horizyn_release_v1`,
`reactzyme_time_release_v2`, `reactzyme_enzyme_smi_release_v2`, and
`reactzyme_reaction_smi_release_v2`.

Run the source-only audit with:

```bash
python scripts/audit_level1_benchmark_data.py
```

## Validation versus final test

The released test arrangements are exactly reproducible; the original
ReactZyme validation membership is not. ReactZyme calls unseeded
`torch.random_split` on a concatenated positive/negative classification dataset
and does not publish the selected indices. The local protocol generator now
uses the same PyTorch permutation and length rule on positive pairs with a
recorded seed, and records this honestly as a deterministic positive-only
analogue. It must not be described as the original validation membership.

Horizyn releases train and test data but no separate development split. For new
Level-1 experiments, model selection must be completed without the official
test labels, and the frozen final checkpoint is evaluated once on the registered
test protocol. The older upstream training script exposes its test file through
the validation data loader; reproducing that implementation detail is not a
reason to select a new method on the test set.

The local trainer supports this second phase with
`training.validation_enabled: false`. In that mode it loads training data only,
rejects validation metrics and early stopping, saves `last.ckpt` every epoch,
and does not define a validation-selected "best" checkpoint. The intended flow
is:

1. Choose architecture, hyperparameters, and epoch budget on a deterministic
   split made only from the released training set.
2. Refit from scratch on all released training pairs for that fixed epoch
   budget with validation disabled.
3. Freeze `last.ckpt`, run `scripts/audit_level1_benchmark_data.py`, and evaluate
   the registered test suite once.

The completed `horizyn_f3_f4_paper_v1` runs predate this guard: their logs
contain test-set validation measurements. They used the final epoch and no
early stopping, so the test metric did not choose the saved final checkpoint,
but those runs should not be described as fully test-blind development.

## Deterministic implementation choices

The local evaluator preserves input/HDF5 candidate order and uses stable
descending sorting. ReactZyme's upstream code constructs candidates through
Python sets, so its tie ordering is not reproducible. The deterministic local
tie policy can differ only when candidate scores tie exactly and is always
recorded by the code revision in artifact provenance.

The local split audit found zero exact train/test positive-pair overlap for the
Horizyn test and all three ReactZyme test protocols. Validation rejects any
future exact pair overlap, missing positive representation, or positive outside
the declared candidate universe.

| Local protocol | Train pairs | Test pairs | Test queries | Test proteins | Exact overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Horizyn | 257,733 | 33,996 | 1,012 raw / 2,024 directional | 32,100 | 0 |
| ReactZyme time | 166,172 | 12,287 | 2,634 | 12,277 | 0 |
| ReactZyme sequence similarity (`enzyme_smi`) | 169,720 | 8,739 | 1,573 | 8,734 | 0 |
| ReactZyme molecule similarity (`reaction_smi`) | 163,770 | 14,689 | 386 | 14,688 | 0 |

The legacy local ReactZyme candidate files each contain 178,327 proteins and
therefore do not reproduce upstream ReactZyme retrieval. Official suite entries
now derive the smaller split-specific universes shown above directly from the
test positives.

Likewise, the 394,459-protein union is a useful Level-2 deployment stress test,
but it is not the published Horizyn Level-1 candidate universe.
