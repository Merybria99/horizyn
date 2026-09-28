# Chemistry-only ReactZyme CGR recovery

This is an isolated preparation/audit pipeline, **not a change to the running
retrieval model**. It reads reaction tables and the released Rhea equations only.
It never reads enzyme sequences, EC labels, pair tables, or UniProt annotations.

## Recovery policy

1. Canonical, stereochemistry/isotope-preserving participant-multiset lookup.
2. If unmatched, ignore water; then water/protons; finally duplicate participants
   **for lookup only**. Recover the original full Rhea equation, never a stripped
   equation. Record which relaxation was needed.
3. Merge source identifiers only when full equations are chemically identical up
   to reversing the two sides. Different partitions remain ambiguous.
4. For unmatched collections, search for bounded, irredundant combinations of
   Rhea equations whose participant sets cover the query. These are explicitly
   **decomposition hypotheses**, even when only one cover is found. They are not
   automatically mapped or used as model features. Default bounds: 32 candidates,
   four equations, 3,000 search nodes; stop after finding two distinct covers.

The equation gate rejects generic atoms, invalid structures, element/isotope/H
or charge imbalance, and identical sides. No automatic stoichiometric repair is
attempted. This is deliberately stricter than merely finding a source match.

## Atom mapping and graphs

RXNMapper runs on **CPU with four threads**, using the installed bundled weights.
Each unique candidate equation is processed once, across train and validation.
Model files and source/code hashes are recorded; changed inputs require a new
run directory. Atomic per-equation receipts allow resume after interruption.

Mappings must preserve each side's chemistry and provide a bijective,
element/isotope-consistent correspondence for every heavy atom. The exported CGR
stores before/after atom and bond attributes, a reaction-center mask, and a
one-hop center-context mask. It does not subtract molecule embeddings. Stereo
orientation is computed relative to mapped neighbors, not SMILES ordering.

`structurally_valid_cgr` means **automated structural checks passed**, not that a
model's atom mapping has been experimentally verified. RXNMapper confidence is
recorded but is not a calibrated probability. `--min-confidence` is an optional
filter, not a substitute for validation. The default 0 does not filter by score.
Large equations are explicitly deferred (`--max-atoms 200` and mapper token limit).
No dummy CGR is produced for failed or unavailable mappings.

## Running

From `/datastor2/deep-proteins/EnzymeDiscovery/horizyn`:

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../env/bin/python -u scripts/recover_reactzyme_cgr.py all \
  --split reaction_smi --output runs/reactzyme_cgr_recovery_v1
```

Stages: `recover`, `map`, `report`, `all`. `all` reuses validated recovery files;
`map` resumes existing receipts. `--limit 10` bounds new mapping records for a
smoke test. `--retry-failed` retries mapper exceptions, not chemistry rejections.
Concurrent controllers for the same output are prohibited by a file lock.

`--split time` and `--split enzyme_smi` use their own reaction tables. Defaults
inspect train/validation only; `--partitions train validation test` must be
explicitly requested to include the held-out test chemistry. Use separate output
directories for different splits/settings. This does not change any benchmark IDs,
positive associations, candidate pools, or splits.

## Outputs and interpretation

- `recovery.json`: one result per original query, including all ambiguities.
- `equations.json`: original, canonical source equations, source IDs and checks.
- `mapped/eq_*.json`: mapping outcomes; successful records contain the CGR.
- `{partition}_cgr_index.json`: availability masks and lookup IDs for every query.
- `report.json`: recovery/mapping counts, usable coverage, match tiers and overlap
  of candidate equations with training. Updated periodically during mapping.

Missing features remain masked; they are not zero-change reactions or negatives.
Relaxed match candidates require audit before use in training. Report both full
benchmark and covered/uncovered performance. Original query-disjointness need
not imply equation-disjointness after reconstruction: the report exposes this
overlap without silently redefining the benchmark. Physiological direction and
the catalytic mechanism are not inferred from Rhea orientation or a net CGR.
