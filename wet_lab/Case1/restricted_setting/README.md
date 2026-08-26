# Case1 Restricted Setting

This experiment applies the **F3 Set Chemistry** checkpoint in reaction-to-enzyme
(R->E) mode to the D-fructose -> D-tagatose reaction.

## Protocol

1. Select every resolved row from the workbook's `Homologs` sheet: 144 candidates from
   145 rows.
2. Preserve candidates as workbook rows. Identical sequences under different Homolog IDs
   remain separate ranked candidates and therefore receive tied model scores.
3. Generate residue-level ProtT5 embeddings locally.
4. Rank all 144 candidates by F3 cosine similarity to the reaction embedding.
5. Cross-reference the model top 25 with `Ranked Candidates` in two ways:
   - `evidence_id`: the Homolog ID is named by the Ranked Candidate's evidence row.
   - `exact_sequence`: the resolved protein sequences have identical SHA-256 values.

The Ranked Candidates list is treated as a comparison set, not exhaustive ground truth.
Unique-sequence recovery is reported separately so duplicated Homolog rows do not inflate
the overlap statistic.

## Run

From the `horizyn` project root:

```bash
../.capability-run-py/bin/python \
  wet_lab/Case1/restricted_setting/run_experiment.py --gpu 0
```

The runner does not use the account home directory. Model caches, temporary files, and
outputs stay below this folder.

Use `--force-embeddings` to regenerate candidate ProtT5 embeddings and
`--force-features` to regenerate the reaction features.

## Outputs

- `candidate_pool/`: exact ranked pool, metadata, FASTA, ProtT5 embeddings, and manifest.
- `runs/*/rankings.csv`: complete 144-candidate F3 ranking.
- `runs/*/top_25.csv`: raw F3 top 25.
- `results/top25_vs_ranked_candidates.csv`: annotated comparison table.
- `results/report.md`: readable result summary and top-25 table.
- `experiment_manifest.json`: paths, provenance, and machine-readable summary.
- `logs/`: ProtT5 and F3 query logs.

## All F3 splits

Run the same restricted query concurrently with the `time` and `enzyme_smi` F3
checkpoints and compare both with the existing `reaction_smi` result:

```bash
../.capability-run-py/bin/python \
  wet_lab/Case1/restricted_setting/run_all_f3_splits.py --gpus 0 3
```

The comparison under `all_splits/` contains split-specific top-25 overlap, full-ranking
agreement, and a diagnostic reciprocal-rank-fusion consensus. Raw cosine scores are not
averaged across independently trained checkpoints.

## Structures

The only retained structure layout is the canonical tree at
`structures/F3_set_chemistry/D-fructose_to_D-tagatose/`, with `overlapping/` and
`non-overlapping/` under each split. Repeated sequences and candidates across splits are
represented as hard links, so they do not duplicate PDB storage. Split manifests retain
the checkpoint, rank, score, overlap basis, structure source, and confidence.
