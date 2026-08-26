# F3 Restricted Setting Across All Splits

The same D-fructose -> D-tagatose reaction and the same 144 Homolog rows were scored independently with the validation-selected F3 checkpoint from each ReactZyme split.

Cosine values are checkpoint-specific and are not averaged. The consensus uses reciprocal-rank fusion with k=60 and is diagnostic rather than a trained model.

## Ranked Candidates overlap

| Checkpoint split | Top-25 matching rows | Direct evidence | Exact sequence | Eligible unique recovery |
|---|---:|---:|---:|---:|
| time | 4 | 3 | 4 | 3/25 (12.0%) |
| enzyme_smi | 12 | 7 | 12 | 7/25 (28.0%) |
| reaction_smi | 12 | 7 | 12 | 7/25 (28.0%) |
| RRF consensus | 10 | 6 | 10 | 6/25 (24.0%) |

## Rank agreement

| Pair | Spearman (144) | Top-10 overlap | Top-25 overlap |
|---|---:|---:|---:|
| time vs enzyme_smi | 0.313 | 2/10 | 10/25 |
| time vs reaction_smi | 0.456 | 1/10 | 10/25 |
| enzyme_smi vs reaction_smi | 0.756 | 5/10 | 24/25 |

## Consensus top 25

| Consensus rank | Candidate | Time | Enzyme-Sim | Reaction-Sim | Mean rank |
|---:|---|---:|---:|---:|---:|
| 1 | P9 | 1 | 1 | 4 | 2.0 |
| 2 | H044 | 12 | 2 | 1 | 5.0 |
| 3 | H045 | 13 | 4 | 2 | 6.3 |
| 4 | H043 | 15 | 3 | 5 | 7.7 |
| 5 | H048 | 14 | 6 | 3 | 7.7 |
| 6 | P23 | 5 | 5 | 13 | 7.7 |
| 7 | H035 | 19 | 14 | 9 | 14.0 |
| 8 | H005 | 20 | 15 | 10 | 15.0 |
| 9 | H012 | 25 | 17 | 6 | 16.0 |
| 10 | H007 | 17 | 23 | 11 | 17.0 |
| 11 | H011 | 27 | 19 | 7 | 17.7 |
| 12 | H069 | 28 | 18 | 8 | 18.0 |
| 13 | H039 | 2 | 33 | 27 | 20.7 |
| 14 | H037 | 3 | 31 | 29 | 21.0 |
| 15 | H042 | 4 | 30 | 34 | 22.7 |
| 16 | H047 | 30 | 24 | 12 | 22.0 |
| 17 | H049 | 49 | 16 | 14 | 26.3 |
| 18 | H067 | 56 | 12 | 21 | 29.7 |
| 19 | P336 | 58 | 13 | 22 | 31.0 |
| 20 | H041 | 8 | 34 | 54 | 32.0 |
| 21 | P373 | 66 | 21 | 15 | 34.0 |
| 22 | H022 | 115 | 7 | 18 | 46.7 |
| 23 | P8 | 116 | 8 | 17 | 47.0 |
| 24 | H006 | 67 | 22 | 16 | 35.0 |
| 25 | H050 | 57 | 25 | 19 | 33.7 |
