# F3 D-fructose to D-tagatose Structures

Each split contains its top-25 structures partitioned against the workbook's resolved `Ranked Candidates` sheet.

`overlapping` means an evidence-ID or exact-sequence match. `non-overlapping` means neither match is present.

| Split | Overlapping | Non-overlapping | Total |
|---|---:|---:|---:|
| time | 4 | 21 | 25 |
| enzyme_smi | 12 | 13 | 25 |
| reaction_smi | 12 | 13 | 25 |

PDB files are hard-linked to validated source structures when supported by the filesystem, with ordinary copies as fallback. Each split manifest records the checkpoint, model score, match basis, structure source, confidence, and original structure path.
