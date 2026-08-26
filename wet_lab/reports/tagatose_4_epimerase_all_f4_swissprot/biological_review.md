# Swiss-Prot Biological Review

## Search Scope

- UniProtKB/Swiss-Prot release: 2026_02 (10-Jun-2026)
- Reviewed canonical proteins ranked: 575,503
- Missing candidate embeddings: 0
- Query: mapped ketose C4 epimerization in `all_f4.yaml`
- Scores are cosine retrieval similarities, not catalytic probabilities.

## Model Top Hits

| F4 checkpoint | Top protein | Score | Swiss-Prot annotation |
|:--|:--|--:|:--|
| ReactZyme time | A0A7H0DN72 | 0.451803 | Virion assembly protein OPG100 |
| ReactZyme enzyme_smi | P56504 | 0.427062 | Mating-type pheromone BAP1(1) |
| ReactZyme reaction_smi | Q9WYP7 | 0.385249 | 5-keto-L-gluconate epimerase |
| Horizyn in-domain | D4GPW5 | 0.463329 | Putative ABC transporter glucose-binding protein TsgA13 |

## Reaction-SMILES Biological Shortlist

| Rank | Protein | Score | Swiss-Prot annotation |
|---:|:--|--:|:--|
| 1 | Q9WYP7 | 0.385249 | 5-keto-L-gluconate epimerase |
| 2 | A8RG82 | 0.372461 | D-psicose 3-epimerase |
| 3 | A9CH28 | 0.360396 | D-psicose 3-epimerase |
| 4 | B8I944 | 0.358340 | D-psicose 3-epimerase |
| 5 | P73599 | 0.357669 | Probable ketose 3-epimerase |
| 8 | Q83JB2 | 0.354321 | Fructoselysine 3-epimerase |
| 13 | C1KKR1 | 0.346145 | D-tagatose 3-epimerase |
| 16 | O50580 | 0.344633 | D-tagatose 3-epimerase |

## Interpretation

The reaction-similarity checkpoint is the only F4 variant whose leading
Swiss-Prot neighborhood is consistently enriched for chemically related ketose
epimerases. The other checkpoints place several short, viral, membrane, or
non-enzymatic proteins near the query. Consequently, the unweighted
reciprocal-rank consensus in `consensus.csv` is retained as a model-comparison
artifact, but it should not be used as the primary wet-lab shortlist.

For experimental selection, prioritize the reaction-similarity candidates and
then screen for catalytic-family membership, metal/cofactor requirements,
expression feasibility, oligomeric state, and the exact substrate
stereochemistry. The ranking alone does not establish activity.
