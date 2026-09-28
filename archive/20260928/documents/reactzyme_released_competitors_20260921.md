# Public competitors for a ReactZyme comparison

Release audit: 21 September 2026. This note checks official publications and repositories; it does not claim that every released training pipeline or checkpoint has been reproduced locally.

The recommended core comparison is the official ReactZyme baseline families, CLIPZyme, and the original Horizyn model, all fitted on the same ReactZyme training splits. EnzymeCAGE and CREEP provide additional public comparators, with greater adaptation requirements. EnzGFM is a relevant published ReactZyme backbone comparison. TIGER and FGW-CLIP remain important published-score references, but an official public implementation/checkpoint was not verified in this audit.

## 1. Baselines released with ReactZyme

The [official repository](https://github.com/WillHua127/ReactZyme) provides training and retrieval scripts for four families:

| Family | Training entry point | Evaluation entry point |
|---|---|---|
| MLP | `train.py` | `retrieval.py` |
| Contrastive MLP | `train_contra.py` | `retrieval.py` |
| Transformer | `train_tfmr.py` | `retrieval_tfmr.py` |
| Bi-RNN | `train_rnn.py` | `retrieval_rnn.py` |

The [ReactZyme paper](https://arxiv.org/abs/2408.13659) also varies molecular representations (MAT/UniMol, 2D/3D) and protein representations (ESM/SaProt, including geometric enhancements). These combinations should be identified by architecture and features rather than counted as unrelated named methods.

The repository releases the time, sequence-similarity, and molecular-similarity splits and documents feature preparation. Training code is verified; a complete collection of trained retrieval checkpoints covering every split/configuration was not verified. These baselines are the most direct benchmark reproduction targets.

## 2. Public models to retrain on ReactZyme

| Model | Public release evidence | Role and adaptation requirement |
|---|---|---|
| **CLIPZyme** | [Official code and training configuration](https://github.com/pgmikhael/clipzyme); [model/data archive](https://zenodo.org/records/11187747). | Major external enzyme-screening comparator. Its released model uses EnzymeMap training data. Retrain on ReactZyme and document reaction preprocessing and protein-structure coverage. |
| **Horizyn** | [Official code, training command, and checkpoint downloader](https://github.com/dayhofflabs/horizyn). Development and full-data inference checkpoints are distinguished. | Essential parent-method baseline for our modified codebase. Reproduce the original fingerprint/ProtT5 dual encoder with MLNCE on the target splits; do not substitute our F3 configuration for the original method. |
| **EnzymeCAGE** | [Official repository](https://github.com/GENTEL-lab/EnzymeCAGE) contains `train.py`, data/checkpoint links, and structural preprocessing instructions. | Useful structural competitor. Its native evaluations include Orphan-335 and Enzyme-405; these are not the official ReactZyme splits. Target-data training and candidate structure/pocket coverage must be established. |
| **CREEP, released with CARE** | [Training and extraction instructions](https://github.com/jsunn-y/CARE/tree/main/task2_baselines) and [advertised pretrained models](https://huggingface.co/jsunn-y/CARE_pretrained). | Additional reaction/protein representation competitor. CARE's EC-mediated evaluation needs adaptation to the exact ReactZyme associations and candidate pools. Text inputs, if used, must be reported as additional information. |

Release links and instructions establish documented availability; they do not establish identical training data, absence of overlap, or successful end-to-end reproduction. In particular, Horizyn's full-data inference checkpoint should not be used as a supposedly held-out ReactZyme baseline.

## 3. Other relevant methods and release limits

- **EnzGFM:** The [Nature Communications paper](https://www.nature.com/articles/s41467-026-75283-3) reports ReactZyme evaluation, and [public code](https://github.com/DeepBxM/EnzGFM) includes enzyme–reaction retrieval scripts. It is especially relevant as an enzyme-specific protein-backbone control. Availability of the exact downstream retrieval checkpoints was not verified; a checkpoint path in source code alone is insufficient evidence of a release.
- **TIGER:** A main published ReactZyme competitor. An official public training implementation/checkpoint was not identified in the checked [paper and publication record](https://aclanthology.org/2026.acl-long.1643/). Its benchmark-adapted baseline rows should not automatically be attributed to the original released methods/checkpoints. Keep reported and locally reproduced scores distinct.
- **FGW-CLIP:** Another main published comparator, including the EnzymeMap screening setting. An official public implementation/checkpoint was not identified from the checked [paper](https://arxiv.org/abs/2512.08508). This is a qualified availability finding, not proof that no repository exists.
- **VenusRXN:** [Public code](https://github.com/zy-zhou/VenusRXN) exists, but the checked README says pretrained checkpoints will be provided in future updates. It does not currently meet the same verified code-plus-model criterion as CLIPZyme.
- **CLIPZyme+:** The current CLIPZyme repository also describes a cofactor-prediction extension. It is not a separate direct ReactZyme retrieval baseline.

## 4. Controlled comparison design

These are proposed experimental controls, not claims that the runs are already complete:

1. Train each retrieval method separately on exactly the official training associations for each of the three ReactZyme splits. Start the downstream retrieval model from a fresh initialization; disclose every pretrained feature extractor.
2. Keep test associations, query sets, candidate pools, positive handling, and metric implementations identical for both retrieval directions. Document how reaction inputs are constructed, especially whether direction or atom mapping is supplied.
3. Select configurations and checkpoints using validation data, with comparable tuning budgets. Do not pick architecture variants using test performance.
4. Preserve each complete method's intended objective in the main comparison. Add shared-feature/shared-loss ablations when making a narrower claim that architecture alone explains an improvement.
5. Report structures, text, EC labels, cofactors, and other auxiliary information explicitly. Matching association data is necessary, but does not by itself equalize all information available to the methods.
6. Use native pretrained checkpoints only for clearly labeled transfer experiments or implementation checks. Treat published TIGER/FGW-CLIP numbers as reported results until they can be reproduced under the same evaluator.

Recommended order: official ReactZyme baselines and original Horizyn first; CLIPZyme next; EnzymeCAGE and CREEP as expanded external comparisons; EnzGFM as a backbone control once its exact required weights are resolved.
