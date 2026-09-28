# Fresh ReactZyme controls with matched validation cadence

The unannotated F3 models use the same seed42, ten epochs, batch512, frozen SLEEC and fast validation cadence as the fresh biological models. Original V4 used an additional retrieval validation iterator, which can alter random-number consumption. These fresh controls remove that known protocol confound from the biological-loss comparison. They do not isolate how much of the difference from original V4 is caused by validation cadence alone. It remains one seed and exploratory repeated test evaluation.

The primary comparison below uses the original phase2 objective for every model. Consequently the biological contrast is confined to F3 training. Existing inference fusion, dictionary and score coefficients are unchanged. No test result chooses a checkpoint or seed.

| Split / direction | Fresh unannotated | Attraction 0.1 | Relative 0.1 | Relative 1 |
| --- | ---: | ---: | ---: | ---: |
| reaction_smi / reaction_to_enzyme | 0.405064 | 0.402602 (-0.002462) | 0.398129 (-0.006935) | 0.401638 (-0.003426) |
| reaction_smi / enzyme_to_reaction | 0.539420 | 0.532584 (-0.006836) | 0.546721 (+0.007301) | 0.541618 (+0.002198) |
| enzyme_smi / reaction_to_enzyme | 0.667073 | 0.664804 (-0.002269) | 0.665099 (-0.001973) | 0.664423 (-0.002650) |
| enzyme_smi / enzyme_to_reaction | 0.971730 | 0.954584 (-0.017146) | 0.971280 (-0.000450) | 0.957043 (-0.014687) |
| time / reaction_to_enzyme | 0.539458 | 0.539845 (+0.000387) | 0.539334 (-0.000124) | 0.539148 (-0.000309) |
| time / enzyme_to_reaction | 0.811951 | 0.794002 (-0.017949) | 0.796441 (-0.015510) | 0.792256 (-0.019694) |

The parentheses show biological minus matched unannotated MRR. Enzyme-Sim and Time E→R are sensitive to near-tied scores for distinct reaction IDs with chemically equivalent released inputs. Their raw MRR changes cannot automatically be interpreted as improved chemistry.

## Phase2-only secondary control

These models share the fresh unannotated F3 weights above and add relative biological weight0.1 only to the existing phase2 objective. They are reported separately from the primary causal contrast.

| Split | R→E | E→R |
| --- | ---: | ---: |
| reaction_smi | 0.405054 | 0.539383 |
| enzyme_smi | 0.667073 | 0.972818 |
| time | 0.539452 | 0.803642 |

The labels `f3_biology` in the raw control directories refer to a pipeline slot: its configured biological weight is zero. It must not be described as biologically supervised F3. No Case1 comparison is inferred from these benchmark scores.

[Architecture](v4_biological_signal_architecture.md) · [Fresh EnzymeMap control and annotation specificity](v4_biology_additional_controls_20260921.md)

[Direct Case1 evaluations of qualified controls](v4_matched_control_case1_20260921.md)

[Source hashes and paired differences](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_matched_reactzyme_control_20260921_v1/comparison.json)
