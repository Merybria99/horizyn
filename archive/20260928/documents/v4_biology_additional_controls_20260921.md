# Matched biological-loss controls beyond seed42

All controls use the unchanged architecture, target-training associations and fixed 10 epochs. The first phase2 variant uses the original unannotated phase2 objective; the second also applies biological geometry to phase2. For shuffled and removed-family controls, their changed annotations apply throughout training. No control or seed is selected by its test result.

## f3_biology

Each difference below is the corresponding correct-label model minus the indicated control. The fresh unannotated seed42 row therefore compares F3 biological supervision with no F3 supervision; the other rows test annotation assignment or mechanism removal.

| Control | T1 BEDROC85 | T1 BEDROC20 | T1 EF5 | T1 EF10 | T2 BEDROC85 | T2 BEDROC20 | T2 EF5 | T2 EF10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fresh_control_seed42 | +0.019622 | +0.007139 | -0.180084 | +0.007355 | +0.021680 | +0.007134 | -0.228225 | +0.014252 |
| shuffled_seed43 | +0.021818 | +0.014885 | +0.078533 | +0.155555 | +0.023508 | +0.016585 | +0.090567 | +0.212286 |
| shuffled_seed44 | -0.006181 | -0.007268 | -0.277806 | -0.115600 | -0.008050 | -0.009207 | -0.319378 | -0.108162 |
| without_mechanism_seed43 | +0.007817 | +0.001245 | -0.042102 | +0.016311 | +0.007910 | +0.001050 | -0.060071 | +0.038539 |
## f3_phase2_biology

Each difference below is the corresponding correct-label model minus the indicated control. The fresh unannotated seed42 row therefore compares F3 biological supervision with no F3 supervision; the other rows test annotation assignment or mechanism removal.

| Control | T1 BEDROC85 | T1 BEDROC20 | T1 EF5 | T1 EF10 | T2 BEDROC85 | T2 BEDROC20 | T2 EF5 | T2 EF10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fresh_control_seed42 | +0.019615 | +0.007126 | -0.175387 | +0.007355 | +0.021675 | +0.007124 | -0.224485 | +0.014252 |
| shuffled_seed43 | +0.021682 | +0.014812 | +0.078533 | +0.162129 | +0.023358 | +0.016506 | +0.088074 | +0.219765 |
| shuffled_seed44 | -0.006299 | -0.007423 | -0.284165 | -0.115600 | -0.008173 | -0.009375 | -0.334923 | -0.105749 |
| without_mechanism_seed43 | +0.007882 | +0.001287 | -0.042102 | +0.020375 | +0.007978 | +0.001097 | -0.059240 | +0.038539 |

## Current versus archived unannotated seed42

The largest absolute difference across the eight screening metrics is 0.00399762721. The two BEDROC85 differences are -0.000064766 and -0.000070602. The end-to-end pipelines therefore do not reproduce every ranking exactly.

[Archived result](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_geometry_20260921_v1/enzymemap/control/test_summary.json) · [Fresh control](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_additional_controls_20260921_v1/fresh_control_seed42/followup/enzymemap/enzymemap/f3_biology/test_summary.json)

A separate parameter audit verifies that all 120 F3 state tensors are bit-for-bit identical: True. Maximum parameter difference is 0. Thus the small end-to-end differences arise after the identical F3 state, in feature export, phase2 fitting and/or scoring; the exact downstream source has not been isolated. [Checkpoint hashes and tensor comparison](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_additional_controls_20260921_v1/fresh_control_seed42/checkpoint_reproduction.json).

## Annotation specificity across seeds: f3_biology

Correct labels minus shuffled labels, using the same F3 weight 1 and seed within each pair. The same fixed annotation shuffle is used across seeds; this is not variability over shuffles.

| Seed | T1 BEDROC85 | T1 BEDROC20 | T1 EF5 | T1 EF10 | T2 BEDROC85 | T2 BEDROC20 | T2 EF5 | T2 EF10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | +0.006608 | +0.001166 | -0.196392 | +0.031954 | +0.006586 | +0.000953 | -0.236636 | +0.047446 |
| 43 | +0.021818 | +0.014885 | +0.078533 | +0.155555 | +0.023508 | +0.016585 | +0.090567 | +0.212286 |
| 44 | -0.006181 | -0.007268 | -0.277806 | -0.115600 | -0.008050 | -0.009207 | -0.319378 | -0.108162 |
| Mean | +0.007415 | +0.002928 | -0.131888 | +0.023970 | +0.007348 | +0.002777 | -0.155149 | +0.050523 |

## Annotation specificity across seeds: f3_phase2_biology

Correct labels minus shuffled labels, using the same F3 weight 1 and seed within each pair. The same fixed annotation shuffle is used across seeds; this is not variability over shuffles.

| Seed | T1 BEDROC85 | T1 BEDROC20 | T1 EF5 | T1 EF10 | T2 BEDROC85 | T2 BEDROC20 | T2 EF5 | T2 EF10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | +0.006400 | +0.001048 | -0.189817 | +0.031954 | +0.006349 | +0.000822 | -0.221678 | +0.047446 |
| 43 | +0.021682 | +0.014812 | +0.078533 | +0.162129 | +0.023358 | +0.016506 | +0.088074 | +0.219765 |
| 44 | -0.006299 | -0.007423 | -0.284165 | -0.115600 | -0.008173 | -0.009375 | -0.334923 | -0.105749 |
| Mean | +0.007261 | +0.002812 | -0.131817 | +0.026161 | +0.007178 | +0.002651 | -0.156176 | +0.053821 |

Correct annotation assignment improves BEDROC85 in two of three seeds, but worsens all eight metrics at seed44. The mechanism-removal seed43 check improves BEDROC85 when mechanism supervision is retained, while EF5 falls. These are conditional, mixed contributions; they do not support a universal benefit for all annotation families.

These controls were declared after earlier outcomes, so they remain exploratory. A positive full-loss-versus-removal contrast may include a regularization-strength effect. The shuffled-label control addresses whether the true annotation assignment matters. No result establishes a named biochemical meaning for an individual residue query.

[All absolute metrics, paired changes and source hashes](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_additional_controls_20260921_v1/comparison.json) · [Three-seed results](v4_relative_biology_seed_sensitivity_20260921.md)
