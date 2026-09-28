# Existing residue-query behavior on unseen validation proteins

This fixed diagnostic uses 128 validation proteins absent from target training by exact protein ID. It compares native F3 models before phase2, semantic scoring or the fixed inference fusion adjustment. Selection uses sorted hashed IDs, not attention or activity outcomes. No test or Case1 data are used.

| Model | Mean normalized raw attention entropy | Effective support / length | Centered head cosine | Global-output cosine |
| --- | ---: | ---: | ---: | ---: |
| V4 | 0.9431 | 0.7347 | 0.5591 | 0.9951 |
| attraction_0p1 | 0.9441 | 0.7388 | 0.5502 | 0.9951 |
| relative_0p1 | 0.9436 | 0.7368 | 0.5515 | 0.9951 |
| relative_1 | 0.9448 | 0.7415 | 0.5492 | 0.9951 |

Entropy is divided by log(sequence length); one means uniform raw attention. Effective support is exp(entropy) after the uniform mixture. Centered head cosine measures similarity after subtracting uniform attention; smaller values indicate more distinct residue weighting. Global-output cosine measures proximity to the existing global branch at the native training fusion scale.

| Model | Mean global gate | Mean SLEEC gate | Combined four learned-view gates | Permutation error, example |
| --- | ---: | ---: | ---: | ---: |
| V4 | 0.1809 | 0.3309 | 0.4882 | 2.24e-08 |
| attraction_0p1 | 0.1810 | 0.3300 | 0.4891 | 2.24e-08 |
| relative_0p1 | 0.1807 | 0.3276 | 0.4917 | 4.47e-08 |
| relative_1 | 0.1813 | 0.3314 | 0.4873 | 2.98e-08 |

Gate magnitudes are descriptive, not an additive decomposition of prediction importance. The gates act by feature dimension before another projection. The four queries select contextual residue content and are shared across proteins; they are not position embeddings. Reversing the already computed residue vectors preserves the pooled output within floating-point tolerance. Reversing an amino-acid sequence before ProtT5 would be a different operation.

Neither diverse attention nor changed attention proves a catalytic-site assignment. These data do not justify naming individual queries EC, cofactor or mechanism queries. Biological supervision reaches the final shared embedding and can be distributed across existing views.

[Per-protein values, attention example and checkpoint hashes](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_query_audit_20260921_v1/query_audit.json) · [Architecture explanation](v4_biological_signal_architecture.md)
