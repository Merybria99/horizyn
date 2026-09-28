# Biological supervision: local gradient diagnostic

This evaluates the fitted phase2 weight3 heads on their original training graph. No parameter is updated. The same positive CE, identity weight10, relative margin0.1 and fixed family divisor3 are used as in training. All four target-trained models are reported.

| Target / term | Applied loss | Gradient norm / retrieval norm | Gradient cosine with retrieval |
| --- | ---: | ---: | ---: |
| reaction_smi / ec | 0.00097319 | 0.011143 | -0.116578 |
| reaction_smi / cofactor | 1.505335e-05 | 0.004931 | -0.147927 |
| reaction_smi / mechanism | 0.003678834 | 0.043250 | -0.164913 |
| reaction_smi / identity_weighted | 0.09291233 | 0.999214 | -0.999094 |
| enzyme_smi / ec | 0.001165334 | 0.012462 | -0.107318 |
| enzyme_smi / cofactor | 0 | 0.000000 | +0.000000 |
| enzyme_smi / mechanism | 0.003519751 | 0.042015 | -0.197464 |
| enzyme_smi / identity_weighted | 0.09526333 | 1.004139 | -0.999175 |
| time / ec | 0.001042771 | 0.015661 | -0.128291 |
| time / cofactor | 8.008895e-06 | 0.004302 | -0.077452 |
| time / mechanism | 0.004131572 | 0.020454 | -0.038378 |
| time / identity_weighted | 0.1107204 | 1.000413 | -0.998874 |
| enzymemap / ec | 1.676764e-05 | 0.002464 | +0.001578 |
| enzymemap / cofactor | 7.076521e-06 | 0.034979 | +0.012075 |
| enzymemap / mechanism | 0.02429692 | 0.253576 | -0.087577 |
| enzymemap / identity_weighted | 0.04969936 | 1.017300 | -0.947688 |

A negative cosine means the two local gradients oppose each other at this fitted point. It does not establish that the biological term harms generalization: regularization can deliberately oppose the fitted retrieval objective. Equal scalar loss coefficients need not yield equal gradient magnitudes. Neither the loss value nor the gradient norm is a causal importance percentage.

These are gradients with respect to the existing phase2 parameters before gradient clipping. Adam momentum, second-moment scaling and prior training steps are not represented. The original F3 and its learned residue queries stay frozen in this study. Benchmark/Case1 removal controls provide outcome evidence separately.

[Raw gradients statistics and source hashes](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_biology_phase2_gradient_audit_20260921_v1/gradient_audit.json) · [Architecture and outcome interpretation](v4_biological_signal_architecture.md)
