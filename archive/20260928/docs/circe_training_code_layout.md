# CIRCE training code layout

Existing commands, model/loss settings, checkpoint keys and feature formats are
preserved. This cleanup does not change the training recipe or restart active runs.

| Responsibility | Source |
| --- | --- |
| CLI, reporting, constructors, callbacks and training | `scripts/train_protein_pooling.py` |
| Fast-I/O wrapper and throughput benchmark | `scripts/train_protein_pooling_fast_io.py` |
| YAML, overrides and public validation API | `horizyn/config.py` |
| Ordered validation sections | `horizyn/config_validation.py` |
| Config-to-constructor mappings and shared aliases | `horizyn/training_options.py` |
| CPU threading and DDP setup | `horizyn/training_runtime.py` |
| Partial model and BioFP warm starts | `horizyn/training_warm_start.py` |
| Recovery checkpoints and resume-safe early stopping | `horizyn/training_checkpoints.py` |
| Encoder architecture | `horizyn/model.py` |
| Encoding dispatcher and training objectives | `horizyn/protein_pooling_lightning_module.py` |

## What was simplified

The direct constructors and literal defaults are restored. The duplicate
defaults registry and constructor option-context/helper scaffolding are removed.

Configuration forwarding uses one helper, `_with_defaults`. It accepts only
explicitly listed fields and can prefix output names, for example translating
`enabled` into `e2r_adapter_enabled`. Defaults stay beside their mappings.
Explicit `None`, `False` and zero remain unchanged. Derived fallbacks and aliases
stay explicit; the CLI's historical defaults need not equal constructor defaults.

The uncached loss path retains `_encode_training_inputs`: one call per tower,
with the original attention/pooling-detail conditions. Cached and distributed
loss paths, auxiliary penalties and loss arithmetic are unchanged.

## Compatibility boundaries

- `horizyn.config.validate_config` retains validation order and error messages.
- Builders import no Torch, construct no models and do not mutate configuration.
- ChIRo/chirality/ChIENN share one resolver with the existing precedence rules.
- Experimental variants, inactive options and public constructor arguments remain supported.
- Constructor calls, the config loader and source-replay loading remain in the
  training entry point so the fast-I/O wrapper can still replace them.
- A partial warm start remains distinct from `--resume`; optimizer restoration,
  validation cadence, early stopping and recovery/best checkpoints are unchanged.
- New manifests hash the relevant code, including `model.py`. Existing run
  artifacts are not rewritten.

## Verification and size

| Production file | Before this reduction | After |
| --- | ---: | ---: |
| `model.py` | 5,912 | 5,776 |
| `protein_pooling_lightning_module.py` | 4,404 | 4,226 |
| `training_options.py` | 948 | 779 |
| Redundant defaults registry | 246 | 0 |
| Fast-I/O wrapper | 181 | 180 |
| **Total** | **11,691** | **10,961** |

This removes 730 production lines, excluding tests/docs. The result is also 95
lines smaller than the version before the growth-producing refactor.

The focused suite passed 106 tests. Comparisons against the saved original
matched 3,920 configuration cases (values, key order, exceptions and non-mutation)
and 12,288 encoding cases (calls, arguments, returned details and statistics).
Original constructor ASTs match; `model.py` matches its pre-scaffolding source
byte-for-byte. Tensor-level/distributed testing remains a separate gate; the
previous full-ML checks stalled while importing PyTorch from shared storage.

Dependency-light checks:

```bash
python -B tests/unit/test_training_options.py -v
python -B tests/unit/test_config_validation_sections.py -v
python -B tests/unit/test_protein_pooling_core_structure.py -v
```

Full-environment regression suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES= python -B -m pytest \
  tests/unit/test_config.py \
  tests/unit/test_training_options.py \
  tests/unit/test_config_validation_sections.py \
  tests/unit/test_protein_pooling_core_structure.py \
  tests/unit/test_horizyn1_multimodal_training_smoke.py \
  tests/unit/test_training_epoch_resume.py \
  tests/unit/test_h200_training_performance.py \
  tests/unit/test_biological_residual.py \
  tests/unit/test_training_io.py \
  -q -o addopts= -p no:cacheprovider
```
