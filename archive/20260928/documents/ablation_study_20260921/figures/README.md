# Ablation figure exports

PNG files are embedded in the report. SVG preserves editable vector text; PDF is suitable for manuscript inclusion. Each plot uses the audited experiment records, with source hashes in `manifest.json`. The images have no titles, subtitles, panel titles or footnotes. Explanations and scope notes appear in the [manuscript captions](../report.md).

Figure 04 uses one shared symmetric logarithmic color scale (linear within ±0.001 ΔMRR×100); its printed cell values are unchanged. Figure 01 retains a linear color scale. See the manuscript captions for interpretation.

Style: [SciencePlots 2.2.1](https://github.com/garrettj403/SciencePlots), using `science` and `no-latex`, serif fonts, inward ticks and thin axes. The selected custom gradient runs from orange `#B86B2D` through pale orange `#E8C3A3` and neutral `#FAFAFA` to pale blue `#ADC8E2` and blue `#28679C`. Signed labels, marker shapes and hatching provide redundant cues. See [color-vision diagnostics](palette_accessibility.json) and the [simulation preview](palette_accessibility_preview.png).

Install the small plotting-only dependencies without changing the training environment:

```bash
python3 -m pip install --no-deps --target .deps/ablation-figures -r documents/ablation_study_20260921/requirements-figures.txt
python3 documents/ablation_study_20260921/plot_study.py
```

| Figure | PNG | SVG | PDF |
| --- | --- | --- | --- |
| 01 reactzyme inference | [PNG](01_reactzyme_inference.png) | [SVG](01_reactzyme_inference.svg) | [PDF](01_reactzyme_inference.pdf) |
| 02 enzymemap inference | [PNG](02_enzymemap_inference.png) | [SVG](02_enzymemap_inference.svg) | [PDF](02_enzymemap_inference.pdf) |
| 03 temperature paired seeds | [PNG](03_temperature_paired_seeds.png) | [SVG](03_temperature_paired_seeds.svg) | [PDF](03_temperature_paired_seeds.pdf) |
| 04 reactzyme phase2 biology | [PNG](04_reactzyme_phase2_biology.png) | [SVG](04_reactzyme_phase2_biology.svg) | [PDF](04_reactzyme_phase2_biology.pdf) |
| 05 enzymemap phase2 biology | [PNG](05_enzymemap_phase2_biology.png) | [SVG](05_enzymemap_phase2_biology.svg) | [PDF](05_enzymemap_phase2_biology.pdf) |
| 06 reactzyme near tie sensitivity | [PNG](06_reactzyme_near_tie_sensitivity.png) | [SVG](06_reactzyme_near_tie_sensitivity.svg) | [PDF](06_reactzyme_near_tie_sensitivity.pdf) |
| 07 f3 biology paired seeds | [PNG](07_f3_biology_paired_seeds.png) | [SVG](07_f3_biology_paired_seeds.svg) | [PDF](07_f3_biology_paired_seeds.pdf) |
| 08 case1 controls | [PNG](08_case1_controls.png) | [SVG](08_case1_controls.svg) | [PDF](08_case1_controls.pdf) |
