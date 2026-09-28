"""Compare palette options on the same audited ReactZyme inference heatmap.

Static previews only. Keep the numerical data, scale, SciencePlots style and
minimal layout identical, so the only comparison is the two-root colour palette.
Signed cell values and a labelled zero-centred colour bar preserve interpretation
without relying on hue. Titles and explanation stay outside the images.
"""
from __future__ import annotations

import csv
import hashlib
import json

import plot_study as study
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np


PALETTES = {
    'a_navy_rose': {
        'label': 'A — Muted navy and dusty rose',
        'anchors': ['#B56587', '#E5B4C8', '#FAFAFA', '#B3CCDF', '#355C7D'],
    },
    'b_blue_coral': {
        'label': 'B — Cool blue and soft coral',
        'anchors': ['#C96E66', '#EDBDB6', '#FAFAFA', '#AFCCE5', '#3979AD'],
    },
    'c_blue_orange': {
        'label': 'C — Blue and orange',
        'anchors': ['#B86B2D', '#E8C3A3', '#FAFAFA', '#ADC8E2', '#28679C'],
    },
}


def main():
    study.style()
    output = study.HERE / 'palette_previews'
    output.mkdir(exist_ok=True)
    source = study.FIGURES / 'plotted_contrasts.csv'
    rows = [r for r in csv.DictReader(source.open())
            if r['figure'] == '01_reactzyme_inference']
    assert len(rows) == 18
    data = np.asarray([float(r['delta']) * 100 for r in rows]).reshape(6, 3)
    norm = TwoSlopeNorm(vmin=-np.abs(data).max(), vcenter=0,
                        vmax=np.abs(data).max())
    labels = [f'{label}  {direction}' for _, label in study.SPLITS
              for direction in ('R→E', 'E→R')]
    metadata = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'data': data.tolist(), 'previews': []}
    for name, palette in PALETTES.items():
        cmap = LinearSegmentedColormap.from_list(name, palette['anchors'], N=513)
        ramp = cmap(np.linspace(0, 1, 513))[:, :3]
        simulations = {}
        for view in ('standard', 'protanomaly', 'deuteranomaly', 'tritanomaly'):
            rgb = ramp if view == 'standard' else np.clip(study.cspace_convert(
                ramp, {'name': 'sRGB1+CVD', 'cvd_type': view, 'severity': 100},
                'sRGB1'), 0, 1)
            luminance = study.relative_luminance(rgb)
            monotonic = bool(np.all(np.diff(luminance[:257]) >= -1e-6)
                             and np.all(np.diff(luminance[256:]) <= 1e-6))
            assert monotonic, (name, view)
            simulations[view] = {'luminance_monotonic_toward_neutral': monotonic}
        fig, ax = plt.subplots(figsize=(12.8, 5.2))
        fig.subplots_adjust(left=.235, right=.88, top=.98, bottom=.20)
        im = ax.imshow(data, aspect='auto', cmap=cmap, norm=norm)
        ax.set_xticks(range(3), [c[2] for c in study.CONTRASTS], fontsize=10)
        ax.set_yticks(range(6), labels, fontsize=11)
        ax.minorticks_off()
        ax.tick_params(length=0, pad=10)
        for i in range(6):
            for j in range(3):
                ax.text(j, i, f'{data[i,j]:+.4f}', ha='center', va='center',
                        fontsize=11, color=study.annotation_color(cmap(norm(data[i,j]))))
        for y in (1.5, 3.5):
            ax.axhline(y, color='white', linewidth=2)
        fig.colorbar(im, ax=ax, fraction=.045, pad=.035).set_label('ΔMRR × 100')
        for extension in ('png', 'svg'):
            fig.savefig(output / f'{name}.{extension}', dpi=160,
                        bbox_inches='tight', pad_inches=.14)
        plt.close(fig)
        metadata['previews'].append(dict(id=name, **palette, simulations=simulations))
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    text = ['# Palette comparison', '',
            'The same audited ReactZyme inference figure, with identical values and colour scales. '
            'All options retain SciencePlots and the minimal layout. '
            'A explores the original pink–blue direction; B is warmer; C changes pink to orange and is the user-selected option. '
            'Signed labels support interpretation without hue. '
            'Simulations check gradient luminance, not universal accessibility.', '']
    for name, palette in PALETTES.items():
        text.extend([f"## {palette['label']}", '', f'![{palette["label"]}]({name}.png)', ''])
    (output / 'README.md').write_text('\n'.join(text))
    print(f'Saved 3 palette previews in {output}')


if __name__ == '__main__':
    main()
