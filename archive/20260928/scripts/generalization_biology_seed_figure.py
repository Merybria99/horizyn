#!/usr/bin/env python3
"""Export the paired-seed screening comparison as a standalone scientific figure."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CROSS = ROOT / 'runs/generalization_20260919_2251/cross_paper_retraining'


def main():
    source = CROSS / 'v4_biological_f3_seed_controls_20260921_v1/seed_comparison.json'
    data = json.loads(source.read_text())
    records = {(r['seed'], r['supervision']): r['summary'] for r in data['records']}
    target = json.loads((CROSS / 'goal_primary_comparators_20260921.json').read_text())['enzymemap']
    colors = {42: '#087f8c', 43: '#c56b16', 44: '#7a5195'}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), sharey=True)
    for ax, table, title in zip(axes, ('table1', 'table2'), ('Full screening pool', 'Training enzymes removed')):
        for seed in (42, 43, 44):
            y = [records[seed, kind][table]['bedroc85'] for kind in ('control', 'biology')]
            ax.plot([0, 1], y, 'o-', color=colors[seed], lw=1.8, ms=5, label=f'Seed {seed}')
            ax.annotate(f'{y[1]-y[0]:+.4f}', (.5, sum(y)/2), xytext=(0, 8),
                        textcoords='offset points', ha='center', color=colors[seed], fontsize=9)
        ax.axhline(target[table]['bedroc85'], color='#555555', linestyle='--', lw=1.1,
                   label='Primary published comparator')
        ax.set(xticks=[0, 1], xticklabels=['V4 control', '+ biological loss'], xlim=(-.18, 1.18),
               ylim=(.385, .62), title=title)
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('Test BEDROC85 ↑')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='lower center', bbox_to_anchor=(.5, -.01), frameon=False,
               fontsize=9)
    fig.suptitle('Variation across seeds exceeds most biological-loss gains', fontsize=12)
    fig.tight_layout(rect=(0, .08, 1, .95))
    directory = ROOT / 'documents/figures'; directory.mkdir(exist_ok=True)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(directory / f'v4_biology_seed_bedroc85.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
