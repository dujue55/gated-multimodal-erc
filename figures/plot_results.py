"""Draw figures/results.png from the results/ CSV files (copied from the notebook output).

Usage: python figures/plot_results.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folds = pd.read_csv(os.path.join(ROOT, 'results', 'macro_f1_per_fold.csv'))
summary = pd.read_csv(os.path.join(ROOT, 'results', 'summary.csv'), index_col='setting')

INK, MUTED, GRID = '#1f1f1d', '#6b6a66', '#e6e6e3'
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left', 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'axes.linewidth': 0.8,
    'xtick.color': MUTED, 'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
})

# (label, setting); None marks a gap between groups. Ablations are reported in the README table only.
ROWS = [
    ('Text only', 'Text-Only'),
    ('Speech only (E2V)', 'Speech-Only (E2V)'),
    ('Speech only (WavLM)', 'Speech-Only (WavLM)'),
    None,
    ('Fixed 0.5 fusion (E2V)', 'Fixed Fusion (E2V)'),
    ('Gated fusion (E2V)', 'Gated Fusion (E2V)'),
    ('Gated fusion (WavLM)', 'Gated Fusion (WavLM)'),
]
LIGHT_GREY = '#b4b3ae'


def colour(setting):
    if setting.startswith('Gated'):
        return BLUE
    return ORANGE if setting.startswith('Fixed') else LIGHT_GREY


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.6), gridspec_kw={'width_ratios': [1.05, 1]})

# (a) Mean macro-F1 over three seeds, error bar = standard deviation across seeds
y, pos = [], 0.0
for row in ROWS:
    if row is None:
        pos += 0.6
        continue
    y.append(pos)
    pos += 1
y = -np.array(y)
rows = [r for r in ROWS if r]
means = [summary.loc[s, 'macro_f1_mean'] for _, s in rows]
stds = [summary.loc[s, 'macro_f1_std'] for _, s in rows]
best = int(np.argmax(means))
ax1.barh(y, means, height=0.68, color=[colour(s) for _, s in rows], xerr=stds,
         error_kw={'ecolor': INK, 'elinewidth': 0.9, 'capsize': 2.5})
for i, (yi, m, sd) in enumerate(zip(y, means, stds)):
    ax1.text(m + sd + 0.02, yi, f'{m:.3f}', va='center', fontsize=8.8, color=INK,
             fontweight='bold' if i == best else 'normal')
ax1.set_yticks(y)
ax1.set_yticklabels([label for label, _ in rows])
ax1.get_yticklabels()[best].set_fontweight('bold')
ax1.tick_params(axis='y', length=0)
ax1.set_xlim(0, 0.9)
ax1.set_xlabel('Test macro-F1 (mean ± std over 3 seeds)')
ax1.grid(axis='x', color=GRID, linewidth=0.6)
ax1.set_axisbelow(True)
ax1.spines['left'].set_visible(False)
ax1.set_title('(a) Fusion vs. single modality', pad=10)

# (b) Per test session: gated fusion vs. fixed fusion vs. the better single modality
sessions = sorted(folds.test_session.unique())
x = np.arange(len(sessions))
for name, label, c, marker, ls in [
    ('Gated Fusion (E2V)', 'Gated fusion', BLUE, 'o', '-'),
    ('Fixed Fusion (E2V)', 'Fixed 0.5 fusion', ORANGE, 's', '-'),
    ('Speech-Only (E2V)', 'Speech only', LIGHT_GREY, 'D', '--'),
]:
    vals = folds[folds.setting == name].groupby('test_session').macro_f1.mean().reindex(sessions).values
    ax2.plot(x, vals, color=c, marker=marker, markersize=6, linewidth=2, linestyle=ls, label=label)
ax2.set_xticks(x)
ax2.set_xticklabels([s.replace('Session', 'Session ') for s in sessions])
ax2.set_xlim(-0.3, len(x) - 0.7)
ax2.set_ylabel('Test macro-F1')
ax2.grid(axis='y', color=GRID, linewidth=0.6)
ax2.set_axisbelow(True)
ax2.set_title('(b) Per test session (E2V speech features)', pad=10)
ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.34), ncol=3, fontsize=8.5, handlelength=2.2)

fig.tight_layout(w_pad=3)
fig.savefig(os.path.join(ROOT, 'figures', 'results.png'))
