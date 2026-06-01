
# ── Export tables as high-resolution PNGs for Word ───────────────────────
# Paste this as a new cell in any of your notebooks and run it.
# Saves two PNGs to results/analysis/ — ready to Insert → Picture in Word.

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

ANALYSIS_DIR = '/content/drive/MyDrive/Consultant/Colab_Notebooks/Obrown_Dissertation_NU_25/OBrown_DIS9300_v2/results/analysis'

# ── Colour maps ───────────────────────────────────────────────────────────
EXP_COLOURS = {
    '243-class':   '#b8d4ee',
    '1683-class':  '#b5ddb5',
    'Signer-ctrl': '#f7d199',
    'S11-only':    '#f2b8b8',
}

SPEED_COLOURS = {   # keyed on Mean (ms) tier
    'red':    '#f5bcbc',   # >= 500
    'amber':  '#f7d199',   # 50–500
    'yellow': '#fff3b0',   # 10–50
    'green':  '#b5ddb5',   # < 10
}

HEADER_BG  = '#1a252f'
HEADER_FG  = '#ffffff'
BEST_FG    = '#0a4a0a'
TEXT_FG    = '#111111'
DPI        = 180          # high-res — crisp when pasted into Word


# ═══════════════════════════════════════════════════════════════════════════
# Helper: draw a table from a DataFrame
# ═══════════════════════════════════════════════════════════════════════════
def render_table(df, row_colours, best_cols_min=None, best_cols_max=None,
                 title='', col_widths=None, fontsize=7.5):
    """
    Render df as a matplotlib table image.
    row_colours : list of hex colour strings, one per row.
    best_cols_min : column names where the LOWEST value gets bold green.
    best_cols_max : column names where the HIGHEST value gets bold green.
    """
    best_cols_min = best_cols_min or []
    best_cols_max = best_cols_max or []

    n_rows, n_cols = df.shape
    fig_w = sum(col_widths) if col_widths else n_cols * 1.2
    fig_h = (n_rows + 1) * 0.32 + 0.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=fontsize + 1.5, fontweight='bold',
                     color=TEXT_FG, x=0.01, ha='left', y=0.99)

    col_w = col_widths if col_widths else [1.2] * n_cols
    col_w_norm = [w / sum(col_w) for w in col_w]

    # ── find best values per column ───────────────────────────────────────
    best_vals = {}
    for col in best_cols_min:
        if col in df.columns:
            try:
                vals = pd.to_numeric(df[col].astype(str).str.replace('%',''), errors='coerce')
                best_vals[col] = ('min', vals.min())
            except: pass
    for col in best_cols_max:
        if col in df.columns:
            try:
                vals = pd.to_numeric(df[col].astype(str).str.replace('%',''), errors='coerce')
                best_vals[col] = ('max', vals.max())
            except: pass

    row_height = 1 / (n_rows + 1)
    x_positions = []
    x = 0
    for w in col_w_norm:
        x_positions.append(x)
        x += w

    def draw_cell(ax, text, x, y, w, h, bg, fg, bold=False, fontsize=fontsize, align='left'):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle='square,pad=0',
            facecolor=bg, edgecolor='#cccccc', linewidth=0.4,
            transform=ax.transAxes, clip_on=False))
        ha = 'left' if align == 'left' else 'center'
        ax.text(x + 0.005, y + h / 2, str(text),
                transform=ax.transAxes,
                va='center', ha=ha,
                fontsize=fontsize,
                fontweight='bold' if bold else 'normal',
                color=fg, clip_on=False)

    # ── header row ────────────────────────────────────────────────────────
    y_header = 1 - row_height
    for ci, col in enumerate(df.columns):
        draw_cell(ax, col,
                  x_positions[ci], y_header, col_w_norm[ci], row_height,
                  bg=HEADER_BG, fg=HEADER_FG, bold=True)

    # ── data rows ─────────────────────────────────────────────────────────
    for ri, (_, row) in enumerate(df.iterrows()):
        y_row = 1 - row_height * (ri + 2)
        bg = row_colours[ri] if ri < len(row_colours) else '#ffffff'
        for ci, col in enumerate(df.columns):
            val = row[col]
            val_str = str(val) if pd.notna(val) else '—'

            # check if this cell is best
            is_best = False
            if col in best_vals:
                direction, bv = best_vals[col]
                try:
                    num = float(str(val).replace('%',''))
                    if direction == 'min' and abs(num - bv) < 1e-6:
                        is_best = True
                    elif direction == 'max' and abs(num - bv) < 1e-6:
                        is_best = True
                except: pass

            draw_cell(ax, val_str,
                      x_positions[ci], y_row, col_w_norm[ci], row_height,
                      bg=bg, fg=BEST_FG if is_best else TEXT_FG,
                      bold=is_best)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# TABLE 1 — Precision / Recall / F1 / Accuracy
# ═══════════════════════════════════════════════════════════════════════════
metrics_path = os.path.join(ANALYSIS_DIR, 'full_metrics_percentages.csv')
df_m = pd.read_csv(metrics_path)

# Format columns
pct_cols = ['Accuracy (%)', 'Precision_macro (%)', 'Recall_macro (%)',
            'F1_macro (%)', 'F1_weighted (%)', 'Top5_Acc (%)']
for c in pct_cols:
    if c in df_m.columns:
        df_m[c] = df_m[c].apply(lambda x: f'{x:.2f}%' if pd.notna(x) else '—')

# Rename for display
df_m = df_m.rename(columns={
    'Accuracy (%)':       'Accuracy',
    'Precision_macro (%)':'Precision\n(macro)',
    'Recall_macro (%)':   'Recall\n(macro)',
    'F1_macro (%)':       'F1\n(macro)',
    'F1_weighted (%)':    'F1\n(weighted)',
    'Top5_Acc (%)':       'Top-5\nAcc',
})

row_colours_m = [EXP_COLOURS.get(e, '#ffffff') for e in df_m['Experiment']]
metric_display_cols = ['Accuracy','Precision\n(macro)','Recall\n(macro)',
                       'F1\n(macro)','F1\n(weighted)','Top-5\nAcc']

fig1 = render_table(
    df_m,
    row_colours=row_colours_m,
    best_cols_max=metric_display_cols,
    title='Table 1 — Model Performance: Accuracy, Precision, Recall & F1  (macro-averaged)',
    col_widths=[1.1, 1.5, 0.6, 0.9, 0.9, 0.75, 0.75, 0.9, 0.75, 2.0],
    fontsize=7,
)

out1 = os.path.join(ANALYSIS_DIR, 'table1_metrics.png')
fig1.savefig(out1, dpi=DPI, bbox_inches='tight',
             facecolor='white', edgecolor='none')
plt.close(fig1)
print(f'✅ Table 1 saved → {out1}')


# ═══════════════════════════════════════════════════════════════════════════
# TABLE 2 — Inference Speed  (slowest → fastest)
# ═══════════════════════════════════════════════════════════════════════════
speed_path = os.path.join(ANALYSIS_DIR, 'inference_speed_full.csv')
df_s = pd.read_csv(speed_path)

# ── Label correction: S11-only CIF used full 68 channels, no reduction ───
# The benchmark registry logged '25 trees · 68 ch · No PCA · augmented train'
# but nb7d confirms the model ran on the full (1043, 68, 40) tensor with no
# channel subsetting.  '68 ch' described the input shape, not a reduction step.
# Corrected to match what the notebook actually did.
mask_s11_cif = (
    (df_s['Experiment'] == 'S11-only') &
    (df_s['Model'] == 'CIF (25 trees)')
)
df_s.loc[mask_s11_cif, 'Special conditions'] = '25 trees · No PCA · augmented train'

df_s = df_s.sort_values('Mean (ms)', ascending=False, na_position='last').reset_index(drop=True)

def speed_colour(mean_ms):
    try:
        v = float(mean_ms)
        if v >= 500: return SPEED_COLOURS['red']
        if v >= 50:  return SPEED_COLOURS['amber']
        if v >= 10:  return SPEED_COLOURS['yellow']
        return SPEED_COLOURS['green']
    except: return '#e0e0e0'

row_colours_s = [speed_colour(r) for r in df_s['Mean (ms)']]

# Round numeric cols for display
for c in ['Mean (ms)','Median (ms)','Std (ms)','Min (ms)','Max (ms)','P95 (ms)','Throughput (FPS)','Model size (MB)']:
    if c in df_s.columns:
        df_s[c] = df_s[c].apply(lambda x: f'{x:.1f}' if pd.notna(x) else '—')

fig2 = render_table(
    df_s,
    row_colours=row_colours_s,
    best_cols_min=['Mean (ms)', 'Median (ms)', 'P95 (ms)'],
    best_cols_max=['Throughput (FPS)'],
    title='Table 2 — CPU Inference Speed (single sample, 100 trials)  —  sorted slowest → fastest',
    col_widths=[1.1, 1.5, 0.55, 0.85, 2.0, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 1.0],
    fontsize=6.5,
)

out2 = os.path.join(ANALYSIS_DIR, 'table2_inference_speed.png')
fig2.savefig(out2, dpi=DPI, bbox_inches='tight',
             facecolor='white', edgecolor='none')
plt.close(fig2)
print(f'✅ Table 2 saved → {out2}')


# ═══════════════════════════════════════════════════════════════════════════
# Legend reminder
# ═══════════════════════════════════════════════════════════════════════════
print()
print('Table 1 row colours:')
for exp, c in EXP_COLOURS.items():
    print(f'  {c}  →  {exp}')
print()
print('Table 2 row colours (speed tiers):')
print('  #b5ddb5 green  — fast      < 10 ms')
print('  #fff3b0 yellow — moderate  10–50 ms')
print('  #f7d199 amber  — slow      50–500 ms')
print('  #f5bcbc red    — very slow >= 500 ms')
print()
print('Bold dark-green = best value in that column.')
print()
print('In Word:  Insert → Pictures → select the PNG → OK')
print('Tip: right-click the image → Wrap Text → In Line with Text')
print('     then resize to fit your page width.')
