"""
Draw the paper's two figures at IEEE single-column width:
  Fig. 1  fig1_bootstrap.png  bootstrap intervals for the three metrics
  Fig. 2  fig2_latency.png    latency against batch size, on-device and cloud

Series differ in line style and marker as well as color, so the figures still
read when printed in grayscale. Colors come from the Okabe-Ito palette, and
check_cvd() measures how well they separate under simulated deuteranopia and
protanopia.

The plotted numbers are typed in by hand from benchmark/*_results.json,
cloud/loopback_results.json, the round-trip times in cloud/compose_estimate.py
and a full run of model/bootstrap_significance.py.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COL_W = 3.45          # IEEE single column, inches
DPI = 400

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.2,
})

# Okabe-Ito subset; its worst pair separates by dE 12.6. Sets that included
# yellow scored higher, but yellow lines are too faint on a white background.
BLACK, GREEN, SKY, ORANGE, PURPLE = ('#000000', '#009E73', '#56B4E9',
                                     '#E69F00', '#CC79A7')
GRAY = '#444444'

BATCH = [1, 32, 256, 1024]
MLX = [0.2425, 0.2213, 0.2218, 0.2192]
COREML = [0.0675, 0.0743, 0.0871, 0.1027]
LOOPBACK = [0.450, 0.352, 0.521, 1.190]
RTT = {'Cloud: fiber, idle': 7.0, 'Cloud: 5G mobile edge': 44.62,
       'Cloud: 5G distant region': 78.49}


def srgb_to_lin(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def hex_to_rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


def to_oklab(rgb):
    r, g, b = srgb_to_lin(rgb)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.array([0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
                     1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
                     0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_])


def simulate_cvd(rgb, kind):
    # Brettel/Vienot-style approximation in linear RGB
    r, g, b = srgb_to_lin(rgb)
    if kind == 'deuter':
        m = np.array([[0.625, 0.375, 0.0], [0.7, 0.3, 0.0], [0.0, 0.3, 0.7]])
    else:  # protan
        m = np.array([[0.567, 0.433, 0.0], [0.558, 0.442, 0.0], [0.0, 0.242, 0.758]])
    out = m @ np.array([r, g, b])
    out = np.clip(out, 0, 1)
    srgb = np.where(out <= 0.0031308, out * 12.92, 1.055 * out ** (1 / 2.4) - 0.055)
    return srgb * 255


def check_cvd(colors, label):
    print(f'  CVD separation check ({label}):')
    worst = 999
    for kind in ('normal', 'deuter', 'protan'):
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                a, b = hex_to_rgb(colors[i]), hex_to_rgb(colors[j])
                if kind != 'normal':
                    a, b = simulate_cvd(a, kind), simulate_cvd(b, kind)
                d = np.linalg.norm(to_oklab(a) - to_oklab(b)) * 100
                worst = min(worst, d)
    print(f'    worst pair OKLab dE across normal/deuter/protan: {worst:.1f}')
    print(f'    {"PASS" if worst >= 8 else "FAIL"} (target dE >= 8)')
    return worst


def fig_latency():
    fig, ax = plt.subplots(figsize=(COL_W, 2.35))
    x = np.arange(len(BATCH))

    series = [
        ('On-device: Core ML', COREML, BLACK, '-', 'o'),
        ('On-device: MLX', MLX, GREEN, '-', 's'),
    ]
    cloud = [
        ('Cloud: fiber, idle', [l + RTT['Cloud: fiber, idle'] for l in LOOPBACK],
         SKY, '--', '^'),
        ('Cloud: 5G mobile edge', [l + RTT['Cloud: 5G mobile edge'] for l in LOOPBACK],
         ORANGE, '--', 'D'),
        ('Cloud: 5G distant region', [l + RTT['Cloud: 5G distant region'] for l in LOOPBACK],
         PURPLE, '--', 'v'),
    ]

    for name, ys, c, ls, mk in series + cloud:
        ax.plot(x, ys, color=c, linestyle=ls, marker=mk, markersize=3.2,
                label=name, markeredgewidth=0)

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in BATCH])
    ax.set_xlabel('Batch size (predictions per call)')
    ax.set_ylabel('Latency (ms, log scale)')
    ax.grid(True, which='major', axis='y', color='#cccccc', linestyle=':')
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc='center', bbox_to_anchor=(0.58, 0.47),
              handlelength=2.2, labelspacing=0.25)

    # label the gap at batch 1
    ax.annotate('', xy=(0, COREML[0]), xytext=(0, LOOPBACK[0] + RTT['Cloud: 5G distant region']),
                arrowprops=dict(arrowstyle='<->', color=GRAY, lw=0.7))
    ax.text(0.06, 2.4, '1169x', fontsize=6, color=GRAY, rotation=90,
            va='center', ha='left')

    fig.tight_layout(pad=0.3)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(HERE, f'fig2_latency.{ext}'), dpi=DPI,
                    bbox_inches='tight')
    plt.close(fig)
    print('  wrote fig2_latency.png/pdf')


def fig_bootstrap():
    # the three metrics differ in scale, so each gets its own panel and axis
    metrics = [
        ('MAE (lower is better)', 0.0018, 0.0017, 0.0019, True),
        ('CORh (higher is better)', 0.0165, 0.0134, 0.0195, True),
        ('AUC (higher is better)', -0.0012, -0.0014, -0.0010, False),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(COL_W, 2.5))
    for ax, (name, pt, lo, hi, favorable) in zip(axes, metrics):
        color = BLACK if favorable else ORANGE
        ax.axvline(0, color=GRAY, linestyle='--', linewidth=0.8, zorder=1)
        ax.plot([lo, hi], [0, 0], color=color, linewidth=1.6,
                solid_capstyle='round', zorder=2)
        ax.plot([pt], [0], marker='o', markersize=4.5, color=color,
                markeredgecolor='white', markeredgewidth=0.8, zorder=3)
        span = max(abs(hi), abs(lo)) * 1.9
        ax.set_xlim(-span, span)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_ylabel('')
        ax.text(0.015, 0.80, name, transform=ax.transAxes, fontsize=6.5,
                va='top', ha='left')
        ax.text(0.985, 0.80, f'{pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]',
                transform=ax.transAxes, fontsize=6, va='top', ha='right',
                color=GRAY)
        ax.tick_params(axis='x', length=2, pad=1.5)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.6)
    axes[-1].set_xlabel('Neural minus linear (95% bootstrap CI, resampled by learner)')
    fig.tight_layout(pad=0.25, h_pad=0.6)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(HERE, f'fig1_bootstrap.{ext}'), dpi=DPI,
                    bbox_inches='tight')
    plt.close(fig)
    print('  wrote fig1_bootstrap.png/pdf')


if __name__ == '__main__':
    check_cvd([BLACK, GREEN, SKY, ORANGE, PURPLE], 'Fig. 2, five series')
    check_cvd([BLACK, ORANGE], 'Fig. 1, two states')
    fig_latency()
    fig_bootstrap()
