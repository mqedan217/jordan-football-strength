"""Shared plotting style, so every figure in the project reads as one system.

Colours are the reference data-viz palette, used only within its documented
pre-validated subset: categorical slots 1-3 clear the all-pairs colour-vision
gates in both modes, and of those, blue and orange are the two that also clear
3:1 contrast on a light surface. Notebook figures render on white, so those two
are what charts here use. Anything needing more series gets faceted instead of
reaching for a fourth hue.

Text never wears a series colour -- values and labels stay in ink tokens and a
coloured mark beside them carries identity.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e1"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
SERIES = [BLUE, ORANGE]

# sequential blue ramp, for magnitude
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

SPLIT = "2022-07-01"


def use_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 120,
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.constrained_layout.use": True,
    })


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.grid(axis="x", visible=False)


def mark_split(ax, when=SPLIT, label="mid-2022"):
    """The treatment date, drawn as context rather than as a series."""
    import pandas as pd
    x = pd.Timestamp(when)
    ax.axvline(x, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=0)
    ax.annotate(label, xy=(x, 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, -10), textcoords="offset points",
                fontsize=8, color=INK_MUTED, ha="left", va="top")
