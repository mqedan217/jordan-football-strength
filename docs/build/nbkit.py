"""Shared builder for the five notebooks.

Each notebook is a Python script of `md(...)` and `code(...)` calls, so the
prose and the code that produces a figure sit next to each other in one file
and the whole set can be rebuilt from the command line:

    python docs/build/build_all.py

`figure(caption, src)` is the only way a plot should be added. It writes the
caption cell and the code cell as a pair, and `write()` refuses to emit a
notebook where a `plt.show()` is not preceded by a markdown cell -- the rule
that every figure is introduced by a concise description of what is plotted is
enforced here rather than remembered.
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]


class Nb:
    def __init__(self, name):
        self.name = name
        self.cells = []

    def md(self, s):
        self.cells.append(nbf.v4.new_markdown_cell(s.strip()))
        return self

    def code(self, s):
        self.cells.append(nbf.v4.new_code_cell(s.strip()))
        return self

    def figure(self, caption, src):
        """A caption and the figure it describes, always in that order."""
        self.md(caption)
        self.code(src)
        return self

    def write(self):
        n_fig = 0
        for i, c in enumerate(self.cells):
            if c.cell_type == "code" and "plt.show()" in c.source:
                n_fig += c.source.count("plt.show()")
                prev = self.cells[i - 1] if i else None
                if prev is None or prev.cell_type != "markdown":
                    raise AssertionError(
                        f"{self.name}: figure at cell {i} has no caption cell "
                        f"before it -- use figure() rather than code()")
        nb = nbf.v4.new_notebook(cells=self.cells)
        nb.metadata.kernelspec = {"display_name": "Python 3",
                                  "language": "python", "name": "python3"}
        nb.metadata.language_info = {"name": "python"}
        dest = ROOT / "notebooks" / f"{self.name}.ipynb"
        dest.write_text(nbf.writes(nb), encoding="utf-8")
        print(f"wrote {dest.name:<44} {len(self.cells):>3} cells  "
              f"{n_fig:>2} figures")
        return dest


# The preamble every notebook opens with. Kept here so the five agree on the
# window, the split date and the palette rather than drifting apart.
PRELUDE = '''
import sys, json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "JordanData"))
import viz
viz.use_style()

OUT = ROOT / "outputs"
SQUAD = OUT / "squad"
START, CUT = pd.Timestamp("2015-01-01"), pd.Timestamp(viz.SPLIT)
rng = np.random.default_rng(41)
'''.strip()

# Loading the match/goal/appearance tables on the 2015+ window, with eras and
# six-month blocks attached. Notebooks 3-5 all start from exactly this frame.
LOAD_SQUAD = '''
mt = pd.read_csv(SQUAD / "match_table.csv", parse_dates=["date"]).sort_values("date")
goals = pd.read_csv(SQUAD / "goals.csv", parse_dates=["date"])
ap = pd.read_csv(SQUAD / "appearances_combined.csv")
ja = ap[ap.squad_id == "3e22f0fa"].copy()
ja["Min"] = pd.to_numeric(ja.Min, errors="coerce")

mt = mt[mt.date >= START].copy()
goals = goals[goals.date >= START].copy()
for d in (mt, goals):
    d["era"] = np.where(d.date < CUT, "pre", "post")
# six-month blocks: matches inside a window share a squad, a manager and a set
# of opponents, so the block is the unit a bootstrap resamples (Kunsch 1989)
mt["blk"] = mt.date.dt.year * 2 + (mt.date.dt.month > 6)

# A scorer is a player id, never a name -- the same man is transliterated
# several ways across sources (notebook 01, section 6).
gj = goals.dropna(subset=["scorer_player_id"]).copy()
NAMES = ja.dropna(subset=["player_id"]).groupby("player_id").Player.agg(
    lambda s: s.mode().iloc[0])
BLOCKS = {e: [d for _, d in mt[mt.era == e].groupby("blk")] for e in ("pre", "post")}


def block_boot(fn, nb=2000, frame=None):
    """Resample six-month blocks within each era; return (observed, lo, hi)."""
    src = mt if frame is None else frame
    blocks = {e: [d for _, d in src[src.era == e].groupby("blk")]
              for e in ("pre", "post")}
    obs = fn(src[src.era == "post"]) - fn(src[src.era == "pre"])
    vals = []
    for _ in range(nb):
        s = {e: pd.concat([blocks[e][i] for i in
                           rng.integers(0, len(blocks[e]), len(blocks[e]))])
             for e in ("pre", "post")}
        v = fn(s["post"]) - fn(s["pre"])
        if np.isfinite(v):
            vals.append(v)
    return (obs, *np.percentile(vals, [2.5, 97.5]))
'''.strip()
