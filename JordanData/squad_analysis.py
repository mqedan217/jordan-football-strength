"""Does squad composition change across the strength shift?

Part 3's question, answered the way the rest of the project answers things:
with an interval, and with the reasons it might be wrong stated next to it.

Two comparisons, both on per-match features:

  1. pre/post the treatment date CLAUDE.md fixes at 2022-07-01
  2. correlation between composition and contemporaneous strength

Both bootstrap over six-month blocks rather than over matches. Consecutive
matches share players almost by definition -- that is what continuity measures
-- so treating matches as independent would give intervals several times too
narrow. Blocks are the honest unit.

What this cannot do, stated plainly:

* it is not a placebo. Whether other AFC sides show the same pattern over the
  same period is unknown until their squads are scraped too, and CLAUDE.md is
  right that without it no pattern here is demonstrably Jordan's.
* the treatment date is fixed in advance, but only one date is tested here.
* 20% of Jordan's matches since 2015 have no FBref lineup, and the missing
  ones are disproportionately Arab Cup and WAFF -- rotation-heavy fixtures.
  Turnover is therefore biased downward, and not evenly across time.
* strength and composition are both slow-moving, so a correlation between them
  is close to a correlation between two trends. It is a description of
  co-movement, never evidence of a mechanism.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SQUAD_DIR = ROOT / "outputs" / "squad"
SPLIT = pd.Timestamp("2022-07-01")
SEED = 20260909
FEATURES = ["turnover", "core_continuity", "age_mean", "hhi_minutes", "n_debuts"]


def _blocks(dates):
    """Six-month block id, so the bootstrap resamples periods not matches."""
    return dates.dt.year * 2 + (dates.dt.month > 6).astype(int)


def prepost(feat, split=SPLIT, n_boot=4000, seed=SEED):
    d = feat.dropna(subset=["date"]).copy()
    d["block"] = _blocks(d.date)
    d["post"] = d.date >= split
    ids = d.block.unique()
    groups = {b: g for b, g in d.groupby("block")}
    rng = np.random.default_rng(seed)

    rows = []
    for col in FEATURES:
        sub = d.dropna(subset=[col])
        obs = (sub.loc[sub.post, col].mean() - sub.loc[~sub.post, col].mean())
        draws = np.empty(n_boot)
        for i in range(n_boot):
            take = rng.choice(ids, len(ids), replace=True)
            s = pd.concat([groups[b] for b in take], ignore_index=True).dropna(subset=[col])
            a, b_ = s.loc[s.post, col], s.loc[~s.post, col]
            draws[i] = a.mean() - b_.mean() if len(a) and len(b_) else np.nan
        ok = draws[~np.isnan(draws)]
        lo, hi = np.percentile(ok, [2.5, 97.5])
        # Standardised by the pre-period spread. These measures are in
        # incommensurable units -- years, an HHI, a 0-1 share -- so a raw
        # difference cannot be compared across rows or plotted on one axis
        # without the largest unit swamping the rest.
        sd_pre = sub.loc[~sub.post, col].std(ddof=1)
        scale = sd_pre if sd_pre and np.isfinite(sd_pre) else np.nan
        rows.append({
            "feature": col,
            "pre": sub.loc[~sub.post, col].mean(),
            "post": sub.loc[sub.post, col].mean(),
            "diff": obs, "lo": lo, "hi": hi,
            "sd_pre": sd_pre,
            "diff_sd": obs / scale, "lo_sd": lo / scale, "hi_sd": hi / scale,
            "crosses_zero": bool(lo * hi <= 0),
            "n_pre": int((~sub.post).sum()), "n_post": int(sub.post.sum()),
        })
    return pd.DataFrame(rows)


def correlate(aligned, n_boot=4000, seed=SEED, block=4):
    """Spearman correlation with strength, moving-block bootstrapped.

    Both series are slow-moving and serially dependent: strength comes from
    overlapping ten-year fit windows, and continuity is serial by construction.
    Resampling the 28 points independently would treat each as fresh evidence
    and give intervals several times too narrow. A moving-block bootstrap
    (blocks of `block` consecutive periods, here two years) keeps the local
    dependence intact.

    Even so these intervals are optimistic. Two trending series correlate
    because they trend; this describes co-movement, not mechanism.
    """
    d = aligned.dropna(subset=["strength"]).copy()
    rng = np.random.default_rng(seed)
    rows = []
    for col in FEATURES:
        s = d.dropna(subset=[col]).reset_index(drop=True)
        if len(s) < 6:
            continue
        obs = s[col].corr(s.strength, method="spearman")
        n = len(s)
        n_blocks = int(np.ceil(n / block))
        starts = np.arange(max(n - block + 1, 1))
        draws = np.empty(n_boot)
        for i in range(n_boot):
            take = rng.choice(starts, n_blocks, replace=True)
            idx = np.concatenate([np.arange(t, min(t + block, n)) for t in take])[:n]
            t_ = s.iloc[idx]
            draws[i] = t_[col].corr(t_.strength, method="spearman")
        ok = draws[~np.isnan(draws)]
        lo, hi = np.percentile(ok, [2.5, 97.5])
        rows.append({"feature": col, "spearman": obs, "lo": lo, "hi": hi,
                     "crosses_zero": bool(lo * hi <= 0), "n": len(s)})
    return pd.DataFrame(rows)


def build():
    feat = pd.read_csv(SQUAD_DIR / "features_per_match.csv", parse_dates=["date"])
    aligned = pd.read_csv(SQUAD_DIR / "features_aligned.csv", parse_dates=["date"])

    pp = prepost(feat)
    pp.to_csv(SQUAD_DIR / "analysis_prepost.csv", index=False)
    print("=== composition before vs after 2022-07-01 (per match) ===")
    print(pp.round(3).to_string(index=False))

    cc = correlate(aligned)
    cc.to_csv(SQUAD_DIR / "analysis_correlation.csv", index=False)
    print("\n=== correlation with contemporaneous strength (6-month points) ===")
    print(cc.round(3).to_string(index=False))
    return pp, cc


if __name__ == "__main__":
    build()
