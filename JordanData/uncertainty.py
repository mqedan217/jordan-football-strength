"""Bootstrap intervals for the strength trace.

The trace in `trajectory.py` is a sequence of point estimates with nothing
around them. With ~15 Jordan matches a year, a point estimate on its own
cannot support any of the claims the project wants to make -- least of all a
difference between two periods. This module supplies the intervals.

Resampling scheme: a draw resamples *matches* (with replacement) from the full
49k-row dataset once, and then runs the entire trace on that resampled dataset.
Resampling once per draw rather than once per reference date is what makes the
draws paired across dates: draw b gives a whole trajectory, so a difference
between two dates can be read off draw by draw and gets an interval of its own.
Resampling independently at each date would destroy that.

Nonparametric rather than parametric on purpose. Simulating goals from the
fitted Poisson would hold the fixture list fixed and propagate only scoring
noise; resampling matches also propagates "which matches happened", which for a
team playing 15 a year is the larger source of uncertainty.

Percentile intervals throughout. No bias correction -- with 500 draws the
BCa acceleration term is noisier than the correction is worth.
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from data import load_results, index_teams
from baseline import PoissonStrength

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260909
N_BOOT = 500


def _pack(df, idx):
    """Full dataset as flat arrays, so a draw is just fancy indexing."""
    return {
        "h": df.home_team.map(idx).to_numpy(),
        "a": df.away_team.map(idx).to_numpy(),
        "gh": df.home_score.to_numpy(float),
        "ga": df.away_score.to_numpy(float),
        "flag": (~df.neutral).to_numpy(float),
        "day": df.date.to_numpy("datetime64[D]").astype(np.int64),
    }


def _fit_at(cols, order, n_teams, ti, ref_day, lookback_days, half_life, ridge):
    """Fit one reference date on the rows named by `order`. None if degenerate.

    `ti` may be a single team index or a sequence of them. Every fit already
    estimates all 337 teams, so tracking several costs nothing beyond the
    bookkeeping -- which is what makes the placebo comparison free once the
    bootstrap is running at all.
    """
    day = cols["day"][order]
    win = (day < ref_day) & (day >= ref_day - lookback_days)
    if win.sum() < 100:
        return None
    sel = order[win]
    m = PoissonStrength(n_teams, half_life_days=half_life, ridge=ridge)
    m.fit(cols["h"][sel], cols["a"][sel], cols["gh"][sel], cols["ga"][sel],
          cols["flag"][sel], (ref_day - cols["day"][sel]).astype(float))
    idxs = np.atleast_1d(ti)
    return (m.atk[idxs], m.dfn[idxs],
            np.array([m.strength(i) for i in idxs]),
            np.full(len(idxs), m.adv), int(win.sum()))


def bootstrap(team="Jordan", start="2012-01-01", end="2026-07-01", freq="6MS",
              half_life=730.0, ridge=0.05, lookback_years=10, n_boot=N_BOOT,
              seed=SEED):
    """`team` may be one name or a list; extra teams are free (see _fit_at)."""
    df = load_results()
    teams, idx = index_teams(df)
    names = [team] if isinstance(team, str) else list(team)
    ti = [idx[t] for t in names]
    cols = _pack(df, idx)
    n_rows = len(df)
    lookback_days = 365 * lookback_years

    refs = pd.date_range(start, end, freq=freq)
    ref_days = refs.to_numpy("datetime64[D]").astype(np.int64)

    identity = np.arange(n_rows)
    fitted = [_fit_at(cols, identity, len(teams), ti, rd, lookback_days,
                      half_life, ridge) for rd in ref_days]
    # (date, param, team)
    point = np.array([np.stack(f[:4]) for f in fitted])
    n_win = np.array([f[4] for f in fitted])

    rng = np.random.default_rng(seed)
    draws = np.full((n_boot, len(refs), 4, len(ti)), np.nan)
    t0 = time.perf_counter()
    for b in range(n_boot):
        order = np.sort(rng.integers(0, n_rows, n_rows))
        for j, rd in enumerate(ref_days):
            got = _fit_at(cols, order, len(teams), ti, rd, lookback_days,
                          half_life, ridge)
            if got is not None:
                draws[b, j] = np.stack(got[:4])
        if (b + 1) % 25 == 0:
            el = time.perf_counter() - t0
            print(f"  draw {b + 1}/{n_boot}  {el:.0f}s elapsed  "
                  f"{el / (b + 1) * (n_boot - b - 1):.0f}s left", flush=True)

    return refs, n_win, point, draws, names


def summarise(refs, n_win, point, draws, names, lo=2.5, hi=97.5):
    """One frame per team, concatenated with a `team` column."""
    params = ["attack", "defence", "strength", "home_adv"]
    frames = []
    for t, team in enumerate(names):
        out = pd.DataFrame({"team": team, "date": refs, "n_matches": n_win})
        for k, name in enumerate(params):
            d = draws[:, :, k, t]
            out[name] = point[:, k, t]
            out[f"{name}_lo"] = np.nanpercentile(d, lo, axis=0)
            out[f"{name}_hi"] = np.nanpercentile(d, hi, axis=0)
            out[f"{name}_se"] = np.nanstd(d, axis=0, ddof=1)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


PLACEBO = ["Jordan", "Iraq", "Syria", "United Arab Emirates"]

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_boot = int(args[0]) if args else N_BOOT
    names = PLACEBO if "--placebo" in sys.argv else ["Jordan"]

    refs, n_win, point, draws, names = bootstrap(team=names, n_boot=n_boot)
    (ROOT / "outputs").mkdir(exist_ok=True)
    summary = summarise(refs, n_win, point, draws, names)

    jordan = summary[summary.team == "Jordan"].drop(columns="team")
    jordan.to_csv(ROOT / "outputs/jordan_strength_ci.csv", index=False)
    if len(names) > 1:
        summary.to_csv(ROOT / "outputs/strength_ci_multi.csv", index=False)

    np.savez_compressed(
        ROOT / "outputs/bootstrap_draws.npz",
        dates=refs.to_numpy("datetime64[D]").astype(np.int64),
        point=point, draws=draws, seed=SEED, n_boot=n_boot,
        teams=np.array(names),
        params=np.array(["attack", "defence", "strength", "home_adv"]))
    print(f"wrote jordan_strength_ci.csv"
          + (", strength_ci_multi.csv" if len(names) > 1 else "")
          + ", bootstrap_draws.npz")
