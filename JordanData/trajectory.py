"""Refit the baseline at regular reference dates to trace a team's strength.

Each fit uses only matches strictly before the reference date, so nothing here
peeks at the future. That is the whole point: the curve has to be something you
could have drawn on the day.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from data import load_results, index_teams
from baseline import PoissonStrength
from tournament import design, VARIANTS


def build_arrays(df, idx, ref_date, lookback_years=10):
    win = df[(df.date < ref_date) &
             (df.date >= ref_date - pd.Timedelta(days=365 * lookback_years))]
    h = win.home_team.map(idx).to_numpy()
    a = win.away_team.map(idx).to_numpy()
    gh = win.home_score.to_numpy(float)
    ga = win.away_score.to_numpy(float)
    flag = (~win.neutral).to_numpy(float)
    days = (ref_date - win.date).dt.days.to_numpy(float)
    return h, a, gh, ga, flag, days, len(win), win


def trace(team="Jordan", start="2012-01-01", end="2026-07-01", freq="6MS",
          half_life=730.0, ridge=0.05, variant="M0"):
    """`variant` selects a tournament-effect design; see tournament.VARIANTS.

    M0 is the published model. The others exist so the trace can be checked for
    sensitivity to match importance, not because they predict better -- they do
    not (outputs/tournament_comparison.csv).
    """
    df = load_results()
    teams, idx = index_teams(df)
    ti = idx[team]
    rows = []
    for ref in pd.date_range(start, end, freq=freq):
        h, a, gh, ga, flag, days, n, win = build_arrays(df, idx, ref)
        shared, home_only, comp = design(win, variant)
        m = PoissonStrength(len(teams), half_life_days=half_life, ridge=ridge,
                            **VARIANTS[variant][1])
        m.fit(h, a, gh, ga, flag, days, shared=shared, home_only=home_only,
              competitive=comp)
        rows.append({
            "date": ref,
            "n_matches": n,
            "attack": m.atk[ti],
            "defence": m.dfn[ti],
            "importance": m.imp[ti],
            "strength": m.strength(ti),
            "home_adv": m.adv,
            "converged": m.converged,
        })
        print(f"{ref.date()}  n={n:6d}  atk={m.atk[ti]:+.3f}  "
              f"def={m.dfn[ti]:+.3f}  strength={m.strength(ti):+.3f}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = trace()
    out.to_csv(Path(__file__).resolve().parents[1] / "outputs/jordan_strength_trace.csv", index=False)
