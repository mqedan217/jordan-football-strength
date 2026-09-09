"""Load international match results and build the modelling frame."""

from pathlib import Path
import numpy as np
import pandas as pd

RAW = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def load_results(path="data/results.csv", refresh=False):
    p = Path(__file__).resolve().parents[1] / path
    if refresh or not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(RAW).to_csv(p, index=False)
    df = pd.read_csv(p, parse_dates=["date"])
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    return df.sort_values("date").reset_index(drop=True)


def index_teams(df):
    teams = sorted(set(df.home_team) | set(df.away_team))
    idx = {t: i for i, t in enumerate(teams)}
    return teams, idx


def team_view(df, team="Jordan"):
    """One row per match from `team`'s perspective."""
    m = df[(df.home_team == team) | (df.away_team == team)].copy()
    at_home = m.home_team == team
    m["gf"] = np.where(at_home, m.home_score, m.away_score)
    m["ga"] = np.where(at_home, m.away_score, m.home_score)
    m["opponent"] = np.where(at_home, m.away_team, m.home_team)
    m["venue"] = np.where(m.neutral, "neutral", np.where(at_home, "home", "away"))
    m["gd"] = m.gf - m.ga
    m["result"] = np.select([m.gd > 0, m.gd == 0], ["W", "D"], "L")
    return m.reset_index(drop=True)
