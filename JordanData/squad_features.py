"""Squad composition measures, per match and aligned to the strength trace.

Everything here is built from lineups and minutes only -- no advanced stats.
That is deliberate: FBref's provider changed from StatsBomb to Opta near the
2022 treatment date, so any rate statistic is unattributable across the
boundary, while who started and how long they played are structural facts that
a provider change cannot move.

Player identity is the 8-hex FBref id, never the name. Arabic names get
re-transliterated between seasons and name-keying would invent debuts and
corrupt every continuity measure.

Per-match measures:

  turnover          1 - Jaccard(XI_t, XI_{t-1}); 0 = unchanged side, 1 = all new
  core_continuity   share of today's XI that also started k matches ago
  age_mean/age_sd   starting XI age in years
  n_debuts          starters with no prior appearance in the record
  hhi_minutes       Herfindahl index of minutes across a trailing window;
                    high = minutes concentrated in a settled group
  n_players_used    distinct players appearing in the trailing window

Aligned measures use the trailing six months before each strength reference
date -- the same no-lookahead convention the trace itself uses, so a row
describes only what was knowable on the day.

Caveat carried from the coverage audit: matches FBref does not hold (Arab Cup,
WAFF and friends) are invisible here. Those are competitive, often
rotation-heavy fixtures, so turnover measured from this data is biased
downward. `n_matches_missing` on the aligned frame keeps that visible.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SQUAD_DIR = ROOT / "outputs" / "squad"
TEAM = "Jordan"
SQUAD_ID = "3e22f0fa"
CONTINUITY_K = 5
WINDOW = 10


def _load():
    matches = pd.read_csv(SQUAD_DIR / "matches.csv")
    lineups = pd.read_csv(SQUAD_DIR / "lineups.csv")
    apps = pd.read_csv(SQUAD_DIR / "appearances.csv")
    matches["date"] = pd.to_datetime(matches["Date"], errors="coerce")
    return matches, lineups, apps


def per_match(matches, lineups, apps):
    dates = matches.set_index("match_id").date.to_dict()
    comp = matches.set_index("match_id").Comp.to_dict() if "Comp" in matches else {}

    mine = lineups[lineups.team == TEAM].copy()
    mine["date"] = mine.match_id.map(dates)
    mine = mine[mine.date.notna()].sort_values("date")

    ap = apps[apps.squad_id == SQUAD_ID].copy()
    ap["date"] = ap.match_id.map(dates)
    ap = ap[ap.date.notna()]
    ap["minutes"] = pd.to_numeric(ap.get("Min"), errors="coerce")
    ap["age"] = (pd.to_numeric(ap.get("age_years"), errors="coerce")
                 + pd.to_numeric(ap.get("age_days"), errors="coerce") / 365.25)

    order = mine[["match_id", "date"]].drop_duplicates().sort_values("date")
    xi = {m: set(g.loc[g.role == "starter", "player_id"].dropna())
          for m, g in mine.groupby("match_id")}

    seen = set()
    rows, history = [], []
    for _, r in order.iterrows():
        mid, date = r.match_id, r.date
        today = xi.get(mid, set())
        prev = xi.get(history[-1], set()) if history else set()
        back = xi.get(history[-CONTINUITY_K], set()) if len(history) >= CONTINUITY_K else set()

        union = today | prev
        turnover = 1 - (len(today & prev) / len(union)) if union else np.nan
        continuity = (len(today & back) / len(today)) if today and back else np.nan

        ages = ap.loc[(ap.match_id == mid) & ap.player_id.isin(today), "age"].dropna()
        debuts = len(today - seen) if today else np.nan

        win = history[-WINDOW:] + [mid]
        wap = ap[ap.match_id.isin(win)]
        mins = wap.groupby("player_id").minutes.sum().dropna()
        share = mins / mins.sum() if mins.sum() else mins
        hhi = float((share ** 2).sum()) if len(share) else np.nan

        rows.append({
            "match_id": mid, "date": date, "comp": comp.get(mid),
            "n_starters": len(today), "turnover": turnover,
            "core_continuity": continuity, "n_debuts": debuts,
            "age_mean": ages.mean() if len(ages) else np.nan,
            "age_sd": ages.std(ddof=1) if len(ages) > 1 else np.nan,
            "hhi_minutes": hhi, "n_players_used": int(len(mins)),
        })
        seen |= today
        history.append(mid)
    return pd.DataFrame(rows)


def align_to_trace(feat, trace_path=None, spine_path=None):
    """Summarise the six months before each strength reference date."""
    trace_path = trace_path or ROOT / "outputs" / "jordan_strength_ci.csv"
    trace = pd.read_csv(trace_path, parse_dates=["date"])

    spine = None
    if spine_path is None:
        spine_path = ROOT / "JordanData" / "jordan_matches.csv"
    if Path(spine_path).exists():
        spine = pd.read_csv(spine_path, parse_dates=["date"])

    out = []
    for _, t in trace.iterrows():
        lo = t.date - pd.DateOffset(months=6)
        w = feat[(feat.date >= lo) & (feat.date < t.date)]
        row = {"date": t.date, "strength": t.strength,
               "strength_lo": t.get("strength_lo"), "strength_hi": t.get("strength_hi"),
               "n_matches_fbref": len(w)}
        if spine is not None:
            n_spine = ((spine.date >= lo) & (spine.date < t.date)).sum()
            row["n_matches_spine"] = int(n_spine)
            row["n_matches_missing"] = int(n_spine) - len(w)
        for col in ["turnover", "core_continuity", "age_mean", "age_sd",
                    "hhi_minutes", "n_players_used"]:
            row[col] = w[col].mean() if len(w) else np.nan
        row["n_debuts"] = w.n_debuts.sum() if len(w) else np.nan
        out.append(row)
    return pd.DataFrame(out)


def build():
    matches, lineups, apps = _load()
    feat = per_match(matches, lineups, apps)
    feat.to_csv(SQUAD_DIR / "features_per_match.csv", index=False)
    print(f"features_per_match.csv  {len(feat)} matches")

    aligned = align_to_trace(feat)
    aligned.to_csv(SQUAD_DIR / "features_aligned.csv", index=False)
    print(f"features_aligned.csv    {len(aligned)} reference dates")
    return feat, aligned


if __name__ == "__main__":
    f, a = build()
    cols = ["date", "n_matches_fbref", "n_matches_missing", "turnover",
            "core_continuity", "age_mean", "hhi_minutes", "n_debuts", "strength"]
    print("\n" + a[[c for c in cols if c in a.columns]].round(3).to_string(index=False))
