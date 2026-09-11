"""Flatten the squad data into two analysis-ready tables.

    match_table.csv   one row per Jordan match
    goals.csv         one row per Jordan goal

Two grains because two questions. Match-level measures -- formation, who
managed, how many substitutes and when, the age and turnover of the XI -- belong
on the match. "Where did the extra goals come from" is a question about goals,
and squeezing it onto a match row would throw away the scorer.

Both are built from `*_combined.csv`, so the ten Wikipedia-sourced Arab Cup
matches are included and every row says which source it came from.

🚨 Goal attribution does NOT use the event feed's `side` field. That field is
null on neutral ground, and Jordan plays a third of its matches there --
including every Asian Cup and World Cup fixture. Attributing by `side` silently
drops those, which biases exactly the period under study. Goals are attributed
by checking the scorer against Jordan's own appearance list for that match.

Own goals are handled explicitly rather than by sign convention: FBref files an
own goal under the team that benefits, so an own goal whose scorer IS a Jordan
player is a goal *against* Jordan, and one whose scorer is not is a goal *for*.

Each match also carries the model's expectation. The fit is per match on
matches strictly before that date, so nothing leaks backwards: `expected_gf`
and `expected_ga` are what the model would have predicted on the morning of the
game, which is what makes "goals above expectation" meaningful.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_results, index_teams, team_view
from baseline import PoissonStrength

ROOT = Path(__file__).resolve().parents[1]
SQUAD = ROOT / "outputs" / "squad"
JORDAN = "3e22f0fa"
TEAM = "Jordan"
LOOKBACK_DAYS = 3650
COARSE = {"GK": "GK", "CB": "DF", "LB": "DF", "RB": "DF", "DF": "DF", "WB": "DF",
          "DM": "MF", "CM": "MF", "AM": "MF", "LM": "MF", "RM": "MF", "MF": "MF",
          "LW": "FW", "RW": "FW", "FW": "FW", "CF": "FW", "SS": "FW",
          "LF": "FW", "RF": "FW"}


def coarse_position(p):
    """Collapse to GK/DF/MF/FW.

    FBref mixes generic labels (DF, MF) with specific ones (CB, DM), and the mix
    drifts from all-generic in 2011 to about half specific from 2021, so a
    fine-grained position measure would show a trend that is only relabelling.
    """
    if not isinstance(p, str):
        return None
    for part in (x.strip() for x in p.split(",")):
        if part in COARSE:
            return COARSE[part]
    return None


def _pick(name):
    combined = SQUAD / f"{name}_combined.csv"
    return pd.read_csv(combined if combined.exists() else SQUAD / f"{name}.csv")


def _expectations(dates_by_match):
    """Model expectation per match, fit only on matches strictly before it."""
    df = load_results()
    teams, idx = index_teams(df)
    ti = idx[TEAM]
    spine = team_view(df, TEAM).set_index("date")

    out = {}
    for mid, date in sorted(dates_by_match.items(), key=lambda kv: kv[1]):
        if pd.isna(date):
            continue
        win = df[(df.date < date) & (df.date >= date - pd.Timedelta(days=LOOKBACK_DAYS))]
        if len(win) < 100:
            continue
        m = PoissonStrength(len(teams)).fit(
            win.home_team.map(idx).to_numpy(), win.away_team.map(idx).to_numpy(),
            win.home_score.to_numpy(float), win.away_score.to_numpy(float),
            (~win.neutral).to_numpy(float),
            (date - win.date).dt.days.to_numpy(float))
        if date not in spine.index:
            continue
        row = spine.loc[date]
        row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
        oi = idx.get(row.opponent)
        if oi is None:
            continue
        flag = 1.0 if row.venue == "home" else 0.0
        if row.venue == "away":
            mo, mj = m.rates(np.array([oi]), np.array([ti]), np.array([1.0]))
        else:
            mj, mo = m.rates(np.array([ti]), np.array([oi]), np.array([flag]))
        out[mid] = (float(mj[0]), float(mo[0]), float(m.strength(ti)))
    return out


def build_match_table():
    matches = _pick("matches")
    apps = _pick("appearances")
    events = pd.read_csv(SQUAD / "events.csv")
    matches["date"] = pd.to_datetime(matches["Date"], errors="coerce")
    matches = matches.dropna(subset=["date", "match_id"])

    spine = team_view(load_results(), TEAM)[
        ["date", "opponent", "tournament", "venue", "gf", "ga", "gd", "result"]]

    ja = apps[apps.squad_id == JORDAN].copy()
    ja["minutes"] = pd.to_numeric(ja.get("Min"), errors="coerce")
    ja["age"] = (pd.to_numeric(ja.age_years, errors="coerce")
                 + pd.to_numeric(ja.age_days, errors="coerce") / 365.25)
    ja["grp"] = ja.Pos.map(coarse_position)
    starters = ja[ja.role == "starter"]

    g = starters.groupby("match_id")
    per = pd.DataFrame({
        "n_starters": g.size(),
        "age_mean": g.age.mean(),
        "age_sd": g.age.std(ddof=1),
        "n_subs_used": ja[ja.role == "sub"].groupby("match_id").size(),
    })
    shape = (starters.pivot_table(index="match_id", columns="grp", values="Player",
                                 aggfunc="count", fill_value=0)
             .rename(columns=lambda c: f"n_{c.lower()}"))
    per = per.join(shape)

    # substitution and card timing, Jordan's own only where side is known
    ev = events.merge(matches[["match_id", "date"]], on="match_id", how="left")
    jnames = set(ja.Player.dropna())
    ev["is_jordan"] = ev.player.isin(jnames) | ev.player_2.isin(jnames)
    jev = ev[ev.is_jordan]
    subs = jev[jev.type == "sub"].groupby("match_id").minute
    cards = jev[jev.type.isin(["yellow", "red", "second_yellow"])]
    per = per.join(pd.DataFrame({
        "first_sub_minute": subs.min(),
        "median_sub_minute": subs.median(),
        "n_cards": cards.groupby("match_id").size(),
    }))

    keep = ["match_id", "date", "Comp", "Round", "Formation", "Opp Formation",
            "Poss", "Attendance", "Referee", "Captain", "home_manager", "source"]
    keep = [c for c in keep if c in matches.columns]
    out = (matches[keep].merge(per, on="match_id", how="left")
           .merge(spine, on="date", how="left"))
    out = out.rename(columns={"Comp": "comp", "Round": "round",
                              "Formation": "formation",
                              "Opp Formation": "opp_formation",
                              "Poss": "possession", "Attendance": "attendance",
                              "Referee": "referee", "Captain": "captain",
                              "home_manager": "manager"})
    out["year"] = out.date.dt.year
    out["has_xi"] = out.n_starters.eq(11)

    exp = _expectations(dict(zip(out.match_id, out.date)))
    out["expected_gf"] = out.match_id.map(lambda m: exp.get(m, (np.nan,) * 3)[0])
    out["expected_ga"] = out.match_id.map(lambda m: exp.get(m, (np.nan,) * 3)[1])
    out["strength_at_match"] = out.match_id.map(lambda m: exp.get(m, (np.nan,) * 3)[2])
    out["expected_gd"] = out.expected_gf - out.expected_ga
    out["gd_above_expected"] = out.gd - out.expected_gd
    out["gf_above_expected"] = out.gf - out.expected_gf

    return out.sort_values("date").reset_index(drop=True)


GOAL_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]\s*((?:\{\{goal\|[^}]*\}\}\s*)+)")
GOAL_TPL = re.compile(r"\{\{goal\|([^}]*)\}\}")


def _minutes_from_template(body):
    """Every goal in one {{goal|...}} template.

    A brace is written as a single template: {{goal|48||68|pen.}} is two goals,
    at 48 and at 68, the second a penalty. Taking only the first number loses
    the rest, so walk the arguments: a numeric one starts a goal, a following
    word annotates the goal before it.
    """
    out = []
    for arg in body.split("|"):
        arg = arg.strip()
        if not arg:
            continue
        m = re.match(r"^(\d+)(?:\+(\d+))?$", arg)
        if m:
            out.append({"minute": int(m.group(1)),
                        "minute_stoppage": int(m.group(2)) if m.group(2) else 0,
                        "type": "goal"})
        elif out and "pen" in arg.lower():
            out[-1]["type"] = "penalty_goal"
    return out


def _arabcup_goals(apps):
    """Arab Cup scorers, parsed from the Wikipedia match box."""
    f = SQUAD / "arabcup_matches.csv"
    if not f.exists():
        return pd.DataFrame()
    acm = pd.read_csv(f, parse_dates=["date"])
    rows = []
    for _, r in acm.iterrows():
        if not isinstance(r.goals_raw, str):
            continue
        for target, display, goalblob in GOAL_RE.findall(r.goals_raw):
            name = (display or target).strip()
            for tpl in GOAL_TPL.findall(goalblob):
                for g in _minutes_from_template(tpl):
                    rows.append({
                        "match_id": r.match_id, "date": r.date,
                        "scorer_wiki": name, "assist": None,
                        "source": "wikipedia", **g,
                    })
    return pd.DataFrame(rows)


def build_goals(match_table):
    apps = _pick("appearances")
    events = pd.read_csv(SQUAD / "events.csv")
    dates = dict(zip(match_table.match_id, match_table.date))

    ja = apps[apps.squad_id == JORDAN].copy()
    ja["grp"] = ja.Pos.map(coarse_position)
    ja["age"] = (pd.to_numeric(ja.age_years, errors="coerce")
                 + pd.to_numeric(ja.age_days, errors="coerce") / 365.25)
    # who played for Jordan in each match, by name and by id
    by_match = {m: set(g.Player.dropna()) for m, g in ja.groupby("match_id")}

    ev = events[events.type.isin(["goal", "penalty_goal", "own_goal"])].copy()
    ev["date"] = ev.match_id.map(dates)
    ev = ev[ev.date.notna()]

    # Matches whose FBref report carries a single appearance row give no squad
    # to check a scorer against. For those only, fall back to the event feed's
    # side -- usable at home and away, never on neutral ground, which is why it
    # is a fallback and not the primary rule.
    venue = dict(zip(match_table.match_id, match_table.venue))
    thin = {m for m, sq in by_match.items() if len(sq) < 5}
    thin |= set(match_table.match_id) - set(by_match)

    rows = []
    for _, e in ev.iterrows():
        squad = by_match.get(e.match_id, set())
        if e.match_id in thin:
            v = venue.get(e.match_id)
            if v == "home":
                scorer_is_jordan = e.side == "home"
            elif v == "away":
                scorer_is_jordan = e.side == "away"
            else:
                continue          # neutral + no squad list: not attributable
        else:
            scorer_is_jordan = e.player in squad
        if e.type == "own_goal":
            # an own goal by a Jordan player is a goal against Jordan
            if scorer_is_jordan:
                continue
            rows.append({"match_id": e.match_id, "date": e.date, "minute": e.minute,
                         "minute_stoppage": e.minute_stoppage, "scorer": None,
                         "type": "own_goal_for", "assist": None, "source": "fbref"})
        elif scorer_is_jordan:
            rows.append({"match_id": e.match_id, "date": e.date, "minute": e.minute,
                         "minute_stoppage": e.minute_stoppage, "scorer": e.player,
                         "type": e.type,
                         "assist": e.player_2 if e.type != "sub" else None,
                         "source": "fbref"})

    g = pd.DataFrame(rows)

    ac = _arabcup_goals(apps)
    if len(ac):
        ac = ac.rename(columns={"scorer_wiki": "scorer"})
        g = pd.concat([g, ac], ignore_index=True)

    # attach scorer attributes from the appearance row for that match
    attr = ja.set_index(["match_id", "Player"])[["grp", "age", "role", "player_id"]]
    idx = pd.MultiIndex.from_arrays([g.match_id, g.scorer])
    joined = attr.reindex(idx)
    g["scorer_position"] = joined.grp.values
    g["scorer_age"] = joined.age.values
    g["scorer_role"] = joined.role.values
    g["scorer_player_id"] = joined.player_id.values
    return g.sort_values(["date", "minute"]).reset_index(drop=True)


def build():
    mt = build_match_table()
    mt.to_csv(SQUAD / "match_table.csv", index=False)
    print(f"match_table.csv  {len(mt)} matches, {mt.has_xi.sum()} with a full XI")

    goals = build_goals(mt)
    goals.to_csv(SQUAD / "goals.csv", index=False)
    print(f"goals.csv        {len(goals)} Jordan goals")
    print(goals.type.value_counts().to_string())
    return mt, goals


if __name__ == "__main__":
    build()
