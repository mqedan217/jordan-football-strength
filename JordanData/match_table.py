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

import json
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
COARSE = {"GK": "GK",
          # wing-backs group with defenders, as bare WB already did. RWB/LWB
          # were missing, which silently dropped 12 starters from the shape
          # counts and left several matches summing to 9 players rather than 11.
          "CB": "DF", "LB": "DF", "RB": "DF", "DF": "DF",
          "WB": "DF", "RWB": "DF", "LWB": "DF",
          "DM": "MF", "CM": "MF", "AM": "MF", "LM": "MF", "RM": "MF", "MF": "MF",
          "LW": "FW", "RW": "FW", "FW": "FW", "CF": "FW", "SS": "FW",
          "LF": "FW", "RF": "FW"}


# A second grouping, used ONLY for the band counts behind xi_shape. A winger is
# a forward when you ask who scored, and a midfielder when you ask what shape
# the team lined up in -- in a 4-2-3-1 the wide players are the "3". Tested
# against the 97 matches where FBref states a formation: sending wingers to FW
# reproduces its bands 58% of the time, to MF 69%. Neither is good enough to
# call the result a formation, which is why xi_shape is a separate column.
SHAPE = {**{k: v for k, v in COARSE.items()}, "LW": "MF", "RW": "MF"}


def shape_position(p):
    if not isinstance(p, str):
        return None
    for part in (x.strip() for x in p.split(",")):
        if part in SHAPE:
            return SHAPE[part]
    return None


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


def _as_json(group):
    """A squad list as a JSON array of player objects.

    Carries the player id as well as the name, so the field can be joined back
    to appearances without re-deriving identity from a name -- names are not a
    safe key here. `appearances_combined.csv` remains authoritative; this is a
    self-contained copy for reading a match without a join.
    """
    out = []
    for _, r in group.iterrows():
        entry = {"id": r.get("player_id"), "name": r.get("Player"),
                 "pos": r.get("grp"), "min": r.get("minutes")}
        no = r.get("#")
        if pd.notna(no):
            entry["no"] = int(no)
        if pd.notna(r.get("age")):
            entry["age"] = round(float(r["age"]), 1)
        out.append({k: (None if pd.isna(v) else
                        (int(v) if isinstance(v, float) and k == "min" and
                         float(v).is_integer() else v))
                    for k, v in entry.items()})
    return json.dumps(out, ensure_ascii=False)


def _pick(name):
    combined = SQUAD / f"{name}_combined.csv"
    return pd.read_csv(combined if combined.exists() else SQUAD / f"{name}.csv")


def _expectations(fixtures):
    """Model expectation per match, fit only on matches strictly before it.

    Takes opponent and venue from the match row rather than looking them up in
    the results spine. Three FBref matches (the March and June 2019 friendlies
    against Syria, Iraq and Slovakia) are absent from results.csv entirely, and
    a spine lookup silently skipped them.
    """
    df = load_results()
    teams, idx = index_teams(df)
    ti = idx[TEAM]

    out = {}
    for mid, date, opponent, venue in sorted(fixtures, key=lambda r: r[1]):
        if pd.isna(date) or not isinstance(opponent, str):
            continue
        win = df[(df.date < date) & (df.date >= date - pd.Timedelta(days=LOOKBACK_DAYS))]
        if len(win) < 100:
            continue
        m = PoissonStrength(len(teams)).fit(
            win.home_team.map(idx).to_numpy(), win.away_team.map(idx).to_numpy(),
            win.home_score.to_numpy(float), win.away_score.to_numpy(float),
            (~win.neutral).to_numpy(float),
            (date - win.date).dt.days.to_numpy(float))
        oi = idx.get(opponent)
        if oi is None:
            continue
        flag = 1.0 if venue == "home" else 0.0
        if venue == "away":
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

    # The XI as text on the match row. The authoritative per-player data is
    # appearances_combined.csv, one row per player per match -- this is a
    # readable copy so a match can be eyeballed without a join, which is what
    # anyone checking these rows by hand actually needs.
    ja = ja.sort_values(["match_id", "Min"], ascending=[True, False])
    starters = ja[ja.role == "starter"]
    subs = ja[ja.role == "sub"]

    g = starters.groupby("match_id")
    per = pd.DataFrame({
        "n_starters": g.size(),
        "age_mean": g.age.mean(),
        "age_sd": g.age.std(ddof=1),
        "n_subs_used": subs.groupby("match_id").size(),
        "starting_xi": g.apply(_as_json, include_groups=False),
        "subs_used": subs.groupby("match_id").apply(_as_json, include_groups=False),
    })
    shape = (starters.pivot_table(index="match_id", columns="grp", values="Player",
                                 aggfunc="count", fill_value=0)
             .rename(columns=lambda c: f"n_{c.lower()}"))
    per = per.join(shape)

    # band counts for xi_shape use the shape grouping, not the scoring one
    starters_shape = starters.assign(band=starters.Pos.map(shape_position))
    bands = (starters_shape.pivot_table(index="match_id", columns="band",
                                        values="Player", aggfunc="count",
                                        fill_value=0)
             .rename(columns=lambda c: f"band_{c.lower()}"))
    per = per.join(bands)

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

    # Fall back to the appearance rows where the event feed is silent. The ten
    # Arab Cup matches come from Wikipedia and have no FBref events, but their
    # lineups carry on_minute for every substitute and a card minute for every
    # booking, so the same measures are derivable directly.
    if "on_minute" in ja.columns:
        on = pd.to_numeric(ja.on_minute, errors="coerce")
        sub_on = ja.assign(on=on)[ja.role.eq("sub") & on.notna()].groupby("match_id").on
        per["first_sub_minute"] = per.first_sub_minute.fillna(sub_on.min())
        per["median_sub_minute"] = per.median_sub_minute.fillna(sub_on.median())
    card_cols = [c for c in ["yellow_minute", "red_minute"] if c in ja.columns]
    if card_cols:
        n_card = (ja[card_cols].apply(pd.to_numeric, errors="coerce").notna()
                  .sum(axis=1).groupby(ja.match_id).sum())
        has_any = n_card.index[n_card.gt(0)]
        per["n_cards"] = per.n_cards.fillna(n_card.reindex(per.index).where(
            per.index.isin(has_any)))
    if "is_captain" in ja.columns:
        cap = (ja[ja.is_captain.fillna(False).astype(bool)]
               .drop_duplicates("match_id").set_index("match_id").Player)
        per["captain_from_lineup"] = cap

    # Opponent/Venue/GF/GA are FBref's own copies of the spine columns. They are
    # kept so the three matches results.csv lacks can be filled from them, then
    # dropped once used.
    keep = ["match_id", "date", "Comp", "Round", "Formation", "Opp Formation",
            "Poss", "Attendance", "Referee", "Captain", "home_manager", "source",
            "Opponent", "Venue", "GF", "GA"]
    keep = [c for c in keep if c in matches.columns]
    out = (matches[keep].merge(per, on="match_id", how="left")
           .merge(spine, on="date", how="left"))
    out = out.rename(columns={"Comp": "comp", "Round": "round",
                              "Formation": "formation",
                              "Opp Formation": "opp_formation",
                              "Poss": "possession", "Attendance": "attendance",
                              "Referee": "referee", "Captain": "captain",
                              "home_manager": "manager"})
    # Zero versus not-recorded. A count is a real 0 when we hold the source for
    # that match and nothing of that kind occurred; it stays null only when we
    # have no source. Leaving a card-free match null would drop it from any
    # average, which silently inflates cards per match.
    has_events = set(events.match_id.dropna().unique())
    has_lineup_detail = set()
    if "on_minute" in ja.columns:
        detail = ja[["match_id", "on_minute", "yellow_minute", "red_minute"]]
        has_lineup_detail = set(detail.match_id[detail.notna().any(axis=1)])
    known = has_events | has_lineup_detail
    known_mask = out.match_id.isin(known)

    out["n_cards"] = out.n_cards.where(~(out.n_cards.isna() & known_mask), 0)
    out["n_subs_used"] = out.n_subs_used.where(
        ~(out.n_subs_used.isna() & out.n_starters.eq(11)), 0)

    if "captain_from_lineup" in out.columns:
        out["captain"] = out.captain.fillna(out.captain_from_lineup)
        out = out.drop(columns=["captain_from_lineup"])
    # FBref holds three matches results.csv does not, so their spine columns are
    # null. FBref's own fixture row carries the same facts -- fill from there
    # rather than leave the rows unusable. Zero extra fetching, and the values
    # come from a page already on disk.
    fb_opp = (out.Opponent.astype(str).str.replace(r"^[a-z]{2}\s+", "", regex=True)
              if "Opponent" in out.columns else pd.Series(dtype=object))
    if "Opponent" in out.columns:
        out["opponent"] = out.opponent.fillna(fb_opp.where(fb_opp.ne("nan")))
    if "Venue" in out.columns:
        out["venue"] = out.venue.fillna(out.Venue.str.lower())
    for col, src in (("gf", "GF"), ("ga", "GA")):
        if src in out.columns:
            out[col] = out[col].fillna(pd.to_numeric(out[src], errors="coerce"))
    out["gd"] = out.gd.fillna(out.gf - out.ga)
    out["result"] = out.result.fillna(
        pd.Series(np.select([out.gd > 0, out.gd == 0], ["W", "D"], "L"),
                  index=out.index).where(out.gd.notna()))
    out["tournament"] = out.tournament.fillna(out.comp)

    out = out.drop(columns=[c for c in ["Opponent", "Venue", "GF", "GA"]
                            if c in out.columns])
    out["year"] = out.date.dt.year
    out["has_xi"] = out.n_starters.eq(11)

    # A shape derived by counting the positions the source states, for the 77
    # matches where FBref records no formation string and no other source has
    # one. It is NOT the same thing as `formation` and is kept in its own
    # column: 4-2-3-1 and 4-5-1 both count as four defenders, five midfielders
    # and one forward, so the shape cannot distinguish them. It says how many
    # players occupied each band, which is a fact about the team sheet, not a
    # guess at the manager's intent.
    cols = ["band_gk", "band_df", "band_mf", "band_fw"]
    for c in cols:
        if c not in out:
            out[c] = pd.NA
    ok = out[cols].notna().all(axis=1) & out[cols].sum(axis=1).eq(11)
    out["xi_shape"] = None
    out.loc[ok, "xi_shape"] = (
        out.loc[ok, "band_df"].astype(int).astype(str) + "-"
        + out.loc[ok, "band_mf"].astype(int).astype(str) + "-"
        + out.loc[ok, "band_fw"].astype(int).astype(str))
    # A single usable formation column: FBref's where it exists, the derived
    # band count elsewhere, with the origin of every value recorded. Deliberately
    # NOT a third-party fill -- four sources were checked on the 2025 Arab Cup
    # final and returned 3-6-1 (BeSoccer), 3-4-2-1 (search summary), 3-4-3
    # (derived) and nothing (FBref) for the same eleven players. Formation is a
    # provider's interpretation, so mixing providers would make the column
    # measure who recorded it rather than how the team lined up.
    #
    # 🚨 The two methods agree only 69% of the time, and their mix shifts across
    # the treatment date: 50/50 fbref/derived before mid-2022, 85/15 after.
    # A pre/post comparison over the full span therefore partly measures the
    # METHOD changing. Restricted to 2019 onward the mix is 82% vs 85% and the
    # comparison is safe. Use `formation_source` to check this before trusting
    # any trend, or filter to formation_source == "fbref".
    out["formation_source"] = None
    out.loc[out.formation.notna(), "formation_source"] = "fbref"
    out.loc[out.formation.isna() & out.xi_shape.notna(),
            "formation_source"] = "derived_from_lineup"
    out["formation_best"] = out.formation.fillna(out.xi_shape)
    # attendance is a count; FBref writes it as "68,527", which makes the
    # column a string and blocks any later numeric fill
    if "attendance" in out.columns:
        out["attendance"] = pd.to_numeric(
            out.attendance.astype(str).str.replace(r"[^\d]", "", regex=True),
            errors="coerce").astype("Int64")
    if "possession" in out.columns:
        out["possession"] = pd.to_numeric(out.possession, errors="coerce")

    exp = _expectations(list(zip(out.match_id, out.date, out.opponent, out.venue)))
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

    g = _fill_goal_gaps(g, match_table)

    # attach scorer attributes from the appearance row for that match
    attr = ja.set_index(["match_id", "Player"])[["grp", "age", "role", "player_id"]]
    idx = pd.MultiIndex.from_arrays([g.match_id, g.scorer])
    joined = attr.reindex(idx)
    g["scorer_position"] = joined.grp.values
    g["scorer_age"] = joined.age.values
    g["scorer_role"] = joined.role.values
    g["scorer_player_id"] = joined.player_id.values
    return g.sort_values(["date", "minute"]).reset_index(drop=True)


def _fill_goal_gaps(goals, match_table):
    """Recover goals for matches whose FBref report contains no events.

    Six friendlies have an FBref report with no event feed at all, so their
    goals are simply absent -- 14 of them. Wikipedia's Jordan results articles
    record scorers for most matches, so they can be recovered.

    The rule is strict on purpose, because these rows are meant to be checked:
    a match is only filled if we currently have NO goals for it AND the number
    of goals the article lists equals the scoreline we already hold. A source
    that disagrees about the score is not a source to take scorers from -- the
    2025 Mali friendly is recorded on Wikipedia as 0-0 where the results
    dataset has 2-3, and is therefore left alone rather than reconciled.
    """
    try:
        from wiki_results import jordan_goals
    except Exception:                                   # noqa: BLE001
        return goals

    have = goals.groupby("match_id").size() if len(goals) else pd.Series(dtype=int)
    need = match_table[match_table.gf.notna()].copy()
    need["have"] = need.match_id.map(have).fillna(0).astype(int)
    need = need[(need.have == 0) & (need.gf > 0)]
    if need.empty:
        return goals

    wiki = jordan_goals()
    if wiki.empty:
        return goals

    added = []
    for _, r in need.iterrows():
        cand = wiki[wiki.date == r.date]
        if len(cand) != int(r.gf):
            continue                       # count must match the scoreline
        for _, g in cand.iterrows():
            added.append({
                "match_id": r.match_id, "date": r.date, "minute": g.minute,
                "minute_stoppage": g.minute_stoppage, "scorer": g.scorer,
                "type": g.type, "assist": None, "source": "wikipedia_results",
            })
    if not added:
        return goals
    out = pd.concat([goals, pd.DataFrame(added)], ignore_index=True)
    print(f"  recovered {len(added)} goals for "
          f"{len({a['match_id'] for a in added})} matches from Wikipedia results")
    return out


def _fill_from_results_articles(out):
    """Attendance and referee from Wikipedia's Jordan results articles.

    Only nulls are filled, and only where the article's scoreline agrees with
    ours, so a date collision cannot attach the wrong row. Provenance is kept
    per field.
    """
    try:
        from wiki_results import propose
    except Exception:                                   # noqa: BLE001
        return out
    try:
        fills, clashes, _ = propose(out)
    except Exception as exc:                            # noqa: BLE001
        print(f"  !! results-article fill skipped: {type(exc).__name__}: {exc}")
        return out

    for field in ("attendance", "referee"):
        out[f"{field}_source"] = None
        out.loc[out[field].notna(), f"{field}_source"] = "fbref"
        out[field] = out[field].astype(object)
    for _, f in fills.iterrows():
        i = out.index[out.match_id == f.match_id]
        out.loc[i, f.field] = f.proposed
        out.loc[i, f"{f.field}_source"] = "wikipedia"
    out["attendance"] = pd.to_numeric(out.attendance, errors="coerce").astype("Int64")
    if len(clashes):
        print(f"  {len(clashes)} rows rejected: article scoreline disagrees")
    print(f"  filled {len(fills)} cells from the results articles")
    return out


def build():
    mt = build_match_table()
    mt = _fill_from_results_articles(mt)
    mt.to_csv(SQUAD / "match_table.csv", index=False)
    print(f"match_table.csv  {len(mt)} matches, {mt.has_xi.sum()} with a full XI")

    goals = build_goals(mt)
    goals.to_csv(SQUAD / "goals.csv", index=False)
    print(f"goals.csv        {len(goals)} Jordan goals")
    print(goals.type.value_counts().to_string())
    return mt, goals


if __name__ == "__main__":
    build()
