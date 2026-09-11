"""Merge the Wikipedia-sourced Arab Cup matches into the FBref squad tables.

`appearances.csv` stays exactly what `fbref_parse.py` produces -- FBref only,
regenerable from the page cache. This module writes a separate
`appearances_combined.csv` that adds the ten Arab Cup matches FBref does not
carry, every row tagged with `source`. Downstream analysis reads the combined
file; provenance is never lost and neither input is mutated.

Two things Wikipedia does not give, and how each is handled:

**Player ids.** Names are not a safe key (transliteration varies), so the
crosswalk in `arabcup_name_crosswalk.csv` must be human-confirmed before a row
is merged. Anything still unconfirmed is dropped, loudly, rather than guessed.

**Age.** Wikipedia carries no birth date. FBref's own appearance rows do carry
age as `YY-DDD` on a known match date, so a date of birth can be recovered
exactly by subtracting. Doing it from every appearance and taking the median
guards against a stray bad row. Players who never appear in FBref data have no
recoverable age and are left null rather than imputed.

Minutes are derived, not observed: a starter who is not substituted plays the
whole match, one who is plays until their off-minute, a substitute plays from
their on-minute. Match length is taken as the latest event minute seen, so
matches that went to extra time are not truncated at 90.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "squad"
JORDAN = "3e22f0fa"


def player_dob():
    """Recover each FBref player's date of birth from their age on match days."""
    apps = pd.read_csv(OUT / "appearances.csv")
    matches = pd.read_csv(OUT / "matches.csv")
    dates = dict(zip(matches.match_id, pd.to_datetime(matches["Date"], errors="coerce")))

    a = apps[apps.squad_id == JORDAN].dropna(subset=["player_id"]).copy()
    a["date"] = a.match_id.map(dates)
    a = a.dropna(subset=["date"])
    a["yrs"] = pd.to_numeric(a.age_years, errors="coerce")
    a["dys"] = pd.to_numeric(a.age_days, errors="coerce")
    a = a.dropna(subset=["yrs", "dys"])

    born = [d - relativedelta(years=int(y)) - pd.Timedelta(days=int(dd))
            for d, y, dd in zip(a.date, a.yrs, a.dys)]
    a["dob"] = born
    # median guards against a single mistyped age
    out = a.groupby("player_id").dob.median().rename("dob").reset_index()
    names = a.drop_duplicates("player_id")[["player_id", "Player"]]
    return out.merge(names, on="player_id", how="left")


def confirmed_crosswalk():
    cw = pd.read_csv(OUT / "arabcup_name_crosswalk.csv")
    ok = cw.status == "exact"
    conf = cw.confirmed.astype(str).str.strip().str.lower().isin(
        {"y", "yes", "true", "1"})
    # a confirmed row takes its suggested match
    cw.loc[conf & cw.player_id.isna(), "fbref_name"] = cw.loc[conf & cw.player_id.isna(),
                                                              "suggested_fbref_name"]
    resolved = cw[ok | conf].copy()
    unresolved = cw[~(ok | conf)]
    return resolved, unresolved


def arabcup_appearances():
    lin = pd.read_csv(OUT / "arabcup_lineups.csv", parse_dates=["date"])
    resolved, unresolved = confirmed_crosswalk()

    if len(unresolved):
        print(f"  !! {len(unresolved)} names still unconfirmed, their rows are "
              f"DROPPED: {sorted(unresolved.wiki_name)}")

    # attach ids: exact matches already carry one; confirmed ones resolve by name
    apps = pd.read_csv(OUT / "appearances.csv")
    fb = (apps[apps.squad_id == JORDAN][["Player", "player_id"]]
          .dropna().drop_duplicates("Player"))
    byname = dict(zip(fb.Player, fb.player_id))
    resolved["player_id"] = resolved.apply(
        lambda r: r.player_id if pd.notna(r.player_id) else byname.get(r.fbref_name),
        axis=1)
    idmap = dict(zip(resolved.wiki_name, resolved.player_id))
    namemap = dict(zip(resolved.wiki_name, resolved.fbref_name))

    lin = lin[lin.player.isin(idmap)].copy()
    lin["player_id"] = lin.player.map(idmap)
    lin["fbref_name"] = lin.player.map(namemap)

    # match length from the latest event minute, so extra time is not truncated
    mins = pd.concat([lin.on_minute, lin.off_minute, lin.yellow_minute,
                      lin.red_minute]).dropna()
    full = lin.groupby("match_id").apply(
        lambda g: max(90, int(np.nanmax(
            np.concatenate([g.on_minute.values, g.off_minute.values,
                            g.yellow_minute.values, g.red_minute.values,
                            [90]]).astype(float)))),
        include_groups=False).rename("full_time")
    lin = lin.merge(full, on="match_id", how="left")

    played = lin[lin.role.isin(["starter", "sub"])].copy()
    played["Min"] = np.where(
        played.role == "starter",
        played.off_minute.fillna(played.full_time),
        played.full_time - played.on_minute)

    dob = player_dob().set_index("player_id").dob
    played["dob"] = played.player_id.map(dob)
    age = (played.date - played.dob)
    played["age_years"] = (age.dt.days // 365.25).astype("Float64")
    played["age_days"] = (age.dt.days - (played.age_years * 365.25)).astype("Float64")

    out = pd.DataFrame({
        "squad_id": JORDAN,
        "Player": played.fbref_name,
        "#": played.shirt_no,
        "Pos": played.position,
        "Min": played.Min.astype(float),
        "player_id": played.player_id,
        "age_years": played.age_years,
        "age_days": played.age_days,
        "match_id": played.match_id,
        "role": played.role,
        "role_source": "wikipedia_lineup",
        "source": "wikipedia",
    })
    return out


def build():
    fb = pd.read_csv(OUT / "appearances.csv")
    fb["source"] = "fbref"
    ac = arabcup_appearances()

    combined = pd.concat([fb, ac], ignore_index=True)
    combined.to_csv(OUT / "appearances_combined.csv", index=False)

    # matches: the Arab Cup fixtures, in the shape matches.csv uses
    acm = pd.read_csv(OUT / "arabcup_matches.csv", parse_dates=["date"])
    acm_out = pd.DataFrame({
        "Date": acm.date.dt.strftime("%Y-%m-%d"), "Comp": "Arab Cup",
        "Venue": "Neutral", "GF": acm.gf, "GA": acm.ga,
        "Opponent": acm.opponent_code, "Attendance": acm.attendance,
        "Referee": acm.referee, "match_id": acm.match_id,
        "home_manager": acm.manager, "source": "wikipedia",
    })
    m = pd.read_csv(OUT / "matches.csv")
    m["source"] = "fbref"
    pd.concat([m, acm_out], ignore_index=True).to_csv(
        OUT / "matches_combined.csv", index=False)

    n_ac = ac.match_id.nunique()
    print(f"appearances_combined.csv  {len(combined)} rows "
          f"({len(fb)} fbref + {len(ac)} wikipedia)")
    print(f"matches_combined.csv      {len(m) + len(acm_out)} matches "
          f"({len(m)} fbref + {len(acm_out)} wikipedia)")

    xi = ac[ac.role == "starter"].groupby("match_id").size()
    print(f"\nArab Cup matches merged: {n_ac}; all with 11 starters: "
          f"{bool((xi == 11).all())}")
    print(f"age recovered for {ac.age_years.notna().mean():.0%} of merged rows")

    coverage(combined)
    return combined


def coverage(combined):
    """Recompute spine coverage now that the Arab Cup matches are present."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data import load_results, team_view

    allm = pd.read_csv(OUT / "matches_combined.csv")
    allm["date"] = pd.to_datetime(allm["Date"], errors="coerce")
    # merge_asof refuses null keys, and one FBref fixture row has no date
    allm = allm.dropna(subset=["date"])
    xi = (combined[(combined.squad_id == JORDAN) & (combined.role == "starter")]
          .groupby("match_id").size())
    allm["n_starters"] = allm.match_id.map(xi).fillna(0).astype(int)

    spine = team_view(load_results(), "Jordan")[
        ["date", "opponent", "tournament", "venue", "gf", "ga", "result"]]
    out = pd.merge_asof(
        spine.sort_values("date"),
        allm[["date", "match_id", "Comp", "n_starters", "source"]].sort_values("date"),
        on="date", direction="nearest", tolerance=pd.Timedelta(days=1))
    out["has_xi"] = out.n_starters.eq(11)
    out.to_csv(OUT / "match_coverage_combined.csv", index=False)

    since = out[out.date >= "2015-01-01"]
    print(f"\nmatch_coverage_combined.csv  since 2015: {int(since.has_xi.sum())} "
          f"of {len(since)} matches with a full XI "
          f"({since.has_xi.mean():.0%}), was 81% before the merge")
    gaps = since[~since.has_xi].tournament.value_counts()
    for t, n in gaps.items():
        print(f"    {n:3d}  {t}")


if __name__ == "__main__":
    build()
