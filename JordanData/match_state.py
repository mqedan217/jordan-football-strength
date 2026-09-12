"""Match state: what the scoreboard said, minute by minute.

`events.csv` carries every goal in the match -- Jordan's and the opponent's --
with a minute and a running score. That is enough to reconstruct the scoreboard
through the match and ask a different class of question from the squad tables:
not who was on the pitch, but what situation they were in.

Two decisions make this trustworthy:

1. **Which side was Jordan?** The event feed is match-relative (home/away), not
   team-relative. The final running score is compared against Jordan's own gf/ga:
   whichever orientation reproduces the result is Jordan's side. Level
   scorelines reproduce both, and fall back to `venue` where it is decisive.
   This agrees with `venue` on 98% of matches where venue is decisive, and no
   match fails both orientations -- the feed is complete wherever it exists.

2. **State comes from the running score, never from the `side` label.** FBref
   files an own goal under the team that *benefits*, so `side` means different
   things for different event types (see CLAUDE.md). The score columns have no
   such ambiguity: whichever of them increased is the team that scored. Goals
   are attributed by score delta, so own goals need no special handling.

Scoring-time results are read against Armatas et al. (2009), who find goals
rise through a match and peak in the final 15 minutes -- the expected shape
that any era difference here has to be judged against. Match-state effects on
play are established in Lago-Penas & Gomez-Lopez (2014).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SQUAD = ROOT / "outputs" / "squad"

GOAL_TYPES = ("goal", "penalty_goal", "own_goal")
BUCKETS = [0, 15, 30, 45, 60, 75, 91]
BUCKET_LABELS = ["1-15", "16-30", "31-45", "46-60", "61-75", "76-90"]


def _load():
    ev = pd.read_csv(SQUAD / "events.csv")
    mt = pd.read_csv(SQUAD / "match_table.csv", parse_dates=["date"])
    return ev, mt


def resolve_side(ev, mt):
    """Jordan's orientation in the event feed, per match.

    Returns a frame of match_id, jordan_side, side_source. A match appears only if it
    could be resolved; a guessed orientation would silently invert every
    scored/conceded number for that match, so an unresolved one is dropped.
    """
    order = ev.sort_values(["match_id", "minute", "minute_stoppage"])
    last = (order.groupby("match_id")
            .agg(sh=("score_home", "last"), sa=("score_away", "last"))
            .reset_index())
    j = last.merge(mt[["match_id", "gf", "ga", "venue"]], on="match_id")

    as_home = j.sh.eq(j.gf) & j.sa.eq(j.ga)
    as_away = j.sa.eq(j.gf) & j.sh.eq(j.ga)
    if (~as_home & ~as_away).any():
        bad = j.match_id[~as_home & ~as_away].tolist()
        raise ValueError(f"event feed disagrees with the scoreline: {bad}")

    side = pd.Series(np.where(as_home & ~as_away, "home",
                              np.where(as_away & ~as_home, "away", None)),
                     index=j.index, dtype=object)
    src = pd.Series(np.where(side.notna(), "scoreline", None),
                    index=j.index, dtype=object)

    # A level scoreline reads the same either way round. Venue settles it
    # except at neutral grounds, which stay unresolved.
    tie = side.isna() & j.venue.isin(["home", "away"])
    side[tie] = j.venue[tie]
    src[tie] = "venue"

    j["jordan_side"], j["side_source"] = side, src
    return j.dropna(subset=["jordan_side"])[["match_id", "jordan_side", "side_source"]]


def _timeline(rows, side):
    """Goal minutes as (minute, +1 for Jordan / -1 against), in order."""
    out = []
    prev_h = prev_a = 0
    for _, r in rows.iterrows():
        h, a = r.score_home, r.score_away
        if pd.isna(h) or pd.isna(a):
            continue
        dh, da = h - prev_h, a - prev_a
        prev_h, prev_a = h, a
        # whichever score advanced is the scoring team, whatever the label says
        if dh > 0 and da <= 0:
            scorer = "home"
        elif da > 0 and dh <= 0:
            scorer = "away"
        else:
            continue
        out.append((float(r.minute), 1 if scorer == side else -1))
    return out


def build():
    ev, mt = _load()
    sides = resolve_side(ev, mt)
    goals = ev[ev.type.isin(GOAL_TYPES)].sort_values(
        ["match_id", "minute", "minute_stoppage"])
    goals = goals.merge(sides, on="match_id")

    # A goalless match has no goal events but is still 90 minutes of match
    # state -- level throughout. Dropping those would bias every state measure
    # toward matches where something happened. All 18 are genuine 0-0 draws.
    goalless = sides[~sides.match_id.isin(goals.match_id)]

    rows = []
    for mid in goalless.match_id:
        src = goalless.loc[goalless.match_id.eq(mid), "side_source"].iloc[0]
        side = goalless.loc[goalless.match_id.eq(mid), "jordan_side"].iloc[0]
        rows.append({
            "match_id": mid, "jordan_side": side, "side_source": src,
            "n_goal_events": 0, "first_goal": None, "first_goal_minute": np.nan,
            "min_leading": 0.0, "min_level": 90.0, "min_trailing": 0.0,
            "ever_led": False, "ever_trailed": False, "match_end": 90.0,
        })

    for mid, grp in goals.groupby("match_id"):
        side = grp.jordan_side.iloc[0]
        tl = _timeline(grp, side)
        # Match length: 90, or later if a goal came in stoppage time. Halves are
        # not separable here, so this is deliberately coarse.
        end = max(90.0, max((m for m, _ in tl), default=0.0))

        lead = level = trail = 0.0
        diff, t = 0, 0.0
        for minute, delta in tl:
            span = minute - t
            bucket = "lead" if diff > 0 else ("trail" if diff < 0 else "level")
            lead += span if bucket == "lead" else 0
            level += span if bucket == "level" else 0
            trail += span if bucket == "trail" else 0
            diff += delta
            t = minute
        span = end - t
        lead += span if diff > 0 else 0
        level += span if diff == 0 else 0
        trail += span if diff < 0 else 0

        deltas = [d for _, d in tl]
        running = np.cumsum(deltas) if deltas else np.array([])
        rows.append({
            "match_id": mid,
            "jordan_side": side,
            "side_source": grp.side_source.iloc[0],
            "n_goal_events": len(tl),
            "first_goal": ("for" if deltas[0] > 0 else "against") if deltas else None,
            "first_goal_minute": tl[0][0] if tl else np.nan,
            "min_leading": lead, "min_level": level, "min_trailing": trail,
            "ever_led": bool((running > 0).any()),
            "ever_trailed": bool((running < 0).any()),
            "match_end": end,
        })

    st = pd.DataFrame(rows)

    # Goals for and against per 15-minute bucket, wide.
    per = []
    for mid, grp in goals.groupby("match_id"):
        side = grp.jordan_side.iloc[0]
        for minute, delta in _timeline(grp, side):
            per.append((mid, minute, "for" if delta > 0 else "against"))
    pg = pd.DataFrame(per, columns=["match_id", "minute", "dir"])
    pg["bucket"] = pd.cut(pg.minute.clip(0, 90), BUCKETS,
                          labels=BUCKET_LABELS, include_lowest=True)
    wide = (pg.groupby(["match_id", "dir", "bucket"], observed=False).size()
            .unstack(["dir", "bucket"], fill_value=0).astype(int))
    wide.columns = [f"{d}_{b}" for d, b in wide.columns]

    st = st.merge(wide.reset_index(), on="match_id", how="left")
    out = mt.merge(st, on="match_id", how="left")

    # Outcome given the state reached. `result` is Jordan's, so no re-derivation.
    out["recovered"] = np.where(out.ever_trailed, out.result.isin(["W", "D"]), np.nan)
    out["held_lead"] = np.where(out.ever_led, out.result.eq("W"), np.nan)

    context = ["date", "opponent", "venue", "tournament", "result", "gf", "ga",
               "expected_gd", "gd_above_expected", "recovered", "held_lead"]
    cols = ["match_id"] + [c for c in context if c in out.columns] + \
           [c for c in out.columns if c in set(st.columns) - {"match_id"}]
    out[cols].to_csv(SQUAD / "match_state.csv", index=False)

    have = st.match_id.nunique()
    print(f"match_state.csv  {have} matches with a reconstructed scoreboard "
          f"({len(mt)} in the table)")
    print("  side resolved by:", st.side_source.value_counts().to_dict())
    print(f"  goals reconstructed: {int(st.n_goal_events.sum())}")
    return out


if __name__ == "__main__":
    build()
