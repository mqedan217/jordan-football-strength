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
import re
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


GOAL_TPL = re.compile(r"\{\{goal\|([^}]*)\}\}")


def _arabcup_timelines():
    """Goal timelines for the Arab Cup, from the Wikipedia match boxes.

    One template can hold several goals -- `{{goal|48||68|pen.}}` is two -- so
    every numeric field counts, not every template. Stoppage time is written
    `90+3`, which int() on the leading digits resolves to the minute the goal
    belongs in. Returns {match_id: [(minute, +1 for Jordan / -1 against)]}.
    """
    path = SQUAD / "arabcup_matches.csv"
    if not path.exists():
        return {}
    ac = pd.read_csv(path)
    if "opp_goals_raw" not in ac.columns:
        return {}

    def minutes(blob):
        if not isinstance(blob, str):
            return []
        out = []
        for m in GOAL_TPL.finditer(blob):
            for part in m.group(1).split("|"):
                hit = re.match(r"\s*(\d+)(?:\+(\d+))?", part)
                if hit:
                    out.append(float(hit.group(1)) +
                               (float(hit.group(2)) if hit.group(2) else 0.0))
        return out

    tl = {}
    for r in ac.itertuples():
        ours = [(m, 1) for m in minutes(r.goals_raw)]
        theirs = [(m, -1) for m in minutes(r.opp_goals_raw)]
        if len(ours) != r.gf or len(theirs) != r.ga:
            raise ValueError(
                f"Arab Cup {r.date}: parsed {len(ours)}-{len(theirs)}, "
                f"scoreline says {r.gf}-{r.ga}")
        tl[r.match_id] = sorted(ours + theirs)
    return tl


def _state_row(mid, tl, side, src):
    """State measures for one match from its goal timeline."""
    end = max(90.0, max((m for m, _ in tl), default=0.0))
    lead = level = trail = 0.0
    diff, t = 0, 0.0
    for minute, delta in tl:
        span = minute - t
        if diff > 0:
            lead += span
        elif diff < 0:
            trail += span
        else:
            level += span
        diff += delta
        t = minute
    span = end - t
    if diff > 0:
        lead += span
    elif diff < 0:
        trail += span
    else:
        level += span

    deltas = [d for _, d in tl]
    running = np.cumsum(deltas) if deltas else np.array([])
    return {
        "match_id": mid, "jordan_side": side, "side_source": src,
        "n_goal_events": len(tl),
        "first_goal": ("for" if deltas[0] > 0 else "against") if deltas else None,
        "first_goal_minute": tl[0][0] if tl else np.nan,
        "min_leading": lead, "min_level": level, "min_trailing": trail,
        "ever_led": bool((running > 0).any()),
        "ever_trailed": bool((running < 0).any()),
        "match_end": end,
    }


def build():
    ev, mt = _load()
    sides = resolve_side(ev, mt)
    goals = ev[ev.type.isin(GOAL_TYPES)].sort_values(
        ["match_id", "minute", "minute_stoppage"]).merge(sides, on="match_id")

    # One dict of goal timelines, whatever the source. Everything downstream
    # reads this, so a match from Wikipedia is treated exactly like one from
    # the event feed.
    timelines = {}
    for mid, grp in goals.groupby("match_id"):
        timelines[mid] = (_timeline(grp, grp.jordan_side.iloc[0]),
                          grp.jordan_side.iloc[0], grp.side_source.iloc[0])

    # A goalless match has no goal events but is still 90 minutes of match
    # state -- level throughout. Dropping those would bias every state measure
    # toward matches where something happened. All 18 are genuine 0-0 draws.
    for r in sides[~sides.match_id.isin(goals.match_id)].itertuples():
        timelines[r.match_id] = ([], r.jordan_side, r.side_source)

    # The Arab Cup, which the event feed does not carry at all.
    for mid, tl in _arabcup_timelines().items():
        timelines.setdefault(mid, (tl, "home", "arabcup_box"))

    rows = [_state_row(mid, tl, side, src)
            for mid, (tl, side, src) in timelines.items()]

    st = pd.DataFrame(rows)

    # Goals for and against per 15-minute bucket, wide.
    per = []
    for mid, (tl, _side, _src) in timelines.items():
        for minute, delta in tl:
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
