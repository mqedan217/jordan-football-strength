# Squad layer — extraction spec

What we fetch, what we keep, and what we know is missing before we start.
Written against real pages (one history page, one season page, one Jordan match
report), not against assumptions. Verified 2026-09-09.

## Scope

Jordan men's national team, **2014 → 2026**. FBref holds continuous seasons
2014–2023, 2025, 2026 and nothing usable before 2014 (2011, 2010, 2007, 2006,
2004, 2002, 2000 exist but are isolated).

**Season labels are competition cycles, not calendar years.** There is no 2024
season: the 2023 Asian Cup (played Jan–Feb 2024) files under season 2023, and
the 2024–25 World Cup qualifiers file under season 2026. Everything in this
pipeline keys on **match date**, never on season label.

## Source of truth

`data/results.csv` is the spine — every Jordan match 2014+ gets a row whether
or not FBref has it. FBref data is joined onto that spine by date and opponent.
This keeps coverage holes visible instead of silently absorbing them.

Known holes: FBref tracks only World Cup, WCQ-AFC, AFC Asian Cup, AFC Asian Cup
qualification and Friendlies for Jordan. It does **not** track Arab Cup (10
matches), WAFF Championship (6), Palestine International Championship (4),
King's Cup (2), Jordan International Tournament (2). All 24 are competitive
matches; none are friendlies. Arab Cup is to be hand-entered — every table
below carries a `source` column (`fbref` | `manual`) so hand-entered rows are
never mistaken for scraped ones.

## Fetch plan

| step | pages | notes |
|---|---|---|
| 1 | 13 season pages | `/en/squads/3e22f0fa/<season>/Jordan-Men-Stats` |
| 2 | ~126 match reports | `/en/matches/<8-hex-id>/<slug>` |

Roughly 139 requests, three already cached. Every response is written to
`cache/raw/` and never re-requested; the pipeline is resumable and re-running
it with a warm cache makes zero network calls.

FBref blocks plain HTTP clients, so `soccerdata` drives a real headless Chrome
and paces itself. Expect 15–30 minutes for a cold run.

## Tables produced

### `matches.csv` — one row per match

`match_id`, `date`, `kickoff_venue_time`, `competition`, `stage`,
`season_label`, `home_team`, `away_team`, `home_score`, `away_score`,
`venue`, `city`, `attendance`, `neutral`, `referee`, `ar1`, `ar2`, `fourth`,
`var`, `home_manager`, `away_manager`, `home_captain`, `away_captain`,
`home_formation`, `away_formation`, `home_possession`, `away_possession`,
`fbref_url`, `fetched_at`, `source`

Formation comes from the lineup-table header (e.g. `Jordan (4-2-3-1)`), not a
field of its own. Possession exists only from ~Sept 2024; expect null before.
Attendance is present for the World Cup match checked (68,527) but is not
guaranteed for AFC fixtures.

### `appearances.csv` — one row per player per match

`match_id`, `team`, `squad_id`, `player`, `player_id`, `shirt_no`, `nation`,
`position`, `started`, `minutes`, `age_years`, `age_days`, then the
performance block: `gls`, `ast`, `pk`, `pkatt`, `sh`, `sot`, `crdy`, `crdr`,
`fls`, `fld`, `off`, `crs`, `tklw`, `int`, `og`, `pkwon`, `pkcon`; for
keepers `sota`, `ga`, `saves`, `save_pct`; plus `source`.

Age arrives as `YY-DDD` (e.g. `33-159`) and splits into two integer columns.
`pkwon`/`pkcon` came back non-numeric even in 2026 — treat as uncollected
until a season proves otherwise.

### `events.csv` — one row per event

`match_id`, `minute`, `minute_stoppage`, `team`, `type`, `player`, `player_2`,
`score_home`, `score_away`, `source`

`type` ∈ {`goal`, `own_goal`, `penalty_goal`, `sub`, `yellow`, `red`,
`second_yellow`}. `player_2` is the assist provider for goals and the player
replaced for substitutions. Parsed from the `#events_wrap` feed, which gives
minute, running score and event kind directly — this is the match timeline.

**The trailing label is unreliable and this is verified, not theoretical.** In
the Austria–Jordan report the feed reads:

```
76’  2:1  Yazan Al-Arab Own Goal        — Substitute
90+12’ 3:1 Marko Arnautović Penalty Kick — Substitute
81’  2:1  Saed Al-Rosan for Abdallah Nasib — Substitute
```

Own goals, penalty goals and actual substitutions all trail with
`— Substitute`. Everything that identifies the event lives *before* the em
dash, so classification reads the head (`Own Goal`, `Penalty Kick`, `X for Y`)
and uses the trailer only to disambiguate goals from cards. Keying on the
trailer or the CSS class silently turns two goals into substitutions.

One semantic caveat: FBref files an own goal under the team that *benefits*,
so `side` names the scoring team while `player` names the opponent who scored
it. That is FBref's convention, preserved rather than corrected.

### `unused_subs.csv`

`match_id`, `team`, `player`, `position`, `source`

### `coverage.csv` — the missingness audit

`season`, `column`, `n_matches`, `frac_matches_all_zero`, `verdict`

**Applied per season, not per match.** A single match legitimately has zero red
cards, zero penalties and zero own goals; flagging those as missing would
discard real data. A column is `not_collected` only when it is entirely zero or
empty across **100%** of a season's matches, `suspect` above 90%, else `ok`.

### `match_coverage.csv`

Spine join: every `results.csv` Jordan match 2014+ with `fbref_match_id` or
null, plus `reason` for known gaps. This is what any downstream analysis must
consult before claiming a rate or a share.

## Derived features (Part 3, built after the above)

Per match, from lineups only — nothing here depends on advanced stats:

- starting-XI turnover (Jaccard distance vs previous match)
- minutes concentration (top-N share; HHI)
- mean and spread of starting-XI age
- debut count per window
- core continuity (share of XI who also started N matches ago)

## Hard limits, stated up front

- **No advanced stats across the 2022 boundary.** FBref credits StatsBomb in
  2019-season pages and Opta in 2023-season pages. The provider change lands
  near the treatment date, so any shift in a rate stat is unattributable.
  Lineups, minutes, goals, subs and cards are structural facts and unaffected.
- **No shot data before ~2024** for Jordan's AFC fixtures, so "did they create
  more chances" is unanswerable. Only "who was on the pitch".
- **The coverage hole is not random.** All 24 missing matches are competitive,
  and Arab Cup and WAFF are exactly the rotation-heavy tournaments where a
  second-string XI is most likely. Squad turnover measured from FBref alone
  will be biased downward. Hand-entering Arab Cup closes 10 of the 24.
- FBref launched June 2018 and backfilled; it warns its own totals may be
  incomplete.
