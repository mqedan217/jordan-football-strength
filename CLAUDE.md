# Jordan National Team — squad composition and team strength, 2015–2026

## What this project is

Jordan's results improved sharply from roughly mid-2022: 2023 Asian Cup final,
first World Cup qualification. The question is whether that shift is real once
opponent quality is controlled for, and if so, whether it coincides with a
change in squad composition.

Two halves:

1. **Strength model (built, working).** Opponent-adjusted team strength over
   time from scores alone. Establishes whether and when a shift happened.
2. **Squad layer (not built).** Per-match lineups from FBref, used to measure
   squad turnover, minutes concentration, and age profile across the shift.

The claim at the end should be a coincidence-with-uncertainty, not a causal
one. See "Limitations" — this design cannot identify a cause.

---

## Part 1 — what already exists

### Data
`martj42/international_results` (GitHub, CSV). 49,547 international matches,
1872 to Aug 2026. Zero nulls. Columns: date, home_team, away_team, home_score,
away_score, tournament, city, country, neutral.

Jordan: 488 matches, 230 since 2010. Venue split 175 home / 161 neutral /
152 away. Record 180W 134D 174L.

Two companion files in the same repo:
- `goalscorers.csv` — **only covers ~25% of Jordan matches** (159 of 524 goals
  since 2000). Do not rely on it.
- `shootouts.csv` — 683 rows, `first_shooter` 62% null. Ignore.

Caveat: team names are normalised to *current* names, so historical matches
appear under modern country names. Harmless for 2015+, note it in the README.

### Model
Time-weighted Poisson attack/defence, penalised MLE, fit on all 49k matches:

```
mu_home = exp(atk_home - def_away + adv * is_home)
mu_away = exp(atk_away - def_home)
```

- Matches weighted by `0.5 ** (days_before / half_life)`, half_life = 730d
- L2 penalty (ridge = 0.05) identifies the model and shrinks sparse teams
  toward the global mean — frequentist stand-in for partial pooling
- Refit every 6 months, each fit using ONLY matches before the reference date
  (no lookahead)
- This is Dixon–Coles (1997) with the time-decay term; the Bayesian version
  planned for later follows Baio & Blangiardo (2010). Cite both.

Files: `src/data.py`, `src/baseline.py`, `src/trajectory.py`.
Output: `outputs/jordan_strength_trace.csv`.

### The finding that motivates everything

| year | raw GD/match | model strength |
|------|--------------|----------------|
| 2015 | +0.94 | 0.61 |
| 2023 | −0.71 | 0.93 |
| 2025 | +0.37 | 1.14 |
| 2026 | **−1.43** | **1.36** |

2026 is Jordan's worst raw goal difference and the model's highest strength
estimate. Both are correct: the opponents were Argentina, Austria, Algeria,
Switzerland, Colombia. Raw results say collapse, opponent adjustment says peak.

That divergence is the project's headline. It is also the argument for why the
squad layer is worth building.

### ⚠️ A result that must NOT be reused

An earlier attack-vs-defence decomposition (attack +0.53, defence −0.08,
"the improvement is entirely attacking") was computed by averaging point
estimates across two periods **with no standard errors at all**. With ~15
matches/year that difference may not be distinguishable from zero. Do not
carry this claim forward unless it is recomputed with uncertainty. It is
currently unsupported.

---

## Part 2 — the squad layer (to build)

### Source: FBref

Squad id `3e22f0fa`. Competition ids: AFC Asian Cup `664`, Asian Cup
qualification `665`. Friendlies exist as their own season slug (e.g.
`/squads/3e22f0fa/2017/s3691/`).

Match report URL pattern (verified):
`fbref.com/en/matches/<match_id>/Jordan-Cambodia-March-28-2017-Asian-Cup-qualification`

### Verified contents of a 2017 match report

Checked directly: Jordan 7–0 Cambodia, 2017-03-28, match id `bc1d8695`.

**Populated:**
- Starting XI — player name, shirt number, position (MF/FW/DF/GK), minutes
- Player age on match day, `YY-DDD` format (years-days) — parse this, it's useful
- Unused reserves, by name and position
- Goals with minute and scorer
- Substitutions with minute, player on, player off
- Cards with minute
- Manager, captain, referee + assistants, attendance, venue
- Goalkeeper: GA only

**Present as columns but EMPTY for 2017:**
`Sh, SoT, Ast, PK, PKatt, Fls, Fld, Off, Crs, TklW, Int, PKwon, PKcon`,
and GK `SoTA, Saves, Save%`.

### 🚨 Data trap: empty renders as 0

Jordan won 7–0 and every `Ast` cell reads 0. Assists were not recorded in this
era — they are missing, not zero. **Any column that is uniformly 0 across a
match must be treated as missing until proven otherwise.** Write an explicit
check: for each column, per season, compute the fraction of matches where the
column is entirely 0. Anything at 100% is not collected.

### 🚨 Confound: FBref changed data provider

2019-season pages credit advanced data to **StatsBomb**. 2023-season pages
credit **Opta**. The provider change lands near the 2022 treatment date. Any
measured shift in a stat could be a collection change rather than a real
change. This does not affect lineups/minutes/goals/subs (structural facts),
but it rules out using any advanced stat across the boundary.

### Coverage boundaries (verified)

- Lineups, minutes, subs, cards: back to at least 2017, friendlies included
- Possession: only from ~Sept 2024
- Formation / opponent formation: recent seasons only
- xG: not available for AFC competitions
- FBref launched June 2018 and backfilled; it warns totals may be incomplete

**Open question:** the season-by-season history page
(`/squads/3e22f0fa/history/`) lists every season FBref holds for Jordan. This
has NOT been checked and determines the true start year. Check it first.

### Scraping

**Use `soccerdata`, not raw requests.** FBref actively blocks automated access
— it started returning bot-detection errors after two consecutive fetches
during research. `soccerdata` respects their rate limits.

```
pip install soccerdata
```

Read FBref's Data Usage Terms before automating. Cache every response to disk;
never re-fetch a page you already have.

---

## Part 3 — analysis plan

From lineups, per match, build:
- squad turnover (Jaccard distance of starting XI vs previous match)
- minutes concentration (share of minutes to top-N players; HHI)
- mean and spread of starting XI age
- debut count per window
- core continuity (share of XI who also started N matches ago)

Then align these to the 6-month strength trace and ask whether composition
changes track the strength shift.

Required checks:
- **Placebo**: run the same pipeline on 3–4 comparable AFC nations (Iraq,
  Palestine, Oman, Bahrain). If every team shows the same pattern, there is no
  Jordan effect.
- **Uncertainty**: every reported difference needs an interval. n ≈ 15
  matches/year. Wide intervals are the honest result.
- **Missingness audit**: the uniform-zero check above, per column per season.

---

## Part 4 — dead ends (do NOT re-explore)

All checked during research. Each failed for the reason given.

| source | verdict |
|---|---|
| RSSSF | Only 23 of 67 listed players have match-by-match pages. Coverage 64% (2011) → 13% (2025). **Al-Taamari and Al-Naimat have no pages.** Main results page dead since 2008. |
| Wikipedia | 137 of 184 category members have no international by-year table. Tables truncated at source (Ali Olwan shows 31 caps vs 59 actual). Parser was fixed and validated — Amer Shafi 171/171, Al-Dardour 113/113 — the source is what's incomplete. |
| FootyStats | 10 friendlies, season aggregates only, corners/cards paywalled, percentages computed on n=2. Paid product, scraping breaches terms. |
| API-Football | Free tier serves **2022–2024 only**. Lineups/events flagged true from 2011 but need paid tier. Pro is $19/mo, 7500 req/day — ~460 requests would backfill everything. **Terms forbid redistributing raw data** (scraper + derived features are fine). This remains the only complete option. |
| StatsBomb open | Rich (3947 events/match, xG, freeze frames) but La Liga = Messi dataset (Barcelona only, except full 2015/16). UCL = finals only. No Jordan. |
| Understat | robots.txt disallows automated access. La Liga only. |
| openfootball | FT + HT scores only, no minutes. Public domain. Fine as a spine, no squad data. |
| football-data.co.uk | HT/FT + shots + cards + **bookmaker odds**, ~30 seasons. Domestic leagues only, no internationals. Shots are match totals so cannot be split by half. |
| national-football-teams.com | Stale (shows 51 caps for a player with 84). |
| AFC technical reports | 95pp narrative + interviews. Not a dataset. The 200pp "post-tournament report" is hotels and accreditation. |
| Google sports panel | Has the data back to 2015 but no public API; scraping breaches ToS. |
| `withqwerty/reep` | ID crosswalk only (444k people, 45k teams, 212 competitions). No match data. Useful only for joining across providers. v0 register frozen since Apr 2026. |

---

## Limitations for the README

State these plainly rather than working around them:

- **No causal identification.** Squad change, coaching change, and scheduling
  all move together. With ~15 matches/year and many candidate explanations,
  this design shows coincidence, not cause. Say so.
- **No shot data before ~2024**, so "did they create more" is unanswerable.
  Only "who was on the pitch and how did that change".
- **FBref coverage is incomplete by its own admission** and backfilled from a
  2018 launch, so pre-2018 data may be thinner or absent.
- **Provider change (StatsBomb → Opta)** near the treatment date rules out
  advanced stats across the boundary.
- **A repo with no events is not necessarily a failure** — the analogous
  caution applies to any absence-based inference here.

## Conventions

- Cache every fetched page to disk; the pipeline must be resumable and must
  never re-request a cached page.
- No claim without an interval.
- Any column that is uniformly zero is missing until proven otherwise.
- Cite Dixon & Coles (1997), Maher (1982), Baio & Blangiardo (2010).
