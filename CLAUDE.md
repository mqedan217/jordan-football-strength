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

Files live in `JordanData/`, not `src/`: `data.py`, `baseline.py`,
`trajectory.py`, plus `uncertainty.py` (bootstrap), `tournament.py` (match
importance), `importance.py` (the DiD), `fbref_fetch.py`, `fbref_parse.py`,
`squad_features.py`.
Outputs: `outputs/jordan_strength_trace.csv` (point estimates),
`outputs/jordan_strength_ci.csv` (with intervals — prefer this one).

Environment: `.venv` at the repo root, pinned in `requirements.txt`
(numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, soccerdata 1.9.1 for Part 2).
`data/results.csv` is the cached 49k-row spine and is gitignored.

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

### ✅ The attack-vs-defence claim — recomputed, now supported

An earlier decomposition (attack +0.53, defence −0.08, "the improvement is
entirely attacking") was computed by averaging point estimates across two
periods **with no standard errors at all**, and was banned pending a proper
recomputation. That condition has now been met (`JordanData/uncertainty.py`, 500
paired bootstrap draws), using the exact additive split
`strength = e^atk − e^−def`:

| component | 2015→2026 | 95% CI | |
|---|---|---|---|
| attacking contribution | +0.744 | [+0.238, +1.212] | excludes 0 |
| defensive contribution | −0.027 | [−0.255, +0.196] | includes 0 |
| **attack minus defence** | **+0.771** | **[+0.229, +1.282]** | **excludes 0** |

The third row is the test that matters — it compares two uncertain quantities
rather than eyeballing two point estimates, which is what the original failed
to do. Attack accounts for ~104% of the rise. **The claim may now be carried
forward.** The magnitudes differ from the banned version because that used a
different period and method; the conclusion survives.

### ⚠️ A claim that must NOT be made

**"2026 is Jordan's peak" is not supportable.** It is the highest point
estimate, but not distinguishable from its neighbours:

- 2026 vs 2023: +0.411, CI [−0.031, +0.867]
- 2026 vs 2025: +0.225, CI [−0.116, +0.562]

The defensible claim is the long-run rise (+0.717, CI [+0.203, +1.220]).
Describe 2026 as "at the top of the range", never as the peak. The raw-GD
divergence story is unaffected — that is about point estimates.

### Tournament effects — tested, rejected as a model term

Four variants (global friendly offset, + friendly-specific home advantage,
+ finals offset, and a per-team importance parameter) were tested on
rolling-origin held-out log score. **All four rejected.** See
`JordanData/tournament.py` and `outputs/tournament_comparison.csv`.

The per-team version is the interesting failure. Jordan does show a
competitive-match premium post-2022 (+1.09, CI [+0.27, +1.90]) that survives
opponent adjustment and holds within opponent-strength bands. But:

- the difference-in-differences vs pre-2022 is +0.86, CI [−0.06, +1.74] — and
  it grows monotonically the later the split date is placed
- across 161 teams the spread of that DiD has sd 0.735, while the average
  within-team estimation noise is sd 0.758. **Between-team variation is
  indistinguishable from zero** — no team is separable from any other on this
  measure, so Jordan's 88th-percentile rank is an artefact
- a single time-invariant per-team term would also smear a post-2022 behaviour
  back across 2015, corrupting the baseline the whole project measures from

Report it as a description of what happened in those 57 matches, tiered:
*supported* (the post-2022 premium), *not established* (that it is a change),
*ruled out* (that it is distinctive to Jordan). No further work on scorelines
can settle this — the signal is below the noise floor of ~15 matches a year.

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
era — they are missing, not zero.

**Apply this per season, never per match.** Verified against the 2026
Austria–Jordan report: Jordan's `PK`, `PKatt`, `CrdY` and `CrdR` are all zero
in that match because Jordan took no penalties and got no cards — those are
true zeros. Applied per match the rule deletes real data. The per-season form
is the one that works: for each column, the fraction of a season's matches
where it is entirely 0; 100% means not collected. Implemented in
`fbref_parse.coverage_audit`, written to `outputs/squad/coverage.csv`.

### 🚨 Data trap: the event feed's trailing label lies

Verified in the same report. FBref labels own goals *and* penalty goals with
the same trailer as real substitutions:

```
76’    Yazan Al-Arab Own Goal          — Substitute
90+12’ Marko Arnautović Penalty Kick   — Substitute
81’    Saed Al-Rosan for Abdallah Nasib — Substitute
```

Classify on the text *before* the em dash. Keying on the trailer or the CSS
class turns two goals into substitutions. Also: FBref files an own goal under
the team that benefits, so `side` is the scoring team while `player` is the
opponent who scored it.

### 🚨 Data trap: the scorebox is positional, and half of it is the opponent's

The scorebox carries **both** teams' manager and captain. `fbref_parse._scorebox`
collects them with two independent `re.findall`s and assigns by position —
`managers[0]` is home, `managers[1]` is away. `match_table` then renamed
`home_manager` straight to `manager`, which for every **away** match returns the
**opponent's** manager. The column held 81 distinct names where Jordan has had
about 13, with Roberto Mancini, Ralf Rangnick, Zlatko Dalić and Ange Postecoglou
filed as Jordan managers, and changed value 123 times in 174 matches.

The tell is spell structure: a manager column is correct when it forms
contiguous blocks, not when it alternates. It now gives **15 spells over 166
matches**, which reconciles with Jordan's actual coaching history.

Which side Jordan was on is resolved in `match_table._resolve_manager`, in order:

1. **the captain** — the season fixtures page is Jordan's, so its `Captain` is
   Jordan's by construction; the scorebox side whose captain it matches is
   Jordan's. Also accepts a captain recognised from Jordan's own appearance rows,
   which covers matches with no `Captain`. 135 matches.
2. **`Venue`** — FBref's own column, already Jordan-relative. 20 matches.
3. **`source == "wikipedia"`** — the Arab Cup rows are written from Jordan's
   match article, so their manager field is Jordan's already and neither rule
   above can see them (no opposing scorebox, neutral venue). 10 matches.
4. nothing — the row keeps a **null** manager rather than a guessed one. 8
   matches, all of which have no manager in the source at all.

Rules 1 and 2 agree wherever both fire, and there are zero rows resolving to
both sides. The chosen rule is recorded per match in `manager_side_source`, and
the opposing manager is kept in `opp_manager` (105 distinct, free of Jordan's).

🚨 **A lone block carries no side information.** Where FBref populated only one
team's metadata, exactly one manager and one captain parse, and both land in the
`home_` slots whatever team they belong to. Position is then meaningless, but the
pair came from the *same* block — so recognising the captain identifies the
manager. Without this, 2016-09-04 vs Bahrain files Abdullah Abu Zema, Jordan's
own caretaker, as the opposition. Note this is the *opposite* failure to rules 1
and 2, which is why the captain rule is restricted to rows with **both** captains.

**`Abdulla Mesfer` and `Abdullah Ahmed Mesfer` are one man** — the only manager
who alternates inside a single unbroken spell (2017-01-25 to 2017-10-10), which
is what a transliteration variant looks like and what a real change of manager
does not. Merged through an explicit `MANAGER_ALIASES` dict, visible in the
source, never a fuzzy match. Managers have no FBref id, so the
"never key by name" rule below cannot be applied to them — treat any manager
comparison as name-keyed and check new spells by hand.

### 🚨 Never key players by name

Arabic names get re-transliterated between seasons. Name-keying invents debuts
and corrupts every continuity measure. Use FBref's 8-hex player id, extracted
from the `/en/players/<id>/` hrefs — verified present on every row of both the
lineup and stats tables.

### 🚨 Confound: FBref changed data provider

2019-season pages credit advanced data to **StatsBomb**. 2023-season pages
credit **Opta**. The provider change lands near the 2022 treatment date. Any
measured shift in a stat could be a collection change rather than a real
change. This does not affect lineups/minutes/goals/subs (structural facts),
but it rules out using any advanced stat across the boundary.

### Coverage boundaries (verified)

- Lineups, minutes, subs, cards: back to at least 2017, friendlies included
- Possession: **27 of 165 matches, back to 2015** -- not "from ~Sept 2024"
  as previously recorded. Like every advanced stat here it is gated by
  competition, not date: Asian Cup 14/14, World Cup 3/3, WCQ 10/50, Asian Cup
  qualification 0/15, friendlies 0/83. A pre-2022 baseline therefore exists
  (Asian Cup 2015 and 2019) but is only 7 matches, against 20 after -- far too
  few to establish a trend given how much possession moves with the opponent
- Ball location (territory, touches by third, field tilt, passing networks):
  absent entirely, in every era
- Formation / opponent formation: recent seasons only
- xG: not available for AFC competitions
- FBref launched June 2018 and backfilled; it warns totals may be incomplete

**Open question — ANSWERED (2026-09-09).** The history page was fetched and
parsed. FBref holds Jordan continuously for seasons **2014–2023, 2025, 2026**,
totalling **165 distinct match reports**. Pre-2014 is sparse and unusable
(2011, 2010, 2007, 2006, 2004, 2002, 2000 only). The squad layer can therefore
cover both the 2015 baseline and the 2022 shift.

Formation and opponent formation turn out to be available much further back
than "recent seasons only" — the season fixtures table (`matchlogs_for`)
carries Date, Comp, Round, Venue, Result, GF, GA, Opponent, Poss, Attendance,
Captain, Formation, Opp Formation and Referee, so most match metadata comes
from 12 season pages rather than 165 match reports.

**Season labels are competition cycles, not calendar years.** There is no 2024
season: the 2023 Asian Cup (played Jan–Feb 2024) files under season 2023, and
the 2024–25 World Cup qualifiers file under season 2026. Key everything on
match date, never on season label.

**Coverage hole, and it is not random.** FBref tracks only World Cup, WCQ-AFC,
AFC Asian Cup, AFC Asian Cup qualification and Friendlies for Jordan. It does
not track Arab Cup (10), WAFF Championship (6), Palestine International
Championship (4), King's Cup (2), Jordan International Tournament (2). **All
24 are competitive; none are friendlies**, and Arab Cup and WAFF are exactly
the rotation-heavy tournaments where a second-string XI is most likely — so
squad turnover measured from FBref alone is biased downward. Use
`data/results.csv` as the spine and carry a per-match coverage flag.
Full extraction spec: `docs/squad_layer_spec.md`.

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
| Wikipedia (player pages) | 137 of 184 category members have no international by-year table. Tables truncated at source (Ali Olwan shows 31 caps vs 59 actual). Parser was fixed and validated — Amer Shafi 171/171, Al-Dardour 113/113 — the source is what's incomplete. |
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

### Arab Cup — recovered from Wikipedia match articles (2026-09-11)

FBref carries no Arab Cup at all: absent from Jordan's squad pages and from
FBref's own competition index. Wikipedia's *tournament* articles do carry full
match detail, and this is a different body of content from the per-player
by-year tables listed as a dead end above.

All **10** Arab Cup matches since 2015 are now extracted (`JordanData/wiki_arabcup.py`
-> `outputs/squad/arabcup_*.csv`): 2021 group x3 + quarter-final, 2025 group x3 +
quarter-final, semi-final and final. Exactly 11 starters per match, and all ten
scores agree with `data/results.csv`.

Parsing notes, each one a bug that failed silently before it was caught:
- lineups are plain wikitables, not templates; split on any table opening and
  keep the ones containing lineup rows, because the 2021 articles write
  `font-size: 90%` and the 2025 ones `font-size:90%`
- the bench separator is `'''Substitutes:'''` in some articles and
  `'''Substitutions:'''` in others, sometimes on the same page; matching one
  spelling counts the whole bench as starters (16 a side)
- a `^`-anchored row regex needs `re.M` when searched against a whole table

🚨 **Wikipedia supplies no player id.** Rows carry `player_id = NA` and
`source = "wikipedia"`. 29 of 35 names match an FBref player exactly; the other
6 are transliteration variants (`Yaseen Al-Bakhit` / `Yaseen Al Bakheet`) and
are listed in `arabcup_name_crosswalk.csv` with a suggested match and a
similarity score, **unconfirmed**. Do not merge these rows into the main
appearances table until a human sets `confirmed`. A close string is not proof,
and a wrong accept invents a player history.

### 🚫 Possession — checked exhaustively, NOT obtainable (2026-09-11)

Do not re-explore. Possession exists for **27 of 174** matches and cannot be
completed, because for most of Jordan's fixtures the measurement was never
taken. Sources checked:

| source | result |
|---|---|
| FBref | 27 matches, gated by competition (Asian Cup 14/14, World Cup 3/3, WCQ 10/50, Asian Cup qualification 0/15, friendlies 0/83) |
| Wikipedia results + tournament articles | **zero** occurrences of "possession", four probes |
| ESPN (2015 Jordan–Malta friendly) | lineups, scorers, referee — no possession, and a "1 shot" total in a 2–0 win |
| FootballCritic match-stats page (same match) | lineups, scorers, referee, stadium — no possession |
| `stats.the-afc.com` | domain no longer resolves |
| BeSoccer | **has it for recent matches** (2025 Arab Cup final: Jordan 42%) |
| Sofascore | checked manually — does not have it for the old matches |

Coverage by era is the tell: **0% (2011-14), 6% (2015-18), 9% (2019-22), 44%
(2023-26)**. That is the history of when these fixtures were instrumented, not
a gap in any one provider.

🚨 **A partial fill would make things worse, not better.** Anything obtainable is
concentrated in recent matches, so topping up 2023-26 from BeSoccer while
2011-18 stays empty turns any pre/post comparison into a comparison of "matches
with data" against "matches without" — and that boundary sits right at the
treatment date. There is no safe window to retreat to, unlike formation.

**Use possession descriptively for the 27 matches, never as a trend variable.**
It does say something real: Jordan is under 50% in 23 of the 27, including 30%
in the 2024 Asian Cup semi-final win over Korea and 27% against Argentina.
Whatever changed, it is not ball control — consistent with the rise being
attacking rather than defensive.

### Match state — built, and it says nothing (2026-09-13)

`JordanData/match_state.py` -> `outputs/squad/match_state.csv`. `events.csv`
carries **both** teams' goals with a minute and a running score, which is enough
to reconstruct the scoreboard through a match: time spent leading / level /
trailing, who scored first and when, goals for and against per 15-minute
window, and what happened after Jordan led or fell behind. **145 of 174
matches** (111 pre, 34 post); 377 goals.

Two decisions make it trustworthy, and both are reusable:

- **Which side was Jordan?** The feed is match-relative. Compare its final
  running score against Jordan's own `gf`/`ga` — the orientation that
  reproduces the result is Jordan's. 116 matches resolve this way with **no
  match failing both orientations**, which also proves the feed is complete
  wherever it exists. Level scorelines reproduce both and fall back to `venue`
  (29 more); level scorelines at neutral grounds stay unresolved rather than
  guessed, because a wrong orientation silently inverts every scored/conceded
  number for that match.
- **State comes from the running score, never the `side` label**, which means
  different things for different event types (own goals are filed under the
  team that benefits). Whichever score column advanced is the scoring team, so
  own goals need no special case.

🚨 **Goalless matches are not absent data.** A 0-0 has no goal events but is 90
minutes of match state — level throughout. Eighteen matches, all genuine 0-0
draws. Dropping them biases every state measure toward matches where something
happened.

**The result is a null, and it should be reported as one.** Block bootstrap on
six-month windows, pre vs post 2022-07:

| | pre | post | diff | 95% |
|---|---|---|---|---|
| share of time leading | 0.32 | 0.29 | −0.03 | [−0.23, +0.12] |
| share of time trailing | 0.17 | 0.24 | +0.07 | [−0.05, +0.23] |
| scores first | 0.54 | 0.47 | −0.07 | [−0.36, +0.14] |
| trailed → avoided defeat | 0.16 | 0.20 | +0.04 | [−0.16, +0.25] |
| led → won | 0.85 | 0.67 | −0.18 | [−0.72, +0.08] |

Twelve scoring/conceding timing comparisons were run; two cross zero narrowly
(conceded 31-45 +0.23 [+0.00, +0.52]; conceded 46-60 −0.11 [−0.20, −0.02]).
**At 95% across 12 tests that is what chance produces** — do not report either
as a finding. In particular the eye-catching raw pattern, conceding 0.34 → 0.56
per match in the 76-90 window, has interval [−0.06, +0.66] and does **not**
survive; read it against Armatas et al. (2009), who find 22.5% of all goals
fall in that window regardless.

Match state is therefore a **"what it isn't"** section, not a finding. That is
worth keeping: with n=34 post-2022 matches these intervals are the honest
result, and a null with an interval is more useful than a point estimate.

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

---

## Citations

Every method, data source and software dependency this project leans on is
listed in **`docs/references.md`**, with the file that uses it. Cite from there
rather than re-deriving a reference; add to it when a new method is introduced.

The three that carry the most weight: **Dixon & Coles (1997)** for the
time-decayed Poisson model, **Künsch (1989)** for the block bootstrap behind
every interval, and **Gotelli & Colwell (2001)** for rarefaction — without which
"fewer different scorers after 2022" is an artefact of the post era being
shorter, not a finding.
