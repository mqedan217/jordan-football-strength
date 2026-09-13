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

### 🚨 `gf_above_expected` is a model-lag statistic, not an improvement measure

`match_table` carries `gf_above_expected` = goals scored minus the model's
expectation for that fixture. It reads like "how much better than expected they
played" and it is **not** a measure of improvement. Do not use it as the outcome
for a pre/post comparison.

The model is refit every six months on data strictly before each match, so the
residual measures how far Jordan beat its own **contemporaneous forecast**. On
the 2015+ window:

| | pre | post |
|---|---|---|
| goals for | 1.449 | 1.756 |
| model expected_gf | 1.376 | **1.338** |
| gf_above_expected | +0.073 | **+0.417** |

Expected goals-for is essentially flat across the boundary, because Jordan's
attack rising and its opponents getting much stronger cancel out. A team
improving faster than a 730-day half-life can track shows a positive residual
for that reason alone, so the post-era value is largely the model lagging.

**The improvement measure is the model's own attack parameter**, which rises
0.229 [0.018, 0.399] in 2015 to 0.710 [0.468, 0.866] in 2026 — the +0.744
[+0.238, +1.212] already recorded above.

This cost a real analysis: a "symmetric top-k scorer removal" test was built on
`gf_above_expected` as the outcome, which answers *how much of the model's
forecast error those players accounted for* rather than whether the improvement
depended on them. Dropped. Questions about who scored are answerable in
**shares and counts**, which need no expectation at all — that is what notebook
03 uses.

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

### Match state — built, and the coverage trap it exposed (2026-09-13)

`JordanData/match_state.py` -> `outputs/squad/match_state.csv`. Reconstructs the
scoreboard through each match from goal minutes on **both** sides: time leading
/ level / trailing, first goal, goals for and against per 15-minute window, and
what followed a lead or a deficit. **155 of 174 matches, 409 goals.**

Three decisions make it trustworthy, all reusable:

- **Which side was Jordan?** The event feed is match-relative. Compare its final
  running score against Jordan's own `gf`/`ga` — the orientation reproducing the
  result is Jordan's (116 matches). No match fails both, which also proves the
  feed is complete wherever it exists. Level scorelines reproduce both and fall
  back to `venue` (29); level scorelines at neutral grounds stay unresolved
  rather than guessed, since a wrong orientation silently inverts every
  scored/conceded number.
- **State comes from the running score, never the `side` label**, which means
  different things for different event types. Whichever score column advanced is
  the scoring team, so own goals need no special case.
- **Goalless matches are not absent data.** A 0-0 is 90 minutes of state, level
  throughout. 18 matches, all genuine 0-0 draws.

🚨 **The coverage trap — check representativeness BEFORE reporting a null.**
The first version covered 145 matches and returned a clean null on everything.
It was wrong, not because the numbers were wrong but because the covered subset
was not representative:

| | n | gd_above_expected |
|---|---|---|
| post, with event data | 34 | **+0.126** |
| post, without | 11 | **+1.062** |

FBref carries no Arab Cup, so the entire 2025 run — UAE, Kuwait, Egypt (+3.47),
Iraq, Saudi Arabia, Morocco — plus the 2024 Asian Cup draw with South Korea sat
outside the feed. On covered matches the improvement read +0.126 post against
+0.125 pre: **no improvement at all**, because the improvement lived in the
matches the feed could not see. Same failure mode as possession: coverage
correlated with the outcome, boundary near the treatment date.

Fixed by reading the opponent's goals from the Wikipedia `Football box`, which
was already cached — only Jordan's half had been parsed. Coverage 145 -> 155 and
**89% in both eras** rather than 76% post against 86% pre. Covered subset now
reads +0.293 post against +0.126 pre.

**Findings, on the repaired data:**

- **Jordan concedes 33.9% of all goals against in the 76-90 window**, against
  22.0% of its own goals scored there — and Armatas et al. (2009) put the
  all-football benchmark at 22.5%. The attack's timing is unremarkable; the
  defence's is not. This is a description of all 155 matches, not a change.
- **Scoring rate per 90 minutes actually spent in each state** — a rate, so
  "trails more, therefore scores more while trailing" cannot fake it:

  | | leading | level | trailing |
  |---|---|---|---|
  | pre | 2.47 | 1.13 | **0.85** |
  | post | 1.96 | 1.53 | **1.40** |

  Pre-2022 Jordan looks like a front-runner: ruthless ahead, and *below its
  level-state rate* when behind, the opposite of the Lago-Penas pattern.
  ❌ **Bootstrapped, and it does not survive.** On the 2015+ window the
  trailing/level ratio difference is +0.19, 95% [-1.01, +0.99], resting on 24
  goals scored while trailing in eleven years. The point estimates tell a good
  story and the interval cannot carry it. Reported in notebook 03 as a null.
- **Blowouts did not change.** Win by 3+ −0.008 [−0.152, +0.124]; lose by 3+
  −0.002 [−0.068, +0.097]; margin sd 2.34 -> 2.36. What changed is total goals
  per match, **2.50 -> 3.05 [+0.08, +1.04]** — games got more open.
- **Still null:** time leading, time trailing, scores first, recovery from
  behind. Led->won falls 0.841 -> 0.708 [−0.446, +0.082], suggestive only. The
  "mentality" story stays dead; the change is in how flat the scoring rate is
  across states, which is a squad-quality story.

🚨 **Opponent-adjust state measures before comparing eras.** Raw time-trailing
rises +0.067 pre to post; simulating each scoreline from the model's own
`expected_gf`/`expected_ga` (goal minutes drawn from the pooled empirical
distribution, identical across eras) collapses it to **−0.006**. The raw rise
was Argentina and Switzerland, not Jordan. Scratch implementation and a power
calculation are in `docs/scratch/state_adjusted.py` -- it runs from the repo
root with hardcoded relative paths; give it the ROOT/SQUAD treatment and move
it into `JordanData/` before relying on it.

**Still uncovered: 19 matches** — 14 friendlies and 5 competitive, all FBref
reports with no event feed. Coverage is now balanced across eras, so this is
noise rather than bias, but it caps n at 40 post-2022.

### Notebook 03 — where the goals came from (2026-09-14)

`notebooks/03_where_the_goals_came_from.ipynb`, built by a script in the session
scratchpad, 27 cells and 4 figures. Window 2015+, split 2022-07-01, 98 pre / 45
post matches. Every code cell recomputes its own numbers so the prose cannot
drift from the output.

**Supported:**

| | estimate | 95% |
|---|---|---|
| midfielder share of goals | +0.148 | [+0.039, +0.315] |
| minutes: top-11 share | +0.247 | [+0.135, +0.351] |
| familiarity rate (pairs) | +0.217 | [+0.113, +0.337] |
| appearance rate (starters) | +0.209 | [+0.106, +0.319] |

Rarefied to equal goals (75): distinct scorers 16 vs [21, 28], top-3 share 0.640
vs [0.333, 0.480], HHI 0.153 vs [0.068, 0.113] — all **outside**.

🚨 **Top-1 share is 0.253 against a rarefied [0.147, 0.267] — inside.** Jordan is
NOT a one-man team, and the popular version of this claim is not supportable.
The breadth measures move and the single-player measure does not; that is a
narrowing onto a small *group*.

🚨 **It is not a clean replacement of scorers.** Al-Dardour (29 goals, then 0)
and Baha' Faisal (17, then 0) vanish, but the new top three were all already
scoring before 2022 — Olwan 8 -> 19, Al-Naimat 5 -> 15, and Al-Taamari is in the
top three of **both** eras (10 -> 14). A wide distribution collapsing onto three
players who were already in it, not a new generation arriving. Earlier notes in
this file described a replacement; that was an artefact of using the 2011+
window, where Al-Taamari's pre-2022 share looks smaller.

Nulls reported as nulls: match state, heavy wins and defeats, goalkeeper,
discipline, and the state scoring rates.

### ✅ The level-scoreline collapse — the project's one actionable finding

Time and state were used separately at first, and neither alone explains the
late conceding. **Crossed, they do.** `match_state.state_time_cells()` ->
`outputs/squad/state_time_cells.csv` gives minutes AT RISK and goals per
(15-minute window x scoreboard state), which is what a rate needs: a team that
spends the last fifteen minutes level has more chance to concede a level-state
goal simply by being there.

Goals conceded per 90 minutes at risk, 2015+:

| window | leading | level | trailing |
|---|---|---|---|
| 61-75 | 0.89 | 1.10 | 1.45 |
| **76-90** | **0.90** | **4.30** | 2.42 |

- **while level: 3.92x, 95% [1.80, 11.47]**
- **while leading: 1.01x, 95% [0.45, 2.22]** — the control, flat

The control is what makes this more than "goals rise late". Armatas et al.
(2009) find that everywhere, but a general effect would lift the leading row
too, and it does not.

🚨 **The internal control is the decisive one.** If level matches simply get
frantic, Jordan should score more late as well. It does the opposite: while
level, 61-75 -> 76-90, scoring goes **1.41 -> 0.21** per 90 while conceding goes
**1.10 -> 4.30**. Since 2015 Jordan has scored **one** goal while level in the
last fifteen minutes and conceded **twenty**.

Verified a second way, independent of the rate arithmetic: **36 matches level at
the 75th minute -> 2W 20D 14L, 0.72 points of a possible 3.** Present in both
eras (pre 1-15-8, post 1-5-6), so this is **not** a pre/post change — the model
says Jordan improved a lot over this period and this failure came with it.

Opponent-adjusted, because that check has caught several claims in this project:
those 36 fixtures have mean `expected_gd` **+0.20**, i.e. Jordan was *favoured*.
Simulating the remaining 15 minutes at each fixture's own model goal rates gives
an expected **1.19** points against **0.72** observed — a shortfall of
**-0.47, 95% [-0.69, -0.22]**, about 17 points across the 36.

Robustness: minutes at risk stop at 90 for every match while stoppage is always
played, so the last window's rate is inflated for everyone. Adding 8 minutes of
stoppage to every level match still leaves the spike at **2.69x**.

🚨 **Do not explain it.** The obvious reading — a team chasing a winner is caught
on the break — needs shot locations or possession to test, and neither exists
for these matches. It is a description of what happens, not a mechanism.

`goal_timelines()` is now the public way to get {match_id: (timeline, side,
source)}; three scripts had been rebuilding it from private helpers.

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
