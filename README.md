# Jordan National Team — team strength and squad composition, 2015–2026

Jordan's results improved sharply from around mid-2022: a 2023 Asian Cup final
and a first World Cup qualification. This project asks whether that shift is
real once opponent quality is controlled for, when it happened, and whether it
coincides with a change in squad composition.

**It cannot and does not identify a cause.** Squad change, coaching change and
scheduling all move together, and with ~15 matches a year there is no design
here that separates them. Everything below is coincidence with uncertainty.

## The headline

2026 is simultaneously Jordan's **worst raw goal difference** and the model's
**highest strength estimate**. Both are correct — the 2026 opponents were
Argentina, Austria, Algeria, Switzerland and Colombia, and Jordan lost all five
narrowly. Raw results say collapse; opponent adjustment says otherwise.

| year | raw GD/match | model strength | 95% CI |
|------|--------------|----------------|--------|
| 2015 | +0.94 | 0.61 | [+0.28, +0.90] |
| 2023 | −0.71 | 0.93 | [+0.57, +1.27] |
| 2025 | +0.37 | 1.14 | [+0.68, +1.50] |
| 2026 | **−1.43** | **1.36** | [+0.91, +1.74] |

## What the numbers do and do not support

**Supported — the rise is real.** 2015 → 2026 is **+0.717, 95% CI
[+0.203, +1.220]**.

**Supported — the improvement is attacking, not defensive.** Using the exact
additive split `strength = e^atk − e^−def`:

| component | change | 95% CI | |
|---|---|---|---|
| attacking contribution | +0.744 | [+0.238, +1.212] | excludes 0 |
| defensive contribution | −0.027 | [−0.255, +0.196] | includes 0 |
| **attack minus defence** | **+0.771** | **[+0.229, +1.282]** | **excludes 0** |

**Not supported — "2026 is the peak."** It is the highest point estimate but
not separable from its neighbours: 2026 vs 2023 is +0.411 [−0.031, +0.867],
2026 vs 2025 is +0.225 [−0.116, +0.562]. Say "at the top of the range".

**Described, but not established — the competitive-match premium.** Post-2022
Jordan beats its own rating in competitive matches relative to friendlies by
+1.09 [+0.27, +1.90], and this survives opponent adjustment. But the
difference-in-differences against the pre-2022 period is +0.86 [−0.06, +1.74],
it grows the later the split date is placed, and — decisively — across 161
teams the spread of that statistic (sd 0.735) is *smaller* than the average
within-team estimation noise (sd 0.758). Between-team variation is
indistinguishable from zero, so no team is separable from any other on this
measure. Report it as a description of 57 matches, not a trait.

## Method

Time-weighted Poisson attack/defence ratings, penalised MLE, fit on all 49,547
international matches:

```
mu_home = exp(atk_home - def_away + adv * is_home)
mu_away = exp(atk_away - def_home)
```

- matches weighted `0.5 ** (days_before / 730)`, so ratings track form
- ridge (0.05) identifies the model and shrinks sparse teams toward the mean —
  a frequentist stand-in for partial pooling
- refit every 6 months on **only** matches before the reference date, so the
  curve is one you could have drawn on the day
- intervals from 500 bootstrap draws, resampling matches and re-running the
  whole trace per draw so the draws are paired across dates and differences
  between dates get intervals of their own

This is Dixon & Coles (1997) with the time-decay term, after Maher (1982); the
Bayesian version follows Baio & Blangiardo (2010).

**Tournament effects were tested and rejected.** Four variants — a global
friendly offset, a friendly-specific home advantage, a finals offset, and a
per-team importance parameter — all failed on rolling-origin held-out log
score. Details in `JordanData/tournament.py`.

## Running it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt

cd JordanData
python trajectory.py      # strength trace (point estimates)
python uncertainty.py     # bootstrap intervals, ~30 min
python tournament.py      # model comparison
python importance.py      # competitive/friendly difference-in-differences

python fbref_fetch.py     # squad layer: scrape (slow, resumable, cached)
python fbref_parse.py     # HTML -> tidy tables
python squad_features.py  # turnover, minutes concentration, age, debuts
```

`fbref_fetch.py` waits 12–25s between pages and rests every 20. It never
re-requests anything already in `cache/raw/`, so interrupting it is safe —
re-run to resume.

## Limitations

- **No causal identification.** Stated first because it matters most.
- **No claim without an interval.** With ~15 matches a year, wide intervals are
  the honest result, not a failure.
- **A uniformly-zero column is missing, not zero** — but this must be judged
  per season, not per match. A single match legitimately has no red cards.
- **No shot data before ~2024** for Jordan's AFC fixtures, so "did they create
  more chances" is unanswerable. Only "who was on the pitch".
- **FBref changed provider** (StatsBomb → Opta) near the 2022 treatment date,
  which rules out any advanced stat across the boundary. Lineups, minutes,
  goals, subs and cards are structural facts and unaffected.
- **The FBref coverage hole is not random.** All 24 uncovered matches are
  competitive — Arab Cup, WAFF and similar — and those are exactly the
  rotation-heavy fixtures, so squad turnover measured from FBref alone is
  biased downward.
- **Team names are normalised to current names** in the source data, so
  historical matches appear under modern country names. Harmless for 2015+.
- **The placebo is not yet run.** Comparing Jordan against Iraq, Palestine,
  Oman and Bahrain requires scraping those squads too; until then, no claim
  that a squad pattern is distinctive to Jordan is supportable.

## Data

`martj42/international_results` (49,547 matches, 1872–Aug 2026) for scores.
FBref squad `3e22f0fa` for lineups. The companion `goalscorers.csv` covers only
~25% of Jordan matches and is not used; `shootouts.csv` is 62% null and is not
used.
