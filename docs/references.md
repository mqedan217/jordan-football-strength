# References

Everything this project borrows, and where it is used. Each entry names the
file or notebook that depends on it, so a claim can be traced back to the method
it rests on.

---

## Data sources

| source | what it gives | where | licence / terms |
|---|---|---|---|
| [martj42/international_results](https://github.com/martj42/international_results) | 49,547 international matches, 1872–2026: date, teams, score, tournament, venue, neutral flag | `data/results.csv`, the spine for every model fit | CC BY 4.0 |
| [FBref / Sports Reference](https://fbref.com/en/squads/3e22f0fa/Jordan-Men-Stats) | Jordan squad pages and match reports: lineups, minutes, substitutions, cards, goals with minute, formations, captain, manager, attendance, referee | `fbref_fetch.py`, `fbref_parse.py` | see FBref's Data Usage Terms; pages cached to disk, never re-requested |
| Wikipedia — [2021](https://en.wikipedia.org/wiki/2021_FIFA_Arab_Cup) and [2025 Arab Cup](https://en.wikipedia.org/wiki/2025_FIFA_Arab_Cup) tournament articles | the 10 Arab Cup matches FBref does not carry at all: lineups, scorers, subs, referee, attendance | `wiki_arabcup.py` | CC BY-SA 4.0 |
| Wikipedia — Jordan national football team results articles | attendance, referee and goalscorers for matches with no FBref report | `wiki_results.py` | CC BY-SA 4.0 |

### Software

| package | role |
|---|---|
| [`soccerdata`](https://github.com/probberechts/soccerdata) (Probberechts) | FBref access with rate limiting. FBref blocks plain HTTP clients; this is the supported route |
| numpy, pandas, scipy | the model, the bootstrap, the tables |
| matplotlib | figures |

---

## The strength model

The attack/defence Poisson model with time decay, penalised MLE, refit on a
rolling origin. `baseline.py`, `trajectory.py`.

- **Maher, M. J. (1982).** Modelling association football scores.
  *Statistica Neerlandica*, 36(3), 109–118.
  The independent-Poisson attack/defence formulation this is built on.
- **Dixon, M. J. & Coles, S. G. (1997).** Modelling association football scores
  and inefficiencies in the football betting market. *Journal of the Royal
  Statistical Society: Series C*, 46(2), 265–280.
  The exponential time-decay weighting — `0.5 ** (days_before / half_life)`,
  half-life 730 days — and the low-score dependence correction (which this
  project does not apply; see Limitations).
- **Baio, G. & Blangiardo, M. (2010).** Bayesian hierarchical model for the
  prediction of football results. *Journal of Applied Statistics*, 37(2),
  253–264. The hierarchical version, and the shrinkage argument the ridge
  penalty here stands in for.
- **Hoerl, A. E. & Kennard, R. W. (1970).** Ridge regression: biased estimation
  for nonorthogonal problems. *Technometrics*, 12(1), 55–67.
  The L2 penalty (ridge = 0.05) that identifies the model and pulls sparse
  teams toward the global mean.
- **Efron, B. & Morris, C. (1975).** Data analysis using Stein's estimator and
  its generalizations. *JASA*, 70(350), 311–319.
  Why shrinking a noisy per-team estimate toward the mean beats not shrinking
  it — the justification for treating the ridge term as partial pooling.
- **Gneiting, T. & Raftery, A. E. (2007).** Strictly proper scoring rules,
  prediction, and estimation. *JASA*, 102(477), 359–378.
  The held-out log score used to accept or reject model variants
  (`tournament.py`). All four tournament variants were rejected on it.

## Uncertainty

`uncertainty.py`, `importance.py`, and every interval in the notebooks.

- **Efron, B. (1979).** Bootstrap methods: another look at the jackknife.
  *Annals of Statistics*, 7(1), 1–26.
- **Künsch, H. R. (1989).** [The jackknife and the bootstrap for general
  stationary observations](https://projecteuclid.org/journals/annals-of-statistics/volume-17/issue-3/The-Jackknife-and-the-Bootstrap-for-General-Stationary-Observations/10.1214/aos/1176347265.full).
  *Annals of Statistics*, 17(3), 1217–1241.
  The moving-block bootstrap. Match results within a six-month window are not
  independent, so blocks are resampled rather than matches.
- **Politis, D. N. & Romano, J. P. (1994).** The stationary bootstrap. *JASA*,
  89(428), 1303–1313. Variable block length, for the autocorrelated series.

The **paired** bootstrap design — resample the full 49k dataset once per draw
and re-run the entire trace, so that differences between dates and between
teams get their own intervals rather than being differenced across independent
resamples — follows standard practice for comparing dependent estimates.

## Concentration and diversity

Used in notebook 04 to ask whether the goals came from fewer players.

- **Gotelli, N. J. & Colwell, R. K. (2001).** [Quantifying biodiversity:
  procedures and pitfalls in the measurement and comparison of species
  richness](https://www.uvm.edu/~ngotelli/manuscriptpdfs/Chapter%204.pdf).
  *Ecology Letters*, 4(4), 379–391.
  **The method this project leans on hardest outside the model itself.**
  "How many distinct scorers" is species richness, and richness rises with
  sample size no matter what the underlying diversity is. Jordan's post-2022
  era has 75 goals against the pre era's 182, so the two counts are not
  comparable as observed. Rarefaction — repeatedly down-sampling the larger
  era to the smaller era's size and comparing distributions — is the
  correction. Without it the finding is an artefact of the era lengths.
- **Hurlbert, S. H. (1971).** The nonconcept of species diversity: a critique
  and alternative parameters. *Ecology*, 52(4), 577–586.
  The original statement of why raw richness comparisons mislead.
- **Herfindahl–Hirschman index.** Sum of squared shares, borrowed from
  industrial concentration. Used for both goal shares and minutes shares
  because it is sample-size sensitive in the same way and gets the same
  rarefaction treatment.

## Squad composition, continuity and familiarity

- **Reassessing the impact of shared experience in football** (2025).
  *Journal of Big Data*, 12.
  [doi:10.1186/s40537-025-01239-x](https://link.springer.com/article/10.1186/s40537-025-01239-x).
  Defines team familiarity as pairwise prior appearances together, aggregated
  over the team. Used in notebook 03, §1.3. This project's `fam_rate` is that
  measure, restricted to a
  24-month lookback and divided by matches available in the window — raw
  pairwise counts accumulate with time and FBref's coverage starts in 2011, so
  the unadjusted version rises for reasons that have nothing to do with Jordan.
  The paper's own conclusion is a caution worth carrying: at match level,
  shared experience did **not** predict results once team strength was
  controlled for.
- **How team familiarity mitigates negative consequences of team composition
  disruptions: an analysis of Premier League teams** (2025).
  [PMC11999323](https://pmc.ncbi.nlm.nih.gov/articles/PMC11999323/).
- **Does stability foster team performance?**
  [van Vugt et al.](https://www.professormarkvanvugt.com/images/files/DoesStabilityFosterTeamPerformance.pdf)
  Evidence for an inverted-U between stability and performance — the reason
  "more settled is better" is not assumed here.
- **CIES Football Observatory**, [squad stability monthly report](https://football-observatory.com/New-Monthly-Report-highlights-the-importance-of).
  Measures stability as the share of players recruited within a year, and puts
  champions near 25%. A club-transfer metric with no direct national-team
  analogue, cited for the shape of the relationship, not the threshold.
  Used in notebook 05, §3, as the caution against reading "send players abroad"
  off a null: moving players abroad reduces the availability and continuity that
  notebook 03 found most clearly went up.

## Match state and goal timing

`match_state.py`, and part 3 of notebook 04.

- **Lago-Peñas, C. & Gómez-López, M. (2014).** [How important is it to score a
  goal? The influence of the scoreline on match
  performance in elite soccer](https://journals.sagepub.com/doi/abs/10.2466/23.27.PMS.119c32z1).
  *Perceptual and Motor Skills*, 119(3), 774–784.
  Match status drives possession and territory: teams that go behind take on
  more of the ball. This is why the possession figures in this project are
  descriptive only — a low possession share may be the scoreline, not a style.
  It is also why notebook 04's level-scoreline expectation is qualified: a
  Poisson simulation at a constant rate is state-blind by construction, and
  match status is precisely what it ignores.
- **Armatas, V., Yiannakos, A., Skoufas, D. & Papadopoulou, S. (2009).**
  Evaluation of goals scored in top ranking soccer matches: Greek "Super
  League" 2006-07. *Serbian Journal of Sports Sciences*, 3(1), 39–43.
  Goals rise steadily through a match, peaking at **22.5% in the 76–90 minute
  window**. Any era difference in Jordan's scoring or conceding profile has to
  be read against that baseline rather than against a flat expectation.

  🚨 These two between them motivate the qualification on notebook 04's
  level-scoreline shortfall, but they do not carry it. Neither paper reports a
  late-goal rate conditioned on being level at the 75th minute specifically, so
  the argument is made on Jordan's own data instead: the model predicts 0.38
  goals and 0.71 draws after the 75th where 0.69 and 0.56 are observed. Cite the
  calibration check, not the literature, for the size of the effect.
- **Myers, B. R. (2012).** A proposed decision rule for the timing of soccer
  substitutions. *Journal of Quantitative Analysis in Sports*, 8(1).
  The 58/73/79-minute rule. **Not applied here** — see below. Both papers are
  cited in notebook 05, §2, as the reason substitutions were excluded rather
  than tested.
- **Silva, R. M. & Swartz, T. B. (2016).** [Analysis of substitution times in
  soccer](https://www.sfu.ca/~tswartz/papers/substitution.pdf).
  *Journal of Quantitative Analysis in Sports*.
  The critique of Myers: substitution timing cannot be judged without team
  strength and game state. Between this and the 2020 five-substitute rule
  change, which sits two years before the treatment date, substitution timing
  was **dropped from this project** rather than reported.

## Design

- **Placebo tests.** The strength model is run on Iraq, Syria and UAE over the
  same window (notebook 02, §3), and the competitive-match DiD is run across
  161 teams (`importance.py`). The second one is what ruled out the "turns up
  when it matters" claim: the cross-team spread of the effect (sd 0.735) is
  indistinguishable from within-team estimation noise (sd 0.758).
- **Difference-in-differences** is used descriptively for the competitive/
  friendly split and is reported as *not established*, because the estimate
  grows monotonically with the split date — a sign the parallel-trends
  assumption does not hold here.

---

## Where each reference is used, by notebook

| notebook | leans on |
|---|---|
| **01 the data** | no methods; the data-source table at the top of this file |
| **02 the strength model** | Maher (1982), Dixon & Coles (1997), Baio & Blangiardo (2010), Hoerl & Kennard (1970), Efron & Morris (1975), Gneiting & Raftery (2007), Efron (1979), Künsch (1989) |
| **03 the squad and the opposition** | Künsch (1989); *Journal of Big Data* (2025) shared experience; van Vugt et al.; CIES Football Observatory |
| **04 the goals** | Gotelli & Colwell (2001), Hurlbert (1971), HHI; Armatas et al. (2009); Lago-Peñas & Gómez-López (2014); Künsch (1989) |
| **05 how Jordan can improve** | Lago-Peñas & Gómez-López (2014); Myers (2012) and Silva & Swartz (2016), both as exclusions; CIES and van Vugt as the caution on moving players abroad |

---

## What is deliberately not cited

No expected-goals model, no possession-based metric, and no passing-network
measure appears in this project's conclusions. Each is standard in the
literature above and each is unavailable for Jordan across the 2022 boundary:
shots are non-zero in 1.4% of pre-2022 appearance rows against 21% after,
possession exists for 27 of 174 matches, and ball location is absent in every
era. Using them would compare coverage rather than football. See CLAUDE.md,
Part 4.
