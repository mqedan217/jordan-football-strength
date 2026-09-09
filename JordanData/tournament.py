"""Does match importance belong in the model?

First attempt tested a *global* friendly effect -- one offset applied to every
team. It failed, and it deserved to: across 158 teams the competitive-minus-
friendly residual averages -0.011 goals. There is no global effect to find.

That was the wrong test. The football claim is not "friendlies are lower
scoring", it is "a friendly result is weaker evidence about how good a team is
when it matters", and how much weaker is a property of the team -- who rotates,
who experiments, who turns up. Pooling across teams averages that to zero by
construction. Jordan, held out and after opponent adjustment, beats its rating
by +0.53 goals in competitive matches relative to friendlies (95% CI [+0.04,
+1.01]), which is the 94th percentile of that distribution.

So the term here is per-team: one importance parameter each, lifting that
team's attack and defence together in competitive matches. It enters each
scoring rate as the difference of the two teams' terms, so it is identified
only up to a constant and the ridge centres it -- the same device the model
already uses for attack and defence. `ridge_imp` is tuned on held-out score,
not chosen.

Two changes to the test itself, both because the earlier one was too generous:
the score is reported on held-out COMPETITIVE matches (predicting friendlies
well is not the goal), and separately for Jordan, since a term justified by a
team-specific effect should be checked on that team.

Classes are coarse -- 86 distinct tournament names since 2016, 64 of them under
50 matches, so anything finer fits names not football.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

from data import load_results, index_teams
from baseline import PoissonStrength

ROOT = Path(__file__).resolve().parents[1]
LOOKBACK_DAYS = 3650
SEED = 20260909

# variant -> (description, PoissonStrength kwargs, design name)
VARIANTS = {
    "M0": ("baseline, no tournament term", {}, "none"),
    "M3": ("global friendly and finals offsets, friendly home adv", {}, "global"),
    "M4a": ("per-team importance, ridge 0.10",
            dict(team_importance=True, ridge_imp=0.10), "none"),
    "M4b": ("per-team importance, ridge 0.25",
            dict(team_importance=True, ridge_imp=0.25), "none"),
    "M4c": ("per-team importance, ridge 1.00",
            dict(team_importance=True, ridge_imp=1.00), "none"),
}


def classify(tournament):
    """Friendly / qualification / finals. Reference class is qualification."""
    t = tournament.str.lower()
    return np.where(t.eq("friendly"), "friendly",
                    np.where(t.str.contains("qualification|qualifier"),
                             "qualification", "finals"))


def is_competitive(frame):
    return (classify(frame.tournament) != "friendly").astype(float)


def design(frame, variant):
    """Covariate arrays for a variant: (shared, home_only, competitive)."""
    _, _, kind = VARIANTS[variant]
    comp = is_competitive(frame)
    if kind == "none":
        return None, None, comp
    klass = classify(frame.tournament)
    friendly = (klass == "friendly").astype(float)
    finals = (klass == "finals").astype(float)
    home_flag = (~frame.neutral).to_numpy(float)
    return (np.column_stack([friendly, finals]), friendly * home_flag, comp)


def fit_variant(frame, idx, n_teams, ref, variant, half_life=730.0, ridge=0.05):
    _, kw, _ = VARIANTS[variant]
    shared, home_only, comp = design(frame, variant)
    m = PoissonStrength(n_teams, half_life_days=half_life, ridge=ridge, **kw)
    m.fit(frame.home_team.map(idx).to_numpy(),
          frame.away_team.map(idx).to_numpy(),
          frame.home_score.to_numpy(float),
          frame.away_score.to_numpy(float),
          (~frame.neutral).to_numpy(float),
          (ref - frame.date).dt.days.to_numpy(float),
          shared=shared, home_only=home_only, competitive=comp)
    return m


def log_score(m, frame, idx, variant):
    """Poisson log likelihood per held-out match (both scorelines)."""
    shared, home_only, comp = design(frame, variant)
    mh, ma = m.rates(frame.home_team.map(idx).to_numpy(),
                     frame.away_team.map(idx).to_numpy(),
                     (~frame.neutral).to_numpy(float),
                     shared=shared, home_only=home_only, competitive=comp)
    gh = frame.home_score.to_numpy(float)
    ga = frame.away_score.to_numpy(float)
    return (gh * np.log(mh) - mh - gammaln(gh + 1)
            + ga * np.log(ma) - ma - gammaln(ga + 1))


def validate(start="2014-01-01", end="2026-01-01", freq="6MS", team="Jordan"):
    df = load_results()
    teams, idx = index_teams(df)
    ti = idx[team]
    rows, blocks = [], []
    scores = {v: [] for v in VARIANTS}
    masks = {"competitive": [], "team": []}
    imp_track = []

    for b, ref in enumerate(pd.date_range(start, end, freq=freq)):
        train = df[(df.date < ref) & (df.date >= ref - pd.Timedelta(days=LOOKBACK_DAYS))]
        test = df[(df.date >= ref) & (df.date < ref + pd.DateOffset(months=6))]
        if len(test) < 50:
            continue
        is_team = ((test.home_team == team) | (test.away_team == team)).to_numpy()
        masks["competitive"].append(is_competitive(test).astype(bool))
        masks["team"].append(is_team)
        blocks.append(np.full(len(test), len(rows)))

        row = {"cutoff": ref.date(), "n_train": len(train), "n_test": len(test),
               "n_test_team": int(is_team.sum())}
        for v in VARIANTS:
            m = fit_variant(train, idx, len(teams), ref, v)
            scores[v].append(log_score(m, test, idx, v))
            row[v] = scores[v][-1].mean()
            if v == "M4b":
                imp_track.append({"cutoff": ref.date(), "imp_team": m.imp[ti],
                                  "imp_sd_all": m.imp.std()})
        rows.append(row)
        print(f"{ref.date()}  n_test={len(test):5d}  "
              + "  ".join(f"{v}={row[v]:+.5f}" for v in VARIANTS), flush=True)

    cat = lambda d: {k: np.concatenate(x) for k, x in d.items()}
    return (pd.DataFrame(rows), cat(scores), cat(masks),
            np.concatenate(blocks), pd.DataFrame(imp_track))


def compare(scores, blocks, subset=None, baseline="M0", n_boot=2000, seed=SEED):
    """Held-out log-score gain over the baseline, block-bootstrapped by cutoff.

    Matches inside one six-month window share a fitted model, so they are not
    independent and a per-match SE would be too narrow.
    """
    rng = np.random.default_rng(seed)
    keep = np.ones(len(blocks), bool) if subset is None else subset
    ids = np.unique(blocks[keep])
    picks = [np.flatnonzero(keep & (blocks == i)) for i in ids]
    picks = [p for p in picks if len(p)]
    out = []
    for v in scores:
        if v == baseline:
            continue
        d = scores[v] - scores[baseline]
        draws = np.empty(n_boot)
        for i in range(n_boot):
            take = rng.integers(0, len(picks), len(picks))
            draws[i] = d[np.concatenate([picks[j] for j in take])].mean()
        out.append({"variant": v, "description": VARIANTS[v][0],
                    "n_matches": int(keep.sum()), "mean_gain": d[keep].mean(),
                    "lo": np.percentile(draws, 2.5),
                    "hi": np.percentile(draws, 97.5),
                    "p_better": float((draws > 0).mean())})
    return pd.DataFrame(out)


def sensitivity(variants=("M0", "M3", "M4b"), **kw):
    """Does the trace move under each term, even if it predicts no better?

    Prediction and estimation are different questions: a term can fail on
    held-out log score and still shift the quantity being reported.
    """
    from trajectory import trace
    runs = {v: trace(variant=v, **kw).set_index("date") for v in variants}
    out = pd.DataFrame({v: runs[v].strength for v in variants})
    for v in variants[1:]:
        out[f"{v}_minus_M0"] = out[v] - out["M0"]
    return out


if __name__ == "__main__":
    table, scores, masks, blocks, imp = validate()
    (ROOT / "outputs").mkdir(exist_ok=True)
    table.to_csv(ROOT / "outputs/tournament_validation.csv", index=False)

    subsets = {
        "all held-out matches": None,
        "competitive only": masks["competitive"],
        "Jordan only": masks["team"],
        "Jordan competitive only": masks["competitive"] & masks["team"],
    }
    frames = []
    for label, sub in subsets.items():
        c = compare(scores, blocks, subset=sub)
        c.insert(0, "subset", label)
        frames.append(c)
        print(f"\n=== log-score gain over M0: {label} ===")
        print(c.to_string(index=False))
    allc = pd.concat(frames, ignore_index=True)
    allc.to_csv(ROOT / "outputs/tournament_comparison.csv", index=False)

    print("\n=== Jordan's fitted importance parameter (M4b) ===")
    print(imp.round(4).to_string(index=False))

    print("\n=== Trace sensitivity ===")
    sens = sensitivity()
    sens.to_csv(ROOT / "outputs/tournament_sensitivity.csv")
    print(sens.round(4).to_string())
    print("\nlargest shift: " + ", ".join(
        f"{c} {sens[c].abs().max():.4f}"
        for c in sens.columns if c.endswith("_minus_M0")))
