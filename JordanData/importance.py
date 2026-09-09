"""Did Jordan become a team that turns up when it matters?

The claim: relative to its own rating, Jordan performs better in competitive
matches than in friendlies, and this is a post-2022 trait rather than a
permanent one. tournament.py established that this does NOT belong in the model
as a fitted parameter -- a single time-invariant term would smear a post-2022
behaviour back across 2015 and corrupt the baseline the project measures from.
It belongs here instead, as a measured claim with an interval on it.

The statistic is a difference-in-differences on held-out forecast errors:

    gap(era)  = mean residual in competitive matches
              - mean residual in friendlies
    DiD       = gap(post) - gap(pre)

Residuals come from rolling-origin fits -- the model at each cutoff has never
seen the matches it is scored on -- so a positive gap means the team beat what
was predicted of it, opponent quality already accounted for.

Three things this module does that a single number cannot:

  1. puts a block-bootstrap interval on the DiD, not just on each era
  2. sweeps the split date, since a result that only exists at one cutoff is
     an artefact of choosing that cutoff
  3. runs every team through the same pipeline, so Jordan's DiD can be read
     against the distribution it came from -- the placebo CLAUDE.md requires

Note the split date is not fished for: CLAUDE.md fixes the treatment at
"roughly mid-2022" on the basis of results, before any of this was computed.

Limitation: the bootstrap resamples cutoff windows, holding the fitted ratings
at each cutoff fixed. It therefore carries sampling uncertainty in the
residuals but not estimation uncertainty in the ratings themselves. With ~9,600
matches behind each fit and one team's residuals in front, that second term is
second order -- but it is not zero, so these intervals are mildly optimistic.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data import load_results, index_teams
from baseline import PoissonStrength
from tournament import classify, LOOKBACK_DAYS

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260909
SPLIT = "2022-07-01"
PEERS = ["Iraq", "Palestine", "Oman", "Bahrain", "Syria", "Lebanon",
         "United Arab Emirates", "Kuwait", "Qatar", "Saudi Arabia"]


def residual_frame(start="2014-01-01", end="2026-01-01", freq="6MS"):
    """Held-out forecast errors for every team, one row per team per match."""
    df = load_results()
    teams, idx = index_teams(df)
    out = []
    for block, ref in enumerate(pd.date_range(start, end, freq=freq)):
        train = df[(df.date < ref) & (df.date >= ref - pd.Timedelta(days=LOOKBACK_DAYS))]
        test = df[(df.date >= ref) & (df.date < ref + pd.DateOffset(months=6))]
        if len(test) < 50:
            continue
        m = PoissonStrength(len(teams)).fit(
            train.home_team.map(idx).to_numpy(), train.away_team.map(idx).to_numpy(),
            train.home_score.to_numpy(float), train.away_score.to_numpy(float),
            (~train.neutral).to_numpy(float),
            (ref - train.date).dt.days.to_numpy(float))
        h = test.home_team.map(idx).to_numpy()
        a = test.away_team.map(idx).to_numpy()
        mh, ma = m.rates(h, a, (~test.neutral).to_numpy(float))
        pred = mh - ma
        act = (test.home_score - test.away_score).to_numpy(float)
        comp = classify(test.tournament) != "friendly"
        strength = np.array([m.strength(i) for i in range(len(teams))])
        for side, sign, me, them in (("home", 1.0, h, a), ("away", -1.0, a, h)):
            out.append(pd.DataFrame({
                "block": block, "date": test.date.to_numpy(),
                "team": np.asarray(teams)[me], "opponent": np.asarray(teams)[them],
                "resid": sign * (act - pred), "competitive": comp,
                "opp_strength": strength[them], "side": side}))
        print(f"  {ref.date()}  n_test={len(test)}", flush=True)
    return pd.concat(out, ignore_index=True)


#  cell order: 0 = pre/competitive, 1 = pre/friendly,
#              2 = post/competitive, 3 = post/friendly
def _cells(d, split):
    """Per-block sums and counts for the four cells.

    Reducing each block to sums and counts first means a bootstrap draw is a
    matrix product rather than a reshuffle of rows -- the statistic is
    identical, it just makes the all-teams placebo tractable.
    """
    post = (d.date >= np.datetime64(pd.Timestamp(split))).to_numpy()
    comp = d.competitive.to_numpy(bool)
    cell = post.astype(int) * 2 + (~comp).astype(int)
    blocks, bi = np.unique(d.block.to_numpy(), return_inverse=True)
    sums = np.zeros((len(blocks), 4))
    counts = np.zeros((len(blocks), 4))
    np.add.at(sums, (bi, cell), d.resid.to_numpy(float))
    np.add.at(counts, (bi, cell), 1.0)
    return sums, counts


def _stats(sums, counts):
    """(gap_pre, gap_post, DiD) from cell totals; NaN where a cell is empty."""
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(counts > 0, sums / np.where(counts > 0, counts, 1), np.nan)
    gp = m[..., 0] - m[..., 1]
    gq = m[..., 2] - m[..., 3]
    return gp, gq, gq - gp


def did_for(frame, team, split=SPLIT, n_boot=4000, seed=SEED):
    """DiD for one team, with a bootstrap over cutoff windows."""
    d = frame[frame.team == team]
    sums, counts = _cells(d, split)
    gp, gq, obs = _stats(sums.sum(0), counts.sum(0))

    n_blocks = len(sums)
    rng = np.random.default_rng(seed)
    sel = rng.integers(0, n_blocks, (n_boot, n_blocks))
    w = np.zeros((n_boot, n_blocks))
    np.add.at(w, (np.repeat(np.arange(n_boot), n_blocks), sel.ravel()), 1.0)
    draws = _stats(w @ sums, w @ counts)[2]

    ok = draws[~np.isnan(draws)]
    n = counts.sum(0)
    return {
        "team": team, "gap_pre": gp, "gap_post": gq, "did": obs,
        "lo": np.percentile(ok, 2.5) if len(ok) else np.nan,
        "hi": np.percentile(ok, 97.5) if len(ok) else np.nan,
        "p_positive": float((ok > 0).mean()) if len(ok) else np.nan,
        "n_pre": int(n[0] + n[1]), "n_post": int(n[2] + n[3]),
        "n_pre_comp": int(n[0]), "n_post_comp": int(n[2]),
        "boot_usable": len(ok) / n_boot,
    }


def split_sweep(frame, team, dates, **kw):
    """The claim should not live or die on one cutoff date."""
    return pd.DataFrame([{**did_for(frame, team, split=s, **kw), "split": s}
                         for s in dates])


def placebo(frame, teams, split=SPLIT, min_per_cell=5, **kw):
    rows = []
    for t in teams:
        d = frame[frame.team == t]
        post = d.date >= np.datetime64(pd.Timestamp(split))
        cells = [((~post) & d.competitive).sum(), ((~post) & ~d.competitive).sum(),
                 (post & d.competitive).sum(), (post & ~d.competitive).sum()]
        if min(cells) < min_per_cell:
            continue
        rows.append(did_for(frame, t, split=split, **kw))
    return pd.DataFrame(rows).sort_values("did", ascending=False)


if __name__ == "__main__":
    print("building held-out residuals for all teams...")
    frame = residual_frame()
    (ROOT / "outputs").mkdir(exist_ok=True)
    frame.to_csv(ROOT / "outputs/heldout_residuals.csv", index=False)

    print("\n=== Jordan: difference-in-differences ===")
    jordan = did_for(frame, "Jordan")
    print(pd.Series(jordan).to_string())

    print("\n=== split-date sweep (Jordan) ===")
    sweep = split_sweep(frame, "Jordan",
                        ["2021-01-01", "2021-07-01", "2022-01-01", "2022-07-01",
                         "2023-01-01", "2023-07-01"], n_boot=2000)
    print(sweep[["split", "gap_pre", "gap_post", "did", "lo", "hi",
                 "n_pre", "n_post"]].round(3).to_string(index=False))
    sweep.to_csv(ROOT / "outputs/importance_split_sweep.csv", index=False)

    print("\n=== placebo: AFC peers ===")
    peers = placebo(frame, ["Jordan"] + PEERS, n_boot=2000)
    print(peers[["team", "gap_pre", "gap_post", "did", "lo", "hi",
                 "n_pre", "n_post"]].round(3).to_string(index=False))
    peers.to_csv(ROOT / "outputs/importance_placebo.csv", index=False)

    print("\n=== all teams with enough matches ===")
    counts = frame.groupby("team").size()
    everyone = placebo(frame, counts[counts >= 60].index.tolist(), n_boot=2000)
    everyone.to_csv(ROOT / "outputs/importance_all_teams.csv", index=False)
    j = everyone[everyone.team == "Jordan"].did.iloc[0]
    print(f"n teams = {len(everyone)}   Jordan DiD = {j:+.3f}   "
          f"percentile = {(everyone.did < j).mean() * 100:.0f}")
    print(everyone.did.describe(percentiles=[.05, .5, .95]).round(3).to_string())
    print("\nwrote outputs/heldout_residuals.csv, importance_split_sweep.csv, "
          "importance_placebo.csv, importance_all_teams.csv")
