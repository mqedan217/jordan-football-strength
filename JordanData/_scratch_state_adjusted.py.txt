"""Opponent-adjusted match state.

Raw state measures are dominated by who the opponent was: a team playing
Argentina trails for most of the match whatever it does. The model already
gives a per-match expected goals-for and goals-against, so the same adjustment
used for `gd_above_expected` can be applied to the scoreboard itself.

For each match, simulate the scoreline as two Poisson counts with the model's
own means, and place the goals in time by drawing from the POOLED empirical
distribution of goal minutes across every match in the dataset. That keeps the
rising-through-the-match shape Armatas et al. (2009) describe without assuming
a functional form for it, and it is identical across eras so it cannot itself
manufacture an era difference.

Residual = observed - simulated expectation. A positive `trailing_above` means
Jordan spent longer behind than a team of its estimated strength should have
against that opponent.
"""
import numpy as np
import pandas as pd

CUT = pd.Timestamp("2022-07-01")
N_SIM = 4000
N_BOOT = 3000
rng = np.random.default_rng(17)

ms = pd.read_csv("outputs/squad/match_state.csv", parse_dates=["date"])
ms = ms.dropna(subset=["jordan_side", "expected_gd"]).copy()
mt = pd.read_csv("outputs/squad/match_table.csv", parse_dates=["date"])
ms = ms.merge(mt[["match_id", "expected_gf", "expected_ga"]], on="match_id", how="left")
ms["era"] = np.where(ms.date < CUT, "pre", "post")
ms["blk"] = ms.date.dt.year * 2 + (ms.date.dt.month > 6)

# pooled empirical goal-minute distribution, both sides, every era
gm = []
for d in ("for", "against"):
    for b, (lo, hi) in zip(["1-15", "16-30", "31-45", "46-60", "61-75", "76-90"],
                           [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90)]):
        col = f"{d}_{b}"
        if col in ms.columns:
            gm += [(lo + hi) / 2] * int(ms[col].fillna(0).sum())
gm = np.array(gm, dtype=float)
print(f"pooled goal-minute pool: {len(gm)} goals, mean minute {gm.mean():.1f}")


def simulate(mu_f, mu_a, n=N_SIM):
    """Expected state measures for one fixture, under the model's own means."""
    kf = rng.poisson(mu_f, n)
    ka = rng.poisson(mu_a, n)
    out = np.zeros((n, 5))
    for i in range(n):
        tf = rng.choice(gm, kf[i]) if kf[i] else np.empty(0)
        ta = rng.choice(gm, ka[i]) if ka[i] else np.empty(0)
        ev = sorted([(t, 1) for t in tf] + [(t, -1) for t in ta])
        lead = level = trail = 0.0
        diff, t = 0, 0.0
        ever_l = ever_t = False
        first = 0
        for m, d in ev:
            span = m - t
            if diff > 0:
                lead += span
            elif diff < 0:
                trail += span
            else:
                level += span
            diff += d
            t = m
            if first == 0:
                first = d
            ever_l |= diff > 0
            ever_t |= diff < 0
        span = 90.0 - t
        if diff > 0:
            lead += span
        elif diff < 0:
            trail += span
        else:
            level += span
        out[i] = (lead / 90, level / 90, trail / 90, 1.0 if first > 0 else 0.0,
                  1.0 if (ever_t and diff >= 0) else 0.0)
    return out.mean(axis=0)


print("simulating expectations per match ...")
exp = np.array([simulate(r.expected_gf, r.expected_ga) for r in ms.itertuples()])
ms[["e_lead", "e_level", "e_trail", "e_first", "e_recover"]] = exp

ms["o_lead"] = ms.min_leading / ms.match_end
ms["o_level"] = ms.min_level / ms.match_end
ms["o_trail"] = ms.min_trailing / ms.match_end
ms["o_first"] = ms.first_goal.eq("for").astype(float)

for k in ("lead", "level", "trail", "first"):
    ms[f"{k}_above"] = ms[f"o_{k}"] - ms[f"e_{k}"]

BLOCKS = {e: [d for _, d in ms[ms.era == e].groupby("blk")] for e in ("pre", "post")}


def boot(fn, n=N_BOOT):
    out = []
    for _ in range(n):
        s = {}
        for e in ("pre", "post"):
            bs = BLOCKS[e]
            s[e] = pd.concat([bs[i] for i in rng.integers(0, len(bs), len(bs))])
        v = fn(s["post"]) - fn(s["pre"])
        if np.isfinite(v):
            out.append(v)
    return np.percentile(out, [2.5, 97.5])


print(f"\nmatches: {ms.era.value_counts().to_dict()}")
print("\n=== opponent-adjusted match state (observed minus model expectation) ===")
for k, lab in (("lead", "share of time LEADING"), ("level", "share of time LEVEL"),
               ("trail", "share of time TRAILING"), ("first", "scores first")):
    col = f"{k}_above"
    fn = lambda d, c=col: d[c].mean()
    a, b = fn(ms[ms.era == "pre"]), fn(ms[ms.era == "post"])
    lo, hi = boot(fn)
    star = "*" if (lo > 0 or hi < 0) else " "
    print(f" {star} {lab:26s} pre {a:+.3f}  post {b:+.3f}  "
          f"diff {b-a:+.3f}  95% [{lo:+.3f}, {hi:+.3f}]")

print("\n  (for reference, the RAW unadjusted versions)")
for k, lab in (("lead", "time leading"), ("trail", "time trailing"),
               ("first", "scores first")):
    fn = lambda d, c=f"o_{k}": d[c].mean()
    a, b = fn(ms[ms.era == "pre"]), fn(ms[ms.era == "post"])
    lo, hi = boot(fn, 1500)
    print(f"    {lab:24s} pre {a:+.3f}  post {b:+.3f}  diff {b-a:+.3f}  "
          f"95% [{lo:+.3f}, {hi:+.3f}]")

# ---- power: what could 34 post matches have detected? --------------------
print("\n=== power: minimum detectable difference, n_post = 34 ===")
for k in ("lead", "trail", "first"):
    s_pre = ms[ms.era == "pre"][f"{k}_above"].std()
    s_post = ms[ms.era == "post"][f"{k}_above"].std()
    n1, n2 = (ms.era == "pre").sum(), (ms.era == "post").sum()
    se = np.sqrt(s_pre**2 / n1 + s_post**2 / n2)
    print(f"  {k:6s} sd pre {s_pre:.3f} post {s_post:.3f}  SE of diff {se:.3f}  "
          f"-> 80% power needs |diff| >= {2.8 * se:.3f}")
print("  (that SE ignores block clustering, so it is optimistic)")

ms.to_csv("outputs/squad/match_state_adjusted.csv", index=False)
print("\nwrote outputs/squad/match_state_adjusted.csv")
