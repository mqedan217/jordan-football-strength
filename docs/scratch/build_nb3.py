import nbformat as nbf
from pathlib import Path

ROOT = Path(r"C:\Users\Marwan\Projects\JordansFB")
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# Where the goals came from

Notebook 01 established the shift: Jordan's opponent-adjusted strength rose
about 0.7 goals of expected goal difference between 2015 and 2026, the rise came
from **attack** rather than defence (+0.744, 95% CI [+0.238, +1.212]), and no
regional peer matched it. Notebook 02 assembled the squad data and marked its
edges.

This notebook asks the next question: **the extra goals — who scored them, and
how many players did it take?**

Two answers are possible and they mean very different things. If the goals came
from a broad group, Jordan built a better team. If they came from a handful,
Jordan found a few players. The second is more fragile, and it is the one the
data points to.

Everything is on the project's stated window, **2015 onward**, split at
**2022-07-01** — a date section 0 puts on trial before anything is allowed to
rest on it. Every difference carries a block-bootstrap interval clustered on
six-month windows, because matches inside a window share opponents, a manager
and a squad.
""")

code("""
import sys, json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "JordanData"))
import viz
viz.use_style()

SQUAD = ROOT / "outputs" / "squad"
START, CUT = pd.Timestamp("2015-01-01"), pd.Timestamp(viz.SPLIT)
rng = np.random.default_rng(41)

mt = pd.read_csv(SQUAD / "match_table.csv", parse_dates=["date"]).sort_values("date")
goals = pd.read_csv(SQUAD / "goals.csv", parse_dates=["date"])
ap = pd.read_csv(SQUAD / "appearances_combined.csv")
ja = ap[ap.squad_id == "3e22f0fa"].copy()
ja["Min"] = pd.to_numeric(ja.Min, errors="coerce")

mt = mt[mt.date >= START].copy()
goals = goals[goals.date >= START].copy()
for d in (mt, goals):
    d["era"] = np.where(d.date < CUT, "pre", "post")
mt["blk"] = mt.date.dt.year * 2 + (mt.date.dt.month > 6)

# A scorer is a player id, never a name -- the same man is transliterated
# several ways across sources (notebook 02, section 5).
gj = goals.dropna(subset=["scorer_player_id"]).copy()
NAMES = ja.dropna(subset=["player_id"]).groupby("player_id").Player.agg(
    lambda s: s.mode().iloc[0])

print(f"window   {mt.date.min().date()} .. {mt.date.max().date()}")
print(f"matches  {mt.era.value_counts().reindex(['pre','post']).to_dict()}")
print(f"goals    {goals.era.value_counts().reindex(['pre','post']).to_dict()}"
      f"   with a scorer id: {len(gj)}")
""")

md("""
## 0. Is 2022 real?

Everything below splits the record at **2022-07-01**, and a single date picked
after seeing the results is exactly the kind of choice that manufactures
findings. This project has already retired one effect on that ground: the
competitive-match premium in notebook 01 grew monotonically the later the cut
was placed, which is what slicing a smooth trend looks like rather than what a
change looks like.

So the check comes first, before anything that depends on it. Slide the cut
every six months from 2018 to 2024 and recompute each headline measure. The two
cases leave different signatures:

- a **change** at a date gives a ramp up to that date and then a **plateau** --
  once the cut is past the real break, moving it further only trades matches
  between two groups that already differ
- a **trend** with no break gives a line that **keeps climbing**, because every
  step of the cut puts more of the rise on one side

The cut moves; the data does not. This can say whether mid-2022 is where things
changed. It cannot say why, and nothing here identifies a cause.
""")

code("""
# One function per measure, each over an arbitrary set of matches, so the same
# code can be pointed at any candidate date. Sections 1 and 3 reuse them.
pos_by_match = {m: s.scorer_position.dropna().tolist() for m, s in goals.groupby("match_id")}
mins_by_match = {m: s.groupby("player_id").Min.sum() for m, s in ja.groupby("match_id")}
xi = mt.dropna(subset=["starting_xi"]).copy()
xi["ids"] = xi.starting_xi.map(lambda s: [p["id"] for p in json.loads(s) if p.get("id")])
XI = xi.set_index("match_id")

def mf_share(d):
    v = [p for m in d.match_id for p in pos_by_match.get(m, [])]
    return np.mean([x == "MF" for x in v]) if v else np.nan

def top11(d):
    tot = {}
    for m in d.match_id:
        s = mins_by_match.get(m)
        if s is None:
            continue
        for pid, v in s.items():
            tot[pid] = tot.get(pid, 0) + v
    if not tot:
        return np.nan
    p = pd.Series(tot).sort_values(ascending=False)
    return p.head(11).sum() / p.sum()

def top3_share(d):
    g = gj[gj.match_id.isin(set(d.match_id))].scorer_player_id
    if len(g) < 20:                    # too few goals to form a share
        return np.nan
    return g.value_counts(normalize=True).head(3).sum()

# Familiarity and appearance rate depend on the match and the FULL history
# behind it, never on which matches a bootstrap draw happens to pick -- so they
# are computed once per match here, and resampling only averages them.
# Both are divided by the matches available in the 24-month window, making them
# rates: raw counts accumulate with time and FBref's coverage starts in 2011, so
# the unadjusted versions rise for reasons that have nothing to do with Jordan.
fam_rate, caps_rate = {}, {}
for r in xi.itertuples():
    prior = xi[(xi.date >= r.date - pd.Timedelta(days=730)) & (xi.date < r.date)]
    if len(prior) < 15:                # too little history to form a rate
        continue
    pc, cc = {}, {}
    for q in prior.itertuples():
        for pair in combinations(sorted(q.ids), 2):
            pc[pair] = pc.get(pair, 0) + 1
        for i in q.ids:
            cc[i] = cc.get(i, 0) + 1
    pairs = list(combinations(sorted(r.ids), 2))
    fam_rate[r.match_id] = np.mean([pc.get(p, 0) for p in pairs]) / len(prior)
    caps_rate[r.match_id] = np.mean([cc.get(i, 0) for i in r.ids]) / len(prior)

def windowed(d, which):
    src = fam_rate if which == "fam" else caps_rate
    v = [src[m] for m in d.match_id if m in src]
    return np.mean(v) if v else np.nan

print(f"windowed rates computed for {len(fam_rate)} of {len(xi)} matches with an XI")
""")

code("""
SWEEP = [("minutes top-11", top11),
         ("familiarity", lambda d: windowed(d, "fam")),
         ("appearance rate", lambda d: windowed(d, "caps")),
         ("top-3 goal share", top3_share),
         ("midfielder share", mf_share)]

rows = []
for cut in pd.date_range("2018-07-01", "2024-07-01", freq="6MS"):
    pre, post = mt[mt.date < cut], mt[mt.date >= cut]
    if len(pre) < 25 or len(post) < 25:     # a side too thin to compare
        continue
    r = {"cut": cut.date(), "n_pre": len(pre), "n_post": len(post)}
    for lab, fn in SWEEP:
        x, y = fn(pre), fn(post)
        r[lab] = y - x if np.isfinite(x) and np.isfinite(y) else np.nan
    rows.append(r)
sweep = pd.DataFrame(rows)
print("post minus pre, at every candidate split date:")
print(sweep.round(3).to_string(index=False))
print()
print("rows repeat where no match falls between two cuts (the 2020 shutdown,")
print("and the gap after 2022-07) -- the same split, reached two ways.")
""")

code("""
fig, ax = plt.subplots(figsize=(7.6, 4.4))
x = pd.to_datetime(sweep.cut)
# The shaded band is where the cut is already past the break: if the change is
# real, the lines go flat inside it.
ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01"),
           color=viz.GRID, zorder=0)
ax.annotate("cut is past the break \u2192", xy=(pd.Timestamp("2022-01-01"), 1.0),
            xycoords=("data", "axes fraction"), xytext=(-5, -10),
            textcoords="offset points", fontsize=8, color=viz.INK_MUTED,
            ha="right", va="top")
PALETTE = [viz.BLUE, viz.ORANGE, viz.AQUA, viz.VIOLET, viz.INK_MUTED]
for (lab, _), col in zip(SWEEP, PALETTE):
    odd = lab == "midfielder share"          # the one that never settles
    ax.plot(x, sweep[lab], color=col, marker="o", ms=3.5, label=lab,
            lw=1.6 if odd else 2.2, ls=(0, (4, 3)) if odd else "-")
ax.axhline(0, color=viz.INK_MUTED, lw=0.9, zorder=0)
viz.mark_split(ax, label="the cut used below")
ax.set_xlabel("where the cut is placed")
ax.set_ylabel("difference, post minus pre")
ax.set_title("Move the cut: four measures stop moving after early 2022, one never does")
ax.legend(fontsize=8, ncol=3, loc="lower center")
viz.despine(ax)
plt.show()
""")

code("""
# Ramp  = how far a measure travels from the earliest cut to the one we use.
# Flat  = how much it still moves once the cut is PAST the break (shaded band).
# A real change has flat much smaller than ramp. A trend being sliced has the
# two comparable, because there is no break for the cut to get past.
band = pd.to_datetime(sweep.cut).between("2022-01-01", "2024-01-01")
chosen = sweep[pd.to_datetime(sweep.cut) == CUT].iloc[0]
for lab, _ in SWEEP:
    ramp = chosen[lab] - sweep[lab].iloc[0]
    flat = sweep.loc[band, lab].max() - sweep.loc[band, lab].min()
    if abs(ramp) < 0.05:
        verdict = "NO BREAK TO FIND -- as large at the 2018 cut as at 2022"
    else:
        ratio = flat / abs(ramp)
        verdict = f"flat/ramp {ratio:4.2f}   " + ("PLATEAU" if ratio < 0.25 else "still moving")
    print(f"  {lab:17s} ramp {ramp:+.3f}   flat {flat:.3f}   {verdict}")
""")

md("""
**Four of the five behave the way a real change behaves.** Minutes
concentration, familiarity, appearance rate and top-3 goal share all climb from
2018, arrive by **2022-01**, and then sit still — moving the cut a further two
years changes minutes top-11 by 0.007 on a rise of 0.275. Wherever the break is,
it is no later than early 2022, and the exact date is not carrying the result.

**The fifth does not, and that is worth saying plainly.** The midfielder share of
goals is as large at a 2018 cut as at 2022, sags in between, and is largest at
the very last cut — where only 27 matches remain on the post side. The sweep
cannot locate a date for it. Its interval in section 3 still excludes zero, so
the post-2022 team really does take more of its goals from midfield; but that is
a statement about two groups of matches, not evidence that anything changed in
mid-2022. Treat it as the weakest of the four measures there.

What this cannot do is separate mid-2022 from the months either side. The
plateau is already reached at the 2022-01 cut, and at ~15 matches a year those
two dates are a handful of fixtures apart. **"No later than early 2022"** is the
claim the sweep supports; "July 2022 specifically" is not, and nothing below
depends on it.
""")

md("""
## 1. The goals came from fewer players

"How many different men scored" is a **species-richness** problem, and richness
rises with sample size whatever the underlying diversity. The post-2022 era has
far fewer goals than the pre era, so the two counts are not comparable as
observed — the smaller sample would show fewer scorers even if nothing changed.

The correction is **rarefaction** (Gotelli & Colwell 2001; Hurlbert 1971):
repeatedly draw the larger era down to the smaller era's size and compare
distributions rather than points. Four measures, from breadth to concentration.
""")

code("""
pre_s = gj[gj.era == "pre"].scorer_player_id
post_s = gj[gj.era == "post"].scorer_player_id
n_post = len(post_s)

def profile(s):
    vc = s.value_counts(); p = vc / vc.sum()
    return np.array([len(vc), p.iloc[0], p.head(3).sum(), (p ** 2).sum()])

obs = profile(post_s)
draws = np.array([profile(pre_s.sample(n_post, replace=False,
                                       random_state=int(rng.integers(1e9))))
                  for _ in range(4000)])

LABELS = ["distinct scorers", "top-1 share", "top-3 share", "HHI"]
rows = []
for i, lab in enumerate(LABELS):
    lo, hi = np.percentile(draws[:, i], [2.5, 97.5])
    rows.append({"measure": lab, "post": obs[i], "pre_lo": lo, "pre_hi": hi,
                 "verdict": "outside" if not (lo <= obs[i] <= hi) else "inside"})
rar = pd.DataFrame(rows)
print(f"post: {n_post} goals, {int(obs[0])} scorers")
print(f"pre : {len(pre_s)} goals, {pre_s.nunique()} scorers, rarefied to {n_post}\\n")
print(rar.round(3).to_string(index=False))
""")

code("""
# Rarefaction curve: how many scorers a sample of k goals yields, pre vs post.
def curve(s, ks, reps=400):
    out = []
    for k in ks:
        if k > len(s):
            out.append((np.nan, np.nan, np.nan)); continue
        v = [s.sample(k, replace=False,
                      random_state=int(rng.integers(1e9))).nunique()
             for _ in range(reps)]
        out.append((np.mean(v), *np.percentile(v, [2.5, 97.5])))
    return np.array(out)

ks = np.arange(5, len(pre_s) + 1, 5)
c_pre = curve(pre_s, ks)
ks_post = np.arange(5, n_post + 1, 5)
c_post = curve(post_s, ks_post)

fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.fill_between(ks, c_pre[:, 1], c_pre[:, 2], color=viz.BLUE, alpha=0.16, lw=0)
ax.plot(ks, c_pre[:, 0], color=viz.BLUE, label="pre-2022")
ax.fill_between(ks_post, c_post[:, 1], c_post[:, 2], color=viz.ORANGE, alpha=0.16, lw=0)
ax.plot(ks_post, c_post[:, 0], color=viz.ORANGE, label="post-2022")
ax.axvline(n_post, color=viz.INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=0)
ax.annotate(f"{n_post} goals\\n(post-2022 total)", xy=(n_post, 6),
            xytext=(6, 0), textcoords="offset points",
            fontsize=8, color=viz.INK_MUTED, va="bottom")
ax.annotate("pre-2022", xy=(ks[-1], c_pre[-1, 0]), xytext=(-4, 6),
            textcoords="offset points", color=viz.BLUE, fontsize=9,
            ha="right", fontweight="semibold")
ax.annotate("post-2022", xy=(ks_post[-1], c_post[-1, 0]), xytext=(6, -2),
            textcoords="offset points", color=viz.ORANGE, fontsize=9,
            fontweight="semibold")
ax.set_xlabel("goals drawn"); ax.set_ylabel("different scorers")
ax.set_title("Jordan's goals come from a narrower group after 2022")
viz.despine(ax)
plt.show()
""")

md("""
The curves separate well before the cut, so this is not an artefact of the eras
being different lengths: **at any common number of goals, the post-2022 era uses
fewer scorers.**

**But it is not one man.** The top-1 share sits *inside* the rarefied interval —
Jordan's leading scorer takes no more of the load than a pre-2022 sample of the
same size would give. The breadth measures move and the single-player measure
does not, which is a narrowing to a small **group**, not a dependence on an
individual. Worth stating plainly, because "one player carries them" is the
popular version of this claim and the data does not support it.
""")

md("""
## 2. Who

It is tempting to call this a changing of the guard, and the table below only
half supports that. Two high-volume pre-2022 scorers vanish completely. The
three who take over were **already there** — all of them scoring before 2022,
then scaling up sharply. So it is partly a replacement and partly an emergence,
and the distinction matters: a group that grew into the role is a different
proposition from one that arrived.
""")

code("""
tab = (gj.pivot_table(index="scorer_player_id", columns="era",
                      values="match_id", aggfunc="size")
       .reindex(columns=["pre", "post"]).fillna(0).astype(int))
tab["total"] = tab.sum(axis=1)
tab.insert(0, "player", tab.index.map(NAMES))
top = tab.sort_values("total", ascending=False).head(10)
print(top.to_string(index=False))

for e in ("pre", "post"):
    s = gj[gj.era == e].scorer_player_id
    t3 = s.value_counts().head(3)
    print(f"\\n{e:4s}  {len(s)} goals, top three = {t3.sum()/len(s):.0%}: "
          + ", ".join(f"{NAMES.get(i, i)} ({c})" for i, c in t3.items()))
""")

code("""
fig, ax = plt.subplots(figsize=(7.6, 5.0))
d = top.iloc[::-1]
y = np.arange(len(d))
ax.barh(y - 0.2, d.pre, height=0.38, color=viz.BLUE, label="pre-2022")
ax.barh(y + 0.2, d.post, height=0.38, color=viz.ORANGE, label="post-2022")
ax.set_yticks(y); ax.set_yticklabels(d.player)
ax.set_xlabel("goals")
ax.set_title("The scoring load changed hands, it did not shift")
ax.legend(loc="lower right")
viz.despine(ax)
ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
plt.show()
""")

md("""
Two patterns, and they should not be blurred together.

**The old top two disappear.** Jordan's leading scorer of the pre-2022 era and
the man third behind him contribute nothing at all afterwards — not reduced
roles, zero goals.

**The new top three were already scoring.** Every one of them appears in the
pre-2022 column, then roughly doubles or triples his output. Al-Taamari in
particular is in the top three of *both* eras, which is why "Jordan replaced its
forwards" would be an overstatement.

So the narrowing in section 1 is not a squad being rebuilt. It is a wide, fairly
flat scoring distribution collapsing onto three players who were already part of
it. Whether that is a team finding its best attackers or a team running out of
alternatives, this data cannot say.
""")

md("""
## 3. A narrower group, a more settled team

If the goals come from fewer players, the obvious next question is whether the
*team* also narrowed. Three measures, each of which has to be built carefully
because the naive version is confounded by time.

- **Minutes concentration** — share of all minutes going to the top eleven
  players. Compared at equal match counts.
- **Team familiarity** — the standard measure is pairwise prior appearances
  together (*Journal of Big Data*, 2025). Raw counts accumulate with time and
  FBref's coverage starts in 2011, so both rise for reasons unrelated to Jordan.
  Here it is restricted to a **24-month lookback** and divided by the matches
  available in that window, making it a rate that cannot drift.
- **Appearance rate** — the same treatment for individual starters.

Plus **where the goals came from positionally**, which needs no such correction.
""")

code("""
BLOCKS = {e: [d for _, d in mt[mt.era == e].groupby("blk")] for e in ("pre", "post")}

def block_boot(fn, nb=600):
    \"\"\"Resample six-month blocks within each era; matches in a window are not
    independent draws, so the block is the unit (Kunsch 1989).\"\"\"
    obs = fn(mt[mt.era == "post"]) - fn(mt[mt.era == "pre"])
    vals = []
    for _ in range(nb):
        s = {e: pd.concat([BLOCKS[e][i]
                           for i in rng.integers(0, len(BLOCKS[e]), len(BLOCKS[e]))])
             for e in ("pre", "post")}
        v = fn(s["post"]) - fn(s["pre"])
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return obs, lo, hi

MEASURES = [
    ("midfielder share of goals", mf_share, 600),
    ("minutes: top-11 share", top11, 600),
    ("familiarity rate (pairs)", lambda d: windowed(d, "fam"), 600),
    ("appearance rate (starters)", lambda d: windowed(d, "caps"), 600),
]
res = []
for lab, fn, nb in MEASURES:
    o, lo, hi = block_boot(fn, nb)
    res.append({"measure": lab, "pre": fn(mt[mt.era == "pre"]),
                "post": fn(mt[mt.era == "post"]),
                "diff": o, "lo": lo, "hi": hi,
                "excludes_0": bool(lo > 0 or hi < 0)})
settled = pd.DataFrame(res)
print(settled.round(3).to_string(index=False))
""")

code("""
fig, ax = plt.subplots(figsize=(7.6, 3.4))
d = settled.iloc[::-1].reset_index(drop=True)
y = np.arange(len(d))
for i, r in d.iterrows():
    col = viz.ORANGE if r.excludes_0 else viz.INK_MUTED
    ax.plot([r.lo, r.hi], [i, i], color=col, lw=2.4, solid_capstyle="round")
    ax.plot(r["diff"], i, "o", color=col, ms=7, zorder=3)
    ax.annotate(f"{r['diff']:+.2f}", xy=(r["diff"], i), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=8, color=col)
ax.axvline(0, color=viz.INK_2, lw=1.0, zorder=0)
ax.set_yticks(y); ax.set_yticklabels(d.measure)
ax.set_xlabel("change, pre-2022 to post-2022  (95% block-bootstrap interval)")
ax.set_title("A narrower, more settled team")
viz.despine(ax)
ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
plt.show()
""")

md("""
All four move the same way and their intervals clear zero: the post-2022 team
gives more of its minutes to fewer players, those players have played together
more recently, and a larger share of the goals comes from midfield.

The last one is worth separating from the rest. The others describe *stability*;
the midfield share describes *shape*. They are consistent with one another but
they are not the same evidence, and the notebook does not treat them as such.
""")

md("""
## 4. When the goals happen

A different cut of the same matches: not who scored, but when — and what the
scoreboard said at the time. Reconstructed in `JordanData/match_state.py` from
goal minutes on **both** sides, including the ten Arab Cup matches recovered
from Wikipedia, which FBref does not carry at all.

Coverage is 155 of 174 matches across the whole file and somewhat fewer inside
the 2015 window used here; the cell prints the exact count it works from. The
gap is FBref reports with no event feed, and it is balanced across the two eras
— which it was **not** before the Arab Cup was recovered, when the missing
matches were disproportionately Jordan's best.
""")

code("""
ms = pd.read_csv(SQUAD / "match_state.csv", parse_dates=["date"])
ms = ms[(ms.date >= START) & ms.jordan_side.notna()].copy()
ms["era"] = np.where(ms.date < CUT, "pre", "post")
BUCK = ["1-15", "16-30", "31-45", "46-60", "61-75", "76-90"]
f_tot = ms[[f"for_{b}" for b in BUCK]].sum().values
a_tot = ms[[f"against_{b}" for b in BUCK]].sum().values
prof = pd.DataFrame({"window": BUCK,
                     "scored": f_tot / f_tot.sum(),
                     "conceded": a_tot / a_tot.sum()})
print(f"{len(ms)} matches, {int(f_tot.sum())} scored, {int(a_tot.sum())} conceded\\n")
print(prof.round(3).to_string(index=False))
print("\\nArmatas et al. (2009), all football, 76-90 window: 0.225")
""")

code("""
fig, ax = plt.subplots(figsize=(7.6, 4.4))
x = np.arange(len(BUCK))
ax.bar(x - 0.2, prof.scored, width=0.38, color=viz.BLUE, label="scored")
ax.bar(x + 0.2, prof.conceded, width=0.38, color=viz.ORANGE, label="conceded")
ax.axhline(0.225, color=viz.INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=0)
ax.annotate("22.5% — all-football rate for the 76-90 window\\n(Armatas et al. 2009)",
            xy=(0, 0.225), xytext=(0, 5), textcoords="offset points",
            fontsize=8, color=viz.INK_MUTED, va="bottom")
ax.set_xticks(x); ax.set_xticklabels(BUCK)
ax.set_xlabel("minute"); ax.set_ylabel("share of that side's goals")
ax.set_title("Jordan's scoring is unremarkably timed. Its conceding is not.")
ax.legend(loc="upper left")
viz.despine(ax)
ax.grid(axis="x", visible=False); ax.grid(axis="y", visible=True)
plt.show()
""")

md("""
Jordan's own late scoring sits almost exactly on the all-football benchmark —
the attack has no special relationship with the clock. The conceding column is
the outlier: **a third of every goal Jordan concedes arrives in the last fifteen
minutes**, around half again the rate the literature expects.

This is a description of all 155 matches, **not** a change across 2022 — the era
difference in that window does not clear zero. It is a standing feature of this
team over the whole period, and it is the clearest thing in the notebook that
points at something to fix rather than something to celebrate.
""")

md("""
## 5. Crossing the two: when *and* in what state

Section 4 showed **when** Jordan concedes and the earlier sections showed **in
what state**. Neither alone explains the late-conceding result, because "a third
of goals conceded arrive after the 75th minute" is compatible with two opposite
stories: a team that throws away leads, or a team that gets put away in matches
it was already losing.

Crossing them settles it. Rates need **minutes at risk** in each cell, not goal
counts — a team that spends the last fifteen minutes level has more chance to
concede a level-state goal simply by being there.
""")

code("""
cells = pd.read_csv(SQUAD / "state_time_cells.csv")
cells = cells[cells.match_id.isin(set(mt.match_id))]          # 2015+ window
grid = (cells.groupby(["window", "state"])
        .agg(minutes=("minutes", "sum"), ga=("ga", "sum"), gf=("gf", "sum"))
        .reset_index())
grid["conceded_per90"] = grid.ga / grid.minutes * 90
grid["scored_per90"] = grid.gf / grid.minutes * 90
ORDER = ["1-15", "16-30", "31-45", "46-60", "61-75", "76-90"]

print("conceded per 90 minutes AT RISK:")
print(grid.pivot(index="window", columns="state",
                 values="conceded_per90").reindex(ORDER).round(2).to_string())
print("\\ngoals conceded in each cell:")
print(grid.pivot(index="window", columns="state",
                 values="ga").reindex(ORDER).astype(int).to_string())
print("\\nminutes at risk:")
print(grid.pivot(index="window", columns="state",
                 values="minutes").reindex(ORDER).round(0).to_string())
""")

code("""
fig, ax = plt.subplots(figsize=(7.6, 4.6))
x = np.arange(len(ORDER))
for st, col in (("leading", viz.BLUE), ("level", viz.ORANGE), ("trailing", viz.AQUA)):
    y = grid[grid.state == st].set_index("window").reindex(ORDER).conceded_per90
    ax.plot(x, y.values, color=col, marker="o")
    ax.annotate(st, xy=(x[-1], y.values[-1]), xytext=(7, 0),
                textcoords="offset points", color=col, fontsize=9,
                va="center", fontweight="semibold")
ax.set_xticks(x); ax.set_xticklabels(ORDER)
ax.set_xlim(-0.3, len(ORDER) + 0.9)
ax.set_xlabel("minute"); ax.set_ylabel("goals conceded per 90 min at risk")
ax.set_title("Jordan does not lose leads. It loses level matches, late.")
viz.despine(ax)
ax.grid(axis="x", visible=False)
plt.show()
""")

md("""
The blue line is the control and it is flat: **with a lead, Jordan's conceding
rate in the last fifteen minutes is indistinguishable from the fifteen before
it.** Leads are not the problem.

The orange line is the finding. While **level**, the conceding rate roughly
quadruples going into the final window.
""")

code("""
# The contrast, bootstrapped over matches in six-month blocks.
by_match = {m: d for m, d in cells.groupby("match_id")}
blocks = [d.match_id.tolist() for _, d in mt.groupby("blk")]

def rate(ms_, window, state, direction="ga"):
    d = pd.concat([by_match[m] for m in ms_ if m in by_match])
    s = d[(d.window == window) & (d.state == state)]
    return s[direction].sum() / s.minutes.sum() * 90 if s.minutes.sum() > 0 else np.nan

def ratio(ms_, state):
    late, before = rate(ms_, "76-90", state), rate(ms_, "61-75", state)
    return late / before if before and np.isfinite(before) else np.nan

allm = mt.match_id.tolist()
for st in ("level", "leading"):
    obs = ratio(allm, st)
    draws = []
    for _ in range(1500):
        pick = [m for i in rng.integers(0, len(blocks), len(blocks)) for m in blocks[i]]
        v = ratio(pick, st)
        if np.isfinite(v):
            draws.append(v)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    tag = "FINDING" if lo > 1 else "control, flat"
    print(f"  while {st:8s}: 61-75 {rate(allm,'61-75',st):.2f} -> "
          f"76-90 {rate(allm,'76-90',st):.2f} per 90   "
          f"ratio {obs:.2f}x  95% [{lo:.2f}, {hi:.2f}]   {tag}")

print("\\n  internal control -- does Jordan also SCORE more while level late?")
print("  If level matches simply get frantic, both rates rise together.")
for w in ("61-75", "76-90"):
    print(f"    {w}: scored {rate(allm, w, 'level', 'gf'):.2f}  "
          f"conceded {rate(allm, w, 'level', 'ga'):.2f} per 90")
""")

md("""
So it is not that level matches get frantic and goals fly both ways. Jordan's
*scoring* rate while level collapses in the same window its conceding rate
quadruples. The asymmetry is the point.

Two checks before trusting it.
""")

code("""
# The goal timelines, the same ones match_state builds its own table from.
from match_state import goal_timelines
TL = {m: v for m, v in goal_timelines().items() if m in set(mt.match_id)}
info = mt.set_index("match_id")

# Check 1 -- stoppage time. Minutes at risk stop at 90 for every match, but
# stoppage is always played, so the last window's rate is inflated for
# everyone. Adding it back can only shrink the effect. How much does it take?
level_at_90 = sum(1 for tl, _s, _x in TL.values()
                  if sum(d for m, d in tl if m <= 90) == 0)
lvl = grid[grid.state == "level"].set_index("window")
print(f"matches still level at 90': {level_at_90}")
for extra in (0, 3, 5, 8):
    m = lvl.loc["76-90", "minutes"] + extra * level_at_90
    r = lvl.loc["76-90", "ga"] / m * 90
    print(f"  +{extra} min stoppage on each -> {r:.2f} per 90 "
          f"({r / lvl.loc['61-75', 'conceded_per90']:.2f}x vs 61-75)")
""")

code("""
# Check 2 -- outcomes, independent of the rate arithmetic entirely.
rows = []
for mid, (tl, _s, _x) in TL.items():
    if sum(d for m, d in tl if m <= 75) != 0:      # level at 75' only
        continue
    dft = sum(d for _, d in tl)
    r = info.loc[mid]
    rows.append({"match_id": mid, "date": r.date, "era": r.era,
                 "opponent": r.opponent, "expected_gf": r.expected_gf,
                 "expected_ga": r.expected_ga, "expected_gd": r.expected_gd,
                 "pts": 3 if dft > 0 else (1 if dft == 0 else 0),
                 "result": "W" if dft > 0 else ("D" if dft == 0 else "L")})
L = pd.DataFrame(rows)
print(f"matches level at the 75th minute: {len(L)}  {L.era.value_counts().to_dict()}")
print(pd.crosstab(L.era, L.result).reindex(columns=["W", "D", "L"])
      .fillna(0).astype(int).to_string())
print(f"\\n  points per match from that position: {L.pts.mean():.2f} of 3")
print(f"  mean expected_gd of those fixtures: {L.expected_gd.mean():+.2f} "
      f"(all matches {mt.expected_gd.mean():+.2f})")
print("  -> these are not fixtures Jordan was losing on paper")
""")

code("""
# Opponent-adjusted: what should a level scoreline at 75' be worth against
# THESE opponents? Simulate the remaining 15 minutes at each fixture's own
# model goal rates, which is the same adjustment gd_above_expected applies to
# a full match.
sim = []
for r in L.itertuples():
    kf = rng.poisson(r.expected_gf * 15 / 90, 40000)
    ka = rng.poisson(r.expected_ga * 15 / 90, 40000)
    d = kf - ka
    sim.append(np.where(d > 0, 3, np.where(d == 0, 1, 0)).mean())
sim = np.array(sim)
obs, exp = L.pts.mean(), sim.mean()
draws = [(L.pts.values[p].mean() - sim[p].mean())
         for p in (rng.integers(0, len(L), len(L)) for _ in range(5000))]
lo, hi = np.percentile(draws, [2.5, 97.5])
print(f"  observed        {obs:.2f} points per match")
print(f"  model-expected  {exp:.2f}")
print(f"  shortfall       {obs - exp:+.2f}   95% [{lo:+.2f}, {hi:+.2f}]")
print(f"\\n  across {len(L)} matches that is {abs(obs - exp) * len(L):.0f} points.")
""")

md("""
A level scoreline at the 75th minute, against these opponents, should be worth
about 1.2 points. Jordan takes 0.72 — a shortfall of roughly half a point per
match, interval excluding zero, across 36 matches.

It is not an artefact of the clock convention: even adding eight minutes of
stoppage to every one of those matches, the late level-state spike remains about
2.7×. And it is not opponent quality — the adjustment above already handles
that, and these are fixtures Jordan was on average favoured in.

🚨 **It is partly an artefact of the model.** The simulation that sets the 1.2
draws goals at a constant rate whatever the scoreboard says, and that is known
to be wrong for level matches specifically. How much of the shortfall that costs
is measured in **What we can still improve**, at the end of this notebook; the
number above should be read with that qualification attached.

### Did this change between the eras?

That is the notebook's actual question, so it has to be tested rather than
asserted from the pooled number.
""")

code("""
cells["era"] = cells.match_id.map(mt.set_index("match_id").era)
print("the level-state late cell, by era:")
for e in ("pre", "post"):
    d = cells[(cells.era == e) & (cells.state == "level")]
    for w in ("61-75", "76-90"):
        s = d[d.window == w]
        print(f"  {e:5s} {w}: {int(s.ga.sum()):2d} conceded, {int(s.gf.sum())} scored"
              f" / {s.minutes.sum():6.0f} min = {s.ga.sum()/s.minutes.sum()*90:.2f} per 90")

def late_level(ms_):
    return rate(ms_, "76-90", "level")

pre_m = mt[mt.era == "pre"].match_id.tolist()
post_m = mt[mt.era == "post"].match_id.tolist()
blk = {e: [d.match_id.tolist() for _, d in mt[mt.era == e].groupby("blk")]
       for e in ("pre", "post")}
draws = []
for _ in range(2000):
    s = {e: [m for i in rng.integers(0, len(blk[e]), len(blk[e])) for m in blk[e][i]]
         for e in ("pre", "post")}
    v = late_level(s["post"]) - late_level(s["pre"])
    if np.isfinite(v):
        draws.append(v)
lo, hi = np.percentile(draws, [2.5, 97.5])
print(f"\\n  conceding rate while level, 76-90: pre {late_level(pre_m):.2f} -> "
      f"post {late_level(post_m):.2f}")
print(f"  difference {late_level(post_m)-late_level(pre_m):+.2f} per 90  "
      f"95% [{lo:+.2f}, {hi:+.2f}]  -- spans zero, so NOT shown to have worsened")
""")

code("""
# The interpretable version: the points shortfall, by era.
L["era"] = L.match_id.map(mt.set_index("match_id").era)
L["blk"] = L.match_id.map(mt.set_index("match_id").blk)
L["expected_pts"] = sim
L["short"] = L.pts - L.expected_pts
print(L.groupby("era").agg(n=("pts", "size"), observed=("pts", "mean"),
                           model_expected=("expected_pts", "mean"),
                           shortfall=("short", "mean"),
                           opponent_exp_gd=("expected_gd", "mean")).round(2).to_string())

LB = {e: [d.match_id.tolist() for _, d in L[L.era == e].groupby("blk")]
      for e in ("pre", "post")}
lm = L.set_index("match_id")
d_obs = L[L.era == "post"].short.mean() - L[L.era == "pre"].short.mean()
dr = []
for _ in range(4000):
    s = {e: [m for i in rng.integers(0, len(LB[e]), len(LB[e])) for m in LB[e][i]]
         for e in ("pre", "post")}
    v = lm.loc[s["post"]].short.mean() - lm.loc[s["pre"]].short.mean()
    if np.isfinite(v):
        dr.append(v)
lo, hi = np.percentile(dr, [2.5, 97.5])
print(f"\\n  change in shortfall, pre -> post: {d_obs:+.2f}  95% [{lo:+.2f}, {hi:+.2f}]")
for e in ("pre", "post"):
    d = L[L.era == e]
    b = [d.short.values[p].mean()
         for p in (rng.integers(0, len(d), len(d)) for _ in range(5000))]
    l2, h2 = np.percentile(b, [2.5, 97.5])
    print(f"    {e:5s} n={len(d):2d}  shortfall {d.short.mean():+.2f}  "
          f"95% [{l2:+.2f}, {h2:+.2f}]")
""")

code("""
# How often does Jordan even arrive at the 75th minute in each state?
at75 = []
for mid, (tl, _s, _x) in TL.items():
    d75 = sum(d for m, d in tl if m <= 75)
    at75.append({"era": info.loc[mid, "era"],
                 "at75": "leading" if d75 > 0 else ("trailing" if d75 < 0 else "level")})
at75 = pd.DataFrame(at75)
print("scoreboard state at the 75th minute, share of matches:")
print(pd.crosstab(at75.era, at75.at75, normalize="index")
      .reindex(index=["pre", "post"])[["leading", "level", "trailing"]].round(3).to_string())
print("\\nand the opposition faced, all matches:")
print(mt.groupby("era")[["expected_gd", "expected_gf", "expected_ga"]]
      .mean().reindex(["pre", "post"]).round(3).to_string())
""")

md("""
**The answer: this is the part that did not improve.**

The shortfall is **−0.50** before 2022 and **−0.41** after, a change of **+0.09,
95% [−0.36, +0.77]** — no change worth the name. The raw conceding rate in that
cell rises from 3.04 to 7.32 per 90, which looks alarming until you see the
interval, [−0.42, +11.66]: ten goals in 123 minutes cannot distinguish anything.
Pre-2022 the shortfall clears zero on its own; post-2022 it does not, but that
is twelve matches, not evidence of a fix.

The state distribution barely moves either — Jordan arrives at the 75th minute
leading in **47%** of matches in both eras, and level in about a quarter.

**And that is the interesting part.** Over the same period the opposition got
dramatically harder: mean expected goal difference across all matches falls from
**+0.48** to **+0.02**, and among the matches that reach 75 minutes level, from
**+0.50** to **−0.41** — from fixtures Jordan was favoured in to fixtures it was
not. Reaching the same scoreboard positions against that much better opposition
*is* the improvement, and it is the same phenomenon as notebook 01's headline:
the raw numbers hold flat while the quality behind them rises.

So this section's contribution to the notebook's question is a negative one, and
a useful one. The results got better, but **none of the improvement came from
managing tight matches better.** Jordan converts a level scoreline at 75 minutes
exactly as badly as it did before — roughly half a point per match below what
the fixture is worth, in both eras. Whatever drove the rise, it was not this.

What the failure *is* cannot be settled here. The obvious reading — a team
chasing a winner gets caught on the break — needs shot locations or possession
to test, and neither exists for these matches (notebook 02). It stays a
description of what happens, not an explanation.
""")

md("""
## 6. What it isn't

The measures below were built and returned nothing. They are reported because a
null with an interval is a result, and because each one closes off an
explanation that would otherwise stay live.
""")

code("""
ms["blk"] = ms.date.dt.year * 2 + (ms.date.dt.month > 6)
ms["margin"] = ms.gf - ms.ga     # before the blocks are cut, or the draws lack it
SB = {e: [d for _, d in ms[ms.era == e].groupby("blk")] for e in ("pre", "post")}

def sboot(fn, nb=1200):
    obs = fn(ms[ms.era == "post"]) - fn(ms[ms.era == "pre"])
    vals = []
    for _ in range(nb):
        s = {e: pd.concat([SB[e][i] for i in rng.integers(0, len(SB[e]), len(SB[e]))])
             for e in ("pre", "post")}
        v = fn(s["post"]) - fn(s["pre"])
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return obs, lo, hi

NULLS = [
    ("share of time leading", lambda d: (d.min_leading / d.match_end).mean()),
    ("share of time trailing", lambda d: (d.min_trailing / d.match_end).mean()),
    ("scores first", lambda d: d.first_goal.eq("for").mean()),
    ("trailed -> avoided defeat", lambda d: d.recovered.dropna().mean()),
    ("win by 3+", lambda d: (d.margin >= 3).mean()),
    ("lose by 3+", lambda d: (d.margin <= -3).mean()),
]
rows = []
for lab, fn in NULLS:
    o, lo, hi = sboot(fn)
    rows.append({"measure": lab, "diff": o, "lo": lo, "hi": hi,
                 "excludes_0": bool(lo > 0 or hi < 0)})
nulls = pd.DataFrame(rows)
print(nulls.round(3).to_string(index=False))
""")

code("""
# Scoring rate per 90 minutes ACTUALLY SPENT in each state. Both numerator and
# denominator move when a match is resampled, so the rate is what is bootstrapped.
def rate(d, s, direction="gf"):
    m = d[f"min_{s}"].sum()
    return d[f"{direction}_{s}"].sum() / m * 90 if m > 0 else np.nan

for s in ("leading", "level", "trailing"):
    o, lo, hi = sboot(lambda d, ss=s: rate(d, ss))
    a, b = rate(ms[ms.era == "pre"], s), rate(ms[ms.era == "post"], s)
    print(f"  scored while {s:9s} pre {a:.2f}  post {b:.2f}  "
          f"diff {b-a:+.2f}  95% [{lo:+.2f}, {hi:+.2f}]")

# The comparison of interest as a single quantity: does the team push when behind?
def ratio(d):
    lv = rate(d, "level")
    return rate(d, "trailing") / lv if lv > 0 else np.nan

o, lo, hi = sboot(ratio)
print(f"\\n  trailing/level ratio  pre {ratio(ms[ms.era=='pre']):.2f}  "
      f"post {ratio(ms[ms.era=='post']):.2f}  diff {o:+.2f}  95% [{lo:+.2f}, {hi:+.2f}]")
print(f"  goals scored while trailing, both eras: "
      f"{int(ms.gf_trailing.sum())} -- the whole basis of this comparison")
""")

code("""
# The keeper: a complete handover, and no effect on goals conceded
kp = pd.read_csv(SQUAD / "keeper.csv")
kj = kp[kp.squad_id == "3e22f0fa"].merge(
    mt[["match_id", "era", "date", "ga", "expected_ga"]], on="match_id")
kj["above"] = kj.ga - kj.expected_ga
k = (kj.groupby("Player").agg(matches=("match_id", "size"),
                              pre=("era", lambda s: (s == "pre").sum()),
                              post=("era", lambda s: (s == "post").sum()),
                              first=("date", "min"), last=("date", "max"),
                              ga=("ga", "mean"), ga_above_expected=("above", "mean"))
     .sort_values("matches", ascending=False))
k["first"] = k["first"].dt.date
k["last"] = k["last"].dt.date
print("goalkeepers, 8+ matches:")
print(k[k.matches >= 8].round(3).to_string())
print("\\ncards per match:", mt.groupby("era").n_cards.mean().round(2).to_dict())
""")

md("""
- **Match state.** Time leading, time trailing, scoring first, recovering from
  behind — every interval spans zero. Jordan did not become a team that manages
  the scoreboard differently.
- **Scoring rate by state.** The point estimates are seductive: pre-2022 Jordan
  looks like a front-runner, scoring freely with a lead and *below its own
  level-state rate* when chasing — the opposite of the pattern Lago-Peñas &
  Gómez-López (2014) describe, where teams that go behind take on more of the
  game. Post-2022 looks flatter. But the whole comparison rests on the handful
  of goals Jordan has scored while trailing in eleven years, and the interval on
  the trailing/level ratio is far too wide to support any of it. A good story
  that the data cannot carry.
- **Margins.** Winning and losing heavily happen at the old rates. What changed
  is that matches became higher-scoring overall.
- **The goalkeeper.** The position does change hands — Amer Shafi plays 47
  matches and none after February 2021, and Yazeed Abulaila, who had been
  sharing the role since 2017, takes it over outright. But goals conceded above
  expectation are essentially identical between the two. A plausible mechanism,
  measured and ruled out.
- **Discipline.** Cards per match, unchanged.

Two more, tested elsewhere and excluded here for reasons worth recording:
**substitutions** are confounded by the 2020 five-substitute rule, which lands
two years before the treatment date; **manager** gives 15 spells across the
window and cannot be separated from squad change at this sample size.
""")

md("""
## Summary

**Supported** — each with an interval that excludes zero, or a rarefied
comparison that falls outside:

- the goals after 2022 come from a **narrower group** of scorers: fewer distinct
  scorers, higher top-3 share, higher HHI, all outside the rarefied pre-2022
  range at equal sample size
- the team is **more settled**: minutes concentrated in fewer players, and
  starters who have played together more in the preceding two years
- a larger share of goals comes from **midfield**

**Unchanged across the eras — the part of the game that did not improve:**

- Jordan concedes **a third of its goals in the last fifteen minutes**, against
  an all-football benchmark of 22.5%
- crossed with state, that is **entirely a level-scoreline problem**. With a
  lead the late conceding rate is flat (1.01×); while level it roughly
  quadruples (3.92×), and Jordan's own scoring rate collapses in the same window
- from a **level scoreline at the 75th minute** Jordan takes about 0.7 points
  where the model expects about 1.2 — and that shortfall is **−0.50 before 2022
  and −0.41 after, a change of +0.09, 95% [−0.36, +0.77]**. It did not get
  better and it did not get worse.

**Not established** — measured, interval spans zero:

- that Jordan relies on **one** player: the top-1 share is ordinary
- any change in match-state behaviour, in the rate of heavy wins or defeats, in
  the goalkeeper's contribution, or in discipline

**Ruled out as the mechanism:** goalkeeping and discipline.

### So how did the results get better?

Putting the sections together, the shape of the answer is:

**Through the attack, from a narrower group, with a more settled team.** More of
the goals come from a small set of players and from midfield, minutes are
concentrated in fewer men, and those men have played together more in the
preceding two years. Against opposition that got substantially harder — mean
expected goal difference falling from +0.48 to +0.02 — that produced better
results.

**Not through game management.** Jordan reaches the 75th minute in the same
scoreboard positions as before, and converts a level one exactly as badly. It
does not hold leads better, recover from behind better, score first more often,
win more heavily or concede more heavily. Every one of those is a measured null,
and the level-scoreline shortfall is unchanged at about half a point a match.

**Not through the goalkeeper or discipline**, both of which changed hands or
could have and neither of which moved the numbers.

That last group matters as much as the first. The list of things that *didn't*
change is what stops "Jordan got better" from being a story you can tell any way
you like.

### What this does and does not license

The honest reading is a **coincidence with uncertainty**, which is what this
design can support and no more. Jordan's attack improved; over the same period
the scoring narrowed onto a small group of players and the team became more
settled. Those things happened together.

They are not shown to cause one another. Squad change, coaching change and
scheduling all move together across mid-2022, there are ~40 matches after it,
and no design here separates them. The one thing worth saying beyond the
association is a **caution rather than a conclusion**: a rise carried by a
narrow group is more fragile than one carried by a broad one, and the top-3
share is the number to watch if the level slips.
""")

md("""
## What we can still improve

Ranked by how much each would change what is written above. The first one
qualifies a claim already made; the rest are things this notebook does not know.

### 1. The shortfall in section 5 is partly the model's fault, not Jordan's

Section 5 sets the expectation for a level scoreline by simulating the last
fifteen minutes at each fixture's own model goal rates. That simulation draws
goals at a **constant rate whatever the scoreboard says** — and the match-status
literature says that assumption is wrong for exactly this situation. Late goals
are more common when a match is level, and level teams underperform leading ones
on shot quality, for everyone, everywhere. If the model under-predicts late goals
in level matches in general, then part of the −0.47 points is the model being
state-blind rather than Jordan being bad.

Two diagnostics, both of which the notebook can run on its own data:
""")

code("""
# Does the constant-rate model fit the last fifteen minutes of a level match?
late = pd.DataFrame([{"match_id": m,
                      "gf75": sum(1 for x, d in TL[m][0] if x > 75 and d > 0),
                      "ga75": sum(1 for x, d in TL[m][0] if x > 75 and d < 0)}
                     for m in L.match_id]).merge(L, on="match_id")
sg, sd = [], []
for r in late.itertuples():
    kf = rng.poisson(r.expected_gf * 15 / 90, 40000)
    ka = rng.poisson(r.expected_ga * 15 / 90, 40000)
    sg.append((kf + ka).mean())
    sd.append((kf == ka).mean())
print("CALIBRATION -- the model vs what actually happens, same 36 matches")
print(f"  goals after 75'   observed {(late.gf75 + late.ga75).mean():.2f} per match"
      f"    model {np.mean(sg):.2f}")
print(f"  draws             observed {(late.pts == 1).mean():.2f}"
      f"                model {np.mean(sd):.2f}")
print("  -> it predicts too few goals and too many draws. The shortfall")
print("     inherits that mis-specification; it is not all Jordan's.")
""")

code("""
# DIRECTION is the part no general effect explains. "Level matches produce more
# late goals" predicts volume. It does not predict them going one way.
f, g = int(late.gf75.sum()), int(late.ga75.sum())
sh = []
for _ in range(20000):
    d = late.sample(len(late), replace=True, random_state=int(rng.integers(1e9)))
    t = d.gf75.sum() + d.ga75.sum()
    if t:
        sh.append(d.gf75.sum() / t)
lo, hi = np.percentile(sh, [2.5, 97.5])
mdl = (late.expected_gf / (late.expected_gf + late.expected_ga)).mean()
print(f"DIRECTION -- after the 75th minute in those {len(late)} level matches")
print(f"  Jordan scored {f}, conceded {g}")
print(f"  Jordan's share {f / (f + g):.3f}   95% [{lo:.3f}, {hi:.3f}]"
      f"   {'EXCLUDES' if hi < 0.5 else 'includes'} 0.5")
print(f"  the model's own implied share for these fixtures: {mdl:.3f}")
for e in ("pre", "post"):
    d = late[late.era == e]
    t = d.gf75.sum() + d.ga75.sum()
    print(f"    {e:5s} n={len(d):2d}  {int(t):2d} late goals, "
          f"Jordan's share {d.gf75.sum() / t:.3f}")
""")

md("""
So section 5 should be **re-centred on the direction, not the volume**. That the
last fifteen minutes of a level match are busy is ordinary football. That those
goals go one way three times in four, against fixtures the model makes Jordan
the favourite in, is not, and it holds in both eras. The rate ratio and the
points shortfall are still worth showing, but as symptoms — they cannot be
quoted as if the comparison were clean.

### 2. There is no placebo in this notebook

The analysis plan calls for running the same pipeline on three or four
comparable AFC nations, and notebook 01 does exactly that for strength. Nothing
here does. Every measure in this notebook is Jordan against its own past, so a finding that
is really about international football in the 2020s — squads settling, scoring
narrowing, late goals rising — would look identical. This is the single
largest hole, and it is the one the split-date sweep in section 0 cannot fill,
because that check moves the date and not the team.

What it costs: roughly 600 match reports for the peer nations, most of which are
not in the cache. A cheaper partial version would take only the level-at-75
finding, which needs goal minutes and nothing else.

### 3. Three squad measures from the plan were never built

`age of the starting XI`, `debuts per window`, and `squad turnover (Jaccard of
consecutive XIs)`. The data for all three is already on disk — `player_dob.csv`
and the appearance rows — so this is an afternoon, not a project. Debuts in
particular look like they would say something: keyed on player id, first
appearances run **13 in 2021, then 1 in 2022 and 1 in 2023**, reopening to 6 and
9 in 2024-25. Those years hold very different numbers of covered matches (21, 6,
5), so per match it is 0.62 → 0.17 → 0.20 — a smaller drop than the raw counts
suggest, and exactly the kind of thing that needs building properly with an
interval rather than quoting. It is currently in no notebook.

### 4. The mechanism behind section 5 is out of reach, and will stay there

The obvious reading of the level-scoreline collapse — a team pushing for a
winner and getting caught — needs shot locations or possession, and neither
exists for these matches. Possession has been checked exhaustively and is not
obtainable; the 27 matches that have it are concentrated after 2023, so filling
it in would put the coverage boundary on the treatment date. Section 5 stays a
description of what happens, not an explanation.

### 5. Nineteen matches have no event feed at all

14 friendlies and 5 competitive fixtures with an FBref report but no timeline,
so they contribute to the squad measures and not to anything in sections 4-6.
Coverage is balanced across the eras at 89%, which makes this noise rather than
bias — but it caps the post-2022 side at about 40 matches, and that is why the
state-crossed intervals are as wide as they are.

### 6. Smaller things, recorded so they are not rediscovered

- **Minutes at risk stop at 90** while stoppage is always played, inflating the
  last window's rate for every team equally. Section 5 shows the effect survives
  eight added minutes, but the underlying clock is still wrong.
- **Managers have no stable id**, so the manager column is name-keyed and any
  manager-level comparison has to be checked by hand — the opposite of the rule
  the rest of the project follows for players.
- **Six Arab Cup names are unconfirmed transliteration matches** and are
  deliberately not merged into the appearance table. Confirming them by hand
  would add ten matches to the squad measures.
- **Notebooks 01 and 02 have not been revised** since the manager column was
  fixed; notebook 02's column-completeness figure predates it.
""")

md("""
## Appendix — one measure this notebook deliberately does not use

`match_table` carries `gf_above_expected` — goals scored minus the model's
expectation for that fixture. It is tempting as an outcome, and it is the wrong
tool for this question.

The model is refit every six months on data *before* each match, so the residual
measures **how far Jordan beat its own contemporaneous forecast**, not how much
it improved. And because Jordan's attack rose while its opponents got much
stronger, the two effects cancel in the expectation: expected goals-for is
essentially *flat* across the boundary while actual goals-for rises. A team
improving faster than a 730-day half-life can track will show a positive
residual for that reason alone.

So the residual is a statement about the model's lag, and using it here would
answer "how much of the forecast error did these players account for" — a
different question, wearing the same clothes.
""")

code("""
print(mt.groupby("era")[["gf", "expected_gf", "gf_above_expected",
                         "strength_at_match"]].mean().round(3).to_string())
print()
print("expected_gf barely moves; the model's own ATTACK estimate is what rose.")
tr = pd.read_csv(ROOT / "outputs" / "jordan_strength_ci.csv", parse_dates=["date"])
sel = tr[tr.date.isin(pd.to_datetime(["2015-01-01", "2022-07-01", "2026-07-01"]))]
print(sel[["date", "attack", "attack_lo", "attack_hi"]].round(3).to_string(index=False))
""")

md("""
This is why the notebook works in **shares and counts** — how goals were
distributed across players and positions, and how settled the team was. Those
need no expectation at all, so sections 1 to 3 do not depend on the model being
well calibrated at all. Sections 4 and 5 do use the model, to adjust for
opponent quality, and that is where its limits start to matter — which is the
first item in **What we can still improve**.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python",
                          "name": "python3"}
nb.metadata.language_info = {"name": "python"}

dest = ROOT / "notebooks" / "03_where_the_goals_came_from.ipynb"
dest.write_text(nbf.writes(nb), encoding="utf-8")
print(f"wrote {dest} with {len(cells)} cells")
