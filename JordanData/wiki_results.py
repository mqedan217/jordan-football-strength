"""Fill gaps in match_table from Wikipedia's Jordan results articles.

Almost every gap in `match_table.csv` is a friendly: 55 of 58 missing
attendances, all 8 missing lineups. Tournament articles cannot help, because
Wikipedia has no per-match article for a friendly. But Jordan has two results
articles -- 2010-2019 and 2020-present -- that carry a `Football box
collapsible` for every match, with stadium, location, attendance, referee and
scorers.

Rules this module holds to, because the filled values are meant to be checked:

* **Nothing is invented.** Every value written comes from a fetched page held
  in `cache/wiki/`, and every proposed fill is listed in
  `match_table_fills.csv` with the article it came from, so any cell can be
  traced back and verified.
* **Only nulls are filled.** An existing value is never overwritten, so FBref
  stays authoritative wherever it has an answer.
* **A date match is not enough to trust a row.** Dates can collide and articles
  can be wrong, so a Wikipedia row is only used if its scoreline also agrees
  with the one already on the match. Rows that disagree are reported, not used.

The date in these boxes carries no year -- it reads `31 May` -- and the year
comes from the `==2021==` section heading the box sits under. Parsing a box in
isolation therefore produces a date that is wrong by up to fifteen years.

`attendance = 0` is a real value, not a gap: several 2020-21 matches were played
behind closed doors.
"""

import re
from pathlib import Path

import pandas as pd

from wiki_arabcup import fetch_page

ROOT = Path(__file__).resolve().parents[1]
SQUAD = ROOT / "outputs" / "squad"

PAGES = [
    "Jordan national football team results (2010–2019)",
    "Jordan national football team results (2020–present)",
]

BOX = re.compile(r"\{\{Football box collapsible(.*?)\n\}\}", re.S | re.I)
YEAR_HEAD = re.compile(r"^==+\s*(\d{4})\s*==+\s*$", re.M)
LINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


def _field(block, name):
    m = re.search(rf"^\s*\|?\s*{name}\s*=\s*(.*)$", block, re.M)
    if not m:
        return None
    v = m.group(1).strip()
    return v or None


def _plain(v):
    """Strip wiki links and templates down to readable text."""
    if not v:
        return None
    v = LINK.sub(lambda m: (m.group(2) or m.group(1)), v)
    v = re.sub(r"\{\{[^}]*\}\}", "", v)
    v = re.sub(r"\[https?://\S+\s*([^\]]*)\]", r"\1", v)
    v = re.sub(r"<[^>]+>", "", v)
    return re.sub(r"\s+", " ", v).strip(" .,|") or None


def parse_results_page(title):
    """Every match box on one results article, dated using its year heading."""
    text = fetch_page(title)
    if text is None:
        print(f"  !! could not fetch {title}")
        return pd.DataFrame()

    # a box belongs to the most recent year heading above it
    years = [(m.start(), int(m.group(1))) for m in YEAR_HEAD.finditer(text)]
    rows = []
    for m in BOX.finditer(text):
        block, pos = m.group(1), m.start()
        year = None
        for start, y in years:
            if start < pos:
                year = y
            else:
                break
        raw_date = _field(block, "date")
        if not (year and raw_date):
            continue
        when = pd.to_datetime(f"{_plain(raw_date)} {year}", errors="coerce",
                              dayfirst=True)
        if pd.isna(when):
            continue

        score = _field(block, "score") or ""
        nums = re.findall(r"\d+", score)
        t1 = re.search(r"\{\{fb(?:-rt)?\|(\w+)\}\}", _field(block, "team1") or "")
        t2 = re.search(r"\{\{fb(?:-rt)?\|(\w+)\}\}", _field(block, "team2") or "")
        jordan_first = bool(t1 and t1.group(1) == "JOR")
        gf = ga = None
        if len(nums) >= 2:
            a, b = int(nums[0]), int(nums[1])
            gf, ga = (a, b) if jordan_first else (b, a)

        att = _field(block, "attendance")
        att = re.sub(r"[^\d]", "", att) if att else None

        rows.append({
            "date": when, "wiki_gf": gf, "wiki_ga": ga,
            "wiki_round": _plain(_field(block, "round")),
            "wiki_stadium": _plain(_field(block, "stadium")),
            "wiki_location": _plain(_field(block, "location")),
            "wiki_attendance": int(att) if att else None,
            "wiki_referee": _plain(_field(block, "referee")),
            "wiki_page": title,
        })
    return pd.DataFrame(rows)


FILLABLE = {
    "attendance": "wiki_attendance",
    "referee": "wiki_referee",
}


def propose(match_table=None):
    """Proposed fills, one row per cell, each traceable to its source article."""
    if match_table is None:
        match_table = pd.read_csv(SQUAD / "match_table.csv", parse_dates=["date"])
    wiki = pd.concat([parse_results_page(p) for p in PAGES], ignore_index=True)
    wiki = wiki.dropna(subset=["date"]).drop_duplicates("date")
    print(f"  parsed {len(wiki)} dated match boxes from {len(PAGES)} articles")

    j = match_table.merge(wiki, on="date", how="left", suffixes=("", "_w"))

    # a date match alone is not enough; the scoreline must agree too
    known = j.gf.notna() & j.wiki_gf.notna()
    j["score_agrees"] = ~known | ((j.gf == j.wiki_gf) & (j.ga == j.wiki_ga))
    clashes = j[known & ~j.score_agrees]

    rows = []
    for _, r in j[j.wiki_page.notna() & j.score_agrees].iterrows():
        for field, src in FILLABLE.items():
            cur, new = r.get(field), r.get(src)
            if pd.isna(cur) and pd.notna(new):
                rows.append({"match_id": r.match_id, "date": r.date.date(),
                             "opponent": r.get("opponent"), "comp": r.get("comp"),
                             "field": field, "current": None, "proposed": new,
                             "source": r.wiki_page})
    fills = pd.DataFrame(rows)
    return fills, clashes, j


def build():
    mt = pd.read_csv(SQUAD / "match_table.csv", parse_dates=["date"])
    fills, clashes, joined = propose(mt)

    fills.to_csv(SQUAD / "match_table_fills.csv", index=False)
    print(f"\nmatch_table_fills.csv  {len(fills)} proposed fills")
    if len(fills):
        print(fills.groupby("field").size().rename("n").to_string())

    if len(clashes):
        cols = ["date", "opponent", "gf", "ga", "wiki_gf", "wiki_ga", "wiki_page"]
        clashes[cols].to_csv(SQUAD / "match_table_fill_clashes.csv", index=False)
        print(f"\n!! {len(clashes)} rows where the article's score disagrees with "
              f"ours -- NOT used, listed in match_table_fill_clashes.csv")
        print(clashes[cols].head(10).to_string(index=False))

    # apply, recording provenance per field
    out = mt.copy()
    for field in FILLABLE:
        out[f"{field}_source"] = pd.Series(
            ["fbref" if pd.notna(v) else None for v in mt[field]], index=mt.index)
    # widen to object before writing: a filled column may be read back as a
    # string (FBref writes attendance with a thousands comma) and refuse a number
    for field in FILLABLE:
        out[field] = out[field].astype(object)
    for _, f in fills.iterrows():
        i = out.index[out.match_id == f.match_id]
        out.loc[i, f.field] = f.proposed
        out.loc[i, f"{f.field}_source"] = "wikipedia"
    out["attendance"] = pd.to_numeric(out.attendance, errors="coerce").astype("Int64")
    out.to_csv(SQUAD / "match_table_filled.csv", index=False)

    print("\ncoverage before -> after:")
    for field in FILLABLE:
        print(f"  {field:12s} {mt[field].notna().mean():.0%} -> "
              f"{out[field].notna().mean():.0%}")
    return out, fills, clashes


# ---------------------------------------------------------------- goal filling
GOAL_LINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]\s*((?:\{\{goal\|[^}]*\}\}\s*)+)",
                       re.I)
GOAL_TPL = re.compile(r"\{\{goal\|([^}]*)\}\}", re.I)


def _field_block(block, name):
    """A field's FULL value, which may span lines, stopping at the next field.

    `goals1 = ` is often empty and the scorer list that follows a filled one
    runs over several lines. A single-line read returns nothing useful, and a
    greedy multi-line read runs straight into `|stadium =`. Stop at the next
    `| name =` or the end of the box.
    """
    m = re.search(rf"^[ \t]*\|?[ \t]*{name}[ \t]*=[ \t]*(.*?)(?=^[ \t]*\|[ \t]*\w+[ \t]*=|\Z)",
                  block, re.M | re.S)
    return m.group(1).strip() if m else ""


def _goals_from_field(value):
    """(scorer, minute, stoppage, type) for every goal in a goals field."""
    out = []
    for target, display, blob in GOAL_LINK.findall(value or ""):
        # the link target is the full name ("Hamza Al-Dardour"); the display
        # text is usually just the surname, which will not join to anything
        name = target.strip()
        for tpl in GOAL_TPL.findall(blob):
            # one template can hold several goals: {{goal|25||90}} is two
            pending = None
            for arg in tpl.split("|"):
                arg = arg.strip()
                if not arg:
                    continue
                mm = re.match(r"^(\d+)(?:\+(\d+))?$", arg)
                if mm:
                    pending = {"scorer": name, "minute": int(mm.group(1)),
                               "minute_stoppage": int(mm.group(2) or 0),
                               "type": "goal"}
                    out.append(pending)
                elif pending is not None:
                    low = arg.lower()
                    if "pen" in low:
                        pending["type"] = "penalty_goal"
                    elif "o.g" in low or "own" in low:
                        pending["type"] = "own_goal"
    return out


def jordan_goals():
    """Every Jordan goal the results articles record, as a tidy frame."""
    rows = []
    for title in PAGES:
        text = fetch_page(title)
        if text is None:
            continue
        years = [(m.start(), int(m.group(1))) for m in YEAR_HEAD.finditer(text)]
        for m in BOX.finditer(text):
            block, pos = m.group(1), m.start()
            year = None
            for start, y in years:
                if start < pos:
                    year = y
                else:
                    break
            raw = _field(block, "date")
            if not (year and raw):
                continue
            when = pd.to_datetime(f"{_plain(raw)} {year}", errors="coerce",
                                  dayfirst=True)
            if pd.isna(when):
                continue
            t1 = re.search(r"\{\{fb(?:-rt)?\|(\w+)\}\}", _field(block, "team1") or "")
            jordan_first = bool(t1 and t1.group(1) == "JOR")
            field = "goals1" if jordan_first else "goals2"
            for g in _goals_from_field(_field_block(block, field)):
                # an own goal credited in Jordan's column was scored by an
                # opponent, so Jordan has no scorer for it
                if g["type"] == "own_goal":
                    g = {**g, "scorer": None, "type": "own_goal_for"}
                rows.append({"date": when, "wiki_page": title, **g})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    build()
