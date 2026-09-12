"""Recover Jordan's Arab Cup lineups from Wikipedia.

FBref does not carry the Arab Cup for Jordan at all -- it is absent from the
squad pages and from FBref's competition index -- so ten competitive matches
since 2015 have no lineup in `outputs/squad/`. Six of them are the December 2025
tournament, where Jordan won five of six and reached the final. Leaving that out
biases every squad measure, and it biases it in the worst direction: Arab Cup is
exactly the kind of fixture where a second-string XI appears.

Wikipedia's tournament articles carry full match detail -- starting XI with
shirt numbers and positions, substitutions with minutes, cards, captain,
manager, goalscorers. Note this is a *different* source from the one CLAUDE.md
lists as a dead end: that entry concerns per-player by-year tables, which are
incomplete. Match articles are a separate body of content and are complete for
these tournaments.

The lineups are plain wikitables rather than templates, so they are parsed by
line shape:

    |GK ||'''23'''|| [[Yazeed Abulaila]] ([[Captain...|c]])
    |LB ||'''21'''|| [[Mohannad Abu Taha]] || {{yel|40}} || {{suboff|72}}
    |colspan=3|'''Substitutions:'''
    |MF ||'''22'''|| [[Ali Azaizeh]] || || {{subon|61}}
    |colspan=3|'''Manager:'''
    |colspan=3|[[Jamal Sellami]]

🚨 The one thing this source cannot supply is FBref's player id. Wikipedia gives
names only, and CLAUDE.md is explicit that names are not a safe key here because
Arabic transliteration varies between sources and seasons. So rows produced here
carry `player_id = NA` and `source = "wikipedia"`, and `crosswalk()` proposes
name matches against the FBref player universe **for review** rather than
silently accepting them. Nothing downstream should join these rows on name
without consulting that review table.
"""

import difflib
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
WIKI = ROOT / "cache" / "wiki"
OUT = ROOT / "outputs" / "squad"
UA = {"User-Agent": "JordanFootballResearch/1.0 (academic; contact marwanqedan21@gmail.com)"}

PAGES = [
    "2021 FIFA Arab Cup Group C",
    "2021 FIFA Arab Cup knockout stage",
    "2025 FIFA Arab Cup Group C",
    "2025 FIFA Arab Cup knockout stage",
    "2025 FIFA Arab Cup final",
]

# MULTILINE matters: this is used both per-line (.match) and to test whether a
# whole table contains lineup rows (.search), and without it the ^ anchor makes
# the second use silently fail on every table.
POS_ROW = re.compile(r"^\|\s*([A-Z]{1,3})\s*\|\|\s*'''(\d+)'''\s*\|\|(.*)$",
                     re.M)
LINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
BOX = re.compile(r"\{\{football box", re.I)


def fetch_page(title, pause=1.0):
    """Raw wikitext, cached. Never re-requested."""
    WIKI.mkdir(parents=True, exist_ok=True)
    f = WIKI / (title.replace(" ", "_") + ".wikitext")
    if f.exists() and f.stat().st_size > 500:
        return f.read_text(encoding="utf-8")
    r = requests.get("https://en.wikipedia.org/w/index.php",
                     params={"title": title, "action": "raw"}, headers=UA, timeout=30)
    time.sleep(pause)
    if r.status_code != 200 or len(r.text) < 500:
        return None
    f.write_text(r.text, encoding="utf-8")
    return r.text


def _clean_name(cell):
    """Display name from the first wiki link in a cell."""
    for target, display in LINK.findall(cell):
        if "Captain" in target or target.startswith(("File:", "Image:")):
            continue
        return (display or target).strip()
    return None


def _parse_lineup(block):
    """One team's lineup table -> rows plus the manager."""
    rows, on_bench, manager = [], False, None
    lines = block.splitlines()
    for i, line in enumerate(lines):
        # Both spellings occur, sometimes on the same page: the 2021 group
        # articles say "Substitutes:" and the 2025 ones "Substitutions:".
        # Matching only one silently counted the whole bench as starters,
        # giving 16 starters a side.
        if "'''Substitut" in line:
            on_bench = True
            continue
        if "'''Manager:'''" in line:
            for nxt in lines[i + 1:]:
                name = _clean_name(nxt)
                if name:
                    manager = name
                    break
            break
        m = POS_ROW.match(line.strip())
        if not m:
            continue
        pos, num, rest = m.group(1), int(m.group(2)), m.group(3)
        name = _clean_name(rest)
        if not name:
            continue
        subon = re.search(r"\{\{subon\|(\d+)", rest)
        suboff = re.search(r"\{\{suboff\|(\d+)", rest)
        yel = re.search(r"\{\{yel\|(\d+)", rest)
        sentoff = re.search(r"\{\{sentoff\|\d+\|(\d+)", rest)
        rows.append({
            "position": pos, "shirt_no": num, "player": name,
            "role": "sub" if on_bench else "starter",
            "is_captain": "|c]]" in rest,
            "on_minute": int(subon.group(1)) if subon else None,
            "off_minute": int(suboff.group(1)) if suboff else None,
            "yellow_minute": int(yel.group(1)) if yel else None,
            "red_minute": int(sentoff.group(1)) if sentoff else None,
        })
    # a bench name with no subon never actually played
    for r in rows:
        if r["role"] == "sub" and r["on_minute"] is None:
            r["role"] = "unused"
    return rows, manager


def parse_page(text, team_code="JOR"):
    """Every match on the page involving `team_code`, with that team's lineup."""
    starts = [m.start() for m in BOX.finditer(text)]
    out = []
    for k, s in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(text)
        chunk = text[s:end]
        head = chunk[:900]

        d = re.search(r"date\s*=\s*\{\{Start date\|(\d{4})\|(\d{1,2})\|(\d{1,2})", head)
        t1 = re.search(r"team1\s*=\s*\{\{fb(?:-rt)?\|(\w+)\}\}", head)
        t2 = re.search(r"team2\s*=\s*\{\{fb(?:-rt)?\|(\w+)\}\}", head)
        if not (d and t1 and t2):
            continue
        if team_code not in (t1.group(1), t2.group(1)):
            continue

        side = "team1" if t1.group(1) == team_code else "team2"
        # Capture the WHOLE goals field, not one line. Two traps here:
        # a scorer list spans several lines, so `.` must cross newlines and
        # the match must stop at the next "| field =" rather than run on;
        # and \s* after the = would itself cross a newline, so when Jordan
        # scored none it swallowed the blank line and captured the
        # OPPONENT's goals field instead. Horizontal whitespace only.
        goals = re.search(
            rf"goals{side[-1]}[ \t]*=[ \t]*(.*?)(?=\n[ \t]*\|[ \t]*\w+[ \t]*=|\n\}}\}})",
            chunk, re.S)
        # The same field for the other team. Needed to rebuild the scoreboard:
        # without the opponent's goal minutes there is no match state, and
        # these ten matches are where the post-2022 outperformance lives.
        other = "1" if side == "team2" else "2"
        opp_goals = re.search(
            rf"goals{other}[ \t]*=[ \t]*(.*?)(?=\n[ \t]*\|[ \t]*\w+[ \t]*=|\n\}}\}})",
            chunk, re.S)
        sc = re.search(r"score\s*=\s*(?:\{\{score link\|[^|]*\|)?\s*([0-9]+)[^0-9]+([0-9]+)", head)
        stadium = re.search(r"stadium\s*=\s*(.+)", head)
        att = re.search(r"attendance\s*=\s*([\d,]+)", head)
        ref = re.search(r"referee\s*=\s*(.+)", head)

        # lineup tables follow the box, team1 first
        # Split on ANY table opening and keep the ones that actually contain
        # lineup rows, rather than matching a style string: the 2021 articles
        # write "font-size: 90%" with a space and the 2025 ones without, and
        # keying on that spelling silently returned zero lineups for 2021.
        tables = re.split(r"\n\{\|", chunk)[1:]
        tables = [t for t in tables if POS_ROW.search(t)]
        if len(tables) < 2:
            continue
        block = tables[0] if side == "team1" else tables[1]
        rows, manager = _parse_lineup(block)
        if not rows:
            continue

        gf, ga = (int(sc.group(1)), int(sc.group(2))) if sc else (None, None)
        if side == "team2":
            gf, ga = ga, gf

        out.append({
            "date": f"{d.group(1)}-{int(d.group(2)):02d}-{int(d.group(3)):02d}",
            "opponent_code": t2.group(1) if side == "team1" else t1.group(1),
            "home_or_away": "home" if side == "team1" else "away",
            "gf": gf, "ga": ga,
            "stadium": _clean_name(stadium.group(1)) if stadium else None,
            "attendance": int(att.group(1).replace(",", "")) if att else None,
            "referee": _clean_name(ref.group(1)) if ref else None,
            "manager": manager,
            "goals_raw": goals.group(1).strip() if goals else None,
            "opp_goals_raw": opp_goals.group(1).strip() if opp_goals else None,
            "lineup": rows,
        })
    return out


def collect(pages=PAGES):
    matches = []
    for title in pages:
        text = fetch_page(title)
        if text is None:
            print(f"  !! could not fetch {title}")
            continue
        found = parse_page(text)
        print(f"  {title:40s} {len(found)} Jordan matches")
        for f in found:
            f["wiki_page"] = title
        matches += found
    matches.sort(key=lambda m: m["date"])
    return matches


def to_frames(matches):
    mrows, lrows = [], []
    for m in matches:
        mid = "ac_" + m["date"].replace("-", "")
        mrows.append({k: v for k, v in m.items() if k != "lineup"} | {"match_id": mid})
        for r in m["lineup"]:
            lrows.append(r | {"match_id": mid, "date": m["date"],
                              "player_id": pd.NA, "source": "wikipedia"})
    return pd.DataFrame(mrows), pd.DataFrame(lrows)


def crosswalk(lineups):
    """Propose FBref player ids for Wikipedia names -- for review, not for use.

    Exact and normalised matching only. Anything that does not match exactly is
    listed as unresolved rather than guessed at: a wrong match here invents a
    player history, which is precisely the failure mode CLAUDE.md bans
    name-keying to avoid.
    """
    apps = pd.read_csv(OUT / "appearances.csv")
    fb = (apps[apps.squad_id == "3e22f0fa"][["Player", "player_id"]]
          .dropna().drop_duplicates())

    def norm(s):
        s = str(s).lower().replace("-", " ").replace("'", "").replace("’", "")
        return re.sub(r"\s+", " ", re.sub(r"[^a-z\s]", "", s)).strip()

    fb["key"] = fb.Player.map(norm)
    lookup = fb.drop_duplicates("key").set_index("key")

    names = sorted(lineups.player.dropna().unique())
    keys = list(lookup.index)
    rows = []
    for n in names:
        k = norm(n)
        hit = lookup.loc[k] if k in lookup.index else None
        # For anything that does not match exactly, offer the closest FBref
        # name as a SUGGESTION with its similarity score. It is deliberately
        # not applied: "Yaseen Al-Bakhit" and "Yaseen Al Bakheet" are plainly
        # the same person, but a close string is not proof, and a wrong accept
        # here invents a career. A human confirms these.
        suggestion = score = pd.NA
        if hit is None:
            close = difflib.get_close_matches(k, keys, n=1, cutoff=0.6)
            if close:
                suggestion = lookup.loc[close[0], "Player"]
                score = round(difflib.SequenceMatcher(None, k, close[0]).ratio(), 3)
        rows.append({"wiki_name": n,
                     "fbref_name": hit.Player if hit is not None else pd.NA,
                     "player_id": hit.player_id if hit is not None else pd.NA,
                     "status": "exact" if hit is not None else "UNRESOLVED",
                     "suggested_fbref_name": suggestion,
                     "similarity": score,
                     "confirmed": pd.NA})
    return pd.DataFrame(rows)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    matches = collect()
    mdf, ldf = to_frames(matches)
    mdf.to_csv(OUT / "arabcup_matches.csv", index=False)
    ldf.to_csv(OUT / "arabcup_lineups.csv", index=False)
    print(f"\narabcup_matches.csv  {len(mdf)} matches")
    print(f"arabcup_lineups.csv  {len(ldf)} player rows")
    print(ldf.groupby("role").size().to_string())

    cw = crosswalk(ldf)
    # Carry forward any confirmations already recorded. Rebuilding the
    # crosswalk from scratch would silently discard human review, and the
    # merge step then drops those players -- which is exactly what happened
    # the first time this module was re-run.
    prior_path = OUT / "arabcup_name_crosswalk.csv"
    if prior_path.exists():
        prior = pd.read_csv(prior_path)
        if "confirmed" in prior.columns:
            keep = dict(zip(prior.wiki_name, prior.confirmed))
            cw["confirmed"] = cw.wiki_name.map(keep).fillna(cw.confirmed)
            n_kept = cw.confirmed.notna().sum()
            if n_kept:
                print(f"  carried forward {n_kept} existing confirmation(s)")
    cw.to_csv(prior_path, index=False)
    n_ok = (cw.status == "exact").sum()
    print(f"\narabcup_name_crosswalk.csv  {n_ok}/{len(cw)} names matched to an "
          f"FBref player id; {len(cw) - n_ok} unresolved")
    return mdf, ldf, cw


if __name__ == "__main__":
    build()
