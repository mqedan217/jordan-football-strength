"""Turn cached FBref HTML into tidy tables. Reads disk only -- never the network.

Produces, in outputs/squad/:

    matches.csv        one row per match, all metadata
    appearances.csv    one row per player per match
    events.csv         goals, cards, substitutions with minutes
    lineups.csv        starting XI and bench as listed, with formation
    coverage.csv       per season per column, is this actually collected?
    match_coverage.csv every results.csv match, with its FBref id or a null

Design notes worth knowing before editing:

* FBref hides most tables inside HTML comments, so every lookup goes through
  `all_tables`, which digs them out.
* Two things named the same are not the same: the fixtures table on a season
  page already carries formation, possession, attendance, captain and referee.
  Match reports add lineups, per-player stats and the event timeline.
* Event kinds are classified from the text, not the CSS class -- FBref marks
  own goals with the substitution class, which would silently mislabel them.
* Nothing here treats absence as zero. A column that is empty stays empty, and
  `coverage.csv` decides per season whether it was ever collected at all.
"""

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, Comment

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "cache" / "raw"
OUT = ROOT / "outputs" / "squad"
SQUAD = "3e22f0fa"

FORMATION_RE = re.compile(r"^(?P<team>.+?)\s*\((?P<formation>\d(?:-\d+)+)\)\s*$")
MINUTE_RE = re.compile(r"(\d+)(?:\+(\d+))?")


def all_tables(html):
    soup = BeautifulSoup(html, "lxml")
    tables = list(soup.find_all("table"))
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in c:
            tables += BeautifulSoup(c, "lxml").find_all("table")
    return soup, tables


def flat(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(str(x) for x in c if "Unnamed" not in str(x)).strip()
                      for c in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    return df


def read(table):
    return flat(pd.read_html(io.StringIO(str(table)))[0])


PLAYER_HREF = re.compile(r"^/en/players/([0-9a-f]{8})/")


def clean(s):
    """FBref pads names with non-breaking spaces; they break every join."""
    if not isinstance(s, str):
        return s
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def player_id(tr):
    """FBref's stable player id from a row, or None.

    Names are not a safe key here -- Arabic names get re-transliterated between
    seasons (Al-Taamari / Al Taamari / Al-Tamari), which would invent debuts
    and break continuity measures. The 8-hex id is stable.
    """
    for a in tr.find_all("a", href=True):
        m = PLAYER_HREF.match(a["href"])
        if m:
            return m.group(1)
    return None


def body_rows(table):
    tb = table.find("tbody")
    return tb.find_all("tr", recursive=False) if tb else []


# --------------------------------------------------------------- season pages
def parse_season(path):
    """Fixtures table from one season page, plus its match ids."""
    html = path.read_text(encoding="utf-8", errors="replace")
    season = int(re.search(r"season_(\d{4})", path.name).group(1))
    _, tables = all_tables(html)
    for t in tables:
        if not str(t.get("id", "")).startswith("matchlogs"):
            continue
        df = read(t)
        ids = []
        for tr in t.find_all("tr"):
            a = tr.find("a", href=re.compile(r"^/en/matches/[0-9a-f]{8}/"))
            if tr.find(["td", "th"]) and tr.find("td"):
                ids.append(a["href"].split("/")[3] if a else None)
        df = df[df.Date.notna() & (df.Date != "Date")].copy()
        df["match_id"] = (ids + [None] * len(df))[:len(df)]
        df["season_label"] = season
        return df
    return pd.DataFrame()


# --------------------------------------------------------------- match report
def _scorebox(soup):
    sb = soup.find("div", class_="scorebox")
    if sb is None:
        return {}
    txt = sb.get_text(" | ", strip=True)
    out = {}
    for key, pat in (("attendance", r"Attendance\s*\|\s*:\s*\|\s*([\d,]+)"),
                     ("venue", r"Venue\s*\|\s*:\s*\|\s*([^|]+)"),
                     ("officials", r"Officials\s*\|\s*:\s*\|\s*(.+?)(?:\s*\|\s*[A-Z]|$)")):
        m = re.search(pat, txt)
        if m:
            out[key] = m.group(1).strip()
    managers = re.findall(r"Manager\s*\|\s*:\s*([^|]+)", txt)
    captains = re.findall(r"Captain\s*\|\s*:\s*\|\s*([^|]+)", txt)
    for i, side in enumerate(("home", "away")):
        if i < len(managers):
            out[f"{side}_manager"] = managers[i].strip()
        if i < len(captains):
            out[f"{side}_captain"] = captains[i].strip()
    for label in ("Referee", "AR1", "AR2", "4th", "VAR"):
        m = re.search(rf"([A-Z][^|·]*?)\s*\({re.escape(label)}\)", txt)
        if m:
            out[{"4th": "fourth"}.get(label, label.lower())] = m.group(1).strip()
    return out


def _lineups(tables):
    """Lineup tables, identified by the '<Team> (4-2-3-1)' header."""
    rows = []
    for t in tables:
        head = t.find("th")
        if head is None:
            continue
        m = FORMATION_RE.match(head.get_text(" ", strip=True))
        if not m:
            continue
        team, formation = m.group("team"), m.group("formation")
        on_bench = False
        for tr in t.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if not cells:
                continue
            joined = " ".join(cells).strip().lower()
            if joined.startswith("bench"):
                on_bench = True
                continue
            if FORMATION_RE.match(" ".join(cells).strip()):
                continue
            pid = player_id(tr)
            if len(cells) >= 2 and pid:
                rows.append({"team": clean(team), "formation": formation,
                             "shirt_no": int(cells[0]) if cells[0].isdigit() else None,
                             "player": clean(cells[1]), "player_id": pid,
                             "role": "bench" if on_bench else "starter"})
    return pd.DataFrame(rows)


def _events(soup):
    rows = []
    wrap = soup.find(id="events_wrap")
    if wrap is None:
        return pd.DataFrame(rows)
    for ev in wrap.select(".event"):
        text = " ".join(ev.get_text(" ", strip=True).split())
        cls = " ".join(ev.get("class", []))
        side = "home" if re.search(r"\ba\b", cls) else (
            "away" if re.search(r"\bb\b", cls) else None)
        m = MINUTE_RE.search(text)
        minute = int(m.group(1)) if m else None
        stoppage = int(m.group(2)) if m and m.group(2) else 0
        score = re.search(r"(\d+):(\d+)", text)

        # Strip the "12’ 3:1 " prefix, then split on the em dash. Everything
        # that identifies the event kind reliably lives BEFORE the dash: FBref
        # labels own goals and penalty kicks alike as "— Substitute", so the
        # trailer alone would mislabel both.
        body = re.sub(r"^\s*\d+(?:\+\d+)?\s*[’']\s*", "", text)
        body = re.sub(r"^\s*\d+:\d+\s*", "", body)
        head, _, trailer = body.partition("—")
        head, trailer = head.strip(), trailer.strip().lower()

        player = player2 = None
        if re.search(r"\bown goal\b", head, re.I):
            kind = "own_goal"
            player = re.sub(r"\s*own goal\s*", " ", head, flags=re.I).strip()
        elif re.search(r"\bpenalty kick\b", head, re.I):
            kind = "penalty_goal"
            player = re.sub(r"\s*penalty kick\s*", " ", head, flags=re.I).strip()
        elif re.search(r"\bpenalty miss", head, re.I) or "miss" in trailer:
            kind = "penalty_missed"
            player = re.sub(r"\s*penalty miss\w*\s*", " ", head, flags=re.I).strip()
        elif " for " in head and trailer.startswith("substitute"):
            kind = "sub"
            m2 = re.match(r"(.+?)\s+for\s+(.+)$", head)
            if m2:
                player, player2 = m2.group(1).strip(), m2.group(2).strip()
        elif trailer.startswith("goal"):
            kind = "goal"
        elif "second yellow" in trailer or "2nd yellow" in trailer:
            kind = "second_yellow"
        elif "yellow" in trailer:
            kind = "yellow"
        elif "red" in trailer:
            kind = "red"
        else:
            kind = "other"

        if player is None:
            m2 = re.match(r"(.+?)\s*Assist:\s*(.+)$", head)
            if m2:
                player, player2 = m2.group(1).strip(), m2.group(2).strip()
            else:
                player = head or None
        rows.append({"minute": minute, "minute_stoppage": stoppage, "side": side,
                     "type": kind, "player": player, "player_2": player2,
                     "score_home": int(score.group(1)) if score else None,
                     "score_away": int(score.group(2)) if score else None,
                     "raw": text})
    return pd.DataFrame(rows)


def _split_age(s):
    s = str(s)
    m = re.match(r"^(\d+)-(\d+)$", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    return (int(s), np.nan) if s.isdigit() else (np.nan, np.nan)


def parse_match(path):
    html = path.read_text(encoding="utf-8", errors="replace")
    mid = re.search(r"match_([0-9a-f]{8})", path.name).group(1)
    soup, tables = all_tables(html)

    meta = {"match_id": mid, **{k: clean(v) for k, v in _scorebox(soup).items()}}

    apps = []
    for t in tables:
        tid = str(t.get("id", ""))
        m = re.match(r"^stats_([0-9a-f]{8})_summary$", tid)
        if m:
            df = read(t)
            ids = [player_id(tr) for tr in body_rows(t)]
            # read_html returns tbody rows plus the tfoot totals row; ids cover
            # tbody only, so trim to the same length before pairing.
            if len(ids) and len(df) >= len(ids):
                df = df.iloc[:len(ids)].copy()
                df["player_id"] = ids
            df = df[df.Player.notna() &
                    ~df.Player.astype(str).str.contains(r"^\d+ Players$", regex=True)]
            df.insert(0, "squad_id", m.group(1))
            apps.append(df)
    appearances = pd.concat(apps, ignore_index=True) if apps else pd.DataFrame()
    if len(appearances):
        ages = appearances["Age"].map(_split_age)
        appearances["age_years"] = [a[0] for a in ages]
        appearances["age_days"] = [a[1] for a in ages]
        appearances["match_id"] = mid

    keepers = []
    for t in tables:
        m = re.match(r"^keeper_stats_([0-9a-f]{8})$", str(t.get("id", "")))
        if m:
            df = read(t)
            df.insert(0, "squad_id", m.group(1))
            keepers.append(df)
    keeper = pd.concat(keepers, ignore_index=True) if keepers else pd.DataFrame()
    if len(keeper):
        keeper["match_id"] = mid

    lineups = _lineups(tables)
    if len(lineups):
        lineups["match_id"] = mid
    events = _events(soup)
    if len(events):
        events["match_id"] = mid

    # Starter/substitute, derived rather than read off the lineup tables.
    # Match reports before ~2017 carry no lineup tables at all, so keying
    # features on those would silently drop the entire baseline period. The
    # appearance list contains only players who actually played, and the event
    # feed names everyone who came on, so the difference is the starting XI.
    # Names are safe to match here because both come from the same page.
    if len(appearances):
        came_on = set()
        if len(events):
            came_on = {clean(p) for p in
                       events.loc[events.type == "sub", "player"].dropna()}
        appearances["Player"] = appearances["Player"].map(clean)
        appearances["role"] = np.where(
            appearances["Player"].isin(came_on), "sub", "starter")
        appearances["role_source"] = "events" if came_on else "assumed_all_start"

    for frame, cols in ((appearances, ["Player"]), (keeper, ["Player"]),
                        (events, ["player", "player_2"])):
        for c in cols:
            if len(frame) and c in frame.columns:
                frame[c] = frame[c].map(clean)

    return meta, appearances, keeper, lineups, events


# ------------------------------------------------------------------- coverage
def coverage_audit(appearances, season_of):
    """Per season per column: is this column ever actually populated?

    CLAUDE.md's rule is that a uniformly-zero column is missing until proven
    otherwise. Applied per match that is far too aggressive -- a single game
    legitimately has no red cards and no penalties. Applied per season it does
    the intended job: a column that is zero or empty in EVERY match of a season
    was not collected that season.
    """
    df = appearances.copy()
    df["season"] = df.match_id.map(season_of)
    skip = {"squad_id", "match_id", "season", "Player", "Nation", "Pos", "Age",
            "age_years", "age_days", "#"}
    rows = []
    for season, g in df.groupby("season"):
        n_matches = g.match_id.nunique()
        for col in g.columns:
            if col in skip:
                continue
            s = pd.to_numeric(g[col], errors="coerce")
            if s.notna().sum() == 0:
                frac, verdict = 1.0, "not_collected"
            else:
                per_match = g.assign(v=s).groupby("match_id").v.apply(
                    lambda x: x.fillna(0).eq(0).all())
                frac = float(per_match.mean())
                verdict = ("not_collected" if frac == 1.0
                           else "suspect" if frac > 0.9 else "ok")
            rows.append({"season": season, "column": col, "n_matches": n_matches,
                         "frac_matches_all_zero": round(frac, 3),
                         "verdict": verdict})
    return pd.DataFrame(rows).sort_values(["season", "column"])


def build():
    OUT.mkdir(parents=True, exist_ok=True)

    season_files = sorted(RAW.glob("season_*.html"))
    fixtures = pd.concat([parse_season(p) for p in season_files], ignore_index=True)
    fixtures.to_csv(OUT / "matches.csv", index=False)
    print(f"matches.csv       {len(fixtures)} rows from {len(season_files)} seasons")

    season_of = dict(zip(fixtures.match_id, fixtures.season_label))
    want = set(fixtures.match_id.dropna())
    files = [p for p in sorted(RAW.glob("match_*.html"))
             if re.search(r"match_([0-9a-f]{8})", p.name).group(1) in want]

    metas, apps, keeps, lines, evs = [], [], [], [], []
    for p in files:
        try:
            meta, a, k, l, e = parse_match(p)
        except Exception as exc:                      # noqa: BLE001
            print(f"  !! {p.name}: {type(exc).__name__}: {exc}")
            continue
        metas.append(meta)
        for acc, part in ((apps, a), (keeps, k), (lines, l), (evs, e)):
            if len(part):
                acc.append(part)

    meta_df = pd.DataFrame(metas)
    appearances = pd.concat(apps, ignore_index=True) if apps else pd.DataFrame()
    keeper = pd.concat(keeps, ignore_index=True) if keeps else pd.DataFrame()
    lineups = pd.concat(lines, ignore_index=True) if lines else pd.DataFrame()
    events = pd.concat(evs, ignore_index=True) if evs else pd.DataFrame()

    if len(meta_df):
        merged = fixtures.merge(meta_df, on="match_id", how="left")
        merged.to_csv(OUT / "matches.csv", index=False)
    for name, df in (("appearances", appearances), ("keeper", keeper),
                     ("lineups", lineups), ("events", events)):
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name + '.csv':18s}{len(df)} rows")

    if len(appearances):
        cov = coverage_audit(appearances, season_of)
        cov.to_csv(OUT / "coverage.csv", index=False)
        bad = cov[cov.verdict != "ok"]
        print(f"coverage.csv      {len(cov)} rows, "
              f"{len(bad)} season/column pairs flagged")
    print(f"\nparsed {len(files)} match reports -> {OUT}")


if __name__ == "__main__":
    build()
