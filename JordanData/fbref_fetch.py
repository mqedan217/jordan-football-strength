"""Fetch Jordan's FBref pages to disk. Slowly, once, and resumably.

Two phases, strictly ordered:

  1. season pages   -- /en/squads/3e22f0fa/<season>/Jordan-Men-Stats
  2. match reports  -- /en/matches/<8-hex>/<slug>, discovered from phase 1

Nothing here parses anything beyond what it needs to find the next URL, and
nothing here ever re-requests a page that is already on disk. Deleting a file
in cache/raw/ is the only way to refetch it. That makes the whole run
resumable: interrupt it, run it again, it picks up where it stopped.

FBref blocks plain HTTP clients, so soccerdata drives a real headless browser.
On top of its own pacing this module waits DELAY seconds between requests and
takes a longer break every REST_EVERY pages. The point is not to finish fast --
being blocked halfway costs far more than the extra minutes.

If a response comes back looking like a block or a captcha, the fetcher backs
off once and then stops rather than hammering. Run it again later; the cache
keeps everything already collected.
"""

import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
RAW = CACHE / "raw"
os.environ.setdefault("SOCCERDATA_DIR", str(CACHE))

import pandas as pd  # noqa: E402
from bs4 import BeautifulSoup, Comment  # noqa: E402

SQUAD = "3e22f0fa"
BASE = "https://fbref.com"
DELAY = (12.0, 25.0)        # seconds between requests
REST_EVERY = 20             # pages
REST_FOR = (60.0, 100.0)    # seconds
BLOCK_MARKERS = ("rate limited", "too many requests", "captcha",
                 "403 forbidden", "access denied")
MIN_BYTES = 20_000

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import soccerdata as sd
        _reader = sd.FBref(leagues="INT-World Cup", seasons="2022")
    return _reader


def looks_blocked(html):
    if len(html) < MIN_BYTES:
        return f"suspiciously small ({len(html)} bytes)"
    low = html[:6000].lower()
    for m in BLOCK_MARKERS:
        if m in low:
            return f"contains {m!r}"
    return None


def fetch(url, dest, counter=None):
    """Download `url` to `dest` unless already present. Returns the HTML."""
    if dest.exists() and dest.stat().st_size > MIN_BYTES:
        return dest.read_text(encoding="utf-8", errors="replace")

    if counter is not None and counter["n"] > 0:
        if counter["n"] % REST_EVERY == 0:
            nap = random.uniform(*REST_FOR)
            print(f"    ... resting {nap:.0f}s after {counter['n']} pages",
                  flush=True)
            time.sleep(nap)
        else:
            time.sleep(random.uniform(*DELAY))

    print(f"  fetching {url}", flush=True)
    with _get_reader().get(url, filepath=dest) as fh:
        fh.read()
    if counter is not None:
        counter["n"] += 1

    html = dest.read_text(encoding="utf-8", errors="replace")
    why = looks_blocked(html)
    if why:
        print(f"  !! response looks blocked: {why}. Backing off 300s.", flush=True)
        dest.unlink(missing_ok=True)
        time.sleep(300)
        with _get_reader().get(url, filepath=dest) as fh:
            fh.read()
        html = dest.read_text(encoding="utf-8", errors="replace")
        why = looks_blocked(html)
        if why:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"blocked twice on {url} ({why}). Stopping; rerun later to resume.")
    return html


def all_tables(html):
    """FBref hides most tables inside HTML comments."""
    soup = BeautifulSoup(html, "lxml")
    tables = list(soup.find_all("table"))
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in c:
            tables += BeautifulSoup(c, "lxml").find_all("table")
    return soup, tables


def seasons_from_history(html):
    _, tables = all_tables(html)
    for t in tables:
        if t.get("id") == "nat_tm_summary":
            df = pd.read_html(__import__("io").StringIO(str(t)))[0]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [" ".join(str(x) for x in c if "Unnamed" not in str(x)).strip()
                              for c in df.columns]
            yrs = sorted(int(s) for s in df.Season.dropna().astype(int) if s >= 2014)
            return yrs
    return []


def match_links(season_html):
    """Match-report URLs from the fixtures table only.

    The page also carries FBref's site-wide 'today's matches' sidebar; taking
    every /en/matches/ href off the page pulls in unrelated fixtures, so this
    reads matchlogs_for and nothing else.
    """
    _, tables = all_tables(season_html)
    out = []
    for t in tables:
        if t.get("id", "").startswith("matchlogs"):
            for a in t.find_all("a", href=True):
                if re.match(r"^/en/matches/[0-9a-f]{8}/", a["href"]):
                    out.append(a["href"])
    return sorted(set(out))


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    counter = {"n": 0}

    hist = fetch(f"{BASE}/en/squads/{SQUAD}/history/", RAW / "jordan_history.html",
                 counter)
    seasons = seasons_from_history(hist)
    print(f"seasons to fetch ({len(seasons)}): {seasons}\n")

    print("=== phase 1: season pages ===", flush=True)
    links = {}
    for yr in seasons:
        dest = RAW / f"season_{yr}.html"
        html = fetch(f"{BASE}/en/squads/{SQUAD}/{yr}/Jordan-Men-Stats", dest, counter)
        links[yr] = match_links(html)
        print(f"  season {yr}: {len(links[yr])} matches", flush=True)

    every = sorted({h for v in links.values() for h in v})
    print(f"\n{len(every)} distinct match reports "
          f"({sum(len(v) for v in links.values())} rows across seasons)")

    have = sum(1 for h in every
               if (RAW / f"match_{h.split('/')[3]}.html").exists())
    print(f"{have} already cached, {len(every) - have} to fetch\n")

    print("=== phase 2: match reports ===", flush=True)
    for i, href in enumerate(every, 1):
        mid = href.split("/")[3]
        dest = RAW / f"match_{mid}.html"
        if dest.exists() and dest.stat().st_size > MIN_BYTES:
            continue
        print(f"[{i}/{len(every)}] {mid}", flush=True)
        fetch(BASE + href, dest, counter)

    pd.DataFrame(
        [{"season": yr, "href": h, "match_id": h.split("/")[3]}
         for yr, v in links.items() for h in v]
    ).to_csv(CACHE / "match_index.csv", index=False)
    print(f"\ndone. {counter['n']} pages fetched this run. "
          f"index -> cache/match_index.csv")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"\nSTOPPED: {e}", file=sys.stderr)
        sys.exit(1)
