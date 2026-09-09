"""Print a readable match report from the parsed tables.

The cached HTML in cache/raw/ is a DOM capture with no stylesheets attached,
so opening one in a browser gives an unstyled wall of text behind a cookie
banner. The content is all there -- that is what the parser reads -- but it is
not a page you can look at. This renders the same match as something legible,
for spot-checking a parse or just reading a game back.

    python match_report.py                  # most recent Jordan match
    python match_report.py 140e19d9         # by match id
    python match_report.py 2026-06-16       # by date
    python match_report.py Argentina        # by opponent (most recent)
"""

import sys
from pathlib import Path

import pandas as pd

# Windows consoles default to cp1252, which cannot encode names like
# Arnautovic or Kalajdzic as they are actually spelled.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SQUAD_DIR = ROOT / "outputs" / "squad"
JORDAN = "3e22f0fa"


def _load():
    m = pd.read_csv(SQUAD_DIR / "matches.csv")
    m["date"] = pd.to_datetime(m["Date"], errors="coerce")
    return (m.dropna(subset=["match_id"]).sort_values("date"),
            pd.read_csv(SQUAD_DIR / "appearances.csv"),
            pd.read_csv(SQUAD_DIR / "events.csv"),
            pd.read_csv(SQUAD_DIR / "lineups.csv"))


def find(matches, query=None):
    if not query:
        return matches.iloc[-1]
    hit = matches[matches.match_id == query]
    if len(hit):
        return hit.iloc[0]
    hit = matches[matches.date.astype(str).str.startswith(query)]
    if len(hit):
        return hit.iloc[-1]
    hit = matches[matches.Opponent.astype(str).str.contains(query, case=False, na=False)]
    if len(hit):
        return hit.iloc[-1]
    raise SystemExit(f"no match found for {query!r}")


def report(query=None):
    matches, apps, events, lineups = _load()
    m = find(matches, query)
    mid = m.match_id
    opp = str(m.Opponent)
    opp = opp[3:] if len(opp) > 3 and opp[2] == " " else opp

    home_away = {"Home": "vs", "Away": "away to", "Neutral": "vs (neutral)"}
    line = f"Jordan {home_away.get(m.Venue, 'vs')} {opp}   {m.GF}-{m.GA}"
    print("=" * 68)
    print(line)
    print(f"{m.date.date()}   {m.Comp}" + (f" - {m.Round}" if pd.notna(m.Round) else ""))
    print("=" * 68)

    bits = []
    for label, col in (("Venue", "venue"), ("Attendance", "attendance"),
                       ("Referee", "referee")):
        if col in m.index and pd.notna(m.get(col)):
            bits.append(f"{label}: {m[col]}")
    if pd.notna(m.get("Formation")):
        bits.append(f"Formation: {m.Formation} (opp {m.get('Opp Formation')})")
    if pd.notna(m.get("Poss")):
        bits.append(f"Possession: {m.Poss}%")
    for b in bits:
        print("  " + b)

    mg = [c for c in ["home_manager", "away_manager"] if pd.notna(m.get(c))]
    if mg:
        print("  Managers: " + " / ".join(str(m[c]) for c in mg))
    if pd.notna(m.get("Captain")):
        print(f"  Captain: {m.Captain}")

    a = apps[(apps.match_id == mid) & (apps.squad_id == JORDAN)]
    if len(a):
        print("\nJORDAN")
        xi = a[a.role == "starter"]
        for _, p in xi.iterrows():
            num = "" if pd.isna(p["#"]) else f"{int(p['#']):>2} "
            print(f"  {num}{str(p.Pos):<6} {p.Player:<24} {p.Min:>3}'  age {p.Age}")
        sub = a[a.role == "sub"]
        if len(sub):
            print("  -- came on --")
            for _, p in sub.iterrows():
                num = "" if pd.isna(p["#"]) else f"{int(p['#']):>2} "
                print(f"  {num}{str(p.Pos):<6} {p.Player:<24} {p.Min:>3}'  age {p.Age}")

    e = events[events.match_id == mid].sort_values(["minute", "minute_stoppage"])
    if len(e):
        print("\nTIMELINE")
        icon = {"goal": "GOAL", "own_goal": "OWN GOAL", "penalty_goal": "PEN",
                "sub": "sub", "yellow": "yellow", "red": "RED",
                "second_yellow": "2nd yellow", "penalty_missed": "pen missed"}
        for _, ev in e.iterrows():
            mins = f"{int(ev.minute)}" + (f"+{int(ev.minute_stoppage)}"
                                          if ev.minute_stoppage else "")
            who = ev.player if pd.notna(ev.player) else "?"
            extra = ""
            if pd.notna(ev.player_2):
                extra = (f" for {ev.player_2}" if ev.type == "sub"
                         else f" (assist {ev.player_2})")
            score = ""
            if pd.notna(ev.score_home):
                score = f"  [{int(ev.score_home)}-{int(ev.score_away)}]"
            print(f"  {mins:>5}'  {icon.get(ev.type, ev.type):<10} {who}{extra}{score}")

    print(f"\nfbref: https://fbref.com/en/matches/{mid}/")
    print(f"cached: cache/raw/match_{mid}.html")


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else None)
