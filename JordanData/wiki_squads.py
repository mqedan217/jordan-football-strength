# -*- coding: utf-8 -*-
"""Where Jordan's players earn their club football, from tournament squad lists.

FBref cannot answer this. Its soccerdata reader covers nine leagues -- the Big 5
plus a few international tournaments -- and none of the Jordanian, Gulf or Asian
leagues a Jordan international actually plays in, so there is no endpoint that
maps a player to a club. Per-player FBref pages would be ~95 raw fetches against
a host that returns bot-detection errors after two consecutive ones.

Wikipedia's tournament squad-list articles carry it already, one templated row
per player:

    {{nat fs g player|no=2|pos=DF|name=[[Mohammad Abu Hasheesh]]
      |age=...|caps=26|goals=0|club=[[Al Ahed FC|Al-Ahed]]|clubnat=LBN}}

`pos` gives the line, `club` the club, `clubnat` the country of that club. Five
pages cover Jordan's five tournaments in the window.

🚨 This is a SNAPSHOT SERIES, not a career average. It records the club a player
was at when a squad was named, at five moments. It cannot see friendly call-ups,
and it cannot see a transfer between tournaments. Do not describe it as "the
league he played in while called up".

🚨 Count PLAYERS, not squad places. A regular named in three squads appears three
times, which weights him triply; collapsing to one row per player per era moves
the forward result from 56%->56% to 47%->29%. `player_eras()` does the collapse.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from wiki_arabcup import fetch_page

ROOT = Path(__file__).resolve().parents[1]
SQUAD = ROOT / "outputs" / "squad"

# (article, month the squad was named, era relative to the 2022-07-01 cut)
SQUADS = [
    ("2015 AFC Asian Cup squads", "2015-01", "pre"),
    ("2019 AFC Asian Cup squads", "2019-01", "pre"),
    ("2021 FIFA Arab Cup squads", "2021-11", "pre"),
    ("2023 AFC Asian Cup squads", "2024-01", "post"),   # played January 2024
    ("2025 FIFA Arab Cup squads", "2025-12", "post"),
]

LINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")

# the source writes some countries long and some as codes, on the same page
CODE_FIX = {"BAHRAIN": "BHR", "JORDAN": "JOR", "SAUDI ARABIA": "KSA",
            "IRAQ": "IRQ", "QATAR": "QAT", "KUWAIT": "KUW",
            "OMAN": "OMA", "LEBANON": "LBN", "LIBYA": "LBY",
            "MALAYSIA": "MAS", "CYPRUS": "CYP", "FRANCE": "FRA"}

# Region, for the question "did anyone go to Europe". Deliberately explicit
# rather than inferred: a wrong region silently invents a migration.
EUROPE = {"FRA", "CYP", "GER", "ESP", "ITA", "ENG", "POR", "NED", "BEL", "SUI",
          "AUT", "GRE", "TUR", "POL", "CZE", "CRO", "SRB", "ROU", "BUL", "SWE",
          "NOR", "DEN", "FIN", "HUN", "SVK", "SVN", "UKR", "RUS", "SCO", "WAL"}
ARAB_ASIA = {"IRQ", "BHR", "KSA", "QAT", "KUW", "UAE", "OMA", "LBN", "LBY",
             "EGY", "SYR", "PLE", "YEM", "MAS", "IDN", "THA", "IND", "CHN",
             "KOR", "JPN", "UZB", "IRN"}


def _field(key: str, line: str):
    m = re.search(r"\|\s*" + key + r"\s*=\s*([^|}\n]*)", line)
    return m.group(1).strip() if m else None


def _linktext(v: str | None) -> str:
    m = LINK.search(v or "")
    return (m.group(2) or m.group(1)).strip() if m else (v or "").strip()


def _jordan_section(txt: str) -> str | None:
    """Jordan's block, ending at the NEXT HEADING OF ANY LEVEL.

    These pages head each team at level 3 (``=== Jordan ===``), so a boundary
    anchored on level 2 never fires and the parse runs on through every squad
    that follows -- it showed up as 69 players in a 23-man squad, and as Saudi
    Arabia holding 19% of Jordan's places.
    """
    m = re.search(r"==+\s*Jordan\s*==+", txt)
    if not m:
        return None
    seg = txt[m.end():]
    nxt = re.search(r"\n=+[^=\n]+=+[ \t]*\n", seg)
    return seg[:nxt.start()] if nxt else seg


def region(nat: str) -> str:
    if nat == "JOR":
        return "Jordan"
    if nat in EUROPE:
        return "Europe"
    if nat in ARAB_ASIA:
        return "Arab / Asia"
    return "other"


def squad_places() -> pd.DataFrame:
    """One row per player per tournament squad."""
    rows = []
    for title, when, era in SQUADS:
        txt = fetch_page(title)
        if not txt:
            print(f"  !! {title}: not fetched")
            continue
        seg = _jordan_section(txt)
        if seg is None:
            print(f"  !! {title}: no Jordan heading")
            continue
        n = 0
        for line in seg.splitlines():
            if "nat fs" not in line or "player" not in line:
                continue
            pos, club = _field("pos", line), _field("club", line)
            if not (pos and club):
                continue
            nat = (_field("clubnat", line) or "JOR").upper()
            nat = CODE_FIX.get(nat, nat)
            rows.append({"squad": title, "when": when, "era": era,
                         "player": _linktext(_field("name", line)),
                         "pos": pos.upper(),
                         "club": _linktext(club),
                         "nat": nat, "region": region(nat)})
            n += 1
        print(f"  {title:32s} {when}  {era:4s}  {n:2d} players")
    return pd.DataFrame(rows)


def player_eras(places: pd.DataFrame) -> pd.DataFrame:
    """One row per player per era: his modal club country in that era.

    Squad places over-weight the regulars -- a man in three squads counts three
    times -- and the forward result changes sign depending on which you use.
    """
    def modal_nat(g):
        """Most common club country in the era; ties go to the latest squad.

        Taking mode() of `nat` and mode() of `region` independently is a bug:
        a player split between a Jordanian and a foreign club inside one era
        ties both, pandas resolves each alphabetically, and the row ends up
        saying nat=JOR with region="Arab / Asia". Seven of ninety-one rows
        contradicted themselves that way. One source of truth, region derived.
        """
        c = g.nat.value_counts()
        top = c[c == c.max()].index
        if len(top) == 1:
            return top[0]
        return g.sort_values("when").nat.iloc[-1]

    out = (places.groupby(["player", "era"], group_keys=True)
           .apply(lambda g: pd.Series(
               {"nat": modal_nat(g),
                "pos": g.pos.mode().iloc[0],
                "squads": g.squad.nunique()}), include_groups=False)
           .reset_index())
    out["region"] = out.nat.map(region)
    return out[["player", "era", "nat", "region", "pos", "squads"]]


if __name__ == "__main__":
    SQUAD.mkdir(parents=True, exist_ok=True)
    places = squad_places()
    if places.empty:
        raise SystemExit("parsed nothing")
    pe = player_eras(places)
    places.to_csv(SQUAD / "tournament_squad_places.csv", index=False)
    pe.to_csv(SQUAD / "tournament_player_eras.csv", index=False)
    print(f"\n{len(places)} squad places, {places.player.nunique()} players, "
          f"{len(pe)} player-eras")
    print(f"wrote {SQUAD / 'tournament_squad_places.csv'}")
    print(f"wrote {SQUAD / 'tournament_player_eras.csv'}")
