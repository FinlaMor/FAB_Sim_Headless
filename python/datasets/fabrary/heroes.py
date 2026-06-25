"""Canonical hero identifiers, parsed from Talishar's source of truth
(talishar/Libraries/LegalHeroesHelper.php). The file's own comment notes
these slugs are the ones "used by Bazaar/Fabrary in matchup payloads", i.e.
they double as fabrary `heroIdentifier`s.

Classic Constructed is played with the ADULT hero (young => false); Blitz /
Silver Age use the young hero (young => true). Querying the young slug is why
an earlier scrape found no CC decks.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_HELPER = (Path(__file__).resolve().parents[3] / "talishar"
           / "Libraries" / "LegalHeroesHelper.php")
# Authoritative card/format index (provided in the repo root): every card with
# its legalFormats, so we can pick exactly the CC-legal heroes (excludes young
# heroes AND adult heroes that have reached Living Legend / are CC-banned, e.g.
# Kano, Viserai).
_SLUG_INDEX = Path(__file__).resolve().parents[3] / "slug_index.json"

_ROW = re.compile(
    r"'heroId'\s*=>\s*'([^']+)'\s*,\s*'name'\s*=>\s*'([^']+)'\s*,\s*'young'\s*=>\s*(true|false)")


def all_heroes() -> list[dict]:
    if not _HELPER.is_file():
        return []
    txt = _HELPER.read_text(encoding="utf-8", errors="ignore")
    return [{"heroId": hid, "name": name, "young": yng == "true"}
            for hid, name, yng in _ROW.findall(txt)]


def cc_hero_ids() -> list[str]:
    """Adult heroes (young == false) from Talishar's helper. NOTE: this is the
    broad adult set (~54) and still includes Living-Legend heroes that are no
    longer CC-legal. Prefer cc_legal_hero_ids() for scraping."""
    return [h["heroId"] for h in all_heroes() if not h["young"]]


def cc_legal_hero_ids() -> list[str]:
    """The authoritative Classic-Constructed-legal hero pool: heroes whose
    `legalFormats` in slug_index.json include 'Classic Constructed'. Excludes
    young heroes and CC-banned (Living Legend) adults. ~38 heroes.

    slug_index.json is kept in the fabrary/cards NATIVE format (see
    tools/data_update/gen_slug_index.mjs): keys are the dashed `cardIdentifier`,
    which is *exactly* fabrary's `heroIdentifier` — so the returned ids are
    used directly as scrape queries (no underscore/dash conversion)."""
    if not _SLUG_INDEX.is_file():
        return []
    by_slug = json.loads(_SLUG_INDEX.read_text(encoding="utf-8"))["by_slug"]
    out = [s for s, c in by_slug.items()
           if "Hero" in (c.get("types") or [])
           and "Classic Constructed" in (c.get("legalFormats") or [])]
    return sorted(out)


if __name__ == "__main__":
    cc = cc_legal_hero_ids()
    print(f"{len(cc)} CC-legal heroes:")
    for h in cc:
        print("  ", h)
