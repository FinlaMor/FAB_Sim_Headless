"""Import a fabrary deck (URL or id) into the engine's decks/ directory.

    python -m python.datasets.fabrary.import_deck <url-or-id> [--dry]

Fetches via the public AppSync API, converts to the engine deck format, and
validates every slug against Talishar's card universe before writing.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from .client import Fabrary
from .convert import convert_deck, validate, write_deck, talishar_slugs

_REPO = Path(__file__).resolve().parents[3]
_DECKS = _REPO / "decks"


def deck_id_of(url_or_id: str) -> str:
    """Accept a full fabrary URL or a bare deck id."""
    s = url_or_id.strip()
    if "fabrary" in s or "/" in s:
        s = re.split(r"[?#]", s.rsplit("/", 1)[-1])[0]
    return s


def import_one(url_or_id: str, fab: Fabrary | None = None, known=None,
               write: bool = True, min_registered: int = 0) -> dict:
    fab = fab or Fabrary()
    did = deck_id_of(url_or_id)
    raw = fab.get_deck(did)
    deck = convert_deck(raw)
    unknown = validate(deck, known)
    registered = deck["registered_total"]
    info = {
        "deckId": did, "hero": deck["hero"], "format": raw.get("format"),
        "name": raw.get("name"), "equipment": len(deck["equipment"]),
        "deck_cards": deck["maindeck_count"], "registered_total": registered,
        "sideboard": len(deck["sideboard"]) + len(deck["sideboard_equipment"]),
        "unknown_slugs": unknown,
    }
    info["below_min"] = registered < min_registered
    if write and not unknown and not info["below_min"]:
        out = _DECKS / f"cc_{deck['hero']}_{did[-6:].lower()}.json"
        write_deck(deck, out)
        info["written"] = str(out.relative_to(_REPO))
    return info


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry" in sys.argv
    known = talishar_slugs()
    fab = Fabrary()
    for a in args:
        info = import_one(a, fab=fab, known=known, write=not dry)
        print(f"\n=== {info['name']} [{info['format']}] hero={info['hero']} ===")
        print(f"   maindeck={info['deck_cards']} equipment={info['equipment']} "
              f"sideboard={info['sideboard']} registered_total={info['registered_total']}")
        if info["unknown_slugs"]:
            print(f"   !! {len(info['unknown_slugs'])} UNKNOWN slugs: {info['unknown_slugs'][:12]}")
        else:
            print("   all slugs valid against Talishar universe [OK]",
                  f"-> {info.get('written','(dry run, not written)')}")
