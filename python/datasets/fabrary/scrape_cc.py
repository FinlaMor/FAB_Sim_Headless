"""Pull a handful of Classic Constructed public decks across adult heroes and
import them into the engine's decks/ directory.

    python -m python.datasets.fabrary.scrape_cc [N] [--dry]

CC is played with the ADULT hero; fabrary's heroIdentifier uses dashes (the
Talishar heroId uses underscores), so we flip '_' -> '-' for the query.
"""
from __future__ import annotations

import sys

from .client import Fabrary
from .heroes import cc_legal_hero_ids
from .convert import talishar_slugs
from .import_deck import import_one


def fabrary_hero_id(hero_id: str) -> str:
    # slug_index.json keys are now the native fabrary cardIdentifier (dashed),
    # which IS the heroIdentifier — so this is a pass-through. (Kept for the
    # old underscore call sites; harmless on already-dashed ids.)
    return hero_id.replace("_", "-")


def pull(target: int = 8, write: bool = True, min_registered: int = 75,
         candidates_per_hero: int = 6) -> list[dict]:
    """Keep the first CC deck per hero whose registered pool is >= min_registered
    (so the future sideboard bot has a full main+sideboard pool to pick from)."""
    fab = Fabrary()
    known = talishar_slugs()
    kept: list[dict] = []
    for hid in cc_legal_hero_ids():
        if len(kept) >= target:
            break
        fhid = fabrary_hero_id(hid)
        try:
            decks = fab.decks_by_hero(fhid, max_decks=10)
        except Exception as e:  # noqa: BLE001
            print(f"  {fhid}: ERROR {e}")
            continue
        cc = [d for d in decks if d.get("format") == "Classic Constructed"]
        chosen = None
        for d in cc[:candidates_per_hero]:
            info = import_one(d["deckId"], fab=fab, known=known, write=write,
                              min_registered=min_registered)
            if info["unknown_slugs"]:
                print(f"  {fhid}: skip {d['deckId']} ({len(info['unknown_slugs'])} unknown slugs)")
                continue
            if info["below_min"]:
                print(f"  {fhid}: skip {d['deckId']} (registered {info['registered_total']} < {min_registered})")
                continue
            chosen = info
            break
        if chosen:
            kept.append(chosen)
            print(f"  {fhid}: KEPT main={chosen['deck_cards']} equip={chosen['equipment']} "
                  f"side={chosen['sideboard']} reg={chosen['registered_total']} -> {chosen.get('written')}")
        else:
            print(f"  {fhid}: no qualifying CC deck")
    print(f"\nkept {len(kept)}/{target} decks (>= {min_registered} registered cards)")
    return kept


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if args else 8
    pull(target=n, write="--dry" not in sys.argv)
