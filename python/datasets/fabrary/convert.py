"""Convert a fabrary getDeck response into the engine's decks/*.json format.

Engine format (see adapter/lib/TalisharBoot.php::writeTalisharDeckFile and
decks/bravo.json):
    {"hero": <slug>, "equipment": [<slug>...], "deck": [<slug>... expanded]}
Line 1 of the 11-line file = hero + equipment (weapons live here too, NOT in
deck). Line 2 = the main deck cards.

Slug mapping: fabrary `cardIdentifier` is the Talishar identifier with dashes;
Talishar import does str_replace('-','_', ...) (talishar/APIs/AddFavoriteDeck.php).
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def to_slug(identifier: str | None) -> str:
    return identifier.replace("-", "_") if identifier else ""


def _kind(card: dict) -> str:
    types = [str(t).lower() for t in (card.get("types") or [])]
    if "weapon" in types:
        return "weapon"
    if "equipment" in types:
        return "equipment"
    return "deck"


def convert_deck(deck: dict) -> dict:
    """fabrary getDeck response -> engine deck dict.

    Keeps the FULL registered pool, not just a fixed 60. `equipment`/`deck` are
    the author's registered loadout (maindeck `quantity`); `sideboard` /
    `sideboard_equipment` are the extra registered copies (`sideboardQuantity`)
    that a future CC sideboard bot picks from after heroes are revealed.
    `registered_total` counts all registered cards (main + side) for the >=75
    pull filter; `maindeck_count` is the playable-deck size as registered.
    """
    hero = to_slug((deck.get("hero") or {}).get("cardIdentifier"))
    equipment: list[str] = []
    main: list[str] = []
    side: list[str] = []
    side_equip: list[str] = []
    registered = 0
    # Per-matchup maindeck quantity overrides: {matchupId: {slug: quantity}}.
    # Only cards the author flexed per matchup appear here (deltas).
    matchup_q: dict[str, dict[str, int]] = {}
    for dc in deck.get("deckCards") or []:
        qty = int(dc.get("quantity") or 0)
        sb = int(dc.get("sideboardQuantity") or 0)
        registered += qty + sb
        slug = to_slug(dc.get("cardIdentifier"))
        is_equip = _kind(dc.get("card") or {}) in ("weapon", "equipment")
        if is_equip:
            equipment.extend([slug] * qty)
            side_equip.extend([slug] * sb)
        else:
            main.extend([slug] * qty)
            side.extend([slug] * sb)
        for mq in dc.get("matchupQuantities") or []:
            mid = mq.get("matchupId")
            if mid is not None:
                matchup_q.setdefault(mid, {})[slug] = int(mq.get("quantity") or 0)
    # Matchup id -> opponent hero(s) + label, hero ids normalised to talishar slugs.
    matchups = [{"matchupId": m.get("matchupId"),
                 "heroes": [to_slug(h) for h in (m.get("heroIdentifiers") or [])],
                 "name": m.get("name")}
                for m in (deck.get("matchups") or [])]
    return {
        "hero": hero,
        "comment": (f"Imported from fabrary: {deck.get('name')!r} "
                    f"[{deck.get('format')}] deckId={deck.get('deckId')}"),
        "equipment": equipment,
        "deck": main,
        "sideboard": side,
        "sideboard_equipment": side_equip,
        "registered_total": registered,
        "maindeck_count": len(main),
        "matchups": matchups,
        "matchup_quantities": matchup_q,
    }


# ---------------------------------------------------------------------------
# Validation against Talishar's own card universe
# ---------------------------------------------------------------------------
_GEN_PHP = (Path(__file__).resolve().parents[3] / "talishar"
            / "GeneratedCode" / "GeneratedCardDictionaries.php")


def talishar_slugs() -> set[str]:
    """Every card id Talishar knows, parsed from the generated dictionaries
    (`"slug" => value,` match-table rows across all generated tables)."""
    if not _GEN_PHP.is_file():
        return set()
    txt = _GEN_PHP.read_text(encoding="utf-8", errors="ignore")
    return set(re.findall(r'"([a-z0-9_]+)"\s*=>', txt))


def validate(engine_deck: dict, known: set[str] | None = None) -> list[str]:
    """Return the list of slugs not found in Talishar's card universe."""
    known = known if known is not None else talishar_slugs()
    if not known:
        return []  # can't validate without the dictionary; treat as OK
    unknown = []
    for slug in [engine_deck["hero"], *engine_deck["equipment"], *engine_deck["deck"],
                 *engine_deck.get("sideboard", []), *engine_deck.get("sideboard_equipment", [])]:
        if slug and slug not in known:
            unknown.append(slug)
    return sorted(set(unknown))


def write_deck(engine_deck: dict, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(engine_deck, indent=2), encoding="utf-8")
    return out
