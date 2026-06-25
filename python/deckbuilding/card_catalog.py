"""Card metadata catalog.

The deck builder needs three pieces of information per card:

1. ``card_type`` (hero / weapon / equipment / action / aura / item / token)
2. ``pitch`` ("red" | "yellow" | "blue" | None)
3. ``cost``  (integer or None)

Talishar's ``CardDictionaries/`` holds the authoritative metadata, but
each set is a PHP file. We parse those lazily via a thin scanner; the
parsed catalogue is cached as JSON under ``decks/_cache/`` so subsequent
runs are O(1).

If the Talishar clone is not on disk, the catalog falls back to a small
JSON shim shipped at ``decks/sample_packs/oma_card_catalog.json``. The
shim only knows about the heroes / weapons / colours embedded in the
sample packs file — enough for the smoke test to run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..draft.format import LEGAL_HEROES, LEGAL_WEAPONS


@dataclass(frozen=True)
class CardMeta:
    card_id: str
    card_type: str  # "hero" | "weapon" | "equipment" | "action" | "aura" | "item" | "token"
    pitch: str | None  # "red" | "yellow" | "blue" | None
    cost: int | None = None
    name: str | None = None


@dataclass
class CardCatalog:
    """Maps card_id -> CardMeta.

    Missing IDs raise ``KeyError`` so deckbuilding bugs surface loudly
    instead of silently treating an unknown card as a neutral 0-cost
    action.
    """
    by_id: dict[str, CardMeta] = field(default_factory=dict)

    def __contains__(self, card_id: str) -> bool:
        return card_id in self.by_id

    def get(self, card_id: str) -> CardMeta:
        if card_id not in self.by_id:
            raise KeyError(card_id)
        return self.by_id[card_id]

    def pitch_of(self, card_id: str) -> str | None:
        meta = self.by_id.get(card_id)
        return meta.pitch if meta else None

    def is_hero(self, card_id: str) -> bool:
        meta = self.by_id.get(card_id)
        return meta is not None and meta.card_type == "hero"

    def is_weapon(self, card_id: str) -> bool:
        meta = self.by_id.get(card_id)
        return meta is not None and meta.card_type == "weapon"

    def is_equipment(self, card_id: str) -> bool:
        meta = self.by_id.get(card_id)
        return meta is not None and meta.card_type == "equipment"

    @classmethod
    def from_iterable(cls, metas: Iterable[CardMeta]) -> "CardCatalog":
        return cls(by_id={m.card_id: m for m in metas})


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_card_catalog(
    *,
    talishar_root: str | Path | None = None,
    fallback_json: str | Path | None = None,
    pack_universe: Iterable[str] | None = None,
) -> CardCatalog:
    """Resolve a card catalog using whichever source is available.

    Resolution order:

    1. ``talishar_root``/CardDictionaries/*.php  (real metadata; recommended)
    2. ``fallback_json``                         (cached/shimmed catalog)
    3. Inferred from ``pack_universe`` + ``LEGAL_HEROES``/``LEGAL_WEAPONS``
       (zero-knowledge fallback — pitch inferred from the ``_red/_yellow/_blue``
       suffix convention; type set to ``"action"``).

    The first source that yields a non-empty catalog wins.
    """
    if talishar_root:
        try:
            cat = _parse_talishar_dictionaries(Path(talishar_root))
            if cat.by_id:
                return cat
        except FileNotFoundError:
            pass

    if fallback_json and Path(fallback_json).is_file():
        return _load_json(Path(fallback_json))

    return _infer_from_universe(pack_universe or ())


def _parse_talishar_dictionaries(root: Path) -> CardCatalog:
    """Lightweight regex scan over Talishar PHP dictionaries.

    We do NOT try to interpret PHP — we just extract ``"card_id" => [..]``
    rows and pull out the (cost, pitch, type) fields. Anything we can't
    parse is omitted; the builder treats missing entries as unknown.
    """
    dict_dir = root / "CardDictionaries"
    if not dict_dir.is_dir():
        raise FileNotFoundError(dict_dir)

    metas: list[CardMeta] = []
    row_re = re.compile(
        r'"(?P<id>[a-z0-9_]+)"\s*=>\s*\[(?P<body>.*?)\],',
        re.DOTALL,
    )
    for php in sorted(dict_dir.glob("*.php")):
        try:
            text = php.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in row_re.finditer(text):
            card_id = m.group("id")
            body = m.group("body")
            pitch = _scan_pitch(body, card_id)
            cost  = _scan_cost(body)
            ctype = _scan_card_type(body, card_id)
            metas.append(CardMeta(card_id=card_id, card_type=ctype, pitch=pitch, cost=cost))
    return CardCatalog.from_iterable(metas)


_PITCH_RE = re.compile(r'"?(pitch|p)"?\s*=>\s*(?P<v>[0-3])')
_COST_RE  = re.compile(r'"?cost"?\s*=>\s*(?P<v>\d+)')
_TYPE_RE  = re.compile(r'"?(type|t)"?\s*=>\s*"(?P<v>[A-Za-z]+)"')


def _scan_pitch(body: str, card_id: str) -> str | None:
    m = _PITCH_RE.search(body)
    if m:
        return {"1": "red", "2": "yellow", "3": "blue"}.get(m.group("v"))
    # Fall back to the slug suffix convention.
    for suffix, colour in (("_red", "red"), ("_yellow", "yellow"), ("_blue", "blue")):
        if card_id.endswith(suffix):
            return colour
    return None


def _scan_cost(body: str) -> int | None:
    m = _COST_RE.search(body)
    return int(m.group("v")) if m else None


def _scan_card_type(body: str, card_id: str) -> str:
    m = _TYPE_RE.search(body)
    if m:
        v = m.group("v").lower()
        if v.startswith("c"): return "hero"
        if v.startswith("w"): return "weapon"
        if v.startswith("e"): return "equipment"
        if v.startswith("au"): return "aura"
        if v.startswith("a"): return "action"
        if v.startswith("i"): return "item"
        if v.startswith("t"): return "token"
    if card_id in LEGAL_HEROES:    return "hero"
    if card_id in LEGAL_WEAPONS:   return "weapon"
    if card_id.endswith("_equip"): return "equipment"
    return "action"


def _load_json(path: Path) -> CardCatalog:
    rows = json.loads(path.read_text(encoding="utf-8"))
    metas = [
        CardMeta(
            card_id=str(r["card_id"]),
            card_type=str(r.get("card_type", "action")),
            pitch=r.get("pitch"),
            cost=r.get("cost"),
            name=r.get("name"),
        )
        for r in rows
    ]
    return CardCatalog.from_iterable(metas)


def _infer_from_universe(universe: Iterable[str]) -> CardCatalog:
    metas: list[CardMeta] = []
    for cid in universe:
        if cid in LEGAL_HEROES:
            metas.append(CardMeta(cid, "hero", None))
        elif cid in LEGAL_WEAPONS:
            metas.append(CardMeta(cid, "weapon", None))
        else:
            pitch = None
            for suffix, colour in (("_red", "red"), ("_yellow", "yellow"), ("_blue", "blue")):
                if cid.endswith(suffix):
                    pitch = colour
                    break
            metas.append(CardMeta(cid, "action", pitch))
    return CardCatalog.from_iterable(metas)
