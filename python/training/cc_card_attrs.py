"""Per-card attribute features for **Classic Constructed**, sourced from
``slug_index.json`` (the fabrary/cards native metadata) + the engine's
generated combat tables.

Why a separate module from :mod:`card_attrs`:

* ``card_attrs`` parses the **OMN cube**'s ``[CustomCards]`` block and uses a
  tiny schema (4 classes, 1 talent) — it only knows the ~200-card draft cube.
* CC spans the whole game: **19 classes, 12 talents, 14 types, 55 subtypes**
  across ~4,800 cards. The OMN table maps every CC card to UNK (0/60 coverage),
  so the CC gameplay model is *card-blind* — it can't even see a card's
  cost/power/type. This module gives every CC card a real attribute vector.
* Changing ``card_attrs.ATTR_DIM`` would break the resumable OMN/draft
  checkpoints (their ``attr_proj`` Linear is sized to the old dim), so the CC
  schema lives here with its own ``CC_ATTR_DIM``.

Source split (mirrors the OMN design): **categorical** metadata
(type/subtype/class/talent/colour/cost/pitch) from ``slug_index.json``;
**combat** stats (printed power/defense + keywords) from
:mod:`card_stats` (GeneratedCardDictionaries), which is already keyed by the
exact game slugs and is complete for the full pool.

Game slugs use ``_`` (e.g. ``codex_of_frailty_yellow``); slug_index keys use
``-`` (e.g. ``codex-of-frailty-yellow``). The two differ only by that swap —
verified 822/822 on the resolved CC decks — so mapping is a literal replace.

``build_cc_attr_matrix(vocab, attrs)`` returns a ``[len(vocab), CC_ATTR_DIM]``
float32 matrix aligned to a :class:`CardVocab`, ready to bake into a CC
checkpoint as the ``attr_table`` buffer (inference then needs no slug_index).
"""

from __future__ import annotations

import json
from pathlib import Path

from .features import CardVocab
from .card_stats import load_card_stats, KEYWORDS

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_SLUG_INDEX = _REPO / "slug_index.json"

# --- Attribute schema (order matters — it defines CC_ATTR_DIM) ---------------
_COLOURS = ("red", "yellow", "blue")

# Card TYPES (fabrary `types`). The full 14; rare ones kept so nothing collapses.
_TYPES = ("Action", "Instant", "Attack Reaction", "Defense Reaction", "Block",
          "Equipment", "Weapon", "Resource", "Hero", "Token", "Demi-Hero",
          "Mentor", "Macro", "Companion")

# Curated SUBTYPES that bear on play/legality: attack-ness, the aura/item/ally
# engine pieces, Mechanologist Evo, and every equipment slot + 1H/2H hands.
_SUBTYPES = ("Attack", "Non-Attack", "Aura", "Item", "Ally", "Evo", "Arrow",
             "Trap", "Invocation", "Base", "Head", "Chest", "Arms", "Legs",
             "Off-Hand", "Quiver", "1H", "2H")

# All meaningful CC CLASSES (multi-hot; a card may be multi-class). A final
# "other" slot catches the long tail (Adjudicator/Merchant/Shapeshifter/Thief/
# NotClassed) so those still differ from a known class.
_CLASSES = ("Generic", "Guardian", "Warrior", "Mechanologist", "Runeblade",
            "Ninja", "Brute", "Illusionist", "Assassin", "Wizard", "Ranger",
            "Pirate", "Necromancer", "Bard")

# All TALENTS (fabrary `talents`; multi-hot). These gate hero compatibility and
# many effects — a real signal the OMN schema almost entirely lacked.
_TALENTS = ("Mystic", "Revered", "Light", "Draconic", "Ice", "Elemental",
            "Earth", "Lightning", "Chaos", "Reviled", "Shadow", "Royal")

# Combat-stat tail (from card_stats): [power/7, hasPower, defense/4, hasDefense,
# <keyword multi-hot>].
_STAT_DIM = 2 + 2 + len(KEYWORDS)

# layout: [cost(2), pitch(2), colour(4), types, subtypes, classes+other,
#          talents, stats]
CC_ATTR_DIM = (2 + 2 + (len(_COLOURS) + 1)
               + len(_TYPES) + len(_SUBTYPES)
               + (len(_CLASSES) + 1) + len(_TALENTS) + _STAT_DIM)


def _colour_of(slug: str) -> str | None:
    for c in _COLOURS:
        if slug.endswith("_" + c):
            return c
    return None


class CCCardAttributes:
    """Game-slug -> attribute vector, sourced from slug_index + card_stats."""

    def __init__(self, by_slug: dict[str, dict], stats=None) -> None:
        # by_slug is keyed by DASH cardIdentifier (slug_index native).
        self._raw = by_slug
        self._stats = stats if stats is not None else load_card_stats()

    @classmethod
    def from_slug_index(cls, path: str | Path | None = None) -> "CCCardAttributes":
        p = Path(path) if path else _DEFAULT_SLUG_INDEX
        data = json.loads(p.read_text(encoding="utf-8"))
        by_slug = data.get("by_slug", data)  # tolerate either shape
        return cls(by_slug)

    def _entry(self, game_slug: str) -> dict | None:
        return self._raw.get(game_slug.replace("_", "-"))

    def vec(self, slug: str) -> list[float]:
        v = [0.0] * CC_ATTR_DIM
        e = self._entry(slug) or {}
        stat = self._stats.get(slug)  # card_stats is keyed by underscore game slug
        i = 0

        # cost (slug_index, fallback card_stats). The engine uses -1 as a
        # "no printed cost" sentinel (equipment/weapons/heroes; 974 cards) — a
        # printed cost of 0 is real, so the guard is >= 0, not > 0.
        cost = e.get("cost")
        if cost is None:
            cost = stat.get("cost")
        if isinstance(cost, (int, float)) and cost >= 0:
            v[i] = float(cost) / 6.0; v[i + 1] = 1.0
        i += 2

        # pitch (slug_index, fallback card_stats). FAB pitch is 1/2/3; 0 (and the
        # engine's sentinel) means the card has no pitch -> leave hasPitch off.
        pitch = e.get("pitch")
        if pitch is None:
            pitch = stat.get("pitch")
        if isinstance(pitch, (int, float)) and pitch >= 1:
            v[i] = float(pitch) / 3.0; v[i + 1] = 1.0
        i += 2

        # colour from slug suffix; fallback to pitch value (1=red/2=yel/3=blue)
        col = _colour_of(slug)
        if col is None and isinstance(pitch, (int, float)) and int(pitch) in (1, 2, 3):
            col = _COLOURS[int(pitch) - 1]
        for k, c in enumerate(_COLOURS):
            v[i + k] = 1.0 if col == c else 0.0
        v[i + len(_COLOURS)] = 1.0 if col is None else 0.0
        i += len(_COLOURS) + 1

        types = set(e.get("types") or [])
        for k, t in enumerate(_TYPES):
            v[i + k] = 1.0 if t in types else 0.0
        i += len(_TYPES)

        subs = set(e.get("subtypes") or [])
        for k, s in enumerate(_SUBTYPES):
            v[i + k] = 1.0 if s in subs else 0.0
        i += len(_SUBTYPES)

        classes = set(e.get("classes") or [])
        matched = False
        for k, cl in enumerate(_CLASSES):
            if cl in classes:
                v[i + k] = 1.0; matched = True
        # "other": a non-empty class set that none of the named slots captured
        v[i + len(_CLASSES)] = 1.0 if (classes and not matched) else 0.0
        i += len(_CLASSES) + 1

        talents = set(e.get("talents") or [])
        for k, tl in enumerate(_TALENTS):
            v[i + k] = 1.0 if tl in talents else 0.0
        i += len(_TALENTS)

        # combat stats (printed) from card_stats
        pw = stat.get("power")
        if pw is not None:
            v[i] = max(min(pw / 7.0, 1.5), -0.5); v[i + 1] = 1.0
        i += 2
        df = stat.get("defense")
        if df is not None:
            v[i] = max(min(df / 4.0, 1.5), -0.5); v[i + 1] = 1.0
        i += 2
        for k, kw in enumerate(KEYWORDS):
            v[i + k] = 1.0 if stat.get(kw) else 0.0

        return v

    def covers(self, slug: str) -> bool:
        """True if this card has real metadata (in slug_index OR card_stats)."""
        return self._entry(slug) is not None or bool(self._stats.get(slug))


def build_cc_attr_matrix(vocab: CardVocab, attrs: CCCardAttributes):
    """Return a numpy [len(vocab), CC_ATTR_DIM] float32 matrix aligned to vocab
    order (index 0=<pad>, 1=<unk> -> zero rows)."""
    import numpy as np
    rows = []
    for slug in vocab.itos:
        if slug in ("<pad>", "<unk>"):
            rows.append([0.0] * CC_ATTR_DIM)
        else:
            rows.append(attrs.vec(slug))
    return np.asarray(rows, dtype="float32")
