"""Explicit per-card attribute features.

Cards are otherwise pure learned id-embeddings, which the models can't
learn well from a few hundred drafts / 100k transitions. This table gives
every card a small fixed attribute vector (cost, pitch colour, subtypes,
class, talent) parsed from the cube's ``[CustomCards]`` metadata, so the
nets can generalise across cards that share attributes instead of
memorising ids.

``build_attr_matrix(vocab)`` returns a ``[len(vocab), ATTR_DIM]`` float
matrix aligned to a :class:`CardVocab`; it's baked into checkpoints so
inference needs no cube file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

from .features import CardVocab
from .card_stats import load_card_stats, KEYWORDS

# Attribute schema (order matters — it defines ATTR_DIM).
_COLOURS = ("red", "yellow", "blue")
_SUBTYPES = ("attack", "action", "instant", "aura", "equipment",
             "item", "ally", "weapon", "defense reaction", "attack reaction")
_CLASSES = ("illusionist", "wizard", "runeblade", "generic")
_TALENTS = ("lightning",)

# Engine-derived combat stats (the big missing signal): the model can now see
# how hard a card hits / blocks and its combat keywords, not just its cost.
# layout tail: [power/7, hasPower, defense/4, hasDefense, <keyword multi-hot>]
_STAT_DIM = 2 + 2 + len(KEYWORDS)

# layout: [cost, hasCost, red, yellow, blue, noColour, <subtypes>, <classes>,
#          otherClass, <talents>, <stats>]
ATTR_DIM = (1 + 1 + (len(_COLOURS) + 1) + len(_SUBTYPES)
            + (len(_CLASSES) + 1) + len(_TALENTS) + _STAT_DIM)


def _colour_of(slug: str) -> str | None:
    for c in _COLOURS:
        if slug.endswith("_" + c):
            return c
    return None


class CardAttributes:
    """slug -> attribute vector, parsed from the cube CustomCards."""

    def __init__(self, by_slug: dict[str, dict], stats=None) -> None:
        self._raw = by_slug  # slug -> {"cost": str, "type": str, "name": str}
        # Base/printed combat stats from the engine's generated tables. These
        # are STATIC card-identity values (printed power/defense/keywords), the
        # correct thing for a per-card table; in-game buffs are reflected in the
        # game STATE (the combat cache), not here.
        self._stats = stats if stats is not None else load_card_stats()

    @classmethod
    def from_cube(cls, cube_path: str | Path) -> "CardAttributes":
        from .features import loads  # noqa: F401 (kept local to avoid cycle confusion)
        txt = Path(cube_path).read_text(encoding="utf-8")
        m = re.search(r"\[CustomCards\]\s*(\[.*?\])\s*\[", txt, re.S)
        by_slug: dict[str, dict] = {}
        if m:
            from .features import CardVocab  # noqa: F401
            from python.draft.draftmancer import slugify
            for c in json.loads(m.group(1)):
                by_slug[slugify(c["name"])] = {
                    "cost": c.get("mana_cost", ""),
                    "type": (c.get("type", "") or "").lower(),
                    "name": c.get("name", ""),
                }
        return cls(by_slug)

    def vec(self, slug: str) -> list[float]:
        v = [0.0] * ATTR_DIM
        info = self._raw.get(slug)
        typ = (info or {}).get("type", "")
        # cost
        cost = (info or {}).get("cost", "")
        i = 0
        try:
            v[i] = float(cost) / 6.0; v[i + 1] = 1.0
        except (TypeError, ValueError):
            v[i] = 0.0; v[i + 1] = 0.0
        i += 2
        # colour (red/yellow/blue/none)
        col = _colour_of(slug)
        for k, c in enumerate(_COLOURS):
            v[i + k] = 1.0 if col == c else 0.0
        v[i + len(_COLOURS)] = 1.0 if col is None else 0.0
        i += len(_COLOURS) + 1
        # subtypes (substring match in the type string)
        for k, st in enumerate(_SUBTYPES):
            v[i + k] = 1.0 if st in typ else 0.0
        i += len(_SUBTYPES)
        # class
        matched = False
        for k, cl in enumerate(_CLASSES):
            if cl in typ:
                v[i + k] = 1.0; matched = True
        v[i + len(_CLASSES)] = 0.0 if matched else 1.0  # "other/unknown class"
        i += len(_CLASSES) + 1
        # talent
        for k, tl in enumerate(_TALENTS):
            v[i + k] = 1.0 if tl in typ else 0.0
        i += len(_TALENTS)
        # engine combat stats (base/printed): power, defense, keyword multi-hot
        st = self._stats.get(slug)
        pw = st.get("power")
        if pw is not None:
            v[i] = max(min(pw / 7.0, 1.5), -0.5); v[i + 1] = 1.0
        i += 2
        df = st.get("defense")
        if df is not None:
            v[i] = max(min(df / 4.0, 1.5), -0.5); v[i + 1] = 1.0
        i += 2
        for k, kw in enumerate(KEYWORDS):
            v[i + k] = 1.0 if st.get(kw) else 0.0
        return v


def build_attr_matrix(vocab: CardVocab, attrs: CardAttributes):
    """Return a numpy [len(vocab), ATTR_DIM] matrix aligned to vocab order."""
    import numpy as np
    rows = []
    for slug in vocab.itos:  # index 0=<pad>, 1=<unk>, then cards
        if slug in ("<pad>", "<unk>"):
            rows.append([0.0] * ATTR_DIM)
        else:
            rows.append(attrs.vec(slug))
    return np.asarray(rows, dtype="float32")
