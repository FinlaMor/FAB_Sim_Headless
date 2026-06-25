"""Authoritative per-card stats parsed from the engine's generated tables.

Talishar ships ``GeneratedCode/GeneratedCardDictionaries.php`` — big
``match($cardID) { "slug" => value, ... }`` tables keyed by the exact card
slugs our game state uses. We parse the ones that matter for play decisions
(power, defense/block, cost, pitch, and a focused set of combat keywords) so
the models can finally *see* how strong a card is, not just its cost/colour.

Pure text parse — no engine boot, no Docker — so it runs anywhere the repo
is checked out. Cached per process.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Default generated-dictionary location inside the bundled engine clone.
_DEFAULT_GEN = (Path(__file__).resolve().parents[2]
                / "talishar" / "GeneratedCode" / "GeneratedCardDictionaries.php")

# Numeric stat tables: function name -> output key.
_NUM_FUNCS = {
    "GeneratedPowerValue": "power",
    "GeneratedBlockValue": "defense",
    "GeneratedCardCost":   "cost",
    "GeneratedPitchValue": "pitch",
}

# Boolean keyword tables that bear on combat / aggression decisions.
_KW_FUNCS = {
    "GeneratedGoAgain":          "go_again",
    "GeneratedHasDominate":      "dominate",
    "GeneratedHasAmbush":        "ambush",
    "GeneratedHasCombo":         "combo",
    "GeneratedHasCrush":         "crush",
    "GeneratedHasBoost":         "boost",
    "GeneratedHasArcaneBarrier": "arcane_barrier",
    "GeneratedHasBattleworn":    "battleworn",
}
KEYWORDS: tuple[str, ...] = tuple(_KW_FUNCS.values())

_ARM = re.compile(r'"([a-z0-9_]+)"\s*=>\s*([^,\n]+?),')


def _func_body(text: str, name: str) -> str:
    start = text.find(f"function {name}(")
    if start < 0:
        return ""
    nxt = text.find("\nfunction ", start + 1)
    return text[start: nxt if nxt > 0 else len(text)]


def _parse_num(text: str, name: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for slug, val in _ARM.findall(_func_body(text, name)):
        val = val.strip()
        if re.fullmatch(r"-?\d+", val):
            out[slug] = int(val)
    return out


def _parse_bool(text: str, name: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for slug, val in _ARM.findall(_func_body(text, name)):
        v = val.strip().lower()
        if v in ("true", "false"):
            out[slug] = (v == "true")
    return out


class CardStats:
    """slug -> {power, defense, cost, pitch, <keyword bools>}."""

    def __init__(self, by_slug: dict[str, dict[str, Any]]) -> None:
        self._by_slug = by_slug

    @classmethod
    def load(cls, gen_path: str | Path | None = None) -> "CardStats":
        p = Path(gen_path) if gen_path else _DEFAULT_GEN
        by_slug: dict[str, dict[str, Any]] = {}
        if not p.is_file():
            return cls(by_slug)
        text = p.read_text(encoding="utf-8", errors="ignore")
        for fn, key in _NUM_FUNCS.items():
            for slug, v in _parse_num(text, fn).items():
                by_slug.setdefault(slug, {})[key] = v
        for fn, key in _KW_FUNCS.items():
            for slug, v in _parse_bool(text, fn).items():
                by_slug.setdefault(slug, {})[key] = v
        return cls(by_slug)

    def get(self, slug: str) -> dict[str, Any]:
        return self._by_slug.get(slug, {})

    def __len__(self) -> int:
        return len(self._by_slug)


_CACHE: CardStats | None = None


def load_card_stats(gen_path: str | Path | None = None) -> CardStats:
    global _CACHE
    if _CACHE is None or gen_path is not None:
        _CACHE = CardStats.load(gen_path)
    return _CACHE
