"""Final-deck dataclass used by tournament + dataset writer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class DeckEvaluation:
    """Builder-emitted heuristics about the deck.

    These are NOT authoritative ratings — they are the deck builder's
    internal scoring rubric, persisted so analytics can correlate
    builder rationale with tournament performance.
    """
    pitch_distribution: dict[str, int] = field(default_factory=dict)
    curve_histogram:    dict[str, int] = field(default_factory=dict)
    synergy_notes:      list[str]      = field(default_factory=list)
    weapon_alignment_score: float      = 0.0
    overall_score:      float          = 0.0


@dataclass
class Deck:
    """A complete limited deck.

    Notes
    -----
    The ``deck`` list is the main deck (60-card maximum for Living Legend
    limited; 30-card minimum for sealed/draft).  ``sideboard`` is the
    leftover card pool that may be swapped in via a `/sideboard`
    mechanism (not modelled in this MVP — sideboard is informational).

    Talishar's deck file format is line-delimited
    ``hero\nequipment1\nequipment2\n\ndeck_card1\n...`` — the
    ``to_talishar_dict`` helper produces a payload acceptable to
    ``TalisharBoot::writeDeckFile``.
    """
    hero:      str
    weapon:    str
    deck:      list[str]
    sideboard: list[str] = field(default_factory=list)
    equipment: list[str] = field(default_factory=list)
    evaluation: DeckEvaluation = field(default_factory=DeckEvaluation)

    @property
    def size(self) -> int:
        return len(self.deck)

    def to_talishar_dict(self) -> dict[str, Any]:
        """Shape expected by ``adapter/lib/TalisharBoot::writeDeckFile``."""
        return {
            "hero":      self.hero,
            "equipment": [self.weapon, *self.equipment],
            "deck":      list(self.deck),
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def save_json(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_talishar_dict(), indent=2), encoding="utf-8")
