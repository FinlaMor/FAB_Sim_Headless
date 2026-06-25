"""Tournament participant — deck + gameplay bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..deckbuilding.deck import Deck
from ..gameplay.bots.base import Bot


@dataclass
class Player:
    """One participant in the tournament bracket.

    Parameters
    ----------
    seat:
        0..7 in pod order. Maps directly to bracket label A..H.
    label:
        Bracket label (A..H). Computed from seat by default but
        overridable for custom pairings.
    deck:
        The deck the player drafted + built.
    bot_factory:
        Returns a fresh :class:`Bot` per game. We instantiate per-game so
        bots that hold internal seed state (RandomBot RNG, transformer
        caches) don't leak between games.
    name:
        Free-form identifier carried into the dataset for joinability
        with draft picks / analytics.
    """
    seat: int
    deck: Deck
    bot_factory: Callable[[int], Bot]  # (game_seed) -> Bot
    label: str = ""
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = chr(ord("A") + self.seat)
        if not self.name:
            self.name = f"seat{self.seat}_{self.deck.hero}"
