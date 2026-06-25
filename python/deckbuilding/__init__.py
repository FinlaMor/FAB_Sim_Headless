"""Pool-to-deck construction for OMA limited.

Public surface
--------------
* :class:`Deck`              — final decklist (hero, weapon, deck, sideboard).
* :class:`DeckBuilder`       — ABC; ``build_deck(card_pool) -> Deck``.
* :class:`HeuristicDeckBuilder` — pitch-balanced curve, hero+sig-weapon lock.
* :class:`RandomDeckBuilder`    — random legal selection (training baseline).
* :class:`CardCatalog`       — looks up card metadata from
                               talishar/CardDictionaries (or a JSON cache).
* :mod:`python.deckbuilding.legality` — deck-rule checks (min size, hero
                                         lock, weapon match, set legality).

Talishar is still the *ultimate* judge of legality: whatever the
deckbuilder emits is fed to ``TalisharBoot::createGame`` and any illegal
play surfaces as a ``ProcessInput`` rejection. The client-side legality
module is a fast pre-filter so we don't waste a /step round-trip on
deck construction bugs.
"""

from .card_catalog import CardCatalog, load_card_catalog
from .deck import Deck, DeckEvaluation
from .legality import LegalityError, validate_deck
from .builder import DeckBuilder, RandomDeckBuilder, HeuristicDeckBuilder

__all__ = [
    "Deck", "DeckEvaluation",
    "CardCatalog", "load_card_catalog",
    "LegalityError", "validate_deck",
    "DeckBuilder", "RandomDeckBuilder", "HeuristicDeckBuilder",
]
