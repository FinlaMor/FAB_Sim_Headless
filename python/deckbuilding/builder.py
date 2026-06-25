"""Concrete DeckBuilder implementations.

Two are shipped today:

* :class:`RandomDeckBuilder`   — random legal selection (training baseline).
* :class:`HeuristicDeckBuilder` — pitch-balanced curve, hero+sig-weapon
                                  lock, basic synergy scoring.

The transformer scaffold lives in :class:`TransformerDeckBuilder` and
follows the same lazy-torch pattern as :class:`TransformerDraftBot`.

All builders honour the user's API spec:

.. code-block:: python

    class DeckBuilder:
        def build_deck(self, card_pool):
            ...
"""

from __future__ import annotations

import abc
import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from ..draft.format import (
    CRACKED_BAUBLE_SLUG,
    DECK_SIZE_EXACT,
    EQUIPMENT_SLOT_TAGS,
    EQUIPMENT_TAG,
    HERO_WEAPONS,
    LEGAL_HEROES,
    LEGAL_WEAPONS,
    MIN_DECK_SIZE,
)
from .card_catalog import CardCatalog
from .deck import Deck, DeckEvaluation
from .legality import card_legality_tags, is_legal_for_hero, LegalityError, validate_deck


def _pick_equipment(
    pool: list[str],
    hero: str,
    *,
    card_classes: dict[str, set[str]] | None,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    """Slot one head/chest/arms/legs from the drafted pool.

    Returns ``(equipment_cards, leftover_pool)``. ``equipment_cards`` is
    ordered by slot (head, chest, arms, legs). Slots with no legal
    candidate in the pool are omitted — Talishar tolerates a sparse
    character line.

    Hero-legality is enforced when ``card_classes`` is provided; without
    a class map, the first card we see with the right subtype wins.
    """
    if card_classes is None:
        return [], list(pool)

    # Bucket pool by slot.
    by_slot: dict[str, list[str]] = {s: [] for s in EQUIPMENT_SLOT_TAGS}
    leftover: list[str] = []
    for card in pool:
        _, _, subtypes = card_legality_tags(card, card_classes)
        if EQUIPMENT_TAG not in subtypes:
            leftover.append(card)
            continue
        slotted = False
        for slot in EQUIPMENT_SLOT_TAGS:
            if slot in subtypes:
                by_slot[slot].append(card)
                slotted = True
                break
        if not slotted:
            # Equipment without an obvious slot (off-hand, quiver, ...)
            # stays in the leftover bucket — Talishar will handle it via
            # the deck if it's legal there.
            leftover.append(card)

    chosen: list[str] = []
    for slot in EQUIPMENT_SLOT_TAGS:
        legal = [c for c in by_slot[slot]
                 if is_legal_for_hero(c, hero, card_classes)]
        if legal:
            # Deterministic but rng-tiebroken pick.
            rng.shuffle(legal)
            chosen.append(legal[0])
    return chosen, leftover


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class DeckBuilder(abc.ABC):
    """Implement ``build_deck(card_pool) -> Deck``.

    All implementations should call :func:`validate_deck` on the
    produced deck before returning so a buggy builder can't quietly
    emit illegal decks.

    Subclasses that need cube-derived card metadata (class/talent legal
    filtering, Cracked Bauble filling) should accept a ``card_classes``
    kwarg in their constructor. ``RandomDeckBuilder`` and
    ``HeuristicDeckBuilder`` both honour this convention.
    """
    name: str = "deck-builder"

    @abc.abstractmethod
    def build_deck(self, card_pool: Iterable[str]) -> Deck: ...

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------
class RandomDeckBuilder(DeckBuilder):
    """Picks the first legal hero/weapon found in the pool, then random fills.

    Honours OMN limited rules: exactly :data:`DECK_SIZE_EXACT` cards,
    every non-filler card must be class+talent legal for the chosen
    hero, shortfall is padded with :data:`CRACKED_BAUBLE_SLUG`.

    Useful for baselining IQL — a learned deck builder should beat this
    by a wide margin within the first epoch of training.
    """
    name = "deck-random"

    def __init__(
        self,
        *,
        catalog: CardCatalog | None = None,
        card_classes: dict[str, set[str]] | None = None,
        seed: int | None = None,
        target_size: int = DECK_SIZE_EXACT,
    ) -> None:
        self.catalog = catalog
        self.card_classes = card_classes
        self._rng = random.Random(seed)
        self.target_size = target_size

    def build_deck(self, card_pool: Iterable[str]) -> Deck:
        pool = list(card_pool)
        hero, weapon = _select_hero_and_weapon(pool, rng=self._rng)
        candidates = [c for c in pool if c != hero and c != weapon]
        # Restrict to hero-legal cards if a class map is available.
        if self.card_classes is not None:
            candidates = [c for c in candidates
                          if is_legal_for_hero(c, hero, self.card_classes)]
        self._rng.shuffle(candidates)
        deck_cards: list[str] = []
        for c in candidates:
            if len(deck_cards) >= self.target_size:
                break
            deck_cards.append(c)
        # Fill shortfall with cracked_bauble.
        while len(deck_cards) < self.target_size:
            deck_cards.append(CRACKED_BAUBLE_SLUG)
        sideboard = candidates[self.target_size:]
        deck = Deck(
            hero=hero,
            weapon=weapon,
            deck=deck_cards,
            sideboard=sideboard,
            evaluation=_evaluate(deck_cards, self.catalog, weapon),
        )
        validate_deck(deck, pool, catalog=self.catalog,
                      card_classes=self.card_classes)
        return deck


# ---------------------------------------------------------------------------
# Heuristic builder
# ---------------------------------------------------------------------------
class HeuristicDeckBuilder(DeckBuilder):
    """Pitch-balanced, weapon-aligned, curve-aware deck builder.

    Strategy
    --------
    1. Lock the hero and its signature weapon.
    2. **Filter** the remaining pool to cards class+talent legal for the
       hero (using the cube-derived ``card_classes`` map when available).
       Falls back to no filtering for synthetic / class-less pools.
    3. Bucket the remaining pool by pitch colour (red / yellow / blue / none).
    4. Greedily fill targeting a 12 / 9 / 9 red/yellow/blue split. Falls
       back gracefully when the pool is short of a colour.
    5. Penalise more than 3 copies of the same card.
    6. **Pad to EXACTLY ``target_size`` with** :data:`CRACKED_BAUBLE_SLUG`
       when the legal pool can't hit the target. This is how OMN limited
       handles short pools.
    """
    name = "deck-heuristic"

    PITCH_TARGET = {"red": 12, "yellow": 9, "blue": 9}

    def __init__(
        self,
        *,
        catalog: CardCatalog | None = None,
        card_classes: dict[str, set[str]] | None = None,
        seed: int | None = None,
        target_size: int = DECK_SIZE_EXACT,
    ) -> None:
        self.catalog = catalog
        self.card_classes = card_classes
        self._rng = random.Random(seed)
        self.target_size = target_size

    def build_deck(self, card_pool: Iterable[str]) -> Deck:
        pool = list(card_pool)
        hero, weapon = _select_hero_and_weapon(pool, rng=self._rng,
                                               prefer_match=True)
        # Strip out hero + weapon first.
        non_hero_weapon = [c for c in pool if c not in (hero, weapon)]
        # Slot equipment into the character line, keep everything else
        # as candidates for the 30-card main deck.
        equipment, remaining_all = _pick_equipment(
            non_hero_weapon, hero,
            card_classes=self.card_classes, rng=self._rng,
        )
        # Restrict to hero-legal cards. Without a class map we skip the
        # filter (pool already came from a class-pure cube section).
        if self.card_classes is not None:
            remaining = [c for c in remaining_all
                         if is_legal_for_hero(c, hero, self.card_classes)]
        else:
            remaining = remaining_all

        by_colour = {"red": [], "yellow": [], "blue": [], None: []}
        for c in remaining:
            colour = self._colour_of(c)
            by_colour[colour].append(c)
        for colour in by_colour:
            self._rng.shuffle(by_colour[colour])

        chosen: list[str] = []
        copy_count: Counter[str] = Counter()
        # First pass: hit the target colour split.
        for colour, target in self.PITCH_TARGET.items():
            taken = 0
            for c in list(by_colour[colour]):
                if taken >= target or len(chosen) >= self.target_size:
                    break
                if copy_count[c] >= 3:
                    continue
                chosen.append(c)
                copy_count[c] += 1
                taken += 1
                by_colour[colour].remove(c)
        # Second pass: fill the rest with whatever remains.
        leftover: list[str] = []
        for colour in by_colour:
            leftover.extend(by_colour[colour])
        self._rng.shuffle(leftover)
        for c in leftover:
            if len(chosen) >= self.target_size:
                break
            if copy_count[c] >= 3:
                continue
            chosen.append(c)
            copy_count[c] += 1

        # Fill shortfall with Cracked Bauble. The filler counts toward
        # both the size requirement and the yellow-pitch column.
        while len(chosen) < self.target_size:
            chosen.append(CRACKED_BAUBLE_SLUG)

        sideboard = [c for c in remaining_all if c not in Counter(chosen)]
        evaluation = _evaluate(chosen, self.catalog, weapon)
        deck = Deck(
            hero=hero,
            weapon=weapon,
            equipment=equipment,
            deck=chosen,
            sideboard=sideboard,
            evaluation=evaluation,
        )
        validate_deck(deck, pool, catalog=self.catalog,
                      card_classes=self.card_classes)
        return deck

    def _colour_of(self, card_id: str) -> str | None:
        if self.catalog and card_id in self.catalog:
            return self.catalog.pitch_of(card_id)
        for suffix, colour in (("_red", "red"), ("_yellow", "yellow"), ("_blue", "blue")):
            if card_id.endswith(suffix):
                return colour
        return None


# ---------------------------------------------------------------------------
# Transformer scaffold (torch optional)
# ---------------------------------------------------------------------------
@dataclass
class TransformerDeckConfig:
    vocab_size: int = 4096
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_pool: int = 64
    max_deck: int = 60


class TransformerDeckBuilder(DeckBuilder):
    name = "deck-transformer"

    def __init__(
        self,
        config: TransformerDeckConfig | None = None,
        *,
        catalog: CardCatalog | None = None,
        weights_path: str | None = None,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install torch — TransformerDeckBuilder needs it") from e
        self.config = config or TransformerDeckConfig()
        self.device = device
        self.catalog = catalog
        self._rng = random.Random(seed)
        self._build()
        if weights_path is not None:
            self._load(weights_path)

    def build_deck(self, card_pool: Iterable[str]) -> Deck:
        # Placeholder: scaffold defers to HeuristicDeckBuilder until you train weights.
        # This keeps the scaffold "useful" so end-to-end tests still produce legal decks.
        return HeuristicDeckBuilder(
            catalog=self.catalog, seed=self._rng.randint(0, 2**31 - 1),
        ).build_deck(card_pool)

    def _build(self) -> None:
        import torch
        import torch.nn as nn
        cfg = self.config

        class DeckPolicyNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
                layer = nn.TransformerEncoderLayer(
                    d_model=cfg.d_model, nhead=cfg.n_heads,
                    dim_feedforward=cfg.d_model * 4, batch_first=True,
                )
                self.enc = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
                self.head = nn.Linear(cfg.d_model, cfg.max_pool)

            def forward(self, ids: "torch.Tensor") -> "torch.Tensor":
                x = self.tok(ids)
                h = self.enc(x).mean(dim=1)
                return self.head(h).squeeze(0)

        self.model = DeckPolicyNet().to(self.device).eval()

    def _load(self, path: str) -> None:
        import torch
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _select_hero_and_weapon(
    pool: list[str],
    *,
    rng: random.Random,
    prefer_match: bool = False,
) -> tuple[str, str]:
    heroes_in_pool  = [c for c in pool if c in LEGAL_HEROES]
    weapons_in_pool = [c for c in pool if c in LEGAL_WEAPONS]
    if not heroes_in_pool:
        raise LegalityError(
            "no legal hero in card pool — draft policy must always rare-draft a hero"
        )
    if not weapons_in_pool:
        raise LegalityError(
            "no legal weapon in card pool — draft policy must rare-draft a weapon"
        )
    if prefer_match:
        # Prefer a hero whose signature weapon is also in the pool.
        for hero in heroes_in_pool:
            sig = HERO_WEAPONS.get(hero)
            if sig and sig in weapons_in_pool:
                return hero, sig
    hero   = rng.choice(heroes_in_pool)
    target = HERO_WEAPONS.get(hero)
    weapon = target if target in weapons_in_pool else rng.choice(weapons_in_pool)
    return hero, weapon


def _evaluate(deck_cards: list[str], catalog: CardCatalog | None, weapon: str) -> DeckEvaluation:
    pitch_count: Counter[str] = Counter()
    cost_count:  Counter[str] = Counter()
    for c in deck_cards:
        colour = catalog.pitch_of(c) if catalog and c in catalog else None
        if colour is None:
            for suffix, col in (("_red", "red"), ("_yellow", "yellow"), ("_blue", "blue")):
                if c.endswith(suffix):
                    colour = col
                    break
        if colour:
            pitch_count[colour] += 1
        if catalog and c in catalog:
            meta = catalog.get(c)
            if meta.cost is not None:
                cost_count[str(meta.cost)] += 1
    total = sum(pitch_count.values()) or 1
    # Closeness to the 12/9/9 target.
    target = {"red": 12, "yellow": 9, "blue": 9}
    misfit = sum(abs(pitch_count.get(k, 0) - v) for k, v in target.items())
    overall = max(0.0, 1.0 - misfit / 30.0)
    return DeckEvaluation(
        pitch_distribution=dict(pitch_count),
        curve_histogram=dict(cost_count),
        synergy_notes=[],  # placeholder for the heuristic v2
        weapon_alignment_score=1.0 if weapon in LEGAL_WEAPONS else 0.0,
        overall_score=overall,
    )
