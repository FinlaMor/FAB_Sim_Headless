"""Uniform-random draft baseline."""

from __future__ import annotations

import random
from typing import Any

from .base import DraftBot, DraftDecision, DraftPodView


class RandomDraftBot(DraftBot):
    name = "draft-random"

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng.seed(seed)

    def choose_card(
        self,
        pack: tuple[str, ...],
        drafted_cards: tuple[str, ...],
        seat_position: int,
        pick_number: int,
        pack_number: int,
        pod_state: DraftPodView,
    ) -> str | DraftDecision:
        if not pack:
            raise RuntimeError("RandomDraftBot received empty pack")
        choice = self._rng.choice(pack)
        return DraftDecision(card_id=choice, info={"policy": "uniform", "pack_size": len(pack)})

    def pick_hero(
        self,
        drafted_cards: tuple[str, ...],
        available_heroes: tuple[str, ...],
        card_classes: dict[str, set[str]],
    ) -> str | None:
        if not available_heroes:
            return None
        return self._rng.choice(list(available_heroes))
