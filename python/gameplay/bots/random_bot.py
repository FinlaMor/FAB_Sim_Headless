"""Uniform-random baseline.

This is the canonical sanity-check bot. If self-play produces sensible
trajectories with two RandomBots the rest of the pipeline is wired
correctly. It's also the right thing to load when bootstrapping IQL
from scratch (the offline-RL paper "lets random data alone train the
critic; bootstrap from there").
"""

from __future__ import annotations

import random
from typing import Any

from ..env import Action
from .base import Bot, BotDecision


class RandomBot(Bot):
    name = "random"

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng.seed(seed)

    def choose(
        self,
        state: dict[str, Any],
        legal_actions: list[Action],
        *,
        player_id: int,
    ) -> BotDecision:
        if not legal_actions:
            raise RuntimeError("RandomBot received zero legal actions — terminal state?")
        a = self._rng.choice(legal_actions)
        return BotDecision(
            action_id=a.action_id,
            info={"policy": "uniform", "n_legal": len(legal_actions)},
        )
