"""Balanced bot — like AggroBot, but it sometimes blocks.

AggroBot never blocks, so a dataset collected from it contains almost no
defensive lines and the value function never learns what blocking is
worth. BalancedBot keeps the decisive-aggression behaviour (attack in M,
pitch in P, resolve forced choices) but in the block/defence phases it
blocks with probability ``block_prob`` (otherwise it takes the hit). That
injects defensive trajectories into the data (item 2: behavioural
diversity) while keeping games decisive enough to terminate.
"""

from __future__ import annotations

import random
from typing import Any

from ..env import Action
from .base import Bot, BotDecision

_ACT_PHASES = {"M", "P"}
_DEFENCE_PHASES = {"B", "D"}
_CHOICE_PREFIXES = ("CHOOSE", "DECISION", "ARSENAL")


class BalancedBot(Bot):
    name = "balanced"

    def __init__(self, *, seed: int = 0, block_prob: float = 0.5) -> None:
        self._rng = random.Random(seed)
        self.block_prob = block_prob

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)

    def choose(self, state: dict[str, Any], legal_actions: list[Action], *, player_id: int) -> BotDecision:
        phase = str(state.get("phase", ""))
        non_pass = [a for a in legal_actions if a.type != "PASS"]
        pass_actions = [a for a in legal_actions if a.type == "PASS"]

        if phase in _ACT_PHASES and non_pass:
            chosen = self._rng.choice(non_pass)
        elif phase in _DEFENCE_PHASES and non_pass and pass_actions:
            # Block with probability block_prob, else take the hit.
            chosen = self._rng.choice(non_pass) if self._rng.random() < self.block_prob else pass_actions[0]
        elif phase in _DEFENCE_PHASES and non_pass:
            chosen = self._rng.choice(non_pass)
        elif any(a.type.startswith(_CHOICE_PREFIXES) for a in non_pass):
            choices = [a for a in non_pass if a.type.startswith(_CHOICE_PREFIXES)]
            chosen = self._rng.choice(choices)
        elif pass_actions:
            chosen = pass_actions[0]
        else:
            chosen = self._rng.choice(legal_actions)

        return BotDecision(action_id=chosen.action_id, info={"phase": phase, "policy": "balanced"})
