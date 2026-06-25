"""Rule-of-thumb scaffold bot.

Implements a small ordered preference over action types, then breaks
ties randomly. Override ``_score`` to plug in stronger heuristics
without touching the orchestrator.

The default ranking, lowest-cost-first:

    PLAY_FROM_HAND > ACTIVATE_HERO_OR_EQUIP > DECISION/CHOOSE_* > PASS

This is intentionally shallow — real FAB play requires deck/archetype-
aware heuristics. The point of the scaffold is to demonstrate where
those plug in.
"""

from __future__ import annotations

import random
from typing import Any, Callable

from ..env import Action
from .base import Bot, BotDecision


_DEFAULT_PRIORITY: list[str] = [
    "PLAY_FROM_HAND",
    "ACTIVATE_HERO_OR_EQUIP",
    "DECISION",
    "PASS",
]


class HeuristicBot(Bot):
    name = "heuristic"

    def __init__(
        self,
        *,
        priority: list[str] | None = None,
        scorer: Callable[[dict[str, Any], Action, int], float] | None = None,
        seed: int | None = None,
    ) -> None:
        self.priority = priority or _DEFAULT_PRIORITY
        self._rng = random.Random(seed)
        self._scorer = scorer

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
            raise RuntimeError("HeuristicBot received zero legal actions — terminal state?")

        scores: list[tuple[float, Action]] = []
        for a in legal_actions:
            score = self._score(state, a, player_id)
            # Tiny jitter to break ties stochastically without seeding bias.
            score += self._rng.random() * 1e-6
            scores.append((score, a))
        scores.sort(key=lambda kv: kv[0], reverse=True)
        chosen = scores[0][1]

        return BotDecision(
            action_id=chosen.action_id,
            info={
                "policy": "heuristic",
                "score": scores[0][0],
                "n_legal": len(legal_actions),
                "top_type": chosen.type,
            },
        )

    # ------------------------------------------------------------------
    # Extension points
    # ------------------------------------------------------------------
    def _score(self, state: dict[str, Any], action: Action, player_id: int) -> float:
        if self._scorer is not None:
            return float(self._scorer(state, action, player_id))

        # Type priority (higher = more preferred)
        try:
            tier = (len(self.priority) - self.priority.index(action.type))
        except ValueError:
            # Unknown type — sandwich it just above PASS.
            tier = 1
        # Soft preference for actions targeting opponent and away from PASS.
        me = next((p for p in state.get("players", []) if int(p.get("player_id", 0)) == player_id), {})
        opp = next((p for p in state.get("players", []) if int(p.get("player_id", 0)) and int(p["player_id"]) != player_id), {})
        opp_low_hp_bonus = 0.0
        if opp.get("health") is not None and opp["health"] <= 10 and action.type != "PASS":
            opp_low_hp_bonus = 0.5
        return float(tier) + opp_low_hp_bonus
