"""Aggressive heuristic — almost never passes when it can act.

Motivation: the recorded data showed the policy collapsing toward PASS. Even
after dropping forced single-action windows, the *exploration* policy used
during data collection (previously ``BalancedBot``) still passed in every
defence/reaction/instant window, so the dataset under-represented proactive
lines. ``AggressiveBot`` flips that default: whenever a non-PASS action
exists it takes one, strongly preferring offence (play attacks, activate
equipment) and resolving forced choices, so the IQL policy sees plenty of
"do something" trajectories to learn from.

It is *not* mindlessly suicidal: when it is the defender and the incoming
attack is lethal (read from the engine's combat cache via the ``combat``
field), it blocks instead of taking the killing blow. A tiny ``pass_prob``
keeps a sliver of "take the hit" diversity so two AggressiveBots don't
deadlock into perfectly mirrored blocking.
"""

from __future__ import annotations

import random
from typing import Any

from ..env import Action
from .base import Bot, BotDecision

# Offence first, then anything that resolves an effect/choice, then the rest.
_OFFENCE = ("PLAY_FROM_HAND", "ACTIVATE_HERO_OR_EQUIP")
_RESOLVE = ("CHOOSE", "DECISION", "ARSENAL")


class AggressiveBot(Bot):
    name = "aggressive"

    def __init__(self, *, seed: int = 0, pass_prob: float = 0.05,
                 hold_back_prob: float = 0.35) -> None:
        self._rng = random.Random(seed)
        # Small chance to pass even when actions exist — keeps mirror matches
        # from deadlocking and injects a little defensive variety.
        self.pass_prob = pass_prob
        # Arsenal coverage: the engine only opens the end-of-turn arsenal
        # window (turn[0]=="ARS") when the turn player still has cards in hand
        # (NetworkingLibraries.php::PassTurn). A pure dump-everything policy
        # therefore NEVER generates arsenal data — 0 arsenal states in 296
        # games (2026-06-10 audit) even though the mechanic verifiably works.
        # With this probability, when down to the last 1-2 hand cards in the
        # main phase, hold them (PASS) so the ARS window fires and the
        # ARSENAL_FROM_HAND / PLAY_FROM_ARSENAL lines enter the dataset.
        self.hold_back_prob = hold_back_prob

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    def _facing_lethal(self, state: dict[str, Any], me: int) -> bool:
        cb = state.get("combat") or {}
        if not cb.get("active"):
            return False
        attacker = int(cb.get("attacker", 0) or 0)
        if attacker in (0, me):       # I'm the attacker, not defending.
            return False
        pend = float(cb.get("pending_damage", 0) or 0)
        my = next((p for p in state.get("players", [])
                   if int(p.get("player_id", 0)) == me), {})
        life = float(my.get("health") or 0)
        return life > 0 and pend >= life

    def _by_tier(self, actions: list[Action], prefixes: tuple[str, ...]) -> list[Action]:
        return [a for a in actions if a.type.startswith(prefixes)]

    def _hand_size(self, state: dict[str, Any], me: int) -> int:
        p = next((p for p in state.get("players", [])
                  if int(p.get("player_id", 0)) == me), {})
        return len([c for c in (p.get("hand") or []) if c])

    def choose(self, state: dict[str, Any], legal_actions: list[Action],
               *, player_id: int) -> BotDecision:
        non_pass = [a for a in legal_actions if a.type != "PASS"]
        pass_actions = [a for a in legal_actions if a.type == "PASS"]

        if not non_pass:
            chosen = pass_actions[0] if pass_actions else legal_actions[0]
            return BotDecision(action_id=chosen.action_id,
                               info={"policy": "aggressive", "forced": True})

        # Never take a lethal hit when we could block.
        if self._facing_lethal(state, player_id):
            chosen = self._rng.choice(non_pass)
        # Hold the last 1-2 hand cards through our own main phase so the
        # end-of-turn arsenal window can fire (see __init__.hold_back_prob).
        # Safe: PASS in M with no pending payment just passes priority.
        elif (pass_actions
              and str(state.get("phase", "")) == "M"
              and int(state.get("active_player", 0) or 0) == player_id
              and 1 <= self._hand_size(state, player_id) <= 2
              and self._rng.random() < self.hold_back_prob):
            chosen = pass_actions[0]
        # A sliver of "take the hit" diversity.
        elif pass_actions and self._rng.random() < self.pass_prob:
            chosen = pass_actions[0]
        else:
            offence = self._by_tier(non_pass, _OFFENCE)
            resolve = self._by_tier(non_pass, _RESOLVE)
            pool = offence or resolve or non_pass
            chosen = self._rng.choice(pool)

        return BotDecision(action_id=chosen.action_id,
                           info={"policy": "aggressive", "top_type": chosen.type})
