"""Phase-aware aggro bot — produces decisive games.

The default ``HeuristicBot`` plays the first ``PLAY_FROM_HAND`` it sees in
*any* phase, which means it also blocks with every card during the block
step. Two such bots fully block each other, no damage lands, and games
stall to the step cap (a draw). That gives a round-robin nothing to rank.

``AggroBot`` fixes that with a tiny phase policy:

* **M (main):** play an attack from hand (first non-PASS action).
* **P (pitch):** pitch to pay (first non-PASS action).
* **CHOOSE_* / DECISION / popups:** pick the first concrete option so
  card effects resolve instead of stalling.
* **everything else (A/D/B/INSTANT/ARS/...):** PASS — never block, never
  react. Attacks therefore connect and games end in lethal.

It still breaks tie-less situations randomly (seeded) so repeated games
between the same decks aren't identical. This is a *decisive* baseline,
not a strong one — exactly what the data-collection loop needs to start.
"""

from __future__ import annotations

import random
from typing import Any

from ..env import Action
from .base import Bot, BotDecision


# Phases where we actively want to do something other than pass.
_ACT_PHASES = {"M", "P"}
# Action types that represent resolving a forced/utility choice (targeting,
# modes, card selection). Picking one keeps the decision queue moving.
_CHOICE_PREFIXES = ("CHOOSE", "DECISION", "ARSENAL")


class AggroBot(Bot):
    name = "aggro"

    def __init__(self, *, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)

    def choose(
        self,
        state: dict[str, Any],
        legal_actions: list[Action],
        *,
        player_id: int,
    ) -> BotDecision:
        phase = str(state.get("phase", ""))
        non_pass = [a for a in legal_actions if a.type != "PASS"]
        pass_actions = [a for a in legal_actions if a.type == "PASS"]

        chosen: Action
        if phase in _ACT_PHASES and non_pass:
            # Prefer attacks/plays in the main phase; pitches in P.
            chosen = self._rng.choice(non_pass)
        elif any(a.type.startswith(_CHOICE_PREFIXES) for a in non_pass):
            # A forced choice popup (targeting, mode, pitch selection).
            choices = [a for a in non_pass if a.type.startswith(_CHOICE_PREFIXES)]
            chosen = self._rng.choice(choices)
        elif pass_actions:
            # Defence / reaction / arsenal / instant windows: take the hit.
            chosen = pass_actions[0]
        else:
            chosen = self._rng.choice(legal_actions)

        return BotDecision(action_id=chosen.action_id, info={"phase": phase, "policy": "aggro"})
