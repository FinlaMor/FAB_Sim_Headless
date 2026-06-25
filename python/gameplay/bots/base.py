"""Bot abstract base class.

Bots are pure functions of (state, legal_actions) -> chosen action_id +
optional metadata. They MUST NOT mutate the environment state or call
``env.step`` themselves; the self-play orchestrator owns the step loop.

Bots may keep internal state (e.g. RNG seeds, neural nets, action
histograms). The orchestrator calls ``reset()`` between games.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from ..env import Action


@dataclass
class BotDecision:
    """Bot output: which action_id to play, plus optional debug metadata.

    ``info`` lets advanced bots emit per-step diagnostics (Q-values,
    attention weights, MCTS visit counts) that the replay buffer can
    persist alongside the transition.
    """
    action_id: int
    info: dict[str, Any] = field(default_factory=dict)


class Bot(abc.ABC):
    """Override ``choose`` to implement a new bot."""

    name: str = "bot"

    def reset(self, *, seed: int | None = None) -> None:
        """Called once at the start of every game. Default: no-op."""

    @abc.abstractmethod
    def choose(
        self,
        state: dict[str, Any],
        legal_actions: list[Action],
        *,
        player_id: int,
    ) -> BotDecision:
        ...

    # Helpful default repr for trajectory metadata
    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
