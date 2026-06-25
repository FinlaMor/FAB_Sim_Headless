"""Draft bot abstract base + immutable view of the pod state.

A bot receives a :class:`DraftPodView` snapshot rather than the live
simulator, so bot code can never mutate the pod by accident. The view
also exposes neighbouring seats' drafted cards (the "signals" passed by
players upstream/downstream), which is how strong human drafters infer
which archetypes are open.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DraftPodView:
    """Per-pick observation handed to the bot.

    All sequence fields are tuples (i.e. immutable) so accidental mutation
    by the bot never leaks back into the simulator.
    """
    seat: int
    pack_number: int                # 1, 2, or 3
    pick_number: int                # 1-indexed within the pack
    current_pack: tuple[str, ...]   # the cards the bot may pick from
    drafted_so_far: tuple[str, ...]
    left_neighbour_seat: int
    right_neighbour_seat: int
    left_neighbour_drafted: tuple[str, ...]
    right_neighbour_drafted: tuple[str, ...]
    n_seats: int
    pass_direction: int             # +1 (LEFT) or -1 (RIGHT)
    # Bots can write debug info here; the simulator persists it into the dataset.
    bot_decision_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DraftDecision:
    """Optional structured return value from a draft bot.

    Bots may also return a bare card-id string; the simulator normalises
    both shapes.
    """
    card_id: str
    info: dict[str, Any] = field(default_factory=dict)


class DraftBot(abc.ABC):
    """Override ``choose_card`` to implement a new draft bot.

    The pick signature matches the user's spec:

    .. code-block:: python

        class DraftBot:
            def choose_card(
                self,
                pack,
                drafted_cards,
                seat_position,
                pick_number,
                pack_number,
                pod_state,
            ):
                ...

    The optional ``pick_hero(...)`` method participates in the cascade
    used by ``python.pipeline.default_hero_assignment``: bots that
    return a concrete hero short-circuit the cascade; bots that return
    ``None`` (the default) defer to class-count then random.
    """
    name: str = "draft-bot"

    def reset(self, *, seed: int | None = None) -> None:
        """Called once per pod before the first pick. Default: no-op."""

    @abc.abstractmethod
    def choose_card(
        self,
        pack: tuple[str, ...],
        drafted_cards: tuple[str, ...],
        seat_position: int,
        pick_number: int,
        pack_number: int,
        pod_state: DraftPodView,
    ) -> str | DraftDecision:
        ...

    def pick_hero(
        self,
        drafted_cards: tuple[str, ...],
        available_heroes: tuple[str, ...],
        card_classes: dict[str, set[str]],
    ) -> str | None:
        """Optional hero preference for the cascade.

        Return one of ``available_heroes`` to bind that hero to the seat,
        or ``None`` to defer to the next layer (class-count, then
        random). Default: ``None``.
        """
        return None

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(name={self.name!r})"
