"""Run a full 8-player draft pod with deterministic pack-passing.

Direction convention
--------------------
Following standard FaB draft rules (and the user's spec):

* **Pack 1** passes **LEFT**  (each seat's leftovers go to seat ``(i+1) % n``)
* **Pack 2** passes **RIGHT** (leftovers go to seat ``(i-1) % n``)
* **Pack 3** passes **LEFT**

"Seat 0" is the conceptual "Player A". The bracket module's ``A..H``
labels map to seats 0..7.

Signals & wheel tracking
------------------------
Every pick records, in addition to the chosen card:

* ``seat``           — seat index (0..n-1)
* ``pack_number``    — 1, 2, or 3
* ``pick_number``    — 1-indexed within the pack (1..cards_in_pack)
* ``pack_remaining`` — cards still in the pack after the pick (so a bot
                       inspecting a future draft can "see" what wheeled)
* ``neighbours``     — pre-computed left/right seat IDs (signal source)

Bots receive a :class:`DraftPodView` snapshot each call so they can do
context-aware reasoning (e.g. archetype signals, hero locks).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from .bots.base import DraftBot, DraftPodView
from .pack_loader import Pack, PackPool, sample_player_packs


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DraftPodConfig:
    """Per-pod configuration.

    The defaults match the user's spec for OMA limited (8 players, 3 packs,
    LRL pass order).
    """
    n_players: int = 8
    packs_per_player: int = 3
    pass_directions: tuple[int, ...] = (+1, -1, +1)
    record_neighbours: bool = True
    seed: int = 0
    pod_id: str = ""

    def __post_init__(self) -> None:
        if len(self.pass_directions) != self.packs_per_player:
            raise ValueError(
                f"pass_directions length ({len(self.pass_directions)}) "
                f"must equal packs_per_player ({self.packs_per_player})"
            )


@dataclass
class DraftSeat:
    """The state of a single player throughout the draft."""
    seat: int
    bot: DraftBot
    drafted: list[str] = field(default_factory=list)
    bot_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DraftPick:
    """One pick decision recorded in the pod history."""
    seat: int
    pack_number: int
    pick_number: int
    pack_contents: list[str]
    chosen_card: str
    remaining_after_pick: list[str]
    left_neighbour: int
    right_neighbour: int
    bot_info: dict[str, Any] = field(default_factory=dict)
    ts_unix: float = field(default_factory=time.time)


@dataclass
class DraftPodResult:
    """Output of one finished pod."""
    pod_id: str
    seed: int
    config: DraftPodConfig
    seats: list[DraftSeat]
    picks: list[DraftPick]
    pack_assignments: list[list[Pack]]
    started_at: float
    ended_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def drafted_pool(self, seat: int) -> list[str]:
        return list(self.seats[seat].drafted)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class DraftSimulator:
    """Drives one draft pod to completion.

    Usage::

        pool = load_pack_pool("decks/sample_packs/oma_sample.json")
        bots = [RandomDraftBot(seed=i) for i in range(8)]
        sim  = DraftSimulator(pool, bots, DraftPodConfig(seed=42, pod_id="pod001"))
        result = sim.run()

    The simulator never touches Talishar; drafting is a pure-Python
    information game. Talishar enters the picture during the subsequent
    tournament phase.
    """

    def __init__(
        self,
        pool: PackPool,
        bots: list[DraftBot],
        config: DraftPodConfig | None = None,
    ) -> None:
        self.pool = pool
        self.config = config or DraftPodConfig()
        if len(bots) != self.config.n_players:
            raise ValueError(
                f"Expected {self.config.n_players} bots, got {len(bots)}"
            )
        self.bots = bots

    # ------------------------------------------------------------------
    # Public driver
    # ------------------------------------------------------------------
    def run(self) -> DraftPodResult:
        started = time.time()
        assignments = sample_player_packs(
            self.pool,
            n_players=self.config.n_players,
            packs_per_player=self.config.packs_per_player,
            seed=self.config.seed,
        )
        seats = [DraftSeat(seat=i, bot=self.bots[i]) for i in range(self.config.n_players)]
        for s in seats:
            s.bot.reset(seed=self.config.seed + s.seat)

        picks: list[DraftPick] = []
        for pack_round in range(self.config.packs_per_player):
            direction = self.config.pass_directions[pack_round]
            picks += self._run_pack_round(seats, assignments, pack_round, direction)

        ended = time.time()
        return DraftPodResult(
            pod_id=self.config.pod_id or f"pod_seed{self.config.seed}",
            seed=self.config.seed,
            config=self.config,
            seats=seats,
            picks=picks,
            pack_assignments=assignments,
            started_at=started,
            ended_at=ended,
        )

    # ------------------------------------------------------------------
    # Per-pack-round driver
    # ------------------------------------------------------------------
    def _run_pack_round(
        self,
        seats: list[DraftSeat],
        assignments: list[list[Pack]],
        pack_round: int,
        direction: int,
    ) -> list[DraftPick]:
        """Open pack ``pack_round`` for every seat and pass until empty."""
        # Each seat starts with their own pack
        in_hand: list[list[str]] = [list(assignments[s][pack_round].cards) for s in range(len(seats))]
        pack_size = max(len(p) for p in in_hand)
        picks: list[DraftPick] = []

        for pick_no in range(1, pack_size + 1):
            decisions: list[tuple[int, str]] = []
            for seat in range(len(seats)):
                pack = in_hand[seat]
                if not pack:
                    continue
                view = self._view(seats, pack_round, pick_no, seat, pack)
                chosen = self._safe_choose(seats[seat], pack, view)
                if chosen not in pack:
                    raise ValueError(
                        f"Bot at seat {seat} returned {chosen!r} which is not in pack {pack!r}"
                    )
                # Record BEFORE removing so pack_contents reflects what the bot saw.
                left_n, right_n = self._neighbours(seat, len(seats))
                remaining = [c for c in pack if c != chosen]  # remove ONE occurrence below
                # Remove first occurrence only (cards can repeat in a pack).
                idx = pack.index(chosen)
                _ = pack.pop(idx)
                picks.append(
                    DraftPick(
                        seat=seat,
                        pack_number=pack_round + 1,
                        pick_number=pick_no,
                        pack_contents=list(view.current_pack),
                        chosen_card=chosen,
                        remaining_after_pick=list(pack),
                        left_neighbour=left_n,
                        right_neighbour=right_n,
                        bot_info=dict(view.bot_decision_info),
                    )
                )
                seats[seat].drafted.append(chosen)
                decisions.append((seat, chosen))

            # Pass remainders. Capture into a new list first so we don't
            # mutate during iteration.
            new_hand: list[list[str]] = [[] for _ in seats]
            for seat in range(len(seats)):
                target = (seat + direction) % len(seats)
                new_hand[target] = in_hand[seat]
            in_hand = new_hand
            # Stop early if every pack is now empty.
            if not any(in_hand):
                break

        return picks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _view(
        self,
        seats: list[DraftSeat],
        pack_round: int,
        pick_no: int,
        seat: int,
        pack: list[str],
    ) -> DraftPodView:
        left_n, right_n = self._neighbours(seat, len(seats))
        return DraftPodView(
            seat=seat,
            pack_number=pack_round + 1,
            pick_number=pick_no,
            current_pack=tuple(pack),
            drafted_so_far=tuple(seats[seat].drafted),
            left_neighbour_seat=left_n,
            right_neighbour_seat=right_n,
            left_neighbour_drafted=tuple(seats[left_n].drafted),
            right_neighbour_drafted=tuple(seats[right_n].drafted),
            n_seats=len(seats),
            pass_direction=self.config.pass_directions[pack_round],
        )

    def _safe_choose(self, seat: DraftSeat, pack: list[str], view: DraftPodView) -> str:
        """Ask the bot, with defensive fallback to the first card."""
        try:
            decision = seat.bot.choose_card(
                pack=tuple(pack),
                drafted_cards=tuple(seat.drafted),
                seat_position=view.seat,
                pick_number=view.pick_number,
                pack_number=view.pack_number,
                pod_state=view,
            )
        except Exception as e:  # noqa: BLE001
            decision = pack[0]
            view.bot_decision_info["fallback_reason"] = f"bot raised: {e!r}"
        # Bots return either a DraftDecision or a bare card id; normalise.
        if hasattr(decision, "card_id"):
            view.bot_decision_info.update(getattr(decision, "info", {}) or {})
            return getattr(decision, "card_id")
        return str(decision)

    @staticmethod
    def _neighbours(seat: int, n: int) -> tuple[int, int]:
        return ((seat + 1) % n, (seat - 1) % n)


# ---------------------------------------------------------------------------
# Convenience: batch run multiple pods
# ---------------------------------------------------------------------------
def run_batch(
    pool: PackPool,
    *,
    bot_factory,  # Callable[[int seat, int seed], DraftBot]
    base_seed: int,
    n_pods: int,
    n_players: int = 8,
    packs_per_player: int = 3,
) -> Iterable[DraftPodResult]:
    """Yield N pods deterministically (seed = base_seed + pod_index)."""
    for k in range(n_pods):
        seed = base_seed + k
        bots = [bot_factory(seat, seed) for seat in range(n_players)]
        sim = DraftSimulator(
            pool,
            bots,
            DraftPodConfig(
                n_players=n_players,
                packs_per_player=packs_per_player,
                seed=seed,
                pod_id=f"pod_{seed:08d}",
            ),
        )
        yield sim.run()
