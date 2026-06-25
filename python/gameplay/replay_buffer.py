"""In-memory trajectory buffer for offline-RL data collection.

Each transition captures everything an offline-RL algorithm such as IQL
needs: the observation (game state), the legal-action set + boolean mask
(so the critic can mask out impossible actions), the chosen action, the
scalar reward, the next observation, ``done``, the player on the move,
and per-step wall-clock timestamps.

Why not stream directly to disk?
--------------------------------
Streaming row-by-row to parquet would force ``pyarrow`` to flush every
step and kill throughput. Instead we buffer one full game in RAM and
hand the whole trajectory to ``dataset_writer.DatasetWriter`` which can
batch many games into a single rowgroup.

A finished trajectory is a ``Trajectory`` object that the dataset writer
treats as a single row group; per-step records live in lists keyed
positionally.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Transition:
    """One (s, a, r, s', done) tuple plus RL metadata.

    Notes
    -----
    * ``state`` and ``next_state`` are the *serialised* JSON dicts we got
      back from the adapter; encoding to tensors is the policy's job.
    * ``legal_action_ids`` is the integer action_ids list returned by the
      adapter (1-indexed). Use ``action_mask`` for fixed-width models.
    * ``ts_unix`` lets you compute per-step latency offline.
    """
    state: dict[str, Any]
    legal_actions: list[dict[str, Any]]
    legal_action_ids: list[int]
    action_mask: list[bool]
    chosen_action: dict[str, Any]
    chosen_action_id: int
    reward: float
    next_state: dict[str, Any]
    done: bool
    player_to_move: int
    step_index: int
    ts_unix: float


@dataclass
class Trajectory:
    """A complete self-play game.

    Attributes
    ----------
    game_id, seed:
        Identifiers carried into parquet for joinability.
    hero1, hero2, deck1, deck2:
        Game setup that produced the trajectory.
    winner:
        Final terminal winner (1, 2, or 0 for draw / unfinished).
    transitions:
        Ordered list of ``Transition`` records.
    metadata:
        Free-form per-game annotations (bot names, branch, model hash...).
    """
    game_id: str
    seed: int
    hero1: str
    hero2: str
    deck1: str
    deck2: str
    transitions: list[Transition] = field(default_factory=list)
    winner: int = 0
    final_reward_p1: float = 0.0
    final_reward_p2: float = 0.0
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.transitions)

    def add(self, t: Transition) -> None:
        self.transitions.append(t)

    def finalise(self, winner: int) -> None:
        self.winner = winner
        self.ended_at = time.time()
        self.final_reward_p1 = 1.0 if winner == 1 else (-1.0 if winner == 2 else 0.0)
        self.final_reward_p2 = -self.final_reward_p1


class ReplayBuffer:
    """Holds completed trajectories until ``DatasetWriter`` flushes them.

    Keep this lean — heavyweight features (windowed sampling, prioritised
    replay, importance weights) belong in the training pipeline, not the
    self-play data collector.
    """

    def __init__(self, max_trajectories: int = 1024) -> None:
        self.max_trajectories = max_trajectories
        self._buf: list[Trajectory] = []

    def __len__(self) -> int:
        return len(self._buf)

    def append(self, t: Trajectory) -> None:
        self._buf.append(t)
        if len(self._buf) > self.max_trajectories:
            # FIFO drop — caller should flush more often
            self._buf.pop(0)

    def drain(self) -> list[Trajectory]:
        out = self._buf
        self._buf = []
        return out

    def total_transitions(self) -> int:
        return sum(len(t) for t in self._buf)


def make_action_mask(legal_action_ids: list[int], width: int) -> list[bool]:
    """Build a fixed-width boolean mask suitable for transformer policies."""
    mask = [False] * width
    for aid in legal_action_ids:
        i = aid - 1
        if 0 <= i < width:
            mask[i] = True
    return mask
