"""Single-elimination 8-player bracket runner.

::

    runner = TournamentRunner(env=env, players=players, seed=42)
    result = runner.run_bracket()
    for match in result.matches:
        print(match.match_id, match.winner_label)

``players`` is a list of length 8 in seat order [A..H]. The runner does
not shuffle — bracket seeding is intentionally static so identical
seeds across pods always produce comparable matchup statistics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..gameplay.env import TalisharEnv
from .bracket import BracketSlot, BracketTree, eight_player_bracket, render_text
from .match import MatchResult, run_match
from .player import Player


@dataclass
class TournamentResult:
    tournament_id: str
    seed: int
    players: list[Player]
    matches: list[MatchResult] = field(default_factory=list)
    bracket: BracketTree = field(default_factory=eight_player_bracket)
    placements: dict[str, int] = field(default_factory=dict)
    started_at: float = 0.0
    ended_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def champion(self) -> Player | None:
        for p in self.players:
            if self.placements.get(p.label) == 1:
                return p
        return None

    def render(self) -> str:
        return render_text(self.bracket)


class TournamentRunner:
    """Drives the 8-player single-elimination bracket.

    Parameters
    ----------
    env:
        A *single* connected :class:`TalisharEnv`. Matches are played
        sequentially through one adapter. For higher throughput, run
        multiple TournamentRunners against multiple adapter ports.
    players:
        Length-8 list in seat order [A..H].
    seed:
        Tournament seed — combined with ``match_id`` to produce
        per-game deterministic seeds.
    best_of:
        Match length. 1 = Bo1. For Bo3 the runner plays until someone
        wins ``ceil(best_of/2)`` games.
    """

    def __init__(
        self,
        *,
        env: TalisharEnv,
        players: list[Player],
        seed: int,
        tournament_id: str = "",
        best_of: int = 1,
    ) -> None:
        if len(players) != 8:
            raise ValueError(f"TournamentRunner expects 8 players; got {len(players)}")
        for i, p in enumerate(players):
            expected = chr(ord("A") + i)
            if p.label != expected:
                p.label = expected  # auto-relabel
        self.env = env
        self.players = players
        self.seed = seed
        self.tournament_id = tournament_id or f"t{seed:08d}"
        self.best_of = best_of

    # ------------------------------------------------------------------
    # Public driver
    # ------------------------------------------------------------------
    def run_bracket(self) -> TournamentResult:
        started = time.time()
        bracket = eight_player_bracket()
        by_label = {p.label: p for p in self.players}
        result = TournamentResult(
            tournament_id=self.tournament_id,
            seed=self.seed,
            players=self.players,
            bracket=bracket,
            started_at=started,
        )

        # Walk each round in order, resolving "W(...)" references as we go.
        for round_slots in bracket.rounds:
            for slot in round_slots:
                p1 = self._resolve(slot.p1_label, by_label, bracket)
                p2 = self._resolve(slot.p2_label, by_label, bracket)
                winner_label = self._play_best_of(slot, p1, p2, result)
                slot.winner_label = winner_label

        # Assign placements.
        # IMPORTANT: resolve "W(...)" placeholders to concrete A..H labels
        # FIRST. Mixing placeholders and resolved labels (e.g. via set diff)
        # used to swallow the champion when set.pop returned the placeholder
        # for the path the champion took to the final.
        final_slot = bracket.find("F")
        final_p1 = self._resolve_label(final_slot.p1_label, bracket)
        final_p2 = self._resolve_label(final_slot.p2_label, bracket)
        champion = final_slot.winner_label or ""
        runner_up = final_p1 if champion == final_p2 else final_p2 if champion == final_p1 else ""
        if champion:
            result.placements[champion] = 1
        if runner_up:
            result.placements[runner_up] = 2
        # Semi-finalists who lost in SF -> 3rd-4th.
        for slot in bracket.rounds[1]:
            if slot.winner_label:
                loser = self._loser(slot, bracket)
                if loser:
                    result.placements.setdefault(loser, 3)
        # Quarter-finalists who lost in QF -> 5th-8th.
        for slot in bracket.rounds[0]:
            if slot.winner_label:
                loser = self._loser(slot, bracket)
                if loser:
                    result.placements.setdefault(loser, 5)

        result.ended_at = time.time()
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _play_best_of(
        self,
        slot: BracketSlot,
        p1: Player, p2: Player,
        result: TournamentResult,
    ) -> str:
        needed = self.best_of // 2 + 1
        wins = {p1.label: 0, p2.label: 0}
        game_idx = 0
        while max(wins.values()) < needed:
            game_seed = self._game_seed(slot.match_id, game_idx)
            m = run_match(
                env=self.env, p1=p1, p2=p2,
                match_id=f"{slot.match_id}.g{game_idx}",
                seed=game_seed,
            )
            result.matches.append(m)
            if m.winner_label:
                wins[m.winner_label] = wins.get(m.winner_label, 0) + 1
            else:
                # Draw / error — break to avoid infinite loop.
                break
            game_idx += 1
            if game_idx > 10:  # belt-and-braces
                break
        winner = max(wins.items(), key=lambda kv: kv[1])[0]
        return winner

    def _resolve(self, label: str, by_label: dict[str, Player], bracket: BracketTree) -> Player:
        if label.startswith("W(") and label.endswith(")"):
            source = bracket.find(label[2:-1])
            if not source.winner_label:
                raise RuntimeError(f"upstream match {source.match_id} not yet resolved")
            return by_label[source.winner_label]
        return by_label[label]

    def _resolve_label(self, label: str, bracket: BracketTree) -> str:
        if label.startswith("W(") and label.endswith(")"):
            return bracket.find(label[2:-1]).winner_label or ""
        return label

    def _loser(self, slot: BracketSlot, bracket: BracketTree) -> str:
        p1 = self._resolve_label(slot.p1_label, bracket)
        p2 = self._resolve_label(slot.p2_label, bracket)
        if slot.winner_label == p1: return p2
        if slot.winner_label == p2: return p1
        return ""

    def _game_seed(self, match_id: str, game_idx: int) -> int:
        # Mix tournament seed + match id + game idx into a 32-bit value.
        h = (self.seed * 1315423911) & 0xFFFFFFFF
        for ch in match_id.encode():
            h = ((h << 5) + h + ch) & 0xFFFFFFFF
        return (h ^ (game_idx * 2654435761)) & 0x7FFFFFFF
