"""Round-robin tournament runner with a win-by-2 tiebreak.

Every unordered pair of players plays ``games_per_pair`` games (>=10 by
default), alternating who takes the first-player seat each game (first
player is a real advantage in FAB, so we balance it). The pair's head-to-
head winner is whoever has more game wins after the base games; if those
are tied, additional games are played until one side **leads by 2**
(capped so a pathological all-draw matchup can't loop forever).

Standings rank players by pair-wins (head-to-head series won), breaking
ties by total individual game wins. ``placements`` maps label -> rank
(1 = champion) so the rest of the pipeline (draft-IQL reward, parquet
``placement`` column) works unchanged.

The result object intentionally mirrors the public surface of
:class:`tournament.runner.TournamentResult` (``.matches``, ``.players``,
``.placements``, ``.tournament_id``, ``.seed``) so ``pipeline`` can
persist it without branching.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from ..gameplay.env import TalisharEnv
from .match import MatchResult, run_match
from .player import Player


@dataclass
class PairResult:
    label_a: str
    label_b: str
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    games: int = 0
    went_to_winby: bool = False  # base games were tied -> extra games played
    winner_label: str = ""       # "" only if capped while still tied


@dataclass
class RoundRobinResult:
    tournament_id: str
    seed: int
    players: list[Player]
    matches: list[MatchResult] = field(default_factory=list)
    pairs: list[PairResult] = field(default_factory=list)
    placements: dict[str, int] = field(default_factory=dict)
    # label -> {"pair_wins", "pair_losses", "pair_draws", "game_wins", "game_losses", "game_draws"}
    standings: dict[str, dict[str, int]] = field(default_factory=dict)
    started_at: float = 0.0
    ended_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def champion(self) -> Player | None:
        for p in self.players:
            if self.placements.get(p.label) == 1:
                return p
        return None

    def render(self) -> str:
        lines = [f"Round-robin {self.tournament_id} (seed={self.seed})",
                 f"{'rank':>4}  {'label':<5} {'player':<32} {'pairW-L-D':>10} {'gameW-L-D':>12}"]
        ordered = sorted(self.players, key=lambda p: self.placements.get(p.label, 999))
        by_label = {p.label: p for p in self.players}
        for p in ordered:
            s = self.standings.get(p.label, {})
            rank = self.placements.get(p.label, 0)
            pw = f"{s.get('pair_wins',0)}-{s.get('pair_losses',0)}-{s.get('pair_draws',0)}"
            gw = f"{s.get('game_wins',0)}-{s.get('game_losses',0)}-{s.get('game_draws',0)}"
            lines.append(f"{rank:>4}  {p.label:<5} {p.name[:32]:<32} {pw:>10} {gw:>12}")
        return "\n".join(lines)


class RoundRobinRunner:
    def __init__(
        self,
        *,
        env: TalisharEnv,
        players: list[Player],
        seed: int,
        tournament_id: str = "",
        games_per_pair: int = 10,
        win_by: int = 2,
        max_extra_games: int = 20,
        step_cap: int = 2000,
        progress: bool = True,
    ) -> None:
        if len(players) < 2:
            raise ValueError("round-robin needs at least 2 players")
        for i, p in enumerate(players):
            if not p.label:
                p.label = chr(ord("A") + i)
        self.env = env
        self.players = players
        self.seed = seed
        self.tournament_id = tournament_id or f"rr{seed:08d}"
        self.games_per_pair = games_per_pair
        self.win_by = win_by
        self.max_extra_games = max_extra_games
        self.step_cap = step_cap
        self.progress = progress

    # ------------------------------------------------------------------
    def run(self) -> RoundRobinResult:
        started = time.time()
        result = RoundRobinResult(
            tournament_id=self.tournament_id, seed=self.seed,
            players=self.players, started_at=started,
        )
        standings = {
            p.label: dict(pair_wins=0, pair_losses=0, pair_draws=0,
                          game_wins=0, game_losses=0, game_draws=0)
            for p in self.players
        }

        for pa, pb in combinations(self.players, 2):
            pair = self._play_pair(pa, pb, result)
            result.pairs.append(pair)
            # Accumulate game-level tallies.
            standings[pa.label]["game_wins"] += pair.wins_a
            standings[pa.label]["game_losses"] += pair.wins_b
            standings[pa.label]["game_draws"] += pair.draws
            standings[pb.label]["game_wins"] += pair.wins_b
            standings[pb.label]["game_losses"] += pair.wins_a
            standings[pb.label]["game_draws"] += pair.draws
            # Pair-level (head-to-head series) tally.
            if pair.winner_label == pa.label:
                standings[pa.label]["pair_wins"] += 1
                standings[pb.label]["pair_losses"] += 1
            elif pair.winner_label == pb.label:
                standings[pb.label]["pair_wins"] += 1
                standings[pa.label]["pair_losses"] += 1
            else:
                standings[pa.label]["pair_draws"] += 1
                standings[pb.label]["pair_draws"] += 1

        result.standings = standings
        # Rank: pair_wins desc, then game_wins desc, then label asc (stable).
        ordered = sorted(
            self.players,
            key=lambda p: (-standings[p.label]["pair_wins"],
                           -standings[p.label]["game_wins"],
                           p.label),
        )
        for rank, p in enumerate(ordered, start=1):
            result.placements[p.label] = rank
        result.ended_at = time.time()
        return result

    # ------------------------------------------------------------------
    def _play_pair(self, pa: Player, pb: Player, result: RoundRobinResult) -> PairResult:
        pair = PairResult(label_a=pa.label, label_b=pb.label)
        wins = {pa.label: 0, pb.label: 0}

        def play_game(g: int) -> None:
            # Alternate first player every game to balance the FP advantage.
            if g % 2 == 0:
                first, second = pa, pb
            else:
                first, second = pb, pa
            game_seed = self._game_seed(f"{pa.label}v{pb.label}", g)
            m = run_match(
                env=self.env, p1=first, p2=second,
                match_id=f"{self.tournament_id}.{pa.label}v{pb.label}.g{g}",
                seed=game_seed, step_cap=self.step_cap,
            )
            result.matches.append(m)
            pair.games += 1
            if m.winner_label in wins:
                wins[m.winner_label] += 1
            else:
                pair.draws += 1

        # Base games.
        for g in range(self.games_per_pair):
            play_game(g)

        # Win-by-2 tiebreak when the base series is tied.
        g = self.games_per_pair
        extra = 0
        while wins[pa.label] == wins[pb.label] and extra < self.max_extra_games:
            pair.went_to_winby = True
            play_game(g); g += 1; extra += 1
            play_game(g); g += 1; extra += 1  # play in pairs so margin can hit 2
            # If after a pair someone leads by >=win_by, stop.
            if abs(wins[pa.label] - wins[pb.label]) >= self.win_by:
                break

        pair.wins_a = wins[pa.label]
        pair.wins_b = wins[pb.label]
        if wins[pa.label] > wins[pb.label]:
            pair.winner_label = pa.label
        elif wins[pb.label] > wins[pa.label]:
            pair.winner_label = pb.label
        else:
            pair.winner_label = ""  # capped while tied -> drawn series

        if self.progress:
            print(f"  [{self.tournament_id}] {pa.label} vs {pb.label}: "
                  f"{pair.wins_a}-{pair.wins_b} (draws={pair.draws}, games={pair.games})"
                  f" -> {pair.winner_label or 'TIE'}"
                  + (" [win-by-2]" if pair.went_to_winby else ""))
        return pair

    def _game_seed(self, pair_key: str, game_idx: int) -> int:
        h = (self.seed * 1315423911) & 0xFFFFFFFF
        for ch in pair_key.encode():
            h = ((h << 5) + h + ch) & 0xFFFFFFFF
        return (h ^ (game_idx * 2654435761)) & 0x7FFFFFFF
