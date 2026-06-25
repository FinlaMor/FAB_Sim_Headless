"""Round-robin runner that shards pairs across N adapter workers.

Same semantics as :class:`RoundRobinRunner` (every pair plays
``games_per_pair`` games, ties go to win-by-2, standings ranked by
pair-wins then game-wins) — but the 28 pairs are played concurrently
across N ``TalisharEnv`` instances, one per worker port.

Safe because Talishar game state is file-based on shared volumes and
game ids are unique: each pair (and all its games) is handled by exactly
one env/thread at a time, so no single game is ever touched by two
workers concurrently. Each env owns its own ``requests.Session``.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from queue import Queue

from ..gameplay.env import TalisharEnv
from .match import run_match
from .player import Player
from .round_robin import PairResult, RoundRobinResult


class ParallelRoundRobinRunner:
    def __init__(
        self,
        *,
        envs: list[TalisharEnv],
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
        if not envs:
            raise ValueError("need at least one env")
        for i, p in enumerate(players):
            if not p.label:
                p.label = chr(ord("A") + i)
        self.envs = envs
        self.players = players
        self.seed = seed
        self.tournament_id = tournament_id or f"rr{seed:08d}"
        self.games_per_pair = games_per_pair
        self.win_by = win_by
        self.max_extra_games = max_extra_games
        self.step_cap = step_cap
        self.progress = progress
        self._print_lock = threading.Lock()

    # ------------------------------------------------------------------
    def run(self) -> RoundRobinResult:
        started = time.time()
        result = RoundRobinResult(
            tournament_id=self.tournament_id, seed=self.seed,
            players=self.players, started_at=started,
        )
        pairs = list(combinations(self.players, 2))

        env_q: Queue = Queue()
        for e in self.envs:
            env_q.put(e)

        def work(pa: Player, pb: Player):
            env = env_q.get()
            try:
                return self._play_pair(env, pa, pb)
            finally:
                env_q.put(env)

        pair_results: list[PairResult] = []
        all_matches = []
        failed_pairs: list[tuple[str, str, str]] = []
        with ThreadPoolExecutor(max_workers=len(self.envs)) as ex:
            futures = {ex.submit(work, pa, pb): (pa, pb) for pa, pb in pairs}
            for fut in as_completed(futures):
                pa, pb = futures[fut]
                try:
                    pair, matches = fut.result()
                except Exception as e:  # noqa: BLE001
                    # One adapter hiccup (read timeout, dropped connection)
                    # must never sink the whole tournament. Record the pair
                    # as a no-contest and keep the other 27 — standings and
                    # training still run on everything that finished.
                    failed_pairs.append((pa.label, pb.label, repr(e)))
                    with self._print_lock:
                        print(f"  [pair FAILED] {pa.label} vs {pb.label}: "
                              f"{type(e).__name__}: {e}")
                    continue
                pair_results.append(pair)
                all_matches.extend(matches)

        result.pairs = pair_results
        result.matches = all_matches
        if failed_pairs:
            result.metadata["failed_pairs"] = failed_pairs

        # Standings (identical to the serial runner).
        standings = {
            p.label: dict(pair_wins=0, pair_losses=0, pair_draws=0,
                          game_wins=0, game_losses=0, game_draws=0)
            for p in self.players
        }
        by_label = {p.label: p for p in self.players}
        for pair in pair_results:
            a, b = pair.label_a, pair.label_b
            standings[a]["game_wins"] += pair.wins_a
            standings[a]["game_losses"] += pair.wins_b
            standings[a]["game_draws"] += pair.draws
            standings[b]["game_wins"] += pair.wins_b
            standings[b]["game_losses"] += pair.wins_a
            standings[b]["game_draws"] += pair.draws
            if pair.winner_label == a:
                standings[a]["pair_wins"] += 1; standings[b]["pair_losses"] += 1
            elif pair.winner_label == b:
                standings[b]["pair_wins"] += 1; standings[a]["pair_losses"] += 1
            else:
                standings[a]["pair_draws"] += 1; standings[b]["pair_draws"] += 1

        result.standings = standings
        ordered = sorted(
            self.players,
            key=lambda p: (-standings[p.label]["pair_wins"],
                           -standings[p.label]["game_wins"], p.label),
        )
        for rank, p in enumerate(ordered, start=1):
            result.placements[p.label] = rank
        result.ended_at = time.time()
        result.metadata["workers"] = len(self.envs)
        return result

    # ------------------------------------------------------------------
    def _play_pair(self, env: TalisharEnv, pa: Player, pb: Player):
        pair = PairResult(label_a=pa.label, label_b=pb.label)
        matches = []
        wins = {pa.label: 0, pb.label: 0}

        def play_game(g: int) -> None:
            first, second = (pa, pb) if g % 2 == 0 else (pb, pa)
            m = run_match(
                env=env, p1=first, p2=second,
                match_id=f"{self.tournament_id}.{pa.label}v{pb.label}.g{g}",
                seed=self._game_seed(f"{pa.label}v{pb.label}", g), step_cap=self.step_cap,
            )
            matches.append(m)
            pair.games += 1
            if m.winner_label in wins:
                wins[m.winner_label] += 1
            else:
                pair.draws += 1

        for g in range(self.games_per_pair):
            play_game(g)

        g = self.games_per_pair
        extra = 0
        while wins[pa.label] == wins[pb.label] and extra < self.max_extra_games:
            pair.went_to_winby = True
            play_game(g); g += 1; extra += 1
            play_game(g); g += 1; extra += 1
            if abs(wins[pa.label] - wins[pb.label]) >= self.win_by:
                break

        pair.wins_a, pair.wins_b = wins[pa.label], wins[pb.label]
        if wins[pa.label] > wins[pb.label]:
            pair.winner_label = pa.label
        elif wins[pb.label] > wins[pa.label]:
            pair.winner_label = pb.label
        else:
            pair.winner_label = ""

        if self.progress:
            with self._print_lock:
                print(f"  [{self.tournament_id}] {pa.label} vs {pb.label}: "
                      f"{pair.wins_a}-{pair.wins_b} (draws={pair.draws}, games={pair.games})"
                      f" -> {pair.winner_label or 'TIE'}"
                      + (" [win-by-2]" if pair.went_to_winby else ""), flush=True)
        return pair, matches

    def _game_seed(self, pair_key: str, game_idx: int) -> int:
        h = (self.seed * 1315423911) & 0xFFFFFFFF
        for ch in pair_key.encode():
            h = ((h << 5) + h + ch) & 0xFFFFFFFF
        return (h ^ (game_idx * 2654435761)) & 0x7FFFFFFF
