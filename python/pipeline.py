"""End-to-end limited testing orchestrator.

Wires together the modules so a single call drives:

::

    Draft Pod -> Build Decks -> Tournament -> Record everything -> (Train)

The orchestrator is intentionally synchronous. To run many pods in
parallel, spawn N orchestrators (one process each) and partition them
by adapter port. Per project memory: cap at 32 parallel adapter workers
on this host.

Public API
----------
* :class:`LimitedPipeline` — coordinates one cycle.
* :func:`run_cycle(...)`    — convenience wrapper.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .deckbuilding.builder import DeckBuilder, HeuristicDeckBuilder, RandomDeckBuilder
from .deckbuilding.card_catalog import CardCatalog, load_card_catalog
from .deckbuilding.deck import Deck
from .draft.bots.base import DraftBot
from .draft.bots.random_bot import RandomDraftBot
from .draft.bots.heuristic_bot import HeuristicDraftBot
from .draft.dataset import DraftDatasetWriter
from .draft.draftmancer import (
    DraftmancerCube,
    class_map_from_cube,
    load_pack_pool_draftmancer,
    parse_draftmancer,
)
from .draft.format import (
    CLASS_HERO, DECISIVE_CLASSES, HERO_CLASS, HERO_WEAPONS,
    LEGAL_HEROES, LEGAL_WEAPONS,
)
from .draft.pack_loader import PackPool, load_pack_pool
from .draft.simulator import DraftPodConfig, DraftPodResult, DraftSimulator
from .gameplay.bots.base import Bot
from .gameplay.bots.random_bot import RandomBot
from .gameplay.bots.heuristic_bot import HeuristicBot
from .gameplay.bots.aggro_bot import AggroBot
from .gameplay.dataset_writer import DatasetWriter
from .gameplay.env import TalisharEnv, wait_for_adapter
from .gameplay.replay_buffer import Trajectory
from .tournament.player import Player
from .tournament.runner import TournamentResult, TournamentRunner
from .tournament.round_robin import RoundRobinResult, RoundRobinRunner
from .tournament.parallel_round_robin import ParallelRoundRobinRunner


# ---------------------------------------------------------------------------
# Bot factories
# ---------------------------------------------------------------------------
def default_draft_bot(seat: int, seed: int) -> DraftBot:
    # Mixed pod: alternate random / heuristic so analytics is non-trivial.
    return HeuristicDraftBot(seed=seed + seat) if seat % 2 == 0 \
        else RandomDraftBot(seed=seed + seat)


def default_gameplay_bot_factory(player_seed: int) -> Callable[[int], Bot]:
    """Return a factory that the tournament can call per-game.

    Each match instantiates a fresh bot to keep RNG state isolated.
    Uses :class:`AggroBot` so games are *decisive* (it attacks in the
    main phase and never over-blocks, so attacks connect and someone
    dies) — a round-robin needs decisive games to rank decks. The
    ``player_seed`` is folded into the per-game seed for reproducibility.
    """
    def _make(game_seed: int) -> Bot:
        return AggroBot(seed=game_seed ^ (player_seed * 2654435761 & 0x7FFFFFFF))
    return _make


from collections import Counter as _Counter
import random as _random


# Type alias for the hero-assignment callable. Receives the seat index,
# the completed pod (so the assignment can inspect drafted_pool and the
# bot that drafted it), and a card->classes map (empty dict for legacy
# JSON cubes that don't carry class metadata). Returns the seat's
# (hero, signature weapon) pair — both must be in LEGAL_HEROES /
# LEGAL_WEAPONS or the deck builder will reject the deck.
HeroAssignmentFn = Callable[
    [int, "DraftPodResult", dict[str, set[str]]],
    tuple[str, str],
]


def default_hero_assignment(
    seat: int,
    pod: "DraftPodResult",
    card_classes: dict[str, set[str]],
) -> tuple[str, str]:
    """Cascade hero/weapon assignment.

    Layers, in order:

    1. **Draft-bot preference.** Call ``bot.pick_hero(...)`` for the
       seat. If it returns one of ``LEGAL_HEROES``, that's the seat's
       hero.
    2. **Drafted-class plurality.** Count drafted cards by decisive
       class (Illusionist / Wizard / Runeblade). If exactly one class
       has the most cards, the corresponding hero is chosen.
    3. **Random fallback.** Seeded deterministically from
       ``(pod.seed, seat)`` so re-running the same pod produces the
       same assignment.

    The signature weapon comes from :data:`HERO_WEAPONS` once the hero
    is decided.
    """
    bot = pod.seats[seat].bot
    drafted = tuple(pod.seats[seat].drafted)

    # Layer 1: bot preference.
    choice: str | None = None
    pick_hero = getattr(bot, "pick_hero", None)
    if callable(pick_hero):
        try:
            choice = pick_hero(drafted, LEGAL_HEROES, card_classes)
        except Exception:  # noqa: BLE001 — bots should never crash the pipeline
            choice = None
    if choice in LEGAL_HEROES:
        return choice, _weapon_for(choice, pod.seed, seat)

    # Layer 2: class-count plurality (skipped when no metadata).
    counts: _Counter[str] = _Counter()
    for card in drafted:
        for cls in card_classes.get(card, ()):
            if cls in DECISIVE_CLASSES:
                counts[cls] += 1
    if counts:
        ranked = counts.most_common()
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            top_class = ranked[0][0]
            candidate = CLASS_HERO.get(top_class)
            if candidate in LEGAL_HEROES:
                return candidate, _weapon_for(candidate, pod.seed, seat)

    # Layer 3: deterministic random.
    rng = _random.Random((pod.seed << 8) ^ seat)
    hero = rng.choice(list(LEGAL_HEROES))
    return hero, _weapon_for(hero, pod.seed, seat)


def round_robin_hero_assignment(
    seat: int,
    pod: "DraftPodResult | None" = None,
    card_classes: dict[str, set[str]] | None = None,
) -> tuple[str, str]:
    """Legacy round-robin assignment, kept as a named alternative.

    Rotates ``LEGAL_HEROES`` reproducibly. Accepts the new cascade
    signature so it remains plug-compatible with
    :data:`PipelineConfig.hero_assignment`.
    """
    hero = LEGAL_HEROES[seat % len(LEGAL_HEROES)]
    return hero, _weapon_for(hero, getattr(pod, "seed", 0) if pod else 0, seat)


def _weapon_for(hero: str, pod_seed: int, seat: int) -> str:
    weapon = HERO_WEAPONS.get(hero)
    if weapon is None or weapon not in LEGAL_WEAPONS:
        # Defensive: mapping drift shouldn't crash a cycle.
        rng = _random.Random((pod_seed << 16) ^ seat)
        weapon = rng.choice(list(LEGAL_WEAPONS))
    return weapon


def _detect_pack_loader(path: str) -> Callable[[], PackPool]:
    """Pick the right pack loader based on file extension.

    ``.json`` -> :func:`python.draft.pack_loader.load_pack_pool` (legacy).
    ``.txt`` / ``.draft`` -> draftmancer cube parser. The draftmancer
    branch needs ``n_packs`` and ``seed`` at load time, so the returned
    callable closes over a 24-pack default; pipelines that want a
    different size should provide ``pack_pool_factory`` explicitly.
    """
    lower = path.lower()
    if lower.endswith(".json"):
        return lambda: load_pack_pool(path)
    if lower.endswith((".txt", ".draft", ".draftmancer")):
        return lambda: load_pack_pool_draftmancer(path, n_packs=32, seed=0)
    raise ValueError(f"Unknown pack file extension: {path}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    adapter_url: str = "http://localhost:8000"
    # Multiple adapter worker URLs for parallel round-robin. When set with
    # >1 entry, the round-robin shards its pairs across these workers.
    adapter_urls: list[str] | None = None
    packs_path: str = "decks/sample_packs/oma_sample.json"
    out_dir: str = "outputs"
    catalog_fallback: str | None = None
    seed: int = 0
    n_pods: int = 1
    n_players: int = 8
    packs_per_player: int = 3
    best_of: int = 1

    # Tournament format. "bracket" = single-elimination Bo``best_of``.
    # "round_robin" = every deck pair plays ``games_per_pair`` games; on a
    # tied series, play on until one deck leads by ``win_by``.
    tournament_mode: str = "bracket"
    games_per_pair: int = 10
    win_by: int = 2
    rr_max_extra_games: int = 20
    step_cap: int = 2000
    deck_builder_factory: Callable[[CardCatalog, int], DeckBuilder] | None = None
    draft_bot_factory: Callable[[int, int], DraftBot] = default_draft_bot
    gameplay_bot_factory: Callable[[int], Callable[[int], Bot]] = default_gameplay_bot_factory
    dataset_fmt: str = "parquet"

    # Hero / weapon assignment for cubes that don't include them in packs
    # (e.g. the OMN cube, which only ships Lightning-themed Equipment +
    # actions and assumes players bring their own hero outside the draft).
    # Receives (seat, pod, card_classes) and returns (hero, weapon). The
    # default :func:`default_hero_assignment` cascades:
    #   1. bot.pick_hero(...)
    #   2. drafted-class plurality
    #   3. deterministic random
    # Set to None to disable injection — the deck builder will then
    # expect both cards to appear in the drafted pool.
    hero_assignment: HeroAssignmentFn | None = default_hero_assignment

    # Custom pack-pool factory. When set, takes precedence over
    # ``packs_path``. Use this to plug in alternative cube formats or
    # in-process synthetic pools (the OMN smoke test uses this).
    pack_pool_factory: Callable[[], PackPool] | None = None

    # Optional pre-parsed cube. When set, ``LimitedPipeline`` reads its
    # class map from this object instead of re-parsing the cube file.
    # Particularly useful for the OMN smoke test which already parses
    # the cube to build the pack pool.
    cube: DraftmancerCube | None = None


@dataclass
class CycleResult:
    cycle_id: str
    pods: list[DraftPodResult] = field(default_factory=list)
    decks_by_pod_seat: dict[tuple[str, int], Deck] = field(default_factory=dict)
    tournaments: list[TournamentResult] = field(default_factory=list)
    artefacts: dict[str, list[str]] = field(default_factory=dict)
    started_at: float = 0.0
    ended_at: float = 0.0


class LimitedPipeline:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        if cfg.pack_pool_factory is not None:
            self.pool = cfg.pack_pool_factory()
        else:
            self.pool = _detect_pack_loader(cfg.packs_path)()
        # Class map: prefer an explicit pre-parsed cube; else try to
        # parse the cube file if it's a draftmancer .txt; else empty
        # (the cascade then skips class-count and goes straight to random).
        self.cube: DraftmancerCube | None = cfg.cube
        if self.cube is None and cfg.packs_path.lower().endswith((".txt", ".draft", ".draftmancer")):
            try:
                self.cube = parse_draftmancer(cfg.packs_path)
            except Exception:  # noqa: BLE001
                self.cube = None
        self.card_classes: dict[str, set[str]] = (
            self.cube.class_map() if self.cube is not None else {}
        )
        self.catalog: CardCatalog = load_card_catalog(
            talishar_root=Path(__file__).resolve().parents[1] / "talishar",
            fallback_json=cfg.catalog_fallback,
            pack_universe=self.pool.card_universe(),
        )

    # ------------------------------------------------------------------
    def run_cycle(self) -> CycleResult:
        cfg = self.cfg
        cycle_id = f"c{cfg.seed:08d}_{uuid.uuid4().hex[:6]}"
        started = time.time()
        result = CycleResult(cycle_id=cycle_id, started_at=started)

        # 1) Drafts
        pods = self._run_drafts(result, cycle_id)

        # 2) Deck construction
        decks_by_pod_seat = self._build_decks(pods)
        result.decks_by_pod_seat = decks_by_pod_seat
        self._persist_decks(decks_by_pod_seat, result)

        # 3) Tournament per pod
        # One env per adapter worker. With multiple workers + round-robin,
        # the 28 pairs are sharded across them (ParallelRoundRobinRunner);
        # game state is file-based on shared volumes so any worker serves
        # any game safely.
        urls = cfg.adapter_urls or [cfg.adapter_url]
        for u in urls:
            wait_for_adapter(u, timeout_s=30.0)
        envs = [TalisharEnv(u) for u in urls]
        try:
            for pod in pods:
                players = self._players_for_pod(pod, decks_by_pod_seat)
                tour_seed = cfg.seed + hash(pod.pod_id) % 2**31
                if cfg.tournament_mode == "round_robin":
                    if len(envs) > 1:
                        runner = ParallelRoundRobinRunner(
                            envs=envs, players=players, seed=tour_seed,
                            tournament_id=f"t_{pod.pod_id}",
                            games_per_pair=cfg.games_per_pair, win_by=cfg.win_by,
                            max_extra_games=cfg.rr_max_extra_games, step_cap=cfg.step_cap,
                        )
                    else:
                        runner = RoundRobinRunner(
                            env=envs[0], players=players, seed=tour_seed,
                            tournament_id=f"t_{pod.pod_id}",
                            games_per_pair=cfg.games_per_pair, win_by=cfg.win_by,
                            max_extra_games=cfg.rr_max_extra_games, step_cap=cfg.step_cap,
                        )
                    tour = runner.run()
                    print(tour.render())
                else:
                    runner = TournamentRunner(
                        env=envs[0], players=players, seed=tour_seed,
                        tournament_id=f"t_{pod.pod_id}",
                        best_of=cfg.best_of,
                    )
                    tour = runner.run_bracket()
                result.tournaments.append(tour)

            # 4) Persist trajectories + bracket summaries
            self._persist_games(result)
            self._persist_tournaments(result)
            self._persist_matches(result)
            self._persist_drafts(pods, result)  # re-write with placement column
        finally:
            for e in envs:
                e.close()

        result.ended_at = time.time()
        return result

    # ------------------------------------------------------------------
    # 1. Drafts
    # ------------------------------------------------------------------
    def _run_drafts(self, result: CycleResult, cycle_id: str) -> list[DraftPodResult]:
        cfg = self.cfg
        pods: list[DraftPodResult] = []
        for k in range(cfg.n_pods):
            seed = cfg.seed + k
            bots = [cfg.draft_bot_factory(seat, seed) for seat in range(cfg.n_players)]
            sim = DraftSimulator(
                self.pool, bots,
                DraftPodConfig(
                    n_players=cfg.n_players,
                    packs_per_player=cfg.packs_per_player,
                    seed=seed,
                    pod_id=f"{cycle_id}_pod{k:03d}",
                ),
            )
            pod = sim.run()
            pods.append(pod)
        result.pods = pods
        return pods

    # ------------------------------------------------------------------
    # 2. Decks
    # ------------------------------------------------------------------
    def _build_decks(self, pods: list[DraftPodResult]) -> dict[tuple[str, int], Deck]:
        cfg = self.cfg
        out: dict[tuple[str, int], Deck] = {}
        # The default builder factory now threads the cube-derived class
        # map through so HeuristicDeckBuilder can enforce OMN's
        # class+talent legality and pad shortfalls with Cracked Bauble.
        card_classes = self.card_classes
        if cfg.deck_builder_factory is not None:
            builder_factory = cfg.deck_builder_factory
        else:
            builder_factory = lambda catalog, seed: HeuristicDeckBuilder(
                catalog=catalog, card_classes=card_classes, seed=seed,
            )
        for pod in pods:
            for seat in range(cfg.n_players):
                pool = pod.drafted_pool(seat)
                # If the format assigns heroes outside the booster pool,
                # inject the assigned hero + signature weapon so the
                # builder + legality checks see a complete card pool.
                if cfg.hero_assignment is not None:
                    hero, weapon = cfg.hero_assignment(seat, pod, self.card_classes)
                    pool = [hero, weapon, *pool]
                builder = builder_factory(self.catalog, pod.seed + seat)
                deck = builder.build_deck(pool)
                out[(pod.pod_id, seat)] = deck
        return out

    # ------------------------------------------------------------------
    # 3. Tournament
    # ------------------------------------------------------------------
    def _players_for_pod(
        self,
        pod: DraftPodResult,
        decks: dict[tuple[str, int], Deck],
    ) -> list[Player]:
        cfg = self.cfg
        players: list[Player] = []
        for seat in range(cfg.n_players):
            deck = decks[(pod.pod_id, seat)]
            players.append(Player(
                seat=seat,
                deck=deck,
                bot_factory=cfg.gameplay_bot_factory(pod.seed + seat),
                name=f"{pod.pod_id}_{deck.hero}_seat{seat}",
            ))
        return players

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _persist_drafts(
        self,
        pods: list[DraftPodResult],
        result: CycleResult,
    ) -> None:
        cfg = self.cfg
        # Build a placement map from the recorded tournaments.
        placements: dict[tuple[str, int], int] = {}
        for tour in result.tournaments:
            for player in tour.players:
                placement = tour.placements.get(player.label, 0)
                # Pod id is encoded into the tournament id; we tagged players by name above.
                pod_id = tour.tournament_id.removeprefix("t_")
                placements[(pod_id, player.seat)] = placement

        writer = DraftDatasetWriter(cfg.out_dir, fmt=cfg.dataset_fmt)  # type: ignore[arg-type]
        path = writer.write_pods(pods, placements=placements)
        if path:
            result.artefacts.setdefault("drafts", []).append(str(path))

    def _persist_decks(
        self,
        decks: dict[tuple[str, int], Deck],
        result: CycleResult,
    ) -> None:
        cfg = self.cfg
        out_dir = Path(cfg.out_dir) / "parquet" / "decks"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for (pod_id, seat), deck in decks.items():
            rows.append({
                "pod_id": pod_id,
                "seat":   seat,
                "hero":   deck.hero,
                "weapon": deck.weapon,
                "deck_json": json.dumps(deck.deck, separators=(",", ":")),
                "sideboard_json": json.dumps(deck.sideboard, separators=(",", ":")),
                "pitch_distribution_json": json.dumps(deck.evaluation.pitch_distribution, separators=(",", ":")),
                "curve_histogram_json": json.dumps(deck.evaluation.curve_histogram, separators=(",", ":")),
                "overall_score": deck.evaluation.overall_score,
            })
        if not rows:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            return  # graceful no-op if pyarrow missing
        path = out_dir / f"{int(time.time())}-{uuid.uuid4().hex[:6]}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), str(path))
        result.artefacts.setdefault("decks", []).append(str(path))

    def _persist_games(self, result: CycleResult) -> None:
        cfg = self.cfg
        writer = DatasetWriter(cfg.out_dir, fmt=cfg.dataset_fmt)  # type: ignore[arg-type]
        trajectories: list[Trajectory] = []
        for tour in result.tournaments:
            for m in tour.matches:
                trajectories.append(m.trajectory)
        path = writer.write_batch(trajectories)
        if path:
            result.artefacts.setdefault("games", []).append(str(path))

    def _persist_tournaments(self, result: CycleResult) -> None:
        cfg = self.cfg
        out_dir = Path(cfg.out_dir) / "parquet" / "tournaments"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for tour in result.tournaments:
            for p in tour.players:
                rows.append({
                    "tournament_id": tour.tournament_id,
                    "seed":          tour.seed,
                    "label":         p.label,
                    "seat":          p.seat,
                    "hero":          p.deck.hero,
                    "weapon":        p.deck.weapon,
                    "placement":     int(tour.placements.get(p.label, 0)),
                    "name":          p.name,
                })
        if not rows:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            return
        path = out_dir / f"{int(time.time())}-{uuid.uuid4().hex[:6]}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), str(path))
        result.artefacts.setdefault("tournaments", []).append(str(path))

    def _persist_matches(self, result: CycleResult) -> None:
        """One row per game with its TERMINATION REASON, so engine/adapter
        aborts (bad_priority / no_legal / *_error) can be told apart from real
        step-cap stalemates and lethal wins. The games parquet only carries
        per-transition rows + winner, which collapses every winner-0 outcome
        into an indistinguishable "draw"; this is the authoritative per-game
        outcome ledger. The trainer never reads it, so it's purely diagnostic.
        """
        cfg = self.cfg
        out_dir = Path(cfg.out_dir) / "parquet" / "matches"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for tour in result.tournaments:
            for m in getattr(tour, "matches", []) or []:
                md = m.metadata or {}
                rows.append({
                    "tournament_id": tour.tournament_id,
                    "match_id":      m.match_id,
                    "game_id":       m.trajectory.game_id,
                    "game_seed":     m.game_seed,
                    "p1_name":       m.p1_name,
                    "p2_name":       m.p2_name,
                    "winner_label":  m.winner_label,
                    "winner_seat":   m.winner_seat,
                    "engine_winner": int(md.get("engine_winner") or 0),
                    "term_reason":   md.get("term_reason") or "",
                    "steps":         int(md.get("steps") or 0),
                    "recoveries":    int(md.get("recoveries") or 0),
                    "tiebreak":      bool(md.get("tiebreak")),
                    "error":         m.error or "",
                })
        if not rows:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            return
        path = out_dir / f"{int(time.time())}-{uuid.uuid4().hex[:6]}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), str(path))
        result.artefacts.setdefault("matches", []).append(str(path))


def run_cycle(cfg: PipelineConfig) -> CycleResult:
    return LimitedPipeline(cfg).run_cycle()
