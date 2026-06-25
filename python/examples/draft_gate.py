"""Draft-skill gate: IQL draft bot vs HeuristicDraftBot, gameplay held fixed.

Both bots draft in the SAME pod (interleaved seats, so they pass packs to
each other exactly like a real pod), decks are built by the SAME
HeuristicDeckBuilder with the SAME hero-assignment cascade, and every game
is piloted on BOTH sides by the SAME frozen gameplay bot. The only varying
factor is who drafted the deck — so the decisive-game winrate delta is pure
DRAFTING skill.

Seat parity alternates per pod (pod 0: IQL on even seats; pod 1: odd) so
table-position effects cancel. Games are sharded across all adapter
workers. Draws are ignored (decisive wins only — house rule).

Run:  python -m python.examples.draft_gate [--pods 2] [--games-per-pair 2]
          [--draft-ckpt outputs/models/draft/latest.pt]
          [--gameplay-ckpt outputs/models/gameplay/latest.pt]
          [--workers 8] [--max-games 0]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.pipeline import (  # noqa: E402
    LimitedPipeline, PipelineConfig, default_hero_assignment,
)
from python.draft.simulator import DraftSimulator, DraftPodConfig  # noqa: E402
from python.draft.bots.iql_bot import IQLDraftBot  # noqa: E402
from python.draft.bots.heuristic_bot import HeuristicDraftBot  # noqa: E402
from python.deckbuilding.builder import HeuristicDeckBuilder  # noqa: E402
from python.gameplay.env import TalisharEnv  # noqa: E402
from python.gameplay.bots.iql_bot import IQLGameplayBot  # noqa: E402
from python.tournament.player import Player  # noqa: E402
from python.tournament.match import run_match  # noqa: E402

CUBE = "OMN_Draft_3.5.txt"


def run_gate(*, pods: int = 2, games_per_pair: int = 2,
             draft_ckpt: str = "outputs/models/draft/latest.pt",
             gameplay_ckpt: str = "outputs/models/gameplay/latest.pt",
             workers: int = 8, step_cap: int = 600,
             seed: int = 700000, max_games: int = 0,
             verbose: bool = True) -> tuple[int, int, int]:
    """Importable core (used standalone and by the continuous loop's
    periodic draft check). Returns (iql_wins, heu_wins, draws)."""
    from types import SimpleNamespace
    args = SimpleNamespace(pods=pods, games_per_pair=games_per_pair,
                           draft_ckpt=draft_ckpt, gameplay_ckpt=gameplay_ckpt,
                           workers=workers, step_cap=step_cap, seed=seed,
                           max_games=max_games)

    pipe = LimitedPipeline(PipelineConfig(packs_path=CUBE, seed=args.seed))
    urls = [f"http://localhost:{8000 + w}" for w in range(max(1, args.workers))]

    # ---- 1. draft pods with interleaved bots --------------------------
    tasks = []     # (game_idx, deckA(iql), deckB(heu), iql_first)
    g = 0
    for k in range(args.pods):
        seed = args.seed + 17 * k
        iql_even = (k % 2 == 0)
        bots = []
        for seat in range(8):
            is_iql = (seat % 2 == 0) == iql_even
            bots.append(IQLDraftBot(checkpoint=args.draft_ckpt, seed=seed + seat)
                        if is_iql else HeuristicDraftBot(seed=seed + seat))
        pod = DraftSimulator(pipe.pool, bots, DraftPodConfig(
            seed=seed, pod_id=f"draftgate_pod{k}")).run()

        decks = {}
        for seat in range(8):
            pool = pod.drafted_pool(seat)
            hero, weapon = default_hero_assignment(seat, pod, pipe.card_classes)
            builder = HeuristicDeckBuilder(catalog=pipe.catalog,
                                           card_classes=pipe.card_classes,
                                           seed=pod.seed + seat)
            decks[seat] = builder.build_deck([hero, weapon, *pool])

        iql_seats = [s for s in range(8) if (s % 2 == 0) == iql_even]
        heu_seats = [s for s in range(8) if (s % 2 == 0) != iql_even]
        for si in iql_seats:
            for sh in heu_seats:
                for j in range(args.games_per_pair):
                    tasks.append((g, decks[si], decks[sh], j % 2 == 0))
                    g += 1
        if verbose:
            print(f"[pod {k}] drafted; IQL seats={iql_seats} heroes="
                  f"{[decks[s].hero for s in iql_seats]} | HEU heroes="
                  f"{[decks[s].hero for s in heu_seats]}")

    if args.max_games:
        tasks = tasks[:args.max_games]
    if verbose:
        print(f"[draft-gate] {len(tasks)} games across {len(urls)} workers "
              f"(pilot: {args.gameplay_ckpt}, temp=0)")

    # ---- 2. play: same frozen pilot both sides ------------------------
    pilot = lambda s: IQLGameplayBot(checkpoint=args.gameplay_ckpt, seed=s,
                                     temperature=0.0, epsilon=0.0)

    def _run_shard(url: str, shard: list) -> tuple:
        iw = hw = dr = 0
        with TalisharEnv(url, timeout=30.0) as env:
            for gi, d_iql, d_heu, iql_first in shard:
                dA, dB = (d_iql, d_heu) if iql_first else (d_heu, d_iql)
                lA = "IQLDRAFT" if iql_first else "HEUDRAFT"
                lB = "HEUDRAFT" if iql_first else "IQLDRAFT"
                p1 = Player(seat=0, deck=dA, bot_factory=pilot, label=lA)
                p2 = Player(seat=1, deck=dB, bot_factory=pilot, label=lB)
                try:
                    m = run_match(env=env, p1=p1, p2=p2,
                                  match_id=f"draftgate.g{gi}",
                                  seed=args.seed + 31 * gi, step_cap=args.step_cap)
                except Exception:  # noqa: BLE001
                    continue
                if m.winner_label == "IQLDRAFT":
                    iw += 1
                elif m.winner_label == "HEUDRAFT":
                    hw += 1
                else:
                    dr += 1
        return iw, hw, dr

    shards = [(u, tasks[i::len(urls)]) for i, u in enumerate(urls)]
    iw = hw = dr = 0
    with ThreadPoolExecutor(max_workers=len(urls)) as ex:
        for a, b, c in ex.map(lambda t: _run_shard(*t), shards):
            iw += a; hw += b; dr += c
    return iw, hw, dr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pods", type=int, default=2)
    ap.add_argument("--games-per-pair", type=int, default=2,
                    help="Games per IQL-deck x HEU-deck pairing (>=2 swaps seats).")
    ap.add_argument("--draft-ckpt", default="outputs/models/draft/latest.pt")
    ap.add_argument("--gameplay-ckpt", default="outputs/models/gameplay/latest.pt")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--step-cap", type=int, default=600)
    ap.add_argument("--seed", type=int, default=700000)
    ap.add_argument("--max-games", type=int, default=0, help="0 = no cap (smoke runs).")
    args = ap.parse_args()

    iw, hw, dr = run_gate(pods=args.pods, games_per_pair=args.games_per_pair,
                          draft_ckpt=args.draft_ckpt, gameplay_ckpt=args.gameplay_ckpt,
                          workers=args.workers, step_cap=args.step_cap,
                          seed=args.seed, max_games=args.max_games)
    n = iw + hw
    print("\n==== DRAFT GATE ====")
    print(f"IQL-drafted decks:       {iw} wins")
    print(f"Heuristic-drafted decks: {hw} wins")
    print(f"draws (ignored):         {dr}")
    if n:
        print(f"IQL draft winrate (decisive): {100*iw/n:.1f}%  (n={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
