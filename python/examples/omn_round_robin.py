"""Full OMN pipeline with a round-robin tournament.

Drives the real orchestrator end-to-end against the live Talishar adapter:

    8 draft bots draft from N packs  ->  build 8 decks  ->  round-robin
    (every deck pair plays >=`games_per_pair`; tied series go to win-by-2)
    ->  persist draft / game / tournament parquet  ->  (train separately)

Usage::

    # small validation
    python -m python.examples.omn_round_robin --players 4 --games-per-pair 4
    # full run
    python -m python.examples.omn_round_robin --players 8 --games-per-pair 10

Requires the adapter up in ADAPTER_MODE=real on :8000.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.draft.draftmancer import load_pack_pool_draftmancer, parse_draftmancer  # noqa: E402
from python.gameplay.env import wait_for_adapter  # noqa: E402
from python.pipeline import LimitedPipeline, PipelineConfig  # noqa: E402

CUBE = PROJECT_ROOT / "OMN_Draft_3.5.txt"
ADAPTER = "http://localhost:8000"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=8)
    ap.add_argument("--packs-per-player", type=int, default=3)
    ap.add_argument("--games-per-pair", type=int, default=10)
    ap.add_argument("--win-by", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260530)
    ap.add_argument("--step-cap", type=int, default=1200)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--gameplay-ckpt", default=None,
                    help="Trained IQL gameplay checkpoint; play with it instead of AggroBot.")
    ap.add_argument("--draft-ckpt", default=None,
                    help="Trained IQL draft checkpoint; draft with it instead of the default mix.")
    args = ap.parse_args()

    h = wait_for_adapter(ADAPTER, timeout_s=30.0)
    if h.get("mode") != "real":
        print(f"FATAL: adapter mode={h.get('mode')!r}; set ADAPTER_MODE=real")
        return 2
    print(f"[rr] adapter ok: {h}")

    n_packs = args.players * args.packs_per_player
    cube = parse_draftmancer(str(CUBE))
    print(f"[rr] cube universe = {len(cube.card_universe())} cards; building {n_packs} packs")

    cfg = PipelineConfig(
        adapter_url=ADAPTER,
        packs_path=str(CUBE),
        out_dir=args.out,
        seed=args.seed,
        n_pods=1,
        n_players=args.players,
        packs_per_player=args.packs_per_player,
        tournament_mode="round_robin",
        games_per_pair=args.games_per_pair,
        win_by=args.win_by,
        rr_max_extra_games=20,
        step_cap=args.step_cap,
        cube=cube,
        pack_pool_factory=lambda: load_pack_pool_draftmancer(
            str(CUBE), n_packs=n_packs, seed=args.seed),
    )

    # Close the improvement loop: if trained checkpoints are supplied,
    # draft + play with the learned IQL policies instead of the defaults.
    if args.gameplay_ckpt:
        from python.gameplay.bots.iql_bot import IQLGameplayBot
        ck = args.gameplay_ckpt
        cfg.gameplay_bot_factory = (
            lambda player_seed: (lambda game_seed: IQLGameplayBot(checkpoint=ck, seed=game_seed)))
        print(f"[rr] gameplay bot: IQL @ {ck}")
    if args.draft_ckpt:
        from python.draft.bots.iql_bot import IQLDraftBot
        dk = args.draft_ckpt
        cfg.draft_bot_factory = lambda seat, seed: IQLDraftBot(checkpoint=dk, seed=seed + seat)
        print(f"[rr] draft bot: IQL @ {dk}")

    t0 = time.time()
    result = LimitedPipeline(cfg).run_cycle()
    dt = time.time() - t0

    print(f"\n[rr] cycle {result.cycle_id} done in {dt:.1f}s")
    for tour in result.tournaments:
        champ = tour.champion()
        print(f"\n=== {tour.tournament_id} ===")
        print(tour.render())
        print(f"champion: {champ.name if champ else '(none)'}")
        n_games = len(tour.matches)
        n_decisive = sum(1 for m in tour.matches if m.winner_label)
        n_tb = sum(1 for m in tour.matches if m.metadata.get("tiebreak"))
        print(f"games={n_games} decisive={n_decisive} life-tiebreaks={n_tb}")
    print("\nartefacts:")
    for kind, paths in result.artefacts.items():
        for p in paths:
            print(f"  {kind}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
