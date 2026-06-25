"""End-to-end self-improvement loop.

One iteration =
    8 draft bots draft from N packs
    -> build 8 decks
    -> round-robin (every pair >= games_per_pair; ties go to win-by-2)
    -> persist draft/game/tournament parquet
    -> train IQL draft + gameplay policies on the accumulated parquet
    -> next iteration drafts & plays with those trained policies.

    python -m python.examples.self_play_loop --iters 2 --players 8 --games-per-pair 10

Iteration 0 uses the default draft mix + AggroBot (decisive baseline). Each
later iteration loads the previous iteration's ``models/{draft,gameplay}/
latest.pt`` so the bots actually improve from the data they generated.

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
from python.training import iql_gameplay, iql_draft  # noqa: E402
from python.models import registry  # noqa: E402

CUBE = PROJECT_ROOT / "OMN_Draft_3.5.txt"
ADAPTER = "http://localhost:8000"


def run_one_cycle(args, it: int, cube, gp_ckpt: str | None, dr_ckpt: str | None):
    n_packs = args.players * args.packs_per_player
    urls = [f"http://localhost:{8000 + w}" for w in range(max(1, args.workers))]
    cfg = PipelineConfig(
        adapter_url=urls[0], adapter_urls=urls,
        packs_path=str(CUBE), out_dir=args.out,
        seed=args.seed + it, n_pods=1,
        n_players=args.players, packs_per_player=args.packs_per_player,
        tournament_mode="round_robin", games_per_pair=args.games_per_pair,
        # With the life tiebreak gone, step-cap games are draws, so tied pairs
        # are common; cap extra win-by-2 games low so a stalemated pair settles
        # as a draw instead of grinding 20 extra games and ballooning the loop.
        win_by=args.win_by, rr_max_extra_games=6, step_cap=args.step_cap,
        cube=cube,
        pack_pool_factory=lambda: load_pack_pool_draftmancer(
            str(CUBE), n_packs=n_packs, seed=args.seed + it),
    )
    if len(urls) > 1:
        print(f"  parallel workers: {len(urls)} -> {urls}")
    if gp_ckpt:
        from python.gameplay.bots.iql_bot import IQLGameplayBot
        eps, temp, bp, expl = args.epsilon, args.temperature, args.block_prob, args.explorer
        cfg.gameplay_bot_factory = (
            lambda ps: (lambda gs: IQLGameplayBot(
                checkpoint=gp_ckpt, seed=gs, epsilon=eps, temperature=temp,
                block_prob=bp, explorer=expl)))
        print(f"  gameplay bot: IQL @ {gp_ckpt} (epsilon={eps} temp={temp} explorer={expl})")
    if dr_ckpt:
        from python.draft.bots.iql_bot import IQLDraftBot
        dtemp = getattr(args, "draft_temperature", 0.35)
        cfg.draft_bot_factory = lambda seat, seed: IQLDraftBot(
            checkpoint=dr_ckpt, seed=seed + seat, temperature=dtemp)
        print(f"  draft bot: IQL @ {dr_ckpt} (temp={dtemp})")

    result = LimitedPipeline(cfg).run_cycle()
    for tour in result.tournaments:
        print(tour.render())
        champ = tour.champion()
        print(f"  champion: {champ.name if champ else '(none)'}")
    return result


def train(args) -> tuple[str | None, str | None]:
    import shutil
    out = Path(args.out)
    staging = out / "models" / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    gp = dr = None
    if list((out / "parquet" / "games").glob("*.parquet")):
        ck = iql_gameplay.train(
            parquet_dir=out / "parquet" / "games", out_dir=staging,
            hyper=iql_gameplay.IQLHyperparams(
                n_steps=args.gameplay_steps, window=args.window,
                use_shaped_reward=not args.no_shaped_reward,
                aggression_weight=args.aggression_weight))
        registry.save_checkpoint(ck, root=out, role="gameplay")
        gp = str(out / "models" / "gameplay" / "latest.pt")
        shutil.copy2(ck, gp)
    if list((out / "parquet" / "drafts").glob("*.parquet")):
        ck = iql_draft.train(
            parquet_dir=out / "parquet" / "drafts", out_dir=staging,
            hyper=iql_draft.DraftIQLHyperparams(n_steps=args.draft_steps, window=args.window))
        registry.save_checkpoint(ck, root=out, role="draft")
        dr = str(out / "models" / "draft" / "latest.pt")
        shutil.copy2(ck, dr)
    return gp, dr


def _existing_latest(out: Path) -> tuple[str | None, str | None]:
    gp = out / "models" / "gameplay" / "latest.pt"
    dr = out / "models" / "draft" / "latest.pt"
    return (str(gp) if gp.exists() else None, str(dr) if dr.exists() else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--players", type=int, default=8)
    ap.add_argument("--packs-per-player", type=int, default=3)
    ap.add_argument("--games-per-pair", type=int, default=10)
    ap.add_argument("--win-by", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260530)
    ap.add_argument("--step-cap", type=int, default=400)
    ap.add_argument("--gameplay-steps", type=int, default=4000)
    ap.add_argument("--draft-steps", type=int, default=3000)
    ap.add_argument("--window", type=int, default=0,
                    help="Train on the most recent N cycles' parquet; 0 = all.")
    ap.add_argument("--no-shaped-reward", action="store_true")
    ap.add_argument("--epsilon", type=float, default=0.15,
                    help="Exploration: prob the gameplay bot defers to BalancedBot.")
    ap.add_argument("--temperature", type=float, default=0.5,
                    help="Softmax temperature when sampling the IQL policy (0 = argmax).")
    ap.add_argument("--block-prob", type=float, default=0.5,
                    help="BalancedBot block probability in defence phases.")
    ap.add_argument("--explorer", choices=["balanced", "aggressive"], default="balanced",
                    help="Exploration policy the IQL gameplay bot defers to. "
                         "'aggressive' almost never passes when it can act.")
    ap.add_argument("--aggression-weight", type=float, default=0.0,
                    help="Dense reward per step for increasing the mover's life lead "
                         "(potential-based tempo/damage shaping). 0 = terminal reward only.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel adapter workers (expects them on localhost:8000..800N-1).")
    ap.add_argument("--warm-start", action="store_true",
                    help="Train (or load latest.pt) BEFORE iteration 0 so the loop "
                         "drafts/plays with trained models from the start.")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    h = wait_for_adapter(ADAPTER, timeout_s=30.0)
    if h.get("mode") != "real":
        print(f"FATAL: adapter mode={h.get('mode')!r}; set ADAPTER_MODE=real")
        return 2
    cube = parse_draftmancer(str(CUBE))

    gp_ckpt = dr_ckpt = None
    if args.warm_start:
        # Seed from already-trained models; if none exist but parquet does,
        # train once up front. Either way, iteration 0 plays with models.
        gp_ckpt, dr_ckpt = _existing_latest(Path(args.out))
        if gp_ckpt or dr_ckpt:
            print(f"[warm-start] using existing models: gameplay={gp_ckpt} draft={dr_ckpt}")
        else:
            print("[warm-start] no existing models; training on existing parquet first")
            gp_ckpt, dr_ckpt = train(args)
            print(f"[warm-start] trained: gameplay={gp_ckpt} draft={dr_ckpt}")

    for it in range(args.iters):
        print(f"\n############ ITERATION {it} ############")
        t0 = time.time()
        run_one_cycle(args, it, cube, gp_ckpt, dr_ckpt)
        print(f"\n--- training (window={args.window or 'all'}) after iter {it} ---")
        gp_ckpt, dr_ckpt = train(args)
        print(f"iteration {it} done in {time.time()-t0:.0f}s; "
              f"gameplay={gp_ckpt} draft={dr_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
