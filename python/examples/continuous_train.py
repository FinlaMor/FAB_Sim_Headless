"""Autonomous self-improvement loop: gen -> train -> promote -> repeat, forever.

Wraps ``self_play_loop``'s per-iteration machinery in a resilient outer loop
so the bots keep getting better unattended. Each iteration:

  * warm-starts from the freshly promoted ``models/{role}/latest.pt`` (the
    improvement compounds),
  * generates a fresh round-robin with the aggressive explorer + tempo reward,
  * trains on a rolling RECENCY WINDOW (``--window``) so the policy learns from
    recent strong play instead of being diluted by ancient weak-bot games,
  * snapshots each promoted model to ``models/{role}/snapshots/iter_{n}.pt``
    (rollback + a record of progress),
  * appends per-iteration metrics (walltime, champion, lethal/tiebreak/draw %,
    decisions kept) to ``outputs/continuous_metrics.jsonl`` so we can watch the
    trend,
  * survives transient failures (adapter hiccup, one bad iteration): it logs
    the traceback, waits for the adapters to come back, and continues.

Run (after a normal warm-started run has produced models):

    python -u -m python.examples.continuous_train --workers 8 --window 6 \
        --explorer aggressive --aggression-weight 0.3

Stop it with Ctrl-C (or kill the process); state is on disk, so re-launching
resumes from the next iteration.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.draft.draftmancer import parse_draftmancer  # noqa: E402
from python.gameplay.env import wait_for_adapter  # noqa: E402
from python.examples.self_play_loop import (  # noqa: E402
    run_one_cycle, _existing_latest, CUBE, ADAPTER,
)
from python.training import iql_gameplay, iql_draft  # noqa: E402
from python.models import registry  # noqa: E402

METRICS = "continuous_metrics.jsonl"


def _finish_stats(out: str) -> dict:
    """Lethal / tiebreak / draw rates for the newest games parquet."""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(Path(out) / "parquet" / "games" / "*.parquet")),
                   key=os.path.getmtime)
    if not files:
        return {}
    t = pq.read_table(files[-1], columns=["game_id", "winner", "step_index",
                                          "next_state_json"]).to_pylist()
    last: dict = {}
    for r in t:
        g = r["game_id"]
        if g not in last or r["step_index"] > last[g]["step_index"]:
            last[g] = r
    lethal = tiebreak = draw = 0
    for r in last.values():
        w = int(r["winner"] or 0)
        ns = json.loads(r["next_state_json"])
        hps = [int(p.get("health") or 0) for p in ns.get("players", [])]
        if w == 0:
            draw += 1
        elif hps and min(hps) <= 0:
            lethal += 1
        else:
            tiebreak += 1
    n = len(last) or 1
    stats = {"games": len(last), "lethal_pct": round(100 * lethal / n, 1),
             "tiebreak_pct": round(100 * tiebreak / n, 1),
             "draw_pct": round(100 * draw / n, 1),
             "decisions_recorded": len(t)}
    stats.update(_abort_stats(out))
    return stats


def _abort_stats(out: str) -> dict:
    """Split the winner-0 'draws' into real step-cap stalemates vs engine/adapter
    ABORTS, using the per-match ledger (parquet/matches/, written by the
    pipeline). Without this the games parquet makes a turn-0 abort and a genuine
    400-step draw look identical, which silently starved the promotion gate.
    Returns {} if no matches parquet exists yet (older runs)."""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(Path(out) / "parquet" / "matches" / "*.parquet")),
                   key=os.path.getmtime)
    if not files:
        return {}
    rows = pq.read_table(files[-1], columns=["term_reason"]).to_pylist()
    n = len(rows) or 1
    counts: dict = {}
    for r in rows:
        counts[r["term_reason"] or "unknown"] = counts.get(r["term_reason"] or "unknown", 0) + 1
    # Anything that isn't a real engine result or a clean step-cap draw is an abort.
    abort = sum(c for k, c in counts.items() if k not in ("engine_winner", "step_cap"))
    return {"matches_logged": len(rows),
            "abort_pct": round(100 * abort / n, 1),
            "stepcap_draw_pct": round(100 * counts.get("step_cap", 0) / n, 1),
            "term_reasons": counts}


def _snapshot(out: str, it: int) -> None:
    for role in ("gameplay", "draft"):
        src = Path(out) / "models" / role / "latest.pt"
        if src.exists():
            dst = Path(out) / "models" / role / "snapshots" / f"iter_{it:04d}.pt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _next_iter(out: str) -> int:
    """Resume: continue after the highest existing snapshot iteration."""
    snaps = glob.glob(str(Path(out) / "models" / "gameplay" / "snapshots" / "iter_*.pt"))
    best = -1
    for s in snaps:
        try:
            best = max(best, int(Path(s).stem.split("_")[1]))
        except (ValueError, IndexError):
            pass
    return best + 1


def _load_deck(path: str):
    from python.deckbuilding.deck import Deck
    d = json.load(open(path, encoding="utf-8"))
    eq = d.get("equipment", []) or []
    return Deck(hero=d["hero"], weapon=(eq[0] if eq else ""),
                deck=list(d.get("deck", [])), equipment=list(eq[1:]))


def _benchmark_pair(out: str):
    """One FIXED deck pair for the regression gate — snapshot once, reuse, so
    every iteration's candidate is judged on the same yardstick."""
    bdir = Path(out) / "benchmark_decks"
    a, b = bdir / "bench_p1.json", bdir / "bench_p2.json"
    if a.is_file() and b.is_file():
        return str(a), str(b)
    bdir.mkdir(parents=True, exist_ok=True)
    src = sorted(Path("decks/_tmp_matches").glob("*_p1.json"),
                 key=os.path.getmtime, reverse=True)
    for p1 in src:
        p2 = Path(str(p1).replace("_p1.json", "_p2.json"))
        if p2.exists():
            shutil.copy2(p1, a); shutil.copy2(p2, b)
            return str(a), str(b)
    return None, None


def _gate_pool(out: str) -> list[str]:
    """Top human-drafted decks for the gate gauntlet (outputs/gate_decks/,
    written by python.examples.select_gate_decks). A fixed pool = a stable
    yardstick across iterations, but far more varied than one synthetic pair."""
    gdir = Path(out) / "gate_decks"
    return sorted(str(p) for p in gdir.glob("*.json"))


def _gate(cand_ckpt: str, champ_ckpt: str, urls: list[str], out: str,
          decisive_target: int = 100, step_cap: int = 400, max_games: int = 600):
    """Gauntlet: candidate vs champion across the top human-drafted decks. Each
    deck matchup is played BOTH ways (the two models swap seats on the SAME
    decks) so deck and seat bias cancel and only PLAY skill is measured. ONLY
    decisive (lethal/engine) wins count — draws are ignored and never broken by
    life totals. Games are SHARDED across all adapter workers.

    Plays in ROUNDS until `decisive_target` DECISIVE games are reached (or the
    `max_games` safety cap). A fixed 200-game gate left only ~60-130 decisive
    games once draws climbed (SE ~6% — promotions were coin flips); targeting a
    fixed count of decisive games keeps the gate's statistical power constant
    regardless of draw rate. Returns (cand_wins, champ_wins, draws)."""
    from concurrent.futures import ThreadPoolExecutor
    from python.gameplay.env import TalisharEnv
    from python.gameplay.bots.iql_bot import IQLGameplayBot
    from python.tournament.player import Player
    from python.tournament.match import run_match

    pool = _gate_pool(out)
    if len(pool) < 2:                       # fall back to the legacy fixed pair
        pa, pb = _benchmark_pair(out)
        if not pa:
            return 0, 0, 0
        pool = [pa, pb]
    decks = [_load_deck(p) for p in pool]
    cand = lambda s: IQLGameplayBot(checkpoint=cand_ckpt, seed=s,
                                    temperature=0.0, epsilon=0.0)
    champ = lambda s: IQLGameplayBot(checkpoint=champ_ckpt, seed=s,
                                     temperature=0.0, epsilon=0.0)

    def _make_task(g: int):
        """Deterministic game g: pair (g//2) plays decks (2i, 2i+1), the two
        orientations on consecutive g so seat bias cancels."""
        i = g // 2
        dA = decks[(2 * i) % len(decks)]
        dB = decks[(2 * i + 1) % len(decks)]
        return (g, dA, dB, (g % 2 == 0))

    def _run_shard(url: str, shard: list) -> tuple:
        scw = shw = sdr = 0
        with TalisharEnv(url, timeout=30.0) as env:
            for gi, dA, dB, cand_first in shard:
                f0, f1 = (cand, champ) if cand_first else (champ, cand)
                p1 = Player(seat=0, deck=dA, bot_factory=f0,
                            label="CAND" if cand_first else "CHAMP")
                p2 = Player(seat=1, deck=dB, bot_factory=f1,
                            label="CHAMP" if cand_first else "CAND")
                try:
                    m = run_match(env=env, p1=p1, p2=p2,
                                  match_id=f"gate.g{gi}", seed=900000 + gi,
                                  step_cap=step_cap)
                except Exception:  # noqa: BLE001 — a bad gate game shouldn't crash the loop
                    continue
                if m.winner_label == "CAND":
                    scw += 1
                elif m.winner_label == "CHAMP":
                    shw += 1
                else:
                    sdr += 1
        return scw, shw, sdr

    cw = hw = dr = 0
    g = 0
    with ThreadPoolExecutor(max_workers=len(urls)) as ex:
        while (cw + hw) < decisive_target and g < max_games:
            # Size the next round to close the decisive gap, inflating by the
            # observed draw rate so far (default 2x) so we don't under-shoot.
            need = decisive_target - (cw + hw)
            draw_rate = dr / max(1, g)
            batch = min(int(need / max(0.15, 1.0 - draw_rate)) + len(urls),
                        max_games - g)
            tasks = [_make_task(g + k) for k in range(batch)]
            g += batch
            shards = [(u, tasks[k::len(urls)]) for k, u in enumerate(urls)]
            for scw, shw, sdr in ex.map(lambda a: _run_shard(*a), shards):
                cw += scw; hw += shw; dr += sdr
    return cw, hw, dr


def _train_and_promote(args, out: str, urls: list[str]) -> tuple:
    """Train candidate gameplay + draft models. The gameplay model is promoted
    to latest.pt ONLY if it beats the current champion head-to-head (regression
    gate); otherwise the champion is kept. Draft promotes normally. Returns
    (gp_latest|None, dr_latest|None, gate_dict|None)."""
    staging = Path(out) / "models" / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    gp_latest = Path(out) / "models" / "gameplay" / "latest.pt"
    dr_latest = Path(out) / "models" / "draft" / "latest.pt"
    gate = None

    if list((Path(out) / "parquet" / "games").glob("*.parquet")):
        cand = iql_gameplay.train(
            parquet_dir=Path(out) / "parquet" / "games", out_dir=staging,
            hyper=iql_gameplay.IQLHyperparams(
                n_steps=args.gameplay_steps, window=args.window,
                use_shaped_reward=not args.no_shaped_reward,
                aggression_weight=args.aggression_weight,
                draw_penalty=args.draw_penalty,
                time_penalty=args.time_penalty))
        if gp_latest.exists():
            cw, hw, dr = _gate(str(cand), str(gp_latest), urls, out,
                               decisive_target=args.gate_decisive,
                               step_cap=args.step_cap)
            promoted = cw > hw            # decisive-win majority; ties keep champ
            gate = {"cand_wins": cw, "champ_wins": hw, "draws": dr,
                    "promoted": promoted}
            if promoted:
                registry.save_checkpoint(cand, root=out, role="gameplay")
                shutil.copy2(cand, gp_latest)
        else:
            gp_latest.parent.mkdir(parents=True, exist_ok=True)
            registry.save_checkpoint(cand, root=out, role="gameplay")
            shutil.copy2(cand, gp_latest)
            gate = {"cand_wins": 0, "champ_wins": 0, "draws": 0,
                    "promoted": True, "cold_start": True}

    if list((Path(out) / "parquet" / "drafts").glob("*.parquet")):
        ck = iql_draft.train(
            parquet_dir=Path(out) / "parquet" / "drafts", out_dir=staging,
            decks_dir=Path(out) / "parquet" / "decks",
            matches_dir=Path(out) / "parquet" / "matches",
            hyper=iql_draft.DraftIQLHyperparams(
                n_steps=args.draft_steps,
                # extra draft-only pods double the parquet FILES per
                # iteration; window counts files, so scale it to keep the
                # same number of ITERATIONS in the draft window.
                window=getattr(args, "draft_window", args.window)))
        dr_latest.parent.mkdir(parents=True, exist_ok=True)
        registry.save_checkpoint(ck, root=out, role="draft")
        shutil.copy2(ck, dr_latest)

    return (str(gp_latest) if gp_latest.exists() else None,
            str(dr_latest) if dr_latest.exists() else None, gate)


_extra_pool_cache = [None]


def _gen_extra_draft_pods(out: str, dr_ckpt: str | None, it: int,
                          n_pods: int, temp: float) -> None:
    """Draft-only data pods: drafting is pure Python (~seconds/pod), so each
    iteration can multiply DRAFT data without playing any games. Written with
    placement=0 (neutral placement reward) — the deck-quality auxiliary
    terminal still grades the pools, and the picks feed AWR/BC diversity."""
    if not n_pods or not dr_ckpt:
        return
    try:
        from python.pipeline import LimitedPipeline, PipelineConfig
        from python.draft.simulator import DraftSimulator, DraftPodConfig
        from python.draft.bots.iql_bot import IQLDraftBot
        from python.draft.dataset import DraftDatasetWriter
        if _extra_pool_cache[0] is None:
            _extra_pool_cache[0] = LimitedPipeline(
                PipelineConfig(packs_path=str(CUBE), seed=0)).pool
        pool = _extra_pool_cache[0]
        pods = []
        for k in range(n_pods):
            seed = 800_000 + 1000 * it + k
            bots = [IQLDraftBot(checkpoint=dr_ckpt, seed=seed + s, temperature=temp)
                    for s in range(8)]
            pods.append(DraftSimulator(pool, bots, DraftPodConfig(
                seed=seed, pod_id=f"xdraft{it:04d}_{k}")).run())
        DraftDatasetWriter(out).write_pods(pods)
        print(f"  [extra-drafts] wrote {n_pods} draft-only pods "
              f"({n_pods * 8 * 42} picks)", flush=True)
    except Exception as e:  # noqa: BLE001 — bonus data must never kill the loop
        print(f"  [extra-drafts] failed: {e!r}", flush=True)


def _adapters_healthy(workers: int, settle_s: float = 300.0) -> bool:
    """Block until adapter :8000 answers (others share the image)."""
    try:
        h = wait_for_adapter(ADAPTER, timeout_s=settle_s)
        return h.get("mode") == "real"
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=8)
    ap.add_argument("--packs-per-player", type=int, default=3)
    ap.add_argument("--games-per-pair", type=int, default=10)
    ap.add_argument("--win-by", type=int, default=2)
    ap.add_argument("--step-cap", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--gameplay-steps", type=int, default=12000)
    ap.add_argument("--draft-steps", type=int, default=3000)
    ap.add_argument("--gate-decisive", type=int, default=100,
                    help="Promotion gate plays in rounds until this many "
                         "DECISIVE games (draws excluded), sharded across "
                         "workers — keeps statistical power constant as the "
                         "draw rate varies. ~SE 5%% at 100 decisive.")
    # Recency window: train on the last N game/draft parquets so the policy
    # tracks the current (stronger) generation rather than averaging in weak
    # early data. 0 = all (not recommended for a long-running loop).
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--draft-temperature", type=float, default=0.35,
                    help="Pick-sampling temperature for DRAFT data collection "
                         "(argmax drafts were near-clones every cycle).")
    ap.add_argument("--extra-draft-pods", type=int, default=4,
                    help="Draft-only data pods per iteration (no games; "
                         "deck-quality reward only).")
    ap.add_argument("--draft-gate-every", type=int, default=5,
                    help="Run the IQL-vs-heuristic draft gate every N "
                         "iterations (0 = off; informational).")
    ap.add_argument("--draft-gate-games", type=int, default=32)
    ap.add_argument("--block-prob", type=float, default=0.5)
    ap.add_argument("--explorer", choices=["balanced", "aggressive"], default="aggressive")
    ap.add_argument("--aggression-weight", type=float, default=0.3)
    ap.add_argument("--draw-penalty", type=float, default=0.0,
                    help="Terminal penalty for a step-cap draw (makes draws < neutral).")
    ap.add_argument("--time-penalty", type=float, default=0.0,
                    help="Per-decision living cost on all games (rewards faster wins).")
    ap.add_argument("--no-shaped-reward", action="store_true")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--max-iters", type=int, default=0, help="0 = run forever.")
    cli = ap.parse_args()

    # self_play_loop's helpers read a flat args object; mirror it.
    args = SimpleNamespace(**vars(cli))
    # Extra draft-only pods add a second drafts parquet per iteration;
    # double the draft window (file-counted) to span the same iterations.
    args.draft_window = cli.window * (2 if cli.extra_draft_pods else 1)

    if not _adapters_healthy(cli.workers):
        print("FATAL: adapter not up in mode=real on :8000")
        return 2
    cube = parse_draftmancer(str(CUBE))
    urls = [f"http://localhost:{8000 + w}" for w in range(max(1, cli.workers))]

    it = _next_iter(cli.out)
    metrics_path = Path(cli.out) / METRICS
    print(f"[continuous] starting at iteration {it}; window={cli.window} "
          f"explorer={cli.explorer} aggression_weight={cli.aggression_weight}")

    done = 0
    while cli.max_iters == 0 or done < cli.max_iters:
        t0 = time.time()
        print(f"\n================ CONTINUOUS ITER {it} ================", flush=True)
        try:
            # Always warm-start from the latest promoted models.
            gp_ckpt, dr_ckpt = _existing_latest(Path(cli.out))
            # Each iteration gets a distinct seed for fresh packs/games.
            args.seed = cli.seed + it
            res = run_one_cycle(args, 0, cube, gp_ckpt, dr_ckpt)
            champ = None
            try:
                champ = res.tournaments[0].champion()
                champ = champ.name if champ else None
            except Exception:  # noqa: BLE001
                pass

            # Cheap draft-data multiplier (uses the generation checkpoint).
            _gen_extra_draft_pods(cli.out, dr_ckpt, it,
                                  cli.extra_draft_pods, cli.draft_temperature)

            gp_ckpt, dr_ckpt, gate = _train_and_promote(args, cli.out, urls)
            _snapshot(cli.out, it)

            # Draft skill proxy: human pick agreement on a fixed subset of
            # the real reference drafts (CPU-only, ~1 min). The draft model
            # has no promotion gate, so this is its per-iteration health
            # metric — selfplay-only training sat at random (~25% top-1).
            draft_agree = {}
            if dr_ckpt:
                try:
                    from python.examples.draft_agreement import compute_agreement
                    draft_agree = compute_agreement(dr_ckpt, max_drafts=40)
                except Exception as e:  # noqa: BLE001 — metrics must not kill the loop
                    print(f"[continuous] draft agreement failed: {e!r}")

            # Periodic informational draft gate (IQL vs heuristic drafting,
            # frozen pilot both sides) — the draft model's no-gate blind spot.
            draft_gate_res = None
            if (cli.draft_gate_every and dr_ckpt and gp_ckpt
                    and it % cli.draft_gate_every == 0):
                try:
                    from python.examples.draft_gate import run_gate
                    iw, hw, drg = run_gate(
                        pods=1, games_per_pair=2, draft_ckpt=dr_ckpt,
                        gameplay_ckpt=gp_ckpt, workers=cli.workers,
                        step_cap=args.step_cap, seed=910_000 + it,
                        max_games=cli.draft_gate_games, verbose=False)
                    draft_gate_res = {"iql": iw, "heu": hw, "draws": drg}
                    print(f"  [draft-gate] IQL {iw} - {hw} HEU "
                          f"(draws {drg})", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[continuous] draft gate failed: {e!r}")

            row = {"iter": it, "ts": time.time(),
                   "walltime_s": round(time.time() - t0),
                   "champion": champ, "gate": gate,
                   "draft_agreement": draft_agree,
                   "draft_gate": draft_gate_res, **_finish_stats(cli.out)}
            with open(metrics_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            gtxt = ("cold-start promote" if gate and gate.get("cold_start")
                    else (f"PROMOTED {gate['cand_wins']}-{gate['champ_wins']}"
                          if gate and gate.get("promoted")
                          else (f"kept champ {gate['cand_wins']}-{gate['champ_wins']}"
                                if gate else "n/a")))
            print(f"[continuous] iter {it} OK in {row['walltime_s']}s | gate: {gtxt} | "
                  f"champ={champ} | lethal={row.get('lethal_pct')}% "
                  f"draw={row.get('draw_pct')}% | "
                  f"draft_top1={draft_agree.get('top1_pct', 'n/a')}%", flush=True)
            done += 1
            it += 1
        except KeyboardInterrupt:
            print("[continuous] interrupted; stopping.")
            return 0
        except Exception:  # noqa: BLE001 — one bad iteration must not stop the loop
            print(f"[continuous] iter {it} FAILED:\n{traceback.format_exc()}", flush=True)
            # Wait for the adapters to recover before trying again.
            if not _adapters_healthy(cli.workers, settle_s=600.0):
                print("[continuous] adapters still down; waiting 60s.", flush=True)
                time.sleep(60)
            it += 1  # move on so a permanently-poisoned seed can't wedge us
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
