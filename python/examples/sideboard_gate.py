"""Stage-2 sideboard GATE — does the winrate-RL sideboard model build better decks than BC?

For each sampled hero matchup we build BOTH sides' decks two ways (RL-sideboarded and
BC-sideboarded, via `sideboard.resolve`) and play them head-to-head with the SAME gameplay
model (cc_warm3) on BOTH seats — so the only thing that differs between the two decks is the
sideboard model. Across the two seat orientations RL and BC each pilot each hero once, so
hero/seat bias cancels (the same trick the gameplay gate uses).

Scoring is DECISIVE wins only — draws are dropped and never broken by life totals
(project rule: success = winning, not surviving). Verdict = does RL win more than BC.

    python -m python.examples.sideboard_gate \
        --rl outputs/models/sideboard/sideboard_rl.pt \
        --bc outputs/models/sideboard/sideboard_bc.pt \
        --gameplay-model outputs/models/cc_warm3/iql_gameplay.pt \
        --adapters 8000-8007 --matchups 50 --games 1
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from python.gameplay.cc_selfplay import (
    load_pools, _pick_variant, make_pairings, _make_bot, _parse_adapters, _REPO)
from python.deckbuilding.sideboard import resolve, cc_legal_issues
from python.deckbuilding.sideboard_model import SideboardModel
from python.gameplay.selfplay import run_selfplay_batch

_GATE_DIR = _REPO / "decks" / "_cc_gate"      # under ./decks so the adapter (ro mount) can read
_OUT_DIR = "datasets/cc_gate"                 # keep gate games OUT of the training corpus


def _build(pool: dict, opp_hero: str, model: SideboardModel, tag: str):
    """Build a legal deck file for `pool` vs `opp_hero` using `model` (argmax — we are
    EVALUATING, not exploring). Returns (repo-rel path, legality issues)."""
    ov = model.predict_overrides(pool, opp_hero)
    deck = resolve(pool, opp_hero=opp_hero, overrides=ov)
    issues = cc_legal_issues(deck)
    _GATE_DIR.mkdir(parents=True, exist_ok=True)
    out = _GATE_DIR / f"{tag}.json"
    out.write_text(json.dumps({"hero": deck["hero"], "equipment": deck["equipment"],
                               "deck": deck["deck"]}, indent=2), encoding="utf-8")
    return str(out.relative_to(_REPO)).replace("\\", "/"), issues


def run_gate(*, rl_ckpt, bc_ckpt, gameplay_model, adapters, matchups, games,
             base_seed, step_cap):
    pools = load_pools()
    heroes = sorted(pools)
    rl = SideboardModel.load(rl_ckpt)
    bc = SideboardModel.load(bc_ckpt)
    pairs = make_pairings(heroes, "random", matchups, base_seed)

    # Build every deck + play-task up front (pure-Python, ~ms each) so the parallel
    # play phase never races on deck-file writes.
    tasks = []          # (hero1, deck1, hero2, deck2, rl_seat, seed)
    skipped = 0
    for i, (hA, hB) in enumerate(pairs):
        poolA = _pick_variant(pools[hA], base_seed + i)
        poolB = _pick_variant(pools[hB], base_seed + i + 7)
        # Orientation 0: RL builds A, BC builds B  -> hA(=seat1) win counts for RL.
        a0, ia0 = _build(poolA, hB, rl, f"g{i}_rlA")
        b0, ib0 = _build(poolB, hA, bc, f"g{i}_bcB")
        # Orientation 1: BC builds A, RL builds B  -> hB(=seat2) win counts for RL.
        a1, ia1 = _build(poolA, hB, bc, f"g{i}_bcA")
        b1, ib1 = _build(poolB, hA, rl, f"g{i}_rlB")
        if ia0 or ib0:
            skipped += 1
        else:
            tasks.append((hA, a0, hB, b0, "A", base_seed + 1000 * i))
        if ia1 or ib1:
            skipped += 1
        else:
            tasks.append((hA, a1, hB, b1, "B", base_seed + 1000 * i + 500))

    urls = _parse_adapters(adapters)
    tally = {"rl": 0, "bc": 0, "draw": 0}
    lock = threading.Lock()
    done = [0]

    def _run_one(url, task):
        hA, dA, hB, dB, rl_seat, seed = task
        local = {0: 0, 1: 0, 2: 0}
        run_selfplay_batch(
            adapter_url=url, hero1=hA, hero2=hB, deck1=dA, deck2=dB,
            bot1=_make_bot("heuristic", gameplay_model, seed),
            bot2=_make_bot("heuristic", gameplay_model, seed + 999),
            n_games=games, base_seed=seed, out_dir=_OUT_DIR,
            game_format="cc", flush_every=games,
            on_game=lambda tr: local.__setitem__(int(tr.winner or 0),
                                                  local.get(int(tr.winner or 0), 0) + 1),
            step_cap=step_cap, no_progress_cap=60, life_stall_cap=0)
        with lock:
            for w, c in local.items():
                if w == 0:
                    tally["draw"] += c
                elif (w == 1 and rl_seat == "A") or (w == 2 and rl_seat == "B"):
                    tally["rl"] += c
                else:
                    tally["bc"] += c
            done[0] += 1
            print(f"[gate] {done[0]}/{len(tasks)} tasks | "
                  f"RL {tally['rl']} - {tally['bc']} BC (draws {tally['draw']})", flush=True)

    def _worker(url, my_tasks):
        for t in my_tasks:
            try:
                _run_one(url, t)
            except Exception as e:  # noqa: BLE001 — one bad task must not kill the gate
                print(f"[gate] task failed on {url}: {e!r}", flush=True)

    shards = [(u, tasks[k::len(urls)]) for k, u in enumerate(urls)]
    with ThreadPoolExecutor(max_workers=len(urls)) as ex:
        list(ex.map(lambda a: _worker(*a), shards))
    return tally, len(tasks), skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage-2 sideboard winrate gate: RL vs BC.")
    ap.add_argument("--rl", default="outputs/models/sideboard/sideboard_rl.pt")
    ap.add_argument("--bc", default="outputs/models/sideboard/sideboard_bc.pt")
    ap.add_argument("--gameplay-model", default="outputs/models/cc_warm3/iql_gameplay.pt")
    ap.add_argument("--adapters", default="8000-8007")
    ap.add_argument("--matchups", type=int, default=50)
    ap.add_argument("--games", type=int, default=1, help="games per (matchup, orientation)")
    ap.add_argument("--base-seed", type=int, default=700000)
    ap.add_argument("--step-cap", type=int, default=800)
    args = ap.parse_args(argv)

    t0 = time.time()
    tally, n_tasks, skipped = run_gate(
        rl_ckpt=args.rl, bc_ckpt=args.bc, gameplay_model=args.gameplay_model,
        adapters=args.adapters, matchups=args.matchups, games=args.games,
        base_seed=args.base_seed, step_cap=args.step_cap)
    dec = tally["rl"] + tally["bc"]
    print("\n================ SIDEBOARD GATE RESULT ================", flush=True)
    print(f"tasks played: {n_tasks} (skipped illegal: {skipped}) | "
          f"walltime {round(time.time() - t0)}s")
    print(f"RL wins:  {tally['rl']}")
    print(f"BC wins:  {tally['bc']}")
    print(f"draws:    {tally['draw']} (ignored)")
    if dec:
        wr = 100 * tally["rl"] / dec
        se = 100 * math.sqrt((wr / 100) * (1 - wr / 100) / dec)
        verdict = ("RL BETTER" if tally["rl"] > tally["bc"]
                   else ("BC BETTER" if tally["bc"] > tally["rl"] else "TIE"))
        print(f"RL decisive winrate: {wr:.1f}% +/- {se:.1f}% (of {dec} decisive games)")
        print(f"VERDICT: {verdict}")
    else:
        print("no decisive games — inconclusive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
