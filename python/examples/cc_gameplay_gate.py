"""CC gameplay GATE — does a candidate gameplay model beat the champion head-to-head?

For each sampled hero matchup we build BOTH decks ONCE (BC sideboard, the collection
standard) and play them with the CANDIDATE model on one seat and the CHAMPION on the
other — then swap seats. The decks are identical across the two orientations, so the
only thing that differs is which gameplay model pilots which hero; the seat swap cancels
hero/seat bias (the same design as the gameplay gate in continuous_train and the
sideboard gate).

Scoring is DECISIVE wins only — draws dropped, never broken by life (project rule:
success = winning, not surviving). Promote the candidate iff it wins MORE decisive games.

    python -m python.examples.cc_gameplay_gate \
        --cand outputs/models/cc_warm4/iql_gameplay.pt \
        --champ outputs/models/cc_warm3/iql_gameplay.pt \
        --sideboard outputs/models/sideboard/sideboard_bc.pt \
        --adapters 8000-8007 --matchups 60 --games 1
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from python.gameplay.cc_selfplay import (
    load_pools, _pick_variant, make_pairings, _parse_adapters, _REPO)
from python.deckbuilding.sideboard import resolve, cc_legal_issues
from python.deckbuilding.sideboard_model import SideboardModel
from python.gameplay.bots.iql_bot import IQLGameplayBot
from python.gameplay.selfplay import run_selfplay_batch

_GATE_DIR = _REPO / "decks" / "_cc_gp_gate"   # under ./decks so the ro adapter mount can read
_OUT_DIR = "datasets/cc_gp_gate"              # keep gate games OUT of the training corpus


def _build(pool: dict, opp_hero: str, sb: SideboardModel | None, tag: str):
    """Build a legal deck for `pool` vs `opp_hero` (BC sideboard argmax, or author
    data if sb is None). Returns (repo-rel path, legality issues)."""
    ov = sb.predict_overrides(pool, opp_hero) if sb is not None else None
    deck = resolve(pool, opp_hero=opp_hero, overrides=ov)
    issues = cc_legal_issues(deck)
    _GATE_DIR.mkdir(parents=True, exist_ok=True)
    out = _GATE_DIR / f"{tag}.json"
    out.write_text(json.dumps({"hero": deck["hero"], "equipment": deck["equipment"],
                               "deck": deck["deck"]}, indent=2), encoding="utf-8")
    return str(out.relative_to(_REPO)).replace("\\", "/"), issues


def run_gate(*, cand_ckpt, champ_ckpt, sb_ckpt, adapters, matchups, games,
             base_seed, step_cap):
    pools = load_pools()
    heroes = sorted(pools)
    sb = SideboardModel.load(sb_ckpt) if sb_ckpt else None
    pairs = make_pairings(heroes, "random", matchups, base_seed)

    # Build each matchup's two decks once (pure-Python), then make two play-tasks
    # (orientations) that reuse the SAME deck files — only the bots differ.
    tasks = []   # (hero1, deck1, hero2, deck2, cand_seat, seed)
    skipped = 0
    for i, (hA, hB) in enumerate(pairs):
        poolA = _pick_variant(pools[hA], base_seed + i)
        poolB = _pick_variant(pools[hB], base_seed + i + 7)
        dA, iA = _build(poolA, hB, sb, f"g{i}_A")
        dB, iB = _build(poolB, hA, sb, f"g{i}_B")
        if iA or iB:
            skipped += 1
            continue
        # Orientation 0: cand pilots seat1 (hA); champ pilots seat2 (hB).
        tasks.append((hA, dA, hB, dB, 1, base_seed + 1000 * i))
        # Orientation 1: champ pilots seat1 (hA); cand pilots seat2 (hB).
        tasks.append((hA, dA, hB, dB, 2, base_seed + 1000 * i + 500))

    urls = _parse_adapters(adapters)
    tally = {"cand": 0, "champ": 0, "draw": 0}
    lock = threading.Lock()
    done = [0]

    def _bot(ckpt, seed):
        return IQLGameplayBot(checkpoint=ckpt, seed=seed, temperature=0.0, epsilon=0.0)

    def _run_one(url, task):
        hA, dA, hB, dB, cand_seat, seed = task
        # cand_seat says which seat the candidate plays this orientation.
        if cand_seat == 1:
            bot1, bot2 = _bot(cand_ckpt, seed), _bot(champ_ckpt, seed + 999)
        else:
            bot1, bot2 = _bot(champ_ckpt, seed), _bot(cand_ckpt, seed + 999)
        local = {0: 0, 1: 0, 2: 0}
        run_selfplay_batch(
            adapter_url=url, hero1=hA, hero2=hB, deck1=dA, deck2=dB,
            bot1=bot1, bot2=bot2, n_games=games, base_seed=seed, out_dir=_OUT_DIR,
            game_format="cc", flush_every=games,
            on_game=lambda tr: local.__setitem__(int(tr.winner or 0),
                                                 local.get(int(tr.winner or 0), 0) + 1),
            step_cap=step_cap, no_progress_cap=60, life_stall_cap=0)
        with lock:
            for w, c in local.items():
                if w == 0:
                    tally["draw"] += c
                elif w == cand_seat:
                    tally["cand"] += c
                else:
                    tally["champ"] += c
            done[0] += 1
            print(f"[gp-gate] {done[0]}/{len(tasks)} tasks | "
                  f"CAND {tally['cand']} - {tally['champ']} CHAMP "
                  f"(draws {tally['draw']})", flush=True)

    def _worker(url, my_tasks):
        for t in my_tasks:
            try:
                _run_one(url, t)
            except Exception as e:  # noqa: BLE001 — one bad task must not kill the gate
                print(f"[gp-gate] task failed on {url}: {e!r}", flush=True)

    shards = [(u, tasks[k::len(urls)]) for k, u in enumerate(urls)]
    with ThreadPoolExecutor(max_workers=len(urls)) as ex:
        list(ex.map(lambda a: _worker(*a), shards))
    return tally, len(tasks), skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description="CC gameplay gate: candidate vs champion.")
    ap.add_argument("--cand", default="outputs/models/cc_warm4/iql_gameplay.pt")
    ap.add_argument("--champ", default="outputs/models/cc_warm3/iql_gameplay.pt")
    ap.add_argument("--sideboard", default="outputs/models/sideboard/sideboard_bc.pt",
                    help="sideboard model for deck construction (both sides); '' = author data")
    ap.add_argument("--adapters", default="8000-8007")
    ap.add_argument("--matchups", type=int, default=60)
    ap.add_argument("--games", type=int, default=1, help="games per (matchup, orientation)")
    ap.add_argument("--base-seed", type=int, default=600000)
    ap.add_argument("--step-cap", type=int, default=800)
    args = ap.parse_args(argv)

    t0 = time.time()
    tally, n_tasks, skipped = run_gate(
        cand_ckpt=args.cand, champ_ckpt=args.champ, sb_ckpt=args.sideboard or None,
        adapters=args.adapters, matchups=args.matchups, games=args.games,
        base_seed=args.base_seed, step_cap=args.step_cap)
    dec = tally["cand"] + tally["champ"]
    print("\n================ CC GAMEPLAY GATE RESULT ================", flush=True)
    print(f"candidate: {args.cand}")
    print(f"champion:  {args.champ}")
    print(f"tasks played: {n_tasks} (skipped illegal: {skipped}) | "
          f"walltime {round(time.time() - t0)}s")
    print(f"CAND wins:  {tally['cand']}")
    print(f"CHAMP wins: {tally['champ']}")
    print(f"draws:      {tally['draw']} (ignored)")
    if dec:
        wr = 100 * tally["cand"] / dec
        se = 100 * math.sqrt((wr / 100) * (1 - wr / 100) / dec)
        verdict = ("PROMOTE cand" if tally["cand"] > tally["champ"]
                   else ("KEEP champ" if tally["champ"] > tally["cand"] else "TIE — keep champ"))
        print(f"CAND decisive winrate: {wr:.1f}% +/- {se:.1f}% (of {dec} decisive games)")
        print(f"VERDICT: {verdict}")
    else:
        print("no decisive games — inconclusive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
