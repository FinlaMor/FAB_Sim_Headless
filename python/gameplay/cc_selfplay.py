"""Classic Constructed self-play orchestration.

Pairs the scraped CC heroes, resolves each side's pool into a legal game deck
*for that specific opponent* (matchup-aware: author matchup data + class
fallback, or the BC sideboard model's predicted overrides), writes the resolved
decks where the adapters can read them, and runs games at format='cc' sharded
across the 8 worker adapters. Trajectories persist via the normal DatasetWriter,
so this also generates the data the Stage-2 winrate signal will need.

    python -m python.gameplay.cc_selfplay --adapters 8000-8007 \
        --games 2 --pairs 8 --bot heuristic --model outputs/models/sideboard/sideboard_bc.pt
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

GRAVY = "gravy_bones_shipwrecked_looter"

from .selfplay import _build_bot, run_selfplay_batch
from ..deckbuilding.sideboard import resolve, cc_legal_issues

_REPO = Path(__file__).resolve().parents[2]
_POOLS = _REPO / "decks"
_GAMES_DIR = _REPO / "decks" / "_cc_games"   # under ./decks (mounted into adapters)
_CC_STATUS = _REPO / "outputs" / "cc_selfplay.jsonl"  # one row per completed matchup


_CC_DRAWS = _REPO / "outputs" / "cc_draws.log"

# One id per process invocation = one self-play run. Stamped into every status
# row so the monitor can segment runs unambiguously even when two runs overlap
# in time (pid keeps it unique; the timestamp keeps it human-readable/sortable).
_RUN_ID = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"


def _append_status(rec: dict) -> None:
    _CC_STATUS.parent.mkdir(parents=True, exist_ok=True)
    rec = {**rec, "run_id": _RUN_ID, "ts": time.time()}
    with open(_CC_STATUS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


# Stage-2 sideboard winrate signal: one row per matchup recording the sideboard
# override CHOICE each side made + the game outcome, so an AWR trainer can learn
# which choices WIN (see python/training/sideboard_rl.py). Distinct from the
# gameplay parquet (which records in-game transitions, not the deck decision).
_SB_MATCHES = _REPO / "outputs" / "cc_sideboard_matches.jsonl"


def _append_sb_match(rec: dict) -> None:
    _SB_MATCHES.parent.mkdir(parents=True, exist_ok=True)
    rec = {**rec, "run_id": _RUN_ID, "ts": time.time()}
    with open(_SB_MATCHES, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def _life(state: dict, pid: int):
    for p in state.get("players") or []:
        if int(p.get("player_id", 0)) == pid:
            return p.get("health")
    return None


def draw_report(traj, tail: int = 100) -> str:
    """Verbose diagnostic for a drawn game (winner==0) — built to surface engine
    WEDGES (e.g. the OMN pay-phase deadlock): what phase/action the game got
    stuck repeating, and whether life stopped moving. Extra detail for Gravy
    Bones games (Necromancer soul/graveyard mechanics are a wedge suspect)."""
    A, B = traj.hero1, traj.hero2
    trs = traj.transitions
    n = len(trs)
    gravy = GRAVY in (A, B)
    if not trs:
        return f"DRAW {A} vs {B} | steps=0 | (no transitions){'  <<GRAVY>>' if gravy else ''}"
    seg = trs[-tail:]
    phases = Counter(t.state.get("phase") for t in seg)
    atypes = Counter(t.chosen_action.get("type") for t in seg)
    sig = Counter((t.state.get("phase"), t.chosen_action.get("type"),
                   t.chosen_action.get("card_id")) for t in seg)
    l1s, l2s = _life(seg[0].state, 1), _life(seg[0].state, 2)
    l1e, l2e = _life(trs[-1].next_state, 1), _life(trs[-1].next_state, 2)
    stuck = (l1s == l1e and l2s == l2e)
    aborted = (traj.metadata or {}).get("aborted")
    L = [f"DRAW {A} vs {B} | steps={n} | life P1={l1e} P2={l2e} | "
         f"{'STUCK no-life-change/last%d' % len(seg) if stuck else 'life moving'}"
         f"{' | abort=' + aborted if aborted else ' | hit step_cap'}"
         f"{'  <<GRAVY BONES>>' if gravy else ''}",
         f"  last{len(seg)} phases: {dict(phases.most_common())}",
         f"  last{len(seg)} action-types: {dict(atypes.most_common())}",
         f"  most-repeated (phase,type,card): {sig.most_common(4)}"]
    if gravy:
        gp = 1 if A == GRAVY else 2
        end = trs[-1].next_state
        zc = {z: len((next((p for p in end.get('players', [])
                            if int(p.get('player_id', 0)) == gp), {}) or {}).get(z) or [])
              for z in ("graveyard", "soul", "banished", "pitch", "arsenal", "hand")}
        L.append(f"  [gravy P{gp}] end zones: {zc}")
        L.append("  [gravy] last 12 actions: " + str(
            [(t.step_index, f"P{t.player_to_move}", t.state.get("phase"),
              t.chosen_action.get("type"), t.chosen_action.get("card_id")) for t in trs[-12:]]))
    return "\n".join(L)


def _log_draw(report: str) -> None:
    print(report, file=sys.stderr)
    _CC_DRAWS.parent.mkdir(parents=True, exist_ok=True)
    with open(_CC_DRAWS, "a", encoding="utf-8") as fh:
        fh.write(report + "\n" + "-" * 70 + "\n")


def load_pools() -> dict[str, list[dict]]:
    """hero slug -> list of registered pool variants, from decks/cc_*.json.
    Multiple files per hero (different fabrary deckIds) are ALL kept so self-play
    can rotate decks per hero across games for deck diversity (see
    _pick_variant). Sorted for a deterministic, reproducible variant order."""
    out: dict[str, list[dict]] = {}
    for fp in sorted(glob.glob(str(_POOLS / "cc_*.json"))):
        p = json.loads(Path(fp).read_text(encoding="utf-8"))
        p.setdefault("_src", Path(fp).stem[-6:].lower())
        out.setdefault(p["hero"], []).append(p)
    return out


def _pick_variant(variants: list[dict], seed: int) -> dict:
    """Deterministically pick one of a hero's deck variants by seed, so a hero
    brings different lists across its matchups (reproducible per base_seed)."""
    return variants[seed % len(variants)]


def _resolve_and_write(pool: dict, opp_hero: str, model,
                       explore_temp: float = 0.0, rng=None) -> tuple[str, list[str], dict | None]:
    """Resolve `pool` vs `opp_hero` and write the game deck; return (repo-rel
    path, legality issues, the sideboard override used). With explore_temp>0 the
    sideboard model SAMPLES the override (Stage-2 exploration) instead of argmax."""
    if model is None:
        overrides = None
    elif explore_temp and explore_temp > 0:
        overrides = model.sample_overrides(pool, opp_hero, temperature=explore_temp, rng=rng)
    else:
        overrides = model.predict_overrides(pool, opp_hero)
    deck = resolve(pool, opp_hero=opp_hero, overrides=overrides)
    issues = cc_legal_issues(deck)
    _GAMES_DIR.mkdir(parents=True, exist_ok=True)
    out = _GAMES_DIR / f"{pool['hero']}__vs__{opp_hero}.json"
    out.write_text(json.dumps(
        {"hero": deck["hero"],
         "comment": f"CC game deck: {pool['hero']} vs {opp_hero} (matchup={deck.get('matchup')})",
         "equipment": deck["equipment"], "deck": deck["deck"]}, indent=2), encoding="utf-8")
    return str(out.relative_to(_REPO)).replace("\\", "/"), issues, overrides


def _make_bot(bot: str, gameplay_model: str, seed: int):
    """Build the gameplay-acting bot. If `gameplay_model` is set, use the trained
    IQL policy (argmax, no exploration) for a transfer test; else the heuristic/
    random/transformer bot named by `bot`. IQLGameplayBot self-falls-back to a
    BalancedBot if the checkpoint is missing/incompatible, so this never crashes."""
    if gameplay_model:
        from .bots.iql_bot import IQLGameplayBot
        return IQLGameplayBot(checkpoint=gameplay_model, seed=seed,
                              temperature=0.0, epsilon=0.0)
    return _build_bot(bot, seed)


def run_matchup(adapter_url: str, pools: dict, heroA: str, heroB: str, *,
                model, games: int, base_seed: int, out_dir: str, bot: str,
                gameplay_model: str = "",
                step_cap: int = 600, report_draws: bool = True,
                no_progress_cap: int = 60, life_stall_cap: int = 0,
                explore_sideboard: float = 0.0) -> dict:
    import numpy as np
    rng = np.random.default_rng(base_seed) if explore_sideboard > 0 else None
    poolA = _pick_variant(pools[heroA], base_seed)
    poolB = _pick_variant(pools[heroB], base_seed + 7)
    deckA, iA, ovA = _resolve_and_write(poolA, heroB, model, explore_sideboard, rng)
    deckB, iB, ovB = _resolve_and_write(poolB, heroA, model, explore_sideboard, rng)
    if iA or iB:
        rec = {"heroA": heroA, "heroB": heroB, "completed": 0,
               "error": f"illegal deck(s): A={iA} B={iB}",
               "deckA": poolA.get("_src"), "deckB": poolB.get("_src")}
        _append_status(rec)
        return {"matchup": f"{heroA} vs {heroB}", "completed": 0, "error": rec["error"]}

    tally = {0: 0, 1: 0, 2: 0}   # 0=draw, 1=heroA win, 2=heroB win

    def _on_game(traj):
        w = int(traj.winner or 0)
        tally[w] = tally.get(w, 0) + 1
        if w == 0 and report_draws:
            _log_draw(draw_report(traj))

    n = run_selfplay_batch(
        adapter_url=adapter_url, hero1=heroA, hero2=heroB,
        deck1=deckA, deck2=deckB,
        bot1=_make_bot(bot, gameplay_model, base_seed),
        bot2=_make_bot(bot, gameplay_model, base_seed + 999),
        n_games=games, base_seed=base_seed, out_dir=out_dir,
        game_format="cc", flush_every=games, on_game=_on_game, step_cap=step_cap,
        no_progress_cap=no_progress_cap, life_stall_cap=life_stall_cap,
    )
    _append_status({"heroA": heroA, "heroB": heroB, "completed": n,
                    "winA": tally[1], "winB": tally[2], "draws": tally[0],
                    "model": bool(model), "adapter": adapter_url,
                    "deckA": poolA.get("_src"), "deckB": poolB.get("_src")})
    if model is not None:
        # Stage-2 winrate signal: the sideboard override CHOICE each side made
        # this matchup + the outcome. AWR learns which choices win.
        _append_sb_match({"heroA": heroA, "heroB": heroB,
                          "deckA": poolA.get("_src"), "deckB": poolB.get("_src"),
                          "overrideA": ovA or {}, "overrideB": ovB or {},
                          "winA": tally[1], "winB": tally[2], "draws": tally[0],
                          "explore_temp": explore_sideboard})
    return {"matchup": f"{heroA} vs {heroB}", "completed": n,
            "winA": tally[1], "winB": tally[2], "draws": tally[0]}


def make_pairings(heroes: list[str], mode: str, n: int, seed: int,
                  focus: str | None = None) -> list[tuple[str, str]]:
    pairs = list(itertools.combinations(sorted(heroes), 2))
    if focus:
        pairs = [p for p in pairs if focus in p]   # only matchups vs the focus hero
    if mode == "random":
        random.Random(seed).shuffle(pairs)
    return pairs[:n] if n else pairs


def _parse_adapters(spec: str) -> list[str]:
    """'8000-8007' or '8000,8001' or full URLs -> list of adapter URLs."""
    urls = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part and part.replace("-", "").isdigit():
            lo, hi = part.split("-")
            urls += [f"http://localhost:{p}" for p in range(int(lo), int(hi) + 1)]
        elif part.isdigit():
            urls.append(f"http://localhost:{part}")
        else:
            urls.append(part)
    return urls


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Classic Constructed self-play.")
    ap.add_argument("--adapters", default="8000-8007", help="ports range/list or URLs")
    ap.add_argument("--games", type=int, default=2, help="games per matchup")
    ap.add_argument("--pairs", type=int, default=8, help="number of matchups (0=all)")
    ap.add_argument("--pairing", choices=["random", "roundrobin"], default="random")
    ap.add_argument("--bot", default="heuristic")
    ap.add_argument("--gameplay-model", default="",
                    help="trained IQL gameplay ckpt to ACT with (argmax, no exploration); "
                         "e.g. outputs/models/gameplay/latest.pt. Overrides --bot. "
                         "Use for the draft->CC policy transfer test.")
    ap.add_argument("--model", default="", help="sideboard BC ckpt (optional; else author matchup data)")
    ap.add_argument("--out", default="datasets/cc")
    ap.add_argument("--base-seed", type=int, default=5000)
    ap.add_argument("--step-cap", type=int, default=600,
                    help="per-game step cap (CC; low surfaces wedges fast vs 2000)")
    ap.add_argument("--no-progress-cap", type=int, default=60,
                    help="abort a game after N steps with no game-progress (hard-wedge guard; 0=off)")
    ap.add_argument("--life-stall-cap", type=int, default=0,
                    help="abort if neither life total changes for N steps (soft-stall/loop guard; 0=off, "
                         "now default off since the real wedges are capped - lets control mirrors resolve; "
                         "no-progress-cap=60 still catches hard engine deadlocks)")
    ap.add_argument("--no-draw-report", action="store_true",
                    help="disable verbose per-draw wedge diagnostics (default ON)")
    ap.add_argument("--focus-hero", default="",
                    help="only run matchups involving this hero slug "
                         "(e.g. gravy_bones_shipwrecked_looter)")
    ap.add_argument("--explore-sideboard", type=float, default=0.0,
                    help="Stage-2: SAMPLE sideboard overrides from --model at this softmax "
                         "temperature (0=off/argmax). Logs choice+outcome to "
                         "outputs/cc_sideboard_matches.jsonl for the winrate AWR trainer.")
    args = ap.parse_args(argv)

    # CPU throughput: each game-worker THREAD runs its own torch inference, so
    # torch's default intra-op parallelism (num_threads = #physical cores)
    # oversubscribes hard -- workers x cores threads fighting over the HW threads
    # (measured: 8 workers x 6 = 48 threads on 12, host pinned at ~91% thrashing).
    # The ThreadPool already supplies the parallelism, so ONE intra-op thread per
    # worker is far faster for this many-parallel-streams CPU inference workload.
    if args.gameplay_model:
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass

    pools = load_pools()
    if len(pools) < 2:
        print("need >=2 CC pools in decks/cc_*.json", file=sys.stderr)
        return 1
    model = None
    if args.model:
        from ..deckbuilding.sideboard_model import SideboardModel
        model = SideboardModel.load(args.model)
        print(f"[cc-selfplay] sideboard model: {args.model}", file=sys.stderr)

    if args.gameplay_model:
        print(f"[cc-selfplay] gameplay policy (acting bot): {args.gameplay_model} "
              f"(argmax, eps=0) -- overrides --bot={args.bot}", file=sys.stderr)

    adapters = _parse_adapters(args.adapters)
    pairs = make_pairings(list(pools), args.pairing, args.pairs, args.base_seed,
                          focus=args.focus_hero or None)
    print(f"[cc-selfplay] {len(pairs)} matchups x {args.games} games across "
          f"{len(adapters)} workers | step_cap={args.step_cap} | "
          f"draw_report={'off' if args.no_draw_report else 'ON'}"
          f"{' | focus=' + args.focus_hero if args.focus_hero else ''}", file=sys.stderr)

    results = []
    with ThreadPoolExecutor(max_workers=len(adapters)) as ex:
        futs = {}
        for i, (a, b) in enumerate(pairs):
            url = adapters[i % len(adapters)]
            futs[ex.submit(run_matchup, url, pools, a, b, model=model,
                           games=args.games, base_seed=args.base_seed + 100 * i,
                           out_dir=args.out, bot=args.bot,
                           gameplay_model=args.gameplay_model, step_cap=args.step_cap,
                           report_draws=not args.no_draw_report,
                           no_progress_cap=args.no_progress_cap,
                           life_stall_cap=args.life_stall_cap,
                           explore_sideboard=args.explore_sideboard)] = (a, b)
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(f"[cc-selfplay] {r['matchup']}: {r.get('completed')} games"
                  f"{' ERROR ' + r['error'] if r.get('error') else ''}", file=sys.stderr)

    total = sum(r.get("completed", 0) for r in results)
    print(f"[cc-selfplay] DONE: {total} games over {len(results)} matchups", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
