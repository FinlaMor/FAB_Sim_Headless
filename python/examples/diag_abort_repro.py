"""Root-cause the turn-0 abort: replay real DRAFTED decks, log every opening
step, and on the first abort dump the adapter's TRUE current state/actions plus
the engine gamelog so we can see WHY the engine handed back no legal move.

    python -m python.examples.diag_abort_repro --bot iql --max-games 20
    python -m python.examples.diag_abort_repro --bot random --max-games 20   # control
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.gameplay.env import TalisharEnv  # noqa: E402
from python.gameplay.bots.random_bot import RandomBot  # noqa: E402


def _mk_bot(kind: str, seed: int):
    if kind == "random":
        return RandomBot(seed=seed)
    from python.gameplay.bots.iql_bot import IQLGameplayBot
    return IQLGameplayBot(checkpoint="outputs/models/gameplay/latest.pt", seed=seed,
                          epsilon=0.15, temperature=0.5, explorer="aggressive")


def _summ(state: dict) -> str:
    ps = state.get("players", [])
    hp = [p.get("health") for p in ps]
    return (f"turn={state.get('turn')} phase={state.get('phase')}/"
            f"{state.get('subphase')} active={state.get('active_player')} "
            f"prio={state.get('priority_player')} ap={state.get('action_points')} hp={hp}")


def _dump_gamelog(game_id: str, n: int = 30) -> None:
    for base in ("talishar/Games", "datasets/games"):
        gl = PROJECT_ROOT / base / game_id / "gamelog.txt"
        if gl.exists():
            import re
            lines = gl.read_text(encoding="utf-8", errors="ignore").splitlines()
            clean = [re.sub(r"<[^>]+>", "", ln) for ln in lines][-n:]
            print(f"--- gamelog tail ({gl}) ---")
            for ln in clean:
                if ln.strip():
                    print("  ", ln.strip()[:160])
            return
    print(f"--- no gamelog found for {game_id} ---")


def play_one(env: TalisharEnv, d1: str, d2: str, h1: str, h2: str,
             seed: int, bot_kind: str, step_cap: int, verbose_opening: int):
    init = env.reset(hero1=h1, hero2=h2, deck1=d1, deck2=d2, seed=seed)
    bot1, bot2 = _mk_bot(bot_kind, seed), _mk_bot(bot_kind, seed + 1)
    bot1.reset(seed=seed); bot2.reset(seed=seed + 1)
    state, legal = init.state, init.legal_actions
    step = 0
    while not env.done and step < step_cap:
        prio = int(state.get("priority_player", 0))
        if prio not in (1, 2):
            print(f"\n*** ABORT @step{step} reason=bad_priority  {_summ(state)}")
            _dump_abort(env)
            return "bad_priority", step
        if not legal:
            legal = env.get_actions(refresh=True)
            if not legal:
                print(f"\n*** ABORT @step{step} reason=no_legal  {_summ(state)}")
                _dump_abort(env)
                return "no_legal", step
        bot = bot1 if prio == 1 else bot2
        if step < verbose_opening:
            atypes = [a.type for a in legal]
            print(f" step{step:>3} {_summ(state)} | nlegal={len(legal)} types={atypes}")
        if len(legal) == 1:
            chosen = legal[0]
        else:
            dec = bot.choose(state, legal, player_id=prio)
            chosen = next((a for a in legal if a.action_id == dec.action_id), legal[0])
        try:
            res = env.step(chosen.action_id)
        except Exception as e:  # noqa: BLE001
            print(f"\n*** ABORT @step{step} reason=step_exception {e!r}  {_summ(state)}")
            _dump_abort(env)
            return "exception", step
        state, legal = res.state, res.legal_actions
        step += 1
    return ("engine_winner" if env.done else "step_cap"), step


def _dump_abort(env: TalisharEnv) -> None:
    print("  [refetch] /state (refresh):")
    try:
        st = env.get_state(refresh=True)
        print("   ", _summ(st))
        print("    raw state keys:", list(st.keys()))
    except Exception as e:  # noqa: BLE001
        print("    state refetch failed:", repr(e))
    print("  [refetch] /actions (refresh):")
    try:
        acts = env.get_actions(refresh=True)
        print(f"    n={len(acts)}")
        for a in acts[:12]:
            print("    ", json.dumps(a.raw)[:200])
    except Exception as e:  # noqa: BLE001
        print("    actions refetch failed:", repr(e))
    _dump_gamelog(env.game_id)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", choices=["iql", "random"], default="iql")
    ap.add_argument("--max-games", type=int, default=20)
    ap.add_argument("--step-cap", type=int, default=400)
    ap.add_argument("--verbose-opening", type=int, default=12)
    ap.add_argument("--url", default="http://localhost:8000")
    a = ap.parse_args()

    pairs = sorted({p[:-8] for p in glob.glob("decks/_tmp_matches/t_*_p1.json")})
    pairs = [Path(p).name for p in pairs]
    if not pairs:
        print("no t_* drafted deck pairs in decks/_tmp_matches/")
        return 1
    print(f"{len(pairs)} drafted pairs; bot={a.bot}")

    reasons: dict = {}
    g = 0
    with TalisharEnv(a.url, timeout=30.0) as env:
        for pair in pairs:
            p1 = f"decks/_tmp_matches/{pair}_p1.json"
            p2 = f"decks/_tmp_matches/{pair}_p2.json"
            j1 = json.load(open(p1, encoding="utf-8"))
            j2 = json.load(open(p2, encoding="utf-8"))
            for rep in range(2):
                if g >= a.max_games:
                    break
                seed = 700000 + g
                print(f"\n=== game {g} pair={pair} seed={seed} "
                      f"{j1['hero']} vs {j2['hero']} ===")
                reason, step = play_one(env, p1, p2, j1["hero"], j2["hero"],
                                        seed, a.bot, a.step_cap, a.verbose_opening)
                reasons[reason] = reasons.get(reason, 0) + 1
                print(f"  -> {reason} @ step {step}")
                g += 1
                if reason in ("no_legal", "bad_priority", "exception"):
                    print("\n!!! captured a turn-0/early abort; stopping for inspection")
                    print("reasons so far:", reasons)
                    return 0
            if g >= a.max_games:
                break
    print("\n=== reasons ===", reasons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
