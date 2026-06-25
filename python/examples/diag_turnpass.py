"""Diagnostic: always-PASS a real game and dump turn-transition internals.

Creates one game from the existing _tmp_real_smoke decks, then repeatedly
chooses the PASS action and prints, per step:

  step  active(main)  priority  phase  subphase  #stack  dq[0]  hands

Goal: see exactly where/why the turn fails to flip from player 1 to
player 2 (the M->INSTANT->ARS loop observed in real_pipeline_steps.log).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.gameplay.env import TalisharEnv  # noqa: E402

ADAPTER = "http://localhost:8000"
DECK_DIR = "decks/_tmp_attack_smoke"
STEP_CAP = 120


def fmt_stack(stack: list) -> str:
    # layers are flat tokens; show the leading kind tokens
    if not stack:
        return "[]"
    return "[" + ",".join(str(x) for x in stack[:6]) + ("..." if len(stack) > 6 else "") + "]"


def main() -> int:
    env = TalisharEnv(ADAPTER, timeout=30.0)
    h = env.health()
    if h.get("mode") != "real":
        print(f"FATAL: adapter mode={h.get('mode')!r}")
        return 2

    init = env.reset(
        hero1="zyggy", hero2="zyggy",
        deck1=f"{DECK_DIR}/seat0_deck.json",
        deck2=f"{DECK_DIR}/seat1_deck.json",
        seed=777_777,
    )
    print(f"game_id={env.game_id}")
    state = init.state
    legal = init.legal_actions

    for step in range(STEP_CAP):
        dq = state.get("decision_queue") or {}
        dq_queue = dq.get("queue") or []
        turn_arr = dq.get("turn") or []
        p1 = next((p for p in state.get("players", []) if p["player_id"] == 1), {})
        p2 = next((p for p in state.get("players", []) if p["player_id"] == 2), {})
        print(
            f"{step:>3} act={state.get('active_player')} pri={state.get('priority_player')} "
            f"phase={state.get('phase')!r:>10} sub={state.get('subphase')!r:>4} "
            f"hp1={p1.get('health')} hp2={p2.get('health')} "
            f"turn={turn_arr} stack={fmt_stack(state.get('stack') or [])} "
            f"cc={len(state.get('combat_chain') or [])} "
            f"h1={len(p1.get('hand') or [])} h2={len(p2.get('hand') or [])} "
            f"pit1={len(p1.get('pitch') or [])} pit2={len(p2.get('pitch') or [])} "
            f"res1={p1.get('resources')} "
            f"nleg={len(legal)} acts={[(a.type, a.card_id) for a in legal][:5]}"
        )
        if not legal:
            print("  no legal actions; stop")
            break
        # AGGRESSIVE: prefer playing/attacking over passing to exercise
        # the cost/pitch/combat machinery.
        non_pass = [a for a in legal if a.type != "PASS"]
        chosen = non_pass[0] if non_pass else legal[0]
        print(f"    -> chose {chosen.type} card={chosen.card_id} "
              f"mode={chosen.raw.get('talishar_mode')} cardid={chosen.raw.get('talishar_card_id')}")
        result = env.step(chosen.action_id)
        state = result.state
        legal = result.legal_actions
        if result.done:
            print(f"GAME OVER winner={result.winner}")
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
