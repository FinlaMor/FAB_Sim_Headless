"""Controlled game: attack in M phase, never block — proves a winner.

Strategy for both players: in the main phase, play the first attack from
hand (and pitch when asked, since the only non-PASS options in P are
pitches); in every other phase, PASS (so the defender never blocks).
Unblocked attacks connect, health drops, and the game ends with a winner.
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
STEP_CAP = 2000


def main() -> int:
    env = TalisharEnv(ADAPTER, timeout=30.0)
    seed = 909
    init = env.reset(
        hero1="zyggy", hero2="zyggy",
        deck1=f"{DECK_DIR}/seat0_deck.json",
        deck2=f"{DECK_DIR}/seat1_deck.json",
        seed=seed,
    )
    state, legal = init.state, init.legal_actions
    last_hp = (20, 20)
    for step in range(STEP_CAP):
        if env.done:
            break
        if not legal:
            legal = env.get_actions(refresh=True)
            if not legal:
                print(f"step {step}: no legal actions; abort")
                break
        phase = state.get("phase")
        # In M (main) or P (pitch), take the first non-PASS action (attack
        # or pitch). Everywhere else, PASS (never block / never react).
        chosen = None
        if phase in ("M", "P"):
            chosen = next((a for a in legal if a.type != "PASS"), None)
        if chosen is None:
            chosen = next((a for a in legal if a.type == "PASS"), legal[0])
        result = env.step(chosen.action_id)
        state, legal = result.state, result.legal_actions
        p1 = next((p for p in state.get("players", []) if p["player_id"] == 1), {})
        p2 = next((p for p in state.get("players", []) if p["player_id"] == 2), {})
        hp = (p1.get("health"), p2.get("health"))
        if hp != last_hp:
            print(f"step {step:>4}: HP {last_hp} -> {hp}  turn={state.get('turn')} phase={phase}")
            last_hp = hp
        if result.done:
            print(f"\nGAME OVER at step {step}: winner={result.winner} reward={result.reward}")
            break

    print(f"\nfinal hp={last_hp} done={env.done} winner={env.winner}")
    return 0 if env.winner in (1, 2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
