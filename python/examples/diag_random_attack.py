"""Random-vs-random game on the pure-attack decks; report damage + winner.

Validates that real combat damage now flows after the CCS-globals fix:
a RandomBot defender will sometimes decline to block, so attacks should
connect and someone should eventually die.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.gameplay.env import TalisharEnv  # noqa: E402
from python.gameplay.bots.random_bot import RandomBot  # noqa: E402

ADAPTER = "http://localhost:8000"
DECK_DIR = "decks/_tmp_attack_smoke"
STEP_CAP = 1500


def main() -> int:
    env = TalisharEnv(ADAPTER, timeout=30.0)
    seed = 4242
    init = env.reset(
        hero1="zyggy", hero2="zyggy",
        deck1=f"{DECK_DIR}/seat0_deck.json",
        deck2=f"{DECK_DIR}/seat1_deck.json",
        seed=seed,
    )
    bot1, bot2 = RandomBot(seed=seed), RandomBot(seed=seed + 1)
    bot1.reset(seed=seed)
    bot2.reset(seed=seed + 1)

    state, legal = init.state, init.legal_actions
    last_hp = (20, 20)
    damage_events = 0
    for step in range(STEP_CAP):
        if env.done:
            break
        pri = int(state.get("priority_player") or 0)
        if pri not in (1, 2):
            print(f"step {step}: bad priority {pri}; abort")
            break
        if not legal:
            legal = env.get_actions(refresh=True)
            if not legal:
                print(f"step {step}: no legal actions; abort")
                break
        bot = bot1 if pri == 1 else bot2
        decision = bot.choose(state, legal, player_id=pri)
        result = env.step(decision.action_id)
        state, legal = result.state, result.legal_actions
        p1 = next((p for p in state.get("players", []) if p["player_id"] == 1), {})
        p2 = next((p for p in state.get("players", []) if p["player_id"] == 2), {})
        hp = (p1.get("health"), p2.get("health"))
        if hp != last_hp:
            damage_events += 1
            print(f"step {step:>4}: HP changed {last_hp} -> {hp}  (turn {state.get('turn')})")
            last_hp = hp
        if result.done:
            print(f"\nGAME OVER at step {step}: winner={result.winner} reward={result.reward}")
            break

    print(f"\nfinal hp = {last_hp}  damage_events={damage_events}  done={env.done} winner={env.winner}")
    return 0 if (env.winner in (1, 2)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
