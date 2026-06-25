"""Verbose controlled-attack trace: WHY does damage stop after the opening?
Same policy as diag_controlled_attack (first attack in M, PASS everywhere
else, never block) but logs every step so we can see whether main phases
recur, whether attacks are legal, and what the bot actually does."""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from python.gameplay.env import TalisharEnv  # noqa: E402

env = TalisharEnv("http://localhost:8000", timeout=30.0)
init = env.reset(hero1="zyggy", hero2="zyggy",
                 deck1="decks/_tmp_attack_smoke/seat0_deck.json",
                 deck2="decks/_tmp_attack_smoke/seat1_deck.json", seed=909)
state, legal = init.state, init.legal_actions
last_hp = None
N = 120
for step in range(N):
    if env.done:
        print(f"DONE at step {step} winner={env.winner}"); break
    if not legal:
        legal = env.get_actions(refresh=True)
        if not legal:
            print(f"step {step}: NO LEGAL -> abort"); break
    phase = str(state.get("phase", ""))
    prio = state.get("priority_player"); turn = state.get("turn")
    p1 = next((p for p in state.get("players", []) if p["player_id"] == 1), {})
    p2 = next((p for p in state.get("players", []) if p["player_id"] == 2), {})
    hp = (p1.get("health"), p2.get("health"))
    chosen = None
    if phase in ("M", "P"):
        chosen = next((a for a in legal if a.type != "PASS"), None)
    if chosen is None:
        chosen = next((a for a in legal if a.type == "PASS"), legal[0])
    types = [a.type for a in legal]
    cardc = chosen.card_id
    mark = "  <== attack-window" if (phase == "M" and any(a.type != "PASS" for a in legal)) else ""
    print(f"s{step:>3} t{turn} {phase:<22} pr{prio} hp{hp} nleg={len(legal)} "
          f"chose={chosen.type}:{cardc}{mark}")
    res = env.step(chosen.action_id)
    state, legal = res.state, res.legal_actions
print(f"\nfinal hp={hp} done={env.done} winner={env.winner}")
