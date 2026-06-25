"""Play the controlled-attack game to the wedge and dump the paying player's
resource/hand state + the pending layer, to confirm WHY it's stuck."""
from __future__ import annotations
import json, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from python.gameplay.env import TalisharEnv  # noqa: E402

env = TalisharEnv("http://localhost:8000", timeout=30.0)
init = env.reset(hero1="zyggy", hero2="zyggy",
                 deck1="decks/_tmp_attack_smoke/seat0_deck.json",
                 deck2="decks/_tmp_attack_smoke/seat1_deck.json", seed=909)
state, legal = init.state, init.legal_actions
pass_run = 0
for step in range(400):
    if env.done:
        break
    if not legal:
        legal = env.get_actions(refresh=True)
        if not legal:
            break
    phase = str(state.get("phase", ""))
    chosen = None
    if phase in ("M", "P"):
        chosen = next((a for a in legal if a.type != "PASS"), None)
    if chosen is None:
        chosen = next((a for a in legal if a.type == "PASS"), legal[0])
    only_pass = len(legal) == 1 and legal[0].type == "PASS"
    pass_run = pass_run + 1 if only_pass else 0
    if pass_run >= 6:
        st = env.get_state(refresh=True)
        print(f"WEDGE at step {step}: phase={st.get('phase')} sub={st.get('subphase')} "
              f"prio={st.get('priority_player')} turn={st.get('turn')} ap={st.get('action_points')}")
        for p in st.get("players", []):
            print(f"  player {p.get('player_id')}: resources={p.get('resources')} "
                  f"hand={p.get('hand')} hand_n={len(p.get('hand') or [])}")
            print(f"     pitch={p.get('pitch')}")
            for k in ("effects", "class_state", "permanents", "items"):
                v = p.get(k)
                if v:
                    print(f"     {k}={v}")
        print(f"  only legal action raw: {json.dumps(legal[0].raw)[:300]}")
        # full state dump of any layer/stack-ish keys
        for k in st:
            if k not in ("players",):
                print(f"  state.{k} = {st[k]}")
        break
    res = env.step(chosen.action_id)
    state, legal = res.state, res.legal_actions
