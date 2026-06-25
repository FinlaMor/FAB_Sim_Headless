"""Smoke test: load two resolved CC decks into the engine (format='cc') and
play random actions to confirm the format wiring works end-to-end."""
import random
from python.gameplay.env import TalisharEnv

random.seed(7)
env = TalisharEnv("http://localhost:8009")
res = env.reset(
    hero1="kano_dracai_of_aether", hero2="bravo_showstopper",
    deck1="decks/resolved/cc_kano_dracai_of_aether_p2pjcd_game.json",
    deck2="decks/resolved/cc_bravo_showstopper_kpgc30_game.json",
    seed=12345, format="cc")


def summary(st):
    out = []
    for p in st.get("players", []):
        out.append(f"P{p.get('player_id')} {p.get('hero')} hp={p.get('health')} "
                   f"deck={p.get('deck_count')} hand={len(p.get('hand') or [])}")
    return (f"phase={st.get('phase')} turn={st.get('turn')} | "
            + " | ".join(out) + f" | landmarks={st.get('landmarks')}")


print("INITIAL:", summary(res.state))
r = res
for i in range(120):
    if r.done or not r.legal_actions:
        break
    a = random.choice(r.legal_actions)
    r = env.step(a.action_id)
    if i % 20 == 0:
        print(f"step {i:>3}:", summary(r.state), "done=", r.done)
print("FINAL:", summary(r.state), "| done=", r.done, "winner=", r.winner)
