"""Find and dissect a residual wedge on REAL drafted decks. Plays real-deck
pairs with AggroBot until a game stalls (same turn/phase/hp for K steps), then
dumps the lead-up actions, the stuck engine state (phase/stack/decision queue/
resources/hand), and the gamelog tail — to identify the deadlock mechanism the
unpayable-first-entry cancel does NOT cover.

    python -m python.examples.diag_wedge_trace --max-games 30
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.deckbuilding.deck import Deck  # noqa: E402
from python.gameplay.env import TalisharEnv  # noqa: E402
from python.gameplay.bots.aggro_bot import AggroBot  # noqa: E402


def _load_deck(path: str) -> Deck:
    d = json.load(open(path, encoding="utf-8"))
    eq = d.get("equipment", []) or []
    return Deck(hero=d["hero"], weapon=(eq[0] if eq else ""),
               deck=list(d.get("deck", [])), equipment=list(eq[1:]))


def _hp(state):
    return [p.get("health") for p in state.get("players", [])]


def _dump_gamelog(gid, n=22):
    for base in ("talishar/Games", "datasets/games"):
        gl = PROJECT_ROOT / base / gid / "gamelog.txt"
        if gl.exists():
            lines = [re.sub(r"<[^>]+>", "", x).strip()
                     for x in gl.read_text(encoding="utf-8", errors="ignore").splitlines()]
            print(f"--- gamelog tail ({gid}) ---")
            for x in [x for x in lines if x][-n:]:
                print("  ", x[:150])
            return
    print(f"(no gamelog for {gid})")


def play(env, dA, dB, seed, step_cap, wedge_k):
    init = env.reset(hero1=dA.hero, hero2=dB.hero, deck1="decks/_real_draft/_a.json",
                     deck2="decks/_real_draft/_b.json", seed=seed)
    b1, b2 = AggroBot(seed=seed), AggroBot(seed=seed + 1)
    state, legal = init.state, init.legal_actions
    hist = []
    sig = None
    same = 0
    for step in range(step_cap):
        if env.done:
            return "engine_winner", step, None
        if not legal:
            legal = env.get_actions(refresh=True)
            if not legal:
                return "no_legal", step, hist
        prio = int(state.get("priority_player", 0))
        cur = (state.get("turn"), str(state.get("phase")), _hp(state)[0], _hp(state)[1])
        same = same + 1 if cur == sig else 0
        sig = cur
        if same >= wedge_k:
            return "wedge", step, hist
        bot = b1 if prio == 1 else b2
        if len(legal) == 1:
            chosen = legal[0]
        else:
            d = bot.choose(state, legal, player_id=prio)
            chosen = next((a for a in legal if a.action_id == d.action_id), legal[0])
        hist.append((step, cur[0], cur[1], prio, len(legal), chosen.type, chosen.card_id))
        if len(hist) > 60:
            hist.pop(0)
        res = env.step(chosen.action_id)
        state, legal = res.state, res.legal_actions
    return "step_cap", step_cap, hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-games", type=int, default=30)
    ap.add_argument("--step-cap", type=int, default=400)
    ap.add_argument("--wedge-k", type=int, default=40)
    a = ap.parse_args()

    files = sorted(glob.glob("decks/_real_draft/*.json"))
    decks = [_load_deck(f) for f in files]
    env = TalisharEnv("http://localhost:8000", timeout=30.0)
    for g in range(a.max_games):
        dA, dB = decks[g % len(decks)], decks[(g + 1) % len(decks)]
        # write the chosen pair to fixed paths the adapter can resolve
        Path("decks/_real_draft/_a.json").write_text(json.dumps(
            {"hero": dA.hero, "equipment": [dA.weapon, *dA.equipment], "deck": dA.deck}))
        Path("decks/_real_draft/_b.json").write_text(json.dumps(
            {"hero": dB.hero, "equipment": [dB.weapon, *dB.equipment], "deck": dB.deck}))
        reason, step, hist = play(env, dA, dB, 970000 + g, a.step_cap, a.wedge_k)
        print(f"game {g}: {dA.hero[:12]} vs {dB.hero[:12]} -> {reason} @ {step}")
        if reason == "wedge":
            print(f"\n=== WEDGE on game {g}, seed {970000+g} ===")
            print("--- last steps before wedge ---")
            for h in hist[-26:]:
                print(f"  s{h[0]:>3} t{h[1]} {h[2]:<22} pr{h[3]} nleg={h[4]} chose={h[5]}:{h[6]}")
            print("\n--- stuck /state ---")
            st = env.get_state(refresh=True)
            print("  ", {k: st.get(k) for k in ("turn", "phase", "subphase", "priority_player",
                                                "action_points")})
            for k in ("stack", "decision_queue", "combat", "links", "last_played"):
                if st.get(k):
                    print(f"  {k} = {st.get(k)}")
            for p in st.get("players", []):
                print(f"  p{p.get('player_id')}: hp={p.get('health')} res={p.get('resources')} "
                      f"hand={p.get('hand')}")
            acts = env.get_actions(refresh=True)
            print("  legal now:", [(x.type, x.card_id) for x in acts][:10])
            _dump_gamelog(env.game_id)
            return 0
    print("no wedge encountered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
