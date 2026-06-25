"""Diagnostic: prove the arsenal step works end-to-end through the adapter.

Always PASS every decision so hands stay full. Per the engine
(NetworkingLibraries.php::PassTurn), end of turn with a non-empty hand and a
non-full arsenal MUST set turn[0] = "ARS" and offer the arsenal step. When the
ARS window appears we TAKE an ARSENAL_FROM_HAND action, verify the card is in
the arsenal zone next state, and then watch for PLAY_FROM_ARSENAL to be
offered on that player's next main phase.

Exit codes: 0 = full arsenal lifecycle observed; 1 = ARS seen but lifecycle
incomplete; 2 = ARS window NEVER appeared (the old bug).
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
STEP_CAP = 200


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
        seed=424_242,
    )
    print(f"game_id={env.game_id}")
    state, legal = init.state, init.legal_actions

    saw_ars_phase = False
    saw_ars_offer = False
    arsenalled = None          # (player_id, card_id) we arsenalled
    in_arsenal_zone = False
    saw_play_from_ars_offer = False
    played_from_ars = False

    for step in range(STEP_CAP):
        ph = state.get("phase")
        types = [a.type for a in legal]
        if ph == "ARS" or "ARSENAL_FROM_HAND" in types:
            saw_ars_phase = saw_ars_phase or ph == "ARS"
            pri = state.get("priority_player")
            print(f"[{step}] phase={ph} pri={pri} offered={types}")
        if not legal:
            print(f"[{step}] no legal actions; stop")
            break

        ars_acts = [a for a in legal if a.type == "ARSENAL_FROM_HAND"]
        pfa_acts = [a for a in legal if a.type == "PLAY_FROM_ARSENAL"]
        if ars_acts and arsenalled is None:
            saw_ars_offer = True
            chosen = ars_acts[0]
            arsenalled = (chosen.raw.get("player_id"), chosen.card_id)
            print(f"[{step}] TAKING ARSENAL_FROM_HAND card={chosen.card_id} "
                  f"for p{arsenalled[0]}")
        elif pfa_acts and arsenalled and not played_from_ars:
            saw_play_from_ars_offer = True
            chosen = pfa_acts[0]
            played_from_ars = True
            print(f"[{step}] TAKING PLAY_FROM_ARSENAL card={chosen.card_id}")
        else:
            # pure PASS policy keeps hands full so ARS must trigger
            passes = [a for a in legal if a.type == "PASS"]
            chosen = passes[0] if passes else legal[0]

        result = env.step(chosen.action_id)
        state, legal = result.state, result.legal_actions

        if arsenalled and not in_arsenal_zone:
            pid, card = arsenalled
            for p in state.get("players", []):
                if int(p.get("player_id", 0)) == int(pid or 0):
                    if card in (p.get("arsenal") or []):
                        in_arsenal_zone = True
                        print(f"    VERIFIED: {card} is in p{pid} arsenal zone")
        if pfa_acts and not played_from_ars:
            saw_play_from_ars_offer = True

        if result.done:
            print(f"GAME OVER winner={result.winner}")
            break

    print("\n==== ARSENAL LIFECYCLE ====")
    print(f"ARS phase reached:          {saw_ars_phase}")
    print(f"ARSENAL_FROM_HAND offered:  {saw_ars_offer}")
    print(f"card landed in arsenal:     {in_arsenal_zone}")
    print(f"PLAY_FROM_ARSENAL offered:  {saw_play_from_ars_offer}")
    print(f"played from arsenal:        {played_from_ars}")
    if not (saw_ars_phase or saw_ars_offer):
        print("RESULT: FAIL — ARS window never appeared (old bug symptom)")
        return 2
    if saw_ars_offer and in_arsenal_zone and saw_play_from_ars_offer:
        print("RESULT: PASS — full arsenal lifecycle works through the adapter")
        return 0
    print("RESULT: PARTIAL — see flags above")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
