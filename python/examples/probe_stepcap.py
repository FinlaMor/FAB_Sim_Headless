"""Probe: lethal-rate vs wall-time as a function of step_cap.

Plays the SAME two real drafted decks with the SAME warm-started IQL
gameplay bots (greedy, to remove sampling noise) for N games at each of
several step caps, alternating first player and reusing seeds across caps
so the only variable is the cap. Reports, per cap: lethal% / tiebreak% /
draw%, median steps, and median wall-seconds per game.

    python -u -m python.examples.probe_stepcap --games 10 --caps 400,1000
"""
from __future__ import annotations

import argparse
import json
import statistics as S
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.deckbuilding.deck import Deck  # noqa: E402
from python.gameplay.bots.iql_bot import IQLGameplayBot  # noqa: E402
from python.gameplay.env import TalisharEnv, wait_for_adapter  # noqa: E402
from python.tournament.match import run_match  # noqa: E402
from python.tournament.player import Player  # noqa: E402

ADAPTER = "http://localhost:8000"
CKPT = "outputs/models/gameplay/latest.pt"


def load_deck(path: str) -> Deck:
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    equip = list(j.get("equipment", []))
    weapon = equip[0] if equip else ""
    return Deck(hero=j["hero"], weapon=weapon, deck=list(j["deck"]), equipment=equip[1:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--caps", default="400,1000")
    ap.add_argument("--deck-dir", default="decks/_tmp_real_smoke")
    ap.add_argument("--seed", type=int, default=31337)
    args = ap.parse_args()

    wait_for_adapter(ADAPTER, timeout_s=30.0)
    caps = [int(c) for c in args.caps.split(",")]
    d0 = load_deck(f"{args.deck_dir}/seat0_deck.json")
    d1 = load_deck(f"{args.deck_dir}/seat1_deck.json")

    def make_bot(game_seed: int):
        return IQLGameplayBot(checkpoint=CKPT, seed=game_seed, epsilon=0.0, temperature=0.0)

    env = TalisharEnv(ADAPTER, timeout=60.0)
    print(f"decks: {d0.hero} vs {d1.hero} | ckpt={CKPT} | {args.games} games/cap\n")
    try:
        for cap in caps:
            lethal = tiebreak = draw = 0
            steps_list, wall_list = [], []
            for g in range(args.games):
                # alternate first player; reuse seed across caps for comparability
                if g % 2 == 0:
                    p1 = Player(seat=0, deck=d0, bot_factory=make_bot, name=d0.hero)
                    p2 = Player(seat=1, deck=d1, bot_factory=make_bot, name=d1.hero)
                else:
                    p1 = Player(seat=0, deck=d1, bot_factory=make_bot, name=d1.hero)
                    p2 = Player(seat=1, deck=d0, bot_factory=make_bot, name=d0.hero)
                m = run_match(env=env, p1=p1, p2=p2, match_id=f"probe_c{cap}_g{g}",
                              seed=args.seed + g, step_cap=cap)
                meta = m.metadata
                if meta.get("engine_winner"):
                    lethal += 1
                elif meta.get("tiebreak"):
                    tiebreak += 1
                else:
                    draw += 1
                steps_list.append(meta.get("steps", 0))
                wall_list.append(m.ended_at - m.started_at)
            n = args.games
            print(f"cap={cap:>4}: lethal={lethal}/{n} ({100*lethal/n:.0f}%)  "
                  f"tiebreak={tiebreak} ({100*tiebreak/n:.0f}%)  draw={draw}  "
                  f"| median steps={S.median(steps_list):.0f}  "
                  f"median wall={S.median(wall_list):.1f}s  mean wall={S.mean(wall_list):.1f}s")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
