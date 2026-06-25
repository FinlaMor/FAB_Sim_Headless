"""Measure how often a gameplay policy/deck combination actually CLOSES games
to lethal. Self-play across deck pairs, concurrent over the adapter workers.
Reports lethal-win rate vs step-cap-draw rate and mean true game length.

Examples
--------
# trained policy on drafted decks (default):
python -m python.examples.validate_lethal --bot iql --ckpt outputs/models/_validate/baseline.pt --games 64
# deck-ABILITY ceiling: aggressive pilot on decks built to close:
python -m python.examples.validate_lethal --bot aggro \
    --deck1 decks/_tmp_attack_smoke/seat0_deck.json --deck2 decks/_tmp_attack_smoke/seat1_deck.json --games 32
# does the trained policy close on a deck built to close?
python -m python.examples.validate_lethal --bot iql --ckpt outputs/models/_validate/baseline.pt \
    --deck1 decks/_tmp_attack_smoke/seat0_deck.json --deck2 decks/_tmp_attack_smoke/seat1_deck.json --games 32
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.deckbuilding.deck import Deck  # noqa: E402
from python.gameplay.env import TalisharEnv  # noqa: E402
from python.gameplay.bots.aggro_bot import AggroBot  # noqa: E402
from python.gameplay.bots.random_bot import RandomBot  # noqa: E402
from python.tournament.player import Player  # noqa: E402
from python.tournament.match import run_match  # noqa: E402


def _load_deck(path: str) -> Deck:
    d = json.load(open(path, encoding="utf-8"))
    eq = d.get("equipment", []) or []
    return Deck(hero=d["hero"], weapon=(eq[0] if eq else ""),
               deck=list(d.get("deck", [])), equipment=list(eq[1:]))


def _bot_factory(kind: str, ckpt: str | None):
    if kind == "aggro":
        return lambda s: AggroBot(seed=s)
    if kind == "random":
        return lambda s: RandomBot(seed=s)
    if kind == "iql":
        from python.gameplay.bots.iql_bot import IQLGameplayBot
        return lambda s: IQLGameplayBot(checkpoint=ckpt, seed=s, temperature=0.0, epsilon=0.0)
    raise SystemExit(f"unknown bot {kind!r}")


def play(url, mk, dA, dB, seed, step_cap):
    env = TalisharEnv(url, timeout=30.0)
    try:
        p1 = Player(seat=0, deck=dA, bot_factory=mk, label="A", name="A")
        p2 = Player(seat=1, deck=dB, bot_factory=mk, label="B", name="B")
        m = run_match(env=env, p1=p1, p2=p2, match_id=f"val.s{seed}", seed=seed, step_cap=step_cap)
        md = m.metadata
        return (md.get("term_reason"), int(md.get("steps") or 0), m.winner_label)
    except Exception as e:  # noqa: BLE001
        return ("driver_error", 0, repr(e)[:60])
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", choices=["iql", "aggro", "random"], default="iql")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--deck1", default=None, help="explicit p1 deck json (else drafted glob)")
    ap.add_argument("--deck2", default=None, help="explicit p2 deck json")
    ap.add_argument("--deck-dir", default=None, help="dir of deck jsons; pairs them up")
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--ports", type=int, default=8)
    ap.add_argument("--step-cap", type=int, default=400)
    a = ap.parse_args()
    if a.bot == "iql" and not a.ckpt:
        raise SystemExit("--bot iql requires --ckpt")

    if a.deck1 and a.deck2:
        decks = [(_load_deck(a.deck1), _load_deck(a.deck2))]
        src = f"{Path(a.deck1).name} vs {Path(a.deck2).name}"
    elif a.deck_dir:
        files = sorted(glob.glob(str(Path(a.deck_dir) / "*.json")))
        loaded = [_load_deck(f) for f in files]
        decks = [(loaded[i], loaded[(i + 1) % len(loaded)]) for i in range(len(loaded))]
        src = f"{len(loaded)} decks from {a.deck_dir}"
    else:
        pairs = sorted({p[:-8] for p in glob.glob("decks/_tmp_matches/t_*_p1.json")})
        decks = []
        for pp in pairs:
            name = Path(pp).name
            decks.append((_load_deck(f"decks/_tmp_matches/{name}_p1.json"),
                          _load_deck(f"decks/_tmp_matches/{name}_p2.json")))
        src = f"{len(decks)} drafted pairs"

    mk = _bot_factory(a.bot, a.ckpt)
    urls = [f"http://localhost:{8000 + i}" for i in range(a.ports)]
    reasons: Counter = Counter()
    steps_all = []
    g = 0
    print(f"validating bot={a.bot} ckpt={a.ckpt} decks=({src}): "
          f"{a.games} self-play games, step_cap={a.step_cap}")
    while g < a.games:
        jobs = []
        for url in urls:
            if g >= a.games:
                break
            dA, dB = decks[g % len(decks)]
            jobs.append((url, mk, dA, dB, 950000 + g, a.step_cap))
            g += 1
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            for reason, steps, wl in ex.map(lambda j: play(*j), jobs):
                reasons[reason] += 1
                steps_all.append(steps)
    tot = sum(reasons.values()) or 1
    lethal = reasons.get("engine_winner", 0)
    draws = reasons.get("step_cap", 0)
    print("\n=== result ===")
    print("reasons:", dict(reasons))
    print(f"LETHAL: {lethal}/{tot} = {100*lethal/tot:.1f}%   "
          f"step_cap DRAW: {draws}/{tot} = {100*draws/tot:.1f}%")
    print(f"mean true steps: {sum(steps_all)/len(steps_all):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
