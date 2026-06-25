"""Diagnostic: run N games through run_match and report each game's
termination reason (new `metadata["term_reason"]`). Confirms the abort
plumbing end-to-end and quantifies how many games die in the opening.

    python -m python.examples.diag_term_reason --games 12
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.deckbuilding.deck import Deck  # noqa: E402
from python.gameplay.env import TalisharEnv  # noqa: E402
from python.gameplay.bots.iql_bot import IQLGameplayBot  # noqa: E402
from python.tournament.player import Player  # noqa: E402
from python.tournament.match import run_match  # noqa: E402


def _load_deck(path: str) -> Deck:
    d = json.load(open(path, encoding="utf-8"))
    eq = d.get("equipment", []) or []
    return Deck(hero=d["hero"], weapon=(eq[0] if eq else ""),
               deck=list(d.get("deck", [])), equipment=list(eq[1:]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--ckpt", default="outputs/models/gameplay/latest.pt")
    ap.add_argument("--step-cap", type=int, default=400)
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--explorer", default="aggressive")
    a = ap.parse_args()

    dA = _load_deck("outputs/benchmark_decks/bench_p1.json")
    dB = _load_deck("outputs/benchmark_decks/bench_p2.json")
    mk = lambda s: IQLGameplayBot(checkpoint=a.ckpt, seed=s, epsilon=a.epsilon,
                                  temperature=a.temperature, explorer=a.explorer)

    reasons: Counter = Counter()
    with TalisharEnv(a.url, timeout=30.0) as env:
        for g in range(a.games):
            p1 = Player(seat=0, deck=dA, bot_factory=mk, label="A")
            p2 = Player(seat=1, deck=dB, bot_factory=mk, label="B")
            m = run_match(env=env, p1=p1, p2=p2, match_id=f"diag.g{g}",
                          seed=900000 + g, step_cap=a.step_cap)
            md = m.metadata
            reasons[md.get("term_reason")] += 1
            print(f"g{g:02d} winner={m.winner_label or '-':<3} "
                  f"reason={md.get('term_reason'):<14} steps={md.get('steps'):<4} "
                  f"recover={md.get('recoveries')} err={(m.error or '')[:60]}")
    print("\n=== term_reason counts ===")
    for k, v in reasons.most_common():
        print(f"  {k:<16} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
