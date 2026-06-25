"""ACTION-SPACE AUDIT: play a batch of games on a (fixed) adapter and record,
for every priority window, the turn phase and the legal action types offered.
Flags phases where the engine repeatedly hands a player priority but the
serializer offers ONLY pass -> a likely enumeration gap (the player can't act).

Used to verify the ActionSerializer covers every decision point for the OMN
draft format BEFORE retraining. Run against the dedicated adapter (8008), never
the pipeline workers.

    python -m python.examples.diag_action_audit --url http://localhost:8008 --games 15
"""
from __future__ import annotations

import argparse
import collections
import glob
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.gameplay.env import TalisharEnv                # noqa: E402
from python.examples.play_vs_champion import _load_deck    # noqa: E402

_DECK_REL_DIR = "decks/_tmp_matches"
_DECK_DIR = PROJECT_ROOT / _DECK_REL_DIR


def _audit_policy(legal, rng):
    """Coverage-maximising policy: take ARSENAL when offered (to populate the
    arsenal so play-from-arsenal windows occur), otherwise usually act, so the
    game reaches as many distinct phases/actions as possible."""
    by_type = collections.defaultdict(list)
    for a in legal:
        by_type[(a.type or "").upper()].append(a)
    if by_type.get("ARSENAL_FROM_HAND") and rng.random() < 0.6:
        return rng.choice(by_type["ARSENAL_FROM_HAND"])
    nonpass = [a for a in legal if (a.type or "").upper() != "PASS"]
    if nonpass and rng.random() < 0.8:
        return rng.choice(nonpass)
    return next((a for a in legal if (a.type or "").upper() == "PASS"), legal[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8008")
    ap.add_argument("--games", type=int, default=15)
    ap.add_argument("--step-cap", type=int, default=1500)
    args = ap.parse_args()

    gate = sorted(glob.glob(str(PROJECT_ROOT / "outputs/gate_decks/*.json")))
    decks = [_load_deck(p) for p in gate[:6]] or [_load_deck(gate[0])]
    _DECK_DIR.mkdir(parents=True, exist_ok=True)

    phase_windows = collections.Counter()
    phase_onlypass = collections.Counter()
    phase_types: dict = collections.defaultdict(collections.Counter)
    type_total = collections.Counter()
    rng = random.Random(12345)

    def report():
        print("=" * 70, flush=True)
        print(f"ACTION-SPACE AUDIT  ({args.url})")
        print("=" * 70)
        print(f"{'phase':<14}{'windows':>9}{'only-PASS':>11}{'%':>6}   non-pass types")
        for ph, w in phase_windows.most_common():
            op = phase_onlypass[ph]
            pct = 100.0 * op / max(w, 1)
            tt = ", ".join(f"{t}({c})" for t, c in phase_types[ph].most_common(6))
            flag = "  <-- ONLY-PASS GAP?" if pct > 90 and w >= 5 else ""
            print(f"{ph:<14}{w:>9}{op:>11}{pct:>5.0f}%   {tt}{flag}")
        print("-" * 70)
        print("distinct non-pass action TYPES emitted across all phases:")
        for t, c in type_total.most_common():
            print(f"   {t:<24}{c}")
        print("-" * 70, flush=True)

    with TalisharEnv(args.url, timeout=60.0) as env:
      try:
        for g in range(args.games):
            dA = decks[(2 * g) % len(decks)]
            dB = decks[(2 * g + 1) % len(decks)]
            f1 = _DECK_DIR / f"audit_p1_{g}.json"; dA.save_json(str(f1))
            f2 = _DECK_DIR / f"audit_p2_{g}.json"; dB.save_json(str(f2))
            init = env.reset(hero1=dA.hero, hero2=dB.hero,
                             deck1=f"{_DECK_REL_DIR}/{f1.name}",
                             deck2=f"{_DECK_REL_DIR}/{f2.name}",
                             seed=1000 + g)
            state, legal = init.state, init.legal_actions
            steps = 0
            dead: set = set()
            cancel = 0
            while not env.done and steps < args.step_cap:
                prio = int(state.get("priority_player", 0))
                if prio not in (1, 2):
                    break
                if not legal:
                    legal = env.get_actions(refresh=True)
                    if not legal:
                        break
                phase = str(state.get("phase", "") or "<empty>")
                choosable = [a for a in legal if (a.card_id, a.type) not in dead] or legal
                types = [(a.type or "").upper() for a in choosable]
                phase_windows[phase] += 1
                nonpass = [t for t in types if t != "PASS"]
                if not nonpass:
                    phase_onlypass[phase] += 1
                for t in nonpass:
                    phase_types[phase][t] += 1
                    type_total[t] += 1

                chosen = _audit_policy(choosable, rng)
                pre = (state.get("turn"), state.get("phase"), prio)
                try:
                    res = env.step(chosen.action_id)
                except Exception:  # noqa: BLE001
                    break
                post = res.state
                # engine-cancel detection (mirror run_match) so the audit
                # doesn't wedge on an unpayable play
                if (chosen.type or "").upper() != "PASS" and \
                   (post.get("turn"), post.get("phase"), prio) == pre and \
                   len(post.get("players", [{}])) and \
                   _samehp(state, post):
                    dead.add((chosen.card_id, chosen.type))
                    cancel += 1
                    if cancel >= 25:
                        break
                else:
                    dead.clear(); cancel = 0
                state, legal = res.state, res.legal_actions
                steps += 1
            print(f"  [audit] game {g} done: {steps} steps", flush=True)
      finally:
        report()
        print("NOTE: a phase that is ~100% only-PASS with many windows is a likely")
        print("serializer gap (engine gave priority but no action was enumerated).")
    return 0


def _samehp(s0: dict, s1: dict) -> bool:
    def hp(s):
        return tuple(sorted((int(p.get("player_id", 0)), p.get("health"))
                            for p in s.get("players", []) or []))
    return hp(s0) == hp(s1)


if __name__ == "__main__":
    raise SystemExit(main())
