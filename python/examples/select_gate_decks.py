"""Pick the TOP-N human-drafted decks (by the heuristic builder's deck-quality
score) and write them to outputs/gate_decks/ for the promotion gate gauntlet.
Reuses the real-deck build logic; ranks by deck.evaluation.overall_score."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.examples.build_real_decks import log_to_pools, infer_hero  # noqa: E402
from python.draft.draftmancer import parse_draftmancer  # noqa: E402
from python.draft.format import HERO_WEAPONS, LEGAL_HEROES  # noqa: E402
from python.deckbuilding.builder import HeuristicDeckBuilder  # noqa: E402
from python.deckbuilding.card_catalog import load_card_catalog  # noqa: E402

CUBE = PROJECT_ROOT / "OMN_Draft_3.5.txt"
REFS = PROJECT_ROOT / "real_draft_references"
OUT = PROJECT_ROOT / "outputs" / "gate_decks"
TOP_N = 50


def main() -> int:
    cube = parse_draftmancer(str(CUBE))
    card_classes = cube.class_map()
    files = sorted(glob.glob(str(REFS / "*.txt")))

    universe = set(LEGAL_HEROES) | set(HERO_WEAPONS.values()) | {"cracked_bauble_yellow"}
    parsed = {f: log_to_pools(f) for f in files}
    for pools in parsed.values():
        for pool in pools.values():
            universe |= set(pool)
    catalog = load_card_catalog(talishar_root=PROJECT_ROOT / "talishar", pack_universe=universe)

    scored = []
    for f in files:
        for uname, pool in parsed[f].items():
            if len(pool) < 30:
                continue
            hero = infer_hero(pool, card_classes)
            weapon = HERO_WEAPONS[hero]
            builder = HeuristicDeckBuilder(catalog=catalog, card_classes=card_classes,
                                           seed=(hash(uname) & 0xFFFF))
            try:
                deck = builder.build_deck([hero, weapon, *pool])
            except Exception:  # noqa: BLE001
                continue
            scored.append((float(deck.evaluation.overall_score), uname, deck))

    scored.sort(key=lambda t: t[0], reverse=True)
    if not scored:
        print("no decks built"); return 1

    # reset the gate dir
    if OUT.exists():
        for p in OUT.glob("*.json"):
            p.unlink()
    OUT.mkdir(parents=True, exist_ok=True)

    top = scored[:TOP_N]
    from collections import Counter
    heroes: Counter = Counter()
    for i, (score, uname, deck) in enumerate(top):
        out = {"hero": deck.hero, "equipment": [deck.weapon, *deck.equipment], "deck": deck.deck}
        json.dump(out, open(OUT / f"gate_{i:02d}_{deck.hero[:6]}.json", "w"), indent=1)
        heroes[deck.hero] += 1

    print(f"selected top {len(top)} of {len(scored)} decks -> {OUT}")
    print(f"score range: top={top[0][0]:.2f}  cut={top[-1][0]:.2f}  "
          f"overall min={scored[-1][0]:.2f} max={scored[0][0]:.2f}")
    print("hero mix in gate:", dict(heroes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
