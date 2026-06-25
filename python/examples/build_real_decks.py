"""Build playable decks from the real Draftmancer logs in
real_draft_references/. Each DraftLog (v2.1) records every user's picks as
indices into the booster they saw; the picked pool is booster[pick]. OMN heroes
aren't in the booster, so we infer the hero by drafted-class plurality (the same
cascade the pipeline uses) and pair the signature weapon, then build a legal
deck with the heuristic builder. Writes deck JSONs to decks/_real_draft/.

    python -m python.examples.build_real_decks
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.draft.draftmancer import parse_draftmancer, slugify  # noqa: E402
from python.draft.format import (  # noqa: E402
    CLASS_HERO, DECISIVE_CLASSES, HERO_WEAPONS, LEGAL_HEROES,
)
from python.deckbuilding.builder import HeuristicDeckBuilder  # noqa: E402
from python.deckbuilding.card_catalog import load_card_catalog  # noqa: E402

CUBE = PROJECT_ROOT / "OMN_Draft_3.5.txt"
REFS = PROJECT_ROOT / "real_draft_references"
OUT = PROJECT_ROOT / "decks" / "_real_draft"


def log_to_pools(path: str) -> dict[str, list[str]]:
    d = json.load(open(path, encoding="utf-8"))
    pools: dict[str, list[str]] = {}
    for uid, u in (d.get("users") or {}).items():
        if u.get("isBot"):
            continue
        pool: list[str] = []
        for pk in u.get("picks", []) or []:
            booster = pk.get("booster", []) or []
            for idx in pk.get("pick", []) or []:
                if 0 <= idx < len(booster):
                    pool.append(slugify(booster[idx].split("_custom_")[0]))
        pools[u.get("userName") or uid] = pool
    return pools


def infer_hero(pool: list[str], card_classes: dict[str, set[str]]) -> str:
    counts: Counter = Counter()
    for c in pool:
        for cls in card_classes.get(c, ()):
            if cls in DECISIVE_CLASSES:
                counts[cls] += 1
    if counts:
        ranked = counts.most_common()
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            h = CLASS_HERO.get(ranked[0][0])
            if h in LEGAL_HEROES:
                return h
    return LEGAL_HEROES[0]


def main() -> int:
    cube = parse_draftmancer(str(CUBE))
    card_classes = cube.class_map()

    files = sorted(glob.glob(str(REFS / "*.txt")))
    if not files:
        print("no draft logs in real_draft_references/"); return 1

    universe: set[str] = set(LEGAL_HEROES) | set(HERO_WEAPONS.values()) | {"cracked_bauble_yellow"}
    parsed = {}
    for f in files:
        parsed[f] = log_to_pools(f)
        for pool in parsed[f].values():
            universe |= set(pool)
    catalog = load_card_catalog(talishar_root=PROJECT_ROOT / "talishar", pack_universe=universe)

    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    heroes: Counter = Counter()
    for f in files:
        tag = re.sub(r"[^A-Za-z0-9]", "_", Path(f).stem)[:24]
        for uname, pool in parsed[f].items():
            if len(pool) < 30:   # a full OMN draft is ~40 picks; skip stubs
                continue
            hero = infer_hero(pool, card_classes)
            weapon = HERO_WEAPONS[hero]
            builder = HeuristicDeckBuilder(catalog=catalog, card_classes=card_classes,
                                           seed=(hash(uname) & 0xFFFF))
            try:
                deck = builder.build_deck([hero, weapon, *pool])
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {tag}/{uname}: {e!r}")
                continue
            out = {"hero": deck.hero, "equipment": [deck.weapon, *deck.equipment],
                   "deck": deck.deck}
            safe = re.sub(r"[^A-Za-z0-9]", "_", uname)[:18]
            json.dump(out, open(OUT / f"{tag}__{safe}.json", "w"), indent=1)
            heroes[deck.hero] += 1
            n += 1
    print(f"built {n} real decks -> {OUT}")
    print("hero distribution:", dict(heroes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
