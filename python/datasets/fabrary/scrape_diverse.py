"""Pull N ADDITIONAL diverse CC decks per hero we already have.

For each hero with an existing decks/cc_*.json pool, fetch a pool of fabrary CC
candidates and keep the N that are most DIFFERENT from what we already have,
with preference ordered: WEAPON >> SIDEBOARD cards >> EQUIPMENT. Selection is
farthest-point greedy — each pick maximises its minimum weighted distance to the
existing deck(s) AND the decks already picked this run, so the kept set spreads
out instead of clustering.

    python -m python.datasets.fabrary.scrape_diverse [--per-hero 3]
        [--candidates 15] [--hero <slug>] [--dry]

Network-heavy (one API call per candidate, ~0.8s each) — run in background.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

from .client import Fabrary
from .convert import convert_deck, validate, write_deck, talishar_slugs

_REPO = Path(__file__).resolve().parents[3]
_DECKS = _REPO / "decks"
_SLUGIDX = _REPO / "slug_index.json"

# Diversity weights: a different WEAPON is the strongest signal of a different
# deck, then a different SIDEBOARD, then different EQUIPMENT (armor). Each unit
# is one slug of symmetric difference, so weapon outranks any plausible sideboard
# gap and sideboard outranks any equipment gap.
W_WEAPON, W_SIDE, W_EQUIP = 100.0, 1.0, 0.02
MIN_REGISTERED = 75


def _load_slug_meta() -> dict:
    return json.loads(_SLUGIDX.read_text(encoding="utf-8")).get("by_slug", {})


def _types(meta: dict, slug: str) -> list[str]:
    # pool slugs are underscore; slug_index keys are dashed.
    e = meta.get(slug.replace("_", "-")) or {}
    return [str(t).lower() for t in (e.get("types") or [])]


def _deck_sets(pool: dict, meta: dict) -> tuple[frozenset, frozenset, frozenset]:
    """(weapons, armor, sideboard) slug sets for one pool."""
    eq = pool.get("equipment") or []
    weapons, armor = set(), set()
    for s in eq:
        t = _types(meta, s)
        if "weapon" in t:
            weapons.add(s)
        elif "equipment" in t:
            armor.add(s)
    side = set(pool.get("sideboard") or []) | set(pool.get("sideboard_equipment") or [])
    return frozenset(weapons), frozenset(armor), frozenset(side)


def _distance(a: tuple, b: tuple) -> float:
    wa, ea, sa = a
    wb, eb, sb = b
    return (W_WEAPON * len(wa ^ wb)
            + W_SIDE * len(sa ^ sb)
            + W_EQUIP * len(ea ^ eb))


def _existing_by_hero() -> dict[str, list[tuple[str, dict]]]:
    out: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for fp in glob.glob(str(_DECKS / "cc_*.json")):
        p = json.loads(Path(fp).read_text(encoding="utf-8"))
        out[p["hero"]].append((Path(fp).stem[-6:].lower(), p))
    return out


def pull_diverse(per_hero: int = 3, candidates: int = 15, write: bool = True,
                 only_hero: str | None = None) -> dict:
    fab = Fabrary()
    known = talishar_slugs()
    meta = _load_slug_meta()
    existing = _existing_by_hero()
    summary = {"heroes": 0, "added": 0, "short": []}

    heroes = [only_hero] if only_hero else sorted(existing)
    for hero in heroes:
        have = existing.get(hero) or []
        if not have:
            print(f"{hero}: no existing pool, skip", flush=True)
            continue
        summary["heroes"] += 1
        have_ids = {hid for hid, _ in have}
        refs = [_deck_sets(p, meta) for _, p in have]

        try:
            stubs = fab.decks_by_hero(hero.replace("_", "-"), max_decks=candidates * 3)
        except Exception as e:  # noqa: BLE001
            print(f"{hero}: list ERROR {e}", flush=True)
            continue
        cc = [d for d in stubs if d.get("format") == "Classic Constructed"]

        cands: list[tuple[str, dict, tuple]] = []
        for d in cc:
            did = d["deckId"]
            if did[-6:].lower() in have_ids:
                continue
            if len(cands) >= candidates:
                break
            try:
                pool = convert_deck(fab.get_deck(did))
            except Exception as e:  # noqa: BLE001
                print(f"  {hero}: get {did} ERROR {e}", flush=True)
                continue
            if pool["hero"] != hero or validate(pool, known):
                continue
            if pool["registered_total"] < MIN_REGISTERED:
                continue
            cands.append((did, pool, _deck_sets(pool, meta)))

        picked: list[tuple[str, dict, tuple]] = []
        sel_refs = list(refs)
        while cands and len(picked) < per_hero:
            best = max(cands, key=lambda c: min(_distance(c[2], r) for r in sel_refs))
            cands.remove(best)
            picked.append(best)
            sel_refs.append(best[2])

        for did, pool, sets in picked:
            w, a, s = sets
            if write:
                write_deck(pool, _DECKS / f"cc_{hero}_{did[-6:].lower()}.json")
            print(f"  {hero}: +{did[-6:].lower()} weapon={sorted(w)} "
                  f"side={len(s)} armor={len(a)}", flush=True)
        summary["added"] += len(picked)
        if len(picked) < per_hero:
            summary["short"].append(f"{hero}({len(picked)})")
        print(f"{hero}: kept {len(picked)}/{per_hero} (had {len(have)})", flush=True)

    print(f"\n=== added {summary['added']} decks across {summary['heroes']} heroes; "
          f"short of {per_hero}: {summary['short'] or 'none'} ===", flush=True)
    return summary


def _arg(flag: str, default: int) -> int:
    if flag in sys.argv:
        return int(sys.argv[sys.argv.index(flag) + 1])
    return default


if __name__ == "__main__":
    oh = sys.argv[sys.argv.index("--hero") + 1] if "--hero" in sys.argv else None
    pull_diverse(per_hero=_arg("--per-hero", 3), candidates=_arg("--candidates", 15),
                 write="--dry" not in sys.argv, only_hero=oh)
