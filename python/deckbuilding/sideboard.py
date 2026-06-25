"""CC sideboard / equipment resolver.

A scraped fabrary CC deck is a *registered pool* (~80 cards: a maindeck under
`quantity` plus a sideboard, and equipment options). Talishar will not start a
game without a CC-legal deck, so this resolves a pool into a concrete legal
game deck (`hero` + `equipment` + 60-card `deck`) the engine adapter loads.

Two jobs:
  1. **Deck size** — keep the registered maindeck, top up to 60 from the
     sideboard (respecting 3-copies-per-name-and-pitch).
  2. **Equipment selection** — pick a legal loadout (<=1 per slot, weapon(s)).
     This is where Mechanologist *Evo* parts matter: an "Equipment" card that is
     ALSO a playable type (Action/Instant — e.g. Evo pieces) is really a
     maindeck card, not a starting-equipment slot, so it's moved into the deck.
     Card types/subtypes come from slug_index.json (native fabrary card data).

v1: matchup-agnostic. `opp_hero` is reserved for the v2 matchup-aware version
(pick equipment + tech cards against the revealed opponent).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

CC_MIN_DECK = 60
# CR: up to 3 copies of any card, where two cards are "copies" only if they
# share BOTH name AND pitch. The Talishar slug encodes name+pitch, so the
# per-slug count is the copy identity — do NOT strip the pitch suffix.
MAX_COPIES = 3

_REPO = Path(__file__).resolve().parents[2]
_SLUG_INDEX = _REPO / "slug_index.json"
_ARMOR_SLOTS = ("Head", "Chest", "Arms", "Legs", "Off-Hand", "Quiver")
_idx_cache: dict | None = None


def _by_slug() -> dict:
    global _idx_cache
    if _idx_cache is None:
        _idx_cache = (json.loads(_SLUG_INDEX.read_text(encoding="utf-8"))["by_slug"]
                      if _SLUG_INDEX.is_file() else {})
    return _idx_cache


def _meta(slug: str) -> dict:
    """Card metadata from slug_index.json. Deck slugs are talishar underscore
    format; slug_index keys are native fabrary dashes — try both."""
    idx = _by_slug()
    return idx.get(slug.replace("_", "-")) or idx.get(slug) or {}


def _hero_classes(hero_slug: str) -> set[str]:
    return set(_meta(hero_slug).get("classes") or [])


_class_tags_cache: set[str] | None = None


def _all_class_tags() -> set[str]:
    """Every FaB class tag, derived from the hero cards in slug_index."""
    global _class_tags_cache
    if _class_tags_cache is None:
        _class_tags_cache = {t for c in _by_slug().values()
                             if "Hero" in (c.get("types") or [])
                             for t in (c.get("classes") or [])}
    return _class_tags_cache


def _matchup_classes(m: dict) -> set[str]:
    """The opponent class(es) a matchup targets: from its linked hero ids when
    present, else parsed from a class tag in its free-text name."""
    cls: set[str] = set()
    for h in m.get("heroes") or []:
        cls |= _hero_classes(h)
    if not cls:
        nl = str(m.get("name") or "").lower()
        cls = {t for t in _all_class_tags() if t.lower() in nl}
    return cls


def _is_starting_equipment(slug: str) -> bool:
    """True iff the card is real starting equipment (line 1), i.e. its types are
    only Equipment/Weapon. Evo/instant 'equipment' (also Action/Instant) is a
    deck card, not a slot."""
    types = _meta(slug).get("types") or []
    return bool(types) and set(types) <= {"Equipment", "Weapon"}


def _slot(slug: str) -> str | None:
    meta = _meta(slug)
    subs = set(meta.get("subtypes") or [])
    for s in _ARMOR_SLOTS:
        if s in subs:
            return s
    if "Weapon" in (meta.get("types") or []):
        return "Weapon"
    return None


def _is_unlimited(slug: str) -> bool:
    """Cards with the 'Unlimited' designation are exempt from the 3-copy rule
    (e.g. Copper Cog), so a legal deck can run any number of them."""
    meta = _meta(slug)
    return ("Unlimited" in (meta.get("keywords") or [])
            or "**Unlimited**" in str(meta.get("functionalText") or ""))


def _short_name(hero_slug: str) -> str:
    """The hero's display short-name from slug_index (e.g. 'Boltyn'), used to
    fuzzy-match free-text matchup labels."""
    return str(_meta(hero_slug).get("hero") or hero_slug.split("_")[0]).lower()


def pick_matchup(pool: dict, opp_hero: str | None) -> str | None:
    """Find the author's matchupId for the opponent hero, most-specific first:
      1. exact opponent-hero match (heroIdentifiers)
      2. fuzzy match on the free-text matchup name (hero's short name)
      3. class-similarity fallback — the author's nearest matchup by opponent
         CLASS, among matchups that actually carry tech overrides. This closes
         the coverage gap so a novel opponent still gets archetype-appropriate
         sideboarding instead of the plain default."""
    if not opp_hero:
        return None
    for m in pool.get("matchups") or []:
        if opp_hero in (m.get("heroes") or []):
            return m.get("matchupId")
    short = _short_name(opp_hero)
    for m in pool.get("matchups") or []:
        if short and short in str(m.get("name") or "").lower():
            return m.get("matchupId")
    # 3. nearest by class, restricted to matchups with actual overrides.
    opp_cls = _hero_classes(opp_hero)
    mq = pool.get("matchup_quantities") or {}
    best_id, best_score = None, 0
    for m in pool.get("matchups") or []:
        mid = m.get("matchupId")
        if not mq.get(mid):
            continue  # no tech to contribute -> useless as a fallback
        score = len(opp_cls & _matchup_classes(m))
        if score > best_score:
            best_id, best_score = mid, score
    return best_id


def _select_equipment(candidates: list[str]) -> list[str]:
    """Legal loadout from starting-equipment candidates: <=1 per recognized
    armor slot, up to 2 weapons, unknown-slot pieces kept; de-duped by slug."""
    equipment: list[str] = []
    used_slots: set[str] = set()
    weapons = 0
    for slug in candidates:
        if slug in equipment:
            continue
        slot = _slot(slug)
        if slot in _ARMOR_SLOTS:
            if slot in used_slots:
                continue
            used_slots.add(slot)
        elif slot == "Weapon":
            if weapons >= 2:
                continue
            weapons += 1
        equipment.append(slug)
    return equipment


def resolve(pool: dict, target: int = CC_MIN_DECK, opp_hero: str | None = None,
            overrides: dict | None = None) -> dict:
    """Pool dict (from fabrary convert_deck) -> playable engine deck dict
    {hero, equipment, deck, matchup}.

    Per-card maindeck count `overrides` come from (in priority): an explicit
    `overrides` arg (e.g. the BC sideboard model's predictions), else the
    author's registered matchup that best fits `opp_hero` (see pick_matchup),
    else none (base/default loadout). Either way the result is made CC-legal
    (Evo->deck, <=1 per slot, 60-card maindeck, copy limit w/ Unlimited)."""
    hero = pool["hero"]
    if overrides is not None:
        mid = "model"
    else:
        mid = pick_matchup(pool, opp_hero)
        overrides = (pool.get("matchup_quantities") or {}).get(mid, {}) if mid else {}

    # Base registered counts (maindeck + equipment), then apply matchup deltas.
    counts: Counter = Counter(pool.get("deck") or []) + Counter(pool.get("equipment") or [])
    for slug, qty in overrides.items():
        counts[slug] = int(qty)        # override (0 = cut for this matchup)

    # Split into starting-equipment candidates vs maindeck cards (Evo/instant
    # "equipment" is not pure Equipment/Weapon -> falls into the deck).
    deck: list[str] = []
    start_candidates: list[str] = []
    for slug, c in counts.items():
        if c <= 0:
            continue
        if _is_starting_equipment(slug):
            start_candidates.append(slug)   # 1 per slot anyway; expanded copies irrelevant
        else:
            deck.extend([slug] * c)

    equipment = _select_equipment(start_candidates)
    if not equipment:
        # No starting loadout, two real causes -> "no equipment" illegal:
        #  (a) overrides cut every equipment slug — the BC sideboard model
        #      predicts maindeck-style quantities and can wrongly zero equipment;
        #  (b) the author registered ALL equipment as SIDEBOARD (sideboardQuantity,
        #      maindeck quantity 0) — flexible armor picked per matchup — so the
        #      base `equipment` list is empty.
        # Equipment is a starting SLOT, not a maindeck count, so fall back to the
        # pool's full registered equipment (maindeck + sideboard); _select_equipment
        # then builds a legal <=1-per-slot loadout. If the pool truly registers no
        # equipment at all, it stays [] and cc_legal_issues flags it.
        base_eq = [s for s in ((pool.get("equipment") or []) + (pool.get("sideboard_equipment") or []))
                   if _is_starting_equipment(s)]
        equipment = _select_equipment(base_eq)

    # Top the maindeck up to target from the sideboard (3-per-slug, Unlimited-exempt).
    if len(deck) < target:
        seen = Counter(deck)
        for slug in pool.get("sideboard") or []:
            if len(deck) >= target:
                break
            if seen[slug] >= MAX_COPIES and not _is_unlimited(slug):
                continue
            deck.append(slug)
            seen[slug] += 1

    # Legality backstop: a CC maindeck must reach `target`. If overrides (esp. the
    # BC model) cut so much that even the sideboard top-up can't refill to 60,
    # restore from the pool's registered maindeck — the model's cuts are
    # PREFERENCES, the 60-card minimum is a hard rule. (Equipment never enters
    # the maindeck.) If the whole registered pool still can't reach 60 the deck is
    # genuinely under-registered and cc_legal_issues flags it.
    if len(deck) < target:
        seen = Counter(deck)
        for slug in pool.get("deck") or []:
            if len(deck) >= target:
                break
            if _is_starting_equipment(slug):
                continue
            if seen[slug] >= MAX_COPIES and not _is_unlimited(slug):
                continue
            deck.append(slug)
            seen[slug] += 1

    return {"hero": hero, "equipment": equipment, "deck": deck, "matchup": mid}


def cc_legal_issues(deck: dict) -> list[str]:
    """Hard legality problems that would stop Talishar from starting a game.
    (Per-card hero legality is trusted from the source tournament deck.)"""
    issues: list[str] = []
    if not deck.get("hero"):
        issues.append("missing hero")
    if not deck.get("equipment"):
        issues.append("no equipment")
    n = len(deck.get("deck") or [])
    if n < CC_MIN_DECK:
        issues.append(f"maindeck {n} < {CC_MIN_DECK}")
    over = [f"{s} x{c}" for s, c in Counter(deck.get("deck") or []).items()
            if c > MAX_COPIES and not _is_unlimited(s)]
    if over:
        issues.append(f"over copy limit: {over}")
    slot_counts = Counter(s for s in (_slot(x) for x in deck.get("equipment") or [])
                          if s in _ARMOR_SLOTS)
    dup_slots = [f"{sl} x{c}" for sl, c in slot_counts.items() if c > 1]
    if dup_slots:
        issues.append(f"duplicate equipment slots: {dup_slots}")
    return issues


# ---------------------------------------------------------------------------
# CLI: resolve scraped pools into playable game decks
# ---------------------------------------------------------------------------
def resolve_file(pool_path: str | Path, out_dir: str | Path) -> dict:
    pool = json.loads(Path(pool_path).read_text(encoding="utf-8"))
    game = resolve(pool)
    issues = cc_legal_issues(game)
    out = Path(out_dir) / (Path(pool_path).stem + "_game.json")
    if not issues:
        out.parent.mkdir(parents=True, exist_ok=True)
        game_out = {"hero": game["hero"],
                    "comment": f"Resolved from {Path(pool_path).name} (sideboard bot v1)",
                    "equipment": game["equipment"], "deck": game["deck"]}
        out.write_text(json.dumps(game_out, indent=2), encoding="utf-8")
    return {"pool": Path(pool_path).name, "hero": game["hero"],
            "maindeck": len(game["deck"]), "equipment": len(game["equipment"]),
            "issues": issues, "written": (str(out.relative_to(_REPO)) if not issues else None)}


if __name__ == "__main__":
    import glob
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pools = args or sorted(glob.glob(str(_REPO / "decks" / "cc_*.json")))
    out_dir = _REPO / "decks" / "resolved"
    ok = 0
    for p in pools:
        info = resolve_file(p, out_dir)
        status = f"OK -> {info['written']}" if not info["issues"] else f"ILLEGAL: {info['issues']}"
        ok += not info["issues"]
        print(f"  {info['hero']:<34} main={info['maindeck']:>3} equip={info['equipment']}  {status}")
    print(f"\n{ok}/{len(pools)} resolved to legal game decks")
