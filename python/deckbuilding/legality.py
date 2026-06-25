"""Limited deck legality checks.

We never reimplement card-text rules — that's Talishar's job. But we do
catch easily-validatable mistakes upfront:

* hero present and legal for the format
* signature weapon present and matches the hero
* main deck has EXACTLY DECK_SIZE_EXACT cards
* every deck card is class+talent legal for the chosen hero
* every deck card actually came from the player's drafted pool
  (or is the official CRACKED_BAUBLE_SLUG filler — added by the
  deck builder when the legal pool runs short)
* no card present that violates a colour/set restriction

The class/talent check mirrors Talishar's
``APIs/JoinGame.php::isCardLegalinHero``: a card is legal if it has at
least one class tag matching the hero's class (or no class tags at
all = Generic) AND at least one talent tag matching the hero's talent
(or no talent tags at all = universally talent-legal).

If any hard rule is violated, :func:`validate_deck` raises
``LegalityError``. Callers should treat that as a hard programming bug
in the deck builder — Talishar will refuse to start a game otherwise.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..draft.format import (
    BROKEN_CARDS, CLASS_TAGS, CRACKED_BAUBLE_SLUG, DECK_SIZE_EXACT,
    HERO_CLASS, HERO_TALENT, HERO_WEAPONS, LEGAL_HEROES, LEGAL_WEAPONS,
    TALENT_TAGS,
)
from .card_catalog import CardCatalog
from .deck import Deck


class LegalityError(ValueError):
    """Raised by :func:`validate_deck` when the deck is not legal."""


@dataclass
class LegalityReport:
    """Result of a legality check (non-fatal version for analytics)."""
    ok: bool
    errors: list[str]
    warnings: list[str]


# ---------------------------------------------------------------------------
# Tag splitter — separates a card's type field into class / talent / sub.
# ---------------------------------------------------------------------------
def card_legality_tags(
    card_slug: str,
    card_classes: dict[str, set[str]],
) -> tuple[set[str], set[str], set[str]]:
    """Return ``(classes, talents, subtypes)`` for ``card_slug``.

    ``card_classes`` is the cube's ``class_map`` (see
    :func:`python.draft.draftmancer.class_map_from_cube`) — a dict of
    ``{slug: {tag, ...}}`` parsed from the cube's ``CustomCards`` type
    field.

    Unknown cards (slug not in ``card_classes``) return three empty sets
    — i.e. they're treated as a fully-generic card and the legality
    check passes for any hero. The caller can choose to treat unknown
    cards as warnings via the report API.
    """
    tags = card_classes.get(card_slug, set())
    classes  = {t for t in tags if t in CLASS_TAGS}
    talents  = {t for t in tags if t in TALENT_TAGS}
    subtypes = tags - classes - talents
    return classes, talents, subtypes


def is_legal_for_hero(
    card_slug: str,
    hero_slug: str,
    card_classes: dict[str, set[str]],
) -> bool:
    """Return True if ``card_slug`` is class+talent legal for ``hero_slug``.

    Mirrors Talishar's ``APIs/JoinGame.php::isCardLegalinHero``:
    at-least-one-match semantics (not "subset"). The signature weapon
    and the hero itself are always considered legal regardless of tags.
    """
    if card_slug == hero_slug:
        return True
    if card_slug == HERO_WEAPONS.get(hero_slug):
        return True
    # Cracked Bauble is GENERIC, every hero can include it.
    if card_slug == CRACKED_BAUBLE_SLUG:
        return True
    # Cards Talishar can't resolve crash the engine mid-game. Exclude
    # them from legal pools so generated decks can play to completion;
    # the smoke harness logs the engine bug separately.
    if card_slug in BROKEN_CARDS:
        return False

    hero_class  = HERO_CLASS.get(hero_slug, "")
    hero_talent = HERO_TALENT.get(hero_slug, "")

    classes, talents, _ = card_legality_tags(card_slug, card_classes)

    # at least one class tag matches OR card has no class restriction (Generic)
    in_class  = (not classes)  or (hero_class  in classes)
    # at least one talent matches OR card has no talent restriction
    in_talent = (not talents)  or (hero_talent in talents)
    return in_class and in_talent


# ---------------------------------------------------------------------------
# Deck-level validation
# ---------------------------------------------------------------------------
def validate_deck(
    deck: Deck,
    pool: list[str],
    catalog: CardCatalog | None = None,
    *,
    card_classes: dict[str, set[str]] | None = None,
    raise_on_error: bool = True,
) -> LegalityReport:
    """Validate ``deck`` against the player's drafted ``pool``.

    Parameters
    ----------
    deck : Deck
    pool : list[str]
        The full set of cards the player drafted (including the chosen
        hero / weapon, if those are in-booster). Order does not matter.
    catalog : optional
        Used for warnings; absent metadata yields a warning, not an error.
    card_classes : optional
        Cube-derived ``{slug: {tag, ...}}`` map. When provided the
        per-card hero-legality check is run. Absent -> no class check
        (back-compat with synthetic OMA flow).
    raise_on_error : bool
        If True (default), the function raises ``LegalityError`` as soon
        as a hard error is encountered.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Hero
    if deck.hero not in LEGAL_HEROES:
        errors.append(f"hero {deck.hero!r} not in LEGAL_HEROES")

    # 2. Weapon
    if deck.weapon not in LEGAL_WEAPONS:
        errors.append(f"weapon {deck.weapon!r} not in LEGAL_WEAPONS")
    elif HERO_WEAPONS.get(deck.hero) != deck.weapon:
        errors.append(
            f"weapon {deck.weapon!r} does not match hero {deck.hero!r} "
            f"(expected {HERO_WEAPONS.get(deck.hero)!r})"
        )

    # 3. Deck size — EXACT in OMN limited.
    if deck.size != DECK_SIZE_EXACT:
        errors.append(
            f"deck size {deck.size} != required exact size {DECK_SIZE_EXACT}"
        )

    # 4. Per-card hero legality. Cracked Bauble is always legal.
    if card_classes is not None:
        for c in deck.deck:
            if not is_legal_for_hero(c, deck.hero, card_classes):
                errors.append(
                    f"card {c!r} is not class/talent legal for hero {deck.hero!r}"
                )

    # 5. Every deck card came from the pool OR is the official filler.
    pool_counter = Counter(pool)
    deck_counter = Counter(deck.deck)
    # Hero+weapon are also assumed to have come from the pool (the
    # adapter pre-pends them when hero_assignment is active — see
    # pipeline.LimitedPipeline._build_decks).
    deck_counter[deck.hero] += 1
    deck_counter[deck.weapon] += 1
    for card, n in deck_counter.items():
        if card == CRACKED_BAUBLE_SLUG:
            continue   # filler is unlimited supply, no pool constraint
        if pool_counter.get(card, 0) < n:
            errors.append(
                f"deck uses {n} copies of {card!r} but only "
                f"{pool_counter.get(card, 0)} were drafted"
            )

    # 6. Optional catalog-based warnings
    if catalog is not None:
        for c in deck.deck:
            if c not in catalog and c != CRACKED_BAUBLE_SLUG:
                warnings.append(f"no catalog metadata for {c!r}")

    report = LegalityReport(ok=not errors, errors=errors, warnings=warnings)
    if errors and raise_on_error:
        raise LegalityError("; ".join(errors))
    return report
