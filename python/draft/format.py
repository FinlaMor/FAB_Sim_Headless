"""Static metadata for the Omens of the Third Age limited format.

Card IDs use Talishar's lowercase_underscored convention. The exact IDs
for hero/weapon objects must match the entries Talishar's
CardDictionaries register; if a deckbuilder/draft tool emits a different
slug, ``card_catalog.alias()`` will normalise it.

If you patch in a new legal hero/weapon, append it here. The simulator
and deck builder both source legality from these constants — there is
exactly one place to edit.
"""

from __future__ import annotations

from typing import Final

FORMAT_NAME: Final[str] = "Omens of the Third Age"
FORMAT_CODE: Final[str] = "oma"

# Heroes that may be picked during draft and registered for deck building.
LEGAL_HEROES: Final[tuple[str, ...]] = (
    "zyggy",
    "aurora_emissary_of_lightning",
    "oscilio_scion_of_the_third_age",
)

# Signature weapons. Each weapon is bound to exactly one hero; the deck
# builder enforces the correspondence in `deckbuilding.legality`.
#
# NOTE: slugs match Talishar's `Classes/CardObjects/OMNCards.php`
# verbatim. The Scorpio weapon class is `scorpio_comet_tail` (singular),
# not "scorpio_comet_tails" — confirmed via grep on the class definitions.
LEGAL_WEAPONS: Final[tuple[str, ...]] = (
    "aphrodias",
    "scorpio_comet_tail",
    "volzar_meteor_storm",
)

# Hero -> signature weapon mapping. Used by the heuristic deck builder and
# by analytics that bucket performance by weapon archetype.
HERO_WEAPONS: Final[dict[str, str]] = {
    "zyggy":                              "aphrodias",
    "aurora_emissary_of_lightning":       "scorpio_comet_tail",
    "oscilio_scion_of_the_third_age":     "volzar_meteor_storm",
}

# Hero -> talents/class. Used by the cascade hero-assignment logic in
# pipeline.default_hero_assignment to map "this seat drafted mostly
# Wizard cards" -> the Wizard hero.
#
# Source: the OMN draft cube's CustomCards "type" field, which carries
# tags like "Lightning, Wizard, Action, Attack" for Wizard cards and
# "Lightning, Illusionist, Equipment, Arms" for an Illusionist arm.
HERO_CLASS: Final[dict[str, str]] = {
    "zyggy":                              "Illusionist",
    "aurora_emissary_of_lightning":       "Wizard",
    "oscilio_scion_of_the_third_age":     "Runeblade",
}

# Hero -> talent. All three OMN heroes share the Lightning talent.
# Cards a hero may include must satisfy the talent check below.
HERO_TALENT: Final[dict[str, str]] = {
    "zyggy":                              "Lightning",
    "aurora_emissary_of_lightning":       "Lightning",
    "oscilio_scion_of_the_third_age":     "Lightning",
}

# Class -> hero (inverse of HERO_CLASS). Convenience for the cascade.
CLASS_HERO: Final[dict[str, str]] = {v: k for k, v in HERO_CLASS.items()}

# Canonical FaB class tags. Used by `card_legality_tags` to split a
# card's type field into {classes, talents, other}. Anything in the
# type field not in CLASS_TAGS / TALENT_TAGS is treated as a subtype
# (Action / Attack / Aura / Equipment / Head / Chest / Arms / Legs / ...).
CLASS_TAGS: Final[frozenset[str]] = frozenset({
    "Illusionist", "Wizard", "Runeblade", "Guardian", "Warrior", "Brute",
    "Ninja", "Ranger", "Mechanologist", "Merchant", "Pirate", "Assassin",
    "Druid", "Bard", "Necromancer", "Shapeshifter", "Adjudicator",
})

# Canonical FaB talent tags. Cards with NO talent are treated as
# universally talent-legal; same for cards with NO class (Generic).
TALENT_TAGS: Final[frozenset[str]] = frozenset({
    "Lightning", "Ice", "Earth", "Light", "Shadow", "Elemental", "Draconic",
})

# Equipment-slot subtype tags. A card with one of these subtypes lives in
# the corresponding character-equipment slot, not in the deck. The deck
# builder slots one per zone and keeps non-equipment cards for the main
# 30-card stack. Talishar's gamestate stride pulls them off the
# character line in JoinGame's order: head, chest, arms, legs.
EQUIPMENT_SLOT_TAGS: Final[tuple[str, ...]] = ("Head", "Chest", "Arms", "Legs")
EQUIPMENT_TAG: Final[str] = "Equipment"

# Slug Talishar uses for the limited filler card (Welcome to Rathe set,
# yellow pitch, GENERIC class). The OMN deck builder falls back to this
# whenever the legal pool can't hit DECK_SIZE_EXACT.
CRACKED_BAUBLE_SLUG: Final[str] = "cracked_bauble_yellow"

# Cards whose Talishar OMN implementation is known to crash on
# resolution (typically because the relevant $CS_* class-state global
# is not registered in talishar/Constants.php::ResetMainClassState or
# the ClassState init line in MenuFiles/StartHelper.php is short of an
# entry, leading to `IncrementClassState(player, NULL)` -> "string + int"
# TypeError).
#
# The deck builder excludes these from the legal pool so games we run
# can play to completion; once a card's upstream support lands, remove
# it here. Track new crashes in `outputs/engine_card_bugs.log`.
BROKEN_CARDS: Final[frozenset[str]] = frozenset({
    "nourishing_glow_blue",
    "nourishing_glow_yellow",
    "nourishing_glow_red",
    "heaven_s_claws_red",
    "heaven_s_claws_yellow",
    "heaven_s_claws_blue",
})

# Classes the cascade considers when counting drafted cards. Anything
# outside this set (e.g. "Lightning" alone, "Generic") is treated as
# uninformative — we don't want a Lightning-soup pool to dominate.
DECISIVE_CLASSES: Final[tuple[str, ...]] = tuple(HERO_CLASS.values())

# Set-code prefixes used to filter the legal card pool. Patch in additional
# small-print supplemental sets if the format expands.
LEGAL_SET_PREFIXES: Final[tuple[str, ...]] = ("oma",)

# Limited deck minimums (Flesh and Blood Living Legend deck rules). The
# real rules engine (Talishar) will catch any violation when the first
# card is played, but the deck builder also enforces this client-side
# for faster feedback during training.
MIN_DECK_SIZE: Final[int] = 30      # Limited minimum (kept for back-compat).
MAX_DECK_SIZE: Final[int] = 60
# OMN draft format: decks must contain EXACTLY this many cards. The
# heuristic deck builder enforces equality, falling back to the
# `CRACKED_BAUBLE_SLUG` filler if the legal pool runs short.
DECK_SIZE_EXACT: Final[int] = 30
PACKS_PER_PLAYER: Final[int] = 3
CARDS_PER_PACK_DEFAULT: Final[int] = 13   # OMA boosters are 13-card; override per pack.
