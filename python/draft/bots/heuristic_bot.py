"""Rule-of-thumb heuristic draft bot.

This bot is intentionally a SCAFFOLD — it shows where archetype, signal,
and curve logic plug in. Replace ``_score`` with a real evaluator trained
on tournament outcomes once you have data.

What it does today
------------------
* Picks heroes / signature weapons at top priority if seen.
* Locks an archetype after pick 3 (whichever pitch colour dominates the
  drafted pool) and weights cards by colour match.
* Soft penalty for cards already 4-of in the drafted pool (curve sanity).
* Tiny RNG jitter to break ties stochastically.

The bot never references absolute card IDs beyond what's in
``python.draft.format`` — patching a new format only requires editing
that module.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from ..format import (
    CLASS_HERO, DECISIVE_CLASSES, HERO_CLASS, HERO_WEAPONS,
    LEGAL_HEROES, LEGAL_WEAPONS,
)
from .base import DraftBot, DraftDecision, DraftPodView


# Pitch colour heuristic: card IDs that end in _red / _yellow / _blue.
def _pitch(card_id: str) -> str | None:
    for suffix in ("_red", "_yellow", "_blue"):
        if card_id.endswith(suffix):
            return suffix[1:]
    return None


class HeuristicDraftBot(DraftBot):
    name = "draft-heuristic"

    def __init__(
        self,
        *,
        seed: int | None = None,
        hero_preference: tuple[str, ...] = LEGAL_HEROES,
    ) -> None:
        self._rng = random.Random(seed)
        self.hero_preference = tuple(hero_preference)

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng.seed(seed)

    def choose_card(
        self,
        pack: tuple[str, ...],
        drafted_cards: tuple[str, ...],
        seat_position: int,
        pick_number: int,
        pack_number: int,
        pod_state: DraftPodView,
    ) -> str | DraftDecision:
        if not pack:
            raise RuntimeError("HeuristicDraftBot received empty pack")

        pitch_dist = Counter(_pitch(c) for c in drafted_cards if _pitch(c))
        # Dominant colour locked after we have at least one of each? Use top-1.
        dominant_colour = pitch_dist.most_common(1)[0][0] if pitch_dist else None
        # Hero lock: any hero already drafted?
        hero_locked = next((h for h in drafted_cards if h in LEGAL_HEROES), None)
        weapon_locked = next((w for w in drafted_cards if w in LEGAL_WEAPONS), None)

        scored: list[tuple[float, str]] = []
        for card in pack:
            s = self._score(
                card,
                drafted_cards=drafted_cards,
                hero_locked=hero_locked,
                weapon_locked=weapon_locked,
                dominant_colour=dominant_colour,
                pod_state=pod_state,
            )
            s += self._rng.random() * 1e-6
            scored.append((s, card))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        chosen = scored[0][1]
        return DraftDecision(
            card_id=chosen,
            info={
                "policy": "heuristic",
                "top_score": scored[0][0],
                "hero_locked": hero_locked,
                "weapon_locked": weapon_locked,
                "dominant_colour": dominant_colour,
                "pack_size": len(pack),
            },
        )

    # ------------------------------------------------------------------
    # Advisor API: score every card in the pack (higher = better pick).
    # Used by the interactive draft assistant.
    # ------------------------------------------------------------------
    def score_cards(
        self,
        pack: tuple[str, ...],
        drafted_cards: tuple[str, ...],
        seat_position: int,
        pick_number: int,
        pack_number: int,
        pod_state: DraftPodView,
    ) -> dict[str, float]:
        pitch_dist = Counter(_pitch(c) for c in drafted_cards if _pitch(c))
        dominant_colour = pitch_dist.most_common(1)[0][0] if pitch_dist else None
        hero_locked = next((h for h in drafted_cards if h in LEGAL_HEROES), None)
        weapon_locked = next((w for w in drafted_cards if w in LEGAL_WEAPONS), None)
        out: dict[str, float] = {}
        for card in pack:
            out[card] = self._score(
                card, drafted_cards=drafted_cards, hero_locked=hero_locked,
                weapon_locked=weapon_locked, dominant_colour=dominant_colour,
                pod_state=pod_state,
            )
        return out

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _score(
        self,
        card: str,
        *,
        drafted_cards: tuple[str, ...],
        hero_locked: str | None,
        weapon_locked: str | None,
        dominant_colour: str | None,
        pod_state: DraftPodView,
    ) -> float:
        # 1. Hero / weapon priority — pick if we don't have one yet.
        if card in LEGAL_HEROES and hero_locked is None:
            return 100.0 + self.hero_preference.index(card) * -0.1
        if card in LEGAL_WEAPONS:
            if weapon_locked is None and hero_locked is not None:
                if HERO_WEAPONS.get(hero_locked) == card:
                    return 95.0
                return 80.0
            return 50.0

        # 2. Pitch colour alignment.
        colour = _pitch(card)
        if dominant_colour and colour and colour == dominant_colour:
            score = 10.0
        elif colour and dominant_colour is None:
            score = 5.0
        else:
            score = 3.0

        # 3. Soft cap: avoid 5+ copies of the same card (limited curve).
        n_copies = sum(1 for c in drafted_cards if c == card)
        if n_copies >= 3:
            score -= 4.0

        # 4. Wheel awareness: late-pack picks should chase open archetypes
        #    rather than splash. The signal here is crude (just pick number).
        if pod_state.pick_number > 9:
            score += 0.5

        # 5. SIGNAL: read the neighbours' pools. A pitch colour the players
        #    around you are NOT loading up on is an open lane (bonus); a
        #    colour they're heavy in is contested and will dry up (penalty).
        if colour:
            neigh = list(pod_state.left_neighbour_drafted) + list(pod_state.right_neighbour_drafted)
            ncol = Counter(_pitch(c) for c in neigh if _pitch(c))
            total = sum(ncol.values())
            if total >= 3:
                share = ncol.get(colour, 0) / total
                score += (0.33 - share) * 3.0  # ~ +1 fully open .. -2 fully contested

        return score

    # ------------------------------------------------------------------
    # Hero pick: class-aware, defers to cascade on a tie.
    # ------------------------------------------------------------------
    def pick_hero(
        self,
        drafted_cards: tuple[str, ...],
        available_heroes: tuple[str, ...],
        card_classes: dict[str, set[str]],
    ) -> str | None:
        if not available_heroes:
            return None
        # Honour the bot's preference order over any heroes the seat
        # already explicitly drafted (rare in formats where heroes are
        # in the booster, but cheap to support).
        for hero in drafted_cards:
            if hero in available_heroes:
                return hero
        # Count drafted cards by decisive class. The cascade owner does
        # this same count if we return None, so returning a top-class
        # hero here is just a strong hint, not a hard override.
        counts: Counter[str] = Counter()
        for c in drafted_cards:
            for cls in card_classes.get(c, ()):
                if cls in DECISIVE_CLASSES:
                    counts[cls] += 1
        if not counts:
            return None
        ranked = counts.most_common()
        # Strict tie -> let the cascade pick (returns None on ambiguity).
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        top_class = ranked[0][0]
        candidate = CLASS_HERO.get(top_class)
        if candidate in available_heroes:
            return candidate
        return None
