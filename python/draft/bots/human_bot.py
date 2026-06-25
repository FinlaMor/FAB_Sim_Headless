"""Interactive human draft seat.

Plugs into :class:`DraftSimulator` like any other :class:`DraftBot`, but
``choose_card`` prints the current pack and reads the pick from stdin. The
other seats are filled by ordinary bots, so a person can practice drafting
against them.

``input_fn`` / ``output_fn`` are injectable so the bot can be unit-tested
with scripted picks (and so a future GUI/web front-end can drive it).
"""

from __future__ import annotations

from typing import Any, Callable

from .base import DraftBot, DraftDecision, DraftPodView


class HumanDraftBot(DraftBot):
    name = "human"

    def __init__(
        self,
        *,
        card_info: dict[str, dict[str, Any]] | None = None,
        advisor: Any | None = None,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        # card_info: slug -> {"name": str, "cost": str, "type": str}
        self.card_info = card_info or {}
        # advisor: any object with score_cards(pack, drafted, seat, pick,
        # pack_number, pod_state) -> {slug: float}. When set, the pack is
        # annotated with the advisor's rank/score and its top pick.
        self.advisor = advisor
        self._input = input_fn
        self._output = output_fn

    # ------------------------------------------------------------------
    def _label(self, slug: str) -> str:
        info = self.card_info.get(slug, {})
        name = info.get("name", slug)
        cost = info.get("cost")
        typ = info.get("type", "")
        bits = name
        if cost not in (None, ""):
            bits += f"  ({cost})"
        if typ:
            bits += f"  [{typ}]"
        return bits

    def _show_pool(self, drafted: tuple[str, ...]) -> None:
        if not drafted:
            self._output("  (pool empty)")
            return
        # Group identical cards for a compact summary.
        from collections import Counter
        counts = Counter(drafted)
        lines = []
        for slug, n in sorted(counts.items(), key=lambda kv: self.card_info.get(kv[0], {}).get("name", kv[0])):
            nm = self.card_info.get(slug, {}).get("name", slug)
            lines.append(f"{n}x {nm}")
        self._output("  pool (" + str(len(drafted)) + "): " + ", ".join(lines))

    # ------------------------------------------------------------------
    def choose_card(
        self,
        pack: tuple[str, ...],
        drafted_cards: tuple[str, ...],
        seat_position: int,
        pick_number: int,
        pack_number: int,
        pod_state: DraftPodView,
    ) -> str | DraftDecision:
        o = self._output
        # Optional draft-assistant: rank/score the pack.
        scores: dict[str, float] = {}
        ranks: dict[str, int] = {}
        if self.advisor is not None and hasattr(self.advisor, "score_cards"):
            try:
                scores = self.advisor.score_cards(
                    tuple(pack), tuple(drafted_cards), seat_position,
                    pick_number, pack_number, pod_state) or {}
                for r, slug in enumerate(
                    sorted(pack, key=lambda c: scores.get(c, float("-inf")), reverse=True),
                    start=1,
                ):
                    ranks.setdefault(slug, r)
            except Exception as e:  # noqa: BLE001 — never let the advisor break the draft
                o(f"  (advisor unavailable: {e!r})")

        o("")
        o("=" * 72)
        adv = "  |  advisor: ON" if scores else ""
        o(f" Pack {pack_number}  |  Pick {pick_number}  |  your seat {seat_position}{adv}")
        o("-" * 72)
        for i, slug in enumerate(pack, start=1):
            rank_tag = ""
            if scores:
                star = " *" if ranks.get(slug) == 1 else "  "
                rank_tag = f"  #{ranks.get(slug,'?'):>2} score={scores.get(slug, 0.0):>7.2f}{star}"
            o(f"  [{i:>2}] {self._label(slug):<48}{rank_tag}")
        o("-" * 72)
        if scores:
            best = max(scores, key=scores.get)
            o(f"  advisor's pick: {self.card_info.get(best, {}).get('name', best)}")
        self._show_pool(drafted_cards)
        o("=" * 72)

        while True:
            raw = self._input("Pick a card (number or name fragment): ").strip()
            if not raw:
                continue
            # By index.
            if raw.isdigit():
                idx = int(raw)
                if 1 <= idx <= len(pack):
                    chosen = pack[idx - 1]
                    o(f"  -> you picked {self.card_info.get(chosen, {}).get('name', chosen)}")
                    return DraftDecision(card_id=chosen, info={"source": "human"})
                o(f"  ! number out of range (1-{len(pack)})")
                continue
            # By name/slug fragment (case-insensitive), unique match required.
            q = raw.lower()
            matches = []
            for slug in pack:
                nm = self.card_info.get(slug, {}).get("name", slug).lower()
                if q in slug.lower() or q in nm:
                    matches.append(slug)
            uniq = list(dict.fromkeys(matches))
            if len(uniq) == 1:
                chosen = uniq[0]
                o(f"  -> you picked {self.card_info.get(chosen, {}).get('name', chosen)}")
                return DraftDecision(card_id=chosen, info={"source": "human"})
            if len(uniq) == 0:
                o("  ! no card matches that; try a number or a different fragment")
            else:
                names = ", ".join(self.card_info.get(s, {}).get("name", s) for s in uniq[:6])
                o(f"  ! ambiguous ({len(uniq)} matches): {names} — be more specific")
