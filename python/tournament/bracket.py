"""Static 8-player bracket structure.

The user's spec uses fixed seed labels A..H with this pairing:

    QF: A vs E,   C vs G,   B vs F,   D vs H
    SF: W(AE) vs W(CG),     W(BF) vs W(DH)
    F : SF1 vs SF2

We expose the bracket as a list of "rounds", where each round is a list
of ``BracketSlot`` records. The tournament runner walks the rounds in
order, pulling the previous round's winners into the next round's slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


BracketLabel = str   # "A".."H" or "W(AE)" etc.


@dataclass
class BracketSlot:
    """One match slot in the bracket."""
    round_name: str        # "QF" | "SF" | "F"
    match_id: str          # stable identifier ("QF1", "SF2", "F")
    p1_label: BracketLabel
    p2_label: BracketLabel
    winner_label: BracketLabel | None = None


@dataclass
class BracketTree:
    rounds: list[list[BracketSlot]] = field(default_factory=list)

    def all_slots(self) -> Iterable[BracketSlot]:
        for r in self.rounds:
            yield from r

    def find(self, match_id: str) -> BracketSlot:
        for s in self.all_slots():
            if s.match_id == match_id:
                return s
        raise KeyError(match_id)


def eight_player_bracket() -> BracketTree:
    """Construct the 8-player bracket per the user's spec.

    The semifinal winners' source ``W(...)`` strings are resolved at
    runtime by :class:`TournamentRunner` once the prior round has been
    decided.
    """
    qf = [
        BracketSlot("QF", "QF1", "A", "E"),
        BracketSlot("QF", "QF2", "C", "G"),
        BracketSlot("QF", "QF3", "B", "F"),
        BracketSlot("QF", "QF4", "D", "H"),
    ]
    sf = [
        BracketSlot("SF", "SF1", "W(QF1)", "W(QF2)"),
        BracketSlot("SF", "SF2", "W(QF3)", "W(QF4)"),
    ]
    final = [BracketSlot("F", "F", "W(SF1)", "W(SF2)")]
    return BracketTree(rounds=[qf, sf, final])


def render_text(tree: BracketTree) -> str:
    """ASCII representation, used by analytics + smoke test output."""
    out = []
    for r in tree.rounds:
        out.append(f"-- {r[0].round_name} " + "-" * 30)
        for s in r:
            w = s.winner_label or "?"
            out.append(f"  {s.match_id:>4}: {s.p1_label:>8} vs {s.p2_label:<8}   winner: {w}")
    return "\n".join(out)
