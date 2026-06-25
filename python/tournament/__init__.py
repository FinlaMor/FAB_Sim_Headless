"""Single-elimination tournament runner.

Bracket structure (8 players, fixed seeding from the user's spec):

::

  Quarterfinals       Semifinals     Final
  A vs E   -----+
                +-- W(AE) vs W(CG) --+
  C vs G   -----+                    |
                                      +-- Champion
  B vs F   -----+                    |
                +-- W(BF) vs W(DH) --+
  D vs H   -----+

The runner drives matches through the existing :class:`TalisharEnv`
adapter — every game is played by the actual Talishar rules engine; the
tournament module owns ONLY pairing, scoring, and result aggregation.

Public surface
--------------
* :class:`TournamentRunner`  — drives the bracket.
* :class:`Player`            — wraps a deck + a gameplay bot factory.
* :class:`MatchResult`       — per-match outcome.
* :class:`TournamentResult`  — final placements + bracket tree.
* :class:`BracketTree`       — render the bracket as text/dict.
"""

from .player import Player
from .bracket import (
    BracketLabel, BracketSlot, BracketTree, eight_player_bracket,
)
from .match import MatchResult, run_match
from .runner import TournamentRunner, TournamentResult

__all__ = [
    "Player",
    "BracketLabel", "BracketSlot", "BracketTree", "eight_player_bracket",
    "MatchResult", "run_match",
    "TournamentRunner", "TournamentResult",
]
