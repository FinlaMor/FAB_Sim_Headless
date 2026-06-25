"""Analytics over recorded drafts / games / tournaments.

Each module exposes a single ``compute(...)`` function that reads from
parquet and returns a pandas DataFrame (or dict). Plotting is left to
the caller — analytics here is data, not pixels.

Modules
-------
* :mod:`hero_winrate`       — overall WR per hero across tournaments
* :mod:`archetype_winrate`  — WR per pitch-colour archetype
* :mod:`seat_ev`            — placement EV vs. seat A..H
* :mod:`pick_order_ev`      — chosen-card EV vs. pick number
* :mod:`matchup_matrix`     — hero1 vs hero2 win rates
* :mod:`signal_detection`   — heuristic "open archetype" detection
"""

from . import (
    archetype_winrate,
    hero_winrate,
    matchup_matrix,
    pick_order_ev,
    seat_ev,
    signal_detection,
)

__all__ = [
    "archetype_winrate", "hero_winrate", "matchup_matrix",
    "pick_order_ev", "seat_ev", "signal_detection",
]
