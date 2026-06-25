"""Heuristic "open archetype" signal detection.

For each pod & seat, compute the rate at which a colour wheels back to
the same player on subsequent picks. Strong wheeling = the colour is
open at that seat.

Output schema
-------------
(pod_id, seat, colour, picks_seen_pack2, wheel_rate)
"""

from __future__ import annotations

import json
from pathlib import Path

from ..datasets.reader import DatasetReader


def _pitch(card_id: str) -> str | None:
    for suffix, colour in (("_red", "red"), ("_yellow", "yellow"), ("_blue", "blue")):
        if card_id.endswith(suffix):
            return colour
    return None


def compute(root: str | Path):
    import pandas as pd
    drafts = DatasetReader(root, artefact="drafts").load_pandas()
    if drafts.empty:
        return pd.DataFrame(columns=["pod_id", "seat", "colour", "picks_seen", "chosen_in_colour", "rate"])
    rows = []
    for (pod_id, seat), seat_df in drafts.groupby(["pod_id", "seat"]):
        colour_seen = {"red": 0, "yellow": 0, "blue": 0}
        colour_chosen = {"red": 0, "yellow": 0, "blue": 0}
        for _, row in seat_df.iterrows():
            try:
                pack = json.loads(row["pack_contents_json"]) or []
            except Exception:  # noqa: BLE001
                pack = []
            for c in pack:
                p = _pitch(c)
                if p:
                    colour_seen[p] += 1
            chosen_p = _pitch(row["chosen_card"])
            if chosen_p:
                colour_chosen[chosen_p] += 1
        for colour in ("red", "yellow", "blue"):
            rows.append({
                "pod_id": pod_id,
                "seat":   seat,
                "colour": colour,
                "picks_seen": colour_seen[colour],
                "chosen_in_colour": colour_chosen[colour],
                "rate": (colour_chosen[colour] / colour_seen[colour]) if colour_seen[colour] else 0.0,
            })
    return pd.DataFrame(rows)
