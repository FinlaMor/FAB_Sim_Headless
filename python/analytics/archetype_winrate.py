"""Pitch-colour archetype win rate.

We approximate an archetype as the dominant pitch colour in the deck.
Reads from the ``decks/`` artefact written by the orchestrator.
"""

from __future__ import annotations

from pathlib import Path

from ..datasets.reader import DatasetReader


def compute(root: str | Path):
    import json
    import pandas as pd
    decks = DatasetReader(root, artefact="decks").load_pandas()
    games = DatasetReader(root, artefact="games").load_pandas()
    if decks.empty or games.empty:
        return pd.DataFrame(columns=["archetype", "games_played", "wins", "win_rate"])
    # decks parquet schema must include a "pitch_distribution_json" column.
    def _dominant_colour(j: str) -> str:
        try:
            d = json.loads(j) or {}
        except Exception:  # noqa: BLE001
            return "unknown"
        if not d:
            return "colorless"
        return max(d.items(), key=lambda kv: kv[1])[0]

    decks["archetype"] = decks["pitch_distribution_json"].map(_dominant_colour)
    # Join to games on (hero, deck-owner)
    arche_p1 = decks.rename(columns={"hero": "hero1", "archetype": "archetype_p1"})[["hero1", "archetype_p1"]]
    arche_p2 = decks.rename(columns={"hero": "hero2", "archetype": "archetype_p2"})[["hero2", "archetype_p2"]]
    merged = games.merge(arche_p1, on="hero1", how="left").merge(arche_p2, on="hero2", how="left")
    long = pd.concat([
        merged[["archetype_p1", "winner"]].rename(columns={"archetype_p1": "archetype"}).assign(
            win=(merged["winner"] == 1).astype(int)
        ),
        merged[["archetype_p2", "winner"]].rename(columns={"archetype_p2": "archetype"}).assign(
            win=(merged["winner"] == 2).astype(int)
        ),
    ], ignore_index=True).dropna(subset=["archetype"])
    grouped = long.groupby("archetype").agg(
        games_played=("win", "size"), wins=("win", "sum")
    ).reset_index()
    grouped["win_rate"] = grouped["wins"] / grouped["games_played"]
    return grouped.sort_values("win_rate", ascending=False)
