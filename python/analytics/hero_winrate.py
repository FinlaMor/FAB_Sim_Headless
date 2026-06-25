"""Per-hero win rate across recorded tournaments."""

from __future__ import annotations

from pathlib import Path

from ..datasets.reader import DatasetReader


def compute(root: str | Path):
    """Return a dataframe of (hero, games_played, wins, win_rate)."""
    import pandas as pd
    games = DatasetReader(root, artefact="games").load_pandas()
    if games.empty:
        return pd.DataFrame(columns=["hero", "games_played", "wins", "win_rate"])
    # winner column is 1 / 2 / 0; hero1 = p1 hero, hero2 = p2 hero
    hero1 = games[["hero1", "winner"]].rename(columns={"hero1": "hero"})
    hero1["win"] = (hero1["winner"] == 1).astype(int)
    hero2 = games[["hero2", "winner"]].rename(columns={"hero2": "hero"})
    hero2["win"] = (hero2["winner"] == 2).astype(int)
    long = pd.concat([hero1[["hero", "win"]], hero2[["hero", "win"]]], ignore_index=True)
    grouped = long.groupby("hero").agg(games_played=("win", "size"), wins=("win", "sum")).reset_index()
    grouped["win_rate"] = grouped["wins"] / grouped["games_played"]
    return grouped.sort_values("win_rate", ascending=False)
