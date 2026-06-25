"""Hero-vs-hero win-rate matrix."""

from __future__ import annotations

from pathlib import Path

from ..datasets.reader import DatasetReader


def compute(root: str | Path):
    import pandas as pd
    games = DatasetReader(root, artefact="games").load_pandas()
    if games.empty:
        return pd.DataFrame()
    # Mirror so both orders contribute to the same cell.
    p1_wins = games.assign(win=(games["winner"] == 1).astype(int))[["hero1", "hero2", "win"]]
    p2_wins = games.assign(win=(games["winner"] == 2).astype(int)) \
                  .rename(columns={"hero1": "hero2", "hero2": "hero1"})[["hero1", "hero2", "win"]]
    long = pd.concat([p1_wins, p2_wins], ignore_index=True)
    pivot = long.groupby(["hero1", "hero2"]).agg(
        games=("win", "size"), wins=("win", "sum")
    ).reset_index()
    pivot["win_rate"] = pivot["wins"] / pivot["games"]
    return pivot.pivot(index="hero1", columns="hero2", values="win_rate")
