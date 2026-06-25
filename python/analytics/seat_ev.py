"""Tournament placement EV vs. draft seat A..H."""

from __future__ import annotations

from pathlib import Path

from ..datasets.reader import DatasetReader


def compute(root: str | Path):
    import pandas as pd
    tournaments = DatasetReader(root, artefact="tournaments").load_pandas()
    if tournaments.empty:
        return pd.DataFrame(columns=["seat", "mean_placement", "champion_rate", "samples"])
    # tournaments parquet schema: one row per (tournament_id, label, placement)
    grouped = tournaments.groupby("label").agg(
        mean_placement=("placement", "mean"),
        champion_rate=("placement", lambda s: (s == 1).mean()),
        samples=("placement", "size"),
    ).reset_index().rename(columns={"label": "seat"})
    return grouped.sort_values("mean_placement")
