"""Pick-order EV: for each card, what placement do drafters who picked it achieve?

Reads the draft parquet (which carries the post-tournament placement
column) and bins by ``pick_number``.
"""

from __future__ import annotations

from pathlib import Path

from ..datasets.reader import DatasetReader


def compute(root: str | Path):
    import pandas as pd
    drafts = DatasetReader(root, artefact="drafts").load_pandas()
    if drafts.empty:
        return pd.DataFrame(columns=["chosen_card", "pick_number", "mean_placement", "samples"])
    df = drafts[drafts["placement"] > 0]
    if df.empty:
        return pd.DataFrame(columns=["chosen_card", "pick_number", "mean_placement", "samples"])
    grouped = df.groupby(["chosen_card", "pick_number"]).agg(
        mean_placement=("placement", "mean"),
        samples=("placement", "size"),
    ).reset_index()
    return grouped.sort_values(["chosen_card", "pick_number"])
