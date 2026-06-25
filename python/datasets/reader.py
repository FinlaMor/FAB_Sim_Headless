"""Parquet directory reader.

The training and analytics layers all consume parquet files written by
the draft / gameplay / tournament writers. This module hides the
filesystem layout behind a uniform ``DatasetReader`` so swapping to a
remote object store later (S3, GCS) only requires changing the path
resolver.

Layout
------
::

    outputs/
      parquet/
        drafts/        <- DraftDatasetWriter
        games/         <- gameplay DatasetWriter (per-step)
        tournaments/   <- TournamentRunner summary
        decks/         <- DeckBuilder output snapshots

``DatasetReader("outputs", artefact="drafts").iter_files()`` yields each
parquet path so a training loop can shuffle at file granularity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal


_ArtefactT = Literal["drafts", "games", "tournaments", "decks"]


class DatasetReader:
    def __init__(self, root: str | Path, *, artefact: _ArtefactT) -> None:
        self.root = Path(root) / "parquet" / artefact
        self.artefact: _ArtefactT = artefact

    def exists(self) -> bool:
        return self.root.is_dir()

    def iter_files(self) -> Iterator[Path]:
        if not self.exists():
            return iter(())
        return iter(sorted(self.root.glob("*.parquet")))

    def load_pandas(self):
        """Eager-load every file into a single pandas DataFrame.

        Raises ``RuntimeError`` if pyarrow + pandas aren't installed.
        For large datasets prefer ``iter_files`` and stream.
        """
        try:
            import pandas as pd  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pandas required; pip install pandas") from e
        try:
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pyarrow required; pip install pyarrow") from e
        files = list(self.iter_files())
        if not files:
            return pd.DataFrame()
        return pd.concat((pd.read_parquet(p) for p in files), ignore_index=True)


def list_artifacts(root: str | Path) -> dict[str, int]:
    """How many parquet files we have for each artefact type."""
    out: dict[str, int] = {}
    for art in ("drafts", "games", "tournaments", "decks"):
        out[art] = sum(1 for _ in DatasetReader(root, artefact=art).iter_files())
    return out
