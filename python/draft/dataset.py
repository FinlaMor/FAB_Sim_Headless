"""Persist draft trajectories.

A draft "row" is a single pick: (pod_id, seat, pack_no, pick_no,
pack_contents, chosen_card, remaining_after_pick, neighbour seats,
bot_info, final deck snapshot, tournament placement if known).

The "final deck" and "tournament placement" fields are populated later
by the orchestrator after the bracket runs — they're stored as
columns rather than separate tables so a trainer can do
``df[df.placement == 1].chosen_card`` directly.

Storage formats
---------------
The writer mirrors :class:`python.gameplay.dataset_writer.DatasetWriter`:
parquet preferred, msgpack/npz fallbacks. Pass ``fmt="parquet"`` (the
default) for offline-RL pipelines.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from .simulator import DraftPodResult


_FormatT = Literal["parquet", "msgpack", "npz"]


@dataclass
class DraftRecord:
    """One pick row, flattened for parquet."""
    pod_id: str
    seed: int
    seat: int
    pack_number: int
    pick_number: int
    chosen_card: str
    pack_contents_json: str
    remaining_after_pick_json: str
    left_neighbour_seat: int
    right_neighbour_seat: int
    bot_info_json: str
    final_deck_json: str = "[]"
    placement: int = 0
    bracket_label: str = ""
    ts_unix: float = 0.0


class DraftDatasetWriter:
    def __init__(
        self,
        out_dir: str | os.PathLike[str],
        *,
        fmt: _FormatT = "parquet",
        compression: str = "zstd",
    ) -> None:
        self.fmt: _FormatT = fmt
        self.compression = compression
        self.out_dir = Path(out_dir) / fmt / "drafts"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write_pods(
        self,
        pods: Iterable[DraftPodResult],
        *,
        placements: dict[tuple[str, int], int] | None = None,
        bracket_labels: dict[int, str] | None = None,
    ) -> Path | None:
        """Persist a batch of pods.

        ``placements`` maps ``(pod_id, seat)`` to a tournament placement
        (1 = champion, 2 = finalist, etc.). Pass ``None`` to leave the
        column zero-filled.
        """
        rows = []
        for pod in pods:
            for pick in pod.picks:
                record = DraftRecord(
                    pod_id=pod.pod_id,
                    seed=pod.seed,
                    seat=pick.seat,
                    pack_number=pick.pack_number,
                    pick_number=pick.pick_number,
                    chosen_card=pick.chosen_card,
                    pack_contents_json=json.dumps(pick.pack_contents, separators=(",", ":")),
                    remaining_after_pick_json=json.dumps(pick.remaining_after_pick, separators=(",", ":")),
                    left_neighbour_seat=pick.left_neighbour,
                    right_neighbour_seat=pick.right_neighbour,
                    bot_info_json=json.dumps(pick.bot_info, separators=(",", ":")),
                    final_deck_json=json.dumps(pod.drafted_pool(pick.seat), separators=(",", ":")),
                    placement=int(placements.get((pod.pod_id, pick.seat), 0)) if placements else 0,
                    bracket_label=(bracket_labels or {}).get(pick.seat, ""),
                    ts_unix=pick.ts_unix,
                )
                rows.append(record)
        if not rows:
            return None
        path = self.out_dir / f"{int(time.time())}-{uuid.uuid4().hex[:8]}.{_ext(self.fmt)}"
        if self.fmt == "parquet":
            self._write_parquet(rows, path)
        elif self.fmt == "msgpack":
            self._write_msgpack(rows, path)
        else:
            self._write_npz(rows, path)
        return path

    # ------------------------------------------------------------------
    # Backend writers
    # ------------------------------------------------------------------
    def _write_parquet(self, rows: list[DraftRecord], path: Path) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pyarrow required for parquet output.") from e
        table = pa.Table.from_pylist([asdict(r) for r in rows])
        pq.write_table(table, str(path), compression=self.compression)

    def _write_msgpack(self, rows: list[DraftRecord], path: Path) -> None:
        try:
            import msgpack
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("msgpack required for msgpack output.") from e
        with open(path, "wb") as f:
            f.write(msgpack.packb([asdict(r) for r in rows], use_bin_type=True))

    def _write_npz(self, rows: list[DraftRecord], path: Path) -> None:
        try:
            import numpy as np
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("numpy required for npz output.") from e
        np.savez_compressed(
            path,
            records=np.array([json.dumps(asdict(r), separators=(",", ":")) for r in rows], dtype=object),
        )


def _ext(fmt: _FormatT) -> str:
    return {"parquet": "parquet", "msgpack": "mpk", "npz": "npz"}[fmt]
