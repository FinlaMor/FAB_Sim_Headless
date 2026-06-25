"""Persist completed self-play trajectories to disk.

Three output formats are supported, picked at construction time:

* ``parquet`` — preferred. Each call to ``write_batch`` produces one
  rowgroup. Columns include the game-level metadata and a JSON-serialised
  blob for each per-step record. Requires ``pyarrow``.
* ``msgpack`` — single ``.mpk`` file per batch. Useful when you want to
  ship trajectories between processes or to a remote queue.
* ``npz``    — compressed numpy archive. Mostly for legacy pipelines;
  the JSON blobs are stored as object arrays.

The writer never blocks the self-play loop on encoding cost: callers
should accumulate a batch (e.g. 64 games) in the ``ReplayBuffer`` and
then call ``DatasetWriter.write_batch(buffer.drain())``.

Files land in ``datasets/<format>/<batch_id>.<ext>`` by default.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Iterable, Literal

from .replay_buffer import Trajectory


_FormatT = Literal["parquet", "msgpack", "npz"]


class DatasetWriter:
    def __init__(
        self,
        out_dir: str | os.PathLike[str],
        *,
        fmt: _FormatT = "parquet",
        compression: str = "zstd",
        subdir: str = "games",
    ) -> None:
        self.fmt: _FormatT = fmt
        self.compression = compression
        # Mirror the per-artefact layout used by DraftDatasetWriter etc.:
        #   <out_dir>/<fmt>/<subdir>/  e.g. outputs/parquet/games/
        # Older callers that omit subdir still land under games/.
        self.out_dir = Path(out_dir) / fmt / subdir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def write_batch(self, trajectories: Iterable[Trajectory]) -> Path | None:
        trajs = list(trajectories)
        if not trajs:
            return None
        batch_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        path = self.out_dir / f"{batch_id}.{_ext(self.fmt)}"
        if self.fmt == "parquet":
            self._write_parquet(trajs, path)
        elif self.fmt == "msgpack":
            self._write_msgpack(trajs, path)
        elif self.fmt == "npz":
            self._write_npz(trajs, path)
        else:
            raise ValueError(f"Unknown format: {self.fmt}")
        return path

    # ------------------------------------------------------------------
    # Parquet (preferred)
    # ------------------------------------------------------------------
    def _write_parquet(self, trajs: list[Trajectory], path: Path) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "pyarrow is required for parquet output. Install it with "
                "`pip install pyarrow`."
            ) from e

        # Schema is intentionally "one row per transition" so trainers can
        # window/shuffle at transition granularity without re-exploding.
        rows = []
        for tr in trajs:
            for step in tr.transitions:
                rows.append({
                    "game_id": tr.game_id,
                    "seed": tr.seed,
                    "hero1": tr.hero1,
                    "hero2": tr.hero2,
                    "winner": tr.winner,
                    "step_index": step.step_index,
                    "player_to_move": step.player_to_move,
                    "chosen_action_id": step.chosen_action_id,
                    "reward": step.reward,
                    "done": step.done,
                    "ts_unix": step.ts_unix,
                    "state_json": json.dumps(step.state, separators=(",", ":")),
                    "next_state_json": json.dumps(step.next_state, separators=(",", ":")),
                    "legal_actions_json": json.dumps(step.legal_actions, separators=(",", ":")),
                    "chosen_action_json": json.dumps(step.chosen_action, separators=(",", ":")),
                    "action_mask": step.action_mask,
                })

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, str(path), compression=self.compression)

    # ------------------------------------------------------------------
    # Msgpack
    # ------------------------------------------------------------------
    def _write_msgpack(self, trajs: list[Trajectory], path: Path) -> None:
        try:
            import msgpack
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "msgpack-python is required for msgpack output. Install "
                "`pip install msgpack`."
            ) from e
        payload = [{
            "game_id": tr.game_id,
            "seed": tr.seed,
            "hero1": tr.hero1, "hero2": tr.hero2,
            "deck1": tr.deck1, "deck2": tr.deck2,
            "winner": tr.winner,
            "started_at": tr.started_at,
            "ended_at": tr.ended_at,
            "metadata": tr.metadata,
            "transitions": [step.__dict__ for step in tr.transitions],
        } for tr in trajs]
        with open(path, "wb") as f:
            f.write(msgpack.packb(payload, use_bin_type=True))

    # ------------------------------------------------------------------
    # NPZ
    # ------------------------------------------------------------------
    def _write_npz(self, trajs: list[Trajectory], path: Path) -> None:
        try:
            import numpy as np
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("numpy is required for npz output.") from e

        records = []
        for tr in trajs:
            for step in tr.transitions:
                records.append(json.dumps({
                    "game_id": tr.game_id,
                    "seed": tr.seed,
                    "hero1": tr.hero1, "hero2": tr.hero2,
                    "winner": tr.winner,
                    "step_index": step.step_index,
                    "player_to_move": step.player_to_move,
                    "chosen_action_id": step.chosen_action_id,
                    "reward": step.reward,
                    "done": step.done,
                    "ts_unix": step.ts_unix,
                    "action_mask": step.action_mask,
                    "state": step.state,
                    "next_state": step.next_state,
                    "legal_actions": step.legal_actions,
                    "chosen_action": step.chosen_action,
                }, separators=(",", ":")))
        np.savez_compressed(path, records=np.array(records, dtype=object))


# --------------------------------------------------------------------------
def _ext(fmt: _FormatT) -> str:
    return {"parquet": "parquet", "msgpack": "mpk", "npz": "npz"}[fmt]
