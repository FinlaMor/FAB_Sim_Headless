"""8-player draft pod simulator for Omens of the Third Age limited.

Public surface
--------------
* :class:`PackPool`           — loads/holds the full set of packs.
* :class:`DraftPodConfig`     — per-pod parameters (n_players, n_packs, ...).
* :class:`DraftSimulator`     — runs a full pod end-to-end.
* :class:`DraftBot`           — abstract base; implementations live in `bots/`.
* :class:`DraftRecord`        — per-pick row written to parquet/msgpack/npz.

Heroes and signature weapons legal for Omens of the Third Age are defined
in :mod:`python.draft.format` as plain constants so heuristic bots can
reference them by symbolic name.
"""

from .format import LEGAL_HEROES, LEGAL_WEAPONS, FORMAT_NAME, FORMAT_CODE
from .pack_loader import PackPool, load_pack_pool, sample_player_packs
from .simulator import (
    DraftPodConfig,
    DraftPodResult,
    DraftSimulator,
    DraftSeat,
)
from .dataset import DraftRecord, DraftDatasetWriter
from .bots import DraftBot, RandomDraftBot, HeuristicDraftBot

__all__ = [
    "LEGAL_HEROES", "LEGAL_WEAPONS", "FORMAT_NAME", "FORMAT_CODE",
    "PackPool", "load_pack_pool", "sample_player_packs",
    "DraftPodConfig", "DraftPodResult", "DraftSimulator", "DraftSeat",
    "DraftRecord", "DraftDatasetWriter",
    "DraftBot", "RandomDraftBot", "HeuristicDraftBot",
]
