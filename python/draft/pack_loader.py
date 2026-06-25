"""Load and deterministically assign booster packs to draft seats.

Expected JSON file shape
------------------------
::

    [
      {"pack_id": 1, "cards": ["card_a", "card_b", ...]},
      {"pack_id": 2, "cards": [...]},
      ...
    ]

The loader is intentionally tolerant about extra keys (release set, draft
weights, etc.) — only ``pack_id`` and ``cards`` are required.

Deterministic sampling
----------------------
``sample_player_packs(pool, n_players, packs_per_player, seed)`` draws
``n_players * packs_per_player`` packs *without replacement* using a
local :class:`random.Random` instance seeded with ``seed``. Identical
``(seed, pool size)`` inputs always produce identical assignments.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Pack:
    """A single booster: an immutable card list plus its identifier."""

    pack_id: int
    cards: tuple[str, ...]

    @classmethod
    def from_json(cls, j: dict) -> "Pack":
        if "pack_id" not in j or "cards" not in j:
            raise ValueError(f"Pack JSON missing pack_id/cards: {j!r}")
        return cls(int(j["pack_id"]), tuple(str(c) for c in j["cards"]))


@dataclass
class PackPool:
    """Full collection of packs available for sampling."""

    packs: list[Pack] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.packs)

    def add(self, pack: Pack) -> None:
        self.packs.append(pack)

    def card_universe(self) -> set[str]:
        """Distinct card IDs across all packs (used by analytics)."""
        return {c for p in self.packs for c in p.cards}


def load_pack_pool(path: str | Path) -> PackPool:
    """Read a packs JSON file from disk into a :class:`PackPool`."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Packs JSON must be a top-level list, got {type(raw)}")
    pool = PackPool()
    seen_ids: set[int] = set()
    for entry in raw:
        pack = Pack.from_json(entry)
        if pack.pack_id in seen_ids:
            raise ValueError(f"Duplicate pack_id {pack.pack_id} in {path}")
        seen_ids.add(pack.pack_id)
        pool.add(pack)
    return pool


def sample_player_packs(
    pool: PackPool,
    *,
    n_players: int,
    packs_per_player: int,
    seed: int,
) -> list[list[Pack]]:
    """Deterministically assign ``packs_per_player`` packs to each seat.

    Returns a list of length ``n_players``; element ``i`` is the ordered
    list of packs that seat ``i`` will open in turn. Sampling is without
    replacement, so the total pool must contain at least
    ``n_players * packs_per_player`` packs.

    The shuffle is stable for a given (seed, pool order) pair — useful
    for replaying the exact same pod for debugging.
    """
    total = n_players * packs_per_player
    if len(pool) < total:
        raise ValueError(
            f"Need {total} packs ({n_players} players * {packs_per_player}) "
            f"but pool only has {len(pool)}"
        )
    rng = random.Random(seed)
    indices = list(range(len(pool.packs)))
    rng.shuffle(indices)
    drawn = [pool.packs[i] for i in indices[:total]]
    # Round-robin so seat 0 gets packs [0, n, 2n], seat 1 gets [1, n+1, 2n+1], etc.
    # This matches how a real pod opens "Pack 1 / Pack 2 / Pack 3" together.
    assignments: list[list[Pack]] = [[] for _ in range(n_players)]
    for pack_round in range(packs_per_player):
        for seat in range(n_players):
            assignments[seat].append(drawn[pack_round * n_players + seat])
    return assignments
