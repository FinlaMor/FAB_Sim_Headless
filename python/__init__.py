"""FAB_Sim_Headless Python package.

The limited-format ecosystem is split into focused subpackages:

* :mod:`python.gameplay`     — Gym-style env + bots + self-play + replay buffer.
* :mod:`python.draft`        — 8-player draft pod simulator + draft bots.
* :mod:`python.deckbuilding` — Pool-to-deck construction + deck bots.
* :mod:`python.tournament`   — Single-elimination bracket runner.
* :mod:`python.training`     — IQL / imitation training scaffolds (torch-opt).
* :mod:`python.datasets`     — Parquet readers/writers for every artefact type.
* :mod:`python.models`       — Trained-weights registry.
* :mod:`python.analytics`    — WR/EV/matchup analytics over recorded datasets.
* :mod:`python.pipeline`     — Orchestrator: draft -> deck -> tournament -> record.

Heavyweight deps (torch / pyarrow) are imported lazily inside the modules
that need them so a minimal install stays small.
"""

__version__ = "0.2.0"
