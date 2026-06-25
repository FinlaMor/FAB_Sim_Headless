"""Training scaffolds for the draft and gameplay policies.

Each module is **import-safe without torch** — the heavy deps come in
inside the trainer's ``__init__``. The scaffolds give you:

* :mod:`iql_gameplay` — Implicit Q-Learning over the gameplay parquet schema.
* :mod:`iql_draft`    — IQL applied to draft picks (advantage = final placement).
* :mod:`supervised_imitation` — Behaviour-cloning baseline for both heads.

Each trainer exposes a ``train(...)`` function with consistent kwargs so
the orchestrator can swap implementations without touching the call site.
"""

from . import iql_draft, iql_gameplay, supervised_imitation

__all__ = ["iql_draft", "iql_gameplay", "supervised_imitation"]
