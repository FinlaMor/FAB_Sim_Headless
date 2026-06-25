"""Versioned model-weights registry.

Keeps trained checkpoints in ``outputs/models/<role>/<version>/`` where
``role`` is ``"draft" | "gameplay"`` and ``version`` is a sortable
timestamp. The orchestrator's training loop calls
:func:`save_checkpoint` after each cycle; bots load the latest via
:func:`latest_path`.

Intentionally tiny — no MLflow, no DVC. Add one of those when you need
experiment-tracking; the registry's only contract is the on-disk layout.
"""

from .registry import latest_path, list_versions, save_checkpoint

__all__ = ["latest_path", "list_versions", "save_checkpoint"]
