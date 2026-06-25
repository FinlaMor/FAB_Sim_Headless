"""On-disk versioned weights store."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Literal


_RoleT = Literal["draft", "gameplay", "deck"]


def _role_dir(root: str | Path, role: _RoleT) -> Path:
    p = Path(root) / "models" / role
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(
    src: str | Path,
    *,
    root: str | Path,
    role: _RoleT,
    version: str | None = None,
) -> Path:
    """Copy a weights file into the registry.

    Returns the destination path. ``version`` defaults to a millisecond
    timestamp ("20260528-104500-123").
    """
    version = version or time.strftime("%Y%m%d-%H%M%S")
    dest_dir = _role_dir(root, role) / version
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_p = Path(src)
    dest = dest_dir / src_p.name
    shutil.copy2(src_p, dest)
    return dest


def list_versions(root: str | Path, role: _RoleT) -> list[str]:
    return sorted(p.name for p in _role_dir(root, role).iterdir() if p.is_dir())


def latest_path(root: str | Path, role: _RoleT) -> Path | None:
    versions = list_versions(root, role)
    if not versions:
        return None
    last_dir = _role_dir(root, role) / versions[-1]
    files = sorted(last_dir.iterdir())
    return files[0] if files else None
