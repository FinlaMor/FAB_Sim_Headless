"""Supervised imitation baselines.

The orchestrator's iterative-improvement loop uses BC as the warm-start
when no IQL policy exists yet. Given a parquet of trajectories, BC
fits ``π(a|s) ≈ p(chosen | state)`` weighted by terminal reward.

This is the baseline IQL must beat by ≥1% absolute win rate before the
orchestrator promotes the new IQL weights into the bot registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BCHyperparams:
    lr: float          = 3e-4
    batch_size: int    = 256
    n_epochs: int      = 5
    weight_by_reward: bool = True


def train(
    *,
    parquet_dir: str | Path,
    out_dir: str | Path,
    role: str,                  # "draft" | "gameplay"
    hyper: BCHyperparams | None = None,
    device: str = "cpu",
) -> Path:
    try:
        import torch  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Install torch to train BC: pip install torch") from e
    hyper = hyper or BCHyperparams()
    out_path = Path(out_dir) / f"bc_{role}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        f"supervised_imitation.train({role}) is a scaffold. Use BC as a "
        "warm-start for IQL; the resulting weights are loaded by the "
        f"matching ``python.{role}.bots.transformer_bot``."
    )
