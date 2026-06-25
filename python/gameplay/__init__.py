"""Gameplay layer — Gym-style env, bots, self-play, replay capture."""

from .env import Action, StepResult, TalisharEnv, wait_for_adapter
from .replay_buffer import ReplayBuffer, Trajectory, Transition, make_action_mask
from .dataset_writer import DatasetWriter
from .selfplay import GameSpec, run_one_game, run_selfplay_batch

__all__ = [
    "Action", "StepResult", "TalisharEnv", "wait_for_adapter",
    "ReplayBuffer", "Trajectory", "Transition", "make_action_mask",
    "DatasetWriter",
    "GameSpec", "run_one_game", "run_selfplay_batch",
]
