"""Bot zoo for self-play.

Importing this package is side-effect-free; heavy deps (torch) are loaded
lazily inside ``TransformerBot`` so a self-play worker that only uses the
random bot doesn't pay for them.
"""

from .base import Bot, BotDecision
from .random_bot import RandomBot
from .heuristic_bot import HeuristicBot

__all__ = ["Bot", "BotDecision", "RandomBot", "HeuristicBot"]
