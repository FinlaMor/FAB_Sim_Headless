# Adding a new bot

The orchestrator (`python/selfplay.py`) is intentionally bot-agnostic.
A new bot only needs three things:

1. A class that inherits `python.bots.base.Bot`.
2. A `choose(state, legal_actions, *, player_id)` method that returns a
   `BotDecision(action_id=..., info={...})`.
3. (Optional) Registration in `python/selfplay.py::_BOT_REGISTRY` so the
   CLI can construct it by name.

## Minimal example

```python
# python/bots/aggro_bot.py
from typing import Any
from .base import Bot, BotDecision
from ..env import Action

class AggroBot(Bot):
    """Always picks the first PLAY_FROM_HAND if any; otherwise PASS."""
    name = "aggro"

    def choose(self, state: dict[str, Any], legal_actions: list[Action], *, player_id: int) -> BotDecision:
        plays = [a for a in legal_actions if a.type == "PLAY_FROM_HAND"]
        if plays:
            return BotDecision(action_id=plays[0].action_id, info={"policy": "aggro/first-play"})
        pass_a = next((a for a in legal_actions if a.type == "PASS"), legal_actions[0])
        return BotDecision(action_id=pass_a.action_id, info={"policy": "aggro/fallback-pass"})
```

Then register it:

```python
# python/selfplay.py (snippet)
from .bots.aggro_bot import AggroBot
_BOT_REGISTRY["aggro"] = lambda seed: AggroBot()
```

And run self-play with it:

```bash
python -m python.selfplay --hero1 Bravo --hero2 Dash \
    --deck1 decks/bravo.json --deck2 decks/dash.json \
    --bot1 aggro --bot2 random --games 100 --base-seed 1000 \
    --adapter http://localhost:8000 --out datasets
```

## Decision metadata

Anything you place in `BotDecision.info` gets persisted in the trajectory
parquet under `chosen_action_json` (if you also stash it in
`chosen_action.raw`) or you can extend `replay_buffer.Transition` and
the parquet writer to track it as a first-class column.

## Action types

The adapter currently emits these action types:

| `type`                   | When                                              |
| ------------------------ | ------------------------------------------------- |
| `PLAY_FROM_HAND`         | Main / action phase — playing a card from hand    |
| `ACTIVATE_HERO_OR_EQUIP` | Activating a hero ability or equipment            |
| `DECISION`               | Responding to a Talishar BUTTONINPUT/YESNO popup  |
| `CHOOSE_<phase>`         | Choosing a single card during a zone-pick popup   |
| `PASS`                   | The catch-all pass / OK / skip                    |

The `raw` field of every `Action` carries the underlying Talishar mode
codes if your bot needs to inspect them.
