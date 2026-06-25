"""Gym-style HTTP client for the headless Talishar adapter.

The environment is *thin*: it doesn't cache state, it doesn't try to
recompute legality, and it doesn't shape rewards. All of that is the
adapter's job (and ultimately Talishar's). The Python side just observes
and chooses.

Typical usage
-------------
>>> env = TalisharEnv("http://localhost:8000")
>>> obs = env.reset(hero1="Bravo", hero2="Dash",
...                 deck1="decks/bravo.json", deck2="decks/dash.json",
...                 seed=12345)
>>> while not env.done:
...     legal = env.get_actions()
...     action = legal[0]  # bots replace this
...     obs, reward, done, info = env.step(action.action_id)

Threading / processes
---------------------
Every ``TalisharEnv`` instance owns its own ``requests.Session`` so it is
safe to use in a thread/process pool. For self-play at high throughput,
prefer multiple processes each holding one env each — that's what
``selfplay.py`` does.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------------------------------------------------------------------
# Typed records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    """A single legal action returned by the adapter.

    ``raw`` retains the full JSON record (including ``talishar_*`` fields
    so the bot's choice can be round-tripped through replay tools).
    """
    action_id: int
    type: str
    player_id: int
    card_id: str | None
    targets: list[Any]
    cost: dict[str, Any]
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def from_json(cls, j: dict[str, Any]) -> "Action":
        return cls(
            action_id=int(j["action_id"]),
            type=str(j["type"]),
            player_id=int(j.get("player_id") or 0),
            card_id=j.get("card_id"),
            targets=list(j.get("targets") or []),
            cost=dict(j.get("cost") or {}),
            raw=j,
        )


@dataclass
class StepResult:
    state: dict[str, Any]
    legal_actions: list[Action]
    reward: float
    done: bool
    winner: int | None
    info: dict[str, Any]


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------
class TalisharEnv:
    """OpenAI Gym–style wrapper over the adapter REST API.

    Required adapter endpoints:

    * ``POST /new_game``
    * ``POST /step``
    * ``GET /state``
    * ``GET /actions``
    * ``POST /reset``

    Parameters
    ----------
    base_url:
        Adapter HTTP origin, e.g. ``http://localhost:8000``.
    timeout:
        Per-request timeout in seconds. The adapter is a single-threaded
        ``php -S`` dev server; under N-worker contention a single /step can
        legitimately take several seconds, so the default is intentionally
        roomy. A 5 s default once killed an 8-worker run when port 8000
        briefly stalled — keep this generous and rely on pair-level
        isolation for the rare hard failure.
    max_retries:
        Total transport-level retries (connection refused, transient 5xx).
    """

    # Sentinel values used until reset() is called.
    _SENTINEL_GAME_ID = ""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.game_id: str = self._SENTINEL_GAME_ID
        self.last_state: dict[str, Any] = {}
        self.last_actions: list[Action] = []
        self.done: bool = False
        self.winner: int | None = None
        self.info: dict[str, Any] = {}

        self._session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=0.25,
            status_forcelist=(502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        hero1: str,
        hero2: str,
        deck1: str,
        deck2: str,
        seed: int,
        format: str = "draft",
    ) -> StepResult:
        """Start a new game and return the initial observation.

        ``format`` selects the Talishar ruleset: "draft" (OMN limited, the
        default) or "cc" (Classic Constructed).
        """
        body = {
            "hero1": hero1, "hero2": hero2,
            "deck1": deck1, "deck2": deck2,
            "seed": int(seed), "format": format,
        }
        resp = self._post("/new_game", body)
        self.game_id = str(resp["game_id"])
        return self._absorb(resp, default_reward=0.0)

    def reset_same(self) -> StepResult:
        """Reset the *current* game to its initial state (same seed/decks/heroes)."""
        if not self.game_id:
            raise RuntimeError("Call reset(...) first; nothing to reset_same() from.")
        resp = self._post("/reset", {"game_id": self.game_id})
        self.game_id = str(resp["game_id"])
        return self._absorb(resp, default_reward=0.0)

    def step(self, action_id: int) -> StepResult:
        if not self.game_id:
            raise RuntimeError("Call reset(...) before step(...).")
        resp = self._post("/step", {"game_id": self.game_id, "action_id": int(action_id)})
        return self._absorb(resp, default_reward=resp.get("reward", 0.0))

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------
    def get_state(self, *, refresh: bool = False) -> dict[str, Any]:
        if not refresh and self.last_state:
            return self.last_state
        resp = self._get("/state", {"game_id": self.game_id})
        self.last_state = resp["state"]
        return self.last_state

    def get_actions(self, *, refresh: bool = False) -> list[Action]:
        if not refresh and self.last_actions:
            return self.last_actions
        resp = self._get("/actions", {"game_id": self.game_id})
        self.last_actions = [Action.from_json(a) for a in resp["legal_actions"]]
        return self.last_actions

    def health(self) -> dict[str, Any]:
        return self._get("/health", {})

    # ------------------------------------------------------------------
    # Conveniences
    # ------------------------------------------------------------------
    def priority_player(self) -> int:
        return int(self.last_state.get("priority_player", 0))

    def action_mask(self, max_actions: int) -> list[bool]:
        """Boolean mask of length ``max_actions`` over ``action_id - 1`` slots.

        Useful for transformer policies that have a fixed action-space head.
        Falls back to all-False outside the legal set.
        """
        mask = [False] * max_actions
        for a in self.last_actions:
            i = a.action_id - 1
            if 0 <= i < max_actions:
                mask[i] = True
        return mask

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _absorb(self, resp: dict[str, Any], default_reward: float) -> StepResult:
        self.last_state = resp.get("state", {})
        self.last_actions = [Action.from_json(a) for a in resp.get("legal_actions", [])]
        self.done = bool(resp.get("done", False))
        self.winner = resp.get("winner")
        self.info = dict(resp.get("info", {}))
        reward = float(resp.get("reward", default_reward))
        return StepResult(
            state=self.last_state,
            legal_actions=self.last_actions,
            reward=reward,
            done=self.done,
            winner=self.winner,
            info=self.info,
        )

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url + path
        r = self._session.post(url, json=body, timeout=self.timeout)
        return self._parse(r, url, body)

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url + path
        r = self._session.get(url, params=params, timeout=self.timeout)
        return self._parse(r, url, params)

    @staticmethod
    def _parse(r: requests.Response, url: str, payload: Any) -> dict[str, Any]:
        try:
            j = r.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Non-JSON response from {url}: status={r.status_code}, body[:200]={r.text[:200]!r}"
            ) from e
        if r.status_code >= 400:
            raise RuntimeError(
                f"{url} returned {r.status_code}: {j} (payload={payload!r})"
            )
        return j

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "TalisharEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Convenience: spin until the adapter is reachable (used by selfplay.py).
# ---------------------------------------------------------------------------
def wait_for_adapter(base_url: str, *, timeout_s: float = 30.0, poll_s: float = 0.5) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return TalisharEnv(base_url).health()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(poll_s)
    raise RuntimeError(f"Adapter at {base_url} never came up within {timeout_s}s") from last_err
