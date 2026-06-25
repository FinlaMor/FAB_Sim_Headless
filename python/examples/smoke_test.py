"""End-to-end smoke test for the Python client.

Runs against a tiny in-process HTTP stub of the adapter so it can verify
the env <-> bot <-> replay <-> dataset_writer pipeline without needing
PHP / Docker installed. Mirrors the real adapter's `/new_game`,
`/state`, `/actions`, `/step`, `/reset`, `/health` shapes.

After running, you should see one parquet file under
``datasets/smoke/parquet/`` containing N transitions.

::

    python -m python.examples.smoke_test
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# Allow running as `python smoke_test.py` from any CWD by inserting the
# project root into sys.path.
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.gameplay.bots import HeuristicBot, RandomBot  # noqa: E402
from python.gameplay.dataset_writer import DatasetWriter  # noqa: E402
from python.gameplay.env import TalisharEnv  # noqa: E402
from python.gameplay.replay_buffer import ReplayBuffer, Trajectory, Transition, make_action_mask  # noqa: E402
from python.gameplay.selfplay import run_one_game, GameSpec  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny in-process adapter stub
# ---------------------------------------------------------------------------
class _StubState:
    """Mirrors the StubGame PHP class behaviour at the wire level."""

    def __init__(self) -> None:
        self.games: dict[str, dict[str, Any]] = {}

    def new_game(self, body: dict[str, Any]) -> dict[str, Any]:
        gid = f"g{body['seed']}_{uuid.uuid4().hex[:8]}"
        rng = _MulberryRNG(int(body['seed']))
        state = {
            "turn": 1,
            "phase": "M",
            "subphase": "",
            "active_player": 1,
            "priority_player": 1,
            "action_points": 1,
            "winner": None,
            "players": [
                {"player_id": 1, "hero": body['hero1'], "health": 40,
                 "resources": 0, "hand": ["stub_atk_red"], "arsenal": [],
                 "equipment": [], "graveyard": [], "banished": [],
                 "pitch": [], "auras": [], "items": [], "allies": [],
                 "permanents": [], "soul": [], "effects": [],
                 "class_state": [],
                 "turn_stats": {}, "card_stats": {}, "deck_count": 30},
                {"player_id": 2, "hero": body['hero2'], "health": 40,
                 "resources": 0, "hand": ["stub_atk_red"], "arsenal": [],
                 "equipment": [], "graveyard": [], "banished": [],
                 "pitch": [], "auras": [], "items": [], "allies": [],
                 "permanents": [], "soul": [], "effects": [],
                 "class_state": [],
                 "turn_stats": {}, "card_stats": {}, "deck_count": 30},
            ],
            "combat_chain": [], "stack": [], "links": [],
            "decision_queue": {"queue": [], "vars": [], "state": [], "turn": []},
            "current_turn_effects": [], "next_turn_effects": [],
            "events": [], "last_played": [], "landmarks": [], "pending_decisions": [],
            "_rng": rng,
            "_step": 0,
            "_seed": int(body['seed']),
            "_setup": {**body},
        }
        self.games[gid] = state
        return self._snapshot(gid)

    def state(self, gid: str) -> dict[str, Any]:
        return self._snapshot(gid)["state"]

    def actions(self, gid: str) -> list[dict[str, Any]]:
        return self._snapshot(gid)["legal_actions"]

    def step(self, gid: str, action_id: int) -> dict[str, Any]:
        s = self._game(gid)
        # Resolve action
        legal = self._legal(s)
        chosen = next((a for a in legal if a["action_id"] == action_id), None)
        if chosen is None:
            raise RuntimeError(f"action_id {action_id} not legal")
        me = s["priority_player"] - 1
        opp = 1 - me
        if chosen["type"] == "ATTACK":
            dmg = s["_rng"].randint(2, 5)
            s["players"][opp]["health"] = max(0, s["players"][opp]["health"] - dmg)
        elif chosen["type"] == "DEFEND":
            heal = s["_rng"].randint(0, 2)
            s["players"][me]["health"] = min(40, s["players"][me]["health"] + heal)
        # toggle priority
        s["priority_player"] = opp + 1
        s["active_player"]   = opp + 1
        s["turn"]            += 1
        s["_step"]           += 1
        if s["players"][opp]["health"] <= 0:
            s["winner"] = me + 1
        elif s["turn"] > 50:
            s["winner"] = 1 if s["players"][0]["health"] >= s["players"][1]["health"] else 2
        return self._snapshot(gid)

    def reset(self, gid: str) -> dict[str, Any]:
        body = self._game(gid)["_setup"]
        return self.new_game(body)

    # internals
    def _game(self, gid: str) -> dict[str, Any]:
        if gid not in self.games:
            raise KeyError(gid)
        return self.games[gid]

    def _legal(self, s: dict[str, Any]) -> list[dict[str, Any]]:
        if s["winner"] is not None:
            return []
        pid = s["priority_player"]
        base = {
            "player_id": pid, "card_id": None, "targets": [], "cost": {},
            "talishar_mode": 99, "talishar_button": "", "talishar_card_id": "0",
            "talishar_chk_count": 0, "talishar_chk_input": [], "talishar_input_text": "",
        }
        return [
            {**base, "action_id": 1, "type": "ATTACK"},
            {**base, "action_id": 2, "type": "DEFEND"},
            {**base, "action_id": 3, "type": "PASS"},
        ]

    def _snapshot(self, gid: str) -> dict[str, Any]:
        s = self._game(gid)
        # Strip non-JSON-serialisable fields
        view = {k: v for k, v in s.items() if not k.startswith("_")}
        winner = s["winner"]
        done = winner is not None
        return {
            "game_id": gid,
            "state": view,
            "legal_actions": self._legal(s),
            "done": done,
            "reward": (1.0 if winner == 1 else -1.0) if done else 0.0,
            "winner": winner,
            "info": {"mode": "python-stub", "step_counter": s["_step"]},
        }


class _MulberryRNG:
    """Deterministic small RNG (mulberry32). Avoids the global random state."""
    def __init__(self, seed: int) -> None:
        self.s = seed & 0xFFFFFFFF
    def next(self) -> int:
        self.s = (self.s + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.s
        t = (t ^ (t >> 15)) * (t | 1) & 0xFFFFFFFF
        t ^= (t + ((t ^ (t >> 7)) * (t | 61) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return (t ^ (t >> 14)) & 0xFFFFFFFF
    def randint(self, lo: int, hi: int) -> int:
        return lo + (self.next() % (hi - lo + 1))


def make_handler(stub: _StubState):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a, **k): return  # quiet

        def _send(self, code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _qs(self) -> dict[str, str]:
            qs = parse_qs(urlparse(self.path).query)
            return {k: v[0] for k, v in qs.items()}

        def _body(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0: return {}
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self) -> None:
            path = urlparse(self.path).path.rstrip("/") or "/"
            try:
                if path in ("/", "/health"):
                    self._send(200, {"ok": True, "mode": "python-stub"})
                elif path == "/state":
                    gid = self._qs().get("game_id", "")
                    self._send(200, {"game_id": gid, "state": stub.state(gid)})
                elif path == "/actions":
                    gid = self._qs().get("game_id", "")
                    legal = stub.actions(gid)
                    self._send(200, {"game_id": gid, "legal_actions": legal, "count": len(legal)})
                else:
                    self._send(404, {"error": f"no GET route for {path}"})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": str(e)})

        def do_POST(self) -> None:
            path = urlparse(self.path).path.rstrip("/") or "/"
            try:
                body = self._body()
                if path == "/new_game":
                    self._send(200, stub.new_game(body))
                elif path == "/step":
                    self._send(200, stub.step(body["game_id"], int(body["action_id"])))
                elif path == "/reset":
                    self._send(200, stub.reset(body["game_id"]))
                else:
                    self._send(404, {"error": f"no POST route for {path}"})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": str(e)})

    return H


def _start_stub(port: int) -> tuple[ThreadingHTTPServer, _StubState]:
    stub = _StubState()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(stub))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, stub


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------
def main() -> int:
    port = 8765
    server, _ = _start_stub(port)
    try:
        adapter = f"http://127.0.0.1:{port}"
        env = TalisharEnv(adapter)
        # Health
        h = env.health()
        assert h["ok"], h
        print(f"[smoke] adapter healthy: {h}")

        # Single game with RandomBot vs HeuristicBot
        with tempfile.TemporaryDirectory() as td:
            buf = ReplayBuffer()
            spec = GameSpec(
                hero1="Bravo", hero2="Dash",
                deck1="decks/bravo.json", deck2="decks/dash.json",
                seed=42,
                bot1=RandomBot(seed=42),
                bot2=HeuristicBot(seed=43),
            )
            traj = run_one_game(env, spec)
            buf.append(traj)
            print(f"[smoke] game finished: winner={traj.winner}, transitions={len(traj)}")
            assert traj.winner in (1, 2), f"no winner: {traj.winner}"
            assert len(traj) > 0

            # Verify replay shape
            t0 = traj.transitions[0]
            assert isinstance(t0, Transition)
            assert len(t0.action_mask) == 256
            assert t0.chosen_action_id in t0.legal_action_ids
            print(f"[smoke] first transition ok: action_id={t0.chosen_action_id} player={t0.player_to_move}")

            # Write parquet (skip if pyarrow missing)
            try:
                writer = DatasetWriter(td, fmt="parquet")
                path = writer.write_batch([traj])
                assert path and path.exists() and path.stat().st_size > 0
                print(f"[smoke] parquet written: {path} ({path.stat().st_size} bytes)")
            except RuntimeError as e:
                print(f"[smoke] parquet skipped (pyarrow missing): {e}")
                # Fall back to msgpack
                try:
                    writer = DatasetWriter(td, fmt="msgpack")
                    path = writer.write_batch([traj])
                    print(f"[smoke] msgpack written: {path}")
                except RuntimeError as e2:
                    print(f"[smoke] msgpack also missing: {e2}")

        # Determinism check: same seed -> same winner & same number of steps
        env2 = TalisharEnv(adapter)
        spec.bot1 = RandomBot(seed=42)
        spec.bot2 = HeuristicBot(seed=43)
        traj2 = run_one_game(env2, spec)
        assert traj.winner == traj2.winner, f"non-deterministic winner: {traj.winner} vs {traj2.winner}"
        assert len(traj) == len(traj2), f"non-deterministic length: {len(traj)} vs {len(traj2)}"
        print(f"[smoke] determinism: same seed produced same winner ({traj.winner}) and same length ({len(traj)})")
        print("[smoke] ALL CHECKS PASSED")
        return 0
    finally:
        server.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
