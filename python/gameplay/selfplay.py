"""Self-play orchestration.

This module is the *only* place that owns the (env, bot1, bot2, replay)
loop. Bots themselves stay pure functions; the orchestrator does the
HTTP I/O and the replay capture.

CLI usage
---------
::

    python -m python.selfplay --adapter http://localhost:8000 \\
        --hero1 Bravo --hero2 Dash \\
        --deck1 decks/bravo.json --deck2 decks/dash.json \\
        --bot1 random --bot2 random \\
        --games 10 --base-seed 1000 --out datasets

The module is also importable so other scripts (e.g. a parallel runner
or notebook) can call ``run_selfplay_batch(...)`` directly.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from .bots import Bot, HeuristicBot, RandomBot
from .dataset_writer import DatasetWriter
from .env import Action, TalisharEnv, wait_for_adapter
from .replay_buffer import ReplayBuffer, Trajectory, Transition, make_action_mask


# ---------------------------------------------------------------------------
# Bot resolution
# ---------------------------------------------------------------------------
_BOT_REGISTRY = {
    "random":      lambda seed: RandomBot(seed=seed),
    "heuristic":   lambda seed: HeuristicBot(seed=seed),
}


def _build_bot(name: str, seed: int) -> Bot:
    name = name.strip().lower()
    if name == "transformer":
        # Lazy import keeps torch optional for the common random/heuristic case.
        from .bots.transformer_bot import TransformerBot, TransformerConfig
        return TransformerBot(TransformerConfig(), seed=seed)
    try:
        return _BOT_REGISTRY[name](seed)
    except KeyError as e:
        raise SystemExit(f"Unknown bot: {name!r} (choices: {list(_BOT_REGISTRY) + ['transformer']})") from e


# ---------------------------------------------------------------------------
# One-game loop
# ---------------------------------------------------------------------------
@dataclass
class GameSpec:
    hero1: str
    hero2: str
    deck1: str
    deck2: str
    seed: int
    bot1: Bot
    bot2: Bot
    max_actions_mask: int = 256
    step_cap: int = 2000  # safety belt against infinite-loop bugs
    game_format: str = "draft"  # "draft" (OMN) or "cc" (Classic Constructed)
    # Abort the game if coarse game-progress markers don't change for this many
    # consecutive steps (0 = off). Catches HARD engine wedges (everything
    # frozen) that the OMN run_match loop's rewind handles but this lean loop
    # doesn't — turns a step_cap grind into a fast, logged draw.
    no_progress_cap: int = 0
    # Abort if NEITHER player's life changes for this many steps (0 = off), even
    # if other state (zones/deck) churns. Catches SOFT stalls/loops the
    # full-signature no_progress guard misses — a card replayed every turn or
    # two bots cycling without closing (life is the true progress signal).
    # Higher than no_progress_cap: life legitimately sits still during setup.
    life_stall_cap: int = 0


def _progress_sig(state: dict) -> tuple:
    """Coarse game-progress markers. If these don't change for many consecutive
    steps the game is wedged (engine stuck re-offering a no-op), not just slow.
    Deliberately excludes the stack (it churns during a wedge loop)."""
    parts = [state.get("phase"), state.get("turn")]
    for p in state.get("players") or []:
        parts += [p.get("health"), p.get("deck_count"),
                  len(p.get("hand") or []), len(p.get("graveyard") or []),
                  len(p.get("pitch") or []), len(p.get("arsenal") or [])]
    return tuple(str(x) for x in parts)


def _life_pair(state: dict) -> tuple:
    return tuple(p.get("health") for p in (state.get("players") or []))


def run_one_game(env: TalisharEnv, spec: GameSpec) -> Trajectory:
    """Drive a single game to completion and return the trajectory."""
    spec.bot1.reset(seed=spec.seed)
    spec.bot2.reset(seed=spec.seed + 1)

    init = env.reset(
        hero1=spec.hero1, hero2=spec.hero2,
        deck1=spec.deck1, deck2=spec.deck2,
        seed=spec.seed,
        format=spec.game_format,
    )
    state = init.state
    legal = init.legal_actions

    traj = Trajectory(
        game_id=env.game_id,
        seed=spec.seed,
        hero1=spec.hero1, hero2=spec.hero2,
        deck1=spec.deck1, deck2=spec.deck2,
        metadata={
            "bot1": repr(spec.bot1),
            "bot2": repr(spec.bot2),
            "adapter_base": env.base_url,
        },
    )

    step_idx = 0
    last_sig = None
    stale = 0
    last_life = None
    life_stale = 0
    # Loop-breaker state: per progress-signature, the action_ids we've OBSERVED
    # to revert to that exact signature (true no-ops, e.g. a PASS the engine
    # won't honour for a mandatory CHOOSEMULTIZONE). Once a state has clearly
    # stalled we hide those from the bot so a deterministic (argmax) policy is
    # forced to try something that makes progress. Only ever prunes actions seen
    # to be no-ops, and never empties the action set.
    tried_noop: dict = {}
    loopbreak_after = 4   # grace steps before pruning (avoid acting on a fluke)
    while not env.done and step_idx < spec.step_cap:
        priority = int(state.get("priority_player", 0))
        if priority not in (1, 2):
            break  # malformed state — abort cleanly
        bot = spec.bot1 if priority == 1 else spec.bot2

        if not legal:
            # Engine wants no input — request a refresh; if still empty, terminate.
            legal = env.get_actions(refresh=True)
            if not legal:
                break

        # Hide known no-op actions for this state once we're clearly stalled.
        cur_sig = _progress_sig(state) if spec.no_progress_cap > 0 else None
        avail = legal
        if cur_sig is not None and stale >= loopbreak_after:
            noops = tried_noop.get(cur_sig)
            if noops:
                pruned = [a for a in legal if a.action_id not in noops]
                if pruned:           # never hand the bot an empty set
                    avail = pruned

        decision = bot.choose(state, avail, player_id=priority)
        legal_ids = [a.action_id for a in avail]
        mask = make_action_mask(legal_ids, spec.max_actions_mask)

        # Find chosen Action for trajectory metadata.
        chosen = next((a for a in avail if a.action_id == decision.action_id), None)
        if chosen is None:
            raise RuntimeError(
                f"Bot {bot.name} returned action_id {decision.action_id} not in legal set "
                f"{legal_ids} on step {step_idx}"
            )

        ts = time.time()
        result = env.step(decision.action_id)

        traj.add(Transition(
            state=state,
            legal_actions=[a.raw for a in avail],
            legal_action_ids=legal_ids,
            action_mask=mask,
            chosen_action=chosen.raw,
            chosen_action_id=decision.action_id,
            reward=result.reward,
            next_state=result.state,
            done=result.done,
            player_to_move=priority,
            step_index=step_idx,
            ts_unix=ts,
        ))

        state = result.state
        legal = result.legal_actions
        step_idx += 1

        if spec.no_progress_cap > 0:
            sig = _progress_sig(state)
            if sig == last_sig:
                stale += 1
                # The action we just took left the signature unchanged -> it's a
                # no-op at cur_sig; remember it so we can prune it next time.
                if cur_sig is not None:
                    tried_noop.setdefault(cur_sig, set()).add(decision.action_id)
                if stale >= spec.no_progress_cap:
                    traj.metadata["aborted"] = f"no_progress_{stale}_steps"
                    break
            else:
                stale = 0
                last_sig = sig

        if spec.life_stall_cap > 0:
            lp = _life_pair(state)
            if lp == last_life:
                life_stale += 1
                if life_stale >= spec.life_stall_cap:
                    traj.metadata["aborted"] = f"life_stall_{life_stale}_steps"
                    break
            else:
                life_stale = 0
                last_life = lp

    traj.finalise(winner=env.winner or 0)
    return traj


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------
def run_selfplay_batch(
    *,
    adapter_url: str,
    hero1: str,
    hero2: str,
    deck1: str,
    deck2: str,
    bot1: Bot,
    bot2: Bot,
    n_games: int,
    base_seed: int,
    out_dir: str,
    fmt: str = "parquet",
    game_format: str = "draft",
    max_actions_mask: int = 256,
    flush_every: int = 16,
    on_game=None,
    step_cap: int = 2000,
    no_progress_cap: int = 0,
    life_stall_cap: int = 0,
) -> int:
    """Run N games against a single adapter and persist trajectories.

    Returns the number of games actually completed.
    """
    wait_for_adapter(adapter_url, timeout_s=30.0)
    writer = DatasetWriter(out_dir, fmt=fmt)  # type: ignore[arg-type]
    buf = ReplayBuffer(max_trajectories=flush_every)

    completed = 0
    with TalisharEnv(adapter_url) as env:
        for i in range(n_games):
            spec = GameSpec(
                hero1=hero1, hero2=hero2, deck1=deck1, deck2=deck2,
                seed=base_seed + i,
                bot1=bot1, bot2=bot2,
                max_actions_mask=max_actions_mask,
                game_format=game_format,
                step_cap=step_cap,
                no_progress_cap=no_progress_cap,
                life_stall_cap=life_stall_cap,
            )
            t0 = time.monotonic()
            try:
                traj = run_one_game(env, spec)
            except Exception as e:  # noqa: BLE001
                print(f"[selfplay] game {i} failed: {e}", file=sys.stderr)
                continue
            dt = time.monotonic() - t0
            print(
                f"[selfplay] game {completed:04d}/{n_games} seed={spec.seed} "
                f"winner={traj.winner} steps={len(traj)} dt={dt:.2f}s",
                file=sys.stderr,
            )
            buf.append(traj)
            completed += 1
            if on_game is not None:
                try:
                    on_game(traj)
                except Exception:  # noqa: BLE001 — telemetry must never break the run
                    pass

            if len(buf) >= flush_every:
                path = writer.write_batch(buf.drain())
                if path:
                    print(f"[selfplay] flushed batch -> {path}", file=sys.stderr)

        # Final flush.
        path = writer.write_batch(buf.drain())
        if path:
            print(f"[selfplay] final flush -> {path}", file=sys.stderr)

    return completed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run FAB_Sim_Headless self-play.")
    p.add_argument("--adapter", default=os.getenv("ADAPTER_URL", "http://localhost:8000"))
    p.add_argument("--hero1", required=True)
    p.add_argument("--hero2", required=True)
    p.add_argument("--deck1", required=True)
    p.add_argument("--deck2", required=True)
    p.add_argument("--bot1", default="random")
    p.add_argument("--bot2", default="random")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=1000)
    p.add_argument("--out", default="datasets")
    p.add_argument("--fmt", choices=["parquet", "msgpack", "npz"], default="parquet")
    p.add_argument("--max-actions-mask", type=int, default=256)
    p.add_argument("--flush-every", type=int, default=16)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    bot1 = _build_bot(args.bot1, seed=args.base_seed)
    bot2 = _build_bot(args.bot2, seed=args.base_seed + 999)
    n = run_selfplay_batch(
        adapter_url=args.adapter,
        hero1=args.hero1, hero2=args.hero2,
        deck1=args.deck1, deck2=args.deck2,
        bot1=bot1, bot2=bot2,
        n_games=args.games,
        base_seed=args.base_seed,
        out_dir=args.out,
        fmt=args.fmt,
        max_actions_mask=args.max_actions_mask,
        flush_every=args.flush_every,
    )
    print(f"[selfplay] completed {n}/{args.games} games", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
