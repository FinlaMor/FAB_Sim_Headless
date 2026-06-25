"""One-match driver.

Plays a single Bo1 (best-of-one) match between two players via the
headless Talishar adapter. Best-of-N is layered on top by
:class:`TournamentRunner` (just call ``run_match`` repeatedly until
someone hits the win threshold).

Talishar owns the rules; this module only:

* opens a new game via ``env.reset(hero1=..., deck1=..., ...)``,
* drives each priority window through the player's gameplay bot,
* returns the winner and the captured trajectory.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..gameplay.env import TalisharEnv
from ..gameplay.replay_buffer import Trajectory, Transition, make_action_mask
from .player import Player

# Project root = .../FAB_Sim_Headless (this file is python/tournament/match.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Decks MUST live under a directory bind-mounted into the adapter
# container. docker-compose mounts ./decks at /srv/decks (read-only is
# fine — the host writes, the container reads). We write here and pass a
# path relative to the project root; the adapter resolves it against
# PROJECT_ROOT (=/srv inside the container). A host tempdir is invisible
# to the container, so we cannot use tempfile here.
_DECK_REL_DIR = "decks/_tmp_matches"
_DECK_DIR = _PROJECT_ROOT / _DECK_REL_DIR


@dataclass
class MatchResult:
    match_id: str
    p1_label: str
    p2_label: str
    p1_name: str
    p2_name: str
    winner_label: str
    winner_seat: int
    game_seed: int
    trajectory: Trajectory
    started_at: float
    ended_at: float
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _life_of(state: dict[str, Any], seat0_or_1: int) -> int | None:
    """Return the health of the player in `players` whose id == seat+1."""
    pid = seat0_or_1 + 1
    for p in state.get("players", []) or []:
        if int(p.get("player_id", 0)) == pid:
            h = p.get("health")
            return int(h) if h is not None else None
    return None


def _hand_count(state: dict[str, Any], player_id: int) -> int:
    """Number of cards in `player_id`'s hand (used to tell a real play, which
    consumes a card, from an engine-cancelled no-op, which restores it)."""
    for p in state.get("players", []) or []:
        if int(p.get("player_id", 0)) == player_id:
            return len(p.get("hand") or [])
    return -1


def _progress_sig(state: dict[str, Any], priority: int) -> tuple:
    """Visible-progress signature for engine-cancel detection. MUST cover every
    way a real action can change the game, or a legitimate move gets falsely
    dead-listed and dropped from training (2026-06-10 audit: abilities whose
    only effect was resources/auras/stack looked identical under the old
    turn/phase/life/hand-count signature). A true engine cancel restores ALL
    of these, so a richer signature only removes false positives."""
    boards = []
    for p in state.get("players", []) or []:
        boards.append((
            int(p.get("resources") or 0),
            len(p.get("hand") or []), len(p.get("pitch") or []),
            len(p.get("arsenal") or []), len(p.get("graveyard") or []),
            len(p.get("banished") or []),
            len(p.get("auras") or []) + len(p.get("items") or [])
            + len(p.get("allies") or []) + len(p.get("permanents") or []),
            int(p.get("deck_count") or 0),
        ))
    return (state.get("turn"), state.get("phase"), priority,
            _life_of(state, 0), _life_of(state, 1),
            len(state.get("combat_chain") or []), len(state.get("stack") or []),
            tuple(boards))


def run_match(
    *,
    env: TalisharEnv,
    p1: Player,
    p2: Player,
    match_id: str,
    seed: int,
    max_actions_mask: int = 256,
    step_cap: int = 2000,
    life_tiebreak: bool = False,
    wedge_limit: int = 60,
) -> MatchResult:
    """Play one Bo1 game and return the result + replay.

    ``env`` MUST already be connected to a healthy adapter. The decks
    are written to a temp dir and passed by path to ``env.reset``.

    A game is won only the way real FAB games are: by an engine winner
    (lethal / deck-out / concede). If the step cap is reached with no
    winner the game is a genuine DRAW (winner 0, zero terminal reward).
    The life-total tiebreak is OFF by default — it rewarded out-lifing
    the opponent on the clock, which let the policy coast to safe
    life-margin "wins" instead of actually closing games. ``life_tiebreak``
    remains as an opt-in escape hatch but is not used by the loop.
    """
    started = time.time()
    _DECK_DIR.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", match_id) + "_" + uuid.uuid4().hex[:6]
    d1 = _DECK_DIR / f"{tag}_p1.json"
    d2 = _DECK_DIR / f"{tag}_p2.json"
    rel1 = f"{_DECK_REL_DIR}/{d1.name}"  # path the (containerised) adapter resolves
    rel2 = f"{_DECK_REL_DIR}/{d2.name}"
    try:
        p1.deck.save_json(str(d1))
        p2.deck.save_json(str(d2))

        init = env.reset(
            hero1=p1.deck.hero, hero2=p2.deck.hero,
            deck1=rel1, deck2=rel2,
            seed=seed,
        )
        bot1 = p1.bot_factory(seed)
        bot2 = p2.bot_factory(seed + 1)
        bot1.reset(seed=seed)
        bot2.reset(seed=seed + 1)

        traj = Trajectory(
            game_id=env.game_id,
            seed=seed,
            hero1=p1.deck.hero, hero2=p2.deck.hero,
            deck1=rel1, deck2=rel2,
            metadata={
                "match_id": match_id,
                "p1_label": p1.label, "p2_label": p2.label,
                "p1_name": p1.name, "p2_name": p2.name,
            },
        )
        state = init.state
        legal = init.legal_actions

        error: str | None = None
        # Why the game loop exited. Set at each break; resolved to
        # "engine_winner"/"step_cap" after the loop. This is the ONLY thing
        # that separates a real stalemate (step_cap) from an engine/adapter
        # abort (bad_priority / no_legal / *_error) — both otherwise look like
        # an identical winner-0 "draw" downstream.
        term_reason: str | None = None
        step_idx = 0
        wedge_count = 0      # consecutive no-progress forced-PASS windows
        wedge_sig = None     # (turn, phase, p1hp, p2hp) progress signature
        dead_actions: set = set()  # (card_id, type) unpayable no-ops at this state
        cancel_count = 0     # consecutive engine-cancelled (no-op) plays
        recover = 0          # consecutive stale-action recoveries
        recover_total = 0    # total recoveries this game (for diagnostics)
        while not env.done and step_idx < step_cap:
            priority = int(state.get("priority_player", 0))
            if priority not in (1, 2):
                term_reason = "bad_priority"
                break
            bot = bot1 if priority == 1 else bot2
            if not legal:
                legal = env.get_actions(refresh=True)
                if not legal:
                    term_reason = "no_legal"
                    break
            # Wedge guard: the engine can DEADLOCK in the pay ("P") phase when a
            # player activates an ability/plays a card it can't fully fund — only
            # PASS is offered, PASS neither pays nor cancels, and the game burns
            # its whole step budget passing with turn/health frozen (otherwise
            # miscounted as a step-cap DRAW). Detect a run of no-progress
            # forced-PASS windows and end the game with a distinct term_reason.
            sig = (state.get("turn"), state.get("phase"),
                   _life_of(state, 0), _life_of(state, 1))
            if len(legal) == 1 and sig == wedge_sig:
                wedge_count += 1
                if wedge_count >= wedge_limit:
                    term_reason = "wedge"
                    break
            else:
                wedge_count = 0
                wedge_sig = sig
            # Forced windows — exactly one legal action (pitch prompts, the
            # opponent's turn, an empty stack) — are not decisions: the only
            # move is almost always PASS. Take it WITHOUT invoking the policy
            # net (this is ~85% of all priority windows, so it's most of the
            # per-game inference cost) and do NOT record it as a training
            # transition, so the policy learns from real choices instead of
            # drowning in forced PASSes. A forced step that ends the game is
            # still recorded below so the terminal reward attaches.
            # Drop actions already proven to be unpayable no-ops at THIS state:
            # the engine cancels them (returns the card, restores the phase), so
            # re-choosing one just loops. `dead_actions` is cleared whenever the
            # state advances (below). PASS is never excluded, so this can't empty
            # the menu.
            choosable = [a for a in legal if (a.card_id, a.type) not in dead_actions] or legal
            forced = len(choosable) == 1
            if forced:
                chosen = choosable[0]
            else:
                try:
                    decision = bot.choose(state, choosable, player_id=priority)
                except Exception as e:  # noqa: BLE001
                    error = f"bot {priority} crashed at step {step_idx}: {e!r}"
                    term_reason = "bot_crash"
                    break
                chosen = next((a for a in choosable if a.action_id == decision.action_id), None)
                if chosen is None:
                    error = f"illegal action {decision.action_id} at step {step_idx}"
                    term_reason = "illegal_action"
                    break
            pre_sig = _progress_sig(state, priority)
            try:
                result = env.step(chosen.action_id)
            except RuntimeError as e:
                # The legal set can shift between the read that produced
                # `legal` and this step (rare: the decision queue auto-
                # advanced, or a phase flipped). A 409 means the action was
                # rejected and NOT applied, so the game is intact — refresh
                # the actual current actions/state and re-choose. One bad
                # step must never kill the whole tournament.
                recover += 1
                recover_total += 1
                if recover > 25:
                    error = f"unrecoverable stale action at step {step_idx}: {e!r}"
                    term_reason = "stale_action"
                    break
                legal = env.get_actions(refresh=True)
                state = env.get_state(refresh=True)
                continue
            except Exception as e:  # noqa: BLE001
                # Transport-level failure (read timeout / dropped connection):
                # the adapter stalled or died. Abort just THIS game with an
                # error — the pair keeps its other games, and pair-level
                # isolation in the parallel runner absorbs a hard env death.
                error = f"transport error at step {step_idx}: {e!r}"
                term_reason = "transport"
                break
            recover = 0
            # Engine-cancel detection / rewind: an unpayable play is cancelled
            # in-engine (card returned to hand, phase restored — see
            # NetworkingLibraries.php::PlayCard), so a NON-PASS action that
            # leaves the visible state identical was a reverted no-op. Do NOT
            # record it (the model must not train on attempted-but-cancelled
            # plays) and remember it so the bot doesn't re-pick it and loop.
            post = result.state
            post_sig = _progress_sig(post, priority)
            if chosen.type != "PASS" and post_sig == pre_sig:
                dead_actions.add((chosen.card_id, chosen.type))
                cancel_count += 1
                if cancel_count >= 20:
                    term_reason = "cancel_loop"
                    break
                state = post
                legal = result.legal_actions
                step_idx += 1
                continue
            # A real step advanced the game — clear the per-state cancel memory.
            dead_actions.clear()
            cancel_count = 0
            # Record real decisions, plus any terminal step (even a forced one)
            # so the terminal reward is never lost.
            if not forced or result.done:
                # Record the FILTERED menu: dead-listed (engine-cancelled)
                # actions are not really available, and training on menus
                # that contain them teaches the policy phantom options.
                legal_ids = [a.action_id for a in choosable]
                traj.add(Transition(
                    state=state,
                    legal_actions=[a.raw for a in choosable],
                    legal_action_ids=legal_ids,
                    action_mask=make_action_mask(legal_ids, max_actions_mask),
                    chosen_action=chosen.raw,
                    chosen_action_id=chosen.action_id,
                    reward=result.reward,
                    next_state=result.state,
                    done=result.done,
                    player_to_move=priority,
                    step_index=step_idx,
                    ts_unix=time.time(),
                ))
            state = result.state
            legal = result.legal_actions
            step_idx += 1

        # Resolve the exit reason for the paths that didn't break explicitly:
        # the engine declared a winner, or we hit the step cap (a real draw).
        if env.done:
            term_reason = "engine_winner"
        elif term_reason is None:
            term_reason = "step_cap"

        engine_winner = env.winner or 0  # 1, 2, or 0
        winner = engine_winner
        tiebreak = False
        if winner == 0 and life_tiebreak and error is None:
            l1, l2 = _life_of(state, 0), _life_of(state, 1)
            if l1 is not None and l2 is not None and l1 != l2:
                winner = 1 if l1 > l2 else 2
                tiebreak = True

        winner_seat = winner - 1  # -1 for a genuine draw
        winner_label = p1.label if winner == 1 else (p2.label if winner == 2 else "")
        # Terminal reward reflects the real match outcome: a win only for an
        # actual engine winner, zero for a step-cap draw. With no tiebreak
        # consolation, the only path to positive reward is to close the game.
        traj.finalise(winner=winner)
    finally:
        # Best-effort cleanup of the per-match deck files.
        for d in (d1, d2):
            try:
                d.unlink()
            except OSError:
                pass

    return MatchResult(
        match_id=match_id,
        p1_label=p1.label, p2_label=p2.label,
        p1_name=p1.name, p2_name=p2.name,
        winner_label=winner_label, winner_seat=winner_seat,
        game_seed=seed, trajectory=traj,
        started_at=started, ended_at=time.time(),
        error=error,
        metadata={"tiebreak": tiebreak, "engine_winner": engine_winner,
                  "steps": step_idx, "recoveries": recover_total,
                  "term_reason": term_reason},
    )
