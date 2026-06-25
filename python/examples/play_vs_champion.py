"""Play a live game against the current IQL champion, on the real gate decks.

The champion (outputs/models/gameplay/latest.pt by default) drives one seat via
the SAME IQLGameplayBot the promotion gate uses; YOU drive the other seat from a
numbered legal-action menu. The game runs on a dedicated adapter instance (a
separate port from the training pipeline) so nothing here disturbs continuous
training. Point the real Talishar FE at the same game id to watch the board.

This mirrors python.tournament.match.run_match exactly — forced single-action
windows auto-resolve and unpayable plays are engine-cancelled/rewound the same
way — so the champion you face behaves identically to the one in the gate.

Example
-------
    python -m python.examples.play_vs_champion \
        --url http://localhost:8090 \
        --my-deck outputs/gate_decks/gate_01_zyggy.json \
        --champ-deck outputs/gate_decks/gate_00_oscili.json \
        --my-seat 1 --seed 4242
"""
from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Windows consoles default to cp1252 and choke on non-ASCII; make stdout robust.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from python.deckbuilding.deck import Deck                       # noqa: E402
from python.gameplay.env import Action, TalisharEnv            # noqa: E402
from python.gameplay.bots.iql_bot import IQLGameplayBot        # noqa: E402

_DECK_REL_DIR = "decks/_tmp_matches"
_DECK_DIR = PROJECT_ROOT / _DECK_REL_DIR


# ---------------------------------------------------------------------------
# deck / rendering helpers
# ---------------------------------------------------------------------------
def _load_deck(path: str) -> Deck:
    import json
    d = json.load(open(path, encoding="utf-8"))
    eq = d.get("equipment", []) or []
    return Deck(hero=d["hero"], weapon=(eq[0] if eq else ""),
               deck=list(d.get("deck", [])), equipment=list(eq[1:]))


def _player(state: dict, pid: int) -> dict:
    for p in state.get("players", []) or []:
        if int(p.get("player_id", 0)) == pid:
            return p
    return {}


def _pretty_card(cid: str) -> str:
    return (cid or "").replace("_", " ").strip() or "?"


def _action_label(a: Action) -> str:
    """One readable line for a legal action."""
    t = (a.type or "").upper()
    if t == "PASS":
        return "PASS / end of options"
    card = _pretty_card(a.card_id) if a.card_id else ""
    cost = ""
    c = a.cost or {}
    if isinstance(c, dict):
        rc = c.get("resource_cost", c.get("cost"))
        if rc not in (None, "", 0, "0"):
            cost = f"  (cost {rc})"
    pieces = [t]
    if card:
        pieces.append(card)
    return " ".join(pieces) + cost


def _render_board(state: dict, my_pid: int, champ_pid: int) -> str:
    me, opp = _player(state, my_pid), _player(state, champ_pid)
    turn = state.get("turn"); phase = state.get("phase"); sub = state.get("subphase") or ""
    lines = []
    lines.append(f"  -- turn {turn}  phase {phase}{('/' + sub) if sub else ''}"
                 f"   priority P{state.get('priority_player')} --")

    def row(tag: str, p: dict) -> str:
        return (f"  {tag} P{p.get('player_id')} {p.get('hero','?'):<26} "
                f"HP {str(p.get('health')):>3}  "
                f"pitch {p.get('resources',0)}  "
                f"hand {len(p.get('hand') or [])}  "
                f"ars {len(p.get('arsenal') or [])}  "
                f"deck {p.get('deck_count',0)}")

    lines.append(row("YOU  ", me))
    lines.append(row("CHAMP", opp))

    cb = state.get("combat") or {}
    if cb.get("active"):
        lines.append(f"  [combat] attacker P{cb.get('attacker')} "
                     f"power {cb.get('attack_power')} vs block {cb.get('total_block')} "
                     f"-> {cb.get('pending_damage')} pending"
                     + ("  GO-AGAIN" if cb.get('go_again') else "")
                     + ("  DOMINATE" if cb.get('dominate') else ""))
    hand = me.get("hand") or []
    if hand:
        lines.append("  your hand: " + ", ".join(_pretty_card(c) for c in hand))
    ars = me.get("arsenal") or []
    if ars:
        lines.append("  your arsenal: " + ", ".join(_pretty_card(c) for c in ars))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# human input
# ---------------------------------------------------------------------------
def _human_choose(state: dict, legal: list[Action], my_pid: int,
                  champ_pid: int) -> Action:
    print("\n" + _render_board(state, my_pid, champ_pid))
    print("\n  YOUR MOVE:")
    for i, a in enumerate(legal):
        print(f"    [{i}] {_action_label(a)}")
    while True:
        try:
            raw = input("  > ").strip().lower()
        except EOFError:
            raw = "q"
        if raw in ("q", "quit", "exit"):
            raise KeyboardInterrupt
        if raw in ("b", "board"):
            print("\n" + _render_board(state, my_pid, champ_pid))
            continue
        if raw == "":
            # bare Enter = take PASS if available, else the first option
            idx = next((i for i, a in enumerate(legal)
                        if (a.type or "").upper() == "PASS"), 0)
            return legal[idx]
        try:
            idx = int(raw)
            if 0 <= idx < len(legal):
                return legal[idx]
        except ValueError:
            pass
        print(f"  ? enter 0-{len(legal) - 1}  (Enter=pass, b=board, q=quit)")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _life(state: dict, pid: int):
    p = _player(state, pid)
    h = p.get("health")
    return int(h) if h is not None else None


def _hand_count(state: dict, pid: int) -> int:
    return len(_player(state, pid).get("hand") or [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8090",
                    help="adapter base url (a DEDICATED instance, not 8000-8007)")
    ap.add_argument("--my-deck", default="")
    ap.add_argument("--champ-deck", default="")
    ap.add_argument("--champ-ckpt",
                    default=str(PROJECT_ROOT / "outputs/models/gameplay/latest.pt"))
    ap.add_argument("--my-seat", type=int, default=1, choices=(1, 2))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step-cap", type=int, default=2000)
    args = ap.parse_args()

    seed = args.seed or random.randint(1, 10_000_000)
    gate = sorted(glob.glob(str(PROJECT_ROOT / "outputs/gate_decks/*.json")))
    my_deck_path = args.my_deck or (gate[1] if len(gate) > 1 else gate[0])
    champ_deck_path = args.champ_deck or gate[0]

    my_pid = args.my_seat
    champ_pid = 2 if my_pid == 1 else 1
    my_deck = _load_deck(my_deck_path)
    champ_deck = _load_deck(champ_deck_path)
    # seat 1 = hero1, seat 2 = hero2
    d1, d2 = (my_deck, champ_deck) if my_pid == 1 else (champ_deck, my_deck)

    _DECK_DIR.mkdir(parents=True, exist_ok=True)
    p1_file = _DECK_DIR / f"play_p1_{seed}.json"
    p2_file = _DECK_DIR / f"play_p2_{seed}.json"
    d1.save_json(str(p1_file))
    d2.save_json(str(p2_file))

    champ = IQLGameplayBot(checkpoint=args.champ_ckpt, seed=seed,
                           temperature=0.0, epsilon=0.0)
    using_net = champ._net is not None
    print("=" * 64)
    print(f"  YOU   P{my_pid}: {my_deck.hero}   ({Path(my_deck_path).name})")
    print(f"  CHAMP P{champ_pid}: {champ_deck.hero}   ({Path(champ_deck_path).name})")
    print(f"  champion policy: {'IQL net' if using_net else 'FALLBACK BalancedBot (ckpt failed to load!)'}")
    print(f"  ckpt: {args.champ_ckpt}")
    print(f"  seed: {seed}")
    print("=" * 64)

    with TalisharEnv(args.url, timeout=60.0) as env:
        init = env.reset(hero1=d1.hero, hero2=d2.hero,
                         deck1=f"{_DECK_REL_DIR}/{p1_file.name}",
                         deck2=f"{_DECK_REL_DIR}/{p2_file.name}",
                         seed=seed)
        # Drop a marker so spectate.php auto-serves this game LIVE to the FE.
        live_marker = PROJECT_ROOT / "talishar" / "Games" / env.game_id / ".headless_live"
        try:
            live_marker.write_text("1", encoding="utf-8")
        except OSError:
            live_marker = None
        print(f"\n  game id: {env.game_id}")
        print(f"  WATCH IN FE (spectate):  "
              f"http://localhost:5173/game/play/{env.game_id}\n")

        state = init.state
        legal = init.legal_actions
        steps = 0
        dead: set = set()           # engine-cancelled no-ops at this state
        cancel_run = 0
        recover = 0
        while not env.done and steps < args.step_cap:
            prio = int(state.get("priority_player", 0))
            if prio not in (1, 2):
                print(f"\n  [adapter] bad priority ({prio}); stopping.")
                break
            if not legal:
                legal = env.get_actions(refresh=True)
                if not legal:
                    print("\n  [adapter] no legal actions; stopping.")
                    break
            choosable = [a for a in legal if (a.card_id, a.type) not in dead] or legal
            forced = len(choosable) == 1

            if forced:
                chosen = choosable[0]
                if prio == my_pid and (chosen.type or "").upper() != "PASS":
                    print(f"  (auto: your only option — {_action_label(chosen)})")
            elif prio == champ_pid:
                try:
                    dec = champ.choose(state, choosable, player_id=prio)
                except Exception as e:  # noqa: BLE001
                    print(f"  [champion crashed: {e!r}] passing")
                    dec = None
                chosen = (next((a for a in choosable if a.action_id == dec.action_id), None)
                          if dec else None) or choosable[0]
                print(f"  CHAMPION > {_action_label(chosen)}")
            else:
                try:
                    chosen = _human_choose(state, choosable, my_pid, champ_pid)
                except KeyboardInterrupt:
                    print("\n  quit.")
                    break

            pre = (state.get("turn"), state.get("phase"), prio,
                   _life(state, 1), _life(state, 2), _hand_count(state, prio))
            try:
                result = env.step(chosen.action_id)
            except RuntimeError as e:
                recover += 1
                if recover > 25:
                    print(f"\n  [adapter] unrecoverable stale action: {e!r}")
                    break
                legal = env.get_actions(refresh=True)
                state = env.get_state(refresh=True)
                continue
            recover = 0

            post = result.state
            post_sig = (post.get("turn"), post.get("phase"), prio,
                        _life(post, 1), _life(post, 2), _hand_count(post, prio))
            if (chosen.type or "").upper() != "PASS" and post_sig == pre:
                # engine returned the card (unpayable / illegal) — reverted no-op
                dead.add((chosen.card_id, chosen.type))
                cancel_run += 1
                if prio == my_pid:
                    print("  (that play was reverted — couldn't be paid for / not legal)")
                if cancel_run >= 20:
                    print("\n  [adapter] cancel loop; stopping.")
                    break
                state = post
                legal = result.legal_actions
                steps += 1
                continue
            dead.clear()
            cancel_run = 0
            state = result.state
            legal = result.legal_actions
            steps += 1

        # outcome
        print("\n" + "=" * 64)
        w = env.winner or 0
        if w == my_pid:
            print("  RESULT: you WIN !")
        elif w == champ_pid:
            print("  RESULT: champion WINS")
        elif w == 0:
            print(f"  RESULT: draw / no winner after {steps} steps")
        print(f"  final  YOU P{my_pid} HP {_life(state, my_pid)}   "
              f"CHAMP P{champ_pid} HP {_life(state, champ_pid)}   (steps {steps})")
        print("=" * 64)
        if live_marker:
            try:
                live_marker.unlink()
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
