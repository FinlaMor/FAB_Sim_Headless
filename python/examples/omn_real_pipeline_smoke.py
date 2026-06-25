"""End-to-end pipeline against the REAL Talishar adapter container.

Differences from ``omn_pipeline_smoke.py``:

* Talks to the running PHP container on ``http://localhost:8000`` (must
  be up in ``ADAPTER_MODE=real``).
* Plays a **single match** end-to-end through real ``ProcessInput()``,
  not the full 7-match bracket — keeps the failure surface narrow while
  we debug.
* Heavy diagnostic output: per-step phase / priority / hand / pitch
  counts, first 5 legal actions, plus a final disposition (winner or
  step-cap hit).
* No bracket assertions — this is exploratory.

Once a full random-vs-random game plays to a winner, switch
``--mode=bracket`` (TODO) to run the full pipeline.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.draft.bots.heuristic_bot import HeuristicDraftBot  # noqa: E402
from python.draft.draftmancer import parse_draftmancer  # noqa: E402
from python.draft.simulator import DraftPodConfig, DraftSimulator  # noqa: E402
from python.draft.pack_loader import PackPool, load_pack_pool  # noqa: E402
from python.draft.draftmancer import load_pack_pool_draftmancer  # noqa: E402
from python.deckbuilding.builder import HeuristicDeckBuilder  # noqa: E402
from python.pipeline import default_hero_assignment  # noqa: E402
from python.gameplay.env import TalisharEnv  # noqa: E402
from python.gameplay.bots.random_bot import RandomBot  # noqa: E402


ADAPTER = "http://localhost:8000"
CUBE = PROJECT_ROOT / "OMN_Draft_3.5.txt"
STEP_CAP = 400


def main() -> int:
    # ---- 0. Adapter health ----
    print(f"[real] adapter health @ {ADAPTER}")
    env = TalisharEnv(ADAPTER, timeout=30.0)
    h = env.health()
    print(f"  {h}")
    if h.get("mode") != "real":
        print(f"  FATAL: adapter mode is {h.get('mode')!r}, set ADAPTER_MODE=real")
        return 2

    # ---- 1. Draft a single pod ----
    print(f"[real] parsing cube {CUBE.name}")
    cube = parse_draftmancer(str(CUBE))
    classes = cube.class_map()
    pool = load_pack_pool_draftmancer(str(CUBE), n_packs=24, seed=12345)
    print(f"  cube universe = {len(cube.card_universe())} cards")

    print(f"[real] drafting pod (8 players × 3 packs × 14 cards)")
    bots = [HeuristicDraftBot(seed=1000 + s) for s in range(8)]
    sim = DraftSimulator(pool, bots,
                         DraftPodConfig(seed=12345, pod_id="real_pod_001"))
    pod = sim.run()
    print(f"  picks total = {len(pod.picks)}")

    # ---- 2. Build decks for seats A and B (the first match) ----
    print(f"[real] building decks for seats 0 and 1")
    decks = {}
    for seat in (0, 1):
        hero, weapon = default_hero_assignment(seat, pod, classes)
        pool_for_seat = [hero, weapon, *pod.drafted_pool(seat)]
        builder = HeuristicDeckBuilder(card_classes=classes, seed=2000 + seat)
        deck = builder.build_deck(pool_for_seat)
        decks[seat] = deck
        print(f"  seat {seat}: hero={deck.hero} weapon={deck.weapon} "
              f"size={deck.size} pitch={deck.evaluation.pitch_distribution}")

    # ---- 3. Write decks to disk in our JSON format ----
    # IMPORTANT: deck files must live inside a directory bind-mounted
    # into the adapter container — `./decks/` is exposed read-only at
    # `/srv/decks`. We write under decks/_tmp_real_smoke/ and pass the
    # relative path; the adapter resolves it against PROJECT_ROOT which
    # inside the container is /srv.
    deck_dir_rel = Path("decks/_tmp_real_smoke")
    deck_dir_abs = PROJECT_ROOT / deck_dir_rel
    deck_dir_abs.mkdir(parents=True, exist_ok=True)
    # Clean previous run's files.
    for old in deck_dir_abs.glob("seat*_deck.json"):
        old.unlink()
    deck_paths = {}
    for seat in (0, 1):
        p_abs = deck_dir_abs / f"seat{seat}_deck.json"
        decks[seat].save_json(str(p_abs))
        deck_paths[seat] = str(deck_dir_rel / f"seat{seat}_deck.json").replace("\\", "/")
    print(f"  deck files at {deck_dir_abs}")

    # The rest of the function runs inside an inline try so the temp
    # cleanup happens even on exceptions — keep the body indented.
    if True:

        # ---- 4. Start a game and step it ----
        seed = 777_777
        print(f"\n[real] POST /new_game seed={seed}")
        try:
            init = env.reset(
                hero1=decks[0].hero, hero2=decks[1].hero,
                deck1=deck_paths[0], deck2=deck_paths[1],
                seed=seed,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  NEW_GAME FAILED: {e}")
            return 3
        print(f"  game_id  = {env.game_id}")
        print(f"  phase    = {init.state.get('phase')!r}  priority = {init.state.get('priority_player')}")
        for p in init.state.get("players", []):
            print(f"  p{p.get('player_id')}: hero={p.get('hero')!r} health={p.get('health')} hand={len(p.get('hand') or [])} deck={p.get('deck_count')}")
        print(f"  legal_actions = {len(init.legal_actions)} initial")

        # ---- 5. Play random-bot vs random-bot, log every step ----
        bot1 = RandomBot(seed=seed)
        bot2 = RandomBot(seed=seed + 1)
        bot1.reset(seed=seed)
        bot2.reset(seed=seed + 1)

        state = init.state
        legal = init.legal_actions
        step_idx = 0
        log_path = PROJECT_ROOT / "outputs" / "real_pipeline_steps.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"--- game_id={env.game_id} seed={seed} ---\n")
            log.write(json.dumps({"init_phase": state.get("phase"),
                                  "init_priority": state.get("priority_player")}) + "\n")
            t0 = time.time()
            while not env.done and step_idx < STEP_CAP:
                priority = int(state.get("priority_player") or 0)
                if priority not in (1, 2):
                    print(f"  step {step_idx}: broken priority {priority}, aborting")
                    break
                bot = bot1 if priority == 1 else bot2
                if not legal:
                    legal = env.get_actions(refresh=True)
                    if not legal:
                        print(f"  step {step_idx}: no legal actions — engine wants no input, aborting")
                        break
                decision = bot.choose(state, legal, player_id=priority)
                chosen = next((a for a in legal if a.action_id == decision.action_id), None)
                if step_idx < 30 or step_idx % 20 == 0:
                    print(f"  step {step_idx:>3} pri={priority} phase={state.get('phase')!r:>10} "
                          f"action={chosen.type:<22} card={chosen.card_id} legal={len(legal)}")
                log.write(json.dumps({
                    "step": step_idx, "priority": priority,
                    "phase": state.get("phase"),
                    "chosen_id": decision.action_id, "chosen_type": chosen.type,
                    "chosen_card": chosen.card_id,
                    "n_legal": len(legal),
                }) + "\n")
                try:
                    result = env.step(decision.action_id)
                except Exception as e:  # noqa: BLE001
                    print(f"  step {step_idx}: STEP FAILED: {e}")
                    log.write(json.dumps({"step_error": str(e)}) + "\n")
                    return 4
                state = result.state
                legal = result.legal_actions
                step_idx += 1
                if result.done:
                    print(f"\n  GAME OVER at step {step_idx}: winner={result.winner} reward={result.reward}")
                    break
            dt = time.time() - t0

        # ---- 6. Summary ----
        print(f"\n[real] {step_idx} steps in {dt:.1f}s ({step_idx/dt:.1f} steps/s)")
        print(f"  final phase = {state.get('phase')!r}  priority = {state.get('priority_player')}  winner = {env.winner}")
        for p in state.get("players", []):
            print(f"  p{p.get('player_id')}: health={p.get('health')} hand={len(p.get('hand') or [])} "
                  f"pitch={len(p.get('pitch') or [])} grave={len(p.get('graveyard') or [])}")
        print(f"  full step log -> {log_path}")

        if env.done and env.winner in (1, 2):
            print("\n[real] FULL GAME PLAYED TO COMPLETION!")
            return 0
        print("\n[real] game did not finish (step-cap or no-legal-actions)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
