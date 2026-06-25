"""End-to-end smoke test against the real **OMN_Draft_3.5.txt** cube.

Differences from ``limited_pipeline_smoke.py``:

* Uses the Draftmancer cube file (`OMN_Draft_3.5.txt`) instead of the
  synthetic JSON pool.
* Heroes / signature weapons are assigned outside the booster pool via
  :func:`python.pipeline.round_robin_hero_assignment` because the OMN
  cube ships no hero cards — players bring their own.
* Same stub-adapter HTTP server (no PHP / Docker required to verify the
  Python plumbing; swap to your running adapter to exercise Talishar).

::

    python -m python.examples.omn_pipeline_smoke
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the gameplay smoke test's in-process stub HTTP adapter.
from python.examples.smoke_test import _start_stub  # noqa: E402
from python.pipeline import (  # noqa: E402
    LimitedPipeline, PipelineConfig, default_hero_assignment,
)
from python.draft.draftmancer import (  # noqa: E402
    load_pack_pool_draftmancer, parse_draftmancer,
)
from python.draft.bots.heuristic_bot import HeuristicDraftBot  # noqa: E402
from python.datasets.reader import list_artifacts  # noqa: E402
from python.analytics import hero_winrate, seat_ev  # noqa: E402
from python.tournament.bracket import render_text  # noqa: E402


CUBE_PATH = PROJECT_ROOT / "OMN_Draft_3.5.txt"
N_PLAYERS = 8
PACKS_PER_PLAYER = 3


def _heuristic_drafters(seat: int, seed: int):
    """All-heuristic to make pack contents irrelevant to draft-bot variance."""
    return HeuristicDraftBot(seed=seed + seat)


def main() -> int:
    if not CUBE_PATH.is_file():
        print(f"[omn-smoke] cube file missing: {CUBE_PATH}", file=sys.stderr)
        return 2

    port = 8767
    server, _ = _start_stub(port)
    try:
        with tempfile.TemporaryDirectory(prefix="omn_pipeline_") as td:
            # Generate exactly enough packs for one pod (seed = pipeline seed).
            n_packs = N_PLAYERS * PACKS_PER_PLAYER
            cube = parse_draftmancer(str(CUBE_PATH))
            cfg = PipelineConfig(
                adapter_url=f"http://127.0.0.1:{port}",
                packs_path=str(CUBE_PATH),
                out_dir=td,
                seed=2026_05_29,
                n_pods=1,
                n_players=N_PLAYERS,
                packs_per_player=PACKS_PER_PLAYER,
                best_of=1,
                draft_bot_factory=_heuristic_drafters,
                hero_assignment=default_hero_assignment,
                pack_pool_factory=lambda: load_pack_pool_draftmancer(
                    str(CUBE_PATH), n_packs=n_packs, seed=2026_05_29,
                ),
                cube=cube,
            )
            print(f"[omn-smoke] cube       = {CUBE_PATH.name}")
            print(f"[omn-smoke] out_dir    = {td}")
            pipeline = LimitedPipeline(cfg)
            print(f"[omn-smoke] pool size  = {len(pipeline.pool)} packs "
                  f"(universe = {len(pipeline.pool.card_universe())} distinct cards)")
            print(f"[omn-smoke] pack size  = {len(pipeline.pool.packs[0].cards)} cards/pack")

            result = pipeline.run_cycle()

            # --- draft pod ---
            assert len(result.pods) == 1
            pod = result.pods[0]
            picks_per_seat = [len(s.drafted) for s in pod.seats]
            assert all(n == picks_per_seat[0] for n in picks_per_seat), picks_per_seat
            print(f"[omn-smoke] drafted    = 8 seats * {picks_per_seat[0]} cards "
                  f"= {sum(picks_per_seat)} total")

            # --- decks ---
            assert len(result.decks_by_pod_seat) == 8
            from python.draft.format import HERO_CLASS, HERO_WEAPONS, LEGAL_HEROES
            class_seen: dict[str, int] = {}
            for (pod_id, seat), deck in result.decks_by_pod_seat.items():
                assert deck.hero in LEGAL_HEROES, f"seat {seat} got hero {deck.hero!r}"
                assert deck.weapon == HERO_WEAPONS[deck.hero], \
                    f"seat {seat} weapon {deck.weapon!r} mismatches hero {deck.hero!r}"
                assert deck.size >= 30, f"seat {seat} deck size = {deck.size}"
                cls = HERO_CLASS[deck.hero]
                class_seen[cls] = class_seen.get(cls, 0) + 1
            print(f"[omn-smoke] decks      = 8 legal decks (>= 30 cards, hero+weapon match)")
            print(f"[omn-smoke] hero spread by class: {class_seen}")

            # --- bracket ---
            tour = result.tournaments[0]
            print("[omn-smoke] bracket:")
            for line in render_text(tour.bracket).splitlines():
                print("  " + line)
            champ = tour.champion()
            assert champ is not None
            print(f"[omn-smoke] champion   = {champ.label} ({champ.deck.hero} / {champ.deck.weapon})")

            # --- artefacts ---
            arts = list_artifacts(td)
            print(f"[omn-smoke] artefacts  = {arts}")
            for k in ("drafts", "decks", "games", "tournaments"):
                assert arts[k] >= 1, f"missing {k}: {arts}"

            # --- analytics ---
            wr = hero_winrate.compute(td)
            print(f"[omn-smoke] hero win-rate:\n{wr.to_string(index=False)}")
            sev = seat_ev.compute(td)
            print(f"[omn-smoke] seat EV (top 4):\n{sev.head(4).to_string(index=False)}")

            print("[omn-smoke] ALL CHECKS PASSED")
        return 0
    finally:
        server.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
