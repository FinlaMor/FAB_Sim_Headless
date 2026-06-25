"""End-to-end smoke test for the **full limited pipeline**.

Runs::

    Draft pod (8 players)
      -> Deck construction (HeuristicDeckBuilder)
      -> Single-elimination bracket (Talishar stub adapter)
      -> Parquet output for drafts / decks / games / tournaments
      -> Analytics over the recorded data

Like ``smoke_test.py``, this test spins up an in-process Python HTTP
stub that mimics the real PHP adapter's wire protocol, so the pipeline
can be exercised without Docker / PHP. Switch ``ADAPTER_URL`` to your
running adapter for the real-Talishar end-to-end check.

::

    python -m python.examples.limited_pipeline_smoke
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the gameplay smoke test's in-process stub HTTP adapter.
from python.examples.smoke_test import _start_stub  # noqa: E402
from python.pipeline import LimitedPipeline, PipelineConfig  # noqa: E402
from python.datasets.reader import list_artifacts  # noqa: E402
from python.analytics import hero_winrate, seat_ev, signal_detection  # noqa: E402
from python.tournament.bracket import render_text  # noqa: E402
from python.draft.bots.heuristic_bot import HeuristicDraftBot  # noqa: E402


def _all_heuristic_drafters(seat: int, seed: int):
    """Smoke-test draft factory: deterministic + always rare-drafts hero/weapon.

    The default factory in pipeline.py alternates random/heuristic for
    analytics variety; the smoke test forces all-heuristic so we never
    end up with a seat that has no hero/weapon in their pool.
    """
    return HeuristicDraftBot(seed=seed + seat)


def main() -> int:
    port = 8766
    server, _ = _start_stub(port)
    try:
        with tempfile.TemporaryDirectory(prefix="fab_pipeline_") as td:
            cfg = PipelineConfig(
                adapter_url=f"http://127.0.0.1:{port}",
                packs_path=str(PROJECT_ROOT / "decks" / "sample_packs" / "oma_sample.json"),
                out_dir=td,
                seed=42,
                n_pods=1,
                n_players=8,
                packs_per_player=3,
                best_of=1,
                draft_bot_factory=_all_heuristic_drafters,
            )
            print(f"[pipeline-smoke] cfg.out_dir = {td}")
            pipeline = LimitedPipeline(cfg)
            result = pipeline.run_cycle()

            # --- assertions on pod / decks ---
            assert len(result.pods) == 1
            pod = result.pods[0]
            assert len(pod.seats) == 8
            assert all(len(s.drafted) == 39 for s in pod.seats), \
                [len(s.drafted) for s in pod.seats]
            print(f"[pipeline-smoke] drafted: 8 seats * 39 cards = {sum(len(s.drafted) for s in pod.seats)}")
            assert len(result.decks_by_pod_seat) == 8

            for (pod_id, seat), deck in result.decks_by_pod_seat.items():
                assert deck.size >= 30, f"seat {seat} deck too small: {deck.size}"
                assert deck.hero, f"seat {seat} missing hero"
                assert deck.weapon, f"seat {seat} missing weapon"
            print(f"[pipeline-smoke] 8 decks built, all >= 30 cards, hero+weapon set")

            # --- tournament ---
            assert len(result.tournaments) == 1
            tour = result.tournaments[0]
            assert tour.placements, "no placements recorded"
            print("[pipeline-smoke] bracket:")
            for line in render_text(tour.bracket).splitlines():
                print("  " + line)
            champ = tour.champion()
            assert champ is not None, "no champion crowned"
            print(f"[pipeline-smoke] champion: {champ.label} ({champ.deck.hero})")

            # --- artefacts ---
            artefacts = list_artifacts(td)
            print(f"[pipeline-smoke] artefacts on disk: {artefacts}")
            assert artefacts["drafts"] >= 1,       f"no drafts parquet: {artefacts}"
            assert artefacts["decks"] >= 1,        f"no decks parquet: {artefacts}"
            assert artefacts["games"] >= 1,        f"no games parquet: {artefacts}"
            assert artefacts["tournaments"] >= 1,  f"no tournaments parquet: {artefacts}"

            # --- analytics ---
            wr = hero_winrate.compute(td)
            print(f"[pipeline-smoke] hero win-rate:\n{wr.to_string(index=False)}")
            sev = seat_ev.compute(td)
            print(f"[pipeline-smoke] seat EV:\n{sev.to_string(index=False)}")
            sig = signal_detection.compute(td)
            print(f"[pipeline-smoke] signal-detection rows: {len(sig)}")

            print("[pipeline-smoke] ALL CHECKS PASSED")
        return 0
    finally:
        server.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
