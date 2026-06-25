"""Human pick agreement: replay real Draftmancer logs through the draft bots.

For every HUMAN player in real_draft_references/DraftLog_*.txt, replays
their draft pick-by-pick: the bot sees the same booster, the same pool so
far (the human's actual picks), and the same seen-cards history, and we
score whether the bot's top-1 / top-3 choices contain the human's pick.

This is a stable, training-free skill proxy: a draft bot that "thinks like
the humans who know the format" scores high; a memorizing or drifting bot
scores near the random baseline.

Run:  python -m python.examples.draft_agreement
          [--draft-ckpt outputs/models/draft/latest.pt] [--refs real_draft_references]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.draft.bots.base import DraftPodView  # noqa: E402
from python.draft.bots.iql_bot import IQLDraftBot  # noqa: E402
from python.draft.bots.heuristic_bot import HeuristicDraftBot  # noqa: E402


def _view(booster, pool, pack_no, pick_no) -> DraftPodView:
    """Minimal single-seat view for replaying a logged pick (no neighbour
    info in Draftmancer logs — empty tuples keep the heuristics neutral)."""
    return DraftPodView(
        seat=0, pack_number=pack_no + 1, pick_number=pick_no + 1,
        current_pack=tuple(booster), drafted_so_far=tuple(pool),
        left_neighbour_seat=1, right_neighbour_seat=7,
        left_neighbour_drafted=(), right_neighbour_drafted=(),
        n_seats=8, pass_direction=1,
    )


def slug(name: str) -> str:
    # "Nebula Duality (red)_custom_OMN121" -> "nebula_duality_red"
    base = name.split("_custom_")[0]
    return re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")


def load_human_drafts(refs_dir: Path):
    """Yield (file, user_name, picks) where picks is an ordered list of
    (pack_number, pick_number, booster_slugs, human_pick_slug)."""
    for fp in sorted(refs_dir.glob("DraftLog_*.txt")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for u in (data.get("users") or {}).values():
            if u.get("isBot"):
                continue
            picks = []
            for p in u.get("picks") or []:
                booster = [slug(c) for c in (p.get("booster") or [])]
                sel = p.get("pick") or []
                if not booster or not sel:
                    continue
                idx = int(sel[0])
                if not (0 <= idx < len(booster)):
                    continue
                picks.append((int(p.get("packNum", 0)), int(p.get("pickNum", 0)),
                              booster, booster[idx]))
            picks.sort(key=lambda t: (t[0], t[1]))
            if picks:
                yield fp.name, str(u.get("userName") or "?"), picks


def compute_agreement(draft_ckpt: str, refs_dir: str = "real_draft_references",
                      max_drafts: int | None = None) -> dict:
    """IQL-only human pick agreement (importable; used by the continuous
    loop's per-iteration metrics). Returns {top1_pct, top3_pct, n_picks,
    n_drafts}. Cheap: forward passes only."""
    drafts = list(load_human_drafts(Path(refs_dir)))
    if max_drafts:
        drafts = drafts[:max_drafts]
    if not drafts:
        return {}
    top1 = top3 = n = 0
    for _, _, picks in drafts:
        bot = IQLDraftBot(checkpoint=draft_ckpt, seed=0)
        bot.reset(seed=0)
        pool: list[str] = []
        for pack_no, pick_no, booster, human in picks:
            n += 1
            view = _view(booster, pool, pack_no, pick_no)
            scores = bot.score_cards(booster, list(pool), 0, pick_no, pack_no, view)
            ranked = sorted(scores, key=scores.get, reverse=True)
            if ranked and ranked[0] == human:
                top1 += 1
            if human in ranked[:3]:
                top3 += 1
            pool.append(human)
    return {"top1_pct": round(100 * top1 / n, 1),
            "top3_pct": round(100 * top3 / n, 1),
            "n_picks": n, "n_drafts": len(drafts)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-ckpt", default="outputs/models/draft/latest.pt")
    ap.add_argument("--refs", default="real_draft_references")
    args = ap.parse_args()

    drafts = list(load_human_drafts(Path(args.refs)))
    if not drafts:
        print("no human drafts found"); return 1
    print(f"[refs] {len(drafts)} human drafts in {args.refs}")

    stats = {b: defaultdict(int) for b in ("iql", "heuristic")}
    rand_top1 = rand_top3 = 0.0
    n_picks = 0
    by_pack = defaultdict(lambda: defaultdict(int))

    for fname, user, picks in drafts:
        bots = {
            "iql": IQLDraftBot(checkpoint=args.draft_ckpt, seed=0),
            "heuristic": HeuristicDraftBot(seed=0),
        }
        for b in bots.values():
            b.reset(seed=0)
        pool: list[str] = []
        for pack_no, pick_no, booster, human in picks:
            n_picks += 1
            rand_top1 += 1.0 / len(booster)
            rand_top3 += min(3, len(booster)) / len(booster)
            view = _view(booster, pool, pack_no, pick_no)
            for name, bot in bots.items():
                # score_cards both observes the pack (seen-signal tracking)
                # and gives a full ranking; fall back to choose_card.
                try:
                    scores = bot.score_cards(booster, list(pool), 0,
                                             pick_no, pack_no, view)
                    ranked = sorted(scores, key=scores.get, reverse=True)
                except Exception:  # noqa: BLE001
                    d = bot.choose_card(booster, list(pool), 0,
                                        pick_no, pack_no, view)
                    ranked = [d.card_id]
                if ranked and ranked[0] == human:
                    stats[name]["top1"] += 1
                    by_pack[pack_no][name + "_top1"] += 1
                if human in ranked[:3]:
                    stats[name]["top3"] += 1
            by_pack[pack_no]["n"] += 1
            pool.append(human)   # the human's actual pool, not the bot's

    print(f"\n==== PICK AGREEMENT (n={n_picks} human picks) ====")
    print(f"{'bot':>10} | top-1 | top-3")
    print(f"{'random':>10} | {100*rand_top1/n_picks:5.1f}% | {100*rand_top3/n_picks:5.1f}%")
    for name in ("heuristic", "iql"):
        s = stats[name]
        print(f"{name:>10} | {100*s['top1']/n_picks:5.1f}% | {100*s['top3']/n_picks:5.1f}%")
    print("\ntop-1 by pack number (early picks are the signal-free skill test):")
    for pk in sorted(by_pack):
        row = by_pack[pk]
        n = row["n"] or 1
        print(f"  pack {pk}: n={n:>4}  iql={100*row['iql_top1']/n:5.1f}%  "
              f"heuristic={100*row['heuristic_top1']/n:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
