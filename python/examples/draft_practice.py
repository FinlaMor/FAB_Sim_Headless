"""Practice drafting against the draft bots — with an optional advisor.

You take one seat; the other seats are filled by draft bots. Each pick,
your pack is shown and you choose a card. With ``--advisor`` on, the bot
also ranks/scores every card in the pack so you can compare your pick to
the bot's evaluation (a draft-assistant mode).

    # draft with the heuristic advisor showing rankings
    python -m python.examples.draft_practice --advisor heuristic

    # use the trained IQL draft model as the advisor
    python -m python.examples.draft_practice --advisor iql

    # no advisor, just draft
    python -m python.examples.draft_practice --advisor none --seat 3

At the end it prints your drafted pool and (with --build) builds your deck.
Run it in a real terminal (it reads your picks from stdin).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.draft.draftmancer import slugify, parse_draftmancer, load_pack_pool_draftmancer  # noqa: E402
from python.draft.simulator import DraftPodConfig, DraftSimulator  # noqa: E402
from python.draft.bots.heuristic_bot import HeuristicDraftBot  # noqa: E402
from python.draft.bots.human_bot import HumanDraftBot  # noqa: E402

CUBE = PROJECT_ROOT / "OMN_Draft_3.5.txt"


def build_card_info(cube_path: Path) -> dict[str, dict]:
    """slug -> {name, cost, type} parsed from the cube's [CustomCards]."""
    txt = cube_path.read_text(encoding="utf-8")
    m = re.search(r"\[CustomCards\]\s*(\[.*?\])\s*\[", txt, re.S)
    info: dict[str, dict] = {}
    if not m:
        return info
    for c in json.loads(m.group(1)):
        info[slugify(c["name"])] = {
            "name": c.get("name", ""),
            "cost": c.get("mana_cost", ""),
            "type": c.get("type", ""),
        }
    return info


def make_advisor(kind: str, ckpt: str, card_info: dict):
    if kind == "none":
        return None
    if kind == "iql":
        from python.draft.bots.iql_bot import IQLDraftBot
        bot = IQLDraftBot(checkpoint=ckpt, seed=0)
        if bot._net is None:
            print(f"[advisor] IQL model not loadable at {ckpt}; using heuristic advisor instead")
            return HeuristicDraftBot(seed=0)
        return bot
    return HeuristicDraftBot(seed=0)  # "heuristic" (default)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seat", type=int, default=0, help="your seat (0..players-1)")
    ap.add_argument("--players", type=int, default=8)
    ap.add_argument("--packs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--advisor", choices=["heuristic", "iql", "none"], default="heuristic")
    ap.add_argument("--advisor-ckpt", default="outputs/models/draft/latest.pt")
    ap.add_argument("--build", action="store_true", help="build your deck after the draft")
    args = ap.parse_args()

    card_info = build_card_info(CUBE)
    cube = parse_draftmancer(str(CUBE))
    n_packs = args.players * args.packs
    pool = load_pack_pool_draftmancer(str(CUBE), n_packs=n_packs, seed=args.seed)

    advisor = make_advisor(args.advisor, args.advisor_ckpt, card_info)
    if not (0 <= args.seat < args.players):
        print(f"--seat must be 0..{args.players-1}"); return 2

    bots = []
    for s in range(args.players):
        if s == args.seat:
            bots.append(HumanDraftBot(card_info=card_info, advisor=advisor))
        else:
            bots.append(HeuristicDraftBot(seed=1000 + s))

    print(f"\nDrafting: you are seat {args.seat} of {args.players}; "
          f"advisor={args.advisor if advisor else 'off'}; {args.packs} packs.\n")
    sim = DraftSimulator(pool, bots, DraftPodConfig(
        n_players=args.players, packs_per_player=args.packs, seed=args.seed,
        pod_id="practice"))
    pod = sim.run()

    my_pool = list(pod.drafted_pool(args.seat))
    print("\n" + "#" * 60)
    print(f"DRAFT COMPLETE — your pool ({len(my_pool)} cards):")
    from collections import Counter
    for slug, n in sorted(Counter(my_pool).items(), key=lambda kv: card_info.get(kv[0], {}).get("name", kv[0])):
        print(f"  {n}x {card_info.get(slug, {}).get('name', slug)}")

    if args.build:
        from python.deckbuilding.builder import HeuristicDeckBuilder
        from python.pipeline import default_hero_assignment
        classes = cube.class_map()
        hero, weapon = default_hero_assignment(args.seat, pod, classes)
        builder = HeuristicDeckBuilder(card_classes=classes, seed=args.seed)
        deck = builder.build_deck([hero, weapon, *my_pool])
        out = PROJECT_ROOT / "decks" / "practice" / f"seat{args.seat}_deck.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        deck.save_json(str(out))
        print(f"\nBuilt deck: hero={deck.hero} weapon={deck.weapon} size={deck.size}")
        print(f"  saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
