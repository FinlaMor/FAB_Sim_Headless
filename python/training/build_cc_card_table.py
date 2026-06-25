"""Build the Classic Constructed training vocabulary + attribute table and bake
it into a single artifact the CC gameplay retrain can load.

The vocab is the union of two sources, so the model can both (a) see every CC
card a future deck might field and (b) handle the non-card STATE tokens that
only appear at play time:

* **Card universe** — every CC-legal card slug in ``slug_index.json``, plus
  every slug in the resolved CC decks (decks/resolved/cc_*_game.json).
* **State tokens** — every identity token actually emitted by the tokenizer
  over the recorded CC games (datasets/cc/parquet): hero/zone cards, chain/
  stack/effect ``card_id``s, and engine sentinels like ``RESOLUTIONSTEP`` that
  ride the same token slot. Collected with the *exact* training-time functions
  (``iql_gameplay._iter_cards``) so the vocab matches what the net will see.

Non-card tokens (sentinels, any OOV) simply get a zero attribute row — the same
treatment as pad/unk — which is correct: they carry no card metadata.

Output (torch): ``outputs/models/cc/cc_card_table.pt`` with::

    {"vocab": {"itos": [...]},        # CardVocab.to_dict()
     "attr_matrix": float32[V, D],    # aligned to vocab order; pad/unk = 0
     "attr_dim": D,                   # == cc_card_attrs.CC_ATTR_DIM
     "n_cards": V}

Run: ``python -m python.training.build_cc_card_table [--no-scan] [--window N]``
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

from . import features as F
from .features import CardVocab
from .cc_card_attrs import CCCardAttributes, build_cc_attr_matrix, CC_ATTR_DIM
from .iql_gameplay import _iter_cards

_REPO = Path(__file__).resolve().parents[2]
_OUT = _REPO / "outputs" / "models" / "cc" / "cc_card_table.pt"
_CC_PARQUET = _REPO / "datasets" / "cc" / "parquet" / "games"

# A scanned state token earns a vocab slot only if it carries identity: a real
# card (has metadata) OR a bounded alphabetic engine sentinel (ATTACKSTEP,
# RESOLUTIONSTEP, Go_Again, WATERYGRAVE, ...). Everything else the tokenizer
# emits — pure numerics, unbounded zone-target refs (THEIRDISCARD-147), and
# comma-compound payloads (gauntlets_of_iron_will,ACTIVE) — is targeting/index
# NOISE of unbounded cardinality that can never be fully covered, so it's left
# to map to UNK instead of bloating the vocab. (See state-encoding gaps memo.)
_SENTINEL_RE = re.compile(r"[A-Za-z][A-Za-z_]*")


def _keep_state_token(tok: str, attrs: "CCCardAttributes") -> bool:
    if attrs.covers(tok):
        return True
    if re.fullmatch(r"-?\d+", tok) or "," in tok:
        return False
    if re.fullmatch(r"(THEIR|MY)[A-Z]+-\d+", tok):
        return False
    return bool(_SENTINEL_RE.fullmatch(tok))


def cc_card_universe() -> set[str]:
    """All CC-legal card slugs (game-slug/underscore form), unioned with the
    slugs present in the resolved CC decks."""
    idx = json.loads((_REPO / "slug_index.json").read_text(encoding="utf-8"))["by_slug"]
    slugs = {k.replace("-", "_") for k, e in idx.items()
             if "Classic Constructed" in (e.get("legalFormats") or [])}
    for f in glob.glob(str(_REPO / "decks" / "resolved" / "cc_*_game.json")):
        j = json.loads(Path(f).read_text(encoding="utf-8"))
        slugs.update(j.get("deck") or [])
        slugs.update(j.get("equipment") or [])
        if j.get("hero"):
            slugs.add(j["hero"])
    return {s for s in slugs if s}


def scan_state_tokens(parquet_dir: Path = _CC_PARQUET, window: int = 0) -> set[str]:
    """Set of every identity token emitted over the recorded CC games. Reads
    file-by-file (only the 3 token-bearing columns) to bound memory; uses the
    training-time ``_iter_cards`` so tokens match the net's view exactly."""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(parquet_dir / "*.parquet")), key=lambda p: Path(p).stat().st_mtime)
    if window:
        files = files[-window:]
    cols = ["state_json", "chosen_action_json", "legal_actions_json"]
    toks: set[str] = set()
    for fp in files:
        rows = pq.read_table(fp, columns=cols).to_pylist()
        for t in _iter_cards(rows):
            if t:
                toks.add(str(t))
    return toks


def build(out: Path = _OUT, *, scan: bool = True, window: int = 0) -> dict:
    import torch

    attrs = CCCardAttributes.from_slug_index()
    cards = cc_card_universe()
    raw_state = scan_state_tokens(window=window) if scan else set()
    # Filter scanned tokens: keep real cards + meaningful sentinels, drop the
    # numeric / target-index / comma-compound noise (-> UNK at inference).
    extra = sorted(raw_state - cards)
    kept = {t for t in extra if _keep_state_token(t, attrs)}
    dropped = len(extra) - len(kept)
    sentinels = sorted(t for t in kept if not attrs.covers(t))
    recovered_cards = sorted(t for t in kept if attrs.covers(t))

    tokens = cards | kept
    vocab = CardVocab(tokens)
    M = build_cc_attr_matrix(vocab, attrs)

    # TRUE metadata coverage = tokens with real card attributes (slug_index OR
    # card_stats). Sentinels legitimately carry none -> zero attr row.
    with_meta = sum(1 for t in tokens if attrs.covers(t))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vocab": vocab.to_dict(),
        "attr_matrix": torch.from_numpy(M),
        "attr_dim": CC_ATTR_DIM,
        "n_cards": len(vocab),
    }
    torch.save(payload, out)

    print(f"[cc-card-table] CC-legal card universe : {len(cards)}")
    print(f"[cc-card-table] state tokens scanned    : {len(raw_state)}"
          f"{'  (scan off)' if not scan else ''}")
    print(f"[cc-card-table]   kept sentinels        : {len(sentinels)} "
          f"(e.g. {sentinels[:6]})")
    print(f"[cc-card-table]   cards recovered       : {len(recovered_cards)} "
          f"(e.g. {recovered_cards[:4]})")
    print(f"[cc-card-table]   dropped as noise->UNK : {dropped}")
    print(f"[cc-card-table] vocab (incl pad/unk)    : {len(vocab)}")
    print(f"[cc-card-table] attribute dim           : {CC_ATTR_DIM}")
    print(f"[cc-card-table] cards w/ real metadata  : {with_meta}/{len(tokens)} "
          f"({100*with_meta/max(len(tokens),1):.1f}%; rest = sentinels, zero attr row)")
    print(f"[cc-card-table] matrix shape            : {tuple(M.shape)}")
    print(f"[cc-card-table] saved -> {out.relative_to(_REPO)}")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build CC vocab + attribute table.")
    ap.add_argument("--no-scan", action="store_true",
                    help="skip the parquet state-token scan (card universe only)")
    ap.add_argument("--window", type=int, default=0,
                    help="scan only the most recent N CC parquet files (0=all)")
    args = ap.parse_args(argv)
    build(scan=not args.no_scan, window=args.window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
