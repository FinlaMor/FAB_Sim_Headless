"""Convert the real Draftmancer logs (real_draft_references/) into draft-pick
parquet for training a HUMAN-BASELINE draft bot. Each human pick becomes one row
in the schema iql_draft expects:
  pod_id, seat, pack_number, pick_number, chosen_card, pack_contents_json, placement
No tournament placement is known (these are drafts, not played-out events), so
placement=0 -> the IQL trainer's advantage weighting is uniform = behaviour
cloning of the human picks. Writes outputs/parquet/drafts_human/human.parquet.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.draft.draftmancer import slugify  # noqa: E402

REFS = PROJECT_ROOT / "real_draft_references"
OUT = PROJECT_ROOT / "outputs" / "parquet" / "drafts_human"


def _slug(name: str) -> str:
    return slugify(name.split("_custom_")[0])


def main() -> int:
    rows: list[dict] = []
    seats = 0
    files = sorted(glob.glob(str(REFS / "*.txt")))
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        pod_id = re.sub(r"[^A-Za-z0-9]", "_", Path(f).stem)[:32]
        seat = 0
        for uid, u in (d.get("users") or {}).items():
            if u.get("isBot"):
                continue
            picks = u.get("picks", []) or []
            if len(picks) < 30:
                continue
            wrote = False
            for pk in picks:
                booster = pk.get("booster", []) or []
                idxs = pk.get("pick", []) or []
                if not booster or not idxs:
                    continue
                ci = idxs[0]
                if not (0 <= ci < len(booster)):
                    continue
                rows.append({
                    "pod_id": pod_id,
                    "seat": seat,
                    "pack_number": int(pk.get("packNum", 0)),
                    "pick_number": int(pk.get("pickNum", 0)),
                    "chosen_card": _slug(booster[ci]),
                    "pack_contents_json": json.dumps([_slug(c) for c in booster],
                                                     separators=(",", ":")),
                    "placement": 0,
                })
                wrote = True
            if wrote:
                seat += 1
                seats += 1

    if not rows:
        print("no human picks found"); return 1
    OUT.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows), str(OUT / "human.parquet"))
    print(f"{len(rows)} pick rows from {seats} human seats across {len(files)} logs -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
