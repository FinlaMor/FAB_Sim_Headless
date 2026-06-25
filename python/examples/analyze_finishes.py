"""Quick analysis: lethal vs life-tiebreak vs draw for the newest games parquet."""
from __future__ import annotations
import glob, json, os, sys
import pyarrow.parquet as pq

files = sorted(glob.glob("outputs/parquet/games/*.parquet"), key=os.path.getmtime)
f = sys.argv[1] if len(sys.argv) > 1 else files[-1]
print("analyzing", os.path.basename(f))
t = pq.read_table(f, columns=["game_id", "winner", "step_index", "next_state_json"]).to_pylist()
last: dict = {}
for r in t:
    g = r["game_id"]
    if g not in last or r["step_index"] > last[g]["step_index"]:
        last[g] = r
lethal = tiebreak = draw = 0
for g, r in last.items():
    w = int(r["winner"] or 0)
    ns = json.loads(r["next_state_json"])
    hps = [int(p.get("health") or 0) for p in ns.get("players", [])]
    if w == 0:
        draw += 1
    elif hps and min(hps) <= 0:
        lethal += 1
    else:
        tiebreak += 1
tot = len(last) or 1
print(f"games={tot}  lethal={lethal} ({100*lethal/tot:.0f}%)  "
      f"life-tiebreak={tiebreak} ({100*tiebreak/tot:.0f}%)  draw={draw}")
print("prior AggroBot full run: games=290 lethal~80 (28%) life-tiebreak=199 (69%) draw=11")
