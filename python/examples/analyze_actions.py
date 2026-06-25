"""Scan training data: which action types actually appear (legal vs chosen)."""
from __future__ import annotations
import glob, json, os, collections
import pyarrow.parquet as pq

f = max(glob.glob("outputs/parquet/games/*.parquet"), key=os.path.getmtime)
print("analyzing", os.path.basename(f))
t = pq.read_table(f, columns=["legal_actions_json", "chosen_action_json"]).to_pylist()

legal_types = collections.Counter()
chosen_types = collections.Counter()
activate_examples = []
for r in t:
    legals = json.loads(r["legal_actions_json"]) or []
    for a in legals:
        legal_types[a.get("type", "?")] += 1
    ch = json.loads(r["chosen_action_json"])
    chosen_types[ch.get("type", "?")] += 1
    if ch.get("type") == "ACTIVATE_HERO_OR_EQUIP":
        activate_examples.append(ch.get("card_id"))

print("\nLEGAL action types offered (count over all steps):")
for k, v in legal_types.most_common():
    print(f"  {k:28} {v}")
print("\nCHOSEN action types (what bots actually did):")
for k, v in chosen_types.most_common():
    print(f"  {k:28} {v}")
print(f"\nACTIVATE_HERO_OR_EQUIP chosen {len(activate_examples)}x; sample cards: "
      f"{collections.Counter(activate_examples).most_common(8)}")
